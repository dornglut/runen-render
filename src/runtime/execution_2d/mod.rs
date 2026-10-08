//! Private 2D semantic admission, retained field cache, and transactional ordered lowering.

mod field;
mod image;
mod intrinsic;
mod lowering;
mod vector;

use self::field::{FieldSetKey, QualityTier, ResourceFields};
use self::lowering::GlyphOccurrence;
pub(crate) use self::lowering::add_target_boundary;
use crate::composition_2d::{
    Render2dComposition, Render2dEntry, Render2dOpacity, Render2dPrimitive,
    Render2dResourceBindings, Render2dResourceId, Render2dResourceValue,
};
use crate::execution_2d::{
    Render2dExecutionError, Render2dPreparedContribution, Render2dTarget,
    Render2dUnsupportedContent,
};
use runen_gpu::GpuContext;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

#[derive(Debug, Default)]
pub(crate) struct Render2dExecutionState {
    observed: BTreeMap<Render2dResourceId, Render2dResourceValue>,
    fields: BTreeMap<FieldSetKey, Arc<ResourceFields>>,
}

impl Render2dExecutionState {
    pub(crate) fn discard_cache(&mut self) {
        self.fields.clear();
    }

    pub(crate) fn prepare(
        &mut self,
        context: &GpuContext,
        composition: &Render2dComposition,
        bindings: &Render2dResourceBindings,
        target: &Render2dTarget,
    ) -> Result<Render2dPreparedContribution, Render2dExecutionError> {
        composition.validate_bindings(bindings)?;
        let runs = admit_runs(composition)?;
        let admitted_target = lowering::admit_target(context, target, !runs.is_empty())?;
        let unique_resources = runs
            .iter()
            .map(|run| run.resource_id)
            .collect::<BTreeSet<_>>();
        let image_resources = composition
            .root_entries()
            .iter()
            .filter_map(|entry| match entry {
                Render2dEntry::Item(item) => match item.primitive() {
                    Render2dPrimitive::Image(image) => Some(image.resource_id()),
                    _ => None,
                },
                Render2dEntry::Group(_) => None,
            })
            .collect::<BTreeSet<_>>();

        let mut observed_updates = Vec::new();
        let mut field_updates = Vec::new();
        let mut resolved_fields = BTreeMap::<FieldSetKey, Arc<ResourceFields>>::new();

        for resource_id in unique_resources {
            let value = bindings
                .get(resource_id)
                .expect("composition validation proves every shaped resource is bound");
            let Render2dResourceValue::ShapedText(resource) = value else {
                unreachable!("F2 semantic admission retains only shaped-text requirements");
            };

            if let Some(existing) = self.observed.get(&resource_id) {
                if existing != value {
                    return Err(Render2dExecutionError::ResourceIdentityRebound { resource_id });
                }
            } else {
                observed_updates.push((resource_id, value.clone()));
            }

            let quality = QualityTier::select(resource.font_size(), admitted_target.raster_scale())
                .ok_or(Render2dUnsupportedContent::FontSize { resource_id })?;
            let key = FieldSetKey::new(resource_id, quality);
            if let Some(existing) = self.fields.get(&key) {
                existing
                    .validate_texture_limit(resource_id, admitted_target.max_texture_dimension_2d())
                    .map_err(crate::execution_2d::Render2dExecutionError::ShapedText)?;
                resolved_fields.insert(key, Arc::clone(existing));
                continue;
            }

            let realized = Arc::new(field::realize(
                resource_id,
                resource,
                quality,
                admitted_target.max_texture_dimension_2d(),
            )?);
            resolved_fields.insert(key, Arc::clone(&realized));
            field_updates.push((key, realized));
        }

        // Resource identity is observed transactionally for admitted image items too,
        // including empty/fully transparent mappings.
        for resource_id in image_resources {
            let value = bindings
                .get(resource_id)
                .expect("composition validation proves every image binding exists");
            if let Some(existing) = self.observed.get(&resource_id) {
                if existing != value {
                    return Err(Render2dExecutionError::ResourceIdentityRebound { resource_id });
                }
            } else {
                observed_updates.push((resource_id, value.clone()));
            }
        }

        let mut occurrences = BTreeMap::<usize, Vec<GlyphOccurrence>>::new();
        for run in &runs {
            let value = bindings
                .get(run.resource_id)
                .expect("composition validation proves every shaped resource is bound");
            let Render2dResourceValue::ShapedText(resource) = value else {
                unreachable!("F2 semantic admission retains only shaped-text requirements");
            };
            let quality = QualityTier::select(resource.font_size(), admitted_target.raster_scale())
                .ok_or(Render2dUnsupportedContent::FontSize {
                    resource_id: run.resource_id,
                })?;
            let key = FieldSetKey::new(run.resource_id, quality);
            let fields = resolved_fields
                .get(&key)
                .expect("every admitted shaped resource has a resolved field set");

            for glyph in resource.glyphs() {
                let Some(field) = fields.glyph(glyph.id()) else {
                    continue;
                };
                let logical_x = field.origin_x().mul_add(
                    resource.font_size(),
                    run.origin_x + run.translate_x + glyph.x(),
                );
                let logical_y = field.origin_y().mul_add(
                    resource.font_size(),
                    run.origin_y + run.translate_y + glyph.y(),
                );
                let logical_width =
                    f64::from(field.width()) / quality.pixels_per_em() * resource.font_size();
                let logical_height =
                    f64::from(field.height()) / quality.pixels_per_em() * resource.font_size();
                if ![logical_x, logical_y, logical_width, logical_height]
                    .into_iter()
                    .all(f64::is_finite)
                {
                    return Err(Render2dExecutionError::Gpu {
                        stage: "logical glyph realization",
                        detail: format!(
                            "2D shaped resource {} glyph {} produced non-finite logical bounds",
                            run.resource_id.get(),
                            glyph.id()
                        ),
                    });
                }
                occurrences
                    .entry(run.root_index)
                    .or_default()
                    .push(GlyphOccurrence {
                        resource_id: run.resource_id,
                        field: Arc::clone(field),
                        logical_x,
                        logical_y,
                        logical_width,
                        logical_height,
                        color: run.color,
                    });
            }
        }

        let mut ordered = Vec::new();
        for (root_index, entry) in composition.root_entries().iter().enumerate() {
            let Render2dEntry::Item(item) = entry else {
                unreachable!("admitted root item")
            };
            if matches!(item.primitive(), Render2dPrimitive::ShapedText(_)) {
                ordered.extend(
                    occurrences
                        .remove(&root_index)
                        .unwrap_or_default()
                        .into_iter()
                        .map(lowering::OrderedItem::Glyph),
                );
            } else if let Render2dPrimitive::Image(image) = item.primitive() {
                let value = bindings
                    .get(image.resource_id())
                    .expect("validated image binding");
                let Render2dResourceValue::ImageRgba8Srgb(source) = value else {
                    unreachable!("validated image binding kind")
                };
                for patch in image::realize(
                    item,
                    image,
                    source,
                    root_index,
                    admitted_target.raster_scale(),
                    admitted_target.canvas(),
                    admitted_target.max_texture_dimension_2d(),
                )? {
                    ordered.push(lowering::OrderedItem::Image(patch));
                }
            } else if let Some(mesh) = vector::realize(
                item,
                root_index,
                admitted_target.raster_scale(),
                admitted_target.canvas(),
                admitted_target.max_buffer_bytes(),
            )? {
                ordered.push(lowering::OrderedItem::Vector(mesh));
            }
        }
        let lowered = lowering::lower_ordered(&admitted_target, ordered)?;

        for (resource_id, value) in observed_updates {
            let previous = self.observed.insert(resource_id, value);
            debug_assert!(previous.is_none());
        }
        for (key, fields) in field_updates {
            let previous = self.fields.insert(key, fields);
            debug_assert!(previous.is_none());
        }

        Ok(Render2dPreparedContribution::new(
            lowered,
            target.view().clone(),
        ))
    }
}

#[derive(Clone, Copy, Debug)]
struct AdmittedRun {
    root_index: usize,
    resource_id: Render2dResourceId,
    origin_x: f64,
    origin_y: f64,
    color: crate::composition_2d::Render2dColorRgba8,
    translate_x: f64,
    translate_y: f64,
}

fn admit_runs(
    composition: &Render2dComposition,
) -> Result<Vec<AdmittedRun>, Render2dExecutionError> {
    let mut runs = Vec::with_capacity(composition.root_entries().len());
    for (root_index, entry) in composition.root_entries().iter().enumerate() {
        let Render2dEntry::Item(item) = entry else {
            return Err(Render2dUnsupportedContent::Group { root_index }.into());
        };
        if !item.clips().is_empty() {
            return Err(Render2dUnsupportedContent::Clips { root_index }.into());
        }
        match item.primitive() {
            Render2dPrimitive::Fill { .. } | Render2dPrimitive::Stroke { .. } => continue,
            Render2dPrimitive::Image(_) => continue,
            Render2dPrimitive::ShapedText(_) => {}
        }
        if item.opacity() != Render2dOpacity::OPAQUE {
            return Err(Render2dUnsupportedContent::Opacity { root_index }.into());
        }
        let [m11, m12, m21, m22, translate_x, translate_y] = item.local_to_parent().components();
        if m11 != 1.0 || m12 != 0.0 || m21 != 0.0 || m22 != 1.0 {
            return Err(Render2dUnsupportedContent::Transform { root_index }.into());
        }
        let Render2dPrimitive::ShapedText(text) = item.primitive() else {
            return Err(Render2dUnsupportedContent::Primitive { root_index }.into());
        };
        runs.push(AdmittedRun {
            root_index,
            resource_id: text.resource_id(),
            origin_x: text.origin().x(),
            origin_y: text.origin().y(),
            color: text.color(),
            translate_x,
            translate_y,
        });
    }
    Ok(runs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::{
        Render2dAffineTransform, Render2dColorRgba8, Render2dEntry, Render2dFontBinding,
        Render2dGlyph, Render2dItem, Render2dPoint, Render2dResourceBinding,
        Render2dShapedTextPrimitive, Render2dShapedTextResource,
    };

    fn shaped_composition(
        transform: Render2dAffineTransform,
    ) -> (Render2dComposition, Render2dResourceBindings) {
        let resource_id = Render2dResourceId::new(1).expect("resource id");
        let font = Render2dFontBinding::new(vec![1], 0, Vec::new(), false, None)
            .expect("semantic font binding");
        let glyph = Render2dGlyph::new(1, 0.0, 0.0, 8.0).expect("glyph");
        let resource =
            Render2dShapedTextResource::new(font, 16.0, vec![glyph]).expect("shaped resource");
        let bindings = Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
            resource_id,
            Render2dResourceValue::ShapedText(resource),
        )])
        .expect("bindings");
        let primitive = Render2dShapedTextPrimitive::new(
            resource_id,
            Render2dPoint::new(2.0, 3.0).expect("origin"),
            Render2dColorRgba8::WHITE,
        );
        let composition = Render2dComposition::new(vec![Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::ShapedText(primitive),
            transform,
            Vec::new(),
            Render2dOpacity::OPAQUE,
        ))])
        .expect("composition");
        (composition, bindings)
    }

    #[test]
    fn semantic_gate_accepts_translation_without_widening_linear_transform() {
        let (composition, _) = shaped_composition(
            Render2dAffineTransform::translation(4.0, -2.0).expect("translation"),
        );
        let runs = admit_runs(&composition).expect("translation is inside F2");
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].translate_x, 4.0);
        assert_eq!(runs[0].translate_y, -2.0);

        let (composition, _) = shaped_composition(
            Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, 0.0, 0.0).expect("finite transform"),
        );
        assert!(matches!(
            admit_runs(&composition),
            Err(Render2dExecutionError::UnsupportedContent(
                Render2dUnsupportedContent::Transform { root_index: 0 }
            ))
        ));
    }
}
