//! Bounded invocation-local lookup over the last successful field working set.
//!
//! Resource identities remain distinct. Only font/face/variation/synthesis,
//! quality and glyph facts select reusable physical fields. This index owns no
//! cross-invocation residency and may conservatively omit entries at its limit.

use super::{FieldSetKey, GlyphField, QualityTier, ResourceFields};
use crate::composition_2d::{Render2dFontBinding, Render2dResourceId, Render2dResourceValue};
use std::{cmp::Ordering, collections::BTreeMap, sync::Arc};

// Bound additional metadata independently of bitmap bytes, including valid
// non-painting glyph entries which charge no field pixels.
const MAX_INDEX_FONTS: usize = 64;
const MAX_INDEX_GLYPHS: usize = 4096;

struct FontKey<'a> {
    font: &'a Render2dFontBinding,
    quality: QualityTier,
}

impl Ord for FontKey<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        let bytes = if std::ptr::eq(self.font.bytes(), other.font.bytes()) {
            Ordering::Equal
        } else {
            self.font.bytes().cmp(other.font.bytes())
        };
        bytes
            .then_with(|| self.font.face_index().cmp(&other.font.face_index()))
            .then_with(|| {
                self.font
                    .normalized_coords()
                    .cmp(other.font.normalized_coords())
            })
            .then_with(|| self.font.faux_bold().cmp(&other.font.faux_bold()))
            // Exact bits are conservative even for semantically equal signed zero.
            .then_with(|| {
                self.font
                    .faux_skew()
                    .map(f64::to_bits)
                    .cmp(&other.font.faux_skew().map(f64::to_bits))
            })
            .then_with(|| self.quality.cmp(&other.quality))
    }
}

impl PartialOrd for FontKey<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for FontKey<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for FontKey<'_> {}

pub(in crate::runtime::execution_2d) struct FieldReuseIndex<'a> {
    by_font: BTreeMap<FontKey<'a>, ResourceFields>,
}

/// Constructible only by exact lookup through the index, never by a resource ID.
#[derive(Clone, Copy)]
pub(in crate::runtime::execution_2d) struct MatchingFields<'a>(&'a ResourceFields);

impl MatchingFields<'_> {
    pub(super) fn glyph(&self, id: u32) -> Option<&Option<Arc<GlyphField>>> {
        self.0.by_glyph.get(&id)
    }
}

impl<'a> FieldReuseIndex<'a> {
    pub(in crate::runtime::execution_2d) fn new(
        fields: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
        observed: &'a BTreeMap<Render2dResourceId, Render2dResourceValue>,
    ) -> Self {
        let mut by_font = BTreeMap::<FontKey<'a>, ResourceFields>::new();
        let mut indexed = 0;
        for (key, fields) in fields {
            let Some(Render2dResourceValue::ShapedText(resource)) = observed.get(&key.resource_id)
            else {
                continue;
            };
            let key = FontKey {
                font: resource.font(),
                quality: key.quality,
            };
            if by_font.len() == MAX_INDEX_FONTS && !by_font.contains_key(&key) {
                continue;
            }
            let shared = by_font.entry(key).or_insert_with(|| ResourceFields {
                by_glyph: BTreeMap::new(),
            });
            for (id, field) in &fields.by_glyph {
                if shared.by_glyph.contains_key(id) {
                    continue;
                }
                if indexed == MAX_INDEX_GLYPHS {
                    return Self { by_font };
                }
                shared.by_glyph.insert(*id, field.clone());
                indexed += 1;
            }
        }
        Self { by_font }
    }

    pub(in crate::runtime::execution_2d) fn resolve<'b>(
        &'b self,
        font: &'b Render2dFontBinding,
        quality: QualityTier,
    ) -> Option<MatchingFields<'b>> {
        self.by_font
            .get(&FontKey { font, quality })
            .map(MatchingFields)
    }
}

#[cfg(test)]
mod tests {
    use super::super::{FieldBudget, MAX_TEXT_FIELD_BYTES, realize};
    use super::*;
    use crate::composition_2d::{Render2dGlyph, Render2dShapedTextResource};
    use crate::execution_2d::{Render2dExecutionError, Render2dShapedTextError};

    const FONT: &[u8] = include_bytes!("../../../../tests/fixtures/f2_outline.ttf");
    fn font() -> Render2dFontBinding {
        Render2dFontBinding::new(FONT.to_vec(), 0, vec![], false, None).unwrap()
    }
    fn resource(font: Render2dFontBinding, size: f64, ids: &[u32]) -> Render2dShapedTextResource {
        Render2dShapedTextResource::new(
            font,
            size,
            ids.iter()
                .map(|id| Render2dGlyph::new(*id, 4., 3., size).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }
    fn prior() -> (
        BTreeMap<FieldSetKey, Arc<ResourceFields>>,
        BTreeMap<Render2dResourceId, Render2dResourceValue>,
    ) {
        let id = Render2dResourceId::new(1).unwrap();
        let r = resource(font(), 24., &[1, 2]);
        let fields = realize(
            id,
            &r,
            QualityTier::P24,
            4096,
            &mut FieldBudget::default(),
            None,
        )
        .unwrap();
        (
            BTreeMap::from([(FieldSetKey::new(id, QualityTier::P24), Arc::new(fields))]),
            BTreeMap::from([(id, Render2dResourceValue::ShapedText(r))]),
        )
    }

    #[test]
    fn new_identity_and_placement_reuse_exact_immutable_glyphs() {
        let (fields, observed) = prior();
        let index = FieldReuseIndex::new(&fields, &observed);
        let changed = resource(font(), 20., &[2, 1, 2]);
        let id = Render2dResourceId::new(2).unwrap();
        let result = realize(
            id,
            &changed,
            QualityTier::P24,
            4096,
            &mut FieldBudget::default(),
            index.resolve(changed.font(), QualityTier::P24),
        )
        .unwrap();
        let prior = fields.values().next().unwrap();
        assert!(Arc::ptr_eq(
            prior.glyph(2).unwrap(),
            result.glyph(2).unwrap()
        ));
        assert_eq!(result.by_glyph.len(), 2);
        assert!(
            matches!(result.by_glyph.get(&1), Some(None)),
            "valid empty outline is not a failed admission"
        );
        let fresh = realize(
            id,
            &changed,
            QualityTier::P24,
            4096,
            &mut FieldBudget::default(),
            None,
        )
        .unwrap();
        assert_eq!(
            result.glyph(2).unwrap().rgba8(),
            fresh.glyph(2).unwrap().rgba8()
        );
    }

    #[test]
    fn every_exact_font_fact_and_quality_separate_reuse() {
        let (fields, observed) = prior();
        let index = FieldReuseIndex::new(&fields, &observed);
        assert!(
            index.resolve(&font(), QualityTier::P24).is_some(),
            "equal bytes in an independent allocation are valid"
        );
        assert!(index.resolve(&font(), QualityTier::P32).is_none());
        let mut changed = FONT.to_vec();
        *changed.last_mut().unwrap() ^= 1;
        for changed in [
            Render2dFontBinding::new(changed, 0, vec![], false, None).unwrap(),
            Render2dFontBinding::new(FONT.to_vec(), 1, vec![], false, None).unwrap(),
            Render2dFontBinding::new(FONT.to_vec(), 0, vec![8192], false, None).unwrap(),
            Render2dFontBinding::new(FONT.to_vec(), 0, vec![], true, None).unwrap(),
            Render2dFontBinding::new(FONT.to_vec(), 0, vec![], false, Some(0.2)).unwrap(),
        ] {
            assert!(index.resolve(&changed, QualityTier::P24).is_none());
        }
    }

    #[test]
    fn reused_fields_pay_current_budget_and_obey_later_texture_limits() {
        let (fields, observed) = prior();
        let index = FieldReuseIndex::new(&fields, &observed);
        let next = resource(font(), 24., &[2]);
        let id = Render2dResourceId::new(3).unwrap();
        let shared = index.resolve(next.font(), QualityTier::P24);
        assert!(
            matches!(realize(id,&next,QualityTier::P24,1,&mut FieldBudget::default(),shared),Err(Render2dExecutionError::ShapedText(Render2dShapedTextError::GlyphExtentExceedsLimit {resource_id,..})) if resource_id==id)
        );
        let glyph = fields.values().next().unwrap().glyph(2).unwrap();
        let bytes = u64::from(glyph.width()) * u64::from(glyph.height()) * 4;
        let mut budget = FieldBudget {
            bytes: MAX_TEXT_FIELD_BYTES - bytes + 1,
        };
        let before = budget.bytes;
        assert!(
            matches!(realize(id,&next,QualityTier::P24,4096,&mut budget,shared),Err(Render2dExecutionError::ShapedText(Render2dShapedTextError::FieldBudgetExceeded{resource_id,..})) if resource_id==id)
        );
        assert_eq!(budget.bytes, before);
        let mut budget = FieldBudget::default();
        realize(id, &next, QualityTier::P24, 4096, &mut budget, shared).unwrap();
        assert_eq!(budget.bytes, bytes);
    }

    #[test]
    fn missing_and_invalid_glyphs_use_authoritative_realization() {
        let (fields, observed) = prior();
        let index = FieldReuseIndex::new(&fields, &observed);
        let bad = resource(font(), 24., &[u32::MAX]);
        let id = Render2dResourceId::new(4).unwrap();
        assert!(
            matches!(realize(id,&bad,QualityTier::P24,4096,&mut FieldBudget::default(),index.resolve(bad.font(),QualityTier::P24)),Err(Render2dExecutionError::ShapedText(Render2dShapedTextError::InvalidOutline{resource_id,..})) if resource_id==id)
        );
        assert!(
            index
                .resolve(bad.font(), QualityTier::P24)
                .unwrap()
                .glyph(u32::MAX)
                .is_none()
        );
        let valid = resource(font(), 24., &[2]);
        assert!(
            realize(
                id,
                &valid,
                QualityTier::P24,
                4096,
                &mut FieldBudget::default(),
                index.resolve(valid.font(), QualityTier::P24)
            )
            .is_ok()
        );
    }

    #[test]
    fn optional_index_bounds_distinct_font_metadata() {
        let mut fields = BTreeMap::new();
        let mut observed = BTreeMap::new();
        for number in 0..=MAX_INDEX_FONTS {
            let id = Render2dResourceId::new(number as u64 + 1).unwrap();
            let font = Render2dFontBinding::new(
                FONT.to_vec(),
                0,
                vec![],
                false,
                Some(number as f64 / 1000.),
            )
            .unwrap();
            observed.insert(
                id,
                Render2dResourceValue::ShapedText(resource(font, 24., &[1])),
            );
            fields.insert(
                FieldSetKey::new(id, QualityTier::P24),
                Arc::new(ResourceFields {
                    by_glyph: BTreeMap::from([(1, None)]),
                }),
            );
        }
        let index = FieldReuseIndex::new(&fields, &observed);
        assert_eq!(index.by_font.len(), MAX_INDEX_FONTS);
        let omitted = observed.values().last().unwrap();
        let Render2dResourceValue::ShapedText(omitted) = omitted else {
            unreachable!()
        };
        assert!(index.resolve(omitted.font(), QualityTier::P24).is_none());
        assert!(
            realize(
                Render2dResourceId::new(1000).unwrap(),
                omitted,
                QualityTier::P24,
                4096,
                &mut FieldBudget::default(),
                index.resolve(omitted.font(), QualityTier::P24)
            )
            .is_ok()
        );
    }

    #[test]
    fn optional_index_bounds_nonpainting_metadata() {
        let id = Render2dResourceId::new(1).unwrap();
        let r = resource(font(), 24., &[1]);
        let fields = BTreeMap::from([(
            FieldSetKey::new(id, QualityTier::P24),
            Arc::new(ResourceFields {
                by_glyph: (0..(MAX_INDEX_GLYPHS as u32 + 1))
                    .map(|g| (g, None))
                    .collect(),
            }),
        )]);
        let observed = BTreeMap::from([(id, Render2dResourceValue::ShapedText(r))]);
        let index = FieldReuseIndex::new(&fields, &observed);
        assert_eq!(
            index
                .by_font
                .values()
                .map(|x| x.by_glyph.len())
                .sum::<usize>(),
            MAX_INDEX_GLYPHS
        );
        let f = font();
        assert!(
            index
                .resolve(&f, QualityTier::P24)
                .unwrap()
                .glyph(MAX_INDEX_GLYPHS as u32)
                .is_none()
        );
    }
}
