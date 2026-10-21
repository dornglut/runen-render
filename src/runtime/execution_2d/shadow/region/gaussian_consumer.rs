//! Original-boundary consumer admission. No disposable mesh supplies eligibility.
use super::*;
use crate::runtime::execution_2d::shadow::{
    GaussianTarget, PreparedGaussianCaster, PreparedGaussianField,
};

impl NeutralRegionBuilder {
    /// A retained signed effect must be rejected by the existing polygon
    /// correspondence rule, and every source boundary must belong to the
    /// numerical compiler's original straight-boundary class. This is only
    /// consumer availability; the compiler still proves precision/resources.
    pub(in crate::runtime::execution_2d::shadow) fn original_gaussian_required(
        &mut self,
        root: &NeutralRegion<'_>,
    ) -> Result<bool, Render2dExecutionError> {
        fn shape(
            builder: &mut NeutralRegionBuilder,
            shape: &crate::composition_2d::Render2dShape,
            path: &[usize],
        ) -> Result<bool, Render2dExecutionError> {
            if shape.as_rect().is_some() {
                return Ok(true);
            }
            let Some(commands) = shape.as_path().map(|path| path.commands()) else {
                return Ok(false);
            };
            builder.charge_preparation(64 + commands.len() * 16, path)?;
            Ok(commands.iter().all(|command| {
                matches!(
                    command,
                    crate::composition_2d::Render2dPathCommand::MoveTo(_)
                        | crate::composition_2d::Render2dPathCommand::LineTo(_)
                        | crate::composition_2d::Render2dPathCommand::Close
                )
            }))
        }
        fn clips(
            builder: &mut NeutralRegionBuilder,
            clips: &[Render2dClip],
            path: &[usize],
        ) -> Result<bool, Render2dExecutionError> {
            builder.charge_preparation(64 + clips.len() * 64, path)?;
            for clip in clips {
                if !shape(builder, clip.shape(), path)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        struct Eligibility {
            polygon_possible: bool,
            requires_original: bool,
            supported_boundaries: bool,
        }
        fn visit(
            builder: &mut NeutralRegionBuilder,
            node: &NeutralRegion<'_>,
        ) -> Result<Eligibility, Render2dExecutionError> {
            builder.charge_preparation(1024, &node.path)?;
            Ok(match &node.kind {
                Kind::Source(_) => {
                    let Some(item) = node.semantic_item else {
                        return Ok(Eligibility {
                            polygon_possible: false,
                            requires_original: false,
                            supported_boundaries: false,
                        });
                    };
                    let supported = match item.primitive() {
                        crate::composition_2d::Render2dPrimitive::Fill {
                            shape: source, ..
                        } => shape(builder, source, &node.path)?,
                        crate::composition_2d::Render2dPrimitive::Image(image) => {
                            builder
                                .charge_preparation(64 + image.patches().len() * 32, &node.path)?;
                            true
                        }
                        _ => false,
                    } && clips(builder, item.clips(), &node.path)?;
                    Eligibility {
                        polygon_possible: NeutralRegionBuilder::original_polygon_source_possible(
                            item,
                        ),
                        requires_original: false,
                        supported_boundaries: supported,
                    }
                }
                Kind::Union(children) => {
                    builder.charge_preparation(children.len() * 64, &node.path)?;
                    let mut result = Eligibility {
                        polygon_possible: true,
                        requires_original: false,
                        supported_boundaries: true,
                    };
                    for child in children {
                        let child = visit(builder, child)?;
                        result.polygon_possible &= child.polygon_possible;
                        result.requires_original |= child.requires_original;
                        result.supported_boundaries &= child.supported_boundaries;
                    }
                    result
                }
                Kind::Affine { source, .. } => visit(builder, source)?,
                Kind::Clip {
                    source,
                    clips: constraints,
                } => {
                    let child = visit(builder, source)?;
                    Eligibility {
                        polygon_possible: child.polygon_possible && constraints.is_empty(),
                        requires_original: child.requires_original,
                        supported_boundaries: child.supported_boundaries
                            && clips(builder, constraints, &node.path)?,
                    }
                }
                Kind::Effect { source, shadow } => {
                    let child = visit(builder, source)?;
                    let empty = certified_complete_disk_erosion(source.envelope, shadow.spread());
                    let blocked = shadow.spread() < 0.0 && !child.polygon_possible && !empty;
                    Eligibility {
                        polygon_possible: false,
                        requires_original: !empty && (child.requires_original || blocked),
                        supported_boundaries: child.supported_boundaries
                            && (shadow.spread() < 0.0
                                || shadow.sigma() == 0.0 && shadow.spread() == 0.0),
                    }
                }
            })
        }
        let result = visit(self, root)?;
        Ok(result.requires_original && result.supported_boundaries)
    }

    pub(in crate::runtime::execution_2d::shadow) fn prepare_gaussian_caster(
        &mut self,
        source: &NeutralRegion<'_>,
        effects: &[Render2dDropShadow],
        target: GaussianTarget,
        path: &[usize],
    ) -> Result<PreparedGaussianCaster, Render2dExecutionError> {
        self.charge_preparation(1024 + effects.len() * 64, path)?;
        let extent = target.physical_extent.map(|value| value.checked_mul(4));
        let [Some(width), Some(height)] = extent else {
            return Err(resource(path, "Gaussian target extent overflow"));
        };
        let scale = target.raster_scale * 4.0;
        if !scale.is_finite() || scale <= 0.0 {
            return Err(precision(path, "Gaussian sample scale invalid"));
        }
        let mut result = PreparedGaussianCaster {
            planes: Vec::new(),
            effect_planes: Vec::new(),
        };
        let mut keys = Vec::<[u64; 3]>::new();
        self.host
            .reserve_vector(&mut result.planes, effects.len(), path)?;
        self.host
            .reserve_vector(&mut result.effect_planes, effects.len(), path)?;
        self.host.reserve_vector(&mut keys, effects.len(), path)?;
        for effect in effects {
            self.charge_preparation(2048 + keys.len() * 16, path)?;
            let Some(envelope) = source.certified_logical_envelope() else {
                return Err(unsupported(path));
            };
            let logical_envelope = nested_effect_envelope(envelope, *effect, path)?;
            let mut root = [0_u32; 4];
            for axis in 0..2 {
                let end = if axis == 0 { width } else { height };
                let lower = interval::Interval::scalar(logical_envelope[axis])
                    .and_then(|x| x.mul(interval::Interval::scalar(scale)?))
                    .ok_or_else(|| precision(path, "Gaussian root window lower bound invalid"))?
                    .lo
                    .floor();
                let upper = interval::Interval::scalar(logical_envelope[axis + 2])
                    .and_then(|x| x.mul(interval::Interval::scalar(scale)?))
                    .ok_or_else(|| precision(path, "Gaussian root window upper bound invalid"))?
                    .hi
                    .ceil();
                root[axis] = lower.clamp(0.0, f64::from(end)) as u32;
                root[axis + 2] = upper.clamp(0.0, f64::from(end)) as u32;
            }
            if root[0] >= root[2] || root[1] >= root[3] {
                result.effect_planes.push(None);
                continue;
            }
            let key = [
                effect.sigma().to_bits(),
                effect.offset_x().to_bits(),
                effect.offset_y().to_bits(),
            ];
            if let Some(index) = keys.iter().position(|previous| *previous == key) {
                result.effect_planes.push(Some(index));
                continue;
            }
            let request = GaussianPlaneRequest {
                root,
                scale,
                sigma: effect.sigma(),
                offset: [effect.offset_x(), effect.offset_y()],
                parent_to_root: Render2dAffineTransform::IDENTITY,
            };
            let peak = request
                .preparation_bytes()
                .ok_or_else(|| resource(path, "Gaussian numerical peak overflow"))?;
            self.host.reserve(peak, path)?;
            let mut fuel = MAX_REGION_GRAPH_WORK
                .checked_sub(self.graph_work)
                .ok_or_else(|| resource(path, "Gaussian source work exceeds admission"))?;
            let plane = source.prepare_gaussian_plane(request, &mut fuel, peak);
            self.graph_work = MAX_REGION_GRAPH_WORK - fuel;
            let plane = plane.map_err(|error| match error {
                AreaQueryError::Unsupported => unsupported(path),
                AreaQueryError::Resource => resource(
                    path,
                    "Gaussian source preparation exceeds invocation resources",
                ),
                AreaQueryError::Precision => {
                    precision(path, "Gaussian source preparation cannot certify precision")
                }
            })?;
            let retained = plane
                .retained_bytes()
                .ok_or_else(|| resource(path, "Gaussian plane inventory overflow"))?;
            self.host.verify_reserved_capacity(retained, peak, path)?;
            self.host.release(peak - retained);
            result.effect_planes.push(Some(result.planes.len()));
            keys.push(key);
            result.planes.push(PreparedGaussianField {
                plane,
                logical_envelope,
            });
        }
        let key_bytes = keys.capacity() * size_of::<[u64; 3]>();
        drop(keys);
        self.host.release(key_bytes);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::*;
    use crate::runtime::execution_2d::support::PreparationWork;
    use std::collections::BTreeMap;
    fn builder() -> NeutralRegionBuilder {
        NeutralRegionBuilder::new(
            shadow::NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            64 * 1024 * 1024,
        )
    }
    fn item(shape: Render2dShape) -> Render2dItem {
        Render2dItem::new(
            Render2dPrimitive::Fill {
                shape,
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        )
    }
    fn ring() -> Render2dItem {
        let point = |x, y| Render2dPoint::new(x, y).unwrap();
        let mut commands = Vec::new();
        for [x0, y0, x1, y1] in [[10.0, 10.0, 30.0, 30.0], [18.0, 18.0, 22.0, 22.0]] {
            commands.extend([
                Render2dPathCommand::MoveTo(point(x0, y0)),
                Render2dPathCommand::LineTo(point(x1, y0)),
                Render2dPathCommand::LineTo(point(x1, y1)),
                Render2dPathCommand::LineTo(point(x0, y1)),
                Render2dPathCommand::Close,
            ]);
        }
        item(Render2dShape::path(
            Render2dPath::new(Render2dFillRule::EvenOdd, commands).unwrap(),
        ))
    }
    #[test]
    fn selector_preserves_plain_gaussian_classes_and_requires_a_retained_blocked_effect() {
        let rect = item(Render2dShape::rect(
            Render2dRect::new(0.0, 0.0, 4.0, 4.0).unwrap(),
        ));
        let ellipse = item(Render2dShape::ellipse(
            Render2dRect::new(0.0, 0.0, 4.0, 4.0).unwrap(),
        ));
        let compound = ring();
        let stroke = Render2dItem::new(
            Render2dPrimitive::Stroke {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 4.0, 4.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::BLACK),
                style: Render2dStrokeStyle::new(
                    1.0,
                    Render2dStrokeCap::Butt,
                    Render2dStrokeJoin::Miter,
                    4.0,
                )
                .unwrap(),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::OPAQUE,
        );
        let (_, _, glyph) = super::super::tests::glyph_source_fixture(2, 1);
        for input in [&rect, &ellipse, &compound, &stroke, &glyph] {
            let mut graph = builder();
            let source = graph.register_source(input, &[0, 0]).unwrap().unwrap();
            assert!(!graph.original_gaussian_required(&source).unwrap());
        }
        let mut graph = builder();
        let source = graph.register_source(&compound, &[0, 0]).unwrap().unwrap();
        let negative =
            Render2dDropShadow::new(8.0, 0.0, 0.03125, -1.0, Render2dColorRgba8::TRANSPARENT)
                .unwrap();
        let effect = graph
            .effect(Arc::clone(&source), negative, &[0, 0])
            .unwrap();
        assert!(graph.original_gaussian_required(&effect).unwrap());
        let erased =
            Render2dDropShadow::new(8.0, 0.0, 0.03125, -1000.0, Render2dColorRgba8::TRANSPARENT)
                .unwrap();
        let empty = graph.effect(source, erased, &[0, 0]).unwrap();
        assert!(!graph.original_gaussian_required(&empty).unwrap());
        let source = graph.register_source(&ellipse, &[0, 0]).unwrap().unwrap();
        let effect = graph.effect(source, negative, &[0, 0]).unwrap();
        assert!(!graph.original_gaussian_required(&effect).unwrap());
    }
    #[test]
    fn selector_rejects_work_before_visiting_sources_and_numerical_peak_before_allocation() {
        let input = ring();
        let mut graph = builder();
        let source = graph.register_source(&input, &[0, 0]).unwrap().unwrap();
        graph.graph_work = MAX_REGION_GRAPH_WORK;
        assert!(graph.original_gaussian_required(&source).is_err());
        let mut graph = builder();
        let source = graph.register_source(&input, &[0, 0]).unwrap().unwrap();
        let live = graph.host.live();
        graph
            .host
            .reserve(support::MAX_SHADOW_HOST_BYTES - live - 4096, &[0])
            .unwrap();
        let effect =
            Render2dDropShadow::new(0.0, 0.0, 0.1, 0.0, Render2dColorRgba8::BLACK).unwrap();
        assert!(
            graph
                .prepare_gaussian_caster(
                    &source,
                    &[effect],
                    GaussianTarget {
                        canvas: [64.0; 2],
                        raster_scale: 1.0,
                        physical_extent: [64; 2]
                    },
                    &[0]
                )
                .is_err()
        );
        assert!(graph.host.peak() <= support::MAX_SHADOW_HOST_BYTES);
    }
    #[test]
    fn numerical_owner_preserves_exact_and_one_short_invocation_work() {
        let input = item(Render2dShape::rect(
            Render2dRect::new(-10.0, -10.0, 20.0, 20.0).unwrap(),
        ));
        let effect =
            Render2dDropShadow::new(0.0, 0.0, 0.1, 0.0, Render2dColorRgba8::BLACK).unwrap();
        let target = GaussianTarget {
            canvas: [1.0; 2],
            raster_scale: 1.0,
            physical_extent: [1; 2],
        };
        let mut baseline = builder();
        let source = baseline.register_source(&input, &[0, 0]).unwrap().unwrap();
        let before = baseline.graph_work;
        let caster = baseline
            .prepare_gaussian_caster(&source, &[effect], target, &[0])
            .unwrap();
        let cost = baseline.graph_work - before;
        assert_eq!(caster.planes.len(), 1);
        assert_eq!(caster.planes[0].plane.query_data().1, 0);
        for short in [false, true] {
            let mut graph = builder();
            let source = graph.register_source(&input, &[0, 0]).unwrap().unwrap();
            graph.graph_work = MAX_REGION_GRAPH_WORK - cost + usize::from(short);
            let prepared = graph.prepare_gaussian_caster(&source, &[effect], target, &[0]);
            assert_eq!(prepared.is_err(), short);
            assert_eq!(graph.graph_work, MAX_REGION_GRAPH_WORK);
        }
    }

    #[test]
    fn original_gaussian_consumer_prepares_once_for_distinct_colors_without_polygon_factories() {
        let child = Render2dGroup::new(
            [Render2dEntry::item(ring())],
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::OPAQUE,
            vec![
                Render2dDropShadow::new(8.0, 0.0, 0.03125, -1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
            ],
        );
        let effects = [
            Render2dColorRgba8::new(255, 0, 0, 255),
            Render2dColorRgba8::new(0, 0, 255, 128),
        ]
        .map(|color| Render2dDropShadow::new(0.0, 0.0, 0.1, 0.0, color).unwrap());
        let root = Render2dGroup::new(
            [Render2dEntry::group(child)],
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::OPAQUE,
            effects.to_vec(),
        );
        let composition = Render2dComposition::new([Render2dEntry::group(root)]).unwrap();
        let bindings = Render2dResourceBindings::default();
        let fields = BTreeMap::new();
        let mut preparation = shadow::SourcePreparation::new(
            shadow::NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            64 * 1024 * 1024,
        );
        assert_eq!(preparation.admit_effects(&composition).unwrap(), 3);
        let dag =
            shadow::build_group_sources(&composition, 0, &bindings, &fields, preparation).unwrap();
        let target = GaussianTarget {
            canvas: [64.0; 2],
            raster_scale: 1.0,
            physical_extent: [64; 2],
        };
        let sources = dag.prepare_consumers(target).unwrap();
        let shadow::NeutralCaster::Gaussian(caster) = sources.caster(0).unwrap() else {
            panic!("root must use original numerical source");
        };
        assert_eq!(caster.planes.len(), 1);
        assert_eq!(caster.effect_planes, [Some(0), Some(0)]);
        assert!(matches!(
            sources.caster(1),
            Some(shadow::NeutralCaster::Symbolic(_))
        ));
        assert!(sources.work.remaining(usize::MAX) > 0);
        assert!(sources.host.peak() <= support::MAX_SHADOW_HOST_BYTES);
        println!(
            "ORIGINAL_GAUSSIAN_SHARED used={} remaining={} retained={} host_peak={}",
            sources.work.used(),
            sources.work.remaining(usize::MAX),
            caster.planes[0].plane.retained_bytes().unwrap(),
            sources.host.peak()
        );
        println!(
            "ORIGINAL_GAUSSIAN_QUERY count={} node_visits={} samples={} root={:?}",
            caster.planes[0].plane.query_data().1,
            caster.planes[0].plane.query_node_visits(),
            caster.planes[0].plane.samples().len(),
            caster.planes[0].plane.root()
        );
        // Neither selected source nor invisible child's source requested a
        // physical realization. The Symbolic leaf still owns original F1.
        let shadow::NeutralCaster::Symbolic(child) = sources.caster(1).unwrap() else {
            unreachable!()
        };
        fn unmaterialized(node: &NeutralRegion<'_>) {
            match &node.kind {
                Kind::Source(source) => assert!(source.get().is_none()),
                Kind::Union(children) => {
                    for child in children {
                        unmaterialized(child);
                    }
                }
                Kind::Affine { source, .. }
                | Kind::Clip { source, .. }
                | Kind::Effect { source, .. } => unmaterialized(source),
            }
        }
        unmaterialized(&child.expression);
    }
}
