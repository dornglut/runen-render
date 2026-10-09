//! Renderer-private continuous neutral geometry graph for F3F group effects.
//!
//! Only F1's borrowed painter events and immutable font/image bindings own
//! meaning. This module materializes bounded group-parent geometry *before*
//! ancestor affines, non-semantic paint alpha and target culling. Every
//! owner clip is a geometric intersection, not a source color or an AABB.

use super::{
    field::{FieldSetKey, QualityTier, ResourceFields},
    image, scene,
    support::NeutralMesh,
    vector,
};
use crate::composition_2d::{
    Render2dAffineTransform, Render2dBrush, Render2dClip, Render2dColorRgba8, Render2dDropShadow,
    Render2dItem, Render2dOpacity, Render2dPrimitive, Render2dResourceBindings,
    Render2dResourceValue,
};
use crate::execution_2d::{
    Render2dExecutionError, Render2dSampleSpaceError, Render2dUnsupportedContent,
};
use std::{collections::BTreeMap, sync::Arc};

const MAX_NEUTRAL_VERTICES: usize = 1_048_576;
const MAX_NEUTRAL_CLIP_WORK: usize = 16_777_216;

fn failed(
    path: &[usize],
    kind: Render2dSampleSpaceError,
    detail: &'static str,
) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    }
}

fn precision(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    failed(path, Render2dSampleSpaceError::PrecisionLimit, detail)
}

fn resource(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    failed(path, Render2dSampleSpaceError::ResourceLimit, detail)
}

/// Per-invocation retained structural materialization budget. Counting
/// intermediate copies prevents a deep sequence of individually bounded
/// nested groups from retaining an unbounded aggregate working set.
fn charge_vertices(
    used: &mut usize,
    count: usize,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    let next = used
        .checked_add(count)
        .ok_or_else(|| resource(path, "neutral source graph vertex count overflow"))?;
    if next > MAX_NEUTRAL_VERTICES {
        return Err(resource(
            path,
            "neutral source graph exceeds aggregate geometry budget",
        ));
    }
    *used = next;
    Ok(())
}

struct SourceFrame<'a> {
    begin_event: usize,
    path: Vec<usize>,
    group: &'a crate::composition_2d::Render2dGroup,
    /// All children are in this group's LOCAL frame, without opacity.
    child_source: NeutralMesh,
}

fn neutral_item(
    item: &Render2dItem,
    path: &[usize],
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    resolution: f64,
    max_buffer_bytes: u64,
) -> Result<Option<NeutralMesh>, Render2dExecutionError> {
    let root_index = path[0];
    let result = match item.primitive() {
        Render2dPrimitive::Fill { .. } | Render2dPrimitive::Stroke { .. } => {
            vector::neutral_support(item, root_index, resolution, max_buffer_bytes)?
        }
        Render2dPrimitive::Image(primitive) => {
            image::neutral_support(item, primitive, root_index, resolution, max_buffer_bytes)?
        }
        Render2dPrimitive::ShapedText(primitive) => {
            let value = bindings
                .get(primitive.resource_id())
                .ok_or_else(|| precision(path, "missing validated shaped-text binding"))?;
            let Render2dResourceValue::ShapedText(resource) = value else {
                return Err(precision(path, "shaped-text binding kind changed"));
            };
            let quality = QualityTier::select(resource.font_size(), resolution)
                .ok_or_else(|| precision(path, "shaped-text quality is not representable"))?;
            let key = FieldSetKey::new(primitive.resource_id(), quality);
            let fields = field_sets
                .get(&key)
                .ok_or_else(|| precision(path, "missing validated immutable font outlines"))?;
            let mut result = empty_mesh();
            for glyph in resource.glyphs() {
                let Some(field) = fields.glyph(glyph.id()) else {
                    // An admitted empty glyph outline has no neutral area.
                    continue;
                };
                let origin = [
                    primitive.origin().x() + glyph.x(),
                    primitive.origin().y() + glyph.y(),
                ];
                let instance = vector::NeutralGlyphInstance {
                    field,
                    resource_id: primitive.resource_id(),
                    origin,
                    font_size: resource.font_size(),
                    to_parent: item.local_to_parent(),
                };
                if let Some(mesh) =
                    vector::neutral_shaped_glyph(instance, path, resolution, max_buffer_bytes)?
                {
                    append(&mut result, &mesh, Render2dAffineTransform::IDENTITY, path)?;
                }
            }
            (!result.triangles.is_empty()).then_some(result)
        }
    };
    let Some(mesh) = result else {
        return Ok(None);
    };
    let clipped = intersect_clips(
        &mesh,
        item.clips(),
        root_index,
        path,
        resolution,
        max_buffer_bytes,
    )?;
    Ok((!clipped.triangles.is_empty()).then_some(clipped))
}

/// Derives F1 group child geometry in each attached group's immediate-parent
/// frame, with no root flattening, pixel-alpha sampling or final canvas cull.
///
/// The result identifies each SHADOW-BEARING group's one pre-shadow C source
/// by its BeginGroup event index. A shadow-bearing nested group cannot yet
/// be propagated to an ancestor: returning an explicit unsupported error is
/// mandatory until its own expanded effect geometry is representable. This
/// method never substitutes the child's original geometry for that effect.
pub(super) fn group_child_sources(
    plan: &scene::Plan<'_>,
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    resolution: f64,
    max_buffer_bytes: u64,
) -> Result<BTreeMap<usize, NeutralMesh>, Render2dExecutionError> {
    let mut active = Vec::<SourceFrame<'_>>::new();
    let mut sources = BTreeMap::new();
    let mut retained_vertices = 0_usize;
    for (index, event) in plan.events.iter().enumerate() {
        match event {
            scene::Event::BeginGroup { path, group, .. } => {
                active.push(SourceFrame {
                    begin_event: index,
                    path: path.clone(),
                    group,
                    child_source: empty_mesh(),
                });
            }
            scene::Event::Item { path, item, .. } => {
                if active.is_empty() {
                    continue;
                }
                if let Some(mesh) = neutral_item(
                    item,
                    path,
                    bindings,
                    field_sets,
                    resolution,
                    max_buffer_bytes,
                )? {
                    charge_vertices(&mut retained_vertices, mesh.triangles.len(), path)?;
                    append(
                        &mut active.last_mut().expect("active source group").child_source,
                        &mesh,
                        Render2dAffineTransform::IDENTITY,
                        path,
                    )?;
                }
            }
            scene::Event::EndGroup => {
                let frame = active.pop().expect("F1 balanced group plan");
                let mut in_parent = empty_mesh();
                append(
                    &mut in_parent,
                    &frame.child_source,
                    frame.group.local_to_parent(),
                    &frame.path,
                )?;
                charge_vertices(
                    &mut retained_vertices,
                    in_parent.triangles.len(),
                    &frame.path,
                )?;
                if !frame.group.shadows().is_empty() {
                    // C is before this group's own clips; those must later
                    // clip completed shadow+children output exactly once.
                    if !active.is_empty() {
                        return Err(super::unsupported_at(
                            &frame.path,
                            Render2dUnsupportedContent::Shadows {
                                root_index: frame.path[0],
                            },
                        ));
                    }
                    sources.insert(frame.begin_event, in_parent);
                    continue;
                }
                let clipped = intersect_clips(
                    &in_parent,
                    frame.group.clips(),
                    frame.path[0],
                    &frame.path,
                    resolution,
                    max_buffer_bytes,
                )?;
                if let Some(parent) = active.last_mut() {
                    charge_vertices(&mut retained_vertices, clipped.triangles.len(), &frame.path)?;
                    append(
                        &mut parent.child_source,
                        &clipped,
                        Render2dAffineTransform::IDENTITY,
                        &frame.path,
                    )?;
                }
            }
        }
    }
    debug_assert!(active.is_empty());
    Ok(sources)
}

/// A literal empty mesh never introduces a fake shadow caster.
pub(super) fn empty_mesh() -> NeutralMesh {
    NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: Vec::new(),
        bounds: [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
    }
}

/// Finite transform in one owning immediate-parent frame. The source's
/// disposable tessellation scale cancels *before* applying this affine.
pub(super) fn append(
    output: &mut NeutralMesh,
    source: &NeutralMesh,
    source_to_parent: Render2dAffineTransform,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    if source.triangles.is_empty() {
        return Ok(());
    }
    if !source.triangles.len().is_multiple_of(3)
        || !source.units_per_parent_logical_unit.is_finite()
        || source.units_per_parent_logical_unit <= 0.0
    {
        return Err(precision(path, "invalid neutral mesh source or scale"));
    }
    let required = output
        .triangles
        .len()
        .checked_add(source.triangles.len())
        .ok_or_else(|| resource(path, "neutral geometry vertex count overflow"))?;
    if required > MAX_NEUTRAL_VERTICES {
        return Err(resource(
            path,
            "neutral geometry exceeds bounded vertex admission",
        ));
    }
    let [a, b, c, d, tx, ty] = source_to_parent.components();
    let scale = source.units_per_parent_logical_unit;
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(source.triangles.len())
        .map_err(|_| resource(path, "neutral geometry allocation failed"))?;
    for &[x, y] in &source.triangles {
        let x = x / scale;
        let y = y / scale;
        let transformed = [
            a.mul_add(x, c.mul_add(y, tx)),
            b.mul_add(x, d.mul_add(y, ty)),
        ];
        if !transformed.iter().all(|v| v.is_finite()) {
            return Err(precision(
                path,
                "neutral parent-space transform lost precision",
            ));
        }
        prepared.push(transformed);
    }
    // Two-dimensional rank-deficient transforms have zero area and cannot
    // create a false positive-area source for finite ordinary shadows.
    output
        .triangles
        .try_reserve(prepared.len())
        .map_err(|_| resource(path, "neutral destination geometry allocation failed"))?;
    for triangle in prepared.as_chunks::<3>().0 {
        if orient(triangle[0], triangle[1], triangle[2]) == 0.0 {
            continue;
        }
        for &point in triangle {
            output.bounds[0] = output.bounds[0].min(point[0]);
            output.bounds[1] = output.bounds[1].min(point[1]);
            output.bounds[2] = output.bounds[2].max(point[0]);
            output.bounds[3] = output.bounds[3].max(point[1]);
            output.triangles.push(point);
        }
    }
    Ok(())
}

fn orient(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]).mul_add(p[1] - a[1], -(b[1] - a[1]) * (p[0] - a[0]))
}

/// A *conservative extent*, never geometric shadow membership. It bounds
/// positive Euclidean spread and finite 3-sigma blur before ancestor affines.
/// Negative erosion may empty the support; its source extent is retained only
/// as a safe overestimate for tile/work admission.
pub(super) fn shadow_envelope(
    source: &NeutralMesh,
    effect: Render2dDropShadow,
    path: &[usize],
) -> Result<Option<[f64; 4]>, Render2dExecutionError> {
    if source.triangles.is_empty() {
        return Ok(None);
    }
    let scale = source.units_per_parent_logical_unit;
    if !scale.is_finite() || scale <= 0.0 {
        return Err(precision(path, "invalid neutral geometry frame scale"));
    }
    let padding = effect.spread().max(0.0) + 3.0 * effect.sigma();
    let bounds = [
        source.bounds[0] / scale + effect.offset_x() - padding,
        source.bounds[1] / scale + effect.offset_y() - padding,
        source.bounds[2] / scale + effect.offset_x() + padding,
        source.bounds[3] / scale + effect.offset_y() + padding,
    ];
    if !padding.is_finite()
        || !bounds.iter().all(|v| v.is_finite())
        || bounds[0] > bounds[2]
        || bounds[1] > bounds[3]
    {
        return Err(precision(
            path,
            "shadow parent-frame envelope is not representable",
        ));
    }
    Ok(Some(bounds))
}

/// Project a previously completed effect's bounding extent through an
/// ancestor affine. Never apply effect kernels to this projected AABB.
pub(super) fn transform_envelope(
    bounds: [f64; 4],
    ancestor: Render2dAffineTransform,
    path: &[usize],
) -> Result<[f64; 4], Render2dExecutionError> {
    let [a, b, c, d, tx, ty] = ancestor.components();
    let mut result = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for x in [bounds[0], bounds[2]] {
        for y in [bounds[1], bounds[3]] {
            let point = [
                a.mul_add(x, c.mul_add(y, tx)),
                b.mul_add(x, d.mul_add(y, ty)),
            ];
            if !point.iter().all(|v| v.is_finite()) {
                return Err(precision(
                    path,
                    "ancestor shadow envelope is not representable",
                ));
            }
            result[0] = result[0].min(point[0]);
            result[1] = result[1].min(point[1]);
            result[2] = result[2].max(point[0]);
            result[3] = result[3].max(point[1]);
        }
    }
    Ok(result)
}

/// The intersection of two *convex triangles*, without a rectangular/alpha
/// approximation. Clipping against a polygon-union clip means retaining each
/// triangle intersection's union (overlaps are resolved by area coverage).
fn triangle_intersection(
    source: [[f64; 2]; 3],
    clip: [[f64; 2]; 3],
    path: &[usize],
) -> Result<Vec<[f64; 2]>, Render2dExecutionError> {
    let winding = orient(clip[0], clip[1], clip[2]);
    if winding == 0.0 || orient(source[0], source[1], source[2]) == 0.0 {
        return Ok(Vec::new());
    }
    let mut polygon = source.to_vec();
    for i in 0..3 {
        let a = clip[i];
        let b = clip[(i + 1) % 3];
        let mut result = Vec::new();
        result
            .try_reserve_exact(9)
            .map_err(|_| resource(path, "neutral polygon intersection allocation failed"))?;
        for j in 0..polygon.len() {
            let p = polygon[j];
            let q = polygon[(j + 1) % polygon.len()];
            let d0 = orient(a, b, p) * winding.signum();
            let d1 = orient(a, b, q) * winding.signum();
            let inside_p = d0 >= 0.0;
            let inside_q = d1 >= 0.0;
            if inside_p != inside_q {
                let denominator = d0 - d1;
                if denominator == 0.0 {
                    return Err(precision(
                        path,
                        "neutral clip edge intersection is singular",
                    ));
                }
                let t = d0 / denominator;
                let intersection = [
                    (q[0] - p[0]).mul_add(t, p[0]),
                    (q[1] - p[1]).mul_add(t, p[1]),
                ];
                if !intersection.iter().all(|x| x.is_finite()) {
                    return Err(precision(path, "neutral clip intersection lost precision"));
                }
                result.push(intersection);
            }
            if inside_q {
                result.push(q);
            }
        }
        polygon = result;
        if polygon.len() < 3 {
            break;
        }
    }
    Ok(polygon)
}

/// Applies every structural clip once in its owner's parent logical frame.
/// Each clip mesh is a union of fill-rule tessellation triangles; successive
/// clips intersect conjunctively, independent of all source paint alpha.
pub(super) fn intersect_clips(
    source: &NeutralMesh,
    clips: &[Render2dClip],
    root_index: usize,
    path: &[usize],
    resolution: f64,
    max_buffer_bytes: u64,
) -> Result<NeutralMesh, Render2dExecutionError> {
    if clips.is_empty() || source.triangles.is_empty() {
        let mut result = empty_mesh();
        append(&mut result, source, Render2dAffineTransform::IDENTITY, path)?;
        return Ok(result);
    }
    let mut current = empty_mesh();
    append(
        &mut current,
        source,
        Render2dAffineTransform::IDENTITY,
        path,
    )?;
    let mut work = 0_usize;
    for clip in clips {
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: clip.shape().clone(),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            clip.clip_to_parent(),
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let Some(clip_mesh) =
            vector::neutral_support(&item, root_index, resolution, max_buffer_bytes)?
        else {
            return Ok(empty_mesh());
        };
        let mut clip_local = empty_mesh();
        append(
            &mut clip_local,
            &clip_mesh,
            Render2dAffineTransform::IDENTITY,
            path,
        )?;
        let mut next = empty_mesh();
        let cost = (current.triangles.len() / 3)
            .checked_mul(clip_local.triangles.len() / 3)
            .ok_or_else(|| resource(path, "neutral conjunctive clip work overflow"))?;
        work = work
            .checked_add(cost)
            .ok_or_else(|| resource(path, "neutral clip accumulation overflow"))?;
        if work > MAX_NEUTRAL_CLIP_WORK {
            return Err(resource(
                path,
                "neutral clip triangle intersection work exceeds admission",
            ));
        }
        for source in current.triangles.as_chunks::<3>().0 {
            for clipping in clip_local.triangles.as_chunks::<3>().0 {
                let polygon = triangle_intersection(
                    [source[0], source[1], source[2]],
                    [clipping[0], clipping[1], clipping[2]],
                    path,
                )?;
                if polygon.len() < 3 {
                    continue;
                }
                for index in 1..polygon.len() - 1 {
                    let triangle = [polygon[0], polygon[index], polygon[index + 1]];
                    if orient(triangle[0], triangle[1], triangle[2]) == 0.0 {
                        continue;
                    }
                    append(
                        &mut next,
                        &NeutralMesh {
                            units_per_parent_logical_unit: 1.0,
                            bounds: [0.0; 4],
                            triangles: triangle.to_vec(),
                        },
                        Render2dAffineTransform::IDENTITY,
                        path,
                    )?;
                }
            }
        }
        current = next;
        if current.triangles.is_empty() {
            break;
        }
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::{Render2dRect, Render2dShape};

    #[test]
    fn neutral_geometry_transforms_before_ancestor_effects_and_keeps_disjoint_sources() {
        let source = NeutralMesh {
            units_per_parent_logical_unit: 4.0,
            triangles: vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]],
            bounds: [0.0, 0.0, 4.0, 4.0],
        };
        let mut union = empty_mesh();
        let parent = Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, -3.0, 5.0).unwrap();
        append(&mut union, &source, parent, &[1, 3]).unwrap();
        assert_eq!(union.triangles, vec![[-3.0, 5.0], [-1.0, 5.0], [-3.0, 6.0]]);
        append(&mut union, &source, parent, &[1, 4]).unwrap();
        assert_eq!(union.triangles.len(), 6);
    }

    #[test]
    fn parent_effect_precedes_ancestor_affine_even_under_shear_and_anisotropy() {
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            bounds: [0.0, 0.0, 1.0, 1.0],
        };
        let mut parent = empty_mesh();
        append(
            &mut parent,
            &source,
            Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap(),
            &[2, 0],
        )
        .unwrap();
        let effect =
            Render2dDropShadow::new(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let envelope = shadow_envelope(&parent, effect, &[2]).unwrap().unwrap();
        assert_eq!(envelope, [-1.0, -1.0, 3.0, 2.0]);
        let ancestor = Render2dAffineTransform::new(2.0, 0.0, 1.0, 1.0, 0.0, 0.0).unwrap();
        assert_eq!(
            transform_envelope(envelope, ancestor, &[2]).unwrap(),
            [-3.0, -1.0, 8.0, 2.0]
        );
    }

    #[test]
    fn negative_spread_envelope_remains_conservative_and_overflow_is_typed() {
        let source = NeutralMesh {
            units_per_parent_logical_unit: 4.0,
            triangles: vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]],
            bounds: [0.0, 0.0, 4.0, 4.0],
        };
        let effect =
            Render2dDropShadow::new(2.0, -3.0, 0.5, -5.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        assert_eq!(
            shadow_envelope(&source, effect, &[0]).unwrap().unwrap(),
            [0.5, -4.5, 4.5, -0.5]
        );
        let extreme =
            Render2dDropShadow::new(0.0, 0.0, f64::MAX, 0.0, Render2dColorRgba8::TRANSPARENT)
                .unwrap();
        assert!(matches!(
            shadow_envelope(&source, extreme, &[0, 7]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [0, 7]
        ));
    }

    #[test]
    fn nested_group_source_is_in_the_immediate_parent_frame_and_pre_group_clip() {
        use crate::composition_2d::{
            Render2dComposition, Render2dEntry, Render2dGroup, Render2dRect, Render2dShape,
        };
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 1.0, 1.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![Render2dClip::new(
                Render2dShape::rect(Render2dRect::new(0.25, 0.0, 1.0, 1.0).unwrap()),
                Render2dAffineTransform::IDENTITY,
            )],
            Render2dOpacity::TRANSPARENT,
        );
        let inner = Render2dGroup::new(
            vec![Render2dEntry::item(item)],
            Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap(),
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            Vec::new(),
        );
        let effect =
            Render2dDropShadow::new(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let root = Render2dGroup::new(
            vec![Render2dEntry::group(inner)],
            Render2dAffineTransform::IDENTITY,
            vec![Render2dClip::new(
                Render2dShape::rect(Render2dRect::new(1.5, 0.0, 1.0, 1.0).unwrap()),
                Render2dAffineTransform::IDENTITY,
            )],
            Render2dOpacity::TRANSPARENT,
            vec![effect],
        );
        let composition = Render2dComposition::new(vec![Render2dEntry::group(root)]).unwrap();
        let plan = scene::analyze(&composition).unwrap();
        let empty = Render2dResourceBindings::new(Vec::new()).unwrap();
        let sources = group_child_sources(&plan, &empty, &BTreeMap::new(), 4.0, 1_048_576).unwrap();
        assert_eq!(sources.len(), 1);
        let caster = sources.get(&0).unwrap();
        for (actual, expected) in caster.bounds.iter().zip([0.5, 0.0, 2.0, 1.0]) {
            assert!((*actual - expected).abs() < 1.0e-6);
        }
        assert_eq!(
            shadow_envelope(caster, effect, &[0]).unwrap().unwrap(),
            [-0.5, -1.0, 3.0, 2.0]
        );
    }

    #[test]
    fn nested_effect_support_is_not_silently_replaced_with_child_geometry() {
        use crate::composition_2d::{Render2dComposition, Render2dEntry, Render2dGroup};
        let effect =
            Render2dDropShadow::new(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let nested = Render2dGroup::new(
            Vec::new(),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            vec![effect],
        );
        let parent = Render2dGroup::new(
            vec![Render2dEntry::group(nested)],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
            vec![effect],
        );
        let composition =
            Render2dComposition::new(vec![Render2dEntry::group(parent)]).unwrap();
        let plan = scene::analyze(&composition).unwrap();
        assert!(matches!(
            group_child_sources(
                &plan,
                &Render2dResourceBindings::default(),
                &BTreeMap::new(),
                4.0,
                1_048_576,
            ),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 0 },
            }) if path == [0, 0]
        ));
    }

    #[test]
    fn offscreen_parent_geometry_can_reenter_canvas_after_ancestor_translation() {
        let triangle = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            bounds: [0.0, 0.0, 1.0, 1.0],
        };
        let mut translated = empty_mesh();
        append(
            &mut translated,
            &triangle,
            Render2dAffineTransform::translation(1.0e10, 0.0).unwrap(),
            &[3],
        )
        .unwrap();
        assert_eq!(translated.bounds, [1.0e10, 0.0, 1.0e10 + 1.0, 1.0]);
        let root = transform_envelope(
            translated.bounds,
            Render2dAffineTransform::translation(-1.0e10, 0.0).unwrap(),
            &[3],
        )
        .unwrap();
        assert_eq!(root, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn convex_polygon_intersection_obeys_actual_geometry_not_bbox() {
        let clip = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let disjoint_corner = [[1.6, 1.6], [2.2, 1.6], [1.6, 2.2]];
        assert!(
            triangle_intersection(disjoint_corner, clip, &[1])
                .unwrap()
                .len()
                < 3
        );
        let crossing = [[0.8, 0.8], [1.6, 0.8], [0.8, 1.6]];
        let clipped = triangle_intersection(crossing, clip, &[1]).unwrap();
        assert!(clipped.len() >= 3);
        for point in clipped {
            assert!(point[0] >= 0.0 && point[1] >= 0.0);
            assert!(point[0] + point[1] <= 2.0 + 1.0e-12);
        }
    }

    #[test]
    fn transparent_parent_space_shape_and_clip_retain_only_intersection() {
        let support = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 3.0, 3.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let mesh = vector::neutral_support(&support, 1, 4.0, 1_048_576)
            .unwrap()
            .unwrap();
        let clip = Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(2.0, 1.0, 2.0, 2.0).unwrap()),
            Render2dAffineTransform::IDENTITY,
        );
        let clipped = intersect_clips(&mesh, &[clip], 1, &[1], 4.0, 1_048_576).unwrap();
        let coverage = super::super::support::prepare_untranslated_shadow_coverage(
            &clipped,
            0.0,
            0.0,
            4.0,
            &[1],
        )
        .unwrap()
        .unwrap();
        let total = coverage.values.iter().sum::<f64>();
        assert!((total - 32.0).abs() < 1.0e-7);
    }
}
