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
use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};
use std::{collections::BTreeMap, sync::Arc};

mod region;

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

/// Aggregate geometric clip-pair work is owned by one contribution, not by
/// individual item, group, or clipping invocation. Like vertex retention, this
/// budget is charged before expensive triangle intersections allocate storage.
fn charge_clip_work(
    used: &mut usize,
    count: usize,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    let next = used
        .checked_add(count)
        .ok_or_else(|| resource(path, "neutral clip accumulation overflow"))?;
    if next > MAX_NEUTRAL_CLIP_WORK {
        return Err(resource(
            path,
            "neutral clip triangle intersection work exceeds aggregate admission",
        ));
    }
    *used = next;
    Ok(())
}

struct SourceFrame<'a> {
    begin_event: usize,
    path: Vec<usize>,
    group: &'a crate::composition_2d::Render2dGroup,
    /// Structural source-neutral support in this group's LOCAL frame.
    /// No ancestor completion, opacity or disk morphology has been flattened.
    children: Vec<Arc<region::NeutralRegion<'a>>>,
}

/// Immutable F2 glyph-field quality and F3F disposable geometric sample
/// density are distinct: a 4x correlated mask must not remint the font cache.
#[derive(Clone, Copy)]
pub(super) struct NeutralPreparationScale {
    pub(super) field_raster_scale: f64,
    pub(super) geometry_sample_scale: f64,
}

fn neutral_item(
    item: &Render2dItem,
    path: &[usize],
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    scales: NeutralPreparationScale,
    max_buffer_bytes: u64,
    clip_work_used: &mut usize,
) -> Result<Option<NeutralMesh>, Render2dExecutionError> {
    let root_index = path[0];
    let result = match item.primitive() {
        Render2dPrimitive::Fill { .. } | Render2dPrimitive::Stroke { .. } => {
            vector::neutral_support(
                item,
                root_index,
                scales.geometry_sample_scale,
                max_buffer_bytes,
            )?
        }
        Render2dPrimitive::Image(primitive) => image::neutral_support(
            item,
            primitive,
            root_index,
            scales.geometry_sample_scale,
            max_buffer_bytes,
        )?,
        Render2dPrimitive::ShapedText(primitive) => {
            let value = bindings
                .get(primitive.resource_id())
                .ok_or_else(|| precision(path, "missing validated shaped-text binding"))?;
            let Render2dResourceValue::ShapedText(resource) = value else {
                return Err(precision(path, "shaped-text binding kind changed"));
            };
            let quality = QualityTier::select(resource.font_size(), scales.field_raster_scale)
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
                if let Some(mesh) = vector::neutral_shaped_glyph(
                    instance,
                    path,
                    scales.geometry_sample_scale,
                    max_buffer_bytes,
                )? {
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
        scales.geometry_sample_scale,
        max_buffer_bytes,
        clip_work_used,
    )?;
    Ok((!clipped.triangles.is_empty()).then_some(clipped))
}

/// An exact geometric Minkowski sum of a triangle UNION with the finite
/// parent-axis Gaussian cutoff square. The 3sigma square is SUPPORT only:
/// Gaussian alpha remains the separate continuous normalized integral.
/// A square dilated convex triangle is its 12 translated vertices' convex
/// hull, so the resulting triangulated union is exact (not an AABB).
fn gaussian_square_support(
    source: &NeutralMesh,
    radius: f64,
    path: &[usize],
) -> Result<NeutralMesh, Render2dExecutionError> {
    if !radius.is_finite() || radius <= 0.0 {
        return Err(precision(path, "nested Gaussian support radius invalid"));
    }
    if !source.units_per_parent_logical_unit.is_finite()
        || source.units_per_parent_logical_unit <= 0.0
        || !source.triangles.len().is_multiple_of(3)
    {
        return Err(precision(path, "nested Gaussian source mesh invalid"));
    }
    let largest = source
        .triangles
        .len()
        .checked_mul(10)
        .ok_or_else(|| resource(path, "nested Gaussian support growth overflow"))?;
    if largest > MAX_NEUTRAL_VERTICES {
        return Err(resource(
            path,
            "nested Gaussian support growth exceeds geometry budget",
        ));
    }
    let mut expanded = empty_mesh();
    expanded
        .triangles
        .try_reserve(largest)
        .map_err(|_| resource(path, "nested Gaussian support allocation failed"))?;
    for tri in source.triangles.as_chunks::<3>().0 {
        if orient(tri[0], tri[1], tri[2]) == 0.0 {
            continue;
        }
        let mut points = Vec::with_capacity(12);
        for &[x, y] in tri {
            let x = x / source.units_per_parent_logical_unit;
            let y = y / source.units_per_parent_logical_unit;
            for dx in [-radius, radius] {
                for dy in [-radius, radius] {
                    let p = [x + dx, y + dy];
                    if !p.iter().all(|n| n.is_finite()) {
                        return Err(precision(path, "nested Gaussian square loses precision"));
                    }
                    points.push(p);
                }
            }
        }
        points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
        points.dedup();
        let mut hull: Vec<[f64; 2]> = Vec::with_capacity(points.len() * 2);
        for &point in &points {
            while hull.len() >= 2 {
                let n = hull.len();
                let turn = orient(hull[n - 2], hull[n - 1], point);
                if !turn.is_finite() {
                    return Err(precision(path, "nested Gaussian hull orientation overflow"));
                }
                if turn > 0.0 {
                    break;
                }
                hull.pop();
            }
            hull.push(point);
        }
        let lower = hull.len();
        for &point in points.iter().rev().skip(1) {
            while hull.len() > lower {
                let n = hull.len();
                let turn = orient(hull[n - 2], hull[n - 1], point);
                if !turn.is_finite() {
                    return Err(precision(path, "nested Gaussian hull orientation overflow"));
                }
                if turn > 0.0 {
                    break;
                }
                hull.pop();
            }
            hull.push(point);
        }
        hull.pop();
        if hull.len() < 3 {
            return Err(precision(path, "nested Gaussian hull collapsed"));
        }
        for index in 1..hull.len() - 1 {
            for point in [hull[0], hull[index], hull[index + 1]] {
                expanded.bounds[0] = expanded.bounds[0].min(point[0]);
                expanded.bounds[1] = expanded.bounds[1].min(point[1]);
                expanded.bounds[2] = expanded.bounds[2].max(point[0]);
                expanded.bounds[3] = expanded.bounds[3].max(point[1]);
                expanded.triangles.push(point);
            }
        }
    }
    Ok(expanded)
}

/// Derives source-neutral F1 group support in each immediate-parent frame,
/// independent of color alpha, materialization details or final-target culling.
///
/// Every shadow-bearing group's pre-shadow C is materialized once for the
/// unchanged F3E correlated compositor. Descendant completion is represented
/// by a bounded immutable DAG: only certified polygonal operations are lowered
/// to triangles, with shared pristine sources and typed unsupported curved
/// propagation. A positive disk is never silently replaced with child C,
/// an axis-aligned bound, cached alpha or an approximate distance field.
///
/// Retaining graph operations here is a structural cutover, NOT admission of
/// previously unsupported mixed signed morphology or repeated curved effects.
/// One renderer-private physical realization of an immutable F1 caster.
/// The qualified disk remains an exact curve, never triangle/AABB/alpha support.
pub(super) enum NeutralCaster {
    Mesh(Arc<NeutralMesh>),
    RoundedRectangle(region::CertifiedRoundedRectangle),
}

pub(super) fn group_child_sources<'a>(
    plan: &scene::Plan<'a>,
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    scales: NeutralPreparationScale,
    max_buffer_bytes: u64,
) -> Result<BTreeMap<usize, NeutralCaster>, Render2dExecutionError> {
    group_child_sources_with_probe(
        plan,
        bindings,
        field_sets,
        scales,
        max_buffer_bytes,
        |_, _| {},
    )
}

/// A read-only, transient observer for deterministic tests of the REAL
/// borrowed-F1 graph construction. It never writes or replaces F1 authority,
/// does not retain renderer-owned nodes, and cannot bypass materialization.
/// The production path supplies a no-op observer.
fn group_child_sources_with_probe<'a>(
    plan: &scene::Plan<'a>,
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    scales: NeutralPreparationScale,
    max_buffer_bytes: u64,
    mut inspect: impl FnMut(&Arc<region::NeutralRegion<'a>>, &[usize]),
) -> Result<BTreeMap<usize, NeutralCaster>, Render2dExecutionError> {
    let mut active = Vec::<SourceFrame<'a>>::new();
    let mut sources = BTreeMap::new();
    let mut graph = region::NeutralRegionBuilder::new(scales, max_buffer_bytes);
    for (index, event) in plan.events.iter().enumerate() {
        match event {
            scene::Event::BeginGroup { path, group, .. } => {
                active.push(SourceFrame {
                    begin_event: index,
                    path: path.clone(),
                    group,
                    children: Vec::new(),
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
                    scales,
                    max_buffer_bytes,
                    &mut graph.clip_work,
                )? {
                    let leaf = graph.source(mesh, path)?;
                    active
                        .last_mut()
                        .expect("active source group")
                        .children
                        .push(leaf);
                }
            }
            scene::Event::EndGroup => {
                let frame = active.pop().expect("F1 balanced group plan");
                let children = graph.union(frame.children, &frame.path)?;
                // A group first maps its entire local C into its immediate
                // parent frame. Only THEN do its Euclidean disk, offset and
                // finite Gaussian square operators act in that parent frame.
                let in_parent =
                    graph.affine(children, frame.group.local_to_parent(), &frame.path)?;
                let nested = !active.is_empty();
                if !frame.group.shadows().is_empty() {
                    inspect(&in_parent, &frame.path);
                    let source = match graph.materialize(&in_parent) {
                        Ok(mesh) => NeutralCaster::Mesh(mesh),
                        Err(error @ Render2dExecutionError::UnsupportedEntry { .. }) if !nested => {
                            match graph.qualified_rounded_rectangle(&in_parent)? {
                                Some(qualified) => NeutralCaster::RoundedRectangle(qualified),
                                None => return Err(error),
                            }
                        }
                        Err(error) => return Err(error),
                    };
                    sources.insert(frame.begin_event, source);
                    if !nested {
                        // A root group has no ancestor caster consumer. Its
                        // paint shadows are already owned by the same F3E
                        // compositor, not by this neutral support graph.
                        continue;
                    }
                }
                if !nested {
                    continue;
                }
                let mut components = vec![Arc::clone(&in_parent)];
                if !frame.group.shadows().is_empty() {
                    // Do not reject a positive disk before its symbolic
                    // neutral support reaches the consuming ancestor's DAG.
                    // Polygonal materialization of that root still fails
                    // closed with the disk's exact original F1 owner path;
                    // no cached alpha, AABB or faceted substitute is admitted.
                    components
                        .try_reserve(frame.group.shadows().len())
                        .map_err(|_| {
                            resource(&frame.path, "neutral effect graph allocation failed")
                        })?;
                    for &effect in frame.group.shadows() {
                        let effect_region =
                            graph.effect(Arc::clone(&in_parent), effect, &frame.path)?;
                        if effect.spread() <= 0.0 {
                            // Preserve the existing admitted subset: convex
                            // negative erosion, zero spread and Gaussian
                            // support must actually qualify now. A negative
                            // nonconvex caster still fails with its owner path,
                            // even when no ancestor would paint from it.
                            graph.materialize(&effect_region)?;
                        }
                        // Positive disks remain symbolic until a certified
                        // ancestor region query is independently qualified;
                        // their own continuous paint path remains unchanged.
                        components.push(effect_region);
                    }
                }
                let completed = graph.union(components, &frame.path)?;
                let clipped = graph.clip(completed, frame.group.clips(), &frame.path)?;
                if let Some(parent) = active.last_mut() {
                    parent.children.push(clipped);
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
    // A rank-deficient authored affine truly maps positive-area 2D support
    // to no filled area. A FULL-RANK affine whose triangles *numerically*
    // collapse instead is a representability failure, never an empty caster.
    // Complete the precision preflight before changing the output mesh.
    let determinant = a.mul_add(d, -(b * c));
    for (before, after) in source
        .triangles
        .as_chunks::<3>()
        .0
        .iter()
        .zip(prepared.as_chunks::<3>().0)
    {
        if orient(after[0], after[1], after[2]) == 0.0
            && determinant != 0.0
            && orient(before[0], before[1], before[2]) != 0.0
        {
            return Err(precision(
                path,
                "full-rank parent affine loses positive-area neutral geometry",
            ));
        }
    }
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
    clip_work_used: &mut usize,
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
        charge_clip_work(clip_work_used, cost, path)?;
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
    use crate::execution_2d::Render2dUnsupportedContent;

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
    fn nondegenerate_neutral_support_cannot_silently_collapse_at_large_translation() {
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            bounds: [0.0, 0.0, 1.0, 1.0],
        };
        let mut retained = empty_mesh();
        append(
            &mut retained,
            &source,
            Render2dAffineTransform::IDENTITY,
            &[3],
        )
        .unwrap();
        let before = retained.triangles.clone();
        let too_large = Render2dAffineTransform::translation(1.0e17, 0.0).unwrap();
        assert!(matches!(
            append(&mut retained, &source, too_large, &[3, 7]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [3, 7]
        ));
        assert_eq!(
            retained.triangles, before,
            "failed geometry preflight is atomic"
        );
        let rank_deficient = Render2dAffineTransform::new(1.0, 0.0, 2.0, 0.0, 0.0, 0.0).unwrap();
        let mut empty = empty_mesh();
        append(&mut empty, &source, rank_deficient, &[3, 8]).unwrap();
        assert!(
            empty.triangles.is_empty(),
            "mathematically singular 2D fill has no area"
        );
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
        let sources = group_child_sources(
            &plan,
            &empty,
            &BTreeMap::new(),
            NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            1_048_576,
        )
        .unwrap();
        assert_eq!(sources.len(), 1);
        let NeutralCaster::Mesh(caster) = sources.get(&0).unwrap() else {
            panic!("clipped caster must remain polygonal");
        };
        for (actual, expected) in caster.bounds.iter().zip([0.5, 0.0, 2.0, 1.0]) {
            assert!((*actual - expected).abs() < 1.0e-6);
        }
        assert_eq!(
            shadow_envelope(caster, effect, &[0]).unwrap().unwrap(),
            [-0.5, -1.0, 3.0, 2.0]
        );
    }

    #[test]
    fn outlined_text_uses_f2_cached_field_tier_independent_of_4x_shadow_density() {
        use crate::composition_2d::{
            Render2dComposition, Render2dEntry, Render2dFontBinding, Render2dGlyph, Render2dGroup,
            Render2dPoint, Render2dResourceBinding, Render2dResourceId,
            Render2dShapedTextPrimitive, Render2dShapedTextResource,
        };
        const FONT: &[u8] = include_bytes!("../../../tests/fixtures/f2_outline.ttf");
        let id = Render2dResourceId::new(91).unwrap();
        let resource = Render2dShapedTextResource::new(
            Render2dFontBinding::new(FONT.to_vec(), 0, Vec::new(), false, None).unwrap(),
            24.0,
            vec![Render2dGlyph::new(2, 0.0, 0.0, 24.0).unwrap()],
        )
        .unwrap();
        let tier = QualityTier::select(24.0, 1.0).unwrap();
        assert_ne!(tier, QualityTier::select(24.0, 4.0).unwrap());
        let fields = super::super::field::realize(
            id,
            &resource,
            tier,
            4096,
            &mut super::super::field::FieldBudget::default(),
        )
        .unwrap();
        let mut cached = BTreeMap::new();
        cached.insert(FieldSetKey::new(id, tier), Arc::new(fields));
        let bindings = Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
            id,
            Render2dResourceValue::ShapedText(resource),
        )])
        .unwrap();
        let glyph = Render2dItem::new(
            Render2dPrimitive::ShapedText(Render2dShapedTextPrimitive::new(
                id,
                Render2dPoint::new(1.0, 2.0).unwrap(),
                Render2dColorRgba8::TRANSPARENT,
            )),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let shadow =
            Render2dDropShadow::new(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let group = Render2dGroup::new(
            vec![Render2dEntry::item(glyph)],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            vec![shadow],
        );
        let composition = Render2dComposition::new(vec![Render2dEntry::group(group)]).unwrap();
        let plan = scene::analyze(&composition).unwrap();
        let sources = group_child_sources(
            &plan,
            &bindings,
            &cached,
            NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            1_048_576,
        )
        .unwrap();
        let NeutralCaster::Mesh(mesh) = sources.get(&0).unwrap() else {
            panic!("outlined glyph caster must remain polygonal");
        };
        assert!(!mesh.triangles.is_empty());
        assert!(mesh.bounds[0].is_finite() && mesh.bounds[2] > mesh.bounds[0]);
    }

    #[test]
    fn exact_gaussian_square_support_of_triangle_does_not_fill_its_aabb() {
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let expanded = gaussian_square_support(&source, 0.5, &[1, 2]).unwrap();
        assert_eq!(expanded.bounds, [-0.5, -0.5, 2.5, 2.5]);
        let contains = |p: [f64; 2]| {
            expanded.triangles.as_chunks::<3>().0.iter().any(|tri| {
                let signs = [
                    orient(tri[0], tri[1], p),
                    orient(tri[1], tri[2], p),
                    orient(tri[2], tri[0], p),
                ];
                signs.iter().all(|n| *n >= 0.0) || signs.iter().all(|n| *n <= 0.0)
            })
        };
        assert!(contains([1.0, 1.0]));
        assert!(contains([2.3, 0.2]));
        assert!(
            !contains([2.4, 2.4]),
            "the extent is not geometric membership"
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
        let composition = Render2dComposition::new(vec![Render2dEntry::group(parent)]).unwrap();
        let plan = scene::analyze(&composition).unwrap();
        assert!(matches!(
            group_child_sources(
                &plan,
                &Render2dResourceBindings::default(),
                &BTreeMap::new(),
                NeutralPreparationScale {
                    field_raster_scale: 1.0,
                    geometry_sample_scale: 4.0,
                },
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
    fn clip_intersection_admission_is_contribution_wide_and_failure_is_atomic() {
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let clip = Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(0.0, 0.0, 1.0, 1.0).unwrap()),
            Render2dAffineTransform::IDENTITY,
        );
        // A rectangle has two triangles; one input triangle therefore
        // contributes two independently charged triangle-pair inspections.
        let one_clip_cost = 2;
        let mut used = MAX_NEUTRAL_CLIP_WORK - one_clip_cost;
        let first = intersect_clips(
            &source,
            std::slice::from_ref(&clip),
            0,
            &[0, 0],
            4.0,
            1_048_576,
            &mut used,
        )
        .unwrap();
        assert!(!first.triangles.is_empty());
        assert_eq!(used, MAX_NEUTRAL_CLIP_WORK);
        assert!(matches!(
            intersect_clips(
                &source,
                &[clip],
                0,
                &[0, 1],
                4.0,
                1_048_576,
                &mut used,
            ),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [0, 1]
        ));
        assert_eq!(
            used, MAX_NEUTRAL_CLIP_WORK,
            "failed admission never consumes further work budget"
        );
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
        let mut clip_work_used = 0;
        let clipped =
            intersect_clips(&mesh, &[clip], 1, &[1], 4.0, 1_048_576, &mut clip_work_used).unwrap();
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
