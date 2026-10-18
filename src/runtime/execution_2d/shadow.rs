//! Renderer-private continuous neutral geometry graph for F3F group effects.
//!
//! Only F1's borrowed painter events and immutable font/image bindings own
//! meaning. This module materializes bounded group-parent geometry *before*
//! ancestor affines, non-semantic paint alpha and target culling. Every
//! owner clip is a geometric intersection, not a source color or an AABB.

use super::{
    field::{FieldSetKey, QualityTier, ResourceFields},
    image, scene,
    support::{MAX_COMPOSITIONAL_SHADOW_WORK, NeutralMesh, PreparationWork},
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

pub(in crate::runtime::execution_2d) mod region;

const MAX_NEUTRAL_VERTICES: usize = 1_048_576;
const MAX_NEUTRAL_CLIP_WORK: usize = 16_777_216;

/// Numerical work and the clip-pair inventory are distinct quantities. Both
/// belong to this invocation; neither starts again for an item or group.
pub(super) struct NeutralPreparationWork<'a> {
    graph: &'a mut usize,
    clip_pairs: &'a mut usize,
}

impl NeutralPreparationWork<'_> {
    fn charge(&mut self, cost: usize, path: &[usize]) -> Result<(), Render2dExecutionError> {
        self.graph.charge(
            cost,
            MAX_COMPOSITIONAL_SHADOW_WORK,
            path,
            "neutral source preparation exceeds shared graph work",
        )
    }

    fn append(
        &mut self,
        output: &mut NeutralMesh,
        source: &NeutralMesh,
        path: &[usize],
    ) -> Result<(), Render2dExecutionError> {
        self.charge(append_work(output, source, path)?, path)?;
        append(output, source, Render2dAffineTransform::IDENTITY, path)
    }
}

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

/// Preserve nested occurrence ownership when root-oriented physical helpers
/// decline neutral source preparation. Root paint diagnostics keep their API.
fn neutral_source_error(error: Render2dExecutionError, path: &[usize]) -> Render2dExecutionError {
    if path.len() <= 1 {
        return error;
    }
    use crate::execution_2d::{Render2dImageError, Render2dVectorError};
    match error {
        Render2dExecutionError::Image {
            kind: Render2dImageError::PrecisionLimit,
            ..
        }
        | Render2dExecutionError::Vector {
            kind: Render2dVectorError::PrecisionLimit,
            ..
        } => precision(
            path,
            "nested original neutral source cannot preserve physical precision",
        ),
        Render2dExecutionError::Image {
            kind: Render2dImageError::ResourceLimit,
            ..
        }
        | Render2dExecutionError::Vector {
            kind: Render2dVectorError::ResourceLimit,
            ..
        } => resource(
            path,
            "nested original neutral source exceeds physical resource admission",
        ),
        other => other,
    }
}

fn neutral_item(
    item: &Render2dItem,
    path: &[usize],
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    scales: NeutralPreparationScale,
    max_buffer_bytes: u64,
    work: &mut NeutralPreparationWork<'_>,
) -> Result<Option<NeutralMesh>, Render2dExecutionError> {
    let root_index = path[0];
    let result = match item.primitive() {
        Render2dPrimitive::Fill { .. } | Render2dPrimitive::Stroke { .. } => {
            vector::neutral_support(
                item,
                root_index,
                scales.geometry_sample_scale,
                max_buffer_bytes,
            )
            .map_err(|error| neutral_source_error(error, path))?
        }
        Render2dPrimitive::Image(primitive) => image::neutral_support(
            item,
            primitive,
            root_index,
            scales.geometry_sample_scale,
            max_buffer_bytes,
        )
        .map_err(|error| neutral_source_error(error, path))?,
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
            let mut has_original_outline = false;
            for glyph in resource.glyphs() {
                let Some(field) = fields.glyph(glyph.id()) else {
                    // An admitted empty glyph outline has no neutral area.
                    continue;
                };
                has_original_outline = true;
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
                    work.append(&mut result, &mesh, path)?;
                }
            }
            if !has_original_outline {
                // The immutable admitted glyph facts prove absence. An empty
                // disposable realization of a PRESENT outline does not.
                return Ok(None);
            }
            Some(result)
        }
    };
    // Missing physical triangles cannot delete borrowed original F1. Keep a
    // leaf even when tessellation/culling produced no disposable geometry;
    // materialization must independently prove emptiness or stay symbolic.
    let mesh = result.unwrap_or_else(empty_mesh);
    if region::trivially_empty_original_source(item) {
        return Ok(None);
    }
    let clipped = intersect_clips(
        &mesh,
        item.clips(),
        root_index,
        path,
        scales.geometry_sample_scale,
        max_buffer_bytes,
        work,
    )?;
    Ok(Some(clipped))
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

/// Prove that a completed NEGATIVE disk shadow cannot add any neutral
/// support to its unchanged source, including sources with holes, curves,
/// clips or nested effects. If E=C⊖disk(r), then E+v⊆C for every
/// displacement |v|₂≤r. The physical finite Gaussian's SUPPORT is
/// translated E plus the entire parent-axis square [-3σ,3σ]², so
/// |v|₂≤|offset_x|+|offset_y|+6σ suffices.
///
/// Use the stronger r≥2(|ox|+|oy|)+8σ certificate, preserving the
/// existing offset-zero 8σ bound. The positive slack is deliberate
/// against floating-point boundary uncertainty. Unrepresentable sums or
/// products fail closed; this check never derives C from alpha or an AABB.
fn negative_shadow_absorbed_by_source(effect: Render2dDropShadow) -> bool {
    let radius = -effect.spread();
    if radius <= 0.0 {
        return false;
    }
    let displacement = effect.offset_x().abs() + effect.offset_y().abs();
    let conservative = (2.0 * displacement) + (8.0 * effect.sigma());
    conservative.is_finite() && radius >= conservative
}

/// Derives source-neutral F1 group support in each immediate-parent frame,
/// independent of color alpha, materialization details or final-target culling.
///
/// Every shadow-bearing group's pre-shadow C retains one immutable source
/// expression. Certified polygonal operations may use disposable triangles;
/// curved support is queried directly for bounded continuous cell areas by
/// the unchanged F3E correlated compositor. A positive disk is never silently replaced with child C,
/// an axis-aligned bound, cached alpha or an approximate distance field.
///
/// Deliberately opaque borrowed original-F1 source-neutral expression.
/// Its consumer must independently certify continuous coverage, physical
/// precision and whole-scene resources. Keeping the DAG private prevents
/// bounds or disposable alpha from becoming geometric membership.
pub(super) struct SymbolicCaster<'a> {
    expression: Arc<region::NeutralRegion<'a>>,
    first_unqualified_owner: Vec<usize>,
}

impl SymbolicCaster<'_> {
    pub(super) fn logical_envelope(&self) -> Option<[f64; 4]> {
        self.expression.certified_logical_envelope()
    }

    pub(super) fn owner_path(&self) -> &[usize] {
        &self.first_unqualified_owner
    }

    /// Retains original borrowed F1 region authority across coherent sample
    /// blocks. The result is uncertain interval evidence, not paint alpha.
    pub(super) fn continuous_grid_enclosures(
        &self,
        root_grid: [u32; 4],
        samples_per_logical_unit: f64,
        owner_to_root: Option<Render2dAffineTransform>,
        max_cell_depth: u8,
        max_cells: usize,
        fuel: &mut usize,
    ) -> Result<Vec<[f64; 2]>, region::AreaQueryError> {
        self.expression.continuous_grid_enclosures(
            root_grid,
            samples_per_logical_unit,
            owner_to_root,
            max_cell_depth,
            max_cells,
            fuel,
        )
    }
}

/// Retaining graph operations here is a structural cutover, NOT admission of
/// previously unsupported mixed signed morphology or repeated curved effects.
/// One renderer-private physical realization of an immutable F1 caster.
/// The qualified disk remains an exact curve, never triangle/AABB/alpha support.
pub(super) enum NeutralCaster<'a> {
    Mesh(Arc<NeutralMesh>),
    /// Every original F1 leaf is an exact straight rectangle or triangle fill.
    /// Signed disk integration may consume this geometry after original-F1 proof;
    /// this does not re-author a neutral support representation.
    OriginalStraightPolygonUnion(Arc<NeutralMesh>),
    RoundedRectangle(region::CertifiedRoundedRectangle),
    /// The ONE immutable borrowed-F1 neutral expression has reached its
    /// consuming F3E boundary without being falsified into a disposable
    /// polygon. Continuous area intervals qualify its disposable paint;
    /// unsupported operations retain the original authored owner path.
    Symbolic(SymbolicCaster<'a>),
}

/// Preserve one borrowed original-F1 expression when polygonalization
/// cannot represent its exact geometry. This is a transport boundary, NOT
/// permission to paint or sample unknown curved regions. Resource/precision
/// errors never become Symbolic. The consumer independently admits continuous
/// coverage before any target mutation and preserves authored-path failures.
fn materialized_or_symbolic<'a>(
    graph: &mut region::NeutralRegionBuilder,
    source: &Arc<region::NeutralRegion<'a>>,
    original_straight: bool,
) -> Result<NeutralCaster<'a>, Render2dExecutionError> {
    match graph.materialize(source) {
        Ok(mesh) if original_straight => Ok(NeutralCaster::OriginalStraightPolygonUnion(mesh)),
        Ok(mesh) => Ok(NeutralCaster::Mesh(mesh)),
        Err(Render2dExecutionError::UnsupportedEntry {
            path,
            kind: Render2dUnsupportedContent::Shadows { .. },
        }) if !source.contains_positive_disk => Ok(NeutralCaster::Symbolic(SymbolicCaster {
            expression: Arc::clone(source),
            first_unqualified_owner: path,
        })),
        Err(error) => Err(error),
    }
}

/// One immutable source-preparation result. Graph work has already been
/// admitted and paid; its consuming coverage policy must retain that charge.
pub(super) struct GroupChildSources<'a> {
    pub(super) casters: BTreeMap<usize, NeutralCaster<'a>>,
    pub(super) graph_work: usize,
}

pub(super) fn group_child_sources<'a>(
    plan: &scene::Plan<'a>,
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    scales: NeutralPreparationScale,
    max_buffer_bytes: u64,
    root_canvas: Option<([f64; 2], f64)>,
) -> Result<GroupChildSources<'a>, Render2dExecutionError> {
    group_child_sources_with_probe(
        plan,
        bindings,
        field_sets,
        scales,
        max_buffer_bytes,
        root_canvas,
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
    root_canvas: Option<([f64; 2], f64)>,
    mut inspect: impl FnMut(&Arc<region::NeutralRegion<'a>>, &[usize]),
) -> Result<GroupChildSources<'a>, Render2dExecutionError> {
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
                if let Some(leaf) = graph.prepare_source_from_item(item, path, |work| {
                    neutral_item(
                        item,
                        path,
                        bindings,
                        field_sets,
                        scales,
                        max_buffer_bytes,
                        work,
                    )
                })? {
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
                    // Resolve the proven curved DAG before asking for a
                    // polygonal mesh: a disk cannot be made into triangles
                    // and an expected rejection is not control flow.
                    let offscreen = if !nested {
                        if let Some((canvas, raster_scale)) = root_canvas {
                            in_parent.all_root_shadows_provably_offscreen(
                                frame.group.shadows(),
                                canvas,
                                raster_scale,
                            )?
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    let source = if !nested
                        && (in_parent.all_root_shadows_provably_empty(frame.group.shadows())
                            || offscreen)
                    {
                        // Certified-empty or fully offscreen root effects
                        // cannot reach the requested target. Preserve all
                        // original F3E child painting and nested support.
                        NeutralCaster::Mesh(Arc::new(empty_mesh()))
                    } else if !nested {
                        match graph.qualified_rounded_rectangle(&in_parent)? {
                            Some(qualified) => NeutralCaster::RoundedRectangle(qualified),
                            None => {
                                // Root shadow paint is not exempt from
                                // original-F1 negative-erosion provenance.
                                // A convex Lyon tessellation never proves a
                                // curved F1 fill/stroke is polygonal support.
                                let proven = frame
                                    .group
                                    .shadows()
                                    .iter()
                                    .any(|shadow| shadow.spread() < 0.0)
                                    && graph.proven_original_straight_polygon_union(&in_parent)?;
                                materialized_or_symbolic(&mut graph, &in_parent, proven)?
                            }
                        }
                    } else {
                        let exact_original_polygons = frame
                            .group
                            .shadows()
                            .iter()
                            .any(|effect| effect.spread() < 0.0)
                            && graph.proven_original_straight_polygon_union(&in_parent)?;
                        materialized_or_symbolic(&mut graph, &in_parent, exact_original_polygons)?
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
                        // Certified containment is valid even for a
                        // translated finite Gaussian over a truly holed,
                        // concave or curved original C: E(C,-r)+offset+
                        // finite halo ⊆ C when the whole displacement
                        // fits inside the erosion disk. Retain pristine C;
                        // the independent physical shadow remains painted.
                        if negative_shadow_absorbed_by_source(effect) {
                            continue;
                        }
                        let effect_region =
                            graph.effect(Arc::clone(&in_parent), effect, &frame.path)?;
                        if effect.spread() <= 0.0 {
                            // Existing fully polygonal cases still materialize
                            // eagerly and account for real geometry/work.
                            // An unmaterializable SIGNED disk stays symbolic
                            // through the ancestor F1 DAG instead of being
                            // silently approximated or rejected before F3E
                            // can inspect the complete authored expression.
                            // A missing proof is NOT a resource/precision
                            // exception; those errors still fail immediately.
                            match graph.materialize(&effect_region) {
                                Ok(_) => {}
                                Err(Render2dExecutionError::UnsupportedEntry {
                                    kind: Render2dUnsupportedContent::Shadows { .. },
                                    ..
                                }) if effect.spread() < 0.0 => {}
                                Err(error) => return Err(error),
                            }
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
    Ok(GroupChildSources {
        casters: sources,
        graph_work: graph.graph_work,
    })
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
    append_triangles(
        output,
        &source.triangles,
        source.units_per_parent_logical_unit,
        source_to_parent,
        path,
    )
}

fn append_triangles(
    output: &mut NeutralMesh,
    triangles: &[[f64; 2]],
    scale: f64,
    source_to_parent: Render2dAffineTransform,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    if triangles.is_empty() {
        return Ok(());
    }
    if !triangles.len().is_multiple_of(3) || !scale.is_finite() || scale <= 0.0 {
        return Err(precision(path, "invalid neutral mesh source or scale"));
    }
    let required = output
        .triangles
        .len()
        .checked_add(triangles.len())
        .ok_or_else(|| resource(path, "neutral geometry vertex count overflow"))?;
    if required > MAX_NEUTRAL_VERTICES {
        return Err(resource(
            path,
            "neutral geometry exceeds bounded vertex admission",
        ));
    }
    let [a, b, c, d, tx, ty] = source_to_parent.components();
    if region::interval::exact_products_equal(a, d, b, c) == Some(true) {
        return Ok(());
    }
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(triangles.len())
        .map_err(|_| resource(path, "neutral geometry allocation failed"))?;
    for &[x, y] in triangles {
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
    for (before, after) in triangles
        .as_chunks::<3>()
        .0
        .iter()
        .zip(prepared.as_chunks::<3>().0)
    {
        if orient(after[0], after[1], after[2]) == 0.0
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

/// Primitive preparation work for the transform, positive-area checks and
/// destination writes in append. Reserve can also relocate retained output.
/// Geometry retention and numerical work remain separate admission concerns.
pub(super) fn append_work(
    output: &NeutralMesh,
    source: &NeutralMesh,
    path: &[usize],
) -> Result<usize, Render2dExecutionError> {
    append_vertices_work(output, source.triangles.len(), path)
}

fn append_vertices_work(
    output: &NeutralMesh,
    source_vertices: usize,
    path: &[usize],
) -> Result<usize, Render2dExecutionError> {
    let required = output
        .triangles
        .len()
        .checked_add(source_vertices)
        .ok_or_else(|| resource(path, "neutral append work extent overflow"))?;
    let relocation = if required > output.triangles.capacity() {
        output.triangles.len()
    } else {
        0
    };
    source_vertices
        .checked_mul(64)
        .and_then(|cost| {
            relocation
                .checked_mul(4)
                .and_then(|copy| cost.checked_add(copy))
        })
        .and_then(|cost| cost.checked_add(256))
        .ok_or_else(|| resource(path, "neutral append preparation work overflow"))
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

/// Three triangle half-planes need at most nine vertices. Rounded predicates
/// are not an authority for that combinatorial proof: every insertion checks
/// the fixed capacity, so uncertain topology cannot grow a temporary heap.
struct ClipPolygon {
    points: [[f64; 2]; 9],
    len: usize,
}

impl ClipPolygon {
    fn empty() -> Self {
        Self {
            points: [[0.0; 2]; 9],
            len: 0,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn points(&self) -> &[[f64; 2]] {
        &self.points[..self.len]
    }

    fn push(&mut self, point: [f64; 2], path: &[usize]) -> Result<(), Render2dExecutionError> {
        let Some(slot) = self.points.get_mut(self.len) else {
            return Err(precision(
                path,
                "neutral clip polygon exceeds bounded representation",
            ));
        };
        *slot = point;
        self.len += 1;
        Ok(())
    }
}

/// The intersection of two *convex triangles*, without a rectangular/alpha
/// approximation. Clipping against a polygon-union clip means retaining each
/// triangle intersection's union (overlaps are resolved by area coverage).
fn triangle_intersection(
    source: &[[f64; 2]; 3],
    clip: &[[f64; 2]; 3],
    path: &[usize],
    work: &mut NeutralPreparationWork<'_>,
) -> Result<ClipPolygon, Render2dExecutionError> {
    // Borrow the triangles: no argument arrays are copied before admission.
    // Two orientations, fixed stack initialization, winding, source insertion,
    // and enclosing iterator/bookkeeping work fit this fixed setup bound.
    work.charge(256, path)?;
    let winding = orient(clip[0], clip[1], clip[2]);
    if winding == 0.0 || orient(source[0], source[1], source[2]) == 0.0 {
        return Ok(ClipPolygon::empty());
    }
    let mut polygon = ClipPolygon::empty();
    for &point in source {
        polygon.push(point, path)?;
    }
    for i in 0..3 {
        // Each visited edge has two orientations, sign/inside tests, at most
        // one division and two FMAs, finite guards, two bounded insertions,
        // and index/loop work. 128 per edge bounds all of those operations;
        // 128 also covers the fixed output initialization and pass bookkeeping.
        // Charge the ACTUAL current length before starting this pass.
        work.charge(128 + 128 * polygon.len(), path)?;
        let a = clip[i];
        let b = clip[(i + 1) % 3];
        let mut result = ClipPolygon::empty();
        for j in 0..polygon.len() {
            let p = polygon.points()[j];
            let q = polygon.points()[(j + 1) % polygon.len()];
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
                result.push(intersection, path)?;
            }
            if inside_q {
                result.push(q, path)?;
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
    work: &mut NeutralPreparationWork<'_>,
) -> Result<NeutralMesh, Render2dExecutionError> {
    if clips.is_empty() || source.triangles.is_empty() {
        let mut result = empty_mesh();
        work.append(&mut result, source, path)?;
        return Ok(result);
    }
    let mut current = empty_mesh();
    work.append(&mut current, source, path)?;
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
            vector::neutral_support(&item, root_index, resolution, max_buffer_bytes)
                .map_err(|error| neutral_source_error(error, path))?
        else {
            return Ok(empty_mesh());
        };
        let mut clip_local = empty_mesh();
        work.append(&mut clip_local, &clip_mesh, path)?;
        let mut next = empty_mesh();
        let cost = (current.triangles.len() / 3)
            .checked_mul(clip_local.triangles.len() / 3)
            .ok_or_else(|| resource(path, "neutral conjunctive clip work overflow"))?;
        charge_clip_work(work.clip_pairs, cost, path)?;
        for source in current.triangles.as_chunks::<3>().0 {
            for clipping in clip_local.triangles.as_chunks::<3>().0 {
                let polygon = triangle_intersection(source, clipping, path, work)?;
                if polygon.len() < 3 {
                    continue;
                }
                for index in 1..polygon.len() - 1 {
                    // Fan lookup/copy, orientation and loop bookkeeping precede
                    // the separately charged transform/copy of an emitted fan.
                    work.charge(64, path)?;
                    let points = polygon.points();
                    let triangle = [points[0], points[index], points[index + 1]];
                    if orient(triangle[0], triangle[1], triangle[2]) == 0.0 {
                        continue;
                    }
                    work.charge(append_vertices_work(&next, 3, path)?, path)?;
                    append_triangles(
                        &mut next,
                        &triangle,
                        1.0,
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

    #[test]
    fn unfunded_neutral_source_must_reject_before_physical_path_reconstruction() {
        use crate::composition_2d::{
            Render2dFillRule, Render2dPath, Render2dPathCommand, Render2dPoint, Render2dShape,
        };
        let mut commands = vec![Render2dPathCommand::MoveTo(
            Render2dPoint::new(0.0, 0.0).unwrap(),
        )];
        for index in 0..3_000 {
            commands.push(Render2dPathCommand::LineTo(
                Render2dPoint::new(f64::from(index % 2), 1.0).unwrap(),
            ));
        }
        // This legal finite F1 coordinate fails only when disposable physical
        // path reconstruction reaches it. Exhausted invocation work must win
        // before that reconstruction visits any of the preceding commands.
        commands.push(Render2dPathCommand::LineTo(
            Render2dPoint::new(1.0e20, 1.0).unwrap(),
        ));
        commands.push(Render2dPathCommand::Close);
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::path(
                    Render2dPath::new(Render2dFillRule::NonZero, commands).unwrap(),
                ),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let scales = NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        };
        let mut graph = region::NeutralRegionBuilder::new(scales, 16 * 1024 * 1024);
        graph.graph_work = MAX_COMPOSITIONAL_SHADOW_WORK;
        let result = graph.prepare_source_from_item(&item, &[4, 2], |work| {
            neutral_item(
                &item,
                &[4, 2],
                &Render2dResourceBindings::default(),
                &BTreeMap::new(),
                scales,
                16 * 1024 * 1024,
                work,
            )
        });
        assert!(
            matches!(&result, Err(Render2dExecutionError::SampleSpace {
            kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
        }) if path == &[4, 2]),
            "physical reconstruction ran before source admission"
        );
        assert_eq!(graph.graph_work, MAX_COMPOSITIONAL_SHADOW_WORK);
    }

    #[test]
    fn original_group_clip_retains_positive_strip_when_tessellation_disappears() {
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 4.0, 4.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let width = 2.0_f64.powi(-24);
        let clips = [Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(1.0, 1.0, width, 1.0).unwrap()),
            Render2dAffineTransform::IDENTITY,
        )];
        let scales = NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        };
        let mut graph = region::NeutralRegionBuilder::new(scales, 1_048_576);
        let mesh = vector::neutral_support(&item, 0, 4.0, 1_048_576)
            .unwrap()
            .unwrap();
        let source = graph.source_from_item(mesh, &item, &[0, 0, 0]).unwrap();
        let clipped = graph.clip(source, &clips, &[0, 0]).unwrap();
        let NeutralCaster::Symbolic(caster) =
            materialized_or_symbolic(&mut graph, &clipped, false).unwrap()
        else {
            panic!("positive original group clip cannot become an empty mesh");
        };
        assert_eq!(caster.owner_path(), [0, 0]);
        let mut fuel = 16_777_216;
        let areas = caster
            .continuous_grid_enclosures([4, 4, 8, 8], 4.0, None, 8, 16, &mut fuel)
            .unwrap();
        assert!(areas[0][0] > 0.0 && areas[0][0] <= 4.0 * width && 4.0 * width <= areas[0][1]);
    }

    #[test]
    fn original_f1_missing_realization_keeps_thin_rectangle_and_lens() {
        use crate::composition_2d::{
            Render2dFillRule, Render2dPath, Render2dPathCommand, Render2dPoint, Render2dRect,
            Render2dShape,
        };
        let width = f64::from_bits((1023_u64 - 24) << 52);
        let p = |x, y| Render2dPoint::new(x, y).unwrap();
        let shapes = [
            Render2dShape::rect(Render2dRect::new(1.0, 1.0, width, 1.0).unwrap()),
            Render2dShape::path(
                Render2dPath::new(
                    Render2dFillRule::NonZero,
                    vec![
                        Render2dPathCommand::MoveTo(p(1.0, 1.0)),
                        Render2dPathCommand::QuadraticTo {
                            control: p(1.5, 1.0 + width / 2.0),
                            to: p(2.0, 1.0),
                        },
                        Render2dPathCommand::QuadraticTo {
                            control: p(1.5, 1.0 - width / 2.0),
                            to: p(1.0, 1.0),
                        },
                        Render2dPathCommand::Close,
                    ],
                )
                .unwrap(),
            ),
        ];
        let bindings = Render2dResourceBindings::new(Vec::new()).unwrap();
        let scales = NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        };
        for (index, shape) in shapes.into_iter().enumerate() {
            let item = Render2dItem::new(
                Render2dPrimitive::Fill {
                    shape,
                    brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
                },
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
            );
            assert!(
                vector::neutral_support(&item, 3, 4.0, 1_048_576)
                    .unwrap()
                    .is_none(),
                "fixture must exercise an absent disposable realization"
            );
            let owner = [3, index];
            let mesh = neutral_item(
                &item,
                &owner,
                &bindings,
                &BTreeMap::new(),
                scales,
                1_048_576,
                &mut NeutralPreparationWork {
                    graph: &mut 0,
                    clip_pairs: &mut 0,
                },
            )
            .unwrap()
            .unwrap();
            assert!(mesh.triangles.is_empty());
            let mut graph = region::NeutralRegionBuilder::new(scales, 1_048_576);
            let source = graph.source_from_item(mesh, &item, &owner).unwrap();
            let NeutralCaster::Symbolic(caster) =
                materialized_or_symbolic(&mut graph, &source, false).unwrap()
            else {
                panic!("missing physical mesh cannot materialize as an empty caster");
            };
            assert_eq!(caster.owner_path(), owner);
            let reach = caster.logical_envelope().unwrap();
            assert!(reach[0] < reach[2] && reach[1] < reach[3]);
            if index == 0 {
                let mut fuel = 16_777_216;
                let areas = caster
                    .continuous_grid_enclosures([4, 4, 8, 8], 4.0, None, 8, 16, &mut fuel)
                    .unwrap();
                let expected = 4.0 * width;
                assert!(areas[0][0] > 0.0 && areas[0][0] <= expected && expected <= areas[0][1]);
            }
        }
    }

    #[test]
    fn original_f1_empty_receipts_preserve_empty_and_disjoint_clip_cases() {
        use crate::composition_2d::{
            Render2dRect, Render2dShape, Render2dStrokeCap, Render2dStrokeJoin, Render2dStrokeStyle,
        };
        let fill = |rect| Render2dPrimitive::Fill {
            shape: Render2dShape::rect(rect),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        };
        let ordinary = Render2dRect::new(0.0, 0.0, 4.0, 4.0).unwrap();
        let items = [
            Render2dItem::new(
                fill(Render2dRect::new(0.0, 0.0, 0.0, 4.0).unwrap()),
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
            ),
            Render2dItem::new(
                fill(ordinary),
                Render2dAffineTransform::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0).unwrap(),
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
            ),
            Render2dItem::new(
                Render2dPrimitive::Stroke {
                    shape: Render2dShape::rect(ordinary),
                    brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
                    style: Render2dStrokeStyle::new(
                        0.0,
                        Render2dStrokeCap::Round,
                        Render2dStrokeJoin::Round,
                        4.0,
                    )
                    .unwrap(),
                },
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
            ),
            Render2dItem::new(
                fill(ordinary),
                Render2dAffineTransform::IDENTITY,
                vec![Render2dClip::new(
                    Render2dShape::rect(Render2dRect::new(10.0, 10.0, 1.0, 1.0).unwrap()),
                    Render2dAffineTransform::IDENTITY,
                )],
                Render2dOpacity::TRANSPARENT,
            ),
        ];
        let scales = NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        };
        let bindings = Render2dResourceBindings::new(Vec::new()).unwrap();
        for (index, item) in items.iter().enumerate() {
            let mut graph = region::NeutralRegionBuilder::new(scales, 1_048_576);
            let source = graph
                .source_from_item(empty_mesh(), item, &[7, index])
                .unwrap();
            assert!(
                graph.materialize(&source).unwrap().triangles.is_empty(),
                "an original empty fact must remain accepted"
            );
            let realized = neutral_item(
                item,
                &[7, index],
                &bindings,
                &BTreeMap::new(),
                scales,
                1_048_576,
                &mut NeutralPreparationWork {
                    graph: &mut 0,
                    clip_pairs: &mut 0,
                },
            )
            .unwrap();
            if index < 3 {
                assert!(realized.is_none());
            } else {
                assert!(realized.unwrap().triangles.is_empty());
            }
        }
    }
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
            None,
        )
        .unwrap();
        assert_eq!(sources.casters.len(), 1);
        let NeutralCaster::Mesh(caster) = sources.casters.get(&0).unwrap() else {
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
        let mut exhausted = MAX_COMPOSITIONAL_SHADOW_WORK;
        assert!(matches!(
            neutral_item(
                &glyph,
                &[0, 0],
                &bindings,
                &cached,
                NeutralPreparationScale { field_raster_scale: 1.0, geometry_sample_scale: 4.0 },
                1_048_576,
                &mut NeutralPreparationWork { graph: &mut exhausted, clip_pairs: &mut 0 },
            ),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [0, 0]
        ));
        assert_eq!(exhausted, MAX_COMPOSITIONAL_SHADOW_WORK);
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
            None,
        )
        .unwrap();
        let NeutralCaster::Mesh(mesh) = sources.casters.get(&0).unwrap() else {
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
                None,
            ),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 0 },
            }) if path == [0, 0]
        ));
    }

    #[test]
    fn nonabsorbed_holed_signed_region_reaches_f3e_as_borrowed_symbolic_dag() {
        use crate::composition_2d::{
            Render2dComposition, Render2dEntry, Render2dGroup, Render2dOpacity, Render2dRect,
            Render2dShape,
        };
        let make_rect = |x, y, width, height| {
            Render2dEntry::item(Render2dItem::new(
                Render2dPrimitive::Fill {
                    shape: Render2dShape::rect(Render2dRect::new(x, y, width, height).unwrap()),
                    brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
                },
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
            ))
        };
        // True touching-seam F1 rectangular ring, not its bounding box.
        // Disk erosion rounds the inner hole and cannot be represented as
        // finite exact triangles. The offset causes nonabsorbed support.
        let ring = [
            make_rect(10.0, 10.0, 20.0, 8.0),
            make_rect(10.0, 22.0, 20.0, 8.0),
            make_rect(10.0, 18.0, 8.0, 4.0),
            make_rect(22.0, 18.0, 8.0, 4.0),
        ];
        let inner = Render2dGroup::new(
            ring,
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            vec![
                Render2dDropShadow::new(8.0, 0.0, 0.01, -1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
            ],
        );
        let outer = Render2dGroup::new(
            vec![Render2dEntry::group(inner)],
            Render2dAffineTransform::new(1.0, 0.0, 0.5, 1.0, 0.0, 0.0).unwrap(),
            Vec::new(),
            Render2dOpacity::OPAQUE,
            vec![
                Render2dDropShadow::new(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    Render2dColorRgba8::new(0, 0, 255, 255),
                )
                .unwrap(),
            ],
        );
        let scene = Render2dComposition::new(vec![Render2dEntry::group(outer)]).unwrap();
        let plan = scene::analyze(&scene).unwrap();
        let sources = group_child_sources(
            &plan,
            &Render2dResourceBindings::default(),
            &BTreeMap::new(),
            NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            1_048_576,
            None,
        )
        .unwrap();
        assert!(matches!(
            sources.casters.get(&0),
            Some(NeutralCaster::Symbolic(symbolic))
                if symbolic.owner_path() == [0, 0]
        ));
        assert!(matches!(
            sources.casters.get(&1),
            Some(NeutralCaster::OriginalStraightPolygonUnion(_))
        ));
    }

    #[test]
    fn bounded_translated_erosion_absorption_is_conservative_and_fail_closed() {
        let effect = |offset_x, offset_y, sigma, spread| {
            Render2dDropShadow::new(
                offset_x,
                offset_y,
                sigma,
                spread,
                Render2dColorRgba8::TRANSPARENT,
            )
            .unwrap()
        };
        assert!(negative_shadow_absorbed_by_source(effect(
            1.0, 0.0, 0.05, -2.5
        )));
        assert!(negative_shadow_absorbed_by_source(effect(
            -1.0, 0.0, 0.05, -2.5
        )));
        assert!(negative_shadow_absorbed_by_source(effect(
            0.0, 0.0, 0.25, -2.0
        )));
        assert!(negative_shadow_absorbed_by_source(effect(
            0.0, 0.0, 0.0, -0.25
        )));
        assert!(!negative_shadow_absorbed_by_source(effect(
            1.0, 0.0, 0.1, -2.5
        )));
        assert!(!negative_shadow_absorbed_by_source(effect(
            2.0, 0.0, 0.05, -2.5
        )));
        assert!(!negative_shadow_absorbed_by_source(effect(
            0.0, 0.0, 0.25, -1.0
        )));
        assert!(!negative_shadow_absorbed_by_source(effect(
            0.0, 0.0, 0.0, 0.0
        )));
        assert!(!negative_shadow_absorbed_by_source(effect(
            1.0, 0.0, 0.0, 0.5
        )));
        // Bounds that overflow f64 never certify containment.
        assert!(!negative_shadow_absorbed_by_source(effect(
            1.0e308, 1.0e308, 0.0, -1.0
        )));
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
            triangle_intersection(
                &disjoint_corner,
                &clip,
                &[1],
                &mut NeutralPreparationWork {
                    graph: &mut 0,
                    clip_pairs: &mut 0
                }
            )
            .unwrap()
            .len()
                < 3
        );
        let crossing = [[0.8, 0.8], [1.6, 0.8], [0.8, 1.6]];
        let clipped = triangle_intersection(
            &crossing,
            &clip,
            &[1],
            &mut NeutralPreparationWork {
                graph: &mut 0,
                clip_pairs: &mut 0,
            },
        )
        .unwrap();
        assert!(clipped.len() >= 3);
        for point in clipped.points() {
            assert!(point[0] >= 0.0 && point[1] >= 0.0);
            assert!(point[0] + point[1] <= 2.0 + 1.0e-12);
        }
    }

    #[test]
    fn clip_copy_and_each_half_plane_require_remaining_graph_work() {
        let triangle = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: triangle.to_vec(),
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let mut exhausted = MAX_COMPOSITIONAL_SHADOW_WORK;
        let mut clip_pairs = 0;
        assert!(matches!(
            intersect_clips(
                &mesh, &[], 2, &[2, 1], 4.0, 1_048_576,
                &mut NeutralPreparationWork { graph: &mut exhausted, clip_pairs: &mut clip_pairs },
            ),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [2, 1]
        ));
        assert_eq!(exhausted, MAX_COMPOSITIONAL_SHADOW_WORK);
        assert_eq!(clip_pairs, 0);
        assert_eq!(mesh.triangles, triangle);

        // Only setup fits. The first clipping pass must fail before visiting
        // an edge, retaining the paid setup rather than refunding it.
        let mut setup_only = MAX_COMPOSITIONAL_SHADOW_WORK - 256;
        assert!(matches!(
            triangle_intersection(&triangle, &triangle, &[2, 3],
                &mut NeutralPreparationWork { graph: &mut setup_only, clip_pairs: &mut clip_pairs }),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [2, 3]
        ));
        assert_eq!(setup_only, MAX_COMPOSITIONAL_SHADOW_WORK);
        assert_eq!(clip_pairs, 0, "pair inventory is not numerical work");
    }

    #[test]
    fn clip_temporary_capacity_rejects_uncertified_growth_without_mutation() {
        let mut polygon = ClipPolygon::empty();
        for _ in 0..9 {
            polygon.push([1.0, 2.0], &[4, 2]).unwrap();
        }
        assert!(matches!(
            polygon.push([3.0, 4.0], &[4, 2]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit, path: Some(path), ..
            }) if path == [4, 2]
        ));
        assert_eq!(polygon.points(), &[[1.0, 2.0]; 9]);
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
            &mut NeutralPreparationWork {
                graph: &mut 0,
                clip_pairs: &mut used,
            },
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
                &mut NeutralPreparationWork { graph: &mut 0, clip_pairs: &mut used },
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
        let clipped = intersect_clips(
            &mesh,
            &[clip],
            1,
            &[1],
            4.0,
            1_048_576,
            &mut NeutralPreparationWork {
                graph: &mut 0,
                clip_pairs: &mut clip_work_used,
            },
        )
        .unwrap();
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
