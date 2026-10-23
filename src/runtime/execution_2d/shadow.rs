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
    Render2dAffineTransform, Render2dBrush, Render2dClip, Render2dColorRgba8, Render2dComposition,
    Render2dDropShadow, Render2dItem, Render2dOpacity, Render2dPrimitive, Render2dResourceBindings,
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
    host: &'a mut super::support::PreparationHostBudget,
}

impl NeutralPreparationWork<'_> {
    pub(super) fn new<'a>(
        graph: &'a mut usize,
        clip_pairs: &'a mut usize,
        host: &'a mut super::support::PreparationHostBudget,
    ) -> NeutralPreparationWork<'a> {
        NeutralPreparationWork {
            graph,
            clip_pairs,
            host,
        }
    }

    pub(super) fn charge(
        &mut self,
        cost: usize,
        path: &[usize],
    ) -> Result<(), Render2dExecutionError> {
        self.graph.charge(
            cost,
            MAX_COMPOSITIONAL_SHADOW_WORK,
            path,
            "neutral source preparation exceeds shared graph work",
        )
    }

    /// Factory-owned buffers use the invocation's existing host owner. A
    /// completed source transfers this reservation into retained admission;
    /// failed factories release storage while keeping their paid CPU work.
    pub(super) fn reserve_vector<T>(
        &mut self,
        values: &mut Vec<T>,
        capacity: usize,
        path: &[usize],
    ) -> Result<(), Render2dExecutionError> {
        self.host.reserve_vector(values, capacity, path)
    }

    pub(super) fn reserve_bytes(
        &mut self,
        bytes: usize,
        path: &[usize],
    ) -> Result<(), Render2dExecutionError> {
        self.host.reserve(bytes, path)
    }

    pub(super) fn release_bytes(&mut self, bytes: usize) {
        self.host.release(bytes);
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
    /// Exact numerical chart of this group's immediate parent, excluding
    /// its own transform. None is lack of a receipt, not invalid F1 geometry.
    parent_chart: Option<Render2dAffineTransform>,
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

/// One lookup contract for the immutable F2 facts used by both original reach
/// and disposable glyph tessellation. No font reconstruction or quality
/// selection authority is duplicated by neutral source admission.
fn shaped_source<'a>(
    primitive: &crate::composition_2d::Render2dShapedTextPrimitive,
    bindings: &'a Render2dResourceBindings,
    field_sets: &'a BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    field_raster_scale: f64,
    path: &[usize],
    work: &mut NeutralPreparationWork<'_>,
) -> Result<
    (
        &'a crate::composition_2d::Render2dShapedTextResource,
        &'a ResourceFields,
    ),
    Render2dExecutionError,
> {
    let bits = |len: usize| (usize::BITS - len.leading_zeros()) as usize;
    work.charge(
        512 + 512 * (bits(bindings.len()) + bits(field_sets.len())),
        path,
    )?;
    let value = bindings
        .get(primitive.resource_id())
        .ok_or_else(|| precision(path, "missing validated shaped-text binding"))?;
    let Render2dResourceValue::ShapedText(resource) = value else {
        return Err(precision(path, "shaped-text binding kind changed"));
    };
    let quality = QualityTier::select(resource.font_size(), field_raster_scale)
        .ok_or_else(|| precision(path, "shaped-text quality is not representable"))?;
    let key = FieldSetKey::new(primitive.resource_id(), quality);
    let fields = field_sets
        .get(&key)
        .ok_or_else(|| precision(path, "missing validated immutable font outlines"))?;
    Ok((resource, fields))
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
            path,
            work,
        )
        .map_err(|error| neutral_source_error(error, path))?,
        Render2dPrimitive::ShapedText(primitive) => {
            let (resource, fields) = shaped_source(
                primitive,
                bindings,
                field_sets,
                scales.field_raster_scale,
                path,
                work,
            )?;
            let mut result = empty_mesh();
            let mut has_original_outline = false;
            let lookup_work = fields.outline_lookup_work();
            for glyph in resource.glyphs() {
                work.charge(lookup_work, path)?;
                let Some(field) = fields
                    .admitted_outline(glyph.id())
                    .map_err(|()| precision(path, "missing validated glyph outline record"))?
                else {
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
                    work,
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
    // An image factory already emits the complete destination triangles. With
    // no clips there is no reason to allocate a second normalized copy. The
    // mesh's explicit scale is preserved by every later physical consumer.
    if matches!(item.primitive(), Render2dPrimitive::Image(_)) && item.clips().is_empty() {
        return Ok(Some(mesh));
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
///
/// The result's retained vector is reserved on the invocation's host owner
/// before allocation; its reservation transfers to the region materializer.
/// Each triangle's 12 points and <=23 hull entries stay on the stack.
fn gaussian_square_support(
    source: &NeutralMesh,
    radius: f64,
    path: &[usize],
    host: &mut super::support::PreparationHostBudget,
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
    host.reserve_vector(&mut expanded.triangles, largest, path)?;
    let reserved = expanded
        .triangles
        .capacity()
        .checked_mul(size_of::<[f64; 2]>())
        .ok_or_else(|| resource(path, "nested Gaussian result storage overflow"))?;
    let result = (|| -> Result<(), Render2dExecutionError> {
        for tri in source.triangles.as_chunks::<3>().0 {
            if orient(tri[0], tri[1], tri[2]) == 0.0 {
                continue;
            }
            let mut points = [[0.0_f64; 2]; 12];
            let mut point_count = 0;
            for &[x, y] in tri {
                let x = x / source.units_per_parent_logical_unit;
                let y = y / source.units_per_parent_logical_unit;
                for dx in [-radius, radius] {
                    for dy in [-radius, radius] {
                        let point = [x + dx, y + dy];
                        if !point.iter().all(|n| n.is_finite()) {
                            return Err(precision(path, "nested Gaussian square loses precision"));
                        }
                        points[point_count] = point;
                        point_count += 1;
                    }
                }
            }
            debug_assert_eq!(point_count, points.len());
            points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
            let mut unique = 0;
            for index in 0..points.len() {
                let point = points[index];
                if unique == 0 || point != points[unique - 1] {
                    points[unique] = point;
                    unique += 1;
                }
            }
            let mut hull = [[0.0_f64; 2]; 24];
            let mut len = 0;
            for &point in &points[..unique] {
                while len >= 2 {
                    let turn = orient(hull[len - 2], hull[len - 1], point);
                    if !turn.is_finite() {
                        return Err(precision(path, "nested Gaussian hull orientation overflow"));
                    }
                    if turn > 0.0 {
                        break;
                    }
                    len -= 1;
                }
                hull[len] = point;
                len += 1;
            }
            let lower = len;
            for &point in points[..unique].iter().rev().skip(1) {
                while len > lower {
                    let turn = orient(hull[len - 2], hull[len - 1], point);
                    if !turn.is_finite() {
                        return Err(precision(path, "nested Gaussian hull orientation overflow"));
                    }
                    if turn > 0.0 {
                        break;
                    }
                    len -= 1;
                }
                hull[len] = point;
                len += 1;
            }
            len -= 1;
            if len < 3 {
                return Err(precision(path, "nested Gaussian hull collapsed"));
            }
            for index in 1..len - 1 {
                for point in [hull[0], hull[index], hull[index + 1]] {
                    expanded.bounds[0] = expanded.bounds[0].min(point[0]);
                    expanded.bounds[1] = expanded.bounds[1].min(point[1]);
                    expanded.bounds[2] = expanded.bounds[2].max(point[0]);
                    expanded.bounds[3] = expanded.bounds[3].max(point[1]);
                    expanded.triangles.push(point);
                }
            }
        }
        Ok(())
    })();
    if let Err(error) = result {
        // The failed output is dropped on return; only its temporary storage
        // lease is released. The invocation's already charged work remains.
        host.release(reserved);
        return Err(error);
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

    /// Original owner-frame reach for validity before invisible-paint elision.
    /// The consuming preparation owner prepays the interval envelope work.
    pub(super) fn effect_envelope(
        &self,
        effect: Render2dDropShadow,
    ) -> Result<Option<[f64; 4]>, Render2dExecutionError> {
        self.expression.effect_envelope(effect)
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
#[derive(Clone, Copy)]
pub(super) struct GaussianTarget {
    pub(super) canvas: [f64; 2],
    pub(super) raster_scale: f64,
    pub(super) physical_extent: [u32; 2],
}

pub(super) struct PreparedGaussianField {
    pub(super) plane: region::PreparedPlane,
    /// Completed parent-frame effect reach projected only for root cropping.
    pub(super) logical_envelope: [f64; 4],
}

pub(super) struct PreparedGaussianCaster {
    pub(super) planes: Vec<PreparedGaussianField>,
    /// Authored occurrence order; repeated geometry indexes one owned plane.
    pub(super) effect_planes: Vec<Option<usize>>,
}

impl PreparedGaussianCaster {
    fn retained_bytes(&self) -> Option<usize> {
        let mut bytes = self
            .planes
            .capacity()
            .checked_mul(size_of::<PreparedGaussianField>())?
            .checked_add(
                self.effect_planes
                    .capacity()
                    .checked_mul(size_of::<Option<usize>>())?,
            )?;
        for field in &self.planes {
            bytes = bytes.checked_add(field.plane.retained_bytes()?)?;
        }
        Some(bytes)
    }
}

pub(super) enum NeutralCaster<'a> {
    Gaussian(PreparedGaussianCaster),
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
    let diagnostic = graph.reserve_diagnostic_path(source.owner_path())?;
    match graph.materialize(source) {
        Ok(mesh) if original_straight => {
            graph.host.release(diagnostic);
            Ok(NeutralCaster::OriginalStraightPolygonUnion(mesh))
        }
        Ok(mesh) => {
            graph.host.release(diagnostic);
            Ok(NeutralCaster::Mesh(mesh))
        }
        Err(Render2dExecutionError::UnsupportedEntry {
            path,
            kind: Render2dUnsupportedContent::Shadows { .. },
        }) if !source.contains_positive_disk => {
            let actual = path.capacity() * size_of::<usize>();
            graph
                .host
                .verify_reserved_capacity(actual, diagnostic, &path)?;
            graph.host.release(diagnostic - actual);
            Ok(NeutralCaster::Symbolic(SymbolicCaster {
                expression: Arc::clone(source),
                first_unqualified_owner: path,
            }))
        }
        Err(error) => {
            graph.host.release(diagnostic);
            Err(error)
        }
    }
}

/// One immutable source-preparation result. Graph work has already been
/// admitted and paid; its consuming coverage policy must retain that charge.
pub(super) struct GroupChildSources<'a> {
    pub(super) casters: Vec<(usize, NeutralCaster<'a>)>,
    pub(super) work: super::support::PreparationBudget,
    pub(super) retained_bytes: usize,
    pub(super) host: super::support::PreparationHostBudget,
}

pub(super) const MAX_AUTHORED_SHADOWS: usize = 256;

/// Actual comparisons and record moves are charged before each operation.
/// Explicit insertion uses no allocator or implementation-dependent sort bound.
fn order_sources<T>(
    entries: &mut [(usize, T)],
    graph: &mut region::NeutralRegionBuilder,
) -> Result<(), Render2dExecutionError> {
    for first in 1..entries.len() {
        let mut position = first;
        while position > 0 {
            graph.charge_preparation(32, &[])?;
            if entries[position - 1].0 <= entries[position].0 {
                break;
            }
            graph.charge_preparation(3 * size_of::<(usize, T)>() + 32, &[])?;
            entries.swap(position - 1, position);
            position -= 1;
        }
    }
    Ok(())
}

impl<'a> GroupChildSources<'a> {
    pub(super) fn caster(&self, index: usize) -> Option<&NeutralCaster<'a>> {
        self.casters
            .binary_search_by_key(&index, |entry| entry.0)
            .ok()
            .map(|position| &self.casters[position].1)
    }
}

/// Linear owner of the original-source phase, including admission performed
/// before fields exist. Moving this into graph construction preserves every
/// paid debit; neither preparation nor failure can manufacture a fresh counter.
pub(super) struct SourcePreparation {
    graph: region::NeutralRegionBuilder,
    scales: NeutralPreparationScale,
    max_buffer_bytes: u64,
}

impl SourcePreparation {
    pub(super) fn new(scales: NeutralPreparationScale, max_buffer_bytes: u64) -> Self {
        let mut graph = region::NeutralRegionBuilder::new(scales, max_buffer_bytes);
        graph
            .charge_preparation(64, &[])
            .expect("fixed source owner setup fits work admission");
        graph
            .host
            .reserve(
                size_of::<Self>() - size_of::<region::NeutralRegionBuilder>(),
                &[],
            )
            .expect("fixed source owner fits host admission");
        Self {
            graph,
            scales,
            max_buffer_bytes,
        }
    }

    /// Complete authored cardinality admission precedes every font or source
    /// factory. This is the canonical lexical cursor, with the same source
    /// phase's 16M counter, including each skipped inert occurrence.
    pub(super) fn admit_effects(
        &mut self,
        composition: &Render2dComposition,
    ) -> Result<usize, Render2dExecutionError> {
        let bytes = scene::OccurrenceCursor::storage_bytes();
        self.graph.charge_preparation(64, &[])?;
        self.graph.host.reserve(bytes, &[])?;
        let result = self.admit_effects_inner(composition);
        self.graph.host.release(bytes);
        result
    }

    fn admit_effects_inner(
        &mut self,
        composition: &Render2dComposition,
    ) -> Result<usize, Render2dExecutionError> {
        let mut cursor = scene::OccurrenceCursor::new(composition, &mut |cost, path| {
            self.graph.charge_preparation(cost, path)
        })?;
        let mut effects = 0_usize;
        while let Some(event) =
            cursor.next(&mut |cost, path| self.graph.charge_preparation(cost, path))?
        {
            if let scene::Occurrence::BeginGroup { group, path } = event {
                self.graph.charge_preparation(64, path)?;
                effects = effects
                    .checked_add(group.shadows().len())
                    .ok_or_else(|| resource(path, "shadow effect count overflow"))?;
                if effects > MAX_AUTHORED_SHADOWS {
                    return Err(resource(
                        path,
                        "authored shadows exceed bounded effect admission",
                    ));
                }
            }
        }
        Ok(effects)
    }
}

#[cfg(test)]
pub(super) fn group_child_sources<'a>(
    plan: &scene::Plan<'a>,
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    scales: NeutralPreparationScale,
    max_buffer_bytes: u64,
    root_canvas: Option<([f64; 2], f64)>,
) -> Result<GroupChildSources<'a>, Render2dExecutionError> {
    build_group_sources(
        plan.composition(),
        plan.retained_bytes(),
        bindings,
        field_sets,
        SourcePreparation::new(scales, max_buffer_bytes),
    )?
    .prepare_polygonal_consumers(None, root_canvas, max_buffer_bytes)
}

/// Source-owned capacity bounds. Caller-owned F1 values and immutable resource
/// allocations are borrowed and must be inventoried by their owning caller.
/// Construction covers graph/frontier allocation lifetimes. Physical source
/// factory scratch (including Lyon) is a separate consumer-phase obligation.
#[derive(Clone, Copy, Debug)]
pub(super) struct SourceStorage {
    /// Caller-owned renderer input retained during this source phase. Current
    /// plan consumers pass their exact plan capacity; a caller preparing from
    /// borrowed F1 before planning passes zero. F1/resource payloads are separate.
    pub(super) input_bytes: usize,
    pub(super) retained_bytes: usize,
    pub(super) construction_peak_bytes: usize,
}

fn push_source_value<T>(
    values: &mut Vec<T>,
    value: T,
    graph: &mut region::NeutralRegionBuilder,
    construction: &mut usize,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    let grows = values.len() == values.capacity();
    let capacity = if grows {
        values.capacity().saturating_mul(2).max(4)
    } else {
        values.capacity()
    };
    let bytes = capacity
        .checked_mul(size_of::<T>())
        .ok_or_else(|| resource(path, "source container bytes overflow"))?;
    if u64::try_from(bytes)
        .ok()
        .is_none_or(|bytes| bytes > graph.storage_limit())
    {
        return Err(resource(path, "source container exceeds buffer capacity"));
    }
    let copies = if grows { values.len() } else { 0 };
    let work = copies
        .checked_add(1)
        .and_then(|v| v.checked_mul(size_of::<T>()))
        .and_then(|v| v.checked_mul(2))
        .and_then(|v| v.checked_add(128))
        .ok_or_else(|| resource(path, "source container work overflow"))?;
    graph.charge_preparation(work, path)?;
    if grows {
        *construction = construction
            .checked_add(bytes)
            .ok_or_else(|| resource(path, "source container construction storage overflow"))?;
        graph.host.reserve_vector(values, capacity, path)?;
    }
    values.push(value);
    Ok(())
}

fn copy_source_path(
    path: &[usize],
    graph: &mut region::NeutralRegionBuilder,
    construction: &mut usize,
) -> Result<Vec<usize>, Render2dExecutionError> {
    let bytes = path
        .len()
        .checked_mul(size_of::<usize>())
        .ok_or_else(|| resource(path, "source path storage overflow"))?;
    graph.charge_preparation(
        bytes
            .checked_mul(2)
            .and_then(|v| v.checked_add(128))
            .ok_or_else(|| resource(path, "source path copy work overflow"))?,
        path,
    )?;
    *construction = construction
        .checked_add(bytes)
        .ok_or_else(|| resource(path, "source path construction storage overflow"))?;
    if u64::try_from(bytes)
        .ok()
        .is_none_or(|bytes| bytes > graph.storage_limit())
    {
        return Err(resource(path, "source path exceeds buffer capacity"));
    }
    let mut result = Vec::new();
    graph.host.reserve_vector(&mut result, path.len(), path)?;
    result.extend_from_slice(path);
    Ok(result)
}

fn source_container_bytes(
    sources: &Vec<(usize, Arc<region::NeutralRegion<'_>>)>,
    demands: &Vec<SourceDemand<'_>>,
) -> Result<usize, Render2dExecutionError> {
    let mut bytes = sources
        .capacity()
        .checked_mul(size_of::<(usize, Arc<region::NeutralRegion<'_>>)>())
        .and_then(|v| {
            v.checked_add(
                demands
                    .capacity()
                    .checked_mul(size_of::<SourceDemand<'_>>())?,
            )
        })
        .ok_or_else(|| resource(&[], "source container storage overflow"))?;
    for demand in demands {
        if let SourceDemand::Caster { path, .. } = demand {
            bytes = bytes
                .checked_add(
                    path.capacity()
                        .checked_mul(size_of::<usize>())
                        .ok_or_else(|| resource(path, "source demand path bytes overflow"))?,
                )
                .ok_or_else(|| resource(path, "source demand path storage overflow"))?;
        }
    }
    Ok(bytes)
}

/// A complete borrowed-F1 source graph with optional physical realizations.
/// Polygon preparation is an explicit consumer; original-region consumers can
/// retain a source without constructing disposable Lyon geometry.
pub(super) struct GroupSourceDag<'scene, 'resources> {
    graph: region::NeutralRegionBuilder,
    sources: Vec<(usize, Arc<region::NeutralRegion<'scene>>)>,
    storage: SourceStorage,
    demands: Vec<SourceDemand<'scene>>,
    bindings: &'resources Render2dResourceBindings,
    field_sets: &'resources BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    scales: NeutralPreparationScale,
    max_buffer_bytes: u64,
}

enum SourceDemand<'a> {
    Leaf(Arc<region::NeutralRegion<'a>>),
    Caster {
        index: usize,
        group: &'a crate::composition_2d::Render2dGroup,
        path: Vec<usize>,
        nested: bool,
        parent_chart: Option<Render2dAffineTransform>,
    },
    Effect(Arc<region::NeutralRegion<'a>>),
}

impl<'scene> GroupSourceDag<'scene, '_> {
    /// A shadow-bearing group's pre-clip C in its immediate parent frame.
    /// The key is the original balanced plan's BeginGroup event index.
    pub(super) fn source(&self, begin_event: usize) -> Option<Arc<region::NeutralRegion<'scene>>> {
        self.sources
            .binary_search_by_key(&begin_event, |entry| entry.0)
            .ok()
            .map(|index| Arc::clone(&self.sources[index].1))
    }

    pub(super) fn graph_work(&self) -> usize {
        self.graph.graph_work
    }

    pub(super) fn storage(&self) -> SourceStorage {
        self.storage
    }

    /// Existing production consumer and eligibility remain unchanged. Keeping
    /// these demands in authored event order preserves physical errors and
    /// once-only cache/work ownership while original consumers need none.
    /// None requests every recorded physical demand, as the existing default
    /// consumer does. Some requests only the listed shadow-bearing BeginGroup
    /// occurrences and their dependencies; an empty set requests no polygons.
    /// Original consumers retain their source Arcs before consuming this DAG.
    /// The host lease bounds the selected source phase (input plan, DAG and
    /// dependency metadata), not subsequent physical factories or GPU work.
    pub(super) fn prepare_polygonal_consumers(
        self,
        selected: Option<&[usize]>,
        root_canvas: Option<([f64; 2], f64)>,
        max_host_bytes: u64,
    ) -> Result<GroupChildSources<'scene>, Render2dExecutionError> {
        self.prepare_polygonal_with_probe(selected, root_canvas, max_host_bytes, None, |_, _| {})
    }

    /// Select the bounded original numerical family before affine Plan. All
    /// remaining consumers keep their existing authored physical replay.
    pub(super) fn prepare_consumers(
        self,
        target: GaussianTarget,
    ) -> Result<GroupChildSources<'scene>, Render2dExecutionError> {
        self.prepare_polygonal_with_probe(
            None,
            Some((target.canvas, target.raster_scale)),
            super::support::MAX_SHADOW_HOST_BYTES as u64,
            Some(target),
            |_, _| {},
        )
    }

    fn selected_dependencies(
        &mut self,
        selected: &[usize],
        max_host_bytes: u64,
    ) -> Result<region::PolygonDependencies<'scene>, Render2dExecutionError> {
        // Prepay scalar receipt arithmetic even if admission then fails.
        self.graph.charge_preparation(32, &[])?;
        let source_phase = self
            .storage
            .input_bytes
            .checked_add(self.storage.construction_peak_bytes)
            .ok_or_else(|| resource(&[], "polygon source phase storage overflow"))?;
        let mut dependencies =
            self.graph
                .polygon_dependencies(self.sources.len(), source_phase, max_host_bytes)?;
        self.storage.construction_peak_bytes = self
            .storage
            .construction_peak_bytes
            .checked_add(dependencies.metadata_bytes())
            .ok_or_else(|| resource(&[], "polygon selection storage receipt overflow"))?;
        let lookup_work = self.consumer_lookup_work();
        // Validate the complete request before any physical callback. Repeated
        // group indices are idempotent set membership, not repeated demands.
        for &index in selected {
            self.graph.charge_preparation(lookup_work, &[])?;
            let position = self
                .sources
                .binary_search_by_key(&index, |entry| entry.0)
                .map_err(|_| precision(&[], "polygon consumer is not a registered shadow group"))?;
            dependencies.select_consumer(position);
        }
        for (position, (_, source)) in self.sources.iter().enumerate() {
            self.graph.charge_preparation(16, &[])?;
            if dependencies.consumer(position) {
                self.graph
                    .require_polygon_dependencies(source, &mut dependencies)?;
            }
        }
        Ok(dependencies)
    }

    fn consumer_lookup_work(&self) -> usize {
        // Bound two binary searches, Arc/flag bookkeeping and dispatch. The
        // second search is the existing source() lookup during caster replay.
        64 + 32 * (usize::BITS - self.sources.len().leading_zeros()) as usize
    }

    fn prepare_polygonal_with_probe(
        mut self,
        selected: Option<&[usize]>,
        root_canvas: Option<([f64; 2], f64)>,
        max_host_bytes: u64,
        gaussian: Option<GaussianTarget>,
        mut inspect: impl FnMut(&Arc<region::NeutralRegion<'scene>>, &[usize]),
    ) -> Result<GroupChildSources<'scene>, Render2dExecutionError> {
        let mut numerical = Vec::<usize>::new();
        let mut physical = Vec::<usize>::new();
        let mut deferred = Vec::<usize>::new();
        if let Some(target) = gaussian {
            self.graph
                .charge_preparation(256 + self.sources.len() * 64 + self.demands.len() * 16, &[])?;
            self.graph
                .host
                .reserve_vector(&mut numerical, self.sources.len(), &[])?;
            self.graph
                .host
                .reserve_vector(&mut physical, self.sources.len(), &[])?;
            for demand in &self.demands {
                if let SourceDemand::Caster {
                    index,
                    group,
                    path,
                    nested,
                    parent_chart,
                } = demand
                {
                    self.graph
                        .charge_preparation(1024 + group.shadows().len() * 1088, path)?;
                    let source = self.source(*index).expect("registered source");
                    // Existing certified absence wins before new capability
                    // selection; it must not acquire numerical allocations.
                    let absent = source.all_shadows_provably_empty(group.shadows())
                        || !nested
                            && source.all_root_shadows_provably_offscreen(
                                group.shadows(),
                                target.canvas,
                                target.raster_scale,
                            )?;
                    // This pays the chart proof even when another eligibility
                    // condition declines. No numerical failure is a fallback.
                    self.graph.charge_preparation(512, path)?;
                    let chart_qualified = parent_chart.is_some_and(|chart| {
                        let raw = chart.components();
                        // Existing complete chart proofs cover d=1. Other
                        // vertical maps await their own numerical evidence.
                        raw[3] == 1.0 && region::interval::exact_row_inverse(raw).is_some()
                    });
                    let candidate = chart_qualified
                        && !absent
                        && !group.shadows().is_empty()
                        && group
                            .shadows()
                            .iter()
                            .all(|effect| effect.spread() == 0.0 && effect.sigma() > 0.0)
                        && source.contains_negative_disk
                        && !source.contains_positive_disk
                        && self.graph.original_gaussian_required(&source)?;
                    if candidate {
                        numerical.push(*index);
                    } else {
                        physical.push(*index);
                    }
                }
            }
            // Nested closing order differs from original BeginGroup order.
            // Meter every comparison and move before binary-search consumers.
            for index in 1..numerical.len() {
                let mut position = index;
                while position > 0 {
                    self.graph.charge_preparation(32, &[])?;
                    if numerical[position - 1] <= numerical[position] {
                        break;
                    }
                    self.graph
                        .charge_preparation(3 * size_of::<usize>() + 32, &[])?;
                    numerical.swap(position - 1, position);
                    position -= 1;
                }
            }
            if !numerical.is_empty() {
                self.graph
                    .charge_preparation(128 + self.demands.len() * 64, &[])?;
                self.graph
                    .host
                    .reserve_vector(&mut deferred, self.sources.len(), &[])?;
                for demand in &self.demands {
                    if let SourceDemand::Caster {
                        index, group, path, ..
                    } = demand
                    {
                        self.graph.charge_preparation(
                            128 + group.shadows().len() * 16 + physical.len() * 16,
                            path,
                        )?;
                        if group
                            .shadows()
                            .iter()
                            .all(|effect| effect.color().channels()[3] == 0)
                            && let Some(position) = physical.iter().position(|entry| entry == index)
                        {
                            physical.swap_remove(position);
                            deferred.push(*index);
                        }
                    }
                }
            }
            self.graph
                .charge_preparation(64 + numerical.len() * 16 + physical.len() * 16, &[])?;
        }
        let selected = if !numerical.is_empty() {
            Some(physical.as_slice())
        } else {
            selected
        };
        let dependencies = selected
            .map(|indices| self.selected_dependencies(indices, max_host_bytes))
            .transpose()?;
        let lookup_work = self.consumer_lookup_work();
        // A fallibly reserved inventory exposes actual retained capacity;
        // no opaque allocator-owned tree nodes survive into Plan admission.
        self.graph.charge_preparation(128, &[])?;
        let mut sources = Vec::new();
        self.graph
            .host
            .reserve_vector(&mut sources, self.sources.len(), &[])?;
        for demand in std::mem::take(&mut self.demands) {
            if let Some(dependencies) = &dependencies {
                let needed = match &demand {
                    SourceDemand::Leaf(source) | SourceDemand::Effect(source) => {
                        self.graph.charge_preparation(16, source.owner_path())?;
                        dependencies.source(source)
                    }
                    SourceDemand::Caster { index, path, .. } => {
                        self.graph.charge_preparation(lookup_work, path)?;
                        let position = self
                            .sources
                            .binary_search_by_key(index, |entry| entry.0)
                            .expect("registered caster demand");
                        self.graph.charge_preparation(
                            64 + 32 * (usize::BITS - numerical.len().leading_zeros()) as usize,
                            path,
                        )?;
                        self.graph
                            .charge_preparation(64 + deferred.len() * 16, path)?;
                        dependencies.consumer(position)
                            || numerical.binary_search(index).is_ok()
                            || deferred.contains(index)
                    }
                };
                if !needed {
                    continue;
                }
            }
            match demand {
                SourceDemand::Leaf(source) => {
                    self.graph
                        .prepare_registered_source(&source, |item, path, work| {
                            neutral_item(
                                item,
                                path,
                                self.bindings,
                                self.field_sets,
                                self.scales,
                                self.max_buffer_bytes,
                                work,
                            )
                        })?;
                }
                SourceDemand::Caster {
                    index,
                    group,
                    path,
                    nested,
                    parent_chart,
                } => {
                    // Dependency selection and this authored caster dispatch
                    // are distinct visits. Prepay both membership searches on
                    // every outcome, including a failed/deferred match.
                    self.graph.charge_preparation(
                        64 + deferred.len() * 16
                            + 32 * (usize::BITS - numerical.len().leading_zeros()) as usize,
                        &path,
                    )?;
                    let in_parent = self.source(index).expect("registered group source");
                    inspect(&in_parent, &path);
                    // Resolve the proven curved DAG before asking for a
                    // polygonal mesh: a disk cannot be made into triangles
                    // and an expected rejection is not control flow.
                    let offscreen = if !nested {
                        if let Some((canvas, raster_scale)) = root_canvas {
                            // At most one interval halo projection per effect,
                            // plus two canvas divisions/setup. 512 covers the
                            // finite endpoint operations and loop comparisons.
                            let cost = group
                                .shadows()
                                .len()
                                .checked_mul(512)
                                .and_then(|cost| cost.checked_add(512))
                                .ok_or_else(|| resource(&path, "offscreen proof work overflow"))?;
                            self.graph.charge_preparation(cost, &path)?;
                            in_parent.all_root_shadows_provably_offscreen(
                                group.shadows(),
                                canvas,
                                raster_scale,
                            )?
                        } else {
                            false
                        }
                    } else {
                        false
                    };
                    // Complete erosion examines up to two interval half-spans
                    // per authored effect. Prepay every candidate, including
                    // the early-false and nested paths, on this same counter.
                    let cost = group
                        .shadows()
                        .len()
                        .checked_mul(512)
                        .and_then(|cost| cost.checked_add(64))
                        .ok_or_else(|| resource(&path, "complete erosion proof work overflow"))?;
                    self.graph.charge_preparation(cost, &path)?;
                    let source = if in_parent.all_shadows_provably_empty(group.shadows())
                        || offscreen
                    {
                        // Complete erosion proves absence of this group's
                        // own painted effects even when it is nested. Its
                        // pristine C remains in the separate ancestor DAG.
                        // Target offscreen admission is still root-only.
                        self.graph
                            .host
                            .reserve(size_of::<NeutralMesh>() + 2 * size_of::<usize>(), &path)?;
                        NeutralCaster::Mesh(Arc::new(empty_mesh()))
                    } else if deferred.contains(&index) {
                        // All original graph/resource/envelope validation is
                        // complete. Preserve C without requesting an invisible
                        // painted realization; ancestors still borrow the DAG.
                        self.graph.charge_preparation(
                            128 + path.len() * 2 + deferred.len() * 16,
                            &path,
                        )?;
                        let mut owner = Vec::new();
                        self.graph
                            .host
                            .reserve_vector(&mut owner, path.len(), &path)?;
                        owner.extend_from_slice(&path);
                        NeutralCaster::Symbolic(SymbolicCaster {
                            expression: Arc::clone(&in_parent),
                            first_unqualified_owner: owner,
                        })
                    } else if numerical.binary_search(&index).is_ok() {
                        NeutralCaster::Gaussian(self.graph.prepare_gaussian_caster(
                            &in_parent,
                            group.shadows(),
                            gaussian.expect("selected numerical target"),
                            parent_chart.expect("selected exact parent chart"),
                            &path,
                        )?)
                    } else if !nested {
                        match self.graph.qualified_rounded_rectangle(&in_parent)? {
                            Some(qualified) => NeutralCaster::RoundedRectangle(qualified),
                            None => {
                                // Root shadow paint is not exempt from
                                // original-F1 negative-erosion provenance.
                                // A convex Lyon tessellation never proves a
                                // curved F1 fill/stroke is polygonal support.
                                let proven =
                                    group.shadows().iter().any(|shadow| shadow.spread() < 0.0)
                                        && self
                                            .graph
                                            .proven_original_straight_polygon_union(&in_parent)?;
                                materialized_or_symbolic(&mut self.graph, &in_parent, proven)?
                            }
                        }
                    } else {
                        let exact_original_polygons =
                            group.shadows().iter().any(|effect| effect.spread() < 0.0)
                                && self
                                    .graph
                                    .proven_original_straight_polygon_union(&in_parent)?;
                        materialized_or_symbolic(
                            &mut self.graph,
                            &in_parent,
                            exact_original_polygons,
                        )?
                    };
                    self.graph.charge_preparation(
                        64 + 2 * size_of::<(usize, NeutralCaster<'_>)>(),
                        &path,
                    )?;
                    sources.push((index, source));
                }
                SourceDemand::Effect(effect_region) => {
                    let negative = effect_region.is_negative_effect();
                    let diagnostic = self
                        .graph
                        .reserve_diagnostic_path(effect_region.owner_path())?;
                    let disposition = match self.graph.materialize(&effect_region) {
                        Ok(_) => Ok(()),
                        Err(Render2dExecutionError::UnsupportedEntry {
                            path,
                            kind: Render2dUnsupportedContent::Shadows { .. },
                        }) if negative => self.graph.host.verify_reserved_capacity(
                            path.capacity() * size_of::<usize>(),
                            diagnostic,
                            &path,
                        ),
                        Err(error) => Err(error),
                    };
                    self.graph.host.release(diagnostic);
                    disposition?;
                }
            }
        }
        order_sources(&mut sources, &mut self.graph)?;
        let (mut retained_bytes, _) = self
            .graph
            .storage_bounds(self.sources.iter().map(|entry| &entry.1))?;
        retained_bytes = retained_bytes
            .checked_add(sources.capacity() * size_of::<(usize, NeutralCaster<'_>)>())
            .and_then(|v| v.checked_add(size_of::<GroupChildSources<'_>>()))
            .ok_or_else(|| resource(&[], "retained caster storage overflow"))?;
        let mut has_symbolic = false;
        for (_, source) in &sources {
            self.graph.charge_preparation(64, &[])?;
            has_symbolic |= matches!(
                source,
                NeutralCaster::Symbolic(_) | NeutralCaster::Gaussian(_)
            );
            if let NeutralCaster::Gaussian(caster) = source {
                self.graph
                    .charge_preparation(64 + caster.planes.len() * 128, &[])?;
                retained_bytes = retained_bytes
                    .checked_add(
                        caster
                            .retained_bytes()
                            .ok_or_else(|| resource(&[], "Gaussian caster retention overflow"))?,
                    )
                    .ok_or_else(|| resource(&[], "Gaussian source inventory overflow"))?;
            }
            let (mesh, path_capacity) = match source {
                NeutralCaster::Mesh(mesh) | NeutralCaster::OriginalStraightPolygonUnion(mesh) => {
                    (Some(mesh), 0)
                }
                NeutralCaster::RoundedRectangle(rounded) => {
                    (Some(&rounded.rectangle), rounded.curved_owner.capacity())
                }
                NeutralCaster::Symbolic(symbolic) => {
                    (None, symbolic.first_unqualified_owner.capacity())
                }
                NeutralCaster::Gaussian(_) => (None, 0),
            };
            let mesh_bytes = mesh
                .map_or(Some(0), |mesh| {
                    mesh.triangles
                        .capacity()
                        .checked_mul(size_of::<[f64; 2]>())
                        .and_then(|v| {
                            v.checked_add(size_of::<NeutralMesh>() + 2 * size_of::<usize>())
                        })
                })
                .ok_or_else(|| resource(&[], "caster mesh storage overflow"))?;
            // Cache/source sharing may count the same mesh twice. This is a
            // conservative live-capacity upper bound, never an undercount.
            retained_bytes = retained_bytes
                .checked_add(mesh_bytes)
                .and_then(|v| v.checked_add(path_capacity.checked_mul(size_of::<usize>())?))
                .ok_or_else(|| resource(&[], "caster storage total overflow"))?;
        }
        self.graph.charge_preparation(64, &[])?;
        let work = super::support::PreparationBudget::from_prepared_sources(
            has_symbolic,
            self.graph_work(),
        );
        // All graph/cache/frontier owners die at this cut. Keep the same
        // budget and its peak, trimming only to the conservative returned
        // caster inventory after those owners have actually been dropped.
        drop(dependencies);
        let selection_bytes =
            (numerical.capacity() + physical.capacity() + deferred.capacity()) * size_of::<usize>();
        drop(numerical);
        drop(physical);
        drop(deferred);
        self.graph.host.release(selection_bytes);
        drop(self.sources);
        let mut host = self.graph.into_host();
        host.retain(
            retained_bytes
                .checked_add(self.storage.input_bytes)
                .ok_or_else(|| resource(&[], "source retained host extent overflow"))?,
            &[],
        )?;
        Ok(GroupChildSources {
            casters: sources,
            work,
            retained_bytes,
            host,
        })
    }
}

/// Observe the default consumer after leaf geometry is prepared, preserving
/// the original test observer's physical-representation contract.
#[cfg(test)]
fn group_child_sources_with_probe<'a>(
    plan: &scene::Plan<'a>,
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    scales: NeutralPreparationScale,
    max_buffer_bytes: u64,
    root_canvas: Option<([f64; 2], f64)>,
    inspect: impl FnMut(&Arc<region::NeutralRegion<'a>>, &[usize]),
) -> Result<GroupChildSources<'a>, Render2dExecutionError> {
    build_group_sources(
        plan.composition(),
        plan.retained_bytes(),
        bindings,
        field_sets,
        SourcePreparation::new(scales, max_buffer_bytes),
    )?
    .prepare_polygonal_with_probe(None, root_canvas, max_buffer_bytes, None, inspect)
}

pub(super) fn build_group_sources<'scene, 'resources>(
    composition: &'scene Render2dComposition,
    input_bytes: usize,
    bindings: &'resources Render2dResourceBindings,
    field_sets: &'resources BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    preparation: SourcePreparation,
) -> Result<GroupSourceDag<'scene, 'resources>, Render2dExecutionError> {
    let mut active = Vec::<SourceFrame<'scene>>::new();
    let mut demands = Vec::new();
    let mut sources = Vec::new();
    let SourcePreparation {
        mut graph,
        scales,
        max_buffer_bytes,
    } = preparation;
    // Bound the host-owner/input transfer and cursor lease bookkeeping.
    graph.charge_preparation(128, &[])?;
    graph.host.reserve(input_bytes, &[])?;
    let owner_growth = size_of::<GroupSourceDag<'_, '_>>() - size_of::<SourcePreparation>();
    graph.host.reserve(owner_growth, &[])?;
    // This fixed host-only cursor is never uploaded to a native GPU buffer.
    // Include its exact bounded capacities in the construction inventory;
    // device buffer limits continue to govern the existing graph containers.
    // Whole-invocation host admission belongs to the caller's combined peak.
    graph.charge_preparation(32, &[])?;
    let cursor_bytes = scene::OccurrenceCursor::storage_bytes();
    graph.host.reserve(cursor_bytes, &[])?;
    let mut construction_bytes = cursor_bytes;
    let mut cursor = scene::OccurrenceCursor::new(composition, &mut |cost, path| {
        graph.charge_preparation(cost, path)
    })?;
    let mut event_count = 0_usize;
    while let Some(event) = cursor.next(&mut |cost, path| graph.charge_preparation(cost, path))? {
        let index = event_count;
        event_count = event_count
            .checked_add(1)
            .ok_or_else(|| resource(&[], "source event count overflow"))?;
        let event_path = match &event {
            scene::Occurrence::BeginGroup { path, .. } | scene::Occurrence::Item { path, .. } => {
                *path
            }
            scene::Occurrence::EndGroup => active.last().map_or(&[][..], |frame| &frame.path),
        };
        // Lexical cursor stepping and builder dispatch are separate work.
        // Include top-level items with no neutral-source consumer in both.
        graph.charge_preparation(64, event_path)?;
        match event {
            scene::Occurrence::BeginGroup { path, group } => {
                // The original lexical ancestry supplies a numerical receipt.
                // Rounded cumulative paint transforms are never source facts.
                graph.charge_preparation(512, path)?;
                let parent_chart = match active.last() {
                    None => Some(Render2dAffineTransform::IDENTITY),
                    Some(parent) => parent.parent_chart.and_then(|chart| {
                        let raw = region::interval::exact_affine_compose(
                            chart.components(),
                            parent.group.local_to_parent().components(),
                        )?;
                        Render2dAffineTransform::new(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5])
                            .ok()
                    }),
                };
                let frame_path = copy_source_path(path, &mut graph, &mut construction_bytes)?;
                push_source_value(
                    &mut active,
                    SourceFrame {
                        begin_event: index,
                        path: frame_path,
                        group,
                        parent_chart,
                        children: Vec::new(),
                    },
                    &mut graph,
                    &mut construction_bytes,
                    path,
                )?;
            }
            scene::Occurrence::Item { path, item } => {
                if active.is_empty() {
                    continue;
                }
                let leaf = if !matches!(item.primitive(), Render2dPrimitive::Image(_)) {
                    let leaf = if let Render2dPrimitive::ShapedText(primitive) = item.primitive() {
                        let (resource, fields) = shaped_source(
                            primitive,
                            bindings,
                            field_sets,
                            scales.field_raster_scale,
                            path,
                            &mut NeutralPreparationWork {
                                graph: &mut graph.graph_work,
                                clip_pairs: &mut graph.clip_work,
                                host: &mut graph.host,
                            },
                        )?;
                        graph.register_shaped_source(item, resource, fields, path)?
                    } else {
                        graph.register_source(item, path)?
                    };
                    if let Some(leaf) = &leaf {
                        push_source_value(
                            &mut demands,
                            SourceDemand::Leaf(Arc::clone(leaf)),
                            &mut graph,
                            &mut construction_bytes,
                            path,
                        )?;
                    }
                    leaf
                } else {
                    // Image destination support retains its existing factory.
                    graph.prepare_source_from_item(item, path, |work| {
                        neutral_item(
                            item,
                            path,
                            bindings,
                            field_sets,
                            scales,
                            max_buffer_bytes,
                            work,
                        )
                    })?
                };
                if let Some(leaf) = leaf {
                    push_source_value(
                        &mut active.last_mut().expect("active source group").children,
                        leaf,
                        &mut graph,
                        &mut construction_bytes,
                        path,
                    )?;
                }
            }
            scene::Occurrence::EndGroup => {
                let frame = active.pop().expect("F1 balanced occurrence cursor");
                let children = graph.union(frame.children, &frame.path)?;
                // A group first maps its entire local C into its immediate
                // parent frame. Only THEN do its Euclidean disk, offset and
                // finite Gaussian square operators act in that parent frame.
                let in_parent =
                    graph.affine(children, frame.group.local_to_parent(), &frame.path)?;
                let nested = !active.is_empty();
                if !frame.group.shadows().is_empty() {
                    push_source_value(
                        &mut sources,
                        (frame.begin_event, Arc::clone(&in_parent)),
                        &mut graph,
                        &mut construction_bytes,
                        &frame.path,
                    )?;
                    let demand_path =
                        copy_source_path(&frame.path, &mut graph, &mut construction_bytes)?;
                    push_source_value(
                        &mut demands,
                        SourceDemand::Caster {
                            index: frame.begin_event,
                            group: frame.group,
                            path: demand_path,
                            nested,
                            parent_chart: frame.parent_chart,
                        },
                        &mut graph,
                        &mut construction_bytes,
                        &frame.path,
                    )?;
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
                let mut components = Vec::new();
                push_source_value(
                    &mut components,
                    Arc::clone(&in_parent),
                    &mut graph,
                    &mut construction_bytes,
                    &frame.path,
                )?;
                if !frame.group.shadows().is_empty() {
                    // Do not reject a positive disk before its symbolic
                    // neutral support reaches the consuming ancestor's DAG.
                    // Polygonal materialization of that root still fails
                    // closed with the disk's exact original F1 owner path;
                    // no cached alpha, AABB or faceted substitute is admitted.
                    for &effect in frame.group.shadows() {
                        // Pay containment arithmetic and loop bookkeeping even
                        // when exact absorption omits the physical operator.
                        graph.charge_preparation(256, &frame.path)?;
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
                            // This is a compatibility demand, not graph meaning.
                            // Its existing typed failure authority runs only in
                            // the selected polygon/default preparation phase.
                            push_source_value(
                                &mut demands,
                                SourceDemand::Effect(Arc::clone(&effect_region)),
                                &mut graph,
                                &mut construction_bytes,
                                &frame.path,
                            )?;
                        }
                        // Positive disks remain symbolic until a certified
                        // ancestor region query is independently qualified;
                        // their own continuous paint path remains unchanged.
                        push_source_value(
                            &mut components,
                            effect_region,
                            &mut graph,
                            &mut construction_bytes,
                            &frame.path,
                        )?;
                    }
                }
                let completed = graph.union(components, &frame.path)?;
                let clipped = graph.clip(completed, frame.group.clips(), &frame.path)?;
                if let Some(parent) = active.last_mut() {
                    push_source_value(
                        &mut parent.children,
                        clipped,
                        &mut graph,
                        &mut construction_bytes,
                        &frame.path,
                    )?;
                }
            }
        }
    }
    debug_assert!(active.is_empty());
    drop(cursor);
    graph.host.release(cursor_bytes);
    order_sources(&mut sources, &mut graph)?;
    let roots = sources
        .iter()
        .map(|entry| &entry.1)
        .chain(demands.iter().filter_map(|demand| match demand {
            SourceDemand::Leaf(source) | SourceDemand::Effect(source) => Some(source),
            SourceDemand::Caster { .. } => None,
        }));
    let (node_bytes, construction) = graph.storage_bounds(roots)?;
    graph.charge_preparation(
        demands
            .len()
            .checked_mul(32)
            .and_then(|v| v.checked_add(event_count.checked_mul(32)?))
            .and_then(|v| v.checked_add(64))
            .ok_or_else(|| resource(&[], "source metadata scan work overflow"))?,
        &[],
    )?;
    let containers = source_container_bytes(&sources, &demands)?;
    let storage = SourceStorage {
        input_bytes,
        retained_bytes: node_bytes
            .checked_add(containers)
            .and_then(|v| v.checked_add(size_of::<GroupSourceDag<'_, '_>>()))
            .ok_or_else(|| resource(&[], "source retained storage overflow"))?,
        construction_peak_bytes: construction
            .checked_add(construction_bytes)
            .and_then(|v| v.checked_add(size_of::<GroupSourceDag<'_, '_>>()))
            .ok_or_else(|| resource(&[], "source construction storage overflow"))?,
    };
    let mut storage = storage;
    storage.construction_peak_bytes = storage
        .construction_peak_bytes
        .max(graph.host.peak().saturating_sub(input_bytes));
    let result = GroupSourceDag {
        graph,
        sources,
        storage,
        demands,
        bindings,
        field_sets,
        scales,
        max_buffer_bytes,
    };
    if result.storage().retained_bytes > result.storage().construction_peak_bytes {
        return Err(resource(
            &[],
            "source construction bound does not cover retained storage",
        ));
    }
    Ok(result)
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
    // This first pass is a complete, allocation-free precision preflight.
    // The destination must not change if a late vertex overflows or a
    // full-rank affine numerically collapses an authored positive triangle.
    // Repeating the same finite transform below avoids retaining a second,
    // invocation-unaccounted heap vector for all prepared vertices.
    let transform = |[x, y]: [f64; 2]| {
        let x = x / scale;
        let y = y / scale;
        [
            a.mul_add(x, c.mul_add(y, tx)),
            b.mul_add(x, d.mul_add(y, ty)),
        ]
    };
    for triangle in triangles.as_chunks::<3>().0 {
        let after = triangle.map(transform);
        if !after.iter().flatten().all(|v| v.is_finite()) {
            return Err(precision(
                path,
                "neutral parent-space transform lost precision",
            ));
        }
        if orient(after[0], after[1], after[2]) == 0.0
            && orient(triangle[0], triangle[1], triangle[2]) != 0.0
        {
            return Err(precision(
                path,
                "full-rank parent affine loses positive-area neutral geometry",
            ));
        }
    }
    output
        .triangles
        .try_reserve(triangles.len())
        .map_err(|_| resource(path, "neutral destination geometry allocation failed"))?;
    // No further fallible work: all source vertices have passed the exact
    // same floating-point conversion, and the destination has reserved room.
    for triangle in triangles.as_chunks::<3>().0 {
        let after = triangle.map(transform);
        if orient(after[0], after[1], after[2]) == 0.0 {
            continue;
        }
        for point in after {
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
    fn symbolic_transport_prepays_diagnostic_path_before_materialization() {
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: crate::composition_2d::Render2dShape::ellipse(
                    crate::composition_2d::Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap(),
                ),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = region::NeutralRegionBuilder::new(
            NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            1_048_576,
        );
        let source = graph
            .source_from_item(empty_mesh(), &item, &[3, 2])
            .unwrap();
        let fill = super::super::support::MAX_SHADOW_HOST_BYTES - graph.host.live();
        graph.host.reserve(fill, &[]).unwrap();
        let before = graph.host.live();
        assert!(
            matches!(materialized_or_symbolic(&mut graph, &source, false), Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), .. }) if path == [3, 2])
        );
        assert_eq!(graph.host.live(), before);
        graph.host.release(fill);
        let NeutralCaster::Symbolic(symbolic) =
            materialized_or_symbolic(&mut graph, &source, false).unwrap()
        else {
            panic!("original ellipse cannot become an empty physical source")
        };
        assert_eq!(symbolic.owner_path(), [3, 2]);
        assert!(graph.host.peak() >= graph.host.live());
    }

    #[test]
    fn source_input_reservation_precedes_cursor_or_graph_construction() {
        let composition =
            Render2dComposition::new(Vec::<crate::composition_2d::Render2dEntry>::new()).unwrap();
        let bindings = Render2dResourceBindings::default();
        let fields = BTreeMap::new();
        let result = build_group_sources(
            &composition,
            super::super::support::MAX_SHADOW_HOST_BYTES,
            &bindings,
            &fields,
            SourcePreparation::new(
                NeutralPreparationScale {
                    field_raster_scale: 1.0,
                    geometry_sample_scale: 4.0,
                },
                32,
            ),
        );
        assert!(
            matches!(result, Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), .. }) if path.is_empty())
        );
    }

    #[test]
    fn effect_admission_has_exact_late_owner_and_preserves_failed_debits() {
        use crate::composition_2d::{Render2dEntry, Render2dGroup};
        let effect =
            Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let group = |effects| {
            Render2dEntry::group(Render2dGroup::new(
                Vec::<Render2dEntry>::new(),
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
                vec![effect; effects],
            ))
        };
        let composition = Render2dComposition::new(vec![group(256), group(1)]).unwrap();
        assert!(composition.has_shadows());
        let scales = NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        };
        let mut preparation = SourcePreparation::new(scales, 1_048_576);
        assert!(
            matches!(preparation.admit_effects(&composition), Err(Render2dExecutionError::SampleSpace {
            kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
        }) if path == [1])
        );
        assert!(preparation.graph.graph_work > 0);
        let empty = Render2dComposition::new(Vec::<Render2dEntry>::new()).unwrap();
        let mut preparation = SourcePreparation::new(scales, 1_048_576);
        let before = MAX_COMPOSITIONAL_SHADOW_WORK - 200;
        preparation.graph.graph_work = before;
        assert!(preparation.admit_effects(&empty).is_err());
        let paid = preparation.graph.graph_work;
        assert!(paid > before && paid <= MAX_COMPOSITIONAL_SHADOW_WORK);
        assert!(preparation.admit_effects(&empty).is_err());
        assert_eq!(preparation.graph.graph_work, paid);
    }

    #[test]
    fn source_ordering_pays_full_records_before_movement() {
        let scales = NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        };
        let mut graph = region::NeutralRegionBuilder::new(scales, 1_048_576);
        let mut entries = [(2, [0_u8; 128]), (1, [1_u8; 128])];
        graph.graph_work = MAX_COMPOSITIONAL_SHADOW_WORK - 32;
        assert!(order_sources(&mut entries, &mut graph).is_err());
        assert_eq!(entries[0].0, 2);
        assert_eq!(graph.graph_work, MAX_COMPOSITIONAL_SHADOW_WORK);
        let mut graph = region::NeutralRegionBuilder::new(scales, 1_048_576);
        order_sources(&mut entries, &mut graph).unwrap();
        assert_eq!(entries[0], (1, [1_u8; 128]));
        assert!(graph.graph_work >= 3 * size_of_val(&entries[0]));
    }

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
                    host: &mut crate::runtime::execution_2d::support::PreparationHostBudget::new(),
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
                    host: &mut crate::runtime::execution_2d::support::PreparationHostBudget::new(),
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
    fn late_neutral_transform_overflow_preserves_destination_before_allocation() {
        let base = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            bounds: [0.0, 0.0, 1.0, 1.0],
        };
        let mut retained = empty_mesh();
        append(&mut retained, &base, Render2dAffineTransform::IDENTITY, &[6]).unwrap();
        let vertices = retained.triangles.clone();
        let bounds = retained.bounds;
        let capacity = retained.triangles.capacity();

        // The first triangle transforms to finite noncollapsed geometry.
        // Only a later triangle overflows the authored full-rank affine.
        // A partial destination write would violate prepare atomicity.
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [0.0, 0.0],
                [1.0e-300, 0.0],
                [0.0, 1.0],
                [10.0, 0.0],
                [11.0, 0.0],
                [10.0, 1.0],
            ],
            bounds: [0.0, 0.0, 11.0, 1.0],
        };
        let huge = Render2dAffineTransform::new(1.0e308, 0.0, 0.0, 1.0, 0.0, 0.0)
            .unwrap();
        assert!(matches!(
            append(&mut retained, &source, huge, &[6, 2]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [6, 2]
        ));
        assert_eq!(retained.triangles, vertices);
        assert_eq!(retained.bounds, bounds);
        assert_eq!(retained.triangles.capacity(), capacity);
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
        let plan = scene::analyze(&composition, &mut ()).unwrap();
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
        let NeutralCaster::Mesh(caster) = sources.caster(0).unwrap() else {
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
                &mut NeutralPreparationWork { graph: &mut exhausted, clip_pairs: &mut 0, host: &mut crate::runtime::execution_2d::support::PreparationHostBudget::new(), },
            ),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [0, 0]
        ));
        assert_eq!(exhausted, MAX_COMPOSITIONAL_SHADOW_WORK);
        let group = Render2dGroup::new(
            vec![Render2dEntry::item(glyph.clone())],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            vec![shadow],
        );
        let composition = Render2dComposition::new(vec![Render2dEntry::group(group)]).unwrap();
        // Original lexical preparation does not need the affine plan or a
        // polygon factory. The borrowed outlines certify reach first; the
        // later polygon consumer reconstructs the same physical mesh once.
        let original = build_group_sources(
            &composition,
            0,
            &bindings,
            &cached,
            SourcePreparation::new(
                NeutralPreparationScale {
                    field_raster_scale: 1.0,
                    geometry_sample_scale: 4.0,
                },
                1_048_576,
            ),
        )
        .unwrap();
        assert_eq!(original.storage().input_bytes, 0);
        let reach = original
            .source(0)
            .unwrap()
            .certified_logical_envelope()
            .unwrap();
        assert!(reach[0] < reach[2] && reach[1] < reach[3]);
        let original = original
            .prepare_polygonal_consumers(None, None, 1_048_576)
            .unwrap();
        let plan = scene::analyze(&composition, &mut ()).unwrap();
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
        let NeutralCaster::Mesh(mesh) = sources.caster(0).unwrap() else {
            panic!("outlined glyph caster must remain polygonal");
        };
        assert!(!mesh.triangles.is_empty());
        assert!(mesh.bounds[0].is_finite() && mesh.bounds[2] > mesh.bounds[0]);
        let NeutralCaster::Mesh(original_mesh) = original.caster(0).unwrap() else {
            panic!("pre-plan outlined glyph caster must remain polygonal");
        };
        assert_eq!(original_mesh.triangles, mesh.triangles);
        assert_eq!(original_mesh.bounds, mesh.bounds);

        let nested = Render2dGroup::new(
            vec![Render2dEntry::item(glyph)],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            vec![
                Render2dDropShadow::new(2000.0, 0.0, 0.1, -1000.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
            ],
        );
        let parent = Render2dGroup::new(
            vec![Render2dEntry::group(nested)],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            vec![
                Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
            ],
        );
        let nested = Render2dComposition::new(vec![Render2dEntry::group(parent)]).unwrap();
        let prepared = build_group_sources(
            &nested,
            0,
            &bindings,
            &cached,
            SourcePreparation::new(
                NeutralPreparationScale {
                    field_raster_scale: 1.0,
                    geometry_sample_scale: 4.0,
                },
                1_048_576,
            ),
        )
        .unwrap()
        .prepare_polygonal_consumers(None, None, 1_048_576)
        .unwrap();
        let NeutralCaster::Mesh(parent) = prepared.caster(0).unwrap() else {
            panic!("original ancestor support")
        };
        let NeutralCaster::Mesh(child) = prepared.caster(1).unwrap() else {
            panic!("complete nested erosion")
        };
        assert!(
            !parent.triangles.is_empty(),
            "a child's absent own shadow cannot erase its original support from the ancestor"
        );
        assert!(child.triangles.is_empty());
    }

    #[test]
    fn exact_gaussian_square_support_of_triangle_does_not_fill_its_aabb() {
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let mut host = super::super::support::PreparationHostBudget::new();
        let result_bytes = 30 * size_of::<[f64; 2]>();
        host.reserve(
            super::super::support::MAX_SHADOW_HOST_BYTES - result_bytes,
            &[1, 2],
        )
        .unwrap();
        let expanded = gaussian_square_support(&source, 0.5, &[1, 2], &mut host).unwrap();
        assert_eq!(host.live(), super::super::support::MAX_SHADOW_HOST_BYTES);
        assert_eq!(host.peak(), super::super::support::MAX_SHADOW_HOST_BYTES);
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
    fn gaussian_square_factory_rejects_one_short_before_result_allocation() {
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let bytes = 30 * size_of::<[f64; 2]>();
        let mut host = super::super::support::PreparationHostBudget::new();
        let previous = super::super::support::MAX_SHADOW_HOST_BYTES - bytes + 1;
        host.reserve(previous, &[1, 2]).unwrap();
        assert!(matches!(
            gaussian_square_support(&source, 0.5, &[1, 2], &mut host),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
        assert_eq!((host.live(), host.peak()), (previous, previous));
    }

    #[test]
    fn gaussian_square_factory_refunds_failed_result_storage() {
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[1.0e308, 0.0], [0.0, 1.0], [0.0, 0.0]],
            bounds: [0.0, 0.0, 1.0e308, 1.0],
        };
        let mut host = super::super::support::PreparationHostBudget::new();
        let before = host.live();
        let error = gaussian_square_support(&source, f64::MAX, &[1, 2], &mut host)
            .expect_err("overflowing translated vertex must fail after reservation");
        assert!(matches!(
            error,
            Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                ..
            }
        ));
        assert_eq!(host.live(), before);
        assert_eq!(host.peak(), 30 * size_of::<[f64; 2]>());
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
        let plan = scene::analyze(&composition, &mut ()).unwrap();
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
        let plan = scene::analyze(&scene, &mut ()).unwrap();
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
            sources.caster(0),
            Some(NeutralCaster::Symbolic(symbolic))
                if symbolic.owner_path() == [0, 0]
        ));
        assert!(matches!(
            sources.caster(1),
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
                    clip_pairs: &mut 0,
                    host: &mut crate::runtime::execution_2d::support::PreparationHostBudget::new(),
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
                host: &mut crate::runtime::execution_2d::support::PreparationHostBudget::new(),
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
                &mut NeutralPreparationWork { graph: &mut exhausted, clip_pairs: &mut clip_pairs, host: &mut crate::runtime::execution_2d::support::PreparationHostBudget::new(), },
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
                &mut NeutralPreparationWork { graph: &mut setup_only, clip_pairs: &mut clip_pairs, host: &mut crate::runtime::execution_2d::support::PreparationHostBudget::new(), }),
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
                host: &mut crate::runtime::execution_2d::support::PreparationHostBudget::new(),
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
                &mut NeutralPreparationWork { graph: &mut 0, clip_pairs: &mut used, host: &mut crate::runtime::execution_2d::support::PreparationHostBudget::new(), },
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
                host: &mut crate::runtime::execution_2d::support::PreparationHostBudget::new(),
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
