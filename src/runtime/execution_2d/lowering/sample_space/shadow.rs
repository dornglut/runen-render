//! F3F renderer-owned neutral shadow masks inside the ONE F3E correlated
//! linear-premultiplied 4x4 sample compositor. No separate color renderer,
//! sampled source alpha, shader module, scene owner or direct WGPU backend.
use super::*;
use crate::composition_2d::Render2dDropShadow;
use crate::runtime::execution_2d::{
    field::{FieldSetKey, ResourceFields},
    shadow as geometry_support,
    support::{self, NeutralCoverage},
};
use std::sync::Arc;
use support::PreparationWork;

mod symbolic;

// Retained floating neutral support is separate from per-tile source-backed
// GPU uploads; both are aggregate limits across one immutable contribution.
const MAX_SHADOW_COVERAGE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SHADOW_MASK_UPLOAD_BYTES: u64 = 128 * 1024 * 1024;
const MAX_AUTHORED_SHADOWS: usize = 256;
const FORMAT: GpuTextureFormat = GpuTextureFormat::R32Float;

pub(super) struct PreparedShadow {
    pub(super) color: [f32; 4],
    pub(super) coverage: Arc<NeutralCoverage>,
    /// The mathematically bounded 3σ support reach in the immediate-parent
    /// continuous logical frame, not a rectangle used as membership.
    pub(super) logical_envelope: [f64; 4],
    /// Clipped physical-pixel envelope, never geometric membership.
    pub(super) bounds: Option<[u32; 4]>,
    /// Group alpha is a visual admission fact; neutral support ignores it.
    pub(super) group_visible: bool,
    pub(super) path: Vec<usize>,
    /// Continuous cell area may remain positive when its center misses reach.
    /// Legacy center-sampled kernels retain their existing envelope guard.
    sample_area_coverage: bool,
}

/// Ordered physical preparations have a fallibly reserved inventory. Event
/// indices are emitted monotonically; lookup never allocates a tree node.
#[derive(Default)]
pub(super) struct ShadowGroups {
    entries: Vec<(usize, Vec<PreparedShadow>)>,
}

impl ShadowGroups {
    pub(super) fn get(&self, index: &usize) -> Option<&Vec<PreparedShadow>> {
        self.entries
            .binary_search_by_key(index, |entry| entry.0)
            .ok()
            .map(|position| &self.entries[position].1)
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &Vec<PreparedShadow>> {
        self.entries.iter().map(|entry| &entry.1)
    }

    fn insert(&mut self, index: usize, values: Vec<PreparedShadow>) {
        assert!(self.entries.last().is_none_or(|entry| entry.0 < index));
        assert!(self.entries.len() < self.entries.capacity());
        self.entries.push((index, values));
    }
}

/// Reserve all persistent occurrence headers and owner paths across the
/// complete contribution, including invisible occurrences. Payload capacity
/// is charged separately as each unique coverage buffer is prepared.
fn occurrence_metadata(
    plan: &scene::Plan<'_>,
    effects: usize,
) -> Result<u64, Render2dExecutionError> {
    let per_effect = std::mem::size_of::<PreparedShadow>()
        + std::mem::size_of::<NeutralCoverage>()
        + 2 * std::mem::size_of::<usize>();
    let mut bytes = effects
        .checked_mul(per_effect + std::mem::size_of::<(usize, Vec<PreparedShadow>)>())
        .ok_or_else(|| shadow_resource(&[], "shadow occurrence metadata extent overflow"))?;
    for event in &plan.events {
        if let scene::Event::BeginGroup { group, path, .. } = event {
            let paths = group
                .shadows()
                .len()
                .checked_mul(path.len())
                .and_then(|units| units.checked_mul(std::mem::size_of::<usize>()))
                .ok_or_else(|| {
                    shadow_resource(path, "shadow owner-path metadata extent overflow")
                })?;
            bytes = bytes
                .checked_add(paths)
                .ok_or_else(|| shadow_resource(path, "shadow metadata total overflow"))?;
        }
    }
    u64::try_from(bytes).map_err(|_| shadow_resource(&[], "shadow metadata exceeds byte counter"))
}

fn shadow_precision(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::PrecisionLimit,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    }
}

fn shadow_resource(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::ResourceLimit,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    }
}

fn intersection(a: [u32; 4], b: [u32; 4]) -> bool {
    a[0] < b[2] && a[2] > b[0] && a[1] < b[3] && a[3] > b[1]
}

/// Convert signed globally phased 4x sample-area bounds to conservative
/// physical pixel bounds. Only this derived extent is cropped to the output;
/// the parent-frame neutral caster itself is never prematurely clipped.
fn output_bounds(
    samples: &NeutralCoverage,
    target: &AdmittedTarget,
    path: &[usize],
) -> Result<Option<[u32; 4]>, Render2dExecutionError> {
    let upper_x = samples
        .origin_x
        .checked_add(
            i64::try_from(samples.width)
                .map_err(|_| shadow_resource(path, "shadow sample width exceeds signed index"))?,
        )
        .ok_or_else(|| shadow_precision(path, "shadow x sample extent overflow"))?;
    let upper_y = samples
        .origin_y
        .checked_add(
            i64::try_from(samples.height)
                .map_err(|_| shadow_resource(path, "shadow sample height exceeds signed index"))?,
        )
        .ok_or_else(|| shadow_precision(path, "shadow y sample extent overflow"))?;
    let upper_x = upper_x
        .checked_add(3)
        .ok_or_else(|| shadow_precision(path, "shadow x pixel ceiling overflow"))?;
    let upper_y = upper_y
        .checked_add(3)
        .ok_or_else(|| shadow_precision(path, "shadow y pixel ceiling overflow"))?;
    let crop = |value: i64, maximum: u32| {
        u32::try_from(value.clamp(0, i64::from(maximum)))
            .expect("cropped pixel bound is within u32")
    };
    let bounds = [
        crop(samples.origin_x.div_euclid(4), target.physical_width),
        crop(samples.origin_y.div_euclid(4), target.physical_height),
        crop(upper_x.div_euclid(4), target.physical_width),
        crop(upper_y.div_euclid(4), target.physical_height),
    ];
    Ok((bounds[0] < bounds[2] && bounds[1] < bounds[3]).then_some(bounds))
}

/// Validate the complete borrowed F1 plan's authored effects BEFORE paint
/// inspection, tessellation and source-neutral morphology can allocate.
/// The first over-budget owner keeps its exact F1 path.
pub(super) fn admit_effect_count(plan: &scene::Plan<'_>) -> Result<usize, Render2dExecutionError> {
    // Admit the total authored effect cardinality before materializing any
    // source-neutral geometry or running any morphology/convolution. Checking
    // group-by-group during preparation permits an inadmissible later sibling
    // to consume up to 256 expensive masks before the budget rejects it.
    let mut authored_effects = 0_usize;
    for event in &plan.events {
        let scene::Event::BeginGroup { group, path, .. } = event else {
            continue;
        };
        authored_effects = authored_effects
            .checked_add(group.shadows().len())
            .ok_or_else(|| shadow_resource(path, "shadow effect count overflow"))?;
        if authored_effects > MAX_AUTHORED_SHADOWS {
            return Err(shadow_resource(
                path,
                "authored shadows exceed bounded effect admission",
            ));
        }
    }
    Ok(authored_effects)
}

/// Establish the immutable invocation policy from the COMPLETE caster
/// inventory, retaining work already spent preparing those same sources.
fn admit_source_work(
    sources: &geometry_support::GroupChildSources<'_>,
) -> Result<support::PreparationBudget, Render2dExecutionError> {
    let has_symbolic = sources
        .casters
        .values()
        .any(|source| matches!(source, geometry_support::NeutralCaster::Symbolic(_)));
    let mut work = support::PreparationBudget::for_contribution(has_symbolic);
    work.charge(
        sources.graph_work,
        usize::MAX,
        &[],
        "neutral source graph exceeds aggregate shadow work admission",
    )?;
    Ok(work)
}

/// Independent F1 shadows each derive from the same immutable pre-shadow C.
/// This preflight runs before caller-target GPU work and retains exact authored
/// event paths; it is never a reinterpreted public support ontology.
pub(super) fn prepare(
    plan: &scene::Plan<'_>,
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    target: &AdmittedTarget,
    admitted_effects: usize,
) -> Result<ShadowGroups, Render2dExecutionError> {
    if admitted_effects == 0 {
        return Ok(ShadowGroups::default());
    }
    let sample_scale = target.raster_scale() * f64::from(SAMPLES);
    if !sample_scale.is_finite() || sample_scale <= 0.0 {
        return Err(shadow_precision(
            &[],
            "shadow group-parent sampling scale is invalid",
        ));
    }
    let metadata = occurrence_metadata(plan, admitted_effects)?;
    if metadata > MAX_SHADOW_COVERAGE_BYTES {
        return Err(shadow_resource(
            &[],
            "shadow occurrence metadata exceeds bounded host storage",
        ));
    }
    let sources = geometry_support::group_child_sources(
        plan,
        bindings,
        field_sets,
        geometry_support::NeutralPreparationScale {
            field_raster_scale: target.raster_scale(),
            geometry_sample_scale: sample_scale,
        },
        target.max_buffer_bytes(),
        Some((target.canvas(), target.raster_scale())),
    )?;
    let mut aggregate_coverage_work = admit_source_work(&sources)?;
    let sources = sources.casters;
    let mut groups = ShadowGroups::default();
    groups
        .entries
        .try_reserve_exact(admitted_effects)
        .map_err(|_| shadow_resource(&[], "shadow group inventory allocation failed"))?;
    if groups.entries.capacity() > admitted_effects {
        return Err(shadow_resource(
            &[],
            "shadow group inventory capacity exceeds admission",
        ));
    }
    let mut retained_coverage_bytes = metadata;
    // This is ONLY paint reachability. Alpha-neutral structural support for
    // ancestors is independently prepared from every borrowed F1 group,
    // including transparent effects and zero-opacity children.
    let mut paintable_ancestors = Vec::<bool>::new();
    for (index, event) in plan.events.iter().enumerate() {
        let (group, path, parent_to_root) = match event {
            scene::Event::BeginGroup {
                group,
                path,
                parent_to_root,
            } => (group, path, parent_to_root),
            scene::Event::EndGroup => {
                paintable_ancestors.pop().expect("balanced F1 group plan");
                continue;
            }
            scene::Event::Item { .. } => continue,
        };
        let paintable =
            paintable_ancestors.last().copied().unwrap_or(true) && group.opacity().get() > 0.0;
        paintable_ancestors.push(paintable);
        if group.shadows().is_empty() {
            continue;
        }
        let source = sources
            .get(&index)
            .expect("the neutral F1 traversal records each shadow-bearing group");
        if let geometry_support::NeutralCaster::Symbolic(source) = source {
            let [a, b, c, d, tx, ty] = parent_to_root.coefficients();
            let ancestor = Render2dAffineTransform::new(a, b, c, d, tx, ty).map_err(|_| {
                shadow_precision(path, "symbolic ancestor affine is unrepresentable")
            })?;
            let prepared = symbolic::prepare(
                source,
                group.shadows(),
                symbolic::Context {
                    parent_to_root: ancestor,
                    paintable,
                    target,
                    path,
                    aggregate_work: &mut aggregate_coverage_work,
                    retained_bytes: &mut retained_coverage_bytes,
                },
            )?;
            groups.insert(index, prepared);
            continue;
        }
        let (mesh, qualified_disk, original_polygons) = match source {
            geometry_support::NeutralCaster::Mesh(mesh) => (mesh.as_ref(), None, false),
            geometry_support::NeutralCaster::OriginalStraightPolygonUnion(mesh) => {
                (mesh.as_ref(), None, true)
            }
            geometry_support::NeutralCaster::RoundedRectangle(qualified) => (
                qualified.rectangle.as_ref(),
                Some((qualified.radius, qualified.curved_owner.as_slice())),
                false,
            ),
            geometry_support::NeutralCaster::Symbolic(symbolic) => {
                unreachable!(
                    "symbolic caster {} was prepared above",
                    symbolic.owner_path()[0]
                );
            }
        };
        let mut prepared = Vec::new();
        prepared
            .try_reserve_exact(group.shadows().len())
            .map_err(|_| shadow_resource(path, "shadow preparation list allocation failed"))?;
        if prepared.capacity() > group.shadows().len() {
            return Err(shadow_resource(
                path,
                "shadow occurrence capacity exceeds admission",
            ));
        }
        for effect in group.shadows() {
            let color = linear_color(effect.color());
            // Build the effect in its immediate-parent frame first.
            // Zero spread and zero blur commute with ancestor affine
            // projection, so a nested translated C remains exact triangles.
            // Unqualified nonpolygonal owner-frame kernels retain their
            // symbolic F1 DAG until this preflight boundary, then reject
            // rather than being incorrectly applied in root space.
            // Exact Euclidean Minkowski associativity:
            // (C union (C plus disk(r))) plus disk(s) =
            // C plus disk(r+s), for nonnegative s. Both radii act in
            // the same parent frame, before the ancestor shadow offset.
            // For a convex C, support-function inequalities give the exact
            // identity (C plus B_r) eroded by B_t =
            // C plus B_(r-t) when t<=r, else C eroded by B_(t-r).
            // The certified rectangular C is convex; this remains exact
            // in its one shared parent frame. A finite Gaussian convolves
            // the resulting support, never the child's paint alpha.
            // Nonrectangular or foreign-frame expressions fail closed.
            let effective = if let Some((radius, owner)) = qualified_disk {
                if path.len() != 1 {
                    return Err(crate::runtime::execution_2d::unsupported_at(
                        owner,
                        crate::execution_2d::Render2dUnsupportedContent::Shadows {
                            root_index: owner[0],
                        },
                    ));
                }
                let combined = radius + effect.spread();
                if !combined.is_finite() {
                    return Err(shadow_precision(
                        owner,
                        "combined Euclidean disk radius overflow",
                    ));
                }
                Render2dDropShadow::new(
                    effect.offset_x(),
                    effect.offset_y(),
                    effect.sigma(),
                    combined,
                    effect.color(),
                )
                .map_err(|_| {
                    shadow_precision(
                        owner,
                        "combined rounded neutral caster cannot be represented",
                    )
                })?
            } else {
                *effect
            };
            let Some(parent_envelope) = geometry_support::shadow_envelope(mesh, effective, path)?
            else {
                continue;
            };
            let parent_affine = if path.len() > 1 {
                let [a, b, c, d, tx, ty] = parent_to_root.coefficients();
                Some(
                    Render2dAffineTransform::new(a, b, c, d, tx, ty).map_err(|_| {
                        shadow_precision(path, "nested shadow ancestor affine is unrepresentable")
                    })?,
                )
            } else {
                None
            };
            let envelope = if let Some(ancestor) = parent_affine {
                geometry_support::transform_envelope(parent_envelope, ancestor, path)?
            } else {
                parent_envelope
            };
            let canvas = target.canvas();
            if envelope[2] <= 0.0
                || envelope[3] <= 0.0
                || envelope[0] >= canvas[0] / target.raster_scale()
                || envelope[1] >= canvas[1] / target.raster_scale()
            {
                continue;
            }
            // An effect with no painted contribution must NOT allocate its
            // physical lattice/halo. The independent group_child_sources
            // pass already preserved the exact neutral support for any
            // ancestor shadow, regardless of this alpha reachability.
            // Still compute/validate every authored parent-frame envelope
            // above; only the disposable raster work is elided.
            if !paintable || color[3] == 0.0 {
                continue;
            }
            // Even when a negative shadow is absorbed for ancestor
            // SUPPORT, its VISIBLE PAINT still needs a source-exact
            // Euclidean erosion. Never promote a convex Lyon mesh,
            // stroked curve, glyph outline or clipped F1 geometry into
            // an exact polygonal distance predicate. A qualified rounded
            // caster is separately proved from its original rectangle.
            if effective.spread() < 0.0 && !original_polygons && qualified_disk.is_none() {
                return Err(crate::runtime::execution_2d::unsupported_at(
                    path,
                    crate::execution_2d::Render2dUnsupportedContent::Shadows {
                        root_index: path[0],
                    },
                ));
            }
            let offset = Render2dAffineTransform::translation(effect.offset_x(), effect.offset_y())
                .map_err(|_| {
                    shadow_precision(path, "shadow offset transform is unrepresentable")
                })?;
            let mut shifted = geometry_support::empty_mesh();
            aggregate_coverage_work.charge(
                geometry_support::append_work(&shifted, mesh, path)?,
                usize::MAX,
                path,
                "shadow offset preparation exceeds invocation work",
            )?;
            geometry_support::append(&mut shifted, mesh, offset, path)?;
            // A certified negative Euclidean disk erosion has EXACT convex
            // polygonal geometry, so carry that into the already-owned
            // parent-frame continuous Gaussian / identity evaluator.
            // Nonconvex unions require original-F1 proof and an independently
            // admitted continuous parent-frame evaluator.
            let axis_isometry = parent_affine.is_some_and(|ancestor| {
                let [a, b, c, d, _, _] = ancestor.components();
                let sign = |value: f64| value == 1.0 || value == -1.0;
                (sign(a) && b == 0.0 && c == 0.0 && sign(d))
                    || (a == 0.0 && sign(b) && sign(c) && d == 0.0)
            });
            // Finite Gaussian negative erosion can be integrated in its
            // owning parent frame for a source-proven original F1 rectangle
            // union, even when an ancestor shears or scales that frame.
            // Sigma-zero negative erosion still needs an independently
            // certified inverse-affine pixel-area evaluator.
            let affine_negative = parent_affine.is_some()
                && original_polygons
                && effect.spread() < 0.0
                && effect.sigma() > 0.0
                && !axis_isometry;
            let erosion = if parent_affine.is_some() && effect.spread() < 0.0 {
                match support::certified_convex_erosion_mesh(
                    &shifted,
                    -effect.spread(),
                    path,
                    &mut aggregate_coverage_work,
                )? {
                    Some(convex) => Some(convex),
                    None if original_polygons && axis_isometry => {
                        // Signed axis permutation plus translation commutes
                        // with signed Euclidean disks AND the finite square
                        // Gaussian, preserving the
                        // original F1 rectangle union and let the already
                        // qualified root continuous orthogonal/nonconvex
                        // evaluator integrate in the translated root frame.
                        // This fast path is restricted to exact isometries;
                        // non-isometric parent-frame kernels are handled by
                        // the separate certified finite-Gaussian path only.
                        None
                    }
                    None if affine_negative => None,
                    None => {
                        return Err(crate::runtime::execution_2d::unsupported_at(
                            path,
                            crate::execution_2d::Render2dUnsupportedContent::Shadows {
                                root_index: path[0],
                            },
                        ));
                    }
                }
            } else {
                None
            };
            let caster = erosion.as_ref().unwrap_or(&shifted);
            if caster.triangles.is_empty() {
                continue;
            }
            let remaining_spread = if erosion.is_some() {
                0.0
            } else {
                effective.spread()
            };
            // A nested positive Euclidean disk and optional finite Gaussian
            // are evaluated in the owner's immediate-parent frame at
            // inverse-mapped globally phased ROOT samples. With zero sigma,
            // the ROOT unit sample cell's area is integrated under the
            // inverse-affine parallelogram, not guessed from its midpoint. Applying the
            // positive disk or Gaussian in root axes after ancestor shear
            // changes the F1 semantics and is not admitted.
            // Retained f64 neutral coverage is invocation-owned. Pass the
            // remaining scene budget through root and nested preparation:
            // late siblings must reject before allocating the float buffer.
            let remaining_coverage_samples = MAX_SHADOW_COVERAGE_BYTES
                .checked_sub(retained_coverage_bytes)
                .and_then(|remaining| usize::try_from(remaining / 8).ok())
                .ok_or_else(|| {
                    shadow_resource(path, "remaining retained coverage budget overflow")
                })?;
            let coverage = if let Some(ancestor) = parent_affine {
                if affine_negative
                    || (remaining_spread >= 0.0 && (effect.sigma() > 0.0 || remaining_spread > 0.0))
                {
                    let [a, b, c, d, _, _] = ancestor.components();
                    aggregate_coverage_work.charge(
                        128,
                        usize::MAX,
                        path,
                        "affine shadow rank proof exceeds invocation work",
                    )?;
                    if geometry_support::region::interval::exact_products_equal(a, d, b, c)
                        == Some(true)
                    {
                        // Rank-deficient ancestor transforms collapse 2D
                        // painted area; the source-neutral propagation pass
                        // separately discards degenerate affine triangles.
                        continue;
                    }
                    support::affine_parent_effect_coverage_admitted(
                        caster,
                        support::ParentEffectKernel {
                            signed_spread: remaining_spread,
                            sigma: effect.sigma(),
                        },
                        support::ParentEffectFrame {
                            to_root: ancestor,
                            root_envelope: envelope,
                            sample_scale,
                        },
                        path,
                        &mut aggregate_coverage_work,
                        remaining_coverage_samples,
                    )?
                } else {
                    let mut projected = geometry_support::empty_mesh();
                    aggregate_coverage_work.charge(
                        geometry_support::append_work(&projected, caster, path)?,
                        usize::MAX,
                        path,
                        "shadow ancestor preparation exceeds invocation work",
                    )?;
                    geometry_support::append(&mut projected, caster, ancestor, path)?;
                    support::prepare_shadow_coverage_admitted(
                        &projected,
                        remaining_spread,
                        effect.sigma(),
                        sample_scale,
                        path,
                        &mut aggregate_coverage_work,
                        remaining_coverage_samples,
                    )?
                }
            } else {
                support::prepare_shadow_coverage_admitted(
                    caster,
                    remaining_spread,
                    effect.sigma(),
                    sample_scale,
                    path,
                    &mut aggregate_coverage_work,
                    remaining_coverage_samples,
                )?
            };
            let Some(coverage) = coverage else {
                continue;
            };
            let bytes = u64::try_from(coverage.values.len())
                .ok()
                .and_then(|count| count.checked_mul(8))
                .ok_or_else(|| shadow_resource(path, "shadow floating coverage bytes overflow"))?;
            retained_coverage_bytes = retained_coverage_bytes
                .checked_add(bytes)
                .ok_or_else(|| shadow_resource(path, "aggregate shadow coverage bytes overflow"))?;
            if retained_coverage_bytes > MAX_SHADOW_COVERAGE_BYTES {
                return Err(shadow_resource(
                    path,
                    "retained neutral coverage exceeds bounded memory",
                ));
            }
            let bounds = output_bounds(&coverage, target, path)?;
            prepared.push(PreparedShadow {
                color,
                coverage: Arc::new(coverage),
                logical_envelope: envelope,
                bounds,
                group_visible: paintable,
                path: path.clone(),
                sample_area_coverage: false,
            });
        }
        groups.insert(index, prepared);
    }
    debug_assert!(paintable_ancestors.is_empty());
    Ok(groups)
}

pub(super) fn required_roles_admitted(context: &GpuContext) -> bool {
    let roles = context
        .device_facts()
        .admission_contract()
        .format_roles()
        .collect::<BTreeSet<_>>();
    [GpuFormatRole::Sampled, GpuFormatRole::CopyDestination]
        .into_iter()
        .all(|role| roles.contains(&(FORMAT, role)))
}

pub(super) fn visible(shadow: &PreparedShadow, tile: [u32; 4]) -> bool {
    shadow.group_visible
        && shadow.color[3] > 0.0
        && shadow
            .bounds
            .is_some_and(|bounds| intersection(bounds, tile))
}

/// Admission includes EACH immutable per-tile uploaded texture, not just one
/// reusable scratch handle that cannot in fact be overwritten in-flight.
/// Return the exact *work* for every admitted visible shadow: one draw node
/// plus all 4x mask samples written to its source-backed upload. The existing
/// compositor charges this alongside authored painter-tree replay. The
/// separate byte/node limits remain independent physical-resource ceilings.
pub(super) fn admit_tile_uploads(
    shadows: &ShadowGroups,
    target: &AdmittedTarget,
    side: u32,
    bounds: [u32; 4],
) -> Result<u64, Render2dExecutionError> {
    let dimension = side
        .checked_mul(SAMPLES)
        .ok_or_else(|| failure("shadow tile sample extent overflow"))?;
    if dimension > target.max_texture_dimension_2d() {
        return Err(failure("shadow mask sample texture exceeds device extent"));
    }
    let bytes = u64::from(dimension)
        .checked_mul(u64::from(dimension))
        .and_then(|cells| cells.checked_mul(4))
        .ok_or_else(|| failure("shadow per-tile upload extent overflow"))?;
    let origin_x = bounds[0] / side * side;
    let origin_y = bounds[1] / side * side;
    let sample_visits = u64::from(dimension)
        .checked_mul(u64::from(dimension))
        .ok_or_else(|| failure("shadow tile sample visit count overflow"))?;
    let per_upload_work = sample_visits
        .checked_add(1)
        .ok_or_else(|| failure("shadow tile upload/work addition overflow"))?;
    let mut total_bytes = 0_u64;
    let mut total_work = 0_u64;
    let mut count = 0_usize;
    for row in 0..u64::from((bounds[3] - origin_y).div_ceil(side)) {
        for col in 0..u64::from((bounds[2] - origin_x).div_ceil(side)) {
            let origin = [
                origin_x + u32::try_from(col).expect("bounded tile column") * side,
                origin_y + u32::try_from(row).expect("bounded tile row") * side,
            ];
            let tile = [
                origin[0],
                origin[1],
                origin[0].saturating_add(side).min(bounds[2]),
                origin[1].saturating_add(side).min(bounds[3]),
            ];
            if tile[0] >= tile[2] || tile[1] >= tile[3] {
                continue;
            }
            for shadow in shadows.values().flatten() {
                if !visible(shadow, tile) {
                    continue;
                }
                total_bytes = total_bytes
                    .checked_add(bytes)
                    .ok_or_else(|| shadow_resource(&shadow.path, "shadow upload byte overflow"))?;
                count = count
                    .checked_add(1)
                    .ok_or_else(|| shadow_resource(&shadow.path, "shadow upload node overflow"))?;
                total_work = total_work.checked_add(per_upload_work).ok_or_else(|| {
                    shadow_resource(&shadow.path, "shadow sample visits overflow")
                })?;
                if total_bytes > MAX_SHADOW_MASK_UPLOAD_BYTES || count > MAX_OPERATIONS / 2 {
                    return Err(shadow_resource(
                        &shadow.path,
                        "all-tile shadow upload/work budget exceeded",
                    ));
                }
            }
        }
    }
    Ok(total_work)
}

/// Source-backed R32Float mask: every 4x physical sample receives the ONE
/// precomputed parent-frame coverage value. The existing F3E
/// fs_sample_mask_fill resolves its alpha in the same Rgba16Float scratch
/// before the once-only group clip/opacity. No shadow-specific compositor.
pub(super) fn upload_tile(
    shadow: &PreparedShadow,
    origin: [u32; 2],
    dimension: u32,
    target: &AdmittedTarget,
    resources: &mut GpuResourceScope,
) -> Result<GpuTextureViewHandle, Render2dExecutionError> {
    let count = usize::try_from(dimension)
        .ok()
        .and_then(|side| side.checked_mul(side))
        .ok_or_else(|| shadow_resource(&shadow.path, "shadow tile sample count overflow"))?;
    let bytes = count
        .checked_mul(4)
        .ok_or_else(|| shadow_resource(&shadow.path, "shadow upload allocation overflow"))?;
    let mut data = Vec::new();
    data.try_reserve_exact(bytes)
        .map_err(|_| shadow_resource(&shadow.path, "shadow upload allocation failed"))?;
    if data.capacity() > bytes {
        return Err(shadow_resource(
            &shadow.path,
            "shadow upload capacity exceeds admitted tile storage",
        ));
    }
    let base_x = i64::from(origin[0]) * i64::from(SAMPLES);
    let base_y = i64::from(origin[1]) * i64::from(SAMPLES);
    let canvas = target.canvas();
    let width = usize::try_from(dimension).expect("bounded tile width");
    for row in 0..width {
        let y = base_y + i64::try_from(row).expect("bounded shadow row");
        for col in 0..width {
            let x = base_x + i64::try_from(col).expect("bounded shadow column");
            let sample_x = x as f64 + 0.5;
            let sample_y = y as f64 + 0.5;
            let logical_x = sample_x / (f64::from(SAMPLES) * target.raster_scale());
            let logical_y = sample_y / (f64::from(SAMPLES) * target.raster_scale());
            let [left, top, right, bottom] = shadow.logical_envelope;
            let value = if x < 0
                || y < 0
                || sample_x >= canvas[0] * f64::from(SAMPLES)
                || sample_y >= canvas[1] * f64::from(SAMPLES)
                || (!shadow.sample_area_coverage
                    && (logical_x < left
                        || logical_x > right
                        || logical_y < top
                        || logical_y > bottom))
            {
                0.0
            } else {
                let sx = x
                    .checked_sub(shadow.coverage.origin_x)
                    .and_then(|delta| usize::try_from(delta).ok());
                let sy = y
                    .checked_sub(shadow.coverage.origin_y)
                    .and_then(|delta| usize::try_from(delta).ok());
                match (sx, sy) {
                    (Some(sx), Some(sy))
                        if sx < shadow.coverage.width && sy < shadow.coverage.height =>
                    {
                        shadow.coverage.values[sy * shadow.coverage.width + sx]
                    }
                    _ => 0.0,
                }
            };
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                return Err(shadow_precision(
                    &shadow.path,
                    "shadow coverage is not normalized",
                ));
            }
            let narrow = value as f32;
            if value > 0.0 && narrow == 0.0 {
                return Err(shadow_precision(
                    &shadow.path,
                    "shadow coverage cannot survive GPU mask precision",
                ));
            }
            data.extend_from_slice(&narrow.to_le_bytes());
        }
    }
    let name = format!("runen-render 4x shadow {}", shadow.path[0]);
    let label = GpuResourceLabel::new(&name).map_err(|e| gpu("shadow mask label", e))?;
    // RunenGPU copies this Vec into a transfer Vec, then an immutable Arc.
    // During conversion all three payloads coexist, alongside every prior
    // tile's descriptor-retained upload and the retained floating coverage.
    // The all-tile payload admission above therefore is not reusable scratch.
    let source = PreparedGpuData::<TransferData>::ordinary_pod_transfer(&name, &data)
        .map_err(|e| gpu("shadow float mask bytes", e))?;
    drop(data);
    let extent = GpuTextureExtent::new(&label, GpuTextureDimension::D2, dimension, dimension, 1)
        .map_err(|e| gpu("shadow mask extent", e))?;
    let row_bytes = dimension
        .checked_mul(4)
        .ok_or_else(|| shadow_resource(&shadow.path, "shadow mask row overflow"))?;
    let prepared = GpuPreparedTextureData::new(&label, source, FORMAT, extent, row_bytes, 0)
        .map_err(|e| gpu("shadow prepared mask", e))?;
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                &name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                dimension,
                dimension,
                FORMAT,
                [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
                GpuTextureInitialization::Prepared(prepared),
            )
            .map_err(|e| gpu("shadow mask descriptor", e))?,
        )
        .map_err(|e| gpu("shadow mask texture", e))?;
    resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(format!("{name} view"), &texture)
                .map_err(|e| gpu("shadow mask view descriptor", e))?,
        )
        .map_err(|e| gpu("shadow mask view", e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::*;
    use crate::execution_2d::Render2dSampleSpaceError;

    fn rectangle([x, y, width, height]: [f64; 4]) -> Render2dEntry {
        Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(x, y, width, height).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        ))
    }

    fn legacy_group() -> Render2dEntry {
        Render2dEntry::group(Render2dGroup::new(
            vec![rectangle([1.0, 1.0, 1.0, 1.0])],
            Render2dAffineTransform::IDENTITY,
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
        ))
    }

    fn symbolic_group() -> Render2dEntry {
        let child = Render2dEntry::group(Render2dGroup::new(
            [
                [10.0, 10.0, 20.0, 8.0],
                [10.0, 22.0, 20.0, 8.0],
                [10.0, 18.0, 8.0, 4.0],
                [22.0, 18.0, 8.0, 4.0],
            ]
            .into_iter()
            .map(rectangle)
            .collect::<Vec<_>>(),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            vec![
                Render2dDropShadow::new(8.0, 0.0, 0.03125, -1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
            ],
        ));
        Render2dEntry::group(Render2dGroup::new(
            vec![child],
            Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap(),
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
        ))
    }

    fn sources<'a>(plan: &scene::Plan<'a>) -> geometry_support::GroupChildSources<'a> {
        geometry_support::group_child_sources(
            plan,
            &Render2dResourceBindings::default(),
            &BTreeMap::new(),
            geometry_support::NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            16 * 1024 * 1024,
            Some(([64.0, 64.0], 1.0)),
        )
        .unwrap()
    }

    #[test]
    fn mixed_graph_receipt_leaves_the_same_coverage_allowance_in_both_orders() {
        for symbolic_first in [false, true] {
            let entries = if symbolic_first {
                vec![symbolic_group(), legacy_group()]
            } else {
                vec![legacy_group(), symbolic_group()]
            };
            let composition = Render2dComposition::new(entries).unwrap();
            let plan = scene::analyze(&composition).unwrap();
            let mut sources = sources(&plan);
            assert!(
                sources.graph_work > 0,
                "actual graph/proof work must leave a receipt"
            );
            assert!(
                sources
                    .casters
                    .values()
                    .any(|s| matches!(s, geometry_support::NeutralCaster::Symbolic(_)))
            );
            assert_eq!(
                admit_source_work(&sources).unwrap().used(),
                sources.graph_work
            );
            // The graph stage already spent all but one unit. Coverage cannot
            // receive a fresh 16M merely because the stage changed.
            sources.graph_work = support::MAX_COMPOSITIONAL_SHADOW_WORK - 1;
            let mut work = admit_source_work(&sources).unwrap();
            assert_eq!(work.remaining(usize::MAX), 1);
            let (&index, mesh) = sources
                .casters
                .iter()
                .find_map(|(index, source)| match source {
                    geometry_support::NeutralCaster::Mesh(mesh) => Some((index, mesh)),
                    _ => None,
                })
                .unwrap();
            let scene::Event::BeginGroup { path, .. } = &plan.events[index] else {
                panic!("source owns a group");
            };
            let error = support::prepare_shadow_coverage_admitted(
                mesh, 0.0, 0.0, 4.0, path, &mut work, 1_048_576,
            )
            .unwrap_err();
            assert!(matches!(error, Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(owner), ..
            } if owner == *path));
            assert_eq!(
                work.used(),
                sources.graph_work,
                "reject before coverage allocation/work"
            );
        }
    }

    #[test]
    fn legacy_graph_receipt_is_paid_without_changing_legacy_policy() {
        let composition = Render2dComposition::new(vec![legacy_group()]).unwrap();
        let plan = scene::analyze(&composition).unwrap();
        let mut sources = sources(&plan);
        assert!(sources.graph_work > 0);
        assert_eq!(
            admit_source_work(&sources).unwrap().used(),
            sources.graph_work
        );
        sources.graph_work = support::MAX_COMPOSITIONAL_SHADOW_WORK;
        let mut work = admit_source_work(&sources).unwrap();
        let geometry_support::NeutralCaster::Mesh(mesh) = sources.casters.get(&0).unwrap() else {
            panic!("legacy source is polygonal");
        };
        let coverage = support::prepare_shadow_coverage_admitted(
            mesh,
            0.0,
            0.0,
            4.0,
            &[0],
            &mut work,
            1_048_576,
        )
        .unwrap()
        .unwrap();
        assert!(work.used() > sources.graph_work);
        assert!((coverage.values.iter().sum::<f64>() - 16.0).abs() < 1.0e-12);
    }
}
