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
use geometry_support::MAX_AUTHORED_SHADOWS;
use std::sync::Arc;
use support::PreparationWork;

mod symbolic;

// Retained floating neutral support is separate from per-tile source-backed
// GPU uploads; both are aggregate limits across one immutable contribution.
pub(super) const MAX_SHADOW_COVERAGE_BYTES: u64 = support::MAX_SHADOW_HOST_BYTES as u64;
pub(super) const MAX_SHADOW_MASK_UPLOAD_BYTES: u64 = 128 * 1024 * 1024;
const FORMAT: GpuTextureFormat = GpuTextureFormat::R32Float;

pub(super) struct PreparedShadow {
    pub(super) color: [f32; 4],
    pub(super) coverage: ShadowCoverage,
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

#[derive(Clone)]
pub(super) enum ShadowCoverage {
    Samples(Arc<NeutralCoverage>),
    Gaussian(usize),
}

pub(super) struct NumericalField {
    pub(super) plane: geometry_support::region::PreparedPlane,
    pub(super) visible_extent: [u32; 2],
    logical_envelope: [f64; 4],
    pub(super) path: Vec<usize>,
}

struct PreparationAdmission {
    work: support::PreparationBudget,
    host: support::PreparationHostBudget,
}

/// Ordered physical preparations have a fallibly reserved inventory. Event
/// indices are emitted monotonically; lookup never allocates a tree node.
#[derive(Default)]
pub(super) struct ShadowGroups {
    entries: Vec<(usize, Vec<PreparedShadow>)>,
    pub(super) numerical: Vec<NumericalField>,
    admission: Option<PreparationAdmission>,
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

/// Prepared once before affine planning. This owning state transports the
/// complete caster inventory, its selected non-copyable work policy and the
/// concurrently live source/Plan host lease into coverage lowering.
pub(in crate::runtime::execution_2d) struct PreparedSources<'a> {
    composition: &'a crate::composition_2d::Render2dComposition,
    sources: geometry_support::GroupChildSources<'a>,
    effects: usize,
}

impl<'a> PreparedSources<'a> {
    fn new(
        composition: &'a crate::composition_2d::Render2dComposition,
        mut sources: geometry_support::GroupChildSources<'a>,
        effects: usize,
    ) -> Result<Self, Render2dExecutionError> {
        sources.host.reserve(
            size_of::<Self>() - size_of::<geometry_support::GroupChildSources<'_>>(),
            &[],
        )?;
        Ok(Self {
            composition,
            sources,
            effects,
        })
    }

    /// Failure consumes and abandons the complete owner, including any
    /// partially reserved storage. A caller cannot retry it with a reset lease.
    pub(in crate::runtime::execution_2d) fn plan(
        mut self,
    ) -> Result<(scene::Plan<'a>, Self), Render2dExecutionError> {
        let plan = scene::analyze(self.composition, &mut self)?;
        self.finish_planning(&plan)?;
        Ok((plan, self))
    }

    fn finish_planning(&mut self, plan: &scene::Plan<'_>) -> Result<(), Render2dExecutionError> {
        self.sources.work.charge(
            64,
            usize::MAX,
            &[],
            "shadow plan receipt exceeds invocation work",
        )?;
        let expected = self
            .sources
            .retained_bytes
            .checked_add(size_of::<Self>() - size_of::<geometry_support::GroupChildSources<'_>>())
            .and_then(|bytes| bytes.checked_add(plan.retained_bytes()))
            .ok_or_else(|| shadow_resource(&[], "shadow plan receipt storage overflow"))?;
        if expected != self.sources.host.live() {
            return Err(shadow_precision(
                &[],
                "shadow planner retained storage does not match lease",
            ));
        }
        Ok(())
    }
}

impl scene::PlanningAdmission for PreparedSources<'_> {
    fn charge(&mut self, work: usize, path: &[usize]) -> Result<(), Render2dExecutionError> {
        self.sources.work.charge(
            work,
            usize::MAX,
            path,
            "shadow scene planning exceeds invocation work",
        )
    }
    fn reserve(&mut self, bytes: usize, path: &[usize]) -> Result<(), Render2dExecutionError> {
        self.sources.host.reserve(bytes, path)
    }
    fn release(&mut self, bytes: usize) {
        self.sources.host.release(bytes);
    }
}

pub(in crate::runtime::execution_2d) fn prepare_sources<'scene>(
    composition: &'scene crate::composition_2d::Render2dComposition,
    bindings: &Render2dResourceBindings,
    field_sets: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
    target: &AdmittedTarget,
    preparation: geometry_support::SourcePreparation,
    effects: usize,
) -> Result<PreparedSources<'scene>, Render2dExecutionError> {
    let sources =
        geometry_support::build_group_sources(composition, 0, bindings, field_sets, preparation)?
            .prepare_consumers(geometry_support::GaussianTarget {
            canvas: target.canvas(),
            raster_scale: target.raster_scale(),
            physical_extent: [target.physical_width, target.physical_height],
        })?;
    PreparedSources::new(composition, sources, effects)
}

/// Independent F1 shadows each derive from the same immutable pre-shadow C.
/// This preflight runs before caller-target GPU work and retains exact authored
/// event paths; it is never a reinterpreted public support ontology.
pub(super) fn prepare(
    plan: &scene::Plan<'_>,
    target: &AdmittedTarget,
    prepared: PreparedSources<'_>,
) -> Result<ShadowGroups, Render2dExecutionError> {
    let live_bytes = prepared.sources.host.live();
    let PreparedSources {
        sources,
        effects: admitted_effects,
        ..
    } = prepared;
    let sample_scale = target.raster_scale() * f64::from(SAMPLES);
    if !sample_scale.is_finite() || sample_scale <= 0.0 {
        return Err(shadow_precision(
            &[],
            "shadow group-parent sampling scale is invalid",
        ));
    }
    let metadata = occurrence_metadata(plan, admitted_effects)?;
    let mut retained_coverage_bytes = (live_bytes as u64)
        .checked_add(metadata)
        .ok_or_else(|| shadow_resource(&[], "shadow metadata host storage overflow"))?;
    if retained_coverage_bytes > MAX_SHADOW_COVERAGE_BYTES {
        return Err(shadow_resource(
            &[],
            "shadow metadata exceeds remaining host storage",
        ));
    }
    let geometry_support::GroupChildSources {
        casters: sources,
        work: mut aggregate_coverage_work,
        mut host,
        ..
    } = sources;
    host.reserve(metadata as usize, &[])?;
    let mut groups = ShadowGroups::default();
    let inventory_work = sources
        .len()
        .checked_mul(32)
        .and_then(|work| work.checked_add(128))
        .ok_or_else(|| shadow_resource(&[], "numerical field inventory work overflow"))?;
    aggregate_coverage_work.charge(
        inventory_work,
        usize::MAX,
        &[],
        "numerical field inventory exceeds invocation work",
    )?;
    let field_count = sources
        .iter()
        .try_fold(0_usize, |count, (_, source)| {
            let fields = match source {
                geometry_support::NeutralCaster::Gaussian(caster) => caster.planes.len(),
                _ => 0,
            };
            count.checked_add(fields)
        })
        .ok_or_else(|| shadow_resource(&[], "numerical field inventory overflow"))?;
    host.reserve_vector(&mut groups.numerical, field_count, &[])?;
    let mut sources = sources.into_iter();
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
        retained_coverage_bytes = host.live() as u64;
        let (source_index, source) = sources
            .next()
            .expect("the neutral F1 traversal records each shadow-bearing group");
        assert_eq!(source_index, index, "caster and painter event order agree");
        if let geometry_support::NeutralCaster::Gaussian(caster) = source {
            let prepared = prepare_numerical(
                caster,
                group,
                &mut groups.numerical,
                NumericalPreparation {
                    path,
                    paintable,
                    target,
                    work: &mut aggregate_coverage_work,
                    host: &mut host,
                },
            )?;
            groups.insert(index, prepared);
            continue;
        }
        if let geometry_support::NeutralCaster::Symbolic(source) = &source {
            let [a, b, c, d, tx, ty] = parent_to_root.coefficients();
            let ancestor = Render2dAffineTransform::new(a, b, c, d, tx, ty).map_err(|_| {
                shadow_precision(path, "symbolic ancestor affine is unrepresentable")
            })?;
            let before = retained_coverage_bytes;
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
            host.reserve((retained_coverage_bytes - before) as usize, path)?;
            groups.insert(index, prepared);
            continue;
        }
        let (mesh, qualified_disk, original_polygons) = match &source {
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
            geometry_support::NeutralCaster::Gaussian(_) => {
                unreachable!("numerical caster prepared above")
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
            host.reserve(bytes as usize, path)?;
            let bounds = output_bounds(&coverage, target, path)?;
            prepared.push(PreparedShadow {
                color,
                coverage: ShadowCoverage::Samples(Arc::new(coverage)),
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
    debug_assert!(sources.next().is_none());
    groups.admission = Some(PreparationAdmission {
        work: aggregate_coverage_work,
        host,
    });
    Ok(groups)
}

struct NumericalPreparation<'a> {
    path: &'a [usize],
    paintable: bool,
    target: &'a AdmittedTarget,
    work: &'a mut support::PreparationBudget,
    host: &'a mut support::PreparationHostBudget,
}

fn prepare_numerical(
    caster: geometry_support::PreparedGaussianCaster,
    group: &Render2dGroup,
    fields: &mut Vec<NumericalField>,
    context: NumericalPreparation<'_>,
) -> Result<Vec<PreparedShadow>, Render2dExecutionError> {
    let NumericalPreparation {
        path,
        paintable,
        target,
        work,
        host,
    } = context;
    work.charge(
        128 + group.shadows().len() * 32,
        usize::MAX,
        path,
        "numerical authored effect traversal exceeds invocation work",
    )?;
    if caster.effect_planes.len() != group.shadows().len() {
        return Err(shadow_precision(
            path,
            "numerical effect inventory does not match authored order",
        ));
    }
    // Every original source/kernel was validated before this paint-only cut.
    if !paintable
        || group
            .shadows()
            .iter()
            .all(|effect| effect.color().channels()[3] == 0)
    {
        return Ok(Vec::new());
    }
    let first = fields.len();
    for field in caster.planes {
        work.charge(
            256 + path.len() * 2,
            usize::MAX,
            path,
            "numerical field transfer exceeds invocation work",
        )?;
        let visible_extent =
            super::gaussian::canvas_extent(target.canvas(), field.plane.root(), work, path)?;
        let mut owner = Vec::new();
        host.reserve_vector(&mut owner, path.len(), path)?;
        owner.extend_from_slice(path);
        assert!(
            fields.len() < fields.capacity(),
            "numerical field inventory was reserved"
        );
        fields.push(NumericalField {
            plane: field.plane,
            visible_extent,
            logical_envelope: field.logical_envelope,
            path: owner,
        });
    }
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(group.shadows().len())
        .map_err(|_| shadow_resource(path, "numerical effect inventory allocation failed"))?;
    if prepared.capacity() > group.shadows().len() {
        return Err(shadow_resource(
            path,
            "numerical effect capacity exceeds admission",
        ));
    }
    for (effect, field) in group.shadows().iter().zip(caster.effect_planes) {
        let color = linear_color(effect.color());
        if color[3] == 0.0 {
            continue;
        }
        let Some(field) = field else { continue };
        let index = first
            .checked_add(field)
            .filter(|&index| index < fields.len())
            .ok_or_else(|| {
                shadow_precision(path, "numerical effect references an unknown field")
            })?;
        work.charge(
            256 + path.len() * 2,
            usize::MAX,
            path,
            "numerical effect preparation exceeds invocation work",
        )?;
        let root = fields[index].plane.root();
        let bounds = [
            root[0] / SAMPLES,
            root[1] / SAMPLES,
            root[2].div_ceil(SAMPLES).min(target.physical_width),
            root[3].div_ceil(SAMPLES).min(target.physical_height),
        ];
        let mut owner = Vec::new();
        // The common occurrence reservation already includes this capacity.
        owner
            .try_reserve_exact(path.len())
            .map_err(|_| shadow_resource(path, "numerical effect owner allocation failed"))?;
        if owner.capacity() > path.len() {
            return Err(shadow_resource(
                path,
                "numerical effect owner capacity exceeds admission",
            ));
        }
        owner.extend_from_slice(path);
        prepared.push(PreparedShadow {
            color,
            coverage: ShadowCoverage::Gaussian(index),
            logical_envelope: fields[index].logical_envelope,
            bounds: (bounds[0] < bounds[2] && bounds[1] < bounds[3]).then_some(bounds),
            group_visible: paintable,
            path: owner,
            sample_area_coverage: true,
        });
    }
    Ok(prepared)
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
pub(super) struct TileAdmission {
    pub(super) work: u64,
    /// Shader-only work has no CPU per-sample loop. It stays in the tile
    /// workload ceiling and must not be billed as CPU preparation as well.
    pub(super) gpu_work: u64,
    pub(super) numerical_uses: Vec<u64>,
    pub(super) numerical_scratch: u64,
}

impl ShadowGroups {
    pub(super) fn charge_replay(
        &mut self,
        work: u64,
        gpu_work: u64,
    ) -> Result<(), Render2dExecutionError> {
        if let Some(admission) = &mut self.admission {
            let work = work.checked_sub(gpu_work).ok_or_else(|| {
                shadow_precision(&[], "GPU tile work exceeds complete replay receipt")
            })?;
            let work = usize::try_from(work)
                .map_err(|_| shadow_resource(&[], "shadow replay work extent overflow"))?;
            admission.work.charge(
                work,
                usize::MAX,
                &[],
                "shadow replay exceeds invocation work",
            )?;
        }
        Ok(())
    }
}

pub(super) fn admit_tile_uploads(
    shadows: &mut ShadowGroups,
    target: &AdmittedTarget,
    context: &GpuContext,
    side: u32,
    bounds: [u32; 4],
) -> Result<TileAdmission, Render2dExecutionError> {
    let ShadowGroups {
        entries,
        numerical,
        admission,
    } = shadows;
    let mut numerical_uses = Vec::<u64>::new();
    if let Some(owner) = admission {
        owner.work.charge(
            // Includes the later physical slot construction, even for fields
            // with no visible use and therefore no field preparation fee.
            128 + numerical.len() * 64,
            usize::MAX,
            &[],
            "shadow tile inventory exceeds invocation work",
        )?;
        owner
            .host
            .reserve_vector(&mut numerical_uses, numerical.len(), &[])?;
    }
    numerical_uses.resize(numerical.len(), 0);
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
    let mut gpu_work = 0_u64;
    let mut count = 0_usize;
    let mut lookup_uploads = 0_u64;
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
            for shadow in entries.iter().flat_map(|(_, effects)| effects) {
                if let Some(owner) = admission {
                    owner.work.charge(
                        32,
                        usize::MAX,
                        &shadow.path,
                        "shadow tile selection exceeds invocation work",
                    )?;
                }
                if !visible(shadow, tile) {
                    continue;
                }
                if let ShadowCoverage::Gaussian(index) = shadow.coverage {
                    let uses = numerical_uses.get_mut(index).ok_or_else(|| {
                        shadow_precision(&shadow.path, "shadow tile has no numerical field")
                    })?;
                    *uses = uses.checked_add(1).ok_or_else(|| {
                        shadow_resource(&shadow.path, "numerical tile use count overflow")
                    })?;
                    total_work = total_work
                        .checked_add(sample_visits)
                        .and_then(|work| work.checked_add(2))
                        .ok_or_else(|| {
                            shadow_resource(&shadow.path, "numerical tile work count overflow")
                        })?;
                    gpu_work = gpu_work.checked_add(sample_visits).ok_or_else(|| {
                        shadow_resource(&shadow.path, "Gaussian lookup workload overflow")
                    })?;
                    lookup_uploads = lookup_uploads
                        .checked_add(super::gaussian::TILE_UNIFORM_BYTES + 6 * VERTEX_STRIDE)
                        .ok_or_else(|| {
                            shadow_resource(&shadow.path, "numerical tile upload count overflow")
                        })?;
                    let owner = admission
                        .as_mut()
                        .expect("numerical field has its preparation owner");
                    owner.work.charge(
                        super::gaussian::lookup_preparation_work(),
                        usize::MAX,
                        &shadow.path,
                        "numerical tile preparation exceeds invocation work",
                    )?;
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
    let mut numerical_uploads = lookup_uploads;
    let mut temporary = if total_bytes == 0 {
        0
    } else {
        bytes
            .checked_mul(2)
            .ok_or_else(|| shadow_resource(&[], "shadow temporary upload bytes overflow"))?
    };
    let mut numerical_scratch = 0;
    for (field, &uses) in numerical.iter().zip(&numerical_uses) {
        if uses == 0 {
            continue;
        }
        let data = super::gaussian::field_data(field);
        super::gaussian::admit_field(&data, &field.path, context)?;
        let map_bytes = (data.samples.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| shadow_resource(&field.path, "numerical map byte count overflow"))?;
        let input_bytes = data.query_input.len() as u64;
        let dummy = if data.query_count == 0 { 16 } else { 0 };
        numerical_uploads = numerical_uploads
            .checked_add(map_bytes)
            .and_then(|bytes| bytes.checked_add(input_bytes))
            .and_then(|bytes| bytes.checked_add(dummy))
            .ok_or_else(|| shadow_resource(&field.path, "numerical field upload bytes overflow"))?;
        temporary = temporary
            .max(map_bytes)
            .max(input_bytes)
            .max(6 * VERTEX_STRIDE)
            .max(super::gaussian::TILE_UNIFORM_BYTES);
        numerical_scratch = bytes;
        let fee = super::gaussian::field_preparation_work(
            data.samples.len(),
            data.query_input.len(),
            data.query_count,
        )
        .ok_or_else(|| shadow_resource(&field.path, "numerical field preparation work overflow"))?;
        let owner = admission
            .as_mut()
            .expect("numerical field has its preparation owner");
        owner.work.charge(
            fee,
            usize::MAX,
            &field.path,
            "numerical field preparation exceeds invocation work",
        )?;
        // Each node has at most two 12-product kernel polynomials and two
        // weighted reductions; each signed Q48 multiply has 16 exact u32
        // multiply-accumulates. Charge every referencing query's run, once per
        // shared field dispatch. This is an arithmetic-work bound, not a claim
        // that GPU execution consists solely of these instructions.
        let arithmetic = field
            .plane
            .query_node_visits()
            .checked_mul(26 * 16)
            .and_then(|work| u64::try_from(work).ok())
            .ok_or_else(|| {
                shadow_resource(&field.path, "expanded Gaussian arithmetic work overflow")
            })?;
        total_work = total_work.checked_add(arithmetic).ok_or_else(|| {
            shadow_resource(&field.path, "combined Gaussian execution work overflow")
        })?;
        gpu_work = gpu_work.checked_add(arithmetic).ok_or_else(|| {
            shadow_resource(&field.path, "combined Gaussian shader workload overflow")
        })?;
        if data.query_count != 0 {
            total_work = total_work.checked_add(2).ok_or_else(|| {
                shadow_resource(&field.path, "Gaussian initialization node count overflow")
            })?;
        }
    }
    let all_uploads = total_bytes
        .checked_add(numerical_uploads)
        .ok_or_else(|| shadow_resource(&[], "combined shadow upload bytes overflow"))?;
    if all_uploads > MAX_SHADOW_MASK_UPLOAD_BYTES
        || all_uploads > context.execution_policy().max_upload_bytes_in_flight()
    {
        return Err(shadow_resource(
            &[],
            "combined shadow uploads exceed admission",
        ));
    }
    if let Some(owner) = admission {
        let metadata = super::gaussian::field_metadata_bytes(numerical.len())
            .ok_or_else(|| shadow_resource(&[], "numerical physical metadata bytes overflow"))?;
        let payload = all_uploads
            .checked_add(temporary)
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| shadow_resource(&[], "combined shadow upload host extent overflow"))?;
        // Immutable transfer Arcs from all uses coexist. One conversion Vec
        // (two for legacy mask expansion) is additional peak-live storage.
        owner.host.reserve(payload, &[])?;
        owner.host.reserve(metadata, &[])?;
    }
    Ok(TileAdmission {
        work: total_work,
        gpu_work,
        numerical_uses,
        numerical_scratch,
    })
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
    let ShadowCoverage::Samples(coverage) = &shadow.coverage else {
        unreachable!("encoded numerical fields use their retained GPU lookup")
    };
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
                    .checked_sub(coverage.origin_x)
                    .and_then(|delta| usize::try_from(delta).ok());
                let sy = y
                    .checked_sub(coverage.origin_y)
                    .and_then(|delta| usize::try_from(delta).ok());
                match (sx, sy) {
                    (Some(sx), Some(sy)) if sx < coverage.width && sy < coverage.height => {
                        coverage.values[sy * coverage.width + sx]
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

    fn before_plan(composition: &Render2dComposition) -> PreparedSources<'_> {
        let mut preparation = geometry_support::SourcePreparation::new(
            geometry_support::NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            16 * 1024 * 1024,
        );
        let effects = preparation.admit_effects(composition).unwrap();
        let sources = geometry_support::build_group_sources(
            composition,
            0,
            &Render2dResourceBindings::default(),
            &BTreeMap::new(),
            preparation,
        )
        .unwrap()
        .prepare_polygonal_consumers(None, Some(([64.0, 64.0], 1.0)), MAX_SHADOW_COVERAGE_BYTES)
        .unwrap();
        PreparedSources::new(composition, sources, effects).unwrap()
    }

    fn gaussian_composition() -> Render2dComposition {
        let mut commands = Vec::new();
        for [x0, y0, x1, y1] in [[10.0, 10.0, 30.0, 30.0], [18.0, 18.0, 22.0, 22.0]] {
            let point = |x, y| Render2dPoint::new(x, y).unwrap();
            commands.extend([
                Render2dPathCommand::MoveTo(point(x0, y0)),
                Render2dPathCommand::LineTo(point(x1, y0)),
                Render2dPathCommand::LineTo(point(x1, y1)),
                Render2dPathCommand::LineTo(point(x0, y1)),
                Render2dPathCommand::Close,
            ]);
        }
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::path(
                    Render2dPath::new(Render2dFillRule::EvenOdd, commands).unwrap(),
                ),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let child = Render2dGroup::new(
            [Render2dEntry::item(item)],
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
            vec![
                Render2dDropShadow::new(8.0, 0.0, 0.03125, -1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
            ],
        );
        let wrapper = Render2dGroup::new(
            [Render2dEntry::group(child)],
            Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap(),
            vec![],
            Render2dOpacity::TRANSPARENT,
            vec![],
        );
        let root = Render2dGroup::new(
            [Render2dEntry::group(wrapper)],
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::OPAQUE,
            vec![
                Render2dDropShadow::new(
                    0.0,
                    0.0,
                    0.1,
                    0.0,
                    Render2dColorRgba8::new(255, 0, 0, 255),
                )
                .unwrap(),
            ],
        );
        Render2dComposition::new([Render2dEntry::group(root)]).unwrap()
    }

    #[test]
    fn original_gaussian_source_and_affine_plan_share_the_complete_sheared_invocation() {
        let composition = gaussian_composition();
        let bindings = Render2dResourceBindings::default();
        let fields = BTreeMap::new();
        let mut preparation = geometry_support::SourcePreparation::new(
            geometry_support::NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            MAX_SHADOW_COVERAGE_BYTES,
        );
        let effects = preparation.admit_effects(&composition).unwrap();
        let sources =
            geometry_support::build_group_sources(&composition, 0, &bindings, &fields, preparation)
                .unwrap()
                .prepare_consumers(geometry_support::GaussianTarget {
                    canvas: [64.0; 2],
                    raster_scale: 1.0,
                    physical_extent: [64; 2],
                })
                .unwrap();
        let before = sources.work.used();
        let prepared = PreparedSources::new(&composition, sources, effects).unwrap();
        let (plan, prepared) = prepared.plan().unwrap();
        let geometry_support::NeutralCaster::Gaussian(caster) = prepared.sources.caster(0).unwrap()
        else {
            panic!("original Gaussian caster")
        };
        assert_eq!(caster.planes.len(), 1);
        let plane = &caster.planes[0].plane;
        let (words, count) = plane.query_data();
        let physical = super::super::gaussian::field_preparation_work(
            plane.samples().len(),
            words.len(),
            count,
        )
        .unwrap()
            + super::super::gaussian::lookup_preparation_work();
        eprintln!(
            "ORDINARY_GAUSSIAN_CPU source={before} with_plan={} remaining={} physical_field_work={physical} root={:?} query_nodes={} Q48_products={} u32_MACs={} plan_events={} host_peak={}",
            prepared.sources.work.used(),
            prepared.sources.work.remaining(usize::MAX),
            plane.root(),
            plane.query_node_visits(),
            plane.query_node_visits() * 26,
            plane.query_node_visits() * 26 * 16,
            plan.events.len(),
            prepared.sources.host.peak()
        );
        assert!(physical < prepared.sources.work.remaining(usize::MAX));
        assert!(prepared.sources.host.peak() <= support::MAX_SHADOW_HOST_BYTES);
    }

    #[test]
    fn empty_invisible_effects_do_not_require_a_nonempty_symbolic_envelope() {
        let mut entries = gaussian_composition().root_entries().to_vec();
        entries.push(Render2dEntry::group(Render2dGroup::new(
            [],
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
            vec![
                Render2dDropShadow::new(0.0, 0.0, 0.1, -1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
            ],
        )));
        let composition = Render2dComposition::new(entries).unwrap();
        let bindings = Render2dResourceBindings::default();
        let field_sets = BTreeMap::new();
        let mut preparation = geometry_support::SourcePreparation::new(
            geometry_support::NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            MAX_SHADOW_COVERAGE_BYTES,
        );
        let effects = preparation.admit_effects(&composition).unwrap();
        let sources = geometry_support::build_group_sources(
            &composition,
            0,
            &bindings,
            &field_sets,
            preparation,
        )
        .unwrap()
        .prepare_consumers(geometry_support::GaussianTarget {
            canvas: [64.0; 2],
            raster_scale: 1.0,
            physical_extent: [64; 2],
        })
        .unwrap();
        assert!(
            matches!(sources.caster(7), Some(geometry_support::NeutralCaster::Symbolic(source))
            if source.logical_envelope().is_none())
        );
        let (plan, sources) = PreparedSources::new(&composition, sources, effects)
            .unwrap()
            .plan()
            .unwrap();
        let mut resources = GpuResourceScope::new();
        let texture = resources
            .texture(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "CPU symbolic preparation target",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    64,
                    64,
                    GpuTextureFormat::Rgba8UnormSrgb,
                    [GpuTextureUsage::ColorAttachment],
                    GpuTextureInitialization::Zeroed,
                )
                .unwrap(),
            )
            .unwrap();
        let view = resources
            .texture_view(
                GpuTextureViewDescriptor::ordinary_full_owned(
                    "CPU symbolic preparation view",
                    &texture,
                )
                .unwrap(),
            )
            .unwrap();
        let target = AdmittedTarget {
            view,
            format: GpuTextureFormat::Rgba8UnormSrgb,
            physical_width: 64,
            physical_height: 64,
            continuous_width: 64.0,
            continuous_height: 64.0,
            raster_scale: 1.0,
            max_texture_dimension_2d: 8192,
            max_buffer_bytes: MAX_SHADOW_COVERAGE_BYTES,
            image_format: true,
            clip_format: true,
        };
        let result = prepare(&plan, &target, sources).unwrap();
        assert_eq!(result.numerical.len(), 1);
        assert!(result.get(&7).unwrap().is_empty());
    }

    #[test]
    fn paid_sources_survive_planning_once_with_the_selected_policy_and_host_lease() {
        let composition = Render2dComposition::new(vec![legacy_group()]).unwrap();
        let prepared = before_plan(&composition);
        let geometry_support::NeutralCaster::Mesh(mesh) = prepared.sources.caster(0).unwrap()
        else {
            panic!("polygonal source")
        };
        let original = Arc::as_ptr(mesh);
        let paid = prepared.sources.work.used();
        let source_bytes = prepared.sources.host.live();
        assert!(paid > 0);
        assert!(prepared.sources.work.maximum() > support::MAX_COMPOSITIONAL_SHADOW_WORK);
        let (plan, prepared) = prepared.plan().unwrap();
        assert!(prepared.sources.work.used() > paid);
        assert_eq!(
            prepared.sources.host.live(),
            source_bytes + plan.retained_bytes()
        );
        let geometry_support::NeutralCaster::Mesh(mesh) = prepared.sources.caster(0).unwrap()
        else {
            panic!("same polygonal source")
        };
        assert_eq!(Arc::as_ptr(mesh), original);
        assert_eq!(plan.events.len(), 3);
    }

    #[test]
    fn mixed_policy_and_host_failures_precede_affine_plan_allocation() {
        let composition = Render2dComposition::new(vec![legacy_group(), symbolic_group()]).unwrap();
        let mut prepared = before_plan(&composition);
        assert_eq!(
            prepared.sources.work.maximum(),
            support::MAX_COMPOSITIONAL_SHADOW_WORK
        );
        prepared
            .sources
            .work
            .record(support::MAX_COMPOSITIONAL_SHADOW_WORK - 1);
        assert!(matches!(
            prepared.plan(),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
        // A fresh owning invocation is independent; the failed one was moved
        // and dropped, so its partial lease/counter cannot be reused.
        let mut fresh = before_plan(&composition);
        let remaining = MAX_SHADOW_COVERAGE_BYTES as usize - fresh.sources.host.live();
        fresh.sources.host.reserve(remaining, &[]).unwrap();
        assert!(matches!(
            fresh.plan(),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
        let (_, fresh) = before_plan(&composition).plan().unwrap();
        assert!(fresh.sources.work.used() < support::MAX_COMPOSITIONAL_SHADOW_WORK);
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
            let plan = scene::analyze(&composition, &mut ()).unwrap();
            let mut sources = sources(&plan);
            assert!(sources.work.used() > 0);
            assert!(
                sources.casters.iter().any(|(_, source)| matches!(
                    source,
                    geometry_support::NeutralCaster::Symbolic(_)
                ))
            );
            assert_eq!(
                sources.work.maximum(),
                support::MAX_COMPOSITIONAL_SHADOW_WORK
            );
            sources
                .work
                .record(support::MAX_COMPOSITIONAL_SHADOW_WORK - 1);
            let paid = sources.work.used();
            let (index, mesh) = sources
                .casters
                .iter()
                .find_map(|(index, source)| match source {
                    geometry_support::NeutralCaster::Mesh(mesh) => Some((*index, mesh)),
                    _ => None,
                })
                .unwrap();
            let scene::Event::BeginGroup { path, .. } = &plan.events[index] else {
                panic!("source owns a group")
            };
            let error = support::prepare_shadow_coverage_admitted(
                mesh,
                0.0,
                0.0,
                4.0,
                path,
                &mut sources.work,
                1_048_576,
            )
            .unwrap_err();
            assert!(matches!(error, Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(owner), ..
            } if owner == *path));
            assert_eq!(sources.work.used(), paid);
        }
    }

    #[test]
    fn legacy_graph_receipt_is_paid_without_changing_legacy_policy() {
        let composition = Render2dComposition::new(vec![legacy_group()]).unwrap();
        let plan = scene::analyze(&composition, &mut ()).unwrap();
        let mut sources = sources(&plan);
        assert!(sources.work.used() > 0);
        assert!(sources.work.maximum() > support::MAX_COMPOSITIONAL_SHADOW_WORK);
        sources.work.record(support::MAX_COMPOSITIONAL_SHADOW_WORK);
        let paid = sources.work.used();
        let (_, geometry_support::NeutralCaster::Mesh(mesh)) = &sources.casters[0] else {
            panic!("legacy source is polygonal")
        };
        let coverage = support::prepare_shadow_coverage_admitted(
            mesh,
            0.0,
            0.0,
            4.0,
            &[0],
            &mut sources.work,
            1_048_576,
        )
        .unwrap()
        .unwrap();
        assert!(sources.work.used() > paid);
        assert!((coverage.values.iter().sum::<f64>() - 16.0).abs() < 1.0e-12);
    }
}
