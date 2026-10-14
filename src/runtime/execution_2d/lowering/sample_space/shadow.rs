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

// Retained floating neutral support is separate from per-tile source-backed
// GPU uploads; both are aggregate limits across one immutable contribution.
const MAX_SHADOW_COVERAGE_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SHADOW_MASK_UPLOAD_BYTES: u64 = 128 * 1024 * 1024;
const MAX_AUTHORED_SHADOWS: usize = 256;
const FORMAT: GpuTextureFormat = GpuTextureFormat::R32Float;

pub(super) struct PreparedShadow {
    pub(super) color: [f32; 4],
    pub(super) coverage: NeutralCoverage,
    /// The mathematically bounded 3σ support reach in the immediate-parent
    /// continuous logical frame, not a rectangle used as membership.
    pub(super) logical_envelope: [f64; 4],
    /// Clipped physical-pixel envelope, never geometric membership.
    pub(super) bounds: Option<[u32; 4]>,
    /// Group alpha is a visual admission fact; neutral support ignores it.
    pub(super) group_visible: bool,
    pub(super) path: Vec<usize>,
}

pub(super) type ShadowGroups = BTreeMap<usize, Vec<PreparedShadow>>;

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

/// First production symbolic-caster cut. A zero-spread, zero-blur ancestor
/// consumes the *completed original-F1 region*, including a child's signed
/// eroded geometry, without reinterpreting that geometry as triangles or alpha.
/// A true finite Gaussian/ancestor disk needs a separate admitted integrator.
/// Conservative worst-case output uncertainty for arbitrary prior target
/// channels and a source-over painter: the largest sRGB8 excursion occurs
/// for a fully dark/bright channel against its opposite. This constrains
/// the FINAL resolved channel, not merely unorm alpha or a center hit.
fn symbolic_srgb8_uncertainty(lower: f64, upper: f64) -> f64 {
    let encode = |linear: f64| {
        if linear <= 0.003_130_8 {
            12.92 * linear
        } else {
            1.055 * linear.powf(1.0 / 2.4) - 0.055
        }
    };
    255.0
        * (encode(upper) - encode(lower))
            .max(encode(1.0 - lower) - encode(1.0 - upper))
}

/// One invocation-owned admission context, not a second scene representation.
struct SymbolicPreparation<'a> {
    group: &'a crate::composition_2d::Render2dGroup,
    path: &'a [usize],
    parent_to_root: scene::Affine,
    target: &'a AdmittedTarget,
    sample_scale: f64,
    paintable: bool,
    authored_effects: usize,
}

fn prepare_symbolic_unblurred(
    symbolic: &geometry_support::SymbolicCaster<'_>,
    request: SymbolicPreparation<'_>,
    retained_coverage_bytes: &mut u64,
    aggregate_coverage_work: &mut usize,
) -> Result<Vec<PreparedShadow>, Render2dExecutionError> {
    let SymbolicPreparation {
        group,
        path,
        parent_to_root,
        target,
        sample_scale,
        paintable,
        authored_effects,
    } = request;
    let unsupported = || {
        let owner = symbolic.owner_path();
        crate::runtime::execution_2d::unsupported_at(
            owner,
            crate::execution_2d::Render2dUnsupportedContent::Shadows {
                root_index: owner[0],
            },
        )
    };
    // Reject even transparent unqualified effects: neutral geometry may be
    // visible to an ancestor regardless of foreground opacity.
    if group.shadows().iter().any(|s| {
        s.spread() != 0.0 || s.sigma() != 0.0 || s.offset_x() != 0.0 || s.offset_y() != 0.0
    }) {
        return Err(unsupported());
    }
    let limit = support::MAX_AGGREGATE_SHADOW_PREPARATION_WORK;
    let mut fuel = limit
        .checked_sub(*aggregate_coverage_work)
        .ok_or_else(|| shadow_resource(path, "symbolic aggregate work already exhausted"))?;
    if !symbolic.qualified_unblurred_fills(&mut fuel) {
        return Err(unsupported());
    }
    let [a, b, c, d, tx, ty] = parent_to_root.coefficients();
    let to_root = Render2dAffineTransform::new(a, b, c, d, tx, ty).map_err(|_| {
        shadow_precision(path, "symbolic owner-to-root transform is unrepresentable")
    })?;
    let det = a.mul_add(d, -(b * c));
    if !det.is_finite() || det == 0.0 {
        return Err(shadow_precision(
            path,
            "symbolic ancestor transform is singular",
        ));
    }
    let Some(owner_envelope) = symbolic.certified_reach_envelope() else {
        *aggregate_coverage_work = limit - fuel;
        return Ok(Vec::new());
    };
    let envelope = geometry_support::transform_envelope(owner_envelope, to_root, path)?;
    let scaled = envelope.map(|v| v * sample_scale);
    if !scaled.iter().all(|v| v.is_finite())
        || scaled.iter().any(|v| v.abs() > f64::from(i32::MAX) / 2.0)
    {
        return Err(shadow_precision(
            path,
            "symbolic root coverage reach exceeds precise indices",
        ));
    }
    let canvas = [
        i64::from(target.physical_width) * i64::from(SAMPLES),
        i64::from(target.physical_height) * i64::from(SAMPLES),
    ];
    // Cropping is permitted only after exact original-F1 support reach is
    // conservatively proven. A shadow can reach the target from outside.
    #[allow(clippy::cast_possible_truncation)]
    let edges = [
        (scaled[0].floor() as i64).clamp(0, canvas[0]),
        (scaled[1].floor() as i64).clamp(0, canvas[1]),
        (scaled[2].ceil() as i64).clamp(0, canvas[0]),
        (scaled[3].ceil() as i64).clamp(0, canvas[1]),
    ];
    if edges[0] >= edges[2] || edges[1] >= edges[3] {
        *aggregate_coverage_work = limit - fuel;
        return Ok(Vec::new());
    }
    let width = usize::try_from(edges[2] - edges[0])
        .map_err(|_| shadow_resource(path, "symbolic mask width overflow"))?;
    let height = usize::try_from(edges[3] - edges[1])
        .map_err(|_| shadow_resource(path, "symbolic mask height overflow"))?;
    let count = width
        .checked_mul(height)
        .ok_or_else(|| shadow_resource(path, "symbolic sample grid overflow"))?;
    let remaining = MAX_SHADOW_COVERAGE_BYTES
        .checked_sub(*retained_coverage_bytes)
        .and_then(|bytes| usize::try_from(bytes / 8).ok())
        .ok_or_else(|| shadow_resource(path, "symbolic retained coverage budget exhausted"))?;
    if count > support::MAX_NEUTRAL_MASK_SAMPLES || count > remaining {
        return Err(shadow_resource(
            path,
            "symbolic coverage exceeds aggregate sample admission",
        ));
    }
    // Charge a conservative half of the existing four-code-value output
    // tolerance to geometric coverage; leave headroom for Rgba16Float
    // blending/resolve. Sum per-effect errors even across 256 ordered layers.
    let effect_count =
        f64::from(u16::try_from(authored_effects.max(1)).expect("bounded 256 effects"));
    let color_error_budget = 2.0 / effect_count;
    let mut prepared = Vec::<PreparedShadow>::new();
    prepared
        .try_reserve_exact(group.shadows().len())
        .map_err(|_| shadow_resource(path, "symbolic effect list allocation failed"))?;
    let mut retained_one: Option<NeutralCoverage> = None;
    for effect in group.shadows() {
        let color = linear_color(effect.color());
        if !paintable || color[3] == 0.0 {
            continue;
        }
        // Every sibling uses pristine C and the same zero-blur identity
        // kernel. Reusing one unmodified coverage is semantically exact.
        if retained_one.is_none() {
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| shadow_resource(path, "symbolic coverage allocation failed"))?;
            for y in edges[1]..edges[3] {
                for x in edges[0]..edges[2] {
                    let x0 = f64::from(i32::try_from(x).expect("bounded symbolic root index"));
                    let y0 = f64::from(i32::try_from(y).expect("bounded symbolic root index"));
                    let bounds = [
                        x0 / sample_scale,
                        y0 / sample_scale,
                        (x0 + 1.0) / sample_scale,
                        (y0 + 1.0) / sample_scale,
                    ];
                    let mut area = symbolic
                        .continuous_area_enclosure(bounds, Some(to_root), 10, &mut fuel)
                        .ok_or_else(|| {
                            shadow_resource(
                                symbolic.owner_path(),
                                "symbolic original-F1 cell proof exhausted shared work",
                            )
                        })?;
                    let valid = |[lo, hi]: [f64; 2]| {
                        lo.is_finite()
                            && hi.is_finite()
                            && lo >= 0.0
                            && hi <= 1.0
                            && lo <= hi
                            && symbolic_srgb8_uncertainty(lo, hi) <= color_error_budget
                    };
                    // Depth 10 admits most boundaries. Refine only genuinely
                    // ambiguous sample cells to depth 12, sharing the same
                    // bounded invocation fuel; never lower alpha to a guess.
                    if !valid(area) {
                        area = symbolic
                            .continuous_area_enclosure(bounds, Some(to_root), 12, &mut fuel)
                            .ok_or_else(|| {
                                shadow_resource(
                                    symbolic.owner_path(),
                                    "symbolic refined F1 cell proof exhausted shared work",
                                )
                            })?;
                    }
                    if !valid(area) {
                        return Err(shadow_precision(
                            symbolic.owner_path(),
                            "symbolic continuous-area uncertainty exceeds final-color bound",
                        ));
                    }
                    values.push((area[0] + area[1]) * 0.5);
                }
            }
            retained_one = Some(NeutralCoverage {
                origin_x: edges[0],
                origin_y: edges[1],
                width,
                height,
                values,
            });
        }
        let bytes = u64::try_from(count)
            .ok()
            .and_then(|count| count.checked_mul(8))
            .ok_or_else(|| shadow_resource(path, "symbolic retained bytes overflow"))?;
        let next_retained = retained_coverage_bytes
            .checked_add(bytes)
            .ok_or_else(|| shadow_resource(path, "symbolic aggregate retained bytes overflow"))?;
        if next_retained > MAX_SHADOW_COVERAGE_BYTES {
            return Err(shadow_resource(
                path,
                "symbolic retained coverage memory exhausted",
            ));
        }
        // Move the first certified mask without doubling peak-live bytes.
        // Other ordered source-over layers consume the same pristine C, but
        // retain separately admitted, fallibly allocated physical buffers.
        let layer = if let Some(first) = retained_one.take() {
            first
        } else {
            let first = &prepared[0].coverage;
            let mut values = Vec::new();
            values
                .try_reserve_exact(count)
                .map_err(|_| shadow_resource(path, "symbolic sibling mask allocation failed"))?;
            values.extend_from_slice(&first.values);
            NeutralCoverage {
                origin_x: first.origin_x,
                origin_y: first.origin_y,
                width,
                height,
                values,
            }
        };
        let bounds = output_bounds(&layer, target, path)?;
        prepared.push(PreparedShadow {
            color,
            coverage: layer,
            logical_envelope: envelope,
            bounds,
            group_visible: paintable,
            path: path.to_vec(),
        });
        *retained_coverage_bytes = next_retained;
    }
    *aggregate_coverage_work = limit - fuel;
    Ok(prepared)
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
        return Ok(BTreeMap::new());
    }
    let sample_scale = target.raster_scale() * f64::from(SAMPLES);
    if !sample_scale.is_finite() || sample_scale <= 0.0 {
        return Err(shadow_precision(
            &[],
            "shadow group-parent sampling scale is invalid",
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
    let mut groups = BTreeMap::new();
    let mut retained_coverage_bytes = 0_u64;
    let mut aggregate_coverage_work = 0_usize;
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
        if let geometry_support::NeutralCaster::Symbolic(symbolic) = source {
            let prepared = prepare_symbolic_unblurred(
                symbolic,
                SymbolicPreparation {
                    group,
                    path,
                    parent_to_root: *parent_to_root,
                    target,
                    sample_scale,
                    paintable,
                    authored_effects: admitted_effects,
                },
                &mut retained_coverage_bytes,
                &mut aggregate_coverage_work,
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
                // The exact original-F1 DAG is retained to this owner boundary,
                // but no continuous whole-pixel evaluator has yet qualified
                // its nonpolygonal nested signed support. Reject before
                // creating a physical mask or mutating the caller target.
                let owner = symbolic.owner_path();
                return Err(crate::runtime::execution_2d::unsupported_at(
                    owner,
                    crate::execution_2d::Render2dUnsupportedContent::Shadows {
                        root_index: owner[0],
                    },
                ));
            }
        };
        let mut prepared = Vec::new();
        prepared
            .try_reserve_exact(group.shadows().len())
            .map_err(|_| shadow_resource(path, "shadow preparation list allocation failed"))?;
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
                    let determinant = a.mul_add(d, -(b * c));
                    if determinant == 0.0 {
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
                coverage,
                logical_envelope: envelope,
                bounds,
                group_visible: paintable,
                path: path.clone(),
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
                || logical_x < left
                || logical_x > right
                || logical_y < top
                || logical_y > bottom
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
    let source = PreparedGpuData::<TransferData>::ordinary_pod_transfer(&name, &data)
        .map_err(|e| gpu("shadow float mask bytes", e))?;
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
