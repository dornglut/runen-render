//! F3F renderer-owned neutral shadow masks inside the ONE F3E correlated
//! linear-premultiplied 4x4 sample compositor. No separate color renderer,
//! sampled source alpha, shader module, scene owner or direct WGPU backend.
use super::*;
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
    )?;
    let mut groups = BTreeMap::new();
    let mut retained_coverage_bytes = 0_u64;
    for (index, event) in plan.events.iter().enumerate() {
        let scene::Event::BeginGroup {
            group,
            path,
            parent_to_root,
        } = event
        else {
            continue;
        };
        if group.shadows().is_empty() {
            continue;
        }
        let source = sources
            .get(&index)
            .expect("the neutral F1 traversal records each shadow-bearing group");
        let mut prepared = Vec::new();
        prepared
            .try_reserve_exact(group.shadows().len())
            .map_err(|_| shadow_resource(path, "shadow preparation list allocation failed"))?;
        for effect in group.shadows() {
            let color = linear_color(effect.color());
            // Build the effect in its immediate-parent frame first.
            // Zero spread and zero blur commute with ancestor affine
            // projection, so a nested translated C remains exact triangles.
            // Nonzero nested kernels reject in group_child_sources rather
            // than being incorrectly applied in root space.
            let Some(parent_envelope) = geometry_support::shadow_envelope(source, *effect, path)?
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
            let offset = Render2dAffineTransform::translation(effect.offset_x(), effect.offset_y())
                .map_err(|_| {
                    shadow_precision(path, "shadow offset transform is unrepresentable")
                })?;
            let mut shifted = geometry_support::empty_mesh();
            geometry_support::append(&mut shifted, source, offset, path)?;
            let mut projected = geometry_support::empty_mesh();
            let raster_source = if let Some(ancestor) = parent_affine {
                geometry_support::append(&mut projected, &shifted, ancestor, path)?;
                &projected
            } else {
                &shifted
            };
            let Some(coverage) = support::prepare_untranslated_shadow_coverage(
                raster_source,
                effect.spread(),
                effect.sigma(),
                sample_scale,
                path,
            )?
            else {
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
                group_visible: group.opacity().get() > 0.0,
                path: path.clone(),
            });
        }
        groups.insert(index, prepared);
    }
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
