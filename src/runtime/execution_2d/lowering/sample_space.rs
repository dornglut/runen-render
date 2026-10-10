//! F3E private correlated-sample group color compilation.
//!
//! The single admitted F1 color path covers root and nested vectors, image
//! patches and retained F2 text with conjunctive clips and atomic opacity.
//! Effects remain fail-closed until their separately owned F3F delivery.
//! Source colors share one correlated 4x4 plane and resolve only at the
//! caller-owned target boundary.
mod gpu;
mod inspection;
mod shadow;

use self::gpu::*;
use self::inspection::*;
use super::*;
use crate::composition_2d::{
    Render2dAffineTransform, Render2dBrush, Render2dClip, Render2dGroup, Render2dItem,
    Render2dPrimitive, Render2dResourceBindings, Render2dResourceId, Render2dResourceValue,
};
use crate::execution_2d::{
    Render2dSampleSpaceError, Render2dTargetAdmissionError,
};
use crate::runtime::execution_2d::{
    clip as clip_geometry,
    field::{FieldSetKey, ResourceFields},
    image as image_semantics, scene, vector as geometry,
};
use crate::runtime::program::retained_vector_source;
use std::{collections::BTreeMap, sync::Arc};

// At RGBA16F each 4x4 sample tile occupies 128 bytes per logical pixel/layer.
// The 4x4 union mask consumes a further 64 bytes/logical pixel. The common
// scratch layers are deliberately REUSED over all sequential tiles and siblings.
const MAX_PRIVATE_SCRATCH_BYTES: u64 = 128 * 1024 * 1024;
const MAX_IMAGE_UPLOAD_BYTES: u64 = 128 * 1024 * 1024;
const MAX_PATCH_PARAMETER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TILE_SIDE: u32 = 256;
const MAX_TILES: u64 = 16384;
const MAX_OPERATIONS: usize = 1_048_576;
// Every tile visits the lexical plan, vector triangles and image patches.
// Bound aggregate CPU replay separately from emitted GPU operations.
const MAX_TILE_WORK_UNITS: u64 = 16_777_216;
const MAX_RETAINED_VECTOR_VERTICES: usize = 1_048_576;
const SAMPLES: u32 = crate::runtime::program::abi::CORRELATED_SAMPLE_AXIS;
const COLOR_BYTES_PER_PIXEL: u64 = 128;
const MASK_BYTES_PER_PIXEL: u64 = 64;

fn failure(detail: impl Into<String>) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::ResourceLimit,
        path: None,
        detail: detail.into(),
    }
}

fn precision_failure(path: &[usize], detail: impl Into<String>) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::PrecisionLimit,
        path: Some(path.to_vec()),
        detail: detail.into(),
    }
}

fn tile_side(target: &AdmittedTarget, depth: usize) -> Result<u32, Render2dExecutionError> {
    let depth = u64::try_from(depth).map_err(|_| failure("sample-depth overflow"))?;
    let bytes_per_pixel = (depth + 1)
        .checked_mul(COLOR_BYTES_PER_PIXEL)
        .and_then(|v| v.checked_add(MASK_BYTES_PER_PIXEL))
        .ok_or_else(|| failure("sample-space allocation overflow"))?;
    let mut side = MAX_TILE_SIDE.min(target.max_texture_dimension_2d() / SAMPLES);
    while side > 0 {
        let bytes = u64::from(side)
            .checked_mul(u64::from(side))
            .and_then(|area| area.checked_mul(bytes_per_pixel))
            .ok_or_else(|| failure("sample-space allocation overflow"))?;
        if bytes <= MAX_PRIVATE_SCRATCH_BYTES {
            return Ok(side);
        }
        side /= 2;
    }
    Err(failure("no feasible bounded correlated-sample tile"))
}

struct GroupFrame<'a> {
    group: &'a Render2dGroup,
    visible: bool,
    painted: bool,
    begin_operations: usize,
    clip: Option<super::clip::ClipGpu>,
}

/// One immutable image patch's source-backed upload, retained across tiles.
struct PreparedPatch {
    bounds: [u32; 4],
    image: GpuTextureViewHandle,
    parameters: GpuBufferHandle,
}

/// The clip owner and its immediate-parent coordinate frame travel together.
#[derive(Clone, Copy)]
struct ClipOwner {
    parent_to_root: scene::Affine,
    root_index: usize,
}

/// One aggregate admission budget across all owner clips and output tiles.
#[derive(Default)]
struct ClipBudget {
    reserved_bytes: u64,
    sample_work: u64,
}

/// A clip remains in the owner's immediate-parent coordinate frame. The
/// existing F3D mask/RunenGPU upload is reused; no second clip semantics.
fn prepare_clip(
    clips: &[Render2dClip],
    owner: ClipOwner,
    tile: [u32; 4],
    target: &AdmittedTarget,
    budget: &mut ClipBudget,
    resources: &mut GpuResourceScope,
) -> Result<Option<super::clip::ClipGpu>, Render2dExecutionError> {
    let Some(mask) = clip_geometry::prepare_bounded(
        clips,
        tile,
        owner.parent_to_root,
        owner.root_index,
        target,
        budget.reserved_bytes,
        &mut budget.sample_work,
    )?
    else {
        return Ok(None);
    };
    let exceeded = || {
        clip_geometry::failure(
            owner.root_index,
            crate::execution_2d::Render2dClipError::ResourceLimit,
        )
    };
    let bytes = u64::try_from(mask.rgba.len()).map_err(|_| exceeded())?;
    budget.reserved_bytes = budget
        .reserved_bytes
        .checked_add(bytes)
        .ok_or_else(exceeded)?;
    super::clip::upload(target, &mask, resources).map(Some)
}

/// Compiles a real F1 group tree through one
/// globally-phased sample plane. This is an admitted *staging subset* of the
/// same future mixed-content F3E compiler, not a second persistent renderer.
/// All nonadmitted semantics reject before an external target is modified.
/// All painting scenes use this one compositor. Missing private physical roles
/// are a typed admission failure rather than a second pixel-formation law.
pub(in crate::runtime::execution_2d) fn admits_sample_plane(context: &GpuContext) -> bool {
    let admitted = context
        .device_facts()
        .admission_contract()
        .format_roles()
        .collect::<BTreeSet<_>>();
    [
        (
            GpuTextureFormat::Rgba16Float,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Sampled),
        (FIELD_FORMAT, GpuFormatRole::ColorAttachment),
        (FIELD_FORMAT, GpuFormatRole::Sampled),
    ]
    .into_iter()
    .all(|role| admitted.contains(&role))
}

pub(in crate::runtime::execution_2d) fn lower(
    context: &GpuContext,
    target: &AdmittedTarget,
    plan: &scene::Plan<'_>,
    bindings: &Render2dResourceBindings,
    glyphs_by_event: &BTreeMap<usize, Vec<super::GlyphOccurrence>>,
    fields: &BTreeMap<FieldSetKey, Arc<ResourceFields>>,
) -> Result<Vec<GpuRenderOperation>, Render2dExecutionError> {
    // Do not realize even the visible paint tessellation of an invocation
    // whose authored shadow cardinality is already inadmissible.
    let admitted_effects = shadow::admit_effect_count(plan)?;
    let Inspected {
        items,
        mut group_bounds,
        mut bounds,
        peak_group_depth: peak,
    } = inspect(plan, target, bindings, glyphs_by_event)?;
    // This work comes from the SAME borrowed F1 painter tree as the visible
    // F3E items, but never from their clipped/colorized physical footprints.
    let shadow_groups = shadow::prepare(plan, bindings, fields, target, admitted_effects)?;
    // Immutable borrowed paths index just the authored group nodes. Each
    // visible shadow can widen at most its 64 semantic ancestors, rather
    // than scanning a potentially million-entry F1 plan per effect.
    let group_events = plan
        .events
        .iter()
        .enumerate()
        .filter_map(|(index, event)| match event {
            scene::Event::BeginGroup { path, .. } => Some((path.as_slice(), index)),
            _ => None,
        })
        .collect::<BTreeMap<_, _>>();
    for effects in shadow_groups.values() {
        for effect in effects {
            if !effect.group_visible || effect.color[3] == 0.0 {
                continue;
            }
            let Some(b) = effect.bounds else { continue };
            for dimension in 0..2 {
                bounds[dimension] = bounds[dimension].min(b[dimension]);
                bounds[dimension + 2] = bounds[dimension + 2].max(b[dimension + 2]);
            }
            // The effect's own owner and every containing group need this
            // conservative bound, never as neutral geometric membership.
            for depth in 1..=effect.path.len() {
                let Some(&index) = group_events.get(&effect.path[..depth]) else {
                    continue;
                };
                let old = group_bounds[index];
                group_bounds[index] = Some(old.map_or(b, |existing| {
                    [
                        existing[0].min(b[0]),
                        existing[1].min(b[1]),
                        existing[2].max(b[2]),
                        existing[3].max(b[3]),
                    ]
                }));
            }
        }
    }
    if bounds[0] >= bounds[2] || bounds[1] >= bounds[3] {
        // Valid no-paint scene does not require transient sample-plane roles.
        return Ok(Vec::new());
    }
    if !admits_sample_plane(context) {
        return Err(Render2dTargetAdmissionError::SamplePlaneFormatUnsupported.into());
    }
    if shadow_groups
        .values()
        .flatten()
        .any(|effect| effect.group_visible && effect.color[3] > 0.0 && effect.bounds.is_some())
        && !shadow::required_roles_admitted(context)
    {
        return Err(Render2dTargetAdmissionError::SamplePlaneFormatUnsupported.into());
    }
    let has_images = items
        .iter()
        .any(|item| matches!(item, Some(PreparedItem::Image(_))));
    let has_text = items
        .iter()
        .any(|item| matches!(item, Some(PreparedItem::Text(_))));
    let has_isolated_items = has_images || has_text;
    let extra_layer = if has_isolated_items { 1 } else { 0 };
    let side = tile_side(target, peak + extra_layer)?;
    let x0 = bounds[0] / side * side;
    let y0 = bounds[1] / side * side;
    let x_end = bounds[2].min(target.physical_width);
    let y_end = bounds[3].min(target.physical_height);
    let cols = u64::from((x_end - x0).div_ceil(side));
    let rows = u64::from((y_end - y0).div_ceil(side));
    let tile_count = cols
        .checked_mul(rows)
        .ok_or_else(|| failure("sample-space tile count overflow"))?;
    if tile_count > MAX_TILES {
        return Err(failure("sample-space tile count exceeds bounded budget"));
    }
    // Each shadow draw emits one GPU work node AND expands an immutable 4x
    // source-backed mask over every tile sample cell. Both resource traversal
    // and per-sample upload preparation share this compiler's replay budget.
    // Account for both before constructing any GPU resource or target work.
    let shadow_work = shadow::admit_tile_uploads(&shadow_groups, target, side, bounds)?;
    admit_tile_work(plan.events.len(), &items, tile_count, shadow_work)?;

    let dimension = side * SAMPLES;
    let physical = [dimension, dimension];
    let mut resources = GpuResourceScope::new();
    let layers = (0..=peak)
        .map(|n| {
            scratch(
                &mut resources,
                &format!("F3E sample color depth {n}"),
                dimension,
                GpuTextureFormat::Rgba16Float,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mask = scratch(
        &mut resources,
        "F3E vector coverage union mask",
        dimension,
        FIELD_FORMAT,
    )?;
    let item_layer = if has_isolated_items {
        Some(scratch(
            &mut resources,
            "F3E isolated item sample plane",
            dimension,
            GpuTextureFormat::Rgba16Float,
        )?)
    } else {
        None
    };
    // One semantic binding -> one immutable sampled GPU source across all
    // occurrences and output tiles. Patch mapping buffers are per occurrence.
    let mut image_views = BTreeMap::<Render2dResourceId, GpuTextureViewHandle>::new();
    let mut total_source_bytes = 0u64;
    let mut total_patch_bytes = 0u64;
    let prepared_images = items
        .iter()
        .map(|item| {
            let Some(PreparedItem::Image(patches)) = item else {
                return Ok(None);
            };
            let mut prepared = Vec::with_capacity(patches.len());
            for patch in patches {
                let image = if let Some(cached) = image_views.get(&patch.resource_id) {
                    cached.clone()
                } else {
                    let fail = || {
                        image_semantics::failure(
                            patch.root_index,
                            crate::execution_2d::Render2dImageError::ResourceLimit,
                        )
                    };
                    let size =
                        u64::try_from(patch.source.rgba8_srgb().len()).map_err(|_| fail())?;
                    total_source_bytes = total_source_bytes.checked_add(size).ok_or_else(fail)?;
                    if total_source_bytes > MAX_IMAGE_UPLOAD_BYTES {
                        return Err(fail());
                    }
                    let view = super::image::upload(target, patch, &mut resources)?;
                    image_views.insert(patch.resource_id, view.clone());
                    view
                };
                let payload = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
                    "runen-render F3E immutable image patch",
                    &patch.payload,
                )
                .map_err(|e| gpu("F3E image patch parameters", e))?;
                let bytes = payload.layout().byte_len();
                let fail = || {
                    image_semantics::failure(
                        patch.root_index,
                        crate::execution_2d::Render2dImageError::ResourceLimit,
                    )
                };
                total_patch_bytes = total_patch_bytes.checked_add(bytes).ok_or_else(fail)?;
                if bytes > target.max_buffer_bytes()
                    || total_patch_bytes > MAX_PATCH_PARAMETER_BYTES
                {
                    return Err(fail());
                }
                let parameters = resources
                    .buffer(
                        GpuBufferDescriptor::ordinary_owned(
                            "runen-render F3E source image patch mapping",
                            GpuResourceLifetime::Transient,
                            GpuReconstruction::SourceBacked,
                            bytes,
                            [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                            GpuBufferInitialization::Prepared(payload),
                        )
                        .map_err(|e| gpu("F3E image patch descriptor", e))?,
                    )
                    .map_err(|e| gpu("F3E image patch buffer", e))?;
                prepared.push(PreparedPatch {
                    bounds: patch.bounds,
                    image,
                    parameters,
                });
            }
            Ok(Some(prepared))
        })
        .collect::<Result<Vec<_>, Render2dExecutionError>>()?;

    // The same immutable F3B stop payload is uploaded once per authored
    // vector occurrence. Tiles reuse its source-backed RunenGPU handle;
    // previously a large scene could upload the same payload per tile.
    const MAX_GRADIENT_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;
    let mut allocated_gradient_bytes = 0_u64;
    let gradient_buffers = items
        .iter()
        .map(|maybe_item| {
            let Some(PreparedItem::Vector(mesh)) = maybe_item else {
                return Ok(None);
            };
            let Some(words) = vector::gradient_payload(mesh)? else {
                return Ok(None);
            };
            let bytes = u64::try_from(words.len())
                .ok()
                .and_then(|length| length.checked_mul(4))
                .ok_or_else(|| failure("gradient payload size overflow"))?;
            allocated_gradient_bytes = allocated_gradient_bytes
                .checked_add(bytes)
                .ok_or_else(|| failure("aggregate gradient resource overflow"))?;
            if bytes > target.max_buffer_bytes()
                || allocated_gradient_bytes > MAX_GRADIENT_PAYLOAD_BYTES
            {
                return Err(failure("gradient stop storage exceeds bounded admission"));
            }
            let data = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
                "runen-render F3E retained gradient stops",
                &words,
            )
            .map_err(|e| gpu("F3E gradient payload", e))?;
            let buffer = resources
                .buffer(
                    GpuBufferDescriptor::ordinary_owned(
                        "runen-render F3E retained gradient stops",
                        GpuResourceLifetime::Transient,
                        GpuReconstruction::SourceBacked,
                        data.layout().byte_len(),
                        [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                        GpuBufferInitialization::Prepared(data),
                    )
                    .map_err(|e| gpu("F3E gradient buffer descriptor", e))?,
                )
                .map_err(|e| gpu("F3E gradient buffer", e))?;
            Ok(Some(buffer))
        })
        .collect::<Result<Vec<_>, Render2dExecutionError>>()?;
    // Reuse F2's already-realized immutable fields. No reshaping, font fallback,
    // alternate text cache or per-tile reupload is introduced here.
    let mut text_views = BTreeMap::<super::FieldTextureKey, GpuTextureViewHandle>::new();
    for item in items.iter().flatten() {
        let PreparedItem::Text(glyphs) = item else {
            continue;
        };
        for glyph in glyphs {
            let key = super::FieldTextureKey::new(
                glyph.occurrence.resource_id,
                glyph.occurrence.field.glyph_id(),
            );
            if text_views.contains_key(&key) {
                continue;
            }
            // The invocation-wide field budget was already enforced before
            // generating any CPU MSDF bitmap or borrowing cached fields.
            let view = super::create_field_view(&mut resources, key, &glyph.occurrence.field)?;
            text_views.insert(key, view);
        }
    }
    let text_sampler = has_text
        .then(|| super::create_sampler(&mut resources))
        .transpose()?;
    let text_pipeline = has_text.then(super::shaped_text_pipeline).transpose()?;

    let coverage_pipeline = pipeline(FIELD_FORMAT, "fs_coverage", None)?;
    let fill_pipeline = pipeline(
        GpuTextureFormat::Rgba16Float,
        "fs_sample_mask_fill",
        Some(0),
    )?;
    let merge_pipeline = pipeline(GpuTextureFormat::Rgba16Float, "fs_sample_merge", Some(6))?;
    let merge_clipped_pipeline = pipeline(
        GpuTextureFormat::Rgba16Float,
        "fs_sample_merge_clipped",
        Some(6),
    )?;
    let fill_clipped_pipeline = pipeline(
        GpuTextureFormat::Rgba16Float,
        "fs_sample_mask_fill_clipped",
        Some(0),
    )?;
    let gradient_pipeline = pipeline(GpuTextureFormat::Rgba16Float, "fs_sample_gradient", Some(0))?;
    let gradient_clipped_pipeline = pipeline(
        GpuTextureFormat::Rgba16Float,
        "fs_sample_gradient_clipped",
        Some(0),
    )?;
    let image_pipeline = pipeline(GpuTextureFormat::Rgba16Float, "fs_sample_image", Some(2))?;
    let resolve_pipeline = pipeline(target.format, "fs_sample_resolve", Some(6))?;

    let mut operations = Vec::new();
    let mut clip_budget = ClipBudget::default();
    for row in 0..rows {
        for col in 0..cols {
            let origin = [
                x0 + u32::try_from(col).expect("bounded tile index") * side,
                y0 + u32::try_from(row).expect("bounded tile index") * side,
            ];
            let end = [
                origin[0].saturating_add(side).min(x_end),
                origin[1].saturating_add(side).min(y_end),
            ];
            if origin[0] >= end[0] || origin[1] >= end[1] {
                continue;
            }
            let tile_bounds = [origin[0], origin[1], end[0], end[1]];
            let has_content = items.iter().flatten().any(|item| {
                let b = item.bounds();
                b[0] < end[0] && b[2] > origin[0] && b[1] < end[1] && b[3] > origin[1]
            }) || shadow_groups
                .values()
                .flatten()
                .any(|effect| shadow::visible(effect, tile_bounds));
            if !has_content {
                continue;
            }
            let tile_operations_start = operations.len();
            let mut tile_painted = false;
            append(&mut operations, operation(&layers[0], true, Vec::new())?)?;
            let mut depth = 0usize;
            let mut group_stack = Vec::<GroupFrame<'_>>::new();
            for (index, event) in plan.events.iter().enumerate() {
                match event {
                    scene::Event::BeginGroup {
                        group,
                        path,
                        parent_to_root,
                    } => {
                        depth += 1;
                        let parent_visible = group_stack.last().is_none_or(|f| f.visible);
                        let group_tile = group_bounds[index]
                            .and_then(|bounds| intersection(bounds, tile_bounds));
                        let viable =
                            parent_visible && group.opacity().get() > 0.0 && group_tile.is_some();
                        let clip = if !viable || group.clips().is_empty() {
                            None
                        } else {
                            prepare_clip(
                                group.clips(),
                                ClipOwner {
                                    parent_to_root: *parent_to_root,
                                    root_index: path[0],
                                },
                                group_tile.expect("viable group has visible bounds"),
                                target,
                                &mut clip_budget,
                                &mut resources,
                            )?
                        };
                        let visible = viable && (group.clips().is_empty() || clip.is_some());
                        let begin_operations = operations.len();
                        group_stack.push(GroupFrame {
                            group,
                            visible,
                            painted: false,
                            begin_operations,
                            clip,
                        });
                        if visible {
                            append(
                                &mut operations,
                                operation(&layers[depth], true, Vec::new())?,
                            )?;
                            // The group scratch starts transparent. Every
                            // authored shadow independently samples C, paints
                            // source-over in list order BEHIND its children,
                            // then the existing group merge clips and applies
                            // opacity exactly once. No separate backend.
                            if let Some(effects) = shadow_groups.get(&index) {
                                for effect in effects {
                                    if !shadow::visible(effect, tile_bounds) {
                                        continue;
                                    }
                                    let coverage = shadow::upload_tile(
                                        effect,
                                        origin,
                                        dimension,
                                        target,
                                        &mut resources,
                                    )?;
                                    let mut quad = Vec::new();
                                    let sample_origin = [
                                        f64::from(origin[0]) * f64::from(SAMPLES),
                                        f64::from(origin[1]) * f64::from(SAMPLES),
                                    ];
                                    rectangle(
                                        &mut quad,
                                        [0.0, 0.0, f64::from(dimension), f64::from(dimension)],
                                        physical,
                                        effect.color,
                                        sample_origin,
                                    );
                                    let paint = draw(
                                        &fill_pipeline,
                                        Some((0, &coverage)),
                                        None,
                                        None,
                                        &quad,
                                        physical,
                                        &mut resources,
                                    )?;
                                    append(
                                        &mut operations,
                                        operation(&layers[depth], false, vec![paint])?,
                                    )?;
                                    tile_painted = true;
                                    group_stack
                                        .last_mut()
                                        .expect("the shadow has an active visible group")
                                        .painted = true;
                                }
                            }
                        }
                    }
                    scene::Event::EndGroup => {
                        let frame = group_stack.pop().expect("balanced F1 group plan");
                        if frame.visible && frame.painted {
                            let mut vertices = Vec::new();
                            let origin_sample = [
                                f64::from(origin[0]) * f64::from(SAMPLES),
                                f64::from(origin[1]) * f64::from(SAMPLES),
                            ];
                            rectangle(
                                &mut vertices,
                                [0.0, 0.0, f64::from(dimension), f64::from(dimension)],
                                physical,
                                [1.0, 1.0, 1.0, f32_from_f64(frame.group.opacity().get())],
                                origin_sample,
                            );
                            let pipeline = if frame.clip.is_some() {
                                &merge_clipped_pipeline
                            } else {
                                &merge_pipeline
                            };
                            let merged = draw(
                                pipeline,
                                Some((6, &layers[depth])),
                                frame.clip.as_ref(),
                                None,
                                &vertices,
                                physical,
                                &mut resources,
                            )?;
                            append(
                                &mut operations,
                                operation(&layers[depth - 1], false, vec![merged])?,
                            )?;
                            if let Some(parent) = group_stack.last_mut() {
                                parent.painted = true;
                            }
                        } else {
                            // Never author an empty group clear/merge simply because
                            // structurally valid content was fully clipped away.
                            operations.truncate(frame.begin_operations);
                        }
                        depth -= 1;
                    }
                    scene::Event::Item {
                        item,
                        path,
                        parent_to_root,
                        ..
                    } => {
                        if group_stack.last().is_some_and(|frame| !frame.visible)
                            || item.opacity().get() == 0.0
                        {
                            continue;
                        }
                        let Some(content) = &items[index] else {
                            continue;
                        };
                        let Some(item_tile) = intersection(content.bounds(), tile_bounds) else {
                            continue;
                        };
                        let item_clip = if item.clips().is_empty() {
                            None
                        } else {
                            let Some(mask) = prepare_clip(
                                item.clips(),
                                ClipOwner {
                                    parent_to_root: *parent_to_root,
                                    root_index: path[0],
                                },
                                item_tile,
                                target,
                                &mut clip_budget,
                                &mut resources,
                            )?
                            else {
                                continue;
                            };
                            Some(mask)
                        };
                        match content {
                            PreparedItem::Vector(mesh) => {
                                let mut coverage_vertices =
                                    Vec::with_capacity(mesh.triangles.len() * 8);
                                for &[x, y] in &mesh.triangles {
                                    let x = (x - f64::from(origin[0])) * f64::from(SAMPLES);
                                    let y = (y - f64::from(origin[1])) * f64::from(SAMPLES);
                                    coverage_vertices.extend([
                                        physical_x_to_ndc(x, dimension),
                                        physical_y_to_ndc(y, dimension),
                                        0.0,
                                        0.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                        1.0,
                                    ]);
                                }
                                let coverage_draw = draw(
                                    &coverage_pipeline,
                                    None,
                                    None,
                                    None,
                                    &coverage_vertices,
                                    physical,
                                    &mut resources,
                                )?;
                                append(
                                    &mut operations,
                                    operation(&mask, true, vec![coverage_draw])?,
                                )?;
                                // F3B's single gradient payload authority supplies exact
                                // premultiplied linear authored stops and inverse brush mapping.
                                let (paint_pipeline, vertex_color) = match &mesh.brush {
                                    Render2dBrush::Solid(color) => {
                                        let mut rgba = linear_color(*color);
                                        rgba[3] *= f32_from_f64(mesh.opacity);
                                        let pipeline = if item_clip.is_some() {
                                            &fill_clipped_pipeline
                                        } else {
                                            &fill_pipeline
                                        };
                                        (pipeline, rgba)
                                    }
                                    Render2dBrush::Linear(_) | Render2dBrush::Radial(_) => {
                                        let pipeline = if item_clip.is_some() {
                                            &gradient_clipped_pipeline
                                        } else {
                                            &gradient_pipeline
                                        };
                                        (pipeline, [1.0; 4])
                                    }
                                };
                                let mut full_quad = Vec::new();
                                let origin_sample = [
                                    f64::from(origin[0]) * f64::from(SAMPLES),
                                    f64::from(origin[1]) * f64::from(SAMPLES),
                                ];
                                rectangle(
                                    &mut full_quad,
                                    [0.0, 0.0, f64::from(dimension), f64::from(dimension)],
                                    physical,
                                    vertex_color,
                                    origin_sample,
                                );
                                let color_draw = draw(
                                    paint_pipeline,
                                    Some((0, &mask)),
                                    item_clip.as_ref(),
                                    gradient_buffers[index].as_ref(),
                                    &full_quad,
                                    physical,
                                    &mut resources,
                                )?;
                                append(
                                    &mut operations,
                                    operation(&layers[depth], false, vec![color_draw])?,
                                )?;
                                tile_painted = true;
                                if let Some(group) = group_stack.last_mut() {
                                    group.painted = true;
                                }
                            }
                            PreparedItem::Image(_) | PreparedItem::Text(_) => {
                                let layer = item_layer
                                    .as_ref()
                                    .expect("image/text item has isolated shared scratch");
                                let mut draws = Vec::new();
                                let origin_sample = [
                                    f64::from(origin[0]) * f64::from(SAMPLES),
                                    f64::from(origin[1]) * f64::from(SAMPLES),
                                ];
                                match content {
                                    PreparedItem::Image(_) => {
                                        let patches = prepared_images[index]
                                            .as_ref()
                                            .expect("every image occurrence has patch resources");
                                        for patch in patches {
                                            let b = patch.bounds;
                                            let left = b[0].max(origin[0]);
                                            let top = b[1].max(origin[1]);
                                            let right = b[2].min(end[0]);
                                            let bottom = b[3].min(end[1]);
                                            if left >= right || top >= bottom {
                                                continue;
                                            }
                                            let mut vertices = Vec::new();
                                            rectangle(
                                                &mut vertices,
                                                [
                                                    f64::from(left - origin[0])
                                                        * f64::from(SAMPLES),
                                                    f64::from(top - origin[1]) * f64::from(SAMPLES),
                                                    f64::from(right - origin[0])
                                                        * f64::from(SAMPLES),
                                                    f64::from(bottom - origin[1])
                                                        * f64::from(SAMPLES),
                                                ],
                                                physical,
                                                [1.0; 4],
                                                origin_sample,
                                            );
                                            draws.push(image_draw(
                                                &image_pipeline,
                                                patch,
                                                &vertices,
                                                physical,
                                                &mut resources,
                                            )?);
                                        }
                                    }
                                    PreparedItem::Text(glyphs) => {
                                        let pipeline =
                                            text_pipeline.as_ref().expect("text pipeline");
                                        let sampler = text_sampler.as_ref().expect("field sampler");
                                        for glyph in glyphs {
                                            let vertices = text_quad(glyph, origin, end, dimension);
                                            if vertices.is_empty() {
                                                continue;
                                            }
                                            let key = super::FieldTextureKey::new(
                                                glyph.occurrence.resource_id,
                                                glyph.occurrence.field.glyph_id(),
                                            );
                                            let view = text_views
                                                .get(&key)
                                                .expect("unique retained field upload");
                                            let bindings = pipeline
                                                .runtime_bindings([
                                                    texture_binding(0, view)?,
                                                    super::sampler_binding(1, sampler)?,
                                                ])
                                                .map_err(|e| {
                                                    gpu("F3E retained text bindings", e)
                                                })?;
                                            draws.push(vector::vector_draw(
                                                pipeline.clone(),
                                                bindings,
                                                &vertices,
                                                physical,
                                                &mut resources,
                                            )?);
                                        }
                                    }
                                    PreparedItem::Vector(_) => {
                                        unreachable!("isolated item is image or text")
                                    }
                                }
                                if !draws.is_empty() {
                                    tile_painted = true;
                                    if let Some(group) = group_stack.last_mut() {
                                        group.painted = true;
                                    }
                                    // Glyphs and image patches composite at original
                                    // alpha. Item opacity and clip apply only after
                                    // their completed isolated item contribution.
                                    append(&mut operations, operation(layer, true, draws)?)?;
                                    let mut merged_quad = Vec::new();
                                    rectangle(
                                        &mut merged_quad,
                                        [0.0, 0.0, f64::from(dimension), f64::from(dimension)],
                                        physical,
                                        [1.0, 1.0, 1.0, f32_from_f64(item.opacity().get())],
                                        origin_sample,
                                    );
                                    let pipeline = if item_clip.is_some() {
                                        &merge_clipped_pipeline
                                    } else {
                                        &merge_pipeline
                                    };
                                    let merged = draw(
                                        pipeline,
                                        Some((6, layer)),
                                        item_clip.as_ref(),
                                        None,
                                        &merged_quad,
                                        physical,
                                        &mut resources,
                                    )?;
                                    append(
                                        &mut operations,
                                        operation(&layers[depth], false, vec![merged])?,
                                    )?;
                                }
                            }
                        }
                    }
                }
            }
            if !tile_painted {
                operations.truncate(tile_operations_start);
                continue;
            }
            let mut resolve_vertices = Vec::new();
            rectangle(
                &mut resolve_vertices,
                [
                    f64::from(origin[0]),
                    f64::from(origin[1]),
                    f64::from(end[0]),
                    f64::from(end[1]),
                ],
                [target.physical_width, target.physical_height],
                [1.0; 4],
                [-f64::from(origin[0]), -f64::from(origin[1])],
            );
            let resolve_draw = draw(
                &resolve_pipeline,
                Some((6, &layers[0])),
                None,
                None,
                &resolve_vertices,
                [target.physical_width, target.physical_height],
                &mut resources,
            )?;
            append(
                &mut operations,
                operation(&target.view, false, vec![resolve_draw])?,
            )?;
        }
    }
    Ok(operations)
}

#[cfg(test)]
mod tile_budget_tests {
    use super::*;

    #[test]
    fn cumulative_tile_replay_is_bounded_before_resource_preparation() {
        assert!(admit_tile_work(1024, &[], MAX_TILE_WORK_UNITS / 1024, 0).is_ok());
        assert!(matches!(
            admit_tile_work(1024, &[], MAX_TILE_WORK_UNITS / 1024 + 1, 0),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
        assert!(admit_tile_work(usize::MAX, &[], u64::MAX, 0).is_err());
        // Previously admitted as independent replay and shadow budgets,
        // although their combined sample visits and draws exceed the one
        // compiler budget. The last scalar is actual shadow work, not just
        // the number of authored shadows.
        assert!(matches!(
            admit_tile_work(1024, &[], MAX_TILE_WORK_UNITS / 1024, 1),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
        assert!(admit_tile_work(1, &[], 1, MAX_TILE_WORK_UNITS - 1).is_ok());
        assert!(admit_tile_work(1, &[], 1, u64::MAX).is_err());
        // One full 1024x1024 4x mask upload visits 1,048,576 sample cells
        // and contributes one draw. Mask uploads are not one CPU work unit.
        let upload = u64::from(1024_u32) * u64::from(1024_u32) + 1;
        assert!(admit_tile_work(100, &[], 1, upload).is_ok());
        assert!(matches!(
            admit_tile_work(100, &[], 1, MAX_TILE_WORK_UNITS),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
    }

    #[test]
    fn retained_vector_vertices_are_charged_across_items() {
        let mut used = MAX_RETAINED_VECTOR_VERTICES - 3;
        charge_vector_geometry(&mut used, 3).expect("last admissible vertices");
        assert_eq!(used, MAX_RETAINED_VECTOR_VERTICES);
        assert!(charge_vector_geometry(&mut used, 1).is_err());
        assert_eq!(used, MAX_RETAINED_VECTOR_VERTICES);
    }
}
