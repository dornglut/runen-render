//! F3E private correlated-sample group color compilation.
//!
//! Initial admitted realization: solid/gradient vector fills and strokes,
//! conjunctive item/group clips, and nested atomic groups. Shaped text, images,
//! effects and unsupported semantic classes fail closed until this same
//! compiler can realize them. No second public renderer or semantic display
//! list is introduced. Root and child colors share one correlated 4x4 sample
//! lattice and one final pixel resolve to the caller-owned target.
use super::*;
use crate::composition_2d::{
    Render2dAffineTransform, Render2dBrush, Render2dClip, Render2dGroup, Render2dItem,
    Render2dPrimitive,
};
use crate::execution_2d::Render2dUnsupportedContent;
use crate::runtime::execution_2d::{clip as clip_geometry, scene, vector as geometry};
use crate::runtime::program::retained_vector_source;

// At RGBA16F each 4x4 sample tile occupies 128 bytes per logical pixel/layer.
// The 4x4 union mask consumes a further 64 bytes/logical pixel. The common
// scratch layers are deliberately REUSED over all sequential tiles and siblings.
const MAX_PRIVATE_SCRATCH_BYTES: u64 = 128 * 1024 * 1024;
const MAX_TILE_SIDE: u32 = 256;
const MAX_TILES: u64 = 16384;
const MAX_OPERATIONS: usize = 1_048_576;
const SAMPLES: u32 = 4;
const COLOR_BYTES_PER_PIXEL: u64 = 128;
const MASK_BYTES_PER_PIXEL: u64 = 64;

fn failure(detail: impl Into<String>) -> Render2dExecutionError {
    Render2dExecutionError::Gpu {
        stage: "F3E correlated sample lowering",
        detail: detail.into(),
    }
}

fn pipeline(
    target_format: GpuTextureFormat,
    entry: &'static str,
    sample_binding: Option<u32>,
) -> Result<GpuRenderPipelineDescriptor, Render2dExecutionError> {
    let source = retained_vector_source().map_err(|e| Render2dExecutionError::Program {
        stage: "F3E maintained vector source",
        detail: format!("{e:?}"),
    })?;
    let vertex = GpuEntryPointName::new("vs_main").map_err(|e| gpu("F3E vertex entry", e))?;
    let fragment = GpuEntryPointName::new(entry).map_err(|e| gpu("F3E fragment entry", e))?;
    let mut refinements = sample_binding
        .map(|binding| {
            GpuBindingKey::try_new(0, u64::from(binding))
                .map(|key| {
                    vec![
                        GpuBindingLayoutRefinement::new(key)
                            .with_texture_sample_class(GpuTextureSampleClass::FloatUnfilterable),
                    ]
                })
                .map_err(|e| gpu("F3E source texture layout key", e))
        })
        .transpose()?
        .unwrap_or_default();
    if matches!(
        entry,
        "fs_sample_mask_fill_clipped" | "fs_sample_merge_clipped" | "fs_sample_gradient_clipped"
    ) {
        let key = GpuBindingKey::try_new(0, 4)
            .map_err(|e| gpu("F3E structural clip binding layout", e))?;
        refinements.push(
            GpuBindingLayoutRefinement::new(key)
                .with_texture_sample_class(GpuTextureSampleClass::FloatUnfilterable),
        );
    }
    let program =
        GpuProgramDescriptor::new(source, [vertex.clone(), fragment.clone()], refinements)
            .map_err(|e| gpu("F3E sample program", e))?;
    let layout = GpuVertexBufferLayoutDescriptor::new(
        0,
        VERTEX_STRIDE,
        GpuVertexStepMode::Vertex,
        [
            GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32x2),
            GpuVertexAttribute::new(1, 8, GpuVertexFormat::Float32x2),
            GpuVertexAttribute::new(2, 16, GpuVertexFormat::Float32x4),
        ],
    )
    .map_err(|e| gpu("F3E vertex layout", e))?;
    let component = GpuBlendComponent::new(
        GpuBlendFactor::One,
        GpuBlendFactor::OneMinusSrcAlpha,
        GpuBlendOperation::Add,
    )
    .map_err(|e| gpu("F3E premultiplied source-over", e))?;
    let output = GpuColorTargetStateDescriptor::new(
        target_format,
        (entry != "fs_coverage").then_some(GpuBlendState::new(component, component)),
        GpuColorWriteMask::ALL,
    )
    .map_err(|e| gpu("F3E attachment format", e))?;
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([layout]).map_err(|e| gpu("F3E vertex input", e))?,
        Some(GpuFragmentOutputStateDescriptor::new([output])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .map_err(|e| gpu("F3E render pipeline state", e))?;
    GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .map_err(|e| gpu("F3E render pipeline", e))
}

fn scratch(
    resources: &mut GpuResourceScope,
    name: &str,
    dimension: u32,
    format: GpuTextureFormat,
) -> Result<GpuTextureViewHandle, Render2dExecutionError> {
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                dimension,
                dimension,
                format,
                [GpuTextureUsage::ColorAttachment, GpuTextureUsage::Sampled],
                GpuTextureInitialization::Uninitialized,
            )
            .map_err(|e| gpu("F3E scratch descriptor", e))?,
        )
        .map_err(|e| gpu("F3E scratch", e))?;
    resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(format!("{name} view"), &texture)
                .map_err(|e| gpu("F3E scratch view descriptor", e))?,
        )
        .map_err(|e| gpu("F3E scratch view", e))
}

fn operation(
    view: &GpuTextureViewHandle,
    clear: bool,
    draws: Vec<GpuRenderDraw>,
) -> Result<GpuRenderOperation, Render2dExecutionError> {
    let load = if clear {
        GpuColorAttachmentLoad::Clear(
            GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0)
                .map_err(|e| gpu("F3E scratch clear color", e))?,
        )
    } else {
        GpuColorAttachmentLoad::Load
    };
    let attachment =
        GpuRenderColorAttachment::new(view.clone(), load, GpuAttachmentStore::Store, None)
            .map_err(|e| gpu("F3E attachment", e))?;
    GpuRenderOperation::new([attachment], None, draws, None).map_err(|e| gpu("F3E pass", e))
}

fn append(
    operations: &mut Vec<GpuRenderOperation>,
    op: GpuRenderOperation,
) -> Result<(), Render2dExecutionError> {
    if operations.len() >= MAX_OPERATIONS {
        return Err(failure("sample-space work-node budget exceeded"));
    }
    operations.push(op);
    Ok(())
}

fn rectangle(
    vertices: &mut Vec<f32>,
    bounds: [f64; 4],
    viewport: [u32; 2],
    color: [f32; 4],
    coordinates: [f64; 2],
) {
    let [left, top, right, bottom] = bounds;
    for [x, y] in [
        [left, top],
        [right, top],
        [left, bottom],
        [left, bottom],
        [right, top],
        [right, bottom],
    ] {
        vertices.extend([
            physical_x_to_ndc(x, viewport[0]),
            physical_y_to_ndc(y, viewport[1]),
            f32_from_f64(x + coordinates[0]),
            f32_from_f64(y + coordinates[1]),
            color[0],
            color[1],
            color[2],
            color[3],
        ]);
    }
}

fn draw(
    pipeline: &GpuRenderPipelineDescriptor,
    sampled: Option<(u32, &GpuTextureViewHandle)>,
    clipped: Option<&super::clip::ClipGpu>,
    gradient: Option<&GpuBufferHandle>,
    vertices: &[f32],
    extent: [u32; 2],
    resources: &mut GpuResourceScope,
) -> Result<GpuRenderDraw, Render2dExecutionError> {
    let mut values = sampled
        .map(|(binding, view)| texture_binding(binding, view))
        .transpose()?
        .into_iter()
        .collect::<Vec<_>>();
    if let Some(clipped) = clipped {
        values.extend(super::clip::bindings(clipped, 4, 5)?);
    }
    if let Some(buffer) = gradient {
        values.push(GpuRuntimeBindingValue::whole_buffer(0, 1, buffer));
    }
    let bindings = pipeline
        .runtime_bindings(values)
        .map_err(|e| gpu("F3E sample bindings", e))?;
    vector::vector_draw(pipeline.clone(), bindings, vertices, extent, resources)
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

/// One derived, preflight-only solid-vector snapshot. Its fields are not
/// authored semantic state; the immutable F1 plan remains the only authority.
struct Inspected {
    meshes: Vec<Option<geometry::VectorMesh>>,
    bounds: [u32; 4],
    peak_group_depth: usize,
}

struct GroupFrame<'a> {
    group: &'a Render2dGroup,
    visible: bool,
    clip: Option<super::clip::ClipGpu>,
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

fn inspect(
    plan: &scene::Plan<'_>,
    target: &AdmittedTarget,
) -> Result<Inspected, Render2dExecutionError> {
    let mut meshes = Vec::with_capacity(plan.events.len());
    let mut bounds = [u32::MAX, u32::MAX, 0, 0];
    let mut depth = 0usize;
    let mut peak = 0usize;
    for event in &plan.events {
        match event {
            scene::Event::BeginGroup { group, path, .. } => {
                if !group.shadows().is_empty() {
                    return Err(Render2dUnsupportedContent::Group {
                        root_index: path[0],
                    }
                    .into());
                }
                depth += 1;
                peak = peak.max(depth);
                meshes.push(None);
            }
            scene::Event::EndGroup => {
                depth -= 1;
                meshes.push(None);
            }
            scene::Event::Item {
                item,
                path,
                to_root,
                ..
            } => {
                if !matches!(
                    item.primitive(),
                    Render2dPrimitive::Fill { .. } | Render2dPrimitive::Stroke { .. }
                ) {
                    return Err(Render2dUnsupportedContent::Group {
                        root_index: path[0],
                    }
                    .into());
                }
                let [a, b, c, d, tx, ty] = to_root.coefficients();
                let transform = Render2dAffineTransform::new(a, b, c, d, tx, ty).map_err(|_| {
                    failure(format!("entry path {path:?}: unrepresentable transform"))
                })?;
                let derived = Render2dItem::new(
                    item.primitive().clone(),
                    transform,
                    Vec::new(),
                    item.opacity(),
                );
                let realized = geometry::realize(
                    &derived,
                    path[0],
                    target.raster_scale(),
                    target.canvas(),
                    target.max_buffer_bytes(),
                )?;
                if let Some(ref mesh) = realized {
                    let [left, top, width, height] = mesh.bounds;
                    bounds[0] = bounds[0].min(left);
                    bounds[1] = bounds[1].min(top);
                    bounds[2] = bounds[2].max(left.saturating_add(width));
                    bounds[3] = bounds[3].max(top.saturating_add(height));
                }
                meshes.push(realized);
            }
        }
    }
    debug_assert_eq!(depth, 0);
    Ok(Inspected {
        meshes,
        bounds,
        peak_group_depth: peak,
    })
}

/// Compiles a real (currently solid-vector-only) F1 group tree through one
/// globally-phased sample plane. This is an admitted *staging subset* of the
/// same future mixed-content F3E compiler, not a second persistent renderer.
/// All nonadmitted semantics reject before an external target is modified.
pub(in crate::runtime::execution_2d) fn lower(
    context: &GpuContext,
    target: &AdmittedTarget,
    plan: &scene::Plan<'_>,
) -> Result<Vec<GpuRenderOperation>, Render2dExecutionError> {
    let roles = context
        .device_facts()
        .admission_contract()
        .format_roles()
        .collect::<BTreeSet<_>>();
    for (format, role) in [
        (
            GpuTextureFormat::Rgba16Float,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Sampled),
        (FIELD_FORMAT, GpuFormatRole::ColorAttachment),
        (FIELD_FORMAT, GpuFormatRole::Sampled),
    ] {
        if !roles.contains(&(format, role)) {
            return Err(failure(format!(
                "sample target requires admitted {format:?} {role:?}"
            )));
        }
    }
    let Inspected {
        meshes,
        bounds,
        peak_group_depth: peak,
    } = inspect(plan, target)?;
    if bounds[0] >= bounds[2] || bounds[1] >= bounds[3] {
        return Ok(Vec::new());
    }
    let side = tile_side(target, peak)?;
    let x0 = bounds[0] / side * side;
    let y0 = bounds[1] / side * side;
    let x_end = bounds[2].min(target.physical_width);
    let y_end = bounds[3].min(target.physical_height);
    let cols = u64::from((x_end - x0).div_ceil(side));
    let rows = u64::from((y_end - y0).div_ceil(side));
    if cols.checked_mul(rows).is_none_or(|count| count > MAX_TILES) {
        return Err(failure("sample-space tile count exceeds bounded budget"));
    }

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
    // The same immutable F3B stop payload is uploaded once per authored
    // vector occurrence. Tiles reuse its source-backed RunenGPU handle;
    // previously a large scene could upload the same payload per tile.
    const MAX_GRADIENT_PAYLOAD_BYTES: u64 = 64 * 1024 * 1024;
    let mut allocated_gradient_bytes = 0_u64;
    let gradient_buffers = meshes
        .iter()
        .map(|maybe_mesh| {
            let Some(mesh) = maybe_mesh else { return Ok(None) };
            let Some(words) = vector::gradient_payload(mesh)? else { return Ok(None) };
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
            let mut any_mesh = false;
            for mesh in meshes.iter().flatten() {
                let b = mesh.bounds;
                if b[0] < end[0]
                    && b[0].saturating_add(b[2]) > origin[0]
                    && b[1] < end[1]
                    && b[1].saturating_add(b[3]) > origin[1]
                {
                    any_mesh = true;
                    break;
                }
            }
            if !any_mesh {
                continue;
            }
            append(&mut operations, operation(&layers[0], true, Vec::new())?)?;
            let mut depth = 0usize;
            let mut group_stack = Vec::<GroupFrame<'_>>::new();
            let tile_bounds = [origin[0], origin[1], end[0], end[1]];
            for (index, event) in plan.events.iter().enumerate() {
                match event {
                    scene::Event::BeginGroup {
                        group,
                        path,
                        parent_to_root,
                    } => {
                        depth += 1;
                        let parent_visible = group_stack.last().is_none_or(|f| f.visible);
                        let clip = if !parent_visible || group.clips().is_empty() {
                            None
                        } else {
                            prepare_clip(
                                group.clips(),
                                ClipOwner {
                                    parent_to_root: *parent_to_root,
                                    root_index: path[0],
                                },
                                tile_bounds,
                                target,
                                &mut clip_budget,
                                &mut resources,
                            )?
                        };
                        let visible =
                            parent_visible && (group.clips().is_empty() || clip.is_some());
                        group_stack.push(GroupFrame {
                            group,
                            visible,
                            clip,
                        });
                        if visible {
                            append(
                                &mut operations,
                                operation(&layers[depth], true, Vec::new())?,
                            )?;
                        }
                    }
                    scene::Event::EndGroup => {
                        let frame = group_stack.pop().expect("balanced F1 group plan");
                        if frame.visible {
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
                        }
                        depth -= 1;
                    }
                    scene::Event::Item {
                        item,
                        path,
                        parent_to_root,
                        ..
                    } => {
                        if group_stack.last().is_some_and(|frame| !frame.visible) {
                            continue;
                        }
                        let Some(mesh) = &meshes[index] else { continue };
                        let [left, top, width, height] = mesh.bounds;
                        if left >= end[0]
                            || left.saturating_add(width) <= origin[0]
                            || top >= end[1]
                            || top.saturating_add(height) <= origin[1]
                        {
                            continue;
                        }
                        let item_clip = if item.clips().is_empty() {
                            None
                        } else {
                            let Some(mask) = prepare_clip(
                                item.clips(),
                                ClipOwner {
                                    parent_to_root: *parent_to_root,
                                    root_index: path[0],
                                },
                                tile_bounds,
                                target,
                                &mut clip_budget,
                                &mut resources,
                            )?
                            else {
                                continue;
                            };
                            Some(mask)
                        };
                        let mut coverage_vertices = Vec::with_capacity(mesh.triangles.len() * 8);
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
                    }
                }
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
