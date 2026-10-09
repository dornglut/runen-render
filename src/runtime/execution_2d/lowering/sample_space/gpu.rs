//! F3E source-backed RunenGPU pipelines, transient sample scratch and ordered work construction.
use super::*;

pub(super) fn pipeline(
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
                        GpuBindingLayoutRefinement::new(key).with_texture_sample_class(
                            if entry == "fs_sample_image" {
                                GpuTextureSampleClass::FloatFilterable
                            } else {
                                GpuTextureSampleClass::FloatUnfilterable
                            },
                        ),
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

pub(super) fn scratch(
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

pub(super) fn operation(
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

pub(super) fn append(
    operations: &mut Vec<GpuRenderOperation>,
    op: GpuRenderOperation,
) -> Result<(), Render2dExecutionError> {
    if operations.len() >= MAX_OPERATIONS {
        return Err(failure("sample-space work-node budget exceeded"));
    }
    operations.push(op);
    Ok(())
}

pub(super) fn rectangle(
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

pub(super) fn draw(
    pipeline: &GpuRenderPipelineDescriptor,
    sampled: Option<(u32, &GpuTextureViewHandle)>,
    clipped: Option<&super::super::clip::ClipGpu>,
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
        values.extend(super::super::clip::bindings(clipped, 4, 5)?);
    }
    if let Some(buffer) = gradient {
        values.push(GpuRuntimeBindingValue::whole_buffer(0, 1, buffer));
    }
    let bindings = pipeline
        .runtime_bindings(values)
        .map_err(|e| gpu("F3E sample bindings", e))?;
    vector::vector_draw(pipeline.clone(), bindings, vertices, extent, resources)
}

pub(super) fn image_draw(
    pipeline: &GpuRenderPipelineDescriptor,
    patch: &PreparedPatch,
    vertices: &[f32],
    extent: [u32; 2],
    resources: &mut GpuResourceScope,
) -> Result<GpuRenderDraw, Render2dExecutionError> {
    let bindings = pipeline
        .runtime_bindings([
            texture_binding(2, &patch.image)?,
            GpuRuntimeBindingValue::whole_buffer(0, 3, &patch.parameters),
        ])
        .map_err(|e| gpu("F3E sampled image source bindings", e))?;
    vector::vector_draw(pipeline.clone(), bindings, vertices, extent, resources)
}

