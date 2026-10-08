//! Contribution-local vector coverage and once-only source-over realization.
use super::{
    AdmittedTarget, FIELD_FORMAT, VERTEX_STRIDE, create_vertex_buffer, f32_from_f64, f32_from_u32,
    gpu, linear_color, physical_x_to_ndc, physical_y_to_ndc, texture_binding, vertex_count,
};
use crate::execution_2d::Render2dExecutionError;
use runen_gpu::*;

fn vector_pipeline(
    format: GpuTextureFormat,
    compose: bool,
) -> Result<GpuRenderPipelineDescriptor, Render2dExecutionError> {
    let source = crate::runtime::program::retained_vector_source().map_err(|error| {
        Render2dExecutionError::Program {
            stage: "retained vector source",
            detail: format!("{error:?}"),
        }
    })?;
    let vertex = GpuEntryPointName::new("vs_main").map_err(|e| gpu("vector entry", e))?;
    let fragment = GpuEntryPointName::new(if compose { "fs_compose" } else { "fs_coverage" })
        .map_err(|e| gpu("vector entry", e))?;
    let refinements = if compose {
        vec![
            GpuBindingLayoutRefinement::new(
                GpuBindingKey::try_new(0, 0).map_err(|e| gpu("mask binding key", e))?,
            )
            .with_texture_sample_class(GpuTextureSampleClass::FloatUnfilterable),
        ]
    } else {
        vec![]
    };
    let program =
        GpuProgramDescriptor::new(source, [vertex.clone(), fragment.clone()], refinements)
            .map_err(|e| gpu("vector program", e))?;
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
    .map_err(|e| gpu("vector vertex layout", e))?;
    let component = GpuBlendComponent::new(
        GpuBlendFactor::One,
        GpuBlendFactor::OneMinusSrcAlpha,
        GpuBlendOperation::Add,
    )
    .map_err(|e| gpu("vector source-over", e))?;
    let color = GpuColorTargetStateDescriptor::new(
        format,
        compose.then_some(GpuBlendState::new(component, component)),
        GpuColorWriteMask::ALL,
    )
    .map_err(|e| gpu("vector color target", e))?;
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([layout]).map_err(|e| gpu("vector vertex input", e))?,
        Some(GpuFragmentOutputStateDescriptor::new([color])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .map_err(|e| gpu("vector pipeline state", e))?;
    GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .map_err(|e| gpu("vector pipeline", e))
}

fn vector_draw(
    pipeline: GpuRenderPipelineDescriptor,
    bindings: GpuRuntimeBindingSet,
    vertices: &[f32],
    extent: [u32; 2],
    resources: &mut GpuResourceScope,
) -> Result<GpuRenderDraw, Render2dExecutionError> {
    let buffer = create_vertex_buffer(resources, vertices)?;
    let binding = GpuVertexBufferBinding::new(
        0,
        &buffer,
        GpuBufferRange::whole(&buffer).map_err(|e| gpu("vector range", e))?,
    )
    .map_err(|e| gpu("vector binding", e))?;
    GpuRenderDraw::new(
        pipeline,
        bindings,
        [binding],
        None,
        GpuDrawIntent::direct(
            GpuDrawRange::new(0, vertex_count(vertices)?)
                .map_err(|e| gpu("vector draw range", e))?,
            GpuDrawRange::new(0, 1).map_err(|e| gpu("vector instance", e))?,
        ),
        GpuViewport::new(
            0.0,
            0.0,
            f32_from_u32(extent[0]),
            f32_from_u32(extent[1]),
            0.0,
            1.0,
        )
        .map_err(|e| gpu("vector viewport", e))?,
        GpuScissorRect::new(0, 0, extent[0], extent[1]).map_err(|e| gpu("vector scissor", e))?,
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).map_err(|e| gpu("vector blend constant", e))?,
        0,
    )
    .map_err(|e| gpu("vector draw", e))
}

pub(super) fn lower(
    target: &AdmittedTarget,
    mesh: &crate::runtime::execution_2d::vector::VectorMesh,
    mask: &GpuTextureViewHandle,
    extent: [u32; 2],
    resources: &mut GpuResourceScope,
) -> Result<[GpuRenderOperation; 2], Render2dExecutionError> {
    let [left, top, width, height] = mesh.bounds;
    let mut vertices = Vec::with_capacity(mesh.triangles.len() * 8);
    for [x, y] in &mesh.triangles {
        let x = (*x - f64::from(left))
            * f64::from(crate::runtime::program::abi::VECTOR_COVERAGE_AXIS_SAMPLES);
        let y = (*y - f64::from(top))
            * f64::from(crate::runtime::program::abi::VECTOR_COVERAGE_AXIS_SAMPLES);
        vertices.extend_from_slice(&[
            physical_x_to_ndc(x, extent[0]),
            physical_y_to_ndc(y, extent[1]),
            0.0,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]);
    }
    let pipeline = vector_pipeline(FIELD_FORMAT, false)?;
    let bindings = pipeline
        .runtime_bindings([])
        .map_err(|e| gpu("coverage bindings", e))?;
    let draw = vector_draw(pipeline, bindings, &vertices, extent, resources)?;
    let attachment = GpuRenderColorAttachment::new(
        mask.clone(),
        GpuColorAttachmentLoad::Clear(
            GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0).map_err(|e| gpu("mask clear", e))?,
        ),
        GpuAttachmentStore::Store,
        None,
    )
    .map_err(|e| gpu("coverage attachment", e))?;
    let coverage = GpuRenderOperation::new([attachment], None, [draw], None)
        .map_err(|e| gpu("coverage operation", e))?;
    let mut color = linear_color(mesh.color);
    color[3] *= f32_from_f64(mesh.opacity);
    let x0 = f64::from(left);
    let y0 = f64::from(top);
    let x1 = x0 + f64::from(width);
    let y1 = y0 + f64::from(height);
    let vertex = |x, y, u, v| {
        [
            physical_x_to_ndc(x, target.physical_width),
            physical_y_to_ndc(y, target.physical_height),
            f32_from_f64(u),
            f32_from_f64(v),
            color[0],
            color[1],
            color[2],
            color[3],
        ]
    };
    let vertices = [
        vertex(x0, y0, 0.0, 0.0),
        vertex(x1, y0, f64::from(width), 0.0),
        vertex(x0, y1, 0.0, f64::from(height)),
        vertex(x0, y1, 0.0, f64::from(height)),
        vertex(x1, y0, f64::from(width), 0.0),
        vertex(x1, y1, f64::from(width), f64::from(height)),
    ]
    .concat();
    let pipeline = vector_pipeline(target.format, true)?;
    let bindings = pipeline
        .runtime_bindings([texture_binding(0, mask)?])
        .map_err(|e| gpu("composition bindings", e))?;
    let draw = vector_draw(
        pipeline,
        bindings,
        &vertices,
        [target.physical_width, target.physical_height],
        resources,
    )?;
    let attachment = GpuRenderColorAttachment::new(
        target.view.clone(),
        GpuColorAttachmentLoad::Load,
        GpuAttachmentStore::Store,
        None,
    )
    .map_err(|e| gpu("composition attachment", e))?;
    let compose = GpuRenderOperation::new([attachment], None, [draw], None)
        .map_err(|e| gpu("composition operation", e))?;
    Ok([coverage, compose])
}
