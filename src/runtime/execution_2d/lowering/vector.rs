//! Reusable source-backed GPU vertex submission for one correlated sample pass.
use super::{create_vertex_buffer, f32_from_u32, gpu, vertex_count};
use crate::execution_2d::Render2dExecutionError;
use runen_gpu::*;

pub(super) fn vector_draw(
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

