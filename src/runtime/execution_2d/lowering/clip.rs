//! Disposable packed 4x4 clip-mask uploads through the ordinary RunenGPU contract.
use super::{AdmittedTarget, f32_from_u32, gpu, texture_binding};
use crate::execution_2d::{Render2dClipError, Render2dExecutionError};
use crate::runtime::execution_2d::clip::{ClipMask, failure};
use runen_gpu::*;

const FORMAT: GpuTextureFormat = GpuTextureFormat::Rgba8Unorm;

pub(super) struct ClipGpu {
    pub(super) texture: GpuTextureViewHandle,
    pub(super) parameters: GpuBufferHandle,
}

pub(super) fn upload(
    target: &AdmittedTarget,
    mask: &ClipMask,
    resources: &mut GpuResourceScope,
) -> Result<ClipGpu, Render2dExecutionError> {
    if !target.clip_format {
        return Err(failure(mask.root_index, Render2dClipError::FormatUnsupported));
    }
    let [width, height] = mask.extent;
    if width > target.max_texture_dimension_2d
        || height > target.max_texture_dimension_2d {
        return Err(failure(mask.root_index, Render2dClipError::ResourceLimit));
    }
    let name = format!("runen-render conjunctive clip {}", mask.root_index);
    let label = GpuResourceLabel::new(&name).map_err(|e| gpu("clip mask label", e))?;
    let transfer = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        &name, &mask.rgba,
    ).map_err(|e| gpu("clip mask bytes", e))?;
    let physical_extent = GpuTextureExtent::new(
        &label, GpuTextureDimension::D2, width, height, 1,
    ).map_err(|e| gpu("clip mask extent", e))?;
    let row=width.checked_mul(4)
        .ok_or_else(||failure(mask.root_index,Render2dClipError::ResourceLimit))?;
    let prepared = GpuPreparedTextureData::new(
        &label, transfer, FORMAT, physical_extent, row, 0,
    ).map_err(|e| gpu("clip mask upload", e))?;
    let texture = resources.texture(
        GpuTextureDescriptor::ordinary_owned_2d(
            &name,
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            width, height, FORMAT,
            [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
            GpuTextureInitialization::Prepared(prepared),
        ).map_err(|e| gpu("clip mask descriptor", e))?,
    ).map_err(|e|gpu("clip mask texture",e))?;
    let texture = resources.texture_view(
        GpuTextureViewDescriptor::ordinary_full_owned(
            format!("{name} view"), &texture,
        ).map_err(|e|gpu("clip mask view descriptor",e))?,
    ).map_err(|e|gpu("clip mask view",e))?;
    let values=[
        f32_from_u32(mask.origin[0]), f32_from_u32(mask.origin[1]),
        f32_from_u32(width), f32_from_u32(height),
    ];
    let param = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "runen-render clip mask coordinates", &values,
    ).map_err(|e|gpu("clip mask coordinates",e))?;
    let parameters=resources.buffer(
        GpuBufferDescriptor::ordinary_owned(
            "runen-render clip mask coordinates",
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            param.layout().byte_len(),
            [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
            GpuBufferInitialization::Prepared(param),
        ).map_err(|e|gpu("clip mask parameter descriptor",e))?,
    ).map_err(|e|gpu("clip mask parameter",e))?;
    Ok(ClipGpu { texture, parameters })
}

pub(super) fn bindings(
    gpu: &ClipGpu,
    mask_binding: u32,
    parameter_binding: u32,
) -> Result<[GpuRuntimeBindingValue; 2], Render2dExecutionError> {
    Ok([
        texture_binding(mask_binding,&gpu.texture)?,
        GpuRuntimeBindingValue::whole_buffer(0, parameter_binding, &gpu.parameters),
    ])
}
