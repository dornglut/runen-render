//! Private source-neutral image texture upload and ordered patch composition.
use super::{AdmittedTarget, gpu, physical_x_to_ndc, physical_y_to_ndc, texture_binding, vector};
use crate::execution_2d::{Render2dExecutionError, Render2dImageError};
use crate::runtime::execution_2d::image::{ImagePatchWork, failure};
use runen_gpu::*;

const FORMAT: GpuTextureFormat = GpuTextureFormat::Rgba8UnormSrgb;

pub(super) fn upload(
    target: &AdmittedTarget,
    patch: &ImagePatchWork,
    resources: &mut GpuResourceScope,
) -> Result<GpuTextureViewHandle, Render2dExecutionError> {
    let extent = patch.source.extent();
    if extent.width() > target.max_texture_dimension_2d
        || extent.height() > target.max_texture_dimension_2d
    {
        return Err(failure(patch.root_index, Render2dImageError::ResourceLimit));
    }
    let row = extent
        .width()
        .checked_mul(4)
        .ok_or_else(|| failure(patch.root_index, Render2dImageError::ResourceLimit))?;
    let name = format!(
        "runen-render immutable 2D image {}",
        patch.resource_id.get()
    );
    let label = GpuResourceLabel::new(&name).map_err(|e| gpu("image label", e))?;
    let values =
        PreparedGpuData::<TransferData>::ordinary_pod_transfer(&name, patch.source.rgba8_srgb())
            .map_err(|e| gpu("image bytes", e))?;
    let physical_extent = GpuTextureExtent::new(
        &label,
        GpuTextureDimension::D2,
        extent.width(),
        extent.height(),
        1,
    )
    .map_err(|e| gpu("image extent", e))?;
    let prepared = GpuPreparedTextureData::new(&label, values, FORMAT, physical_extent, row, 0)
        .map_err(|e| gpu("image upload layout", e))?;
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                &name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                extent.width(),
                extent.height(),
                FORMAT,
                [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
                GpuTextureInitialization::Prepared(prepared),
            )
            .map_err(|e| gpu("image texture descriptor", e))?,
        )
        .map_err(|e| gpu("image texture", e))?;
    resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(format!("{name} view"), &texture)
                .map_err(|e| gpu("image view descriptor", e))?,
        )
        .map_err(|e| gpu("image view", e))
}

pub(super) fn lower(
    target: &AdmittedTarget,
    patch: &ImagePatchWork,
    image_view: &GpuTextureViewHandle,
    clipped: Option<&super::clip::ClipGpu>,
    resources: &mut GpuResourceScope,
) -> Result<GpuRenderOperation, Render2dExecutionError> {
    let prepared = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "runen-render 2D image patch parameters",
        &patch.payload,
    )
    .map_err(|e| gpu("image patch payload", e))?;
    let bytes = prepared.layout().byte_len();
    let parameter_buffer = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "runen-render 2D image patch parameters",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                bytes,
                [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Prepared(prepared),
            )
            .map_err(|e| gpu("image parameter descriptor", e))?,
        )
        .map_err(|e| gpu("image parameter buffer", e))?;
    let pipeline = vector::vector_pipeline(target.format, true, false, true, clipped.is_some())?;
    let mut values = vec![
        texture_binding(2, image_view)?,
        GpuRuntimeBindingValue::whole_buffer(0, 3, &parameter_buffer),
    ];
    if let Some(clip) = clipped {
        values.extend(super::clip::bindings(clip, 4, 5)?);
    }
    let bindings = pipeline
        .runtime_bindings(values)
        .map_err(|e| gpu("image runtime bindings", e))?;
    let [left, top, right, bottom] = patch.bounds;
    let vertex = |x: f64, y: f64| {
        [
            physical_x_to_ndc(x, target.physical_width),
            physical_y_to_ndc(y, target.physical_height),
            0.0_f32,
            0.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ]
    };
    let (x0, y0, x1, y1) = (
        f64::from(left),
        f64::from(top),
        f64::from(right),
        f64::from(bottom),
    );
    let vertices = [
        vertex(x0, y0),
        vertex(x1, y0),
        vertex(x0, y1),
        vertex(x0, y1),
        vertex(x1, y0),
        vertex(x1, y1),
    ]
    .concat();
    let draw = vector::vector_draw(
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
    .map_err(|e| gpu("image target attachment", e))?;
    GpuRenderOperation::new([attachment], None, [draw], None)
        .map_err(|e| gpu("image composition operation", e))
}
