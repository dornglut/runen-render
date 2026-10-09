//! Source-backed immutable image upload for correlated 2D composition.
use super::{AdmittedTarget, gpu};
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

