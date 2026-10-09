//! Private bounded image mapping into independent ordered painter patches.
use super::support::NeutralMesh;
use crate::composition_2d::{
    Render2dImagePrimitive, Render2dImageResource, Render2dItem, Render2dRect, Render2dResourceId,
};
use crate::execution_2d::{Render2dExecutionError, Render2dImageError};

#[derive(Debug)]
pub(super) struct ImagePatchWork {
    pub(super) root_index: usize,
    pub(super) resource_id: Render2dResourceId,
    pub(super) source: Render2dImageResource,
    pub(super) bounds: [u32; 4],
    pub(super) payload: [f32; 20],
}

pub(super) fn failure(root_index: usize, kind: Render2dImageError) -> Render2dExecutionError {
    Render2dExecutionError::Image { root_index, kind }
}

const MAX_IMAGE_PATCHES: usize = 65_536;

fn narrow(value: f64, root_index: usize) -> Result<f32, Render2dExecutionError> {
    let result = value as f32;
    if !result.is_finite()
        || (f64::from(result) - value).abs() > 1.0e-5_f64.max(value.abs() * 1.0e-7)
    {
        Err(failure(root_index, Render2dImageError::PrecisionLimit))
    } else {
        Ok(result)
    }
}

/// Derives signed, off-canvas neutral image-patch geometry before visible-paint
/// opacity, source-texel sampling and final-target clipping.
///
/// This is a disposable private physical approximation of immutable F1
/// destination patches, not a source-authority image-alpha mask. Group effects
/// and item clips consume the patches at a later F3F lowering boundary.
#[allow(dead_code, reason = "awaiting F3F group lowering")]
pub(super) fn neutral_support(
    item: &Render2dItem,
    image: &Render2dImagePrimitive,
    root_index: usize,
    raster_scale: f64,
    max_buffer_bytes: u64,
) -> Result<Option<NeutralMesh>, Render2dExecutionError> {
    if image.patches().len() > MAX_IMAGE_PATCHES {
        return Err(failure(root_index, Render2dImageError::ResourceLimit));
    }
    let [a, b, c, d, tx, ty] = item.local_to_parent().components();
    let det = a.mul_add(d, -(b * c));
    if !det.is_finite() || !raster_scale.is_finite() || raster_scale <= 0.0 {
        return Err(failure(root_index, Render2dImageError::PrecisionLimit));
    }
    if det == 0.0 {
        return Ok(None);
    }
    let mut triangles = Vec::new();
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for patch in image.patches() {
        let dest = patch.destination();
        let src = patch.source();
        if dest.width() == 0.0
            || dest.height() == 0.0
            || src.width() == 0.0
            || src.height() == 0.0
        {
            continue;
        }
        let points = [
            [dest.x(), dest.y()],
            [dest.x() + dest.width(), dest.y()],
            [dest.x() + dest.width(), dest.y() + dest.height()],
            [dest.x(), dest.y() + dest.height()],
        ];
        let mut physical = [[0.0; 2]; 4];
        for (index, [x, y]) in points.into_iter().enumerate() {
            let px = a.mul_add(x, c.mul_add(y, tx)) * raster_scale;
            let py = b.mul_add(x, d.mul_add(y, ty)) * raster_scale;
            if !px.is_finite()
                || !py.is_finite()
                || px.abs().max(py.abs()) * f64::EPSILON >= 1.0 / 4096.0
            {
                return Err(failure(root_index, Render2dImageError::PrecisionLimit));
            }
            physical[index] = [px, py];
        }
        let count = triangles
            .len()
            .checked_add(6)
            .ok_or_else(|| failure(root_index, Render2dImageError::ResourceLimit))?;
        let bytes = u64::try_from(count)
            .ok()
            .and_then(|n| n.checked_mul(crate::runtime::program::abi::COMPOSITION_VERTEX_STRIDE))
            .ok_or_else(|| failure(root_index, Render2dImageError::ResourceLimit))?;
        if bytes > max_buffer_bytes {
            return Err(failure(root_index, Render2dImageError::ResourceLimit));
        }
        for point in [
            physical[0],
            physical[1],
            physical[2],
            physical[0],
            physical[2],
            physical[3],
        ] {
            bounds[0] = bounds[0].min(point[0]);
            bounds[1] = bounds[1].min(point[1]);
            bounds[2] = bounds[2].max(point[0]);
            bounds[3] = bounds[3].max(point[1]);
            triangles.push(point);
        }
    }
    Ok((!triangles.is_empty()).then_some(NeutralMesh { triangles, bounds }))
}

pub(super) fn realize(
    item: &Render2dItem,
    image: &Render2dImagePrimitive,
    source: &Render2dImageResource,
    root_index: usize,
    raster_scale: f64,
    canvas: [f64; 2],
    texture_limit: u32,
) -> Result<Vec<ImagePatchWork>, Render2dExecutionError> {
    if image.patches().is_empty() || item.opacity().get() == 0.0 {
        return Ok(Vec::new());
    }
    let extent = source.extent();
    let byte_len = source.rgba8_srgb().len();
    if extent.width() > texture_limit
        || extent.height() > texture_limit
        || byte_len > 64 * 1024 * 1024
        || image.patches().len() > MAX_IMAGE_PATCHES
    {
        return Err(failure(root_index, Render2dImageError::ResourceLimit));
    }
    let [a, b, c, d, tx, ty] = item.local_to_parent().components();
    let determinant = a.mul_add(d, -(b * c));
    // A singular affine transform has no 2D area to paint.
    if determinant == 0.0 {
        return Ok(Vec::new());
    }
    if !determinant.is_finite() {
        return Err(failure(root_index, Render2dImageError::PrecisionLimit));
    }
    let inverse = [
        d / determinant / raster_scale,
        -c / determinant / raster_scale,
        c.mul_add(ty, -(d * tx)) / determinant,
        -b / determinant / raster_scale,
        a / determinant / raster_scale,
        b.mul_add(tx, -(a * ty)) / determinant,
    ];
    for value in inverse {
        narrow(value, root_index)?;
    }
    let mut output = Vec::new();
    for patch in image.patches() {
        let dest: Render2dRect = patch.destination();
        let src = patch.source();
        if dest.width() == 0.0 || dest.height() == 0.0 || src.width() == 0.0 || src.height() == 0.0
        {
            continue;
        }
        let corners = [
            [dest.x(), dest.y()],
            [dest.x() + dest.width(), dest.y()],
            [dest.x(), dest.y() + dest.height()],
            [dest.x() + dest.width(), dest.y() + dest.height()],
        ];
        let mut left = canvas[0];
        let mut top = canvas[1];
        let mut right = 0.0_f64;
        let mut bottom = 0.0_f64;
        for [x, y] in corners {
            let px = a.mul_add(x, c.mul_add(y, tx)) * raster_scale;
            let py = b.mul_add(x, d.mul_add(y, ty)) * raster_scale;
            if !px.is_finite()
                || !py.is_finite()
                || ![px, py]
                    .into_iter()
                    .all(|v| v.abs() * f64::EPSILON < 1.0 / 4096.0)
            {
                return Err(failure(root_index, Render2dImageError::PrecisionLimit));
            }
            left = left.min(px.max(0.0));
            top = top.min(py.max(0.0));
            right = right.max(px.min(canvas[0]));
            bottom = bottom.max(py.min(canvas[1]));
        }
        let left = left.max(0.0).min(canvas[0]);
        let top = top.max(0.0).min(canvas[1]);
        let right = right.max(0.0).min(canvas[0]);
        let bottom = bottom.max(0.0).min(canvas[1]);
        if right <= left || bottom <= top {
            continue;
        }
        // The fragment shader evaluates local positions in f32. Even exact
        // f32 endpoints can be unusable when a large inverse translation
        // cancels a large authored destination coordinate: the subpixel phase
        // disappears in the intermediate dot products. Bound the worst-case
        // local f32 rounding in physical pixels, rather than checking only
        // the f64-to-f32 conversion of individual parameters.
        let maximum_local_x =
            (inverse[0].abs() * canvas[0] + inverse[1].abs() * canvas[1] + inverse[2].abs())
                .max(dest.x().abs())
                .max((dest.x() + dest.width()).abs());
        let maximum_local_y =
            (inverse[3].abs() * canvas[0] + inverse[4].abs() * canvas[1] + inverse[5].abs())
                .max(dest.y().abs())
                .max((dest.y() + dest.height()).abs());
        let roundoff_pixels = 4.0
            * f64::from(f32::EPSILON)
            * raster_scale
            * (maximum_local_x * (a.abs() + b.abs()) + maximum_local_y * (c.abs() + d.abs()));
        if !roundoff_pixels.is_finite() || roundoff_pixels > 1.0 / 16.0 {
            return Err(failure(root_index, Render2dImageError::PrecisionLimit));
        }
        // Pixel-center quad is conservatively expanded to include every partially
        // covered pixel; the shader tests continuous patch and canvas geometry
        // at the accepted 4x4 sample lattice.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bounds = [
            left.floor() as u32,
            top.floor() as u32,
            right.ceil() as u32,
            bottom.ceil() as u32,
        ];
        let floats = [
            inverse[0],
            inverse[1],
            inverse[2],
            item.opacity().get(),
            inverse[3],
            inverse[4],
            inverse[5],
            0.0,
            src.x(),
            src.y(),
            src.width(),
            src.height(),
            dest.x(),
            dest.y(),
            dest.width(),
            dest.height(),
            canvas[0],
            canvas[1],
            0.0,
            0.0,
        ];
        let mut payload = [0.0_f32; 20];
        for (index, number) in floats.into_iter().enumerate() {
            payload[index] = narrow(number, root_index)?;
        }
        if payload[12] + payload[14] == payload[12] || payload[13] + payload[15] == payload[13] {
            return Err(failure(root_index, Render2dImageError::PrecisionLimit));
        }
        output.push(ImagePatchWork {
            root_index,
            resource_id: image.resource_id(),
            source: source.clone(),
            bounds,
            payload,
        });
    }
    Ok(output)
}

#[cfg(test)]
mod neutral_tests {
    use super::*;
    use crate::composition_2d::{
        Render2dAffineTransform, Render2dImagePatch,
        Render2dImageSourceRect, Render2dOpacity, Render2dPixelExtent, Render2dResourceId,
    };

    #[test]
    fn completely_transparent_offscreen_image_retains_destination_support() {
        let extent = Render2dPixelExtent::new(2, 2).unwrap();
        let image = Render2dImagePrimitive::new(
            Render2dResourceId::new(7).unwrap(),
            extent,
            vec![Render2dImagePatch::new(
                Render2dImageSourceRect::new(0.0, 0.0, 2.0, 2.0).unwrap(),
                Render2dRect::new(-8.0, -3.0, 4.0, 2.0).unwrap(),
            )],
        ).unwrap();
        let item = Render2dItem::new(
            crate::composition_2d::Render2dPrimitive::Image(image.clone()),
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let support = neutral_support(&item, &image, 0, 1.0, 4096)
            .unwrap().expect("opacity and offscreen canvas cannot erase neutral support");
        assert_eq!(support.bounds, [-8.0, -3.0, -4.0, -1.0]);
        assert_eq!(support.triangles.len(), 6);
        let error = neutral_support(&item, &image, 0, 1.0, 32).unwrap_err();
        assert!(matches!(error,
            Render2dExecutionError::Image {
                root_index: 0,
                kind: Render2dImageError::ResourceLimit
            }
        ));
    }

    #[test]
    fn image_neutral_support_preserves_affine_geometry_and_empty_source() {
        let extent = Render2dPixelExtent::new(4, 4).unwrap();
        let image = Render2dImagePrimitive::new(
            Render2dResourceId::new(1).unwrap(), extent,
            vec![
                Render2dImagePatch::new(
                    Render2dImageSourceRect::new(0.0, 0.0, 1.0, 1.0).unwrap(),
                    Render2dRect::new(0.0, 0.0, 1.0, 2.0).unwrap(),
                ),
                Render2dImagePatch::new(
                    Render2dImageSourceRect::new(0.0, 0.0, 0.0, 1.0).unwrap(),
                    Render2dRect::new(6.0, 0.0, 2.0, 2.0).unwrap(),
                )
            ]
        ).unwrap();
        let transform = Render2dAffineTransform::new(
            2.0, 0.0, 0.0, 3.0, 5.0, -2.0
        ).unwrap();
        let item = Render2dItem::new(
            crate::composition_2d::Render2dPrimitive::Image(image.clone()),
            transform, vec![], Render2dOpacity::TRANSPARENT
        );
        let support = neutral_support(&item, &image, 0, 1.0, 4096).unwrap().unwrap();
        assert_eq!(support.bounds, [5.0, -2.0, 7.0, 4.0]);
        assert_eq!(support.triangles.len(), 6);
    }
}
