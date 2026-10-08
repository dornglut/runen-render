//! Private bounded image mapping into independent ordered painter patches.
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
        || image.patches().len() > 65_536
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
