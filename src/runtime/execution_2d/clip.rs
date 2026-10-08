//! Renderer-private exact-lattice conjunctive clips: structural tessellation is
//! disposable, and each physical pixel packs its sixteen binary coverage samples.
use super::{
    lowering::{AdmittedTarget, OrderedItem},
    vector,
};
use crate::composition_2d::{
    Render2dBrush, Render2dColorRgba8, Render2dItem, Render2dOpacity, Render2dPrimitive,
};
use crate::execution_2d::{Render2dClipError, Render2dExecutionError, Render2dVectorError};

const AXIS_SAMPLES: u32 = 4;
const MAX_MASK_BYTES: u64 = 64 * 1024 * 1024;
const MAX_TRIANGLE_SAMPLES: u64 = 1_000_000_000;

#[derive(Debug)]
pub(super) struct ClipMask {
    pub(super) root_index: usize,
    pub(super) origin: [u32; 2],
    pub(super) extent: [u32; 2],
    pub(super) rgba: Vec<u8>,
}

pub(super) fn failure(root_index: usize, kind: Render2dClipError) -> Render2dExecutionError {
    Render2dExecutionError::Clip { root_index, kind }
}

fn vector_failure(root_index: usize, err: Render2dExecutionError) -> Render2dExecutionError {
    let kind = match err {
        Render2dExecutionError::Vector {
            kind: Render2dVectorError::PrecisionLimit,
            ..
        } => Render2dClipError::PrecisionLimit,
        Render2dExecutionError::Vector {
            kind: Render2dVectorError::ResourceLimit,
            ..
        } => Render2dClipError::ResourceLimit,
        _ => Render2dClipError::TessellationFailed,
    };
    failure(root_index, kind)
}

// Physical bounds as [left, top, right, bottom]. All source operations have
// already passed their ordinary renderer admission and typed representability.
fn content_bounds(
    ordered: &[OrderedItem],
    target: &AdmittedTarget,
    root_index: usize,
) -> Result<Option<[u32; 4]>, Render2dExecutionError> {
    let mut result: Option<[u32; 4]> = None;
    for operation in ordered {
        let candidate = match operation {
            OrderedItem::Vector(mesh) => [
                mesh.bounds[0],
                mesh.bounds[1],
                mesh.bounds[0].saturating_add(mesh.bounds[2]),
                mesh.bounds[1].saturating_add(mesh.bounds[3]),
            ],
            OrderedItem::Image(patch) => patch.bounds,
            OrderedItem::Glyph(glyph) => {
                let scale = target.raster_scale();
                let left = glyph.logical_x * scale;
                let top = glyph.logical_y * scale;
                let right = (glyph.logical_x + glyph.logical_width) * scale;
                let bottom = (glyph.logical_y + glyph.logical_height) * scale;
                let [cw, ch] = target.canvas();
                if ![left, top, right, bottom].into_iter().all(f64::is_finite) {
                    return Err(failure(root_index, Render2dClipError::PrecisionLimit));
                }
                if right <= 0.0 || bottom <= 0.0 || left >= cw || top >= ch {
                    continue;
                }
                let bounded = [
                    left.max(0.0).floor(),
                    top.max(0.0).floor(),
                    right.min(cw).ceil(),
                    bottom.min(ch).ceil(),
                ];
                if bounded[2] <= bounded[0] || bounded[3] <= bounded[1] {
                    continue;
                }
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                {
                    [
                        bounded[0] as u32,
                        bounded[1] as u32,
                        bounded[2] as u32,
                        bounded[3] as u32,
                    ]
                }
            }
        };
        result = Some(match result {
            None => candidate,
            Some([x0, y0, x1, y1]) => [
                x0.min(candidate[0]),
                y0.min(candidate[1]),
                x1.max(candidate[2]),
                y1.max(candidate[3]),
            ],
        });
    }
    Ok(result)
}

fn intersect(a: [u32; 4], b: [u32; 4]) -> Option<[u32; 4]> {
    let result = [
        a[0].max(b[0]),
        a[1].max(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ];
    (result[0] < result[2] && result[1] < result[3]).then_some(result)
}

fn orient(a: [f64; 2], b: [f64; 2], point: [f64; 2]) -> f64 {
    (b[0] - a[0]).mul_add(point[1] - a[1], -(b[1] - a[1]) * (point[0] - a[0]))
}

fn raster_triangle(
    triangle: [[f64; 2]; 3],
    bounds: [u32; 4],
    width: usize,
    bits: &mut [u16],
    work: &mut u64,
    root_index: usize,
) -> Result<(), Render2dExecutionError> {
    let [a, b, c] = triangle;
    let area = orient(a, b, c);
    if area == 0.0 {
        return Ok(());
    }
    let left = a[0].min(b[0]).min(c[0]).floor().max(f64::from(bounds[0]));
    let top = a[1].min(b[1]).min(c[1]).floor().max(f64::from(bounds[1]));
    let right = a[0].max(b[0]).max(c[0]).ceil().min(f64::from(bounds[2]));
    let bottom = a[1].max(b[1]).max(c[1]).ceil().min(f64::from(bounds[3]));
    if right <= left || bottom <= top {
        return Ok(());
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let (left, top, right, bottom) = (left as u32, top as u32, right as u32, bottom as u32);
    let area_samples = u64::from(right - left)
        .checked_mul(u64::from(bottom - top))
        .and_then(|value| value.checked_mul(u64::from(AXIS_SAMPLES * AXIS_SAMPLES)))
        .ok_or_else(|| failure(root_index, Render2dClipError::ResourceLimit))?;
    *work = work
        .checked_add(area_samples)
        .ok_or_else(|| failure(root_index, Render2dClipError::ResourceLimit))?;
    if *work > MAX_TRIANGLE_SAMPLES {
        return Err(failure(root_index, Render2dClipError::ResourceLimit));
    }
    for y in top..bottom {
        for x in left..right {
            let index = usize::try_from(y - bounds[1]).expect("bounded row") * width
                + usize::try_from(x - bounds[0]).expect("bounded column");
            for sy in 0..AXIS_SAMPLES {
                for sx in 0..AXIS_SAMPLES {
                    let point = [
                        f64::from(x) + (f64::from(sx) + 0.5) / f64::from(AXIS_SAMPLES),
                        f64::from(y) + (f64::from(sy) + 0.5) / f64::from(AXIS_SAMPLES),
                    ];
                    let e0 = orient(a, b, point);
                    let e1 = orient(b, c, point);
                    let e2 = orient(c, a, point);
                    if (area > 0.0 && e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0)
                        || (area < 0.0 && e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0)
                    {
                        bits[index] |= 1u16 << (sy * AXIS_SAMPLES + sx);
                    }
                }
            }
        }
    }
    Ok(())
}

pub(super) fn prepare(
    item: &Render2dItem,
    ordered: &[OrderedItem],
    root_index: usize,
    target: &AdmittedTarget,
) -> Result<Option<ClipMask>, Render2dExecutionError> {
    if item.clips().is_empty() {
        return Ok(None);
    }
    let Some(mut bounds) = content_bounds(ordered, target, root_index)? else {
        return Ok(None);
    };
    let mut geometries = Vec::with_capacity(item.clips().len());
    for clip in item.clips() {
        let primitive = Render2dPrimitive::Fill {
            shape: clip.shape().clone(),
            brush: Render2dBrush::Solid(Render2dColorRgba8::WHITE),
        };
        let support = Render2dItem::new(
            primitive,
            clip.clip_to_parent(),
            vec![],
            Render2dOpacity::OPAQUE,
        );
        let mesh = vector::realize(
            &support,
            root_index,
            target.raster_scale(),
            target.canvas(),
            target.max_buffer_bytes(),
        )
        .map_err(|e| vector_failure(root_index, e))?;
        let Some(mesh) = mesh else { return Ok(None) };
        let extent = [
            mesh.bounds[0],
            mesh.bounds[1],
            mesh.bounds[0].saturating_add(mesh.bounds[2]),
            mesh.bounds[1].saturating_add(mesh.bounds[3]),
        ];
        let Some(region) = intersect(bounds, extent) else {
            return Ok(None);
        };
        bounds = region;
        geometries.push(mesh);
    }
    let width_u32 = bounds[2] - bounds[0];
    let height_u32 = bounds[3] - bounds[1];
    if width_u32 > target.max_texture_dimension_2d()
        || height_u32 > target.max_texture_dimension_2d()
    {
        return Err(failure(root_index, Render2dClipError::ResourceLimit));
    }
    let bytes = u64::from(width_u32)
        .checked_mul(u64::from(height_u32))
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| failure(root_index, Render2dClipError::ResourceLimit))?;
    if bytes > MAX_MASK_BYTES {
        return Err(failure(root_index, Render2dClipError::ResourceLimit));
    }
    let pixels = usize::try_from(bytes / 4)
        .map_err(|_| failure(root_index, Render2dClipError::ResourceLimit))?;
    let width = usize::try_from(width_u32).expect("mask dimensions bounded");
    let mut intersection = vec![u16::MAX; pixels];
    let mut temporary = vec![0u16; pixels];
    let mut work = 0u64;
    for mesh in geometries {
        temporary.fill(0);
        for triangle in mesh.triangles.chunks_exact(3) {
            raster_triangle(
                [triangle[0], triangle[1], triangle[2]],
                bounds,
                width,
                &mut temporary,
                &mut work,
                root_index,
            )?;
        }
        for (value, next) in intersection.iter_mut().zip(&temporary) {
            *value &= *next;
        }
        if intersection.iter().all(|value| *value == 0) {
            return Ok(None);
        }
    }
    let mut rgba = Vec::with_capacity(
        usize::try_from(bytes)
            .map_err(|_| failure(root_index, Render2dClipError::ResourceLimit))?,
    );
    for value in intersection {
        rgba.extend_from_slice(&[
            u8::try_from(value & 0xff).expect("low clip byte"),
            u8::try_from(value >> 8).expect("high clip byte"),
            0,
            255,
        ]);
    }
    Ok(Some(ClipMask {
        root_index,
        origin: [bounds[0], bounds[1]],
        extent: [width_u32, height_u32],
        rgba,
    }))
}
