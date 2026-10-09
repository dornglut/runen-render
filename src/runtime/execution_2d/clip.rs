//! Renderer-private exact-lattice conjunctive clips: structural tessellation is
//! disposable, and each physical pixel packs its sixteen binary coverage samples.
use super::{lowering::AdmittedTarget, scene, vector};
use crate::composition_2d::{
    Render2dAffineTransform, Render2dBrush, Render2dClip, Render2dColorRgba8, Render2dItem,
    Render2dOpacity, Render2dPrimitive,
};
use crate::execution_2d::{Render2dClipError, Render2dExecutionError, Render2dVectorError};

const AXIS_SAMPLES: u32 = 4;
const MAX_MASK_BYTES: u64 = 64 * 1024 * 1024;
pub(super) const MAX_TOTAL_MASK_BYTES: u64 = 128 * 1024 * 1024;
const MAX_TRIANGLE_SAMPLES: u64 = 1_000_000_000;
const MAX_RETAINED_TRIANGLE_VERTICES: usize = 1_048_576;

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

fn charge_geometry(
    used: &mut usize,
    vertices: usize,
    root_index: usize,
) -> Result<(), Render2dExecutionError> {
    let next = used
        .checked_add(vertices)
        .ok_or_else(|| failure(root_index, Render2dClipError::ResourceLimit))?;
    if next > MAX_RETAINED_TRIANGLE_VERTICES {
        return Err(failure(root_index, Render2dClipError::ResourceLimit));
    }
    *used = next;
    Ok(())
}

fn check_mask_budget(
    bytes: u64,
    previously_reserved: u64,
    root_index: usize,
) -> Result<(), Render2dExecutionError> {
    let aggregate = previously_reserved
        .checked_add(bytes)
        .ok_or_else(|| failure(root_index, Render2dClipError::ResourceLimit))?;
    if bytes > MAX_MASK_BYTES || aggregate > MAX_TOTAL_MASK_BYTES {
        return Err(failure(root_index, Render2dClipError::ResourceLimit));
    }
    Ok(())
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
    let next_work = work
        .checked_add(area_samples)
        .ok_or_else(|| failure(root_index, Render2dClipError::ResourceLimit))?;
    if next_work > MAX_TRIANGLE_SAMPLES {
        return Err(failure(root_index, Render2dClipError::ResourceLimit));
    }
    *work = next_work;
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

/// Rasterizes one owner's conjunctive clips over already-admitted physical
/// content bounds. The owner's parent-to-root affine applies to each clip,
/// but the owner's local-to-parent affine deliberately does not.
pub(super) fn prepare_bounded(
    clips: &[Render2dClip],
    mut bounds: [u32; 4],
    parent_to_root: scene::Affine,
    root_index: usize,
    target: &AdmittedTarget,
    previously_reserved_mask_bytes: u64,
    contribution_work: &mut u64,
) -> Result<Option<ClipMask>, Render2dExecutionError> {
    if clips.is_empty() {
        return Ok(None);
    }
    if bounds[0] >= bounds[2] || bounds[1] >= bounds[3] {
        return Ok(None);
    }
    // Retaining every authored clip mesh must remain bounded independently of mask size.
    let mut geometries = Vec::new();
    let mut retained_vertices = 0usize;
    for clip in clips {
        let primitive = Render2dPrimitive::Fill {
            shape: clip.shape().clone(),
            brush: Render2dBrush::Solid(Render2dColorRgba8::WHITE),
        };
        let clip_to_root = parent_to_root
            .compose(
                scene::Affine::from_source(clip.clip_to_parent()),
                &[root_index],
            )
            .map_err(|_| failure(root_index, Render2dClipError::PrecisionLimit))?;
        let [m11, m12, m21, m22, tx, ty] = clip_to_root.coefficients();
        let transform = Render2dAffineTransform::new(m11, m12, m21, m22, tx, ty)
            .expect("composed F3E clip affine was checked finite");
        let support = Render2dItem::new(primitive, transform, vec![], Render2dOpacity::OPAQUE);
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
        charge_geometry(&mut retained_vertices, mesh.triangles.len(), root_index)?;
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
    // Check the cumulative upload budget before reserving CPU mask storage.
    check_mask_budget(bytes, previously_reserved_mask_bytes, root_index)?;
    let pixels = usize::try_from(bytes / 4)
        .map_err(|_| failure(root_index, Render2dClipError::ResourceLimit))?;
    let width = usize::try_from(width_u32).expect("mask dimensions bounded");
    let mut intersection = vec![u16::MAX; pixels];
    let mut temporary = vec![0u16; pixels];
    // Charge each triangle against the complete prepared contribution, not the root item.
    for mesh in geometries {
        temporary.fill(0);
        for triangle in mesh.triangles.as_chunks::<3>().0 {
            raster_triangle(
                [triangle[0], triangle[1], triangle[2]],
                bounds,
                width,
                &mut temporary,
                contribution_work,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_mask_budget_is_checked_before_allocating_the_next_mask() {
        assert!(check_mask_budget(MAX_MASK_BYTES, MAX_MASK_BYTES, 1).is_ok());
        assert!(matches!(
            check_mask_budget(MAX_MASK_BYTES, MAX_MASK_BYTES + 1, 2),
            Err(Render2dExecutionError::Clip {
                root_index: 2,
                kind: Render2dClipError::ResourceLimit,
            })
        ));
        assert!(matches!(
            check_mask_budget(MAX_MASK_BYTES + 1, 0, 3),
            Err(Render2dExecutionError::Clip {
                root_index: 3,
                kind: Render2dClipError::ResourceLimit,
            })
        ));
    }

    #[test]
    fn retained_clip_geometry_is_bounded_before_another_mesh_is_kept() {
        let mut retained_vertices = MAX_RETAINED_TRIANGLE_VERTICES - 3;
        charge_geometry(&mut retained_vertices, 3, 1)
            .expect("final geometry fits the per-item budget");
        assert_eq!(retained_vertices, MAX_RETAINED_TRIANGLE_VERTICES);
        assert!(matches!(
            charge_geometry(&mut retained_vertices, 1, 2),
            Err(Render2dExecutionError::Clip {
                root_index: 2,
                kind: Render2dClipError::ResourceLimit,
            })
        ));
        assert_eq!(retained_vertices, MAX_RETAINED_TRIANGLE_VERTICES);
    }

    #[test]
    fn raster_work_budget_is_cumulative_and_rejects_before_mutating_coverage() {
        let triangle = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mut bits = [0u16];
        let mut contribution_work = MAX_TRIANGLE_SAMPLES - 16;

        raster_triangle(
            triangle,
            [0, 0, 1, 1],
            1,
            &mut bits,
            &mut contribution_work,
            7,
        )
        .expect("final permitted triangle must fit the contribution budget");
        assert_eq!(contribution_work, MAX_TRIANGLE_SAMPLES);
        assert_ne!(bits[0], 0);
        let accepted_bits = bits;

        assert!(matches!(
            raster_triangle(
                triangle,
                [0, 0, 1, 1],
                1,
                &mut bits,
                &mut contribution_work,
                8,
            ),
            Err(Render2dExecutionError::Clip {
                root_index: 8,
                kind: Render2dClipError::ResourceLimit,
            })
        ));
        assert_eq!(contribution_work, MAX_TRIANGLE_SAMPLES);
        assert_eq!(bits, accepted_bits);
    }
}
