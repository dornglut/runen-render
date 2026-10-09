//! Renderer-private continuous neutral geometry graph for F3F group effects.
//!
//! Only F1's borrowed painter events and immutable font/image bindings own
//! meaning. This module materializes bounded group-parent geometry *before*
//! ancestor affines, non-semantic paint alpha and target culling. Every
//! owner clip is a geometric intersection, not a source color or an AABB.

use super::{support::NeutralMesh, vector};
use crate::composition_2d::{
    Render2dAffineTransform, Render2dBrush, Render2dClip, Render2dColorRgba8, Render2dItem,
    Render2dOpacity, Render2dPrimitive,
};
use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};

const MAX_NEUTRAL_VERTICES: usize = 1_048_576;
const MAX_NEUTRAL_CLIP_WORK: usize = 16_777_216;

fn failed(
    path: &[usize],
    kind: Render2dSampleSpaceError,
    detail: &'static str,
) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    }
}

fn precision(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    failed(path, Render2dSampleSpaceError::PrecisionLimit, detail)
}

fn resource(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    failed(path, Render2dSampleSpaceError::ResourceLimit, detail)
}

/// A literal empty mesh never introduces a fake shadow caster.
pub(super) fn empty_mesh() -> NeutralMesh {
    NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: Vec::new(),
        bounds: [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
    }
}

/// Finite transform in one owning immediate-parent frame. The source's
/// disposable tessellation scale cancels *before* applying this affine.
pub(super) fn append(
    output: &mut NeutralMesh,
    source: &NeutralMesh,
    source_to_parent: Render2dAffineTransform,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    if source.triangles.is_empty() {
        return Ok(());
    }
    if source.triangles.len() % 3 != 0
        || !source.units_per_parent_logical_unit.is_finite()
        || source.units_per_parent_logical_unit <= 0.0
    {
        return Err(precision(path, "invalid neutral mesh source or scale"));
    }
    let required = output
        .triangles
        .len()
        .checked_add(source.triangles.len())
        .ok_or_else(|| resource(path, "neutral geometry vertex count overflow"))?;
    if required > MAX_NEUTRAL_VERTICES {
        return Err(resource(
            path,
            "neutral geometry exceeds bounded vertex admission",
        ));
    }
    let [a, b, c, d, tx, ty] = source_to_parent.components();
    let scale = source.units_per_parent_logical_unit;
    let mut prepared = Vec::new();
    prepared
        .try_reserve_exact(source.triangles.len())
        .map_err(|_| resource(path, "neutral geometry allocation failed"))?;
    for &[x, y] in &source.triangles {
        let x = x / scale;
        let y = y / scale;
        let transformed = [
            a.mul_add(x, c.mul_add(y, tx)),
            b.mul_add(x, d.mul_add(y, ty)),
        ];
        if !transformed.iter().all(|v| v.is_finite()) || transformed.iter().any(|v| v.abs() > 1.0e9)
        {
            return Err(precision(
                path,
                "neutral parent-space transform lost precision",
            ));
        }
        prepared.push(transformed);
    }
    // Two-dimensional rank-deficient transforms have zero area and cannot
    // create a false positive-area source for finite ordinary shadows.
    output
        .triangles
        .try_reserve(prepared.len())
        .map_err(|_| resource(path, "neutral destination geometry allocation failed"))?;
    for triangle in prepared.chunks_exact(3) {
        if orient(triangle[0], triangle[1], triangle[2]) == 0.0 {
            continue;
        }
        for &point in triangle {
            output.bounds[0] = output.bounds[0].min(point[0]);
            output.bounds[1] = output.bounds[1].min(point[1]);
            output.bounds[2] = output.bounds[2].max(point[0]);
            output.bounds[3] = output.bounds[3].max(point[1]);
            output.triangles.push(point);
        }
    }
    Ok(())
}

fn orient(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]).mul_add(p[1] - a[1], -(b[1] - a[1]) * (p[0] - a[0]))
}

/// The intersection of two *convex triangles*, without a rectangular/alpha
/// approximation. Clipping against a polygon-union clip means retaining each
/// triangle intersection's union (overlaps are resolved by area coverage).
fn triangle_intersection(
    source: [[f64; 2]; 3],
    clip: [[f64; 2]; 3],
    path: &[usize],
) -> Result<Vec<[f64; 2]>, Render2dExecutionError> {
    let winding = orient(clip[0], clip[1], clip[2]);
    if winding == 0.0 || orient(source[0], source[1], source[2]) == 0.0 {
        return Ok(Vec::new());
    }
    let mut polygon = source.to_vec();
    for i in 0..3 {
        let a = clip[i];
        let b = clip[(i + 1) % 3];
        let mut result = Vec::new();
        result
            .try_reserve_exact(9)
            .map_err(|_| resource(path, "neutral polygon intersection allocation failed"))?;
        for j in 0..polygon.len() {
            let p = polygon[j];
            let q = polygon[(j + 1) % polygon.len()];
            let d0 = orient(a, b, p) * winding.signum();
            let d1 = orient(a, b, q) * winding.signum();
            let inside_p = d0 >= 0.0;
            let inside_q = d1 >= 0.0;
            if inside_p != inside_q {
                let denominator = d0 - d1;
                if denominator == 0.0 {
                    return Err(precision(
                        path,
                        "neutral clip edge intersection is singular",
                    ));
                }
                let t = d0 / denominator;
                let intersection = [
                    (q[0] - p[0]).mul_add(t, p[0]),
                    (q[1] - p[1]).mul_add(t, p[1]),
                ];
                if !intersection.iter().all(|x| x.is_finite()) {
                    return Err(precision(path, "neutral clip intersection lost precision"));
                }
                result.push(intersection);
            }
            if inside_q {
                result.push(q);
            }
        }
        polygon = result;
        if polygon.len() < 3 {
            break;
        }
    }
    Ok(polygon)
}

/// Applies every structural clip once in its owner's parent logical frame.
/// Each clip mesh is a union of fill-rule tessellation triangles; successive
/// clips intersect conjunctively, independent of all source paint alpha.
pub(super) fn intersect_clips(
    source: &NeutralMesh,
    clips: &[Render2dClip],
    root_index: usize,
    path: &[usize],
    resolution: f64,
    max_buffer_bytes: u64,
) -> Result<NeutralMesh, Render2dExecutionError> {
    if clips.is_empty() || source.triangles.is_empty() {
        let mut result = empty_mesh();
        append(&mut result, source, Render2dAffineTransform::IDENTITY, path)?;
        return Ok(result);
    }
    let mut current = empty_mesh();
    append(
        &mut current,
        source,
        Render2dAffineTransform::IDENTITY,
        path,
    )?;
    let mut work = 0_usize;
    for clip in clips {
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: clip.shape().clone(),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            clip.clip_to_parent(),
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let Some(clip_mesh) =
            vector::neutral_support(&item, root_index, resolution, max_buffer_bytes)?
        else {
            return Ok(empty_mesh());
        };
        let mut clip_local = empty_mesh();
        append(
            &mut clip_local,
            &clip_mesh,
            Render2dAffineTransform::IDENTITY,
            path,
        )?;
        let mut next = empty_mesh();
        let cost = (current.triangles.len() / 3)
            .checked_mul(clip_local.triangles.len() / 3)
            .ok_or_else(|| resource(path, "neutral conjunctive clip work overflow"))?;
        work = work
            .checked_add(cost)
            .ok_or_else(|| resource(path, "neutral clip accumulation overflow"))?;
        if work > MAX_NEUTRAL_CLIP_WORK {
            return Err(resource(
                path,
                "neutral clip triangle intersection work exceeds admission",
            ));
        }
        for source in current.triangles.chunks_exact(3) {
            for clipping in clip_local.triangles.chunks_exact(3) {
                let polygon = triangle_intersection(
                    [source[0], source[1], source[2]],
                    [clipping[0], clipping[1], clipping[2]],
                    path,
                )?;
                if polygon.len() < 3 {
                    continue;
                }
                for index in 1..polygon.len() - 1 {
                    let triangle = [polygon[0], polygon[index], polygon[index + 1]];
                    if orient(triangle[0], triangle[1], triangle[2]) == 0.0 {
                        continue;
                    }
                    append(
                        &mut next,
                        &NeutralMesh {
                            units_per_parent_logical_unit: 1.0,
                            bounds: [0.0; 4],
                            triangles: triangle.to_vec(),
                        },
                        Render2dAffineTransform::IDENTITY,
                        path,
                    )?;
                }
            }
        }
        current = next;
        if current.triangles.is_empty() {
            break;
        }
    }
    Ok(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::{Render2dRect, Render2dShape};

    #[test]
    fn neutral_geometry_transforms_before_ancestor_effects_and_keeps_disjoint_sources() {
        let source = NeutralMesh {
            units_per_parent_logical_unit: 4.0,
            triangles: vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]],
            bounds: [0.0, 0.0, 4.0, 4.0],
        };
        let mut union = empty_mesh();
        let parent = Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, -3.0, 5.0).unwrap();
        append(&mut union, &source, parent, &[1, 3]).unwrap();
        assert_eq!(union.triangles, vec![[-3.0, 5.0], [-1.0, 5.0], [-3.0, 6.0]]);
        append(&mut union, &source, parent, &[1, 4]).unwrap();
        assert_eq!(union.triangles.len(), 6);
    }

    #[test]
    fn convex_polygon_intersection_obeys_actual_geometry_not_bbox() {
        let clip = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let disjoint_corner = [[1.6, 1.6], [2.2, 1.6], [1.6, 2.2]];
        assert!(
            triangle_intersection(disjoint_corner, clip, &[1])
                .unwrap()
                .len()
                < 3
        );
        let crossing = [[0.8, 0.8], [1.6, 0.8], [0.8, 1.6]];
        let clipped = triangle_intersection(crossing, clip, &[1]).unwrap();
        assert!(clipped.len() >= 3);
        for point in clipped {
            assert!(point[0] >= 0.0 && point[1] >= 0.0);
            assert!(point[0] + point[1] <= 2.0 + 1.0e-12);
        }
    }

    #[test]
    fn transparent_parent_space_shape_and_clip_retain_only_intersection() {
        let support = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 3.0, 3.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let mesh = vector::neutral_support(&support, 1, 4.0, 1_048_576)
            .unwrap()
            .unwrap();
        let clip = Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(2.0, 1.0, 2.0, 2.0).unwrap()),
            Render2dAffineTransform::IDENTITY,
        );
        let clipped = intersect_clips(&mesh, &[clip], 1, &[1], 4.0, 1_048_576).unwrap();
        let coverage = super::super::support::prepare_untranslated_shadow_coverage(
            &clipped,
            0.0,
            0.0,
            4.0,
            &[1],
        )
        .unwrap()
        .unwrap();
        let total = coverage.values.iter().sum::<f64>();
        assert!((total - 32.0).abs() < 1.0e-7);
    }
}
