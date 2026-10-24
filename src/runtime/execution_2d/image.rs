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

/// Exact product identity owns rank, independently of rounded determinant
/// cancellation. Full-rank numerical inversion still requires a finite value.
fn admitted_determinant(
    item: &Render2dItem,
    root_index: usize,
) -> Result<Option<f64>, Render2dExecutionError> {
    let [a, b, c, d, _, _] = item.local_to_parent().components();
    use super::shadow::region::interval::exact_products_equal;
    if exact_products_equal(a, d, b, c) == Some(true) {
        return Ok(None);
    }
    let determinant = a.mul_add(d, -(b * c));
    if determinant == 0.0 || !determinant.is_finite() {
        return Err(failure(root_index, Render2dImageError::PrecisionLimit));
    }
    Ok(Some(determinant))
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

/// Derives signed, off-canvas neutral image-patch geometry before visible-paint
/// opacity, source-texel sampling and final-target clipping.
///
/// This is a disposable private physical approximation of immutable F1
/// destination patches, not a source-authority image-alpha mask. Group effects
/// and item clips consume the patches at a later F3F lowering boundary.
pub(super) fn neutral_support(
    item: &Render2dItem,
    image: &Render2dImagePrimitive,
    root_index: usize,
    raster_scale: f64,
    max_buffer_bytes: u64,
    path: &[usize],
    work: &mut super::shadow::NeutralPreparationWork<'_>,
) -> Result<Option<NeutralMesh>, Render2dExecutionError> {
    // Rank/scale guards and loop setup are work even for empty geometry.
    work.charge(512, path)?;
    if image.patches().len() > MAX_IMAGE_PATCHES {
        return Err(failure(root_index, Render2dImageError::ResourceLimit));
    }
    let [a, b, c, d, tx, ty] = item.local_to_parent().components();
    if !raster_scale.is_finite() || raster_scale <= 0.0 {
        return Err(failure(root_index, Render2dImageError::PrecisionLimit));
    }
    if admitted_determinant(item, root_index)?.is_none() {
        return Ok(None);
    }
    // Count before allocating. This pass observes only destination extents;
    // source crop/texel alpha never changes neutral image support.
    work.charge(32 * image.patches().len(), path)?;
    let patches = image
        .patches()
        .iter()
        .filter(|patch| {
            let dest = patch.destination();
            dest.width() != 0.0 && dest.height() != 0.0
        })
        .count();
    let count = patches * 6; // MAX_IMAGE_PATCHES was checked above.
    let bytes = (count as u64)
        .checked_mul(crate::runtime::program::abi::COMPOSITION_VERTEX_STRIDE)
        .ok_or_else(|| failure(root_index, Render2dImageError::ResourceLimit))?;
    if bytes > max_buffer_bytes {
        return Err(failure(root_index, Render2dImageError::ResourceLimit));
    }
    // Per nonempty patch: four corners, sixteen affine FMAs and eight scale products,
    // finite/precision checks, six vertex copies and 24 extrema updates.
    // The second destination scan includes empty entries too. Prepay the
    // entire emission before reserving or entering its first patch.
    work.charge(32 * image.patches().len() + 1024 * patches, path)?;
    let mut triangles = Vec::new();
    work.reserve_vector(&mut triangles, count, path)?;
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for patch in image.patches() {
        let dest = patch.destination();
        // Even a zero-area source crop retains destination geometry under F1:
        // sampled color is absent, but an enclosing effect can use this support.
        if dest.width() == 0.0 || dest.height() == 0.0 {
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
    Ok((!triangles.is_empty()).then_some(NeutralMesh {
        units_per_parent_logical_unit: raster_scale,
        triangles,
        bounds,
    }))
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
    // Only proved rank deficiency has no 2D area to paint.
    let Some(determinant) = admitted_determinant(item, root_index)? else {
        return Ok(Vec::new());
    };
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

    // Isolated factory tests get their own invocation owner; production passes
    // the live source graph owner through the required factory argument.
    fn neutral_support(
        item: &Render2dItem,
        image: &Render2dImagePrimitive,
        root_index: usize,
        scale: f64,
        max_buffer_bytes: u64,
    ) -> Result<Option<NeutralMesh>, Render2dExecutionError> {
        super::neutral_support(
            item,
            image,
            root_index,
            scale,
            max_buffer_bytes,
            &[root_index],
            &mut super::super::shadow::NeutralPreparationWork::new(
                &mut 0,
                &mut 0,
                &mut super::super::support::PreparationHostBudget::new(),
            ),
        )
    }

    use crate::composition_2d::{
        Render2dAffineTransform, Render2dImagePatch, Render2dImageSourceRect, Render2dOpacity,
        Render2dPixelExtent, Render2dResourceId,
    };

    #[test]
    fn neutral_image_factory_admits_complete_emission_before_allocation() {
        use super::super::support::{
            MAX_COMPOSITIONAL_SHADOW_WORK, MAX_SHADOW_HOST_BYTES, PreparationHostBudget,
        };
        let extent = Render2dPixelExtent::new(1, 1).unwrap();
        let image = Render2dImagePrimitive::new(
            Render2dResourceId::new(1).unwrap(),
            extent,
            [
                Render2dRect::new(1.0, 2.0, 3.0, 4.0).unwrap(),
                Render2dRect::new(10.0, 10.0, 0.0, 4.0).unwrap(),
                Render2dRect::new(-4.0, -3.0, 2.0, 2.0).unwrap(),
            ]
            .into_iter()
            .map(|destination| {
                Render2dImagePatch::new(
                    // No sampled source color, but two nonempty destinations.
                    Render2dImageSourceRect::new(0.0, 0.0, 0.0, 0.0).unwrap(),
                    destination,
                )
            })
            .collect::<Vec<_>>(),
        )
        .unwrap();
        let item = Render2dItem::new(
            crate::composition_2d::Render2dPrimitive::Image(image.clone()),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let cost = 512 + 2 * 32 * 3 + 1024 * 2;
        let bytes = 12 * size_of::<[f64; 2]>();
        let path = [7, 2];
        let mut work = MAX_COMPOSITIONAL_SHADOW_WORK - cost;
        let mut host = PreparationHostBudget::new();
        let mesh = super::neutral_support(
            &item,
            &image,
            7,
            4.0,
            4096,
            &path,
            &mut super::super::shadow::NeutralPreparationWork::new(&mut work, &mut 0, &mut host),
        )
        .unwrap()
        .unwrap();
        assert_eq!(work, MAX_COMPOSITIONAL_SHADOW_WORK);
        assert_eq!(mesh.triangles.len(), 12);
        assert_eq!(mesh.triangles.capacity(), 12);
        assert_eq!(mesh.bounds, [-16.0, -12.0, 16.0, 24.0]);
        assert_eq!(host.live(), bytes);
        assert_eq!(host.peak(), bytes);
        drop(mesh);

        let mut work = MAX_COMPOSITIONAL_SHADOW_WORK - cost + 1;
        let initial = work;
        let mut host = PreparationHostBudget::new();
        assert!(matches!(super::neutral_support(
            &item, &image, 7, 4.0, 4096, &path,
            &mut super::super::shadow::NeutralPreparationWork::new(
                &mut work, &mut 0, &mut host,
            ),
        ), Err(Render2dExecutionError::SampleSpace {
            kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            path: Some(owner), ..
        }) if owner == path));
        assert_eq!(work, initial + 512 + 32 * 3);
        assert_eq!((host.live(), host.peak()), (0, 0));

        let mut work = 0;
        let mut host = PreparationHostBudget::new();
        let occupied = MAX_SHADOW_HOST_BYTES - bytes + 1;
        host.reserve(occupied, &path).unwrap();
        assert!(matches!(super::neutral_support(
            &item, &image, 7, 4.0, 4096, &path,
            &mut super::super::shadow::NeutralPreparationWork::new(
                &mut work, &mut 0, &mut host,
            ),
        ), Err(Render2dExecutionError::SampleSpace {
            kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            path: Some(owner), ..
        }) if owner == path));
        assert_eq!(work, cost);
        assert_eq!((host.live(), host.peak()), (occupied, occupied));
    }

    #[test]
    fn rounded_zero_image_determinant_is_precision_not_empty_support() {
        let step = f64::from_bits((1023_u64 - 27) << 52);
        // Real determinant=1-(1-step)(1+step)=2^-54. The rounded
        // intermediate product is1, so the old FMA expression returned0.
        let transform =
            Render2dAffineTransform::new(1.0, 1.0 - step, 1.0 + step, 1.0, 0.0, 0.0).unwrap();
        let [a, b, c, d, _, _] = transform.components();
        assert_eq!(a.mul_add(d, -(b * c)), 0.0);
        let extent = Render2dPixelExtent::new(1, 1).unwrap();
        let image = Render2dImagePrimitive::new(
            Render2dResourceId::new(1).unwrap(),
            extent,
            vec![Render2dImagePatch::new(
                Render2dImageSourceRect::new(0.0, 0.0, 1.0, 1.0).unwrap(),
                Render2dRect::new(0.0, 0.0, 1.0, 1.0).unwrap(),
            )],
        )
        .unwrap();
        let item = Render2dItem::new(
            crate::composition_2d::Render2dPrimitive::Image(image.clone()),
            transform,
            Vec::new(),
            Render2dOpacity::OPAQUE,
        );
        let source = Render2dImageResource::new(extent, vec![255; 4]).unwrap();
        for error in [
            neutral_support(&item, &image, 7, 1.0, 4096).unwrap_err(),
            realize(&item, &image, &source, 7, 1.0, [64.0, 64.0], 8192).unwrap_err(),
        ] {
            assert!(matches!(
                error,
                Render2dExecutionError::Image {
                    root_index: 7,
                    kind: Render2dImageError::PrecisionLimit,
                }
            ));
        }
        for coefficient in [1.0, 1.1] {
            let singular = Render2dItem::new(
                item.primitive().clone(),
                Render2dAffineTransform::new(
                    coefficient,
                    coefficient,
                    coefficient,
                    coefficient,
                    0.0,
                    0.0,
                )
                .unwrap(),
                Vec::new(),
                Render2dOpacity::OPAQUE,
            );
            assert!(
                neutral_support(&singular, &image, 7, 1.0, 4096)
                    .unwrap()
                    .is_none()
            );
            assert!(
                realize(&singular, &image, &source, 7, 1.0, [64.0, 64.0], 8192)
                    .unwrap()
                    .is_empty()
            );
        }
    }

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
        )
        .unwrap();
        let item = Render2dItem::new(
            crate::composition_2d::Render2dPrimitive::Image(image.clone()),
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let support = neutral_support(&item, &image, 0, 1.0, 4096)
            .unwrap()
            .expect("opacity and offscreen canvas cannot erase neutral support");
        assert_eq!(support.bounds, [-8.0, -3.0, -4.0, -1.0]);
        assert_eq!(support.triangles.len(), 6);
        let error = neutral_support(&item, &image, 0, 1.0, 32).unwrap_err();
        assert!(matches!(
            error,
            Render2dExecutionError::Image {
                root_index: 0,
                kind: Render2dImageError::ResourceLimit
            }
        ));
    }

    #[test]
    fn equal_image_destinations_at_different_tessellation_scales_keep_shadow_size() {
        let extent = Render2dPixelExtent::new(2, 2).unwrap();
        let image = Render2dImagePrimitive::new(
            Render2dResourceId::new(9).unwrap(),
            extent,
            vec![Render2dImagePatch::new(
                Render2dImageSourceRect::new(0.0, 0.0, 2.0, 2.0).unwrap(),
                Render2dRect::new(-3.0, 1.0, 4.0, 2.0).unwrap(),
            )],
        )
        .unwrap();
        let item = Render2dItem::new(
            crate::composition_2d::Render2dPrimitive::Image(image.clone()),
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let low = neutral_support(&item, &image, 0, 1.0, 4096)
            .unwrap()
            .unwrap();
        let high = neutral_support(&item, &image, 0, 2.0, 4096)
            .unwrap()
            .unwrap();
        let first = super::super::support::prepare_untranslated_shadow_coverage(
            &low,
            0.5,
            0.25,
            4.0,
            &[1, 2],
        )
        .unwrap()
        .unwrap();
        let second = super::super::support::prepare_untranslated_shadow_coverage(
            &high,
            0.5,
            0.25,
            4.0,
            &[1, 2],
        )
        .unwrap()
        .unwrap();
        assert_eq!(first.origin_x, second.origin_x);
        assert_eq!(first.origin_y, second.origin_y);
        assert_eq!((first.width, first.height), (second.width, second.height));
        assert_eq!(first.values, second.values);
    }

    #[test]
    fn image_neutral_support_preserves_affine_geometry_and_empty_source() {
        let extent = Render2dPixelExtent::new(4, 4).unwrap();
        let image = Render2dImagePrimitive::new(
            Render2dResourceId::new(1).unwrap(),
            extent,
            vec![
                Render2dImagePatch::new(
                    Render2dImageSourceRect::new(0.0, 0.0, 1.0, 1.0).unwrap(),
                    Render2dRect::new(0.0, 0.0, 1.0, 2.0).unwrap(),
                ),
                Render2dImagePatch::new(
                    Render2dImageSourceRect::new(0.0, 0.0, 0.0, 1.0).unwrap(),
                    Render2dRect::new(6.0, 0.0, 2.0, 2.0).unwrap(),
                ),
            ],
        )
        .unwrap();
        let transform = Render2dAffineTransform::new(2.0, 0.0, 0.0, 3.0, 5.0, -2.0).unwrap();
        let item = Render2dItem::new(
            crate::composition_2d::Render2dPrimitive::Image(image.clone()),
            transform,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let support = neutral_support(&item, &image, 0, 1.0, 4096)
            .unwrap()
            .unwrap();
        // Source rectangle #2 has width zero but a positive destination extent.
        // Both destination rectangles are neutral geometry despite zero item opacity.
        assert_eq!(support.bounds, [5.0, -2.0, 21.0, 4.0]);
        assert_eq!(support.triangles.len(), 12);
    }
}
