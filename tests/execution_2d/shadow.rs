//! Independent F3F GPU pixel-reference proofs. Expected colors are derived
//! from F1's authored straight-alpha sRGB and linear-premul source-over law,
//! NOT by reading an implementation-owned mask, intermediate or shader.
use super::*;
use runen_render::composition_2d::{
    Render2dBrush, Render2dDropShadow, Render2dRect, Render2dShape,
};

fn context() -> Option<GpuContext> {
    let mut descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("F3F independent GPU shadow sample reference");
    for (format, role) in [
        (
            GpuTextureFormat::Rgba8UnormSrgb,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Sampled),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Filterable),
        (
            GpuTextureFormat::Rgba8UnormSrgb,
            GpuFormatRole::CopyDestination,
        ),
        (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment),
        (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled),
        (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Filterable),
        (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination),
        (
            GpuTextureFormat::Rgba16Float,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Sampled),
        // Distinct R32Float mask source is disposable F3F GPU work; neither
        // field cache nor semantic support depends on this physical format.
        (GpuTextureFormat::R32Float, GpuFormatRole::Sampled),
        (GpuTextureFormat::R32Float, GpuFormatRole::CopyDestination),
    ] {
        descriptor = descriptor.require_format_role(format, role);
    }
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "F3F actual shadow readback requires software Vulkan"
            );
            None
        }
        Err(error) => panic!("F3F R32Float shadow role admission: {error}"),
    }
}

fn caster(rgba: Render2dColorRgba8, x: f64, y: f64, width: f64, height: f64) -> Render2dEntry {
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(
                Render2dRect::new(x, y, width, height).expect("finite positive rectangle"),
            ),
            brush: Render2dBrush::solid(rgba),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    ))
}

fn effect(
    dx: f64,
    dy: f64,
    sigma: f64,
    spread: f64,
    color: Render2dColorRgba8,
) -> Render2dDropShadow {
    Render2dDropShadow::new(dx, dy, sigma, spread, color).unwrap()
}

fn group(
    content: Vec<Render2dEntry>,
    effects: Vec<Render2dDropShadow>,
    opacity: f64,
) -> Render2dComposition {
    Render2dComposition::new(vec![Render2dEntry::group(Render2dGroup::new(
        content,
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::new(opacity).unwrap(),
        effects,
    ))])
    .unwrap()
}

fn pixel_near(actual: [u8; 4], expected: [u8; 4]) {
    assert!(
        actual
            .into_iter()
            .zip(expected)
            .all(|(a, b)| a.abs_diff(b) <= 4),
        "independent linear-premul shadow oracle: {actual:?}, expected {expected:?}"
    );
}

#[test]
fn transparent_geometry_casts_actual_gpu_shadow_offscreen_and_at_fractional_offset() {
    let Some(context) = context() else { return };
    let composition = group(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            -20.0,
            10.0,
            10.0,
            8.0,
        )],
        vec![effect(
            25.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F offscreen fully transparent geometry GPU",
    );
    pixel_near(pixel(&image, 10, 13), [255, 0, 0, 255]);
    pixel_near(pixel(&image, 2, 13), [0, 0, 0, 0]);
    pixel_near(pixel(&image, 16, 13), [0, 0, 0, 0]);
}

#[test]
fn two_translucent_shadows_have_authored_order_before_child_and_group_opacity() {
    let Some(context) = context() else { return };
    let red = effect(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(255, 0, 0, 128));
    let blue = effect(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 128));
    let src = vec![caster(
        Render2dColorRgba8::TRANSPARENT,
        10.0,
        10.0,
        16.0,
        16.0,
    )];
    let first = execute(
        &context,
        &mut Render2dExecutor::new(),
        &group(src.clone(), vec![red, blue], 1.0),
        &Render2dResourceBindings::default(),
        "F3F red then blue",
    );
    let reverse = execute(
        &context,
        &mut Render2dExecutor::new(),
        &group(src, vec![blue, red], 1.0),
        &Render2dResourceBindings::default(),
        "F3F blue then red",
    );
    // At one fully covered sample:
    // red->blue premul R=0.25, B=0.5, A=0.75.
    // blue->red premul R=0.5, B=0.25, A=0.75.
    pixel_near(pixel(&first, 15, 15), [137, 0, 188, 191]);
    pixel_near(pixel(&reverse, 15, 15), [188, 0, 137, 191]);
}

#[test]
fn translucent_child_is_painted_above_two_shadows_before_once_only_group_opacity() {
    let Some(context) = context() else { return };
    let red = effect(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(255, 0, 0, 128));
    let blue = effect(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 128));
    let child = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(Render2dRect::new(10.0, 10.0, 16.0, 16.0).unwrap()),
            brush: Render2dBrush::solid(Render2dColorRgba8::new(0, 255, 0, 128)),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
    ));
    let binding = Render2dResourceBindings::default();
    let opaque = execute(
        &context,
        &mut Render2dExecutor::new(),
        &group(vec![child.clone()], vec![red, blue], 1.0),
        &binding,
        "F3F child above shadows",
    );
    let half = execute(
        &context,
        &mut Render2dExecutor::new(),
        &group(vec![child], vec![red, blue], 0.5),
        &binding,
        "F3F once-only group opacity",
    );
    // Independent source-over: first R=.25/B=.5/A=.75, then green .5:
    // R=.125, G=.5, B=.25, A=.875 in premultiplied linear space.
    pixel_near(pixel(&opaque, 16, 16), [99, 188, 137, 224]);
    // Group opacity multiplies the completed premultiplied result ONCE.
    pixel_near(pixel(&half, 16, 16), [71, 137, 99, 112]);
}

#[test]
fn group_clips_intersect_shadows_after_child_source_and_opacity_preserves_zero() {
    let Some(context) = context() else { return };
    use runen_render::composition_2d::Render2dClip;
    let clips = vec![
        Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(12.0, 10.0, 4.0, 16.0).unwrap()),
            Render2dAffineTransform::IDENTITY,
        ),
        Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(14.0, 10.0, 4.0, 16.0).unwrap()),
            Render2dAffineTransform::IDENTITY,
        ),
    ];
    let tree = Render2dComposition::new(vec![Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            16.0,
            16.0,
        )],
        Render2dAffineTransform::IDENTITY,
        clips,
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    ))])
    .unwrap();
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &tree,
        &Render2dResourceBindings::default(),
        "F3F conjunctive group shadow clips",
    );
    pixel_near(pixel(&image, 14, 15), [255, 0, 0, 255]);
    assert_eq!(pixel(&image, 13, 15), [0, 0, 0, 0]);
    assert_eq!(pixel(&image, 16, 15), [0, 0, 0, 0]);
}

#[test]
fn shadows_preserve_nontransparent_caller_pixels_and_exact_completion_evidence() {
    let Some(context) = context() else {
        return;
    };
    let tree = group(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            16.0,
            16.0,
        )],
        vec![effect(
            0.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 128),
        )],
        1.0,
    );
    let (texture, target) = target("F3F caller-owned prior opaque blue");
    let image = execute_inline(
        &context,
        &mut Render2dExecutor::new(),
        &tree,
        &Render2dResourceBindings::default(),
        &texture,
        &target,
        Some([0.0, 0.0, 1.0, 1.0]),
    );
    // A red 0.5 source-over the opaque caller-owned blue target in linear light:
    // red .5, blue .5, alpha 1. This path also witnesses complete membership
    // and terminal completion of every authored work node via execute_inline.
    pixel_near(pixel(&image, 15, 15), [188, 0, 187, 255]);
    assert_eq!(pixel(&image, 5, 5), [0, 0, 255, 255]);
}

#[test]
fn nonuniform_group_transform_applies_before_parent_frame_euclidean_spread() {
    let Some(context) = context() else {
        return;
    };
    let child = caster(Render2dColorRgba8::TRANSPARENT, 4.0, 10.0, 4.0, 8.0);
    let group = Render2dGroup::new(
        vec![child],
        Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.0,
            1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    );
    let tree = Render2dComposition::new(vec![Render2dEntry::group(group)]).unwrap();
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &tree,
        &Render2dResourceBindings::default(),
        "F3F parent-frame positive spread after scale",
    );
    // Child geometry [4,8] is mapped to [8,16] BEFORE spread(+1).
    // Parent-frame shadow reaches [7,17], NOT [6,18], which would
    // result from spreading in the child frame and scaling afterwards.
    pixel_near(pixel(&image, 7, 14), [255, 0, 0, 255]);
    assert_eq!(pixel(&image, 6, 14), [0, 0, 0, 0]);
    assert_eq!(pixel(&image, 17, 14), [0, 0, 0, 0]);
}

#[test]
fn transparent_nested_child_group_keeps_neutral_support_for_outer_shadow() {
    let Some(context) = context() else {
        return;
    };
    let nested = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            16.0,
            16.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        Vec::new(),
    ));
    let tree = group(
        vec![nested],
        vec![effect(
            0.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &tree,
        &Render2dResourceBindings::default(),
        "F3F zero-opacity nested child retained as parent caster",
    );
    pixel_near(pixel(&output, 18, 18), [255, 0, 0, 255]);
    assert_eq!(pixel(&output, 8, 18), [0, 0, 0, 0]);
}

#[test]
fn fractional_parent_offset_retains_correlated_half_pixel_phase() {
    let Some(context) = context() else {
        return;
    };
    let tree = group(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            8.0,
            8.0,
        )],
        vec![effect(
            0.5,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &tree,
        &Render2dResourceBindings::default(),
        "F3F fractional parent frame translation",
    );
    // The x=10 pixel has two of four correlated x sub-samples covered,
    // each of its four y sub-samples covered. Alpha is exactly 8/16;
    // premultiplied linear red 0.5 encodes to sRGB 188.
    pixel_near(pixel(&output, 10, 14), [188, 0, 0, 128]);
    assert_eq!(pixel(&output, 9, 14), [0, 0, 0, 0]);
}

#[test]
fn signed_euclidean_disk_spread_has_gpu_erosion_and_non_square_dilation() {
    let Some(context) = context() else {
        return;
    };
    let source = vec![caster(
        Render2dColorRgba8::TRANSPARENT,
        10.0,
        10.0,
        16.0,
        16.0,
    )];
    let eroded = group(
        source.clone(),
        vec![effect(
            0.0,
            0.0,
            0.0,
            -2.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let eroded_image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &eroded,
        &Render2dResourceBindings::default(),
        "F3F signed negative disk GPU",
    );
    pixel_near(pixel(&eroded_image, 18, 18), [255, 0, 0, 255]);
    assert_eq!(pixel(&eroded_image, 11, 18), [0, 0, 0, 0]);
    let dilated = group(
        source,
        vec![effect(
            0.0,
            0.0,
            0.0,
            2.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let dilated_image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &dilated,
        &Render2dResourceBindings::default(),
        "F3F positive Euclidean disk GPU",
    );
    pixel_near(pixel(&dilated_image, 26, 18), [255, 0, 0, 255]);
    // Independently integrate the Euclidean disk at corner (26,26)
    // over pixel [27,28]^2: area int_1^sqrt(3) (sqrt(4-x*x)-1) dx
    // = 0.31514674362772044. Premultiplied red encodes to sRGB 152;
    // linear alpha to 80. Earlier binary 4x center hits gave 6/16
    // (sRGB 165, alpha 96), a visible non-continuous approximation.
    pixel_near(pixel(&dilated_image, 27, 27), [152, 0, 0, 80]);
}

#[test]
fn sheared_parent_frame_erosion_uses_real_union_not_aabb_on_actual_gpu() {
    let Some(context) = context() else { return };
    // A transparent local [8,16]^2 caster is sheared and translated into
    // the group's immediate-parent frame: (x,y) -> (x+.5*y+4, y).
    // Its actual parallelogram is 8 <= x-.5*y-4 <= 16, 8 <= y <= 16.
    // Erosion by .5 uses signed distance to the true parallelogram edges;
    // the enclosing AABB [16,28] x [8,16] is an extent, not geometry.
    let transformed = Render2dGroup::new(
        vec![caster(Render2dColorRgba8::TRANSPARENT, 8.0, 8.0, 8.0, 8.0)],
        Render2dAffineTransform::new(1.0, 0.0, 0.5, 1.0, 4.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.0,
            -0.5,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    );
    let composition = Render2dComposition::new(vec![Render2dEntry::group(transformed)]).unwrap();
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F sheared Euclidean erosion cannot become AABB erosion",
    );
    // Every point in pixel [22,23] x [12,13] is >.5 from all four
    // actual parallelogram edges; the full correlated pixel is red.
    pixel_near(pixel(&output, 22, 12), [255, 0, 0, 255]);
    // Pixel [17,18] x [12,13] lies strictly OUTSIDE the actual
    // parallelogram (x-.5*y-4 < 8 throughout), yet an AABB inset
    // would wrongly paint it. This is not a sampled-color comparison.
    assert_eq!(pixel(&output, 17, 12), [0, 0, 0, 0]);
}

#[test]
fn continuous_negative_euclidean_erosion_uses_actual_subpixel_area() {
    let Some(context) = context() else { return };
    // C=[10,11]^2, r=-.2, sigma=0. Its exact eroded support is
    // [10.2,10.8]^2, of area .36 in logical pixel [10,11]^2.
    // Independent linear-premul resolve: alpha round(255*.36)=92,
    // red sRGB round(255*(1.055*.36^(1/2.4)-.055))=162.
    // Center-grid negative distance would incorrectly keep all 16 samples.
    let composition = group(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            1.0,
            1.0,
        )],
        vec![effect(
            0.0,
            0.0,
            0.0,
            -0.2,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F continuous negative disk erosion GPU subcell area",
    );
    pixel_near(pixel(&image, 10, 10), [162, 0, 0, 92]);
    assert_eq!(pixel(&image, 9, 10), [0, 0, 0, 0]);
}

#[test]
fn continuous_negative_erosion_gaussian_preserves_normalized_inner_samples() {
    let Some(context) = context() else { return };
    // Erosion of [10,11]^2 by r=.2 is exactly [10.2,10.8]^2.
    // With sigma=.025 and the independent finite 3σ cutoff=.075,
    // correlated samples at x/y=.375,.625 have entirely covered kernel
    // footprints; samples at .125,.875 have zero area intersection.
    // Exactly four of sixteen sample contributions survive: alpha=.25,
    // linear-premul red .25 => sRGB 137 and unorm alpha 64.
    let composition = group(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            1.0,
            1.0,
        )],
        vec![effect(
            0.0,
            0.0,
            0.025,
            -0.2,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F continuous negative erosion finite Gaussian normalization",
    );
    pixel_near(pixel(&output, 10, 10), [137, 0, 0, 64]);
    assert_eq!(pixel(&output, 9, 10), [0, 0, 0, 0]);
}

#[test]
fn zero_alpha_image_bytes_and_item_opacity_still_cast_from_resolved_destination() {
    let Some(context) = context() else {
        return;
    };
    use runen_render::composition_2d::{
        Render2dImagePatch, Render2dImagePrimitive, Render2dImageResource, Render2dImageSourceRect,
        Render2dPixelExtent,
    };
    let id = Render2dResourceId::new(915).unwrap();
    let extent = Render2dPixelExtent::new(1, 1).unwrap();
    let patch = Render2dImagePatch::new(
        Render2dImageSourceRect::new(0.0, 0.0, 1.0, 1.0).unwrap(),
        Render2dRect::new(10.0, 10.0, 16.0, 16.0).unwrap(),
    );
    let image = Render2dImagePrimitive::new(id, extent, vec![patch]).unwrap();
    let item = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Image(image),
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    ));
    let bindings = Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
        id,
        Render2dResourceValue::ImageRgba8Srgb(
            Render2dImageResource::new(extent, vec![0, 0, 0, 0]).unwrap(),
        ),
    )])
    .unwrap();
    let composition = group(
        vec![item],
        vec![effect(
            0.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(0, 0, 255, 255),
        )],
        1.0,
    );
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &bindings,
        "F3F RGBA-zero and item-alpha-zero image destination",
    );
    pixel_near(pixel(&output, 16, 16), [0, 0, 255, 255]);
    assert_eq!(pixel(&output, 6, 16), [0, 0, 0, 0]);
}

#[test]
fn excess_authored_shadows_fail_before_caller_work_and_executor_remains_reusable() {
    let Some(context) = context() else {
        return;
    };
    let (texture, target) = target("F3F bounded shadow admission caller target");
    let mut executor = Render2dExecutor::new();
    let caster = vec![caster(
        Render2dColorRgba8::TRANSPARENT,
        10.0,
        10.0,
        16.0,
        16.0,
    )];
    let red = effect(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(255, 0, 0, 128));
    let bindings = Render2dResourceBindings::default();
    let inadmissible = group(caster.clone(), vec![red; 257], 1.0);
    let failure = match executor.prepare(&context, &inadmissible, &bindings, &target) {
        Ok(_) => panic!("257 shadow effects must not prepare GPU work"),
        Err(error) => error,
    };
    assert!(matches!(
        failure,
        Render2dExecutionError::SampleSpace {
            kind: runen_render::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            path: Some(path),
            ..
        } if path == [0]
    ));

    // The failed invocation cannot publish a partial GPU contribution or
    // poison subsequent executor identity; caller owns the clear, and the
    // next valid contribution blends into that exact prior-blue target.
    let valid = group(caster, vec![red], 1.0);
    let image = execute_inline(
        &context,
        &mut executor,
        &valid,
        &bindings,
        &texture,
        &target,
        Some([0.0, 0.0, 1.0, 1.0]),
    );
    pixel_near(pixel(&image, 16, 16), [188, 0, 187, 255]);
    assert_eq!(pixel(&image, 7, 16), [0, 0, 255, 255]);
}

#[test]
fn positive_destination_with_zero_area_image_source_crop_still_casts_exact_shadow() {
    let Some(context) = context() else {
        return;
    };
    use runen_render::composition_2d::{
        Render2dImagePatch, Render2dImagePrimitive, Render2dImageResource, Render2dImageSourceRect,
        Render2dPixelExtent,
    };
    let id = Render2dResourceId::new(916).unwrap();
    let extent = Render2dPixelExtent::new(1, 1).unwrap();
    // No source texel can be sampled. F1 nevertheless retains the 16x16
    // destination rectangle as geometric caster; derived image paint vanishes.
    let patch = Render2dImagePatch::new(
        Render2dImageSourceRect::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        Render2dRect::new(10.0, 10.0, 16.0, 16.0).unwrap(),
    );
    let item = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Image(Render2dImagePrimitive::new(id, extent, vec![patch]).unwrap()),
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
    ));
    let bindings = Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
        id,
        Render2dResourceValue::ImageRgba8Srgb(
            Render2dImageResource::new(extent, vec![255, 255, 255, 255]).unwrap(),
        ),
    )])
    .unwrap();
    let tree = group(
        vec![item],
        vec![effect(
            0.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let rendered = execute(
        &context,
        &mut Render2dExecutor::new(),
        &tree,
        &bindings,
        "F3F zero-area image source positive destination",
    );
    pixel_near(pixel(&rendered, 18, 18), [255, 0, 0, 255]);
    assert_eq!(pixel(&rendered, 7, 18), [0, 0, 0, 0]);
}

#[test]
fn immutable_monochrome_text_casts_from_transparent_foreground_after_cache_loss() {
    let Some(context) = context() else {
        return;
    };
    let id = Render2dResourceId::new(911).unwrap();
    let bindings = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let composition = group(
        vec![shaped_entry(
            id,
            [8.0, 32.0],
            Render2dColorRgba8::TRANSPARENT,
        )],
        vec![effect(
            0.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let mut executor = Render2dExecutor::new();
    let first = execute(
        &context,
        &mut executor,
        &composition,
        &bindings,
        "F3F neutral outlined glyph",
    );
    // The published fixture's opaque box contains this physical pixel. The
    // foreground here has zero alpha, so only immutable outline C can cast.
    pixel_near(pixel(&first, 20, 20), [255, 0, 0, 255]);
    executor.discard_cache();
    let reconstructed = execute(
        &context,
        &mut executor,
        &composition,
        &bindings,
        "F3F outlined glyph after derived cache loss",
    );
    assert_eq!(first.as_bytes(), reconstructed.as_bytes());
}

#[test]
fn narrow_continuous_gaussian_uses_neutral_geometry_not_cell_average() {
    let Some(context) = context() else {
        return;
    };
    let scene = group(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            0.03,
            0.03,
            0.20,
            0.20,
        )],
        vec![effect(
            0.0,
            0.0,
            0.025,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let result = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F-GAUSS-01 exact normalized cutoff on a subsample caster",
    );
    // Independent continuous kernel oracle: only the first of sixteen 4x4
    // sample centers intersects the 3sigma support. Its full
    // [.05,.20]^2 Gaussian footprint lies inside [.03,.23]^2, giving
    // EXACT alpha=1 at that sample, and alpha=0 at the other fifteen.
    // Physical output premul alpha=1/16; linear red=1/16 converts to
    // ~71 in sRGB, straight output alpha ~16.
    pixel_near(pixel(&result, 0, 0), [71, 0, 0, 16]);
    assert_eq!(pixel(&result, 2, 0), [0, 0, 0, 0]);
}

#[test]
fn positive_euclidean_spread_convolves_continuous_geometry_on_actual_gpu() {
    let Some(context) = context() else { return };
    // The center of correlated subpixel (.125,.125) is inside the Gaussian
    // footprint of a 0.1-wide neutral caster after disk dilation +0.015.
    // Independent normalized finite-Gaussian integral across that footprint
    // is (Phi(3)-Phi(-1.6))/(Phi(3)-Phi(-3)) = 0.94640591323.
    // Only one of four x sample columns contributes, all four y rows do.
    // The expected premultiplied physical pixel alpha is 0.94640591323/4,
    // yielding alpha=60 and linear-red->sRGB=134 at 8-bit resolve.
    // The old cell-area/discrete-kernel path gave alpha about 26 instead.
    let composition = group(
        vec![caster(Render2dColorRgba8::TRANSPARENT, 0.1, 0.0, 0.1, 1.0)],
        vec![effect(
            0.0,
            0.0,
            0.025,
            0.015,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F continuous positive spread Gaussian external readback",
    );
    pixel_near(pixel(&image, 0, 0), [134, 0, 0, 60]);
    assert_eq!(pixel(&image, 2, 0), [0, 0, 0, 0]);
}

#[test]
fn finite_gaussian_shadow_retains_continuous_coverage_across_two_gpu_tiles() {
    let Some(context) = context() else { return };
    let composition = group(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            250.0,
            10.0,
            12.0,
            24.0,
        )],
        vec![effect(
            0.0,
            0.0,
            1.0,
            0.0,
            Render2dColorRgba8::new(255, 255, 255, 255),
        )],
        1.0,
    );
    let image = execute_sized(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F finite 3sigma tile seam",
        512,
        64,
    );
    // At least three physical pixels inside both rectangle boundaries:
    // the finite 3sigma Gaussian is exactly normalized here and has no
    // contribution from missing exterior geometry.
    for x in 254..259 {
        pixel_near(pixel_sized(&image, 512, x, 20), [255, 255, 255, 255]);
    }
    // Exact structural finite support is [247,265], irrespective of the
    // disposable raster convolution grid's ±half-sample integration cells.
    for x in [245, 246, 266, 267] {
        assert_eq!(pixel_sized(&image, 512, x, 20), [0, 0, 0, 0]);
    }
}
