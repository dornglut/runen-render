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
