//! F3E external consumer proof: actual immutable F1 group -> public executor
//! -> RunenGPU work/evidence -> unmodified caller-owned RGBA8-sRGB readback.
//! These pixels are computed independently from straight-alpha source facts.
use super::*;
use runen_render::composition_2d::{Render2dBrush, Render2dRect, Render2dShape};

fn context() -> Option<GpuContext> {
    let mut descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("F3E source-neutral grouped solid vector consumer");
    for (format, role) in [
        (
            GpuTextureFormat::Rgba8UnormSrgb,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource),
        (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment),
        (GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled),
        (
            GpuTextureFormat::Rgba16Float,
            GpuFormatRole::ColorAttachment,
        ),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Blendable),
        (GpuTextureFormat::Rgba16Float, GpuFormatRole::Sampled),
    ] {
        descriptor = descriptor.require_format_role(format, role);
    }
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "F3E public group consumer requires a Vulkan adapter"
            );
            None
        }
        Err(error) => panic!("F3E grouped F1 admission: {error}"),
    }
}

fn solid(color: Render2dColorRgba8, x: f64, width: f64) -> Render2dEntry {
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(
                Render2dRect::new(x, 0.0, width, 64.0).expect("positive finite rect"),
            ),
            brush: Render2dBrush::solid(color),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
    ))
}

fn group(entries: Vec<Render2dEntry>, opacity: f64) -> Render2dEntry {
    Render2dEntry::group(Render2dGroup::new(
        entries,
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::new(opacity).unwrap(),
        Vec::new(),
    ))
}

fn pixel_close(actual: [u8; 4], expected: [u8; 4]) {
    assert!(
        actual
            .into_iter()
            .zip(expected)
            .all(|(a, b)| a.abs_diff(b) <= 4),
        "correlated pixel {actual:?}, expected {expected:?}"
    );
}

#[test]
fn grouped_translucent_children_obey_once_only_group_opacity_via_public_executor() {
    let Some(ctx) = context() else {
        return;
    };
    let root = Render2dComposition::new(vec![group(
        vec![
            solid(Render2dColorRgba8::new(255, 0, 0, 128), 0.0, 64.0),
            solid(Render2dColorRgba8::new(0, 0, 255, 128), 0.0, 64.0),
        ],
        0.5,
    )])
    .unwrap();
    let image = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &root,
        &Render2dResourceBindings::default(),
        "F3E public atomic opacity",
    );
    // Linear premul red=.125, blue=.25, alpha=.375 for nominal half-alpha.
    pixel_close(pixel(&image, 10, 20), [99, 0, 137, 96]);
}

#[test]
fn nested_groups_preserve_disjoint_sample_coverage_and_parent_order() {
    let Some(ctx) = context() else {
        return;
    };
    let content = vec![
        solid(Render2dColorRgba8::new(255, 0, 0, 255), 10.0, 0.5),
        solid(Render2dColorRgba8::new(0, 0, 255, 255), 10.5, 0.5),
    ];
    let root = Render2dComposition::new(vec![group(vec![group(content, 1.0)], 1.0)])
        .expect("valid nested F1 painter tree");
    let image = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &root,
        &Render2dResourceBindings::default(),
        "F3E public nested 4x4 coverage",
    );
    // Each of the 16 samples is exactly red OR blue: alpha=1, no 0.75 leak.
    pixel_close(pixel(&image, 10, 20), [188, 0, 188, 255]);
    pixel_close(pixel(&image, 11, 20), [0, 0, 0, 0]);
}
