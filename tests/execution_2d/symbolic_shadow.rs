//! Original-F1 continuous caster references across the physical compositor.
use super::*;
use runen_render::composition_2d::{
    Render2dBrush, Render2dDropShadow, Render2dRect, Render2dShape,
};

fn context() -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .require_format_role(
                GpuTextureFormat::Rgba8UnormSrgb,
                GpuFormatRole::ColorAttachment,
            )
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable)
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource)
            .require_format_role(
                GpuTextureFormat::Rgba16Float,
                GpuFormatRole::ColorAttachment,
            )
            .require_format_role(GpuTextureFormat::Rgba16Float, GpuFormatRole::Blendable)
            .require_format_role(GpuTextureFormat::Rgba16Float, GpuFormatRole::Sampled)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled)
            .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::Sampled)
            .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopyDestination)
            .with_label("F3F continuous original caster physical reference");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1")
            );
            None
        }
        Err(error) => panic!("continuous caster Vulkan admission: {error}"),
    }
}

fn rectangle(x: f64, y: f64, width: f64, height: f64) -> Render2dEntry {
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(Render2dRect::new(x, y, width, height).unwrap()),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    ))
}

fn completed_ring(effects: Vec<Render2dDropShadow>) -> Render2dComposition {
    let child = Render2dEntry::group(Render2dGroup::new(
        vec![
            rectangle(10.0, 10.0, 20.0, 8.0),
            rectangle(10.0, 22.0, 20.0, 8.0),
            rectangle(10.0, 18.0, 8.0, 4.0),
            rectangle(22.0, 18.0, 8.0, 4.0),
        ],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![
            Render2dDropShadow::new(8.0, 0.0, 0.0, -1.0, Render2dColorRgba8::TRANSPARENT).unwrap(),
        ],
    ));
    Render2dComposition::new(vec![Render2dEntry::group(Render2dGroup::new(
        vec![child],
        Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        effects,
    ))])
    .unwrap()
}

#[test]
fn completed_holed_caster_integrates_true_curved_erosion_before_ancestor_shear() {
    let Some(context) = context() else { return };
    let bytes = execute(
        &context,
        &mut Render2dExecutor::new(),
        &completed_ring(vec![
            Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 255))
                .unwrap(),
        ]),
        &Render2dResourceBindings::default(),
        "original continuous completed ring",
    );
    // At this pixel C is absent. E is disk erosion of the original four-bar
    // union, offset +8, then sheared x += y/4. Independent circle-segment
    // integration yields area .5868378429496339. Opaque blue premultiplied
    // by that area encodes to [0,0,201,150] in the final sRGB target.
    for (actual, expected) in pixel(&bytes, 36, 22).into_iter().zip([0_u8, 0, 201, 150]) {
        assert!(
            actual.abs_diff(expected) <= 4,
            "curved pixel channel {actual} vs {expected}"
        );
    }
    assert_eq!(pixel(&bytes, 20, 12), [0, 0, 255, 255]);
    // Pristine C fills the translated E's hole here; completed neutral
    // support is their union even though the entire child paints no alpha.
    assert_eq!(pixel(&bytes, 33, 20), [0, 0, 255, 255]);
    assert_eq!(pixel(&bytes, 0, 0), [0, 0, 0, 0]);
    assert_eq!(pixel(&bytes, 48, 20), [0, 0, 0, 0]);
}

#[test]
fn four_translucent_ancestor_shadows_keep_continuous_cell_correlation() {
    let Some(context) = context() else { return };
    let effect =
        Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 64))
            .unwrap();
    let bytes = execute(
        &context,
        &mut Render2dExecutor::new(),
        &completed_ring(vec![effect; 4]),
        &Render2dResourceBindings::default(),
        "four correlated continuous caster shadows",
    );
    // Apply four source-overs separately at each independently integrated
    // sample: mean(1-(1-(64/255)*alpha_i)^4)=.4137411872262517.
    // Preaveraging geometry gives .47129181416838695 and [0,0,183,120].
    for (actual, expected) in pixel(&bytes, 36, 22).into_iter().zip([0_u8, 0, 172, 106]) {
        assert!(
            actual.abs_diff(expected) <= 4,
            "correlated channel {actual} vs {expected}"
        );
    }
}
