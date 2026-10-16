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
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Sampled)
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Filterable)
            .require_format_role(
                GpuTextureFormat::Rgba8UnormSrgb,
                GpuFormatRole::CopyDestination,
            )
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

fn completed_ring(effects: Vec<Render2dDropShadow>, child_sigma: f64) -> Render2dComposition {
    completed_ring_in_frame(
        effects,
        child_sigma,
        Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap(),
    )
}

fn completed_ring_in_frame(
    effects: Vec<Render2dDropShadow>,
    child_sigma: f64,
    frame: Render2dAffineTransform,
) -> Render2dComposition {
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
            Render2dDropShadow::new(8.0, 0.0, child_sigma, -1.0, Render2dColorRgba8::TRANSPARENT)
                .unwrap(),
        ],
    ));
    Render2dComposition::new(vec![Render2dEntry::group(Render2dGroup::new(
        vec![child],
        frame,
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
        &completed_ring(
            vec![
                Render2dDropShadow::new(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    Render2dColorRgba8::new(0, 0, 255, 255),
                )
                .unwrap(),
            ],
            0.0,
        ),
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
        &completed_ring(vec![effect; 4], 0.0),
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

#[test]
fn finite_child_kernel_integrates_completed_caster_before_ancestor_shear() {
    let Some(context) = context() else { return };
    let effect =
        Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 255))
            .unwrap();
    let bytes = execute(
        &context,
        &mut Render2dExecutor::new(),
        &completed_ring(vec![effect], 0.03125),
        &Render2dResourceBindings::default(),
        "finite-kernel completed caster",
    );
    // Independent circle-segment integration after the child box halo h=3/32
    // and ancestor shear gives original continuous area .73453430260630349.
    for (actual, expected) in pixel(&bytes, 36, 22).into_iter().zip([0_u8, 0, 223, 187]) {
        assert!(
            actual.abs_diff(expected) <= 4,
            "finite curved channel {actual} vs {expected}"
        );
    }
    assert_eq!(pixel(&bytes, 20, 12), [0, 0, 255, 255]);
    assert_eq!(pixel(&bytes, 33, 20), [0, 0, 255, 255]);
    assert_eq!(pixel(&bytes, 0, 0), [0, 0, 0, 0]);
    assert_eq!(pixel(&bytes, 48, 20), [0, 0, 0, 0]);
}

#[test]
fn interval_halo_completed_caster_keeps_independent_curved_boundary_on_vulkan() {
    let Some(context) = context() else { return };
    let effect =
        Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 255))
            .unwrap();
    let bytes = execute(
        &context,
        &mut Render2dExecutor::new(),
        &completed_ring(vec![effect], 0.05),
        &Render2dResourceBindings::default(),
        "interval-valued finite halo original completed caster",
    );
    // Independent exact-binary64(.05) halo h=3*sigma and 110-digit
    // circle-segment integration of all16 cells gives alpha
    // .8103862966359923233; sRGB blue232, unorm alpha207.
    for (actual, expected) in pixel(&bytes, 36, 22).into_iter().zip([0_u8, 0, 232, 207]) {
        assert!(
            actual.abs_diff(expected) <= 4,
            "interval halo {actual} vs {expected}"
        );
    }
    assert_eq!(pixel(&bytes, 20, 12), [0, 0, 255, 255]);
    assert_eq!(pixel(&bytes, 33, 20), [0, 0, 255, 255]);
    assert_eq!(pixel(&bytes, 0, 0), [0, 0, 0, 0]);
    assert_eq!(pixel(&bytes, 48, 20), [0, 0, 0, 0]);
}

#[test]
fn four_finite_child_caster_shadows_preserve_cell_correlation() {
    let Some(context) = context() else { return };
    let effect =
        Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 64))
            .unwrap();
    let bytes = execute(
        &context,
        &mut Render2dExecutor::new(),
        &completed_ring(vec![effect; 4], 0.03125),
        &Render2dResourceBindings::default(),
        "four finite-kernel correlated caster shadows",
    );
    // Independent 16-cell areas, source-over at each cell, then resolve:
    // mean(1-(1-(64/255)*alpha_i)^4)=.5168041040779434.
    for (actual, expected) in pixel(&bytes, 36, 22).into_iter().zip([0_u8, 0, 190, 132]) {
        assert!(
            actual.abs_diff(expected) <= 4,
            "finite correlated channel {actual} vs {expected}"
        );
    }
}

#[test]
fn reflected_finite_child_caster_keeps_original_quadrant_domains() {
    let Some(context) = context() else { return };
    let effect =
        Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 255))
            .unwrap();
    for (a, c, d, tx, ty, x, y) in [
        (-1.0, -0.25, 1.0, 64.0, 0.0, 27, 22),
        (1.0, 0.25, -1.0, 0.0, 64.0, 36, 41),
        (-1.0, -0.25, -1.0, 64.0, 64.0, 27, 41),
    ] {
        let frame = Render2dAffineTransform::new(a, 0.0, c, d, tx, ty).unwrap();
        let bytes = execute(
            &context,
            &mut Render2dExecutor::new(),
            &completed_ring_in_frame(vec![effect], 0.03125, frame),
            &Render2dResourceBindings::default(),
            "reflected finite-kernel completed caster",
        );
        // Exact reflections preserve area and permute the 16 physical cells.
        // Pixel intervals transform by [u,u+1] -> [63-u,64-u].
        for (actual, expected) in pixel(&bytes, x, y).into_iter().zip([0_u8, 0, 223, 187]) {
            assert!(
                actual.abs_diff(expected) <= 4,
                "reflected {a},{d} channel {actual} vs {expected}"
            );
        }
    }
}

#[test]
fn nested_image_rank_precision_keeps_original_occurrence_and_executor_reuse() {
    let Some(context) = context() else { return };
    use runen_render::composition_2d::{
        Render2dImagePatch, Render2dImagePrimitive, Render2dImageResource, Render2dImageSourceRect,
        Render2dPixelExtent,
    };
    let id = Render2dResourceId::new(2801).unwrap();
    let extent = Render2dPixelExtent::new(1, 1).unwrap();
    let primitive = Render2dPrimitive::Image(
        Render2dImagePrimitive::new(
            id,
            extent,
            vec![Render2dImagePatch::new(
                Render2dImageSourceRect::new(0.0, 0.0, 1.0, 1.0).unwrap(),
                Render2dRect::new(1.0, 1.0, 1.0, 1.0).unwrap(),
            )],
        )
        .unwrap(),
    );
    let step = 2.0_f64.powi(-27);
    let scene = |frame| {
        Render2dComposition::new(vec![Render2dEntry::group(Render2dGroup::new(
            vec![Render2dEntry::group(Render2dGroup::new(
                vec![Render2dEntry::item(Render2dItem::new(
                    primitive.clone(),
                    frame,
                    Vec::new(),
                    Render2dOpacity::TRANSPARENT,
                ))],
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
                Vec::new(),
            ))],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
            vec![
                Render2dDropShadow::new(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    Render2dColorRgba8::new(0, 0, 255, 255),
                )
                .unwrap(),
            ],
        ))])
        .unwrap()
    };
    let bindings = Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
        id,
        Render2dResourceValue::ImageRgba8Srgb(
            Render2dImageResource::new(extent, vec![0; 4]).unwrap(),
        ),
    )])
    .unwrap();
    let (_, target) = target("nested original image rank preflight");
    let mut executor = Render2dExecutor::new();
    let uncertain =
        scene(Render2dAffineTransform::new(1.0, 1.0 - step, 1.0 + step, 1.0, 0.0, 0.0).unwrap());
    let error = match executor.prepare(&context, &uncertain, &bindings, &target) {
        Ok(_) => panic!("uncertain inverse must fail preflight"),
        Err(e) => e,
    };
    assert!(
        matches!(error, Render2dExecutionError::SampleSpace {kind:runen_render::execution_2d::Render2dSampleSpaceError::PrecisionLimit,path:Some(path),..} if path==[0,0,0])
    );
    // Identical nonexact products prove actual rank deficiency, even though
    // the rounded FMA determinant is nonzero. It remains genuine no-work.
    let singular = scene(Render2dAffineTransform::new(1.1, 1.1, 1.1, 1.1, 0.0, 0.0).unwrap());
    assert!(
        !executor
            .prepare(&context, &singular, &bindings, &target)
            .unwrap()
            .has_render_work()
    );
    let bytes = execute(
        &context,
        &mut executor,
        &scene(Render2dAffineTransform::IDENTITY),
        &bindings,
        "nested rank failure reuse",
    );
    assert_eq!(pixel(&bytes, 1, 1), [0, 0, 255, 255]);
}

#[test]
fn affine_shadow_rank_uses_original_products_before_culling() {
    let Some(context) = context() else { return };
    let scene = |frame| {
        Render2dComposition::new(vec![Render2dEntry::group(Render2dGroup::new(
            vec![Render2dEntry::group(Render2dGroup::new(
                vec![rectangle(10.0, 10.0, 4.0, 4.0)],
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::OPAQUE,
                vec![
                    Render2dDropShadow::new(
                        0.0,
                        0.0,
                        0.1,
                        1.0,
                        Render2dColorRgba8::new(0, 0, 255, 255),
                    )
                    .unwrap(),
                ],
            ))],
            frame,
            Vec::new(),
            Render2dOpacity::OPAQUE,
            Vec::new(),
        ))])
        .unwrap()
    };
    let (_, target) = target("original shadow affine rank preflight");
    let mut executor = Render2dExecutor::new();
    let step = 2.0_f64.powi(-27);
    let error = match executor.prepare(
        &context,
        &scene(Render2dAffineTransform::new(1.0, 1.0 - step, 1.0 + step, 1.0, 0.0, 0.0).unwrap()),
        &Render2dResourceBindings::default(),
        &target,
    ) {
        Ok(_) => panic!("full-rank shadow may not be culled by rounded zero"),
        Err(error) => error,
    };
    assert!(
        matches!(error,Render2dExecutionError::SampleSpace {kind:runen_render::execution_2d::Render2dSampleSpaceError::PrecisionLimit,path:Some(path),..} if path==[0,0])
    );
    let singular = scene(Render2dAffineTransform::new(1.1, 1.1, 1.1, 1.1, 0.0, 0.0).unwrap());
    assert!(
        !executor
            .prepare(
                &context,
                &singular,
                &Render2dResourceBindings::default(),
                &target
            )
            .unwrap()
            .has_render_work()
    );
}
