//! Original-F1 continuous caster references across the physical compositor.
use super::*;
use runen_render::composition_2d::{
    Render2dBrush, Render2dDropShadow, Render2dRect, Render2dShape,
};
use runen_render::execution_2d::Render2dUnsupportedContent;

fn context() -> Option<GpuContext> {
    context_with_compute(false)
}

fn context_with_compute(compute: bool) -> Option<GpuContext> {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    if compute {
        requirements
            .insert(runen_gpu::GpuCapabilityRequirement::Required(
                runen_gpu::GpuCapabilityFeature::Compute,
            ))
            .unwrap();
    }
    let mut descriptor = GpuContextDescriptor::new(requirements)
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
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Filterable)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination)
        .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::Sampled)
        .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopyDestination)
        .with_label("F3F continuous original caster physical reference");
    if compute {
        descriptor = descriptor
            .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::ColorAttachment);
    }
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

fn numerical_ring(
    effects: Vec<Render2dDropShadow>,
    image: bool,
) -> (Render2dComposition, Render2dResourceBindings) {
    use runen_render::composition_2d::{
        Render2dFillRule, Render2dImagePatch, Render2dImagePrimitive, Render2dImageResource,
        Render2dImageSourceRect, Render2dPath, Render2dPathCommand, Render2dPixelExtent,
    };
    let id = Render2dResourceId::new(2901).unwrap();
    let extent = Render2dPixelExtent::new(1, 1).unwrap();
    let primitive = if image {
        let patches: Vec<_> = [
            [10.0, 10.0, 20.0, 8.0],
            [10.0, 22.0, 20.0, 8.0],
            [10.0, 18.0, 8.0, 4.0],
            [22.0, 18.0, 8.0, 4.0],
        ]
        .into_iter()
        .enumerate()
        .map(|(index, [x, y, w, h])| {
            Render2dImagePatch::new(
                Render2dImageSourceRect::new(0.0, 0.0, if index == 0 { 0.0 } else { 1.0 }, 1.0)
                    .unwrap(),
                Render2dRect::new(x, y, w, h).unwrap(),
            )
        })
        .collect();
        Render2dPrimitive::Image(Render2dImagePrimitive::new(id, extent, patches).unwrap())
    } else {
        let mut commands = Vec::new();
        for [x0, y0, x1, y1] in [[10.0, 10.0, 30.0, 30.0], [18.0, 18.0, 22.0, 22.0]] {
            let p = |x, y| Render2dPoint::new(x, y).unwrap();
            commands.extend([
                Render2dPathCommand::MoveTo(p(x0, y0)),
                Render2dPathCommand::LineTo(p(x1, y0)),
                Render2dPathCommand::LineTo(p(x1, y1)),
                Render2dPathCommand::LineTo(p(x0, y1)),
                Render2dPathCommand::Close,
            ]);
        }
        Render2dPrimitive::Fill {
            shape: Render2dShape::path(
                Render2dPath::new(Render2dFillRule::EvenOdd, commands).unwrap(),
            ),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        }
    };
    let child = Render2dGroup::new(
        [Render2dEntry::item(Render2dItem::new(
            primitive,
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        ))],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![
            Render2dDropShadow::new(8.0, 0.0, 0.03125, -1.0, Render2dColorRgba8::TRANSPARENT)
                .unwrap(),
        ],
    );
    let wrapper = Render2dGroup::new(
        [Render2dEntry::group(child)],
        Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        Vec::new(),
    );
    let composition = Render2dComposition::new([Render2dEntry::group(Render2dGroup::new(
        [Render2dEntry::group(wrapper)],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        effects,
    ))])
    .unwrap();
    let bindings = if image {
        Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
            id,
            Render2dResourceValue::ImageRgba8Srgb(
                Render2dImageResource::new(extent, vec![0; 4]).unwrap(),
            ),
        )])
        .unwrap()
    } else {
        Render2dResourceBindings::default()
    };
    (composition, bindings)
}

#[test]
fn ordinary_gaussian_consumes_original_nested_path_and_image_support_on_vulkan() {
    let Some(context) = context_with_compute(true) else {
        return;
    };
    for image in [false, true] {
        let effect =
            Render2dDropShadow::new(0.0, 0.0, 0.1, 0.0, Render2dColorRgba8::new(255, 0, 0, 255))
                .unwrap();
        let (composition, bindings) = numerical_ring(vec![effect], image);
        let mut executor = Render2dExecutor::new();
        let first = execute(
            &context,
            &mut executor,
            &composition,
            &bindings,
            "ordinary original nested Gaussian",
        );
        // Independent 16-phase Gaussian/circle-segment references, shared with
        // the complete-plane numerical qualification. Only final ordinary
        // contribution color is read back here; no query readback is an input.
        for ([x, y], expected) in [
            ([34, 17], [193_u8, 0, 0, 136]),
            ([23, 18], [184, 0, 0, 122]),
            ([25, 12], [255, 0, 0, 255]),
            ([47, 40], [0, 0, 0, 0]),
        ] {
            let actual = pixel(&first, x, y);
            assert!(
                actual[0].abs_diff(expected[0]) <= 2 && actual[1..] == expected[1..],
                "ordinary original Gaussian image={image}, {x},{y}: {actual:?}"
            );
        }
        executor.discard_cache();
        let recreated = execute(
            &context,
            &mut executor,
            &composition,
            &bindings,
            "recreated ordinary original Gaussian",
        );
        assert_eq!(first.as_bytes(), recreated.as_bytes());
    }
}

#[test]
fn ordinary_gaussian_shared_field_preserves_authored_color_order_on_vulkan() {
    let Some(context) = context_with_compute(true) else {
        return;
    };
    let red = Render2dDropShadow::new(0.0, 0.0, 0.1, 0.0, Render2dColorRgba8::new(255, 0, 0, 128))
        .unwrap();
    let blue = Render2dDropShadow::new(0.0, 0.0, 0.1, 0.0, Render2dColorRgba8::new(0, 0, 255, 128))
        .unwrap();
    for (effects, expected) in [
        (vec![red, blue], [137_u8, 0, 188, 192]),
        (vec![blue, red], [188, 0, 137, 192]),
    ] {
        let (composition, bindings) = numerical_ring(effects, false);
        let bytes = execute(
            &context,
            &mut Render2dExecutor::new(),
            &composition,
            &bindings,
            "ordered shared Gaussian field",
        );
        let actual = pixel(&bytes, 25, 12);
        assert!(
            actual[0].abs_diff(expected[0]) <= 2
                && actual[1] == 0
                && actual[2].abs_diff(expected[2]) <= 2
                && actual[3] == expected[3],
            "same field, authored color order: {actual:?} vs {expected:?}"
        );
        assert_eq!(pixel(&bytes, 47, 40), [0, 0, 0, 0]);
    }
}

#[test]
fn ordinary_gaussian_reuses_one_field_across_output_tile_seams_on_vulkan() {
    let Some(context) = context_with_compute(true) else {
        return;
    };
    let effect =
        Render2dDropShadow::new(0.0, 0.0, 0.1, 0.0, Render2dColorRgba8::new(255, 0, 0, 255))
            .unwrap();
    let (source, bindings) = numerical_ring(vec![effect], false);
    let reference = execute(
        &context,
        &mut Render2dExecutor::new(),
        &source,
        &bindings,
        "ordinary Gaussian single-tile reference",
    );
    let Render2dEntry::Group(root) = &source.root_entries()[0] else {
        unreachable!()
    };
    let translated = Render2dComposition::new([Render2dEntry::group(Render2dGroup::new(
        root.entries().to_vec(),
        Render2dAffineTransform::translation(230.0, 0.0).unwrap(),
        root.clips().to_vec(),
        root.opacity(),
        root.shadows().to_vec(),
    ))])
    .unwrap();
    let tiled = execute_sized(
        &context,
        &mut Render2dExecutor::new(),
        &translated,
        &bindings,
        "ordinary Gaussian shared field across tile boundary",
        512,
        64,
    );
    // Integer translation preserves every global 4x4 sample phase. Compare
    // the complete result across x=256, including zero cutoff/background,
    // rather than selecting only a convenient interior pixel.
    for y in 0..64_usize {
        for x in 0..512_usize {
            let actual = &tiled.as_bytes()[(y * 512 + x) * 4..][..4];
            if (230..294).contains(&x) {
                assert_eq!(
                    actual,
                    &reference.as_bytes()[(y * 64 + x - 230) * 4..][..4],
                    "shared field tile phase at {x},{y}"
                );
            } else {
                assert_eq!(actual, &[0; 4], "outside translated field at {x},{y}");
            }
        }
    }
}

#[test]
fn ordinary_gaussian_missing_compute_rejects_before_work_and_executor_reuses_on_vulkan() {
    let Some(context) = context() else { return };
    let effect =
        Render2dDropShadow::new(0.0, 0.0, 0.1, 0.0, Render2dColorRgba8::new(255, 0, 0, 255))
            .unwrap();
    let (composition, bindings) = numerical_ring(vec![effect], false);
    let (_, target) = target("unadmitted Gaussian compute");
    let mut executor = Render2dExecutor::new();
    let error = executor
        .prepare(&context, &composition, &bindings, &target)
        .unwrap_err();
    assert!(
        matches!(
            &error,
            Render2dExecutionError::UnsupportedContent(Render2dUnsupportedContent::Shadows {
                root_index: 0
            })
        ),
        "Gaussian capability failure: {error:?}"
    );
    // A rejection produces no contribution/work token. The same executor and
    // context still realize the existing non-compute shadow family correctly.
    let valid = Render2dComposition::new([Render2dEntry::group(Render2dGroup::new(
        [rectangle(10.0, 10.0, 8.0, 8.0)],
        Render2dAffineTransform::IDENTITY,
        vec![],
        Render2dOpacity::OPAQUE,
        vec![
            Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 255))
                .unwrap(),
        ],
    ))])
    .unwrap();
    let bytes = execute(
        &context,
        &mut executor,
        &valid,
        &Render2dResourceBindings::default(),
        "executor reuse after Gaussian capability rejection",
    );
    assert_eq!(pixel(&bytes, 12, 12), [0, 0, 255, 255]);
    assert_eq!(pixel(&bytes, 30, 30), [0; 4]);
}

#[test]
fn ordinary_nested_gaussian_keeps_kernel_in_consuming_parent_frame_on_vulkan() {
    let Some(context) = context_with_compute(true) else {
        return;
    };
    let effect = Render2dDropShadow::new(
        0.0,
        0.0,
        0.125,
        0.0,
        Render2dColorRgba8::new(255, 0, 0, 255),
    )
    .unwrap();
    let (fixture, bindings) = numerical_ring(vec![effect], false);
    let Render2dEntry::Group(root) = &fixture.root_entries()[0] else {
        unreachable!()
    };
    let Render2dEntry::Group(wrapper) = &root.entries()[0] else {
        unreachable!()
    };
    // Put the existing finite child directly in the consumer. The shear is
    // outside that consuming group, so it maps the completed Gaussian color.
    let consumer = Render2dGroup::new(
        wrapper.entries().to_vec(),
        Render2dAffineTransform::IDENTITY,
        vec![],
        Render2dOpacity::OPAQUE,
        root.shadows().to_vec(),
    );
    // This larger requested window exhausts the fixed invocation CPU limit
    // (20,493 of21,252 samples prepared at the measured failure). Preserve
    // the truthful rejection; the cropped positives below do not claim it
    // fits. No prepared contribution or work token escapes this failure.
    let larger = Render2dComposition::new([Render2dEntry::group(Render2dGroup::new(
        [Render2dEntry::group(consumer.clone())],
        Render2dAffineTransform::new(2.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap(),
        vec![],
        Render2dOpacity::OPAQUE,
        vec![],
    ))])
    .unwrap();
    let (_, larger_target) = target_sized("nested Gaussian bounded rejection", 96, 64);
    let mut executor = Render2dExecutor::new();
    let error = executor
        .prepare(&context, &larger, &bindings, &larger_target)
        .unwrap_err();
    assert!(matches!(error,Render2dExecutionError::SampleSpace {
        kind: runen_render::execution_2d::Render2dSampleSpaceError::ResourceLimit,
        path:Some(path),..
    } if path == [0,0]));
    // At parent y=16 the displaced finite child ends exactly at x=37+3/32.
    // The sixteen root phases in pixel(41,16) have normalized coverage
    // F(j/2-2i). Independent110-digit series integration gives mean alpha
    // .22039569311145207981876287429553705616 and sRGB code129.23576519.
    // Alpha code56.20090174 is .299 away from a rounding boundary, exceeding
    // the <.064-code combined numerical/half-float/resolve error bound.
    // The scale/reflection cases use the same independent series at phases
    // 7.25-4i+j and -6.5+2i-j/2. The compressed scale is positive coverage;
    // the expanded scale's fixed-budget rejection remains explicit above.
    // both rounding margins also exceed the physical error bound above.
    for (raw, point, expected, inside, outside) in [
        (
            [1.0, 0.0, 0.25, 1.0, 0.0, 0.0],
            41,
            [129_u8, 0, 0, 56],
            39,
            43,
        ),
        (
            [0.5, 0.0, 0.25, 1.0, 0.0, 0.0],
            22,
            [214, 0, 0, 171],
            20,
            24,
        ),
        ([-1.0, 0.0, 0.25, 1.0, 80.0, 0.0], 46, [52, 0, 0, 9], 49, 45),
    ] {
        eprintln!("ordinary nested Gaussian chart {raw:?}");
        let ancestor = Render2dGroup::new(
            [Render2dEntry::group(consumer.clone())],
            Render2dAffineTransform::new(raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]).unwrap(),
            vec![],
            Render2dOpacity::OPAQUE,
            vec![],
        );
        let composition = Render2dComposition::new([Render2dEntry::group(ancestor)]).unwrap();
        let bytes = execute_sized(
            &context,
            &mut executor,
            &composition,
            &bindings,
            "nested Gaussian original parent chart",
            64,
            64,
        );
        let actual = pixel(&bytes, point, 16);
        assert!(
            actual[0].abs_diff(expected[0]) <= 2,
            "parent-frame Gaussian {raw:?}: {actual:?}"
        );
        assert_eq!(&actual[1..], &expected[1..]);
        assert_eq!(pixel(&bytes, inside, 16), [255, 0, 0, 255]);
        assert_eq!(pixel(&bytes, outside, 16), [0; 4]);
    }
    // Consumer clipping happens AFTER its blur, then each of the two group
    // opacities applies once. At pixel40,16 precisely two of16 correlated
    // samples pass x<=36 in the parent frame and have full shadow coverage:
    // alpha=(2/16)*.5*.5=1/32. Applying the clip before blur changes this.
    let clipped = Render2dGroup::new(
        consumer.entries().to_vec(),
        consumer.local_to_parent(),
        vec![runen_render::composition_2d::Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(0.0, 0.0, 36.0, 64.0).unwrap()),
            Render2dAffineTransform::IDENTITY,
        )],
        Render2dOpacity::new(0.5).unwrap(),
        consumer.shadows().to_vec(),
    );
    let composition = Render2dComposition::new([Render2dEntry::group(Render2dGroup::new(
        [Render2dEntry::group(clipped)],
        Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap(),
        vec![],
        Render2dOpacity::new(0.5).unwrap(),
        vec![],
    ))])
    .unwrap();
    let bytes = execute(
        &context,
        &mut executor,
        &composition,
        &bindings,
        "nested Gaussian clip then once-only opacity",
    );
    for (x, red, alpha) in [(38, 137_u8, 64_u8), (40, 49, 8)] {
        let actual = pixel(&bytes, x, 16);
        assert!(
            actual[0].abs_diff(red) <= 2,
            "clipped nested Gaussian {x}: {actual:?}"
        );
        assert_eq!(&actual[1..], &[0, 0, alpha]);
    }
    assert_eq!(pixel(&bytes, 42, 16), [0; 4]);
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
fn original_glyph_survives_completely_eroded_child_effect_and_casts_ancestor_on_vulkan() {
    let Some(context) = context() else { return };
    let id = Render2dResourceId::new(2802).unwrap();
    let bindings = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 8.0));
    let child = Render2dEntry::group(Render2dGroup::new(
        vec![shaped_entry(
            id,
            [8.0, 20.0],
            Render2dColorRgba8::TRANSPARENT,
        )],
        Render2dAffineTransform::translation(2.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![
            Render2dDropShadow::new(
                10.0,
                0.0,
                0.1,
                -8.0,
                Render2dColorRgba8::new(255, 0, 0, 255),
            )
            .unwrap(),
        ],
    ));
    let composition = Render2dComposition::new(vec![Render2dEntry::group(Render2dGroup::new(
        vec![child],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![
            Render2dDropShadow::new(2.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 255))
                .unwrap(),
        ],
    ))])
    .unwrap();
    let mut executor = Render2dExecutor::new();
    let first = execute(
        &context,
        &mut executor,
        &composition,
        &bindings,
        "original glyph after exact complete child erosion",
    );
    // The retained fixture outline has width <8 and height <8. A radius
    // 8 disk cannot fit, so the child's entire translated/blurred effect
    // is empty. Its transparent original glyph remains ancestor support.
    // The +10 offset also defeats the erosion-subset absorption shortcut.
    // Original text casts at x=[12,18.5], y=[13.625,20]. A false uneroded
    // child effect would add blue at x=25; erasing pristine C loses x=14.
    assert_eq!(pixel(&first, 14, 16), [0, 0, 255, 255]);
    assert_eq!(pixel(&first, 5, 16), [0, 0, 0, 0]);
    assert_eq!(pixel(&first, 25, 16), [0, 0, 0, 0]);
    executor.discard_cache();
    let recreated = execute(
        &context,
        &mut executor,
        &composition,
        &bindings,
        "original glyph erosion after field cache loss",
    );
    assert_eq!(first.as_bytes(), recreated.as_bytes());
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
