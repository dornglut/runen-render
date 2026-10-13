//! Independent F3F GPU pixel-reference proofs. Expected colors are derived
//! from F1's authored straight-alpha sRGB and linear-premul source-over law,
//! NOT by reading an implementation-owned mask, intermediate or shader.
use super::*;
use runen_render::composition_2d::{
    Render2dBrush, Render2dClip, Render2dDropShadow, Render2dRect, Render2dShape,
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
fn invisible_root_shadows_skip_halo_work_but_opaque_siblings_still_fail_preflight() {
    let Some(context) = context() else {
        return;
    };
    let caster = vec![caster(
        Render2dColorRgba8::TRANSPARENT,
        10.0,
        10.0,
        16.0,
        16.0,
    )];
    let mut executor = Render2dExecutor::new();
    let bindings = Render2dResourceBindings::default();
    let (_, target) = target("F3F alpha-neutral no-paint halo admission");
    let huge_transparent = group(
        caster.clone(),
        vec![effect(
            0.0,
            0.0,
            500.0,
            0.0,
            Render2dColorRgba8::TRANSPARENT,
        )],
        1.0,
    );
    assert!(
        !executor
            .prepare(&context, &huge_transparent, &bindings, &target)
            .expect("invisible shadow has no physical mask/halo to allocate")
            .has_render_work()
    );
    let huge_zero_opacity_group = group(
        caster.clone(),
        vec![effect(
            0.0,
            0.0,
            500.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        0.0,
    );
    assert!(
        !executor
            .prepare(&context, &huge_zero_opacity_group, &bindings, &target)
            .expect("zero-opacity group has no physical effect work")
            .has_render_work()
    );
    let actually_visible = group(
        caster,
        vec![effect(
            0.0,
            0.0,
            500.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let error = match executor.prepare(&context, &actually_visible, &bindings, &target) {
        Ok(_) => panic!("visible oversized 3sigma halo must remain inadmissible"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Render2dExecutionError::SampleSpace {
            kind: runen_render::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            path: Some(path),
            ..
        } if path == [0]
    ));
}

#[test]
fn invisible_nested_effect_skips_mask_while_neutral_source_remains_admitted() {
    let Some(context) = context() else {
        return;
    };
    // This is much wider than the bounded disposable 4x physical mask,
    // but source-neutral geometry is just two finite triangles.
    let nested = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            0.0,
            0.0,
            900.0,
            900.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    ));
    let tree = Render2dComposition::new(vec![Render2dEntry::group(Render2dGroup::new(
        vec![nested],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        Vec::new(),
    ))])
    .unwrap();
    let (_, target) = target("F3F hidden nested neutral source");
    let prepared = Render2dExecutor::new()
        .prepare(
            &context,
            &tree,
            &Render2dResourceBindings::default(),
            &target,
        )
        .expect("ancestor opacity culls only physical paint, not F1 support verification");
    assert!(!prepared.has_render_work());
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
fn nested_zero_kernel_shadow_maps_through_sheared_ancestor_before_paint() {
    let Some(context) = context() else {
        return;
    };
    let caster = caster(Render2dColorRgba8::TRANSPARENT, 2.0, 10.0, 4.0, 6.0);
    let nested = Render2dEntry::group(Render2dGroup::new(
        vec![caster],
        Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            5.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![nested],
        Render2dAffineTransform::new(2.0, 0.0, 0.5, 1.0, 5.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        Vec::new(),
    ));
    let scene = Render2dComposition::new(vec![outer]).unwrap();
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F nested zero-kernel shadow under shear",
    );
    // Child-local C x=[2,6] maps through its OWN 2x transform to
    // [4,12]. The shadow's immediate-parent offset +5 yields [9,17]
    // (not the incorrect local-offset-then-scale [14,22]). Only then
    // does the ancestor shear x'=2x+0.5y+5 produce [29.5,45.5]
    // at y=13. Test both ends and a deeply covered interior pixel.
    pixel_near(pixel(&output, 35, 13), [255, 0, 0, 255]);
    assert_eq!(pixel(&output, 26, 13), [0, 0, 0, 0]);
    assert_eq!(pixel(&output, 49, 13), [0, 0, 0, 0]);
}

#[test]
fn nested_finite_gaussian_integrates_parent_kernel_under_ancestor_shear() {
    let Some(context) = context() else { return };
    let nested = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.5,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![nested],
        Render2dAffineTransform::new(2.0, 0.0, 0.5, 1.0, 4.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        Vec::new(),
    ));
    let scene = Render2dComposition::new(vec![outer]).unwrap();
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F nested parent-frame Gaussian under nonuniform ancestor shear",
    );
    // Independent analytic separable truncated Gaussian over parent
    // rectangle [10,20]^2, sigma=.5 and cutoff=1.5, after inverse
    // mapping each correlated root point to
    // parent=( (root_x-.5*root_y-4)/2, root_y ).
    // The y-kernel is fully interior at y in [13,14]. Integrating x via
    // normalized erf CDF at all sixteen 4x samples gives pixel alpha:
    // (30,13)=0.405534, (31,13)=0.763495, (32,13)=0.953792.
    // Incorrect root-axis sigma on the sheared parallelogram cannot meet
    // these three independent premultiplied-red sample oracles.
    pixel_near(pixel(&output, 30, 13), [171, 0, 0, 103]);
    pixel_near(pixel(&output, 31, 13), [226, 0, 0, 195]);
    pixel_near(pixel(&output, 32, 13), [250, 0, 0, 243]);
    assert_eq!(pixel(&output, 26, 13), [0, 0, 0, 0]);
}

#[test]
fn nested_positive_disk_zero_blur_integrates_inverse_affine_root_sample_cells() {
    let Some(context) = context() else { return };
    let inner = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.0,
            1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![inner],
        Render2dAffineTransform::new(2.0, 0.0, 0.5, 1.0, 4.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        Vec::new(),
    ));
    let composition = Render2dComposition::new(vec![outer]).unwrap();
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F zero blur positive disk exact root-cell affine area",
    );
    // Parent C=[10,20]^2, Euclidean positive disk radius=1.
    // At y in [15,16], away from the four rounded corners, the
    // parent left boundary is x=9. Map AFTER dilation through
    // ancestor x'=2x+.5y+4, y'=y, giving root boundary x'=22+.5y.
    // Its exact area inside root pixel [29,30] x [15,16] is
    // integral(y=15..16) (30-(22+.5y))dy = 0.25.
    // Inverse-affine ROOT sample-cell integration must preserve it;
    // midpoint tests can lose the triangular area altogether.
    // Linear red .25 -> sRGB 137, alpha .25 -> 64.
    pixel_near(pixel(&output, 29, 15), [137, 0, 0, 64]);
    pixel_near(pixel(&output, 30, 15), [255, 0, 0, 255]);
    assert_eq!(pixel(&output, 28, 15), [0, 0, 0, 0]);
}

#[test]
fn nested_positive_zero_blur_respects_reflected_ancestor_area_and_winding() {
    let Some(context) = context() else { return };
    let inner = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.0,
            1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![inner],
        Render2dAffineTransform::new(-2.0, 0.0, 0.0, 1.0, 50.5, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        Vec::new(),
    ));
    let scene = Render2dComposition::new(vec![outer]).unwrap();
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F nested positive disk zero blur with reflected ancestor",
    );
    // Parent C=[10,20]^2; positive Euclidean disk radius=1.
    // In the y=[15,16] interior, the left parent support boundary
    // is x=9. The negative-determinant ancestor maps it to
    // root x'=-2*9+50.5=32.5. The root pixel [32,33] is therefore
    // covered by exactly half its area. A correctly oriented inverse-
    // affine parallelogram must produce alpha 0.5; clipping its signed
    // polygon area without winding normalization would incorrectly
    // lose the coverage or report a negative area.
    // Premultiplied linear red 0.5 -> sRGB 188; alpha 0.5 -> 128.
    pixel_near(pixel(&output, 32, 15), [188, 0, 0, 128]);
    pixel_near(pixel(&output, 31, 15), [255, 0, 0, 255]);
    assert_eq!(pixel(&output, 33, 15), [0, 0, 0, 0]);
}

#[test]
fn nested_positive_disk_then_finite_gaussian_maps_through_sheared_ancestor() {
    let Some(context) = context() else { return };
    let inner = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.1,
            1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![inner],
        Render2dAffineTransform::new(2.0, 0.0, 0.5, 1.0, 4.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        Vec::new(),
    ));
    let composition = Render2dComposition::new(vec![outer]).unwrap();
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F nested positive disk before parent Gaussian and ancestor shear",
    );
    // Source C=[10,20]^2. At y≈15 its exact Euclidean disk(+1)
    // boundary is parent x=9; the finite normalized sigma=.1 Gaussian
    // has 3sigma=.3 parent-axis support. For each 4x4 root pixel sample:
    // parent x=(root_x-.5*root_y-4)/2, integrate
    // G(x) on [-.3,min(.3,parent_x-9)] / integral[-.3,.3].
    // Independent Simpson integration of this F1 half-plane law over
    // 16 correlated root sample centers gives alpha:
    // root (29,15)=.2662224405, (30,15)=.9837775595.
    // Premultiplied LINEAR red converted to sRGB yields 141 and 253.
    pixel_near(pixel(&output, 29, 15), [141, 0, 0, 68]);
    pixel_near(pixel(&output, 30, 15), [253, 0, 0, 251]);
    pixel_near(pixel(&output, 31, 15), [255, 0, 0, 255]);
    assert_eq!(pixel(&output, 28, 15), [0, 0, 0, 0]);
}

/// An ancestor must consume its child's exact positive Euclidean disk, not
/// transparent paint alpha, the raw rectangle or a bounding-box substitute.
#[test]
fn nested_positive_disk_sigma_zero_consumed_by_ancestor_on_actual_vulkan() {
    let Some(context) = context() else { return };
    let inner = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![effect(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let scene = group(
        vec![inner],
        vec![effect(
            0.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    // The [9,10]x[9,10] corner intersects the radius-one disk in
    // exactly pi/4 area. Linear-premultiplied red resolves to [229,0,0,200].
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F ancestor consumes the real transparent positive disk",
    );
    pixel_near(pixel(&output, 9, 9), [229, 0, 0, 200]);
    pixel_near(pixel(&output, 15, 15), [255, 0, 0, 255]);
    assert_eq!(pixel(&output, 8, 8), [0, 0, 0, 0]);
}

#[test]
fn qualified_nested_disk_respects_ancestor_translation_on_actual_vulkan() {
    let Some(context) = context() else { return };
    let child = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![effect(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let composition = group(
        vec![child],
        vec![effect(
            3.0,
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
        "F3F certified disk precedes ancestor offset",
    );
    // The same 1-radius quarter disk has moved precisely +3 in x.
    pixel_near(pixel(&image, 12, 9), [229, 0, 0, 200]);
    pixel_near(pixel(&image, 17, 15), [255, 0, 0, 255]);
    assert_eq!(pixel(&image, 9, 9), [0, 0, 0, 0]);
}

/// Independent normalized finite 3sigma Gaussian reference. For the
/// rounded C=[10,14]^2 with Euclidean dilation r=1, the bottom-left
/// boundary at y in [9,10] is x=10-sqrt(1-(y-10)^2). Integrate each
/// retained x half-line with the truncated normal CDF at sigma=.25,
/// then integrate y over its curved/straight topology pieces. Averaging
/// the 16 globally phased 4x4 physical centers of pixel [9,9] gives
/// alpha=0.69934700764064, premultiplied sRGB8 [218,0,0,178].
/// This is a genuinely different oracle from unblurred pi/4 alpha=.7854.
#[test]
fn qualified_child_positive_disk_convolves_finite_ancestor_gaussian_on_vulkan() {
    let Some(context) = context() else { return };
    let child = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            4.0,
            4.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![effect(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let composition = group(
        vec![child],
        vec![effect(
            0.0,
            0.0,
            0.25,
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
        "F3F curved child finite ancestor Gaussian real Vulkan",
    );
    pixel_near(pixel(&image, 9, 9), [218, 0, 0, 178]);
    pixel_near(pixel(&image, 11, 11), [255, 0, 0, 255]);
    assert_eq!(pixel(&image, 7, 7), [0, 0, 0, 0]);
}

/// Independent geometric identity: (C ⊕ B1) ⊕ B1 = C ⊕ B2.
/// At the rounded bottom-left corner, pixel [8,9]² relative to C=[10,20]²
/// contains a circle/square intersection of 0.31514674362772044 area,
/// obtained analytically by integrating sqrt(4-x²)-1 over x=[1,sqrt(3)].
/// It premultiplies opaque linear red to sRGB8 [152,0,0,80].
#[test]
fn nested_positive_disks_add_radii_with_independent_corner_oracle_on_vulkan() {
    let Some(context) = context() else { return };
    let child = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![effect(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let composition = group(
        vec![child],
        vec![effect(
            0.0,
            0.0,
            0.0,
            1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F consecutive positive Euclidean disks add exact radii",
    );
    pixel_near(pixel(&image, 8, 8), [152, 0, 0, 80]);
    pixel_near(pixel(&image, 8, 10), [255, 0, 0, 255]);
    assert_eq!(pixel(&image, 7, 10), [0, 0, 0, 0]);
    assert_eq!(pixel(&image, 7, 7), [0, 0, 0, 0]);
}

/// Two different ancestor shadows MUST both sample the same completed
/// child source C ⊕ B1: the first red root spread 1 reaches radius 2,
/// while the second blue root spread 0 reaches only radius 1. The latter
/// paints above the former, but cannot inherit the earlier red shadow.
#[test]
fn two_ancestor_shadows_keep_independent_curved_caster_and_painter_order() {
    let Some(context) = context() else { return };
    let child = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![effect(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let scene = group(
        vec![child],
        vec![
            effect(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::new(255, 0, 0, 255)),
            effect(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(0, 0, 255, 255)),
        ],
        1.0,
    );
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F independent curved ancestor shadows and authored paint order",
    );
    // Outer-only red radius-2 reach cannot leak into the second blue radius-1.
    pixel_near(pixel(&image, 8, 10), [255, 0, 0, 255]);
    // Both disks cover the next cell: later blue overwrites earlier red.
    pixel_near(pixel(&image, 9, 10), [0, 0, 255, 255]);
    pixel_near(pixel(&image, 15, 15), [0, 0, 255, 255]);
    pixel_near(pixel(&image, 8, 8), [152, 0, 0, 80]);
    assert_eq!(pixel(&image, 7, 10), [0, 0, 0, 0]);
}

/// Independent four-layer physical-compositor reference for a rounded
/// *transparent* child. At root pixel [9,9], the source is the intersection
/// of each 0.25x0.25 physical cell with the Euclidean unit quarter-disk.
/// For u,v in [0,1], each cell's coverage is
/// 16 * integral( max(0, min(v1,sqrt(1-u²)) - v0) ) du.
/// The antiderivative of sqrt(1-u²) is
/// F(u) = (u*sqrt(1-u²) + asin(u))/2.
/// For four identical translucent red shadows with alpha=64/255,
/// the independent 4x4 reference is:
/// A = sum_{16 cells} [1-(1-(64/255)*coverage_cell)^4]/16
///   = 0.5530777966844838, linear premul red -> sRGB8 [196,0,0,141].
/// Crucially, applying source-over to the *resolved* pi/4 coverage instead
/// would give A=0.5844688787, and is NOT the accepted F3E 4x4 law.
/// The full-inside pixel is 1-(1-64/255)^4 = 0.6852449206461124.
#[test]
fn four_translucent_curved_ancestor_shadows_composite_before_physical_resolve() {
    let Some(context) = context() else { return };
    let child = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![effect(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let translucent = effect(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::new(255, 0, 0, 64));
    let scene = group(vec![child], vec![translucent; 4], 1.0);
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F four curved translucent shadows, subpixel source-over",
    );
    pixel_near(pixel(&image, 9, 9), [196, 0, 0, 141]);
    pixel_near(pixel(&image, 15, 15), [216, 0, 0, 175]);
    assert_eq!(pixel(&image, 8, 8), [0, 0, 0, 0]);
}

/// Exact convex morphology: (C plus B1) eroded by B0.5 equals C plus B0.5.
/// At the bottom-left pixel [9,9], only the radius-0.5 quarter disk lies
/// within the cell, so its area is pi/16=0.19634954084936207.
#[test]
fn qualified_child_positive_disk_partial_ancestor_erosion_has_gpu_quarter_disk() {
    let Some(context) = context() else { return };
    let child = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![effect(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let scene = group(
        vec![child],
        vec![effect(
            0.0,
            0.0,
            0.0,
            -0.5,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F exact erosion of certified rounded descendant disk",
    );
    pixel_near(pixel(&image, 9, 9), [123, 0, 0, 50]);
    pixel_near(pixel(&image, 12, 12), [255, 0, 0, 255]);
    assert_eq!(pixel(&image, 8, 8), [0, 0, 0, 0]);
}

/// Convex parallel-body law beyond the existing corner radius:
/// (C plus B1) eroded by B1.5 equals C eroded by B0.5.
/// For rectangular C this is the exact inset [10.5,19.5]^2, yielding
/// quarter-cell coverage 0.25 at pixel [10,10] (sRGB8 [137,0,0,64]).
#[test]
fn qualified_child_positive_disk_exhausted_radius_erodes_original_rectangle() {
    let Some(context) = context() else { return };
    let child = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![effect(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let scene = group(
        vec![child],
        vec![effect(
            0.0,
            0.0,
            0.0,
            -1.5,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F disk erosion past curved radius uses exact inset rectangle",
    );
    pixel_near(pixel(&image, 10, 10), [137, 0, 0, 64]);
    pixel_near(pixel(&image, 12, 12), [255, 0, 0, 255]);
    assert_eq!(pixel(&image, 9, 9), [0, 0, 0, 0]);
}

/// The borrowed F1 ellipse is curved; disposable tessellation is not an
/// exact signed Euclidean erosion witness. A radius-6 disk cannot fit in
/// the source's certified 8x8 original reach, so its negative spread is
/// geometrically EMPTY even when the inner group is transparent. The
/// transparent inner shadow cannot erase the pristine caster used by a
/// separately authored opaque ancestor shadow.
#[test]
fn nested_curved_f1_complete_erosion_preserves_ancestor_gpu_paint() {
    let Some(context) = context() else { return };
    let scene = |with_erasure: bool| {
        let ellipse = Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::ellipse(Render2dRect::new(10.0, 10.0, 8.0, 8.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        ));
        let nested = Render2dEntry::group(Render2dGroup::new(
            vec![ellipse],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            if with_erasure {
                vec![effect(
                    2.0,
                    -1.0,
                    0.1,
                    -6.0,
                    Render2dColorRgba8::TRANSPARENT,
                )]
            } else {
                Vec::new()
            },
        ));
        group(
            vec![nested],
            vec![effect(
                0.0,
                0.0,
                0.0,
                0.0,
                Render2dColorRgba8::new(255, 0, 0, 255),
            )],
            1.0,
        )
    };
    let plain = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene(false),
        &Render2dResourceBindings::default(),
        "F3F original ellipse without child erasure",
    );
    let actual = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene(true),
        &Render2dResourceBindings::default(),
        "F3F original ellipse with complete child erosion",
    );
    assert_eq!(
        actual.as_bytes(),
        plain.as_bytes(),
        "certified-empty curved child shadow must not alter original-F1 ancestor paint"
    );
    pixel_near(pixel(&actual, 14, 14), [255, 0, 0, 255]);
    assert_eq!(pixel(&actual, 5, 5), [0, 0, 0, 0]);
}

/// A nonconvex original source needs no polygonal signed inset when a
/// disk larger than half its rigorously enclosing extent erodes it fully.
/// Real Vulkan must render precisely the same pristine L-shaped ancestor
/// caster with or without the fully erased transparent nested shadow.
#[test]
fn nested_nonconvex_l_complete_erosion_preserves_gpu_ancestor_caster() {
    let Some(context) = context() else { return };
    let scene = |with_erasure: bool| {
        let inner = Render2dEntry::group(Render2dGroup::new(
            vec![
                caster(Render2dColorRgba8::TRANSPARENT, 10.0, 10.0, 8.0, 4.0),
                caster(Render2dColorRgba8::TRANSPARENT, 10.0, 14.0, 4.0, 4.0),
            ],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            if with_erasure {
                vec![effect(
                    1.5,
                    -0.5,
                    0.25,
                    -5.0,
                    Render2dColorRgba8::TRANSPARENT,
                )]
            } else {
                Vec::new()
            },
        ));
        group(
            vec![inner],
            vec![effect(
                0.0,
                0.0,
                0.0,
                0.0,
                Render2dColorRgba8::new(255, 0, 0, 255),
            )],
            1.0,
        )
    };
    let reference = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene(false),
        &Render2dResourceBindings::default(),
        "F3F L-shaped original caster without child erasure",
    );
    let actual = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene(true),
        &Render2dResourceBindings::default(),
        "F3F L-shaped original caster with empty signed child shadow",
    );
    assert_eq!(
        actual.as_bytes(),
        reference.as_bytes(),
        "provably empty nested erosion must preserve the original nonconvex caster"
    );
    pixel_near(pixel(&actual, 12, 12), [255, 0, 0, 255]);
    pixel_near(pixel(&actual, 12, 16), [255, 0, 0, 255]);
    assert_eq!(pixel(&actual, 16, 16), [0, 0, 0, 0]);
}

/// Independent complete-root-erasure proof: the outer negative Euclidean
/// disk erases both the pristine 8x8 C and its nested positive disk + finite
/// Gaussian halo BEFORE any attempt to polygonalize those curved regions.
/// The separately authored opaque rectangle must still paint identically.
#[test]
fn completely_eroded_curved_root_caster_skips_only_its_shadow_on_vulkan() {
    let Some(context) = context() else { return };
    let scene = |outer_shadow: Option<Render2dDropShadow>| {
        let child = Render2dEntry::group(Render2dGroup::new(
            vec![caster(
                Render2dColorRgba8::TRANSPARENT,
                10.0,
                10.0,
                8.0,
                8.0,
            )],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            vec![effect(0.0, 0.0, 0.1, 1.0, Render2dColorRgba8::TRANSPARENT)],
        ));
        let opaque = Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(12.0, 12.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::new(0, 255, 0, 255)),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
        ));
        group(vec![child, opaque], outer_shadow.into_iter().collect(), 1.0)
    };
    let reference = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene(None),
        &Render2dResourceBindings::default(),
        "F3F nested curved source without outer negative shadow",
    );
    let actual = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene(Some(effect(
            2.0,
            -1.5,
            0.25,
            -6.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        ))),
        &Render2dResourceBindings::default(),
        "F3F nested curved source with provably empty outer shadow",
    );
    assert_eq!(
        actual.as_bytes(),
        reference.as_bytes(),
        "empty outer shadow cannot mutate the original F3E painter"
    );
    pixel_near(pixel(&actual, 12, 12), [0, 255, 0, 255]);
    assert_eq!(pixel(&actual, 9, 9), [0, 0, 0, 0]);
}

#[test]
fn nested_positive_disk_is_unsupported_when_ancestor_needs_neutral_caster() {
    let Some(context) = context() else { return };
    let inner = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![effect(0.0, 0.0, 0.1, 1.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![inner],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let composition = Render2dComposition::new(vec![outer]).unwrap();
    let (_, target) = target("F3F curved caster must not be replaced with raw C");
    let error = match Render2dExecutor::new().prepare(
        &context,
        &composition,
        &Render2dResourceBindings::default(),
        &target,
    ) {
        Ok(_) => panic!("an ancestor shadow must not consume unrepresented curved C"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Render2dExecutionError::UnsupportedEntry {
            path,
            kind: runen_render::execution_2d::Render2dUnsupportedContent::Shadows {
                root_index: 0,
            },
        } if path == [0, 0]
    ));
}

#[test]
fn nested_convex_signed_disk_erosion_is_exact_before_ancestor_shear() {
    let Some(context) = context() else { return };
    let inner = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.0,
            -1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![inner],
        Render2dAffineTransform::new(2.0, 0.0, 0.5, 1.0, 4.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        Vec::new(),
    ));
    let scene = Render2dComposition::new(vec![outer]).unwrap();
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F nested exact convex Euclidean erosion under shear",
    );
    // Exact C ⊖ disk(1) for square [10,20]^2 is [11,19]^2.
    // Ancestor root x=2*parent_x+.5*parent_y+4: pixel [35,36]
    // y=[14,15] is entirely IN; pixel [32,33] y=[14,15]
    // is entirely OUT of the eroded region but intersects original C.
    pixel_near(pixel(&output, 35, 14), [255, 0, 0, 255]);
    // Exact sheared inset boundary at y=14+t is x=33+0.5*t.
    // Within pixel [33,34]x[14,15] the covered horizontal width
    // is 1-0.5*t, yielding integral area 0.75. This is a continuous
    // Euclidean-inset + affine coverage oracle, not a sampled mask.
    // Straight sRGB red=255 premultiplied in linear light by 0.75
    // resolves to sRGB red≈225 and unorm alpha≈191.
    pixel_near(pixel(&output, 33, 14), [225, 0, 0, 191]);
    assert_eq!(pixel(&output, 32, 14), [0, 0, 0, 0]);
}

#[test]
fn nested_convex_erosion_gaussian_integrates_in_its_parent_frame() {
    let Some(context) = context() else { return };
    let inner = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.1,
            -1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![inner],
        Render2dAffineTransform::new(2.0, 0.0, 0.5, 1.0, 4.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        Vec::new(),
    ));
    let scene = Render2dComposition::new(vec![outer]).unwrap();
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F nested convex erosion then finite Gaussian under shear",
    );
    // Exact eroded C = [11,19]^2. Finite normalized sigma=.1
    // has 3sigma=.3 parent-coordinate cutoff on each axis.
    // All inverse-mapped points in root pixel [35,36]×[14,15]
    // lie >.3 INSIDE that eroded region. Those in [31,32]×[14,15]
    // lie >.3 OUTSIDE its left boundary.
    pixel_near(pixel(&output, 35, 14), [255, 0, 0, 255]);
    assert_eq!(pixel(&output, 31, 14), [0, 0, 0, 0]);
}

#[test]
fn nested_nonconvex_disk_erosion_rejects_with_authored_path_before_gpu_work() {
    let Some(context) = context() else { return };
    let inner = Render2dEntry::group(Render2dGroup::new(
        vec![
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 10.0, 4.0, 2.0),
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 12.0, 2.0, 2.0),
        ],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.0,
            -0.25,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![inner],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
        Vec::new(),
    ));
    let scene = Render2dComposition::new(vec![outer]).unwrap();
    let (texture, target) = target("F3F nonconvex nested signed erosion rejects atomically");
    let mut executor = Render2dExecutor::new();
    let result = executor.prepare(
        &context,
        &scene,
        &Render2dResourceBindings::default(),
        &target,
    );
    let failure = match result {
        Ok(_) => panic!("nonconvex disk erosion cannot become a convex rectangle"),
        Err(error) => error,
    };
    assert!(matches!(
        failure,
        Render2dExecutionError::UnsupportedEntry {
            path,
            kind: runen_render::execution_2d::Render2dUnsupportedContent::Shadows {
                root_index: 0,
            },
        } if path == [0, 0]
    ));
    drop(texture);
}

#[test]
fn invisible_nested_gaussian_expands_exact_neutral_support_for_ancestor() {
    let Some(context) = context() else { return };
    let nested = Render2dEntry::group(Render2dGroup::new(
        vec![caster(
            Render2dColorRgba8::TRANSPARENT,
            10.0,
            10.0,
            10.0,
            10.0,
        )],
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
        vec![effect(0.0, 0.0, 0.5, 0.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![nested],
        Render2dAffineTransform::new(2.0, 0.0, 0.5, 1.0, 4.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(0, 255, 0, 255),
        )],
    ));
    let scene = Render2dComposition::new(vec![outer]).unwrap();
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F invisible nested Gaussian retains expanded ancestor support",
    );
    // Finite blur SUPPORT (not alpha) is C + [-1.5,1.5]^2 in
    // immediate-parent coordinates. Root x=29,y=13 maps to parent
    // x in [9.1,9.7], strictly outside original C x>=10 but inside
    // the dilated support. Child/group/color alpha do not erase it.
    pixel_near(pixel(&output, 29, 13), [0, 255, 0, 255]);
    assert_eq!(pixel(&output, 26, 13), [0, 0, 0, 0]);
}

#[test]
fn transparent_nested_shadow_support_reaches_ancestor_after_own_clip() {
    let Some(context) = context() else {
        return;
    };
    let clip = Render2dClip::new(
        Render2dShape::rect(Render2dRect::new(7.0, 10.0, 2.0, 6.0).unwrap()),
        Render2dAffineTransform::IDENTITY,
    );
    let nested = Render2dEntry::group(Render2dGroup::new(
        vec![caster(Render2dColorRgba8::TRANSPARENT, 2.0, 10.0, 4.0, 6.0)],
        Render2dAffineTransform::IDENTITY,
        vec![clip],
        Render2dOpacity::TRANSPARENT,
        vec![effect(5.0, 0.0, 0.0, 0.0, Render2dColorRgba8::TRANSPARENT)],
    ));
    let outer = Render2dEntry::group(Render2dGroup::new(
        vec![nested],
        Render2dAffineTransform::new(2.0, 0.0, 0.5, 1.0, 5.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
        vec![effect(
            0.0,
            0.0,
            0.0,
            0.0,
            Render2dColorRgba8::new(0, 255, 0, 255),
        )],
    ));
    let scene = Render2dComposition::new(vec![outer]).unwrap();
    let output = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene,
        &Render2dResourceBindings::default(),
        "F3F alpha-neutral nested support after clip",
    );
    // The nested clip removes original C [2,6] and keeps only the
    // transparent effect C+5 intersected with [7,9]. Ancestor
    // x'=2x+0.5y+5 maps it to [25.5,29.5] at y=13.
    pixel_near(pixel(&output, 27, 13), [0, 255, 0, 255]);
    assert_eq!(pixel(&output, 22, 13), [0, 0, 0, 0]);
    assert_eq!(pixel(&output, 32, 13), [0, 0, 0, 0]);
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
fn orthogonal_root_concave_l_erodes_with_true_euclidean_disk_on_vulkan() {
    let Some(context) = context() else { return };
    // Source-neutral alpha-zero L. The missing upper-right quadrant must not
    // be filled by a polygon bounding box or an internal triangle seam.
    let composition = group(
        vec![
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 10.0, 10.0, 5.0),
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 15.0, 5.0, 5.0),
        ],
        vec![effect(
            0.0,
            0.0,
            0.0,
            -1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F certified root concave-L Euclidean negative spread",
    );
    pixel_near(pixel(&image, 12, 12), [255, 0, 0, 255]);
    pixel_near(pixel(&image, 17, 12), [255, 0, 0, 255]);
    pixel_near(pixel(&image, 12, 17), [255, 0, 0, 255]);
    // The [14,15]^2 sample cell abuts the *concave* missing quadrant.
    // Erosion removes exactly one quarter unit disk from its filled unit
    // square: alpha=1-pi/4. F3E writes linear-premul red through sRGB.
    pixel_near(pixel(&image, 14, 14), [128, 0, 0, 55]);
    assert_eq!(pixel(&image, 18, 18), [0, 0, 0, 0]);
    assert_eq!(pixel(&image, 9, 12), [0, 0, 0, 0]);
}

#[test]
fn orthogonal_root_hole_erodes_from_both_exterior_boundaries_on_vulkan() {
    let Some(context) = context() else { return };
    // Exact 20x20 outer ring with an [16,24]^2 hole. Top/bottom/side
    // rectangles touch; all their internal tessellation seams are NOT an
    // erosion boundary. The central hole is genuine union exterior.
    let composition = group(
        vec![
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 10.0, 20.0, 6.0),
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 16.0, 6.0, 8.0),
            caster(Render2dColorRgba8::TRANSPARENT, 24.0, 16.0, 6.0, 8.0),
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 24.0, 20.0, 6.0),
        ],
        vec![effect(
            0.0,
            0.0,
            0.0,
            -1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3F certified root holed-ring Euclidean negative spread",
    );
    pixel_near(pixel(&image, 13, 20), [255, 0, 0, 255]);
    pixel_near(pixel(&image, 27, 20), [255, 0, 0, 255]);
    // The outside of the hole's southwest corner [15,16]^2 also loses
    // exactly pi/4 to the Euclidean disk around (16,16). An AABB/alpha
    // approximation cannot satisfy this fractional geometry reference.
    pixel_near(pixel(&image, 15, 15), [128, 0, 0, 55]);
    assert_eq!(pixel(&image, 20, 20), [0, 0, 0, 0]);
    assert_eq!(pixel(&image, 9, 20), [0, 0, 0, 0]);
}

/// Independent corner, full-interior and hole/outside readback requirements.
/// F3F may not accept a ResourceLimit for these ordinary finite-Gaussian
/// workload classes: every test below requires a real completed Vulkan render.
fn assert_orthogonal_finite_gaussian_on_vulkan(
    composition: &Render2dComposition,
    corner: [u32; 2],
    interior: [u32; 2],
    exterior: [u32; 2],
    label: &str,
) {
    let Some(context) = context() else { return };
    let image = execute(
        &context,
        &mut Render2dExecutor::new(),
        composition,
        &Render2dResourceBindings::default(),
        label,
    );
    // Analytic normalized finite 3sigma Gaussian integration over the
    // radius-1 Euclidean-eroded orthogonal union gives alpha=0.22752904
    // for the 16 correlated 4x4 samples at this fractional corner.
    pixel_near(pixel(&image, corner[0], corner[1]), [131, 0, 0, 58]);
    pixel_near(pixel(&image, interior[0], interior[1]), [255, 0, 0, 255]);
    assert_eq!(pixel(&image, exterior[0], exterior[1]), [0, 0, 0, 0]);
}

#[test]
fn finite_gaussian_concave_l_must_render_independent_oracle_on_vulkan() {
    let composition = group(
        vec![
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 10.0, 10.0, 5.0),
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 15.0, 5.0, 5.0),
        ],
        vec![effect(
            0.0,
            0.0,
            0.1,
            -1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    assert_orthogonal_finite_gaussian_on_vulkan(
        &composition,
        [14, 14],
        [12, 12],
        [18, 18],
        "F3F NS02 L Gaussian actual Vulkan",
    );
}

#[test]
fn finite_gaussian_holed_ring_must_render_independent_oracle_on_vulkan() {
    let composition = group(
        vec![
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 10.0, 20.0, 6.0),
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 16.0, 6.0, 8.0),
            caster(Render2dColorRgba8::TRANSPARENT, 24.0, 16.0, 6.0, 8.0),
            caster(Render2dColorRgba8::TRANSPARENT, 10.0, 24.0, 20.0, 6.0),
        ],
        vec![effect(
            0.0,
            0.0,
            0.1,
            -1.0,
            Render2dColorRgba8::new(255, 0, 0, 255),
        )],
        1.0,
    );
    assert_orthogonal_finite_gaussian_on_vulkan(
        &composition,
        [15, 15],
        [13, 20],
        [20, 20],
        "F3F NS02 ring Gaussian actual Vulkan",
    );
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

    // The bound applies across independent sibling groups, not only to
    // the length of one group's effect list. Admission must reject at the
    // second group before producing either sibling's expensive masks.
    let siblings = Render2dComposition::new(vec![
        Render2dEntry::group(Render2dGroup::new(
            caster.clone(),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
            vec![red; 128],
        )),
        Render2dEntry::group(Render2dGroup::new(
            caster.clone(),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
            vec![red; 129],
        )),
    ])
    .unwrap();
    let sibling_error = match executor.prepare(&context, &siblings, &bindings, &target) {
        Ok(_) => panic!("257 shadows split across siblings must fail admission"),
        Err(error) => error,
    };
    assert!(matches!(
        sibling_error,
        Render2dExecutionError::SampleSpace {
            kind: runen_render::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            path: Some(path),
            ..
        } if path == [1]
    ));

    // Distinguish TRUE preflight cardinality from a late error after
    // spending the first sibling's geometry/morphology budget. Its sigma is
    // individually unadmittable; a late count would report path [0], while
    // invocation-wide effect admission MUST reject 1+256 at path [1]
    // without even attempting the first Gaussian allocation.
    let expensive_first = Render2dComposition::new(vec![
        Render2dEntry::group(Render2dGroup::new(
            caster.clone(),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
            vec![effect(
                0.0,
                0.0,
                500.0,
                0.0,
                Render2dColorRgba8::new(255, 0, 0, 128),
            )],
        )),
        Render2dEntry::group(Render2dGroup::new(
            caster.clone(),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
            vec![red; 256],
        )),
    ])
    .unwrap();
    let prioritized_error = match executor.prepare(&context, &expensive_first, &bindings, &target) {
        Ok(_) => panic!("aggregate effect count must precede expensive first-sibling preparation"),
        Err(error) => error,
    };
    assert!(matches!(
        prioritized_error,
        Render2dExecutionError::SampleSpace {
            kind: runen_render::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            path: Some(path),
            ..
        } if path == [1]
    ));

    // Cardinality must also precede ordinary F3E paint inspection: this
    // first root item has a finite affine whose Frobenius scale is not
    // representable as a physical vector tessellation. Paint inspection
    // would report the first item's precision error if it ran too early.
    let oversized_paint = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 1.0, 1.0).unwrap()),
            brush: Render2dBrush::solid(Render2dColorRgba8::WHITE),
        },
        Render2dAffineTransform::new(1.0e308, 0.0, 0.0, 1.0e308, 0.0, 0.0).unwrap(),
        Vec::new(),
        Render2dOpacity::OPAQUE,
    ));
    let paint_before_effect_limit = Render2dComposition::new(vec![
        oversized_paint,
        Render2dEntry::group(Render2dGroup::new(
            caster.clone(),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
            vec![red; 257],
        )),
    ])
    .unwrap();
    let paint_priority =
        match executor.prepare(&context, &paint_before_effect_limit, &bindings, &target) {
            Ok(_) => panic!("shadow count must preempt costly physical paint inspection"),
            Err(error) => error,
        };
    assert!(matches!(
        paint_priority,
        Render2dExecutionError::SampleSpace {
            kind: runen_render::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            path: Some(path),
            ..
        } if path == [1]
    ));

    // Neither failure can publish partial GPU work or poison subsequent
    // executor identity. The next valid composition retains the caller's
    // exact preexisting blue pixel and source-over order.
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

/// The original rectangle and finite Gaussian are independent of paint alpha.
/// The expected samples below come from the continuous F1 reference, never a
/// renderer-owned raster mask or the test-only region-DAG feasibility helper.
#[test]
fn finite_gaussian_shadows_preserve_order_at_fractional_boundary_on_vulkan() {
    let Some(context) = context() else { return };
    let red = effect(0.0, 0.0, 0.25, 0.0, Render2dColorRgba8::new(255, 0, 0, 128));
    let blue = effect(0.0, 0.0, 0.25, 0.0, Render2dColorRgba8::new(0, 0, 255, 128));

    // Original neutral source: [10,20]^2. For pixel [10,10], the four
    // correlated sample centers are 10.125, 10.375, 10.625, 10.875
    // on each axis. Independent normalized finite Gaussian axis integrals:
    // (Phi(min(3, (center-10)/0.25))-Phi(-3))/(Phi(3)-Phi(-3)) =
    // [0.6919807702016251, 0.9343654969899314,
    //  0.995127076807001, 1.0], with Phi the standard normal CDF.
    //
    // At each of 16 sites, q=(128/255)*mass_x*mass_y. Red then blue
    // is premul [q(1-q),0,q] and alpha 2q-q². Reverse painter order
    // exchanges red and blue. Average all premul site colors BEFORE the
    // once-only linear-sRGB8 resolve: [133,0,172,165] or [172,0,133,165].
    let neutral_rectangle = vec![caster(
        Render2dColorRgba8::TRANSPARENT,
        10.0,
        10.0,
        10.0,
        10.0,
    )];
    for (shadows, boundary, interior, label) in [
        (
            vec![red, blue],
            [133, 0, 172, 165],
            [137, 0, 188, 191],
            "F3F finite Gaussian red then blue",
        ),
        (
            vec![blue, red],
            [172, 0, 133, 165],
            [188, 0, 137, 191],
            "F3F finite Gaussian blue then red",
        ),
    ] {
        let result = execute(
            &context,
            &mut Render2dExecutor::new(),
            &group(neutral_rectangle.clone(), shadows, 1.0),
            &Render2dResourceBindings::default(),
            label,
        );
        pixel_near(pixel(&result, 10, 10), boundary);
        // The full finite Gaussian kernel is inside the neutral rectangle.
        pixel_near(pixel(&result, 15, 15), interior);
        // No tail is permitted outside the exact finite 3σ square support.
        assert_eq!(pixel(&result, 7, 7), [0, 0, 0, 0]);
    }
}

/// General original-F1 reach culling: a symbolic curved descendant cannot
/// reach the canvas through either root shadow. This is not a curved coverage
/// approximation or a special-case shape classifier. Original paint remains.
#[test]
fn wholly_offscreen_curved_root_effects_preserve_actual_vulkan_child_paint() {
    let Some(context) = context() else {
        return;
    };
    let scene = |shadows: Vec<Render2dDropShadow>| {
        let curved = Render2dEntry::group(Render2dGroup::new(
            vec![caster(
                Render2dColorRgba8::TRANSPARENT,
                10.0,
                10.0,
                8.0,
                8.0,
            )],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            vec![effect(0.0, 0.0, 0.1, 1.0, Render2dColorRgba8::TRANSPARENT)],
        ));
        let paint = Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(12.0, 12.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::new(0, 255, 0, 255)),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
        ));
        group(vec![curved, paint], shadows, 1.0)
    };
    let offscreen = effect(
        100.0,
        0.0,
        0.25,
        -0.5,
        Render2dColorRgba8::new(255, 0, 0, 255),
    );
    let reference = execute(
        &context,
        &mut Render2dExecutor::new(),
        &scene(Vec::new()),
        &Render2dResourceBindings::default(),
        "F3F original F1 children without offscreen effects",
    );
    let mut executor = Render2dExecutor::new();
    let actual = execute(
        &context,
        &mut executor,
        &scene(vec![offscreen, offscreen]),
        &Render2dResourceBindings::default(),
        "F3F provably offscreen curved original F1 shadows",
    );
    assert_eq!(actual.as_bytes(), reference.as_bytes());
    pixel_near(pixel(&actual, 12, 12), [0, 255, 0, 255]);
    assert_eq!(pixel(&actual, 9, 9), [0, 0, 0, 0]);

    // One surviving root shadow must still require exact curved coverage.
    // A single offscreen sibling may never make the whole source disappear.
    let visible = effect(
        0.0,
        0.0,
        0.25,
        -0.5,
        Render2dColorRgba8::new(255, 0, 0, 255),
    );
    let (_, target) = target("F3F mixed offscreen and surviving root shadow");
    let error = match executor.prepare(
        &context,
        &scene(vec![offscreen, visible]),
        &Render2dResourceBindings::default(),
        &target,
    ) {
        Ok(_) => panic!("uncertified surviving curved root must reject before target work"),
        Err(error) => error,
    };
    assert!(matches!(
        error,
        Render2dExecutionError::UnsupportedEntry {
            path,
            kind: runen_render::execution_2d::Render2dUnsupportedContent::Shadows {
                root_index: 0
            },
        } if path == [0, 0]
    ));
    let recovered = execute(
        &context,
        &mut executor,
        &scene(Vec::new()),
        &Render2dResourceBindings::default(),
        "F3F executor remains usable after mixed curved rejection",
    );
    assert_eq!(recovered.as_bytes(), reference.as_bytes());
}
