//! F3E external consumer proof: actual immutable F1 group -> public executor
//! -> RunenGPU work/evidence -> unmodified caller-owned RGBA8-sRGB readback.
//! These pixels are computed independently from straight-alpha source facts.
use super::*;
use runen_render::composition_2d::{
    Render2dBrush, Render2dClip, Render2dGradientStop, Render2dGradientStops, Render2dImagePatch,
    Render2dImagePrimitive, Render2dImageResource, Render2dImageSourceRect, Render2dLinearGradient,
    Render2dPixelExtent, Render2dRect, Render2dShape,
};

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

fn clip(left: f64, width: f64) -> Render2dClip {
    Render2dClip::new(
        Render2dShape::rect(Render2dRect::new(left, 0.0, width, 64.0).unwrap()),
        Render2dAffineTransform::IDENTITY,
    )
}

#[test]
fn grouped_half_pixel_and_identical_parent_clip_correlate_once() {
    let Some(ctx) = context() else {
        return;
    };
    let group = Render2dGroup::new(
        vec![solid(Render2dColorRgba8::new(255, 0, 0, 255), 10.0, 0.5)],
        Render2dAffineTransform::IDENTITY,
        vec![clip(10.0, 0.5)],
        Render2dOpacity::OPAQUE,
        Vec::new(),
    );
    let composition = Render2dComposition::new(vec![Render2dEntry::group(group)]).unwrap();
    let image = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3E independent group intersected sample clip",
    );
    // Both binary masks select the same 8/16 samples. Applying the
    // resolved 0.5 mask independently would incorrectly yield 0.25.
    pixel_close(pixel(&image, 10, 20), [188, 0, 0, 128]);
    pixel_close(pixel(&image, 11, 20), [0, 0, 0, 0]);
}

#[test]
fn group_item_clip_uses_its_own_parent_space_after_group_transform() {
    let Some(ctx) = context() else {
        return;
    };
    // The child's local x=[3,3.25) maps through outer x'=2x+4 to
    // global x=[10,10.5). The item clip is in the *outer group's*
    // local space x=[3,3.25), while the group's own clip is in
    // global x=[10,10.5). Neither clip may inherit its owner's
    // own local transform twice.
    let child = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(Render2dRect::new(3.0, 0.0, 0.25, 64.0).unwrap()),
            brush: Render2dBrush::solid(Render2dColorRgba8::new(255, 0, 0, 255)),
        },
        Render2dAffineTransform::IDENTITY,
        vec![clip(3.0, 0.25)],
        Render2dOpacity::OPAQUE,
    ));
    let outer = Render2dGroup::new(
        vec![child],
        Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, 4.0, 0.0).unwrap(),
        vec![clip(10.0, 0.5)],
        Render2dOpacity::OPAQUE,
        Vec::new(),
    );
    let composition = Render2dComposition::new(vec![Render2dEntry::group(outer)]).unwrap();
    let image = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        "F3E independent parent-frame item and group clip",
    );
    pixel_close(pixel(&image, 10, 20), [188, 0, 0, 128]);
    pixel_close(pixel(&image, 11, 20), [0, 0, 0, 0]);
}

fn linear_gradient_item(start: f64, end: f64, left: f64, width: f64) -> Render2dEntry {
    let stops = Render2dGradientStops::new(vec![
        Render2dGradientStop::new(0.0, Render2dColorRgba8::new(255, 0, 0, 255)).unwrap(),
        Render2dGradientStop::new(1.0, Render2dColorRgba8::new(0, 0, 255, 255)).unwrap(),
    ])
    .unwrap();
    let brush = Render2dBrush::Linear(
        Render2dLinearGradient::new(
            Render2dPoint::new(start, 0.0).unwrap(),
            Render2dPoint::new(end, 0.0).unwrap(),
            stops,
        )
        .unwrap(),
    );
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(Render2dRect::new(left, 0.0, width, 64.0).unwrap()),
            brush,
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::OPAQUE,
    ))
}

fn encode_linear(value: f64) -> u8 {
    let encoded = if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Independent F1 oracle: physical centers on fixed 4x4 lattice, straight
/// red/blue stops, followed by group opacity once and a final linear resolve.
fn expected_gradient_pixel(
    pixel_x: u32,
    start: f64,
    end: f64,
    coverage: impl Fn(f64) -> bool,
    opacity: f64,
) -> [u8; 4] {
    let mut red = 0.0;
    let mut blue = 0.0;
    let mut alpha = 0.0;
    for sy in 0..4 {
        for sx in 0..4 {
            let _sample_y = (f64::from(sy) + 0.5) / 4.0;
            let x = f64::from(pixel_x) + (f64::from(sx) + 0.5) / 4.0;
            if coverage(x) {
                let t = ((x - start) / (end - start)).clamp(0.0, 1.0);
                red += (1.0 - t) * opacity / 16.0;
                blue += t * opacity / 16.0;
                alpha += opacity / 16.0;
            }
        }
    }
    [
        encode_linear(red),
        0,
        encode_linear(blue),
        (alpha * 255.0).round() as u8,
    ]
}

#[test]
fn nested_linear_gradient_matches_independent_per_sample_oracle() {
    let Some(ctx) = context() else {
        return;
    };
    let root = Render2dComposition::new(vec![group(
        vec![group(vec![linear_gradient_item(0.0, 64.0, 0.0, 64.0)], 0.5)],
        1.0,
    )])
    .unwrap();
    let result = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &root,
        &Render2dResourceBindings::default(),
        "F3E nested sampled linear gradient",
    );
    for x in [0, 16, 32, 48, 63] {
        pixel_close(
            pixel(&result, x, 20),
            expected_gradient_pixel(x, 0.0, 64.0, |_| true, 0.5),
        );
    }
}

#[test]
fn fractional_group_clip_and_gradient_source_share_exact_samples() {
    let Some(ctx) = context() else {
        return;
    };
    let parent = Render2dGroup::new(
        vec![linear_gradient_item(10.0, 11.0, 10.0, 0.5)],
        Render2dAffineTransform::IDENTITY,
        vec![clip(10.0, 0.5)],
        Render2dOpacity::new(0.5).unwrap(),
        Vec::new(),
    );
    let root = Render2dComposition::new(vec![Render2dEntry::group(parent)]).unwrap();
    let result = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &root,
        &Render2dResourceBindings::default(),
        "F3E fractional gradient clip and opacity",
    );
    pixel_close(
        pixel(&result, 10, 20),
        expected_gradient_pixel(10, 10.0, 11.0, |x| x < 10.5, 0.5),
    );
    pixel_close(pixel(&result, 11, 20), [0, 0, 0, 0]);
}

fn repeated_image_patches(id: Render2dResourceId, item_opacity: f64) -> Render2dEntry {
    repeated_image_patches_at(id, 0.0, 64.0, item_opacity)
}

fn repeated_image_patches_at(
    id: Render2dResourceId,
    x: f64,
    width: f64,
    item_opacity: f64,
) -> Render2dEntry {
    let patch = Render2dImagePatch::new(
        Render2dImageSourceRect::new(0.0, 0.0, 1.0, 1.0).unwrap(),
        Render2dRect::new(x, 0.0, width, 64.0).unwrap(),
    );
    let image = Render2dImagePrimitive::new(
        id,
        Render2dPixelExtent::new(1, 1).unwrap(),
        vec![patch, patch],
    )
    .unwrap();
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Image(image),
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::new(item_opacity).unwrap(),
    ))
}

fn image_binding(id: Render2dResourceId, source: [u8; 4]) -> Render2dResourceBindings {
    Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
        id,
        Render2dResourceValue::ImageRgba8Srgb(
            Render2dImageResource::new(Render2dPixelExtent::new(1, 1).unwrap(), source.to_vec())
                .unwrap(),
        ),
    )])
    .unwrap()
}

#[test]
fn overlapping_image_patches_apply_item_opacity_once_after_ordered_source_over() {
    let Some(ctx) = context() else {
        return;
    };
    let id = Render2dResourceId::new(776).unwrap();
    let composition =
        Render2dComposition::new(vec![group(vec![repeated_image_patches(id, 0.5)], 1.0)]).unwrap();
    let bytes = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &composition,
        &image_binding(id, [255, 0, 0, 128]),
        "F3E-R1 image patch isolation public oracle",
    );
    let alpha = f64::from(128_u8) / 255.0;
    let covered = 1.0 - (1.0 - alpha).powi(2);
    let item_alpha = covered * 0.5;
    let expected = [
        encode_linear(item_alpha),
        0,
        0,
        (255.0 * item_alpha).round() as u8,
    ];
    pixel_close(pixel(&bytes, 20, 20), expected);
}

#[test]
fn image_binding_identity_survives_group_execution_and_cache_independence() {
    let Some(ctx) = context() else {
        return;
    };
    let id = Render2dResourceId::new(777).unwrap();
    let composition =
        Render2dComposition::new(vec![group(vec![repeated_image_patches(id, 0.75)], 1.0)]).unwrap();
    let mut executor = Render2dExecutor::new();
    let binding = image_binding(id, [255, 0, 0, 128]);
    let _ = execute(
        &ctx,
        &mut executor,
        &composition,
        &binding,
        "F3E retained grouped image binding",
    );
    executor.discard_cache();
    let (_, target) = super::target("F3E rebound image target");
    let rebound = image_binding(id, [0, 255, 0, 128]);
    assert!(matches!(
        executor.prepare(&ctx, &composition, &rebound, &target),
        Err(Render2dExecutionError::ResourceIdentityRebound { resource_id })
            if resource_id == id
    ));
}

#[test]
fn multi_tile_image_item_and_group_clips_retain_absolute_sample_phase() {
    let Some(ctx) = context() else {
        return;
    };
    // 576 pixel width forces three output tiles at F3E's <=256px tile side.
    // All sample positions and alpha answers below derive exclusively from F1
    // geometry and the independent straight-alpha source-over equation.
    let id = Render2dResourceId::new(778).unwrap();
    let left_group = Render2dGroup::new(
        vec![repeated_image_patches_at(id, 254.75, 2.5, 0.5)],
        Render2dAffineTransform::IDENTITY,
        vec![clip(255.25, 1.5)],
        Render2dOpacity::OPAQUE,
        Vec::new(),
    );
    let distant_group = group(vec![repeated_image_patches_at(id, 510.0, 4.0, 0.25)], 1.0);
    let composition =
        Render2dComposition::new(vec![Render2dEntry::group(left_group), distant_group]).unwrap();
    let result = super::execute_sized(
        &ctx,
        &mut Render2dExecutor::new(),
        &composition,
        &image_binding(id, [255, 0, 0, 128]),
        "F3E wide three-tile source-and-clip sample phase",
        576,
        64,
    );
    let source_alpha = f64::from(128_u8) / 255.0;
    let two_patch_alpha = 1.0 - (1.0 - source_alpha).powi(2);
    let expected_at_seam = two_patch_alpha * 0.5 * (3.0 / 4.0);
    let expected_distant = two_patch_alpha * 0.25;
    let encoded = |alpha: f64| [encode_linear(alpha), 0, 0, (alpha * 255.0).round() as u8];
    for x in [255, 256] {
        pixel_close(
            super::pixel_sized(&result, 576, x, 20),
            encoded(expected_at_seam),
        );
    }
    for x in [510, 512, 513] {
        pixel_close(
            super::pixel_sized(&result, 576, x, 20),
            encoded(expected_distant),
        );
    }
    for x in [254, 257, 514, 575] {
        pixel_close(super::pixel_sized(&result, 576, x, 20), [0, 0, 0, 0]);
    }
}

#[test]
fn direct_root_and_identity_group_share_disjoint_4x4_source_over_law() {
    let Some(ctx) = context() else {
        return;
    };
    let painter = || {
        vec![
            solid(Render2dColorRgba8::new(255, 0, 0, 255), 10.0, 0.5),
            solid(Render2dColorRgba8::new(0, 0, 255, 255), 10.5, 0.5),
        ]
    };
    let direct = Render2dComposition::new(painter()).unwrap();
    let identity = Render2dComposition::new(vec![group(painter(), 1.0)]).unwrap();
    let ungrouped = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &direct,
        &Render2dResourceBindings::default(),
        "F3E direct-root correlated sibling paints",
    );
    let grouped = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &identity,
        &Render2dResourceBindings::default(),
        "F3E identity-group correlated sibling paints",
    );
    // Opaque disjoint half-pixel rectangles cover every physical sample.
    // Legacy separately resolved per-item source-over incorrectly had alpha .75.
    for result in [&ungrouped, &grouped] {
        pixel_close(pixel(result, 10, 20), [188, 0, 188, 255]);
        pixel_close(pixel(result, 11, 20), [0, 0, 0, 0]);
    }
    assert_eq!(pixel(&ungrouped, 10, 20), pixel(&grouped, 10, 20));
}

#[test]
fn direct_root_image_item_opacity_matches_identity_group_r1() {
    let Some(ctx) = context() else {
        return;
    };
    let id = Render2dResourceId::new(779).unwrap();
    let source = image_binding(id, [255, 0, 0, 128]);
    let direct = Render2dComposition::new(vec![repeated_image_patches(id, 0.5)]).unwrap();
    let grouped =
        Render2dComposition::new(vec![group(vec![repeated_image_patches(id, 0.5)], 1.0)]).unwrap();
    let direct_image = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &direct,
        &source,
        "F3E direct-root once-only image opacity",
    );
    let identity_image = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &grouped,
        &source,
        "F3E grouped once-only image opacity",
    );
    let straight_alpha = f64::from(128_u8) / 255.0;
    let item_alpha = (1.0 - (1.0 - straight_alpha).powi(2)) * 0.5;
    let expected = [
        encode_linear(item_alpha),
        0,
        0,
        (item_alpha * 255.0).round() as u8,
    ];
    for result in [&direct_image, &identity_image] {
        pixel_close(pixel(result, 20, 20), expected);
    }
    assert_eq!(pixel(&direct_image, 20, 20), pixel(&identity_image, 20, 20));
}

#[test]
fn grouped_f2_text_keeps_pixel_center_coverage_and_linear_source_over() {
    let Some(ctx) = context() else {
        return;
    };
    let id = Render2dResourceId::new(780).unwrap();
    let source = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let painter = || {
        vec![
            solid(Render2dColorRgba8::new(255, 0, 0, 128), 0.0, 64.0),
            shaped_entry(id, [8.0, 32.0], Render2dColorRgba8::new(0, 0, 255, 128)),
        ]
    };
    let direct = Render2dComposition::new(painter()).unwrap();
    let isolated = Render2dComposition::new(vec![group(painter(), 1.0)]).unwrap();
    let direct_pixels = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &direct,
        &source,
        "F3E mixed direct-root vector and F2 text",
    );
    let isolated_pixels = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &isolated,
        &source,
        "F3E mixed identity-group vector and F2 text",
    );
    for (x, y) in [(20, 20), (8, 32), (0, 0), (40, 40)] {
        let actual = pixel(&isolated_pixels, x, y);
        pixel_close(actual, pixel(&direct_pixels, x, y));
    }
    // At the fixture box's interior the retained glyph is fully opaque
    // before its authored half-alpha, painted over half-alpha red.
    pixel_close(pixel(&isolated_pixels, 20, 20), [137, 0, 188, 192]);
}

#[test]
fn nested_f2_text_group_clip_and_opacity_apply_at_correlated_samples() {
    let Some(ctx) = context() else {
        return;
    };
    let id = Render2dResourceId::new(781).unwrap();
    let source = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let nested = group(
        vec![shaped_entry(
            id,
            [8.0, 32.0],
            Render2dColorRgba8::new(0, 0, 255, 255),
        )],
        1.0,
    );
    let clipped = Render2dGroup::new(
        vec![nested],
        Render2dAffineTransform::IDENTITY,
        vec![clip(20.0, 0.5)],
        Render2dOpacity::new(0.5).unwrap(),
        Vec::new(),
    );
    let composition = Render2dComposition::new(vec![Render2dEntry::group(clipped)]).unwrap();
    let result = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &composition,
        &source,
        "F3E retained glyph with group sample clip and group alpha",
    );
    pixel_close(pixel(&result, 20, 20), [0, 0, 137, 64]);
    pixel_close(pixel(&result, 19, 20), [0, 0, 0, 0]);
    pixel_close(pixel(&result, 21, 20), [0, 0, 0, 0]);
}

#[test]
fn grouped_text_unsupported_intrinsic_fails_without_observing_identity() {
    let Some(ctx) = context() else {
        return;
    };
    let id = Render2dResourceId::new(782).unwrap();
    let composition = Render2dComposition::new(vec![group(
        vec![shaped_entry(id, [8.0, 32.0], Render2dColorRgba8::WHITE)],
        1.0,
    )])
    .unwrap();
    let rejected = bindings(id, shaped_resource(COLR_V0_FONT, false, BOX_GLYPH, 24.0));
    let replacement = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let (_, target) = super::target("F3E unsupported text atomicity");
    let mut executor = Render2dExecutor::new();
    assert!(matches!(
        executor.prepare(&ctx, &composition, &rejected, &target),
        Err(Render2dExecutionError::ShapedText(
            Render2dShapedTextError::UnsupportedGlyph {
                resource_id,
                kind: Render2dUnsupportedGlyphKind::ColrV0,
                ..
            }
        )) if resource_id == id
    ));
    assert!(
        executor
            .prepare(&ctx, &composition, &replacement, &target)
            .expect("failed group preparation must not observe rejected semantic identity")
            .has_render_work()
    );
}

#[test]
fn missing_sample_plane_roles_reject_root_and_identity_group_equally() {
    // Target admission succeeds, but Rgba16Float sample planes are NOT admitted.
    // The old direct-root fallback would silently render different pixels.
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(
                GpuTextureFormat::Rgba8UnormSrgb,
                GpuFormatRole::ColorAttachment,
            )
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable)
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_label("F3E typed unified sample-plane admission");
    let context = match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => context,
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1")
            );
            return;
        }
        Err(error) => panic!("F3E typed capability test GPU request: {error}"),
    };
    let entries = || {
        vec![
            solid(Render2dColorRgba8::WHITE, 10.0, 0.5),
            solid(Render2dColorRgba8::new(0, 0, 255, 255), 10.5, 0.5),
        ]
    };
    let direct = Render2dComposition::new(entries()).unwrap();
    let nested = Render2dComposition::new(vec![group(entries(), 1.0)]).unwrap();
    let (_, target) = super::target("F3E missing sample roles");
    for composition in [&direct, &nested] {
        assert!(matches!(
            Render2dExecutor::new().prepare(
                &context, composition, &Render2dResourceBindings::default(), &target
            ),
            Err(Render2dExecutionError::Target(
                runen_render::execution_2d::Render2dTargetAdmissionError::SamplePlaneFormatUnsupported
            ))
        ));
    }
    // Purely non-painting input admits no operation and needs no sample scratch.
    let empty = Render2dComposition::new(Vec::new()).unwrap();
    assert!(
        !Render2dExecutor::new()
            .prepare(
                &context,
                &empty,
                &Render2dResourceBindings::default(),
                &target
            )
            .unwrap()
            .has_render_work()
    );
}

#[test]
fn opaque_overlapping_ordered_patches_apply_item_opacity_and_quarter_clip_once() {
    let Some(ctx) = context() else { return };
    let id = Render2dResourceId::new(783).unwrap();
    let patch = |source_x| {
        Render2dImagePatch::new(
            Render2dImageSourceRect::new(source_x, 0.0, 1.0, 1.0).unwrap(),
            Render2dRect::new(10.0, 0.0, 1.0, 64.0).unwrap(),
        )
    };
    let image = Render2dImagePrimitive::new(
        id,
        Render2dPixelExtent::new(2, 1).unwrap(),
        vec![patch(0.0), patch(1.0)],
    )
    .unwrap();
    let binding = Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
        id,
        Render2dResourceValue::ImageRgba8Srgb(
            Render2dImageResource::new(
                Render2dPixelExtent::new(2, 1).unwrap(),
                vec![255, 0, 0, 255, 0, 0, 255, 255],
            )
            .unwrap(),
        ),
    )])
    .unwrap();
    let item = |opacity, clips| {
        Render2dEntry::item(Render2dItem::new(
            Render2dPrimitive::Image(image.clone()),
            Render2dAffineTransform::IDENTITY,
            clips,
            Render2dOpacity::new(opacity).unwrap(),
        ))
    };
    let direct = Render2dComposition::new(vec![item(0.5, Vec::new())]).unwrap();
    let nested = Render2dComposition::new(vec![group(vec![item(0.5, Vec::new())], 1.0)]).unwrap();
    for composition in [&direct, &nested] {
        let bytes = execute(
            &ctx,
            &mut Render2dExecutor::new(),
            composition,
            &binding,
            "F3E-R1 opaque patch opacity",
        );
        pixel_close(pixel(&bytes, 10, 20), [0, 0, encode_linear(0.5), 128]);
    }
    let quarter_clip =
        Render2dComposition::new(vec![group(vec![item(1.0, vec![clip(10.0, 0.25)])], 1.0)])
            .unwrap();
    let result = execute(
        &ctx,
        &mut Render2dExecutor::new(),
        &quarter_clip,
        &binding,
        "F3E-R1 opaque ordered image patch quarter-coverage item clip",
    );
    pixel_close(pixel(&result, 10, 20), [0, 0, encode_linear(0.25), 64]);
    pixel_close(pixel(&result, 11, 20), [0, 0, 0, 0]);
}

#[test]
fn overlapping_opaque_group_children_clipped_once_blend_over_prior_green_target() {
    let Some(ctx) = context() else {
        return;
    };
    let painted_children = vec![
        solid(Render2dColorRgba8::new(255, 0, 0, 255), 0.0, 64.0),
        solid(Render2dColorRgba8::new(0, 0, 255, 255), 0.0, 64.0),
    ];
    let clipped = Render2dGroup::new(
        painted_children,
        Render2dAffineTransform::IDENTITY,
        vec![clip(10.0, 0.5)],
        Render2dOpacity::OPAQUE,
        Vec::new(),
    );
    let composition = Render2dComposition::new(vec![Render2dEntry::group(clipped)]).unwrap();
    let (texture, target) = super::target("F3E D3 prior opaque target");
    let result = super::execute_inline(
        &ctx,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        &texture,
        &target,
        Some([0.0, 1.0, 0.0, 1.0]),
    );
    // D3: blue overwrites red in isolated group first. A half-coverage
    // structural clip then yields blue=.5 / alpha=.5, composited once
    // over the caller's opaque green. Distributing clip to children
    // would leave illicit red and yield wrong alpha before prior blending.
    pixel_close(pixel(&result, 10, 20), [0, 188, 188, 255]);
    pixel_close(pixel(&result, 11, 20), [0, 255, 0, 255]);
}
