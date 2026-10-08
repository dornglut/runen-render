//! Independent RGBA8 image output proofs through the public GPU contribution.
use super::*;
use runen_render::composition_2d::*;
use runen_render::execution_2d::Render2dImageError;

fn context() -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
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
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Filterable)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination)
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("F3C public immutable image");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(ctx) => Some(ctx),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1")
            );
            None
        }
        Err(error) => panic!("F3C Vulkan context: {error}"),
    }
}

fn rect(x: f64, y: f64, w: f64, h: f64) -> Render2dRect {
    Render2dRect::new(x, y, w, h).unwrap()
}

fn patch(source: [f64; 4], destination: [f64; 4]) -> Render2dImagePatch {
    Render2dImagePatch::new(
        Render2dImageSourceRect::new(source[0], source[1], source[2], source[3]).unwrap(),
        rect(
            destination[0],
            destination[1],
            destination[2],
            destination[3],
        ),
    )
}

fn item(
    id: Render2dResourceId,
    patches: Vec<Render2dImagePatch>,
    transform: Render2dAffineTransform,
    opacity: f64,
) -> Render2dEntry {
    let image =
        Render2dImagePrimitive::new(id, Render2dPixelExtent::new(2, 2).unwrap(), patches).unwrap();
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Image(image),
        transform,
        vec![],
        Render2dOpacity::new(opacity).unwrap(),
    ))
}

fn bindings(id: Render2dResourceId, bytes: Vec<u8>) -> Render2dResourceBindings {
    let source =
        Render2dImageResource::new(Render2dPixelExtent::new(2, 2).unwrap(), bytes).unwrap();
    Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
        id,
        Render2dResourceValue::ImageRgba8Srgb(source),
    )])
    .unwrap()
}

fn bytes() -> Vec<u8> {
    vec![
        240, 30, 5, 255, 25, 210, 45, 255, 30, 50, 190, 255, 20, 200, 240, 128,
    ]
}

fn draw(
    ctx: &GpuContext,
    entries: Vec<Render2dEntry>,
    bind: &Render2dResourceBindings,
) -> GpuReadbackBytes {
    execute(
        ctx,
        &mut Render2dExecutor::new(),
        &Render2dComposition::new(entries).unwrap(),
        bind,
        "F3C public image",
    )
}

fn check(observed: [u8; 4], reference: [u8; 4]) {
    assert!(
        observed
            .into_iter()
            .zip(reference)
            .all(|(a, b)| a.abs_diff(b) <= 3),
        "observed {observed:?}, expected {reference:?}"
    );
}

#[test]
fn exact_nearest_source_mapping_paints_four_distinct_linearized_texels() {
    let Some(ctx) = context() else { return };
    let id = Render2dResourceId::new(401).unwrap();
    let image = item(
        id,
        vec![patch([0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 64.0, 64.0])],
        Render2dAffineTransform::IDENTITY,
        1.0,
    );
    let result = draw(&ctx, vec![image], &bindings(id, bytes()));
    check(pixel(&result, 8, 8), [240, 30, 5, 255]);
    check(pixel(&result, 48, 8), [25, 210, 45, 255]);
    check(pixel(&result, 8, 48), [30, 50, 190, 255]);
    // RGBA8 sRGB target writes premultiplied linear RGB, not encoded-space alpha.
    let encoded = |v: u8| {
        let t = f64::from(v) / 255.0;
        let linear = if t <= 0.04045 {
            t / 12.92
        } else {
            ((t + 0.055) / 1.055).powf(2.4)
        };
        let value = linear * 128.0 / 255.0;
        let srgb = if value <= 0.0031308 {
            12.92 * value
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        };
        (srgb * 255.0).round() as u8
    };
    check(
        pixel(&result, 48, 48),
        [encoded(20), encoded(200), encoded(240), 128],
    );
}

#[test]
fn fractional_crop_affine_mapping_preserves_resolved_source_and_local_coordinates() {
    let Some(ctx) = context() else { return };
    let id = Render2dResourceId::new(402).unwrap();
    let transform = Render2dAffineTransform::new(1.0, 0.25, 0.5, 1.0, 4.0, 3.0).unwrap();
    let image = item(
        id,
        vec![patch([0.5, 0.0, 1.5, 2.0], [0.0, 0.0, 32.0, 32.0])],
        transform,
        1.0,
    );
    let result = draw(&ctx, vec![image], &bindings(id, bytes()));
    // Inside affine-transformed destination, nearest texel is still selected in
    // source-pixel space after the inverse affine, never by screen pixel coordinate.
    check(pixel(&result, 13, 14), [240, 30, 5, 255]);
    check(pixel(&result, 35, 14), [25, 210, 45, 255]);
}

#[test]
fn authored_overlapping_image_patches_and_solid_vector_preserve_painter_order() {
    let Some(ctx) = context() else { return };
    let id = Render2dResourceId::new(403).unwrap();
    let red = patch([0.0, 0.0, 1.0, 1.0], [8.0, 8.0, 40.0, 40.0]);
    let blue = patch([0.0, 1.0, 1.0, 1.0], [8.0, 8.0, 40.0, 40.0]);
    let images = item(id, vec![red, blue], Render2dAffineTransform::IDENTITY, 0.5);
    let solid = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(rect(8.0, 8.0, 40.0, 40.0)),
            brush: Render2dBrush::Solid(Render2dColorRgba8::new(0, 250, 0, 128)),
        },
        Render2dAffineTransform::IDENTITY,
        vec![],
        Render2dOpacity::OPAQUE,
    ));
    let bind = bindings(id, bytes());
    let before = draw(&ctx, vec![images.clone(), solid.clone()], &bind);
    let after = draw(&ctx, vec![solid, images], &bind);
    assert_ne!(pixel(&before, 20, 20), pixel(&after, 20, 20));
    // Second patch is last within the authored image primitive.
    assert!(pixel(&after, 20, 20)[2] > pixel(&after, 20, 20)[0]);
}

#[test]
fn nonpainting_patches_do_not_author_synthetic_gpu_work_and_resource_rebind_fails() {
    let Some(ctx) = context() else { return };
    let id = Render2dResourceId::new(404).unwrap();
    let empty = item(
        id,
        vec![patch([0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 0.0, 64.0])],
        Render2dAffineTransform::IDENTITY,
        1.0,
    );
    let composition = Render2dComposition::new(vec![empty]).unwrap();
    let (_, target) = target("F3C nonpaint target");
    let mut executor = Render2dExecutor::new();
    assert!(
        !executor
            .prepare(&ctx, &composition, &bindings(id, bytes()), &target)
            .unwrap()
            .has_render_work()
    );
    executor.discard_cache();
    let mut modified = bytes();
    modified[0] = 0;
    assert!(
        matches!(executor.prepare(&ctx,&composition,&bindings(id,modified),&target),
        Err(Render2dExecutionError::ResourceIdentityRebound {resource_id}) if resource_id==id)
    );
}

#[test]
fn missing_image_capability_rejects_structurally_and_preparation_is_transactional() {
    let Some(ctx) = f2_context() else { return };
    let id = Render2dResourceId::new(405).unwrap();
    let item = item(
        id,
        vec![patch([0.0, 0.0, 2.0, 2.0], [0.0, 0.0, 32.0, 32.0])],
        Render2dAffineTransform::IDENTITY,
        1.0,
    );
    let composition = Render2dComposition::new(vec![item]).unwrap();
    let (_, target) = target("F3C unsupported format");
    let mut executor = Render2dExecutor::new();
    assert!(matches!(
        executor.prepare(&ctx, &composition, &bindings(id, bytes()), &target),
        Err(Render2dExecutionError::Image {
            root_index: 0,
            kind: Render2dImageError::FormatUnsupported
        })
    ));
    // The failed preparation did not observe resource identity.
    let mut changed = bytes();
    changed[0] = 10;
    assert!(matches!(
        executor.prepare(&ctx, &composition, &bindings(id, changed), &target),
        Err(Render2dExecutionError::Image {
            kind: Render2dImageError::FormatUnsupported,
            ..
        })
    ));
}

#[test]
fn mixed_images_shaped_text_and_retained_reconstruction_match() {
    let Some(ctx) = context() else { return };
    let id = Render2dResourceId::new(406).unwrap();
    let text_id = Render2dResourceId::new(407).unwrap();
    let image = item(
        id,
        vec![patch([0.0, 0.0, 2.0, 2.0], [8.0, 8.0, 40.0, 40.0])],
        Render2dAffineTransform::IDENTITY,
        0.75,
    );
    let text = shaped_entry(text_id, [8.0, 32.0], Render2dColorRgba8::WHITE);
    let image_source =
        Render2dImageResource::new(Render2dPixelExtent::new(2, 2).unwrap(), bytes()).unwrap();
    let resources = Render2dResourceBindings::new(vec![
        Render2dResourceBinding::new(id, Render2dResourceValue::ImageRgba8Srgb(image_source)),
        Render2dResourceBinding::new(
            text_id,
            Render2dResourceValue::ShapedText(shaped_resource(
                OUTLINE_FONT,
                false,
                BOX_GLYPH,
                24.0,
            )),
        ),
    ])
    .unwrap();
    let first = Render2dComposition::new(vec![image.clone(), text.clone()]).unwrap();
    let inverse = Render2dComposition::new(vec![text, image]).unwrap();
    let mut executor = Render2dExecutor::new();
    let pixels = execute(
        &ctx,
        &mut executor,
        &first,
        &resources,
        "F3C text image first",
    );
    let opposed = execute(
        &ctx,
        &mut executor,
        &inverse,
        &resources,
        "F3C text image second",
    );
    assert_ne!(pixel(&pixels, 20, 20), pixel(&opposed, 20, 20));
    executor.discard_cache();
    let rebuilt = execute(&ctx, &mut executor, &first, &resources, "F3C cache rebuilt");
    assert_eq!(pixels.as_bytes(), rebuilt.as_bytes());
}

#[test]
fn fractional_destination_resolves_edge_coverage_before_image_alpha() {
    let Some(ctx) = context() else { return };
    let id = Render2dResourceId::new(408).unwrap();
    let image = item(
        id,
        vec![patch([0.0, 0.0, 1.0, 1.0], [10.25, 20.25, 10.0, 10.0])],
        Render2dAffineTransform::IDENTITY,
        1.0,
    );
    let result = draw(&ctx, vec![image], &bindings(id, bytes()));
    // 3/4 of the four samples on each axis are covered, i.e. 9/16.
    // The independently expected output encodes premultiplied linear RGB,
    // not the product of already-encoded RGB and the coverage factor.
    let covered = 9.0 / 16.0;
    let encode = |v: u8| {
        let srgb = f64::from(v) / 255.0;
        let linear = if srgb <= 0.04045 {
            srgb / 12.92
        } else {
            ((srgb + 0.055) / 1.055).powf(2.4)
        };
        let painted = linear * covered;
        let encoded = if painted <= 0.0031308 {
            painted * 12.92
        } else {
            1.055 * painted.powf(1.0 / 2.4) - 0.055
        };
        (encoded * 255.0).round() as u8
    };
    check(
        pixel(&result, 10, 20),
        [encode(240), encode(30), encode(5), 143],
    );
    check(pixel(&result, 9, 20), [0, 0, 0, 0]);
    check(pixel(&result, 11, 21), [240, 30, 5, 255]);
}
