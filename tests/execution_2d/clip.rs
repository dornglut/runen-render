//! GPU-required independent 4x4 conjunctive clip proofs at the public execution API.
use super::*;
use runen_render::composition_2d::*;
use runen_render::execution_2d::Render2dClipError;

fn context() -> Option<GpuContext> {
    let request = GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
        .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable)
        .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource)
        .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Sampled)
        .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Filterable)
        .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopyDestination)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Filterable)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("F3D conjunctive clip");
    match pollster::block_on(GpuContext::request(request)) {
        Ok(ctx) => Some(ctx),
        Err(e) if e.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(), Some("1"));
            None
        },
        Err(e) => panic!("F3D clip Vulkan context: {e}"),
    }
}

fn r(x:f64,y:f64,w:f64,h:f64)->Render2dRect {
    Render2dRect::new(x,y,w,h).unwrap()
}
fn cp(shape:Render2dShape)->Render2dClip {
    Render2dClip::new(shape,Render2dAffineTransform::IDENTITY)
}
fn cp_rect(x:f64,y:f64,w:f64,h:f64)->Render2dClip {
    cp(Render2dShape::rect(r(x,y,w,h)))
}
fn painted(
    brush:Render2dBrush,
    clips:Vec<Render2dClip>,
    transform:Render2dAffineTransform,
    opacity:f64,
)->Render2dEntry {
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape:Render2dShape::rect(r(0.0,0.0,64.0,64.0)),brush,
        },
        transform,clips,Render2dOpacity::new(opacity).unwrap(),
    ))
}
fn render(ctx:&GpuContext,entries:Vec<Render2dEntry>,bindings:&Render2dResourceBindings)->GpuReadbackBytes {
    execute(ctx,&mut Render2dExecutor::new(),
        &Render2dComposition::new(entries).unwrap(),bindings,
        "F3D clip conformance")
}
fn s_linear(byte:u8,alpha:f64)->u8 {
    let e=f64::from(byte)/255.0;
    let l=if e<=0.04045 {e/12.92}else{((e+0.055)/1.055).powf(2.4)};
    let v=l*alpha;
    let mapped=if v<=0.0031308 {12.92*v}else{1.055*v.powf(1.0/2.4)-0.055};
    (mapped*255.0).round() as u8
}
fn check(observed:[u8;4],wanted:[u8;4]){
    assert!(observed.iter().zip(wanted).all(|(a,b)|a.abs_diff(b)<=3),
        "observed {observed:?}, expected {wanted:?}");
}

#[test]
fn two_overlapping_fractional_clips_intersect_sample_bits_not_average_alphas() {
    let Some(ctx)=context() else {return};
    let paint=painted(
        Render2dBrush::solid(Render2dColorRgba8::new(240,30,5,255)),
        vec![
            cp_rect(10.0,0.0,0.5,64.0),
            cp_rect(10.25,0.0,53.75,64.0),
        ],
        // The owner transform is deliberately not applied to clips a second time.
        Render2dAffineTransform::translation(4.0,0.0).unwrap(),
        1.0,
    );
    let actual=render(&ctx,vec![paint],&Render2dResourceBindings::default());
    // Only 1 of 4 horizontal samples satisfies BOTH clips (four vertical
    // samples): 4/16, whereas the product of averaged alphas is 3/8.
    check(pixel(&actual,10,20),
        [s_linear(240,0.25),s_linear(30,0.25),s_linear(5,0.25),64]);
    check(pixel(&actual,11,20),[0;4]);
    check(pixel(&actual,14,20),[0;4]);
}

#[test]
fn evenodd_hole_and_ellipse_clip_use_structural_fill_geometry() {
    let Some(ctx)=context() else {return};
    let point=|x,y|Render2dPoint::new(x,y).unwrap();
    let mut commands=Vec::new();
    for [x0,y0,x1,y1] in [[8.0,8.0,56.0,56.0],[24.0,24.0,40.0,40.0]] {
        commands.extend([
            Render2dPathCommand::MoveTo(point(x0,y0)),
            Render2dPathCommand::LineTo(point(x1,y0)),
            Render2dPathCommand::LineTo(point(x1,y1)),
            Render2dPathCommand::LineTo(point(x0,y1)),
            Render2dPathCommand::Close,
        ]);
    }
    let path=Render2dShape::path(Render2dPath::new(Render2dFillRule::EvenOdd,commands).unwrap());
    let item=painted(Render2dBrush::solid(Render2dColorRgba8::WHITE),
        vec![cp(path),cp(Render2dShape::ellipse(r(8.0,8.0,48.0,48.0)))],
        Render2dAffineTransform::IDENTITY,1.0);
    let actual=render(&ctx,vec![item],&Render2dResourceBindings::default());
    check(pixel(&actual,32,32),[0;4]);
    check(pixel(&actual,10,10),[0;4]);
    check(pixel(&actual,32,14),[255;4]);
}

#[test]
fn image_crops_are_clipped_before_composition_without_leaking_to_sibling() {
    let Some(ctx)=context() else {return};
    let id=Render2dResourceId::new(601).unwrap();
    let extent=Render2dPixelExtent::new(2,2).unwrap();
    let source=Render2dImageResource::new(extent,vec![
        240,30,5,255, 25,210,45,255,
        30,50,190,255, 20,200,240,128
    ]).unwrap();
    let patch=Render2dImagePatch::new(
        Render2dImageSourceRect::new(0.0,0.0,2.0,2.0).unwrap(),
        r(0.0,0.0,64.0,64.0));
    let image=Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Image(Render2dImagePrimitive::new(id,extent,vec![patch]).unwrap()),
        Render2dAffineTransform::IDENTITY,
        vec![cp_rect(8.0,8.0,24.0,24.0)],
        Render2dOpacity::OPAQUE,
    ));
    let sibling=painted(Render2dBrush::solid(Render2dColorRgba8::new(0,255,0,255)),
        vec![],Render2dAffineTransform::IDENTITY,1.0);
    let bind=Render2dResourceBindings::new(vec![Render2dResourceBinding::new(
        id,Render2dResourceValue::ImageRgba8Srgb(source),
    )]).unwrap();
    let single=render(&ctx,vec![image.clone()],&bind);
    check(pixel(&single,4,4),[0;4]);
    check(pixel(&single,16,16),[240,30,5,255]);
    check(pixel(&single,40,40),[0;4]);
    let with_sibling=render(&ctx,vec![image,sibling],&bind);
    check(pixel(&with_sibling,4,4),[0,255,0,255]);
    check(pixel(&with_sibling,40,40),[0,255,0,255]);
}

#[test]
fn clipped_shaped_text_retains_msdf_coverage_and_cache_reconstruction() {
    let Some(ctx) = context() else { return };
    let id = Render2dResourceId::new(602).unwrap();
    let base = shaped_entry(id, [8.0, 32.0], Render2dColorRgba8::WHITE);
    let text = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::ShapedText(Render2dShapedTextPrimitive::new(
            id,
            Render2dPoint::new(8.0, 32.0).unwrap(),
            Render2dColorRgba8::WHITE,
        )),
        Render2dAffineTransform::IDENTITY,
        vec![cp_rect(18.0, 0.0, 46.0, 64.0)],
        Render2dOpacity::OPAQUE,
    ));
    let bind = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let unclipped = render(&ctx, vec![base], &bind);
    let composition = Render2dComposition::new(vec![text]).unwrap();
    let mut executor = Render2dExecutor::new();
    let clipped = execute(&ctx, &mut executor, &composition, &bind, "F3D clipped text");
    let mut removed = false;
    let mut retained = false;
    for y in 0..64 {
        for x in 0..64 {
            let old = pixel(&unclipped, x, y);
            let new = pixel(&clipped, x, y);
            if x < 18 && old[3] > 0 {
                removed = true;
                assert_eq!(new, [0; 4]);
            }
            if x >= 18 && old[3] > 0 && new[3] > 0 {
                retained = true;
            }
        }
    }
    assert!(
        removed && retained,
        "clip removed and preserved ordinary MSDF coverage"
    );
    executor.discard_cache();
    let regenerated = execute(
        &ctx,
        &mut executor,
        &composition,
        &bind,
        "F3D text reconstructed",
    );
    assert_eq!(clipped.as_bytes(), regenerated.as_bytes());
}

#[test]
fn disjoint_clips_produce_no_work_without_invalidating_following_paint() {
    let Some(ctx) = context() else { return };
    let clips = vec![cp_rect(0.0, 0.0, 8.0, 8.0), cp_rect(20.0, 20.0, 8.0, 8.0)];
    let item = painted(
        Render2dBrush::solid(Render2dColorRgba8::WHITE),
        clips,
        Render2dAffineTransform::IDENTITY,
        1.0,
    );
    let composition = Render2dComposition::new(vec![item.clone()]).unwrap();
    let (_, target) = target("F3D disjoint clip target");
    let prepared = Render2dExecutor::new()
        .prepare(
            &ctx,
            &composition,
            &Render2dResourceBindings::default(),
            &target,
        )
        .unwrap();
    assert!(!prepared.has_render_work());
    let sibling = painted(
        Render2dBrush::solid(Render2dColorRgba8::new(0, 0, 255, 255)),
        vec![],
        Render2dAffineTransform::IDENTITY,
        1.0,
    );
    let image = render(
        &ctx,
        vec![item, sibling],
        &Render2dResourceBindings::default(),
    );
    check(pixel(&image, 4, 4), [0, 0, 255, 255]);
    check(pixel(&image, 40, 40), [0, 0, 255, 255]);
}

#[test]
fn unadmitted_clip_format_has_typed_owner_level_failure() {
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
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("F3D missing clip upload role");
    let ctx = match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(ctx) => ctx,
        Err(e) if e.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1")
            );
            return;
        }
        Err(e) => panic!("F3D capability context: {e}"),
    };
    let composition = Render2dComposition::new(vec![painted(
        Render2dBrush::solid(Render2dColorRgba8::WHITE),
        vec![cp_rect(2.0, 2.0, 32.0, 32.0)],
        Render2dAffineTransform::IDENTITY,
        1.0,
    )])
    .unwrap();
    let (_, target) = target("F3D missing clip sampling role");
    assert!(matches!(
        Render2dExecutor::new().prepare(
            &ctx,
            &composition,
            &Render2dResourceBindings::default(),
            &target
        ),
        Err(Render2dExecutionError::Clip {
            root_index: 0,
            kind: Render2dClipError::FormatUnsupported
        })
    ));
}

fn encode_linear(x: f64) -> u8 {
    let v = if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[test]
fn clipped_gradient_interpolates_at_the_same_covered_sample_positions() {
    let Some(ctx) = context() else { return };
    let stops = Render2dGradientStops::new(vec![
        Render2dGradientStop::new(0.0, Render2dColorRgba8::new(255, 0, 0, 255)).unwrap(),
        Render2dGradientStop::new(1.0, Render2dColorRgba8::new(0, 0, 255, 255)).unwrap(),
    ])
    .unwrap();
    let brush = Render2dBrush::Linear(
        Render2dLinearGradient::new(
            Render2dPoint::new(0.0, 0.0).unwrap(),
            Render2dPoint::new(64.0, 0.0).unwrap(),
            stops,
        )
        .unwrap(),
    );
    let item = painted(
        brush,
        vec![
            cp_rect(10.0, 0.0, 0.5, 64.0),
            cp_rect(10.25, 0.0, 53.75, 64.0),
        ],
        Render2dAffineTransform::IDENTITY,
        0.75,
    );
    let actual = render(&ctx, vec![item], &Render2dResourceBindings::default());
    // Only x=10.375 survives all clips, for each of four vertical samples.
    let t = 10.375 / 64.0;
    let coverage = 0.25 * 0.75;
    check(
        pixel(&actual, 10, 20),
        [
            encode_linear((1.0 - t) * coverage),
            0,
            encode_linear(t * coverage),
            48,
        ],
    );
}

#[test]
fn clipped_translucent_self_intersecting_stroke_is_shaded_only_once() {
    let Some(ctx) = context() else { return };
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    let shape = Render2dShape::path(
        Render2dPath::new(
            Render2dFillRule::NonZero,
            vec![
                Render2dPathCommand::MoveTo(p(8.0, 8.0)),
                Render2dPathCommand::LineTo(p(56.0, 56.0)),
                Render2dPathCommand::LineTo(p(8.0, 56.0)),
                Render2dPathCommand::LineTo(p(56.0, 8.0)),
            ],
        )
        .unwrap(),
    );
    let item = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Stroke {
            shape,
            brush: Render2dBrush::solid(Render2dColorRgba8::new(255, 0, 0, 128)),
            style: Render2dStrokeStyle::new(
                8.0,
                Render2dStrokeCap::Round,
                Render2dStrokeJoin::Round,
                4.0,
            )
            .unwrap(),
        },
        Render2dAffineTransform::IDENTITY,
        vec![cp_rect(32.25, 0.0, 0.25, 64.0)],
        Render2dOpacity::new(0.5).unwrap(),
    ));
    let actual = render(&ctx, vec![item], &Render2dResourceBindings::default());
    let alpha = 0.25 * 0.5 * (128.0 / 255.0);
    check(
        pixel(&actual, 32, 32),
        [encode_linear(alpha), 0, 0, (alpha * 255.0).round() as u8],
    );
}
