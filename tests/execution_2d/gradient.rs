//! Independent, GPU-required gradient output oracles: no tessellator, WGSL, or GPU
//! resource-layout internals are used to compute expected colors.
use super::*;
use runen_render::composition_2d::*;

fn gradient_context() -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable)
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled)
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("F3B immutable gradients");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(), Some("1"));
            None
        }
        Err(error) => panic!("F3B Vulkan context: {error}"),
    }
}

fn position(x: f64, y: f64) -> Render2dPoint {
    Render2dPoint::new(x, y).unwrap()
}
fn stops(values: &[(f64, [u8; 4])]) -> Render2dGradientStops {
    Render2dGradientStops::new(values.iter().map(|(t,c)| {
        Render2dGradientStop::new(*t, Render2dColorRgba8::new(c[0],c[1],c[2],c[3])).unwrap()
    }).collect::<Vec<_>>()).unwrap()
}
fn gradient_item(
    brush: Render2dBrush,
    transform: Render2dAffineTransform,
    opacity: f64,
) -> Render2dEntry {
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(Render2dRect::new(0.0,0.0,64.0,64.0).unwrap()),
            brush,
        },
        transform, vec![], Render2dOpacity::new(opacity).unwrap()
    ))
}
fn rendered(context: &GpuContext, items: Vec<Render2dEntry>) -> GpuReadbackBytes {
    execute(context, &mut Render2dExecutor::new(),
        &Render2dComposition::new(items).unwrap(),
        &Render2dResourceBindings::default(), "F3B gradient pixels")
}
fn decode(byte: u8) -> f64 {
    let t=f64::from(byte)/255.0;
    if t <= 0.04045 {t/12.92} else {((t+0.055)/1.055).powf(2.4)}
}
fn encode(linear: f64) -> u8 {
    let v=if linear<=0.0031308 {linear*12.92} else {1.055*linear.powf(1.0/2.4)-0.055};
    (v.clamp(0.0,1.0)*255.0).round() as u8
}
fn expected(stop_list: &[(f64,[u8;4])], t: f64) -> [f64;4] {
    let color=|raw:[u8;4]| {
        let a=f64::from(raw[3])/255.0;
        [decode(raw[0])*a, decode(raw[1])*a, decode(raw[2])*a, a]
    };
    let mut previous=stop_list[0];
    if t<=previous.0 {return color(previous.1)}
    for current in stop_list.iter().copied().skip(1) {
        if t<=current.0 {
            if t==current.0 {return color(current.1)}
            let f=(t-previous.0)/(current.0-previous.0);
            let a=color(previous.1);
            let b=color(current.1);
            return std::array::from_fn(|i| a[i]*(1.0-f)+b[i]*f);
        }
        previous=current;
    }
    color(previous.1)
}
fn expected_pixel(samples: impl Iterator<Item=([f64;2],bool)>, stops: &[(f64,[u8;4])],
   position_to_t: impl Fn([f64;2])->f64, opacity: f64) -> [u8;4] {
    let mut rgba=[0.0;4];
    for (p,covered) in samples {
        if covered {
            let c=expected(stops,position_to_t(p).clamp(0.0,1.0));
            for i in 0..4 {rgba[i]+=c[i]*opacity/16.0}
        }
    }
    [encode(rgba[0]),encode(rgba[1]),encode(rgba[2]),(rgba[3]*255.0).round() as u8]
}
fn lattice(x:u32,y:u32) -> impl Iterator<Item=([f64;2],bool)> {
    (0..4).flat_map(move |sy| (0..4).map(move |sx| {
       ([f64::from(x)+(f64::from(sx)+0.5)/4.0,
         f64::from(y)+(f64::from(sy)+0.5)/4.0],true)
    }))
}
fn check(actual:[u8;4],reference:[u8;4],label:&str){
    assert!(actual.iter().zip(reference).all(|(a,b)|a.abs_diff(b)<=3),
        "{label}: observed {actual:?}, expected {reference:?}");
}

#[test]
fn linear_premultiplied_alpha_and_hardstop_boundary() {
    let Some(ctx)=gradient_context() else {return};
    let raw=[
       (0.0,[255,0,0,255]),
       (0.5,[0,255,0,0]),
       (0.5,[0,0,255,255]),
       (1.0,[255,255,255,255]),
    ];
    let b=Render2dBrush::Linear(Render2dLinearGradient::new(
        position(0.125,0.0),position(64.125,0.0),stops(&raw)).unwrap());
    let output=rendered(&ctx,vec![gradient_item(b,Render2dAffineTransform::IDENTITY,0.75)]);
    for (x,y) in [(8,20),(31,20),(32,20),(40,20),(60,20)] {
        let reference=expected_pixel(lattice(x,y),&raw,
           |p|(p[0]-0.125)/64.0,0.75);
        check(pixel(&output,x,y),reference,&format!("linear pixel {x}"));
    }
}

#[test]
fn radial_gradient_affine_inverse_and_outside_stop_extension() {
    let Some(ctx)=gradient_context() else {return};
    let raw=[(0.25,[0,255,0,255]),(0.75,[0,0,255,128])];
    let gradient=Render2dBrush::Radial(Render2dRadialGradient::new(
        position(16.0,16.0),12.0,stops(&raw)).unwrap());
    let xform=Render2dAffineTransform::new(1.0,0.5,0.25,1.0,4.0,4.0).unwrap();
    let output=rendered(&ctx,vec![gradient_item(gradient,xform,1.0)]);
    for (x,y) in [(24,24),(28,28),(32,40),(16,32)] {
        let reference=expected_pixel(lattice(x,y),&raw,|p|{
            let px=p[0]-4.0;
            let py=p[1]-4.0;
            let lx=(px-0.25*py)/0.875;
            let ly=(-0.5*px+py)/0.875;
            ((lx-16.0).hypot(ly-16.0))/12.0
        },1.0);
        check(pixel(&output,x,y),reference,&format!("radial affine {x},{y}"));
    }
}

#[test]
fn gradient_limit_is_typed_and_solid_path_remains_admitted() {
    let Some(ctx)=gradient_context() else {return};
    let raw=(0..258).map(|i|(f64::from(i)/257.0,[255,255,255,255])).collect::<Vec<_>>();
    let gradient=Render2dBrush::Linear(Render2dLinearGradient::new(
        position(0.0,0.0),position(64.0,0.0),stops(&raw)).unwrap());
    let result=Render2dExecutor::new().prepare(&ctx,
       &Render2dComposition::new(vec![gradient_item(
           gradient,Render2dAffineTransform::IDENTITY,1.0)]).unwrap(),
       &Render2dResourceBindings::default(),&target("bounded F3B gradient").1);
    assert!(matches!(result,Err(Render2dExecutionError::Vector {
       kind:runen_render::execution_2d::Render2dVectorError::ResourceLimit,..})));
    let solid=Render2dBrush::solid(Render2dColorRgba8::WHITE);
    let output=rendered(&ctx,vec![gradient_item(solid,Render2dAffineTransform::IDENTITY,1.0)]);
    check(pixel(&output,20,20),[255;4],"F3A solid regression");
}

fn distance_to_segment(point: [f64; 2], from: [f64; 2], to: [f64; 2]) -> f64 {
    let delta=[to[0]-from[0],to[1]-from[1]];
    let len_sq=delta[0]*delta[0]+delta[1]*delta[1];
    let projection=(((point[0]-from[0])*delta[0]+(point[1]-from[1])*delta[1])
        /len_sq).clamp(0.0,1.0);
    (point[0]-(from[0]+projection*delta[0]))
        .hypot(point[1]-(from[1]+projection*delta[1]))
}

#[test]
fn crossing_translucent_gradient_stroke_is_covered_and_composited_only_once() {
    let Some(ctx)=gradient_context() else {return};
    let corners=[[8.0,8.0],[56.0,56.0],[8.0,56.0],[56.0,8.0]];
    let mut commands=vec![Render2dPathCommand::MoveTo(position(8.0,8.0))];
    commands.extend(corners.iter().skip(1).map(|p|
        Render2dPathCommand::LineTo(position(p[0],p[1]))));
    let shape=Render2dShape::path(Render2dPath::new(
        Render2dFillRule::NonZero,commands).unwrap());
    let raw=[(0.0,[255,0,0,128]),(1.0,[0,0,255,128])];
    let gradient=Render2dBrush::Linear(Render2dLinearGradient::new(
        position(0.0,0.0),position(64.0,0.0),stops(&raw)).unwrap());
    let entry=Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Stroke {shape,brush:gradient,
            style:Render2dStrokeStyle::new(8.0,Render2dStrokeCap::Round,
                Render2dStrokeJoin::Round,4.0).unwrap()},
        Render2dAffineTransform::IDENTITY,vec![],
        Render2dOpacity::new(0.75).unwrap()));
    let output=rendered(&ctx,vec![entry]);
    for y in 25..39 {
        for x in 25..39 {
            let samples=lattice(x,y).map(|(p,_)|{
                let covered=corners.windows(2).any(|w|
                    distance_to_segment(p,w[0],w[1])<4.0);
                (p,covered)
            });
            let oracle=expected_pixel(samples,&raw,|p|p[0]/64.0,0.75);
            check(pixel(&output,x,y),oracle,&format!("crossing {x},{y}"));
        }
    }
}

#[test]
fn gradient_interleaves_with_solid_vectors_and_retained_text() {
    let Some(ctx)=gradient_context() else {return};
    let id=Render2dResourceId::new(777).unwrap();
    let gradient=Render2dBrush::Radial(Render2dRadialGradient::new(
        position(24.0,24.0),24.0,
        stops(&[(0.0,[255,0,0,128]),(1.0,[0,0,255,128])])).unwrap());
    let painted=gradient_item(gradient,Render2dAffineTransform::IDENTITY,1.0);
    let solid=gradient_item(
        Render2dBrush::Solid(Render2dColorRgba8::new(0,255,0,128)),
        Render2dAffineTransform::IDENTITY,1.0);
    let text=shaped_entry(id,[8.0,32.0],Render2dColorRgba8::WHITE);
    let resources=bindings(id,shaped_resource(OUTLINE_FONT,false,BOX_GLYPH,24.0));
    let original=Render2dComposition::new(vec![painted.clone(),text.clone(),solid.clone()]).unwrap();
    let reversed=Render2dComposition::new(vec![solid,text,painted]).unwrap();
    let mut executor=Render2dExecutor::new();
    let pixels=execute(&ctx,&mut executor,&original,&resources,"gradient text mixed");
    let opposite=execute(&ctx,&mut executor,&reversed,&resources,"gradient text reversed");
    assert_ne!(pixel(&pixels,20,20),pixel(&opposite,20,20));
    executor.discard_cache();
    let same=execute(&ctx,&mut executor,&original,&resources,"gradient text rebuilt");
    assert_eq!(pixels.as_bytes(),same.as_bytes());
}

#[test]
fn gradient_unrepresentable_adjacent_stops_fail_closed() {
    let Some(ctx)=gradient_context() else {return};
    let raw=[(0.0,[255;4]),(0.5,[0,0,0,255]),
       (0.500000000001,[255;4]),(1.0,[255;4])];
    let gradient=Render2dBrush::Linear(Render2dLinearGradient::new(
        position(0.0,0.0),position(64.0,0.0),stops(&raw)).unwrap());
    let composition=Render2dComposition::new(vec![gradient_item(
        gradient,Render2dAffineTransform::IDENTITY,1.0)]).unwrap();
    assert!(matches!(
        Render2dExecutor::new().prepare(&ctx,&composition,
            &Render2dResourceBindings::default(),&target("hardstop precision").1),
        Err(Render2dExecutionError::Vector {
            kind:runen_render::execution_2d::Render2dVectorError::PrecisionLimit,..})
    ));
}
