//! Independent Vulkan output oracles use F1 stops and physical sample positions,
//! never the tessellator's triangles or the renderer's private GPU payload.
use super::*;
use runen_render::composition_2d::*;

fn gradient_context() -> Option<GpuContext> {
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
            .with_label("F3B immutable gradients");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1")
            );
            None
        }
        Err(error) => panic!("F3B Vulkan context: {error}"),
    }
}

fn point(x: f64, y: f64) -> Render2dPoint {
    Render2dPoint::new(x, y).unwrap()
}

fn stops(values: &[(f64, [u8; 4])]) -> Render2dGradientStops {
    Render2dGradientStops::new(
        values
            .iter()
            .map(|(t, c)| {
                Render2dGradientStop::new(*t, Render2dColorRgba8::new(c[0], c[1], c[2], c[3]))
                    .unwrap()
            })
            .collect::<Vec<_>>(),
    )
    .unwrap()
}

fn item(brush: Render2dBrush, transform: Render2dAffineTransform, opacity: f64) -> Render2dEntry {
    Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 64.0, 64.0).unwrap()),
            brush,
        },
        transform,
        vec![],
        Render2dOpacity::new(opacity).unwrap(),
    ))
}

fn rendered(context: &GpuContext, items: Vec<Render2dEntry>) -> GpuReadbackBytes {
    execute(
        context,
        &mut Render2dExecutor::new(),
        &Render2dComposition::new(items).unwrap(),
        &Render2dResourceBindings::default(),
        "F3B gradient pixels",
    )
}

fn decode(byte: u8) -> f64 {
    let t = f64::from(byte) / 255.0;
    if t <= 0.04045 {
        t / 12.92
    } else {
        ((t + 0.055) / 1.055).powf(2.4)
    }
}

fn encode(linear: f64) -> u8 {
    let value = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn expected(stops: &[(f64, [u8; 4])], t: f64) -> [f64; 4] {
    let color = |raw: [u8; 4]| {
        let a = f64::from(raw[3]) / 255.0;
        [
            decode(raw[0]) * a,
            decode(raw[1]) * a,
            decode(raw[2]) * a,
            a,
        ]
    };
    let mut previous = stops[0];
    if t <= previous.0 {
        return color(previous.1);
    }
    for current in stops.iter().copied().skip(1) {
        if t <= current.0 {
            if t == current.0 {
                return color(current.1);
            }
            let fraction = (t - previous.0) / (current.0 - previous.0);
            let a = color(previous.1);
            let b = color(current.1);
            return std::array::from_fn(|i| a[i] * (1.0 - fraction) + b[i] * fraction);
        }
        previous = current;
    }
    color(previous.1)
}

fn lattice(x: u32, y: u32) -> impl Iterator<Item = [f64; 2]> {
    (0..4).flat_map(move |sy| {
        (0..4).map(move |sx| {
            [
                f64::from(x) + (f64::from(sx) + 0.5) / 4.0,
                f64::from(y) + (f64::from(sy) + 0.5) / 4.0,
            ]
        })
    })
}

fn expected_pixel(
    x: u32,
    y: u32,
    stops: &[(f64, [u8; 4])],
    t: impl Fn([f64; 2]) -> f64,
    covered: impl Fn([f64; 2]) -> bool,
    opacity: f64,
) -> [u8; 4] {
    let mut rgba = [0.0; 4];
    for position in lattice(x, y) {
        if covered(position) {
            let sampled = expected(stops, t(position).clamp(0.0, 1.0));
            for i in 0..4 {
                rgba[i] += sampled[i] * opacity / 16.0;
            }
        }
    }
    [
        encode(rgba[0]),
        encode(rgba[1]),
        encode(rgba[2]),
        (rgba[3] * 255.0).round() as u8,
    ]
}

fn check(observed: [u8; 4], expected: [u8; 4]) {
    assert!(
        observed.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 3),
        "observed {observed:?}, expected {expected:?}"
    );
}

#[test]
fn linear_premultiplied_alpha_and_duplicate_hardstop_boundary() {
    let Some(context) = gradient_context() else {
        return;
    };
    let raw = [
        (0.0, [255, 0, 0, 255]),
        (0.5, [0, 255, 0, 0]),
        (0.5, [0, 0, 255, 255]),
        (1.0, [255, 255, 255, 255]),
    ];
    let brush = Render2dBrush::Linear(
        Render2dLinearGradient::new(point(0.125, 0.0), point(64.125, 0.0), stops(&raw))
            .unwrap(),
    );
    let actual = rendered(&context, vec![item(brush, Render2dAffineTransform::IDENTITY, 0.75)]);
    for (x, y) in [(8, 20), (31, 20), (32, 20), (40, 20), (60, 20)] {
        check(
            pixel(&actual, x, y),
            expected_pixel(x, y, &raw, |p| (p[0] - 0.125) / 64.0, |_| true, 0.75),
        );
    }
}

#[test]
fn concentric_radial_gradient_tracks_inverse_sheared_affine() {
    let Some(context) = gradient_context() else {
        return;
    };
    let raw = [(0.25, [0, 255, 0, 255]), (0.75, [0, 0, 255, 128])];
    let brush = Render2dBrush::Radial(
        Render2dRadialGradient::new(point(16.0, 16.0), 12.0, stops(&raw)).unwrap(),
    );
    let transform = Render2dAffineTransform::new(1.0, 0.5, 0.25, 1.0, 4.0, 4.0).unwrap();
    let actual = rendered(&context, vec![item(brush, transform, 1.0)]);
    for (x, y) in [(24, 24), (28, 28), (32, 40), (16, 32)] {
        check(
            pixel(&actual, x, y),
            expected_pixel(
                x,
                y,
                &raw,
                |p| {
                    let px = p[0] - 4.0;
                    let py = p[1] - 4.0;
                    let lx = (px - 0.25 * py) / 0.875;
                    let ly = (-0.5 * px + py) / 0.875;
                    (lx - 16.0).hypot(ly - 16.0) / 12.0
                },
                |_| true,
                1.0,
            ),
        );
    }
}

fn segment_distance(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let dx = b[0] - a[0];
    let dy = b[1] - a[1];
    let projection = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy))
        .clamp(0.0, 1.0);
    (p[0] - (a[0] + projection * dx)).hypot(p[1] - (a[1] + projection * dy))
}

#[test]
fn translucent_crossing_gradient_stroke_uses_one_coverage_union() {
    let Some(context) = gradient_context() else {
        return;
    };
    let corners = [[8.0, 8.0], [56.0, 56.0], [8.0, 56.0], [56.0, 8.0]];
    let mut commands = vec![Render2dPathCommand::MoveTo(point(8.0, 8.0))];
    commands.extend(
        corners
            .iter()
            .skip(1)
            .map(|p| Render2dPathCommand::LineTo(point(p[0], p[1]))),
    );
    let shape =
        Render2dShape::path(Render2dPath::new(Render2dFillRule::NonZero, commands).unwrap());
    let raw = [(0.0, [255, 0, 0, 128]), (1.0, [0, 0, 255, 128])];
    let brush = Render2dBrush::Linear(
        Render2dLinearGradient::new(point(0.0, 0.0), point(64.0, 0.0), stops(&raw)).unwrap(),
    );
    let stroke = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Stroke {
            shape,
            brush,
            style: Render2dStrokeStyle::new(
                8.0,
                Render2dStrokeCap::Round,
                Render2dStrokeJoin::Round,
                4.0,
            )
            .unwrap(),
        },
        Render2dAffineTransform::IDENTITY,
        vec![],
        Render2dOpacity::new(0.75).unwrap(),
    ));
    let actual = rendered(&context, vec![stroke]);
    for y in 25..39 {
        for x in 25..39 {
            check(
                pixel(&actual, x, y),
                expected_pixel(
                    x,
                    y,
                    &raw,
                    |p| p[0] / 64.0,
                    |p| corners.windows(2).any(|w| segment_distance(p, w[0], w[1]) < 4.0),
                    0.75,
                ),
            );
        }
    }
}

#[test]
fn gradients_interleave_with_solid_and_retained_text() {
    let Some(context) = gradient_context() else {
        return;
    };
    let id = Render2dResourceId::new(777).unwrap();
    let brush = Render2dBrush::Radial(
        Render2dRadialGradient::new(
            point(24.0, 24.0),
            24.0,
            stops(&[(0.0, [255, 0, 0, 128]), (1.0, [0, 0, 255, 128])]),
        )
        .unwrap(),
    );
    let gradient = item(brush, Render2dAffineTransform::IDENTITY, 1.0);
    let solid = item(
        Render2dBrush::Solid(Render2dColorRgba8::new(0, 255, 0, 128)),
        Render2dAffineTransform::IDENTITY,
        1.0,
    );
    let text = shaped_entry(id, [8.0, 32.0], Render2dColorRgba8::WHITE);
    let resources = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let first =
        Render2dComposition::new(vec![gradient.clone(), text.clone(), solid.clone()]).unwrap();
    let opposite = Render2dComposition::new(vec![solid, text, gradient]).unwrap();
    let mut executor = Render2dExecutor::new();
    let first_pixels = execute(&context, &mut executor, &first, &resources, "gradient/text");
    let reverse_pixels = execute(&context, &mut executor, &opposite, &resources, "text/gradient");
    assert_ne!(pixel(&first_pixels, 20, 20), pixel(&reverse_pixels, 20, 20));
    executor.discard_cache();
    let reconstructed = execute(&context, &mut executor, &first, &resources, "reconstruct");
    assert_eq!(first_pixels.as_bytes(), reconstructed.as_bytes());
}

#[test]
fn unrepresentable_and_oversized_stops_fail_before_execution() {
    let Some(context) = gradient_context() else {
        return;
    };
    for (values, expected) in [
        (
            vec![
                (0.0, [255; 4]),
                (0.5, [0, 0, 0, 255]),
                (0.500000000001, [255; 4]),
                (1.0, [255; 4]),
            ],
            runen_render::execution_2d::Render2dVectorError::PrecisionLimit,
        ),
        (
            (0..258)
                .map(|i| (f64::from(i) / 257.0, [255; 4]))
                .collect(),
            runen_render::execution_2d::Render2dVectorError::ResourceLimit,
        ),
    ] {
        let brush = Render2dBrush::Linear(
            Render2dLinearGradient::new(point(0.0, 0.0), point(64.0, 0.0), stops(&values))
                .unwrap(),
        );
        let composition =
            Render2dComposition::new(vec![item(brush, Render2dAffineTransform::IDENTITY, 1.0)])
                .unwrap();
        assert!(matches!(
            Render2dExecutor::new().prepare(
                &context,
                &composition,
                &Render2dResourceBindings::default(),
                &target("reject gradient").1
            ),
            Err(Render2dExecutionError::Vector { kind, .. }) if kind == expected
        ));
    }
    let solid = Render2dBrush::solid(Render2dColorRgba8::WHITE);
    let actual = rendered(&context, vec![item(solid, Render2dAffineTransform::IDENTITY, 1.0)]);
    check(pixel(&actual, 20, 20), [255; 4]);
}
