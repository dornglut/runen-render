//! Independent output assertions use semantic geometry and known binary sample
//! positions, never the tessellator's vertices, indices, bounds or mask payload.
use super::*;
use runen_render::composition_2d::*;
use runen_render::execution_2d::Render2dVectorError;

fn context() -> Option<GpuContext> {
    context_with_text_roles(true)
}
fn context_with_text_roles(text: bool) -> Option<GpuContext> {
    let mut descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
            .require_format_role(
                GpuTextureFormat::Rgba16Float,
                GpuFormatRole::ColorAttachment,
            )
            .require_format_role(GpuTextureFormat::Rgba16Float, GpuFormatRole::Blendable)
            .require_format_role(GpuTextureFormat::Rgba16Float, GpuFormatRole::Sampled)
            .require_format_role(
                GpuTextureFormat::Rgba8UnormSrgb,
                GpuFormatRole::ColorAttachment,
            )
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable)
            .require_format_role(GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("F3A vector output proofs");
    if text {
        descriptor = descriptor
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Filterable)
            .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination);
    }
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => {
            eprintln!(
                "F3A adapter: {:?} {:?} {:?}",
                context.adapter_facts().backend(),
                context.adapter_facts().diagnostic_name(),
                context.adapter_facts().software()
            );
            Some(context)
        }
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "F3A requires a real Vulkan adapter"
            );
            None
        }
        Err(error) => panic!("F3A context: {error}"),
    }
}
fn point(x: f64, y: f64) -> Render2dPoint {
    Render2dPoint::new(x, y).unwrap()
}
fn rect(x: f64, y: f64, w: f64, h: f64) -> Render2dShape {
    Render2dShape::rect(Render2dRect::new(x, y, w, h).unwrap())
}
fn item(
    primitive: Render2dPrimitive,
    transform: Render2dAffineTransform,
    opacity: f64,
) -> Render2dEntry {
    Render2dEntry::item(Render2dItem::new(
        primitive,
        transform,
        vec![],
        Render2dOpacity::new(opacity).unwrap(),
    ))
}
fn fill(shape: Render2dShape, color: Render2dColorRgba8) -> Render2dEntry {
    item(
        Render2dPrimitive::Fill {
            shape,
            brush: Render2dBrush::solid(color),
        },
        Render2dAffineTransform::IDENTITY,
        1.0,
    )
}
fn stroke(
    shape: Render2dShape,
    cap: Render2dStrokeCap,
    join: Render2dStrokeJoin,
    miter: f64,
    color: Render2dColorRgba8,
) -> Render2dEntry {
    item(
        Render2dPrimitive::Stroke {
            shape,
            brush: Render2dBrush::solid(color),
            style: Render2dStrokeStyle::new(8.0, cap, join, miter).unwrap(),
        },
        Render2dAffineTransform::IDENTITY,
        1.0,
    )
}
fn path(points: &[[f64; 2]], closed: bool, rule: Render2dFillRule) -> Render2dShape {
    let mut commands = vec![Render2dPathCommand::MoveTo(point(
        points[0][0],
        points[0][1],
    ))];
    commands.extend(
        points[1..]
            .iter()
            .map(|p| Render2dPathCommand::LineTo(point(p[0], p[1]))),
    );
    if closed {
        commands.push(Render2dPathCommand::Close);
    }
    Render2dShape::path(Render2dPath::new(rule, commands).unwrap())
}
fn render(context: &GpuContext, entries: Vec<Render2dEntry>) -> GpuReadbackBytes {
    execute(
        context,
        &mut Render2dExecutor::new(),
        &Render2dComposition::new(entries).unwrap(),
        &Render2dResourceBindings::default(),
        "F3A pixels",
    )
}
fn assert_pixel(bytes: &GpuReadbackBytes, x: u32, y: u32, expected: [u8; 4]) {
    let actual = pixel(bytes, x, y);
    assert!(
        actual
            .into_iter()
            .zip(expected)
            .all(|(a, e)| a.abs_diff(e) <= 3),
        "pixel ({x},{y}): {actual:?}, expected {expected:?}"
    );
}

#[test]
fn solid_shapes_and_curved_paths_produce_independent_output() {
    let Some(context) = context() else {
        return;
    };
    let white = Render2dColorRgba8::WHITE;
    let round = Render2dShape::rounded_rect(
        Render2dRect::new(4.0, 4.0, 24.0, 24.0).unwrap(),
        Render2dCornerRadii::new(10.0, 4.0, 8.0, 2.0).unwrap(),
    );
    let ellipse = Render2dShape::ellipse(Render2dRect::new(32.0, 4.0, 28.0, 20.0).unwrap());
    let curve = Render2dShape::path(
        Render2dPath::new(
            Render2dFillRule::NonZero,
            vec![
                Render2dPathCommand::MoveTo(point(4.0, 40.0)),
                Render2dPathCommand::QuadraticTo {
                    control: point(16.0, 20.0),
                    to: point(28.0, 40.0),
                },
                Render2dPathCommand::CubicTo {
                    control1: point(28.0, 60.0),
                    control2: point(4.0, 60.0),
                    to: point(4.0, 40.0),
                },
                Render2dPathCommand::Close,
            ],
        )
        .unwrap(),
    );
    let bytes = render(
        &context,
        vec![
            fill(round, white),
            fill(ellipse, white),
            fill(curve, white),
            fill(rect(36.0, 36.0, 20.0, 20.0), white),
        ],
    );
    for (x, y) in [(16, 16), (46, 14), (16, 42), (46, 46)] {
        assert_pixel(&bytes, x, y, [255; 4]);
    }
    for (x, y) in [(4, 4), (32, 4), (4, 30), (0, 0), (63, 63), (35, 46)] {
        assert_pixel(&bytes, x, y, [0; 4]);
    }
    assert!(
        bytes
            .as_bytes()
            .as_chunks::<4>()
            .0
            .iter()
            .any(|p| p[3] > 0 && p[3] < 255),
        "curves require antialiased edge coverage"
    );
}

#[test]
fn self_intersections_and_open_contours_obey_both_fill_rules() {
    let Some(context) = context() else {
        return;
    };
    // True crossing pentagram: center winding is two, outer tips winding is one.
    let pentagram = [
        [32.0, 4.0],
        [48.0, 56.0],
        [5.0, 24.0],
        [59.0, 24.0],
        [16.0, 56.0],
    ];
    let even = render(
        &context,
        vec![fill(
            path(&pentagram, false, Render2dFillRule::EvenOdd),
            Render2dColorRgba8::WHITE,
        )],
    );
    let nonzero = render(
        &context,
        vec![fill(
            path(&pentagram, true, Render2dFillRule::NonZero),
            Render2dColorRgba8::WHITE,
        )],
    );
    assert_pixel(&even, 32, 32, [0; 4]);
    assert_pixel(&nonzero, 32, 32, [255; 4]);
    for bytes in [&even, &nonzero] {
        assert_pixel(bytes, 32, 12, [255; 4]);
        assert_pixel(bytes, 1, 1, [0; 4]);
    }
    // Independent winding-number oracle over the authored straight contours, away
    // from edges. This detects a fill-rule switch or silently dropped crossings.
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let p = [f64::from(x) + 0.5, f64::from(y) + 0.5];
            if pentagram
                .iter()
                .enumerate()
                .any(|(i, a)| distance_to_segment(p, *a, pentagram[(i + 1) % 5]) < 1.0)
            {
                continue;
            }
            let mut winding = 0_i32;
            for i in 0..5 {
                let a = pentagram[i];
                let b = pentagram[(i + 1) % 5];
                let cross = (b[0] - a[0]) * (p[1] - a[1]) - (p[0] - a[0]) * (b[1] - a[1]);
                if a[1] <= p[1] && b[1] > p[1] && cross > 0.0 {
                    winding += 1;
                }
                if a[1] > p[1] && b[1] <= p[1] && cross < 0.0 {
                    winding -= 1;
                }
            }
            assert_pixel(
                &even,
                x,
                y,
                if winding.abs() % 2 == 1 {
                    [255; 4]
                } else {
                    [0; 4]
                },
            );
            assert_pixel(&nonzero, x, y, if winding != 0 { [255; 4] } else { [0; 4] });
        }
    }
}
fn distance_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let d = [b[0] - a[0], b[1] - a[1]];
    let len = d[0] * d[0] + d[1] * d[1];
    let t = if len == 0.0 {
        0.0
    } else {
        ((p[0] - a[0]) * d[0] + (p[1] - a[1]) * d[1]) / len
    }
    .clamp(0.0, 1.0);
    (p[0] - a[0] - t * d[0]).hypot(p[1] - a[1] - t * d[1])
}

#[test]
fn stroke_caps_joins_and_miter_fallback_are_visible() {
    let Some(context) = context() else {
        return;
    };
    let line = path(
        &[[16.0, 32.0], [48.0, 32.0]],
        false,
        Render2dFillRule::NonZero,
    );
    let butt = render(
        &context,
        vec![stroke(
            line.clone(),
            Render2dStrokeCap::Butt,
            Render2dStrokeJoin::Miter,
            4.0,
            Render2dColorRgba8::WHITE,
        )],
    );
    let round = render(
        &context,
        vec![stroke(
            line.clone(),
            Render2dStrokeCap::Round,
            Render2dStrokeJoin::Miter,
            4.0,
            Render2dColorRgba8::WHITE,
        )],
    );
    let square = render(
        &context,
        vec![stroke(
            line,
            Render2dStrokeCap::Square,
            Render2dStrokeJoin::Miter,
            4.0,
            Render2dColorRgba8::WHITE,
        )],
    );
    assert_pixel(&butt, 13, 32, [0; 4]);
    assert_pixel(&round, 13, 32, [255; 4]);
    assert_pixel(&square, 13, 32, [255; 4]);
    assert_pixel(&round, 12, 28, [0; 4]);
    assert_pixel(&square, 12, 28, [255; 4]);
    let corner = path(
        &[[8.0, 48.0], [32.0, 16.0], [56.0, 48.0]],
        false,
        Render2dFillRule::NonZero,
    );
    let render_join = |join, miter| {
        render(
            &context,
            vec![stroke(
                corner.clone(),
                Render2dStrokeCap::Butt,
                join,
                miter,
                Render2dColorRgba8::WHITE,
            )],
        )
    };
    let miter = render_join(Render2dStrokeJoin::Miter, 4.0);
    let bevel = render_join(Render2dStrokeJoin::Bevel, 4.0);
    let rounded = render_join(Render2dStrokeJoin::Round, 4.0);
    let pressure = path(
        &[[16.0, 56.0], [32.0, 24.0], [36.0, 56.0]],
        false,
        Render2dFillRule::NonZero,
    );
    let fallback = render(
        &context,
        vec![stroke(
            pressure.clone(),
            Render2dStrokeCap::Butt,
            Render2dStrokeJoin::Miter,
            1.0,
            Render2dColorRgba8::WHITE,
        )],
    );
    let bevel_pressure = render(
        &context,
        vec![stroke(
            pressure.clone(),
            Render2dStrokeCap::Butt,
            Render2dStrokeJoin::Bevel,
            4.0,
            Render2dColorRgba8::WHITE,
        )],
    );
    let full = render(
        &context,
        vec![stroke(
            pressure,
            Render2dStrokeCap::Butt,
            Render2dStrokeJoin::Miter,
            4.0,
            Render2dColorRgba8::WHITE,
        )],
    );
    assert_pixel(&full, 32, 16, [255; 4]);
    assert_pixel(&fallback, 32, 16, [0; 4]);
    assert_pixel(&miter, 32, 11, [255; 4]);
    assert_pixel(&bevel, 32, 11, [0; 4]);
    assert_pixel(&rounded, 32, 13, [255; 4]);
    // Bevel edge y=13.6 covers two of the four sample rows.
    assert_pixel(&bevel, 32, 13, [188, 188, 188, 128]);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            assert_eq!(
                pixel(&fallback, x, y),
                pixel(&bevel_pressure, x, y),
                "miter fallback ({x},{y})"
            );
        }
    }
}

#[test]
fn translucent_crossing_stroke_applies_alpha_once_per_union_sample() {
    let Some(context) = context() else {
        return;
    };
    let points = [[8.0, 8.0], [56.0, 56.0], [8.0, 56.0], [56.0, 8.0]];
    let shape = path(&points, false, Render2dFillRule::NonZero);
    let source = stroke(
        shape,
        Render2dStrokeCap::Round,
        Render2dStrokeJoin::Round,
        4.0,
        Render2dColorRgba8::new(255, 0, 0, 128),
    );
    let bytes = render(&context, vec![source]);
    assert_pixel(&bytes, 32, 32, [188, 0, 0, 128]);
    assert_pixel(&bytes, 20, 20, [188, 0, 0, 128]);
    // Independent distance-to-segment union, exact regular sample lattice. Avoid
    // endpoint joins, whose oracle is tested separately. No triangles are consumed.
    for y in 24..40 {
        for x in 24..40 {
            let count = (0..4)
                .flat_map(|sy| (0..4).map(move |sx| (sx, sy)))
                .filter(|(sx, sy)| {
                    let p = [
                        f64::from(x) + (f64::from(*sx) + 0.5) / 4.0,
                        f64::from(y) + (f64::from(*sy) + 0.5) / 4.0,
                    ];
                    points
                        .windows(2)
                        .any(|s| distance_to_segment(p, s[0], s[1]) < 4.0)
                })
                .count();
            let alpha = (128.0 / 255.0) * count as f64 / 16.0;
            assert_pixel(
                &bytes,
                x,
                y,
                [encode(alpha), 0, 0, (alpha * 255.0).round() as u8],
            );
        }
    }
}
fn encode(linear: f64) -> u8 {
    let encoded = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[test]
fn vector_opacity_and_mixed_text_preserve_exact_painter_order_and_reconstruction() {
    let Some(context) = context() else {
        return;
    };
    let id = Render2dResourceId::new(300).unwrap();
    let text = shaped_entry(id, [8.0, 32.0], Render2dColorRgba8::new(0, 255, 0, 128));
    let red = fill(
        rect(8.0, 8.0, 30.0, 30.0),
        Render2dColorRgba8::new(255, 0, 0, 128),
    );
    let blue = item(
        Render2dPrimitive::Fill {
            shape: rect(8.0, 8.0, 30.0, 30.0),
            brush: Render2dBrush::solid(Render2dColorRgba8::new(0, 0, 255, 128)),
        },
        Render2dAffineTransform::IDENTITY,
        0.5,
    );
    let bindings = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let composition =
        Render2dComposition::new(vec![red.clone(), text.clone(), blue.clone()]).unwrap();
    let mut executor = Render2dExecutor::new();
    let first = execute(
        &context,
        &mut executor,
        &composition,
        &bindings,
        "mixed vector text",
    );
    let alpha = 128.0 / 255.0;
    let blue_alpha = alpha * 0.5;
    let expected = [
        encode(alpha * (1.0 - alpha) * (1.0 - blue_alpha)),
        encode(alpha * (1.0 - blue_alpha)),
        encode(blue_alpha),
        ((1.0 - (1.0 - alpha).powi(2) * (1.0 - blue_alpha)) * 255.0).round() as u8,
    ];
    assert_pixel(&first, 20, 20, expected);
    let reverse = Render2dComposition::new(vec![blue, text, red]).unwrap();
    let reversed = execute(
        &context,
        &mut executor,
        &reverse,
        &bindings,
        "reversed mixed vector text",
    );
    assert_ne!(pixel(&first, 20, 20), pixel(&reversed, 20, 20));
    executor.discard_cache();
    let rebuilt = execute(
        &context,
        &mut executor,
        &composition,
        &bindings,
        "rebuilt mixed vector text",
    );
    assert_eq!(first.as_bytes(), rebuilt.as_bytes());
    let Some(fresh) = self::context() else {
        panic!("fresh Vulkan context");
    };
    let rebuilt = execute(
        &fresh,
        &mut Render2dExecutor::new(),
        &composition,
        &bindings,
        "fresh mixed vector text",
    );
    assert_eq!(first.as_bytes(), rebuilt.as_bytes());
}

#[test]
fn vector_affine_transform_and_fractional_canvas_preserve_coverage() {
    let Some(context) = context() else {
        return;
    };
    // 90-degree rotation, non-uniform scale and fractional translation: local
    // [0,0]..[8,12] maps to [8.25,12.5]..[32.25,20.5].
    let entry = item(
        Render2dPrimitive::Fill {
            shape: rect(0.0, 0.0, 8.0, 12.0),
            brush: Render2dBrush::solid(Render2dColorRgba8::WHITE),
        },
        Render2dAffineTransform::new(0.0, 1.0, -2.0, 0.0, 32.25, 12.5).unwrap(),
        1.0,
    );
    let bytes = render(&context, vec![entry]);
    assert_pixel(&bytes, 12, 16, [255; 4]);
    assert_pixel(&bytes, 7, 16, [0; 4]);
    assert_pixel(&bytes, 8, 16, [225, 225, 225, 191]); // 12 of 16 samples
    assert_pixel(&bytes, 12, 12, [188, 188, 188, 128]); // 8 of 16 samples
    let (texture, unit) = target("vector fractional canvas");
    let target = Render2dTarget::new(unit.view().clone(), 50.6, 51.2, 1.25).unwrap();
    let composition = Render2dComposition::new(vec![fill(
        rect(0.0, 0.0, 100.0, 100.0),
        Render2dColorRgba8::WHITE,
    )])
    .unwrap();
    let bytes = execute_inline(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        &texture,
        &target,
        None,
    );
    assert_pixel(&bytes, 62, 25, [255; 4]);
    assert_pixel(&bytes, 63, 25, [137, 137, 137, 64]); // canvas right edge 63.25
}

#[test]
fn vector_nonpainting_and_capability_failures_are_structural() {
    let Some(context) = context() else {
        return;
    };
    let (_, target) = target("vector no-work");
    let mut executor = Render2dExecutor::new();
    let degenerate = Render2dShape::path(
        Render2dPath::new(
            Render2dFillRule::NonZero,
            vec![Render2dPathCommand::MoveTo(point(8.0, 8.0))],
        )
        .unwrap(),
    );
    for entry in [
        fill(rect(0.0, 0.0, 0.0, 10.0), Render2dColorRgba8::WHITE),
        fill(degenerate, Render2dColorRgba8::WHITE),
        fill(rect(100.0, 100.0, 10.0, 10.0), Render2dColorRgba8::WHITE),
        fill(rect(8.0, 8.0, 10.0, 10.0), Render2dColorRgba8::TRANSPARENT),
    ] {
        let composition = Render2dComposition::new(vec![entry]).unwrap();
        let prepared = executor
            .prepare(
                &context,
                &composition,
                &Render2dResourceBindings::default(),
                &target,
            )
            .unwrap();
        assert!(!prepared.has_render_work());
        let (fragment, token) = prepared.into_fragment(&work_binding("no-work")).unwrap();
        assert!(fragment.outputs().is_empty());
        assert!(token.is_none());
    }
    // The coherent F3E target role contract also admits vectors in contexts
    // originally used for F2 text: adding an identity group never changes law.
    let Some(unified) = f2_context() else {
        return;
    };
    let composition = Render2dComposition::new(vec![fill(
        rect(8.0, 8.0, 10.0, 10.0),
        Render2dColorRgba8::WHITE,
    )])
    .unwrap();
    assert!(
        executor
            .prepare(
                &unified,
                &composition,
                &Render2dResourceBindings::default(),
                &target
            )
            .expect("one admitted correlated compositor also covers vector roots")
            .has_render_work()
    );
    let composition = Render2dComposition::new(vec![fill(
        rect(1e30, 0.0, 1e20, 10.0),
        Render2dColorRgba8::WHITE,
    )])
    .unwrap();
    assert!(matches!(
        executor.prepare(
            &context,
            &composition,
            &Render2dResourceBindings::default(),
            &target
        ),
        Err(Render2dExecutionError::Vector {
            kind: Render2dVectorError::PrecisionLimit,
            ..
        })
    ));
}

#[test]
fn late_vector_failure_preserves_target_and_unobserved_text_identity() {
    let Some(context) = context() else {
        return;
    };
    let id = Render2dResourceId::new(500).unwrap();
    let bound = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 24.0));
    let composition = Render2dComposition::new(vec![
        shaped_entry(id, [8.0, 32.0], Render2dColorRgba8::WHITE),
        fill(rect(1e30, 0.0, 1e20, 10.0), Render2dColorRgba8::WHITE),
    ])
    .unwrap();
    let (texture, target) = target("transactional vector rejection");
    let mut executor = Render2dExecutor::new();
    assert!(matches!(
        executor.prepare(&context, &composition, &bound, &target),
        Err(Render2dExecutionError::Vector {
            root_index: 1,
            kind: Render2dVectorError::PrecisionLimit
        })
    ));
    let readback = GpuReadbackOperation::ordinary(
        GpuTextureCopyRegion::whole_base_mip(&texture)
            .unwrap()
            .into(),
    )
    .unwrap();
    let id_readback = readback.id();
    let fragment = GpuWorkFragment::build("rejected preparation target", |work| {
        work.operation("read unchanged target", readback)?;
        Ok(())
    })
    .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("rejection proof").unwrap(),
        [fragment],
    )
    .unwrap();
    let submission = context
        .submit_prepared(pollster::block_on(context.prepare_submission(graph)).unwrap())
        .unwrap();
    assert!(
        await_readback(&context, &submission, id_readback)
            .as_bytes()
            .iter()
            .all(|b| *b == 0)
    );
    let changed = bindings(id, shaped_resource(OUTLINE_FONT, false, BOX_GLYPH, 25.0));
    assert!(
        executor
            .prepare(
                &context,
                &shaped_composition(id, [8.0, 32.0], Render2dColorRgba8::WHITE),
                &changed,
                &target
            )
            .is_ok()
    );
}

#[test]
fn supported_gradient_no_work_and_unsupported_clip_fail_closed() {
    let Some(context) = context() else {
        return;
    };
    let (_, target) = target("unsupported semantics");
    let gradient = Render2dBrush::Linear(
        Render2dLinearGradient::new(
            point(0.0, 0.0),
            point(1.0, 0.0),
            Render2dGradientStops::new(vec![
                Render2dGradientStop::new(0.0, Render2dColorRgba8::TRANSPARENT).unwrap(),
                Render2dGradientStop::new(1.0, Render2dColorRgba8::TRANSPARENT).unwrap(),
            ])
            .unwrap(),
        )
        .unwrap(),
    );
    let empty = item(
        Render2dPrimitive::Fill {
            shape: rect(0.0, 0.0, 0.0, 0.0),
            brush: gradient,
        },
        Render2dAffineTransform::IDENTITY,
        0.0,
    );
    assert!(
        !Render2dExecutor::new()
            .prepare(
                &context,
                &Render2dComposition::new(vec![empty]).unwrap(),
                &Render2dResourceBindings::default(),
                &target,
            )
            .expect("admitted empty gradient must have no render work")
            .has_render_work()
    );
    let clipped = Render2dEntry::item(Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: rect(8.0, 8.0, 10.0, 10.0),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        Render2dAffineTransform::IDENTITY,
        vec![Render2dClip::new(
            rect(0.0, 0.0, 10.0, 10.0),
            Render2dAffineTransform::IDENTITY,
        )],
        Render2dOpacity::TRANSPARENT,
    ));
    assert!(
        !Render2dExecutor::new()
            .prepare(
                &context,
                &Render2dComposition::new(vec![clipped]).unwrap(),
                &Render2dResourceBindings::default(),
                &target,
            )
            .expect("clipped transparent content is admitted and has no work")
            .has_render_work()
    );
}

#[test]
fn all_vector_nodes_execute_with_caller_owned_clear_and_prior_import() {
    let Some(context) = context() else {
        return;
    };
    let composition = Render2dComposition::new(vec![
        fill(
            rect(8.0, 8.0, 20.0, 20.0),
            Render2dColorRgba8::new(255, 0, 0, 128),
        ),
        fill(rect(40.0, 40.0, 8.0, 8.0), Render2dColorRgba8::WHITE),
    ])
    .unwrap();
    let (texture, target) = target("multi-node vector target");
    let bytes = execute_inline(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        &texture,
        &target,
        Some([0.0, 0.0, 1.0, 1.0]),
    );
    assert_pixel(&bytes, 0, 0, [0, 0, 255, 255]);
    assert_pixel(&bytes, 16, 16, [188, 0, 187, 255]);
    assert_pixel(&bytes, 44, 44, [255; 4]);
    let key = GpuExportKey::new("f3a.prior").unwrap();
    let prior = GpuWorkFragment::build("caller prior vector target", |work| {
        work.operation(
            "caller blue clear",
            GpuRenderOperation::new(
                [GpuRenderColorAttachment::new(
                    target.view().clone(),
                    GpuColorAttachmentLoad::Clear(
                        GpuColorClearValue::new(0.0, 0.0, 1.0, 1.0).unwrap(),
                    ),
                    GpuAttachmentStore::Store,
                    None,
                )
                .unwrap()],
                None,
                [],
                None,
            )
            .unwrap(),
        )?;
        work.declare_resource(GpuResourceRef::Texture(texture.clone()))?;
        work.add_output(runen_gpu::GpuWorkOutput::new(
            runen_gpu::GpuExportRelationship::new(
                GpuResourceRef::Texture(texture.clone()),
                key.clone(),
                GpuResourceAccessIntent::Write,
                GpuResourceProvenance::new(
                    GpuResourceLabel::new("caller prior").unwrap(),
                    None,
                    None,
                ),
            ),
            runen_gpu::GpuInitialCoverage::texture_subresources(
                &runen_gpu::GpuTextureAccessResource::Texture(texture.clone()),
                [target.view().descriptor().subresources()],
            )?,
        )?)?;
        Ok(())
    })
    .unwrap();
    let (fragment, token) = Render2dExecutor::new()
        .prepare(
            &context,
            &composition,
            &Render2dResourceBindings::default(),
            &target,
        )
        .unwrap()
        .into_fragment(&work_binding("after prior vector").after(key))
        .unwrap();
    assert_eq!(
        fragment.nodes().len(),
        6,
        "one root sample clear, two coverage/color pairs and one final target resolve"
    );
    let nodes = fragment
        .nodes()
        .iter()
        .map(|n| n.id().clone())
        .collect::<Vec<_>>();
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("reversed vector fragment inventory").unwrap(),
        [fragment, prior],
    )
    .unwrap();
    let submission = context
        .submit_prepared(pollster::block_on(context.prepare_submission(graph)).unwrap())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(30);
    while submission.status() == GpuSubmissionStatus::Accepted {
        assert!(Instant::now() < deadline);
        context.progress();
    }
    assert!(nodes.iter().all(|n| submission.contains_work_node(n)));
    assert_eq!(
        token
            .unwrap()
            .completed_by(&submission)
            .unwrap()
            .submission_id(),
        submission.id()
    );
}

#[test]
fn structural_strokes_and_degenerate_path_caps_render() {
    let Some(context) = context() else {
        return;
    };
    for shape in [
        rect(8.0, 8.0, 40.0, 40.0),
        Render2dShape::rounded_rect(
            Render2dRect::new(8.0, 8.0, 40.0, 40.0).unwrap(),
            Render2dCornerRadii::new(8.0, 8.0, 8.0, 8.0).unwrap(),
        ),
        Render2dShape::ellipse(Render2dRect::new(8.0, 8.0, 40.0, 40.0).unwrap()),
    ] {
        let bytes = render(
            &context,
            vec![stroke(
                shape,
                Render2dStrokeCap::Butt,
                Render2dStrokeJoin::Round,
                4.0,
                Render2dColorRgba8::WHITE,
            )],
        );
        assert_pixel(&bytes, 28, 28, [0; 4]);
        assert_pixel(&bytes, 28, 8, [255; 4]);
        assert_pixel(&bytes, 28, 2, [0; 4]);
    }
    let degenerate = path(
        &[[32.0, 32.0], [32.0, 32.0]],
        false,
        Render2dFillRule::NonZero,
    );
    let bytes = render(
        &context,
        vec![stroke(
            degenerate,
            Render2dStrokeCap::Round,
            Render2dStrokeJoin::Round,
            4.0,
            Render2dColorRgba8::new(255, 0, 0, 128),
        )],
    );
    assert_pixel(&bytes, 32, 32, [188, 0, 0, 128]);
    assert_pixel(&bytes, 36, 32, [0; 4]);
}

#[test]
fn vector_only_context_needs_no_text_field_upload_or_filter_roles() {
    let Some(context) = context_with_text_roles(false) else {
        return;
    };
    let bytes = render(
        &context,
        vec![fill(rect(8.0, 8.0, 10.0, 10.0), Render2dColorRgba8::WHITE)],
    );
    assert_pixel(&bytes, 12, 12, [255; 4]);
}

#[test]
fn large_vector_coverage_is_accepted_with_bounded_tiles() {
    let Some(context) = context() else {
        return;
    };
    let mut resources = GpuResourceScope::new();
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                "bounded vector mask",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                1100,
                1100,
                GpuTextureFormat::Rgba8UnormSrgb,
                [GpuTextureUsage::ColorAttachment],
                GpuTextureInitialization::Zeroed,
            )
            .unwrap(),
        )
        .unwrap();
    let view = resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned("bounded vector mask view", &texture)
                .unwrap(),
        )
        .unwrap();
    let target = Render2dTarget::new(view, 1100.0, 1100.0, 1.0).unwrap();
    let composition = Render2dComposition::new(vec![fill(
        rect(0.0, 0.0, 1100.0, 1100.0),
        Render2dColorRgba8::WHITE,
    )])
    .unwrap();
    assert!(
        Render2dExecutor::new()
            .prepare(&context, &composition, &Render2dResourceBindings::default(), &target)
            .expect("large F3E vector uses bounded reusable tiles rather than one full-surface mask")
            .has_render_work()
    );
}

#[test]
fn nonprimary_srgb_color_and_opacity_blend_over_translucent_destination() {
    let Some(context) = context() else {
        return;
    };
    let entry = item(
        Render2dPrimitive::Fill {
            shape: rect(8.0, 8.0, 20.0, 20.0),
            brush: Render2dBrush::solid(Render2dColorRgba8::new(128, 64, 192, 96)),
        },
        Render2dAffineTransform::IDENTITY,
        0.75,
    );
    let composition = Render2dComposition::new(vec![entry]).unwrap();
    let (texture, target) = target("nonprimary color source-over");
    let bytes = execute_inline(
        &context,
        &mut Render2dExecutor::new(),
        &composition,
        &Render2dResourceBindings::default(),
        &texture,
        &target,
        Some([0.012, 0.06, 0.2, 0.4]),
    );
    // Independently calculated: decode sRGB8, alpha=96/255*0.75, premultiplied
    // source-over over the declared linear destination, then encode RGB only.
    assert_pixel(&bytes, 16, 16, [75, 68, 147, 145]);
}

#[test]
fn curved_stroke_caps_retain_authored_endpoint_tangents() {
    let Some(context) = context() else {
        return;
    };
    let shape = Render2dShape::path(
        Render2dPath::new(
            Render2dFillRule::NonZero,
            vec![
                Render2dPathCommand::MoveTo(point(20.0, 24.0)),
                Render2dPathCommand::CubicTo {
                    control1: point(20.0, 40.0),
                    control2: point(40.0, 40.0),
                    to: point(40.0, 24.0),
                },
            ],
        )
        .unwrap(),
    );
    let square = render(
        &context,
        vec![stroke(
            shape.clone(),
            Render2dStrokeCap::Square,
            Render2dStrokeJoin::Round,
            4.0,
            Render2dColorRgba8::WHITE,
        )],
    );
    let butt = render(
        &context,
        vec![stroke(
            shape,
            Render2dStrokeCap::Butt,
            Render2dStrokeJoin::Round,
            4.0,
            Render2dColorRgba8::WHITE,
        )],
    );
    assert_pixel(&square, 20, 21, [255; 4]);
    assert_pixel(&square, 40, 21, [255; 4]);
    assert_pixel(&butt, 20, 21, [0; 4]);
    assert_pixel(&butt, 40, 21, [0; 4]);
}
