use std::{error::Error, mem::size_of, time::Instant};

use lyon_tessellation::{
    BuffersBuilder, FillOptions, FillRule as LyonFillRule, FillTessellator, FillVertex, LineCap,
    LineJoin, StrokeOptions, StrokeTessellator, StrokeVertex, VertexBuffers,
    path::{Path as LyonPath, math::point},
};
use vello_common::{
    fearless_simd::Level,
    kurbo::{Affine, BezPath, Cap as VelloCap, Join as VelloJoin, Stroke as VelloStroke},
    peniko::Fill as VelloFill,
    strip::Strip,
    strip_generator::{StripGenerator, StripStorage},
};

const VIEWPORT_WIDTH: u16 = 1024;
const VIEWPORT_HEIGHT: u16 = 768;
const REPEATS: usize = 5;
const ITERATIONS: usize = 64;
const SCALES: [f32; 4] = [0.75, 1.0, 2.0, 4.0];

#[derive(Clone, Copy)]
enum Cmd {
    Move(f32, f32),
    Line(f32, f32),
    Quad(f32, f32, f32, f32),
    Cubic(f32, f32, f32, f32, f32, f32),
    Close,
    EndOpen,
}

#[derive(Clone, Copy)]
enum FillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Copy)]
enum Join {
    Miter,
    Bevel,
    Round,
}

#[derive(Clone, Copy)]
enum Cap {
    Butt,
    Square,
    Round,
}

#[derive(Clone, Copy)]
struct StrokeStyle {
    width: f32,
    join: Join,
    cap: Cap,
    miter_limit: f32,
}

#[derive(Clone, Copy)]
enum Operation {
    Fill(FillRule),
    Stroke(StrokeStyle),
}

struct Case {
    label: &'static str,
    commands: Vec<Cmd>,
    operation: Operation,
}

#[derive(Clone, Copy)]
struct Stats {
    hash: u64,
    primary_count: usize,
    secondary_count: usize,
    bytes: usize,
}

fn xy(x: f32, y: f32, scale: f32, translate: (f32, f32)) -> (f32, f32) {
    (x * scale + translate.0, y * scale + translate.1)
}

fn lyon_path(commands: &[Cmd], scale: f32, translate: (f32, f32)) -> LyonPath {
    let mut builder = LyonPath::builder();
    let mut open = false;
    for command in commands {
        match *command {
            Cmd::Move(x, y) => {
                if open {
                    builder.end(false);
                }
                let (x, y) = xy(x, y, scale, translate);
                builder.begin(point(x, y));
                open = true;
            }
            Cmd::Line(x, y) => {
                let (x, y) = xy(x, y, scale, translate);
                builder.line_to(point(x, y));
            }
            Cmd::Quad(cx, cy, x, y) => {
                let (cx, cy) = xy(cx, cy, scale, translate);
                let (x, y) = xy(x, y, scale, translate);
                builder.quadratic_bezier_to(point(cx, cy), point(x, y));
            }
            Cmd::Cubic(cx0, cy0, cx1, cy1, x, y) => {
                let (cx0, cy0) = xy(cx0, cy0, scale, translate);
                let (cx1, cy1) = xy(cx1, cy1, scale, translate);
                let (x, y) = xy(x, y, scale, translate);
                builder.cubic_bezier_to(point(cx0, cy0), point(cx1, cy1), point(x, y));
            }
            Cmd::Close => {
                if open {
                    builder.end(true);
                    open = false;
                }
            }
            Cmd::EndOpen => {
                if open {
                    builder.end(false);
                    open = false;
                }
            }
        }
    }
    if open {
        builder.end(false);
    }
    builder.build()
}

fn vello_path(commands: &[Cmd], scale: f32, translate: (f32, f32)) -> BezPath {
    let mut path = BezPath::new();
    for command in commands {
        match *command {
            Cmd::Move(x, y) => {
                let (x, y) = xy(x, y, scale, translate);
                path.move_to((f64::from(x), f64::from(y)));
            }
            Cmd::Line(x, y) => {
                let (x, y) = xy(x, y, scale, translate);
                path.line_to((f64::from(x), f64::from(y)));
            }
            Cmd::Quad(cx, cy, x, y) => {
                let (cx, cy) = xy(cx, cy, scale, translate);
                let (x, y) = xy(x, y, scale, translate);
                path.quad_to((f64::from(cx), f64::from(cy)), (f64::from(x), f64::from(y)));
            }
            Cmd::Cubic(cx0, cy0, cx1, cy1, x, y) => {
                let (cx0, cy0) = xy(cx0, cy0, scale, translate);
                let (cx1, cy1) = xy(cx1, cy1, scale, translate);
                let (x, y) = xy(x, y, scale, translate);
                path.curve_to(
                    (f64::from(cx0), f64::from(cy0)),
                    (f64::from(cx1), f64::from(cy1)),
                    (f64::from(x), f64::from(y)),
                );
            }
            Cmd::Close => path.close_path(),
            Cmd::EndOpen => {}
        }
    }
    path
}

fn lyon_fill(rule: FillRule) -> LyonFillRule {
    match rule {
        FillRule::NonZero => LyonFillRule::NonZero,
        FillRule::EvenOdd => LyonFillRule::EvenOdd,
    }
}

fn vello_fill(rule: FillRule) -> VelloFill {
    match rule {
        FillRule::NonZero => VelloFill::NonZero,
        FillRule::EvenOdd => VelloFill::EvenOdd,
    }
}

fn lyon_join(join: Join) -> LineJoin {
    match join {
        Join::Miter => LineJoin::Miter,
        Join::Bevel => LineJoin::Bevel,
        Join::Round => LineJoin::Round,
    }
}

fn vello_join(join: Join) -> VelloJoin {
    match join {
        Join::Miter => VelloJoin::Miter,
        Join::Bevel => VelloJoin::Bevel,
        Join::Round => VelloJoin::Round,
    }
}

fn lyon_cap(cap: Cap) -> LineCap {
    match cap {
        Cap::Butt => LineCap::Butt,
        Cap::Square => LineCap::Square,
        Cap::Round => LineCap::Round,
    }
}

fn vello_cap(cap: Cap) -> VelloCap {
    match cap {
        Cap::Butt => VelloCap::Butt,
        Cap::Square => VelloCap::Square,
        Cap::Round => VelloCap::Round,
    }
}

fn lyon_stats(case: &Case, path: &LyonPath, scale: f32) -> Result<Stats, Box<dyn Error>> {
    let mut geometry: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    match case.operation {
        Operation::Fill(rule) => FillTessellator::new().tessellate_path(
            path,
            &FillOptions::default().with_fill_rule(lyon_fill(rule)),
            &mut BuffersBuilder::new(&mut geometry, |v: FillVertex<'_>| v.position().to_array()),
        )?,
        Operation::Stroke(style) => StrokeTessellator::new().tessellate_path(
            path,
            &StrokeOptions::default()
                .with_line_width(style.width * scale)
                .with_line_join(lyon_join(style.join))
                .with_line_cap(lyon_cap(style.cap))
                .with_miter_limit(style.miter_limit),
            &mut BuffersBuilder::new(&mut geometry, |v: StrokeVertex<'_, '_>| {
                v.position().to_array()
            }),
        )?,
    }

    let mut hash = Fnv64::new();
    for vertex in &geometry.vertices {
        hash.write(&vertex[0].to_bits().to_le_bytes());
        hash.write(&vertex[1].to_bits().to_le_bytes());
    }
    for index in &geometry.indices {
        hash.write(&index.to_le_bytes());
    }
    Ok(Stats {
        hash: hash.finish(),
        primary_count: geometry.vertices.len(),
        secondary_count: geometry.indices.len(),
        bytes: geometry.vertices.len() * size_of::<[f32; 2]>()
            + geometry.indices.len() * size_of::<u32>(),
    })
}

fn vello_stats(case: &Case, path: &BezPath, scale: f32) -> Stats {
    let mut generator = StripGenerator::new(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, Level::baseline());
    let mut storage = StripStorage::default();
    match case.operation {
        Operation::Fill(rule) => generator.generate_filled_path(
            path.elements().iter().copied(),
            vello_fill(rule),
            Affine::IDENTITY,
            None,
            &mut storage,
            None,
        ),
        Operation::Stroke(style) => {
            let stroke = VelloStroke {
                width: f64::from(style.width * scale),
                join: vello_join(style.join),
                start_cap: vello_cap(style.cap),
                end_cap: vello_cap(style.cap),
                miter_limit: f64::from(style.miter_limit),
                ..Default::default()
            };
            generator.generate_stroked_path(
                path.elements().iter().copied(),
                &stroke,
                Affine::IDENTITY,
                None,
                &mut storage,
                None,
            );
        }
    }

    let mut hash = Fnv64::new();
    for strip in &storage.strips {
        hash.write(&strip.x.to_le_bytes());
        hash.write(&strip.y.to_le_bytes());
        hash.write(&strip.alpha_idx().to_le_bytes());
        hash.write(&[u8::from(strip.fill_gap())]);
    }
    hash.write(&storage.alphas);
    Stats {
        hash: hash.finish(),
        primary_count: storage.strips.len(),
        secondary_count: storage.alphas.len(),
        bytes: storage.strips.len() * size_of::<Strip>() + storage.alphas.len(),
    }
}

fn verify(case: &Case, scale: f32) -> Result<(Stats, Stats), Box<dyn Error>> {
    let lyon = lyon_path(&case.commands, scale, (96.0, 96.0));
    let vello = vello_path(&case.commands, scale, (96.0, 96.0));
    let first_lyon = lyon_stats(case, &lyon, scale)?;
    let first_vello = vello_stats(case, &vello, scale);
    for _ in 1..REPEATS {
        if lyon_stats(case, &lyon, scale)?.hash != first_lyon.hash {
            return Err(format!("Lyon output was nondeterministic for {}", case.label).into());
        }
        if vello_stats(case, &vello, scale).hash != first_vello.hash {
            return Err(format!("Sparse Strips output was nondeterministic for {}", case.label).into());
        }
    }
    Ok((first_lyon, first_vello))
}

fn time_lyon(case: &Case, scale: f32, changing: bool) -> Result<u128, Box<dyn Error>> {
    match case.operation {
        Operation::Fill(rule) => {
            let options = FillOptions::default().with_fill_rule(lyon_fill(rule));
            let mut tessellator = FillTessellator::new();
            let mut geometry: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
            let mut sink = 0usize;
            let start = Instant::now();
            if changing {
                for i in 0..ITERATIONS {
                    geometry.vertices.clear();
                    geometry.indices.clear();
                    let phase = i as f32 * 0.375;
                    let path = lyon_path(
                        &case.commands,
                        scale,
                        (96.0 + phase, 96.0 + phase * 0.5),
                    );
                    tessellator.tessellate_path(
                        &path,
                        &options,
                        &mut BuffersBuilder::new(&mut geometry, |v: FillVertex<'_>| {
                            v.position().to_array()
                        }),
                    )?;
                    sink ^= geometry.vertices.len() ^ geometry.indices.len();
                }
            } else {
                let path = lyon_path(&case.commands, scale, (96.0, 96.0));
                for _ in 0..ITERATIONS {
                    geometry.vertices.clear();
                    geometry.indices.clear();
                    tessellator.tessellate_path(
                        &path,
                        &options,
                        &mut BuffersBuilder::new(&mut geometry, |v: FillVertex<'_>| {
                            v.position().to_array()
                        }),
                    )?;
                    sink ^= geometry.vertices.len() ^ geometry.indices.len();
                }
            }
            std::hint::black_box(sink);
            Ok(start.elapsed().as_nanos() / ITERATIONS as u128)
        }
        Operation::Stroke(style) => {
            let options = StrokeOptions::default()
                .with_line_width(style.width * scale)
                .with_line_join(lyon_join(style.join))
                .with_line_cap(lyon_cap(style.cap))
                .with_miter_limit(style.miter_limit);
            let mut tessellator = StrokeTessellator::new();
            let mut geometry: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
            let mut sink = 0usize;
            let start = Instant::now();
            if changing {
                for i in 0..ITERATIONS {
                    geometry.vertices.clear();
                    geometry.indices.clear();
                    let phase = i as f32 * 0.375;
                    let path = lyon_path(
                        &case.commands,
                        scale,
                        (96.0 + phase, 96.0 + phase * 0.5),
                    );
                    tessellator.tessellate_path(
                        &path,
                        &options,
                        &mut BuffersBuilder::new(&mut geometry, |v: StrokeVertex<'_, '_>| {
                            v.position().to_array()
                        }),
                    )?;
                    sink ^= geometry.vertices.len() ^ geometry.indices.len();
                }
            } else {
                let path = lyon_path(&case.commands, scale, (96.0, 96.0));
                for _ in 0..ITERATIONS {
                    geometry.vertices.clear();
                    geometry.indices.clear();
                    tessellator.tessellate_path(
                        &path,
                        &options,
                        &mut BuffersBuilder::new(&mut geometry, |v: StrokeVertex<'_, '_>| {
                            v.position().to_array()
                        }),
                    )?;
                    sink ^= geometry.vertices.len() ^ geometry.indices.len();
                }
            }
               std::hint::black_box(sink);
            Ok(start.elapsed().as_nanos() / ITERATIONS as u128)
        }
    }
}

fn generate_vello(
    case: &Case,
    generator: &mut StripGenerator,
    storage: &mut StripStorage,
    path: &BezPath,
    scale: f32,
) {
    storage.clear();
    match case.operation {
        Operation::Fill(rule) => generator.generate_filled_path(
            path.elements().iter().copied(),
            vello_fill(rule),
            Affine::IDENTITY,
            None,
            storage,
            None,
        ),
        Operation::Stroke(style) => {
            let stroke = VelloStroke {
                width: f64::from(style.width * scale),
                join: vello_join(style.join),
                start_cap: vello_cap(style.cap),
                end_cap: vello_cap(style.cap),
                miter_limit: f64::from(style.miter_limit),
                ..Default::default()
            };
            generator.generate_stroked_path(
                path.elements().iter().copied(),
                &stroke,
                Affine::IDENTITY,
                None,
                storage,
                None,
            );
        }
    }
}

fn time_vello(case: &Case, scale: f32, changing: bool) -> u128 {
    let mut generator = StripGenerator::new(VIEWPORT_WIDTH, VIEWPORT_HEIGHT, Level::baseline());
    let mut storage = StripStorage::default();
    let mut sink = 0usize;
    let start = Instant::now();
    if changing {
        for i in 0..ITERATIONS {
            let phase = i as f32 * 0.375;
            let path = vello_path(
                &case.commands,
                scale,
                (96.0 + phase, 96.0 + phase * 0.5),
            );
            generate_vello(case, &mut generator, &mut storage, &path, scale);
            sink ^= storage.strips.len() ^ storage.alphas.len();
        }
    } else {
        let path = vello_path(&case.commands, scale, (96.0, 96.0));
        for _ in 0..ITERATIONS {
            generate_vello(case, &mut generator, &mut storage, &path, scale);
            sink ^= storage.strips.len() ^ storage.alphas.len();
        }
    }
    std::hint::black_box(sink);
    start.elapsed().as_nanos() / ITERATIONS as u128
}

fn op_label(op: Operation) -> &'static str {
    match op {
        Operation::Fill(FillRule::NonZero) => "fill-nonzero",
        Operation::Fill(FillRule::EvenOdd) => "fill-evenodd",
        Operation::Stroke(StrokeStyle { join: Join::Miter, .. }) => "stroke-miter",
        Operation::Stroke(StrokeStyle { join: Join::Bevel, .. }) => "stroke-bevel",
        Operation::Stroke(StrokeStyle { join: Join::Round, .. }) => "stroke-round",
    }
}

fn report(case: &Case, scale: f32, name: &str, stats: Stats, reused: u128, changing: u128) {
    println!(
        "candidate={name} case={} operation={} scale={scale:.2} hash={:016x} primary_count={} secondary_count={} output_bytes={} reused_ns_per_iter={reused} changing_ns_per_iter={changing}",
        case.label,
        op_label(case.operation),
        stats.hash,
        stats.primary_count,
        stats.secondary_count,
        stats.bytes,
    );
}

fn corpus() -> Vec<Case> {
    vec![
        Case {
            label: "rect",
            commands: vec![Cmd::Move(0.0, 0.0), Cmd::Line(180.0, 0.0), Cmd::Line(180.0, 96.0), Cmd::Line(0.0, 96.0), Cmd::Close],
            operation: Operation::Fill(FillRule::NonZero),
        },
        Case {
            label: "rounded-rect-like",
            commands: vec![Cmd::Move(24.0, 0.0), Cmd::Line(176.0, 0.0), Cmd::Quad(200.0, 0.0, 200.0, 24.0), Cmd::Line(200.0, 96.0), Cmd::Quad(200.0, 120.0, 176.0, 120.0), Cmd::Line(24.0, 120.0), Cmd::Quad(0.0, 120.0, 0.0, 96.0), Cmd::Line(0.0, 24.0), Cmd::Quad(0.0, 0.0, 24.0, 0.0), Cmd::Close],
            operation: Operation::Fill(FillRule::NonZero),
        },
        Case {
            label: "ellipse-like-cubic",
            commands: vec![Cmd::Move(110.0, 0.0), Cmd::Cubic(170.75, 0.0, 220.0, 40.3, 220.0, 90.0), Cmd::Cubic(220.0, 139.7, 170.75, 180.0, 110.0, 180.0), Cmd::Cubic(49.25, 180.0, 0.0, 139.7, 0.0, 90.0), Cmd::Cubic(0.0, 40.3, 49.25, 0.0, 110.0, 0.0), Cmd::Close],
            operation: Operation::Fill(FillRule::NonZero),
        },
        Case {
            label: "bezier-icon",
            commands: vec![Cmd::Move(10.0, 70.0), Cmd::Cubic(40.0, 10.0, 100.0, 10.0, 130.0, 70.0), Cmd::Quad(160.0, 130.0, 210.0, 70.0), Cmd::Cubic(180.0, 170.0, 60.0, 170.0, 10.0, 70.0), Cmd::Close],
            operation: Operation::Fill(FillRule::NonZero),
        },
        Case {
            label: "self-intersection-evenodd",
            commands: star(),
            operation: Operation::Fill(FillRule::EvenOdd),
        },
        Case {
            label: "self-intersection-nonzero",
            commands: star(),
            operation: Operation::Fill(FillRule::NonZero),
        },
        Case {
            label: "open-stroke-round",
            commands: vec![Cmd::Move(0.0, 90.0), Cmd::Line(50.0, 10.0), Cmd::Quad(100.0, 160.0, 150.0, 30.0), Cmd::Cubic(190.0, 0.0, 220.0, 180.0, 280.0, 60.0), Cmd::EndOpen],
            operation: Operation::Stroke(StrokeStyle { width: 8.0, join: Join::Round, cap: Cap::Round, miter_limit: 4.0 }),
        },
        Case {
            label: "open-stroke-bevel-square",
            commands: vec![Cmd::Move(0.0, 100.0), Cmd::Line(70.0, 10.0), Cmd::Line(95.0, 150.0), Cmd::Line(165.0, 15.0), Cmd::Line(240.0, 110.0), Cmd::EndOpen],
            operation: Operation::Stroke(StrokeStyle { width: 10.0, join: Join::Bevel, cap: Cap::Square, miter_limit: 4.0 }),
        },
        Case {
            label: "open-stroke-miter-pressure",
            commands: vec![Cmd::Move(0.0, 140.0), Cmd::Line(90.0, 0.0), Cmd::Line(100.0, 145.0), Cmd::Line(190.0, 5.0), Cmd::Line(205.0, 150.0), Cmd::Line(290.0, 20.0), Cmd::EndOpen],
            operation: Operation::Stroke(StrokeStyle { width: 12.0, join: Join::Miter, cap: Cap::Butt, miter_limit: 2.0 }),
        },
        Case {
            label: "off-viewport-culling",
            commands: vec![Cmd::Move(-420.0, -260.0), Cmd::Cubic(-180.0, -320.0, 260.0, -160.0, 520.0, 80.0), Cmd::Line(1300.0, 120.0), Cmd::Line(1300.0, 680.0), Cmd::Cubic(600.0, 720.0, -200.0, 860.0, -420.0, 520.0), Cmd::Close],
            operation: Operation::Fill(FillRule::NonZero),
        },
    ]
}

fn star() -> Vec<Cmd> {
    vec![Cmd::Move(100.0, 0.0), Cmd::Line(124.0, 72.0), Cmd::Line(200.0, 72.0), Cmd::Line(138.0, 116.0), Cmd::Line(162.0, 190.0), Cmd::Line(100.0, 145.0), Cmd::Line(38.0, 190.0), Cmd::Line(62.0, 116.0), Cmd::Line(0.0, 72.0), Cmd::Line(76.0, 72.0), Cmd::Close]
}

#[derive(Clone, Copy)]
struct Fnv64(u64);

impl Fnv64 {
    const fn new() -> Self { Self(0xcbf2_9ce4_8422_2325) }
    fn write(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    const fn finish(self) -> u64 { self.0 }
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("RR23-E1 general-path physical representation evaluation");
    println!("candidates=lyon_tessellation@1.0.22,vello_common@0.3.0 viewport={}x{} repeats={} iterations={} scales=0.75,1,2,4", VIEWPORT_WIDTH, VIEWPORT_HEIGHT, REPEATS, ITERATIONS);
    println!("scope=cpu_preprocessing_and_output_pressure_only no_gpu_performance_claim=true vello_common_production_adoption=false");

    let cases = corpus();
    let mut measurements = 0usize;
    for case in &cases {
        for scale in SCALES {
            let (lyon, vello) = verify(case, scale)?;
            report(case, scale, "lyon_tessellation@1.0.22", lyon, time_lyon(case, scale, false)?, time_lyon(case, scale, true)?);
            report(case, scale, "vello_common@0.3.0-sparse-strips", vello, time_vello(case, scale, false), time_vello(case, scale, true));
            measurements += 2;
        }
    }
    println!("summary cases={} scales={} measurements={} deterministic_repeats={} timed_iterations_per_mode={}", cases.len(), SCALES.len(), measurements, REPEATS, ITERATIONS);
    Ok(())
}
