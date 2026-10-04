use std::{
    error::Error,
    hint::black_box,
    mem::size_of,
    time::{Duration, Instant},
};

use lyon_tessellation::{
    geometry_builder::{BuffersBuilder, VertexBuffers},
    math::point,
    path::Path,
    FillOptions, FillRule as LyonFillRule, FillTessellator, FillVertex,
    LineCap as LyonCap, LineJoin as LyonJoin, StrokeOptions, StrokeTessellator,
    StrokeVertex,
};
use vello_common::{
    fearless_simd::Level,
    kurbo::{Affine, BezPath, Cap as VelloCap, Join as VelloJoin, Stroke},
    peniko::Fill as VelloFill,
    strip_generator::{StripGenerator, StripStorage},
};

const VIEWPORT: u16 = 1024;
const WARMUP: usize = 8;
const SAMPLES: usize = 32;
const SCALES: [f64; 3] = [0.5, 1.0, 4.0];

#[derive(Clone, Copy)]
enum Command {
    Move(f64, f64),
    Line(f64, f64),
    Quad(f64, f64, f64, f64),
    Cubic(f64, f64, f64, f64, f64, f64),
    Close,
}

#[derive(Clone, Copy)]
enum FillRule {
    NonZero,
    EvenOdd,
}

#[derive(Clone, Copy)]
enum Join {
    Bevel,
    Miter,
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
    width: f64,
    join: Join,
    cap: Cap,
    miter_limit: f64,
}

#[derive(Clone, Copy)]
enum Operation {
    Fill(FillRule),
    Stroke(StrokeStyle),
}

struct Case {
    name: &'static str,
    commands: &'static [Command],
    operation: Operation,
    translate: (f64, f64),
}

#[derive(Clone, Copy)]
struct Evidence {
    digest: u64,
    primary_count: usize,
    secondary_count: usize,
    payload_bytes: usize,
}

struct Timing {
    min_ns: u128,
    median_ns: u128,
    p90_ns: u128,
    max_ns: u128,
}

const RECT: &[Command] = &[
    Command::Move(32.0, 32.0),
    Command::Line(288.0, 32.0),
    Command::Line(288.0, 192.0),
    Command::Line(32.0, 192.0),
    Command::Close,
];

const ROUNDED: &[Command] = &[
    Command::Move(56.0, 24.0),
    Command::Line(280.0, 24.0),
    Command::Cubic(300.0, 24.0, 312.0, 36.0, 312.0, 56.0),
    Command::Line(312.0, 184.0),
    Command::Cubic(312.0, 204.0, 300.0, 216.0, 280.0, 216.0),
    Command::Line(56.0, 216.0),
    Command::Cubic(36.0, 216.0, 24.0, 204.0, 24.0, 184.0),
    Command::Line(24.0, 56.0),
    Command::Cubic(24.0, 36.0, 36.0, 24.0, 56.0, 24.0),
    Command::Close,
];

const ELLIPSE: &[Command] = &[
    Command::Move(320.0, 160.0),
    Command::Cubic(320.0, 230.692, 262.692, 288.0, 192.0, 288.0),
    Command::Cubic(121.308, 288.0, 64.0, 230.692, 64.0, 160.0),
    Command::Cubic(64.0, 89.308, 121.308, 32.0, 192.0, 32.0),
    Command::Cubic(262.692, 32.0, 320.0, 89.308, 320.0, 160.0),
    Command::Close,
];

const ICON: &[Command] = &[
    Command::Move(64.0, 40.0),
    Command::Cubic(84.0, 14.0, 126.0, 12.0, 148.0, 38.0),
    Command::Quad(168.0, 62.0, 152.0, 92.0),
    Command::Cubic(136.0, 122.0, 96.0, 142.0, 64.0, 120.0),
    Command::Cubic(34.0, 100.0, 30.0, 64.0, 64.0, 40.0),
    Command::Close,
    Command::Move(78.0, 62.0),
    Command::Line(130.0, 62.0),
    Command::Line(104.0, 104.0),
    Command::Close,
];

const SELF_INTERSECT: &[Command] = &[
    Command::Move(160.0, 16.0),
    Command::Line(198.0, 120.0),
    Command::Line(308.0, 120.0),
    Command::Line(220.0, 184.0),
    Command::Line(254.0, 294.0),
    Command::Line(160.0, 228.0),
    Command::Line(66.0, 294.0),
    Command::Line(100.0, 184.0),
    Command::Line(12.0, 120.0),
    Command::Line(122.0, 120.0),
    Command::Close,
];

const CURVE_STRESS: &[Command] = &[
    Command::Move(20.0, 160.0),
    Command::Cubic(48.0, 16.0, 92.0, 304.0, 124.0, 160.0),
    Command::Cubic(156.0, 16.0, 200.0, 304.0, 232.0, 160.0),
    Command::Cubic(264.0, 16.0, 308.0, 304.0, 340.0, 160.0),
    Command::Cubic(372.0, 16.0, 416.0, 304.0, 448.0, 160.0),
    Command::Line(448.0, 220.0),
    Command::Cubic(416.0, 364.0, 372.0, 76.0, 340.0, 220.0),
    Command::Cubic(308.0, 364.0, 264.0, 76.0, 232.0, 220.0),
    Command::Cubic(200.0, 364.0, 156.0, 76.0, 124.0, 220.0),
    Command::Cubic(92.0, 364.0, 48.0, 76.0, 20.0, 220.0),
    Command::Close,
];

const STROKE_OPEN: &[Command] = &[
    Command::Move(40.0, 80.0),
    Command::Cubic(80.0, 8.0, 128.0, 152.0, 176.0, 80.0),
    Command::Line(240.0, 148.0),
    Command::Quad(304.0, 212.0, 368.0, 92.0),
];

const CASES: &[Case] = &[
    Case { name: "rect_nonzero", commands: RECT, operation: Operation::Fill(FillRule::NonZero), translate: (0.0, 0.0) },
    Case { name: "rounded_nonzero", commands: ROUNDED, operation: Operation::Fill(FillRule::NonZero), translate: (0.0, 0.0) },
    Case { name: "ellipse_nonzero", commands: ELLIPSE, operation: Operation::Fill(FillRule::NonZero), translate: (0.0, 0.0) },
    Case { name: "icon_nonzero", commands: ICON, operation: Operation::Fill(FillRule::NonZero), translate: (0.0, 0.0) },
    Case { name: "self_intersect_nonzero", commands: SELF_INTERSECT, operation: Operation::Fill(FillRule::NonZero), translate: (0.0, 0.0) },
    Case { name: "self_intersect_evenodd", commands: SELF_INTERSECT, operation: Operation::Fill(FillRule::EvenOdd), translate: (0.0, 0.0) },
    Case { name: "curve_stress", commands: CURVE_STRESS, operation: Operation::Fill(FillRule::NonZero), translate: (0.0, 0.0) },
    Case { name: "offviewport_curve_stress", commands: CURVE_STRESS, operation: Operation::Fill(FillRule::NonZero), translate: (-360.0, 0.0) },
    Case {
        name: "stroke_bevel_butt",
        commands: STROKE_OPEN,
        operation: Operation::Stroke(StrokeStyle { width: 8.0, join: Join::Bevel, cap: Cap::Butt, miter_limit: 4.0 }),
        translate: (0.0, 0.0),
    },
    Case {
        name: "stroke_miter_square",
        commands: STROKE_OPEN,
        operation: Operation::Stroke(StrokeStyle { width: 12.0, join: Join::Miter, cap: Cap::Square, miter_limit: 6.0 }),
        translate: (0.0, 0.0),
    },
    Case {
        name: "stroke_round_round",
        commands: STROKE_OPEN,
        operation: Operation::Stroke(StrokeStyle { width: 12.0, join: Join::Round, cap: Cap::Round, miter_limit: 4.0 }),
        translate: (0.0, 0.0),
    },
];

fn transformed(commands: &[Command], scale: f64, translate: (f64, f64)) -> Vec<Command> {
    let p = |x: f64, y: f64| (x * scale + translate.0, y * scale + translate.1);
    commands.iter().map(|command| match *command {
        Command::Move(x, y) => { let (x, y) = p(x, y); Command::Move(x, y) }
        Command::Line(x, y) => { let (x, y) = p(x, y); Command::Line(x, y) }
        Command::Quad(cx, cy, x, y) => {
            let (cx, cy) = p(cx, cy);
            let (x, y) = p(x, y);
            Command::Quad(cx, cy, x, y)
        }
        Command::Cubic(cx0, cy0, cx1, cy1, x, y) => {
            let (cx0, cy0) = p(cx0, cy0);
            let (cx1, cy1) = p(cx1, cy1);
            let (x, y) = p(x, y);
            Command::Cubic(cx0, cy0, cx1, cy1, x, y)
        }
        Command::Close => Command::Close,
    }).collect()
}

fn lyon_path(commands: &[Command]) -> Path {
    let mut builder = Path::builder();
    let mut open = false;
    for command in commands {
        match *command {
            Command::Move(x, y) => {
                if open { builder.end(false); }
                builder.begin(point(x as f32, y as f32));
                open = true;
            }
            Command::Line(x, y) => builder.line_to(point(x as f32, y as f32)),
            Command::Quad(cx, cy, x, y) => builder.quadratic_bezier_to(point(cx as f32, cy as f32), point(x as f32, y as f32)),
            Command::Cubic(cx0, cy0, cx1, cy1, x, y) => builder.cubic_bezier_to(
                point(cx0 as f32, cy0 as f32),
                point(cx1 as f32, cy1 as f32),
                point(x as f32, y as f32),
            ),
            Command::Close => {
                if open {
                    builder.end(true);
                    open = false;
                }
            }
        }
    }
    if open { builder.end(false); }
    builder.build()
}

fn vello_path(commands: &[Command]) -> BezPath {
    let mut path = BezPath::new();
    for command in commands {
        match *command {
            Command::Move(x, y) => path.move_to((x, y)),
            Command::Line(x, y) => path.line_to((x, y)),
            Command::Quad(cx, cy, x, y) => path.quad_to((cx, cy), (x, y)),
            Command::Cubic(cx0, cy0, cx1, cy1, x, y) => path.curve_to((cx0, cy0), (cx1, cy1), (x, y)),
            Command::Close => path.close_path(),
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

fn lyon_join(join: Join) -> LyonJoin {
    match join {
        Join::Bevel => LyonJoin::Bevel,
        Join::Miter => LyonJoin::Miter,
        Join::Round => LyonJoin::Round,
    }
}

fn lyon_cap(cap: Cap) -> LyonCap {
    match cap {
        Cap::Butt => LyonCap::Butt,
        Cap::Square => LyonCap::Square,
        Cap::Round => LyonCap::Round,
    }
}

fn vello_join(join: Join) -> VelloJoin {
    match join {
        Join::Bevel => VelloJoin::Bevel,
        Join::Miter => VelloJoin::Miter,
        Join::Round => VelloJoin::Round,
    }
}

fn vello_cap(cap: Cap) -> VelloCap {
    match cap {
        Cap::Butt => VelloCap::Butt,
        Cap::Square => VelloCap::Square,
        Cap::Round => VelloCap::Round,
    }
}

fn fnv_mix(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

fn lyon_digest(vertices: &[[f32; 2]], indices: &[u32]) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    for [x, y] in vertices {
        hash = fnv_mix(hash, u64::from(x.to_bits()));
        hash = fnv_mix(hash, u64::from(y.to_bits()));
    }
    for index in indices {
        hash = fnv_mix(hash, u64::from(*index));
    }
    hash
}

fn vello_digest(storage: &StripStorage) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    for strip in &storage.strips {
        hash = fnv_mix(hash, u64::from(strip.x));
        hash = fnv_mix(hash, u64::from(strip.y));
        hash = fnv_mix(hash, u64::from(strip.alpha_idx()));
        hash = fnv_mix(hash, if strip.fill_gap() { 1 } else { 0 });
    }
    for alpha in &storage.alphas {
        hash = fnv_mix(hash, u64::from(*alpha));
    }
    hash
}

fn run_lyon(path: &Path, operation: Operation) -> Result<Evidence, Box<dyn Error>> {
    let mut geometry: VertexBuffers<[f32; 2], u32> = VertexBuffers::new();
    match operation {
        Operation::Fill(rule) => {
            FillTessellator::new().tessellate_path(
                path,
                &FillOptions::default().with_fill_rule(lyon_fill(rule)).with_tolerance(0.1),
                &mut BuffersBuilder::new(&mut geometry, |vertex: FillVertex<'_>| vertex.position().to_array()),
            )?;
        }
        Operation::Stroke(style) => {
            StrokeTessellator::new().tessellate_path(
                path,
                &StrokeOptions::default()
                    .with_line_width(style.width as f32)
                    .with_line_join(lyon_join(style.join))
                    .with_line_cap(lyon_cap(style.cap))
                    .with_miter_limit(style.miter_limit as f32)
                    .with_tolerance(0.1),
                &mut BuffersBuilder::new(&mut geometry, |vertex: StrokeVertex<'_, '_>| vertex.position().to_array()),
            )?;
        }
    }
    Ok(Evidence {
        digest: lyon_digest(&geometry.vertices, &geometry.indices),
        primary_count: geometry.vertices.len(),
        secondary_count: geometry.indices.len(),
        payload_bytes: geometry.vertices.len() * size_of::<[f32; 2]>() + geometry.indices.len() * size_of::<u32>(),
    })
}

fn run_vello(path: &BezPath, operation: Operation, level: Level) -> Result<Evidence, Box<dyn Error>> {
    let mut generator = StripGenerator::new(VIEWPORT, VIEWPORT, level);
    let mut storage = StripStorage::default();
    match operation {
        Operation::Fill(rule) => generator.generate_filled_path(
            path.clone(),
            vello_fill(rule),
            Affine::IDENTITY,
            None,
            &mut storage,
            None,
        ),
        Operation::Stroke(style) => {
            let stroke = Stroke {
                width: style.width,
                join: vello_join(style.join),
                start_cap: vello_cap(style.cap),
                end_cap: vello_cap(style.cap),
                miter_limit: style.miter_limit,
                ..Default::default()
            };
            generator.generate_stroked_path(
                path.clone(),
                &stroke,
                Affine::IDENTITY,
                None,
                &mut storage,
                None,
            );
        }
    }
    Ok(Evidence {
        digest: vello_digest(&storage),
        primary_count: storage.strips.len(),
        secondary_count: storage.alphas.len(),
        payload_bytes: storage.strips.len() * size_of::<vello_common::strip::Strip>() + storage.alphas.len(),
    })
}

fn percentile(samples: &[Duration], pct: usize) -> Duration {
    samples[((samples.len() - 1) * pct + 99) / 100]
}

fn measure<F>(mut run: F) -> Result<(Timing, Evidence), Box<dyn Error>>
where
    F: FnMut() -> Result<Evidence, Box<dyn Error>>,
{
    for _ in 0..WARMUP { black_box(run()?); }
    let expected = run()?;
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        let actual = run()?;
        black_box(actual);
        samples.push(start.elapsed());
        if actual.digest != expected.digest
            || actual.primary_count != expected.primary_count
            || actual.secondary_count != expected.secondary_count
            || actual.payload_bytes != expected.payload_bytes
        {
            return Err("candidate output changed across identical repeated generation".into());
        }
    }
    samples.sort_unstable();
    Ok((
        Timing {
            min_ns: samples[0].as_nanos(),
            median_ns: percentile(&samples, 50).as_nanos(),
            p90_ns: percentile(&samples, 90).as_nanos(),
            max_ns: samples[samples.len() - 1].as_nanos(),
        },
        expected,
    ))
}

fn row(case: &Case, scale: f64, mode: &str, candidate: &str, evidence: Evidence, timing: &Timing, fallback: bool) {
    println!(
        "{}\t{scale:.2}\t{mode}\t{candidate}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{fallback}",
        case.name,
        evidence.digest,
        evidence.primary_count,
        evidence.secondary_count,
        evidence.payload_bytes,
        timing.min_ns,
        timing.median_ns,
        timing.p90_ns,
        timing.max_ns,
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let level = Level::new();
    println!("case\tscale\tmode\tcandidate\tdigest\tprimary_count\tsecondary_count\tpayload_bytes\tmin_ns\tmedian_ns\tp90_ns\tmax_ns\tvello_level_fallback");

    for case in CASES {
        for scale in SCALES {
            let commands = transformed(case.commands, scale, case.translate);
            let lyon = lyon_path(&commands);
            let vello = vello_path(&commands);

            let (timing, evidence) = measure(|| run_lyon(&lyon, case.operation))?;
            row(case, scale, "retained", "lyon_tessellation_1.0.22", evidence, &timing, level.is_fallback());

            let (timing, evidence) = measure(|| run_vello(&vello, case.operation, level))?;
            row(case, scale, "retained", "vello_common_0.3.0_sparse_strips", evidence, &timing, level.is_fallback());

            let (timing, evidence) = measure(|| {
                let rebuilt = lyon_path(&commands);
                run_lyon(&rebuilt, case.operation)
            })?;
            row(case, scale, "rebuild", "lyon_tessellation_1.0.22", evidence, &timing, level.is_fallback());

            let (timing, evidence) = measure(|| {
                let rebuilt = vello_path(&commands);
                run_vello(&rebuilt, case.operation, level)
            })?;
            row(case, scale, "rebuild", "vello_common_0.3.0_sparse_strips", evidence, &timing, level.is_fallback());
        }
    }
    Ok(())
}
