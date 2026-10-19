//! Complete test-only sample-plane preparation, without a diagnostic census.
//! Original-region classification remains distinct from disposable Q24 values.
use super::super::super::gaussian_kernel::fixed::{FixedKernel, batch};
use super::*;
use query_compiler::{BoundaryRequest, Compiler, IntegralRequest};

const WIDTH: f64 = 1.0 / 524_288.0;
const Q24: f64 = 16_777_216.0;
const QUERY: u32 = 1 << 31;
const MAX_QUERIES: usize = 4096;

#[derive(Clone, Copy)]
struct Profile {
    run: usize,
    known: Interval,
    error: f64,
    shared: [Option<(usize, f64)>; 2],
}

struct Assembly {
    samples: Vec<u32>,
    runs: Vec<std::ops::Range<usize>>,
    queries: Vec<batch::Query>,
    profiles: Vec<Profile>,
    profile_index: numeric_index::Index<2>,
    unknown: Vec<(Interval, Interval)>,
    unknown_row: Option<u32>,
    unknown_mass: f64,
    literal_radius: f64,
}

/// Disposable physical output. No source/DAG, classifier, numerical compiler,
/// kernel cache or preparation frontier is retained through GPU execution.
pub(in crate::runtime::execution_2d) struct PreparedPlane {
    samples: Vec<u32>,
    batch: batch::Batch,
    extent: [u32; 2],
    root: [u32; 4],
    scale: f64,
    offset: [f64; 2],
    literal_radius: f64,
}

fn midpoint(value: Interval, extra: f64) -> Result<(f64, f64), AreaQueryError> {
    let middle = value.lo * 0.5 + value.hi * 0.5;
    let radius = (middle - value.lo)
        .abs()
        .max((value.hi - middle).abs())
        .next_up();
    Ok((middle, checked(iv(radius).add(iv(extra)))?.hi))
}

impl Assembly {
    fn fixed_bytes(unknown: usize) -> Option<usize> {
        MAX_QUERIES
            .checked_mul(size_of::<std::ops::Range<usize>>() + size_of::<batch::Query>())?
            .checked_add(512 * size_of::<Profile>())?
            .checked_add(numeric_index::Index::<2>::heap_bound()?)?
            .checked_add(unknown.checked_mul(size_of::<(Interval, Interval)>())?)?
            .checked_add(size_of::<Self>())
    }

    fn new(cells: usize, unknown: usize, fuel: &mut usize) -> Result<Self, AreaQueryError> {
        debit(fuel, 1024)?;
        let result = Self {
            samples: bounded_vector(cells).ok_or(AreaQueryError::Resource)?,
            runs: bounded_vector(MAX_QUERIES).ok_or(AreaQueryError::Resource)?,
            queries: bounded_vector(MAX_QUERIES).ok_or(AreaQueryError::Resource)?,
            profiles: bounded_vector(512).ok_or(AreaQueryError::Resource)?,
            profile_index: numeric_index::Index::new(fuel)?,
            unknown: bounded_vector(unknown).ok_or(AreaQueryError::Resource)?,
            unknown_row: None,
            unknown_mass: 0.0,
            literal_radius: 0.0,
        };
        if result.samples.capacity() != cells
            || result.runs.capacity() != MAX_QUERIES
            || result.queries.capacity() != MAX_QUERIES
            || result.profiles.capacity() != 512
            || result.unknown.capacity() != unknown
        {
            return Err(AreaQueryError::Resource);
        }
        Ok(result)
    }

    fn literal(
        &mut self,
        value: Interval,
        count: usize,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        debit(fuel, 128 + count * 8)?;
        let value = checked(Interval::new(value.lo.max(0.0), value.hi.min(1.0)))?;
        let (middle, _) = midpoint(value, 0.0)?;
        let encoded = (middle * Q24).round_ties_even() as u32;
        let decoded = f64::from(encoded) / Q24;
        let radius = (decoded - value.lo)
            .abs()
            .max((value.hi - decoded).abs())
            .next_up();
        if radius * 2.0 > WIDTH || encoded > Q24 as u32 {
            return Err(AreaQueryError::Precision);
        }
        if self.samples.len() + count > self.samples.capacity() {
            return Err(AreaQueryError::Resource);
        }
        self.literal_radius = self.literal_radius.max(radius);
        self.samples.resize(self.samples.len() + count, encoded);
        Ok(())
    }

    fn query(
        &mut self,
        known: Interval,
        error: f64,
        run: usize,
        negate: bool,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        debit(fuel, 192)?;
        let (base, base_error) = midpoint(known, error)?;
        if self.queries.len() == MAX_QUERIES || self.samples.len() == self.samples.capacity() {
            return Err(AreaQueryError::Resource);
        }
        self.samples.push(QUERY | self.queries.len() as u32);
        self.queries.push(batch::Query {
            base,
            base_error,
            run,
            negate,
        });
        Ok(())
    }

    fn finish_run(
        &mut self,
        start: usize,
        end: usize,
        fuel: &mut usize,
    ) -> Result<usize, AreaQueryError> {
        debit(fuel, 48)?;
        if self.runs.len() == MAX_QUERIES {
            return Err(AreaQueryError::Resource);
        }
        self.runs.push(start..end);
        Ok(self.runs.len() - 1)
    }

    fn row_unknown(
        &mut self,
        strips: &Strips,
        span: &classified_spans::Span<'_>,
        compiler: &mut Compiler<'_>,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        debit(fuel, 32)?;
        if self.unknown_row == Some(span.y) {
            return Ok(());
        }
        // Merged unknown slabs are sorted and disjoint. Prepay both binary
        // searches before comparing, rather than replaying every slab/site.
        debit(fuel, 128 + 64 * (strips.unknown.len() + 1).ilog2() as usize)?;
        let first = strips.unknown.partition_point(|slab| slab.hi <= span.ys.lo);
        let end = strips.unknown.partition_point(|slab| slab.lo < span.ys.hi);
        self.unknown.clear();
        let mut total = iv(0.0);
        for slab in &strips.unknown[first..end] {
            debit(fuel, 128)?;
            let local = checked(Interval::new(
                slab.lo.max(span.ys.lo),
                slab.hi.min(span.ys.hi),
            ))?;
            let mass = compiler.mass(local, span.cy, fuel)?;
            total = checked(total.add(mass))?;
            if self.unknown.len() == self.unknown.capacity() {
                return Err(AreaQueryError::Resource);
            }
            self.unknown.push((local, mass));
        }
        self.unknown_mass = total.hi;
        self.unknown_row = Some(span.y);
        Ok(())
    }

    fn profile(
        &mut self,
        distance: f64,
        slope: f64,
        sigma: f64,
        compiler: &mut Compiler<'_>,
        fuel: &mut usize,
    ) -> Result<(usize, Profile), AreaQueryError> {
        let (index, inserted) = self
            .profile_index
            .observe([distance.to_bits(), slope.to_bits()], fuel)?;
        if inserted {
            debit(fuel, 128)?;
            let start = compiler.nodes.len();
            let (known, error) = if distance == 0.0 {
                (iv(0.5), 0.0)
            } else {
                let halo = checked(iv(sigma).mul(iv(3.0)))?.hi;
                let domain = checked(Interval::new(-halo, halo))?;
                compiler.integral(
                    IntegralRequest {
                        map: MappedBoundary::Line {
                            intercept: distance,
                            slope,
                        },
                        domain,
                        y: domain,
                        center: [0.0; 2],
                        terms: 1,
                        sign: 1.0,
                    },
                    fuel,
                )?
            };
            let run = self.finish_run(start, compiler.nodes.len(), fuel)?;
            if index != self.profiles.len() || self.profiles.len() == 512 {
                return Err(AreaQueryError::Resource);
            }
            self.profiles.push(Profile {
                run,
                known,
                error,
                shared: [None; 2],
            });
        }
        Ok((index, self.profiles[index]))
    }

    fn constants(
        &mut self,
        inside: bool,
        count: usize,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        debit(fuel, 64)?;
        let mass = self.unknown_mass;
        let value = if inside {
            checked(Interval::new((1.0 - mass).next_down().max(0.0), 1.0))?
        } else {
            checked(Interval::new(0.0, mass.min(1.0)))?
        };
        self.literal(value, count, fuel)
    }

    fn line(
        &mut self,
        plan: &ScanPlan,
        span: &classified_spans::Span<'_>,
        boundary: (usize, bool),
        sigma: f64,
        compiler: &mut Compiler<'_>,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        let (id, inside_left) = boundary;
        debit(fuel, 384 + 64 * (span.xs.len() + 1).ilog2() as usize)?;
        let CurveKey::ExactLine(intercept, slope) = plan.curves[id].key else {
            return Err(AreaQueryError::Precision);
        };
        let b = exact_add(
            intercept,
            exact_mul(slope, span.cy).ok_or(AreaQueryError::Precision)?,
        )
        .ok_or(AreaQueryError::Precision)?;
        let halo = checked(iv(sigma).mul(iv(3.0)))?;
        let threshold = checked(halo.mul(checked(iv(1.0).add(iv(slope.abs())))?))?.hi;
        let low = checked(iv(b).sub(iv(threshold)))?.lo;
        let high = checked(iv(b).add(iv(threshold)))?.hi;
        let first = span.xs.partition_point(|&x| x < low);
        let end = span.xs.partition_point(|&x| x <= high);
        self.constants(inside_left, first, fuel)?;
        for &x in &span.xs[first..end] {
            debit(fuel, 256)?;
            let distance = exact_add(b, -x).ok_or(AreaQueryError::Precision)?;
            let (profile_id, profile) =
                self.profile(distance.abs(), slope.abs(), sigma, compiler, fuel)?;
            let negate = (distance < 0.0) != !inside_left;
            let mut known = if negate {
                checked(iv(1.0).sub(profile.known))?
            } else {
                profile.known
            };
            // A conservative initial source interval avoids needless CDF
            // reconstruction. The final Batch checks the ACTUAL combined
            // error; this reserve cannot authorize an inaccurate query.
            if self.unknown_mass + WIDTH / 16.0 <= WIDTH / 2.0 {
                // Identical physical value and orientation share a query;
                // its radius retains the largest source uncertainty observed.
                debit(fuel, 96)?;
                let side = usize::from(negate);
                if let Some((query, base_radius)) = profile.shared[side] {
                    let error = checked(iv(base_radius).add(iv(self.unknown_mass)))?.hi;
                    self.queries[query].base_error = self.queries[query].base_error.max(error);
                    if self.samples.len() == self.samples.capacity() {
                        return Err(AreaQueryError::Resource);
                    }
                    self.samples.push(QUERY | query as u32);
                } else {
                    let (_, base_radius) = midpoint(known, profile.error)?;
                    let error = checked(iv(profile.error).add(iv(self.unknown_mass)))?.hi;
                    let query = self.queries.len();
                    self.query(known, error, profile.run, negate, fuel)?;
                    self.profiles[profile_id].shared[side] = Some((query, base_radius));
                }
                continue;
            } else {
                let mut removed = iv(0.0);
                for &(slab, mass) in &self.unknown {
                    let value = compiler.halfplane_slice(
                        [intercept, slope],
                        slab,
                        [x, span.cy],
                        inside_left,
                        mass,
                        fuel,
                    )?;
                    debit(fuel, 48)?;
                    removed = checked(removed.add(value))?;
                }
                debit(fuel, 96)?;
                known = checked(
                    checked(known.sub(removed))?
                        .add(checked(Interval::new(0.0, self.unknown_mass))?),
                )?;
            }
            self.query(known, profile.error, profile.run, negate, fuel)?;
        }
        self.constants(!inside_left, span.xs.len() - end, fuel)
    }

    fn general(
        &mut self,
        plan: &ScanPlan,
        span: &classified_spans::Span<'_>,
        compiler: &mut Compiler<'_>,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        if span.xs.len() != 1 {
            return Err(AreaQueryError::Precision);
        }
        let center = [span.xs[0], span.cy];
        let mut terms = 0;
        for band in span.bands {
            for run in &span.runs[band.first..band.first + band.count] {
                debit(fuel, 32)?;
                if run.x[1].hi > span.window.lo && run.x[0].lo < span.window.hi {
                    terms += 2;
                }
            }
        }
        let start = compiler.nodes.len();
        let mut known = iv(0.0);
        let mut error = 0.0;
        for band in span.bands {
            for run in &span.runs[band.first..band.first + band.count] {
                debit(fuel, 64)?;
                if run.x[1].hi <= span.window.lo || run.x[0].lo >= span.window.hi {
                    continue;
                }
                for side in 0..2 {
                    let (value, radius) = compiler.curve(
                        plan,
                        BoundaryRequest {
                            id: run.ids[side],
                            y: band.y,
                            center,
                            range: run.x[side],
                            terms,
                            sign: if side == 0 { -1.0 } else { 1.0 },
                        },
                        fuel,
                    )?;
                    debit(fuel, 96)?;
                    known = checked(known.add(value))?;
                    error = checked(iv(error).add(iv(radius)))?.hi;
                }
            }
        }
        debit(fuel, 96)?;
        known = checked(known.add(checked(Interval::new(0.0, self.unknown_mass))?))?;
        let run = self.finish_run(start, compiler.nodes.len(), fuel)?;
        self.query(known, error, run, false, fuel)
    }
}

pub(super) fn prepare(
    strips: &Strips,
    plan: &ScanPlan,
    prepared: &PreparedBoundaries,
    region: &NeutralRegion<'_>,
    lattice: Lattice,
    fuel: &mut usize,
) -> Result<PreparedPlane, AreaQueryError> {
    let before = *fuel;
    let root = lattice.root;
    let sigma = lattice.sigma;
    let scale = lattice.scale;
    let offset = lattice.offset;
    let width = root[2]
        .checked_sub(root[0])
        .ok_or(AreaQueryError::Precision)? as usize;
    let height = root[3]
        .checked_sub(root[1])
        .ok_or(AreaQueryError::Precision)? as usize;
    let cells = width.checked_mul(height).ok_or(AreaQueryError::Resource)?;
    if cells == 0
        || cells > 1_048_576
        || !scale.is_finite()
        || scale <= 0.0
        || !sigma.is_finite()
        || sigma <= 0.0
    {
        return Err(AreaQueryError::Precision);
    }
    let kernel = GaussianKernel::compiled(fuel, MAX_BYTES)?;
    let fixed = FixedKernel::compiled(fuel, MAX_BYTES)?;
    debit(fuel, 512)?;
    debit(fuel, 512)?;
    let source_bytes = storage::query_plans(prepared, plan)
        .and_then(|v| v.checked_add(strips.retained_bytes()?))
        .ok_or(AreaQueryError::Resource)?;
    let compiler_bytes = Compiler::reservation_bytes().ok_or(AreaQueryError::Resource)?;
    let assembly_bytes =
        Assembly::fixed_bytes(strips.unknown.len()).ok_or(AreaQueryError::Resource)?;
    let classifier = classified_spans::scratch_bytes(width).ok_or(AreaQueryError::Resource)?;
    let rows = width
        .checked_mul(size_of::<f64>())
        .ok_or(AreaQueryError::Resource)?;
    let output = cells
        .checked_mul(size_of::<u32>())
        .ok_or(AreaQueryError::Resource)?;
    let retained = source_bytes
        .checked_add(compiler_bytes)
        .and_then(|v| v.checked_add(assembly_bytes))
        .and_then(|v| v.checked_add(kernel.retained_bytes()?))
        .and_then(|v| v.checked_add(2 * size_of::<FixedKernel>()))
        .ok_or(AreaQueryError::Resource)?;
    let fixed_query = retained
        .checked_add(
            classifier
                .checked_sub(rows)
                .ok_or(AreaQueryError::Resource)?,
        )
        // Existing query-scratch allowance for scalar frames, headers and
        // bounded nonrecursive numerical calls; not an allocator/backend cap.
        .and_then(|v| v.checked_add(65_536))
        .ok_or(AreaQueryError::Resource)?;
    let traversal_peak = fixed_query
        .checked_add(rows)
        .and_then(|v| v.checked_add(output))
        .ok_or(AreaQueryError::Resource)?;
    if fixed_query > MAX_AREA_QUERY_FIXED_BYTES as usize || traversal_peak > MAX_BYTES {
        return Err(AreaQueryError::Resource);
    }
    let mut center_bound = [f64::from(root[2]) / scale, f64::from(root[3]) / scale];
    if offset != [0.0; 2] {
        debit(fuel, 256)?;
        for axis in 0..2 {
            let range = checked(
                checked(
                    checked(Interval::new(
                        f64::from(root[axis]),
                        f64::from(root[axis + 2]),
                    ))?
                    .div(iv(scale)),
                )?
                .sub(iv(offset[axis])),
            )?;
            center_bound[axis] = range.lo.abs().max(range.hi.abs());
        }
    }
    let mut compiler = Compiler::new(&kernel, sigma, center_bound, fuel)?;
    let mut assembly = Assembly::new(cells, strips.unknown.len(), fuel)?;
    let traversal = classified_spans::visit(strips, plan, region, lattice, fuel, |span, fuel| {
        debit(fuel, 64)?;
        if span.offset != assembly.samples.len() {
            return Err(AreaQueryError::Precision);
        }
        if span.outside_reach {
            return assembly.literal(iv(0.0), span.xs.len(), fuel);
        }
        assembly.row_unknown(strips, &span, &mut compiler, fuel)?;
        match span.class {
            Class::Inside => assembly.constants(true, span.xs.len(), fuel),
            Class::Outside => assembly.constants(false, span.xs.len(), fuel),
            Class::YOnly => {
                let mut known = iv(0.0);
                for (band, piece) in span.bands.iter().zip(span.pieces) {
                    debit(fuel, 64)?;
                    if matches!(piece, Class::Inside) {
                        known = checked(known.add(compiler.mass(band.y, span.cy, fuel)?))?;
                    }
                }
                debit(fuel, 48)?;
                let value =
                    checked(known.add(checked(Interval::new(0.0, assembly.unknown_mass))?))?;
                assembly.literal(value, span.xs.len(), fuel)
            }
            Class::Line(id, side) => {
                assembly.line(plan, &span, (id, side), sigma, &mut compiler, fuel)
            }
            Class::General => assembly.general(plan, &span, &mut compiler, fuel),
        }
    });
    if let Err(error) = traversal {
        eprintln!(
            "COMPLETE_PLANE traversal rejected {error:?}, samples={} queries={} nodes={} remaining={fuel}",
            assembly.samples.len(),
            assembly.queries.len(),
            compiler.nodes.len()
        );
        return Err(error);
    }
    let scratch = traversal?;
    println!(
        "COMPLETE_PLANE before_batch samples={} queries={} runs={} nodes={} remaining={fuel}",
        assembly.samples.len(),
        assembly.queries.len(),
        assembly.runs.len(),
        compiler.nodes.len()
    );
    if assembly.samples.len() != cells {
        return Err(AreaQueryError::Precision);
    }
    let batch = batch::Batch::new(
        &fixed,
        &compiler.nodes,
        &assembly.runs,
        &assembly.queries,
        fuel,
        MAX_BYTES
            .checked_sub(
                retained
                    .checked_add(output)
                    .ok_or(AreaQueryError::Resource)?,
            )
            .ok_or(AreaQueryError::Resource)?
            .min(
                (MAX_AREA_QUERY_FIXED_BYTES as usize)
                    .checked_sub(retained)
                    .ok_or(AreaQueryError::Resource)?,
            ),
    )?;
    for &error in &batch.errors {
        debit(fuel, 16)?;
        if error * 2.0 > WIDTH {
            return Err(AreaQueryError::Precision);
        }
    }
    debit(fuel, batch.field_preparation_work(cells)?)?;
    println!(
        "COMPLETE_PLANE samples={cells} queries={} runs={} nodes={} work={} remaining={} classifier_heap={} maximum_query_width={} numeric_fixed_bytes={fixed_query} numeric_traversal_peak={traversal_peak} caller_source_liveness_and_GPU_data_are_separate_admissions",
        assembly.queries.len(),
        assembly.runs.len(),
        compiler.nodes.len(),
        before - *fuel,
        *fuel,
        scratch,
        2.0 * batch.errors.iter().copied().fold(0.0, f64::max)
    );
    // Moving only these fields drops the compiler, Assembly caches and kernels
    // on return, before any GPU descriptor exists.
    Ok(PreparedPlane {
        samples: assembly.samples,
        batch,
        extent: [width as u32, height as u32],
        root,
        scale,
        offset,
        literal_radius: assembly.literal_radius,
    })
}

impl PreparedPlane {
    pub(super) fn execute_proof(&self, reflected: bool, clipped: bool) {
        let Self {
            samples,
            batch,
            extent,
            root,
            scale,
            offset,
            literal_radius,
        } = self;
        let [width, height] = extent.map(|v| v as usize);
        let scale = *scale;
        // Pixel-reference translation is used only by the integer-offset
        // fixture. The sample/query compiler itself admits exact fractional
        // translations; those require references at their own sample phase.
        assert!(offset.iter().all(|v| v.fract() == 0.0 && v.abs() < 1024.0));
        // Reflection x -> 64-x preserves Euclidean disks, the finite 3-sigma
        // square, the normalized Gaussian and the globally phased lattice.
        // Apply it to the independently integrated references, never to the
        // rendered field. The F1 ancestor already transformed the whole source.
        let logical_x = |x| if reflected { 64.0 - x } else { x };
        let pixel_x = |x| if reflected { width / 4 - 1 - x } else { x };
        // Final rendering is prepared and submitted before the separate
        // query-value conformance readback. No query readback can be an
        // input to this field or the maintained compositor.
        let (query_input, query_count) = batch.field_data();
        let pixels =
            crate::runtime::execution_2d::lowering::sample_space::gaussian_gpu_proof::execute(
                samples,
                [width as u32, height as u32],
                query_input,
                query_count,
            );
        let values = batch.execute_proof();
        let decoded = |index: usize| {
            let word = samples[index];
            if word & QUERY != 0 {
                let i = (word & !QUERY) as usize;
                (f64::from(values[i]), batch.errors[i])
            } else {
                (f64::from(word) / Q24, *literal_radius)
            }
        };
        for &(point, truth) in reference::SAMPLES {
            // For the clipped fixture these points are farther from y=15
            // than child erosion + finite child/parent halos. Thus the new
            // clip cannot affect their original independent values.
            let x = ((logical_x(point[0]) + offset[0]) * scale - 0.5) as usize - root[0] as usize;
            let y = ((point[1] + offset[1]) * scale - 0.5) as usize - root[1] as usize;
            let (value, radius) = decoded(y * width + x);
            assert!(
                (value - truth).abs() <= radius,
                "reference excluded at {point:?}: {value} +/- {radius}, truth={truth}"
            );
        }
        if clipped {
            // In these two pixels the source is exactly y>=15 throughout
            // the finite kernel window: side/hole edges and the translated
            // child shadow are outside it. Test every globally phased sample,
            // including the exact zero/one finite-cutoff cases.
            for (py, truths) in [
                (14, [0.0, 0.0, 0.0, reference::CLIP_HALFPLANE]),
                (15, [1.0 - reference::CLIP_HALFPLANE, 1.0, 1.0, 1.0]),
            ] {
                for (dy, truth) in truths.into_iter().enumerate() {
                    for dx in 0..4 {
                        let (value, radius) = decoded((py * 4 + dy) * width + 25 * 4 + dx);
                        assert!(
                            (value - truth).abs() <= radius,
                            "independent clip sample {py},{dx},{dy}: {value} +/- {radius}, truth={truth}"
                        );
                    }
                }
            }
        }
        let pixel_references = if clipped { 7 } else { 4 };
        let expected_pixels = [
            ([34, 17], [193_u8, 0, 0, 136]),
            ([23, 18], [184, 0, 0, 122]),
            ([25, if clipped { 16 } else { 12 }], [255, 0, 0, 255]),
            ([47, 40], [0, 0, 0, 0]),
            ([25, 14], [45, 0, 0, 7]),
            ([25, 15], [252, 0, 0, 248]),
            ([25, 12], [0, 0, 0, 0]),
        ];
        // Independently generated full-pixel references. This checks the
        // existing correlated resolve/color law numerically; it is not a
        // substitute for the still-required renderer texture readback.
        for ([px, py], expected) in expected_pixels.into_iter().take(pixel_references) {
            let px = (pixel_x(px) as i64 + offset[0] as i64) as usize;
            let py = (py as i64 + offset[1] as i64) as usize;
            let mut total = 0.0;
            for dy in 0..4 {
                for dx in 0..4 {
                    total += decoded((py * 4 + dy) * width + px * 4 + dx).0;
                }
            }
            let alpha = total / 16.0;
            let red = if alpha <= 0.003_130_8 {
                12.92 * alpha
            } else {
                1.055 * alpha.powf(1.0 / 2.4) - 0.055
            };
            let rgba = [
                (red * 255.0).round() as u8,
                0,
                0,
                (alpha * 255.0).round() as u8,
            ];
            assert_eq!(rgba, expected, "independent complete pixel {px},{py}");
        }
        println!(
            "COMPLETE_PLANE independent_phase_references={} separate_query_GPU_conformance",
            reference::SAMPLES.len() + if clipped { 32 } else { 0 }
        );
        for ([px, py], expected) in expected_pixels.into_iter().take(pixel_references) {
            let px = (pixel_x(px) as i64 + offset[0] as i64) as usize;
            let py = (py as i64 + offset[1] as i64) as usize;
            let pixel_offset = (py * (width / 4) + px) * 4;
            let actual = &pixels[pixel_offset..pixel_offset + 4];
            assert!(
                actual.iter().zip(expected).all(|(a, b)| a.abs_diff(b) <= 2),
                "independent final compositor pixel {px},{py}: {actual:?}, expected {expected:?}"
            );
        }
        println!(
            "COMPLETE_PLANE actual_F3E_final_color_readback_PASS independent_pixels={pixel_references} all_six_work_nodes_completed"
        );
    }
}

#[test]
fn original_constant_regions_prepare_without_numerical_queries() {
    use crate::runtime::execution_2d::shadow::region::NeutralRegionBuilder;
    let item = super::tests::original_ring();
    let mut graph = NeutralRegionBuilder::new(
        shadow::NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        },
        MAX_BYTES as u64,
    );
    let source = graph.register_source(&item, &[0, 0]).unwrap().unwrap();
    // Windows are strictly separated from the original ring boundaries by
    // more than 3 sigma. The first is empty, the second fully covered. The
    // original F1 classifier, not these bounds, decides membership.
    let mut fuel = 16_777_216 - graph.graph_work;
    for (root, expected) in [([0, 0, 8, 8], 0), ([44, 44, 52, 52], 1 << 24)] {
        let before = fuel;
        let plane = source
            .prepare_gaussian_plane(root, 4.0, 0.1, [0.0; 2], &mut fuel, MAX_BYTES)
            .unwrap();
        assert!(fuel < before);
        assert_eq!(plane.samples, vec![expected; 64]);
        assert_eq!(plane.batch.field_data(), (&[][..], 0));
        assert!(plane.batch.errors.is_empty());
        assert!(plane.literal_radius * 2.0 < WIDTH);
    }
}

#[test]
fn gaussian_plane_entry_rejects_before_allocating_and_preserves_shared_fuel() {
    use crate::runtime::execution_2d::shadow::region::NeutralRegionBuilder;
    let item = super::tests::original_ring();
    let mut graph = NeutralRegionBuilder::new(
        shadow::NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        },
        MAX_BYTES as u64,
    );
    let source = graph.register_source(&item, &[0, 0]).unwrap().unwrap();
    let root = [44, 44, 52, 52];
    let mut fuel = 16_777_216 - graph.graph_work;
    let before = fuel;
    // Fixed query storage alone omits the output map and row. The caller's
    // lease must cover the complete phase, even for an all-constant result.
    assert!(matches!(
        source.prepare_gaussian_plane(
            root,
            4.0,
            0.1,
            [0.0; 2],
            &mut fuel,
            MAX_AREA_QUERY_FIXED_BYTES as usize
        ),
        Err(AreaQueryError::Resource)
    ));
    assert_eq!(
        before - fuel,
        1024,
        "reject before source/numerical reconstruction"
    );
    let after_rejection = fuel;
    source
        .prepare_gaussian_plane(root, 4.0, 0.1, [0.0; 2], &mut fuel, MAX_BYTES)
        .unwrap();
    assert!(
        fuel < after_rejection,
        "a second consumer inherits the debit"
    );
    let mut short = 1023;
    assert!(matches!(
        source.prepare_gaussian_plane(root, 4.0, 0.1, [0.0; 2], &mut short, MAX_BYTES),
        Err(AreaQueryError::Resource)
    ));
    assert_eq!(short, 0);
    for (root, scale, sigma) in [
        ([0, 0, 0, 8], 4.0, 0.1),
        ([8, 0, 0, 8], 4.0, 0.1),
        ([0, 0, 8, 8], 0.0, 0.1),
        ([0, 0, 8, 8], 4.0, f64::NAN),
        ([0, 0, 8, 8], 4.0, 0.0),
    ] {
        assert!(matches!(
            source.prepare_gaussian_plane(root, scale, sigma, [0.0; 2], &mut fuel, MAX_BYTES),
            Err(AreaQueryError::Precision)
        ));
    }
    assert!(matches!(
        source.prepare_gaussian_plane(root, 4.0, 0.1, [f64::NAN, 0.0], &mut fuel, MAX_BYTES),
        Err(AreaQueryError::Precision)
    ));
    // Never round a parent offset into a different source or query center.
    assert!(matches!(
        source.prepare_gaussian_plane(
            root,
            4.0,
            0.1,
            [f64::EPSILON / 1024.0, 0.0],
            &mut fuel,
            MAX_BYTES
        ),
        Err(AreaQueryError::Precision)
    ));
}
