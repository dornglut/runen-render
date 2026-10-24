//! Continuous area from source-boundary scan events, not per-cell disk replay.
//!
//! Complete candidate boundaries partition each row into open strips. Original
//! F1 membership chooses occupied strips; interval ordering certifies topology.
//! Closed-form conic integrals then bound area. Ambiguous event bands retain
//! their complete possible area and are refined under one invocation budget.
use super::super::interval::{exact_add, exact_div, exact_mul};
use super::conic_integral::VALUE_WORK;
use super::conic_integral::{Conic, sqrt_enclosure};
use super::*;

mod membership;
use membership::MembershipCache;
mod finite_square;

fn exhausted_or_precision(fuel: &usize) -> AreaQueryError {
    if *fuel == 0 {
        AreaQueryError::Resource
    } else {
        AreaQueryError::Precision
    }
}

const MAX_CURVES: usize = 1_024;
const MAX_BANDS: usize = 8_192;

fn charge(fuel: &mut usize, work: usize) -> Option<()> {
    if *fuel < work {
        *fuel = 0;
        return None;
    }
    *fuel -= work;
    Some(())
}

#[derive(Clone, Copy)]
struct Affine([Interval; 6], usize, Option<[f64; 6]>);

impl Affine {
    fn identity() -> Self {
        let raw = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        Self(raw.map(|v| Interval::scalar(v).unwrap()), 0, Some(raw))
    }
    fn after(
        self,
        transform: Render2dAffineTransform,
        frame: usize,
        fuel: &mut usize,
    ) -> Result<Self, AreaQueryError> {
        let raw = transform.components();
        let identity = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        if raw == identity {
            charge(fuel, 64).ok_or(AreaQueryError::Resource)?;
            return Ok(Self(self.0, frame, self.2));
        }
        if self.2 == Some(identity) {
            charge(fuel, 128).ok_or(AreaQueryError::Resource)?;
            return Ok(Self(
                raw.map(|v| Interval::scalar(v).unwrap()),
                frame,
                Some(raw),
            ));
        }
        // Twelve checked products, eight TwoSum additions and scalar record
        // construction precede any interval-composition fallback.
        charge(fuel, 512).ok_or(AreaQueryError::Resource)?;
        let [a, b, c, d, tx, ty] = self.0;
        let [e, f, g, h, ux, uy] = transform.components().map(|v| Interval::scalar(v).unwrap());
        if let Some(raw) = self.2.and_then(|parent| {
            super::super::interval::exact_affine_compose(parent, transform.components())
        }) {
            return Ok(Self(
                raw.map(|v| Interval::scalar(v).unwrap()),
                frame,
                Some(raw),
            ));
        }
        // Only a failed exact proof needs the outward interval composition.
        charge(fuel, 1280).ok_or(AreaQueryError::Resource)?;
        let compose = || {
            Some(Self(
                [
                    a.mul(e)?.add(c.mul(f)?)?,
                    b.mul(e)?.add(d.mul(f)?)?,
                    a.mul(g)?.add(c.mul(h)?)?,
                    b.mul(g)?.add(d.mul(h)?)?,
                    a.mul(ux)?.add(c.mul(uy)?)?.add(tx)?,
                    b.mul(ux)?.add(d.mul(uy)?)?.add(ty)?,
                ],
                frame,
                None,
            ))
        };
        compose().ok_or(AreaQueryError::Precision)
    }
    fn point(self, [x, y]: [f64; 2]) -> Option<[Interval; 2]> {
        if let Some([a, b, c, d, tx, ty]) = self.2 {
            let point = || {
                Some([
                    exact_add(exact_add(exact_mul(a, x)?, exact_mul(c, y)?)?, tx)?,
                    exact_add(exact_add(exact_mul(b, x)?, exact_mul(d, y)?)?, ty)?,
                ])
            };
            if let Some(p) = point() {
                return Some(p.map(|v| Interval::scalar(v).unwrap()));
            }
        }
        let [a, b, c, d, tx, ty] = self.0;
        let [x, y] = [x, y].map(|v| Interval::scalar(v).unwrap());
        Some([
            a.mul(x)?.add(c.mul(y)?)?.add(tx)?,
            b.mul(x)?.add(d.mul(y)?)?.add(ty)?,
        ])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum CurveKey {
    Vertical(usize, f64),
    Horizontal(usize, f64),
    General(usize, [f64; 2], [f64; 2]),
    Circle(usize, [f64; 2], f64, f64),
    ExactLine(f64, f64),
    ExactConic([f64; 6]),
    ConstructedLine(usize, usize, finite_square_geometry::CoordinateOrigin),
    ConstructedCircle(
        usize,
        [finite_square_geometry::CoordinateOrigin; 2],
        f64,
        f64,
    ),
}
struct Curve {
    function: Conic,
    start: Interval,
    end: Interval,
    key: CurveKey,
    reach: Interval,
}

impl Curve {
    fn same_function(&self, other: &Self) -> bool {
        self.key == other.key
    }
}

#[derive(Clone, Copy)]
struct Crossings {
    complete: bool,
    roots: [Option<Interval>; 2],
}

pub(super) struct ScanPlan {
    curves: Vec<Curve>,
    next_frame: usize,
    crossings: Vec<Crossings>,
    // Curve count and completeness are published only after every unordered
    // pair is constructed. Diagonal entries are implicitly complete-empty.
    crossing_completeness: Option<(usize, bool)>,
    extra_y_events: Vec<Interval>,
}

struct GridQuery<'r, 'a> {
    region: &'r NeutralRegion<'a>,
    forward: Option<Render2dAffineTransform>,
    prepared: &'r PreparedBoundaries,
    xs: Vec<f64>,
    cell_width: Interval,
    vertical_uncertainty: Vec<Interval>,
    semantic_cuts: Vec<f64>,
    curve_cuts: Vec<Vec<f64>>,
    membership: Option<MembershipCache>,
}

struct OccupiedStrip<'c> {
    left: &'c Curve,
    right: &'c Curve,
    left_value: Interval,
    right_value: Interval,
}

impl ScanPlan {
    /// Maximum fixed event-query inventory. The consumer separately admits
    /// width-dependent row storage, output intervals and prepared boundaries.
    pub(super) fn fixed_scratch_bytes() -> Option<usize> {
        let curves = MAX_CURVES.checked_mul(std::mem::size_of::<Curve>())?;
        let crossings = 65_536_usize.checked_mul(std::mem::size_of::<Crossings>())?;
        let band_unit = std::mem::size_of::<Interval>()
            .checked_add(std::mem::size_of::<f64>())?
            .checked_add(std::mem::size_of::<(f64, f64, u8)>())?;
        let bands = MAX_BANDS.checked_mul(band_unit)?;
        let membership = MembershipCache::fixed_scratch_bytes()?;
        let active = MAX_CURVES
            .checked_add(2)?
            .checked_mul(std::mem::size_of::<(usize, &Curve, Interval, f64)>())?;
        let bytes = curves
            .checked_add(crossings)?
            .checked_add(bands)?
            .checked_add(MAX_BANDS.checked_mul(std::mem::size_of::<f64>())?)?
            .checked_add(membership)?
            .checked_add(65_536_usize.checked_mul(4 * std::mem::size_of::<f64>())?)?
            .checked_add(256_usize.checked_mul(std::mem::size_of::<Vec<f64>>())?)?
            .checked_add(active)?;
        let bytes = bytes.checked_add(MAX_BANDS.checked_mul(std::mem::size_of::<Interval>())?)?;
        Some(bytes)
    }

    pub(super) fn grid(
        &self,
        region: &NeutralRegion<'_>,
        root: [u32; 4],
        scale: f64,
        forward: Option<Render2dAffineTransform>,
        prepared: &PreparedBoundaries,
        fuel: &mut usize,
    ) -> Result<Vec<[f64; 2]>, AreaQueryError> {
        let width = usize::try_from(
            root[2]
                .checked_sub(root[0])
                .ok_or(AreaQueryError::Precision)?,
        )
        .map_err(|_| AreaQueryError::Resource)?;
        let height = usize::try_from(
            root[3]
                .checked_sub(root[1])
                .ok_or(AreaQueryError::Precision)?,
        )
        .map_err(|_| AreaQueryError::Resource)?;
        let count = width.checked_mul(height).ok_or(AreaQueryError::Resource)?;
        if count == 0 || count > 1_048_576 {
            return Err(AreaQueryError::Resource);
        }
        if !scale.is_finite() || scale <= 0.0 {
            return Err(AreaQueryError::Precision);
        }
        // Precision qualification precedes optional reach exclusion. The
        // requested global lattice must still be representable when all its
        // geometry is outside the target.
        charge(fuel, 192).ok_or(AreaQueryError::Resource)?;
        let step = exact_div(1.0, scale).ok_or(AreaQueryError::Precision)?;
        charge(fuel, 128).ok_or(AreaQueryError::Resource)?;
        let cell_width = Interval::scalar(step).ok_or(AreaQueryError::Precision)?;
        let cell_area = exact_mul(step, step)
            .and_then(Interval::scalar)
            .or_else(|| cell_width.mul(cell_width))
            .ok_or(AreaQueryError::Precision)?;
        if cell_area.lo <= 0.0 {
            return Err(AreaQueryError::Precision);
        }
        for edge in root {
            exact_div(f64::from(edge), scale).ok_or(AreaQueryError::Precision)?;
        }
        let full_root = root;
        let full_width = width;
        let mut root = root;
        // Original certified reach can exclude cells, never supply interior
        // membership. Restrict expensive analytic events to intersecting cells
        // while retaining the complete requested lattice as exact zero outside.
        if let Some(bounds) = region.certified_logical_envelope() {
            charge(fuel, 1024).ok_or(AreaQueryError::Resource)?;
            let calculate = || -> Option<[u32; 4]> {
                let mut cell = Cell::new(bounds)?;
                if let Some(transform) = forward {
                    let [a, b, c, d, tx, ty] = transform.components().map(Interval::scalar);
                    cell = Cell {
                        x: a?.mul(cell.x)?.add(c?.mul(cell.y)?)?.add(tx?)?,
                        y: b?.mul(cell.x)?.add(d?.mul(cell.y)?)?.add(ty?)?,
                    };
                }
                let scale = Interval::scalar(scale)?;
                let x = cell.x.mul(scale)?;
                let y = cell.y.mul(scale)?;
                let clip = |value: f64, low: u32, high: u32| {
                    value.clamp(f64::from(low), f64::from(high)) as u32
                };
                Some([
                    clip(x.lo.floor(), root[0], root[2]),
                    clip(y.lo.floor(), root[1], root[3]),
                    clip(x.hi.ceil(), root[0], root[2]),
                    clip(y.hi.ceil(), root[1], root[3]),
                ])
            };
            if let Some(crop) = calculate() {
                root = crop;
            }
        }
        let width = (root[2] - root[0]) as usize;
        let height = (root[3] - root[1]) as usize;
        let mut output = bounded_vector(count).ok_or(AreaQueryError::Resource)?;
        charge(fuel, 2 * count).ok_or(AreaQueryError::Resource)?;
        output.resize(count, [0.0, 0.0]);
        if width == 0 || height == 0 {
            return Ok(output);
        }
        let y0 = exact_div(f64::from(root[1]), scale).ok_or(AreaQueryError::Precision)?;
        let y1 = exact_div(f64::from(root[3]), scale).ok_or(AreaQueryError::Precision)?;
        let mut xs = bounded_vector(width.checked_add(1).ok_or(AreaQueryError::Resource)?)
            .ok_or(AreaQueryError::Resource)?;
        for index in root[0]..=root[2] {
            charge(fuel, 64).ok_or(AreaQueryError::Resource)?;
            xs.push(exact_div(f64::from(index), scale).ok_or(AreaQueryError::Precision)?);
        }
        if xs.windows(2).any(|p| p[0] >= p[1]) {
            return Err(AreaQueryError::Precision);
        }
        let mut query = GridQuery {
            region,
            forward,
            prepared,
            xs,
            cell_width,
            vertical_uncertainty: bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?,
            semantic_cuts: bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?,
            curve_cuts: Vec::new(),
            membership: None,
        };
        let mut events = bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?;
        let add = |events: &mut Vec<f64>, y: f64| -> Result<(), AreaQueryError> {
            if y0 < y && y < y1 {
                if events.len() == MAX_BANDS {
                    return Err(AreaQueryError::Resource);
                }
                events.push(y);
            }
            Ok(())
        };
        for curve in &self.curves {
            charge(fuel, 8).ok_or(AreaQueryError::Resource)?;
            for y in [curve.start.lo, curve.start.hi, curve.end.lo, curve.end.hi] {
                add(&mut events, y)?;
            }
            // Interval-valued branch endpoints delimit unresolved event
            // strips. Their complete possible area must enter the bound;
            // subdivision cannot resolve coefficient uncertainty.
            charge(fuel, 64).ok_or(AreaQueryError::Resource)?;
            for endpoint in [curve.start, curve.end] {
                if endpoint.lo < endpoint.hi && endpoint.hi > y0 && endpoint.lo < y1 {
                    if query.vertical_uncertainty.len() == MAX_BANDS {
                        return Err(AreaQueryError::Resource);
                    }
                    query.vertical_uncertainty.push(endpoint);
                }
            }
        }
        for &event in &self.extra_y_events {
            charge(fuel, 16).ok_or(AreaQueryError::Resource)?;
            add(&mut events, event.lo)?;
            add(&mut events, event.hi)?;
            if event.lo < event.hi && event.hi > y0 && event.lo < y1 {
                if query.vertical_uncertainty.len() == MAX_BANDS {
                    return Err(AreaQueryError::Resource);
                }
                query.vertical_uncertainty.push(event);
            }
        }
        charge(fuel, events.len()).ok_or(AreaQueryError::Resource)?;
        query.semantic_cuts.extend_from_slice(&events);
        for crossing in &self.crossings {
            charge(fuel, 64).ok_or(AreaQueryError::Resource)?;
            for root in crossing.roots.into_iter().flatten() {
                add(&mut events, root.lo)?;
                add(&mut events, root.hi)?;
                if root.hi > y0 && root.lo < y1 {
                    if query.vertical_uncertainty.len() == MAX_BANDS {
                        return Err(AreaQueryError::Resource);
                    }
                    query.vertical_uncertainty.push(root);
                }
            }
        }
        // Every physical vertical-edge crossing is prepared ONCE for all
        // rows. Each ensuing band fixes both strip topology and cell routing.
        for curve in &self.curves {
            for (column, &x) in query.xs.iter().enumerate() {
                charge(fuel, 4).ok_or(AreaQueryError::Resource)?;
                if x < curve.reach.lo || x > curve.reach.hi {
                    continue;
                }
                let edge = Self::constant_curve(x, y0, y1).ok_or(AreaQueryError::Precision)?;
                charge(fuel, 128 + Self::crossing_work(curve, &edge))
                    .ok_or(AreaQueryError::Resource)?;
                let crossing = Self::crossings(curve, &edge).ok_or(AreaQueryError::Precision)?;

                for root in crossing.roots.into_iter().flatten() {
                    add(&mut events, root.lo)?;
                    add(&mut events, root.hi)?;
                    if column == 0 || column + 1 == query.xs.len() {
                        for y in [root.lo, root.hi] {
                            if y0 < y && y < y1 {
                                if query.semantic_cuts.len() == MAX_BANDS {
                                    return Err(AreaQueryError::Resource);
                                }
                                query.semantic_cuts.push(y);
                            }
                        }
                    }
                    if root.hi > y0 && root.lo < y1 {
                        if query.vertical_uncertainty.len() == MAX_BANDS {
                            return Err(AreaQueryError::Resource);
                        }
                        query.vertical_uncertainty.push(root);
                    }
                }
            }
        }
        sort_admitted(&mut events, f64::total_cmp, 4, fuel).ok_or(AreaQueryError::Resource)?;
        events.dedup();
        sort_admitted(&mut query.semantic_cuts, f64::total_cmp, 4, fuel)
            .ok_or(AreaQueryError::Resource)?;
        query.semantic_cuts.dedup();
        // Between complete source-topology events, an open strip between
        // adjacent candidate branches is connected and has constant original
        // membership. Raster rows/vertical grid edges only subdivide routing.
        // An incomplete crossing inventory cannot license this query reuse.
        if self.complete_crossings(fuel)? {
            let count = self.curves.len();
            query.curve_cuts = bounded_vector(count).ok_or(AreaQueryError::Resource)?;
            for id in 0..count {
                let mut cuts = bounded_vector(4 * count).ok_or(AreaQueryError::Resource)?;
                for other in 0..count {
                    charge(fuel, 64).ok_or(AreaQueryError::Resource)?;
                    let crossing = self.crossing(id, other).ok_or(AreaQueryError::Precision)?;
                    for root in crossing.roots.into_iter().flatten() {
                        if root.hi > y0 && root.lo < y1 {
                            cuts.extend([root.lo, root.hi]);
                        }
                    }
                }
                sort_admitted(&mut cuts, f64::total_cmp, 4, fuel)
                    .ok_or(AreaQueryError::Resource)?;
                cuts.dedup();
                query.curve_cuts.push(cuts);
            }
            query.membership = Some(MembershipCache::new(fuel)?);
        }
        sort_admitted(
            &mut query.vertical_uncertainty,
            |a, b| a.lo.total_cmp(&b.lo).then(a.hi.total_cmp(&b.hi)),
            8,
            fuel,
        )
        .ok_or(AreaQueryError::Resource)?;
        let mut retained = 0_usize;
        for index in 0..query.vertical_uncertainty.len() {
            let root = query.vertical_uncertainty[index];
            if retained > 0 && root.lo <= query.vertical_uncertainty[retained - 1].hi {
                query.vertical_uncertainty[retained - 1].hi =
                    query.vertical_uncertainty[retained - 1].hi.max(root.hi);
            } else {
                query.vertical_uncertainty[retained] = root;
                retained += 1;
            }
        }
        query.vertical_uncertainty.truncate(retained);
        let mut pending = bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?;
        let mut staged = bounded_vector(width).ok_or(AreaQueryError::Resource)?;
        staged.resize(
            width,
            Interval::scalar(0.0).ok_or(AreaQueryError::Precision)?,
        );
        let mut visited = 0_usize;
        for row_index in 0..height {
            charge(fuel, 128).ok_or(AreaQueryError::Resource)?;
            let output_row = (root[1] - full_root[1]) as usize + row_index;
            let output_start = output_row * full_width + (root[0] - full_root[0]) as usize;
            let row = u32::try_from(row_index).map_err(|_| AreaQueryError::Resource)?;
            let lo = exact_div(f64::from(root[1] + row), scale).ok_or(AreaQueryError::Precision)?;
            let hi =
                exact_div(f64::from(root[1] + row + 1), scale).ok_or(AreaQueryError::Precision)?;
            if lo >= hi {
                return Err(AreaQueryError::Precision);
            }
            let start = events.partition_point(|v| *v <= lo);
            let end = events.partition_point(|v| *v < hi);
            let bands = end
                .checked_sub(start)
                .and_then(|n| n.checked_add(1))
                .ok_or(AreaQueryError::Resource)?;
            if bands > MAX_BANDS {
                return Err(AreaQueryError::Resource);
            }
            charge(fuel, bands.checked_mul(8).ok_or(AreaQueryError::Resource)?)
                .ok_or(AreaQueryError::Resource)?;
            let mut previous = lo;
            for &cut in &events[start..end] {
                pending.push((previous, cut, 0_u8));
                previous = cut;
            }
            pending.push((previous, hi, 0_u8));
            while let Some((a, b, depth)) = pending.pop() {
                visited = visited.checked_add(1).ok_or(AreaQueryError::Resource)?;
                // One scene-wide frontier, never a reset allowance per row.
                if visited > MAX_BANDS * 16 {
                    return Err(AreaQueryError::Resource);
                }
                charge(fuel, width + 128).ok_or(AreaQueryError::Resource)?;
                staged.fill(Interval::scalar(0.0).ok_or(AreaQueryError::Precision)?);
                let known = match self.row_band(&mut query, a, b, &mut staged, fuel) {
                    Ok(()) => true,
                    Err(AreaQueryError::Precision) => false,
                    Err(error) => return Err(error),
                };
                if *fuel == 0 {
                    return Err(AreaQueryError::Resource);
                }
                let uncertainty = query
                    .vertical_uncertainty
                    .partition_point(|root| root.hi <= a);
                let unresolved_event = query
                    .vertical_uncertainty
                    .get(uncertainty)
                    .is_some_and(|root| root.lo <= a && root.hi >= b);
                // Subdivision cannot tighten a certified root's coefficient
                // uncertainty. Retain this entire event strip once; the final
                // per-cell precision gate decides whether it is admissible.
                if !known
                    && !unresolved_event
                    && depth < 32
                    && b - a > (hi - lo) / (2048.0 * 256.0 * MAX_BANDS as f64)
                {
                    let middle = a * 0.5 + b * 0.5;
                    if !(a < middle && middle < b) {
                        return Err(AreaQueryError::Precision);
                    }
                    if pending.len() + 2 > MAX_BANDS {
                        return Err(AreaQueryError::Resource);
                    }
                    pending.extend([(a, middle, depth + 1), (middle, b, depth + 1)]);
                    continue;
                }
                // Every lattice coordinate is qualified as the exact quotient
                // of its integer index and scale. Thus all complete cells have
                // this ONE exact logical width, independent of translation.
                // The unresolved band bound is shared across all its columns.
                let unresolved_mass = if known {
                    None
                } else {
                    charge(fuel, 128).ok_or(AreaQueryError::Resource)?;
                    let possible = cell_width
                        .mul(
                            Interval::scalar(b)
                                .and_then(|v| v.sub(Interval::scalar(a)?))
                                .ok_or(AreaQueryError::Precision)?,
                        )
                        .ok_or(AreaQueryError::Precision)?;
                    Some(Interval::new(0.0, possible.hi).ok_or(AreaQueryError::Precision)?)
                };
                for (column, &mass) in staged.iter().enumerate() {
                    charge(fuel, 32).ok_or(AreaQueryError::Resource)?;
                    let mass = unresolved_mass.unwrap_or(mass);
                    let index = output_start + column;
                    let prior = Interval::new(output[index][0], output[index][1])
                        .ok_or(AreaQueryError::Precision)?;
                    let updated = prior.add(mass).ok_or(AreaQueryError::Precision)?;
                    output[index] = [updated.lo, updated.hi];
                }
            }
            for value in &mut output[output_start..output_start + width] {
                // Normalization includes the generic four-quotient interval
                // division, finite reductions, clamping and precision checks.
                charge(fuel, 64).ok_or(AreaQueryError::Resource)?;
                let fraction = Interval::new(value[0], value[1])
                    .and_then(|v| v.div(cell_area))
                    .ok_or(AreaQueryError::Precision)?;
                *value = [fraction.lo.max(0.0), fraction.hi.min(1.0)];
                if value[0] > value[1] || (value[1] - value[0]).next_up() > 1.0 / (2048.0 * 256.0) {
                    return Err(AreaQueryError::Precision);
                }
            }
        }
        Ok(output)
    }

    fn constant_curve(x: f64, y0: f64, y1: f64) -> Option<Curve> {
        Some(Curve {
            function: Conic {
                h: Interval::scalar(x)?,
                a: Interval::scalar(0.0)?,
                b: Interval::scalar(0.0)?,
                k: Interval::scalar(0.0)?,
                r: Interval::scalar(1.0)?,
                c: Interval::scalar(1.0)?,
            },
            start: Interval::scalar(y0)?,
            end: Interval::scalar(y1)?,
            key: CurveKey::ExactLine(x, 0.0),
            reach: Interval::scalar(x)?,
        })
    }

    fn row_band(
        &self,
        query: &mut GridQuery<'_, '_>,
        y0: f64,
        y1: f64,
        staged: &mut [Interval],
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        charge(fuel, 32).ok_or(AreaQueryError::Resource)?;
        let uncertain = query
            .vertical_uncertainty
            .partition_point(|root| root.hi <= y0);
        if query
            .vertical_uncertainty
            .get(uncertain)
            .is_some_and(|root| root.lo < y1)
        {
            return Err(AreaQueryError::Precision);
        }
        let x0 = *query.xs.first().ok_or(AreaQueryError::Precision)?;
        let x1 = *query.xs.last().ok_or(AreaQueryError::Precision)?;
        let left = Self::constant_curve(x0, y0, y1).ok_or(AreaQueryError::Precision)?;
        let right = Self::constant_curve(x1, y0, y1).ok_or(AreaQueryError::Precision)?;
        let ym = y0 * 0.5 + y1 * 0.5;
        if !(y0 < ym && ym < y1) {
            return Err(AreaQueryError::Precision);
        }
        let ys = Interval::new(y0, y1).ok_or(AreaQueryError::Precision)?;
        let mut active = bounded_vector(
            self.curves
                .len()
                .checked_add(2)
                .ok_or(AreaQueryError::Resource)?,
        )
        .ok_or(AreaQueryError::Resource)?;
        for (id, curve) in self
            .curves
            .iter()
            .enumerate()
            .chain([(self.curves.len(), &left), (self.curves.len() + 1, &right)])
        {
            charge(fuel, 16).ok_or(AreaQueryError::Resource)?;
            if curve.reach.hi < x0
                || curve.reach.lo > x1
                || y1 <= curve.start.lo
                || y0 >= curve.end.hi
            {
                continue;
            }
            if y0 < curve.start.hi || y1 > curve.end.lo {
                return Err(AreaQueryError::Precision);
            }
            charge(fuel, VALUE_WORK + 16 * active.len()).ok_or(AreaQueryError::Resource)?;
            let value = curve
                .function
                .value(Interval::scalar(ym).ok_or(AreaQueryError::Precision)?)
                .ok_or(AreaQueryError::Precision)?;
            if !active
                .iter()
                .any(|&(_, old, _, _): &(usize, &Curve, Interval, f64)| old.same_function(curve))
            {
                charge(fuel, 4).ok_or(AreaQueryError::Resource)?;
                active.push((id, curve, value, value.lo * 0.5 + value.hi * 0.5));
            }
        }
        sort_admitted(&mut active, |a, b| a.3.total_cmp(&b.3), 4, fuel)
            .ok_or(AreaQueryError::Resource)?;
        let mut occupied: Option<OccupiedStrip<'_>> = None;
        for pair in active.windows(2) {
            charge(fuel, 128).ok_or(AreaQueryError::Resource)?;
            let [(lid, l, lx, _), (rid, r, rx, _)] = [pair[0], pair[1]];
            if lx.hi >= rx.lo {
                return Err(AreaQueryError::Precision);
            }
            let crossing = if lid < self.curves.len() && rid < self.curves.len() {
                self.crossing(lid, rid).ok_or(AreaQueryError::Precision)?
            } else {
                charge(fuel, Self::crossing_work(l, r)).ok_or(AreaQueryError::Resource)?;
                Self::crossings(l, r).ok_or(AreaQueryError::Precision)?
            };
            if !(crossing.complete
                && crossing
                    .roots
                    .into_iter()
                    .flatten()
                    .all(|v| v.hi <= y0 || v.lo >= y1))
            {
                charge(fuel, 2 * VALUE_WORK).ok_or(AreaQueryError::Resource)?;
                if l.function.value(ys).ok_or(AreaQueryError::Precision)?.hi
                    >= r.function.value(ys).ok_or(AreaQueryError::Precision)?.lo
                {
                    return Err(AreaQueryError::Precision);
                }
            }
            let x = lx.hi * 0.5 + rx.lo * 0.5;
            if x < x0 || x > x1 {
                continue;
            }
            charge(fuel, 128).ok_or(AreaQueryError::Resource)?;
            let mut point = Cell::new([x, ym, x, ym]).ok_or(AreaQueryError::Precision)?;
            if let Some(t) = query.forward {
                charge(fuel, preimage_work(t)).ok_or(AreaQueryError::Resource)?;
                point = point.preimage(t).ok_or(AreaQueryError::Precision)?;
            }
            match query.membership([lid, rid], point, ym, fuel)? {
                CellCertificate::Outside => {
                    if let Some(strip) = occupied.take() {
                        Self::route_occupied_strip(query, strip, [y0, y1], staged, fuel)?;
                    }
                    continue;
                }
                CellCertificate::Unknown => return Err(exhausted_or_precision(fuel)),
                CellCertificate::Inside => {}
            }
            // Every candidate interval keeps its complete ordering and
            // original-source membership proof. Consecutive Inside strips
            // form one exact occupied run; internal false candidate branches
            // need no repeated conic integration or sample-cell routing.
            charge(fuel, 32).ok_or(AreaQueryError::Resource)?;
            if let Some(strip) = occupied.as_mut() {
                strip.right = r;
                strip.right_value = rx;
            } else {
                occupied = Some(OccupiedStrip {
                    left: l,
                    right: r,
                    left_value: lx,
                    right_value: rx,
                });
            }
        }
        if let Some(strip) = occupied {
            Self::route_occupied_strip(query, strip, [y0, y1], staged, fuel)?;
        }
        Ok(())
    }

    fn route_occupied_strip(
        query: &GridQuery<'_, '_>,
        strip: OccupiedStrip<'_>,
        [y0, y1]: [f64; 2],
        staged: &mut [Interval],
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        charge(fuel, 256).ok_or(AreaQueryError::Resource)?;
        let OccupiedStrip {
            left: l,
            right: r,
            left_value: lx,
            right_value: rx,
        } = strip;
        let x1 = *query.xs.last().ok_or(AreaQueryError::Precision)?;
        let column = |value: Interval| -> Option<usize> {
            if value.lo == value.hi && value.lo == x1 {
                return Some(query.xs.len() - 2);
            }
            let a = query.xs.partition_point(|edge| *edge <= value.lo);
            let b = query.xs.partition_point(|edge| *edge <= value.hi);
            (a == b && a > 0 && a < query.xs.len()).then(|| a - 1)
        };
        let lc = column(lx).ok_or(AreaQueryError::Precision)?;
        let rc = column(rx).ok_or(AreaQueryError::Precision)?;
        if lc > rc {
            return Err(AreaQueryError::Precision);
        }
        if lc == rc {
            let mass = Self::difference_integral(l, r, y0, y1, fuel)
                .ok_or_else(|| exhausted_or_precision(fuel))?;
            staged[lc] = staged[lc].add(mass).ok_or(AreaQueryError::Precision)?;
        } else {
            let edge =
                Self::constant_curve(query.xs[lc + 1], y0, y1).ok_or(AreaQueryError::Precision)?;
            let mass = Self::difference_integral(l, &edge, y0, y1, fuel)
                .ok_or_else(|| exhausted_or_precision(fuel))?;
            staged[lc] = staged[lc].add(mass).ok_or(AreaQueryError::Precision)?;
            let edge =
                Self::constant_curve(query.xs[rc], y0, y1).ok_or(AreaQueryError::Precision)?;
            let mass = Self::difference_integral(&edge, r, y0, y1, fuel)
                .ok_or_else(|| exhausted_or_precision(fuel))?;
            staged[rc] = staged[rc].add(mass).ok_or(AreaQueryError::Precision)?;
            // Full interior columns share one strip-height/width product.
            // Only accumulation remains per column; boundary columns above
            // retain their independent analytic integrals.
            charge(fuel, 128).ok_or(AreaQueryError::Resource)?;
            let area = query
                .cell_width
                .mul(
                    Interval::scalar(y1)
                        .and_then(|v| v.sub(Interval::scalar(y0)?))
                        .ok_or(AreaQueryError::Precision)?,
                )
                .ok_or(AreaQueryError::Precision)?;
            for mass in staged.iter_mut().take(rc).skip(lc + 1) {
                charge(fuel, 32).ok_or(AreaQueryError::Resource)?;
                *mass = mass.add(area).ok_or(AreaQueryError::Precision)?;
            }
        }
        Ok(())
    }
    fn frame(&mut self) -> Option<usize> {
        let id = self.next_frame;
        self.next_frame = self.next_frame.checked_add(1)?;
        Some(id)
    }
    pub(super) fn prepare(
        region: &NeutralRegion<'_>,
        forward: Option<Render2dAffineTransform>,
        prepared: &PreparedBoundaries,
        fuel: &mut usize,
    ) -> Result<Self, AreaQueryError> {
        let mut result = Self {
            curves: bounded_vector(MAX_CURVES).ok_or(AreaQueryError::Resource)?,
            next_frame: 1,
            crossings: Vec::new(),
            crossing_completeness: None,
            extra_y_events: bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?,
        };
        let affine = match forward {
            Some(t) => Affine::identity().after(
                t,
                result.frame().ok_or(AreaQueryError::Resource)?,
                fuel,
            )?,
            None => Affine::identity(),
        };
        result.region(region, affine, prepared, fuel)?;
        result.prepare_crossings(fuel)?;
        Ok(result)
    }

    /// Store each unordered pair once, in upper-triangular row order. No
    /// default-filled square table or duplicate symmetric writes are needed.
    /// Failed construction never publishes a completeness receipt.
    fn prepare_crossings(&mut self, fuel: &mut usize) -> Result<(), AreaQueryError> {
        self.crossing_completeness = None;
        charge(fuel, 32).ok_or(AreaQueryError::Resource)?;
        let curves = self.curves.len();
        // Keep the existing square-inventory admission limit unchanged even
        // though the new storage uses fewer entries.
        if curves.checked_mul(curves).ok_or(AreaQueryError::Resource)? > 65_536 {
            return Err(AreaQueryError::Resource);
        }
        let count = curves * curves.saturating_sub(1) / 2;
        self.crossings = bounded_vector(count).ok_or(AreaQueryError::Resource)?;
        let mut complete = true;
        for left in 0..curves {
            for right in left + 1..curves {
                let l = &self.curves[left];
                let r = &self.curves[right];
                // Complete conservative branch domains/reach prove absence of
                // a common point. No approximate curve identity is needed.
                charge(fuel, 128).ok_or(AreaQueryError::Resource)?;
                let disjoint = l.reach.hi < r.reach.lo
                    || r.reach.hi < l.reach.lo
                    || l.end.hi < r.start.lo
                    || r.end.hi < l.start.lo;
                let crossing = if disjoint {
                    Crossings {
                        complete: true,
                        roots: [None, None],
                    }
                } else {
                    charge(fuel, Self::crossing_work(l, r)).ok_or(AreaQueryError::Resource)?;
                    Self::crossings(l, r).unwrap_or(Crossings {
                        complete: false,
                        roots: [None, None],
                    })
                };
                // Prepay the record's words, capacity/length update and
                // completeness update, including proved-empty pairs.
                charge(fuel, 16).ok_or(AreaQueryError::Resource)?;
                complete &= crossing.complete;
                self.crossings.push(crossing);
            }
        }
        self.crossing_completeness = Some((curves, complete));
        Ok(())
    }

    /// Constant-time lookup of the constructor-owned completeness receipt.
    /// Dimension checks also reject failed, partial or changed inventories.
    /// No consumer repeats an O(n^2) completeness scan.
    fn complete_crossings(&self, fuel: &mut usize) -> Result<bool, AreaQueryError> {
        charge(fuel, 16).ok_or(AreaQueryError::Resource)?;
        let (count, complete) = self
            .crossing_completeness
            .ok_or(AreaQueryError::Precision)?;
        if self.curves.len() != count || self.crossings.len() != count * count.saturating_sub(1) / 2
        {
            return Err(AreaQueryError::Precision);
        }
        Ok(complete)
    }

    /// Symmetric lookup in the packed immutable table. Callers prepay the
    /// bounded indexing arithmetic along with their per-pair query work.
    fn crossing(&self, left: usize, right: usize) -> Option<Crossings> {
        let (count, _) = self.crossing_completeness?;
        if count != self.curves.len() || left >= count || right >= count {
            return None;
        }
        if left == right {
            return Some(Crossings {
                complete: true,
                roots: [None, None],
            });
        }
        let (left, right) = if left < right {
            (left, right)
        } else {
            (right, left)
        };
        let offset = left * (2 * count - left - 1) / 2 + right - left - 1;
        self.crossings.get(offset).copied()
    }

    fn push(&mut self, mut curve: Curve, fuel: &mut usize) -> Result<(), AreaQueryError> {
        charge(fuel, 16 * self.curves.len() + 512).ok_or(AreaQueryError::Resource)?;
        let reach = (|| {
            let f = &curve.function;
            let ys = Interval::new(curve.start.lo, curve.end.hi)?;
            let base = f.h.add(f.a.mul(ys.sub(f.k)?)?)?;
            if f.c.lo >= 0.0 && f.r.lo >= 0.0 {
                // The radical is nonnegative and no larger than r. Preserve
                // the admitted branch sign instead of adding a fictitious
                // opposite half-circle to its conservative reach.
                base.add(f.b.mul(Interval::new(0.0, f.r.hi)?)?)
            } else {
                let radius = f.b.lo.abs().max(f.b.hi.abs());
                let margin = Interval::scalar(radius)?.mul(f.r)?.hi;
                base.add(Interval::new(-margin, margin)?)
            }
        })()
        .ok_or(AreaQueryError::Precision)?;
        curve.reach = reach;
        // Exact equality is a geometric identity, never approximate dedup.
        if self.curves.iter().any(|old| {
            old.same_function(&curve)
                && old.start.lo == old.start.hi
                && old.end.lo == old.end.hi
                && curve.start.lo == curve.start.hi
                && curve.end.lo == curve.end.hi
                && old.start.lo == curve.start.lo
                && old.end.lo == curve.end.lo
        }) {
            return Ok(());
        }
        if self.curves.len() == MAX_CURVES {
            return Err(AreaQueryError::Resource);
        }
        self.curves.push(curve);
        Ok(())
    }

    fn line(
        &mut self,
        p: [f64; 2],
        q: [f64; 2],
        affine: Affine,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        // Includes both point proof/fallbacks, slope/intercept, exact key and
        // Curve record. Insertion comparisons are admitted separately.
        charge(fuel, 3072).ok_or(AreaQueryError::Resource)?;
        let curve = (|| -> Option<Option<Curve>> {
            let [px, py] = affine.point(p)?;
            let [qx, qy] = affine.point(q)?;
            let [a, b, c, d, tx, ty] = affine.0;
            let zero = |v: Interval| v.lo == 0.0 && v.hi == 0.0;
            if (p[0] == q[0] && zero(d)) || (p[1] == q[1] && zero(b)) {
                return Some(None);
            }
            let (start, end) = if py.hi < qy.lo {
                (py, qy)
            } else if qy.hi < py.lo {
                (qy, py)
            } else {
                return None;
            };
            let (slope, intercept) = if p[0] == q[0] {
                let slope = c.div(d)?;
                let x = Interval::scalar(p[0])?;
                (
                    slope,
                    a.sub(slope.mul(b)?)?.mul(x)?.add(tx)?.sub(slope.mul(ty)?)?,
                )
            } else if p[1] == q[1] {
                let slope = a.div(b)?;
                let y = Interval::scalar(p[1])?;
                (
                    slope,
                    c.sub(slope.mul(d)?)?.mul(y)?.add(tx)?.sub(slope.mul(ty)?)?,
                )
            } else {
                let slope = qx.sub(px)?.div(qy.sub(py)?)?;
                (slope, px.sub(slope.mul(py)?)?)
            };
            let key = if p[0] == q[0] {
                CurveKey::Vertical(affine.1, p[0])
            } else if p[1] == q[1] {
                CurveKey::Horizontal(affine.1, p[1])
            } else {
                CurveKey::General(affine.1, p, q)
            };
            let exact = || {
                let [a, b, c, d, tx, ty] = affine.2?;
                let (slope, intercept) = if p[0] == q[0] {
                    let slope = exact_div(c, d)?;
                    (
                        slope,
                        exact_add(
                            exact_add(exact_mul(exact_add(a, -exact_mul(slope, b)?)?, p[0])?, tx)?,
                            -exact_mul(slope, ty)?,
                        )?,
                    )
                } else if p[1] == q[1] {
                    let slope = exact_div(a, b)?;
                    (
                        slope,
                        exact_add(
                            exact_add(exact_mul(exact_add(c, -exact_mul(slope, d)?)?, p[1])?, tx)?,
                            -exact_mul(slope, ty)?,
                        )?,
                    )
                } else {
                    return None;
                };
                Some((
                    CurveKey::ExactLine(intercept, slope),
                    Interval::scalar(slope)?,
                    Interval::scalar(intercept)?,
                ))
            };
            let (key, slope, intercept) = exact().unwrap_or((key, slope, intercept));
            Some(Some(Curve {
                key,
                reach: Interval::scalar(0.0)?,
                function: Conic {
                    h: intercept,
                    a: slope,
                    b: Interval::scalar(0.0)?,
                    k: Interval::scalar(0.0)?,
                    r: Interval::scalar(1.0)?,
                    c: Interval::scalar(1.0)?,
                },
                start,
                end,
            }))
        })()
        .ok_or(AreaQueryError::Precision)?;
        if let Some(curve) = curve {
            self.push(curve, fuel)?;
        }
        Ok(())
    }

    fn circle(
        &mut self,
        center: [f64; 2],
        radius: f64,
        affine: Affine,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        // Includes the generic inverse-quadratic map and both branch records.
        charge(fuel, 4096).ok_or(AreaQueryError::Resource)?;
        let curves = (|| -> Option<[Curve; 2]> {
            let [h, k] = affine.point(center)?;
            if let Some([a, 0.0, c, d, _, _]) = affine.2
                && a.abs() == 1.0
                && d.abs() == 1.0
                && h.lo == h.hi
                && k.lo == k.hi
            {
                let slope = c / d;
                let start = Interval::scalar(exact_add(k.lo, -radius)?)?;
                let end = Interval::scalar(exact_add(k.lo, radius)?)?;
                let branch = |sign| -> Option<Curve> {
                    let raw = [h.lo, slope, sign, k.lo, radius, 1.0];
                    let [h, a, b, k, r, c] = raw.map(|v| Interval::scalar(v).unwrap());
                    Some(Curve {
                        function: Conic { h, a, b, k, r, c },
                        start,
                        end,
                        key: CurveKey::ExactConic(raw),
                        reach: Interval::scalar(0.0)?,
                    })
                };
                return Some([branch(-1.0)?, branch(1.0)?]);
            }
            let [a, b, c, d, _, _] = affine.0;
            let determinant = a.mul(d)?.sub(b.mul(c)?)?;
            let determinant2 = squared_interval(determinant)?;
            let qxx = squared_interval(d)?
                .add(squared_interval(b)?)?
                .div(determinant2)?;
            let qxy = d
                .mul(c)?
                .add(b.mul(a)?)?
                .mul(Interval::scalar(-1.0)?)?
                .div(determinant2)?;
            let detq = Interval::scalar(1.0)?.div(determinant2)?;
            let slope = qxy.mul(Interval::scalar(-1.0)?)?.div(qxx)?;
            let radial = Interval::scalar(radius)?.mul(sqrt_enclosure(qxx)?)?;
            let height = Interval::scalar(radius)?.mul(sqrt_enclosure(
                squared_interval(b)?.add(squared_interval(d)?)?,
            )?)?;
            let start = k.sub(height)?;
            let end = k.add(height)?;
            let coefficient = Interval::scalar(1.0)?.div(qxx)?;
            let branch = |sign| -> Option<Curve> {
                Some(Curve {
                    key: CurveKey::Circle(affine.1, center, radius, sign),
                    reach: Interval::scalar(0.0)?,
                    function: Conic {
                        h,
                        a: slope,
                        b: coefficient.mul(Interval::scalar(sign)?)?,
                        k,
                        r: radial,
                        c: detq,
                    },
                    start,
                    end,
                })
            };
            Some([branch(-1.0)?, branch(1.0)?])
        })()
        .ok_or(AreaQueryError::Precision)?;
        for curve in curves {
            self.push(curve, fuel)?;
        }
        Ok(())
    }

    fn capsule(
        &mut self,
        [p, q]: [[f64; 2]; 2],
        radius: f64,
        affine: Affine,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        self.circle(p, radius, affine, fuel)?;
        self.circle(q, radius, affine, fuel)?;
        for sign in [-1.0, 1.0] {
            let (u, v) = if p[0] == q[0] {
                ([p[0] + sign * radius, p[1]], [q[0] + sign * radius, q[1]])
            } else {
                ([p[0], p[1] + sign * radius], [q[0], q[1] + sign * radius])
            };
            // These offset events require exact representability; a rounded
            // parallel edge would invalidate the complete-boundary proof.
            let coordinate = if p[0] == q[0] { p[0] } else { p[1] };
            let sum = coordinate + sign * radius;
            let vb = sum - coordinate;
            if !sum.is_finite() || (coordinate - (sum - vb)) + (sign * radius - vb) != 0.0 {
                return Err(AreaQueryError::Precision);
            }
            self.line(u, v, affine, fuel)?;
        }
        Ok(())
    }

    fn shape(
        &mut self,
        shape: &Render2dShape,
        affine: Affine,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        if let Some(rect) = shape.as_rect() {
            return self.rectangle(rect, affine, fuel);
        }
        let path = shape.as_path().ok_or(AreaQueryError::Unsupported)?;
        let mut first = None;
        let mut previous = None;
        let mut has_segment = false;
        for command in path.commands() {
            charge(fuel, 8).ok_or(AreaQueryError::Resource)?;
            match *command {
                Render2dPathCommand::MoveTo(p) => {
                    if has_segment {
                        self.line(
                            previous.ok_or(AreaQueryError::Precision)?,
                            first.ok_or(AreaQueryError::Precision)?,
                            affine,
                            fuel,
                        )?;
                    }
                    first = Some([p.x(), p.y()]);
                    previous = first;
                    has_segment = false;
                }
                Render2dPathCommand::LineTo(p) => {
                    let q = [p.x(), p.y()];
                    self.line(previous.ok_or(AreaQueryError::Precision)?, q, affine, fuel)?;
                    previous = Some(q);
                    has_segment = true;
                }
                Render2dPathCommand::Close => {
                    if has_segment {
                        self.line(
                            previous.ok_or(AreaQueryError::Precision)?,
                            first.ok_or(AreaQueryError::Precision)?,
                            affine,
                            fuel,
                        )?;
                    }
                    previous = first;
                    has_segment = false;
                }
                _ => return Err(AreaQueryError::Unsupported),
            }
        }
        if has_segment {
            self.line(
                previous.ok_or(AreaQueryError::Precision)?,
                first.ok_or(AreaQueryError::Precision)?,
                affine,
                fuel,
            )?;
        }
        Ok(())
    }

    fn rectangle(
        &mut self,
        rect: Render2dRect,
        affine: Affine,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        charge(fuel, 32).ok_or(AreaQueryError::Resource)?;
        let x1 = exact_add(rect.x(), rect.width()).ok_or(AreaQueryError::Precision)?;
        let y1 = exact_add(rect.y(), rect.height()).ok_or(AreaQueryError::Precision)?;
        for (p, q) in [
            ([rect.x(), rect.y()], [x1, rect.y()]),
            ([x1, rect.y()], [x1, y1]),
            ([x1, y1], [rect.x(), y1]),
            ([rect.x(), y1], [rect.x(), rect.y()]),
        ] {
            self.line(p, q, affine, fuel)?;
        }
        Ok(())
    }

    fn clips(
        &mut self,
        clips: &[Render2dClip],
        affine: Affine,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        for clip in clips {
            let frame = self.frame().ok_or(AreaQueryError::Resource)?;
            let local = affine.after(clip.clip_to_parent(), frame, fuel)?;
            self.shape(clip.shape(), local, fuel)?;
        }
        Ok(())
    }

    fn region(
        &mut self,
        region: &NeutralRegion<'_>,
        affine: Affine,
        prepared: &PreparedBoundaries,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        charge(fuel, 128).ok_or(AreaQueryError::Resource)?;
        match &region.kind {
            Kind::Source(_) => {
                let item = region.semantic_item.ok_or(AreaQueryError::Unsupported)?;
                if !matches!(
                    item.primitive(),
                    Render2dPrimitive::Fill { .. } | Render2dPrimitive::Image(_)
                ) {
                    return Err(AreaQueryError::Unsupported);
                }
                let frame = self.frame().ok_or(AreaQueryError::Resource)?;
                let local = affine.after(item.local_to_parent(), frame, fuel)?;
                match item.primitive() {
                    Render2dPrimitive::Fill { shape, .. } => self.shape(shape, local, fuel)?,
                    Render2dPrimitive::Image(image) => {
                        for patch in image.patches() {
                            charge(fuel, 32).ok_or(AreaQueryError::Resource)?;
                            let rect = patch.destination();
                            if !rect.is_empty() {
                                self.rectangle(rect, local, fuel)?;
                            }
                        }
                    }
                    _ => return Err(AreaQueryError::Unsupported),
                }
                self.clips(item.clips(), affine, fuel)
            }
            Kind::Union(children) => {
                for child in children {
                    self.region(child, affine, prepared, fuel)?;
                }
                Ok(())
            }
            Kind::Affine { source, to_parent } => {
                let frame = self.frame().ok_or(AreaQueryError::Resource)?;
                let local = affine.after(*to_parent, frame, fuel)?;
                self.region(source, local, prepared, fuel)
            }
            Kind::Clip { source, clips } => {
                self.region(source, affine, prepared, fuel)?;
                self.clips(clips, affine, fuel)
            }
            Kind::Effect { source, shadow } => {
                if shadow.sigma() > 0.0 && shadow.spread() < 0.0 {
                    return self.finite_square_effect(region, *shadow, affine, prepared, fuel);
                }
                if shadow.sigma() != 0.0 || shadow.spread() > 0.0 {
                    return Err(AreaQueryError::Unsupported);
                }
                let shift =
                    Render2dAffineTransform::translation(shadow.offset_x(), shadow.offset_y())
                        .map_err(|_| AreaQueryError::Precision)?;
                let frame = self.frame().ok_or(AreaQueryError::Resource)?;
                let shifted = affine.after(shift, frame, fuel)?;
                if shadow.spread() == 0.0 {
                    return self.region(source, shifted, prepared, fuel);
                }
                let boundary = prepared.get(region.id, fuel).ok_or({
                    if *fuel == 0 {
                        AreaQueryError::Resource
                    } else {
                        AreaQueryError::Unsupported
                    }
                })?;
                for &edge in boundary.edges() {
                    self.line(edge[0], edge[1], shifted, fuel)?;
                    self.capsule(edge, -shadow.spread(), shifted, fuel)?;
                }
                Ok(())
            }
        }
    }

    fn crossing_work(left: &Curve, right: &Curve) -> usize {
        let line = |c: &Curve| c.function.b.lo == 0.0 && c.function.b.hi == 0.0;
        match (line(left), line(right)) {
            (true, true) => 384,
            (true, false) | (false, true) => 3072,
            (false, false) => 4096,
        }
    }

    /// Every true crossing is enclosed. Squared equations may introduce
    /// extra cuts, which affect work only and never geometric membership.
    fn crossings(left: &Curve, right: &Curve) -> Option<Crossings> {
        let empty = Crossings {
            complete: true,
            roots: [None, None],
        };
        if left.same_function(right) {
            return Some(empty);
        }
        let f = &left.function;
        let g = &right.function;
        let zero = |v: Interval| v.lo == 0.0 && v.hi == 0.0;
        if zero(f.b) && zero(g.b) {
            let (slope, offset) = match (left.key, right.key) {
                (CurveKey::ExactLine(h, a), CurveKey::ExactLine(k, b)) => (
                    Interval::scalar(exact_add(a, -b)?)?,
                    Interval::scalar(exact_add(h, -k)?)?,
                ),
                _ => {
                    // A scalar slope is a proved real coefficient even when
                    // the parallel line's intercept remains an interval.
                    // Preserve exact cancellation instead of dividing by an
                    // artificial interval around zero and losing topology.
                    let slope = if f.a.lo == f.a.hi && g.a.lo == g.a.hi {
                        exact_add(f.a.lo, -g.a.lo)
                            .and_then(Interval::scalar)
                            .or_else(|| f.a.sub(g.a))?
                    } else {
                        f.a.sub(g.a)?
                    };
                    (slope, f.h.sub(g.h)?)
                }
            };
            if zero(slope) {
                return (offset.lo > 0.0 || offset.hi < 0.0).then_some(empty);
            }
            let root = offset.mul(Interval::scalar(-1.0)?)?.div(slope)?;
            return Some(Crossings {
                complete: true,
                roots: [Some(root), None],
            });
        }
        if zero(f.b) || zero(g.b) {
            let (line_curve, arc_curve) = if zero(f.b) {
                (left, right)
            } else {
                (right, left)
            };
            if let (
                CurveKey::ConstructedLine(frame, 0, line_origin),
                CurveKey::ConstructedCircle(other_frame, center, radius, _),
            ) = (line_curve.key, arc_curve.key)
                && frame == other_frame
            {
                use finite_square_geometry::CoordinateOrigin::{Exact, HaloShift};
                let difference = match (line_origin, center[0]) {
                    (Exact(a), Exact(b)) => exact_add(a, -b),
                    (HaloShift(a, sign), HaloShift(b, other_sign)) if sign == other_sign => {
                        // The SAME real halo cancels exactly, irrespective of
                        // its enclosing interval or the common ancestor shear.
                        exact_add(a, -b)
                    }
                    _ => None,
                };
                if difference.is_some_and(|value| value.abs() == radius) {
                    // A parent vertical tangent meets its circle only at the
                    // center height. Wrong radial branches add harmless cuts.
                    return Some(Crossings {
                        complete: true,
                        roots: [Some(arc_curve.function.k), None],
                    });
                }
            }
            let (line, arc) = if zero(f.b) { (f, g) } else { (g, f) };
            let delta = line.h.add(line.a.mul(arc.k)?)?.sub(arc.h)?;
            let slope = line.a.sub(arc.a)?;
            let aq = squared_interval(slope)?.add(squared_interval(arc.b)?.mul(arc.c)?)?;
            if aq.lo <= 0.0 {
                return None;
            }
            let discriminant = aq
                .mul(squared_interval(arc.r)?)?
                .sub(arc.c.mul(squared_interval(delta)?)?)?;
            if discriminant.hi < 0.0 {
                return Some(empty);
            }
            let radical =
                sqrt_enclosure(Interval::new(discriminant.lo.max(0.0), discriminant.hi)?)?;
            let abs_b = Interval::new(
                arc.b.lo.abs().min(arc.b.hi.abs()),
                arc.b.lo.abs().max(arc.b.hi.abs()),
            )?;
            if arc.b.lo <= 0.0 && arc.b.hi >= 0.0 {
                return None;
            }
            let center = slope.mul(delta)?.mul(Interval::scalar(-1.0)?)?;
            let width = abs_b.mul(radical)?;
            return Some(Crossings {
                complete: true,
                roots: [
                    Some(center.sub(width)?.div(aq)?.add(arc.k)?),
                    Some(center.add(width)?.div(aq)?.add(arc.k)?),
                ],
            });
        }
        // Two circles mapped through the SAME unit-axis shear have exact
        // original circular intersection equations in (x-a*y,y). Reflection
        // exchanges branches without changing the complete intersection set.
        let (CurveKey::ExactConic(p), CurveKey::ExactConic(q)) = (left.key, right.key) else {
            return Self::enclosed_circle_crossings(left, right);
        };
        if p[1] != q[1] || p[2].abs() != 1.0 || q[2].abs() != 1.0 || p[5] != 1.0 || q[5] != 1.0 {
            return None;
        }
        let px = exact_add(p[0], -exact_mul(p[1], p[3])?)?;
        let qx = exact_add(q[0], -exact_mul(q[1], q[3])?)?;
        if px == qx && p[3] == q[3] {
            // Equal circles' opposite branches meet only at the known tips.
            return (p[4] == q[4]).then_some(empty);
        }
        let dx = Interval::scalar(qx)?.sub(Interval::scalar(px)?)?;
        let tangent = || {
            let dx = exact_add(qx, -px)?;
            let dy = exact_add(q[3], -p[3])?;
            let distance2 = exact_add(exact_mul(dx, dx)?, exact_mul(dy, dy)?)?;
            let sum = exact_add(p[4], q[4])?;
            if distance2 != exact_mul(sum, sum)? {
                return None;
            }
            let ratio = exact_div(p[4], sum)?;
            let y = exact_add(p[3], exact_mul(dy, ratio)?)?;
            Some(Crossings {
                complete: true,
                roots: [Some(Interval::scalar(y)?), None],
            })
        };
        if let Some(event) = tangent() {
            return Some(event);
        }
        let dy = Interval::scalar(q[3])?.sub(Interval::scalar(p[3])?)?;
        let d2 = squared_interval(dx)?.add(squared_interval(dy)?)?;
        if d2.lo <= 0.0 {
            return None;
        }
        let distance = sqrt_enclosure(d2)?;
        let r2 = squared_interval(Interval::scalar(p[4])?)?;
        let t = r2
            .sub(squared_interval(Interval::scalar(q[4])?)?)?
            .add(d2)?
            .div(Interval::scalar(2.0)?.mul(distance)?)?;
        let h2 = r2.sub(squared_interval(t)?)?;
        if h2.hi < 0.0 {
            return Some(empty);
        }
        let height = sqrt_enclosure(Interval::new(h2.lo.max(0.0), h2.hi)?)?;
        let base = Interval::scalar(p[3])?.add(dy.mul(t)?.div(distance)?)?;
        let width = dx.mul(height)?.div(distance)?;
        Some(Crossings {
            complete: true,
            roots: [Some(base.sub(width)?), Some(base.add(width)?)],
        })
    }

    /// The same circle intersection equations with uncertain centers.
    /// Roots may over-enclose or be spurious; original DAG membership still
    /// decides occupied strips. Equal boxes never prove circle coincidence.
    fn enclosed_circle_crossings(left: &Curve, right: &Curve) -> Option<Crossings> {
        let qualify = |curve: &Curve| match curve.key {
            CurveKey::ExactConic(raw) if raw[2].abs() == 1.0 && raw[5] == 1.0 => Some(raw[1]),
            CurveKey::ConstructedCircle(_, _, _, _)
                if curve.function.a.lo == curve.function.a.hi =>
            {
                Some(curve.function.a.lo)
            }
            _ => None,
        };
        let shear = qualify(left)?;
        if qualify(right)? != shear {
            return None;
        }
        let empty = Crossings {
            complete: true,
            roots: [None, None],
        };
        if let (
            CurveKey::ConstructedCircle(frame, center, _, _),
            CurveKey::ConstructedCircle(other_frame, other_center, _, _),
        ) = (left.key, right.key)
            && frame == other_frame
            && center == other_center
        {
            // Proven same real center. Equal circles meet at known branch
            // tips; unequal concentric circles have no intersections.
            return Some(empty);
        }
        let f = &left.function;
        let g = &right.function;
        let slope = Interval::scalar(shear)?;
        let px = f.h.sub(slope.mul(f.k)?)?;
        let qx = g.h.sub(slope.mul(g.k)?)?;
        let dx = qx.sub(px)?;
        let dy = g.k.sub(f.k)?;
        let d2 = squared_interval(dx)?.add(squared_interval(dy)?)?;
        if d2.lo <= 0.0 {
            return None;
        }
        let distance = sqrt_enclosure(d2)?;
        let t = squared_interval(f.r)?
            .sub(squared_interval(g.r)?)?
            .add(d2)?
            .div(Interval::scalar(2.0)?.mul(distance)?)?;
        let h2 = squared_interval(f.r)?.sub(squared_interval(t)?)?;
        if h2.hi < 0.0 {
            return Some(empty);
        }
        let height = sqrt_enclosure(Interval::new(h2.lo.max(0.0), h2.hi)?)?;
        let base = f.k.add(dy.mul(t)?.div(distance)?)?;
        let width = dx.mul(height)?.div(distance)?;
        Some(Crossings {
            complete: true,
            roots: [Some(base.sub(width)?), Some(base.add(width)?)],
        })
    }

    fn difference_integral(
        left: &Curve,
        right: &Curve,
        y0: f64,
        y1: f64,
        fuel: &mut usize,
    ) -> Option<Interval> {
        // Local-width coefficient preparation precedes each integral's own
        // admitted numerical work, including the two-conic fallback.
        charge(fuel, 1024)?;
        let f = &left.function;
        let g = &right.function;
        let zero = |v: Interval| v.lo == 0.0 && v.hi == 0.0;
        if zero(g.b) {
            // Direct integral of local strip WIDTH avoids cancellation between
            // two separately integrated large translated x coordinates.
            let width = Conic {
                h: g.h.add(g.a.mul(f.k)?)?.sub(f.h)?,
                a: g.a.sub(f.a)?,
                b: f.b.mul(Interval::scalar(-1.0)?)?,
                k: f.k,
                r: f.r,
                c: f.c,
            };
            return width.integral(y0, y1, fuel);
        }
        if zero(f.b) {
            let width = Conic {
                h: g.h.sub(f.h)?.sub(f.a.mul(g.k)?)?,
                a: g.a.sub(f.a)?,
                b: g.b,
                k: g.k,
                r: g.r,
                c: g.c,
            };
            return width.integral(y0, y1, fuel);
        }
        let origin = Interval::scalar(f.h.lo)?;
        let local_f = Conic {
            h: f.h.sub(origin)?,
            a: f.a,
            b: f.b,
            k: f.k,
            r: f.r,
            c: f.c,
        };
        let local_g = Conic {
            h: g.h.sub(origin)?,
            a: g.a,
            b: g.b,
            k: g.k,
            r: g.r,
            c: g.c,
        };
        local_g
            .integral(y0, y1, fuel)?
            .sub(local_f.integral(y0, y1, fuel)?)
    }

    /// Analytic source-area certificate at the existing region owner boundary.
    /// Every unresolved strip contributes its entire possible area. Numerical
    /// uncertainty is retained through summation and the final normalized bound.
    #[cfg(test)]
    pub(super) fn area(
        &self,
        region: &NeutralRegion<'_>,
        bounds: [f64; 4],
        forward: Option<Render2dAffineTransform>,
        prepared: &PreparedBoundaries,
        fuel: &mut usize,
    ) -> Option<[f64; 2]> {
        let [x0, y0, x1, y1] = bounds;
        if !bounds.iter().all(|v| v.is_finite()) || x0 >= x1 || y0 >= y1 {
            return None;
        }
        let cell_area = Interval::scalar(x1)?
            .sub(Interval::scalar(x0)?)?
            .mul(Interval::scalar(y1)?.sub(Interval::scalar(y0)?)?)?;
        if cell_area.lo <= 0.0 {
            return None;
        }
        let constant = |x: f64| -> Option<Curve> {
            Some(Curve {
                function: Conic {
                    h: Interval::scalar(x)?,
                    a: Interval::scalar(0.0)?,
                    b: Interval::scalar(0.0)?,
                    k: Interval::scalar(0.0)?,
                    r: Interval::scalar(1.0)?,
                    c: Interval::scalar(1.0)?,
                },
                start: Interval::scalar(y0)?,
                end: Interval::scalar(y1)?,
                key: CurveKey::ExactLine(x, 0.0),
                reach: Interval::scalar(x)?,
            })
        };
        let left = constant(x0)?;
        let right = constant(x1)?;
        let mut candidates = bounded_vector(self.curves.len().checked_add(2)?)?;
        for (id, c) in self.curves.iter().enumerate() {
            charge(fuel, 16)?;
            if c.reach.hi >= x0 && c.reach.lo <= x1 && c.start.lo < y1 && c.end.hi > y0 {
                candidates.push((id, c));
            }
        }
        let relevant_count = candidates.len();
        candidates.push((self.curves.len(), &left));
        candidates.push((self.curves.len() + 1, &right));
        let mut cuts = bounded_vector(MAX_BANDS)?;
        cuts.extend([y0, y1]);
        let mut add_cut = |y: f64| -> Option<()> {
            if y0 < y && y < y1 {
                if cuts.len() == MAX_BANDS {
                    return None;
                }
                cuts.push(y);
            }
            Some(())
        };
        for curve in &self.curves {
            charge(fuel, 8)?;
            for y in [curve.start.lo, curve.start.hi, curve.end.lo, curve.end.hi] {
                add_cut(y)?;
            }
        }
        // Global curve intersections were prepared once for the whole grid.
        for crossing in &self.crossings {
            charge(fuel, 64)?;
            for root in crossing.roots.into_iter().flatten() {
                add_cut(root.lo)?;
                add_cut(root.hi)?;
            }
        }
        // Physical cell edges are additional exact straight boundaries.
        for (_, curve) in &candidates[..relevant_count] {
            for edge in [&left, &right] {
                charge(fuel, 128 + Self::crossing_work(curve, edge))?;
                let crossing = Self::crossings(curve, edge)?;
                for root in crossing.roots.into_iter().flatten() {
                    add_cut(root.lo)?;
                    add_cut(root.hi)?;
                }
            }
        }
        sort_admitted(&mut cuts, f64::total_cmp, 4, fuel)?;
        cuts.dedup();
        let mut pending = bounded_vector(MAX_BANDS)?;
        pending.extend(cuts.windows(2).map(|ys| (ys[0], ys[1], 0_u8)));
        let mut total = Interval::scalar(0.0)?;
        let mut visited = 0_usize;
        while let Some((lo, hi, depth)) = pending.pop() {
            visited = visited.checked_add(1)?;
            if visited > MAX_BANDS {
                return None;
            }
            charge(fuel, 128)?;
            let possible = Interval::scalar(x1)?
                .sub(Interval::scalar(x0)?)?
                .mul(Interval::scalar(hi)?.sub(Interval::scalar(lo)?)?)?;
            let resolved = self.band(
                region,
                [x0, lo, x1, hi],
                forward,
                prepared,
                &candidates,
                fuel,
            );
            if let Some(mass) = resolved {
                total = total.add(mass)?;
                continue;
            }
            if *fuel == 0 {
                return None;
            }
            // Small uncertain event strips are geometric enclosures, never
            // guessed membership. Their complete area enters the upper bound.
            if depth == 32 || possible.hi <= cell_area.lo / (2048.0 * 256.0 * MAX_BANDS as f64) {
                total = total.add(Interval::new(0.0, possible.hi)?)?;
                continue;
            }
            let middle = lo * 0.5 + hi * 0.5;
            if !(lo < middle && middle < hi) {
                return None;
            }
            if pending.len() + 2 > MAX_BANDS {
                return None;
            }
            pending.extend([(lo, middle, depth + 1), (middle, hi, depth + 1)]);
        }
        let normalized = total.div(cell_area)?;
        let low = normalized.lo.max(0.0);
        let high = normalized.hi.min(1.0);
        if low > high || high - low > 1.0 / (2048.0 * 256.0) {
            return None;
        }
        Some([low, high])
    }

    #[cfg(test)]
    fn band(
        &self,
        region: &NeutralRegion<'_>,
        bounds: [f64; 4],
        forward: Option<Render2dAffineTransform>,
        prepared: &PreparedBoundaries,
        candidates: &[(usize, &Curve)],
        fuel: &mut usize,
    ) -> Option<Interval> {
        let [x0, y0, x1, y1] = bounds;
        let ys = Interval::new(y0, y1)?;
        let ym = y0 * 0.5 + y1 * 0.5;
        let mut active = bounded_vector(candidates.len())?;
        for &(id, curve) in candidates {
            charge(fuel, 64)?;
            if y1 <= curve.start.lo || y0 >= curve.end.hi {
                continue;
            }
            if y0 < curve.start.hi || y1 > curve.end.lo {
                return None;
            }
            charge(fuel, VALUE_WORK)?;
            let value = curve.function.value(ys)?;
            if value.hi < x0 || value.lo > x1 {
                continue;
            }
            charge(fuel, VALUE_WORK + 16 * active.len())?;
            let at_mid = curve.function.value(Interval::scalar(ym)?)?;
            if !active
                .iter()
                .any(|&(_, old, _, _): &(usize, &Curve, Interval, f64)| old.same_function(curve))
            {
                charge(fuel, 4)?;
                active.push((id, curve, at_mid, at_mid.lo * 0.5 + at_mid.hi * 0.5));
            }
        }
        sort_admitted(&mut active, |a, b| a.3.total_cmp(&b.3), 4, fuel)?;
        let mut mass = Interval::scalar(0.0)?;
        for pair in active.windows(2) {
            charge(fuel, 128)?;
            let [(lid, left, lx, _), (rid, right, rx, _)] = [pair[0], pair[1]];
            if lx.hi >= rx.lo {
                return None;
            }
            let crossing = if lid < self.curves.len() && rid < self.curves.len() {
                self.crossing(lid, rid)?
            } else {
                charge(fuel, Self::crossing_work(left, right))?;
                Self::crossings(left, right)?
            };
            let roots_clear = crossing.complete
                && crossing
                    .roots
                    .into_iter()
                    .flatten()
                    .all(|r| r.hi <= y0 || r.lo >= y1);
            if !roots_clear {
                charge(fuel, 2 * VALUE_WORK)?;
                let lrange = left.function.value(ys)?;
                let rrange = right.function.value(ys)?;
                if lrange.hi >= rrange.lo {
                    return None;
                }
            }
            let x = lx.hi * 0.5 + rx.lo * 0.5;
            if x < x0 || x > x1 {
                continue;
            }
            charge(fuel, 128)?;
            let mut point = Cell::new([x, ym, x, ym])?;
            if let Some(t) = forward {
                charge(fuel, preimage_work(t))?;
                point = point.preimage(t)?;
            }
            match region.certify_prepared(point, fuel, Some(prepared)) {
                CellCertificate::Inside => {
                    mass = mass.add(Self::difference_integral(left, right, y0, y1, fuel)?)?
                }
                CellCertificate::Outside => {}
                CellCertificate::Unknown => return None,
            }
        }
        Some(mass)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn enclosed_circle(origin: finite_square_geometry::CoordinateOrigin, sign: f64) -> Curve {
        let scalar = |v| Interval::scalar(v).unwrap();
        Curve {
            function: Conic {
                h: Interval::new(-0.1, 0.1).unwrap(),
                a: scalar(0.0),
                b: scalar(sign),
                k: Interval::new(-0.1, 0.1).unwrap(),
                r: scalar(1.0),
                c: scalar(1.0),
            },
            start: Interval::new(-1.1, -0.9).unwrap(),
            end: Interval::new(0.9, 1.1).unwrap(),
            key: CurveKey::ConstructedCircle(7, [origin, origin], 1.0, sign),
            reach: Interval::new(-1.1, 1.1).unwrap(),
        }
    }

    #[test]
    fn coincident_interval_boxes_never_supply_circle_or_line_identity() {
        use finite_square_geometry::CoordinateOrigin::HaloShift;
        let left = enclosed_circle(HaloShift(0.0, -1.0), -1.0);
        let right = enclosed_circle(HaloShift(0.0, 1.0), 1.0);
        assert!(!left.same_function(&right));
        assert!(
            ScanPlan::crossings(&left, &right).is_none(),
            "uncertain center separation cannot prove a complete crossing inventory"
        );
        let same = enclosed_circle(HaloShift(0.0, -1.0), 1.0);
        let tips = ScanPlan::crossings(&left, &same).unwrap();
        assert!(tips.complete && tips.roots.iter().all(Option::is_none));
        assert!(
            left.function
                .value(Interval::scalar(1.0).unwrap())
                .is_none(),
            "an uncertain tip radicand cannot be clamped into a value"
        );
        let a = CurveKey::ConstructedLine(7, 0, HaloShift(0.0, -1.0));
        let b = CurveKey::ConstructedLine(7, 0, HaloShift(0.0, 1.0));
        assert_ne!(a, b);
    }

    #[test]
    fn shared_halo_tangent_is_an_exact_expression_proof() {
        use finite_square_geometry::CoordinateOrigin::HaloShift;
        let arc = enclosed_circle(HaloShift(0.0, -1.0), 1.0);
        let mut tangent = ScanPlan::constant_curve(-1.0, -2.0, 2.0).unwrap();
        tangent.key = CurveKey::ConstructedLine(7, 0, HaloShift(-1.0, -1.0));
        let roots = ScanPlan::crossings(&tangent, &arc).unwrap();
        assert!(roots.complete);
        let root = roots.roots[0].unwrap();
        assert_eq!([root.lo, root.hi], [arc.function.k.lo, arc.function.k.hi]);
        tangent.key = CurveKey::ConstructedLine(8, 0, HaloShift(-1.0, -1.0));
        let other = ScanPlan::crossings(&tangent, &arc).unwrap();
        assert!(
            other
                .roots
                .iter()
                .flatten()
                .any(|r| r.hi - r.lo > root.hi - root.lo),
            "a foreign Effect/frame must not inherit the shared halo cancellation"
        );
    }

    fn empty_plan() -> ScanPlan {
        ScanPlan {
            curves: Vec::new(),
            next_frame: 1,
            crossings: Vec::new(),
            crossing_completeness: None,
            extra_y_events: Vec::new(),
        }
    }

    #[test]
    fn packed_crossings_are_symmetric_with_implicit_empty_diagonal() {
        for count in 0_usize..17 {
            let mut plan = empty_plan();
            for id in 0..count {
                plan.curves
                    .push(ScanPlan::constant_curve(id as f64, 0.0, 1.0).unwrap());
            }
            let mut fuel = 1_000_000;
            plan.prepare_crossings(&mut fuel).unwrap();
            assert!(plan.complete_crossings(&mut fuel).unwrap());
            assert_eq!(plan.crossings.len(), count * count.saturating_sub(1) / 2);
            // Give every unordered pair a distinct test-only root, then
            // independently enumerate rows to check every packed index.
            for (id, crossing) in plan.crossings.iter_mut().enumerate() {
                crossing.roots[0] = Some(Interval::scalar(id as f64).unwrap());
            }
            let mut id = 0;
            for left in 0..count {
                assert!(
                    plan.crossing(left, left)
                        .unwrap()
                        .roots
                        .iter()
                        .all(Option::is_none)
                );
                for right in left + 1..count {
                    for (a, b) in [(left, right), (right, left)] {
                        let root = plan.crossing(a, b).unwrap().roots[0].unwrap();
                        assert_eq!([root.lo, root.hi], [id as f64; 2]);
                    }
                    id += 1;
                }
            }
            assert!(plan.crossing(count, 0).is_none());
            // In particular, zero and one curves both have zero stored pairs;
            // table length alone cannot distinguish their dimensions.
            plan.curves
                .push(ScanPlan::constant_curve(100.0, 0.0, 1.0).unwrap());
            assert!(matches!(
                plan.complete_crossings(&mut fuel),
                Err(AreaQueryError::Precision)
            ));
            assert!(plan.crossing(0, 0).is_none());
        }
    }

    #[test]
    fn crossing_construction_prepays_even_empty_records_and_rejects_partial_table() {
        let mut plan = empty_plan();
        for x in 0..3 {
            plan.curves
                .push(ScanPlan::constant_curve(x as f64, 0.0, 1.0).unwrap());
        }
        // Three disjoint pairs cost 32 + 3*(128 + 16). Fail immediately
        // before the last record, after the first two have been stored.
        let mut fuel = 463;
        assert!(matches!(
            plan.prepare_crossings(&mut fuel),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(plan.crossings.len(), 2);
        assert!(plan.crossing(0, 1).is_none());
        let mut query_fuel = 16;
        assert!(matches!(
            plan.complete_crossings(&mut query_fuel),
            Err(AreaQueryError::Precision)
        ));
        assert_eq!(query_fuel, 0);
        let mut fuel = 464;
        plan.prepare_crossings(&mut fuel).unwrap();
        assert_eq!(fuel, 0);
        let mut query_fuel = 15;
        assert!(matches!(
            plan.complete_crossings(&mut query_fuel),
            Err(AreaQueryError::Resource)
        ));
        let mut query_fuel = 16;
        assert!(plan.complete_crossings(&mut query_fuel).unwrap());
    }

    #[test]
    fn incomplete_pair_cannot_license_shared_membership() {
        use finite_square_geometry::CoordinateOrigin::HaloShift;
        let mut plan = empty_plan();
        plan.curves
            .push(enclosed_circle(HaloShift(0.0, -1.0), -1.0));
        plan.curves.push(enclosed_circle(HaloShift(0.0, 1.0), 1.0));
        let mut fuel = 1_000_000;
        plan.prepare_crossings(&mut fuel).unwrap();
        assert!(!plan.complete_crossings(&mut fuel).unwrap());
        assert!(!plan.crossing(0, 1).unwrap().complete);
    }

    #[test]
    fn grid_resource_exhaustion_is_distinct_from_unrepresentable_phase() {
        let mut graph = builder();
        let region = graph.union(Vec::new(), &[0]).unwrap();
        let mut preparation_fuel = 1024;
        let prepared = PreparedBoundaries::prepare(&region, &mut preparation_fuel).unwrap();
        let mut fuel = 0;
        assert!(matches!(
            empty_plan().grid(&region, [0, 0, 1, 1], 4.0, None, &prepared, &mut fuel),
            Err(AreaQueryError::Resource)
        ));
        let mut fuel = 1024;
        assert!(matches!(
            empty_plan().grid(&region, [0, 0, 1, 1], 3.0, None, &prepared, &mut fuel),
            Err(AreaQueryError::Precision)
        ));
        assert!(fuel > 0);
        assert!(matches!(
            empty_plan().grid(&region, [0, 0, 1025, 1025], 4.0, None, &prepared, &mut fuel),
            Err(AreaQueryError::Resource)
        ));
    }

    #[test]
    fn distinct_curve_inventory_limit_keeps_resource_category() {
        let mut plan = empty_plan();
        plan.curves = bounded_vector(MAX_CURVES).unwrap();
        for x in 0..MAX_CURVES {
            plan.curves
                .push(ScanPlan::constant_curve(x as f64, 0.0, 1.0).unwrap());
        }
        let mut fuel = 16_777_216;
        assert!(matches!(
            plan.push(
                ScanPlan::constant_curve(MAX_CURVES as f64, 0.0, 1.0).unwrap(),
                &mut fuel
            ),
            Err(AreaQueryError::Resource)
        ));
        assert!(fuel > 0);
        assert_eq!(plan.curves.len(), MAX_CURVES);
    }

    #[test]
    fn fixed_scratch_inventory_fits_consumer_reservation() {
        let bytes = ScanPlan::fixed_scratch_bytes().unwrap();
        assert!(bytes >= 65_536 * std::mem::size_of::<Crossings>());
        assert!(bytes < 16 * 1024 * 1024);
    }

    #[test]
    fn shared_membership_tracks_both_canvas_clip_crossings() {
        use crate::composition_2d::{
            Render2dBrush, Render2dColorRgba8, Render2dFillRule, Render2dOpacity, Render2dPoint,
        };
        let commands = vec![
            Render2dPathCommand::MoveTo(Render2dPoint::new(-100.0, 0.0).unwrap()),
            Render2dPathCommand::LineTo(Render2dPoint::new(156.0, 64.0).unwrap()),
            Render2dPathCommand::LineTo(Render2dPoint::new(156.0, 0.0).unwrap()),
            Render2dPathCommand::Close,
        ];
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::path(
                    Render2dPath::new(Render2dFillRule::EvenOdd, commands).unwrap(),
                ),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = builder();
        let source = graph
            .source_from_item(
                support::NeutralMesh {
                    units_per_parent_logical_unit: 1.0,
                    triangles: vec![[-100.0, 0.0], [156.0, 64.0], [156.0, 0.0]],
                    bounds: [-100.0, 0.0, 156.0, 64.0],
                },
                &item,
                &[0],
            )
            .unwrap();
        let mut fuel = 16_777_216;
        let areas =
            certified_grid_area_query(&source, [0, 0, 64, 64], 1.0, None, 8, 4096, &mut fuel)
                .unwrap();
        // The original diagonal x=4*y-100 crosses the canvas at y=25 and
        // y=41. The same pair of canvas clip planes bounds an entirely full
        // strip below the first crossing and an empty strip above the last.
        for row in 0..25 {
            for column in 0..64 {
                let value = areas[row * 64 + column];
                assert!(
                    value[0] <= 1.0 && value[1] >= 1.0,
                    "full {column},{row}: {value:?}"
                );
            }
        }
        for row in 41..64 {
            for column in 0..64 {
                let value = areas[row * 64 + column];
                assert!(
                    value[0] == 0.0 && value[1] <= 1.0 / (2048.0 * 256.0),
                    "empty {column},{row}: {value:?}"
                );
            }
        }
    }

    #[test]
    fn shared_membership_preserves_original_bowtie_crossing_and_branch_endpoints() {
        use crate::composition_2d::{
            Render2dBrush, Render2dColorRgba8, Render2dFillRule, Render2dOpacity, Render2dPoint,
        };
        let mut commands = Vec::new();
        for (index, [x, y]) in [[4.0, 4.0], [12.0, 12.0], [4.0, 12.0], [12.0, 4.0]]
            .into_iter()
            .enumerate()
        {
            let point = Render2dPoint::new(x, y).unwrap();
            commands.push(if index == 0 {
                Render2dPathCommand::MoveTo(point)
            } else {
                Render2dPathCommand::LineTo(point)
            });
        }
        commands.push(Render2dPathCommand::Close);
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::path(
                    Render2dPath::new(Render2dFillRule::EvenOdd, commands).unwrap(),
                ),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = builder();
        let source = graph
            .source_from_item(
                support::NeutralMesh {
                    units_per_parent_logical_unit: 1.0,
                    triangles: vec![[0.0, 0.0], [16.0, 0.0], [0.0, 16.0]],
                    bounds: [0.0, 0.0, 16.0, 16.0],
                },
                &item,
                &[0],
            )
            .unwrap();
        let mut fuel = 16_777_216;
        let areas =
            certified_grid_area_query(&source, [0, 0, 64, 64], 4.0, None, 8, 4096, &mut fuel)
                .unwrap();
        for sy in 0..4 {
            for sx in 0..4 {
                // Independent two-triangle geometry, above and below y=8.
                // The diagonal direction changes at the original crossing.
                let top = match sx.cmp(&sy) {
                    std::cmp::Ordering::Less => 0.0,
                    std::cmp::Ordering::Equal => 0.5,
                    std::cmp::Ordering::Greater => 1.0,
                };
                let bottom = match (sx + sy).cmp(&3) {
                    std::cmp::Ordering::Less => 0.0,
                    std::cmp::Ordering::Equal => 0.5,
                    std::cmp::Ordering::Greater => 1.0,
                };
                for (row, reference) in [(28 + sy, top), (32 + sy, bottom)] {
                    let value = areas[row * 64 + 28 + sx];
                    assert!(
                        value[0] <= reference && reference <= value[1],
                        "{sx},{sy}: {value:?} vs {reference}"
                    );
                }
            }
        }
        assert_eq!(areas[0], [0.0, 0.0]);
        assert_eq!(areas[52 * 64 + 28], [0.0, 0.0]);
    }
}

pub(in crate::runtime::execution_2d::shadow::region) mod gaussian;
