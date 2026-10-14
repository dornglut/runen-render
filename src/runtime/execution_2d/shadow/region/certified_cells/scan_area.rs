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
    fn after(self, transform: Render2dAffineTransform, frame: usize) -> Option<Self> {
        let [a, b, c, d, tx, ty] = self.0;
        let [e, f, g, h, ux, uy] = transform.components().map(|v| Interval::scalar(v).unwrap());
        let exact = || {
            let [a, b, c, d, tx, ty] = self.2?;
            let [e, f, g, h, ux, uy] = transform.components();
            Some([
                exact_add(exact_mul(a, e)?, exact_mul(c, f)?)?,
                exact_add(exact_mul(b, e)?, exact_mul(d, f)?)?,
                exact_add(exact_mul(a, g)?, exact_mul(c, h)?)?,
                exact_add(exact_mul(b, g)?, exact_mul(d, h)?)?,
                exact_add(exact_add(exact_mul(a, ux)?, exact_mul(c, uy)?)?, tx)?,
                exact_add(exact_add(exact_mul(b, ux)?, exact_mul(d, uy)?)?, ty)?,
            ])
        };
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
            exact(),
        ))
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

#[derive(Clone, Copy, PartialEq)]
enum CurveKey {
    Vertical(usize, f64),
    Horizontal(usize, f64),
    General(usize, [f64; 2], [f64; 2]),
    Circle(usize, [f64; 2], f64, f64),
    ExactLine(f64, f64),
    ExactConic([f64; 6]),
}
struct Curve {
    function: Conic,
    start: Interval,
    end: Interval,
    key: CurveKey,
    domain: [[f64; 2]; 2],
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
}

struct GridQuery<'r, 'a> {
    region: &'r NeutralRegion<'a>,
    forward: Option<Render2dAffineTransform>,
    prepared: &'r PreparedBoundaries,
    xs: Vec<f64>,
    vertical_uncertainty: Vec<Interval>,
}

impl ScanPlan {
    pub(super) fn grid(
        &self,
        region: &NeutralRegion<'_>,
        root: [u32; 4],
        scale: f64,
        forward: Option<Render2dAffineTransform>,
        prepared: &PreparedBoundaries,
        fuel: &mut usize,
    ) -> Option<Vec<[f64; 2]>> {
        let width = usize::try_from(root[2].checked_sub(root[0])?).ok()?;
        let height = usize::try_from(root[3].checked_sub(root[1])?).ok()?;
        let count = width.checked_mul(height)?;
        if count == 0 || count > 1_048_576 || !scale.is_finite() || scale <= 0.0 {
            return None;
        }
        let y0 = exact_div(f64::from(root[1]), scale)?;
        let y1 = exact_div(f64::from(root[3]), scale)?;
        let mut xs = Vec::new();
        xs.try_reserve_exact(width + 1).ok()?;
        for index in root[0]..=root[2] {
            charge(fuel, 4)?;
            xs.push(exact_div(f64::from(index), scale)?);
        }
        if xs.windows(2).any(|p| p[0] >= p[1]) {
            return None;
        }
        let mut query = GridQuery {
            region,
            forward,
            prepared,
            xs,
            vertical_uncertainty: Vec::new(),
        };
        query
            .vertical_uncertainty
            .try_reserve_exact(MAX_BANDS)
            .ok()?;
        let mut events = Vec::new();
        events.try_reserve_exact(MAX_BANDS).ok()?;
        let mut add = |y: f64| -> Option<()> {
            if y0 < y && y < y1 {
                if events.len() == MAX_BANDS {
                    return None;
                }
                events.push(y);
            }
            Some(())
        };
        for curve in &self.curves {
            charge(fuel, 8)?;
            for y in [curve.start.lo, curve.start.hi, curve.end.lo, curve.end.hi] {
                add(y)?;
            }
        }
        for crossing in &self.crossings {
            charge(fuel, 1)?;
            for root in crossing.roots.into_iter().flatten() {
                add(root.lo)?;
                add(root.hi)?;
                if root.hi > y0 && root.lo < y1 {
                    if query.vertical_uncertainty.len() == MAX_BANDS {
                        return None;
                    }
                    query.vertical_uncertainty.push(root);
                }
            }
        }
        // Every physical vertical-edge crossing is prepared ONCE for all
        // rows. Each ensuing band fixes both strip topology and cell routing.
        for curve in &self.curves {
            for &x in &query.xs {
                charge(fuel, 4)?;
                if x < curve.reach.lo || x > curve.reach.hi {
                    continue;
                }
                let edge = Self::constant_curve(x, y0, y1)?;
                charge(fuel, 384)?;
                let crossing = Self::crossings(curve, &edge)?;
                for root in crossing.roots.into_iter().flatten() {
                    add(root.lo)?;
                    add(root.hi)?;
                    if root.hi > y0 && root.lo < y1 {
                        if query.vertical_uncertainty.len() == MAX_BANDS {
                            return None;
                        }
                        query.vertical_uncertainty.push(root);
                    }
                }
            }
        }
        sort_admitted(&mut events, f64::total_cmp, 4, fuel)?;
        events.dedup();
        sort_admitted(
            &mut query.vertical_uncertainty,
            |a, b| a.lo.total_cmp(&b.lo).then(a.hi.total_cmp(&b.hi)),
            8,
            fuel,
        )?;
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
        let mut output = Vec::new();
        output.try_reserve_exact(count).ok()?;
        charge(fuel, count)?;
        output.resize(count, [0.0, 0.0]);
        let mut pending = Vec::new();
        pending.try_reserve_exact(MAX_BANDS).ok()?;
        let mut staged = Vec::new();
        staged.try_reserve_exact(width).ok()?;
        staged.resize(width, Interval::scalar(0.0)?);
        let mut visited = 0_usize;
        for row_index in 0..height {
            let lo = exact_div(f64::from(root[1] + u32::try_from(row_index).ok()?), scale)?;
            let hi = exact_div(
                f64::from(root[1] + u32::try_from(row_index + 1).ok()?),
                scale,
            )?;
            if lo >= hi {
                return None;
            }
            let start = events.partition_point(|v| *v <= lo);
            let end = events.partition_point(|v| *v < hi);
            let mut previous = lo;
            for &cut in &events[start..end] {
                pending.push((previous, cut, 0_u8));
                previous = cut;
            }
            pending.push((previous, hi, 0_u8));
            let cell_height = Interval::scalar(hi)?.sub(Interval::scalar(lo)?)?;
            while let Some((a, b, depth)) = pending.pop() {
                visited = visited.checked_add(1)?;
                // One scene-wide frontier, never a reset allowance per row.
                if visited > MAX_BANDS * 16 {
                    return None;
                }
                charge(fuel, width + 128)?;
                staged.fill(Interval::scalar(0.0)?);
                let known = self.row_band(&query, a, b, &mut staged, fuel).is_some();
                if *fuel == 0 {
                    return None;
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
                    if !(a < middle && middle < b) || pending.len() + 2 > MAX_BANDS {
                        return None;
                    }
                    pending.extend([(a, middle, depth + 1), (middle, b, depth + 1)]);
                    continue;
                }
                for (column, &mass) in staged.iter().enumerate() {
                    charge(fuel, 32)?;
                    let cell_width = Interval::scalar(query.xs[column + 1])?
                        .sub(Interval::scalar(query.xs[column])?)?;
                    let mass = if known {
                        mass
                    } else {
                        Interval::new(
                            0.0,
                            cell_width
                                .mul(Interval::scalar(b)?.sub(Interval::scalar(a)?)?)?
                                .hi,
                        )?
                    };
                    let index = row_index * width + column;
                    let prior = Interval::new(output[index][0], output[index][1])?;
                    let updated = prior.add(mass)?;
                    output[index] = [updated.lo, updated.hi];
                }
            }
            for (column, value) in output[row_index * width..(row_index + 1) * width]
                .iter_mut()
                .enumerate()
            {
                charge(fuel, 32)?;
                let area = cell_height.mul(
                    Interval::scalar(query.xs[column + 1])?
                        .sub(Interval::scalar(query.xs[column])?)?,
                )?;
                let fraction = Interval::new(value[0], value[1])?.div(area)?;
                *value = [fraction.lo.max(0.0), fraction.hi.min(1.0)];
                if value[0] > value[1] || (value[1] - value[0]).next_up() > 1.0 / (2048.0 * 256.0) {
                    return None;
                }
            }
        }
        Some(output)
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
            domain: [[x, y0], [x, y1]],
            reach: Interval::scalar(x)?,
        })
    }

    fn row_band(
        &self,
        query: &GridQuery<'_, '_>,
        y0: f64,
        y1: f64,
        staged: &mut [Interval],
        fuel: &mut usize,
    ) -> Option<()> {
        charge(fuel, 32)?;
        let uncertain = query
            .vertical_uncertainty
            .partition_point(|root| root.hi <= y0);
        if query
            .vertical_uncertainty
            .get(uncertain)
            .is_some_and(|root| root.lo < y1)
        {
            return None;
        }
        let x0 = *query.xs.first()?;
        let x1 = *query.xs.last()?;
        let left = Self::constant_curve(x0, y0, y1)?;
        let right = Self::constant_curve(x1, y0, y1)?;
        let ym = y0 * 0.5 + y1 * 0.5;
        let ys = Interval::new(y0, y1)?;
        let mut active = Vec::new();
        active.try_reserve_exact(self.curves.len() + 2).ok()?;
        for (id, curve) in self
            .curves
            .iter()
            .enumerate()
            .chain([(self.curves.len(), &left), (self.curves.len() + 1, &right)])
        {
            charge(fuel, 16)?;
            if curve.reach.hi < x0
                || curve.reach.lo > x1
                || y1 <= curve.start.lo
                || y0 >= curve.end.hi
            {
                continue;
            }
            if y0 < curve.start.hi || y1 > curve.end.lo {
                return None;
            }
            charge(fuel, VALUE_WORK + active.len())?;
            let value = curve.function.value(Interval::scalar(ym)?)?;
            if !active
                .iter()
                .any(|&(_, old, _, _): &(usize, &Curve, Interval, f64)| old.same_function(curve))
            {
                charge(fuel, 4)?;
                active.push((id, curve, value, value.lo * 0.5 + value.hi * 0.5));
            }
        }
        sort_admitted(&mut active, |a, b| a.3.total_cmp(&b.3), 4, fuel)?;
        for pair in active.windows(2) {
            charge(fuel, 128)?;
            let [(lid, l, lx, _), (rid, r, rx, _)] = [pair[0], pair[1]];
            if lx.hi >= rx.lo {
                return None;
            }
            let crossing = if lid < self.curves.len() && rid < self.curves.len() {
                self.crossings[lid * self.curves.len() + rid]
            } else {
                charge(fuel, Self::crossing_work(l, r))?;
                Self::crossings(l, r)?
            };
            if !(crossing.complete
                && crossing
                    .roots
                    .into_iter()
                    .flatten()
                    .all(|v| v.hi <= y0 || v.lo >= y1))
            {
                charge(fuel, 2 * VALUE_WORK)?;
                if l.function.value(ys)?.hi >= r.function.value(ys)?.lo {
                    return None;
                }
            }
            let x = lx.hi * 0.5 + rx.lo * 0.5;
            if x < x0 || x > x1 {
                continue;
            }
            charge(fuel, 128)?;
            let mut point = Cell::new([x, ym, x, ym])?;
            if let Some(t) = query.forward {
                point = point.preimage(t)?;
            }
            match query
                .region
                .certify_prepared(point, fuel, Some(query.prepared))
            {
                CellCertificate::Outside => continue,
                CellCertificate::Unknown => return None,
                CellCertificate::Inside => {}
            }
            let column = |value: Interval| -> Option<usize> {
                if value.lo == value.hi && value.lo == x1 {
                    return Some(query.xs.len() - 2);
                }
                let a = query.xs.partition_point(|edge| *edge <= value.lo);
                let b = query.xs.partition_point(|edge| *edge <= value.hi);
                (a == b && a > 0 && a < query.xs.len()).then(|| a - 1)
            };
            let lc = column(lx)?;
            let rc = column(rx)?;
            if lc > rc {
                return None;
            }
            if lc == rc {
                staged[lc] = staged[lc].add(Self::difference_integral(l, r, y0, y1, fuel)?)?;
            } else {
                let edge = Self::constant_curve(query.xs[lc + 1], y0, y1)?;
                staged[lc] = staged[lc].add(Self::difference_integral(l, &edge, y0, y1, fuel)?)?;
                let edge = Self::constant_curve(query.xs[rc], y0, y1)?;
                staged[rc] = staged[rc].add(Self::difference_integral(&edge, r, y0, y1, fuel)?)?;
                for (c, mass) in staged.iter_mut().enumerate().take(rc).skip(lc + 1) {
                    charge(fuel, 32)?;
                    let area = Interval::scalar(query.xs[c + 1])?
                        .sub(Interval::scalar(query.xs[c])?)?
                        .mul(Interval::scalar(y1)?.sub(Interval::scalar(y0)?)?)?;
                    *mass = mass.add(area)?;
                }
            }
        }
        Some(())
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
            curves: Vec::new(),
            next_frame: 1,
            crossings: Vec::new(),
        };
        result
            .curves
            .try_reserve_exact(MAX_CURVES)
            .map_err(|_| AreaQueryError::Resource)?;
        let affine = match forward {
            Some(t) => Affine::identity()
                .after(t, result.frame().ok_or(AreaQueryError::Resource)?)
                .ok_or(AreaQueryError::Precision)?,
            None => Affine::identity(),
        };
        result.region(region, affine, prepared, fuel)?;
        let count = result
            .curves
            .len()
            .checked_mul(result.curves.len())
            .ok_or(AreaQueryError::Resource)?;
        if count > 65_536 {
            return Err(AreaQueryError::Resource);
        }
        result
            .crossings
            .try_reserve_exact(count)
            .map_err(|_| AreaQueryError::Resource)?;
        charge(fuel, count).ok_or(AreaQueryError::Resource)?;
        result.crossings.resize(
            count,
            Crossings {
                complete: true,
                roots: [None, None],
            },
        );
        for left in 0..result.curves.len() {
            for right in left + 1..result.curves.len() {
                let l = &result.curves[left];
                let r = &result.curves[right];
                charge(fuel, Self::crossing_work(l, r)).ok_or(AreaQueryError::Resource)?;
                let crossing = Self::crossings(l, r).unwrap_or(Crossings {
                    complete: false,
                    roots: [None, None],
                });
                result.crossings[left * result.curves.len() + right] = crossing;
                result.crossings[right * result.curves.len() + left] = crossing;
            }
        }
        Ok(result)
    }

    fn push(&mut self, mut curve: Curve, fuel: &mut usize) -> Option<()> {
        charge(fuel, self.curves.len() + 128)?;
        let f = &curve.function;
        let ys = Interval::new(curve.start.lo, curve.end.hi)?;
        let base = f.h.add(f.a.mul(ys.sub(f.k)?)?)?;
        let radius = f.b.lo.abs().max(f.b.hi.abs());
        let margin = Interval::scalar(radius)?.mul(f.r)?.hi;
        curve.reach = base.add(Interval::new(-margin, margin)?)?;
        // Exact equality is a geometric identity, never approximate dedup.
        if self
            .curves
            .iter()
            .any(|old| old.same_function(&curve) && old.domain == curve.domain)
        {
            return Some(());
        }
        if self.curves.len() == MAX_CURVES {
            return None;
        }
        self.curves.push(curve);
        Some(())
    }

    fn line(&mut self, p: [f64; 2], q: [f64; 2], affine: Affine, fuel: &mut usize) -> Option<()> {
        charge(fuel, 128)?;
        let [px, py] = affine.point(p)?;
        let [qx, qy] = affine.point(q)?;
        let [a, b, c, d, tx, ty] = affine.0;
        let zero = |v: Interval| v.lo == 0.0 && v.hi == 0.0;
        if (p[0] == q[0] && zero(d)) || (p[1] == q[1] && zero(b)) {
            return Some(());
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
        self.push(
            Curve {
                key,
                domain: [p, q],
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
            },
            fuel,
        )
    }

    fn circle(
        &mut self,
        center: [f64; 2],
        radius: f64,
        affine: Affine,
        fuel: &mut usize,
    ) -> Option<()> {
        charge(fuel, 256)?;
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
            for sign in [-1.0, 1.0] {
                let raw = [h.lo, slope, sign, k.lo, radius, 1.0];
                let [h, a, b, k, r, c] = raw.map(|v| Interval::scalar(v).unwrap());
                self.push(
                    Curve {
                        function: Conic { h, a, b, k, r, c },
                        start,
                        end,
                        key: CurveKey::ExactConic(raw),
                        domain: [center, center],
                        reach: Interval::scalar(0.0)?,
                    },
                    fuel,
                )?;
            }
            return Some(());
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
        for sign in [-1.0, 1.0] {
            self.push(
                Curve {
                    key: CurveKey::Circle(affine.1, center, radius, sign),
                    domain: [center, center],
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
                },
                fuel,
            )?;
        }
        Some(())
    }

    fn capsule(
        &mut self,
        [p, q]: [[f64; 2]; 2],
        radius: f64,
        affine: Affine,
        fuel: &mut usize,
    ) -> Option<()> {
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
                return None;
            }
            self.line(u, v, affine, fuel)?;
        }
        Some(())
    }

    fn shape(
        &mut self,
        shape: &Render2dShape,
        affine: Affine,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        if let Some(rect) = shape.as_rect() {
            charge(fuel, 32).ok_or(AreaQueryError::Resource)?;
            let x1 = exact_add(rect.x(), rect.width()).ok_or(AreaQueryError::Precision)?;
            let y1 = exact_add(rect.y(), rect.height()).ok_or(AreaQueryError::Precision)?;
            for (p, q) in [
                ([rect.x(), rect.y()], [x1, rect.y()]),
                ([x1, rect.y()], [x1, y1]),
                ([x1, y1], [rect.x(), y1]),
                ([rect.x(), y1], [rect.x(), rect.y()]),
            ] {
                self.line(p, q, affine, fuel)
                    .ok_or_else(|| exhausted_or_precision(fuel))?;
            }
            return Ok(());
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
                        )
                        .ok_or_else(|| exhausted_or_precision(fuel))?;
                    }
                    first = Some([p.x(), p.y()]);
                    previous = first;
                    has_segment = false;
                }
                Render2dPathCommand::LineTo(p) => {
                    let q = [p.x(), p.y()];
                    self.line(previous.ok_or(AreaQueryError::Precision)?, q, affine, fuel)
                        .ok_or_else(|| exhausted_or_precision(fuel))?;
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
                        )
                        .ok_or_else(|| exhausted_or_precision(fuel))?;
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
            )
            .ok_or_else(|| exhausted_or_precision(fuel))?;
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
            let local = affine
                .after(clip.clip_to_parent(), frame)
                .ok_or(AreaQueryError::Precision)?;
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
                let Render2dPrimitive::Fill { shape, .. } = item.primitive() else {
                    return Err(AreaQueryError::Unsupported);
                };
                let frame = self.frame().ok_or(AreaQueryError::Resource)?;
                let local = affine
                    .after(item.local_to_parent(), frame)
                    .ok_or(AreaQueryError::Precision)?;
                self.shape(shape, local, fuel)?;
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
                let local = affine
                    .after(*to_parent, frame)
                    .ok_or(AreaQueryError::Precision)?;
                self.region(source, local, prepared, fuel)
            }
            Kind::Clip { source, clips } => {
                self.region(source, affine, prepared, fuel)?;
                self.clips(clips, affine, fuel)
            }
            Kind::Effect { source, shadow } => {
                if shadow.sigma() != 0.0 || shadow.spread() > 0.0 {
                    return Err(AreaQueryError::Unsupported);
                }
                let shift =
                    Render2dAffineTransform::translation(shadow.offset_x(), shadow.offset_y())
                        .map_err(|_| AreaQueryError::Precision)?;
                let frame = self.frame().ok_or(AreaQueryError::Resource)?;
                let shifted = affine
                    .after(shift, frame)
                    .ok_or(AreaQueryError::Precision)?;
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
                    self.line(edge[0], edge[1], shifted, fuel)
                        .ok_or_else(|| exhausted_or_precision(fuel))?;
                    self.capsule(edge, -shadow.spread(), shifted, fuel)
                        .ok_or_else(|| exhausted_or_precision(fuel))?;
                }
                Ok(())
            }
        }
    }

    fn crossing_work(left: &Curve, right: &Curve) -> usize {
        let line = |c: &Curve| c.function.b.lo == 0.0 && c.function.b.hi == 0.0;
        if line(left) || line(right) { 384 } else { 768 }
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
                _ => (f.a.sub(g.a)?, f.h.sub(g.h)?),
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
            return None;
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

    fn difference_integral(
        left: &Curve,
        right: &Curve,
        y0: f64,
        y1: f64,
        fuel: &mut usize,
    ) -> Option<Interval> {
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
                domain: [[x, y0], [x, y1]],
                reach: Interval::scalar(x)?,
            })
        };
        let left = constant(x0)?;
        let right = constant(x1)?;
        let mut candidates = Vec::new();
        candidates.try_reserve_exact(self.curves.len() + 2).ok()?;
        for (id, c) in self.curves.iter().enumerate() {
            charge(fuel, 16)?;
            if c.reach.hi >= x0 && c.reach.lo <= x1 && c.start.lo < y1 && c.end.hi > y0 {
                candidates.push((id, c));
            }
        }
        let relevant_count = candidates.len();
        candidates.push((self.curves.len(), &left));
        candidates.push((self.curves.len() + 1, &right));
        let mut cuts = Vec::new();
        cuts.try_reserve_exact(MAX_BANDS).ok()?;
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
            charge(fuel, 1)?;
            for root in crossing.roots.into_iter().flatten() {
                add_cut(root.lo)?;
                add_cut(root.hi)?;
            }
        }
        // Physical cell edges are additional exact straight boundaries.
        for (_, curve) in &candidates[..relevant_count] {
            for edge in [&left, &right] {
                charge(fuel, 384)?;
                let crossing = Self::crossings(curve, edge)?;
                for root in crossing.roots.into_iter().flatten() {
                    add_cut(root.lo)?;
                    add_cut(root.hi)?;
                }
            }
        }
        sort_admitted(&mut cuts, f64::total_cmp, 4, fuel)?;
        cuts.dedup();
        let mut pending = Vec::new();
        pending.try_reserve_exact(MAX_BANDS).ok()?;
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
        let mut active = Vec::new();
        active.try_reserve_exact(candidates.len()).ok()?;
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
            charge(fuel, VALUE_WORK + active.len())?;
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
                self.crossings[lid * self.curves.len() + rid]
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
            let mut point = Cell::new([x, ym, x, ym])?;
            if let Some(t) = forward {
                point = point.preimage(t)?;
            }
            charge(fuel, 128)?;
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
