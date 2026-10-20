//! Test-only shared finite-Gaussian integration over original-region strips.
//!
//! A complete boundary arrangement and original DAG membership certify open
//! occupied strips once. Unknown event slabs retain explicit possible mass.
//! Original-boundary cutoff events and analytic quadrature certificates bound
//! the numerical kernel; compiled scalar maps and shared profiles remove
//! repeated preparation. Disposable field values never become source support.
//!
//! The owned plane and maintained F3E GPU readback qualify the bounded numerical
//! path. Production selection remains absent until source and invocation-wide
//! resource obligations are closed. No partial plane escapes on failure.

#![cfg(test)]

use super::super::gaussian_kernel::GaussianKernel;
use super::*;

#[cfg(test)]
mod diagnostic;
#[cfg(test)]
use diagnostic::{Diagnostics, Integrator, MassCertificate};

mod classified_spans;
#[cfg(test)]
mod image_qualification;
mod numeric_index;
mod plane;
mod query_compiler;
#[cfg(test)]
mod reference;
mod scalar_map;
#[cfg(test)]
mod source_qualification;
mod storage;

const MAX_RUNS: usize = 65_536;
const MAX_BYTES: usize = 64 * 1024 * 1024;
const MAX_COMPILED_MAPS: usize = 2 * MAX_CURVES;

fn debit(fuel: &mut usize, amount: usize) -> Result<(), AreaQueryError> {
    charge(fuel, amount).ok_or(AreaQueryError::Resource)
}

fn iv(value: f64) -> Interval {
    Interval::scalar(value).unwrap()
}

fn checked(value: Option<Interval>) -> Result<Interval, AreaQueryError> {
    value.ok_or(AreaQueryError::Precision)
}

/// The constants define an exact finite positive rule, rather than claiming
/// that their rounded values are the mathematical Gauss roots/weights.
/// Symmetric pairs make every odd moment exactly zero. The 65 even residual
/// enclosures below, together with the analytic Cauchy tail, certify the
/// actual rule including all errors in these decimal constants.
const RULE: [(f64, f64); 8] = [
    (0.095_012_509_837_637_44, 0.189_450_610_455_068_5),
    (0.281_603_550_779_258_9, 0.182_603_415_044_923_6),
    (0.458_016_777_657_227_4, 0.169_156_519_395_002_54),
    (0.617_876_244_402_643_8, 0.149_595_988_816_576_74),
    (0.755_404_408_355_003, 0.124_628_971_255_533_88),
    (0.865_631_202_387_831_8, 0.095_158_511_682_492_79),
    (0.944_575_023_073_232_6, 0.062_253_523_938_647_89),
    (0.989_400_934_991_649_9, 0.027_152_459_411_754_096),
];

struct PositiveRule {
    moments: [f64; 65],
    weight_sum: f64,
}

/// An error factor belonging to this module's certified positive rule and the
/// exact ratio ceiling h/R=1/2. Construction is private to PositiveRule; callers
/// can reuse the numerical certificate but cannot pass an arbitrary scalar as
/// an alleged quadrature proof. This eight-byte value owns no heap allocation.
struct HalfRatioFactor {
    upper: f64,
}

impl HalfRatioFactor {
    /// A disc containing the panel is still supplied by the caller. This guard
    /// checks only the ratio required by this certificate, not analyticity or M.
    fn error(
        &self,
        half: f64,
        radius: f64,
        bound: f64,
        fuel: &mut usize,
    ) -> Result<f64, AreaQueryError> {
        // Includes this eight-byte certificate's stack/read bookkeeping, finite
        // and ordering guards, exact doubling, two outward products and return.
        debit(fuel, 224)?;
        if !(half.is_finite()
            && half > 0.0
            && radius.is_finite()
            && radius > half
            && bound.is_finite()
            && bound >= 0.0)
        {
            return Err(AreaQueryError::Precision);
        }
        // Doubling a finite binary64 is exact, including subnormals, unless it
        // overflows. Reject overflow rather than comparing a rounded ratio.
        let twice_half = 2.0 * half;
        if !twice_half.is_finite() || radius < twice_half {
            return Err(AreaQueryError::Precision);
        }
        let error = ((self.upper * half).next_up() * bound).next_up();
        if !error.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        Ok(error)
    }
}

impl PositiveRule {
    fn new(fuel: &mut usize) -> Result<Self, AreaQueryError> {
        debit(fuel, 512)?;
        let mut powers = [iv(1.0); 8];
        let mut squares = [iv(0.0); 8];
        let mut weights = [iv(0.0); 8];
        for (index, &(node, weight)) in RULE.iter().enumerate() {
            debit(fuel, 256)?;
            if !(0.0 < node && node < 1.0 && weight > 0.0) {
                return Err(AreaQueryError::Precision);
            }
            squares[index] = checked(iv(node).mul(iv(node)))?;
            weights[index] = checked(iv(weight).mul(iv(2.0)))?;
        }
        let mut result = Self {
            moments: [0.0; 65],
            weight_sum: 0.0,
        };
        for (degree, residual) in result.moments.iter_mut().enumerate() {
            let mut sum = iv(0.0);
            for index in 0..8 {
                debit(fuel, 272)?;
                sum = checked(sum.add(checked(weights[index].mul(powers[index]))?))?;
                powers[index] = checked(powers[index].mul(squares[index]))?;
            }
            debit(fuel, 192)?;
            let actual = checked(iv(2.0).div(iv((2 * degree + 1) as f64)))?;
            let difference = checked(actual.sub(sum))?;
            *residual = difference.lo.abs().max(difference.hi.abs());
            if degree == 0 {
                result.weight_sum = sum.hi;
            }
        }
        if result.weight_sum > 3.0 {
            return Err(AreaQueryError::Precision);
        }
        Ok(result)
    }

    /// For |f|<=M on the disc radius R around the REAL panel center,
    /// Cauchy gives |a_j|<=M/R^j. Let q=h/R<1. The exact finite rule error
    /// is bounded by h*M*(sum_even |mu_j|q^j +
    /// (2+sum(w))*q^129/(1-q)). The second term includes every moment
    /// beyond 128, including odd ones although symmetry makes them zero.
    /// Nonnegative scalar FMA plus next_up computes an upper bound; this is
    /// intentionally different arithmetic from four-endpoint interval ops.
    #[cfg(test)]
    fn error(
        &self,
        half: f64,
        radius: f64,
        bound: f64,
        fuel: &mut usize,
    ) -> Result<f64, AreaQueryError> {
        debit(fuel, 256)?;
        let q = (half / radius).next_up();
        if !(0.0 < q && q < 1.0 && bound.is_finite() && bound >= 0.0) {
            return Err(AreaQueryError::Precision);
        }
        let factor = self.error_factor(q, fuel)?;
        let answer = ((factor * half).next_up() * bound).next_up();
        if !answer.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        Ok(answer)
    }

    /// All moment coefficients and the tail are nonnegative; hence the exact
    /// Cauchy expression is increasing in q on [0,1). Evaluating it outward at
    /// exact q=1/2 therefore encloses every certified panel with h/R<=1/2.
    /// The RULE-derived moment residuals and weight sum remain its authority.
    fn half_ratio_factor(&self, fuel: &mut usize) -> Result<HalfRatioFactor, AreaQueryError> {
        // Admit certificate storage/construction before evaluating its bound.
        debit(fuel, 64)?;
        Ok(HalfRatioFactor {
            upper: self.error_factor(0.5, fuel)?,
        })
    }

    /// Shared outward nonnegative polynomial and Cauchy-tail calculation.
    /// Callers have already established exact 0<q<1. The final debit includes
    /// the general error caller's two scale products and result check as before;
    /// the compiled-factor caller conservatively pays that same amount.
    fn error_factor(&self, q: f64, fuel: &mut usize) -> Result<f64, AreaQueryError> {
        let q2 = (q * q).next_up();
        let mut polynomial = 0.0_f64;
        let mut power = q;
        for coefficient in self.moments.iter().rev() {
            debit(fuel, 32)?;
            polynomial = polynomial.mul_add(q2, *coefficient).next_up();
            power = (power * q2).next_up();
        }
        // The loop gives q^131; compute q^129 independently, with the
        // exponentiation's actual correctly rounded products paid below.
        debit(fuel, 256)?;
        let mut tail_power = q;
        for _ in 0..7 {
            tail_power = (tail_power * tail_power).next_up();
        }
        tail_power = (tail_power * q).next_up();
        let denominator = (1.0 - q).next_down();
        let tail =
            (((2.0 + self.weight_sum).next_up() * tail_power).next_up() / denominator).next_up();
        let factor = (polynomial + tail).next_up();
        if !factor.is_finite() || !power.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        Ok(factor)
    }
}

/// A source-derived boundary in an integration parameter. Circle quarters
/// use the rational unit-circle parametrization so a tangent never introduces
/// an unbounded real square-root derivative. Coefficients must be exact source
/// identities already certified by ScanPlan; other conics remain unadmitted.
#[derive(Clone, Copy)]
enum MappedBoundary {
    Line {
        intercept: f64,
        slope: f64,
    },
    Circle {
        h: f64,
        a: f64,
        b: f64,
        k: f64,
        r: f64,
        sign: f64,
    },
}

#[derive(Clone, Copy)]
enum Components {
    X,
    Y,
    Position,
    All,
}

struct Coordinates {
    x: Interval,
    y: Interval,
    jacobian: Interval,
}

impl MappedBoundary {
    fn cutoff_events(
        self,
        center: [f64; 2],
        sigma: f64,
        fuel: &mut usize,
    ) -> Result<Option<Vec<Interval>>, AreaQueryError> {
        debit(fuel, 1_024)?;
        let halo = checked(iv(sigma).mul(iv(3.0)))?;
        let mut events = bounded_vector(16).ok_or(AreaQueryError::Resource)?;
        for sign_cut in [-1.0, 1.0] {
            let offset = checked(iv(sign_cut).mul(halo))?;
            let x_cut = checked(iv(center[0]).add(offset))?;
            let y_cut = checked(iv(center[1]).add(offset))?;
            match self {
                Self::Line { intercept, slope } => {
                    events.push(y_cut);
                    if slope != 0.0 {
                        debit(fuel, 192)?;
                        events.push(checked(checked(x_cut.sub(iv(intercept)))?.div(iv(slope)))?);
                    }
                }
                Self::Circle {
                    h,
                    a,
                    b,
                    k,
                    r,
                    sign,
                } => {
                    debit(fuel, 2_560)?;
                    let t = checked(checked(y_cut.sub(iv(k)))?.div(iv(sign * r)))?;
                    if t.hi >= 0.0 && t.lo <= 1.0 {
                        let t = checked(Interval::new(t.lo.max(0.0), t.hi.min(1.0)))?;
                        let radical = checked(iv(1.0).sub(checked(t.mul(t))?))?;
                        let root = sqrt_enclosure(checked(Interval::new(
                            radical.lo.max(0.0),
                            radical.hi,
                        ))?)
                        .ok_or(AreaQueryError::Precision)?;
                        events.push(checked(t.div(checked(iv(1.0).add(root))?))?);
                    }
                    // (h-x-br)u²+2*a*sign*r*u+(h-x+br)=0 is the
                    // exact rational-circle X-cutoff equation. A nonzero
                    // leading interval and a strictly signed discriminant
                    // supply a complete set of enclosed real roots.
                    let difference = checked(iv(h).sub(x_cut))?;
                    let leading = checked(difference.sub(iv(b * r)))?;
                    if leading.lo <= 0.0 && leading.hi >= 0.0 {
                        return Ok(None);
                    }
                    let linear = iv(2.0 * a * sign * r);
                    // The product above must be the exact source value.
                    if exact_mul(
                        exact_mul(exact_mul(2.0, a).ok_or(AreaQueryError::Precision)?, sign)
                            .ok_or(AreaQueryError::Precision)?,
                        r,
                    )
                    .is_none()
                    {
                        return Ok(None);
                    }
                    let constant = checked(difference.add(iv(b * r)))?;
                    let discriminant = checked(
                        checked(linear.mul(linear))?
                            .sub(checked(iv(4.0).mul(checked(leading.mul(constant))?))?),
                    )?;
                    if discriminant.hi < 0.0 {
                        continue;
                    }
                    if discriminant.lo < 0.0 {
                        return Ok(None);
                    }
                    let root = sqrt_enclosure(discriminant).ok_or(AreaQueryError::Precision)?;
                    let denominator = checked(iv(2.0).mul(leading))?;
                    events.push(checked(
                        checked(iv(-linear.lo).sub(root))?.div(denominator),
                    )?);
                    events.push(checked(
                        checked(iv(-linear.lo).add(root))?.div(denominator),
                    )?);
                }
            }
        }
        Ok(Some(events))
    }

    fn evaluate(
        self,
        u: Interval,
        components: Components,
        fuel: &mut usize,
    ) -> Result<Coordinates, AreaQueryError> {
        debit(fuel, 96)?;
        let want_x = !matches!(components, Components::Y);
        let want_y = !matches!(components, Components::X);
        let want_jacobian = matches!(components, Components::All);
        let mut result = Coordinates {
            x: iv(0.0),
            y: iv(0.0),
            jacobian: iv(0.0),
        };
        match self {
            Self::Line { intercept, slope } => {
                if want_x {
                    debit(fuel, 192)?;
                    result.x = checked(iv(intercept).add(checked(iv(slope).mul(u))?))?;
                }
                if want_y {
                    result.y = u;
                }
                if want_jacobian {
                    result.jacobian = iv(1.0);
                }
            }
            Self::Circle {
                h,
                a,
                b,
                k,
                r,
                sign,
            } => {
                // One formula inventory; unrequested coordinates/Jacobian
                // are not computed. Each paid block includes its scalar,
                // interval, branch and record work. The rational basis and
                // 1-u² are shared when more than one output needs them.
                debit(fuel, 224)?;
                let u2 = checked(u.mul(u))?;
                let denominator = checked(iv(1.0).add(u2))?;
                debit(fuel, 336)?;
                let sine = checked(checked(iv(2.0).mul(u))?.div(denominator))?;
                let y_delta = checked(iv(sign * r).mul(sine))?;
                if want_y {
                    debit(fuel, 64)?;
                    result.y = checked(iv(k).add(y_delta))?;
                }
                if want_x || want_jacobian {
                    debit(fuel, 64)?;
                    let complement = checked(iv(1.0).sub(u2))?;
                    if want_x {
                        debit(fuel, 448)?;
                        let cosine = checked(complement.div(denominator))?;
                        result.x = checked(
                            checked(iv(h).add(checked(iv(a).mul(y_delta))?))?
                                .add(checked(iv(b * r).mul(cosine))?),
                        )?;
                    }
                    if want_jacobian {
                        debit(fuel, 320)?;
                        result.jacobian = checked(
                            checked(iv(2.0 * r).mul(complement))?
                                .div(checked(denominator.mul(denominator))?),
                        )?;
                    }
                }
            }
        }
        Ok(result)
    }

    fn at(
        self,
        u: Interval,
        fuel: &mut usize,
    ) -> Result<(Interval, Interval, Interval), AreaQueryError> {
        let value = self.evaluate(u, Components::All, fuel)?;
        Ok((value.x, value.y, value.jacobian))
    }

    fn position(
        self,
        u: Interval,
        fuel: &mut usize,
    ) -> Result<(Interval, Interval), AreaQueryError> {
        let value = self.evaluate(u, Components::Position, fuel)?;
        Ok((value.x, value.y))
    }

    #[cfg(test)]
    fn abscissa(self, u: Interval, fuel: &mut usize) -> Result<Interval, AreaQueryError> {
        Ok(self.evaluate(u, Components::X, fuel)?.x)
    }

    fn ordinate(self, u: Interval, fuel: &mut usize) -> Result<Interval, AreaQueryError> {
        Ok(self.evaluate(u, Components::Y, fuel)?.y)
    }

    fn domain(
        plan: &ScanPlan,
        id: usize,
        y: Interval,
        fuel: &mut usize,
    ) -> Result<(Self, Interval), AreaQueryError> {
        debit(fuel, 256)?;
        match plan.curves[id].key {
            CurveKey::ExactLine(intercept, slope) => Ok((Self::Line { intercept, slope }, y)),
            CurveKey::ExactConic([h, a, b, k, r, c]) if r > 0.0 && c == 1.0 && b.abs() == 1.0 => {
                let sign = if y.lo >= k {
                    1.0
                } else if y.hi <= k {
                    -1.0
                } else {
                    return Err(AreaQueryError::Precision);
                };
                let edge = exact_add(k, sign * r).ok_or(AreaQueryError::Precision)?;
                if y.lo < k.min(edge) || y.hi > k.max(edge) {
                    return Err(AreaQueryError::Precision);
                }
                debit(fuel, 768)?;
                let t = checked(checked(y.sub(iv(k)))?.div(iv(sign * r)))?;
                // The independently certified quarter's real domain implies
                // t in [0,1]; remove only arithmetic excess at its endpoints.
                if t.hi < 0.0 || t.lo > 1.0 {
                    return Err(AreaQueryError::Precision);
                }
                let t = checked(Interval::new(t.lo.max(0.0), t.hi.min(1.0)))?;
                let radical = checked(iv(1.0).sub(checked(t.mul(t))?))?;
                let root = sqrt_enclosure(checked(Interval::new(radical.lo.max(0.0), radical.hi))?)
                    .ok_or(AreaQueryError::Precision)?;
                let u = checked(t.div(checked(iv(1.0).add(root))?))?;
                let u = checked(Interval::new(u.lo.max(0.0), u.hi.min(1.0)))?;
                Ok((
                    Self::Circle {
                        h,
                        a,
                        b,
                        k,
                        r,
                        sign,
                    },
                    u,
                ))
            }
            _ => Err(AreaQueryError::Unsupported),
        }
    }

    /// Bound the analytic complex extension on a radius-R disc around m.
    /// For the rational circle D=1+z², |D|>=1+m²-2|m|R-R².
    /// Integrating derivative bounds from the real center yields imaginary
    /// coordinate bounds IY=R*DY/sigma, IX=R*DX/sigma. Since Z=2G(3)>1,
    /// |k(y)|<=exp(IY²/2)/sigma and |F(x)|<=3+IX exp(IX²/2).
    /// Thus M=DY/sigma*(3+IX)*exp((IX²+IY²)/2) suffices. e^.5<2
    /// gives the exact power-of-two upper bound 2^ceil(IX²+IY²).
    fn complex_bound(
        self,
        middle: Interval,
        radius: f64,
        sigma: f64,
        fuel: &mut usize,
    ) -> Result<Option<f64>, AreaQueryError> {
        debit(fuel, 2_304)?;
        let (dy, dx) = match self {
            Self::Line { slope, .. } => (iv(1.0), iv(slope.abs())),
            Self::Circle { a, b, r, .. } => {
                let m = middle.lo.abs().max(middle.hi.abs());
                let perturb = checked(
                    checked(iv(2.0 * m).mul(iv(radius)))?.add(checked(iv(radius).mul(iv(radius)))?),
                )?;
                let d0 = checked(iv(1.0).add(checked(middle.mul(middle))?))?;
                let lower = checked(d0.sub(perturb))?.lo;
                if lower <= 0.0 {
                    return Ok(None);
                }
                let numerator = checked(iv(1.0).sub(checked(middle.mul(middle))?))?;
                let magnitude = numerator.lo.abs().max(numerator.hi.abs());
                let dy = checked(
                    checked(iv(2.0 * r).mul(checked(iv(magnitude).add(perturb))?))?
                        .div(checked(iv(lower).mul(iv(lower)))?),
                )?;
                let dc = checked(
                    checked(iv(4.0).mul(checked(iv(m).add(iv(radius)))?))?
                        .div(checked(iv(lower).mul(iv(lower)))?),
                )?;
                let dx = checked(
                    checked(iv(a.abs()).mul(dy))?.add(checked(iv((b * r).abs()).mul(dc))?),
                )?;
                (dy, dx)
            }
        };
        let iy = checked(checked(iv(radius).mul(dy))?.div(iv(sigma)))?;
        let ix = checked(checked(iv(radius).mul(dx))?.div(iv(sigma)))?;
        let exponent = checked(checked(iy.mul(iy))?.add(checked(ix.mul(ix))?))?
            .hi
            .ceil();
        if !(0.0..=48.0).contains(&exponent) {
            return Ok(None);
        }
        let power = f64::from_bits((1023 + exponent as u64) << 52);
        let bound = checked(
            checked(checked(dy.div(iv(sigma)))?.mul(checked(iv(3.0).add(ix))?))?.mul(iv(power)),
        )?
        .hi;
        Ok(Some(bound))
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Default, Debug)]
struct CompiledMetrics {
    cache_work: usize,
    build_work: usize,
    node_work: usize,
    point_work: usize,
    panel_work: usize,
    compiled_nodes: usize,
    line_nodes: usize,
    interval_circle_nodes: usize,
    point_queries: usize,
    allocated_bytes: usize,
    entries: usize,
}

struct CompiledEntry {
    key: [u64; 6],
    map: scalar_map::ScalarMap,
}

struct CompiledCache {
    // Entries append, so indices remain stable. Only the separate sorted
    // index moves. Both inventories have one fixed fallible capacity.
    entries: Vec<CompiledEntry>,
    order: Vec<usize>,
    #[cfg(test)]
    metrics: CompiledMetrics,
}

impl CompiledCache {
    fn retained_bound(capacity: usize) -> Option<usize> {
        capacity
            .checked_mul(std::mem::size_of::<CompiledEntry>() + std::mem::size_of::<usize>())?
            .checked_add(std::mem::size_of::<Self>())
    }
    fn peak_bound() -> Option<usize> {
        Self::retained_bound(MAX_COMPILED_MAPS)?
            .checked_add(std::mem::size_of::<CompiledEntry>())?
            .checked_add(std::mem::size_of::<[u64; 6]>())?
            .checked_add(std::mem::size_of::<scalar_map::Panel>())?
            .checked_add(std::mem::size_of::<Coordinates>())?
            // Constructor has no recursive/heap temporary: at most64
            // interval locals and128 scalar/metadata slots, including the
            // transient positive-bound sums and coefficient records.
            .checked_add(64 * std::mem::size_of::<Interval>() + 128 * std::mem::size_of::<f64>())
    }
    fn new(fuel: &mut usize) -> Result<Self, AreaQueryError> {
        Self::with_capacity(MAX_COMPILED_MAPS, fuel)
    }
    fn with_capacity(capacity: usize, fuel: &mut usize) -> Result<Self, AreaQueryError> {
        if capacity > MAX_COMPILED_MAPS {
            return Err(AreaQueryError::Resource);
        }
        let required = Self::peak_bound().ok_or(AreaQueryError::Resource)?;
        if required > MAX_BYTES {
            return Err(AreaQueryError::Resource);
        }
        #[cfg(test)]
        let before = *fuel;
        debit(fuel, 128 + 16 * capacity)?;
        let entries = bounded_vector(capacity).ok_or(AreaQueryError::Resource)?;
        let order = bounded_vector(capacity).ok_or(AreaQueryError::Resource)?;
        let allocated_bytes = entries
            .capacity()
            .checked_mul(std::mem::size_of::<CompiledEntry>())
            .and_then(|v| v.checked_add(order.capacity() * std::mem::size_of::<usize>()))
            .and_then(|v| v.checked_add(std::mem::size_of::<Self>()))
            .ok_or(AreaQueryError::Resource)?;
        if allocated_bytes > Self::retained_bound(capacity).ok_or(AreaQueryError::Resource)? {
            return Err(AreaQueryError::Resource);
        }
        Ok(Self {
            entries,
            order,
            #[cfg(test)]
            metrics: CompiledMetrics {
                cache_work: before - *fuel,
                allocated_bytes,
                ..CompiledMetrics::default()
            },
        })
    }
    fn pay(&mut self, fuel: &mut usize, amount: usize) -> Result<(), AreaQueryError> {
        #[cfg(test)]
        let before = *fuel;
        let result = debit(fuel, amount);
        #[cfg(test)]
        {
            self.metrics.cache_work += before - *fuel;
        }
        result
    }
    fn get(
        &mut self,
        map: MappedBoundary,
        fuel: &mut usize,
    ) -> Result<Option<usize>, AreaQueryError> {
        self.pay(fuel, 32)?;
        let MappedBoundary::Circle {
            h,
            a,
            b,
            k,
            r,
            sign,
        } = map
        else {
            return Ok(None);
        };
        self.pay(fuel, 96)?;
        // Original parameter bits, including the hemisphere. Approximate
        // compiled coefficients or intersecting numeric boxes are not keys.
        let key = [h, a, b, k, r, sign].map(f64::to_bits);
        let mut first = 0;
        let mut end = self.order.len();
        while first < end {
            self.pay(fuel, 96)?;
            let middle = first + (end - first) / 2;
            if self.entries[self.order[middle]].key < key {
                first = middle + 1;
            } else {
                end = middle;
            }
        }
        if let Some(&index) = self.order.get(first)
            && self.entries[index].key == key
        {
            return Ok(Some(index));
        }
        if self.entries.len() == self.entries.capacity()
            || self.order.len() == self.order.capacity()
        {
            return Err(AreaQueryError::Resource);
        }
        #[cfg(test)]
        let before = *fuel;
        let result = scalar_map::ScalarMap::new(map, fuel);
        #[cfg(test)]
        {
            self.metrics.build_work += before - *fuel;
        }
        let compiled = result?;
        // Prepay one full144-byte entry append plus index insert metadata;
        // each shifted index is one word, with bounded pointer/loop work.
        self.pay(fuel, 64 + 8 * (self.order.len() - first))?;
        let index = self.entries.len();
        self.entries.push(CompiledEntry { key, map: compiled });
        self.order.insert(first, index);
        #[cfg(test)]
        {
            self.metrics.entries = self.entries.len();
        }
        Ok(Some(index))
    }
}

#[derive(Clone, Copy)]
struct Band {
    start: f64,
    end: f64,
    first: usize,
    count: usize,
}

struct Strips {
    bands: Vec<Band>,
    runs: Vec<[usize; 2]>,
    unknown: Vec<Interval>,
    bounds: [f64; 4],
}

#[derive(Clone, Copy)]
struct RowRun {
    ids: [usize; 2],
    x: [Interval; 2],
}

#[derive(Clone, Copy)]
struct RowBand {
    y: Interval,
    first: usize,
    count: usize,
    #[cfg(test)]
    mass: Option<MassCertificate>,
}

#[derive(Clone, Copy, Debug)]
enum Class {
    Inside,
    Outside,
    YOnly,
    Line(usize, bool),
    General,
}

struct Lattice {
    root: [u32; 4],
    scale: f64,
    sigma: f64,
    offset: [f64; 2],
}

impl NeutralRegion<'_> {
    /// Prepare one disposable field in this source's owning logical frame.
    /// The caller supplies the existing global sample window, shared remaining
    /// CPU fuel and a host lease AFTER all other live invocation storage.
    /// Translation shifts query centers in this same logical frame. This
    /// entry point does not apply an ancestor affine or spread, select a
    /// physical consumer, or reset its budget.
    pub(in crate::runtime::execution_2d) fn prepare_gaussian_plane(
        &self,
        root: [u32; 4],
        scale: f64,
        sigma: f64,
        offset: [f64; 2],
        fuel: &mut usize,
        host_lease: usize,
    ) -> Result<plane::PreparedPlane, AreaQueryError> {
        // Capacity/extent arithmetic, finite guards, interval bounds and
        // scalar phase selection precede the first numerical allocation.
        debit(fuel, 1024)?;
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
            || !offset.into_iter().all(f64::is_finite)
        {
            return Err(AreaQueryError::Precision);
        }
        let fixed = MAX_AREA_QUERY_FIXED_BYTES as usize;
        let peak = fixed
            .checked_add(
                cells
                    .checked_mul(size_of::<u32>())
                    .ok_or(AreaQueryError::Resource)?,
            )
            .and_then(|v| v.checked_add(width.checked_mul(size_of::<f64>())?))
            .ok_or(AreaQueryError::Resource)?;
        // Boundary construction, scan events and later numeric packing are
        // distinct phases. Each fits the SAME fixed reservation; width-sized
        // map/rows coexist only in the subsequent complete-field phase.
        if PreparedBoundaries::fixed_scratch_bytes().is_none_or(|bytes| bytes > fixed)
            || peak > host_lease.min(MAX_BYTES)
        {
            return Err(AreaQueryError::Resource);
        }
        let halo = checked(iv(sigma).mul(iv(3.0)))?;
        let axis = |lo: u32, hi: u32, shift: f64| -> Result<[f64; 2], AreaQueryError> {
            let lower = checked(
                checked(checked(iv(f64::from(lo)).div(iv(scale)))?.sub(iv(shift)))?.sub(halo),
            )?
            .lo
            .floor();
            let upper = checked(
                checked(checked(iv(f64::from(hi)).div(iv(scale)))?.sub(iv(shift)))?.add(halo),
            )?
            .hi
            .ceil();
            // Integer numerical-domain planes avoid manufacturing a rounded
            // geometric transform. They only enclose the complete kernel
            // windows; original source membership still selects occupancy.
            if lower.abs().max(upper.abs()) > 4_503_599_627_370_496.0 || lower >= upper {
                return Err(AreaQueryError::Precision);
            }
            Ok([lower, upper])
        };
        let [left, right] = axis(root[0], root[2], offset[0])?;
        let [top, bottom] = axis(root[1], root[3], offset[1])?;
        let prepared = PreparedBoundaries::prepare(self, fuel)?;
        let plan = ScanPlan::prepare(self, None, &prepared, fuel)?;
        let strips = Strips::prepare(&plan, self, &prepared, [left, top, right, bottom], fuel)?;
        plane::prepare(
            &strips,
            &plan,
            &prepared,
            self,
            Lattice {
                root,
                scale,
                sigma,
                offset,
            },
            fuel,
        )
    }
}

fn line_range(
    intercept: f64,
    slope: f64,
    y: Interval,
    fuel: &mut usize,
) -> Result<Interval, AreaQueryError> {
    debit(fuel, 64)?;
    let a = slope.mul_add(y.lo, intercept);
    let b = slope.mul_add(y.hi, intercept);
    Interval::new(a.min(b).next_down(), a.max(b).next_up()).ok_or(AreaQueryError::Precision)
}

fn curve_range(
    plan: &ScanPlan,
    bounds: [f64; 4],
    id: usize,
    y: Interval,
    fuel: &mut usize,
) -> Result<Interval, AreaQueryError> {
    debit(fuel, 32)?;
    if id >= plan.curves.len() {
        return Interval::scalar(bounds[if id == plan.curves.len() { 0 } else { 2 }])
            .ok_or(AreaQueryError::Precision);
    }
    let curve = &plan.curves[id];
    if let CurveKey::ExactLine(intercept, slope) = curve.key {
        return line_range(intercept, slope, y, fuel);
    }
    debit(fuel, VALUE_WORK)?;
    // A coefficient enclosure can leave the radical's numeric domain at
    // a tangent. Its already certified complete reach remains conservative;
    // this broadens classification, never invents membership or a value.
    Ok(curve.function.value(y).unwrap_or(curve.reach))
}

fn classify_piece(
    plan: &ScanPlan,
    runs: &[RowRun],
    window: Interval,
    fuel: &mut usize,
) -> Result<Class, AreaQueryError> {
    let mut boundary = None;
    for run in runs {
        debit(fuel, 64)?;
        let [left, right] = run.x;
        if right.hi <= window.lo || left.lo >= window.hi {
            continue;
        }
        if left.hi <= window.lo && right.lo >= window.hi {
            return Ok(Class::Inside);
        }
        let candidate = match (left.hi > window.lo, right.lo < window.hi) {
            (true, false) => (run.ids[0], false),
            (false, true) => (run.ids[1], true),
            _ => return Ok(Class::General),
        };
        if boundary.is_some()
            || !plan
                .curves
                .get(candidate.0)
                .is_some_and(|curve| matches!(curve.key, CurveKey::ExactLine(..)))
        {
            return Ok(Class::General);
        }
        boundary = Some(candidate);
    }
    Ok(boundary.map_or(Class::Outside, |(id, inside_left)| {
        Class::Line(id, inside_left)
    }))
}

fn classify_site(
    plan: &ScanPlan,
    bands: &[RowBand],
    runs: &[RowRun],
    window: Interval,
    classes: &mut Vec<Class>,
    fuel: &mut usize,
) -> Result<Class, AreaQueryError> {
    classes.clear();
    let mut candidate = None;
    let mut all_inside = true;
    let mut all_outside = true;
    for band in bands {
        debit(fuel, 32)?;
        let class = classify_piece(
            plan,
            &runs[band.first..band.first + band.count],
            window,
            fuel,
        )?;
        match class {
            Class::Inside => all_outside = false,
            Class::Outside => all_inside = false,
            Class::Line(id, side) => {
                all_inside = false;
                all_outside = false;
                if let Some((previous, previous_side)) = candidate {
                    if side != previous_side
                        || !plan.curves[id].same_function(&plan.curves[previous])
                    {
                        return Ok(Class::General);
                    }
                } else {
                    candidate = Some((id, side));
                }
            }
            Class::General => return Ok(Class::General),
            Class::YOnly => return Err(AreaQueryError::Precision),
        }
        classes.push(class);
    }
    if all_outside {
        return Ok(Class::Outside);
    }
    if all_inside {
        return Ok(Class::Inside);
    }
    let Some((id, side)) = candidate else {
        return Ok(Class::YOnly);
    };
    let CurveKey::ExactLine(intercept, slope) = plan.curves[id].key else {
        return Err(AreaQueryError::Precision);
    };
    for (band, class) in bands.iter().zip(classes) {
        debit(fuel, 32)?;
        if matches!(class, Class::Line(..)) {
            continue;
        }
        let extent = line_range(intercept, slope, band.y, fuel)?;
        let model_inside = if side {
            extent.lo >= window.hi
        } else {
            extent.hi <= window.lo
        };
        let model_outside = if side {
            extent.hi <= window.lo
        } else {
            extent.lo >= window.hi
        };
        if !(matches!(class, Class::Inside) && model_inside
            || matches!(class, Class::Outside) && model_outside)
        {
            return Ok(Class::General);
        }
    }
    Ok(Class::Line(id, side))
}

impl Strips {
    fn prepare(
        plan: &ScanPlan,
        region: &NeutralRegion<'_>,
        prepared: &PreparedBoundaries,
        bounds: [f64; 4],
        fuel: &mut usize,
    ) -> Result<Self, AreaQueryError> {
        debit(fuel, 256)?;
        if !bounds.iter().all(|v| v.is_finite())
            || bounds[0] >= bounds[2]
            || bounds[1] >= bounds[3]
            || !plan.complete_crossings(fuel)?
        {
            return Err(AreaQueryError::Precision);
        }
        let [x0, y0, x1, y1] = bounds;
        // Admit the complete coexisting construction inventory before its
        // first event, crossing, membership or result buffer is allocated.
        let _peak = storage::strip_preparation(prepared, plan, fuel)?;
        let count = plan.curves.len();
        let left = ScanPlan::constant_curve(x0, y0, y1).ok_or(AreaQueryError::Precision)?;
        let right = ScanPlan::constant_curve(x1, y0, y1).ok_or(AreaQueryError::Precision)?;
        let mut cuts = bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?;
        let mut semantic = bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?;
        let mut uncertainty = bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?;
        cuts.extend([y0, y1]);
        semantic.extend([y0, y1]);
        let mut append =
            |root: Interval, source_event: bool, fuel: &mut usize| -> Result<(), AreaQueryError> {
                debit(fuel, 32)?;
                if root.hi <= y0 || root.lo >= y1 {
                    return Ok(());
                }
                if cuts.len() + 2 > MAX_BANDS || uncertainty.len() == MAX_BANDS {
                    return Err(AreaQueryError::Resource);
                }
                cuts.extend([root.lo.max(y0), root.hi.min(y1)]);
                if source_event {
                    if semantic.len() + 2 > MAX_BANDS {
                        return Err(AreaQueryError::Resource);
                    }
                    semantic.extend([root.lo.max(y0), root.hi.min(y1)]);
                }
                if root.lo < root.hi {
                    uncertainty.push(root);
                }
                Ok(())
            };
        for curve in &plan.curves {
            append(curve.start, true, fuel)?;
            append(curve.end, true, fuel)?;
        }
        for &event in &plan.extra_y_events {
            append(event, true, fuel)?;
        }
        for crossing in &plan.crossings {
            debit(fuel, 16)?;
            for root in crossing.roots.into_iter().flatten() {
                append(root, false, fuel)?;
            }
        }
        // The two finite query-domain planes are numerical integration
        // boundaries. Their complete crossings must partition topology too.
        let mut planes = bounded_vector(count).ok_or(AreaQueryError::Resource)?;
        for curve in &plan.curves {
            let mut pair = [Crossings {
                complete: true,
                roots: [None, None],
            }; 2];
            for (slot, edge) in pair.iter_mut().zip([&left, &right]) {
                debit(fuel, 128 + ScanPlan::crossing_work(curve, edge))?;
                *slot = ScanPlan::crossings(curve, edge).ok_or(AreaQueryError::Precision)?;
                if !slot.complete {
                    return Err(AreaQueryError::Precision);
                }
                for root in slot.roots.into_iter().flatten() {
                    append(root, true, fuel)?;
                }
            }
            planes.push(pair);
        }
        sort_admitted(&mut cuts, f64::total_cmp, 4, fuel).ok_or(AreaQueryError::Resource)?;
        debit(fuel, cuts.len() * 4)?;
        cuts.dedup();
        sort_admitted(&mut semantic, f64::total_cmp, 4, fuel).ok_or(AreaQueryError::Resource)?;
        debit(fuel, semantic.len() * 4)?;
        semantic.dedup();
        sort_admitted(
            &mut uncertainty,
            |a, b| a.lo.total_cmp(&b.lo).then(a.hi.total_cmp(&b.hi)),
            8,
            fuel,
        )
        .ok_or(AreaQueryError::Resource)?;
        let mut retained = 0;
        for index in 0..uncertainty.len() {
            debit(fuel, 32)?;
            let root = uncertainty[index];
            if retained > 0 && root.lo <= uncertainty[retained - 1].hi {
                uncertainty[retained - 1].hi = uncertainty[retained - 1].hi.max(root.hi);
            } else {
                uncertainty[retained] = root;
                retained += 1;
            }
        }
        uncertainty.truncate(retained);
        let mut curve_cuts = bounded_vector(count).ok_or(AreaQueryError::Resource)?;
        for id in 0..count {
            let mut events = bounded_vector(4 * count).ok_or(AreaQueryError::Resource)?;
            for other in 0..count {
                debit(fuel, 64)?;
                let crossing = plan.crossing(id, other).ok_or(AreaQueryError::Precision)?;
                for root in crossing.roots.into_iter().flatten() {
                    if root.hi > y0 && root.lo < y1 {
                        events.extend([root.lo, root.hi]);
                    }
                }
            }
            sort_admitted(&mut events, f64::total_cmp, 4, fuel).ok_or(AreaQueryError::Resource)?;
            debit(fuel, events.len() * 4)?;
            events.dedup();
            curve_cuts.push(events);
        }
        let mut xs = bounded_vector(2).ok_or(AreaQueryError::Resource)?;
        xs.extend([x0, x1]);
        let mut query = GridQuery {
            region,
            forward: None,
            prepared,
            xs,
            cell_width: Interval::new(x1 - x0, x1 - x0).ok_or(AreaQueryError::Precision)?,
            vertical_uncertainty: Vec::new(),
            semantic_cuts: semantic,
            curve_cuts,
            membership: Some(MembershipCache::new(fuel)?),
        };
        let mut result = Self {
            bands: bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?,
            runs: bounded_vector(MAX_RUNS).ok_or(AreaQueryError::Resource)?,
            unknown: bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?,
            bounds,
        };
        let mut active = bounded_vector(count + 2).ok_or(AreaQueryError::Resource)?;
        for pair in cuts.windows(2) {
            debit(fuel, 128)?;
            let [a, b] = [pair[0], pair[1]];
            let unknown = uncertainty.partition_point(|root| root.hi <= a);
            if uncertainty.get(unknown).is_some_and(|root| root.lo < b) {
                result
                    .unknown
                    .push(Interval::new(a, b).ok_or(AreaQueryError::Precision)?);
                continue;
            }
            let middle = a * 0.5 + b * 0.5;
            if !(a < middle && middle < b) {
                result
                    .unknown
                    .push(Interval::new(a, b).ok_or(AreaQueryError::Precision)?);
                continue;
            }
            active.clear();
            let mut ambiguous = false;
            for (id, curve) in plan
                .curves
                .iter()
                .enumerate()
                .chain([(count, &left), (count + 1, &right)])
            {
                debit(fuel, 32)?;
                if curve.reach.hi < x0
                    || curve.reach.lo > x1
                    || b <= curve.start.lo
                    || a >= curve.end.hi
                {
                    continue;
                }
                if a < curve.start.hi || b > curve.end.lo {
                    ambiguous = true;
                    break;
                }
                debit(fuel, VALUE_WORK + 16 * active.len())?;
                let value = curve
                    .function
                    .value(Interval::scalar(middle).ok_or(AreaQueryError::Precision)?)
                    .ok_or(AreaQueryError::Precision)?;
                if !active
                    .iter()
                    .any(|&(_, old, _, _): &(usize, &Curve, Interval, f64)| {
                        old.same_function(curve)
                    })
                {
                    active.push((id, curve, value, value.lo * 0.5 + value.hi * 0.5));
                }
            }
            sort_admitted(&mut active, |a, b| a.3.total_cmp(&b.3), 4, fuel)
                .ok_or(AreaQueryError::Resource)?;
            let first = result.runs.len();
            let mut occupied: Option<[usize; 2]> = None;
            for pair in active.windows(2) {
                debit(fuel, 256)?;
                let [(lid, l, lx, _), (rid, r, rx, _)] = [pair[0], pair[1]];
                if lx.hi >= rx.lo {
                    ambiguous = true;
                    break;
                }
                let crossing = if lid < count && rid < count {
                    plan.crossing(lid, rid).ok_or(AreaQueryError::Precision)?
                } else if lid < count {
                    planes[lid][rid - count]
                } else if rid < count {
                    planes[rid][lid - count]
                } else {
                    Crossings {
                        complete: true,
                        roots: [None, None],
                    }
                };
                if !crossing.complete
                    || crossing
                        .roots
                        .into_iter()
                        .flatten()
                        .any(|root| root.lo < b && root.hi > a)
                {
                    ambiguous = true;
                    break;
                }
                let x = lx.hi * 0.5 + rx.lo * 0.5;
                if x < x0 || x > x1 {
                    continue;
                }
                let point = Cell::new([x, middle, x, middle]).ok_or(AreaQueryError::Precision)?;
                match query.membership([lid, rid], point, middle, fuel)? {
                    CellCertificate::Unknown => {
                        ambiguous = true;
                        break;
                    }
                    CellCertificate::Inside => {
                        if let Some(run) = occupied.as_mut() {
                            run[1] = rid;
                        } else {
                            occupied = Some([lid, rid]);
                        }
                    }
                    CellCertificate::Outside => {
                        if let Some(run) = occupied.take() {
                            if result.runs.len() == MAX_RUNS {
                                return Err(AreaQueryError::Resource);
                            }
                            result.runs.push(run);
                        }
                    }
                }
                // Ordering is certified by the complete intersection matrix,
                // not by the finite midpoint values used only for sorting.
                let _ = (l, r);
            }
            if ambiguous {
                result.runs.truncate(first);
                result
                    .unknown
                    .push(Interval::new(a, b).ok_or(AreaQueryError::Precision)?);
                continue;
            }
            if let Some(run) = occupied {
                if result.runs.len() == MAX_RUNS {
                    return Err(AreaQueryError::Resource);
                }
                result.runs.push(run);
            }
            result.bands.push(Band {
                start: a,
                end: b,
                first,
                count: result.runs.len() - first,
            });
        }
        // Adjacent unresolved subdivisions are one geometric uncertainty
        // interval. Their union, not repeated numerical endpoint queries,
        // supplies the later Gaussian possible-mass allowance.
        let mut retained = 0;
        for index in 0..result.unknown.len() {
            debit(fuel, 32)?;
            let interval = result.unknown[index];
            if retained > 0 && interval.lo <= result.unknown[retained - 1].hi {
                result.unknown[retained - 1].hi = result.unknown[retained - 1].hi.max(interval.hi);
            } else {
                result.unknown[retained] = interval;
                retained += 1;
            }
        }
        result.unknown.truncate(retained);
        Ok(result)
    }

    fn retained_bytes(&self) -> Option<usize> {
        self.bands
            .capacity()
            .checked_mul(std::mem::size_of::<Band>())?
            .checked_add(
                self.runs
                    .capacity()
                    .checked_mul(std::mem::size_of::<[usize; 2]>())?,
            )?
            .checked_add(
                self.unknown
                    .capacity()
                    .checked_mul(std::mem::size_of::<Interval>())?,
            )?
            .checked_add(std::mem::size_of::<Self>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dFillRule, Render2dOpacity, Render2dPoint,
    };

    pub(super) fn original_ring() -> Render2dItem {
        let p = |x, y| Render2dPoint::new(x, y).unwrap();
        let mut commands = Vec::new();
        for [x0, y0, x1, y1] in [[10.0, 10.0, 30.0, 30.0], [18.0, 18.0, 22.0, 22.0]] {
            commands.extend([
                Render2dPathCommand::MoveTo(p(x0, y0)),
                Render2dPathCommand::LineTo(p(x1, y0)),
                Render2dPathCommand::LineTo(p(x1, y1)),
                Render2dPathCommand::LineTo(p(x0, y1)),
                Render2dPathCommand::Close,
            ]);
        }
        Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::path(
                    Render2dPath::new(Render2dFillRule::EvenOdd, commands).unwrap(),
                ),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        )
    }

    #[test]
    fn finite_positive_rule_encloses_independent_dyadic_moment_residuals() {
        // Decimal110 reference evaluates the EXACT f64 constants using
        // Decimal.from_float, then the exact monomial integral 2/(n+1).
        // This is independent of the interval construction/recurrence.
        const REFERENCES: [(usize, f64); 12] = [
            (0, 6.245_004_513_516_506e-17),
            (2, 2.206_076_356_714_017_3e-17),
            (4, 1.173_147_567_545_378_5e-17),
            (8, 1.401_290_262_765_438e-17),
            (16, 2.477_592_367_196_234_5e-17),
            (30, 2.533_442_834_036_622_7e-17),
            (32, 7.204_615_084_085_455e-10),
            (34, 5.938_329_461_197_097e-9),
            (48, 3.314_297_837_988_242e-6),
            (64, 5.437_594_912_950_716e-5),
            (96, 5.715_316_423_942_897e-4),
            (128, 1.536_250_598_987_502_5e-3),
        ];
        let mut fuel = 16_777_216;
        let rule = PositiveRule::new(&mut fuel).unwrap();
        for (degree, reference) in REFERENCES {
            assert!(reference.next_up() <= rule.moments[degree / 2]);
        }
        assert!(rule.weight_sum >= 2.0 && rule.weight_sum < 3.0);
        let error = rule.error(0.25, 1.0, 1.0, &mut fuel).unwrap();
        assert!(error > 0.0 && error < 1.0e-12);
        assert!(matches!(
            rule.error(1.0, 1.0, 1.0, &mut fuel),
            Err(AreaQueryError::Precision)
        ));
        let mut empty = 0;
        assert!(matches!(
            PositiveRule::new(&mut empty),
            Err(AreaQueryError::Resource)
        ));
    }

    #[test]
    fn half_ratio_factor_encloses_independent_rational_polynomial_and_tail() {
        // Independent exact expressions: q=1/2, weight sum=2 gives tail
        // 4*q^129/(1-q)=2^-126. Coefficients 1/8 and 1/4 at degrees 0,2
        // add exactly 1/8+(1/4)*(1/4)=3/16 to that tail.
        let mut fuel = 1_000_000;
        let mut rule = PositiveRule {
            moments: [0.0; 65],
            weight_sum: 2.0,
        };
        let tail = 2.0_f64.powi(-126);
        let compiled = rule.half_ratio_factor(&mut fuel).unwrap();
        assert!(compiled.upper >= tail);
        assert!(compiled.upper <= tail * (1.0 + 2.0_f64.powi(-40)));
        rule.moments[0] = 0.125;
        rule.moments[1] = 0.25;
        let compiled = rule.half_ratio_factor(&mut fuel).unwrap();
        assert!(compiled.upper > 0.1875);
        assert!(compiled.upper < 0.1875 + 2.0_f64.powi(-45));
        let scaled = compiled.error(0.25, 0.5, 2.0, &mut fuel).unwrap();
        assert!(scaled > 0.09375 && scaled < 0.09375 + 2.0_f64.powi(-45));
    }

    #[test]
    fn half_ratio_factor_reuses_bound_with_normal_and_subnormal_scales() {
        let mut fuel = 16_777_216;
        let rule = PositiveRule::new(&mut fuel).unwrap();
        let before = fuel;
        let compiled = rule.half_ratio_factor(&mut fuel).unwrap();
        let compile_work = before - fuel;
        assert_eq!(std::mem::size_of::<HalfRatioFactor>(), 8);
        for half in [
            f64::from_bits(1),
            f64::from_bits(16),
            f64::MIN_POSITIVE * 0.5,
            2.0_f64.powi(-500),
            0.03125,
            0.5,
            2.0_f64.powi(500),
        ] {
            let radius = 2.0 * half;
            let mut previous = 0.0;
            for bound in [0.0, f64::from_bits(1), 1.0, 2.0_f64.powi(500)] {
                let before = fuel;
                let actual = compiled.error(half, radius, bound, &mut fuel).unwrap();
                assert_eq!(before - fuel, 224);
                let before = fuel;
                let general = rule.error(half, radius, bound, &mut fuel).unwrap();
                assert_eq!(before - fuel, 2592);
                // General error rounds q upward from 1/2. The reused certificate
                // uses exact 1/2, so is at least as tight, with identical scaling.
                assert!(actual <= general);
                assert!(actual >= previous);
                previous = actual;
            }
        }
        let at_half = compiled.error(0.25, 0.5, 1.0, &mut fuel).unwrap();
        for radius in [0.5_f64.next_up(), 1.0, 2.0, 16.0] {
            let reused = compiled.error(0.25, radius, 1.0, &mut fuel).unwrap();
            let general = rule.error(0.25, radius, 1.0, &mut fuel).unwrap();
            assert_eq!(reused, at_half);
            assert!(reused >= general);
        }
        println!(
            "Half-ratio Cauchy certificate: compile_work={compile_work}, apply_work=224, general_work=2592, retained_bytes=8, factor={:e}; numerical bound only",
            compiled.upper
        );
    }

    #[test]
    fn half_ratio_factor_rejects_unproved_ratio_invalid_values_and_short_fuel() {
        let mut fuel = 1_000_000;
        let rule = PositiveRule::new(&mut fuel).unwrap();
        let compiled = rule.half_ratio_factor(&mut fuel).unwrap();
        for (half, radius, bound) in [
            (0.0, 1.0, 1.0),
            (-0.25, 1.0, 1.0),
            (f64::NAN, 1.0, 1.0),
            (0.25, 0.5_f64.next_down(), 1.0),
            (0.25, 0.0, 1.0),
            (0.25, f64::INFINITY, 1.0),
            (0.25, f64::NAN, 1.0),
            (0.25, 1.0, -1.0),
            (0.25, 1.0, f64::INFINITY),
            (0.25, 1.0, f64::NAN),
            (f64::MAX * 0.75, f64::MAX, 1.0),
            (1.0e300, 2.0e300, 1.0e300),
        ] {
            assert!(matches!(
                compiled.error(half, radius, bound, &mut fuel),
                Err(AreaQueryError::Precision)
            ));
        }
        let mut short = 223;
        assert!(matches!(
            compiled.error(0.25, 0.5, 1.0, &mut short),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(short, 0);
        let mut short = 2399;
        assert!(matches!(
            rule.half_ratio_factor(&mut short),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(short, 0);
        let mut exact = 2400;
        rule.half_ratio_factor(&mut exact).unwrap();
        assert_eq!(exact, 0);
    }

    #[test]
    fn gaussian_reuse_mass_certificate_rejects_a_different_measure() {
        let mut fuel = 16_777_216;
        let certificate = MassCertificate {
            y: Interval::new(-0.125, 0.125).unwrap(),
            center: 0.0,
            sigma: 0.1,
            value: iv(0.25),
        };
        assert_eq!(
            certificate
                .get(certificate.y, 0.0, 0.1, &mut fuel)
                .unwrap()
                .lo,
            0.25
        );
        for (y, center, sigma) in [
            (Interval::new(-0.25, 0.125).unwrap(), 0.0, 0.1),
            (Interval::new(-0.125, 0.25).unwrap(), 0.0, 0.1),
            (certificate.y, 0.25, 0.1),
            (certificate.y, 0.0, 0.2),
        ] {
            assert!(matches!(
                certificate.get(y, center, sigma, &mut fuel),
                Err(AreaQueryError::Precision)
            ));
        }
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        let mut diagnostics = Diagnostics::default();
        let integrator = Integrator::new(&kernel, 0.1, &mut diagnostics, &mut fuel).unwrap();
        let value = integrator
            .certify_mass(certificate.y, 0.0, &mut fuel)
            .unwrap();
        let direct = kernel.mass(iv(-1.25), iv(1.25), &mut fuel).unwrap();
        assert!(value.value.lo <= direct.hi && direct.lo <= value.value.hi);
    }

    #[test]
    fn gaussian_reuse_components_match_independent_rational_circle_values() {
        for b in [-1.0, 1.0] {
            for sign in [-1.0, 1.0] {
                let map = MappedBoundary::Circle {
                    h: 4.0,
                    a: 0.25,
                    b,
                    k: 2.0,
                    r: 1.0,
                    sign,
                };
                // Rational unit-circle points at t=0,1/2,1 are respectively
                // (C,S)=(1,0),(3/5,4/5),(0,1); |Y'|=2,24/25,0.
                for (u, x, y, jacobian) in [
                    (0.0, 4.0 + b, 2.0, 2.0),
                    (0.5, 4.0 + sign * 0.2 + b * 0.6, 2.0 + sign * 0.8, 0.96),
                    (1.0, 4.0 + sign * 0.25, 2.0 + sign, 0.0),
                ] {
                    let mut fuel = 16_777_216;
                    let all = map.at(iv(u), &mut fuel).unwrap();
                    for (range, expected) in [(all.0, x), (all.1, y), (all.2, jacobian)] {
                        assert!(range.lo <= expected && expected <= range.hi);
                    }
                    let before = fuel;
                    let ordinate = map.ordinate(iv(u), &mut fuel).unwrap();
                    let ordinate_work = before - fuel;
                    let before = fuel;
                    map.at(iv(u), &mut fuel).unwrap();
                    assert!(ordinate_work < before - fuel);
                    let abscissa = map.abscissa(iv(u), &mut fuel).unwrap();
                    let position = map.position(iv(u), &mut fuel).unwrap();
                    for (left, right) in [
                        (ordinate, all.1),
                        (abscissa, all.0),
                        (position.0, all.0),
                        (position.1, all.1),
                    ] {
                        assert_eq!((left.lo, left.hi), (right.lo, right.hi));
                    }
                }
            }
        }
    }

    #[test]
    fn gaussian_reuse_even_halfplane_profile_shares_one_exact_phase() {
        // Independent Decimal200 direct Maclaurin moments: integrate
        // E(z)*G(1.25+z/4) on [-3,3], then .5+integral/Z². The
        // 71 G terms and 150 E terms are independent of runtime patches,
        // quadrature nodes, cutoff event equations and cache arithmetic.
        const REFERENCE: f64 = 0.888_596_387_393_886_6;
        let mut fuel = 16_777_216;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        let mut diagnostics = Diagnostics::default();
        let mut integrator = Integrator::new(&kernel, 0.1, &mut diagnostics, &mut fuel).unwrap();
        let positive = integrator.profile(0.125, 0.25, &mut fuel).unwrap();
        assert!(positive.lo <= REFERENCE && REFERENCE <= positive.hi);
        let panels = integrator.panels;
        let negative = integrator.profile(-0.125, 0.25, &mut fuel).unwrap();
        let slope_reflected = integrator.profile(0.125, -0.25, &mut fuel).unwrap();
        assert!(negative.lo <= 1.0 - REFERENCE && 1.0 - REFERENCE <= negative.hi);
        assert_eq!(
            (positive.lo, positive.hi),
            (slope_reflected.lo, slope_reflected.hi)
        );
        assert_eq!(integrator.profiles.len(), 1);
        assert_eq!(integrator.panels, panels);
    }

    #[test]
    fn gaussian_compiled_cache_preserves_hemispheres_bits_and_stable_indices() {
        let mut fuel = 16_777_216;
        let mut cache = CompiledCache::new(&mut fuel).unwrap();
        let north = MappedBoundary::Circle {
            h: 4.0,
            a: 0.25,
            b: 1.0,
            k: 2.0,
            r: 1.0,
            sign: 1.0,
        };
        let south = MappedBoundary::Circle {
            h: 4.0,
            a: 0.25,
            b: 1.0,
            k: 2.0,
            r: 1.0,
            sign: -1.0,
        };
        let ni = cache.get(north, &mut fuel).unwrap().unwrap();
        let build = cache.metrics.build_work;
        assert_eq!(cache.get(north, &mut fuel).unwrap(), Some(ni));
        assert_eq!(cache.metrics.build_work, build);
        let si = cache.get(south, &mut fuel).unwrap().unwrap();
        assert_ne!(ni, si);
        let before_key = MappedBoundary::Circle {
            h: 0.0,
            a: 0.25,
            b: 1.0,
            k: 2.0,
            r: 1.0,
            sign: 1.0,
        };
        cache.get(before_key, &mut fuel).unwrap();
        assert_eq!(cache.get(north, &mut fuel).unwrap(), Some(ni));
        assert_eq!(cache.get(south, &mut fuel).unwrap(), Some(si));
        let almost = MappedBoundary::Circle {
            h: 4.0,
            a: 0.25_f64.next_up(),
            b: 1.0,
            k: 2.0,
            r: 1.0,
            sign: 1.0,
        };
        assert_ne!(cache.get(almost, &mut fuel).unwrap(), Some(ni));
        let n = cache.entries[ni]
            .map
            .at_exact(0.5, Components::Position, &mut fuel)
            .unwrap();
        let s = cache.entries[si]
            .map
            .at_exact(0.5, Components::Position, &mut fuel)
            .unwrap();
        assert!(n.y.lo <= 2.8 && 2.8 <= n.y.hi);
        assert!(s.y.lo <= 1.2 && 1.2 <= s.y.hi);
        assert!(
            cache
                .order
                .windows(2)
                .all(|pair| cache.entries[pair[0]].key < cache.entries[pair[1]].key)
        );
        assert_eq!(
            cache
                .get(
                    MappedBoundary::Line {
                        intercept: 0.0,
                        slope: 1.0
                    },
                    &mut fuel
                )
                .unwrap(),
            None
        );
        assert_eq!(cache.entries.len(), 4);
        assert_eq!(std::mem::size_of::<CompiledEntry>(), 144);
        assert_eq!(
            cache.metrics.allocated_bytes,
            CompiledCache::retained_bound(MAX_COMPILED_MAPS).unwrap()
        );
        println!(
            "Compiled integration inventory: entry={}, cache_header={}, retained={}, peak={}",
            std::mem::size_of::<CompiledEntry>(),
            std::mem::size_of::<CompiledCache>(),
            cache.metrics.allocated_bytes,
            CompiledCache::peak_bound().unwrap()
        );
    }

    #[test]
    fn gaussian_compiled_cache_capacity_and_fuel_fail_without_growth() {
        let mut fuel = 16_777_216;
        let mut cache = CompiledCache::with_capacity(2, &mut fuel).unwrap();
        for sign in [-1.0, 1.0] {
            cache
                .get(
                    MappedBoundary::Circle {
                        h: 4.0,
                        a: 0.25,
                        b: 1.0,
                        k: 2.0,
                        r: 1.0,
                        sign,
                    },
                    &mut fuel,
                )
                .unwrap();
        }
        let build = cache.metrics.build_work;
        let retained = cache.metrics.allocated_bytes;
        assert!(matches!(
            cache.get(
                MappedBoundary::Circle {
                    h: 0.0,
                    a: 0.25,
                    b: 1.0,
                    k: 2.0,
                    r: 1.0,
                    sign: 1.0
                },
                &mut fuel
            ),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(cache.entries.len(), 2);
        assert_eq!(cache.order.len(), 2);
        assert_eq!(cache.metrics.build_work, build);
        assert_eq!(cache.metrics.allocated_bytes, retained);
        assert!(matches!(
            CompiledCache::with_capacity(MAX_COMPILED_MAPS + 1, &mut fuel),
            Err(AreaQueryError::Resource)
        ));
        let mut empty = 0;
        assert!(matches!(
            CompiledCache::new(&mut empty),
            Err(AreaQueryError::Resource)
        ));
        assert!(matches!(
            cache.get(
                MappedBoundary::Circle {
                    h: 0.0,
                    a: 0.25,
                    b: 1.0,
                    k: 2.0,
                    r: 1.0,
                    sign: 1.0
                },
                &mut empty
            ),
            Err(AreaQueryError::Resource)
        ));
    }

    #[test]
    fn gaussian_compiled_panel_preserves_original_interval_enclosure_and_charges() {
        let mut fuel = 16_777_216;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        let mut diagnostics = Diagnostics::default();
        let mut integrator = Integrator::new(&kernel, 1.0, &mut diagnostics, &mut fuel).unwrap();
        let map = MappedBoundary::Circle {
            h: 4.0,
            a: 0.25,
            b: 1.0,
            k: 2.0,
            r: 1.0,
            sign: 1.0,
        };
        let index = integrator.compiled.get(map, &mut fuel).unwrap();
        let domain = Interval::new(0.0, 0.03125).unwrap();
        let compiled = integrator
            .panel(map, index, domain, [4.0, 2.0], &mut fuel)
            .unwrap()
            .unwrap();
        let original = integrator
            .panel(map, None, domain, [4.0, 2.0], &mut fuel)
            .unwrap()
            .unwrap();
        assert!(compiled.lo <= original.hi && original.lo <= compiled.hi);
        assert_eq!(integrator.compiled.metrics.compiled_nodes, 16);
        assert_eq!(integrator.compiled.metrics.interval_circle_nodes, 16);
        assert_eq!(integrator.compiled.metrics.line_nodes, 0);
        assert_eq!(integrator.compiled.metrics.node_work, 16 * 448);
        assert_eq!(integrator.compiled.metrics.panel_work, 128);
        let before = fuel;
        let ordinate = integrator
            .ordinate_point(map, index, 0.5, &mut fuel)
            .unwrap();
        assert_eq!(before - fuel, 224);
        assert!(ordinate.lo <= 2.8 && 2.8 <= ordinate.hi);
        let before = fuel;
        let position = integrator
            .position_point(map, index, 0.5, &mut fuel)
            .unwrap();
        assert_eq!(before - fuel, 336);
        assert!(position.0.lo <= 4.8 && 4.8 <= position.0.hi);
    }
}
