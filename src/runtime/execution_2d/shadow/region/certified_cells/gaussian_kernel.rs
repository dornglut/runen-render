//! Normalized finite Gaussian CDF/density certificates; no paint.
//!
//! G(z) = integral(0,z) exp(-t*t/2) dt is entire. On a radius-one complex
//! disc about m in [0,3], |z| <= 4 and |Im(z)| <= 1. Integrating along the
//! straight segment from 0 to z gives |G(z)| <= 4 exp(1/2) < 8. The last
//! inequality follows from exp(1/2) < sum(2^-n) = 2. Cauchy's coefficient
//! bound therefore gives the degree-12 Taylor remainder <= 8*h^13/(1-h)
//! for |z-m| <= h=3/32. Coefficient construction, remainder and normalization
//! use outward intervals. Compiled f64 coefficients carry a proved uniform
//! evaluation-error bound. No platform exp/erf precision supplies a proof.
//! Density has its own exp-series coefficient/remainder certificate below;
//! it is never obtained by differentiating a uniform CDF error estimate.
//!
//! Sixteen patches cover [0,3]. Odd G and a single enclosed 2*G(3) supply
//! the normalized CDF on [-3,3], with exact saturation outside the cutoff.
//! This certifies only dimensionless kernel mass, not Gaussian geometry,
//! quadrature, physical sample reconstruction or whole-scene feasibility.

pub(super) mod fixed;
mod precomputed;

use super::{AreaQueryError, Interval};

const PATCHES: usize = 16;
const DEGREE: usize = 12;
#[cfg(test)]
const SERIES_TERMS: usize = 41;
const HALF_WIDTH: f64 = 3.0 / 32.0;
#[cfg(test)]
const PATCH_WIDTH: f64 = 3.0 / 16.0;
const MAX_WORK: usize = 16_777_216;
const MAX_BYTES: usize = 64 * 1024 * 1024;
#[cfg(test)]
const UNIT_ROUNDOFF: f64 = f64::from_bits((1023_u64 - 53) << 52);
#[cfg(test)]
const DELTA_ROUNDOFF: f64 = f64::from_bits((1023_u64 - 56) << 52);
#[cfg(test)]
const UNDERFLOW_ALLOWANCE: f64 = f64::from_bits(32);
// A range bound, not a caller quality allowance: the added normalized
// uncertainty is at most 2^-30/Z < 4e-10. Final integration must still
// satisfy its existing independent error and work admission.
const NARROW_INTERVAL_WIDTH: f64 = f64::from_bits((1023_u64 - 30) << 52);

/// Conservative elementary-operation allowances, including interval identity
/// branches, four endpoint products/divisions, validity guards, rounding,
/// min/max reduction and the returned record. Every arithmetic call below
/// debits its allowance before execution; loop/index/control costs are extra.
struct Arithmetic<'a> {
    fuel: &'a mut usize,
}

impl Arithmetic<'_> {
    fn charge(&mut self, amount: usize) -> Result<(), AreaQueryError> {
        if *self.fuel < amount {
            *self.fuel = 0;
            return Err(AreaQueryError::Resource);
        }
        *self.fuel -= amount;
        Ok(())
    }

    fn scalar(&mut self, value: f64) -> Result<Interval, AreaQueryError> {
        self.charge(16)?;
        Interval::scalar(value).ok_or(AreaQueryError::Precision)
    }

    fn bounds(&mut self, lo: f64, hi: f64) -> Result<Interval, AreaQueryError> {
        self.charge(16)?;
        Interval::new(lo, hi).ok_or(AreaQueryError::Precision)
    }

    fn add(&mut self, a: Interval, b: Interval) -> Result<Interval, AreaQueryError> {
        self.charge(48)?;
        a.add(b).ok_or(AreaQueryError::Precision)
    }

    fn sub(&mut self, a: Interval, b: Interval) -> Result<Interval, AreaQueryError> {
        self.charge(48)?;
        a.sub(b).ok_or(AreaQueryError::Precision)
    }

    #[cfg(test)]
    fn mul(&mut self, a: Interval, b: Interval) -> Result<Interval, AreaQueryError> {
        self.charge(96)?;
        a.mul(b).ok_or(AreaQueryError::Precision)
    }

    fn div(&mut self, a: Interval, b: Interval) -> Result<Interval, AreaQueryError> {
        self.charge(96)?;
        a.div(b).ok_or(AreaQueryError::Precision)
    }
}

#[derive(Clone, Copy)]
struct Patch {
    center: f64,
    cdf: Polynomial,
    #[cfg(test)]
    density: Polynomial,
    // Retained only for the private, separately evaluated interval test
    // oracle. Actual admission/retained_bytes includes this storage too.
    #[cfg(test)]
    oracle_coefficients: [Interval; DEGREE + 1],
}

#[derive(Clone, Copy)]
struct Polynomial {
    coefficients: [f64; DEGREE + 1],
    error: f64,
}

impl Polynomial {
    /// Compile exact f64 midpoint coefficients once. For true coefficients
    /// a_i and stored c_i, each precomputed radius encloses |a_i-c_i|, so
    /// sum radius_i*h^i bounds their polynomial error throughout [-h,h].
    ///
    /// Twelve correctly rounded FMAs give Horner's forward error
    /// gamma_12*sum |c_i|*h^i, gamma_n=n*u/(1-n*u), u=2^-53. Unrolling
    /// fl(a*b+c)=(a*b+c)*(1+epsilon)+eta, |epsilon|<=u, |eta|<=2^-1075,
    /// also bounds all subnormal errors by 2^-1075 times the geometric
    /// sum of [h*(1+u)]^j, j=0..11, which is <2. The separately supplied
    /// 32*2^-1074 allowance is therefore conservative even at cancellation.
    /// Coefficients are guarded by |c_i|<=16; Horner intermediates stay
    /// below 16/(1-h)<18 (up to the bounded rounding), excluding overflow.
    #[cfg(test)]
    fn compile(
        original: [Interval; DEGREE + 1],
        gamma: Interval,
        fixed_error: Interval,
        arithmetic: &mut Arithmetic<'_>,
    ) -> Result<Self, AreaQueryError> {
        arithmetic.charge(128)?;
        let zero = arithmetic.scalar(0.0)?;
        let h = arithmetic.scalar(HALF_WIDTH)?;
        let mut coefficients = [0.0; DEGREE + 1];
        let mut coefficient_error = zero;
        let mut absolute_sum = zero;
        for index in (0..=DEGREE).rev() {
            arithmetic.charge(64)?;
            let coefficient = original[index];
            let midpoint = coefficient.lo * 0.5 + coefficient.hi * 0.5;
            if !midpoint.is_finite()
                || midpoint < coefficient.lo
                || midpoint > coefficient.hi
                || midpoint.abs() > 16.0
            {
                return Err(AreaQueryError::Precision);
            }
            coefficients[index] = midpoint;
            let scalar_midpoint = arithmetic.scalar(midpoint)?;
            let difference = arithmetic.sub(coefficient, scalar_midpoint)?;
            let radius = arithmetic.scalar(difference.lo.abs().max(difference.hi.abs()))?;
            let product = arithmetic.mul(coefficient_error, h)?;
            coefficient_error = arithmetic.add(radius, product)?;
            let absolute = arithmetic.scalar(midpoint.abs())?;
            let product = arithmetic.mul(absolute_sum, h)?;
            absolute_sum = arithmetic.add(absolute, product)?;
        }
        let rounding_error = arithmetic.mul(gamma, absolute_sum)?;
        let numerical_error = arithmetic.add(coefficient_error, rounding_error)?;
        let error = arithmetic.add(numerical_error, fixed_error)?.hi;
        Ok(Self {
            coefficients,
            error,
        })
    }

    fn evaluate(
        &self,
        delta: f64,
        arithmetic: &mut Arithmetic<'_>,
    ) -> Result<Interval, AreaQueryError> {
        let mut value = self.coefficients[DEGREE];
        for &coefficient in self.coefficients[..DEGREE].iter().rev() {
            // One correctly rounded FMA, finite guard, indexing and loop.
            arithmetic.charge(16)?;
            value = value.mul_add(delta, coefficient);
            if !value.is_finite() {
                return Err(AreaQueryError::Precision);
            }
        }
        arithmetic.charge(32)?;
        arithmetic.bounds(
            (value - self.error).next_down(),
            (value + self.error).next_up(),
        )
    }
}

pub(super) struct GaussianKernel {
    patches: std::borrow::Cow<'static, [Patch]>,
    normalization: Interval,
    #[cfg(test)]
    remainder: f64,
}

#[test]
#[ignore = "explicit regeneration of derived canonical coefficient data"]
fn emit_precomputed_gaussian_kernel() {
    let mut fuel = MAX_WORK;
    let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
    println!("BEGIN_GAUSSIAN_DATA");
    println!(
        "//! Derived canonical finite-kernel coefficients; verified bit-for-bit against the interval proof.\nuse super::{{Patch, Polynomial, Interval}};"
    );
    println!(
        "pub(super) const NORMALIZATION: Interval = Interval {{ lo: f64::from_bits({}), hi: f64::from_bits({}) }};",
        kernel.normalization.lo.to_bits(),
        kernel.normalization.hi.to_bits()
    );
    println!(
        "#[cfg(test)] pub(super) const REMAINDER: f64 = f64::from_bits({});",
        kernel.remainder.to_bits()
    );
    println!("pub(super) const DATA: [Patch; 16] = [");
    for patch in kernel.patches.iter() {
        println!(
            "Patch {{ center: f64::from_bits({}),",
            patch.center.to_bits()
        );
        for (name, polynomial) in [("cdf", patch.cdf), ("density", patch.density)] {
            if name == "density" {
                println!("#[cfg(test)]");
            }
            println!("{name}: Polynomial {{ coefficients: [");
            for c in polynomial.coefficients {
                println!("f64::from_bits({}),", c.to_bits());
            }
            println!(
                "], error: f64::from_bits({}) }},",
                polynomial.error.to_bits()
            );
        }
        println!("#[cfg(test)] oracle_coefficients: [");
        for c in patch.oracle_coefficients {
            println!(
                "Interval {{ lo: f64::from_bits({}), hi: f64::from_bits({}) }},",
                c.lo.to_bits(),
                c.hi.to_bits()
            );
        }
        println!("] }},");
    }
    println!("];\nEND_GAUSSIAN_DATA");
}

/// Both alternating series have decreasing magnitudes after n=4 because
/// z*z/2 <= 9/2. Truncation after n=40 is therefore enclosed by term 41.
/// For G the additional factor is z/(2*41+1), bounded by 3/83. The exact
/// series are evaluated independently of the patch recurrence.
#[cfg(test)]
fn series(
    z: f64,
    exp_tail: f64,
    g_tail: f64,
    arithmetic: &mut Arithmetic<'_>,
) -> Result<(Interval, Interval), AreaQueryError> {
    arithmetic.charge(64)?;
    if !(0.0..=3.0).contains(&z) {
        return Err(AreaQueryError::Precision);
    }
    let zero = arithmetic.scalar(0.0)?;
    let one = arithmetic.scalar(1.0)?;
    let negative_half = arithmetic.scalar(-0.5)?;
    let z = arithmetic.scalar(z)?;
    let squared = arithmetic.mul(z, z)?;
    let q = arithmetic.mul(negative_half, squared)?;
    let mut term = one;
    let mut exponential = one;
    let mut primitive = one;
    for n in 1..SERIES_TERMS {
        arithmetic.charge(16)?;
        let denominator = arithmetic.scalar(n as f64)?;
        let product = arithmetic.mul(term, q)?;
        term = arithmetic.div(product, denominator)?;
        exponential = arithmetic.add(exponential, term)?;
        let odd = arithmetic.scalar((2 * n + 1) as f64)?;
        let contribution = arithmetic.div(term, odd)?;
        primitive = arithmetic.add(primitive, contribution)?;
    }
    let primitive = arithmetic.mul(z, primitive)?;
    let exp_error = arithmetic.bounds(-exp_tail, exp_tail)?;
    let g_error = arithmetic.bounds(-g_tail, g_tail)?;
    let exponential = arithmetic.add(exponential, exp_error)?;
    let primitive = arithmetic.add(primitive, g_error)?;
    // exp(-z*z/2) is positive for every finite real z. This removes only
    // arithmetic excess and does not turn an uncertain sign into a proof.
    let exponential = arithmetic.bounds(exponential.lo.max(zero.lo), exponential.hi)?;
    Ok((exponential, primitive))
}

impl GaussianKernel {
    /// The dimensionless +/-3 kernel is immutable across scenes and sigma.
    /// Its derived data is checked against `new` by canonical validation;
    /// admission pays the bounded borrow/metadata work, not proof synthesis.
    pub(super) fn compiled(fuel: &mut usize, max_bytes: usize) -> Result<Self, AreaQueryError> {
        if *fuel > MAX_WORK || max_bytes > MAX_BYTES {
            return Err(AreaQueryError::Resource);
        }
        let data_bytes = PATCHES * std::mem::size_of::<Patch>();
        let required = data_bytes + std::mem::size_of::<Self>();
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(512)?;
        if required > max_bytes {
            return Err(AreaQueryError::Resource);
        }
        Ok(Self {
            patches: std::borrow::Cow::Borrowed(&precomputed::DATA),
            normalization: precomputed::NORMALIZATION,
            #[cfg(test)]
            remainder: precomputed::REMAINDER,
        })
    }

    /// Both F and the interior density have Lipschitz constant <=1/Z.
    /// F'=exp(-z*z/2)/Z <=1/Z. For density on either side of zero,
    /// |rho'|=|z|*exp(-z*z/2)/Z <=1/Z: its numerator increases up to
    /// |z|=1 and then decreases, with maximum exp(-1/2)<1.
    /// Density callers must separately exclude its cutoff discontinuities.
    /// Only the upper endpoint of this outward interval is used below.
    fn narrow_change(
        &self,
        start: f64,
        end: f64,
        arithmetic: &mut Arithmetic<'_>,
    ) -> Result<Option<Interval>, AreaQueryError> {
        // Subtraction, outward rounding, finite/threshold checks and branch.
        // A wide or overflowing difference retains the endpoint evaluator.
        arithmetic.charge(16)?;
        let width = (end - start).next_up();
        if !width.is_finite() || width > NARROW_INTERVAL_WIDTH {
            return Ok(None);
        }
        let width = arithmetic.bounds(0.0, width)?;
        arithmetic.div(width, self.normalization).map(Some)
    }

    #[cfg(test)]
    pub(super) fn new(fuel: &mut usize, max_bytes: usize) -> Result<Self, AreaQueryError> {
        if *fuel > MAX_WORK || max_bytes > MAX_BYTES {
            return Err(AreaQueryError::Resource);
        }
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(128)?;
        let required = PATCHES
            .checked_mul(std::mem::size_of::<Patch>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>()))
            .ok_or(AreaQueryError::Resource)?;
        if required > max_bytes {
            return Err(AreaQueryError::Resource);
        }

        let one = arithmetic.scalar(1.0)?;
        let three = arithmetic.scalar(3.0)?;
        let eighty_three = arithmetic.scalar(83.0)?;
        let nine_halves = arithmetic.scalar(4.5)?;
        let mut exp_tail = one;
        for n in 1..=SERIES_TERMS {
            arithmetic.charge(8)?;
            let n = arithmetic.scalar(n as f64)?;
            let product = arithmetic.mul(exp_tail, nine_halves)?;
            exp_tail = arithmetic.div(product, n)?;
        }
        let numerator = arithmetic.mul(three, exp_tail)?;
        let g_tail = arithmetic.div(numerator, eighty_three)?.hi;
        let exp_tail = exp_tail.hi;

        let h = arithmetic.scalar(HALF_WIDTH)?;
        let eight = arithmetic.scalar(8.0)?;
        let mut power = one;
        for _ in 0..=DEGREE {
            arithmetic.charge(8)?;
            power = arithmetic.mul(power, h)?;
        }
        let numerator = arithmetic.mul(eight, power)?;
        let denominator = arithmetic.sub(one, h)?;
        let remainder = arithmetic.div(numerator, denominator)?.hi;

        let unit_roundoff = arithmetic.scalar(UNIT_ROUNDOFF)?;
        let degree = arithmetic.scalar(DEGREE as f64)?;
        let numerator = arithmetic.mul(degree, unit_roundoff)?;
        let denominator = arithmetic.sub(one, numerator)?;
        let gamma = arithmetic.div(numerator, denominator)?;
        // The exact difference z-center is in [-h,h]. Correct rounding
        // stays inside those representable endpoints, with absolute error
        // <=2^-56 (even at subnormals). Cauchy gives |P_12'| bounded by
        // 8*sum(i*h^(i-1),i>=1)=8/(1-h)^2 < 10 throughout that interval.
        let delta_error = arithmetic.scalar(DELTA_ROUNDOFF)?;
        let ten = arithmetic.scalar(10.0)?;
        let delta_error = arithmetic.mul(ten, delta_error)?;
        let underflow = arithmetic.scalar(UNDERFLOW_ALLOWANCE)?;
        let fixed_error = arithmetic.add(delta_error, underflow)?;
        let remainder_interval = arithmetic.scalar(remainder)?;
        let fixed_error = arithmetic.add(fixed_error, remainder_interval)?;

        // E(z)=exp(-z*z/2) is independently bounded by exp(1/2)<2 on
        // the same complex discs. Its degree-11 Taylor remainder is thus
        // <=2*h^12/(1-h). This is NOT differentiation of a uniform CDF
        // approximation error. True E coefficients are (i+1)*G_{i+1}.
        let two = arithmetic.scalar(2.0)?;
        let power_twelve = arithmetic.div(power, h)?;
        let numerator = arithmetic.mul(two, power_twelve)?;
        let denominator = arithmetic.sub(one, h)?;
        let density_remainder = arithmetic.div(numerator, denominator)?;
        // |P_11,E'| <=2/(1-h)^2 <3; use four for rounded delta error.
        let four = arithmetic.scalar(4.0)?;
        let delta = arithmetic.scalar(DELTA_ROUNDOFF)?;
        let delta = arithmetic.mul(four, delta)?;
        let density_fixed_error = arithmetic.add(delta, underflow)?;
        let density_fixed_error = arithmetic.add(density_fixed_error, density_remainder)?;

        let (_, g_three) = series(3.0, exp_tail, g_tail, &mut arithmetic)?;
        let normalization = arithmetic.mul(two, g_three)?;
        if normalization.lo <= 0.0 {
            return Err(AreaQueryError::Precision);
        }

        arithmetic.charge(64 + PATCHES * 8)?;
        let mut patches = Vec::new();
        patches
            .try_reserve_exact(PATCHES)
            .map_err(|_| AreaQueryError::Resource)?;
        let allocated = patches
            .capacity()
            .checked_mul(std::mem::size_of::<Patch>())
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>()))
            .ok_or(AreaQueryError::Resource)?;
        if allocated > max_bytes {
            return Err(AreaQueryError::Resource);
        }
        for index in 0..PATCHES {
            arithmetic.charge(128)?;
            // All factors and products are small exact dyadic numbers.
            let center = (2 * index + 1) as f64 * HALF_WIDTH;
            let (mut current, g) = series(center, exp_tail, g_tail, &mut arithmetic)?;
            let zero = arithmetic.scalar(0.0)?;
            let mut previous = zero;
            let mut coefficients = [zero; DEGREE + 1];
            coefficients[0] = g;
            let negative_center = arithmetic.scalar(-center)?;
            for n in 0..DEGREE {
                arithmetic.charge(32)?;
                let denominator = arithmetic.scalar((n + 1) as f64)?;
                coefficients[n + 1] = arithmetic.div(current, denominator)?;
                if n + 1 < DEGREE {
                    let product = arithmetic.mul(negative_center, current)?;
                    let numerator = arithmetic.sub(product, previous)?;
                    let next = arithmetic.div(numerator, denominator)?;
                    previous = current;
                    current = next;
                }
            }
            let cdf = Polynomial::compile(coefficients, gamma, fixed_error, &mut arithmetic)?;
            let mut density_coefficients = [zero; DEGREE + 1];
            for index in 0..DEGREE {
                arithmetic.charge(16)?;
                let derivative = arithmetic.scalar((index + 1) as f64)?;
                density_coefficients[index] =
                    arithmetic.mul(coefficients[index + 1], derivative)?;
            }
            let density = Polynomial::compile(
                density_coefficients,
                gamma,
                density_fixed_error,
                &mut arithmetic,
            )?;
            arithmetic.charge(128)?;
            patches.push(Patch {
                center,
                cdf,
                density,
                #[cfg(test)]
                oracle_coefficients: coefficients,
            });
        }
        Ok(Self {
            patches: std::borrow::Cow::Owned(patches),
            normalization,
            #[cfg(test)]
            remainder,
        })
    }

    pub(super) fn retained_bytes(&self) -> Option<usize> {
        (match &self.patches {
            std::borrow::Cow::Owned(patches) => patches.capacity(),
            std::borrow::Cow::Borrowed(patches) => patches.len(),
        })
        .checked_mul(std::mem::size_of::<Patch>())?
        .checked_add(std::mem::size_of::<Self>())
    }

    fn point(&self, z: f64, arithmetic: &mut Arithmetic<'_>) -> Result<Interval, AreaQueryError> {
        arithmetic.charge(128)?;
        if !z.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        if z <= -3.0 {
            return arithmetic.scalar(0.0);
        }
        if z >= 3.0 {
            return arithmetic.scalar(1.0);
        }
        if z == 0.0 {
            return arithmetic.scalar(0.5);
        }
        let positive = z.abs();
        // Compare exact dyadic patch endpoints. Rounded division followed
        // by a floor/cast could select the wrong patch immediately below
        // a seam, invalidating the independently proved |delta| <= h.
        let index = self
            .patches
            .partition_point(|patch| patch.center + HALF_WIDTH <= positive);
        let patch = self.patches.get(index).ok_or(AreaQueryError::Precision)?;
        arithmetic.charge(32)?;
        let delta = positive - patch.center;
        if !(-HALF_WIDTH..=HALF_WIDTH).contains(&delta) {
            return Err(AreaQueryError::Precision);
        }
        let primitive = patch.cdf.evaluate(delta, arithmetic)?;
        let ratio = arithmetic.div(primitive, self.normalization)?;
        let half = arithmetic.scalar(0.5)?;
        let cdf = if z > 0.0 {
            arithmetic.add(half, ratio)?
        } else {
            arithmetic.sub(half, ratio)?
        };
        arithmetic.bounds(cdf.lo.max(0.0), cdf.hi.min(1.0))
    }

    #[cfg(test)]
    fn density_point(
        &self,
        z: f64,
        arithmetic: &mut Arithmetic<'_>,
    ) -> Result<Interval, AreaQueryError> {
        arithmetic.charge(128)?;
        if !z.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        if z.abs() >= 3.0 {
            return arithmetic.scalar(0.0);
        }
        if z == 0.0 {
            let one = arithmetic.scalar(1.0)?;
            return arithmetic.div(one, self.normalization);
        }
        let positive = z.abs();
        let index = self
            .patches
            .partition_point(|patch| patch.center + HALF_WIDTH <= positive);
        let patch = self.patches.get(index).ok_or(AreaQueryError::Precision)?;
        arithmetic.charge(32)?;
        let delta = positive - patch.center;
        if !(-HALF_WIDTH..=HALF_WIDTH).contains(&delta) {
            return Err(AreaQueryError::Precision);
        }
        let primitive = patch.density.evaluate(delta, arithmetic)?;
        let density = arithmetic.div(primitive, self.normalization)?;
        arithmetic.charge(16)?;
        let maximum = (1.0 / self.normalization.lo).next_up();
        arithmetic.bounds(density.lo.max(0.0), density.hi.min(maximum))
    }

    /// Normalized density on the OPEN finite support (-3,3). This chooses
    /// zero at the two cutoff points, an integration-equivalent point
    /// representative; no continuity at either finite cutoff is claimed.
    /// Interval ranges crossing a cutoff retain both zero and interior
    /// density. Monotonicity in |z| handles intervals crossing zero too.
    #[cfg(test)]
    pub(super) fn density(
        &self,
        z: Interval,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        if *fuel > MAX_WORK {
            return Err(AreaQueryError::Resource);
        }
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(96)?;
        if !z.lo.is_finite() || !z.hi.is_finite() || z.lo > z.hi {
            return Err(AreaQueryError::Precision);
        }
        if z.hi <= -3.0 || z.lo >= 3.0 {
            return arithmetic.scalar(0.0);
        }
        let nearest = if z.lo <= 0.0 && z.hi >= 0.0 {
            0.0
        } else {
            z.lo.abs().min(z.hi.abs())
        };
        let farthest = z.lo.abs().max(z.hi.abs());
        let upper = self.density_point(nearest, &mut arithmetic)?;
        let lower = if farthest >= 3.0 {
            arithmetic.scalar(0.0)?
        } else if nearest == farthest {
            upper
        } else if let Some(change) = self.narrow_change(nearest, farthest, &mut arithmetic)? {
            // Interior density decreases with |z|. One enclosed value at
            // the nearest point and its Lipschitz fall enclose this range.
            arithmetic.sub(upper, change)?
        } else {
            self.density_point(farthest, &mut arithmetic)?
        };
        arithmetic.bounds(lower.lo.max(0.0), upper.hi)
    }

    /// Independent arithmetic route retained only as a private test oracle.
    /// It evaluates the ORIGINAL interval coefficients using interval Horner;
    /// primary conformance still uses the independent Decimal references.
    #[cfg(test)]
    fn point_interval(&self, z: f64, fuel: &mut usize) -> Result<Interval, AreaQueryError> {
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(128)?;
        if !z.is_finite() || !(-3.0..=3.0).contains(&z) {
            return Err(AreaQueryError::Precision);
        }
        if z == 0.0 || z.abs() == 3.0 {
            return arithmetic.scalar(if z == 0.0 {
                0.5
            } else if z > 0.0 {
                1.0
            } else {
                0.0
            });
        }
        let positive = z.abs();
        let index = self
            .patches
            .partition_point(|patch| patch.center + HALF_WIDTH <= positive);
        let patch = self.patches.get(index).ok_or(AreaQueryError::Precision)?;
        let positive = arithmetic.scalar(positive)?;
        let center = arithmetic.scalar(patch.center)?;
        let delta = arithmetic.sub(positive, center)?;
        let delta = arithmetic.bounds(delta.lo.max(-HALF_WIDTH), delta.hi.min(HALF_WIDTH))?;
        let mut polynomial = patch.oracle_coefficients[DEGREE];
        for &coefficient in patch.oracle_coefficients[..DEGREE].iter().rev() {
            arithmetic.charge(16)?;
            let product = arithmetic.mul(polynomial, delta)?;
            polynomial = arithmetic.add(coefficient, product)?;
        }
        let error = arithmetic.bounds(-self.remainder, self.remainder)?;
        let primitive = arithmetic.add(polynomial, error)?;
        let ratio = arithmetic.div(primitive, self.normalization)?;
        let half = arithmetic.scalar(0.5)?;
        let cdf = if z > 0.0 {
            arithmetic.add(half, ratio)?
        } else {
            arithmetic.sub(half, ratio)?
        };
        arithmetic.bounds(cdf.lo.max(0.0), cdf.hi.min(1.0))
    }

    /// Monotonicity and a global Lipschitz bound need only one endpoint on
    /// narrow intervals, including patch seams and finite cutoffs. Broad
    /// intervals retain both independently enclosed endpoint evaluations.
    pub(super) fn cdf(&self, z: Interval, fuel: &mut usize) -> Result<Interval, AreaQueryError> {
        if *fuel > MAX_WORK {
            return Err(AreaQueryError::Resource);
        }
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(64)?;
        if !z.lo.is_finite() || !z.hi.is_finite() || z.lo > z.hi {
            return Err(AreaQueryError::Precision);
        }
        let lower = self.point(z.lo, &mut arithmetic)?;
        let upper = if z.lo == z.hi || z.hi <= -3.0 || z.lo >= 3.0 {
            // Preserve exact saturation for entire exterior intervals.
            lower
        } else if let Some(change) = self.narrow_change(z.lo, z.hi, &mut arithmetic)? {
            arithmetic.add(lower, change)?
        } else {
            self.point(z.hi, &mut arithmetic)?
        };
        arithmetic.bounds(lower.lo, upper.hi.min(1.0))
    }

    /// Enclose max(F(right)-F(left),0) for every endpoint in the supplied
    /// intervals. This includes the usual ordered interval's kernel mass.
    pub(super) fn mass(
        &self,
        left: Interval,
        right: Interval,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        let lower = self.cdf(left, fuel)?;
        let upper = self.cdf(right, fuel)?;
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(32)?;
        if upper.hi <= lower.lo {
            return arithmetic.scalar(0.0);
        }
        if left.lo == left.hi && right.lo == right.hi && left.lo == right.lo {
            return arithmetic.scalar(0.0);
        }
        let difference = arithmetic.sub(upper, lower)?;
        arithmetic.bounds(difference.lo.max(0.0), difference.hi.clamp(0.0, 1.0))
    }
}

#[test]
fn compiled_kernel_is_bit_identical_to_complete_interval_construction() {
    let mut proof_fuel = MAX_WORK;
    let proof = GaussianKernel::new(&mut proof_fuel, MAX_BYTES).unwrap();
    let mut fuel = MAX_WORK;
    let compiled = GaussianKernel::compiled(&mut fuel, MAX_BYTES).unwrap();
    assert_eq!(
        [
            proof.normalization.lo.to_bits(),
            proof.normalization.hi.to_bits()
        ],
        [
            compiled.normalization.lo.to_bits(),
            compiled.normalization.hi.to_bits()
        ]
    );
    assert_eq!(proof.remainder.to_bits(), compiled.remainder.to_bits());
    for (a, b) in proof.patches.iter().zip(compiled.patches.iter()) {
        assert_eq!(a.center.to_bits(), b.center.to_bits());
        for (a, b) in [(&a.cdf, &b.cdf), (&a.density, &b.density)] {
            assert_eq!(
                a.coefficients.map(f64::to_bits),
                b.coefficients.map(f64::to_bits)
            );
            assert_eq!(a.error.to_bits(), b.error.to_bits());
        }
        assert_eq!(
            a.oracle_coefficients
                .map(|v| [v.lo.to_bits(), v.hi.to_bits()]),
            b.oracle_coefficients
                .map(|v| [v.lo.to_bits(), v.hi.to_bits()])
        );
    }
    assert!(matches!(
        GaussianKernel::compiled(&mut 0, MAX_BYTES),
        Err(AreaQueryError::Resource)
    ));
    assert!(matches!(
        GaussianKernel::compiled(&mut fuel, 0),
        Err(AreaQueryError::Resource)
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(x: f64) -> Interval {
        Interval::scalar(x).unwrap()
    }

    // Independent 110-decimal-digit reference: Python Decimal, 201 terms
    // z*sum((-z*z/2)^n/(n!*(2*n+1))), divided by twice that series at 3.
    // Its omitted term is < 10^-240 on [0,3]; decimal roundoff < 10^-100.
    // This generator uses neither patch recurrence nor the kernel's code.
    // Values alternate exact patch seams and patch centers, including tails.
    pub(super) const REFERENCE: [f64; 33] = [
        0.5,
        0.537_447_224_424_407_7,
        0.574_567_003_859_365_8,
        0.611_040_457_983_388_9,
        0.646_565_463_534_183_3,
        0.680_864_124_734_826_5,
        0.713_689_215_486_366,
        0.744_829_347_961_864_1,
        0.774_112_696_000_684_1,
        0.801_409_182_853_926_7,
        0.826_631_125_546_345_9,
        0.849_732_406_724_866_7,
        0.870_706_314_311_194_2,
        0.889_582_245_528_924_5,
        0.906_421_512_176_957,
        0.921_312_507_026_467,
        0.934_365_496_989_931_4,
        0.945_707_298_581_831_9,
        0.955_476_067_532_291_8,
        0.963_816_400_329_163,
        0.970_874_904_448_055_5,
        0.976_796_349_643_511_1,
        0.981_720_468_245_383_3,
        0.985_779_430_791_835_7,
        0.989_095_986_764_578_7,
        0.991_782_230_192_669_3,
        0.993_938_927_260_428,
        0.995_655_327_933_840_4,
        0.997_009_375_592_615_8,
        0.998_068_226_883_266_1,
        0.998_888_997_375_922_7,
        0.999_519_655_861_342,
        1.0,
    ];

    // Additional independent Decimal references at random exact dyadics,
    // z=n/2^20. Numerators use the reproducible LCG seed 0x5eed1234,
    // seed=1664525*seed+1013904223 mod 2^32; n=seed mod (3*2^20).
    // The same 110-digit direct Maclaurin integral as above generated the
    // references, without compiled coefficients, FMA or interval Horner.
    const DYADIC_REFERENCE: [(u32, f64); 32] = [
        (2_988_035, 0.999_158_999_562_637_3),
        (1_826_182, 0.960_452_397_141_754_2),
        (1_218_349, 0.878_385_174_620_426_2),
        (1_150_632, 0.864_735_735_185_679_7),
        (1_188_327, 0.872_456_570_408_032_3),
        (767_514, 0.768_628_258_638_239_9),
        (2_068_145, 0.977_002_221_300_300_3),
        (2_930_780, 0.998_751_619_060_087_8),
        (2_422_795, 0.990_896_662_634_625_3),
        (2_551_278, 0.993_847_964_072_145_2),
        (1_301_621, 0.893_819_918_652_577_9),
        (2_409_296, 0.990_534_444_442_71),
        (2_014_831, 0.973_944_735_455_572),
        (383_234, 0.643_009_688_093_434_1),
        (1_295_481, 0.892_731_912_718_489_5),
        (598_916, 0.716_642_959_294_481_7),
        (1_071_379, 0.847_487_729_763_766_5),
        (1_837_910, 0.961_424_796_288_380_5),
        (379_581, 0.641_705_308_776_946_9),
        (2_526_456, 0.993_342_892_679_086_6),
        (453_623, 0.667_803_466_450_06),
        (2_906_346, 0.998_557_848_862_365_2),
        (439_105, 0.662_744_762_347_707_3),
        (717_740, 0.753_852_513_437_225_8),
        (3_136_283, 0.999_959_427_088_768_4),
        (2_938_302, 0.998_808_779_553_987_6),
        (885_253, 0.801_547_314_842_264_7),
        (625_568, 0.725_216_842_067_627),
        (838_271, 0.788_761_216_285_630_6),
        (2_807_250, 0.997_631_513_038_981_1),
        (2_063_113, 0.976_726_447_520_316_1),
        (2_017_492, 0.974_104_593_160_801_8),
    ];

    // Independent 110-digit direct exp series divided by the same independent
    // 201-term integral at 3; values at z=i*3/32. The last is the INTERIOR
    // limit at 3, whereas the chosen cutoff-point representative is zero.
    pub(super) const DENSITY_REFERENCE: [f64; 33] = [
        0.4000222589212848,
        0.3982682055477834,
        0.39305205865318005,
        0.3845098520417716,
        0.37286174453981785,
        0.35840259142492853,
        0.34148954645298896,
        0.32252742184987904,
        0.3019526427108863,
        0.2802166801608124,
        0.2577698333279808,
        0.2350461579376663,
        0.2124502180379972,
        0.19034617948245833,
        0.16904958391927616,
        0.14882195545464103,
        0.12986821335705573,
        0.112336705622607,
        0.09632154944511238,
        0.08186687166171672,
        0.06897248747685518,
        0.057600538211793555,
        0.04768262462288239,
        0.039127015488357605,
        0.03182557439613309,
        0.025660123230532947,
        0.02050804130195359,
        0.016246977827170897,
        0.012758627288557685,
        0.009931578335631606,
        0.007663295152831618,
        0.0058613248497615695,
        0.00444384588957642,
    ];

    #[test]
    fn narrow_kernel_ranges_enclose_independent_decimal_endpoints() {
        let mut fuel = MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        // Independent 120-digit Decimal exp and 201-term G series at the
        // EXACT binary64 endpoints. Neither reference uses patch recurrence
        // nor a Lipschitz enlargement. Include both signs, a patch seam,
        // zero, both discontinuous density cutoffs and the broad fallback.
        for (lo, hi, cdf_reference, density_reference) in [
            (
                -1.3125000002328306,
                -1.3124999997671694,
                [0.09357848778368306, 0.09357848786240291],
                [0.16904958386761626, 0.16904958397093606],
            ),
            (
                0.18749999976716936,
                0.18750000023283064,
                [0.5745670037678513, 0.5745670039508803],
                [0.3930520586360211, 0.39305205867033904],
            ),
            (
                0.0,
                2.3283064365386963e-10,
                [0.5, 0.5000000000931374],
                [0.4000222589212848, 0.4000222589212848],
            ),
            (
                -1.1641532182693481e-10,
                1.1641532182693481e-10,
                [0.4999999999534313, 0.5000000000465687],
                [0.4000222589212848, 0.4000222589212848],
            ),
            (
                2.9999999997671694,
                3.0000000002328306,
                [0.9999999999989654, 1.0],
                [0.0, 0.0044438458926804114],
            ),
            (
                -3.0000000002328306,
                -2.9999999997671694,
                [0.0, 1.034663499130033e-12],
                [0.0, 0.0044438458926804114],
            ),
            (
                -2.8125,
                -2.8124999998835847,
                [0.0011110026240773489, 0.0011110026249694738],
                [0.007663295152831618, 0.007663295155340719],
            ),
            (
                0.75,
                0.7500000001164153,
                [0.7741126960006841, 0.774112696035836],
                [0.3019526426845223, 0.3019526427108863],
            ),
            (
                0.75,
                0.7500000018626451,
                [0.7741126960006841, 0.7741126965631147],
                [0.3019526422890633, 0.3019526427108863],
            ),
        ] {
            let query = Interval::new(lo, hi).unwrap();
            for (range, reference) in [
                (kernel.cdf(query, &mut fuel).unwrap(), cdf_reference),
                (kernel.density(query, &mut fuel).unwrap(), density_reference),
            ] {
                assert!(
                    range.lo <= reference[0] && reference[1] <= range.hi,
                    "[{lo},{hi}]: {range:?} excludes independent {reference:?}"
                );
            }
        }
    }

    #[test]
    fn narrow_kernel_ranges_cover_binary_edges_and_pay_actual_query_work() {
        let mut fuel = MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        let mut maximum_narrow_cdf = 0;
        let mut maximum_narrow_density = 0;
        for index in 0..=PATCHES {
            let seam = index as f64 * PATCH_WIDTH;
            for center in [seam, -seam, f64::from_bits(1), -f64::MIN_POSITIVE] {
                let query = Interval::new(center.next_down(), center.next_up()).unwrap();
                let before = fuel;
                let cdf = kernel.cdf(query, &mut fuel).unwrap();
                maximum_narrow_cdf = maximum_narrow_cdf.max(before - fuel);
                // The separately evaluated endpoint certificates can have
                // different excess rounding. Both enclose the real endpoint;
                // their enclosures must at least overlap, even at a seam.
                let lower = kernel
                    .point(query.lo, &mut Arithmetic { fuel: &mut fuel })
                    .unwrap();
                let upper = kernel
                    .point(query.hi, &mut Arithmetic { fuel: &mut fuel })
                    .unwrap();
                assert!(cdf.lo <= lower.hi && lower.lo <= cdf.hi);
                assert!(cdf.lo <= upper.hi && upper.lo <= cdf.hi);
                assert!(cdf.hi - cdf.lo < 4.0e-10);
                let before = fuel;
                let density = kernel.density(query, &mut fuel).unwrap();
                maximum_narrow_density = maximum_narrow_density.max(before - fuel);
                if query.lo <= -3.0 || query.hi >= 3.0 {
                    assert_eq!(density.lo, 0.0);
                    assert!(density.hi >= DENSITY_REFERENCE[32]);
                } else {
                    assert!(density.hi - density.lo < 4.0e-10);
                }
            }
        }
        for (query, cdf_work, density_work) in [
            (scalar(0.75), 656, 640),
            (
                Interval::new(0.75, 0.75 + 2.0_f64.powi(-33)).unwrap(),
                832,
                816,
            ),
            (
                Interval::new(0.75, (0.75 + NARROW_INTERVAL_WIDTH).next_down()).unwrap(),
                832,
                816,
            ),
            (
                Interval::new(0.75, 0.75 + NARROW_INTERVAL_WIDTH).unwrap(),
                1248,
                1184,
            ),
            (
                Interval::new(0.75, 0.75 + 2.0_f64.powi(-29)).unwrap(),
                1248,
                1184,
            ),
        ] {
            let before = fuel;
            let cdf = kernel.cdf(query, &mut fuel).unwrap();
            assert_eq!(before - fuel, cdf_work);
            let before = fuel;
            let density = kernel.density(query, &mut fuel).unwrap();
            assert_eq!(before - fuel, density_work);
            if cdf_work == 832 {
                assert!(cdf.hi - cdf.lo < 4.0e-10);
                assert!(density.hi - density.lo < 4.0e-10);
            }
        }
        assert_eq!(maximum_narrow_cdf, 832);
        assert_eq!(maximum_narrow_density, 816);
        let narrow = Interval::new(0.75, 0.75 + 2.0_f64.powi(-33)).unwrap();
        let mut insufficient = 831;
        assert!(matches!(
            kernel.cdf(narrow, &mut insufficient),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(insufficient, 0);
        let mut insufficient = 815;
        assert!(matches!(
            kernel.density(narrow, &mut insufficient),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(insufficient, 0);
        for (lo, expected) in [(-4.0, 0.0), (4.0, 1.0)] {
            let query = Interval::new(lo, lo + 2.0_f64.powi(-33)).unwrap();
            let cdf = kernel.cdf(query, &mut fuel).unwrap();
            assert_eq!([cdf.lo, cdf.hi], [expected, expected]);
            let density = kernel.density(query, &mut fuel).unwrap();
            assert_eq!([density.lo, density.hi], [0.0, 0.0]);
        }
    }

    #[test]
    fn gaussian_kernel_density_uses_independent_exp_remainder_and_reference() {
        let mut fuel = MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        let construction = MAX_WORK - fuel;
        let mut max_work = 0;
        let mut max_width = 0.0_f64;
        for (index, &reference) in DENSITY_REFERENCE.iter().enumerate().take(32) {
            let x = index as f64 * HALF_WIDTH;
            for z in [x, -x] {
                let before = fuel;
                let density = kernel.density(scalar(z), &mut fuel).unwrap();
                max_work = max_work.max(before - fuel);
                max_width = max_width.max(density.hi - density.lo);
                assert!(
                    density.lo <= reference.next_down() && reference.next_up() <= density.hi,
                    "density({z})={density:?}, independent={reference}"
                );
            }
        }
        assert!(max_width < 1.0e-10);
        assert!(max_work < 1024);
        println!(
            "Gaussian density: construction_work={construction}, max_point_work={max_work}, maximum_width={max_width:e}, retained_bytes={}",
            kernel.retained_bytes().unwrap()
        );
    }

    #[test]
    fn gaussian_kernel_density_interval_cutoff_is_discontinuous_and_fail_closed() {
        let mut fuel = MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        for z in [3.0, -3.0, 4.0, -4.0, f64::MAX, -f64::MAX] {
            let density = kernel.density(scalar(z), &mut fuel).unwrap();
            assert_eq!([density.lo, density.hi], [0.0, 0.0]);
        }
        for sign in [-1.0_f64, 1.0] {
            let z = sign * 3.0;
            let range = kernel
                .density(
                    Interval::new(z.next_down(), z.next_up()).unwrap(),
                    &mut fuel,
                )
                .unwrap();
            assert_eq!(range.lo, 0.0);
            assert!(range.hi >= DENSITY_REFERENCE[32]);
            let inside = kernel
                .density(scalar(sign * 3.0_f64.next_down()), &mut fuel)
                .unwrap();
            assert!(
                inside.lo > 0.0
                    && inside.lo <= DENSITY_REFERENCE[32]
                    && inside.hi >= DENSITY_REFERENCE[32]
            );
        }
        let symmetric = kernel
            .density(Interval::new(-0.1875, 0.1875).unwrap(), &mut fuel)
            .unwrap();
        assert!(symmetric.lo <= DENSITY_REFERENCE[2] && symmetric.hi >= DENSITY_REFERENCE[0]);
        let full = kernel
            .density(Interval::new(-4.0, 4.0).unwrap(), &mut fuel)
            .unwrap();
        assert_eq!(full.lo, 0.0);
        assert!(full.hi >= DENSITY_REFERENCE[0]);
        let mut exhausted = 100;
        assert!(matches!(
            kernel.density(scalar(1.0), &mut exhausted),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(exhausted, 0);
        assert!(matches!(
            kernel.density(
                Interval {
                    lo: f64::NAN,
                    hi: 1.0
                },
                &mut fuel
            ),
            Err(AreaQueryError::Precision)
        ));
    }

    #[test]
    fn gaussian_kernel_encloses_independent_seams_centers_and_negative_tails() {
        let mut fuel = MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        let construction = MAX_WORK - fuel;
        let mut maximum_width = 0.0_f64;
        let mut maximum_point_work = 0;
        for (index, &reference) in REFERENCE.iter().enumerate() {
            let x = index as f64 * HALF_WIDTH;
            for (x, reference) in [(x, reference), (-x, 1.0 - reference)] {
                let before = fuel;
                let result = kernel.cdf(scalar(x), &mut fuel).unwrap();
                maximum_point_work = maximum_point_work.max(before - fuel);
                maximum_width = maximum_width.max(result.hi - result.lo);
                if x == 0.0 || x.abs() == 3.0 {
                    assert_eq!([result.lo, result.hi], [reference, reference]);
                } else {
                    assert!(
                        result.lo <= reference.next_down() && reference.next_up() <= result.hi,
                        "z={x}: {result:?} excludes independent reference {reference}"
                    );
                }
            }
        }
        assert!(maximum_width < 1.0e-10, "width={maximum_width}");
        assert!(construction < 1_000_000);
        assert!(maximum_point_work < 1_024);
        assert!(kernel.retained_bytes().unwrap() < 8_192);
        println!(
            "Gaussian kernel: construction_work={construction}, max_point_work={maximum_point_work}, retained_bytes={}, maximum_width={maximum_width:e}",
            kernel.retained_bytes().unwrap()
        );
    }

    #[test]
    fn gaussian_kernel_intervals_cross_every_patch_seam_without_losing_mass() {
        let mut fuel = MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        for index in 1..PATCHES {
            let seam = index as f64 * PATCH_WIDTH;
            let lower = seam.next_down();
            let upper = seam.next_up();
            let around = kernel
                .cdf(Interval::new(lower, upper).unwrap(), &mut fuel)
                .unwrap();
            let left = kernel.cdf(scalar(lower), &mut fuel).unwrap();
            let right = kernel.cdf(scalar(upper), &mut fuel).unwrap();
            // A Lipschitz range need not retain all excess rounding from
            // the other patch's independent endpoint evaluation. The real
            // endpoint belongs to both certificates, so they must overlap.
            assert!(around.lo <= left.hi && left.lo <= around.hi);
            assert!(around.lo <= right.hi && right.lo <= around.hi);
            let reference = REFERENCE[index * 2];
            assert!(around.lo <= reference && reference <= around.hi);
        }
        let full = kernel
            .cdf(Interval::new(-4.0, 4.0).unwrap(), &mut fuel)
            .unwrap();
        assert_eq!([full.lo, full.hi], [0.0, 1.0]);
        let cross_zero = kernel
            .cdf(Interval::new(-0.1875, 0.1875).unwrap(), &mut fuel)
            .unwrap();
        assert!(cross_zero.lo <= 1.0 - REFERENCE[2]);
        assert!(cross_zero.hi >= REFERENCE[2]);
        for sign in [-1.0_f64, 1.0] {
            let cutoff = sign * 3.0;
            let around_cutoff = kernel
                .cdf(
                    Interval::new(cutoff.next_down(), cutoff.next_up()).unwrap(),
                    &mut fuel,
                )
                .unwrap();
            if sign > 0.0 {
                assert!(around_cutoff.lo < 1.0);
                assert_eq!(around_cutoff.hi, 1.0);
            } else {
                assert_eq!(around_cutoff.lo, 0.0);
                assert!(around_cutoff.hi > 0.0);
            }
        }
    }

    #[test]
    fn gaussian_kernel_fma_encloses_independent_random_dyadics_and_interval_endpoints() {
        let mut fuel = MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        for (numerator, reference) in DYADIC_REFERENCE {
            let x = f64::from(numerator) / 1_048_576.0;
            for (point, reference) in [(x, reference), (-x, 1.0 - reference)] {
                let fast = kernel.cdf(scalar(point), &mut fuel).unwrap();
                let oracle = kernel.point_interval(point, &mut fuel).unwrap();
                for result in [fast, oracle] {
                    assert!(
                        result.lo <= reference.next_down() && reference.next_up() <= result.hi,
                        "z={point}: {result:?} excludes independent Decimal {reference}"
                    );
                    assert!(result.hi - result.lo < 1.0e-10);
                }
            }
            let interval = kernel
                .cdf(Interval::new(-x, x).unwrap(), &mut fuel)
                .unwrap();
            assert!(interval.lo <= 1.0 - reference);
            assert!(reference <= interval.hi);
        }
    }

    #[test]
    fn gaussian_kernel_fma_binary_edge_queries_agree_with_interval_certificates() {
        let mut fuel = MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        let check = |z: f64, fuel: &mut usize| {
            let fast = kernel.cdf(scalar(z), fuel).unwrap();
            assert!(fast.lo >= 0.0 && fast.lo <= fast.hi && fast.hi <= 1.0);
            if (-3.0..=3.0).contains(&z) {
                let original = kernel.point_interval(z, fuel).unwrap();
                assert!(
                    fast.lo <= original.hi && original.lo <= fast.hi,
                    "z={z}: disjoint FMA {fast:?} and original {original:?}"
                );
            } else {
                let expected = if z > 0.0 { 1.0 } else { 0.0 };
                assert_eq!([fast.lo, fast.hi], [expected, expected]);
            }
        };
        for index in 0..=PATCHES {
            let seam = index as f64 * PATCH_WIDTH;
            for adjacent in [seam.next_down(), seam, seam.next_up()] {
                check(adjacent, &mut fuel);
                check(-adjacent, &mut fuel);
            }
        }
        for positive in [
            0.0,
            f64::from_bits(1),
            f64::MIN_POSITIVE.next_down(),
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE.next_up(),
            HALF_WIDTH.next_down(),
            HALF_WIDTH,
            HALF_WIDTH.next_up(),
            f64::MAX.next_down(),
            f64::MAX,
        ] {
            check(positive, &mut fuel);
            check(-positive, &mut fuel);
        }
    }

    #[test]
    fn gaussian_kernel_mass_normalization_and_finite_cutoff_are_exact() {
        let mut fuel = MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        for (left, right, expected) in [
            (-3.0, 3.0, 1.0),
            (-f64::MAX, f64::MAX, 1.0),
            (3.0, 4.0, 0.0),
            (-4.0, -3.0, 0.0),
            (0.75, 0.75, 0.0),
            (2.0, -2.0, 0.0),
        ] {
            let mass = kernel.mass(scalar(left), scalar(right), &mut fuel).unwrap();
            assert_eq!([mass.lo, mass.hi], [expected, expected]);
        }
        let half = kernel.mass(scalar(-3.0), scalar(0.0), &mut fuel).unwrap();
        assert!(half.lo <= 0.5 && half.hi >= 0.5);
        let central = kernel.mass(scalar(-1.5), scalar(1.5), &mut fuel).unwrap();
        let reference = 2.0 * REFERENCE[16] - 1.0;
        assert!(central.lo <= reference && reference <= central.hi);
        let uncertain = kernel
            .mass(
                Interval::new(-1.5, 0.0).unwrap(),
                Interval::new(0.0, 1.5).unwrap(),
                &mut fuel,
            )
            .unwrap();
        assert_eq!(uncertain.lo, 0.0);
        assert!(uncertain.hi >= reference);
    }

    #[test]
    fn gaussian_kernel_rejects_work_memory_and_invalid_queries_without_partial_values() {
        let mut insufficient = 100;
        assert!(matches!(
            GaussianKernel::new(&mut insufficient, MAX_BYTES),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(insufficient, 0);
        let mut fuel = MAX_WORK;
        let mut excessive = MAX_WORK + 1;
        assert!(matches!(
            GaussianKernel::new(&mut excessive, MAX_BYTES),
            Err(AreaQueryError::Resource)
        ));
        assert!(matches!(
            GaussianKernel::new(&mut fuel, 128),
            Err(AreaQueryError::Resource)
        ));
        let kernel = GaussianKernel::new(&mut fuel, MAX_BYTES).unwrap();
        let mut exhausted = 100;
        assert!(matches!(
            kernel.cdf(scalar(1.0), &mut exhausted),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(exhausted, 0);
        for invalid in [
            Interval { lo: 1.0, hi: 0.0 },
            Interval {
                lo: f64::NAN,
                hi: 1.0,
            },
            Interval {
                lo: 0.0,
                hi: f64::INFINITY,
            },
        ] {
            assert!(matches!(
                kernel.cdf(invalid, &mut fuel),
                Err(AreaQueryError::Precision)
            ));
        }
    }
}
