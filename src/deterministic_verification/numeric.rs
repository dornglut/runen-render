//! Private proof numerics for RR566-EVAL-001.
//!
//! This is not a renderer, a public interval package, or semantic authority. It only provides the
//! bounded conservative arithmetic needed by the maintained deterministic verifier. Every helper
//! fails closed on non-finite/unsupported arithmetic instead of manufacturing fidelity evidence.

use super::super::request::RenderSemanticTolerance;

const CERTIFIED_TAYLOR_X_LIMIT: f64 = 0.786;
const PI_LOWER_BITS: u64 = 0x4009_21fb_5444_2d18;
const PI_UPPER_BITS: u64 = 0x4009_21fb_5444_2d19;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct VerificationInterval {
    lower: f64,
    upper: f64,
}

impl VerificationInterval {
    pub(super) fn singleton(value: f64) -> Option<Self> {
        value.is_finite().then_some(Self {
            lower: value,
            upper: value,
        })
    }

    fn bounds(lower: f64, upper: f64) -> Option<Self> {
        (lower.is_finite() && upper.is_finite() && lower <= upper).then_some(Self { lower, upper })
    }

    pub(super) const fn lower(self) -> f64 {
        self.lower
    }

    pub(super) const fn upper(self) -> f64 {
        self.upper
    }

    pub(super) fn contains(self, value: f64) -> bool {
        value.is_finite() && self.lower <= value && value <= self.upper
    }

    fn is_singleton(self) -> bool {
        self.lower == self.upper
    }

    fn abs(self) -> Self {
        if self.lower >= 0.0 {
            self
        } else if self.upper <= 0.0 {
            self.neg()
        } else {
            Self {
                lower: 0.0,
                upper: (-self.lower).max(self.upper),
            }
        }
    }

    pub(super) fn add(self, other: Self) -> Option<Self> {
        let lower = self.lower + other.lower;
        let upper = self.upper + other.upper;
        outward_bounds(lower, upper)
    }

    pub(super) fn sub(self, other: Self) -> Option<Self> {
        let lower = self.lower - other.upper;
        let upper = self.upper - other.lower;
        outward_bounds(lower, upper)
    }

    pub(super) fn neg(self) -> Self {
        Self {
            lower: -self.upper,
            upper: -self.lower,
        }
    }

    pub(super) fn mul(self, other: Self) -> Option<Self> {
        let products = [
            outward_scalar(self.lower * other.lower)?,
            outward_scalar(self.lower * other.upper)?,
            outward_scalar(self.upper * other.lower)?,
            outward_scalar(self.upper * other.upper)?,
        ];
        Self::bounds(
            products
                .iter()
                .map(|interval| interval.lower)
                .fold(f64::INFINITY, f64::min),
            products
                .iter()
                .map(|interval| interval.upper)
                .fold(f64::NEG_INFINITY, f64::max),
        )
    }

    pub(super) fn div(self, other: Self) -> Option<Self> {
        if other.lower <= 0.0 && other.upper >= 0.0 {
            return None;
        }
        let quotients = [
            outward_scalar(self.lower / other.lower)?,
            outward_scalar(self.lower / other.upper)?,
            outward_scalar(self.upper / other.lower)?,
            outward_scalar(self.upper / other.upper)?,
        ];
        Self::bounds(
            quotients
                .iter()
                .map(|interval| interval.lower)
                .fold(f64::INFINITY, f64::min),
            quotients
                .iter()
                .map(|interval| interval.upper)
                .fold(f64::NEG_INFINITY, f64::max),
        )
    }

    pub(super) fn sqrt(self) -> Option<Self> {
        if self.lower < 0.0 {
            return None;
        }
        let lower = self.lower.sqrt();
        let upper = self.upper.sqrt();
        if !lower.is_finite() || !upper.is_finite() {
            return None;
        }
        Self::bounds(
            if lower == 0.0 {
                0.0
            } else {
                ieee_next_down(lower)
            },
            ieee_next_up(upper),
        )
    }
}

/// Prove one observed finite numeric value satisfies the exact request tolerance for every semantic
/// value enclosed by `reference`.
///
/// The proof is intentionally one-sided: returning `false` means mismatch or inconclusive evidence,
/// never permission to widen the request. Relative tolerance uses the smallest allowed error across
/// the entire reference interval, preserving the request law that an exact zero permits zero error.
pub(super) fn numeric_value_satisfies_tolerance(
    observed: f64,
    reference: VerificationInterval,
    tolerance: RenderSemanticTolerance,
) -> bool {
    if !observed.is_finite() {
        return false;
    }
    if reference.is_singleton() && observed == reference.lower {
        return true;
    }
    if tolerance.is_exact() {
        return false;
    }

    let Some(observed_interval) = VerificationInterval::singleton(observed) else {
        return false;
    };
    let Some(error_interval) = observed_interval
        .sub(reference)
        .map(VerificationInterval::abs)
    else {
        return false;
    };

    if let Some(max_error) = tolerance.absolute_max_error() {
        return error_interval.upper <= max_error;
    }

    let Some(max_fraction) = tolerance.relative_max_fraction() else {
        return false;
    };
    let Some(fraction) = VerificationInterval::singleton(max_fraction) else {
        return false;
    };
    let Some(allowed_error) = reference.abs().mul(fraction) else {
        return false;
    };
    error_interval.upper <= allowed_error.lower
}

/// Fixed binary64 enclosure of mathematical pi.
///
/// The lower endpoint is `3.14159265358979311599...`, the upper endpoint is
/// `3.14159265358979356008...`, while mathematical pi is
/// `3.14159265358979323846...`. The verifier therefore never treats the platform `PI` constant as
/// exact semantic pi.
pub(super) fn mathematical_pi_interval() -> VerificationInterval {
    VerificationInterval {
        lower: f64::from_bits(PI_LOWER_BITS),
        upper: f64::from_bits(PI_UPPER_BITS),
    }
}

/// Conservative enclosure of `tan(full_fov / 2)` for the first certified perspective domain.
///
/// No host `tan`, `sin`, or `cos` call participates. `sin`/`cos` are enclosed with finite
/// alternating Taylor sums plus the next-term remainder bound on the bounded non-negative domain,
/// then divided only after cosine is proven strictly positive.
pub(super) fn certified_tan_half_fov(full_fov: f64) -> Option<VerificationInterval> {
    if !full_fov.is_finite() || full_fov <= 0.0 || full_fov > std::f64::consts::FRAC_PI_2 {
        return None;
    }
    let x =
        VerificationInterval::singleton(full_fov)?.mul(VerificationInterval::singleton(0.5)?)?;
    let sine = sin_small_nonnegative(x)?;
    let cosine = cos_small_nonnegative(x)?;
    if cosine.lower <= 0.0 {
        return None;
    }
    sine.div(cosine)
}

fn sin_small_nonnegative(x: VerificationInterval) -> Option<VerificationInterval> {
    if x.lower < 0.0 || x.upper > CERTIFIED_TAYLOR_X_LIMIT {
        return None;
    }
    let x_squared = x.mul(x)?;
    let mut term = x;
    let mut sum = x;
    for order in 1_u32..=6 {
        let denominator = f64::from((2 * order) * (2 * order + 1));
        term = term
            .mul(x_squared)?
            .div(VerificationInterval::singleton(denominator)?)?;
        sum = if order % 2 == 1 {
            sum.sub(term)?
        } else {
            sum.add(term)?
        };
    }
    let remainder = term
        .mul(x_squared)?
        .div(VerificationInterval::singleton(f64::from(14_u32 * 15_u32))?)?;
    sum.add(symmetric_radius(remainder.upper)?)
}

fn cos_small_nonnegative(x: VerificationInterval) -> Option<VerificationInterval> {
    if x.lower < 0.0 || x.upper > CERTIFIED_TAYLOR_X_LIMIT {
        return None;
    }
    let x_squared = x.mul(x)?;
    let mut term = VerificationInterval::singleton(1.0)?;
    let mut sum = term;
    for order in 1_u32..=6 {
        let denominator = f64::from((2 * order - 1) * (2 * order));
        term = term
            .mul(x_squared)?
            .div(VerificationInterval::singleton(denominator)?)?;
        sum = if order % 2 == 1 {
            sum.sub(term)?
        } else {
            sum.add(term)?
        };
    }
    let remainder = term
        .mul(x_squared)?
        .div(VerificationInterval::singleton(f64::from(13_u32 * 14_u32))?)?;
    sum.add(symmetric_radius(remainder.upper)?)
}

fn symmetric_radius(radius: f64) -> Option<VerificationInterval> {
    (radius.is_finite() && radius >= 0.0).then_some(VerificationInterval {
        lower: -radius,
        upper: radius,
    })
}

fn outward_scalar(value: f64) -> Option<VerificationInterval> {
    if !value.is_finite() {
        return None;
    }
    VerificationInterval::bounds(ieee_next_down(value), ieee_next_up(value))
}

fn outward_bounds(lower: f64, upper: f64) -> Option<VerificationInterval> {
    if !lower.is_finite() || !upper.is_finite() {
        return None;
    }
    VerificationInterval::bounds(ieee_next_down(lower), ieee_next_up(upper))
}

fn ieee_next_up(value: f64) -> f64 {
    debug_assert!(value.is_finite());
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits + 1)
    } else {
        f64::from_bits(bits - 1)
    }
}

fn ieee_next_down(value: f64) -> f64 {
    debug_assert!(value.is_finite());
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    if value > 0.0 {
        f64::from_bits(bits - 1)
    } else {
        f64::from_bits(bits + 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ieee_bit_steps_are_directional_across_zero_and_signs() {
        assert!(ieee_next_down(0.0) < 0.0);
        assert!(ieee_next_up(0.0) > 0.0);
        assert!(ieee_next_down(1.0) < 1.0);
        assert!(ieee_next_up(1.0) > 1.0);
        assert!(ieee_next_down(-1.0) < -1.0);
        assert!(ieee_next_up(-1.0) > -1.0);
    }

    #[test]
    fn basic_interval_arithmetic_encloses_rounded_examples_and_fails_closed() {
        let one = VerificationInterval::singleton(1.0).expect("one");
        let three = VerificationInterval::singleton(3.0).expect("three");
        let quotient = one.div(three).expect("nonzero divisor");
        assert!(quotient.contains(1.0 / 3.0));

        let product = quotient.mul(three).expect("finite product");
        assert!(product.contains(1.0));

        let zero = VerificationInterval::singleton(0.0).expect("zero");
        assert_eq!(one.div(zero), None);
        assert_eq!(VerificationInterval::singleton(-1.0).unwrap().sqrt(), None);
    }

    #[test]
    fn correctly_rounded_sqrt_is_expanded_outward() {
        let root = VerificationInterval::singleton(2.0)
            .expect("two")
            .sqrt()
            .expect("positive square root");
        assert!(root.contains(2.0_f64.sqrt()));
        assert!(root.lower() >= 0.0);
    }

    #[test]
    fn exact_tolerance_requires_a_uniquely_certified_value() {
        let exact = RenderSemanticTolerance::exact();
        let singleton = VerificationInterval::singleton(2.0).expect("singleton");
        assert!(numeric_value_satisfies_tolerance(2.0, singleton, exact));
        assert!(!numeric_value_satisfies_tolerance(
            ieee_next_up(2.0),
            singleton,
            exact
        ));

        let uncertain = VerificationInterval::bounds(1.999, 2.001).expect("uncertain interval");
        assert!(!numeric_value_satisfies_tolerance(2.0, uncertain, exact));
    }

    #[test]
    fn absolute_tolerance_covers_every_reference_value() {
        let reference = VerificationInterval::bounds(9.999, 10.001).expect("reference");
        let accepted = RenderSemanticTolerance::absolute(0.002).expect("absolute tolerance");
        let rejected = RenderSemanticTolerance::absolute(0.0005).expect("absolute tolerance");
        assert!(numeric_value_satisfies_tolerance(10.0, reference, accepted));
        assert!(!numeric_value_satisfies_tolerance(
            10.0, reference, rejected
        ));
    }

    #[test]
    fn relative_tolerance_preserves_strict_zero_semantics() {
        let relative = RenderSemanticTolerance::relative(1.0).expect("relative tolerance");
        let zero = VerificationInterval::singleton(0.0).expect("zero");
        assert!(numeric_value_satisfies_tolerance(0.0, zero, relative));
        assert!(!numeric_value_satisfies_tolerance(
            f64::from_bits(1),
            zero,
            relative
        ));
    }

    #[test]
    fn relative_tolerance_uses_the_smallest_allowance_across_uncertainty() {
        let reference = VerificationInterval::bounds(99.999, 100.001).expect("reference");
        let accepted = RenderSemanticTolerance::relative(0.0001).expect("relative tolerance");
        let rejected = RenderSemanticTolerance::relative(0.000001).expect("relative tolerance");
        assert!(numeric_value_satisfies_tolerance(
            100.0, reference, accepted
        ));
        assert!(!numeric_value_satisfies_tolerance(
            100.0, reference, rejected
        ));
        assert!(!numeric_value_satisfies_tolerance(
            f64::NAN,
            reference,
            accepted
        ));
    }

    #[test]
    fn audited_pi_enclosure_is_fixed_and_non_degenerate() {
        let pi = mathematical_pi_interval();
        assert_eq!(pi.lower().to_bits(), PI_LOWER_BITS);
        assert_eq!(pi.upper().to_bits(), PI_UPPER_BITS);
        assert!(pi.lower() < pi.upper());
    }

    #[test]
    fn founding_perspective_projection_is_certifiable_without_host_trigonometry() {
        let tangent = certified_tan_half_fov(std::f64::consts::FRAC_PI_4)
            .expect("founding perspective must lie in certified domain");
        assert!(tangent.contains(0.414_213_562_373_095));
        assert!(tangent.upper() - tangent.lower() < 1.0e-12);
    }

    #[test]
    fn wider_perspective_projection_fails_closed() {
        assert_eq!(
            certified_tan_half_fov(std::f64::consts::FRAC_PI_2 * 1.5),
            None
        );
    }

    #[test]
    fn reference_source_does_not_delegate_to_host_trigonometry() {
        let source = include_str!("numeric.rs");
        for name in ["tan", "sin", "cos"] {
            let forbidden = [".", name, "("].concat();
            assert!(
                !source.contains(&forbidden),
                "private verification numerics must not call host {name}"
            );
        }
    }

    #[test]
    fn negation_preserves_exact_interval_order() {
        let interval = VerificationInterval::bounds(2.0, 3.0).expect("interval");
        assert_eq!(
            interval.neg(),
            VerificationInterval::bounds(-3.0, -2.0).expect("negated interval")
        );
    }
}
