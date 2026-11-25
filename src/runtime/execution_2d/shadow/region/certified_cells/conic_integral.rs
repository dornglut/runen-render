//! Bounded continuous integrals of one affine circular boundary branch.
//!
//! This is numerical evidence for a source-derived boundary, not source
//! membership or permission to paint. Every returned interval encloses the
//! integral for every admitted coefficient. The consumer must still admit
//! event topology, clipping, final-color error and whole-invocation resources.
//! No unspecified-precision transcendental function supplies a certificate.
use super::Interval;

const ATAN_TERMS: usize = 20;
// A term includes the rational coefficient enclosure, four-endpoint interval
// multiplication and endpoint addition. The allowance also covers loop/sign
// bookkeeping; the separate fixed cost covers powers and the remainder.
const SERIES_WORK: usize = ATAN_TERMS * 96 + 128;
/// The value-only interface has fixed bounded work; its event-query caller
/// must debit this allowance before calling it.
pub(super) const VALUE_WORK: usize = 512;

pub(super) struct Conic {
    pub(super) h: Interval,
    pub(super) a: Interval,
    pub(super) b: Interval,
    pub(super) k: Interval,
    pub(super) r: Interval,
    pub(super) c: Interval,
}

fn charge(fuel: &mut usize, work: usize) -> Option<()> {
    if *fuel < work {
        *fuel = 0;
        return None;
    }
    *fuel -= work;
    Some(())
}

fn scalar(value: f64) -> Option<Interval> {
    Interval::scalar(value)
}

fn valid(value: Interval) -> bool {
    value.lo.is_finite() && value.hi.is_finite() && value.lo <= value.hi
}

/// Rust's f64::sqrt guarantees the rounded infinite-precision IEEE 754
/// squareRoot result. One representable neighbor on each side therefore
/// encloses the real root, including subnormals. Zero and one are exact.
/// Unlike sqrt, f64::asin and f64::atan have unspecified precision.
pub(super) fn sqrt_enclosure(value: Interval) -> Option<Interval> {
    if !valid(value) || value.lo < 0.0 {
        return None;
    }
    if value.lo == value.hi && (value.lo == 0.0 || value.lo == 1.0) {
        return Some(value);
    }
    Interval::new(
        value.lo.sqrt().next_down().max(0.0),
        value.hi.sqrt().next_up(),
    )
}

fn is_zero(value: Interval) -> bool {
    value.lo == 0.0 && value.hi == 0.0
}

/// After PROVING w is in [-1,1], its true 1-w² lies in [0,1]. Intersecting
/// with that known mathematical domain removes only outward-rounding excess.
/// This must never be used to clamp an unqualified conic's negative radicand.
fn unit_circle_height(w: Interval) -> Option<Interval> {
    if !valid(w) || w.lo < -1.0 || w.hi > 1.0 {
        return None;
    }
    if w.lo == w.hi {
        if w.lo == 0.0 {
            return scalar(1.0);
        }
        if w.lo.abs() == 1.0 {
            return scalar(0.0);
        }
    }
    let squared = w.mul(w)?;
    let squared = Interval::new(squared.lo.max(0.0), squared.hi.min(1.0))?;
    let remainder = scalar(1.0)?.sub(squared)?;
    sqrt_enclosure(Interval::new(remainder.lo.max(0.0), remainder.hi.min(1.0))?)
}

struct Numerics<'a> {
    fuel: &'a mut usize,
    half_atan: Option<Interval>,
}

impl Numerics<'_> {
    /// P20(z)=sum_{j=0}^{19} (-1)^j z^(2j+1)/(2j+1).
    /// For |z|<=1/2, the alternating-series remainder has absolute bound
    /// |z|^41/41 <= 2^-41/41. Horner arithmetic AND all rational coefficients
    /// are enclosed separately from this explicit mathematical remainder.
    fn atan_series(&mut self, z: Interval) -> Option<Interval> {
        charge(self.fuel, SERIES_WORK)?;
        if !valid(z) || z.lo < -0.5 || z.hi > 0.5 {
            return None;
        }
        if is_zero(z) {
            return scalar(0.0);
        }
        let squared = z.mul(z)?;
        let mut polynomial = scalar(-1.0)?.div(scalar(39.0)?)?;
        for j in (0..ATAN_TERMS - 1).rev() {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            let denominator = (2 * j + 1) as f64;
            let coefficient = scalar(sign)?.div(scalar(denominator)?)?;
            polynomial = coefficient.add(squared.mul(polynomial)?)?;
        }
        // 2^41 is exactly representable; this is a proved bound, not an
        // empirical epsilon assigned to a platform transcendental result.
        let error = scalar(1.0)?
            .div(scalar(2_199_023_255_552.0)?)?
            .div(scalar(41.0)?)?
            .hi;
        z.mul(polynomial)?.add(Interval::new(-error, error)?)
    }

    fn atan_positive(&mut self, z: Interval) -> Option<Interval> {
        charge(self.fuel, 64)?;
        if !valid(z) || z.lo < 0.0 || z.hi > 1.0 {
            return None;
        }
        if z.hi <= 0.5 {
            return self.atan_series(z);
        }
        // atan(z)=atan(1/2)+atan((z-1/2)/(1+z/2)).
        // For 1/2<z<=1, the reduced argument is in (0,1/3]. No pi
        // constant, atan library call or uncontrolled argument is required.
        let half = match self.half_atan {
            Some(value) => value,
            None => {
                let value = self.atan_series(scalar(0.5)?)?;
                self.half_atan = Some(value);
                value
            }
        };
        let upper_domain = Interval::new(z.lo.max(0.5), z.hi)?;
        let reduced = upper_domain
            .sub(scalar(0.5)?)?
            .div(scalar(1.0)?.add(scalar(0.5)?.mul(upper_domain)?)?)?;
        let upper = half.add(self.atan_series(reduced)?)?;
        if z.lo >= 0.5 {
            return Some(upper);
        }
        // If uncertainty straddles the reduction boundary, evaluate both
        // admitted branches. Interval Horner evaluation encloses the entire
        // polynomial without assuming its coefficients have one sign.
        let lower = self.atan_series(Interval::new(z.lo, 0.5)?)?;
        Interval::new(lower.lo.min(upper.lo), lower.hi.max(upper.hi))
    }

    fn segment_point(&mut self, w: f64) -> Option<Interval> {
        charge(self.fuel, 96)?;
        if !(-1.0..=1.0).contains(&w) {
            return None;
        }
        if w == 0.0 {
            return scalar(0.0);
        }
        let positive = scalar(w.abs())?;
        let height = unit_circle_height(positive)?;
        let denominator = scalar(1.0)?.add(height)?;
        let denominator = Interval::new(denominator.lo.max(1.0), denominator.hi)?;
        let z = positive.div(denominator)?;
        // w is already certified in [0,1] and height is nonnegative, so
        // the exact half-angle quotient is necessarily in [0,1].
        let z = Interval::new(z.lo.max(0.0), z.hi.min(1.0))?;
        // asin(w)=2 atan(w/(1+sqrt(1-w²))). This half-angle formula has
        // a certified bounded argument, even at the vertical tangent w=1.
        let angle = self.atan_positive(z)?;
        let segment = scalar(0.5)?.mul(positive.mul(height)?)?.add(angle)?;
        if w < 0.0 {
            segment.mul(scalar(-1.0)?)
        } else {
            Some(segment)
        }
    }

    /// S(w)=1/2[w sqrt(1-w²)+asin(w)], with S'(w)=sqrt(1-w²)>=0.
    fn segment(&mut self, w: Interval) -> Option<Interval> {
        if !valid(w) || w.lo < -1.0 || w.hi > 1.0 {
            return None;
        }
        let lower = self.segment_point(w.lo)?;
        let upper = if w.lo == w.hi {
            lower
        } else {
            self.segment_point(w.hi)?
        };
        Interval::new(lower.lo, upper.hi)
    }
}

impl Conic {
    fn admitted(&self) -> bool {
        [self.h, self.a, self.b, self.k, self.r, self.c]
            .into_iter()
            .all(valid)
            && self.r.lo > 0.0
            && self.c.lo > 0.0
    }

    fn normalized(&self, u: Interval, root_c: Interval) -> Option<Interval> {
        let numerator = root_c.mul(u)?;
        let w = numerator.div(self.r)?;
        (w.lo >= -1.0 && w.hi <= 1.0).then_some(w)
    }

    pub(super) fn value(&self, y: Interval) -> Option<Interval> {
        if !self.admitted() || !valid(y) {
            return None;
        }
        let u = y.sub(self.k)?;
        let line = self.h.add(self.a.mul(u)?)?;
        if is_zero(self.b) {
            return Some(line);
        }
        let w = self.normalized(u, sqrt_enclosure(self.c)?)?;
        line.add(self.b.mul(self.r)?.mul(unit_circle_height(w)?)?)
    }

    pub(super) fn integral(&self, y0: f64, y1: f64, fuel: &mut usize) -> Option<Interval> {
        // Fixed storage only. This allowance covers admission, normalization,
        // sqrt, linear integration, final interval products and orientation;
        // every actual bounded-series invocation is charged separately.
        charge(fuel, 768)?;
        if !self.admitted() || !y0.is_finite() || !y1.is_finite() || y0 == y1 {
            return None;
        }
        let (lo, hi, orientation) = if y0 < y1 {
            (y0, y1, 1.0)
        } else {
            (y1, y0, -1.0)
        };
        let u0 = scalar(lo)?.sub(self.k)?;
        let u1 = scalar(hi)?.sub(self.k)?;
        // k cancels exactly from u1-u0. Form the difference directly from
        // the original y values rather than subtracting two translated
        // antiderivatives, which would introduce needless cancellation.
        let delta = scalar(hi)?.sub(scalar(lo)?)?;
        let linear = delta.mul(self.h.add(scalar(0.5)?.mul(self.a)?.mul(u0.add(u1)?)?)?)?;
        if is_zero(self.b) {
            return linear.mul(scalar(orientation)?);
        }
        let root_c = sqrt_enclosure(self.c)?;
        let w0 = self.normalized(u0, root_c)?;
        let w1 = self.normalized(u1, root_c)?;
        let mut numerics = Numerics {
            fuel,
            half_atan: None,
        };
        let segment_delta = numerics.segment(w1)?.sub(numerics.segment(w0)?)?;
        let scale = self.b.mul(self.r)?.mul(self.r)?.div(root_c)?;
        linear
            .add(scale.mul(segment_delta)?)?
            .mul(scalar(orientation)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point(value: f64) -> Interval {
        scalar(value).unwrap()
    }

    fn circle() -> Conic {
        Conic {
            h: point(0.0),
            a: point(0.0),
            b: point(1.0),
            k: point(0.0),
            r: point(1.0),
            c: point(1.0),
        }
    }

    fn encloses(interval: Interval, reference: f64) {
        assert!(
            interval.lo <= reference && reference <= interval.hi,
            "[{:.17},{:.17}] missed {:.17}",
            interval.lo,
            interval.hi,
            reference
        );
    }

    #[test]
    fn circle_segments_enclose_independent_geometric_constants() {
        let mut fuel = 100_000;
        // Independent circular-segment identities: pi/4 and sqrt(3)/8+pi/12.
        let quarter = circle().integral(0.0, 1.0, &mut fuel).unwrap();
        encloses(quarter, std::f64::consts::FRAC_PI_4);
        assert!(quarter.hi - quarter.lo < 1.0e-12);
        let partial = circle().integral(0.0, 0.5, &mut fuel).unwrap();
        encloses(partial, 0.478_305_738_745_259_1);
        assert!(partial.hi - partial.lo < 1.0e-12);
        assert!(fuel < 100_000 && fuel > 0);
    }

    #[test]
    fn sheared_conic_integral_matches_independent_whole_pixel_arc() {
        let boundary = Conic {
            h: point(35.5),
            a: point(0.25),
            ..circle()
        };
        let mut fuel = 100_000;
        // Integral of the unclipped boundary: 35.5+1/8+pi/4.
        let integral = boundary.integral(0.0, 1.0, &mut fuel).unwrap();
        encloses(integral, 36.410_398_163_397_45);
        let value = boundary.value(point(0.5)).unwrap();
        encloses(value, 36.491_025_403_784_44);
    }

    #[test]
    fn affine_ellipse_scales_and_negative_branch_preserve_integral() {
        let mut fuel = 100_000;
        let scaled = Conic {
            h: point(3.0),
            a: point(-0.25),
            b: point(-2.0),
            k: point(-1.0),
            r: point(1.0),
            c: point(4.0),
        };
        // Substitution w=2(y+1): [-1.25,-.75] maps to [-.5,.5].
        // Linear contribution=1.5; arc contribution=-2*I(0,.5).
        let integral = scaled.integral(-1.25, -0.75, &mut fuel).unwrap();
        encloses(integral, 0.543_388_522_509_481_8);
        assert!(integral.hi - integral.lo < 1.0e-11);
        let reversed = scaled.integral(-0.75, -1.25, &mut fuel).unwrap();
        encloses(reversed, -0.543_388_522_509_481_8);
    }

    #[test]
    fn rotated_ellipse_branch_has_independent_binomial_integral_oracle() {
        // A radius-one circle scaled (2,1) then rotated 45 degrees has
        // Qxx=Qyy=5/8, Qxy=-3/8. Solving its conic gives
        // x=1+(3/5)y+(8/5)sqrt(5/8-y²/4).
        let rotated = Conic {
            h: point(1.0),
            a: point(0.6),
            b: point(1.6),
            k: point(0.0),
            r: sqrt_enclosure(point(0.625)).unwrap(),
            c: point(0.25),
        };
        let mut fuel = 100_000;
        let area = rotated.integral(-0.5, 0.5, &mut fuel).unwrap();
        // Independent 80-digit binomial integration, 80 terms at q<=1/10.
        encloses(area, 2.243_501_108_793_284_4);
        assert!(area.hi - area.lo < 1.0e-11);
    }

    #[test]
    fn sheared_crossing_height_interval_encloses_the_exact_vertical_edge() {
        let boundary = Conic {
            h: point(35.5),
            a: point(0.25),
            ..circle()
        };
        // At y=8/17, (1/4)y+sqrt(1-y²)=1 exactly. The exact event
        // height is rational; bracket its division rather than inventing
        // an epsilon around a point root or truncating its significant bits.
        let crossing = point(8.0).div(point(17.0)).unwrap();
        let value = boundary.value(crossing).unwrap();
        encloses(value, 36.5);
        assert!(value.hi - value.lo < 1.0e-12);
    }

    #[test]
    fn tangent_endpoints_and_negative_y_are_enclosed_without_asin() {
        let mut fuel = 100_000;
        encloses(circle().value(point(1.0)).unwrap(), 0.0);
        encloses(circle().value(point(-1.0)).unwrap(), 0.0);
        let whole = circle().integral(-1.0, 1.0, &mut fuel).unwrap();
        encloses(whole, std::f64::consts::FRAC_PI_2);
        let near = circle()
            .integral(1.0 - 1.0 / 1_048_576.0, 1.0, &mut fuel)
            .unwrap();
        // An independent geometric bound: height <=sqrt(2*dy), therefore
        // 0 <= the integral <=dy*sqrt(2*dy). No midpoint authority.
        assert!(near.lo <= 1.0e-9 && near.hi > 0.0);
        assert!(near.hi < 1.4e-9);
        assert!(near.hi - near.lo < 1.0e-12);
    }

    #[test]
    fn coefficient_intervals_enclose_an_independent_rectangle_of_parameters() {
        let conic = Conic {
            h: Interval::new(2.0, 2.01).unwrap(),
            a: Interval::new(-0.25, 0.25).unwrap(),
            b: Interval::new(-2.0, -1.0).unwrap(),
            k: point(0.0),
            r: Interval::new(0.99, 1.01).unwrap(),
            c: Interval::new(0.99, 1.01).unwrap(),
        };
        let mut fuel = 100_000;
        let interval = conic.integral(-0.25, 0.25, &mut fuel).unwrap();
        // Independently tabulated extrema using the positive-coefficient
        // binomial series for sqrt(1-q), integrated term by term at 80
        // decimal digits (60 terms, q<.065; residual <10^-70).
        encloses(interval, 0.000_305_367_939_639_831_5);
        encloses(interval, 0.515_366_105_639_616_5);
    }

    #[test]
    fn line_and_unrepresentable_or_unqualified_domain_fail_truthfully() {
        let line = Conic {
            h: point(2.0),
            a: point(0.5),
            b: point(0.0),
            ..circle()
        };
        let mut fuel = 10_000;
        encloses(line.integral(10.0, 12.0, &mut fuel).unwrap(), 15.0);
        assert!(circle().value(point(1.001)).is_none());
        assert!(circle().integral(0.0, 1.001, &mut fuel).is_none());
        // A nominal quarter-unit site has collapsed before any numerical
        // integration; the helper must not manufacture a positive area.
        assert!(
            circle()
                .integral(1.0e300, 1.0e300 + 0.25, &mut fuel)
                .is_none()
        );
        let huge = Conic {
            h: point(f64::MAX),
            b: point(0.0),
            ..circle()
        };
        assert!(huge.integral(0.0, 2.0, &mut fuel).is_none());
    }

    #[test]
    fn numerical_work_is_admitted_before_a_partial_interval_can_escape() {
        for allowance in [0, 1, 767, 768, 1_000, 2_000] {
            let mut fuel = allowance;
            assert!(circle().integral(0.0, 1.0, &mut fuel).is_none());
            assert_eq!(fuel, 0);
        }
        let mut fuel = 100_000;
        assert!(circle().integral(0.0, 1.0, &mut fuel).is_some());
        assert!(fuel < 100_000);
    }

    #[test]
    fn sqrt_enclosure_covers_subnormal_inputs_and_rejects_negative_domain() {
        let tiny = f64::from_bits(1);
        let root = sqrt_enclosure(point(tiny)).unwrap();
        assert!(root.lo > 0.0 && root.hi > root.lo);
        assert!(sqrt_enclosure(Interval::new(-tiny, tiny).unwrap()).is_none());
    }
}
