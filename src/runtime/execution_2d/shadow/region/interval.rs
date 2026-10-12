//! Single outward-rounded finite f64 interval primitive for private F3F
//! geometric certificates. Invalid or overflowing arithmetic is Unknown,
//! never a point estimate masquerading as source-support truth.

#[derive(Clone, Copy, Debug)]
pub(super) struct Interval {
    pub(super) lo: f64,
    pub(super) hi: f64,
}

impl Interval {
    fn is_zero(self) -> bool {
        self.lo == 0.0 && self.hi == 0.0
    }

    fn is_one(self) -> bool {
        self.lo == 1.0 && self.hi == 1.0
    }

    fn is_negative_one(self) -> bool {
        self.lo == -1.0 && self.hi == -1.0
    }

    fn negated(self) -> Option<Self> {
        Self::new(-self.hi, -self.lo)
    }

    pub(super) fn new(lo: f64, hi: f64) -> Option<Self> {
        (lo.is_finite() && hi.is_finite() && lo <= hi).then_some(Self { lo, hi })
    }

    pub(super) fn scalar(value: f64) -> Option<Self> {
        Self::new(value, value)
    }

    pub(super) fn add(self, other: Self) -> Option<Self> {
        // True algebraic identities are exact, including finite f64::MAX.
        // Blind next_up(MAX) would reject a valid zero-offset envelope.
        if self.is_zero() {
            return Some(other);
        }
        if other.is_zero() {
            return Some(self);
        }
        Self::new(
            (self.lo + other.lo).next_down(),
            (self.hi + other.hi).next_up(),
        )
    }

    pub(super) fn sub(self, other: Self) -> Option<Self> {
        if other.is_zero() {
            return Some(self);
        }
        Self::new(
            (self.lo - other.hi).next_down(),
            (self.hi - other.lo).next_up(),
        )
    }

    pub(super) fn mul(self, other: Self) -> Option<Self> {
        // These exact cases prevent a false infinity and make zero matrix
        // coefficients truly zero, without masking any inexact product.
        if self.is_zero() || other.is_zero() {
            return Self::scalar(0.0);
        }
        if self.is_one() {
            return Some(other);
        }
        if other.is_one() {
            return Some(self);
        }
        if self.is_negative_one() {
            return other.negated();
        }
        if other.is_negative_one() {
            return self.negated();
        }
        let candidates = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        if !candidates.iter().all(|x| x.is_finite()) {
            return None;
        }
        let lo = candidates.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = candidates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::new(lo.next_down(), hi.next_up())
    }

    pub(super) fn div(self, other: Self) -> Option<Self> {
        if other.is_one() {
            return Some(self);
        }
        if other.is_negative_one() {
            return self.negated();
        }
        if other.lo <= 0.0 && other.hi >= 0.0 {
            return None;
        }
        let candidates = [
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        ];
        if !candidates.iter().all(|x| x.is_finite()) {
            return None;
        }
        let lo = candidates.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = candidates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::new(lo.next_down(), hi.next_up())
    }
}

#[cfg(test)]
mod tests {
    use super::Interval;

    #[test]
    fn exact_scalar_identities_preserve_maximal_finite_bound() {
        let largest = Interval::scalar(f64::MAX).unwrap();
        let zero = Interval::scalar(0.0).unwrap();
        let one = Interval::scalar(1.0).unwrap();
        let negative_one = Interval::scalar(-1.0).unwrap();
        assert_eq!(largest.add(zero).unwrap().hi, f64::MAX);
        assert_eq!(zero.add(largest).unwrap().hi, f64::MAX);
        assert_eq!(largest.sub(zero).unwrap().hi, f64::MAX);
        assert_eq!(largest.mul(one).unwrap().hi, f64::MAX);
        assert_eq!(one.mul(largest).unwrap().hi, f64::MAX);
        assert_eq!(largest.div(one).unwrap().hi, f64::MAX);
        assert_eq!(largest.mul(zero).unwrap().hi, 0.0);
        assert_eq!(negative_one.mul(largest).unwrap().lo, -f64::MAX);
        assert_eq!(largest.div(negative_one).unwrap().lo, -f64::MAX);
        assert!(
            largest.add(one).is_none(),
            "a truly overflowing sum must reject"
        );
    }

    #[test]
    fn nontrivial_arithmetic_keeps_outward_error_enclosure() {
        let third = Interval::scalar(1.0)
            .unwrap()
            .div(Interval::scalar(3.0).unwrap())
            .unwrap();
        assert!(third.lo < 1.0 / 3.0 && third.hi > 1.0 / 3.0);
        let shifted = Interval::scalar(4.0)
            .unwrap()
            .add(Interval::scalar(2.0_f64.powi(-54)).unwrap())
            .unwrap();
        assert!(shifted.lo <= 4.0 && shifted.hi > 4.0);
    }
}
