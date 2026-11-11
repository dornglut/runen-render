//! Single outward-rounded finite f64 interval primitive for private F3F
//! geometric certificates. Invalid or overflowing arithmetic is Unknown,
//! never a point estimate masquerading as source-support truth.

/// Error-free TwoSum qualifies scalar event identities only when their real
/// sum is exactly representable. Unqualified identities retain interval form.
pub(in crate::runtime::execution_2d) fn exact_add(a: f64, b: f64) -> Option<f64> {
    let sum = a + b;
    let virtual_b = sum - a;
    let residual = (a - (sum - virtual_b)) + (b - virtual_b);
    (sum.is_finite() && residual == 0.0).then_some(sum)
}

/// The exact product residual has at most 106 significant bits. Restricting
/// the product exponent to +/-900 keeps a nonzero residual representable in
/// the correctly rounded FMA; a zero residual therefore proves exactness.
pub(in crate::runtime::execution_2d) fn exact_mul(a: f64, b: f64) -> Option<f64> {
    let product = a * b;
    if a == 0.0 || b == 0.0 || a.abs() == 1.0 || b.abs() == 1.0 {
        return product.is_finite().then_some(product);
    }
    (product.is_finite()
        && product.abs() >= f64::from_bits((1023_u64 - 900) << 52)
        && product.abs() <= f64::from_bits((1023_u64 + 900) << 52)
        && a.mul_add(b, -product) == 0.0)
        .then_some(product)
}

pub(in crate::runtime::execution_2d) fn exact_div(a: f64, b: f64) -> Option<f64> {
    if b == 0.0 {
        return None;
    }
    let quotient = a / b;
    (quotient.is_finite() && exact_mul(quotient, b) == Some(a)).then_some(quotient)
}

/// A numerical chart receipt, not a replacement for authored transform
/// semantics. Return only when every real product and sum is representable.
pub(in crate::runtime::execution_2d) fn exact_affine_compose(
    parent: [f64; 6],
    child: [f64; 6],
) -> Option<[f64; 6]> {
    let [a, b, c, d, tx, ty] = parent;
    let [e, f, g, h, ux, uy] = child;
    Some([
        exact_add(exact_mul(a, e)?, exact_mul(c, f)?)?,
        exact_add(exact_mul(b, e)?, exact_mul(d, f)?)?,
        exact_add(exact_mul(a, g)?, exact_mul(c, h)?)?,
        exact_add(exact_mul(b, g)?, exact_mul(d, h)?)?,
        exact_add(exact_add(exact_mul(a, ux)?, exact_mul(c, uy)?)?, tx)?,
        exact_add(exact_add(exact_mul(b, ux)?, exact_mul(d, uy)?)?, ty)?,
    ])
}

pub(in crate::runtime::execution_2d) fn exact_row_inverse(raw: [f64; 6]) -> Option<[f64; 6]> {
    let [a, b, c, d, tx, ty] = raw;
    if b != 0.0 || a == 0.0 || d == 0.0 {
        return None;
    }
    let ia = exact_div(1.0, a)?;
    let id = exact_div(1.0, d)?;
    let ic = exact_mul(exact_mul(-c, ia)?, id)?;
    let ix = exact_add(-exact_mul(ia, tx)?, -exact_mul(ic, ty)?)?;
    let iy = -exact_mul(id, ty)?;
    Some([ia, 0.0, ic, id, ix, iy])
}

/// Compare two REAL products of finite binary64 inputs without rounding.
/// Each significand has at most 53 bits, so its 106-bit product fits u128.
/// Removing powers of two gives one exact signed-integer/exponent identity,
/// including subnormal inputs and products beyond binary64's exponent range.
/// This is a rank/identity proof only, never a rounded determinant value.
pub(in crate::runtime::execution_2d) fn exact_products_equal(
    a: f64,
    b: f64,
    c: f64,
    d: f64,
) -> Option<bool> {
    fn parts(value: f64) -> Option<(u128, i32, bool)> {
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as i32;
        if exponent == 0x7ff {
            return None;
        }
        let fraction = u128::from(bits & ((1_u64 << 52) - 1));
        Some((
            if exponent == 0 {
                fraction
            } else {
                fraction | (1_u128 << 52)
            },
            if exponent == 0 {
                -1074
            } else {
                exponent - 1075
            },
            bits >> 63 != 0,
        ))
    }
    fn product(a: f64, b: f64) -> Option<(u128, i32, bool)> {
        let (a, ae, an) = parts(a)?;
        let (b, be, bn) = parts(b)?;
        let significand = a * b;
        if significand == 0 {
            return Some((0, 0, false));
        }
        let zeros = significand.trailing_zeros();
        Some((significand >> zeros, ae + be + zeros as i32, an ^ bn))
    }
    Some(product(a, b)? == product(c, d)?)
}

#[derive(Clone, Copy, Debug)]
pub(in crate::runtime::execution_2d) struct Interval {
    pub(in crate::runtime::execution_2d) lo: f64,
    pub(in crate::runtime::execution_2d) hi: f64,
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

    pub(in crate::runtime::execution_2d) fn new(lo: f64, hi: f64) -> Option<Self> {
        (lo.is_finite() && hi.is_finite() && lo <= hi).then_some(Self { lo, hi })
    }

    pub(in crate::runtime::execution_2d) fn scalar(value: f64) -> Option<Self> {
        Self::new(value, value)
    }

    pub(in crate::runtime::execution_2d) fn add(self, other: Self) -> Option<Self> {
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

    pub(in crate::runtime::execution_2d) fn sub(self, other: Self) -> Option<Self> {
        if other.is_zero() {
            return Some(self);
        }
        Self::new(
            (self.lo - other.hi).next_down(),
            (self.hi - other.lo).next_up(),
        )
    }

    pub(in crate::runtime::execution_2d) fn mul(self, other: Self) -> Option<Self> {
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

    pub(in crate::runtime::execution_2d) fn div(self, other: Self) -> Option<Self> {
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
    use super::{Interval, exact_add, exact_div, exact_mul, exact_products_equal};

    #[test]
    fn product_identity_proves_rank_without_underflow_or_cancellation() {
        assert_eq!(exact_products_equal(1.1, 1.1, 1.1, 1.1), Some(true));
        assert_eq!(exact_products_equal(1.1, 2.6, 1.3, 2.2), Some(true));
        let step = 2.0_f64.powi(-27);
        assert_eq!(
            exact_products_equal(1.0, 1.0, 1.0 - step, 1.0 + step),
            Some(false)
        );
        assert_eq!(
            exact_products_equal(f64::MAX, f64::MAX, f64::MAX, f64::MAX),
            Some(true)
        );
        assert_eq!(
            exact_products_equal(f64::from_bits(1), f64::from_bits(1), 0.0, 1.0),
            Some(false)
        );
        assert_eq!(exact_products_equal(-0.0, f64::MAX, 0.0, -1.0), Some(true));
        assert_eq!(exact_products_equal(-1.1, 1.1, 1.1, 1.1), Some(false));
        assert_eq!(exact_products_equal(f64::INFINITY, 0.0, 0.0, 1.0), None);
    }

    #[test]
    fn scalar_event_identities_require_proved_representability() {
        assert_eq!(exact_add(22.0, 0.375), Some(22.375));
        assert!(exact_add(1.0, f64::from_bits((1023_u64 - 54) << 52)).is_none());
        assert_eq!(exact_mul(f64::MAX, -1.0), Some(-f64::MAX));
        assert_eq!(exact_mul(0.5, 0.25), Some(0.125));
        assert!(exact_mul(0.1, 0.1).is_none());
        assert!(exact_mul(f64::from_bits(1), 0.5).is_none());
        assert!(exact_mul(f64::MAX, 2.0).is_none());
        assert_eq!(exact_div(3.0, 4.0), Some(0.75));
        assert!(exact_div(1.0, 3.0).is_none());
        assert!(exact_div(0.0, 0.0).is_none());
    }

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
