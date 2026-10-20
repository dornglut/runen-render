//! Compiled point evaluations of an existing source-derived rational map.
//!
//! The original map remains cutoff, range and complex-bound authority. This
//! module only encloses its values at exact scalar points or exact nodes of
//! an already admitted positive quadrature rule. All coefficient and floating
//! errors are compiled outward; no rounded point is source membership.

use super::{
    AreaQueryError, Components, Coordinates, Interval, MappedBoundary, checked, debit, iv,
};

pub(super) mod query;

const EPSILON: f64 = f64::from_bits((1023_u64 - 53) << 52);
const ETA: f64 = f64::from_bits(1);

/// Nonnegative upper-bound arithmetic. One correctly rounded operation and
/// its upward neighbor enclose the real operation, including underflow.
/// Identity branches preserve exact zeros. Each debit includes validity,
/// arithmetic, rounding, branch and return work, before the operation.
struct Bounds<'a> {
    fuel: &'a mut usize,
}

impl Bounds<'_> {
    fn add(&mut self, a: f64, b: f64) -> Result<f64, AreaQueryError> {
        debit(self.fuel, 24)?;
        if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 {
            return Err(AreaQueryError::Precision);
        }
        if a == 0.0 {
            return Ok(b);
        }
        if b == 0.0 {
            return Ok(a);
        }
        let value = (a + b).next_up();
        if !value.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        Ok(value)
    }
    fn mul(&mut self, a: f64, b: f64) -> Result<f64, AreaQueryError> {
        debit(self.fuel, 24)?;
        if !a.is_finite() || !b.is_finite() || a < 0.0 || b < 0.0 {
            return Err(AreaQueryError::Precision);
        }
        if a == 0.0 || b == 0.0 {
            return Ok(0.0);
        }
        let value = (a * b).next_up();
        if !value.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        Ok(value)
    }
    fn sum(&mut self, terms: &[f64]) -> Result<f64, AreaQueryError> {
        let mut result = 0.0;
        for &term in terms {
            debit(self.fuel, 8)?;
            result = self.add(result, term)?;
        }
        Ok(result)
    }
    fn interval_mul(&mut self, a: Interval, b: Interval) -> Result<Interval, AreaQueryError> {
        debit(self.fuel, 96)?;
        checked(a.mul(b))
    }
    fn scalar(&mut self, value: f64) -> Result<Interval, AreaQueryError> {
        debit(self.fuel, 16)?;
        checked(Interval::scalar(value))
    }
    fn coefficient(&mut self, value: Interval) -> Result<(f64, f64, f64), AreaQueryError> {
        debit(self.fuel, 80)?;
        // A scalar minimum-subnormal coefficient must not be halved twice
        // to zero. Degenerate intervals already supply an exact midpoint.
        let midpoint = if value.lo == value.hi {
            value.lo
        } else {
            value.lo * 0.5 + value.hi * 0.5
        };
        if !midpoint.is_finite() || midpoint < value.lo || midpoint > value.hi {
            return Err(AreaQueryError::Precision);
        }
        let scalar = self.scalar(midpoint)?;
        debit(self.fuel, 48)?;
        let difference = checked(value.sub(scalar))?;
        Ok((
            midpoint,
            difference.lo.abs().max(difference.hi.abs()),
            value.lo.abs().max(value.hi.abs()),
        ))
    }
}

#[derive(Clone, Copy)]
pub(super) struct Panel {
    middle: f64,
    half: f64,
}

impl Panel {
    /// For 0<=a<=b<=1, exact FMA operands satisfy a/2+RN(b/2)<=1
    /// and b/2-RN(a/2) in [-.5,.5]. Thus midpoint error <=eps+2eta
    /// and halfwidth error <=eps/2+2eta, without dropping eps*eta terms.
    /// A final node FMA has exact argument magnitude <=1.5, giving total
    /// input error <=3eps+5eta <4eps+8eta for every exact |node|<=1.
    pub(super) fn new(domain: Interval, fuel: &mut usize) -> Result<Self, AreaQueryError> {
        debit(fuel, 128)?;
        if !domain.lo.is_finite()
            || !domain.hi.is_finite()
            || domain.lo < 0.0
            || domain.lo > domain.hi
            || domain.hi > 1.0
        {
            return Err(AreaQueryError::Precision);
        }
        let middle = domain.lo.mul_add(0.5, domain.hi * 0.5);
        let half = domain.hi.mul_add(0.5, -(domain.lo * 0.5));
        if !(0.0..=1.0).contains(&middle) || half.abs() > 0.5 {
            return Err(AreaQueryError::Precision);
        }
        Ok(Self { middle, half })
    }
    fn node(self, node: f64, fuel: &mut usize) -> Result<UnitInput, AreaQueryError> {
        debit(fuel, 48)?;
        if !node.is_finite() || node.abs() > 1.0 {
            return Err(AreaQueryError::Precision);
        }
        let value = self.half.mul_add(node, self.middle);
        if !value.is_finite() || value.abs() > 1.0 {
            return Err(AreaQueryError::Precision);
        }
        Ok(UnitInput {
            value,
            approximate: true,
        })
    }
}

struct UnitInput {
    value: f64,
    approximate: bool,
}

/// Rounded map values and uniform absolute errors, in x/y/J order. These
/// values are numerical integration inputs, never source membership. The
/// Jacobian value may round slightly below zero; its enclosure alone carries
/// the nonnegative geometric derivative guarantee.
#[derive(Clone, Copy)]
pub(super) struct RawCoordinates {
    pub(super) values: [f64; 3],
    pub(super) errors: [f64; 3],
}

impl UnitInput {
    fn exact(value: f64, fuel: &mut usize) -> Result<Self, AreaQueryError> {
        debit(fuel, 32)?;
        if !value.is_finite() || value.abs() > 1.0 {
            return Err(AreaQueryError::Precision);
        }
        Ok(Self {
            value,
            approximate: false,
        })
    }
}

/// Six midpoint coefficients and two sets of three absolute output errors.
/// No allocated payload. A cache must additionally retain the immutable
/// source-curve identity AND hemisphere sign; this record alone is not a key.
pub(super) struct ScalarMap {
    alpha: f64,
    beta: f64,
    gamma: f64,
    delta: f64,
    h: f64,
    k: f64,
    exact: [f64; 3],
    node: [f64; 3],
}

impl ScalarMap {
    pub(super) fn new(map: MappedBoundary, fuel: &mut usize) -> Result<Self, AreaQueryError> {
        debit(fuel, 192)?;
        let MappedBoundary::Circle {
            h,
            a,
            b,
            k,
            r,
            sign,
        } = map
        else {
            return Err(AreaQueryError::Unsupported);
        };
        if ![h, a, b, k, r, sign].iter().all(|x| x.is_finite())
            || r <= 0.0
            || b.abs() != 1.0
            || sign.abs() != 1.0
        {
            return Err(AreaQueryError::Precision);
        }
        let mut bounds = Bounds { fuel };
        let two = bounds.scalar(2.0)?;
        let ia = bounds.scalar(a)?;
        let ib = bounds.scalar(b)?;
        let ir = bounds.scalar(r)?;
        let isign = bounds.scalar(sign)?;
        let twice_a = bounds.interval_mul(two, ia)?;
        let twice_a_sign = bounds.interval_mul(twice_a, isign)?;
        let alpha = bounds.interval_mul(twice_a_sign, ir)?;
        let beta = bounds.interval_mul(ib, ir)?;
        let twice_sign = bounds.interval_mul(two, isign)?;
        let gamma = bounds.interval_mul(twice_sign, ir)?;
        let delta = bounds.interval_mul(two, ir)?;
        let (alpha, ea, a_upper) = bounds.coefficient(alpha)?;
        let (beta, eb, b_upper) = bounds.coefficient(beta)?;
        let (gamma, eg, g_upper) = bounds.coefficient(gamma)?;
        let (delta, ed, d_upper) = bounds.coefficient(delta)?;

        // Every finite RN operation satisfies |fl(v)-v|<=eps*|v|+eta.
        // Dhat=fma(u,u,1) in [1,2], Chat=fma(-u,u,1) in [0,1],
        // qhat=RN(1/Dhat) in [.5,1], Qhat=RN(qhat²) in [.25,1].
        // These exact ranges handle the cancellation at |u|=1.
        let twice_epsilon = bounds.mul(2.0, EPSILON)?;
        let e_d = bounds.add(twice_epsilon, ETA)?;
        let e_c = bounds.add(EPSILON, ETA)?;
        let e_q = bounds.sum(&[e_d, EPSILON, ETA])?;
        let twice_e_q = bounds.mul(2.0, e_q)?;
        let e_q2 = bounds.sum(&[twice_e_q, EPSILON, ETA])?;

        let rounding = bounds.mul(EPSILON, gamma.abs())?;
        let e_t = bounds.sum(&[eg, rounding, ETA])?;
        let propagated = bounds.mul(g_upper, e_q)?;
        let magnitude = bounds.sum(&[k.abs(), g_upper, e_t])?;
        let rounding = bounds.mul(EPSILON, magnitude)?;
        let e_y = bounds.sum(&[e_t, propagated, rounding, ETA])?;

        let propagated = bounds.mul(b_upper, e_c)?;
        let rounding = bounds.mul(EPSILON, beta.abs())?;
        let e_p = bounds.sum(&[eb, propagated, rounding, ETA])?;
        let magnitude = bounds.sum(&[alpha.abs(), b_upper, e_p])?;
        let rounding = bounds.mul(EPSILON, magnitude)?;
        let e_n = bounds.sum(&[ea, e_p, rounding, ETA])?;
        let numerator = bounds.add(a_upper, b_upper)?;
        let propagated = bounds.mul(numerator, e_q)?;
        let magnitude = bounds.sum(&[h.abs(), a_upper, b_upper, e_n])?;
        let rounding = bounds.mul(EPSILON, magnitude)?;
        let e_x = bounds.sum(&[e_n, propagated, rounding, ETA])?;

        let propagated = bounds.mul(d_upper, e_c)?;
        let rounding = bounds.mul(EPSILON, delta.abs())?;
        let e_pj = bounds.sum(&[ed, propagated, rounding, ETA])?;
        let propagated = bounds.mul(d_upper, e_q2)?;
        let magnitude = bounds.add(d_upper, e_pj)?;
        let rounding = bounds.mul(EPSILON, magnitude)?;
        let e_j = bounds.sum(&[e_pj, propagated, rounding, ETA])?;

        let four_epsilon = bounds.mul(4.0, EPSILON)?;
        let eight_eta = bounds.mul(8.0, ETA)?;
        let tau = bounds.add(four_epsilon, eight_eta)?;
        let four_b = bounds.mul(4.0, b_upper)?;
        let dx = bounds.add(a_upper, four_b)?;
        let dj = bounds.mul(6.0, d_upper)?;
        let input_x = bounds.mul(dx, tau)?;
        let input_y = bounds.mul(g_upper, tau)?;
        let input_j = bounds.mul(dj, tau)?;
        let node = [
            bounds.add(e_x, input_x)?,
            bounds.add(e_y, input_y)?,
            bounds.add(e_j, input_j)?,
        ];

        // Preflight every scalar intermediate, including sums that may
        // cancel to a small final coordinate. No relative-to-final-value
        // gamma bound silently discards this conditioning.
        let _ = bounds.sum(&[gamma.abs(), e_t, k.abs()])?;
        let _ = bounds.sum(&[beta.abs(), e_p, alpha.abs()])?;
        let _ = bounds.sum(&[a_upper, b_upper, e_n, h.abs()])?;
        let _ = bounds.sum(&[delta.abs(), e_pj])?;
        debit(bounds.fuel, 128)?;
        if delta < 0.0 {
            return Err(AreaQueryError::Precision);
        }
        Ok(Self {
            alpha,
            beta,
            gamma,
            delta,
            h,
            k,
            exact: [e_x, e_y, e_j],
            node,
        })
    }

    fn bound(
        value: f64,
        error: f64,
        nonnegative: bool,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        debit(fuel, 48)?;
        if !value.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        let lower = (value - error).next_down();
        let upper = (value + error).next_up();
        checked(Interval::new(
            if nonnegative { lower.max(0.0) } else { lower },
            upper,
        ))
    }

    fn evaluate_raw(
        &self,
        input: UnitInput,
        components: Components,
        fuel: &mut usize,
    ) -> Result<RawCoordinates, AreaQueryError> {
        debit(fuel, 64)?;
        let u = input.value;
        let errors = if input.approximate {
            self.node
        } else {
            self.exact
        };
        let want_x = !matches!(components, Components::Y);
        let want_y = !matches!(components, Components::X);
        let want_j = matches!(components, Components::All);
        let mut result = RawCoordinates {
            values: [0.0; 3],
            errors,
        };
        debit(fuel, 32)?;
        let denominator = u.mul_add(u, 1.0);
        let reciprocal = 1.0 / denominator;
        if want_y {
            debit(fuel, 32)?;
            let product = self.gamma * u;
            let y = product.mul_add(reciprocal, self.k);
            result.values[1] = y;
        }
        if want_x || want_j {
            debit(fuel, 16)?;
            let complement = (-u).mul_add(u, 1.0);
            if want_x {
                debit(fuel, 48)?;
                let product = self.beta * complement;
                let numerator = self.alpha.mul_add(u, product);
                let x = numerator.mul_add(reciprocal, self.h);
                result.values[0] = x;
            }
            if want_j {
                debit(fuel, 48)?;
                let squared = reciprocal * reciprocal;
                let product = self.delta * complement;
                let jacobian = product * squared;
                debit(fuel, 16)?;
                result.values[2] = jacobian;
            }
        }
        // The entry debit includes the component selection, record copies,
        // and three scalar finite guards. Invalid raw output cannot enter a
        // downstream packing path that omits interval construction.
        if !result.values.iter().all(|value| value.is_finite()) {
            return Err(AreaQueryError::Precision);
        }
        Ok(result)
    }

    #[cfg(test)]
    fn evaluate(
        &self,
        input: UnitInput,
        components: Components,
        fuel: &mut usize,
    ) -> Result<Coordinates, AreaQueryError> {
        let raw = self.evaluate_raw(input, components, fuel)?;
        const ZERO: Interval = Interval { lo: 0.0, hi: 0.0 };
        Ok(Coordinates {
            x: if matches!(components, Components::Y) {
                ZERO
            } else {
                Self::bound(raw.values[0], raw.errors[0], false, fuel)?
            },
            y: if matches!(components, Components::X) {
                ZERO
            } else {
                Self::bound(raw.values[1], raw.errors[1], false, fuel)?
            },
            jacobian: if matches!(components, Components::All) {
                Self::bound(raw.values[2], raw.errors[2], true, fuel)?
            } else {
                ZERO
            },
        })
    }

    pub(super) fn raw_at_node(
        &self,
        panel: Panel,
        node: f64,
        fuel: &mut usize,
    ) -> Result<RawCoordinates, AreaQueryError> {
        self.evaluate_raw(panel.node(node, fuel)?, Components::All, fuel)
    }

    #[cfg(test)]
    pub(super) fn at_node(
        &self,
        panel: Panel,
        node: f64,
        fuel: &mut usize,
    ) -> Result<Coordinates, AreaQueryError> {
        self.evaluate(panel.node(node, fuel)?, Components::All, fuel)
    }
    #[cfg(test)]
    pub(super) fn at_exact(
        &self,
        u: f64,
        components: Components,
        fuel: &mut usize,
    ) -> Result<Coordinates, AreaQueryError> {
        self.evaluate(UnitInput::exact(u, fuel)?, components, fuel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_map_cancellation_and_subnormal_coefficients_keep_absolute_errors() {
        let mut fuel = 16_777_216;
        let large = f64::from_bits((1023_u64 + 200) << 52);
        let map = ScalarMap::new(
            MappedBoundary::Circle {
                h: -large,
                a: large,
                b: 1.0,
                k: large,
                r: 1.0,
                sign: 1.0,
            },
            &mut fuel,
        )
        .unwrap();
        let value = map.at_exact(1.0, Components::All, &mut fuel).unwrap();
        assert!(value.x.lo < 0.0 && value.x.hi > 0.0);
        assert!(value.y.lo <= large && value.y.hi >= large.next_up());
        assert_eq!(value.jacobian.lo, 0.0);
        for r in [
            ETA,
            f64::MIN_POSITIVE.next_down(),
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE.next_up(),
        ] {
            let map = ScalarMap::new(
                MappedBoundary::Circle {
                    h: 0.0,
                    a: 0.25,
                    b: 1.0,
                    k: 0.0,
                    r,
                    sign: -1.0,
                },
                &mut fuel,
            )
            .unwrap();
            for (u, x, y, j) in [
                (0.0, r, 0.0, 2.0 * r),
                (0.5, 0.4 * r, -0.8 * r, 0.96 * r),
                (1.0, -0.25 * r, -r, 0.0),
            ] {
                let value = map.at_exact(u, Components::All, &mut fuel).unwrap();
                for (range, reference) in [(value.x, x), (value.y, y), (value.jacobian, j)] {
                    assert!(range.lo <= reference && reference <= range.hi);
                }
            }
        }
    }

    #[test]
    fn scalar_map_finite_domain_overflow_and_fuel_fail_before_output() {
        let original = MappedBoundary::Circle {
            h: 4.0,
            a: 0.25,
            b: 1.0,
            k: 2.0,
            r: 1.0,
            sign: 1.0,
        };
        let mut fuel = 16_777_216;
        for invalid in [
            MappedBoundary::Circle {
                h: 4.0,
                a: f64::MAX,
                b: 1.0,
                k: 2.0,
                r: 1.0,
                sign: 1.0,
            },
            MappedBoundary::Circle {
                h: 4.0,
                a: 0.25,
                b: 1.0,
                k: 2.0,
                r: 0.0,
                sign: 1.0,
            },
            MappedBoundary::Circle {
                h: 4.0,
                a: 0.25,
                b: 1.0,
                k: 2.0,
                r: 1.0,
                sign: 0.0,
            },
            MappedBoundary::Circle {
                h: f64::NAN,
                a: 0.25,
                b: 1.0,
                k: 2.0,
                r: 1.0,
                sign: 1.0,
            },
        ] {
            assert!(matches!(
                ScalarMap::new(invalid, &mut fuel),
                Err(AreaQueryError::Precision)
            ));
        }
        assert!(matches!(
            ScalarMap::new(
                MappedBoundary::Line {
                    intercept: 0.0,
                    slope: 1.0
                },
                &mut fuel
            ),
            Err(AreaQueryError::Unsupported)
        ));
        let map = ScalarMap::new(original, &mut fuel).unwrap();
        for invalid in [
            f64::NAN,
            f64::INFINITY,
            (-1.0_f64).next_down(),
            1.0_f64.next_up(),
        ] {
            assert!(matches!(
                map.at_exact(invalid, Components::All, &mut fuel),
                Err(AreaQueryError::Precision)
            ));
        }
        for domain in [
            Interval {
                lo: f64::NAN,
                hi: 0.0,
            },
            Interval::new(-ETA, 0.0).unwrap(),
            Interval::new(1.0, 1.0_f64.next_up()).unwrap(),
        ] {
            assert!(matches!(
                Panel::new(domain, &mut fuel),
                Err(AreaQueryError::Precision)
            ));
        }
        let panel = Panel::new(Interval::new(0.0, 1.0).unwrap(), &mut fuel).unwrap();
        assert!(matches!(
            map.at_node(panel, 1.0_f64.next_up(), &mut fuel),
            Err(AreaQueryError::Precision)
        ));
        let mut empty = 0;
        assert!(matches!(
            ScalarMap::new(original, &mut empty),
            Err(AreaQueryError::Resource)
        ));
        assert!(matches!(
            Panel::new(iv(0.5), &mut empty),
            Err(AreaQueryError::Resource)
        ));
        assert!(matches!(
            map.at_exact(0.5, Components::All, &mut empty),
            Err(AreaQueryError::Resource)
        ));
        let mut short = 447;
        assert!(matches!(
            map.at_node(panel, 0.5, &mut short),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(short, 0);
    }

    #[test]
    fn scalar_map_actual_cost_and_record_inventory_are_bounded() {
        let mut fuel = 16_777_216;
        let before = fuel;
        let map = ScalarMap::new(
            MappedBoundary::Circle {
                h: 4.0,
                a: 0.25,
                b: 1.0,
                k: 2.0,
                r: 1.0,
                sign: 1.0,
            },
            &mut fuel,
        )
        .unwrap();
        let construction = before - fuel;
        assert!(construction <= 8_192);
        let before = fuel;
        let panel = Panel::new(Interval::new(0.0, 1.0).unwrap(), &mut fuel).unwrap();
        assert_eq!(before - fuel, 128);
        let before = fuel;
        map.at_node(panel, 0.5, &mut fuel).unwrap();
        assert_eq!(before - fuel, 448);
        let before = fuel;
        let raw = map.raw_at_node(panel, 0.0, &mut fuel).unwrap();
        assert_eq!(before - fuel, 304);
        // Exact original rational map at u=1/2: (24/5,14/5,24/25).
        // Neighbors of each rounded rational enclose its real value.
        for (index, reference) in [4.8_f64, 2.8, 0.96].into_iter().enumerate() {
            assert!(raw.values[index] - raw.errors[index] <= reference.next_down());
            assert!(reference.next_up() <= raw.values[index] + raw.errors[index]);
        }
        let mut short = 303;
        assert!(matches!(
            map.raw_at_node(panel, 0.0, &mut short),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(short, 0);
        for (component, cost) in [
            (Components::All, 432),
            (Components::Position, 320),
            (Components::X, 240),
            (Components::Y, 208),
        ] {
            let before = fuel;
            map.at_exact(0.5, component, &mut fuel).unwrap();
            assert_eq!(before - fuel, cost);
        }
        assert_eq!(std::mem::size_of::<ScalarMap>(), 96);
        assert_eq!(std::mem::size_of::<Panel>(), 16);
        assert_eq!(std::mem::size_of::<UnitInput>(), 16);
        println!(
            "Compiled rational map: construction_work={construction}, panel_work=128, node_work=448, raw_node_work=304, exact_all=432, position=320, x=240, y=208, map_bytes=96, panel_bytes=16"
        );
    }

    #[test]
    fn scalar_map_rational_values_and_uniform_node_input_are_enclosed() {
        let mut fuel = 16_777_216;
        for b in [-1.0, 1.0] {
            for sign in [-1.0, 1.0] {
                let original = MappedBoundary::Circle {
                    h: 4.0,
                    a: 0.25,
                    b,
                    k: 2.0,
                    r: 1.0,
                    sign,
                };
                let map = ScalarMap::new(original, &mut fuel).unwrap();
                for (u, x, y, j) in [
                    (0.0, 4.0 + b, 2.0, 2.0),
                    (0.5, 4.0 + sign * 0.2 + b * 0.6, 2.0 + sign * 0.8, 0.96),
                    (1.0, 4.0 + sign * 0.25, 2.0 + sign, 0.0),
                ] {
                    let value = map.at_exact(u, Components::All, &mut fuel).unwrap();
                    for (range, expected) in [(value.x, x), (value.y, y), (value.jacobian, j)] {
                        assert!(range.lo <= expected && expected <= range.hi);
                    }
                    let panel = Panel::new(iv(u), &mut fuel).unwrap();
                    for node in [-1.0, 0.0, 1.0] {
                        let value = map.at_node(panel, node, &mut fuel).unwrap();
                        for (range, expected) in [(value.x, x), (value.y, y), (value.jacobian, j)] {
                            assert!(range.lo <= expected && expected <= range.hi);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scalar_map_decimal120_exact_nodes_and_seams_are_enclosed() {
        // Independent exact-rational source formula evaluated with Python
        // Fraction, converted via Decimal120. Inputs/outputs are exact f64
        // bit patterns; one neighbor on either side encloses the reference.
        // No compiled alpha/beta, runtime interval arithmetic or error bound
        // is reused. Panel references use the EXACT real (a+b)/2+(b-a)*t/2.
        let mut fuel = 16_777_216;
        let map = ScalarMap::new(
            MappedBoundary::Circle {
                h: 1.1,
                a: 0.1,
                b: -1.0,
                k: -2.3,
                r: 0.7,
                sign: -1.0,
            },
            &mut fuel,
        )
        .unwrap();
        const POINTS: &[(u64, [u64; 3])] = &[
            (
                0x0000000000000000,
                [0x3fd999999999999c, 0xc002666666666666, 0x3ff6666666666666],
            ),
            (
                0x0000000000000001,
                [0x3fd999999999999c, 0xc002666666666666, 0x3ff6666666666666],
            ),
            (
                0x0010000000000000,
                [0x3fd999999999999c, 0xc002666666666666, 0x3ff6666666666666],
            ),
            (
                0x3fdfffffffffffff,
                [0x3fe3f7ced916872c, 0xc006e147ae147ae1, 0x3fe5810624dd2f1b],
            ),
            (
                0x3fe0000000000000,
                [0x3fe3f7ced916872c, 0xc006e147ae147ae1, 0x3fe5810624dd2f1a],
            ),
            (
                0x3fe0000000000001,
                [0x3fe3f7ced916872d, 0xc006e147ae147ae1, 0x3fe5810624dd2f18],
            ),
            (
                0x3fefffffffffffff,
                [0x3ff07ae147ae147b, 0xc007ffffffffffff, 0x3c96666666666667],
            ),
            (
                0x3ff0000000000000,
                [0x3ff07ae147ae147b, 0xc008000000000000, 0x0000000000000000],
            ),
        ];
        for &(u, expected) in POINTS {
            let value = map
                .at_exact(f64::from_bits(u), Components::All, &mut fuel)
                .unwrap();
            for (range, reference) in [
                (value.x, expected[0]),
                (value.y, expected[1]),
                (value.jacobian, expected[2]),
            ] {
                let reference = f64::from_bits(reference);
                let lower = if reference == 0.0 && range.lo == 0.0 {
                    0.0
                } else {
                    reference.next_down()
                };
                assert!(range.lo <= lower && reference.next_up() <= range.hi);
            }
        }
        const NODES: &[(u64, u64, u64, [u64; 3])] = &[
            (
                0x0000000000000000,
                0x3ff0000000000000,
                0xbfefa92c264d787e,
                [0x3fd98e16b326e7b1, 0xc00275982a2169f8, 0x3ff665eab7fd933b],
            ),
            (
                0x0000000000000000,
                0x3ff0000000000000,
                0x0000000000000000,
                [0x3fe3f7ced916872c, 0xc006e147ae147ae1, 0x3fe5810624dd2f1a],
            ),
            (
                0x0000000000000000,
                0x3ff0000000000000,
                0x3fefa92c264d787e,
                [0x3ff06ba622a162bf, 0xc007fffad1a92c5a, 0x3f6ea1cf45f5b0da],
            ),
            (
                0x3fd0000000000000,
                0x3fe8000000000000,
                0xbfefa92c264d787e,
                [0x3fdcd9273449ced1, 0xc0050f550bb81c82, 0x3ff287775084ded2],
            ),
            (
                0x3fd0000000000000,
                0x3fe8000000000000,
                0x0000000000000000,
                [0x3fe3f7ced916872c, 0xc006e147ae147ae1, 0x3fe5810624dd2f1a],
            ),
            (
                0x3fd0000000000000,
                0x3fe8000000000000,
                0x3fefa92c264d787e,
                [0x3feab4ed9dcaee02, 0xc007c548edc62d99, 0x3fd048d6eebbd5df],
            ),
            (
                0x3fdfffffffffffff,
                0x3fe0000000000001,
                0xbfefa92c264d787e,
                [0x3fe3f7ced916872c, 0xc006e147ae147ae1, 0x3fe5810624dd2f1b],
            ),
            (
                0x3fdfffffffffffff,
                0x3fe0000000000001,
                0x0000000000000000,
                [0x3fe3f7ced916872c, 0xc006e147ae147ae1, 0x3fe5810624dd2f1a],
            ),
            (
                0x3fdfffffffffffff,
                0x3fe0000000000001,
                0x3fefa92c264d787e,
                [0x3fe3f7ced916872d, 0xc006e147ae147ae1, 0x3fe5810624dd2f18],
            ),
            (
                0x3fefffffffffffff,
                0x3ff0000000000000,
                0xbfefa92c264d787e,
                [0x3ff07ae147ae147b, 0xc007ffffffffffff, 0x3c964802a70183c6],
            ),
            (
                0x3fefffffffffffff,
                0x3ff0000000000000,
                0x0000000000000000,
                [0x3ff07ae147ae147b, 0xc007ffffffffffff, 0x3c86666666666667],
            ),
            (
                0x3fefffffffffffff,
                0x3ff0000000000000,
                0x3fefa92c264d787e,
                [0x3ff07ae147ae147b, 0xc007ffffffffffff, 0x3c1e63bf64e2a0b3],
            ),
            (
                0x0000000000000000,
                0x0000000000000001,
                0xbfefa92c264d787e,
                [0x3fd999999999999c, 0xc002666666666666, 0x3ff6666666666666],
            ),
            (
                0x0000000000000000,
                0x0000000000000001,
                0x0000000000000000,
                [0x3fd999999999999c, 0xc002666666666666, 0x3ff6666666666666],
            ),
            (
                0x0000000000000000,
                0x0000000000000001,
                0x3fefa92c264d787e,
                [0x3fd999999999999c, 0xc002666666666666, 0x3ff6666666666666],
            ),
            (
                0x0000000000000001,
                0x0000000000000003,
                0xbfefa92c264d787e,
                [0x3fd999999999999c, 0xc002666666666666, 0x3ff6666666666666],
            ),
            (
                0x0000000000000001,
                0x0000000000000003,
                0x0000000000000000,
                [0x3fd999999999999c, 0xc002666666666666, 0x3ff6666666666666],
            ),
            (
                0x0000000000000001,
                0x0000000000000003,
                0x3fefa92c264d787e,
                [0x3fd999999999999c, 0xc002666666666666, 0x3ff6666666666666],
            ),
        ];
        for &(a, b, node, expected) in NODES {
            let panel = Panel::new(
                Interval::new(f64::from_bits(a), f64::from_bits(b)).unwrap(),
                &mut fuel,
            )
            .unwrap();
            let value = map.at_node(panel, f64::from_bits(node), &mut fuel).unwrap();
            let raw = map
                .raw_at_node(panel, f64::from_bits(node), &mut fuel)
                .unwrap();
            for (index, &reference) in expected.iter().enumerate() {
                let reference = f64::from_bits(reference);
                assert!(raw.values[index] - raw.errors[index] <= reference.next_down());
                assert!(reference.next_up() <= raw.values[index] + raw.errors[index]);
            }
            for (range, reference) in [
                (value.x, expected[0]),
                (value.y, expected[1]),
                (value.jacobian, expected[2]),
            ] {
                let reference = f64::from_bits(reference);
                let lower = if range.lo >= 0.0 {
                    reference.next_down().max(0.0)
                } else {
                    reference.next_down()
                };
                assert!(range.lo <= lower && reference.next_up() <= range.hi);
            }
        }
    }
}
