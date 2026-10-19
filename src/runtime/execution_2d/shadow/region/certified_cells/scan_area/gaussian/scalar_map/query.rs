//! Uniform panel certificates for scalar kernel inputs. Test-only; no mask.
//! Geometry/cutoff/quadrature authority stays in the original mapped boundary.
use super::*;

enum Geometry<'a> {
    Circle {
        map: &'a ScalarMap,
        panel: Panel,
    },
    Line {
        intercept: f64,
        slope: f64,
        middle: f64,
        half: f64,
    },
}

/// Signed orientation is applied by the query owner after this positive rule.
pub(in super::super) struct Node {
    pub(in super::super) z: [f64; 2],
    pub(in super::super) z_error: [f64; 2],
    pub(in super::super) weight: f64,
    pub(in super::super) weight_error: f64,
}

pub(in super::super) struct QueryPanel<'a> {
    geometry: Geometry<'a>,
    center: [f64; 2],
    normalization: Normalization,
}

/// Numerical bounds compiled for an immutable map/domain and all query
/// centers in a declared box. It carries no borrowed geometry or membership.
#[derive(Clone, Copy)]
pub(in super::super) struct Normalization {
    center_bound: [f64; 2],
    sigma: f64,
    prefactor: f64,
    coordinate_errors: [f64; 3],
    z_error: [f64; 2],
    weight_error: f64,
}

impl Bounds<'_> {
    fn div(&mut self, a: f64, b: f64) -> Result<f64, AreaQueryError> {
        debit(self.fuel, 32)?;
        if !a.is_finite() || a < 0.0 || !b.is_finite() || b <= 0.0 {
            return Err(AreaQueryError::Precision);
        }
        let value = if a == 0.0 { 0.0 } else { (a / b).next_up() };
        if !value.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        Ok(value)
    }
    fn rounding(&mut self, magnitude: f64) -> Result<f64, AreaQueryError> {
        let relative = self.mul(EPSILON, magnitude)?;
        self.add(relative, ETA)
    }
}

impl<'a> QueryPanel<'a> {
    pub(in super::super) fn new(
        map: MappedBoundary,
        compiled: Option<&'a super::super::CompiledEntry>,
        domain: Interval,
        center: [f64; 2],
        sigma: f64,
        fuel: &mut usize,
    ) -> Result<Self, AreaQueryError> {
        debit(fuel, 256)?;
        if !(sigma.is_finite() && sigma > 0.0)
            || !center.into_iter().all(f64::is_finite)
            || !domain.lo.is_finite()
            || !domain.hi.is_finite()
            || domain.lo >= domain.hi
        {
            return Err(AreaQueryError::Precision);
        }
        // These range bounds already belong to the original map, not a new
        // source identity. Every floating/intermediate bound is compiled once.
        let coordinates = map.at(domain, fuel)?;
        let mut bounds = Bounds { fuel };
        let maximum = domain.lo.abs().max(domain.hi.abs());
        let four_epsilon = bounds.mul(4.0, EPSILON)?;
        let relative = bounds.mul(four_epsilon, maximum)?;
        let four_eta = bounds.mul(4.0, ETA)?;
        // Each half-input product has <=eps*M/2+eta error. Final FMA
        // contributes <=eps*(M+eps*M/2+eta)+eta; 4eps*M+4eta dominates.
        let half_error = bounds.add(relative, four_eta)?;
        debit(bounds.fuel, 64)?;
        let middle = domain.lo.mul_add(0.5, domain.hi * 0.5);
        let half = domain.hi.mul_add(0.5, -(domain.lo * 0.5));
        if !middle.is_finite() || !half.is_finite() || half <= 0.0 {
            return Err(AreaQueryError::Precision);
        }
        let (geometry, errors) = match map {
            MappedBoundary::Circle {
                h,
                a,
                b,
                k,
                r,
                sign,
            } => {
                debit(bounds.fuel, 96)?;
                let entry = compiled.ok_or(AreaQueryError::Precision)?;
                // Reuse the cache owner's exact original-parameter key. A
                // rounded coefficient match cannot establish map identity.
                if entry.key != [h, a, b, k, r, sign].map(f64::to_bits) {
                    return Err(AreaQueryError::Precision);
                }
                let compiled = &entry.map;
                let panel = Panel::new(domain, bounds.fuel)?;
                (
                    Geometry::Circle {
                        map: compiled,
                        panel,
                    },
                    compiled.node,
                )
            }
            MappedBoundary::Line { intercept, slope } => {
                if !intercept.is_finite() || !slope.is_finite() {
                    return Err(AreaQueryError::Precision);
                }
                let magnitude = bounds.add(middle.abs(), half.abs())?;
                let node_rounding = bounds.rounding(magnitude)?;
                let two_half_errors = bounds.mul(2.0, half_error)?;
                let ey = bounds.add(two_half_errors, node_rounding)?;
                let y_upper = bounds.add(maximum, ey)?;
                let x_product = bounds.mul(slope.abs(), y_upper)?;
                let x_magnitude = bounds.add(intercept.abs(), x_product)?;
                let x_rounding = bounds.rounding(x_magnitude)?;
                let propagated = bounds.mul(slope.abs(), ey)?;
                let ex = bounds.add(propagated, x_rounding)?;
                (
                    Geometry::Line {
                        intercept,
                        slope,
                        middle,
                        half,
                    },
                    [ex, ey, 0.0],
                )
            }
        };
        let ranges = [coordinates.0, coordinates.1, coordinates.2];
        let mut z_error = [0.0; 2];
        for i in 0..2 {
            debit(bounds.fuel, 32)?;
            let true_upper = ranges[i].lo.abs().max(ranges[i].hi.abs());
            let raw_upper = bounds.add(true_upper, errors[i])?;
            let numerator = bounds.add(raw_upper, center[i].abs())?;
            let subtract_error = bounds.rounding(numerator)?;
            let rounded_numerator = bounds.add(numerator, subtract_error)?;
            let divided = bounds.div(rounded_numerator, sigma)?;
            let divide_error = bounds.rounding(divided)?;
            let input_error = bounds.add(errors[i], subtract_error)?;
            let propagated = bounds.div(input_error, sigma)?;
            z_error[i] = bounds.add(propagated, divide_error)?;
        }
        debit(bounds.fuel, 32)?;
        let prefactor = half / sigma;
        if !prefactor.is_finite() || prefactor <= 0.0 {
            return Err(AreaQueryError::Precision);
        }
        let magnitude = bounds.div(half, sigma)?;
        let rounding = bounds.rounding(magnitude)?;
        let propagated = bounds.div(half_error, sigma)?;
        let prefactor_error = bounds.add(propagated, rounding)?;
        let weight_max = super::super::RULE[0].1;
        let jacobian_max = coordinates.2.hi;
        // A = RN(RN(prefactor*w)*max(raw_J,0)). Because true J>=0,
        // clamping cannot increase its absolute error. Bound both products
        // plus half/normalization and J uncertainty uniformly for all nodes.
        let c = bounds.mul(prefactor, weight_max)?;
        let c_error = bounds.rounding(c)?;
        let c_upper = bounds.add(c, c_error)?;
        let first = bounds.mul(prefactor_error, weight_max)?;
        let first = bounds.add(first, c_error)?;
        let first = bounds.mul(first, jacobian_max)?;
        let second = bounds.mul(c_upper, errors[2])?;
        let j_upper = bounds.add(jacobian_max, errors[2])?;
        let product_upper = bounds.mul(c_upper, j_upper)?;
        let final_rounding = bounds.rounding(product_upper)?;
        let weight_error = bounds.sum(&[first, second, final_rounding])?;
        debit(bounds.fuel, 96)?;
        Ok(Self {
            geometry,
            center,
            normalization: Normalization {
                center_bound: center.map(f64::abs),
                sigma,
                prefactor,
                coordinate_errors: errors,
                z_error,
                weight_error,
            },
        })
    }

    pub(in super::super) fn normalization(&self) -> Normalization {
        self.normalization
    }

    pub(in super::super) fn node(
        &self,
        node: f64,
        weight: f64,
        fuel: &mut usize,
    ) -> Result<Node, AreaQueryError> {
        self.normalize_node(self.raw_node(node, fuel)?, weight, fuel)
    }

    pub(in super::super) fn raw_node(
        &self,
        node: f64,
        fuel: &mut usize,
    ) -> Result<RawCoordinates, AreaQueryError> {
        Ok(match self.geometry {
            Geometry::Circle { map, panel } => map.raw_at_node(panel, node, fuel)?,
            Geometry::Line {
                intercept,
                slope,
                middle,
                half,
            } => {
                debit(fuel, 80)?;
                if !node.is_finite() || node.abs() > 1.0 {
                    return Err(AreaQueryError::Precision);
                }
                let y = half.mul_add(node, middle);
                let x = slope.mul_add(y, intercept);
                if !x.is_finite() || !y.is_finite() {
                    return Err(AreaQueryError::Precision);
                }
                RawCoordinates {
                    values: [x, y, 1.0],
                    errors: self.normalization.coordinate_errors,
                }
            }
        })
    }

    pub(in super::super) fn normalize_node(
        &self,
        raw: RawCoordinates,
        weight: f64,
        fuel: &mut usize,
    ) -> Result<Node, AreaQueryError> {
        self.normalization.node(raw, weight, self.center, fuel)
    }
}

impl Normalization {
    pub(in super::super) fn node(
        &self,
        raw: RawCoordinates,
        weight: f64,
        center: [f64; 2],
        fuel: &mut usize,
    ) -> Result<Node, AreaQueryError> {
        // Paid before arithmetic: finite/identity/weight guards, two coordinate
        // normalizations, J clamp, two weight products and record bookkeeping.
        // Numeric cache identity is supplied by the exact map/domain key; an
        // unrelated error certificate is rejected independently here.
        debit(fuel, 144)?;
        if !raw.values.into_iter().all(f64::is_finite)
            || raw.errors != self.coordinate_errors
            || !center
                .into_iter()
                .zip(self.center_bound)
                .all(|(c, bound)| c.is_finite() && c.abs() <= bound)
            || !weight.is_finite()
            || weight <= 0.0
            || weight > super::super::RULE[0].1
        {
            return Err(AreaQueryError::Precision);
        }
        let z = [
            (raw.values[0] - center[0]) / self.sigma,
            (raw.values[1] - center[1]) / self.sigma,
        ];
        let weight = (self.prefactor * weight) * raw.values[2].max(0.0);
        if !z.into_iter().all(f64::is_finite) || !weight.is_finite() || weight < 0.0 {
            return Err(AreaQueryError::Precision);
        }
        Ok(Node {
            z,
            z_error: self.z_error,
            weight,
            weight_error: self.weight_error,
        })
    }
}

#[test]
fn compiled_scalar_query_nodes_enclose_independent_rational_values() {
    let mut fuel = 16_777_216;
    let map = MappedBoundary::Circle {
        h: 4.0,
        a: 0.25,
        b: 1.0,
        k: 2.0,
        r: 1.0,
        sign: 1.0,
    };
    let compiled = super::super::CompiledEntry {
        key: [4.0_f64, 0.25, 1.0, 2.0, 1.0, 1.0].map(f64::to_bits),
        map: ScalarMap::new(map, &mut fuel).unwrap(),
    };
    let panel = QueryPanel::new(
        map,
        Some(&compiled),
        iv_range(0.0, 1.0),
        [4.0, 2.0],
        0.5,
        &mut fuel,
    )
    .unwrap();
    let before = fuel;
    let value = panel.node(0.0, 0.125, &mut fuel).unwrap();
    assert_eq!(before - fuel, 448);
    // u=1/2: x=24/5,y=14/5,J=24/25; half/sigma=1.
    for (index, reference) in [1.6_f64, 1.6].into_iter().enumerate() {
        assert!(value.z[index] - value.z_error[index] <= reference.next_down());
        assert!(reference.next_up() <= value.z[index] + value.z_error[index]);
    }
    let reference = 0.12_f64;
    assert!(value.weight - value.weight_error <= reference.next_down());
    assert!(reference.next_up() <= value.weight + value.weight_error);
    let line = MappedBoundary::Line {
        intercept: 1.0,
        slope: 0.25,
    };
    let panel = QueryPanel::new(
        line,
        None,
        iv_range(18.0, 18.25),
        [5.0, 18.0],
        0.125,
        &mut fuel,
    )
    .unwrap();
    let before = fuel;
    let value = panel.node(0.0, 0.125, &mut fuel).unwrap();
    assert_eq!(before - fuel, 224);
    for (index, reference) in [4.25_f64, 1.0].into_iter().enumerate() {
        assert!(value.z[index] - value.z_error[index] <= reference);
        assert!(reference <= value.z[index] + value.z_error[index]);
    }
    assert!((value.weight - 0.125).abs() <= value.weight_error);
}

#[cfg(test)]
fn iv_range(lo: f64, hi: f64) -> Interval {
    Interval::new(lo, hi).unwrap()
}

#[test]
fn uniform_normalization_encloses_box_corners_and_rejects_outside_centers() {
    let mut fuel = 16_777_216;
    let panel = QueryPanel::new(
        MappedBoundary::Line {
            intercept: 1.0,
            slope: 0.25,
        },
        None,
        iv_range(-0.5, 0.5),
        [8.0; 2],
        0.25,
        &mut fuel,
    )
    .unwrap();
    let normalization = panel.normalization();
    for t in [-1.0, 0.0, 1.0] {
        let raw = panel.raw_node(t, &mut fuel).unwrap();
        let y = t * 0.5;
        for center in [[-8.0, 8.0], [8.0, -8.0], [0.0; 2]] {
            let node = normalization.node(raw, 0.125, center, &mut fuel).unwrap();
            // Exact dyadic arithmetic, directly from the authored line.
            let truth = [(1.0 + 0.25 * y - center[0]) * 4.0, (y - center[1]) * 4.0];
            for (i, truth) in truth.into_iter().enumerate() {
                assert!((node.z[i] - truth).abs() <= node.z_error[i]);
            }
            assert!((node.weight - 0.25).abs() <= node.weight_error);
        }
        assert!(matches!(
            normalization.node(raw, 0.125, [8.0_f64.next_up(), 0.0], &mut fuel),
            Err(AreaQueryError::Precision)
        ));
        let mut empty = 0;
        assert!(matches!(
            normalization.node(raw, 0.125, [0.0; 2], &mut empty),
            Err(AreaQueryError::Resource)
        ));
    }
}

#[test]
fn scalar_query_fraction_oracle_covers_nonzero_nodes_and_nondyadic_sigma() {
    // Python Fraction: source parameters, domain, rule nodes/weights and sigma
    // are exact binary64 rationals; no map coefficients/bounds are reused.
    // Each stored nearest binary64 and its two neighbors bracket exact truth.
    const EXPECTED: [[u64; 3]; 6] = [
        [0x40249d3910c0ac61, 0x40044c5cea079d69, 0x3fc8cd0168a5e4a9],
        [0x401e7d415c7048ce, 0x4020a610ab259fec, 0x3fe393fa4414b012],
        [0x400ec9b300cd5a8e, 0x4023cf929a261840, 0x3f904507cfd26d38],
        [0x4012c3644680f94b, 0xbff3c9bb97f06b4e, 0x3fa160a8b0f2d94a],
        [0x40141e676cc01454, 0x3fbe676cc0145477, 0x3fce4fe5a211e17b],
        [0x40153c9bb97f06b5, 0x3ff3c9bb97f06b4e, 0x3fa160a8b0f2d94a],
    ];
    let mut fuel = 16_777_216;
    let map = MappedBoundary::Circle {
        h: 4.0,
        a: 0.25,
        b: 1.0,
        k: 2.0,
        r: 1.0,
        sign: 1.0,
    };
    let compiled = super::super::CompiledEntry {
        key: [4.0_f64, 0.25, 1.0, 2.0, 1.0, 1.0].map(f64::to_bits),
        map: ScalarMap::new(map, &mut fuel).unwrap(),
    };
    let panels = [
        QueryPanel::new(
            map,
            Some(&compiled),
            iv_range(0.125, 0.875),
            [4.0, 2.0],
            0.1,
            &mut fuel,
        )
        .unwrap(),
        QueryPanel::new(
            MappedBoundary::Line {
                intercept: 1.0,
                slope: 0.25,
            },
            None,
            iv_range(17.875, 18.125),
            [5.0, 18.0],
            0.1,
            &mut fuel,
        )
        .unwrap(),
    ];
    let rule = super::super::RULE;
    for (kind, panel) in panels.iter().enumerate() {
        for (i, (node, weight)) in [(-rule[7].0, rule[7].1), (rule[0].0, rule[0].1), rule[7]]
            .into_iter()
            .enumerate()
        {
            let value = panel.node(node, weight, &mut fuel).unwrap();
            for (index, (actual, radius)) in [
                (value.z[0], value.z_error[0]),
                (value.z[1], value.z_error[1]),
                (value.weight, value.weight_error),
            ]
            .into_iter()
            .enumerate()
            {
                let truth = f64::from_bits(EXPECTED[3 * kind + i][index]);
                assert!((actual - radius).next_down() <= truth.next_down());
                assert!(truth.next_up() <= (actual + radius).next_up());
            }
        }
    }
}

#[test]
fn scalar_query_rejects_unrelated_compilation_invalid_domain_and_short_fuel() {
    let mut fuel = 16_777_216;
    let map = MappedBoundary::Circle {
        h: 4.0,
        a: 0.25,
        b: 1.0,
        k: 2.0,
        r: 1.0,
        sign: 1.0,
    };
    let mut compiled = super::super::CompiledEntry {
        key: [4.0_f64, 0.25, 1.0, 2.0, 1.0, 1.0].map(f64::to_bits),
        map: ScalarMap::new(map, &mut fuel).unwrap(),
    };
    for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(matches!(
            QueryPanel::new(
                map,
                Some(&compiled),
                iv_range(0.0, 1.0),
                [0.0; 2],
                sigma,
                &mut fuel
            ),
            Err(AreaQueryError::Precision)
        ));
    }
    compiled.key[0] = 5.0_f64.to_bits();
    assert!(matches!(
        QueryPanel::new(
            map,
            Some(&compiled),
            iv_range(0.0, 1.0),
            [0.0; 2],
            0.1,
            &mut fuel
        ),
        Err(AreaQueryError::Precision)
    ));
    assert!(matches!(
        QueryPanel::new(map, None, iv_range(0.0, 1.0), [0.0; 2], 0.1, &mut fuel),
        Err(AreaQueryError::Precision)
    ));
    let line = MappedBoundary::Line {
        intercept: 1.0,
        slope: 0.25,
    };
    assert!(matches!(
        QueryPanel::new(line, None, iv_range(0.0, 0.0), [0.0; 2], 0.1, &mut fuel),
        Err(AreaQueryError::Precision)
    ));
    let panel = QueryPanel::new(line, None, iv_range(0.0, 1.0), [0.0; 2], 0.1, &mut fuel).unwrap();
    let mut short = 207;
    assert!(matches!(
        panel.node(0.0, 0.125, &mut short),
        Err(AreaQueryError::Resource)
    ));
    assert_eq!(short, 0);
    assert!(matches!(
        panel.node(1.0_f64.next_up(), 0.125, &mut fuel),
        Err(AreaQueryError::Precision)
    ));
    let mut raw = panel.raw_node(0.0, &mut fuel).unwrap();
    raw.errors[0] = raw.errors[0].next_up();
    assert!(matches!(
        panel.normalize_node(raw, 0.125, &mut fuel),
        Err(AreaQueryError::Precision)
    ));
}
