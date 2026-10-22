//! Degenerate finite-cutoff certificates; no alternate geometry or renderer.
use super::*;
use query_compiler::{Compiler, IntegralRequest};

#[test]
fn constant_coordinate_integrals_preserve_exact_cutoffs_and_sign() {
    let mut fuel = 16_777_216;
    let kernel = GaussianKernel::compiled(&mut fuel, MAX_BYTES).unwrap();
    let mut compiler = Compiler::new(&kernel, 0.125, [2.0; 2], &mut fuel).unwrap();
    // Independent110-digit normalized finite Gaussian series, G(1)/G(3).
    let mass = 0.684_537_604_065_696_f64;
    let y = Interval::new(-0.125, 0.125).unwrap();
    let cutoff = 0.375_f64;
    let tiny = 1.973_464_009_797_442_7e-18_f64;
    for (intercept, factor) in [
        (-cutoff, 0.0),
        (-cutoff.next_up(), 0.0),
        (-cutoff.next_down(), tiny),
        (0.0, 0.5),
        (cutoff, 1.0),
        (cutoff.next_down(), 1.0),
        (cutoff.next_up(), 1.0),
    ] {
        for sign in [-1.0, 1.0] {
            let (value, remainder) = compiler
                .integral(
                    IntegralRequest {
                        map: MappedBoundary::Line {
                            intercept,
                            slope: 0.0,
                        },
                        domain: y,
                        y,
                        center: [0.0; 2],
                        terms: 1,
                        sign,
                    },
                    &mut fuel,
                )
                .unwrap();
            let truth = sign * factor * mass;
            assert!(
                value.lo <= truth && truth <= value.hi,
                "{intercept},{sign}: {value:?}"
            );
            assert!(value.hi - value.lo < 1.0e-12);
            assert_eq!(remainder, 0.0);
        }
    }
    assert!(
        compiler.nodes.is_empty(),
        "separable constants need no quadrature"
    );
    let domain = Interval::new(y.lo.next_down(), y.hi.next_up()).unwrap();
    compiler
        .integral(
            IntegralRequest {
                map: MappedBoundary::Line {
                    intercept: 0.0,
                    slope: 0.0,
                },
                domain,
                y,
                center: [0.0; 2],
                terms: 1,
                sign: 1.0,
            },
            &mut fuel,
        )
        .unwrap();
    assert!(
        !compiler.nodes.is_empty(),
        "unequal ordinate limits retain generic endpoint proof"
    );
}

#[test]
fn exact_circle_cutoffs_certify_tangency_and_reduced_polynomial_degree() {
    let circle = |a, b| MappedBoundary::Circle {
        h: 0.0,
        a,
        b,
        k: 0.0,
        r: 1.0,
        sign: 1.0,
    };
    let mut fuel = 1_000_000;
    let tangent = circle(0.0, 1.0)
        .cutoff_events([0.625, 10.0], 0.125, &mut fuel)
        .unwrap()
        .unwrap();
    assert!(tangent.iter().any(|root| root.lo <= 0.0 && root.hi >= 0.0));
    let full = circle(1.0, 1.0)
        .cutoff_events([1.25, 0.5], 0.03125, &mut fuel)
        .unwrap()
        .unwrap();
    assert_eq!(full.len(), 6, "both Y events and all four X roots fit");
    let absent = circle(0.0, 1.0)
        .cutoff_events([-1.375, 10.0], 0.125, &mut fuel)
        .unwrap()
        .unwrap();
    assert!(
        absent.is_empty(),
        "constant nonzero reduced equation has no root"
    );
    let linear = circle(1.0, 1.0)
        .cutoff_events([-1.375, 10.0], 0.125, &mut fuel)
        .unwrap()
        .unwrap();
    assert!(linear.iter().any(|root| root.lo <= -1.0 && root.hi >= -1.0));
    assert!(
        circle(0.0, 0.0)
            .cutoff_events([0.375, 10.0], 0.125, &mut fuel)
            .unwrap()
            .is_none()
    );
}

#[test]
fn unresolved_cutoff_panels_keep_complete_signed_mass() {
    let mut fuel = 16_777_216;
    let kernel = GaussianKernel::compiled(&mut fuel, MAX_BYTES).unwrap();
    let mut compiler = Compiler::new(&kernel, 0.125, [2.0; 2], &mut fuel).unwrap();
    let cutoff = 0.375_f64;
    let y = Interval::new(cutoff.next_down(), cutoff.next_up()).unwrap();
    for sign in [-1.0, 1.0] {
        let (value, remainder) = compiler
            .integral(
                IntegralRequest {
                    map: MappedBoundary::Line {
                        intercept: 1.0,
                        slope: 1.0,
                    },
                    domain: y,
                    y,
                    center: [0.0; 2],
                    terms: 1,
                    sign,
                },
                &mut fuel,
            )
            .unwrap();
        assert!(value.lo <= 0.0 && value.hi >= 0.0);
        let truth = sign * 1.973_464_009_797_442_7e-18_f64;
        assert!(
            value.lo <= truth && truth <= value.hi,
            "signed cutoff mass {sign}: {value:?}"
        );
        assert!(
            value.hi - value.lo > 0.0,
            "uncertain mass must not disappear"
        );
        assert!(value.hi - value.lo < 1.0e-12);
        assert!((0.0..f64::MIN_POSITIVE).contains(&remainder));
    }
}
