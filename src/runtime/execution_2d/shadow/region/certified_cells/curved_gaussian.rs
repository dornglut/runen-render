//! Test-only generalizable feasibility of 1D normalized Gaussian-CDF mass
//! integrated across an original-F1 strictly-monotone Bézier boundary.
//!
//! Scope is deliberately narrower than F3F: one x-monotone quadratic/cubic
//! contour with a horizontal lower chord, in the immediate-parent frame,
//! without clips/spread/ancestor effects. The shared certificate API works on
//! any strictly x-monotone authored quadratic/cubic controls. Every x slab's
//! true curve ordinate is enclosed by outward-rounded control hulls. The
//! Gaussian CDF itself, normalization and work/buffer admission come from
//! axis_gaussian; no duplicate raster/SDF/shape authority.
//!
//! This is feasibility evidence, NOT the production general evaluator.
use super::Interval;
use super::axis_gaussian::{MAX_PROOF_BYTES, MAX_SCENE_WORK, NormalizedGaussianCdf};
use super::bezier_winding::certified_graph_y_slab;
use crate::composition_2d::{Render2dPath, Render2dPathCommand};

const STRIPS: usize = 512;
const SLAB_CHARGE: usize = 64;
const MAX_WIDTH: f64 = 0.005;

#[derive(Debug)]
struct GraphMassProof {
    interval: [f64; 2],
    work: usize,
    peak_bytes: usize,
    strips: usize,
}

fn certify_monotone_graph_mass(
    path: &Render2dPath,
    center: [f64; 2],
    sigma: f64,
    work_limit: usize,
    byte_limit: usize,
) -> Option<GraphMassProof> {
    if !center.into_iter().all(f64::is_finite)
        || !sigma.is_finite()
        || sigma <= 0.0
        || work_limit > MAX_SCENE_WORK
        || byte_limit > MAX_PROOF_BYTES
    {
        return None;
    }
    // Consume the ORIGINAL immutable F1 command stream. Admitting a raw
    // polygon/control array would allow a derived approximation to claim
    // semantic region authority. Any other contour/edge topology rejects.
    let commands = path.commands();
    let command_cost = commands.len().checked_mul(40)?;
    if command_cost > work_limit {
        return None;
    }
    let point = |p: &crate::composition_2d::Render2dPoint| [p.x(), p.y()];
    let mut buffer = [[0.0; 2]; 4];
    let size = match commands {
        [
            Render2dPathCommand::MoveTo(a),
            Render2dPathCommand::QuadraticTo { control, to },
            Render2dPathCommand::Close,
        ] => {
            buffer[0] = point(a);
            buffer[1] = point(control);
            buffer[2] = point(to);
            3
        }
        [
            Render2dPathCommand::MoveTo(a),
            Render2dPathCommand::CubicTo {
                control1,
                control2,
                to,
            },
            Render2dPathCommand::Close,
        ] => {
            buffer[0] = point(a);
            buffer[1] = point(control1);
            buffer[2] = point(control2);
            buffer[3] = point(to);
            4
        }
        _ => return None,
    };
    let controls = &buffer[..size];
    let lower_chord = controls[0][1];
    if controls[size - 1][1] != lower_chord {
        return None;
    }
    // Certified domain guard: the ENTIRE finite normalized convolution
    // kernel lies horizontally between authored start and end. Horizontal
    // end caps/other contours require independent certificates.
    let sigma_interval = Interval::scalar(sigma)?;
    let kernel = sigma_interval.mul(Interval::new(-3.0, 3.0)?)?;
    let cx = Interval::scalar(center[0])?;
    let support = cx.add(kernel)?;
    if support.lo <= controls[0][0] || support.hi >= controls[controls.len() - 1][0] {
        return None;
    }
    // This is a lower bound on the retained binary64 interval buffers.
    // Reject impossible byte budgets BEFORE the shared CDF allocation.
    if byte_limit < (16_384 * 2 + 1) * std::mem::size_of::<Interval>() {
        return None;
    }
    let mut work = work_limit - command_cost;
    let cdf = NormalizedGaussianCdf::new(&mut work)?;
    let peak_bytes = cdf.peak_buffer_bytes()?;
    if peak_bytes > byte_limit {
        return None;
    }
    let mut lower = Interval::scalar(0.0)?;
    let mut upper = Interval::scalar(0.0)?;
    let zero = Interval::scalar(0.0)?;
    let one = Interval::scalar(1.0)?;
    let bottom = Interval::scalar(lower_chord)?;
    for strip in 0..STRIPS {
        if work < SLAB_CHARGE {
            return None;
        }
        work -= SLAB_CHARGE;
        // All 512 equally spaced knots are exact dyadics and coincide with
        // knots of the 16,384-bin shared normalized-CDF table.
        let x0 = -3.0 + strip as f64 * (6.0 / STRIPS as f64);
        let x1 = -3.0 + (strip + 1) as f64 * (6.0 / STRIPS as f64);
        let normalized = Interval::new(x0, x1)?;
        let physical = cx.add(sigma_interval.mul(normalized)?)?;
        let curve_y = certified_graph_y_slab(controls, physical, &mut work)?;
        if curve_y.lo < lower_chord {
            // The graph is not shown to remain above the chord over this
            // entire slab; this proof would not apply to that topology.
            return None;
        }
        let x_mass = cdf.equal_strip_mass(strip, STRIPS, &mut work)?;
        let y_lower = cdf.interval_mass(
            bottom,
            Interval::scalar(curve_y.lo)?,
            center[1],
            sigma_interval,
            &mut work,
        )?;
        let y_upper = cdf.interval_mass(
            bottom,
            Interval::scalar(curve_y.hi)?,
            center[1],
            sigma_interval,
            &mut work,
        )?;
        lower = lower.add(Interval::scalar(x_mass.lo)?.mul(Interval::scalar(y_lower.lo)?)?)?;
        upper = upper.add(Interval::scalar(x_mass.hi)?.mul(Interval::scalar(y_upper.hi)?)?)?;
    }
    let lower_bound = lower.lo.max(zero.lo);
    let upper_bound = upper.hi.min(one.hi);
    if lower_bound > upper_bound {
        return None;
    }
    Some(GraphMassProof {
        interval: [lower_bound, upper_bound],
        work: work_limit - work,
        peak_bytes,
        strips: STRIPS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::composition_2d::{Render2dFillRule, Render2dPoint};

    fn authored_graph(points: &[[f64; 2]]) -> Render2dPath {
        let point = |value: [f64; 2]| Render2dPoint::new(value[0], value[1]).unwrap();
        let commands = match points {
            [a, b, c] => vec![
                Render2dPathCommand::MoveTo(point(*a)),
                Render2dPathCommand::QuadraticTo {
                    control: point(*b),
                    to: point(*c),
                },
                Render2dPathCommand::Close,
            ],
            [a, b, c, d] => vec![
                Render2dPathCommand::MoveTo(point(*a)),
                Render2dPathCommand::CubicTo {
                    control1: point(*b),
                    control2: point(*c),
                    to: point(*d),
                },
                Render2dPathCommand::Close,
            ],
            _ => panic!("test fixture expects one original authored F1 curve"),
        };
        Render2dPath::new(Render2dFillRule::NonZero, commands).unwrap()
    }

    #[test]
    fn quadratic_arch_certifies_strict_finite_gaussian_boundary_from_f1_controls() {
        // Independent exact polynomial x=10+40t, y=10+60t(1-t).
        // At (30,25), sigma=.25 and finite [-3,3]^2 kernel:
        // inside Y <= -3 X^2/320 => mass 0.496350400653573.
        let quadratic = authored_graph(&[[10.0, 10.0], [30.0, 40.0], [50.0, 10.0]]);
        let result = certify_monotone_graph_mass(
            &quadratic,
            [30.0, 25.0],
            0.25,
            MAX_SCENE_WORK,
            MAX_PROOF_BYTES,
        )
        .expect("shared 1D Gaussian CDF and exact Bézier enclosures must admit this test");
        assert!(result.interval[0] <= 0.496_350_400_653_573);
        assert!(result.interval[1] >= 0.496_350_400_653_573);
        assert!(result.interval[1] - result.interval[0] < MAX_WIDTH);
        assert_eq!(result.strips, STRIPS);
        assert!(result.work < MAX_SCENE_WORK);
        assert!(result.peak_bytes < MAX_PROOF_BYTES);
    }

    #[test]
    fn cubic_graph_certifies_own_center_mass_against_quadratic_equivalent() {
        // An exact degree elevation of the quadratic, not a polygonized
        // source. Its cubic control y=30, so the actual arch agrees.
        let elevated = authored_graph(&[
            [10.0, 10.0],
            [70.0 / 3.0, 30.0],
            [110.0 / 3.0, 30.0],
            [50.0, 10.0],
        ]);
        let result = certify_monotone_graph_mass(
            &elevated,
            [30.0, 25.0],
            0.25,
            MAX_SCENE_WORK,
            MAX_PROOF_BYTES,
        )
        .expect("degree-elevated source must be supported without faceting");
        assert!(result.interval[0] <= 0.496_350_400_653_573);
        assert!(result.interval[1] >= 0.496_350_400_653_573);
        assert!(result.interval[1] - result.interval[0] < MAX_WIDTH);
    }

    #[test]
    fn topology_uncertainty_work_and_retained_buffer_reject_before_alpha() {
        let quadratic = authored_graph(&[[10.0, 10.0], [30.0, 40.0], [50.0, 10.0]]);
        assert!(certify_monotone_graph_mass(
            &quadratic,
            [30.0, 25.0],
            0.25,
            100,
            MAX_PROOF_BYTES,
        )
        .is_none());
        assert!(
            certify_monotone_graph_mass(
                &quadratic,
                [30.0, 25.0],
                0.25,
                MAX_SCENE_WORK,
                1_024,
            )
            .is_none()
        );
        // The supported proof is only the horizontal-chord topology.
        assert!(
            certify_monotone_graph_mass(
                &authored_graph(&[[10.0, 10.0], [30.0, 0.0], [50.0, 10.0]]),
                [30.0, 10.0],
                0.25,
                MAX_SCENE_WORK,
                MAX_PROOF_BYTES,
            )
            .is_none()
        );
        // Adding a second structural edge changes the F1 region; the
        // restricted graph proof must not guess that its closure is equal.
        let extraneous_edge = Render2dPath::new(
            Render2dFillRule::NonZero,
            vec![
                Render2dPathCommand::MoveTo(Render2dPoint::new(10.0, 10.0).unwrap()),
                Render2dPathCommand::QuadraticTo {
                    control: Render2dPoint::new(30.0, 40.0).unwrap(),
                    to: Render2dPoint::new(50.0, 10.0).unwrap(),
                },
                Render2dPathCommand::LineTo(Render2dPoint::new(45.0, 9.0).unwrap()),
                Render2dPathCommand::Close,
            ],
        )
        .unwrap();
        assert!(
            certify_monotone_graph_mass(
                &extraneous_edge,
                [30.0, 25.0],
                0.25,
                MAX_SCENE_WORK,
                MAX_PROOF_BYTES,
            )
            .is_none()
        );
        // The caller cannot use an x-overhanging source as a certified
        // graph between this same pair of endpoints.
        assert!(
            certify_monotone_graph_mass(
                &authored_graph(&[[10.0, 10.0], [60.0, 40.0], [50.0, 10.0]]),
                [30.0, 25.0],
                0.25,
                MAX_SCENE_WORK,
                MAX_PROOF_BYTES,
            )
            .is_none()
        );
    }
}
