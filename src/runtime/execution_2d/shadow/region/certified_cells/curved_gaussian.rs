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
use super::axis_gaussian::{MAX_PROOF_BYTES, MAX_SCENE_WORK, NormalizedGaussianCdf};
use super::bezier_winding::certified_graph_y_slab;
use super::Interval;

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
    controls: &[[f64; 2]],
    lower_chord: f64,
    center: [f64; 2],
    sigma: f64,
    work_limit: usize,
    byte_limit: usize,
) -> Option<GraphMassProof> {
    if !(3..=4).contains(&controls.len())
        || !lower_chord.is_finite()
        || !center.into_iter().all(f64::is_finite)
        || !sigma.is_finite()
        || sigma <= 0.0
        || work_limit > MAX_SCENE_WORK
        || byte_limit > MAX_PROOF_BYTES
        || controls[0][1] != lower_chord
        || controls[controls.len() - 1][1] != lower_chord
    {
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
    let mut work = work_limit;
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
        let y_lower = cdf.interval_mass(bottom, Interval::scalar(curve_y.lo)?, center[1], sigma_interval, &mut work)?;
        let y_upper = cdf.interval_mass(bottom, Interval::scalar(curve_y.hi)?, center[1], sigma_interval, &mut work)?;
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

    #[test]
    fn quadratic_arch_certifies_strict_finite_gaussian_boundary_from_f1_controls() {
        // Independent exact polynomial x=10+40t, y=10+60t(1-t).
        // At (30,25), sigma=.25 and finite [-3,3]^2 kernel:
        // inside Y <= -3 X^2/320 => mass 0.496350400653573.
        let quadratic = [[10.0, 10.0], [30.0, 40.0], [50.0, 10.0]];
        let result = certify_monotone_graph_mass(
            &quadratic,
            10.0,
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
        let elevated = [
            [10.0, 10.0],
            [70.0 / 3.0, 30.0],
            [110.0 / 3.0, 30.0],
            [50.0, 10.0],
        ];
        let result = certify_monotone_graph_mass(
            &elevated,
            10.0,
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
        let quadratic = [[10.0, 10.0], [30.0, 40.0], [50.0, 10.0]];
        assert!(certify_monotone_graph_mass(
            &quadratic,
            10.0,
            [30.0, 25.0],
            0.25,
            100,
            MAX_PROOF_BYTES,
        )
        .is_none());
        assert!(certify_monotone_graph_mass(
            &quadratic,
            10.0,
            [30.0, 25.0],
            0.25,
            MAX_SCENE_WORK,
            1_024,
        )
        .is_none());
        // The supported proof is only the horizontal-chord topology.
        assert!(certify_monotone_graph_mass(
            &[[10.0, 10.0], [30.0, 0.0], [50.0, 10.0]],
            10.0,
            [30.0, 10.0],
            0.25,
            MAX_SCENE_WORK,
            MAX_PROOF_BYTES,
        )
        .is_none());
        // The caller cannot use an x-overhanging source as a certified
        // graph between this same pair of endpoints.
        assert!(certify_monotone_graph_mass(
            &[[10.0, 10.0], [60.0, 40.0], [50.0, 10.0]],
            10.0,
            [30.0, 25.0],
            0.25,
            MAX_SCENE_WORK,
            MAX_PROOF_BYTES,
        )
        .is_none());
    }
}
