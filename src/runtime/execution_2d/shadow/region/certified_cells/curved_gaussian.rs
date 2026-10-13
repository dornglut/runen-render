//! Test-only certified source-authoritative Gaussian integration for an
//! original F1 x-monotone quadratic/cubic graph with a closing chord.
//!
//! A single table of normalized 1D Gaussian bin masses is shared across
//! all correlated samples; exact-Bézier slab enclosures are shared across
//! consecutive samples with the SAME x center. A y-dependent CDF interval
//! is computed only once per slab/site, without assuming geometric alpha,
//! masks, facets, SDFs or a new renderer owner.
//!
//! Restriction: precisely one curve + horizontal closing chord, no clips,
//! spreads, nested effects, nonmonotone x, arbitrary contour or transforms.
//! This is test-only feasibility, not an admitted F3F production evaluator.
use super::Interval;
use super::axis_gaussian::{BINS, MAX_PROOF_BYTES, MAX_SCENE_WORK, NormalizedGaussianCdf};
use super::bezier_winding::certified_graph_y_slab;
use crate::composition_2d::{Render2dPath, Render2dPathCommand};

const STRIPS: usize = 512;
const SLAB_CHARGE: usize = 64;
const MAX_WIDTH: f64 = 0.005;
const MAX_BATCH_SITES: usize = 256;

#[derive(Debug)]
struct GraphMassProof {
    interval: [f64; 2],
    work: usize,
    peak_bytes: usize,
    strips: usize,
}

#[derive(Debug)]
struct BatchedGraphMassProof {
    intervals: Vec<[f64; 2]>,
    work: usize,
    peak_bytes: usize,
    columns: usize,
}

fn certify_monotone_graph_mass(
    path: &Render2dPath,
    center: [f64; 2],
    sigma: f64,
    work_limit: usize,
    byte_limit: usize,
) -> Option<GraphMassProof> {
    let mut batch =
        certify_monotone_graph_batch(path, &[center], sigma, work_limit, byte_limit)?;
    Some(GraphMassProof {
        interval: batch.intervals.pop()?,
        work: batch.work,
        peak_bytes: batch.peak_bytes,
        strips: STRIPS,
    })
}

/// One SHARED Gaussian CDF for all sites, one borrowed exact F1 command
/// traversal and one source-geometry slab pass per consecutive x-column.
/// Outputs remain atomic: exhausting any work, error or capacity bound
/// rejects the entire batch rather than exposing a partial sample plane.
/// Sites may be supplied in any order; disjoint repetitions of an x-column
/// are conservatively recomputed and charged rather than silently cached.
fn certify_monotone_graph_batch(
    path: &Render2dPath,
    centers: &[[f64; 2]],
    sigma: f64,
    work_limit: usize,
    byte_limit: usize,
) -> Option<BatchedGraphMassProof> {
    if centers.is_empty()
        || centers.len() > MAX_BATCH_SITES
        || !centers.iter().flat_map(|p| p.iter()).all(|v| v.is_finite())
        || !sigma.is_finite()
        || sigma <= 0.0
        || work_limit > MAX_SCENE_WORK
        || byte_limit > MAX_PROOF_BYTES
    {
        return None;
    }
    // Only ORIGINAL immutable F1 commands may establish the graph topology.
    let commands = path.commands();
    let command_cost = commands.len().checked_mul(40)?;
    if command_cost > work_limit {
        return None;
    }
    let point = |p: &crate::composition_2d::Render2dPoint| [p.x(), p.y()];
    let mut control_buffer = [[0.0; 2]; 4];
    let size = match commands {
        [
            Render2dPathCommand::MoveTo(a),
            Render2dPathCommand::QuadraticTo { control, to },
            Render2dPathCommand::Close,
        ] => {
            control_buffer[0] = point(a);
            control_buffer[1] = point(control);
            control_buffer[2] = point(to);
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
            control_buffer[0] = point(a);
            control_buffer[1] = point(control1);
            control_buffer[2] = point(control2);
            control_buffer[3] = point(to);
            4
        }
        _ => return None,
    };
    let controls = &control_buffer[..size];
    let lower_chord = controls[0][1];
    if controls[size - 1][1] != lower_chord {
        return None;
    }

    // Preflight *minimum* table+bin weights+slab enclosures+output capacity
    // BEFORE any Vec allocation. Check actual Vec capacities after reserve.
    let interval_bytes = std::mem::size_of::<Interval>();
    let minimum_intervals = BINS
        .checked_mul(2)?
        .checked_add(1)?
        .checked_add(STRIPS.checked_mul(2)?)?
        .checked_add(centers.len())?;
    if minimum_intervals.checked_mul(interval_bytes)? > byte_limit {
        return None;
    }
    let mut work = work_limit - command_cost;
    let cdf = NormalizedGaussianCdf::new(&mut work)?;
    let mut x_weights = Vec::new();
    x_weights.try_reserve_exact(STRIPS).ok()?;
    for strip in 0..STRIPS {
        x_weights.push(cdf.equal_strip_mass(strip, STRIPS, &mut work)?);
    }
    let mut curve_slabs = Vec::new();
    curve_slabs.try_reserve_exact(STRIPS).ok()?;
    let mut intervals = Vec::new();
    intervals.try_reserve_exact(centers.len()).ok()?;
    let peak_bytes = cdf.peak_buffer_bytes()?.checked_add(
        x_weights
            .capacity()
            .checked_add(curve_slabs.capacity())?
            .checked_add(intervals.capacity())?
            .checked_mul(interval_bytes)?,
    )?;
    if peak_bytes > byte_limit {
        return None;
    }

    let sigma_interval = Interval::scalar(sigma)?;
    let kernel = sigma_interval.mul(Interval::new(-3.0, 3.0)?)?;
    let bottom = Interval::scalar(lower_chord)?;
    let mut last_x = None;
    let mut columns = 0_usize;
    for &[x, y] in centers {
        let cx = Interval::scalar(x)?;
        if last_x != Some(x) {
            columns = columns.checked_add(1)?;
            last_x = Some(x);
            let support = cx.add(kernel)?;
            // Endpoint caps are not covered by this one-graph proof.
            if support.lo <= controls[0][0]
                || support.hi >= controls[size - 1][0]
            {
                return None;
            }
            curve_slabs.clear();
            for strip in 0..STRIPS {
                let x0 = -3.0 + strip as f64 * (6.0 / STRIPS as f64);
                let x1 = -3.0 + (strip + 1) as f64 * (6.0 / STRIPS as f64);
                let normalized = Interval::new(x0, x1)?;
                let physical = cx.add(sigma_interval.mul(normalized)?)?;
                let curve_y = certified_graph_y_slab(controls, physical, &mut work)?;
                if curve_y.lo < lower_chord {
                    return None;
                }
                curve_slabs.push(curve_y);
            }
        }
        let mut mass = Interval::scalar(0.0)?;
        for (x_mass, &curve_y) in x_weights.iter().zip(&curve_slabs) {
            if work < SLAB_CHARGE {
                return None;
            }
            work -= SLAB_CHARGE;
            // The single interval-end CDF call rigorously encloses both
            // curve-y extrema: two separate calls are unnecessary.
            let y_mass =
                cdf.interval_mass(bottom, curve_y, y, sigma_interval, &mut work)?;
            mass = mass.add(x_mass.mul(y_mass)?)?;
        }
        let result = [mass.lo.max(0.0), mass.hi.min(1.0)];
        if result[0] > result[1] || result[1] - result[0] >= MAX_WIDTH {
            return None;
        }
        intervals.push(result);
    }
    Some(BatchedGraphMassProof {
        intervals,
        work: work_limit - work,
        peak_bytes,
        columns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use super::super::axis_gaussian::{AXIS_WORK, BIN_WORK, BINS};
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
    fn independent_f3e_boundary_site_count_proves_per_site_slabs_cannot_admit_64x64() {
        // Exact authored F1 quadratic: x=10+40t, y=10+60t(1-t).
        // Every one of 80 *different* 4x4 globally phased x-columns
        // between x=20 and x=40 has a y-site within 1/8 of the
        // polynomial boundary. The finite sigma=.25 kernel extends
        // 3*sigma=.75 on each axis, hence truly intersects BOTH sides
        // of the boundary at those sites (and never the lower chord).
        // These sites are mixed, not full-kernel 0/1 certificates.
        let path = authored_graph(&[[10.0, 10.0], [30.0, 40.0], [50.0, 10.0]]);
        assert_eq!(path.commands().len(), 3);
        let mut mixed_sites = 0_usize;
        for x_index in (20 * 4)..(40 * 4) {
            let x = (x_index as f64 + 0.5) / 4.0;
            let curve_y = 25.0 - (3.0 / 80.0) * (x - 30.0) * (x - 30.0);
            let y_index = (curve_y * 4.0 - 0.5).round();
            assert!((0.0..256.0).contains(&y_index));
            let center_y = (y_index + 0.5) / 4.0;
            assert!((center_y - curve_y).abs() <= 0.125);
            assert!(center_y - 0.75 > 10.0);
            mixed_sites += 1;
        }
        assert_eq!(mixed_sites, 80);
        // PROOF OF NO-GO for the current PER-SAMPLE 512-slab plan:
        // Give it the strongest possible reuse of its expensive 1D CDF
        // by charging its 16,384-bin construction only ONCE per scene.
        // Also make every x-strip's Gaussian weight FREE, make all
        // geometry subdivision FREE, and ignore command/work/node costs.
        // Even then, each mixed site inevitably pays 512 slab charges
        // and two vertical CDF lookups per strip in THIS ALGORITHM.
        // Thus an optimistic lower bound already breaches the existing
        // 16,777,216 owner-scene model budget. This does NOT exclude a
        // different batched/row-shared algorithm.
        let shared_cdf = BINS.checked_mul(BIN_WORK).unwrap();
        let per_site = STRIPS.checked_mul(SLAB_CHARGE + 2 * AXIS_WORK).unwrap();
        let optimistic_minimum = shared_cdf
            .checked_add(mixed_sites.checked_mul(per_site).unwrap())
            .unwrap();
        assert_eq!(shared_cdf, 8_388_608);
        assert_eq!(per_site, 163_840);
        assert_eq!(optimistic_minimum, 21_495_808);
        assert!(
            optimistic_minimum > MAX_SCENE_WORK,
            "per-site slab integration cannot meet the mandatory 64x64 admission even with CDF and x-weights freely shared"
        );
    }

    #[test]
    fn one_x_column_reuses_exact_f1_slab_geometry_for_distinct_f3e_y_phases() {
        // Two distinct globally phased 4x4 sample sites in the SAME x
        // column. Both straddle the arch, but each has a different
        // continuous Gaussian mass. A single source-CDF and Bézier slab
        // traversal must cover both under one unchanged owner budget.
        let source = authored_graph(&[[10.0, 10.0], [30.0, 40.0], [50.0, 10.0]]);
        let centers = [[30.125, 25.125], [30.125, 24.875]];
        let batch = certify_monotone_graph_batch(
            &source,
            &centers,
            0.25,
            MAX_SCENE_WORK,
            MAX_PROOF_BYTES,
        )
        .expect("one x-column must share authored F1 curve and normalized CDF");
        let independent_masses = [0.304_003_920_263_388, 0.687_900_311_266_600];
        assert_eq!(batch.columns, 1);
        assert_eq!(batch.intervals.len(), centers.len());
        for (interval, reference) in batch.intervals.iter().zip(independent_masses) {
            assert!(interval[0] <= reference && reference <= interval[1]);
            assert!(interval[1] - interval[0] < MAX_WIDTH);
        }
        assert!(batch.work <= MAX_SCENE_WORK);
        assert!(batch.peak_bytes < MAX_PROOF_BYTES);

        // No response is possible from a partial prepared sample plane.
        assert!(
            certify_monotone_graph_batch(
                &source,
                &centers,
                0.25,
                100,
                MAX_PROOF_BYTES,
            )
            .is_none()
        );
        assert!(
            certify_monotone_graph_batch(
                &source,
                &centers,
                0.25,
                MAX_SCENE_WORK,
                1_024,
            )
            .is_none()
        );
    }

    #[test]
    fn topology_uncertainty_work_and_retained_buffer_reject_before_alpha() {
        let quadratic = authored_graph(&[[10.0, 10.0], [30.0, 40.0], [50.0, 10.0]]);
        assert!(
            certify_monotone_graph_mass(&quadratic, [30.0, 25.0], 0.25, 100, MAX_PROOF_BYTES,)
                .is_none()
        );
        assert!(
            certify_monotone_graph_mass(&quadratic, [30.0, 25.0], 0.25, MAX_SCENE_WORK, 1_024,)
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
