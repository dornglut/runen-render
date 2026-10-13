//! Feasibility only: exact-original-F1 quadratic with affine x(t).
//!
//! This is a narrow algebraic law, not physical tessellation, sampled source,
//! SDF or a production Gaussian pipeline. Whole finite 3-sigma normalized
//! integration is still bounded by the shared axis_gaussian CDF.
use super::Interval;
use super::axis_gaussian::{COMPACT_BINS, CompactGaussianCdf, MAX_PROOF_BYTES, MAX_SCENE_WORK};
use crate::composition_2d::{Render2dPath, Render2dPathCommand};

const STRIPS: usize = 512;
const GEOMETRY_WORK: usize = 96;
const SLAB_WORK: usize = 64;
const ERROR: f64 = 0.005;

struct LinearXCurve {
    x0: Interval,
    dx: Interval,
    a: Interval,
    b: Interval,
    c: Interval,
}

impl LinearXCurve {
    fn source(path: &Render2dPath, fuel: &mut usize) -> Option<Self> {
        let commands = path.commands();
        let charge = commands.len().checked_mul(40)?;
        if *fuel < charge {
            return None;
        }
        *fuel -= charge;
        let [
            Render2dPathCommand::MoveTo(start),
            Render2dPathCommand::QuadraticTo { control, to },
            Render2dPathCommand::Close,
        ] = commands
        else {
            return None;
        };
        let [x0, x1, x2] = [start.x(), control.x(), to.x()];
        // Integral binary64 coordinates up to 2^40 have exact 2x and sums.
        // Hence this proves the ORIGINAL quadratic's x(t) is affine,
        // without assuming rounded equality or replacing its F1 source.
        if ![x0, x1, x2]
            .into_iter()
            .all(|x| x.is_finite() && x.fract() == 0.0 && x.abs() <= (1_u64 << 40) as f64)
            || x2 <= x0
            || 2.0 * x1 != x0 + x2
            || start.y() != to.y()
        {
            return None;
        }
        let two = Interval::scalar(2.0)?;
        let a = Interval::scalar(start.y())?;
        let yc = Interval::scalar(control.y())?;
        let end = Interval::scalar(to.y())?;
        Some(Self {
            x0: Interval::scalar(x0)?,
            dx: Interval::scalar(x2)?.sub(Interval::scalar(x0)?)?,
            a,
            b: two.mul(yc.sub(a)?)?,
            c: a.sub(two.mul(yc)?)?.add(end)?,
        })
    }

    fn y_slab(&self, x: Interval, fuel: &mut usize) -> Option<Interval> {
        if *fuel < GEOMETRY_WORK {
            return None;
        }
        *fuel -= GEOMETRY_WORK;
        let t = x.sub(self.x0)?.div(self.dx)?;
        if t.lo < 0.0 || t.hi > 1.0 {
            return None;
        }
        // Mean-value theorem enclosure for EXACT polynomial
        // y(t)=a+b*t+c*t²: y(mid)+(b+2*c*[tlo,thi])*([tlo,thi]-mid).
        let mid = Interval::scalar(t.lo * 0.5 + t.hi * 0.5)?;
        let ymid = self
            .a
            .add(self.b.mul(mid)?)?
            .add(self.c.mul(mid.mul(mid)?)?)?;
        let slope = self.b.add(Interval::scalar(2.0)?.mul(self.c)?.mul(t)?)?;
        ymid.add(slope.mul(t.sub(mid)?)?)
    }
}

struct CompactSceneProof {
    intervals: Vec<[f64; 2]>,
    max_width: f64,
    spent: usize,
    bytes: usize,
}

fn certify_compact_scene(
    path: &Render2dPath,
    centers: &[[f64; 2]],
    sigma: f64,
    budget: usize,
    byte_limit: usize,
) -> Option<CompactSceneProof> {
    if centers.is_empty()
        || centers.len() > 256
        || !centers.iter().all(|p| p.iter().all(|n| n.is_finite()))
        || !sigma.is_finite()
        || sigma <= 0.0
        || budget > MAX_SCENE_WORK
        || byte_limit > MAX_PROOF_BYTES
    {
        return None;
    }
    let mut fuel = budget;
    let curve = LinearXCurve::source(path, &mut fuel)?;
    let required = (2 * COMPACT_BINS + 1 + STRIPS + centers.len())
        .checked_mul(std::mem::size_of::<Interval>())?;
    if required > byte_limit {
        return None;
    }
    let table = CompactGaussianCdf::new(&mut fuel)?;
    let mut weights = Vec::new();
    weights.try_reserve_exact(STRIPS).ok()?;
    for index in 0..STRIPS {
        weights.push(table.equal_strip_mass(index, STRIPS, &mut fuel)?);
    }
    let mut intervals = Vec::new();
    intervals.try_reserve_exact(centers.len()).ok()?;
    let bytes = table.peak_buffer_bytes()?.checked_add(
        weights
            .capacity()
            .checked_add(intervals.capacity())?
            .checked_mul(std::mem::size_of::<Interval>())?,
    )?;
    if bytes > byte_limit {
        return None;
    }
    let ss = Interval::scalar(sigma)?;
    let halo = ss.mul(Interval::new(-3.0, 3.0)?)?;
    let max_x = curve.x0.add(curve.dx)?;
    let mut max_width = 0.0_f64;
    for &[x, y] in centers {
        let center = Interval::scalar(x)?;
        let support = center.add(halo)?;
        if support.lo <= curve.x0.hi || support.hi >= max_x.lo {
            return None;
        }
        let mut mass = Interval::scalar(0.0)?;
        for (i, xweight) in weights.iter().enumerate() {
            if fuel < SLAB_WORK {
                return None;
            }
            fuel -= SLAB_WORK;
            let lo = -3.0 + i as f64 * (6.0 / STRIPS as f64);
            let hi = -3.0 + (i + 1) as f64 * (6.0 / STRIPS as f64);
            let physical = center.add(ss.mul(Interval::new(lo, hi)?)?)?;
            let upper_y = curve.y_slab(physical, &mut fuel)?;
            if upper_y.lo < curve.a.hi {
                return None;
            }
            let yweight = table.interval_mass(curve.a, upper_y, y, ss, &mut fuel)?;
            mass = mass.add(xweight.mul(yweight)?)?;
        }
        let result = [mass.lo.max(0.0), mass.hi.min(1.0)];
        let width = result[1] - result[0];
        if width < 0.0 || width >= ERROR {
            return None;
        }
        max_width = max_width.max(width);
        intervals.push(result);
    }
    Some(CompactSceneProof {
        intervals,
        max_width,
        spent: budget - fuel,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::axis_gaussian::{AXIS_WORK, BIN_WORK, STRIP_BIN_WORK};
    use crate::composition_2d::{Render2dFillRule, Render2dPoint};

    fn source(x_control: f64) -> Render2dPath {
        let p = |x, y| Render2dPoint::new(x, y).unwrap();
        Render2dPath::new(
            Render2dFillRule::NonZero,
            vec![
                Render2dPathCommand::MoveTo(p(10.0, 10.0)),
                Render2dPathCommand::QuadraticTo {
                    control: p(x_control, 40.0),
                    to: p(50.0, 10.0),
                },
                Render2dPathCommand::Close,
            ],
        )
        .unwrap()
    }

    #[test]
    fn exact_original_quadratic_80_independent_x_sites_share_compact_cdf() {
        let mut sites = Vec::new();
        for i in 80..160 {
            let x = (i as f64 + 0.5) / 4.0;
            let actual_y = 25.0 - (3.0 / 80.0) * (x - 30.0) * (x - 30.0);
            let phased = ((4.0 * actual_y - 0.5).round() + 0.5) / 4.0;
            assert!((phased - actual_y).abs() <= 0.125);
            sites.push([x, phased]);
        }
        let cert =
            certify_compact_scene(&source(30.0), &sites, 0.25, MAX_SCENE_WORK, MAX_PROOF_BYTES)
                .expect("all 80 original F1 mixed sites must admit exact compact proof");
        assert_eq!(cert.intervals.len(), 80);
        let expected_work = source(30.0).commands().len() * 40
            + COMPACT_BINS * BIN_WORK
            + COMPACT_BINS * STRIP_BIN_WORK
            + sites.len() * STRIPS * (GEOMETRY_WORK + SLAB_WORK + AXIS_WORK);
        assert_eq!(expected_work, 16_056_440);
        assert_eq!(cert.spent, expected_work);
        assert!(cert.spent < MAX_SCENE_WORK);
        assert!(cert.bytes < MAX_PROOF_BYTES);
        assert!(cert.max_width < ERROR);
        // Independent exact continuous quadratic Gaussian reference,
        // precomputed outside the renderer from the authored polynomial
        // y(x)=25-3*(x-30)^2/80 and finite normalized [-3,3]^2 law.
        // This site is one of the 80 distinct F3E 4x4-phase columns.
        assert_eq!(sites[40], [30.125, 24.875]);
        let exact_mass = 0.687_900_311_266_600_f64;
        assert!(cert.intervals[40][0] <= exact_mass);
        assert!(exact_mass <= cert.intervals[40][1]);
        assert!(cert.intervals[0][0] <= cert.intervals[79][1]);
        assert!(cert.intervals[79][0] <= cert.intervals[0][1]);
    }

    #[test]
    fn source_integer_linear_x_and_budget_reject_fail_closed() {
        let original = source(30.0);
        assert!(
            certify_compact_scene(&original, &[[30.125, 25.125]], 0.25, 100, MAX_PROOF_BYTES)
                .is_none()
        );
        assert!(
            certify_compact_scene(&original, &[[30.125, 25.125]], 0.25, MAX_SCENE_WORK, 128)
                .is_none()
        );
        assert!(
            certify_compact_scene(
                &source(30.25),
                &[[30.125, 25.125]],
                0.25,
                MAX_SCENE_WORK,
                MAX_PROOF_BYTES
            )
            .is_none()
        );
    }
}
