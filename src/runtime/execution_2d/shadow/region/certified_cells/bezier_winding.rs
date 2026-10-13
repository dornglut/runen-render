//! Test-only source-authoritative quadratic/cubic F1 winding certificates.
//!
//! The original Bézier control polygon is subdivided with outward-rounded
//! de Casteljau arithmetic. Convex-hull containment makes each child enclosure
//! rigorous, including under floating-point rounding. A whole query cell is
//! accepted only after EVERY curve is certified disjoint from that cell, and
//! its half-open horizontal-ray crossing is provably isolated. There is no
//! conversion to an approximate polyline, cached mask or signed-distance field.
//!
//! Undecidable boundary contacts, tangencies, interval overflow, depth and
//! aggregate-work exhaustion return Unknown, not fabricated coverage.
use super::{Cell, CellCertificate, Interval, certified_line_crossing};
use crate::composition_2d::{Render2dFillRule, Render2dPath, Render2dPathCommand, Render2dPoint};

const MAX_SUBDIVISION_DEPTH: usize = 12;
const WORK_PER_CURVE_NODE: usize = 128;

#[derive(Clone, Copy)]
struct Bezier {
    controls: [[Interval; 2]; 4],
    degree: usize,
}

impl Bezier {
    fn from_points(points: &[[f64; 2]]) -> Option<Self> {
        if !(3..=4).contains(&points.len()) {
            return None;
        }
        let zero = Interval::scalar(0.0)?;
        let mut controls = [[zero; 2]; 4];
        for (index, &[x, y]) in points.iter().enumerate() {
            controls[index] = [Interval::scalar(x)?, Interval::scalar(y)?];
        }
        Some(Self {
            controls,
            degree: points.len() - 1,
        })
    }

    fn enclosure(self) -> Option<Cell> {
        let mut x0 = f64::INFINITY;
        let mut x1 = f64::NEG_INFINITY;
        let mut y0 = f64::INFINITY;
        let mut y1 = f64::NEG_INFINITY;
        for &[x, y] in &self.controls[..=self.degree] {
            x0 = x0.min(x.lo);
            x1 = x1.max(x.hi);
            y0 = y0.min(y.lo);
            y1 = y1.max(y.hi);
        }
        Some(Cell {
            x: Interval::new(x0, x1)?,
            y: Interval::new(y0, y1)?,
        })
    }

    fn split(self) -> Option<(Self, Self)> {
        let zero = Interval::scalar(0.0)?;
        let half = Interval::scalar(0.5)?;
        let mut levels = [[[zero; 2]; 4]; 4];
        levels[0] = self.controls;
        for level in 1..=self.degree {
            for j in 0..=self.degree - level {
                let mut next = [zero; 2];
                for (axis, coordinate) in next.iter_mut().enumerate() {
                    *coordinate = levels[level - 1][j][axis]
                        .add(levels[level - 1][j + 1][axis])?
                        .mul(half)?;
                }
                levels[level][j] = next;
            }
        }
        let mut left = [[zero; 2]; 4];
        let mut right = [[zero; 2]; 4];
        for i in 0..=self.degree {
            left[i] = levels[i][0];
            right[i] = levels[self.degree - i][i];
        }
        Some((
            Self {
                controls: left,
                degree: self.degree,
            },
            Self {
                controls: right,
                degree: self.degree,
            },
        ))
    }

    fn strict_monotone_x(self) -> bool {
        self.controls[..=self.degree]
            .windows(2)
            .all(|pair| pair[0][0].hi < pair[1][0].lo)
    }

    fn graph_y_slab(
        self,
        x: Interval,
        work: &mut usize,
        depth: usize,
    ) -> Option<Option<Interval>> {
        if *work < WORK_PER_CURVE_NODE {
            *work = 0;
            return None;
        }
        *work -= WORK_PER_CURVE_NODE;
        let hull = self.enclosure()?;
        if hull.x.hi < x.lo || hull.x.lo > x.hi {
            return Some(None);
        }
        if depth == 18 {
            return Some(Some(hull.y));
        }
        let (left, right) = self.split()?;
        let a = left.graph_y_slab(x, work, depth + 1)?;
        let b = right.graph_y_slab(x, work, depth + 1)?;
        match (a, b) {
            (Some(a), Some(b)) => Some(Some(Interval::new(a.lo.min(b.lo), a.hi.max(b.hi))?)),
            (Some(a), None) | (None, Some(a)) => Some(Some(a)),
            (None, None) => Some(None),
        }
    }

    fn monotone_y(self) -> (bool, bool) {
        let mut ascending = true;
        let mut descending = true;
        for edge in self.controls[..=self.degree].windows(2) {
            ascending &= edge[0][1].hi <= edge[1][1].lo;
            descending &= edge[1][1].hi <= edge[0][1].lo;
        }
        (ascending, descending)
    }
}

fn curve_crossings(
    curve: Bezier,
    cell: Cell,
    remaining_work: &mut usize,
    depth: usize,
) -> Option<i64> {
    if *remaining_work < WORK_PER_CURVE_NODE {
        *remaining_work = 0;
        return None;
    }
    *remaining_work -= WORK_PER_CURVE_NODE;
    let bounds = curve.enclosure()?;
    // All points on the true polynomial are in its subdivided control hull.
    let separated = bounds.x.hi < cell.x.lo
        || bounds.x.lo > cell.x.hi
        || bounds.y.hi < cell.y.lo
        || bounds.y.lo > cell.y.hi;
    if separated {
        // A ray crossing wholly to the left or away from the ray height
        // has exactly zero effect on this cell's half-open winding.
        let x = cell.x.lo;
        let y = cell.y.lo;
        if bounds.x.hi < x || bounds.y.hi < y || bounds.y.lo > y {
            return Some(0);
        }
        let (ascending, descending) = curve.monotone_y();
        if bounds.x.lo > x && (ascending || descending) {
            let start = curve.controls[0][1];
            let end = curve.controls[curve.degree][1];
            if ascending && start.hi < y && end.lo > y {
                return Some(1);
            }
            if descending && start.lo > y && end.hi < y {
                return Some(-1);
            }
            if (start.lo > y && end.lo > y) || (start.hi < y && end.hi < y) {
                return Some(0);
            }
        }
    }
    if depth >= MAX_SUBDIVISION_DEPTH {
        return None;
    }
    let (left, right) = curve.split()?;
    let a = curve_crossings(left, cell, remaining_work, depth + 1)?;
    let b = curve_crossings(right, cell, remaining_work, depth + 1)?;
    a.checked_add(b)
}

/// Test-only vertical enclosure of an ORIGINAL authored monotone-x Bézier
/// graph over an entire x slab. Strictly increasing control x coordinates
/// prove one-valued source geometry; general folds/hairpins are not admitted.
/// The existing directed interval de Casteljau implementation is the owner.
pub(super) fn certified_graph_y_slab(
    points: &[[f64; 2]],
    x: Interval,
    work: &mut usize,
) -> Option<Interval> {
    let curve = Bezier::from_points(points)?;
    if !curve.strict_monotone_x() {
        return None;
    }
    let domain = curve.enclosure()?;
    if x.lo < domain.x.lo || x.hi > domain.x.hi {
        return None;
    }
    curve.graph_y_slab(x, work, 0)?
}

fn scalar_point(p: Render2dPoint) -> [f64; 2] {
    [p.x(), p.y()]
}

/// Called only after the parent has preadmitted the complete authored
/// command-class and contour traversal work. Every adaptive curve node
/// additionally spends from the SAME invocation-level query fuel.
pub(super) fn certify_curved_path(
    path: &Render2dPath,
    cell: Cell,
    visits: &mut usize,
) -> CellCertificate {
    let mut first: Option<[f64; 2]> = None;
    let mut previous: Option<[f64; 2]> = None;
    let mut has_segment = false;
    let mut winding = 0_i64;
    let mut add = |crossing: Option<i64>| -> Option<()> {
        winding = winding.checked_add(crossing?)?;
        Some(())
    };
    for command in path.commands() {
        match *command {
            Render2dPathCommand::MoveTo(p) => {
                if has_segment {
                    let (Some(a), Some(b)) = (previous, first) else {
                        return CellCertificate::Unknown;
                    };
                    if add(certified_line_crossing(cell, a, b)).is_none() {
                        return CellCertificate::Unknown;
                    }
                }
                first = Some(scalar_point(p));
                previous = first;
                has_segment = false;
            }
            Render2dPathCommand::LineTo(p) => {
                let Some(start) = previous else {
                    return CellCertificate::Unknown;
                };
                let end = scalar_point(p);
                if add(certified_line_crossing(cell, start, end)).is_none() {
                    return CellCertificate::Unknown;
                }
                previous = Some(end);
                has_segment = true;
            }
            Render2dPathCommand::QuadraticTo { control, to } => {
                let Some(start) = previous else {
                    return CellCertificate::Unknown;
                };
                let end = scalar_point(to);
                let Some(curve) = Bezier::from_points(&[start, scalar_point(control), end]) else {
                    return CellCertificate::Unknown;
                };
                if add(curve_crossings(curve, cell, visits, 0)).is_none() {
                    return CellCertificate::Unknown;
                }
                previous = Some(end);
                has_segment = true;
            }
            Render2dPathCommand::CubicTo {
                control1,
                control2,
                to,
            } => {
                let Some(start) = previous else {
                    return CellCertificate::Unknown;
                };
                let end = scalar_point(to);
                let Some(curve) = Bezier::from_points(&[
                    start,
                    scalar_point(control1),
                    scalar_point(control2),
                    end,
                ]) else {
                    return CellCertificate::Unknown;
                };
                if add(curve_crossings(curve, cell, visits, 0)).is_none() {
                    return CellCertificate::Unknown;
                }
                previous = Some(end);
                has_segment = true;
            }
            Render2dPathCommand::Close => {
                if has_segment {
                    let (Some(a), Some(b)) = (previous, first) else {
                        return CellCertificate::Unknown;
                    };
                    if add(certified_line_crossing(cell, a, b)).is_none() {
                        return CellCertificate::Unknown;
                    }
                }
                previous = first;
                has_segment = false;
            }
        }
    }
    if has_segment {
        let (Some(a), Some(b)) = (previous, first) else {
            return CellCertificate::Unknown;
        };
        if add(certified_line_crossing(cell, a, b)).is_none() {
            return CellCertificate::Unknown;
        }
    }
    let inside = match path.fill_rule() {
        Render2dFillRule::EvenOdd => winding % 2 != 0,
        Render2dFillRule::NonZero => winding != 0,
    };
    if inside {
        CellCertificate::Inside
    } else {
        CellCertificate::Outside
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::Render2dPoint;

    fn p(x: f64, y: f64) -> Render2dPoint {
        Render2dPoint::new(x, y).unwrap()
    }

    fn query(path: &Render2dPath, bounds: [f64; 4]) -> CellCertificate {
        let mut work = 16_777_216;
        super::super::certify_path(path, Cell::new(bounds).unwrap(), &mut work)
    }

    #[test]
    fn quadratic_and_cubic_archs_are_classified_from_polynomials_not_facets() {
        for curve in [
            Render2dPathCommand::QuadraticTo {
                control: p(30.0, 40.0),
                to: p(50.0, 10.0),
            },
            Render2dPathCommand::CubicTo {
                control1: p(10.0, 40.0),
                control2: p(50.0, 40.0),
                to: p(50.0, 10.0),
            },
        ] {
            let path = Render2dPath::new(
                Render2dFillRule::NonZero,
                vec![
                    Render2dPathCommand::MoveTo(p(10.0, 10.0)),
                    curve,
                    Render2dPathCommand::Close,
                ],
            )
            .unwrap();
            assert_eq!(
                query(&path, [25.0, 17.0, 27.0, 19.0]),
                CellCertificate::Inside
            );
            assert_eq!(
                query(&path, [25.0, 34.0, 27.0, 36.0]),
                CellCertificate::Outside
            );
            assert_eq!(
                query(&path, [60.0, 15.0, 62.0, 17.0]),
                CellCertificate::Outside
            );
        }
    }

    #[test]
    fn quadratic_and_cubic_winding_respects_same_and_opposed_hole_direction() {
        for curve in [
            Render2dPathCommand::QuadraticTo {
                control: p(32.0, 48.0),
                to: p(16.0, 16.0),
            },
            Render2dPathCommand::CubicTo {
                control1: p(48.0, 48.0),
                control2: p(16.0, 48.0),
                to: p(16.0, 16.0),
            },
        ] {
            let base = vec![
                Render2dPathCommand::MoveTo(p(0.0, 0.0)),
                Render2dPathCommand::LineTo(p(64.0, 0.0)),
                Render2dPathCommand::LineTo(p(64.0, 64.0)),
                Render2dPathCommand::LineTo(p(0.0, 64.0)),
                Render2dPathCommand::Close,
                Render2dPathCommand::MoveTo(p(48.0, 16.0)),
                curve,
                Render2dPathCommand::Close,
            ];
            let even = Render2dPath::new(Render2dFillRule::EvenOdd, base.clone()).unwrap();
            let nonzero = Render2dPath::new(Render2dFillRule::NonZero, base).unwrap();
            assert_eq!(
                query(&even, [30.0, 20.0, 32.0, 22.0]),
                CellCertificate::Outside
            );
            assert_eq!(
                query(&nonzero, [30.0, 20.0, 32.0, 22.0]),
                CellCertificate::Inside
            );
            assert_eq!(query(&even, [3.0, 3.0, 5.0, 5.0]), CellCertificate::Inside);
        }
    }

    #[test]
    fn authored_cubic_64x64_finite_gaussian_tiles_reuse_interval_proofs() {
        // Original F1 cubic contour; the shared test fixture deliberately
        // supplies a bogus full-canvas physical tessellation. All 65,536
        // correlated sample centers are either certified or remain Unknown.
        // The same unchanged whole-scene query and work caps apply.
        super::super::shared_authored_path_scene_certificate(vec![
            Render2dPathCommand::MoveTo(p(8.0, 8.0)),
            Render2dPathCommand::LineTo(p(50.0, 8.0)),
            Render2dPathCommand::CubicTo {
                control1: p(50.0, 45.0),
                control2: p(8.0, 45.0),
                to: p(8.0, 8.0),
            },
            Render2dPathCommand::Close,
        ]);
    }

    #[test]
    fn subcurve_near_boundary_and_budget_exhaustion_are_unknown() {
        let path = Render2dPath::new(
            Render2dFillRule::EvenOdd,
            vec![
                Render2dPathCommand::MoveTo(p(10.0, 10.0)),
                Render2dPathCommand::CubicTo {
                    control1: p(10.0, 40.0),
                    control2: p(50.0, 40.0),
                    to: p(50.0, 10.0),
                },
                Render2dPathCommand::Close,
            ],
        )
        .unwrap();
        assert_eq!(
            query(&path, [29.0, 32.4, 31.0, 32.6]),
            CellCertificate::Unknown
        );
        let mut fuel = path.commands().len() * 40;
        assert_eq!(
            super::super::certify_path(
                &path,
                Cell::new([25.0, 17.0, 27.0, 19.0]).unwrap(),
                &mut fuel,
            ),
            CellCertificate::Unknown,
        );
        assert_eq!(fuel, 0);
    }
}
