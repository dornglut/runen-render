//! Test-only feasibility of conservative cell classification over the actual
//! F1 neutral-region DAG. This does NOT admit a new production shadow class.
//!
//! Only synthetic physical rectangles may certify source geometry without an F1 item.
//! Authored quadratic/cubic curves use test-only outward-rounded subdivision.
//! A physical triangle inventory never certifies exact F1 source geometry.
//! A production evaluator still needs source, morphology and coverage proofs.
//! A certificate concerns SUPPORT membership, not alpha or Gaussian weights.

mod axis_gaussian;
mod bezier_winding;
mod curved_gaussian;
mod linear_x_gaussian;

use super::interval::Interval;
use super::*;
use crate::composition_2d::{
    Render2dClip, Render2dItem, Render2dPath, Render2dPathCommand, Render2dPrimitive, Render2dRect,
    Render2dShape,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CellCertificate {
    Inside,
    Outside,
    Unknown,
}

// All calculated bounds are outward-rounded. Any arithmetic uncertainty,
// nonfinite result or singular transform yields Unknown, never fake geometry.

#[derive(Clone, Copy)]
struct Cell {
    x: Interval,
    y: Interval,
}

impl Cell {
    fn new([x0, y0, x1, y1]: [f64; 4]) -> Option<Self> {
        Some(Self {
            x: Interval::new(x0, x1)?,
            y: Interval::new(y0, y1)?,
        })
    }

    fn shift(self, x: f64, y: f64) -> Option<Self> {
        Some(Self {
            x: self.x.sub(Interval::scalar(x)?)?,
            y: self.y.sub(Interval::scalar(y)?)?,
        })
    }

    fn expanded(self, margin: f64) -> Option<Self> {
        let r = Interval::scalar(margin)?;
        if margin < 0.0 {
            return None;
        }
        Some(Self {
            x: Interval::new(self.x.lo, self.x.hi)?.add(Interval::new(-r.hi, r.hi)?)?,
            y: Interval::new(self.y.lo, self.y.hi)?.add(Interval::new(-r.hi, r.hi)?)?,
        })
    }

    fn preimage(self, affine: Render2dAffineTransform) -> Option<Self> {
        let [a, b, c, d, tx, ty] = affine.components().map(Interval::scalar);
        let (a, b, c, d, tx, ty) = (a?, b?, c?, d?, tx?, ty?);
        let determinant = a.mul(d)?.sub(b.mul(c)?)?;
        let x = self.x.sub(tx)?;
        let y = self.y.sub(ty)?;
        Some(Self {
            x: d.mul(x)?.sub(c.mul(y)?)?.div(determinant)?,
            y: a.mul(y)?.sub(b.mul(x)?)?.div(determinant)?,
        })
    }

    fn classify_rectangle(self, [x0, y0, x1, y1]: [Interval; 4]) -> CellCertificate {
        // A physical vertex divided by its source scale may round in either
        // direction. A definite answer must account for BOTH possibilities.
        if self.x.lo > x0.hi && self.x.hi < x1.lo && self.y.lo > y0.hi && self.y.hi < y1.lo {
            CellCertificate::Inside
        } else if self.x.hi < x0.lo || self.x.lo > x1.hi || self.y.hi < y0.lo || self.y.lo > y1.hi {
            CellCertificate::Outside
        } else {
            CellCertificate::Unknown
        }
    }

    fn classify_ellipse(self, rect: Render2dRect) -> CellCertificate {
        if rect.is_empty() {
            return CellCertificate::Outside;
        }
        // Normalized ellipse equation, without approximating the curve by
        // an inscribed polygon or by its enclosing rectangle.
        let normalized = |coordinate: Interval, origin: f64, extent: f64| {
            coordinate
                .sub(Interval::scalar(origin)?)?
                .div(Interval::scalar(extent)?)?
                .mul(Interval::scalar(2.0)?)?
                .sub(Interval::scalar(1.0)?)
        };
        let Some(x) = normalized(self.x, rect.x(), rect.width()) else {
            return CellCertificate::Unknown;
        };
        let Some(y) = normalized(self.y, rect.y(), rect.height()) else {
            return CellCertificate::Unknown;
        };
        let Some(square_sum) = squared_interval(x).and_then(|xx| xx.add(squared_interval(y)?))
        else {
            return CellCertificate::Unknown;
        };
        if square_sum.hi < 1.0 {
            CellCertificate::Inside
        } else if square_sum.lo > 1.0 {
            CellCertificate::Outside
        } else {
            CellCertificate::Unknown
        }
    }
}

/// Square an interval without the sign-induced underestimation of a naive
/// interval product. The result still encloses every possible square.
fn squared_interval(v: Interval) -> Option<Interval> {
    let a = v.lo * v.lo;
    let b = v.hi * v.hi;
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let lo = if v.lo <= 0.0 && v.hi >= 0.0 {
        0.0
    } else {
        a.min(b).next_down().max(0.0)
    };
    Interval::new(lo, a.max(b).next_up())
}

/// The sign of a straight edge relative to an interval-enclosed cell is
/// certified from the ORIGINAL authored endpoints, never Lyon's triangles.
fn line_side(a: [f64; 2], b: [f64; 2], x: Interval, y: Interval) -> Option<Interval> {
    let dx = Interval::scalar(b[0])?.sub(Interval::scalar(a[0])?)?;
    let dy = Interval::scalar(b[1])?.sub(Interval::scalar(a[1])?)?;
    let px = x.sub(Interval::scalar(a[0])?)?;
    let py = y.sub(Interval::scalar(a[1])?)?;
    dx.mul(py)?.sub(dy.mul(px)?)
}

/// Return the exact ray-crossing winding increment ONLY after proving that
/// this segment cannot touch any point in the entire query cell. The witness
/// is a representable corner of that enclosing cell. Consequently its winding
/// applies to every point in the cell, including concave and holed paths.
fn certified_line_crossing(cell: Cell, a: [f64; 2], b: [f64; 2]) -> Option<i64> {
    if a == b {
        return Some(0);
    }
    let separate_bounds = cell.x.hi < a[0].min(b[0])
        || cell.x.lo > a[0].max(b[0])
        || cell.y.hi < a[1].min(b[1])
        || cell.y.lo > a[1].max(b[1]);
    if !separate_bounds {
        let side = line_side(a, b, cell.x, cell.y)?;
        // A segment can meet the cell only if its supporting line can.
        // A zero-straddling interval does not authorize either membership.
        if side.lo <= 0.0 && side.hi >= 0.0 {
            return None;
        }
    }
    let y = cell.y.lo;
    let upward = a[1] <= y && y < b[1];
    let downward = b[1] <= y && y < a[1];
    if !upward && !downward {
        return Some(0);
    }
    let side = line_side(a, b, Interval::scalar(cell.x.lo)?, Interval::scalar(y)?)?;
    if upward {
        if side.lo > 0.0 {
            Some(1)
        } else if side.hi < 0.0 {
            Some(0)
        } else {
            None
        }
    } else if side.hi < 0.0 {
        Some(-1)
    } else if side.lo > 0.0 {
        Some(0)
    } else {
        None
    }
}

/// Linear-time, zero-allocation source-authoritative winding certificate for
/// ANY straight-line F1 fill path. Handles concavity, multiple contours,
/// opposite or same contour winding, explicit and implicit closure, and the
/// actual EvenOdd/NonZero law. A boundary/precision ambiguity is Unknown.
fn certify_straight_path(path: &Render2dPath, cell: Cell) -> CellCertificate {
    // The full scan + closure + interval-query work was admitted by certify_path
    // BEFORE even discovering which F1 command classes this path contains.
    let commands = path.commands();
    let mut first = None;
    let mut previous = None;
    let mut has_segment = false;
    let mut winding = 0_i64;
    let mut add_edge = |a: [f64; 2], b: [f64; 2]| -> Option<()> {
        winding = winding.checked_add(certified_line_crossing(cell, a, b)?)?;
        Some(())
    };
    for command in commands {
        match *command {
            Render2dPathCommand::MoveTo(p) => {
                if has_segment {
                    let (Some(a), Some(b)) = (previous, first) else {
                        return CellCertificate::Unknown;
                    };
                    if add_edge(a, b).is_none() {
                        return CellCertificate::Unknown;
                    }
                }
                first = Some([p.x(), p.y()]);
                previous = first;
                has_segment = false;
            }
            Render2dPathCommand::LineTo(p) => {
                let to = [p.x(), p.y()];
                let Some(from) = previous else {
                    return CellCertificate::Unknown;
                };
                if add_edge(from, to).is_none() {
                    return CellCertificate::Unknown;
                }
                previous = Some(to);
                has_segment = true;
            }
            Render2dPathCommand::Close => {
                if has_segment {
                    let (Some(a), Some(b)) = (previous, first) else {
                        return CellCertificate::Unknown;
                    };
                    if add_edge(a, b).is_none() {
                        return CellCertificate::Unknown;
                    }
                }
                previous = first;
                has_segment = false;
            }
            Render2dPathCommand::QuadraticTo { .. } | Render2dPathCommand::CubicTo { .. } => {
                return CellCertificate::Unknown;
            }
        }
    }
    if has_segment {
        let (Some(a), Some(b)) = (previous, first) else {
            return CellCertificate::Unknown;
        };
        if add_edge(a, b).is_none() {
            return CellCertificate::Unknown;
        }
    }
    let inside = match path.fill_rule() {
        crate::composition_2d::Render2dFillRule::EvenOdd => winding % 2 != 0,
        crate::composition_2d::Render2dFillRule::NonZero => winding != 0,
    };
    if inside {
        CellCertificate::Inside
    } else {
        CellCertificate::Outside
    }
}

/// Curve control points certify the conservative exterior even if physical
/// tessellation is arbitrarily coarse. Inside must be proved by a separate
/// exact source query; no polygonization of authored Bézier control hulls.
fn certify_path(path: &Render2dPath, cell: Cell, visits: &mut usize) -> CellCertificate {
    let commands = path.commands();
    // Admit BOTH source-command scans, implicit closure, and worst-case
    // interval geometry in one bounded O(commands) model BEFORE any path
    // traversal. No uncharged per-query prefilter or conditional budget reset.
    let Some(cost) = commands.len().checked_mul(40) else {
        return CellCertificate::Unknown;
    };
    if *visits < cost {
        *visits = 0;
        return CellCertificate::Unknown;
    }
    *visits -= cost;
    if !commands.iter().any(|command| {
        matches!(
            command,
            Render2dPathCommand::QuadraticTo { .. } | Render2dPathCommand::CubicTo { .. }
        )
    }) {
        return certify_straight_path(path, cell);
    }
    let mut hull: Option<[f64; 4]> = None;
    let mut segments = 0_usize;
    let mut point = |p: crate::composition_2d::Render2dPoint| {
        let (x, y) = (p.x(), p.y());
        hull = Some(match hull {
            Some([x0, y0, x1, y1]) => [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
            None => [x, y, x, y],
        });
    };
    for command in commands {
        match *command {
            Render2dPathCommand::MoveTo(p) => point(p),
            Render2dPathCommand::LineTo(p) => {
                segments += 1;
                point(p);
            }
            Render2dPathCommand::QuadraticTo { control, to } => {
                segments += 1;
                point(control);
                point(to);
            }
            Render2dPathCommand::CubicTo {
                control1,
                control2,
                to,
            } => {
                segments += 1;
                point(control1);
                point(control2);
                point(to);
            }
            Render2dPathCommand::Close => {}
        }
    }
    if segments == 0 {
        return CellCertificate::Outside;
    }
    let Some([x0, y0, x1, y1]) = hull else {
        return CellCertificate::Unknown;
    };
    if cell.x.hi < x0 || cell.x.lo > x1 || cell.y.hi < y0 || cell.y.lo > y1 {
        CellCertificate::Outside
    } else {
        bezier_winding::certify_curved_path(path, cell, visits)
    }
}

fn certify_shape(shape: &Render2dShape, cell: Cell, visits: &mut usize) -> CellCertificate {
    if let Some(rect) = shape.as_rect() {
        if rect.is_empty() {
            return CellCertificate::Outside;
        }
        let Some(x0) = Interval::scalar(rect.x()) else {
            return CellCertificate::Unknown;
        };
        let Some(y0) = Interval::scalar(rect.y()) else {
            return CellCertificate::Unknown;
        };
        let Some(x1) = x0.add(Interval::scalar(rect.width()).expect("valid finite width")) else {
            return CellCertificate::Unknown;
        };
        let Some(y1) = y0.add(Interval::scalar(rect.height()).expect("valid finite height")) else {
            return CellCertificate::Unknown;
        };
        cell.classify_rectangle([x0, y0, x1, y1])
    } else if let Some(ellipse) = shape.as_ellipse() {
        cell.classify_ellipse(ellipse)
    } else if let Some(path) = shape.as_path() {
        certify_path(path, cell, visits)
    } else {
        // Centered strokes and rounded corners still need their own
        // independently certified source geometry. Never infer from meshes.
        CellCertificate::Unknown
    }
}

/// A clip is authored in the OWNER PARENT frame, not source-local space.
/// This same operation applies to F1 item clips and immutable group clip DAGs.
fn certify_clipped(
    source: CellCertificate,
    parent_cell: Cell,
    clips: &[Render2dClip],
    visits: &mut usize,
) -> CellCertificate {
    if source == CellCertificate::Outside {
        return CellCertificate::Outside;
    }
    let mut uncertain = source == CellCertificate::Unknown;
    for clip in clips {
        if *visits == 0 {
            return CellCertificate::Unknown;
        }
        *visits -= 1;
        let certificate = parent_cell
            .preimage(clip.clip_to_parent())
            .map(|local| certify_shape(clip.shape(), local, visits))
            .unwrap_or(CellCertificate::Unknown);
        match certificate {
            CellCertificate::Outside => return CellCertificate::Outside,
            CellCertificate::Unknown => uncertain = true,
            CellCertificate::Inside => {}
        }
    }
    if uncertain {
        CellCertificate::Unknown
    } else {
        CellCertificate::Inside
    }
}

/// Return a conservative axis-aligned OUTER halo for positive spread
/// and the normalized Gaussian's finite ±3σ square support. Plain f64
/// multiplication and addition can round down and fabricate Outside at
/// the edge of the kernel. Unrepresentable work is Unknown, not empty.
fn certified_effect_outer_halo(spread: f64, sigma: f64) -> Option<f64> {
    Interval::scalar(3.0)?
        .mul(Interval::scalar(sigma)?)?
        .add(Interval::scalar(spread.max(0.0))?)
        .map(|halo| halo.hi)
}

/// Certified exp(-(x²+y²)/2) on the truncated Gaussian square.
/// Let t=(x²+y²)/32, so 0<=t<=9/16. For positive t the eighth
/// Maclaurin partial sum P8(t) underestimates exp(t); its positive tail
/// is <= t^9/(9! * (1-t/10)), since subsequent term ratios decrease.
/// Thus 1/(P8+tail) <= exp(-t) <= 1/P8. Four interval squarings
/// recover exp(-16t), always rounding outward. No reliance on libm,
/// measured samples, transcendental rounding, or unbudgeted convergence.
fn gaussian_density_enclosure(x: Interval, y: Interval) -> Option<Interval> {
    if x.lo == 0.0 && x.hi == 0.0 && y.lo == 0.0 && y.hi == 0.0 {
        return Interval::scalar(1.0);
    }
    let exponent = squared_interval(x)?
        .add(squared_interval(y)?)?
        .div(Interval::scalar(2.0)?)?;
    let one = Interval::scalar(1.0)?;
    let t = exponent.div(Interval::scalar(16.0)?)?;
    // The true exponent is a sum of squares and is nonnegative. Directed
    // rounding around zero may extend its numeric enclosure below zero;
    // intersecting with this independently proved domain is safe.
    let t = Interval::new(t.lo.max(0.0), t.hi)?;
    if t.hi >= 1.0 {
        return None;
    }
    let mut term = one;
    let mut sum = one;
    for order in 1..=8 {
        term = term.mul(t)?.div(Interval::scalar(order as f64)?)?;
        sum = sum.add(term)?;
    }
    let first_omitted = term.mul(t)?.div(Interval::scalar(9.0)?)?;
    let denominator = one.sub(t.div(Interval::scalar(10.0)?)?)?;
    if denominator.lo <= 0.0 {
        return None;
    }
    let exp_upper = sum.add(first_omitted.div(denominator)?)?;
    let mut density = one.div(Interval::new(sum.lo, exp_upper.hi)?)?;
    for _ in 0..4 {
        density = density.mul(density)?;
    }
    Interval::new(density.lo.max(0.0), density.hi.min(1.0))
}

// One bounded density/area/query-coordinate evaluation entails fewer than
// 256 scalar arithmetic operations, including directed-rounding endpoints.
// The separate region classifier decrements this same fuel for its own work.
const GAUSSIAN_WEIGHT_WORK: usize = 256;

/// Test-only normalized finite Gaussian coverage certificate. The coordinates
/// are in the shadow owner's parent frame, NOT root or raster coordinates.
/// Unknown region cells contribute zero to the lower mass and their full
/// possible mass to the upper bound. There is no guessed alpha or SDF.
/// The caller admits all grid work and every region query through visits.
fn certified_finite_gaussian_coverage(
    region: &NeutralRegion<'_>,
    center: [f64; 2],
    sigma: f64,
    side: usize,
    visits: &mut usize,
) -> Option<[f64; 2]> {
    if !sigma.is_finite() || sigma <= 0.0 || !side.is_power_of_two() || !(8..=64).contains(&side) {
        return None;
    }
    let sigma = Interval::scalar(sigma)?;
    let [qx, qy] = center.map(Interval::scalar);
    let (qx, qy) = (qx?, qy?);
    let mut total = Interval::scalar(0.0)?;
    let mut certain = Interval::scalar(0.0)?;
    let mut unknown = Interval::scalar(0.0)?;
    let mut inside = 0_usize;
    let mut outside = 0_usize;
    // Power-of-two grids have exactly representable finite kernel endpoints.
    let width = 6.0 / side as f64;
    for iy in 0..side {
        let y0 = -3.0 + iy as f64 * width;
        let y1 = y0 + width;
        let ty = Interval::new(y0, y1)?;
        for ix in 0..side {
            if *visits < GAUSSIAN_WEIGHT_WORK {
                return None;
            }
            *visits -= GAUSSIAN_WEIGHT_WORK;
            let x0 = -3.0 + ix as f64 * width;
            let x1 = x0 + width;
            let tx = Interval::new(x0, x1)?;
            let area = Interval::scalar(x1)?
                .sub(Interval::scalar(x0)?)?
                .mul(Interval::scalar(y1)?.sub(Interval::scalar(y0)?)?)?;
            let weight = area.mul(gaussian_density_enclosure(tx, ty)?)?;
            total = total.add(weight)?;
            let query = Cell {
                x: qx.add(sigma.mul(tx)?)?,
                y: qy.add(sigma.mul(ty)?)?,
            };
            match region.certify_cell(query, visits) {
                CellCertificate::Inside => {
                    certain = certain.add(weight)?;
                    inside += 1;
                }
                CellCertificate::Outside => outside += 1,
                CellCertificate::Unknown => unknown = unknown.add(weight)?,
            }
        }
    }
    // Both extrema follow exactly from finite normalized mass, regardless of
    // the numerical integration interval when every cell agrees.
    if inside == side * side {
        return Some([1.0, 1.0]);
    }
    if outside == side * side {
        return Some([0.0, 0.0]);
    }
    if total.lo <= 0.0 {
        return None;
    }
    let lower = Interval::scalar(certain.lo)?
        .div(Interval::scalar(total.hi)?)?
        .lo
        .max(0.0);
    let upper = certain
        .add(unknown)?
        .div(Interval::scalar(total.lo)?)?
        .hi
        .min(1.0);
    if lower > upper {
        return None;
    }
    Some([lower, upper])
}

/// A single bounded owner's-frame Gaussian cell. The rectangle spans a
/// subset of the NORMALIZED [-3,3]^2 domain, not a root-frame image cache.
#[derive(Clone, Copy)]
struct GaussianQuadCell {
    x: Interval,
    y: Interval,
    depth: usize,
}

struct AdaptiveGaussianMass {
    total: Interval,
    definite: Interval,
    uncertain: Interval,
    inside: usize,
    outside: usize,
    leaves: usize,
}

impl AdaptiveGaussianMass {
    fn empty() -> Option<Self> {
        Some(Self {
            total: Interval::scalar(0.0)?,
            definite: Interval::scalar(0.0)?,
            uncertain: Interval::scalar(0.0)?,
            inside: 0,
            outside: 0,
            leaves: 0,
        })
    }

    fn add(&mut self, weight: Interval, certificate: CellCertificate) -> Option<()> {
        self.total = self.total.add(weight)?;
        match certificate {
            CellCertificate::Inside => {
                self.definite = self.definite.add(weight)?;
                self.inside += 1;
            }
            CellCertificate::Outside => self.outside += 1,
            CellCertificate::Unknown => self.uncertain = self.uncertain.add(weight)?,
        }
        self.leaves += 1;
        Some(())
    }

    fn coverage(&self) -> Option<[f64; 2]> {
        if self.inside == self.leaves {
            return Some([1.0, 1.0]);
        }
        if self.outside == self.leaves {
            return Some([0.0, 0.0]);
        }
        if self.total.lo <= 0.0 {
            return None;
        }
        let lower = Interval::scalar(self.definite.lo)?
            .div(Interval::scalar(self.total.hi)?)?
            .lo
            .max(0.0);
        let upper = self
            .definite
            .add(self.uncertain)?
            .div(Interval::scalar(self.total.lo)?)?
            .hi
            .min(1.0);
        if lower > upper {
            return None;
        }
        Some([lower, upper])
    }
}

/// Borrowed immutable Gaussian semantics stay together throughout recursive
/// refinement. This is one private evaluation context, not a second renderer
/// or an alternate scene representation.
struct GaussianRefinement<'r, 'a> {
    region: &'r NeutralRegion<'a>,
    center: [Interval; 2],
    sigma: Interval,
    max_depth: usize,
    mass_threshold: f64,
}

/// Subdivide ALL uncertain geometry cells, and definite cells with a
/// significant remaining Gaussian integration enclosure width. Never refine
/// only geometric boundaries while leaving normalization uncertainty unbound.
/// Every visited node costs its conservative numerical work plus the actual
/// region evaluator's visit charges; depth and total nodes are capped.
fn refine_gaussian_cell(
    context: &GaussianRefinement<'_, '_>,
    cell: GaussianQuadCell,
    remaining_nodes: &mut usize,
    visits: &mut usize,
    accumulator: &mut AdaptiveGaussianMass,
) -> Option<()> {
    if *remaining_nodes == 0 || *visits < GAUSSIAN_WEIGHT_WORK {
        return None;
    }
    *remaining_nodes -= 1;
    *visits -= GAUSSIAN_WEIGHT_WORK;

    let area = Interval::scalar(cell.x.hi)?
        .sub(Interval::scalar(cell.x.lo)?)?
        .mul(Interval::scalar(cell.y.hi)?.sub(Interval::scalar(cell.y.lo)?)?)?;
    let weight = area.mul(gaussian_density_enclosure(cell.x, cell.y)?)?;
    let query = Cell {
        x: context.center[0].add(context.sigma.mul(cell.x)?)?,
        y: context.center[1].add(context.sigma.mul(cell.y)?)?,
    };
    let certificate = context.region.certify_cell(query, visits);
    if certificate == CellCertificate::Unknown && *visits == 0 {
        return None;
    }
    let should_refine = cell.depth < context.max_depth
        && (certificate == CellCertificate::Unknown
            || weight.hi - weight.lo > context.mass_threshold);
    if !should_refine {
        return accumulator.add(weight, certificate);
    }
    let mx = cell.x.lo + (cell.x.hi - cell.x.lo) * 0.5;
    let my = cell.y.lo + (cell.y.hi - cell.y.lo) * 0.5;
    let left = Interval::new(cell.x.lo, mx)?;
    let right = Interval::new(mx, cell.x.hi)?;
    let top = Interval::new(cell.y.lo, my)?;
    let bottom = Interval::new(my, cell.y.hi)?;
    for (x, y) in [(left, top), (right, top), (left, bottom), (right, bottom)] {
        refine_gaussian_cell(
            context,
            GaussianQuadCell {
                x,
                y,
                depth: cell.depth + 1,
            },
            remaining_nodes,
            visits,
            accumulator,
        )?;
    }
    Some(())
}

/// Certify an ENTIRE region of possible correlated 4x4 sample centers
/// against the full finite Gaussian kernel. The owner-frame Minkowski sum
/// encloses all sample kernels in a tile: if the original F1 region is
/// definitely Inside/Outside, every center has exact normalized coverage
/// 1/0 without any per-sample Gaussian density work.
fn certified_uniform_kernel_coverage(
    region: &NeutralRegion<'_>,
    centers: Cell,
    sigma: Interval,
    visits: &mut usize,
) -> Option<[f64; 2]> {
    if sigma.lo <= 0.0 || *visits < GAUSSIAN_WEIGHT_WORK {
        return None;
    }
    *visits -= GAUSSIAN_WEIGHT_WORK;
    let cutoff = Interval::new(-3.0, 3.0)?;
    let halo = sigma.mul(cutoff)?;
    let swept = Cell {
        x: centers.x.add(halo)?,
        y: centers.y.add(halo)?,
    };
    match region.certify_cell(swept, visits) {
        CellCertificate::Inside => Some([1.0, 1.0]),
        CellCertificate::Outside => Some([0.0, 0.0]),
        CellCertificate::Unknown => None,
    }
}

/// Counts only exact all-kernel coverage results, not opaque masks,
/// per-sample colors, GPU costs or an approximation of unresolved pixels.
#[derive(Default, Debug, PartialEq, Eq)]
struct SharedTileCoverage {
    inside_samples: usize,
    outside_samples: usize,
    unresolved_samples: usize,
    queries: usize,
}

/// Spatially share FULL-kernel certificates over correlated sample centers.
/// Subdivision is over pixel tiles, not Gaussian integration cells. Unknown
/// tiles at single-pixel granularity retain all 16 unresolved samples.
fn visit_shared_tile(
    region: &NeutralRegion<'_>,
    origin: [usize; 2],
    side: usize,
    sigma: Interval,
    nodes: &mut usize,
    visits: &mut usize,
    audit: &mut SharedTileCoverage,
) -> Option<()> {
    // Reserve at least the numeric allowance and one root classifier visit.
    // Exact admission is shared across tiles, not charged once per subpixel.
    if *nodes == 0 || *visits <= GAUSSIAN_WEIGHT_WORK {
        return None;
    }
    *nodes -= 1;
    audit.queries = audit.queries.checked_add(1)?;
    let xmax = origin[0].checked_add(side)? as f64 - 0.125;
    let ymax = origin[1].checked_add(side)? as f64 - 0.125;
    let centers = Cell::new([
        origin[0] as f64 + 0.125,
        origin[1] as f64 + 0.125,
        xmax,
        ymax,
    ])?;
    let sites = side.checked_mul(side)?.checked_mul(16)?;
    match certified_uniform_kernel_coverage(region, centers, sigma, visits) {
        Some([1.0, 1.0]) => {
            audit.inside_samples = audit.inside_samples.checked_add(sites)?;
        }
        Some([0.0, 0.0]) => {
            audit.outside_samples = audit.outside_samples.checked_add(sites)?;
        }
        Some(_) => return None,
        None if side == 1 => {
            if *visits == 0 {
                return None;
            }
            audit.unresolved_samples = audit.unresolved_samples.checked_add(sites)?;
        }
        None => {
            let half = side / 2;
            for dy in [0, half] {
                for dx in [0, half] {
                    visit_shared_tile(
                        region,
                        [origin[0].checked_add(dx)?, origin[1].checked_add(dy)?],
                        half,
                        sigma,
                        nodes,
                        visits,
                        audit,
                    )?;
                }
            }
        }
    }
    Some(())
}

/// Atomic test-only feasibility report: no partially counted scene results
/// escape if global node or numerical/geometry admission is exhausted.
fn certified_shared_tile_quadtree(
    region: &NeutralRegion<'_>,
    canvas_side: usize,
    tile_side: usize,
    sigma: f64,
    max_nodes: usize,
    visits: &mut usize,
) -> Option<SharedTileCoverage> {
    if canvas_side == 0
        || canvas_side > 64
        || !canvas_side.is_power_of_two()
        || tile_side == 0
        || tile_side > 8
        || !tile_side.is_power_of_two()
        || !canvas_side.is_multiple_of(tile_side)
        || max_nodes == 0
        || max_nodes > 5_440
        || !sigma.is_finite()
        || sigma <= 0.0
    {
        return None;
    }
    let mut audit = SharedTileCoverage::default();
    let mut remaining_nodes = max_nodes;
    let sigma = Interval::scalar(sigma)?;
    for y in (0..canvas_side).step_by(tile_side) {
        for x in (0..canvas_side).step_by(tile_side) {
            visit_shared_tile(
                region,
                [x, y],
                tile_side,
                sigma,
                &mut remaining_nodes,
                visits,
                &mut audit,
            )?;
        }
    }
    let counted = audit
        .inside_samples
        .checked_add(audit.outside_samples)?
        .checked_add(audit.unresolved_samples)?;
    if counted != canvas_side.checked_mul(canvas_side)?.checked_mul(16)? {
        return None;
    }
    Some(audit)
}

/// Feasibility only: a deterministic bounded quadtree certificate over the
/// normalized finite parent-frame Gaussian. A target error that cannot be
/// proven within admitted query/numeric work returns None, never a guessed
/// midpoint, a false 0 or a false 1.
fn certified_adaptive_finite_gaussian_coverage(
    region: &NeutralRegion<'_>,
    center: [f64; 2],
    sigma: f64,
    max_depth: usize,
    max_nodes: usize,
    max_width: f64,
    visits: &mut usize,
) -> Option<([f64; 2], usize)> {
    if !sigma.is_finite()
        || sigma <= 0.0
        || max_depth > 6
        || max_nodes == 0
        || max_nodes > 5_461
        || !max_width.is_finite()
        || max_width <= 0.0
        || max_width >= 1.0
    {
        return None;
    }
    let center = [Interval::scalar(center[0])?, Interval::scalar(center[1])?];
    let sigma = Interval::scalar(sigma)?;
    // One full-kernel query is also shareable among every correlated sample
    // center in a certified tile, without changing F1 source ownership.
    if let Some(exact) = certified_uniform_kernel_coverage(
        region,
        Cell {
            x: center[0],
            y: center[1],
        },
        sigma,
        visits,
    ) {
        return Some((exact, 1));
    }
    if *visits == 0 {
        return None;
    }
    let mut result = AdaptiveGaussianMass::empty()?;
    let mut remaining_nodes = max_nodes;
    let context = GaussianRefinement {
        region,
        center,
        sigma,
        max_depth,
        mass_threshold: max_width / 128.0,
    };
    refine_gaussian_cell(
        &context,
        GaussianQuadCell {
            x: Interval::new(-3.0, 3.0)?,
            y: Interval::new(-3.0, 3.0)?,
            depth: 0,
        },
        &mut remaining_nodes,
        visits,
        &mut result,
    )?;
    let coverage = result.coverage()?;
    if coverage[1] - coverage[0] > max_width {
        return None;
    }
    Some((coverage, max_nodes - remaining_nodes))
}

/// Certify the full Euclidean disk around an entire query cell from
/// original F1 membership, rather than requiring the larger axis square.
///
/// Slice [-r,r] into a fixed power-of-two number of vertical strips. For
/// each strip, one outward-rounded bounding rectangle encloses ALL disk
/// points (including the full query cell's Minkowski translation). If
/// every rectangle is proven Inside C, Q ⊆ C⊖B_r. Uncertain strips remain
/// Unknown: finite strips can reject near-tangent requests, never admit
/// incorrect erosion. Source queries and conservative strip work share
/// the SAME bounded per-invocation proof fuel.
///
/// This is test-only feasibility for the curved inner corner of a truly
/// holed region. Not an approximate production raster or semantic SDF.
fn certified_negative_interior_disk_cover(
    source: &NeutralRegion<'_>,
    query: Cell,
    radius: f64,
    visits: &mut usize,
) -> bool {
    if !radius.is_finite() || radius <= 0.0 {
        return false;
    }
    const STRIPS: usize = 64;
    const STRIP_COST: usize = 48;
    let Some(r) = Interval::scalar(radius) else {
        return false;
    };
    let Some(radius_squared) = r.mul(r) else {
        return false;
    };
    for i in 0..STRIPS {
        if *visits <= STRIP_COST {
            return false;
        }
        *visits -= STRIP_COST;
        // Binary fractions and the outward interval product enclose
        // every x of the TRUE disk without a piecewise-linear surrogate.
        let x0 = (i as f64) / 32.0 - 1.0;
        let x1 = ((i + 1) as f64) / 32.0 - 1.0;
        let Some(offset_x) = Interval::new(x0, x1).and_then(|unit| unit.mul(r)) else {
            return false;
        };
        let nearest = if offset_x.lo <= 0.0 && offset_x.hi >= 0.0 {
            0.0
        } else {
            offset_x.lo.abs().min(offset_x.hi.abs())
        };
        let Some(nearest_squared) = Interval::scalar(nearest).and_then(|v| v.mul(v)) else {
            return false;
        };
        // The largest disk y at any x in the strip is reached at
        // the nearest horizontal location to the disk origin.
        // Round the *difference* and sqrt outward before building
        // the full strip envelope.
        let y_squared = (radius_squared.hi - nearest_squared.lo).next_up();
        if !y_squared.is_finite() || y_squared < 0.0 {
            return false;
        }
        let y_radius = y_squared.sqrt().next_up();
        let Some(outer) = Interval::new(-y_radius, y_radius) else {
            return false;
        };
        let (Some(x), Some(y)) = (query.x.add(offset_x), query.y.add(outer)) else {
            return false;
        };
        if source.certify_cell(Cell { x, y }, visits) != CellCertificate::Inside {
            return false;
        }
    }
    true
}

/// F3F-NS02 proof-only negative-erosion EXTERIOR witness.
///
/// For any point q definitely outside C, every x with |x-q|₂<r lies
/// outside C⊖disk(r). Thus one q that covers the ENTIRE query cell
/// certifies Outside, including holes, concavities and nonrectangular
/// original F1 contours. An outward-rounded squared Euclidean bound
/// avoids falsely treating a disk as a Manhattan diamond.
/// `query` already includes the child Gaussian
/// square halo in the effect owner's immediate-parent frame.
///
/// Search a bounded set of cardinal, diagonal and inscribed Euclidean
/// exterior witnesses. Each is independently checked against the exact
/// original-F1 source; no sampled SDF or alpha mask is introduced. A missed witness means Unknown, never false Outside.
/// Cost is charged to the existing shared source-query fuel.
fn certified_negative_exterior_witness(
    source: &NeutralRegion<'_>,
    query: Cell,
    radius: f64,
    visits: &mut usize,
) -> bool {
    if !radius.is_finite() || radius <= 0.0 {
        return false;
    }
    let center_x = query.x.lo * 0.5 + query.x.hi * 0.5;
    let center_y = query.y.lo * 0.5 + query.y.hi * 0.5;
    if !center_x.is_finite() || !center_y.is_finite() {
        return false;
    }
    // Orthogonal exterior segments may cover a complete Gaussian halo
    // even when no SINGLE exterior point is near all halo corners.
    // Witnesses at one-fifth and nine-tenths of the source radius
    // are finite deterministic candidates, never a sampled SDF.
    const PROBE_COST: usize = 48;
    for [ux, uy] in [
        [0.0_f64, 0.0],
        [-0.2, 0.0],
        [0.2, 0.0],
        [0.0, -0.2],
        [0.0, 0.2],
        [-0.9, 0.0],
        [0.9, 0.0],
        [0.0, -0.9],
        [0.0, 0.9],
        [-0.95, 0.0],
        [0.95, 0.0],
        [0.0, -0.95],
        [0.0, 0.95],
        // Diagonal witnesses distinguish true Euclidean rounded-hole
        // erosion from a square/Manhattan approximation. Every pair is
        // tested against an outward-rounded squared-distance bound.
        [-0.2, -0.95],
        [-0.2, 0.95],
        [0.2, -0.95],
        [0.2, 0.95],
        [-0.4, -0.9],
        [-0.4, 0.9],
        [0.4, -0.9],
        [0.4, 0.9],
        [-0.7, -0.7],
        [-0.7, 0.7],
        [0.7, -0.7],
        [0.7, 0.7],
        [-0.9, -0.4],
        [-0.9, 0.4],
        [0.9, -0.4],
        [0.9, 0.4],
        [-0.95, -0.2],
        [-0.95, 0.2],
        [0.95, -0.2],
        [0.95, 0.2],
    ] {
        if *visits <= PROBE_COST {
            return false;
        }
        *visits -= PROBE_COST;
        let (qx, qy) = (ux.mul_add(radius, center_x), uy.mul_add(radius, center_y));
        let (Some(x), Some(y)) = (Interval::scalar(qx), Interval::scalar(qy)) else {
            continue;
        };
        let (Some(dx), Some(dy)) = (x.sub(query.x), y.sub(query.y)) else {
            continue;
        };
        let bound_x = dx.lo.abs().max(dx.hi.abs()).next_up();
        let bound_y = dy.lo.abs().max(dy.hi.abs()).next_up();
        // Every (px,py) in the query has a perpendicular exterior
        // witness at (qx,py) if this entire vertical segment is outside C.
        if bound_x < radius
            && source.certify_cell(Cell { x, y: query.y }, visits) == CellCertificate::Outside
        {
            return true;
        }
        // Same proof with one outside horizontal segment (px,qy).
        if bound_y < radius
            && source.certify_cell(Cell { x: query.x, y }, visits) == CellCertificate::Outside
        {
            return true;
        }
        // For a definite exterior POINT q the sufficient condition is
        // Euclidean, not Manhattan: max_{p in query}|p-q|² < r².
        // Outward-rounded interval products ensure that any uncertain
        // corner, overflow or subnormal uncertainty fails closed.
        let Some(radius_i) = Interval::scalar(radius) else {
            continue;
        };
        let Some(squared_bound) = Interval::scalar(bound_x).and_then(|dx| {
            dx.mul(dx)?
                .add(Interval::scalar(bound_y)?.mul(Interval::scalar(bound_y)?)?)
        }) else {
            continue;
        };
        let Some(squared_radius) = radius_i.mul(radius_i) else {
            continue;
        };
        if squared_bound.hi < squared_radius.lo
            && source.certify_cell(Cell { x, y }, visits) == CellCertificate::Outside
        {
            return true;
        }
    }
    // The fixed cardinal/diagonal probes may miss a *narrow curved
    // corner*: the certified exterior exists only in a small angular
    // interval near the original hole corner. A bounded second pass
    // samples an INSCRIBED witness circle; every candidate must still
    // prove both the complete query's Euclidean radius bound and
    // original-F1 exterior membership. Angular sampling is ONLY a
    // witness search and never substitutes a faceted support boundary.
    // No witness is a valid Unknown, not a false region classification.
    const DIRECTIONS: usize = 256;
    for step in 0..DIRECTIONS {
        if *visits <= PROBE_COST {
            return false;
        }
        *visits -= PROBE_COST;
        let angle = (step as f64) * (std::f64::consts::TAU / DIRECTIONS as f64);
        let (qx, qy) = (
            (0.985 * radius * angle.cos()).mul_add(1.0, center_x),
            (0.985 * radius * angle.sin()).mul_add(1.0, center_y),
        );
        let (Some(x), Some(y)) = (Interval::scalar(qx), Interval::scalar(qy)) else {
            continue;
        };
        let (Some(dx), Some(dy)) = (x.sub(query.x), y.sub(query.y)) else {
            continue;
        };
        let bound_x = dx.lo.abs().max(dx.hi.abs()).next_up();
        let bound_y = dy.lo.abs().max(dy.hi.abs()).next_up();
        let Some(squared_bound) = Interval::scalar(bound_x).and_then(|x| {
            x.mul(x)?
                .add(Interval::scalar(bound_y)?.mul(Interval::scalar(bound_y)?)?)
        }) else {
            continue;
        };
        let Some(squared_radius) = Interval::scalar(radius).and_then(|r| r.mul(r)) else {
            continue;
        };
        if squared_bound.hi < squared_radius.lo
            && source.certify_cell(Cell { x, y }, visits) == CellCertificate::Outside
        {
            return true;
        }
    }
    false
}

impl NeutralRegion<'_> {
    fn certify_cell(&self, cell: Cell, visits: &mut usize) -> CellCertificate {
        if *visits == 0 {
            return CellCertificate::Unknown;
        }
        *visits -= 1;
        match &self.kind {
            Kind::Source(mesh) => {
                if let Some(item) = self.semantic_item {
                    return certify_authored_item(item, cell, visits);
                }
                // Synthetic private fixture only. Real renderable F1 leaves
                // always borrow their original item through source_from_item.
                if mesh.triangles.is_empty() {
                    return CellCertificate::Outside;
                }
                let Some(rect) = support::certified_axis_aligned_rectangle(&mesh.triangles) else {
                    return CellCertificate::Unknown;
                };
                let scale = mesh.units_per_parent_logical_unit;
                if !scale.is_finite() || scale <= 0.0 {
                    return CellCertificate::Unknown;
                }
                let Some(scale) = Interval::scalar(scale) else {
                    return CellCertificate::Unknown;
                };
                let intervals = rect.map(|v| Interval::scalar(v).and_then(|v| v.div(scale)));
                let [Some(x0), Some(y0), Some(x1), Some(y1)] = intervals else {
                    return CellCertificate::Unknown;
                };
                cell.classify_rectangle([x0, y0, x1, y1])
            }
            Kind::Union(children) => {
                let mut unknown = false;
                for child in children {
                    match child.certify_cell(cell, visits) {
                        CellCertificate::Inside => return CellCertificate::Inside,
                        CellCertificate::Unknown => unknown = true,
                        CellCertificate::Outside => {}
                    }
                }
                if unknown {
                    CellCertificate::Unknown
                } else {
                    CellCertificate::Outside
                }
            }
            Kind::Affine { source, to_parent } => cell
                .preimage(*to_parent)
                .map(|preimage| source.certify_cell(preimage, visits))
                .unwrap_or(CellCertificate::Unknown),
            Kind::Clip { source, clips } => {
                let child = source.certify_cell(cell, visits);
                certify_clipped(child, cell, clips, visits)
            }
            Kind::Effect { source, shadow } => {
                let Some(translated) = cell.shift(shadow.offset_x(), shadow.offset_y()) else {
                    return CellCertificate::Unknown;
                };
                let spread = shadow.spread();
                // A spread region is contained in its own finite-Gaussian
                // support, but this says NOTHING about blurred alpha.
                let definite_inside = if spread < 0.0 {
                    translated
                        .expanded(-spread)
                        .map(|c| source.certify_cell(c, visits))
                } else {
                    Some(source.certify_cell(translated, visits))
                };
                if definite_inside == Some(CellCertificate::Inside)
                    || (spread < 0.0
                        && certified_negative_interior_disk_cover(
                            source, translated, -spread, visits,
                        ))
                {
                    return CellCertificate::Inside;
                }
                let Some(outer_halo) = certified_effect_outer_halo(spread, shadow.sigma()) else {
                    return CellCertificate::Unknown;
                };
                let Some(outer) = translated.expanded(outer_halo) else {
                    return CellCertificate::Unknown;
                };
                if source.certify_cell(outer, visits) == CellCertificate::Outside
                    || (spread < 0.0
                        && certified_negative_exterior_witness(source, outer, -spread, visits))
                {
                    CellCertificate::Outside
                } else {
                    CellCertificate::Unknown
                }
            }
        }
    }
}

/// Certify original F1 fill geometry first; apply item clips independently
/// in their declared immediate-parent space. Painter color/opacity are ignored.
fn certify_authored_item(item: &Render2dItem, cell: Cell, visits: &mut usize) -> CellCertificate {
    let Render2dPrimitive::Fill { shape, .. } = item.primitive() else {
        return CellCertificate::Unknown;
    };
    let source = cell
        .preimage(item.local_to_parent())
        .map(|local| certify_shape(shape, local, visits))
        .unwrap_or(CellCertificate::Unknown);
    certify_clipped(source, cell, item.clips(), visits)
}

fn certify(region: &NeutralRegion<'_>, bounds: [f64; 4]) -> CellCertificate {
    let Some(cell) = Cell::new(bounds) else {
        return CellCertificate::Unknown;
    };
    let mut visits = 4096;
    region.certify_cell(cell, &mut visits)
}

fn rectangle(
    builder: &mut NeutralRegionBuilder,
    rect: [f64; 4],
    path: &[usize],
) -> Arc<NeutralRegion<'static>> {
    let [x0, y0, x1, y1] = rect;
    builder
        .source(
            support::NeutralMesh {
                units_per_parent_logical_unit: 1.0,
                triangles: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]],
                bounds: rect,
            },
            path,
        )
        .unwrap()
}

fn builder() -> NeutralRegionBuilder {
    NeutralRegionBuilder::new(
        shadow::NeutralPreparationScale {
            field_raster_scale: 1.0,
            geometry_sample_scale: 4.0,
        },
        1_048_576,
    )
}

#[test]
fn certified_cells_handle_holed_union_and_signed_morphology_without_alpha_masks() {
    let mut builder = builder();
    let sources = [
        [0.0, 0.0, 3.0, 10.0],
        [7.0, 0.0, 10.0, 10.0],
        [3.0, 0.0, 7.0, 3.0],
        [3.0, 7.0, 7.0, 10.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, rect)| rectangle(&mut builder, rect, &[0, i]))
    .collect();
    let ring = builder.union(sources, &[0]).unwrap();
    assert_eq!(
        certify(&ring, [4.5, 4.5, 5.5, 5.5]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&ring, [1.0, 4.0, 2.0, 6.0]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&ring, [2.9, 4.0, 3.1, 6.0]),
        CellCertificate::Unknown
    );
    for spread in [-0.5, 0.5] {
        let effect = Render2dDropShadow::new(
            0.0,
            0.0,
            0.0,
            spread,
            crate::composition_2d::Render2dColorRgba8::TRANSPARENT,
        )
        .unwrap();
        let region = builder.effect(Arc::clone(&ring), effect, &[0, 5]).unwrap();
        assert_eq!(
            certify(&region, [4.5, 4.5, 5.5, 5.5]),
            CellCertificate::Outside
        );
        if spread < 0.0 {
            assert_eq!(
                certify(&region, [1.0, 4.0, 2.0, 6.0]),
                CellCertificate::Inside
            );
        }
    }
}

#[test]
fn original_f1_holed_nested_negative_exterior_witness_survives_ancestor_shear() {
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dOpacity, Render2dRect, Render2dShape,
    };
    // A forged physical mesh has no authority over the real F1 source.
    // This ring's actual F1 hole is [18,22]².
    let mut graph = builder();
    let fake = || support::NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: vec![
            [0.0, 0.0],
            [64.0, 0.0],
            [64.0, 64.0],
            [0.0, 0.0],
            [64.0, 64.0],
            [0.0, 64.0],
        ],
        bounds: [0.0, 0.0, 64.0, 64.0],
    };
    let rects = [
        [10.0, 10.0, 20.0, 8.0],
        [10.0, 22.0, 20.0, 8.0],
        [10.0, 18.0, 8.0, 4.0],
        [22.0, 18.0, 8.0, 4.0],
    ];
    let items = rects.map(|[x, y, w, h]| {
        Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(x, y, w, h).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        )
    });
    let sources = items
        .iter()
        .enumerate()
        .map(|(index, item)| graph.source_from_item(fake(), item, &[0, index]).unwrap())
        .collect::<Vec<_>>();
    let ring = graph.union(sources, &[0]).unwrap();
    let child_shadow =
        Render2dDropShadow::new(8.0, 0.0, 0.001, -1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
    let eroded = graph
        .effect(Arc::clone(&ring), child_shadow, &[0, 4])
        .unwrap();
    let union = graph
        .union(vec![Arc::clone(&ring), Arc::clone(&eroded)], &[0])
        .unwrap();
    let ancestor = graph
        .affine(
            union,
            Render2dAffineTransform::new(1.0, 0.0, 0.5, 1.0, 0.0, 0.0).unwrap(),
            &[0],
        )
        .unwrap();
    let next_effect = graph
        .effect(
            ancestor,
            Render2dDropShadow::new(0.0, 0.0, 0.001, 0.0, Render2dColorRgba8::TRANSPARENT).unwrap(),
            &[0],
        )
        .unwrap();

    let mut visits = 16_777_216;
    // Root x=30,y=20 -> parent x=20,y=20, within ORIGINAL
    // empty hole. The translated LEFT eroded arm moves into the hole.
    // An ancestor must observe this otherwise missing support.
    assert_eq!(
        next_effect.certify_cell(Cell::new([30.2, 20.2, 30.3, 20.3]).unwrap(), &mut visits),
        CellCertificate::Inside
    );
    // Root x=40.5,y=20 -> parent x=30.5: outside original C,
    // but translated source x=22.5 lies just INSIDE the right
    // hole boundary's erosion disk. This is genuinely outside
    // nonabsorbed E despite being inside the physical triangles.
    assert_eq!(
        next_effect.certify_cell(
            Cell::new([40.49, 19.99, 40.51, 20.01]).unwrap(),
            &mut visits
        ),
        CellCertificate::Outside
    );
    // Near the outer-right wall, E extends well BEYOND original C.
    assert_eq!(
        next_effect.certify_cell(Cell::new([40.2, 12.2, 40.3, 12.3]).unwrap(), &mut visits),
        CellCertificate::Inside
    );
    assert_eq!(
        next_effect.certify_cell(Cell::new([60.0, 20.0, 60.1, 20.1]).unwrap(), &mut visits),
        CellCertificate::Outside
    );
    // Independent boundary oracle for the NONABSORBED right-hand strip:
    // C outer xmax=30; Euclidean radius 1 erodes it to x<=29.
    // Offset +8.5 then ancestor x_root=x_parent+0.5*y_parent:
    // right support edge is x_root=37.5+0.5*y_root.
    // All original C ends at x_root<=30+0.5*y, far to the left.
    // Both finite Gaussian halos are <=.003 in their own frames.
    // Every correlated site of pixel (43,12) is >=.0625 from
    // this edge, so its NORMALIZED Gaussian coverage is exactly
    // 0 or 1; no quadrature tolerance or physical mesh is involved.
    let boundary_shadow =
        Render2dDropShadow::new(8.5, 0.0, 0.001, -1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
    let boundary_effect = graph
        .effect(Arc::clone(&ring), boundary_shadow, &[0, 5])
        .unwrap();
    let boundary_union = graph
        .union(vec![Arc::clone(&ring), boundary_effect], &[0])
        .unwrap();
    let boundary_affine = graph
        .affine(
            boundary_union,
            Render2dAffineTransform::new(1.0, 0.0, 0.5, 1.0, 0.0, 0.0).unwrap(),
            &[0],
        )
        .unwrap();
    let ancestor_blur =
        Render2dDropShadow::new(0.0, 0.0, 0.001, 0.0, Render2dColorRgba8::TRANSPARENT).unwrap();
    let boundary_ancestor = graph.effect(boundary_affine, ancestor_blur, &[0]).unwrap();
    let mut fully_covered = 0;
    let mut fully_empty = 0;
    for sy in 0..4 {
        for sx in 0..4 {
            let x = 43.0 + (f64::from(sx) + 0.5) / 4.0;
            let y = 12.0 + (f64::from(sy) + 0.5) / 4.0;
            let expected = if x < 37.5 + 0.5 * y {
                fully_covered += 1;
                CellCertificate::Inside
            } else {
                fully_empty += 1;
                CellCertificate::Outside
            };
            assert_eq!(
                boundary_ancestor.certify_cell(Cell::new([x, y, x, y]).unwrap(), &mut visits),
                expected,
                "source-exact nested boundary site ({sx},{sy})"
            );
        }
    }
    assert_eq!((fully_covered, fully_empty), (12, 4));
    assert!(visits < 16_777_216 && visits > 16_000_000);
}

#[test]
fn original_f1_holed_path_negative_disk_curved_corner_has_16_site_shear_oracle() {
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dFillRule, Render2dOpacity, Render2dPath,
        Render2dPathCommand as Command, Render2dPoint, Render2dPrimitive, Render2dShape,
    };
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    // One ORIGINAL F1 EvenOdd path: C=[10,30]² minus hole [18,22]².
    // The actual Euclidean erosion r=1 has a QUARTER-CIRCLE at the
    // hole's upper-right corner (22,22), not a beveled polygon or box.
    let path = Render2dPath::new(
        Render2dFillRule::EvenOdd,
        vec![
            Command::MoveTo(p(10.0, 10.0)),
            Command::LineTo(p(30.0, 10.0)),
            Command::LineTo(p(30.0, 30.0)),
            Command::LineTo(p(10.0, 30.0)),
            Command::Close,
            Command::MoveTo(p(18.0, 18.0)),
            Command::LineTo(p(18.0, 22.0)),
            Command::LineTo(p(22.0, 22.0)),
            Command::LineTo(p(22.0, 18.0)),
            Command::Close,
        ],
    )
    .unwrap();
    let item = Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::path(path),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    );
    let mut graph = builder();
    // Deliberately FALSE all-canvas physical triangle: the source-exact
    // proof cannot accidentally inherit polygonal GPU approximation.
    let fake = support::NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: vec![[0.0, 0.0], [64.0, 0.0], [0.0, 64.0]],
        bounds: [0.0, 0.0, 64.0, 64.0],
    };
    let source = graph.source_from_item(fake, &item, &[0, 0]).unwrap();
    let child = graph
        .effect(
            Arc::clone(&source),
            Render2dDropShadow::new(8.0, 0.0, 0.001, -1.0, Render2dColorRgba8::TRANSPARENT)
                .unwrap(),
            &[0, 1],
        )
        .unwrap();
    let union = graph.union(vec![source, child], &[0]).unwrap();
    let sheared = graph
        .affine(
            union,
            Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap(),
            &[0],
        )
        .unwrap();
    let ancestor = graph
        .effect(
            sheared,
            Render2dDropShadow::new(0.0, 0.0, 0.001, 0.0, Render2dColorRgba8::TRANSPARENT).unwrap(),
            &[0],
        )
        .unwrap();
    let mut visits = 16_777_216;
    let mut inside = 0_usize;
    let mut outside = 0_usize;
    for sy in 0..4 {
        for sx in 0..4 {
            // The globally correlated site is inverse-mapped BEFORE
            // the child's disk, offset or Gaussian is evaluated.
            let root_x = 36.0 + (f64::from(sx) + 0.5) / 4.0;
            let root_y = 22.0 + (f64::from(sy) + 0.5) / 4.0;
            let owner_x = root_x - 0.25 * root_y - 8.0;
            let owner_y = root_y;
            // Exact independent signed distance to the inner-hole
            // axis-aligned corner. Here C's exterior outer edge and
            // original unshifted C are provably far away.
            let dx = (owner_x - 22.0).max(0.0);
            let dy = (owner_y - 22.0).max(0.0);
            let distance = dx.hypot(dy);
            assert!(
                (distance - 1.0).abs() > 0.02,
                "independent oracle site must avoid halo ambiguity"
            );
            let expected = if distance > 1.0 {
                inside += 1;
                CellCertificate::Inside
            } else {
                outside += 1;
                CellCertificate::Outside
            };
            assert_eq!(
                ancestor.certify_cell(
                    Cell::new([root_x, root_y, root_x, root_y]).unwrap(),
                    &mut visits,
                ),
                expected,
                "curved holed F1 signed disk, ancestor shear, 4x4 site ({sx},{sy})"
            );
        }
    }
    // True 3/4 circle-crossing pixel: independent 9/16 covered,
    // 7/16 absent; no sampled SDF, physical triangle or alpha involved.
    assert_eq!((inside, outside), (9, 7));
    assert!(visits < 16_777_216 && visits > 10_000_000);
    let mut exhausted = 1;
    assert_eq!(
        ancestor.certify_cell(
            Cell::new([36.875, 22.625, 36.875, 22.625]).unwrap(),
            &mut exhausted
        ),
        CellCertificate::Unknown
    );
}

#[test]
fn erosion_exclusion_covers_finite_gaussian_halo_not_just_its_center() {
    let mut graph = builder();
    let square = rectangle(&mut graph, [0.0, 0.0, 10.0, 10.0], &[0]);
    let negative = graph
        .effect(
            square,
            Render2dDropShadow::new(
                0.0,
                0.0,
                0.25,
                -1.0,
                crate::composition_2d::Render2dColorRgba8::TRANSPARENT,
            )
            .unwrap(),
            &[0],
        )
        .unwrap();
    let mut visits = 16_777_216;
    // E begins at x=1.0; its 3-sigma envelope begins at x=.25.
    // A center outside E can STILL be inside its finite halo.
    assert_ne!(
        negative.certify_cell(Cell::new([0.8, 5.0, 0.8, 5.0]).unwrap(), &mut visits),
        CellCertificate::Outside
    );
    assert_eq!(
        negative.certify_cell(Cell::new([0.1, 5.0, 0.1, 5.0]).unwrap(), &mut visits),
        CellCertificate::Outside
    );
    assert!(visits < 16_777_216);
}

#[test]
fn affine_cell_preimage_is_outward_bounded_and_rejects_singular_certainty() {
    let mut builder = builder();
    let rect = rectangle(&mut builder, [0.0, 0.0, 2.0, 2.0], &[0, 0]);
    let shear = Render2dAffineTransform::new(1.0, 0.0, 0.5, 1.0, 4.0, 3.0).unwrap();
    let sheared = builder.affine(Arc::clone(&rect), shear, &[0, 1]).unwrap();
    assert_eq!(
        certify(&sheared, [5.25, 3.5, 5.5, 4.0]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&sheared, [0.0, 0.0, 1.0, 1.0]),
        CellCertificate::Outside
    );
    let reflection = Render2dAffineTransform::new(-1.0, 0.0, 0.0, 1.0, 3.0, 0.0).unwrap();
    let reflected = builder.affine(rect, reflection, &[0, 2]).unwrap();
    assert_eq!(
        certify(&reflected, [1.5, 0.5, 2.0, 1.0]),
        CellCertificate::Inside
    );
    let singular = Render2dAffineTransform::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
    let singular_region = builder
        .affine(Arc::clone(&sheared), singular, &[0, 3])
        .unwrap();
    assert_eq!(
        certify(&singular_region, [1.0, 1.0, 2.0, 2.0]),
        CellCertificate::Unknown
    );
}

#[test]
fn actual_f1_rectangle_is_queried_before_disposable_lyon_triangles() {
    use crate::composition_2d::{
        Render2dBrush, Render2dClip, Render2dColorRgba8, Render2dOpacity, Render2dRect,
        Render2dShape,
    };
    use crate::runtime::execution_2d::vector;
    let rectangle = Render2dShape::rect(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap());
    let shear = Render2dAffineTransform::new(1.0, 0.0, 0.5, 1.0, 4.0, 3.0).unwrap();
    let item = Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: rectangle.clone(),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        shear,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    );
    let mesh = vector::neutral_support(&item, 0, 4.0, 1_048_576)
        .unwrap()
        .expect("a real F1 rectangle tessellates");
    let mut graph = builder();
    let region = graph.source_from_item(mesh, &item, &[0, 0]).unwrap();
    assert_eq!(
        certify(&region, [4.5, 3.5, 4.6, 3.6]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&region, [0.0, 0.0, 1.0, 1.0]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&region, [4.0, 3.0, 4.0, 3.0]),
        CellCertificate::Unknown
    );

    let clip = Render2dClip::new(
        Render2dShape::rect(Render2dRect::new(0.0, 0.0, 0.25, 0.25).unwrap()),
        Render2dAffineTransform::IDENTITY,
    );
    let clipped = Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: rectangle,
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        shear,
        vec![clip],
        Render2dOpacity::TRANSPARENT,
    );
    // A deliberately independent physical mesh cannot authorize an Inside
    // result through a real clip without first proving its geometry.
    let physical = support::NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: vec![
            [4.0, 3.0],
            [6.0, 3.0],
            [6.0, 5.0],
            [4.0, 3.0],
            [6.0, 5.0],
            [4.0, 5.0],
        ],
        bounds: [4.0, 3.0, 6.0, 5.0],
    };
    let clipped_region = graph.source_from_item(physical, &clipped, &[0, 1]).unwrap();
    assert_eq!(
        certify(&clipped_region, [4.5, 3.5, 4.6, 3.6]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&clipped_region, [0.0, 0.0, 1.0, 1.0]),
        CellCertificate::Outside
    );
}

#[test]
fn original_f1_ellipse_uses_exact_semantics_instead_of_rectangle_mesh() {
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dOpacity, Render2dRect, Render2dShape,
    };
    let item = Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::ellipse(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap()),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    );
    let mut graph = builder();
    let mesh = support::NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: vec![
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ],
        bounds: [0.0, 0.0, 2.0, 2.0],
    };
    let source = graph.source_from_item(mesh, &item, &[0, 0]).unwrap();
    assert_eq!(
        certify(&source, [0.5, 0.5, 1.5, 1.5]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&source, [-2.0, -2.0, -1.0, -1.0]),
        CellCertificate::Outside
    );
}

#[test]
fn group_ellipse_clip_certifies_inside_outside_and_budget_exhaustion() {
    use crate::composition_2d::{Render2dClip, Render2dRect, Render2dShape};
    let mut graph = builder();
    let source = rectangle(&mut graph, [0.0, 0.0, 6.0, 6.0], &[0, 0]);
    let clips = vec![Render2dClip::new(
        Render2dShape::ellipse(Render2dRect::new(1.0, 1.0, 4.0, 4.0).unwrap()),
        Render2dAffineTransform::IDENTITY,
    )];
    let clipped = graph.clip(source, &clips, &[0]).unwrap();
    assert_eq!(
        certify(&clipped, [2.75, 2.75, 3.25, 3.25]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&clipped, [1.1, 1.1, 1.2, 1.2]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&clipped, [5.0, 3.0, 5.0, 3.0]),
        CellCertificate::Unknown
    );
    let mut work = 2;
    assert_eq!(
        clipped.certify_cell(Cell::new([2.75, 2.75, 3.25, 3.25]).unwrap(), &mut work,),
        CellCertificate::Unknown
    );
    assert_eq!(work, 0);
}

#[test]
fn original_f1_convex_line_path_certifies_both_windings_and_edge_uncertainty() {
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dFillRule, Render2dOpacity, Render2dPath,
        Render2dPoint,
    };
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    let make = |points: [(f64, f64); 3], rule| {
        let shape = Render2dShape::path(
            Render2dPath::new(
                rule,
                vec![
                    Render2dPathCommand::MoveTo(p(points[0].0, points[0].1)),
                    Render2dPathCommand::LineTo(p(points[1].0, points[1].1)),
                    Render2dPathCommand::LineTo(p(points[2].0, points[2].1)),
                    Render2dPathCommand::Close,
                ],
            )
            .unwrap(),
        );
        Render2dItem::new(
            Render2dPrimitive::Fill {
                shape,
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        )
    };
    for (points, rule) in [
        (
            [(0.0, 0.0), (8.0, 0.0), (0.0, 8.0)],
            Render2dFillRule::NonZero,
        ),
        (
            [(0.0, 8.0), (8.0, 0.0), (0.0, 0.0)],
            Render2dFillRule::EvenOdd,
        ),
    ] {
        let item = make(points, rule);
        let mut graph = builder();
        // An intentionally misleading physical rectangle may not override
        // the original triangle's genuine diagonal boundary.
        let source = graph
            .source_from_item(
                support::NeutralMesh {
                    units_per_parent_logical_unit: 1.0,
                    triangles: vec![
                        [0.0, 0.0],
                        [8.0, 0.0],
                        [8.0, 8.0],
                        [0.0, 0.0],
                        [8.0, 8.0],
                        [0.0, 8.0],
                    ],
                    bounds: [0.0, 0.0, 8.0, 8.0],
                },
                &item,
                &[0],
            )
            .unwrap();
        assert_eq!(
            certify(&source, [1.0, 1.0, 2.0, 2.0]),
            CellCertificate::Inside
        );
        assert_eq!(
            certify(&source, [6.0, 6.0, 7.0, 7.0]),
            CellCertificate::Outside
        );
        assert_eq!(
            certify(&source, [3.9, 3.9, 4.1, 4.1]),
            CellCertificate::Unknown
        );
    }
}

#[test]
fn self_intersecting_star_certifies_evenodd_winding_not_local_convexity() {
    use crate::composition_2d::{Render2dFillRule, Render2dPath, Render2dPoint};
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    let star = Render2dShape::path(
        Render2dPath::new(
            Render2dFillRule::EvenOdd,
            vec![
                Render2dPathCommand::MoveTo(p(0.0, 10.0)),
                Render2dPathCommand::LineTo(p(-5.8778525, -8.0901699)),
                Render2dPathCommand::LineTo(p(9.5105652, 3.0901699)),
                Render2dPathCommand::LineTo(p(-9.5105652, 3.0901699)),
                Render2dPathCommand::LineTo(p(5.8778525, -8.0901699)),
                Render2dPathCommand::Close,
            ],
        )
        .unwrap(),
    );
    let mut visits = 4096;
    assert_eq!(
        certify_shape(
            &star,
            Cell::new([-0.1, -0.1, 0.1, 0.1]).unwrap(),
            &mut visits,
        ),
        CellCertificate::Outside
    );
    assert_eq!(
        certify_shape(
            &star,
            Cell::new([12.0, 12.0, 13.0, 13.0]).unwrap(),
            &mut visits,
        ),
        CellCertificate::Outside
    );
}

#[test]
fn authored_concave_and_holed_straight_paths_certify_actual_fill_rules_without_meshes() {
    use crate::composition_2d::{Render2dFillRule, Render2dPath, Render2dPoint};
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    let contour = |points: &[(f64, f64)], commands: &mut Vec<Render2dPathCommand>| {
        commands.push(Render2dPathCommand::MoveTo(p(points[0].0, points[0].1)));
        for &(x, y) in &points[1..] {
            commands.push(Render2dPathCommand::LineTo(p(x, y)));
        }
        commands.push(Render2dPathCommand::Close);
    };
    let query = |path: &Render2dPath, bounds| {
        let mut visits = 16_777_216;
        certify_path(path, Cell::new(bounds).unwrap(), &mut visits)
    };
    let mut l = Vec::new();
    contour(
        &[
            (0.0, 0.0),
            (8.0, 0.0),
            (8.0, 2.0),
            (2.0, 2.0),
            (2.0, 8.0),
            (0.0, 8.0),
        ],
        &mut l,
    );
    let concave = Render2dPath::new(Render2dFillRule::NonZero, l).unwrap();
    assert_eq!(
        query(&concave, [0.5, 0.5, 1.0, 1.0]),
        CellCertificate::Inside
    );
    assert_eq!(
        query(&concave, [5.0, 5.0, 6.0, 6.0]),
        CellCertificate::Outside
    );
    assert_eq!(
        query(&concave, [1.9, 1.9, 2.1, 2.1]),
        CellCertificate::Unknown
    );

    let outer = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
    let inner = [(3.0, 3.0), (7.0, 3.0), (7.0, 7.0), (3.0, 7.0)];
    let mut same = Vec::new();
    contour(&outer, &mut same);
    contour(&inner, &mut same);
    let evenodd = Render2dPath::new(Render2dFillRule::EvenOdd, same.clone()).unwrap();
    let nonzero = Render2dPath::new(Render2dFillRule::NonZero, same).unwrap();
    for path in [&evenodd, &nonzero] {
        assert_eq!(query(path, [1.0, 1.0, 2.0, 2.0]), CellCertificate::Inside);
        assert_eq!(
            query(path, [12.0, 5.0, 13.0, 6.0]),
            CellCertificate::Outside
        );
        assert_eq!(query(path, [2.9, 4.0, 3.1, 5.0]), CellCertificate::Unknown);
    }
    assert_eq!(
        query(&evenodd, [4.0, 4.0, 6.0, 6.0]),
        CellCertificate::Outside
    );
    assert_eq!(
        query(&nonzero, [4.0, 4.0, 6.0, 6.0]),
        CellCertificate::Inside
    );

    let mut opposite = Vec::new();
    contour(&outer, &mut opposite);
    contour(&inner.into_iter().rev().collect::<Vec<_>>(), &mut opposite);
    let opposite = Render2dPath::new(Render2dFillRule::NonZero, opposite).unwrap();
    assert_eq!(
        query(&opposite, [4.0, 4.0, 6.0, 6.0]),
        CellCertificate::Outside
    );

    // An open fill contour MUST close implicitly; neither clipping nor
    // physical tessellation is consulted.
    let open = Render2dPath::new(
        Render2dFillRule::EvenOdd,
        vec![
            Render2dPathCommand::MoveTo(p(0.0, 0.0)),
            Render2dPathCommand::LineTo(p(10.0, 0.0)),
            Render2dPathCommand::LineTo(p(0.0, 10.0)),
        ],
    )
    .unwrap();
    assert_eq!(query(&open, [1.0, 1.0, 2.0, 2.0]), CellCertificate::Inside);
    assert_eq!(query(&open, [8.0, 8.0, 9.0, 9.0]), CellCertificate::Outside);
}

fn shared_authored_path_scene_certificate(commands: Vec<Render2dPathCommand>) {
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dFillRule, Render2dOpacity, Render2dPath,
    };
    let path = Render2dPath::new(Render2dFillRule::EvenOdd, commands).unwrap();
    let item = Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::path(path),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    );
    // The disposable physical triangles DELIBERATELY disagree with F1 path
    // semantics. A proven inside/outside result must come from the path.
    let fake_mesh = support::NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: vec![
            [0.0, 0.0],
            [64.0, 0.0],
            [64.0, 64.0],
            [0.0, 0.0],
            [64.0, 64.0],
            [0.0, 64.0],
        ],
        bounds: [0.0, 0.0, 64.0, 64.0],
    };
    let mut graph = builder();
    let source = graph.source_from_item(fake_mesh, &item, &[0]).unwrap();
    let mut remaining = MAX_REGION_GRAPH_WORK;
    let report = certified_shared_tile_quadtree(&source, 64, 8, 0.1, 5_440, &mut remaining)
        .expect("source-authoritative shared tile classification must fit the unchanged budget");
    assert_eq!(
        report.inside_samples + report.outside_samples + report.unresolved_samples,
        64 * 64 * 16
    );
    assert!(
        report.inside_samples > 0,
        "no original-path interiors proven"
    );
    assert!(
        report.outside_samples > 0,
        "no original-path exteriors proven"
    );
    assert!(
        report.unresolved_samples > 0,
        "boundary uncertainty was erased"
    );
    assert!(report.queries < 2_000, "not enough tile-level sharing");
    let charged = MAX_REGION_GRAPH_WORK - remaining;
    assert!(
        charged < 4_000_000,
        "scene-shared feasibility must avoid per-site query cost"
    );
}

#[test]
fn concave_f1_path_whole_scene_gaussian_tiles_share_certified_support() {
    use crate::composition_2d::Render2dPoint;
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    shared_authored_path_scene_certificate(vec![
        Render2dPathCommand::MoveTo(p(8.0, 8.0)),
        Render2dPathCommand::LineTo(p(48.0, 8.0)),
        Render2dPathCommand::LineTo(p(48.0, 20.0)),
        Render2dPathCommand::LineTo(p(20.0, 20.0)),
        Render2dPathCommand::LineTo(p(20.0, 48.0)),
        Render2dPathCommand::LineTo(p(8.0, 48.0)),
        Render2dPathCommand::Close,
    ]);
}

#[test]
fn holed_f1_path_whole_scene_gaussian_tiles_share_certified_support() {
    use crate::composition_2d::Render2dPoint;
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    shared_authored_path_scene_certificate(vec![
        Render2dPathCommand::MoveTo(p(8.0, 8.0)),
        Render2dPathCommand::LineTo(p(56.0, 8.0)),
        Render2dPathCommand::LineTo(p(56.0, 56.0)),
        Render2dPathCommand::LineTo(p(8.0, 56.0)),
        Render2dPathCommand::Close,
        Render2dPathCommand::MoveTo(p(24.0, 24.0)),
        Render2dPathCommand::LineTo(p(40.0, 24.0)),
        Render2dPathCommand::LineTo(p(40.0, 40.0)),
        Render2dPathCommand::LineTo(p(24.0, 40.0)),
        Render2dPathCommand::Close,
    ]);
}

#[test]
fn linear_winding_certificate_fails_closed_at_exact_global_fuel_boundary() {
    use crate::composition_2d::{Render2dFillRule, Render2dPath, Render2dPoint};
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    let path = Render2dPath::new(
        Render2dFillRule::NonZero,
        vec![
            Render2dPathCommand::MoveTo(p(0.0, 0.0)),
            Render2dPathCommand::LineTo(p(8.0, 0.0)),
            Render2dPathCommand::LineTo(p(0.0, 8.0)),
            Render2dPathCommand::Close,
        ],
    )
    .unwrap();
    let cell = Cell::new([1.0, 1.0, 2.0, 2.0]).unwrap();
    let cost = path.commands().len() * 40;
    let mut insufficient = cost - 1;
    assert_eq!(
        certify_path(&path, cell, &mut insufficient),
        CellCertificate::Unknown
    );
    assert_eq!(insufficient, 0);
    let mut exact = cost;
    assert_eq!(
        certify_path(&path, cell, &mut exact),
        CellCertificate::Inside
    );
    assert_eq!(exact, 0);
}

#[test]
fn curved_f1_path_hull_and_multicontour_boundaries_remain_fail_closed() {
    use crate::composition_2d::{Render2dFillRule, Render2dPath, Render2dPoint};
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    let cubic = Render2dShape::path(
        Render2dPath::new(
            Render2dFillRule::NonZero,
            vec![
                Render2dPathCommand::MoveTo(p(0.0, 0.0)),
                Render2dPathCommand::CubicTo {
                    control1: p(0.0, 5.0),
                    control2: p(5.0, 5.0),
                    to: p(5.0, 0.0),
                },
                Render2dPathCommand::Close,
            ],
        )
        .unwrap(),
    );
    // The curved control-hull query has sufficient fuel. Explicit exhaustion
    // and a multicontour boundary still fail closed below.
    let mut visits = 4096;
    let cell = Cell::new([6.0, 1.0, 7.0, 2.0]).unwrap();
    assert_eq!(
        certify_shape(&cubic, cell, &mut visits),
        CellCertificate::Outside
    );
    assert_eq!(
        certify_shape(
            &cubic,
            Cell::new([1.0, 1.0, 2.0, 2.0]).unwrap(),
            &mut visits
        ),
        CellCertificate::Inside
    );
    let multi = Render2dShape::path(
        Render2dPath::new(
            Render2dFillRule::EvenOdd,
            vec![
                Render2dPathCommand::MoveTo(p(0.0, 0.0)),
                Render2dPathCommand::LineTo(p(4.0, 0.0)),
                Render2dPathCommand::LineTo(p(4.0, 4.0)),
                Render2dPathCommand::Close,
                Render2dPathCommand::MoveTo(p(1.0, 1.0)),
                Render2dPathCommand::LineTo(p(2.0, 1.0)),
                Render2dPathCommand::LineTo(p(2.0, 2.0)),
                Render2dPathCommand::Close,
            ],
        )
        .unwrap(),
    );
    assert_eq!(
        certify_shape(
            &multi,
            Cell::new([1.2, 1.2, 1.4, 1.4]).unwrap(),
            &mut visits
        ),
        CellCertificate::Unknown
    );
    assert_eq!(
        certify_shape(
            &multi,
            Cell::new([7.0, 7.0, 8.0, 8.0]).unwrap(),
            &mut visits
        ),
        CellCertificate::Outside
    );
    let mut exhausted = 1;
    assert_eq!(
        certify_shape(
            &multi,
            Cell::new([7.0, 7.0, 8.0, 8.0]).unwrap(),
            &mut exhausted
        ),
        CellCertificate::Unknown
    );
    assert_eq!(exhausted, 0);
}

#[test]
fn outer_effect_halo_does_not_round_down_or_invent_absence() {
    // This exact pair has a true halo greater than a nearest-rounded f64
    // result. Its certified outer reach MUST include the rounding residue.
    let sigma = 0.3;
    let spread = 0.2;
    let nearest = 3.0 * sigma + spread;
    let certified = certified_effect_outer_halo(spread, sigma).unwrap();
    assert!(certified > nearest);
    assert!(certified >= nearest.next_up());

    // With negative spread, only the finite Gaussian halo enlarges the
    // source; a negative disk erosion cannot enlarge source support.
    let eroded = certified_effect_outer_halo(-2.0, sigma).unwrap();
    assert!(eroded >= 3.0 * sigma);
    assert!(eroded < certified);

    let mut graph = builder();
    let source = rectangle(&mut graph, [0.0, 0.0, 1.0, 1.0], &[0, 0]);
    let outer = graph
        .effect(
            source,
            Render2dDropShadow::new(
                0.0,
                0.0,
                sigma,
                spread,
                crate::composition_2d::Render2dColorRgba8::TRANSPARENT,
            )
            .unwrap(),
            &[0],
        )
        .unwrap();
    assert_eq!(
        certify(&outer, [2.5, 0.25, 2.6, 0.3]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&outer, [1.0, 0.25, 1.0, 0.25]),
        CellCertificate::Unknown
    );
}

#[test]
fn bounded_gaussian_density_encloses_independent_reference_at_kernel_extremes() {
    let samples: [([f64; 2], f64); 3] = [
        ([0.0, 0.0], 1.0),
        ([1.0, 1.0], 0.36787944117144233),
        ([3.0, 3.0], 0.00012340980408667956),
    ];
    for (point, reference) in samples {
        let x = Interval::scalar(point[0]).unwrap();
        let y = Interval::scalar(point[1]).unwrap();
        let bounds = gaussian_density_enclosure(x, y).unwrap();
        if point == [0.0, 0.0] {
            assert_eq!([bounds.lo, bounds.hi], [1.0, 1.0]);
        } else {
            // The independent reference is rounded; permit one representable
            // unit of reference uncertainty rather than trusting host libm.
            assert!(
                bounds.lo <= reference.next_down() && bounds.hi >= reference.next_up(),
                "density enclosure failed for {point:?}: {}..{} vs {reference}",
                bounds.lo,
                bounds.hi
            );
        }
        assert!(
            bounds.hi - bounds.lo < 1.0e-7,
            "kernel density uncertainty must not consume the whole 8-bit budget"
        );
    }
}

#[test]
fn bounded_gaussian_density_keeps_cell_variation_and_numeric_error_distinct() {
    let x = Interval::new(-0.5, 0.5).unwrap();
    let y = Interval::new(-0.5, 0.5).unwrap();
    let bounds = gaussian_density_enclosure(x, y).unwrap();
    assert!(bounds.lo < 1.0 && bounds.hi >= 1.0);
    assert!(
        bounds.lo < 0.8,
        "spatial cell variation still needs refinement"
    );
    assert!(bounds.hi >= bounds.lo);
}

#[test]
fn finite_gaussian_certifies_full_and_empty_kernel_mass() {
    let mut builder = builder();
    let large = rectangle(&mut builder, [-10.0, -10.0, 10.0, 10.0], &[0]);
    let distant = rectangle(&mut builder, [10.0, 10.0, 11.0, 11.0], &[1]);
    let mut visits = 200_000;
    assert_eq!(
        certified_finite_gaussian_coverage(&large, [0.0, 0.0], 0.25, 16, &mut visits),
        Some([1.0, 1.0])
    );
    let mut visits = 200_000;
    assert_eq!(
        certified_finite_gaussian_coverage(&distant, [0.0, 0.0], 0.25, 16, &mut visits),
        Some([0.0, 0.0])
    );
}

#[test]
fn finite_gaussian_symmetry_halfplane_has_certified_half_coverage() {
    let mut builder = builder();
    let half = rectangle(&mut builder, [-10.0, -10.0, 0.0, 10.0], &[0]);
    let mut visits = 1_200_000;
    let [lo, hi] =
        certified_finite_gaussian_coverage(&half, [0.0, 0.0], 0.25, 64, &mut visits).unwrap();
    assert!(lo < 0.5 && hi > 0.5, "{lo}..{hi}");
    assert!(lo > 0.25 && hi < 0.75, "{lo}..{hi}");
    assert!(visits < 1_200_000);
}

#[test]
fn finite_gaussian_uncertainty_and_work_exhaustion_are_not_zero_alpha() {
    let mut builder = builder();
    let source = rectangle(&mut builder, [0.0, 0.0, 1.0, 1.0], &[0]);
    let mut visits = 1;
    assert_eq!(
        certified_finite_gaussian_coverage(&source, [0.5, 0.5], 0.25, 16, &mut visits),
        None
    );
    let mut visits = 200_000;
    let [lo, hi] =
        certified_finite_gaussian_coverage(&source, [1.0, 0.5], 0.25, 16, &mut visits).unwrap();
    assert!(lo >= 0.0 && hi <= 1.0 && lo < hi);
}

#[test]
fn shared_gaussian_tiles_certify_4x4_samples_from_original_f1_geometry() {
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dOpacity, Render2dRect, Render2dShape,
    };
    // Authoritative transparent F1 geometry. The deliberately overlarge
    // physical mesh is NOT allowed to certify support or coverage.
    let item = Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::rect(Render2dRect::new(18.0, 18.0, 28.0, 28.0).unwrap()),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    );
    let physical = support::NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: vec![
            [0.0, 0.0],
            [64.0, 0.0],
            [64.0, 64.0],
            [0.0, 0.0],
            [64.0, 64.0],
            [0.0, 64.0],
        ],
        bounds: [0.0, 0.0, 64.0, 64.0],
    };
    let mut graph = builder();
    let source = graph.source_from_item(physical, &item, &[0]).unwrap();
    let mut work = 20_000;
    let mut exact_inside_tiles = 0_usize;
    let mut exact_outside_tiles = 0_usize;
    let mut uncertain_tiles = 0_usize;
    let halo = 0.75;
    for tile_y in 0..8 {
        for tile_x in 0..8 {
            let x = tile_x as f64 * 8.0;
            let y = tile_y as f64 * 8.0;
            let centers = Cell::new([x + 0.125, y + 0.125, x + 7.875, y + 7.875]).unwrap();
            let result = certified_uniform_kernel_coverage(
                &source,
                centers,
                Interval::scalar(0.25).unwrap(),
                &mut work,
            );
            match result {
                Some([1.0, 1.0]) => exact_inside_tiles += 1,
                Some([0.0, 0.0]) => exact_outside_tiles += 1,
                None => uncertain_tiles += 1,
                _ => panic!("a shared tile may only certify exact zero or one"),
            }
            if let Some([lo, hi]) = result {
                assert_eq!(lo, hi);
                // Independently check EVERY 4x4 sample in the tile against
                // the exact rect⊕finite-Gaussian-square geometry.
                for py in 0..8 {
                    for px in 0..8 {
                        for sy in 0..4 {
                            for sx in 0..4 {
                                let cx = x + px as f64 + (sx as f64 + 0.5) / 4.0;
                                let cy = y + py as f64 + (sy as f64 + 0.5) / 4.0;
                                let certainly_inside = cx - halo > 18.0
                                    && cx + halo < 46.0
                                    && cy - halo > 18.0
                                    && cy + halo < 46.0;
                                let certainly_outside = cx + halo < 18.0
                                    || cx - halo > 46.0
                                    || cy + halo < 18.0
                                    || cy - halo > 46.0;
                                assert!(
                                    if lo == 1.0 {
                                        certainly_inside
                                    } else {
                                        certainly_outside
                                    },
                                    "tile ({tile_x},{tile_y}) falsely certified {lo}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(exact_inside_tiles, 4);
    assert_eq!(exact_outside_tiles, 48);
    assert_eq!(uncertain_tiles, 12);
    let certified_sample_sites = (exact_inside_tiles + exact_outside_tiles) * 8 * 8 * 16;
    assert_eq!(certified_sample_sites, 53_248);
    assert_eq!(
        certified_sample_sites + uncertain_tiles * 8 * 8 * 16,
        64 * 64 * 16
    );
    // 52*8*8*16 = 53,248 certified centers require just 64
    // source-region queries, not one Gaussian integral per center.
    // Every tile charges 256 numerical units plus one source-node visit.
    assert_eq!(20_000 - work, 64 * (GAUSSIAN_WEIGHT_WORK + 1));
}

#[test]
fn adaptively_subdivided_shared_tiles_account_for_all_4x4_centers() {
    let mut graph = builder();
    let region = rectangle(&mut graph, [18.0, 18.0, 46.0, 46.0], &[0]);
    let mut work = 200_000;
    let counts = certified_shared_tile_quadtree(&region, 64, 8, 0.25, 5_440, &mut work).unwrap();

    // Independent rectangular support oracle over the full pixel-center
    // envelope of each correlated 4x4 phase, not an alpha/sample mask.
    let mut independent = SharedTileCoverage::default();
    for py in 0..64 {
        for px in 0..64 {
            let x0 = px as f64 + 0.125 - 0.75;
            let x1 = px as f64 + 0.875 + 0.75;
            let y0 = py as f64 + 0.125 - 0.75;
            let y1 = py as f64 + 0.875 + 0.75;
            if x0 > 18.0 && x1 < 46.0 && y0 > 18.0 && y1 < 46.0 {
                independent.inside_samples += 16;
            } else if x1 < 18.0 || x0 > 46.0 || y1 < 18.0 || y0 > 46.0 {
                independent.outside_samples += 16;
            } else {
                independent.unresolved_samples += 16;
            }
        }
    }
    assert_eq!(counts.inside_samples, independent.inside_samples);
    assert_eq!(counts.outside_samples, independent.outside_samples);
    assert_eq!(counts.unresolved_samples, independent.unresolved_samples);
    assert_eq!(counts.inside_samples, 676 * 16);
    assert_eq!(counts.outside_samples, 3_196 * 16);
    assert_eq!(counts.unresolved_samples, 224 * 16);
    assert_eq!(counts.queries, 672);
    assert_eq!(200_000 - work, counts.queries * (GAUSSIAN_WEIGHT_WORK + 1));
    assert_eq!(
        counts.inside_samples + counts.outside_samples + counts.unresolved_samples,
        64 * 64 * 16
    );
}

#[test]
fn shared_tile_refinement_rejects_partial_node_or_work_admission() {
    let mut graph = builder();
    let region = rectangle(&mut graph, [18.0, 18.0, 46.0, 46.0], &[0]);
    let mut work = 200_000;
    assert_eq!(
        certified_shared_tile_quadtree(&region, 64, 8, 0.25, 64, &mut work),
        None,
        "the partial 64-tile budget cannot escape as an alpha certificate"
    );
    let mut work = GAUSSIAN_WEIGHT_WORK - 1;
    assert_eq!(
        certified_shared_tile_quadtree(&region, 64, 8, 0.25, 5_440, &mut work),
        None,
        "the shared first tile must not bypass its numerical admission"
    );
}

#[test]
fn shared_gaussian_tiles_fail_closed_at_boundaries_and_under_work_exhaustion() {
    let mut graph = builder();
    let source = rectangle(&mut graph, [18.0, 18.0, 46.0, 46.0], &[0]);
    let centers = Cell::new([16.125, 24.125, 23.875, 31.875]).unwrap();
    let sigma = Interval::scalar(0.25).unwrap();
    let mut work = 10_000;
    assert_eq!(
        certified_uniform_kernel_coverage(&source, centers, sigma, &mut work),
        None,
        "a mixed boundary tile cannot be treated as empty"
    );
    let outside = Cell::new([0.125, 0.125, 7.875, 7.875]).unwrap();
    let mut exhausted = GAUSSIAN_WEIGHT_WORK - 1;
    assert_eq!(
        certified_uniform_kernel_coverage(&source, outside, sigma, &mut exhausted),
        None
    );
    assert_eq!(exhausted, GAUSSIAN_WEIGHT_WORK - 1);
}

#[test]
fn adaptive_gaussian_refinement_proves_halfplane_with_finite_total_work() {
    let mut graph = builder();
    let half = rectangle(&mut graph, [-10.0, -10.0, 0.0, 10.0], &[0]);
    let mut fuel = 2_000_000;
    let (bounds, nodes) = certified_adaptive_finite_gaussian_coverage(
        &half,
        [0.0, 0.0],
        0.25,
        6,
        5_461,
        0.35,
        &mut fuel,
    )
    .unwrap();
    assert!(bounds[0] < 0.5 && bounds[1] > 0.5);
    assert!(bounds[1] - bounds[0] <= 0.35);
    assert!(nodes > 1 && nodes <= 5_461 && fuel < 2_000_000);
}

#[test]
fn original_quadratic_mixed_gaussian_boundary_has_independent_truncated_mass_oracle() {
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dFillRule, Render2dOpacity, Render2dPath,
        Render2dPoint,
    };
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    // Exact original quadratic: x(t)=10+40t, y(t)=10+60t(1-t)
    // ==> y=25-(3/80)(x-30)^2. Filled region lies BELOW the arch
    // and above the implicit chord at y=10. For sigma=1/4 centered
    // at the apex, normalized Gaussian coordinates X,Y in [-3,3] obey
    // Y <= -(3/320)X^2. An independent 1D Gaussian/CDF integration,
    // normalized on that exact finite 3-sigma square, gives
    // 0.496350400653573 (not an alpha from the physical mesh).
    let item = Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: Render2dShape::path(
                Render2dPath::new(
                    Render2dFillRule::NonZero,
                    vec![
                        Render2dPathCommand::MoveTo(p(10.0, 10.0)),
                        Render2dPathCommand::QuadraticTo {
                            control: p(30.0, 40.0),
                            to: p(50.0, 10.0),
                        },
                        Render2dPathCommand::Close,
                    ],
                )
                .unwrap(),
            ),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        Render2dAffineTransform::IDENTITY,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    );
    // Forged physical full-canvas mesh is deliberately inconsistent with
    // the original F1 curve. It must not affect this coverage bound.
    let fake = support::NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: vec![
            [0.0, 0.0],
            [64.0, 0.0],
            [64.0, 64.0],
            [0.0, 0.0],
            [64.0, 64.0],
            [0.0, 64.0],
        ],
        bounds: [0.0, 0.0, 64.0, 64.0],
    };
    let mut graph = builder();
    let region = graph.source_from_item(fake, &item, &[0]).unwrap();
    let reference = 0.496_350_400_653_573;
    let mut fuel = 16_777_216;
    let (bounds, nodes) = certified_adaptive_finite_gaussian_coverage(
        &region,
        [30.0, 25.0],
        0.25,
        6,
        5_461,
        0.35,
        &mut fuel,
    )
    .expect("bounded original F1 quadratic must admit a loose but honest Gaussian interval");
    assert!(
        bounds[0] < reference && reference < bounds[1],
        "a certified Gaussian interval must enclose the independent exact-curve mass"
    );
    assert!(bounds[1] - bounds[0] <= 0.35);
    assert!(nodes > 1 && nodes <= 5_461 && fuel < 16_777_216);

    // This deliberately stricter error target exercises the same exact
    // per-invocation budgets and bounded depth. Without a qualified tighter
    // integration proof, it must reject instead of returning a color/alpha.
    let mut tight_fuel = 16_777_216;
    assert_eq!(
        certified_adaptive_finite_gaussian_coverage(
            &region,
            [30.0, 25.0],
            0.25,
            6,
            5_461,
            0.005,
            &mut tight_fuel,
        ),
        None,
        "test-only Gaussian quadtree does not justify production-grade boundary precision"
    );
}

#[test]
fn adaptive_gaussian_never_uses_a_partial_work_budget_as_alpha() {
    let mut graph = builder();
    let source = rectangle(&mut graph, [0.0, 0.0, 1.0, 1.0], &[0]);
    let mut fuel = 1;
    assert_eq!(
        certified_adaptive_finite_gaussian_coverage(
            &source,
            [1.0, 0.5],
            0.25,
            6,
            5_461,
            0.35,
            &mut fuel
        ),
        None
    );
    let mut fuel = 2_000_000;
    assert_eq!(
        certified_adaptive_finite_gaussian_coverage(
            &source,
            [1.0, 0.5],
            0.25,
            6,
            1,
            0.01,
            &mut fuel
        ),
        None,
        "node exhaustion cannot silently certify narrow Gaussian coverage"
    );
}

#[test]
fn adaptive_gaussian_certifies_all_inside_and_outside_without_exhaustive_grid() {
    let mut graph = builder();
    let large = rectangle(&mut graph, [-10.0, -10.0, 10.0, 10.0], &[0]);
    let distant = rectangle(&mut graph, [10.0, 10.0, 11.0, 11.0], &[1]);
    let mut fuel = 2_000;
    assert_eq!(
        certified_adaptive_finite_gaussian_coverage(
            &large,
            [0.0, 0.0],
            0.25,
            6,
            5_461,
            0.01,
            &mut fuel
        ),
        Some(([1.0, 1.0], 1))
    );
    let mut exhausted = 1;
    assert_eq!(
        certified_adaptive_finite_gaussian_coverage(
            &large,
            [0.0, 0.0],
            0.25,
            6,
            5_461,
            0.01,
            &mut exhausted
        ),
        None,
        "even full-support shortcuts charge bounded interval work"
    );
    let mut fuel = 2_000;
    assert_eq!(
        certified_adaptive_finite_gaussian_coverage(
            &distant,
            [0.0, 0.0],
            0.25,
            6,
            5_461,
            0.01,
            &mut fuel
        ),
        Some(([0.0, 0.0], 1))
    );
}

#[test]
fn scaled_source_certificate_uses_outward_rounded_rectangle_division() {
    let mut builder = builder();
    let mesh = support::NeutralMesh {
        units_per_parent_logical_unit: 3.0,
        triangles: vec![
            [0.0, 0.0],
            [9.0, 0.0],
            [9.0, 9.0],
            [0.0, 0.0],
            [9.0, 9.0],
            [0.0, 9.0],
        ],
        bounds: [0.0, 0.0, 9.0, 9.0],
    };
    let source = builder.source(mesh, &[4, 0]).unwrap();
    assert_eq!(
        certify(&source, [1.0, 1.0, 2.0, 2.0]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&source, [-1.0, -1.0, -0.5, -0.5]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&source, [3.0, 1.0, 3.0, 2.0]),
        CellCertificate::Unknown
    );
}

#[test]
fn bounded_cell_query_never_treats_unknown_curve_or_exhaustion_as_empty() {
    let mut builder = builder();
    let rect = rectangle(&mut builder, [0.0, 0.0, 2.0, 2.0], &[0, 0]);
    let effect = Render2dDropShadow::new(
        0.0,
        0.0,
        0.25,
        1.0,
        crate::composition_2d::Render2dColorRgba8::TRANSPARENT,
    )
    .unwrap();
    let curved = builder.effect(rect, effect, &[0, 1]).unwrap();
    assert_eq!(
        certify(&curved, [0.5, 0.5, 1.5, 1.5]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&curved, [-4.0, -4.0, -3.0, -3.0]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&curved, [-0.9, -0.9, -0.8, -0.8]),
        CellCertificate::Unknown
    );
    let mut exhausted = 0;
    assert_eq!(
        curved.certify_cell(Cell::new([0.5, 0.5, 1.5, 1.5]).unwrap(), &mut exhausted),
        CellCertificate::Unknown,
    );
}
