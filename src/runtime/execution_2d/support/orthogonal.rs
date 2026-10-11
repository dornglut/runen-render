//! Certified continuous negative Euclidean erosion for orthogonal polygon unions.
//!
//! The input is already the TRUE exterior of the triangle union; this never
//! interprets source tessellation seams, bounds, cached alpha or an SDF as the
//! caster. A horizontal slice intersects the orthogonal exterior an even
//! number of times. Erosion subtracts the exact horizontal cross-sections
//! of Euclidean-radius capsules around each exposed boundary segment.
//!
//! The 2D cell area is bracketed by 1-Lipschitz radius offsets at a slice
//! midpoint. Best-first refinement stops ONLY when the certified uncertainty
//! is <= 1/2048 of the cell area; no sample-mask approximation is substituted.
use super::{UnionBoundaryEdge, mask_failure};
use crate::execution_2d::{
    Render2dExecutionError, Render2dSampleSpaceError,
};
use std::collections::BinaryHeap;

const MAX_ORTHOGONAL_EDGES: usize = 256;
const MAX_INTERVALS: usize = 32_768;
const MAX_GLOBAL_WORK: usize = 16_777_216;
const ERROR_TARGET: f64 = 1.0 / 2048.0;

fn resource(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail)
}
fn precision(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail)
}
fn charge(
    work: &mut usize,
    count: usize,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    let next = work
        .checked_add(count)
        .ok_or_else(|| resource(path, "orthogonal erosion work counter overflow"))?;
    if next > MAX_GLOBAL_WORK {
        return Err(resource(
            path,
            "orthogonal erosion exceeds bounded whole-scene work",
        ));
    }
    *work = next;
    Ok(())
}

pub(super) struct OrthogonalExterior<'a> {
    edges: &'a [UnionBoundaryEdge],
    y_events: Vec<f64>,
    charge_per_slice: usize,
}

#[derive(Debug, Clone, Copy)]
struct Slice {
    y0: f64,
    y1: f64,
    lower: f64,
    upper: f64,
}
impl Slice {
    fn uncertainty(self) -> f64 {
        self.upper - self.lower
    }
}
impl PartialEq for Slice {
    fn eq(&self, other: &Self) -> bool {
        self.y0 == other.y0 && self.y1 == other.y1
    }
}
impl Eq for Slice {}
impl PartialOrd for Slice {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Slice {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.uncertainty()
            .total_cmp(&other.uncertainty())
            .then_with(|| self.y0.total_cmp(&other.y0))
            .then_with(|| self.y1.total_cmp(&other.y1))
    }
}

impl<'a> OrthogonalExterior<'a> {
    /// None means the TRUE polygonal exterior is not orthogonal, so the
    /// existing general signed-distance integrator retains authority.
    pub(super) fn new(
        edges: &'a [UnionBoundaryEdge],
        path: &[usize],
        work: &mut usize,
    ) -> Result<Option<Self>, Render2dExecutionError> {
        if edges.is_empty() || edges.len() > MAX_ORTHOGONAL_EDGES {
            return Ok(None);
        }
        for &[a, b] in edges {
            if !a.iter().chain(b.iter()).all(|v| v.is_finite()) {
                return Err(precision(path, "orthogonal exterior edge is nonfinite"));
            }
            if (a[0] != b[0] && a[1] != b[1]) || a == b {
                return Ok(None);
            }
        }
        let levels = (usize::BITS - edges.len().leading_zeros()) as usize + 1;
        // One scanline traversal plus THREE bounded sorts (parity crossings
        // and the two interval envelopes); charge worst-case log comparisons.
        let charge_per_slice = edges
            .len()
            .checked_mul(3 * levels + 8)
            .ok_or_else(|| resource(path, "orthogonal slice work overflow"))?;
        charge(work, charge_per_slice, path)?;
        let mut y_events = Vec::new();
        y_events
            .try_reserve_exact(edges.len().saturating_mul(2))
            .map_err(|_| resource(path, "orthogonal exterior event allocation failed"))?;
        for &[a, b] in edges {
            y_events.extend([a[1], b[1]]);
        }
        y_events.sort_by(f64::total_cmp);
        y_events.dedup();
        Ok(Some(Self {
            edges,
            y_events,
            charge_per_slice,
        }))
    }

    fn forbidden_extent(
        radius: f64,
        distance: f64,
        path: &[usize],
    ) -> Result<Option<f64>, Render2dExecutionError> {
        if distance > radius {
            return Ok(None);
        }
        let squared = (radius - distance) * (radius + distance);
        if !squared.is_finite() || squared < 0.0 {
            return Err(precision(path, "orthogonal capsule extent is unrepresentable"));
        }
        Ok(Some(squared.sqrt()))
    }

    /// The original orthogonal interior is fixed between adjacent vertex
    /// y-events. The true forbidden disk-capsule slice at EVERY y in a slab
    /// lies between an inner and outer union of certified x intervals.
    /// This avoids the loose global radius +/- dy certificates that caused
    /// normal complete-scene rectangular edges to exhaust the work budget.
    fn covered_width(
        crossings: &[f64],
        forbidden: &[[f64; 2]],
        x0: f64,
        x1: f64,
    ) -> f64 {
        let mut allowed = 0.0;
        for pair in crossings.chunks_exact(2) {
            let left = pair[0].max(x0);
            let right = pair[1].min(x1);
            if right <= left {
                continue;
            }
            let mut cursor = left;
            for &[start, end] in forbidden {
                if end <= cursor || start >= right {
                    continue;
                }
                allowed += (start.max(cursor).min(right) - cursor).max(0.0);
                cursor = cursor.max(end).min(right);
                if cursor >= right {
                    break;
                }
            }
            allowed += (right - cursor).max(0.0);
        }
        allowed
    }

    fn slice(
        &self,
        y0: f64,
        y1: f64,
        x0: f64,
        x1: f64,
        radius: f64,
        work: &mut usize,
        path: &[usize],
    ) -> Result<Slice, Render2dExecutionError> {
        let mid = (y0 + y1) * 0.5;
        if !(y0 < mid && mid < y1) {
            return Err(precision(path, "orthogonal erosion cannot refine a slice"));
        }
        charge(work, self.charge_per_slice, path)?;
        let mut crossings = Vec::<f64>::new();
        let mut outer_forbidden = Vec::<[f64; 2]>::new();
        let mut inner_forbidden = Vec::<[f64; 2]>::new();
        crossings
            .try_reserve(self.edges.len())
            .map_err(|_| resource(path, "orthogonal crossing allocation failed"))?;
        outer_forbidden
            .try_reserve(self.edges.len())
            .map_err(|_| resource(path, "orthogonal outer-capsule allocation failed"))?;
        inner_forbidden
            .try_reserve(self.edges.len())
            .map_err(|_| resource(path, "orthogonal inner-capsule allocation failed"))?;
        for &[a, b] in self.edges {
            let vertical = a[0] == b[0];
            let (lo_x, hi_x, nearest_y, farthest_y) = if vertical {
                let lo = a[1].min(b[1]);
                let hi = a[1].max(b[1]);
                if mid >= lo && mid < hi {
                    crossings.push(a[0]);
                }
                // Exact min/max of distance(y,[lo,hi]) over [y0,y1].
                let nearest = (lo - y1).max(y0 - hi).max(0.0);
                let dy0 = (lo - y0).max(y0 - hi).max(0.0);
                let dy1 = (lo - y1).max(y1 - hi).max(0.0);
                (a[0], a[0], nearest, dy0.max(dy1))
            } else {
                let edge_y = a[1];
                // Exact min/max of |y-edge_y| over the complete slice.
                let nearest = if edge_y >= y0 && edge_y <= y1 {
                    0.0
                } else {
                    (edge_y - y0).abs().min((edge_y - y1).abs())
                };
                (
                    a[0].min(b[0]),
                    a[0].max(b[0]),
                    nearest,
                    (edge_y - y0).abs().max((edge_y - y1).abs()),
                )
            };
            if let Some(half) = Self::forbidden_extent(radius, nearest_y, path)? {
                let lo = lo_x - half;
                let hi = hi_x + half;
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(precision(path, "orthogonal outer capsule coordinates overflow"));
                }
                outer_forbidden.push([lo, hi]);
            }
            if let Some(half) = Self::forbidden_extent(radius, farthest_y, path)? {
                let lo = lo_x - half;
                let hi = hi_x + half;
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(precision(path, "orthogonal inner capsule coordinates overflow"));
                }
                inner_forbidden.push([lo, hi]);
            }
        }
        crossings.sort_by(f64::total_cmp);
        crossings.dedup();
        if !crossings.len().is_multiple_of(2) {
            return Err(precision(
                path,
                "orthogonal exterior scanline parity is uncertified",
            ));
        }
        outer_forbidden.sort_by(|a, b| a[0].total_cmp(&b[0]));
        inner_forbidden.sort_by(|a, b| a[0].total_cmp(&b[0]));
        // For each y, union(INNER) is contained in true forbidden(y),
        // which is contained in union(OUTER). Therefore the remaining
        // original-inside width is bounded by these two evaluations.
        let lower_width = Self::covered_width(&crossings, &outer_forbidden, x0, x1);
        let upper_width = Self::covered_width(&crossings, &inner_forbidden, x0, x1);
        let height = y1 - y0;
        let lower = lower_width * height;
        let upper = upper_width * height;
        if !lower.is_finite()
            || !upper.is_finite()
            || lower < -1.0e-9
            || upper > (x1 - x0) * height + 1.0e-9
            || lower > upper + 1.0e-9
        {
            return Err(precision(path, "orthogonal disk-slice certificate is invalid"));
        }
        Ok(Slice {
            y0,
            y1,
            lower,
            upper,
        })
    }

    /// Certified continuous area over one actual physical sample cell.
    /// Return value brackets the indicator's exact area to <=1/4096.
    pub(super) fn area(
        &self,
        cell: [f64; 4],
        radius: f64,
        work: &mut usize,
        path: &[usize],
    ) -> Result<f64, Render2dExecutionError> {
        let [x0, y0, x1, y1] = cell;
        if !cell.iter().all(|v| v.is_finite())
            || !(x0 < x1 && y0 < y1)
            || !radius.is_finite()
            || radius <= 0.0
        {
            return Err(precision(path, "orthogonal erosion sample is invalid"));
        }
        let area = (x1 - x0) * (y1 - y0);
        if !area.is_finite() || area <= 0.0 {
            return Err(precision(path, "orthogonal erosion cell area overflow"));
        }
        let mut pending = BinaryHeap::new();
        pending
            .try_reserve(self.y_events.len() + 1)
            .map_err(|_| resource(path, "orthogonal erosion slice allocation failed"))?;
        let mut cuts = Vec::new();
        cuts.try_reserve(self.y_events.len() + 2)
            .map_err(|_| resource(path, "orthogonal erosion event allocation failed"))?;
        cuts.push(y0);
        for &y in &self.y_events {
            if y > y0 && y < y1 {
                cuts.push(y);
            }
        }
        cuts.push(y1);
        let mut total_lower = 0.0;
        let mut total_upper = 0.0;
        for pair in cuts.windows(2) {
            let slab = self.slice(pair[0], pair[1], x0, x1, radius, work, path)?;
            total_lower += slab.lower;
            total_upper += slab.upper;
            pending.push(slab);
        }
        // Splitting at all exterior edge endpoints is essential: between
        // such events, the ORIGINAL orthogonal inside intervals are constant.
        let tolerance = area * ERROR_TARGET;
        while total_upper - total_lower > tolerance {
            let slab = pending
                .pop()
                .ok_or_else(|| precision(path, "orthogonal erosion uncertainty lost its queue"))?;
            let mid = (slab.y0 + slab.y1) * 0.5;
            if !(slab.y0 < mid && mid < slab.y1) {
                return Err(precision(
                    path,
                    "orthogonal erosion precision depth exhausted",
                ));
            }
            if pending.len() + 2 > MAX_INTERVALS {
                return Err(resource(
                    path,
                    "orthogonal erosion exceeds bounded slice count",
                ));
            }
            pending
                .try_reserve(2)
                .map_err(|_| resource(path, "orthogonal erosion slice growth failed"))?;
            let left = self.slice(slab.y0, mid, x0, x1, radius, work, path)?;
            let right = self.slice(mid, slab.y1, x0, x1, radius, work, path)?;
            total_lower += left.lower + right.lower - slab.lower;
            total_upper += left.upper + right.upper - slab.upper;
            pending.push(left);
            pending.push(right);
        }
        let fraction = (total_lower + total_upper) * 0.5 / area;
        if !fraction.is_finite() || !(-1.0e-9..=1.0 + 1.0e-9).contains(&fraction) {
            return Err(precision(path, "orthogonal erosion area is not normalized"));
        }
        Ok(fraction.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Exact exposed L boundary. The concave notch is real source EXTERIOR.
    const L: [UnionBoundaryEdge; 6] = [
        [[0.0, 0.0], [8.0, 0.0]],
        [[8.0, 0.0], [8.0, 4.0]],
        [[8.0, 4.0], [4.0, 4.0]],
        [[4.0, 4.0], [4.0, 8.0]],
        [[4.0, 8.0], [0.0, 8.0]],
        [[0.0, 8.0], [0.0, 0.0]],
    ];

    #[test]
    fn orthogonal_negative_disk_respects_concavity_and_quantified_area() {
        let mut work = 0;
        let shape = OrthogonalExterior::new(&L, &[0, 2], &mut work)
            .unwrap()
            .unwrap();
        assert_eq!(
            shape
                .area([1.0, 1.0, 2.0, 2.0], 0.5, &mut work, &[0, 2])
                .unwrap(),
            1.0
        );
        assert_eq!(
            shape
                .area([6.0, 6.0, 7.0, 7.0], 0.5, &mut work, &[0, 2])
                .unwrap(),
            0.0
        );
        let corner = shape
            .area([3.0, 3.0, 4.0, 4.0], 1.0, &mut work, &[0, 2])
            .unwrap();
        // The inside near a concave corner is a true quarter-disc of
        // forbidden area, not the bounding rectangle.
        assert!((corner - (1.0 - std::f64::consts::FRAC_PI_4)).abs() < 0.001);
    }

    #[test]
    fn hole_boundary_is_an_exterior_even_with_zero_alpha_children() {
        let ring: [UnionBoundaryEdge; 8] = [
            [[0.0, 0.0], [10.0, 0.0]],
            [[10.0, 0.0], [10.0, 10.0]],
            [[10.0, 10.0], [0.0, 10.0]],
            [[0.0, 10.0], [0.0, 0.0]],
            [[3.0, 3.0], [7.0, 3.0]],
            [[7.0, 3.0], [7.0, 7.0]],
            [[7.0, 7.0], [3.0, 7.0]],
            [[3.0, 7.0], [3.0, 3.0]],
        ];
        let mut work = 0;
        let shape = OrthogonalExterior::new(&ring, &[2], &mut work)
            .unwrap()
            .unwrap();
        assert_eq!(
            shape
                .area([4.0, 4.0, 5.0, 5.0], 0.5, &mut work, &[2])
                .unwrap(),
            0.0
        );
        assert_eq!(
            shape
                .area([1.0, 4.0, 2.0, 5.0], 0.5, &mut work, &[2])
                .unwrap(),
            1.0
        );
    }

    #[test]
    fn rejects_oblique_polygon_without_fabricating_orthogonal_membership() {
        let mut work = 0;
        assert!(
            OrthogonalExterior::new(&[[[0.0, 0.0], [1.0, 1.0]]], &[3], &mut work,)
                .unwrap()
                .is_none()
        );
    }
}
