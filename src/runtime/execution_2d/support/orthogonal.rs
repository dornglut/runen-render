//! Certified continuous negative Euclidean erosion for orthogonal polygon unions.
//!
//! The input is already the TRUE exterior of the triangle union; this never
//! interprets source tessellation seams, bounds, cached alpha or an SDF as the
//! caster. A horizontal slice intersects the orthogonal exterior an even
//! number of times. Erosion subtracts the exact horizontal cross-sections
//! of Euclidean-radius capsules around each exposed boundary segment.
//!
//! The 2D area is bracketed by each segment's exact minimum/maximum
//! capsule-slice extent over a bounded vertical interval. Best-first
//! refinement stops ONLY when certified uncertainty is <= 1/2048 of
//! cell area; no sampled-mask approximation is substituted.
use super::{
    PreparationBudget, PreparationWork, UnionBoundaryEdge, gaussian_axis_mass, mask_failure,
};
use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};
use std::collections::BinaryHeap;

const MAX_ORTHOGONAL_EDGES: usize = 256;
const MAX_INTERVALS: usize = 32_768;
// Bound scene-shared neutral crossing storage independently of source edge
// admission. Oversized arrangements keep the previously certified path.
const MAX_CACHED_CROSSINGS: usize = 16_384;
const MAX_GLOBAL_WORK: usize = 16_777_216;
const ERROR_TARGET: f64 = 1.0 / 2048.0;

fn resource(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail)
}
fn precision(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail)
}
fn charge(
    work: &mut dyn PreparationWork,
    count: usize,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    work.charge(
        count,
        MAX_GLOBAL_WORK,
        path,
        "orthogonal erosion exceeds bounded whole-scene work",
    )
}

pub(super) struct OrthogonalExterior<'a> {
    edges: &'a [UnionBoundaryEdge],
    y_events: Vec<f64>,
    charge_per_slice: usize,
    cached_charge_per_slice: usize,
    bands: Option<OrthogonalBands>,
}

/// Pure neutral-geometry arrangement shared by every physical sample in
/// this invocation. Band crossings depend ONLY on the exposed boundary and
/// vertex-height events, not x coverage, alpha, Gaussian or sample caches.
struct OrthogonalBands {
    offsets: Vec<usize>,
    crossings: Vec<f64>,
}
impl OrthogonalBands {
    fn at<'a>(&'a self, events: &[f64], y: f64) -> &'a [f64] {
        if events.len() < 2 || y < events[0] || y >= events[events.len() - 1] {
            return &[];
        }
        let index = events.partition_point(|height| *height <= y) - 1;
        &self.crossings[self.offsets[index]..self.offsets[index + 1]]
    }
}

/// Private physical row parameters; no source-neutral support authority.
pub(super) struct GaussianRow {
    pub(super) y: f64,
    pub(super) origin_x: f64,
    pub(super) width: usize,
    pub(super) radius: f64,
    pub(super) sigma: f64,
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
        self.cmp(other) == std::cmp::Ordering::Equal
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
            .then_with(|| self.lower.total_cmp(&other.lower))
            .then_with(|| self.upper.total_cmp(&other.upper))
    }
}

impl<'a> OrthogonalExterior<'a> {
    /// None means the TRUE polygonal exterior is not orthogonal, so the
    /// existing general signed-distance integrator retains authority.
    pub(super) fn new(
        edges: &'a [UnionBoundaryEdge],
        path: &[usize],
        work: &mut dyn PreparationWork,
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

        // Only eligible, small arrangements are cached. The construction
        // charge is admitted against a LOCAL candidate counter before any
        // cache allocations and committed only after construction succeeds.
        // If the scene budget cannot admit optional caching, the original
        // correct per-slice path remains available with its full work charge.
        let cached_charge_per_slice = edges
            .len()
            .checked_mul(2 * levels + 8)
            .and_then(|cost| cost.checked_add(levels))
            .ok_or_else(|| resource(path, "cached orthogonal slice work overflow"))?;
        let bands_count = y_events.len().saturating_sub(1);
        let max_inventory = bands_count.checked_mul(edges.len());
        let mut bands = None;
        if bands_count > 0 && max_inventory.is_some_and(|n| n <= MAX_CACHED_CROSSINGS) {
            let preparation = max_inventory
                .and_then(|n| n.checked_mul(levels + 2))
                .ok_or_else(|| resource(path, "cached orthogonal crossing preparation overflow"))?;
            let mut admitted = PreparationBudget::snapshot(work);
            if charge(&mut admitted, preparation, path).is_ok() {
                let mut offsets = Vec::new();
                offsets
                    .try_reserve_exact(bands_count + 1)
                    .map_err(|_| resource(path, "orthogonal band offsets allocation failed"))?;
                let mut crossings = Vec::new();
                crossings
                    .try_reserve_exact(max_inventory.unwrap_or(0))
                    .map_err(|_| resource(path, "orthogonal band crossings allocation failed"))?;
                offsets.push(0);
                for window in y_events.windows(2) {
                    let mid = window[0] + (window[1] - window[0]) * 0.5;
                    if !(window[0] < mid && mid < window[1]) {
                        // A numerically collapsed height interval cannot be
                        // made a certified cached band. Retain old behavior.
                        offsets.clear();
                        break;
                    }
                    let base = crossings.len();
                    for &[a, b] in edges {
                        if a[0] != b[0] {
                            continue;
                        }
                        let lo = a[1].min(b[1]);
                        let hi = a[1].max(b[1]);
                        if mid >= lo && mid < hi {
                            crossings.push(a[0]);
                        }
                    }
                    crossings[base..].sort_by(f64::total_cmp);
                    // Deduplicate ONLY this band's suffix. Global Vec
                    // dedup would accidentally merge the last crossing of
                    // an earlier y band with the first of the next band.
                    let mut unique = base;
                    for index in base..crossings.len() {
                        if unique == base || crossings[unique - 1] != crossings[index] {
                            crossings[unique] = crossings[index];
                            unique += 1;
                        }
                    }
                    crossings.truncate(unique);
                    // Each physical exterior crossing toggles once. The
                    // same true segment may appear more than once from
                    // coincident triangulation but must not be counted twice.
                    if !(crossings.len() - base).is_multiple_of(2) {
                        return Err(precision(
                            path,
                            "orthogonal cached exterior parity is uncertified",
                        ));
                    }
                    offsets.push(crossings.len());
                }
                if offsets.len() == bands_count + 1 {
                    work.record(admitted.used());
                    bands = Some(OrthogonalBands { offsets, crossings });
                }
            }
        }
        Ok(Some(Self {
            edges,
            y_events,
            charge_per_slice,
            cached_charge_per_slice,
            bands,
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
            return Err(precision(
                path,
                "orthogonal capsule extent is unrepresentable",
            ));
        }
        Ok(Some(squared.sqrt()))
    }

    /// The original orthogonal interior is fixed between adjacent vertex
    /// y-events. The true forbidden disk-capsule slice at EVERY y in a slab
    /// lies between an inner and outer union of certified x intervals.
    /// This avoids the loose global radius +/- dy certificates that caused
    /// normal complete-scene rectangular edges to exhaust the work budget.
    fn covered_width(crossings: &[f64], forbidden: &[[f64; 2]], x0: f64, x1: f64) -> f64 {
        let mut allowed = 0.0;
        for pair in crossings.as_chunks::<2>().0 {
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
        y: [f64; 2],
        x: [f64; 2],
        radius: f64,
        work: &mut dyn PreparationWork,
        path: &[usize],
    ) -> Result<Slice, Render2dExecutionError> {
        let [y0, y1] = y;
        let [x0, x1] = x;
        let mid = (y0 + y1) * 0.5;
        if !(y0 < mid && mid < y1) {
            return Err(precision(path, "orthogonal erosion cannot refine a slice"));
        }
        charge(
            work,
            if self.bands.is_some() {
                self.cached_charge_per_slice
            } else {
                self.charge_per_slice
            },
            path,
        )?;
        let mut crossings = Vec::<f64>::new();
        let mut outer_forbidden = Vec::<[f64; 2]>::new();
        let mut inner_forbidden = Vec::<[f64; 2]>::new();
        if self.bands.is_none() {
            crossings
                .try_reserve(self.edges.len())
                .map_err(|_| resource(path, "orthogonal crossing allocation failed"))?;
        }
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
                if self.bands.is_none() && mid >= lo && mid < hi {
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
                    return Err(precision(
                        path,
                        "orthogonal outer capsule coordinates overflow",
                    ));
                }
                outer_forbidden.push([lo, hi]);
            }
            if let Some(half) = Self::forbidden_extent(radius, farthest_y, path)? {
                let lo = lo_x - half;
                let hi = hi_x + half;
                if !lo.is_finite() || !hi.is_finite() {
                    return Err(precision(
                        path,
                        "orthogonal inner capsule coordinates overflow",
                    ));
                }
                inner_forbidden.push([lo, hi]);
            }
        }
        let crossings: &[f64] = if let Some(bands) = &self.bands {
            bands.at(&self.y_events, mid)
        } else {
            crossings.sort_by(f64::total_cmp);
            crossings.dedup();
            if !crossings.len().is_multiple_of(2) {
                return Err(precision(
                    path,
                    "orthogonal exterior scanline parity is uncertified",
                ));
            }
            &crossings
        };
        outer_forbidden.sort_by(|a, b| a[0].total_cmp(&b[0]));
        inner_forbidden.sort_by(|a, b| a[0].total_cmp(&b[0]));
        // For each y, union(INNER) is contained in true forbidden(y),
        // which is contained in union(OUTER). Therefore the remaining
        // original-inside width is bounded by these two evaluations.
        let lower_width = Self::covered_width(crossings, &outer_forbidden, x0, x1);
        let upper_width = Self::covered_width(crossings, &inner_forbidden, x0, x1);
        let height = y1 - y0;
        let lower = lower_width * height;
        let upper = upper_width * height;
        if !lower.is_finite()
            || !upper.is_finite()
            || lower < -1.0e-9
            || upper > (x1 - x0) * height + 1.0e-9
            || lower > upper + 1.0e-9
        {
            return Err(precision(
                path,
                "orthogonal disk-slice certificate is invalid",
            ));
        }
        Ok(Slice {
            y0,
            y1,
            lower,
            upper,
        })
    }

    /// Share exact exposed-boundary y sections across all x samples in a
    /// physical row. Gaussian alpha is disposable numerical realization;
    /// neither these intervals nor output values own semantic F1 support.
    /// Split Gaussian integration at real exterior events and disk tangencies.
    pub(super) fn gaussian_row(
        &self,
        query: GaussianRow,
        values: &mut [f64],
        work: &mut dyn PreparationWork,
        path: &[usize],
    ) -> Result<(), Render2dExecutionError> {
        let GaussianRow {
            y,
            origin_x,
            width,
            radius,
            sigma,
        } = query;
        let reach = 3.0 * sigma;
        let y0 = y - reach;
        let y1 = y + reach;
        if !y.is_finite()
            || !origin_x.is_finite()
            || !radius.is_finite()
            || radius <= 0.0
            || !sigma.is_finite()
            || sigma <= 0.0
            || !reach.is_finite()
            || !y0.is_finite()
            || !y1.is_finite()
            || y0 >= y1
            || !(origin_x + width as f64).is_finite()
            || width > super::MAX_NEUTRAL_MASK_SAMPLES
            || values.len() != width
        {
            return Err(precision(
                path,
                "orthogonal Gaussian row is unrepresentable",
            ));
        }
        let mut cuts = Vec::new();
        cuts.try_reserve(self.y_events.len() * 3 + 2)
            .map_err(|_| resource(path, "Gaussian y-event allocation failed"))?;
        cuts.extend([y0, y1]);
        for &e in &self.y_events {
            for cut in [e - radius, e, e + radius] {
                if cut > y0 && cut < y1 && cut.is_finite() {
                    cuts.push(cut);
                }
            }
        }
        cuts.sort_by(f64::total_cmp);
        cuts.dedup();
        // The caller has already admitted and allocated the complete
        // invocation-owned f64 coverage. Reusing its row slice avoids a
        // second width-sized allocation for every Gaussian row.
        values.fill(0.0);
        let mut crossings = Vec::<f64>::new();
        let mut forbidden = Vec::<[f64; 2]>::new();
        let mut allowed = Vec::<[f64; 2]>::new();
        crossings
            .try_reserve(self.edges.len())
            .map_err(|_| resource(path, "Gaussian crossing allocation failed"))?;
        forbidden
            .try_reserve(self.edges.len())
            .map_err(|_| resource(path, "Gaussian capsule allocation failed"))?;
        // At most one original interior span per crossing pair plus one
        // retained segment per exposed forbidden interval endpoint. Reserve
        // the complete bounded inventory before any infallible Vec::push.
        let max_intervals = self
            .edges
            .len()
            .checked_mul(2)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(|| resource(path, "Gaussian interval inventory overflow"))?;
        allowed
            .try_reserve_exact(max_intervals)
            .map_err(|_| resource(path, "Gaussian interval allocation failed"))?;
        let levels = (usize::BITS - self.edges.len().leading_zeros()) as usize + 1;
        let section_charge = self
            .edges
            .len()
            .checked_mul(self.edges.len() + 3 * levels + 16)
            .ok_or_else(|| resource(path, "Gaussian section work overflow"))?;

        for pair in cuts.windows(2) {
            let half = (pair[1] - pair[0]) * 0.5;
            if half <= 0.0 {
                continue;
            }
            let middle = pair[0] + half;
            charge(work, 32, path)?;
            let mass_y = gaussian_axis_mass(pair[0], pair[1], y, sigma);
            if !mass_y.is_finite() || mass_y < 0.0 {
                return Err(precision(path, "Gaussian y mass is invalid"));
            }
            if mass_y == 0.0 {
                continue;
            }
            // Normalize each slab's positive quadrature weights so fully
            // covered interiors preserve the admitted truncated y mass.
            let mut weights = [0.0; 16];
            let mut weight_total = 0.0;
            for (i, &node) in super::GAUSS_NODES.iter().enumerate() {
                let sample_y = node.mul_add(half, middle);
                let z = (sample_y - y) / sigma;
                weights[i] = super::GAUSS_WEIGHTS[i] * (-0.5 * z * z).exp();
                weight_total += weights[i];
            }
            if !weight_total.is_finite() || weight_total <= 0.0 {
                return Err(precision(path, "Gaussian y quadrature is invalid"));
            }
            for (i, &weight) in weights.iter().enumerate() {
                if weight == 0.0 {
                    continue;
                }
                charge(work, section_charge, path)?;
                let sample_y = super::GAUSS_NODES[i].mul_add(half, middle);
                crossings.clear();
                forbidden.clear();
                allowed.clear();
                for &[a, b] in self.edges {
                    let (lo, hi, dy) = if a[0] == b[0] {
                        let low = a[1].min(b[1]);
                        let high = a[1].max(b[1]);
                        if self.bands.is_none() && sample_y >= low && sample_y < high {
                            crossings.push(a[0]);
                        }
                        (a[0], a[0], (low - sample_y).max(sample_y - high).max(0.0))
                    } else {
                        (a[0].min(b[0]), a[0].max(b[0]), (a[1] - sample_y).abs())
                    };
                    if let Some(delta) = Self::forbidden_extent(radius, dy, path)? {
                        let left = lo - delta;
                        let right = hi + delta;
                        if !left.is_finite() || !right.is_finite() {
                            return Err(precision(path, "Gaussian capsule coordinates overflow"));
                        }
                        forbidden.push([left, right]);
                    }
                }
                let active: &[f64] = if let Some(bands) = &self.bands {
                    bands.at(&self.y_events, sample_y)
                } else {
                    crossings.sort_by(f64::total_cmp);
                    crossings.dedup();
                    &crossings
                };
                if !active.len().is_multiple_of(2) {
                    return Err(precision(path, "Gaussian exterior crossing parity invalid"));
                }
                forbidden.sort_by(|a, b| a[0].total_cmp(&b[0]));
                for interval in active.as_chunks::<2>().0 {
                    let mut cursor = interval[0];
                    for &[left, right] in &forbidden {
                        if right <= cursor || left >= interval[1] {
                            continue;
                        }
                        if left > cursor {
                            allowed.push([cursor, left.min(interval[1])]);
                        }
                        cursor = cursor.max(right).min(interval[1]);
                        if cursor >= interval[1] {
                            break;
                        }
                    }
                    if cursor < interval[1] {
                        allowed.push([cursor, interval[1]]);
                    }
                }
                let y_mass = mass_y * weight / weight_total;
                for (column, value) in values.iter_mut().enumerate() {
                    charge(work, 1, path)?;
                    let sample_x = origin_x + column as f64 + 0.5;
                    let window = [sample_x - reach, sample_x + reach];
                    let mut x_mass = 0.0;
                    for &[left, right] in &allowed {
                        charge(work, 1, path)?;
                        let start = left.max(window[0]);
                        let end = right.min(window[1]);
                        if start >= end {
                            continue;
                        }
                        if start <= window[0] && end >= window[1] {
                            x_mass = 1.0;
                            break;
                        }
                        charge(work, 16, path)?;
                        x_mass += gaussian_axis_mass(start, end, sample_x, sigma);
                    }
                    *value += y_mass * x_mass.min(1.0);
                }
            }
        }
        for value in values {
            if !value.is_finite() || !(-1.0e-8..=1.0 + 1.0e-8).contains(value) {
                return Err(precision(path, "Gaussian row coverage not normalized"));
            }
            *value = value.clamp(0.0, 1.0);
        }
        Ok(())
    }

    /// Certified continuous area over one actual physical sample cell.
    /// Return the midpoint of a proved enclosure, with absolute area
    /// error <=1/4096 of this physical cell.
    pub(super) fn area(
        &self,
        cell: [f64; 4],
        radius: f64,
        work: &mut dyn PreparationWork,
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
            let slab = self.slice([pair[0], pair[1]], [x0, x1], radius, work, path)?;
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
            let left = self.slice([slab.y0, mid], [x0, x1], radius, work, path)?;
            let right = self.slice([mid, slab.y1], [x0, x1], radius, work, path)?;
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
    fn slice_heap_total_order_is_consistent_with_equality() {
        let first = Slice {
            y0: 0.0,
            y1: 1.0,
            lower: 0.25,
            upper: 0.75,
        };
        let different = Slice {
            y0: 0.0,
            y1: 1.0,
            lower: 0.0,
            upper: 0.75,
        };
        assert_ne!(first, different);
        assert_ne!(first.cmp(&different), std::cmp::Ordering::Equal);
        assert_eq!(first, first);
        assert_eq!(first.cmp(&first), std::cmp::Ordering::Equal);
    }

    #[test]
    fn shared_band_crossings_preserve_exact_concave_area_and_save_query_work() {
        let mut cached_work = 0;
        let cached = OrthogonalExterior::new(&L, &[6], &mut cached_work)
            .unwrap()
            .unwrap();
        let bands = cached.bands.as_ref().expect("bounded L band inventory");
        assert_eq!(bands.at(&cached.y_events, 2.0), &[0.0, 8.0]);
        assert_eq!(bands.at(&cached.y_events, 6.0), &[0.0, 4.0]);
        assert!(bands.at(&cached.y_events, -2.0).is_empty());
        assert!(bands.at(&cached.y_events, 10.0).is_empty());

        let mut fallback_work = 0;
        let mut fallback = OrthogonalExterior::new(&L, &[6], &mut fallback_work)
            .unwrap()
            .unwrap();
        fallback.bands = None;
        for _ in 0..8 {
            for &cell in &[
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [6.0, 6.0, 7.0, 7.0],
            ] {
                let a = cached.area(cell, 1.0, &mut cached_work, &[6]).unwrap();
                let b = fallback.area(cell, 1.0, &mut fallback_work, &[6]).unwrap();
                assert!((a - b).abs() <= ERROR_TARGET);
            }
        }
        assert!(
            cached_work < fallback_work,
            "shared band construction must amortize"
        );
    }

    #[test]
    fn cached_crossings_remain_source_neutral_across_a_hole_and_disjoint_bands() {
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
        let cached = OrthogonalExterior::new(&ring, &[5], &mut work)
            .unwrap()
            .unwrap();
        let bands = cached.bands.as_ref().unwrap();
        assert_eq!(bands.at(&cached.y_events, 1.0), &[0.0, 10.0]);
        assert_eq!(bands.at(&cached.y_events, 5.0), &[0.0, 3.0, 7.0, 10.0]);
        assert_eq!(bands.at(&cached.y_events, 9.0), &[0.0, 10.0]);
        assert_eq!(
            cached
                .area([4.0, 4.0, 5.0, 5.0], 0.5, &mut work, &[5])
                .unwrap(),
            0.0
        );
        assert_eq!(
            cached
                .area([1.0, 4.0, 2.0, 5.0], 0.5, &mut work, &[5])
                .unwrap(),
            1.0
        );
    }

    #[test]
    fn shared_gaussian_rows_reuse_exact_orthogonal_membership_without_filling_concavity() {
        let mut work = 0;
        let region = OrthogonalExterior::new(&L, &[3], &mut work)
            .unwrap()
            .unwrap();
        let mut near = [9.0; 6];
        region
            .gaussian_row(
                GaussianRow {
                    y: 2.0,
                    origin_x: 1.0,
                    width: 6,
                    radius: 0.5,
                    sigma: 0.1,
                },
                &mut near,
                &mut work,
                &[3],
            )
            .unwrap();
        let mut notch = [9.0; 6];
        region
            .gaussian_row(
                GaussianRow {
                    y: 6.0,
                    origin_x: 1.0,
                    width: 6,
                    radius: 0.5,
                    sigma: 0.1,
                },
                &mut notch,
                &mut work,
                &[3],
            )
            .unwrap();
        assert!((near[0] - 1.0).abs() < 0.001);
        assert!((near[5] - 1.0).abs() < 0.001);
        assert!((notch[0] - 1.0).abs() < 0.001);
        assert!(notch[5].abs() < 0.001);
        assert!(work < MAX_GLOBAL_WORK);
    }

    #[test]
    fn gaussian_row_budget_rejection_keeps_exact_authored_path() {
        let mut work = 0;
        let region = OrthogonalExterior::new(&L, &[8], &mut work)
            .unwrap()
            .unwrap();
        let mut exhausted = MAX_GLOBAL_WORK;
        assert!(matches!(
            region.gaussian_row(
                GaussianRow {
                    y: 2.0,
                    origin_x: 1.0,
                    width: 6,
                    radius: 0.5,
                    sigma: 0.1,
                },
                &mut [0.0; 6],
                &mut exhausted,
                &[8, 4],
            ),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [8, 4]
        ));
        assert_eq!(exhausted, MAX_GLOBAL_WORK);
    }

    #[test]
    fn gaussian_row_requires_exact_caller_admitted_span_without_mutation() {
        let mut work = 0;
        let region = OrthogonalExterior::new(&L, &[6], &mut work)
            .unwrap()
            .unwrap();
        let original_work = work;
        let mut incorrect = [7.0; 5];
        let error = region
            .gaussian_row(
                GaussianRow {
                    y: 2.0,
                    origin_x: 1.0,
                    width: 6,
                    radius: 0.5,
                    sigma: 0.1,
                },
                &mut incorrect,
                &mut work,
                &[6, 2],
            )
            .expect_err("a mismatched destination cannot allocate or rasterize");
        assert!(matches!(
            error,
            Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path), ..
            } if path == [6, 2]
        ));
        assert_eq!(incorrect, [7.0; 5]);
        assert_eq!(work, original_work);
    }

    #[test]
    fn orthogonal_area_resource_rejection_is_atomic_and_retains_owner_path() {
        let mut work = 0;
        let shape = OrthogonalExterior::new(&L, &[4, 1], &mut work)
            .unwrap()
            .unwrap();
        let mut used = MAX_GLOBAL_WORK;
        assert!(matches!(
            shape.area([3.0, 3.0, 4.0, 4.0], 1.0, &mut used, &[4, 7]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [4, 7]
        ));
        assert_eq!(used, MAX_GLOBAL_WORK);
    }

    #[test]
    fn orthogonal_evaluator_keeps_its_limit_under_legacy_policy() {
        let mut work = PreparationBudget::for_contribution(false);
        charge(&mut work, MAX_GLOBAL_WORK, &[4, 1]).unwrap();
        let error = charge(&mut work, 1, &[4, 2]).unwrap_err();
        assert!(matches!(error, Render2dExecutionError::SampleSpace {
            kind: Render2dSampleSpaceError::ResourceLimit,
            path: Some(path), ..
        } if path == [4, 2]));
        assert_eq!(work.used(), MAX_GLOBAL_WORK);
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
