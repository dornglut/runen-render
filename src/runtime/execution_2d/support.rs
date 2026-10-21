//! Disposable neutral primitive support geometry for ordinary F3F effects.
//!
//! These triangles and bounds are physical preparations of F1's one immutable
//! geometry authority. They are never public renderer semantics, paint alpha,
//! an MSDF/image sampled footprint, or a second scene representation.

mod area;
mod orthogonal;
use orthogonal::{GaussianRow, OrthogonalExterior};

#[derive(Debug)]
#[allow(dead_code, reason = "awaiting F3F group lowering")]
pub(super) struct NeutralMesh {
    /// Scale converting this mesh's original immediate-parent logical
    /// coordinates into its disposable stored tessellation coordinates.
    pub(super) units_per_parent_logical_unit: f64,
    pub(super) triangles: Vec<[f64; 2]>,
    pub(super) bounds: [f64; 4],
}

const MAX_GAUSSIAN_RADIUS_SAMPLES: u32 = 512;
// Covers repeated neutral grid traversal AND continuous support integration
// across every authored effect in one contribution, not just one shadow.
const MAX_AGGREGATE_SHADOW_PREPARATION_WORK: usize = 134_217_728;
pub(super) const MAX_COMPOSITIONAL_SHADOW_WORK: usize = 16_777_216;
const MAX_CERTIFIED_RECTANGLE_AXIS_WORK: usize = 16_777_216;

/// One counter carries the immutable contribution policy through every
/// geometric evaluator. Legacy geometry owners retain their established
/// ceiling; mixed symbolic contributions select their stricter policy once.
pub(super) trait PreparationWork {
    fn used(&self) -> usize;
    fn maximum(&self) -> usize;
    fn record(&mut self, used: usize);

    fn remaining(&self, evaluator_maximum: usize) -> usize {
        self.maximum()
            .min(evaluator_maximum)
            .saturating_sub(self.used())
    }

    fn charge(
        &mut self,
        cost: usize,
        evaluator_maximum: usize,
        path: &[usize],
        detail: &'static str,
    ) -> Result<(), crate::execution_2d::Render2dExecutionError> {
        let fail = || {
            mask_failure(
                path,
                crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                detail,
            )
        };
        let next = self.used().checked_add(cost).ok_or_else(fail)?;
        if next > self.maximum().min(evaluator_maximum) {
            return Err(fail());
        }
        self.record(next);
        Ok(())
    }
}

/// Existing geometry-only counters have the same legacy admission policy.
impl PreparationWork for usize {
    fn used(&self) -> usize {
        *self
    }
    fn maximum(&self) -> usize {
        MAX_AGGREGATE_SHADOW_PREPARATION_WORK
    }
    fn record(&mut self, used: usize) {
        *self = used;
    }
}

pub(super) struct PreparationBudget {
    used: usize,
    maximum: usize,
}

impl PreparationBudget {
    pub(super) fn for_contribution(has_symbolic: bool) -> Self {
        Self {
            used: 0,
            maximum: if has_symbolic {
                MAX_COMPOSITIONAL_SHADOW_WORK
            } else {
                MAX_AGGREGATE_SHADOW_PREPARATION_WORK
            },
        }
    }

    /// The completed source owner selects its policy exactly once. The paid
    /// counter becomes the initial counter, rather than charging it again to
    /// a temporary budget that could later be replaced.
    pub(super) fn from_prepared_sources(has_symbolic: bool, used: usize) -> Self {
        let mut budget = Self::for_contribution(has_symbolic);
        assert!(used <= MAX_COMPOSITIONAL_SHADOW_WORK);
        budget.used = used;
        budget
    }
}

/// Borrow one existing evaluator counter and the already-paid graph receipt.
/// Every accepted local debit also consumes graph fuel; neither counter resets.
pub(super) struct SharedPreparationWork<'a> {
    local: &'a mut usize,
    graph: &'a mut usize,
    maximum: usize,
}

impl<'a> SharedPreparationWork<'a> {
    pub(super) fn new(local: &'a mut usize, graph: &'a mut usize) -> Self {
        let remaining = MAX_COMPOSITIONAL_SHADOW_WORK.saturating_sub(*graph);
        let maximum = local
            .saturating_add(remaining)
            .min(MAX_AGGREGATE_SHADOW_PREPARATION_WORK);
        Self {
            local,
            graph,
            maximum,
        }
    }
}

impl PreparationWork for SharedPreparationWork<'_> {
    fn used(&self) -> usize {
        *self.local
    }
    fn maximum(&self) -> usize {
        self.maximum
    }
    fn record(&mut self, used: usize) {
        // PreparationWork::charge checks both limits before this commit.
        // No speculative construction may refund already executed work.
        let delta = used.checked_sub(*self.local).expect("monotone work debit");
        let graph = self.graph.checked_add(delta).expect("admitted graph debit");
        assert!(used <= self.maximum && graph <= MAX_COMPOSITIONAL_SHADOW_WORK);
        *self.local = used;
        *self.graph = graph;
    }
}

impl PreparationWork for PreparationBudget {
    fn used(&self) -> usize {
        self.used
    }
    fn maximum(&self) -> usize {
        self.maximum
    }
    fn record(&mut self, used: usize) {
        self.used = used;
    }
}

/// Retained f64 coverage belongs to the entire invocation, not one shadow.
/// Reject before reserving retained float coverage storage. Bounded temporary
/// geometry work may first certify that a shadow erodes completely.
fn admit_retained_coverage_samples(
    required: usize,
    remaining: usize,
    path: &[usize],
) -> Result<(), crate::execution_2d::Render2dExecutionError> {
    if required > remaining {
        return Err(mask_failure(
            path,
            crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            "aggregate retained shadow coverage exceeds preallocation budget",
        ));
    }
    Ok(())
}

fn charge_shadow_preparation_work(
    used: &mut dyn PreparationWork,
    cost: usize,
    path: &[usize],
) -> Result<(), crate::execution_2d::Render2dExecutionError> {
    // A rejected charge must leave the shared invocation counter intact.
    // The next sibling still owns the remaining work budget and its path.
    // Compute and check the candidate before committing the charge.
    used.charge(
        cost,
        MAX_AGGREGATE_SHADOW_PREPARATION_WORK,
        path,
        "aggregate shadow preparation work exceeds bounded admission",
    )
}

/// Historical finite truncated-Gaussian sampled-kernel comparator for
/// deterministic conformance ONLY; production uses continuous integration.
/// Private finite truncated-Gaussian *physical* kernel. These weights approximate
/// the F1 continuous convolution in a chosen group-parent sample frame; the
/// caller must independently bound complete sample-work and halo allocations.
#[cfg(test)]
#[derive(Debug)]
pub(super) struct GaussianKernel {
    pub(super) radius: usize,
    /// Normalized symmetrical weights indexed by absolute sample offset.
    pub(super) weights: Vec<f64>,
}

/// Generates a reproducible discrete physical approximation of the accepted
/// unit-integral truncated Gaussian, with an exact zero beyond 3 sigma.
///
/// `sample_scale` is physical 4x samples per **group-parent logical unit**,
/// before any ancestor affine. Applying these weights after flattening an
/// ancestor's shear/non-uniform scale would violate F1 morphology frames.
#[cfg(test)]
pub(super) fn gaussian_kernel(
    sigma: f64,
    sample_scale: f64,
    path: &[usize],
) -> Result<GaussianKernel, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};
    let fail = |kind, detail: &'static str| Render2dExecutionError::SampleSpace {
        kind,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    };
    if !sigma.is_finite() || sigma < 0.0 || !sample_scale.is_finite() || sample_scale <= 0.0 {
        return Err(fail(
            Render2dSampleSpaceError::PrecisionLimit,
            "unrepresentable group-parent shadow kernel",
        ));
    }
    if sigma == 0.0 {
        return Ok(GaussianKernel {
            radius: 0,
            weights: vec![1.0],
        });
    }
    let physical_sigma = sigma * sample_scale;
    let cutoff = physical_sigma * 3.0;
    if !cutoff.is_finite() || physical_sigma <= 0.0 {
        return Err(fail(
            Render2dSampleSpaceError::PrecisionLimit,
            "shadow blur sample scale is not representable",
        ));
    }
    if cutoff.ceil() > f64::from(MAX_GAUSSIAN_RADIUS_SAMPLES) {
        return Err(fail(
            Render2dSampleSpaceError::ResourceLimit,
            "shadow blur radius exceeds bounded sample kernel",
        ));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite nonnegative radius was bounded to 512 samples"
    )]
    let radius = cutoff.ceil() as usize;
    let mut weights = Vec::new();
    weights.try_reserve_exact(radius + 1).map_err(|_| {
        fail(
            Render2dSampleSpaceError::ResourceLimit,
            "shadow blur kernel allocation failed",
        )
    })?;
    for offset in 0..=radius {
        let distance = f64::from(u32::try_from(offset).expect("bounded blur sample radius"));
        let weight = if distance <= cutoff {
            (-0.5 * (distance / physical_sigma).powi(2)).exp()
        } else {
            0.0
        };
        weights.push(weight);
    }
    let normalizer = weights[0] + 2.0 * weights.iter().skip(1).sum::<f64>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err(fail(
            Render2dSampleSpaceError::PrecisionLimit,
            "shadow blur kernel normalization failed",
        ));
    }
    for weight in &mut weights {
        *weight /= normalizer;
    }
    Ok(GaussianKernel { radius, weights })
}

/// Private finite binary sample mask on one group-parent aligned sample lattice.
///
/// The origin is signed so neutral off-canvas sources survive until final
/// effects and conservative target intersection. A cell is 0 or 255; these
/// samples approximate, and never replace, continuous F1 Euclidean support.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code, reason = "F3F masks await the group-effect compositor")]
pub(super) struct NeutralMask {
    pub(super) origin_x: i64,
    pub(super) origin_y: i64,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) samples: Vec<u8>,
}

const MAX_NEUTRAL_MASK_SAMPLES: usize = 1_048_576;
const MAX_SPREAD_RADIUS_SAMPLES: u32 = 512;

fn mask_failure(
    path: &[usize],
    kind: crate::execution_2d::Render2dSampleSpaceError,
    detail: &'static str,
) -> crate::execution_2d::Render2dExecutionError {
    crate::execution_2d::Render2dExecutionError::SampleSpace {
        kind,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    }
}

fn filled<T: Clone>(
    count: usize,
    value: T,
    path: &[usize],
) -> Result<Vec<T>, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(count).map_err(|_| {
        mask_failure(
            path,
            Render2dSampleSpaceError::ResourceLimit,
            "neutral mask allocation failed",
        )
    })?;
    if std::mem::size_of::<T>() != 0 && buffer.capacity() > count {
        return Err(mask_failure(
            path,
            Render2dSampleSpaceError::ResourceLimit,
            "neutral mask allocated capacity exceeds admission",
        ));
    }
    buffer.resize(count, value);
    Ok(buffer)
}

fn as_f64(value: usize) -> f64 {
    // Sample area and side are bounded to <= 1_048_576 before any transform.
    f64::from(u32::try_from(value).expect("bounded neutral mask sample index"))
}

/// Squared exact Euclidean distance to the nearest finite seed in one row.
/// The parabolic lower envelope is linear in row length; no O(radius²)
/// neighborhood scan, axis-box kernel, or color-to-support conversion.
fn squared_distance_1d(
    values: &[f64],
    distances: &mut [f64],
    sites: &mut [usize],
    boundaries: &mut [f64],
) {
    let mut count = 0_usize;
    for q in 0..values.len() {
        if !values[q].is_finite() {
            continue;
        }
        let qf = as_f64(q);
        let mut intersection = f64::NEG_INFINITY;
        while count != 0 {
            let p = sites[count - 1];
            let pf = as_f64(p);
            intersection = ((values[q] + qf * qf) - (values[p] + pf * pf)) / (2.0 * (qf - pf));
            if intersection > boundaries[count - 1] {
                break;
            }
            count -= 1;
        }
        if count == 0 {
            intersection = f64::NEG_INFINITY;
        }
        sites[count] = q;
        boundaries[count] = intersection;
        count += 1;
    }
    if count == 0 {
        distances.fill(f64::INFINITY);
        return;
    }
    let mut current = 0_usize;
    for (q, value) in distances.iter_mut().enumerate() {
        let qf = as_f64(q);
        while current + 1 < count && boundaries[current + 1] < qf {
            current += 1;
        }
        let delta = qf - as_f64(sites[current]);
        *value = delta.mul_add(delta, values[sites[current]]);
    }
}

fn squared_distance_2d(
    samples: &[u8],
    width: usize,
    height: usize,
    seeds_present: bool,
    path: &[usize],
) -> Result<Vec<f64>, crate::execution_2d::Render2dExecutionError> {
    let count = width * height; // checked by calling mask admission
    let mut horizontal = filled(count, 0.0_f64, path)?;
    let mut output = filled(count, 0.0_f64, path)?;
    let side = width.max(height);
    let mut values = filled(side, 0.0_f64, path)?;
    let mut results = filled(side, 0.0_f64, path)?;
    let mut sites = filled(side, 0_usize, path)?;
    let mut boundaries = filled(side, 0.0_f64, path)?;
    for y in 0..height {
        let offset = y * width;
        for x in 0..width {
            values[x] = if (samples[offset + x] != 0) == seeds_present {
                0.0
            } else {
                f64::INFINITY
            };
        }
        squared_distance_1d(
            &values[..width],
            &mut results[..width],
            &mut sites[..width],
            &mut boundaries[..width],
        );
        horizontal[offset..offset + width].copy_from_slice(&results[..width]);
    }
    for x in 0..width {
        for y in 0..height {
            values[y] = horizontal[y * width + x];
        }
        squared_distance_1d(
            &values[..height],
            &mut results[..height],
            &mut sites[..height],
            &mut boundaries[..height],
        );
        for y in 0..height {
            output[y * width + x] = results[y];
        }
    }
    Ok(output)
}

/// Rasterizes disposable neutral triangles in one explicitly chosen
/// **group-parent** continuous coordinate frame, not final-target RGBA.
///
/// A globally anchored 4x correlated sample lattice avoids tile-dependent
/// coverage shifts. Caller MUST provide triangles in this frame *before*
/// applying ancestor transforms, spread or blur; do not pass preclipped
/// final-output geometry or infer this support from paint alpha.
#[allow(
    dead_code,
    reason = "awaiting F3F group support and compositor integration"
)]
pub(super) fn rasterize_neutral_mesh(
    mesh: &NeutralMesh,
    samples_per_logical_unit: f64,
    path: &[usize],
) -> Result<Option<NeutralMask>, crate::execution_2d::Render2dExecutionError> {
    rasterize_mesh_with_positive_spread(mesh, samples_per_logical_unit, 0.0, path, None)
}

/// Tests the true tessellated geometric distance at each parent-frame sample,
/// rather than dilating only previously occupied sample centers. A geometric
/// sliver with no original sample-center hit can still cast a spread shadow.
fn rasterize_mesh_with_positive_spread(
    mesh: &NeutralMesh,
    samples_per_logical_unit: f64,
    positive_spread: f64,
    path: &[usize],
    aggregate_work: Option<&mut dyn PreparationWork>,
) -> Result<Option<NeutralMask>, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    if !samples_per_logical_unit.is_finite() || samples_per_logical_unit <= 0.0 {
        return Err(precision("neutral sample spacing is invalid"));
    }
    if !mesh.units_per_parent_logical_unit.is_finite() || mesh.units_per_parent_logical_unit <= 0.0
    {
        return Err(precision("neutral mesh frame scale is invalid"));
    }
    let mesh_to_samples = samples_per_logical_unit / mesh.units_per_parent_logical_unit;
    if !mesh_to_samples.is_finite() || mesh_to_samples <= 0.0 {
        return Err(precision(
            "neutral mesh conversion to sample frame is not finite",
        ));
    }
    if !positive_spread.is_finite() || positive_spread < 0.0 {
        return Err(precision("positive geometric spread is invalid"));
    }
    if !mesh.triangles.len().is_multiple_of(3) {
        return Err(precision("neutral mesh triangle payload is incomplete"));
    }
    if mesh.triangles.is_empty() {
        return Ok(None);
    }
    if !mesh.bounds.iter().all(|value| value.is_finite())
        || !mesh
            .triangles
            .iter()
            .flatten()
            .all(|value| value.is_finite())
    {
        return Err(precision("neutral triangle bounds are not finite"));
    }
    let physical_spread = positive_spread * samples_per_logical_unit;
    let radius_squared = physical_spread * physical_spread;
    if !radius_squared.is_finite() {
        return Err(precision("geometric spread sample radius is not finite"));
    }
    let scaled = [
        mesh.bounds[0].mul_add(mesh_to_samples, -physical_spread),
        mesh.bounds[1].mul_add(mesh_to_samples, -physical_spread),
        mesh.bounds[2].mul_add(mesh_to_samples, physical_spread),
        mesh.bounds[3].mul_add(mesh_to_samples, physical_spread),
    ];
    if !scaled.iter().all(|value| value.is_finite())
        || scaled[0] >= scaled[2]
        || scaled[1] >= scaled[3]
        || scaled
            .iter()
            .any(|value| value.abs() > f64::from(i32::MAX) / 2.0)
    {
        return Err(precision(
            "neutral sample lattice extent is not representable",
        ));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite bounded sample edges are narrowed after exact i32-range admission"
    )]
    let edges = [
        scaled[0].floor() as i32,
        scaled[1].floor() as i32,
        scaled[2].ceil() as i32,
        scaled[3].ceil() as i32,
    ];
    let width = usize::try_from(i64::from(edges[2]) - i64::from(edges[0]))
        .map_err(|_| resource("neutral sample width overflow"))?;
    let height = usize::try_from(i64::from(edges[3]) - i64::from(edges[1]))
        .map_err(|_| resource("neutral sample height overflow"))?;
    let cells = width
        .checked_mul(height)
        .ok_or_else(|| resource("neutral sample area overflow"))?;
    if cells == 0 || cells > MAX_NEUTRAL_MASK_SAMPLES {
        return Err(resource(
            "neutral geometry sample grid exceeds bounded area",
        ));
    }
    let triangles = mesh.triangles.len() / 3;
    let work = cells
        .checked_mul(triangles)
        .ok_or_else(|| resource("neutral triangle/sample work overflow"))?;
    if work > 16_777_216 {
        return Err(resource(
            "neutral triangle/sample work exceeds the bounded budget",
        ));
    }
    if let Some(used) = aggregate_work {
        charge_shadow_preparation_work(used, work, path)?;
    }
    let mut samples = filled(cells, 0_u8, path)?;
    for y in 0..height {
        let sample_y = f64::from(edges[1]) + as_f64(y) + 0.5;
        for x in 0..width {
            let sample_x = f64::from(edges[0]) + as_f64(x) + 0.5;
            let p = [sample_x, sample_y];
            for tri in mesh.triangles.as_chunks::<3>().0 {
                let t = tri.map(|[x, y]| [x * mesh_to_samples, y * mesh_to_samples]);
                let ab = orient(t[0], t[1], p);
                let bc = orient(t[1], t[2], p);
                let ca = orient(t[2], t[0], p);
                let area = orient(t[0], t[1], t[2]);
                if area == 0.0 {
                    continue;
                }
                let inside =
                    (ab >= 0.0 && bc >= 0.0 && ca >= 0.0) || (ab <= 0.0 && bc <= 0.0 && ca <= 0.0);
                if inside
                    || (positive_spread > 0.0
                        && (distance_squared_to_segment(p, t[0], t[1]) <= radius_squared
                            || distance_squared_to_segment(p, t[1], t[2]) <= radius_squared
                            || distance_squared_to_segment(p, t[2], t[0]) <= radius_squared))
                {
                    samples[y * width + x] = u8::MAX;
                    break;
                }
            }
        }
    }
    Ok(Some(NeutralMask {
        origin_x: i64::from(edges[0]),
        origin_y: i64::from(edges[1]),
        width,
        height,
        samples,
    }))
}

fn orient(a: [f64; 2], b: [f64; 2], p: [f64; 2]) -> f64 {
    (b[0] - a[0]).mul_add(p[1] - a[1], -((b[1] - a[1]) * (p[0] - a[0])))
}

/// Closest-point metric in the same physical sample frame as the triangle.
/// This is a Euclidean segment metric, not a box distance or cached paint alpha.
fn distance_squared_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let delta = [b[0] - a[0], b[1] - a[1]];
    let squared_length = delta[0].mul_add(delta[0], delta[1] * delta[1]);
    let t = if squared_length > 0.0 {
        ((p[0] - a[0]).mul_add(delta[0], (p[1] - a[1]) * delta[1]) / squared_length).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let dx = p[0] - delta[0].mul_add(t, a[0]);
    let dy = p[1] - delta[1].mul_add(t, a[1]);
    dx.mul_add(dx, dy * dy)
}

/// Private linear alpha coverage from one sampled neutral shadow support.
/// All values are finite, normalized, and independent of authored paint alpha.
#[derive(Debug)]
#[allow(dead_code, reason = "awaiting unified F3F painter integration")]
pub(super) struct NeutralCoverage {
    pub(super) origin_x: i64,
    pub(super) origin_y: i64,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) values: Vec<f64>,
}

/* Continuous area reconstruction for a disposable parent-frame sample cell.
 *
 * At any open horizontal slab between triangle vertices, edge crossings and
 * cell-boundary crossings, each triangle's horizontal interval endpoints are
 * affine in y. Their union width is therefore affine: the midpoint rule
 * integrates that slab exactly (apart from bounded f64 rounding).
 *
 * Unlike point samples or summed triangle areas, this computes the UNION:
 * overlapping stroke triangles cannot inflate neutral coverage.
 */
/// Bound the one event inventory before either area or Gaussian consumption.
const MAX_NEUTRAL_UNION_EVENTS: usize = 65_536;

/// The one bounded event owner serves both the original cell integrator and
/// its shared-grid consumer. Growth pays old-word relocation before reserve.
fn push_union_event(
    events: &mut Vec<f64>,
    value: f64,
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<(), crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    if events.len() >= MAX_NEUTRAL_UNION_EVENTS {
        return Err(resource("neutral union event inventory exceeds admission"));
    }
    let relocation = if events.len() == events.capacity() {
        events.len()
    } else {
        0
    };
    let cost = relocation
        .checked_mul(2)
        .and_then(|n| n.checked_add(16))
        .ok_or_else(|| resource("neutral union event growth work overflow"))?;
    charge_shadow_preparation_work(work, cost, path)?;
    if events.len() == events.capacity() {
        let capacity = events
            .capacity()
            .saturating_mul(2)
            .clamp(8, MAX_NEUTRAL_UNION_EVENTS);
        events
            .try_reserve_exact(capacity - events.len())
            .map_err(|_| resource("neutral union event allocation failed"))?;
        if events.capacity() > MAX_NEUTRAL_UNION_EVENTS {
            return Err(resource("neutral union event capacity exceeds admission"));
        }
    }
    events.push(value);
    Ok(())
}

/// Single geometric source of truth for continuous neutral triangle-union
/// cross-sections. The area and Gaussian integrators consume exactly the same
/// clipped contours, topology-change events and disjoint horizontal intervals.
/// Only their smooth integration weights differ.
struct NeutralUnionSweep {
    active: Vec<[[f64; 2]; 3]>,
    events: Vec<f64>,
    x0: f64,
    x1: f64,
    fully_covered: bool,
}

impl NeutralUnionSweep {
    fn new(
        triangles: &[[f64; 2]],
        bounds: [f64; 4],
        work_multiplier: usize,
        work: &mut dyn PreparationWork,
        path: &[usize],
    ) -> Result<Self, crate::execution_2d::Render2dExecutionError> {
        use crate::execution_2d::Render2dSampleSpaceError;
        let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
        let precision =
            |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
        let [x0, y0, x1, y1] = bounds;
        if !bounds.iter().all(|v| v.is_finite())
            || x0 >= x1
            || y0 >= y1
            || !triangles.len().is_multiple_of(3)
        {
            return Err(precision(
                "neutral scanline union bounds or triangle payload invalid",
            ));
        }
        // Pay the scan's fixed setup, then each executed source stage before
        // it runs. Disjoint triangles need only their bbox/degeneracy tests;
        // a whole-window proof can stop without inspecting later triangles.
        charge_shadow_preparation_work(work, 64, path)?;
        let mut active = Vec::<[[f64; 2]; 3]>::new();
        for tri in triangles.as_chunks::<3>().0 {
            charge_shadow_preparation_work(work, 64, path)?;
            let xs = [tri[0][0], tri[1][0], tri[2][0]];
            let ys = [tri[0][1], tri[1][1], tri[2][1]];
            if xs.iter().copied().fold(f64::INFINITY, f64::min) >= x1
                || xs.iter().copied().fold(f64::NEG_INFINITY, f64::max) <= x0
                || ys.iter().copied().fold(f64::INFINITY, f64::min) >= y1
                || ys.iter().copied().fold(f64::NEG_INFINITY, f64::max) <= y0
                || orient(tri[0], tri[1], tri[2]) == 0.0
            {
                continue;
            }
            // Up to four three-edge containment tests plus amortized active
            // Vec relocation, copied vertex words, push and loop control.
            charge_shadow_preparation_work(work, 256, path)?;
            // One convex triangle containing the whole integration window
            // proves union coverage of its interior, independent of siblings.
            if [[x0, y0], [x1, y0], [x1, y1], [x0, y1]].iter().all(|&p| {
                let a = orient(tri[0], tri[1], p);
                let b = orient(tri[1], tri[2], p);
                let c = orient(tri[2], tri[0], p);
                (a >= 0.0 && b >= 0.0 && c >= 0.0) || (a <= 0.0 && b <= 0.0 && c <= 0.0)
            }) {
                let mut events = Vec::new();
                push_union_event(&mut events, y0, work, path)?;
                push_union_event(&mut events, y1, work, path)?;
                return Ok(Self {
                    active: Vec::new(),
                    events,
                    x0,
                    x1,
                    fully_covered: true,
                });
            }
            if active.len() == active.capacity() {
                let capacity = active
                    .capacity()
                    .saturating_mul(2)
                    .max(4)
                    .min(triangles.len() / 3);
                active
                    .try_reserve_exact(capacity - active.len())
                    .map_err(|_| resource("neutral scanline triangle allocation failed"))?;
                if active.capacity() > capacity {
                    return Err(resource(
                        "neutral scanline triangle capacity exceeds admission",
                    ));
                }
            }
            active.push(*tri);
        }
        let charge = active
            .len()
            .checked_pow(3)
            .and_then(|n| n.checked_mul(work_multiplier))
            .ok_or_else(|| resource("neutral scanline arrangement work overflow"))?;
        work.charge(
            charge,
            MAX_AGGREGATE_SHADOW_PREPARATION_WORK,
            path,
            "neutral scanline arrangement exceeds bounded work",
        )?;
        let mut events = Vec::new();
        push_union_event(&mut events, y0, work, path)?;
        push_union_event(&mut events, y1, work, path)?;
        let mut edges = Vec::new();
        edges
            .try_reserve_exact(
                active
                    .len()
                    .checked_mul(3)
                    .ok_or_else(|| resource("neutral scanline edges overflow"))?,
            )
            .map_err(|_| resource("neutral scanline edge allocation failed"))?;
        if edges.capacity() > active.len() * 3 {
            return Err(resource("neutral scanline edge capacity exceeds admission"));
        }
        for triangle in &active {
            for i in 0..3 {
                let a = triangle[i];
                let b = triangle[(i + 1) % 3];
                edges.push((a, b));
                if a[1] > y0 && a[1] < y1 {
                    push_union_event(&mut events, a[1], work, path)?;
                }
                if a[0] != b[0] {
                    for x in [x0, x1] {
                        let t = (x - a[0]) / (b[0] - a[0]);
                        if t > 0.0 && t < 1.0 {
                            let y = (b[1] - a[1]).mul_add(t, a[1]);
                            if y > y0 && y < y1 {
                                push_union_event(&mut events, y, work, path)?;
                            }
                        }
                    }
                }
            }
        }
        // The union's interval ordering changes only at real edge crossings.
        for i in 0..edges.len() {
            let (a, b) = edges[i];
            let r = [b[0] - a[0], b[1] - a[1]];
            for &(c, d) in edges.iter().skip(i + 1) {
                let v = [d[0] - c[0], d[1] - c[1]];
                let divisor = r[0].mul_add(v[1], -r[1] * v[0]);
                if divisor == 0.0 {
                    continue;
                }
                let difference = [c[0] - a[0], c[1] - a[1]];
                let t = difference[0].mul_add(v[1], -difference[1] * v[0]) / divisor;
                let u = difference[0].mul_add(r[1], -difference[1] * r[0]) / divisor;
                if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                    let y = r[1].mul_add(t, a[1]);
                    if y > y0 && y < y1 {
                        push_union_event(&mut events, y, work, path)?;
                    }
                }
            }
        }
        events.sort_unstable_by(f64::total_cmp);
        events.dedup();
        Ok(Self {
            active,
            events,
            x0,
            x1,
            fully_covered: false,
        })
    }

    /// Primitive scan, sorting, merging and row-allocation operations for one
    /// horizontal union query. All area/Gaussian schedules pay this same bound.
    fn interval_work(
        &self,
        path: &[usize],
    ) -> Result<usize, crate::execution_2d::Render2dExecutionError> {
        let n = self.active.len();
        n.checked_mul(128)
            .and_then(|linear| {
                n.checked_mul(n)
                    .and_then(|pairs| pairs.checked_mul(16))
                    .and_then(|sort| linear.checked_add(sort))
            })
            .and_then(|cost| cost.checked_add(64))
            .ok_or_else(|| {
                mask_failure(
                    path,
                    crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                    "neutral union interval work overflow",
                )
            })
    }

    /// Disjoint union intervals of immutable triangle interiors at this y.
    /// A single topology and clipping implementation feeds both integrators.
    fn intervals(
        &self,
        y: f64,
        path: &[usize],
    ) -> Result<Vec<[f64; 2]>, crate::execution_2d::Render2dExecutionError> {
        self.intervals_limit(y, true, path)
    }

    /// Horizontal one-sided limit. This also defines a slab whose endpoints
    /// are adjacent floats, without inventing or dropping its geometric area.
    fn intervals_limit(
        &self,
        y: f64,
        from_above: bool,
        path: &[usize],
    ) -> Result<Vec<[f64; 2]>, crate::execution_2d::Render2dExecutionError> {
        use crate::execution_2d::Render2dSampleSpaceError;
        let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
        let mut segments = Vec::<[f64; 2]>::new();
        segments
            .try_reserve_exact(self.active.len())
            .map_err(|_| resource("neutral union row allocation failed"))?;
        if segments.capacity() > self.active.len() {
            return Err(resource("neutral union row capacity exceeds admission"));
        }
        for triangle in &self.active {
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            for i in 0..3 {
                let a = triangle[i];
                let b = triangle[(i + 1) % 3];
                let crosses = if from_above {
                    (a[1] <= y && y < b[1]) || (b[1] <= y && y < a[1])
                } else {
                    (a[1] < y && y <= b[1]) || (b[1] < y && y <= a[1])
                };
                if crosses {
                    let x = (b[0] - a[0]).mul_add((y - a[1]) / (b[1] - a[1]), a[0]);
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                }
            }
            let start = min_x.max(self.x0);
            let end = max_x.min(self.x1);
            if end > start {
                segments.push([start, end]);
            }
        }
        segments.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let mut merged = Vec::<[f64; 2]>::new();
        merged
            .try_reserve_exact(segments.len())
            .map_err(|_| resource("neutral union intervals allocation failed"))?;
        if merged.capacity() > segments.len() {
            return Err(resource(
                "neutral union interval capacity exceeds admission",
            ));
        }
        for [start, end] in segments {
            if let Some(last) = merged.last_mut()
                && start <= last[1]
            {
                last[1] = last[1].max(end);
                continue;
            }
            merged.push([start, end]);
        }
        Ok(merged)
    }
}

/// Exact polygon-union area of one bounded unit lattice cell, independent of
/// final paint alpha and repeated tessellation triangle overlaps.
fn area_sample(
    triangles: &[[f64; 2]],
    x0: f64,
    y0: f64,
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<f64, crate::execution_2d::Render2dExecutionError> {
    let sweep = NeutralUnionSweep::new(triangles, [x0, y0, x0 + 1.0, y0 + 1.0], 8, work, path)?;
    if sweep.fully_covered {
        return Ok(1.0);
    }
    if sweep.active.is_empty() {
        return Ok(0.0);
    }
    // Prepay every slab's union scan/sort/merge and width integration. The
    // constructor's arrangement charge does not also pay these row queries.
    let interval_work = sweep.interval_work(path)?;
    let integration_work = sweep
        .active
        .len()
        .checked_mul(16)
        .and_then(|width| width.checked_add(interval_work))
        .and_then(|row| row.checked_add(32))
        .and_then(|row| row.checked_mul(sweep.events.len().saturating_sub(1)))
        .and_then(|cost| cost.checked_add(32))
        .ok_or_else(|| {
            mask_failure(
                path,
                crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                "continuous area slab work overflow",
            )
        })?;
    charge_shadow_preparation_work(work, integration_work, path)?;
    let mut area = 0.0;
    for slab in sweep.events.windows(2) {
        let height = slab[1] - slab[0];
        if height <= 0.0 {
            continue;
        }
        let y = slab[0] + height * 0.5;
        let width = sweep
            .intervals(y, path)?
            .iter()
            .map(|[a, b]| b - a)
            .sum::<f64>();
        area += width * height;
    }
    Ok(area.clamp(0.0, 1.0))
}

/// Area-aware coverage of an immutable parent-frame triangle union. A true
/// sub-sample caster contributes its *area*, not a fabricated binary center
/// hit or an alpha inferred from visible painting.
fn rasterize_area_coverage(
    mesh: &NeutralMesh,
    scale: f64,
    path: &[usize],
    aggregate_work: &mut dyn PreparationWork,
    remaining_coverage_samples: usize,
) -> Result<Option<NeutralCoverage>, crate::execution_2d::Render2dExecutionError> {
    let Some(grid) =
        rasterize_mesh_with_positive_spread(mesh, scale, 0.0, path, Some(aggregate_work))?
    else {
        return Ok(None);
    };
    admit_retained_coverage_samples(grid.samples.len(), remaining_coverage_samples, path)?;
    let transform = scale / mesh.units_per_parent_logical_unit;
    let mut triangles = filled(mesh.triangles.len(), [0.0; 2], path)?;
    for (out, &[x, y]) in triangles.iter_mut().zip(&mesh.triangles) {
        *out = [x * transform, y * transform];
    }
    let values = area::rasterize(&triangles, &grid, aggregate_work, path)?;
    Ok(Some(NeutralCoverage {
        origin_x: grid.origin_x,
        origin_y: grid.origin_y,
        width: grid.width,
        height: grid.height,
        values,
    }))
}

/// Fixed 16-point Gauss-Legendre rule. On an interval no wider than the
/// truncated 6σ kernel footprint its smooth Gaussian integrand has rapid,
/// reproducible convergence; contour/topology changes are explicitly split.
const GAUSS_NODES: [f64; 16] = [
    -0.9894009349916499,
    -0.9445750230732326,
    -0.8656312023878318,
    -0.755404408355003,
    -0.6178762444026438,
    -0.45801677765722737,
    -0.2816035507792589,
    -0.09501250983763744,
    0.09501250983763744,
    0.2816035507792589,
    0.45801677765722737,
    0.6178762444026438,
    0.755404408355003,
    0.8656312023878318,
    0.9445750230732326,
    0.9894009349916499,
];
const GAUSS_WEIGHTS: [f64; 16] = [
    0.027152459411754176,
    0.062253523938647456,
    0.0951585116824926,
    0.12462897125553407,
    0.1495959888165767,
    0.16915651939500265,
    0.18260341504492364,
    0.1894506104550686,
    0.1894506104550686,
    0.18260341504492364,
    0.16915651939500265,
    0.1495959888165767,
    0.12462897125553407,
    0.0951585116824926,
    0.062253523938647456,
    0.027152459411754176,
];
/// Upper bound in primitive evaluator operations, not machine instructions:
/// `exp` and `mul_add` each count as one primitive. Each quadrature node
/// performs eight arithmetic/intrinsic operations; sixteen units also cover
/// its two input loads, accumulation/store and bounded loop/index control.
/// The fixed 32 cover clipping, setup, normalization and caller finite/range
/// checks. Counting nodes alone is not a bound on their executed arithmetic.
const GAUSSIAN_AXIS_WORK: usize = 32 + 16 * GAUSS_NODES.len();
/// Clipped identity-axis setup, intersection, loop and stored result.
const IDENTITY_AXIS_WORK: usize = 32;
/// One x/y product, clamp, destination index/store and loop progression.
const COVERAGE_PRODUCT_WORK: usize = 16;

/// Analytic 1D integral of exp(-u²/2), u ∈ [-3, 3].
const GAUSS_3SIGMA_NORMALIZER: f64 = 2.499_860_889_483_094_7;

/// Integrate the continuous normalized Gaussian over a UNION of triangle
/// interiors, not cell-averaged alpha. A horizontal sweep partitions every
/// triangle vertex / edge crossing; within each slab union-interval endpoints
/// are affine in y. Both smooth integrations then use fixed Gaussian quadrature.
///
/// Physical integration is disposable; F1 immutable support geometry remains
/// authoritative. Max active-arrangement work is charged before enumeration.
fn gaussian_union_sample(
    triangles: &[[f64; 2]],
    center: [f64; 2],
    sigma: f64,
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<f64, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    let cutoff = sigma * 3.0;
    let sweep = NeutralUnionSweep::new(
        triangles,
        [
            center[0] - cutoff,
            center[1] - cutoff,
            center[0] + cutoff,
            center[1] + cutoff,
        ],
        16 * 16,
        work,
        path,
    )?;
    if sweep.fully_covered {
        return Ok(1.0);
    }
    if sweep.active.is_empty() {
        return Ok(0.0);
    }
    // The arrangement charge above does not pay for the arithmetic inside
    // 16x16 quadrature. Bound every possible disjoint interval by the active
    // triangle count and admit all rows before the first row allocation.
    // Row construction scans three edges/triangle, sorts at most n intervals
    // and merges them; 128n + 16n² covers that work, including comparisons
    // and moves. Each surviving interval then executes one 16-node x integral.
    let interval_work = sweep.interval_work(path)?;
    let row_work = sweep
        .active
        .len()
        .checked_mul(GAUSSIAN_AXIS_WORK)
        .and_then(|axes| axes.checked_add(interval_work));
    let integration_work = row_work
        .and_then(|row| row.checked_mul(GAUSS_NODES.len()))
        .and_then(|slab| slab.checked_add(32))
        .and_then(|slab| slab.checked_mul(sweep.events.len().saturating_sub(1)))
        .and_then(|cost| cost.checked_add(32))
        .ok_or_else(|| {
            mask_failure(
                path,
                Render2dSampleSpaceError::ResourceLimit,
                "continuous Gaussian quadrature work overflow",
            )
        })?;
    charge_shadow_preparation_work(work, integration_work, path)?;
    let mut total = 0.0;
    for interval in sweep.events.windows(2) {
        let half_y = (interval[1] - interval[0]) * 0.5;
        if half_y <= 0.0 {
            continue;
        }
        let mid_y = interval[0] + half_y;
        for (&node_y, &weight_y) in GAUSS_NODES.iter().zip(GAUSS_WEIGHTS.iter()) {
            let y = node_y.mul_add(half_y, mid_y);
            let z = (y - center[1]) / sigma;
            let gy = (-0.5 * z * z).exp();
            let mut row = 0.0;
            for [begin, right] in sweep.intervals(y, path)? {
                let half_x = (right - begin) * 0.5;
                let mid_x = begin + half_x;
                let mut integral = 0.0;
                for (&node_x, &weight_x) in GAUSS_NODES.iter().zip(GAUSS_WEIGHTS.iter()) {
                    let x = node_x.mul_add(half_x, mid_x);
                    let z = (x - center[0]) / sigma;
                    integral += weight_x * (-0.5 * z * z).exp();
                }
                row += half_x * integral;
            }
            total += weight_y * half_y * gy * row;
        }
    }
    let normalizer = sigma * GAUSS_3SIGMA_NORMALIZER;
    let alpha = total / (normalizer * normalizer);
    if !alpha.is_finite() || !(-1.0e-9..=1.0 + 1.0e-9).contains(&alpha) {
        return Err(precision(
            "continuous Gaussian union integral is not normalized",
        ));
    }
    Ok(alpha.clamp(0.0, 1.0))
}

/// A small fixed-capacity convex polygon. A root sample has four corners;
/// clipping against four rectangle sides or two corner quadrants adds at
/// most one vertex per plane. Stack storage avoids per-sample heap churn.
#[derive(Clone, Copy)]
struct DiskAreaCellPolygon {
    vertices: [[f64; 2]; 16],
    len: usize,
}

impl DiskAreaCellPolygon {
    fn quadrilateral(points: [[f64; 2]; 4]) -> Option<Self> {
        if !points.iter().flatten().all(|v| v.is_finite()) {
            return None;
        }
        let mut vertices = [[0.0; 2]; 16];
        vertices[..4].copy_from_slice(&points);
        Some(Self { vertices, len: 4 })
    }

    fn push(&mut self, point: [f64; 2]) -> Option<()> {
        if self.len >= self.vertices.len() || !point.iter().all(|v| v.is_finite()) {
            return None;
        }
        self.vertices[self.len] = point;
        self.len += 1;
        Some(())
    }

    fn clip_axis(self, axis: usize, edge: f64, keep_lower: bool) -> Option<Self> {
        let mut output = Self {
            vertices: [[0.0; 2]; 16],
            len: 0,
        };
        if self.len == 0 {
            return Some(output);
        }
        let signed = |p: [f64; 2]| {
            if keep_lower {
                edge - p[axis]
            } else {
                p[axis] - edge
            }
        };
        for i in 0..self.len {
            let p = self.vertices[i];
            let q = self.vertices[(i + 1) % self.len];
            let dp = signed(p);
            let dq = signed(q);
            if !dp.is_finite() || !dq.is_finite() {
                return None;
            }
            if (dp >= 0.0) != (dq >= 0.0) {
                let t = dp / (dp - dq);
                if !t.is_finite() || !(0.0..=1.0).contains(&t) {
                    return None;
                }
                output.push([
                    (q[0] - p[0]).mul_add(t, p[0]),
                    (q[1] - p[1]).mul_add(t, p[1]),
                ])?;
            }
            if dq >= 0.0 {
                output.push(q)?;
            }
        }
        Some(output)
    }

    fn clipped_rectangle(self, [x0, y0, x1, y1]: [f64; 4]) -> Option<Self> {
        self.clip_axis(0, x0, false)?
            .clip_axis(0, x1, true)?
            .clip_axis(1, y0, false)?
            .clip_axis(1, y1, true)
    }

    fn area(self) -> Option<f64> {
        if self.len < 3 {
            return Some(0.0);
        }
        // Triangulation from a local origin avoids cancellation at large
        // absolute parent coordinates; clipping preserves convexity.
        let origin = self.vertices[0];
        let mut twice = 0.0_f64;
        for i in 1..self.len - 1 {
            twice += orient(origin, self.vertices[i], self.vertices[i + 1]);
        }
        let result = twice.abs() * 0.5;
        result.is_finite().then_some(result)
    }

    /// Green's theorem for intersection with a Euclidean disk. Split each
    /// polygon edge at the true circle crossings. Inside intervals
    /// contribute triangle area, outside intervals a signed circular
    /// sector r² atan2(cross,dot)/2. No faceted circle approximation.
    fn disk_area(self, center: [f64; 2], radius: f64) -> Option<f64> {
        if self.len < 3 {
            return Some(0.0);
        }
        if !radius.is_finite() || radius <= 0.0 {
            return None;
        }
        let radius_squared = radius * radius;
        if !radius_squared.is_finite() {
            return None;
        }
        let mut total = 0.0_f64;
        for i in 0..self.len {
            let p = [
                self.vertices[i][0] - center[0],
                self.vertices[i][1] - center[1],
            ];
            let q = [
                self.vertices[(i + 1) % self.len][0] - center[0],
                self.vertices[(i + 1) % self.len][1] - center[1],
            ];
            let d = [q[0] - p[0], q[1] - p[1]];
            let a = d[0].mul_add(d[0], d[1] * d[1]);
            if !a.is_finite() {
                return None;
            }
            if a == 0.0 {
                continue;
            }
            let b = 2.0 * p[0].mul_add(d[0], p[1] * d[1]);
            let c = p[0].mul_add(p[0], p[1] * p[1]) - radius_squared;
            let discriminant = b.mul_add(b, -4.0 * a * c);
            if !discriminant.is_finite() {
                return None;
            }
            let mut times = [0.0, 1.0, 1.0, 1.0];
            let mut len = 2_usize;
            if discriminant >= 0.0 {
                let root = discriminant.sqrt();
                for t in [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)] {
                    if t > 0.0 && t < 1.0 {
                        times[len] = t;
                        len += 1;
                    }
                }
            }
            times[..len].sort_by(f64::total_cmp);
            for interval in times[..len].windows(2) {
                let lo = interval[0];
                let hi = interval[1];
                let left = [d[0].mul_add(lo, p[0]), d[1].mul_add(lo, p[1])];
                let right = [d[0].mul_add(hi, p[0]), d[1].mul_add(hi, p[1])];
                let mid = [
                    d[0].mul_add((lo + hi) * 0.5, p[0]),
                    d[1].mul_add((lo + hi) * 0.5, p[1]),
                ];
                let cross = left[0].mul_add(right[1], -(left[1] * right[0]));
                let dot = left[0].mul_add(right[0], left[1] * right[1]);
                if mid[0].mul_add(mid[0], mid[1] * mid[1]) <= radius_squared {
                    total += cross * 0.5;
                } else {
                    total += 0.5 * radius_squared * cross.atan2(dot);
                }
            }
        }
        let result = total.abs();
        result.is_finite().then_some(result)
    }
}

/// Exact area of a *certified entire rectangle union* dilated by a disk,
/// intersected with a convex inverse-affine root sample parallelogram.
/// The five rectangular strips and four disjoint quarter-disks partition
/// the true rounded rectangle; neither its AABB nor a faceted polygon
/// replaces the curved Euclidean boundary.
fn certified_rounded_rectangle_area(
    cell: DiskAreaCellPolygon,
    [x0, y0, x1, y1]: [f64; 4],
    radius: f64,
) -> Option<f64> {
    if !(x0 < x1 && y0 < y1 && radius > 0.0) {
        return None;
    }
    let mut area = 0.0_f64;
    for rect in [
        [x0, y0, x1, y1],
        [x0 - radius, y0, x0, y1],
        [x1, y0, x1 + radius, y1],
        [x0, y0 - radius, x1, y0],
        [x0, y1, x1, y1 + radius],
    ] {
        area += cell.clipped_rectangle(rect)?.area()?;
    }
    for (x, lower_x) in [(x0, true), (x1, false)] {
        for (y, lower_y) in [(y0, true), (y1, false)] {
            area += cell
                .clip_axis(0, x, lower_x)?
                .clip_axis(1, y, lower_y)?
                .disk_area([x, y], radius)?;
        }
    }
    area.is_finite().then_some(area)
}

/// Conservatively integrate a positive Euclidean disk over an inverse-affine
/// ROOT unit sample cell, whose immediate-parent image is a parallelogram.
/// Midpoint distance to the exact triangle union is one-Lipschitz. The
/// farthest mapped corner bounds the entire convex cell. Thus every accepted
/// fully covered/uncovered quadrant is mathematically certified; undecided
/// area is refined until absolute error <=1/4096, otherwise a typed error.
struct AffinePositiveDiskArea<'a> {
    triangles: &'a [[f64; 2]],
    certified_rectangle: Option<[f64; 4]>,
    inverse: [f64; 4],
    root_translation: [f64; 2],
    radius: f64,
}

impl AffinePositiveDiskArea<'_> {
    fn sample(
        &self,
        root_lower: [f64; 2],
        work: &mut dyn PreparationWork,
        path: &[usize],
    ) -> Result<f64, crate::execution_2d::Render2dExecutionError> {
        use crate::execution_2d::Render2dSampleSpaceError;
        let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
        let precision =
            |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
        // A quarter-circle boundary requires ~38k visits to certify
        // area within 1/4096 under root-cell quadtree refinement.
        // The independent 16M aggregate geometric-work cap still applies.
        const MAX_VISITS: usize = 65_536;
        const MAX_DEPTH: u8 = 22;
        const UNCERTAIN_MASS_TARGET: f64 = 1.0 / 2048.0;
        if !self.radius.is_finite()
            || self.radius <= 0.0
            || !self.inverse.iter().all(|v| v.is_finite())
            || !self.root_translation.iter().all(|v| v.is_finite())
            || !root_lower.iter().all(|v| v.is_finite())
        {
            return Err(precision(
                "affine disk sample parameters are unrepresentable",
            ));
        }
        if let Some(rect) = self.certified_rectangle {
            // The complete two-triangle UNION (not its AABB) has been
            // certified to be exactly a rectangle. The analytic path has
            // constant per-cell work and no geometric heap allocations.
            // Charge the inexpensive conservative per-cell classification
            // separately from the more costly clipped circular integration.
            // A large rectangle must not exhaust the global work budget
            // solely for cells whose membership is geometrically certain.
            work.charge(
                8,
                MAX_COMPOSITIONAL_SHADOW_WORK,
                path,
                "rounded rectangle area exceeds bounded work",
            )?;
            let [m00, m01, m10, m11] = self.inverse;
            let mapped = |root: [f64; 2]| {
                let p = [
                    root[0] - self.root_translation[0],
                    root[1] - self.root_translation[1],
                ];
                [m00.mul_add(p[0], m01 * p[1]), m10.mul_add(p[0], m11 * p[1])]
            };
            let [x, y] = root_lower;
            let polygon = DiskAreaCellPolygon::quadrilateral([
                mapped([x, y]),
                mapped([x + 1.0, y]),
                mapped([x + 1.0, y + 1.0]),
                mapped([x, y + 1.0]),
            ])
            .ok_or_else(|| precision("rounded rectangle affine cell is invalid"))?;
            let parent_cell_area = polygon
                .area()
                .ok_or_else(|| precision("rounded rectangle parent cell area is invalid"))?;
            if parent_cell_area <= 0.0 {
                return Err(precision("rounded rectangle parent cell collapsed"));
            }
            let [x0, y0, x1, y1] = rect;
            let scale = x0
                .abs()
                .max(y0.abs())
                .max(x1.abs())
                .max(y1.abs())
                .max(self.radius)
                .max(1.0);
            let margin = 16.0 * f64::EPSILON * scale;
            let extended = [
                x0 - self.radius - margin,
                y0 - self.radius - margin,
                x1 + self.radius + margin,
                y1 + self.radius + margin,
            ];
            if !extended.iter().all(|v| v.is_finite()) {
                return Err(precision("rounded rectangle support reach lost precision"));
            }
            let mut min_x = f64::INFINITY;
            let mut min_y = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            let mut max_y = f64::NEG_INFINITY;
            let mut all_inside = true;
            for point in &polygon.vertices[..polygon.len] {
                min_x = min_x.min(point[0]);
                min_y = min_y.min(point[1]);
                max_x = max_x.max(point[0]);
                max_y = max_y.max(point[1]);
                // Rounded rectangle is convex. Four contained corners
                // certify the entire inverse-affine root parallelogram.
                let original_inside =
                    point[0] >= x0 && point[0] <= x1 && point[1] >= y0 && point[1] <= y1;
                let nearest_x = point[0].clamp(x0, x1);
                let nearest_y = point[1].clamp(y0, y1);
                let distance = (point[0] - nearest_x).hypot(point[1] - nearest_y);
                all_inside &= original_inside || distance + margin < self.radius;
            }
            if max_x < extended[0]
                || max_y < extended[1]
                || min_x > extended[2]
                || min_y > extended[3]
            {
                // AABB is used only as a conservative *disjointness* proof,
                // never as geometric membership.
                return Ok(0.0);
            }
            if all_inside {
                return Ok(1.0);
            }
            // 8 units were charged for all cells; 1016 additional units
            // complete the conservative 1024-unit exact clipped integration.
            work.charge(
                1016,
                MAX_COMPOSITIONAL_SHADOW_WORK,
                path,
                "rounded rectangle area exceeds bounded work",
            )?;
            let clipped_area = certified_rounded_rectangle_area(polygon, rect, self.radius)
                .ok_or_else(|| precision("rounded rectangle circle integral is invalid"))?;
            let coverage = clipped_area / parent_cell_area;
            if !coverage.is_finite() || !(-1.0e-7..=1.0 + 1.0e-7).contains(&coverage) {
                return Err(precision("rounded rectangle circle area is not normalized"));
            }
            return Ok(coverage.clamp(0.0, 1.0));
        }
        let mut pending = std::collections::BinaryHeap::new();
        pending
            .try_reserve(1)
            .map_err(|_| resource("affine disk cell queue allocation failed"))?;
        pending.push(SpreadGaussianCell {
            bounds: [
                root_lower[0],
                root_lower[1],
                root_lower[0] + 1.0,
                root_lower[1] + 1.0,
            ],
            mass: 1.0,
            depth: 0,
        });
        let mut unknown = 1.0_f64;
        let mut covered = 0.0_f64;
        let mut visited = 0_usize;
        while unknown > UNCERTAIN_MASS_TARGET {
            let cell = pending
                .pop()
                .ok_or_else(|| precision("affine disk area uncertainty lost its queue"))?;
            unknown -= cell.mass;
            visited = visited
                .checked_add(1)
                .ok_or_else(|| resource("affine disk area visit count overflow"))?;
            if visited > MAX_VISITS {
                return Err(resource("affine disk area exceeds bounded work"));
            }
            work.charge(
                self.triangles.len() / 3,
                MAX_COMPOSITIONAL_SHADOW_WORK,
                path,
                "affine disk area exceeds bounded work",
            )?;
            let [x0, y0, x1, y1] = cell.bounds;
            let half_x = (x1 - x0) * 0.5;
            let half_y = (y1 - y0) * 0.5;
            let center = [
                (x0 + x1) * 0.5 - self.root_translation[0],
                (y0 + y1) * 0.5 - self.root_translation[1],
            ];
            let [m00, m01, m10, m11] = self.inverse;
            let point = [
                m00.mul_add(center[0], m01 * center[1]),
                m10.mul_add(center[0], m11 * center[1]),
            ];
            let dx = m00 * half_x;
            let dy = m10 * half_x;
            let ex = m01 * half_y;
            let ey = m11 * half_y;
            let bound = (dx + ex).hypot(dy + ey).max((dx - ex).hypot(dy - ey));
            if !point.iter().all(|v| v.is_finite()) || !bound.is_finite() {
                return Err(precision("affine disk area projection lost precision"));
            }
            let distance = neutral_union_distance(self.triangles, point);
            if !distance.is_finite() {
                return Err(precision("affine disk area distance is unrepresentable"));
            }
            if distance + bound <= self.radius {
                covered += cell.mass;
                continue;
            }
            if distance > self.radius + bound {
                continue;
            }
            // A single triangle dilated by a Euclidean disk is convex.
            // If all four corners of the inverse-affine parallelogram lie
            // in that convex set, its WHOLE area is known to be inside.
            // This avoids excessive subdivisions of already-covered
            // interiors, including arbitrarily small positive radii.
            // Charge the worst-case four distance queries per triangle.
            let certificate_work = (self.triangles.len() / 3)
                .checked_mul(4)
                .ok_or_else(|| resource("affine disk convex certificate work overflow"))?;
            work.charge(
                certificate_work,
                MAX_COMPOSITIONAL_SHADOW_WORK,
                path,
                "affine disk convex certificate exceeds bounded work",
            )?;
            let corners = [
                [point[0] - dx - ex, point[1] - dy - ey],
                [point[0] - dx + ex, point[1] - dy + ey],
                [point[0] + dx - ex, point[1] + dy - ey],
                [point[0] + dx + ex, point[1] + dy + ey],
            ];
            if !corners.iter().flatten().all(|v| v.is_finite()) {
                return Err(precision("affine disk corner projection lost precision"));
            }
            if self.triangles.as_chunks::<3>().0.iter().any(|tri| {
                orient(tri[0], tri[1], tri[2]) != 0.0
                    && corners
                        .iter()
                        .all(|&point| neutral_union_distance(tri, point) <= self.radius)
            }) {
                covered += cell.mass;
                continue;
            }
            let mx = (x0 + x1) * 0.5;
            let my = (y0 + y1) * 0.5;
            if cell.depth >= MAX_DEPTH || mx <= x0 || mx >= x1 || my <= y0 || my >= y1 {
                return Err(precision(
                    "affine disk area cannot certify coverage tolerance",
                ));
            }
            pending
                .try_reserve(4)
                .map_err(|_| resource("affine disk area queue allocation failed"))?;
            for bounds in [
                [x0, y0, mx, my],
                [mx, y0, x1, my],
                [x0, my, mx, y1],
                [mx, my, x1, y1],
            ] {
                let mass = cell.mass * 0.25;
                pending.push(SpreadGaussianCell {
                    bounds,
                    mass,
                    depth: cell.depth + 1,
                });
                unknown += mass;
            }
        }
        let alpha = covered + unknown * 0.5;
        if !alpha.is_finite() || !(-1.0e-9..=1.0 + 1.0e-9).contains(&alpha) {
            return Err(precision("affine disk area mass is not normalized"));
        }
        Ok(alpha.clamp(0.0, 1.0))
    }
}

/// Evaluate a nested finite Gaussian over the owning parent-frame neutral
/// geometry, with an optional true Euclidean positive disk spread BEFORE
/// convolution. At zero sigma, integrate the inverse-affine root sample
/// cell's exact area. An arbitrary full-rank ancestor affine changes root-space
/// footprints, not the parent's spread or normalized Gaussian:
/// inverse-map each root sample to its parent before continuous integration.
/// An extent is only allocation/reach admission, never a region mask.
/// Private physical kernel for one owning parent-frame effect. A single
/// parameter keeps spread and blur together without minting semantic state.
pub(super) struct ParentEffectKernel {
    pub(super) signed_spread: f64,
    pub(super) sigma: f64,
}

/// One owner-frame projection: these coupled facts describe a single
/// root-sample lattice, not separately configurable shadow semantics.
pub(super) struct ParentEffectFrame {
    pub(super) to_root: crate::composition_2d::Render2dAffineTransform,
    pub(super) root_envelope: [f64; 4],
    pub(super) sample_scale: f64,
}

pub(super) fn affine_parent_effect_coverage_admitted(
    source: &NeutralMesh,
    kernel: ParentEffectKernel,
    frame: ParentEffectFrame,
    path: &[usize],
    aggregate_work: &mut dyn PreparationWork,
    remaining_coverage_samples: usize,
) -> Result<Option<NeutralCoverage>, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let ParentEffectKernel {
        signed_spread,
        sigma,
    } = kernel;
    let ParentEffectFrame {
        to_root: parent_to_root,
        root_envelope,
        sample_scale: scale,
    } = frame;
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    if source.triangles.is_empty() {
        return Ok(None);
    }
    let [a, b, c, d, tx, ty] = parent_to_root.components();
    let determinant = a.mul_add(d, -(b * c));
    let physical_sigma = sigma * scale;
    let physical_spread = signed_spread * scale;
    if !scale.is_finite()
        || scale <= 0.0
        || !source.units_per_parent_logical_unit.is_finite()
        || source.units_per_parent_logical_unit <= 0.0
        || !sigma.is_finite()
        || sigma < 0.0
        || !physical_sigma.is_finite()
        || physical_sigma < 0.0
        || (physical_sigma == 0.0 && physical_spread == 0.0)
        || (signed_spread < 0.0 && sigma == 0.0)
        || !signed_spread.is_finite()
        || !physical_spread.is_finite()
        || !determinant.is_finite()
        || determinant == 0.0
        || !parent_to_root.components().iter().all(|x| x.is_finite())
    {
        return Err(precision(
            "nested Gaussian parent affine or kernel is unrepresentable",
        ));
    }
    if (3.0 * physical_sigma).ceil() > f64::from(MAX_GAUSSIAN_RADIUS_SAMPLES) {
        return Err(resource(
            "nested parent-frame Gaussian window exceeds bounded halo",
        ));
    }
    if physical_spread.abs() > f64::from(MAX_SPREAD_RADIUS_SAMPLES) {
        return Err(resource(
            "nested parent-frame Euclidean disk exceeds bounded radius",
        ));
    }
    let scaled = root_envelope.map(|v| v * scale);
    if !scaled.iter().all(|x| x.is_finite())
        || scaled.iter().any(|x| x.abs() > f64::from(i32::MAX) / 2.0)
        || scaled[0] >= scaled[2]
        || scaled[1] >= scaled[3]
    {
        return Err(precision(
            "nested Gaussian root sample extent cannot be represented",
        ));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "root sample edges have been admitted into the safe i32 range"
    )]
    let edges = [
        scaled[0].floor() as i32,
        scaled[1].floor() as i32,
        scaled[2].ceil() as i32,
        scaled[3].ceil() as i32,
    ];
    let width = usize::try_from(i64::from(edges[2]) - i64::from(edges[0]))
        .map_err(|_| resource("nested Gaussian root width overflow"))?;
    let height = usize::try_from(i64::from(edges[3]) - i64::from(edges[1]))
        .map_err(|_| resource("nested Gaussian root height overflow"))?;
    let area = width
        .checked_mul(height)
        .ok_or_else(|| resource("nested Gaussian root area overflow"))?;
    if area == 0 || area > MAX_NEUTRAL_MASK_SAMPLES {
        return Err(resource(
            "nested Gaussian root sample grid exceeds bounded area",
        ));
    }
    admit_retained_coverage_samples(area, remaining_coverage_samples, path)?;
    let triangles_count = source.triangles.len() / 3;
    let scan_work = area
        .checked_mul(triangles_count)
        .ok_or_else(|| resource("nested Gaussian neutral scan overflow"))?;
    if scan_work > 16_777_216 {
        return Err(resource(
            "nested Gaussian neutral scan exceeds bounded work",
        ));
    }
    charge_shadow_preparation_work(aggregate_work, scan_work, path)?;
    let conversion = scale / source.units_per_parent_logical_unit;
    if !conversion.is_finite() || conversion <= 0.0 {
        return Err(precision("nested Gaussian triangle scale invalid"));
    }
    let mut triangles = filled(source.triangles.len(), [0.0; 2], path)?;
    for (destination, &[x, y]) in triangles.iter_mut().zip(&source.triangles) {
        *destination = [x * conversion, y * conversion];
        if !destination.iter().all(|v| v.is_finite()) {
            return Err(precision(
                "nested Gaussian source projection lost precision",
            ));
        }
    }
    let certified_rectangle = charge_and_certify_rectangle(&triangles, aggregate_work, path)?;
    // A zero-spread rectangle has a separable parent-frame Gaussian even
    // under an arbitrary invertible ancestor. Transform the query point,
    // then integrate the two original axes; no 2D triangle sweep is needed.
    let gaussian_rectangle = if signed_spread == 0.0 && physical_sigma > 0.0 {
        certified_rectangle
    } else {
        None
    };
    if gaussian_rectangle.is_some() {
        let cost = area
            .checked_mul(2 * GAUSSIAN_AXIS_WORK + 64)
            .ok_or_else(|| resource("nested rectangle Gaussian work overflow"))?;
        charge_shadow_preparation_work(aggregate_work, cost, path)?;
    }
    let disk_area = AffinePositiveDiskArea {
        triangles: &triangles,
        certified_rectangle,
        inverse: [
            d / determinant,
            -c / determinant,
            -b / determinant,
            a / determinant,
        ],
        root_translation: [tx * scale, ty * scale],
        radius: physical_spread.max(0.0),
    };
    // Negative spread under a non-isometric ancestor must be evaluated
    // in the original parent frame. Only source-proven original rectangle
    // unions are admitted by the caller: their triangles carry exact F1
    // polygonal support, including true concavities and holes.
    let erosion_boundary = (signed_spread < 0.0)
        .then(|| union_exterior_boundary(&triangles, path, Some(aggregate_work)))
        .transpose()?;
    // For x-only shear (x_root=x_parent+c*y_parent+tx, y_root=y_parent+ty),
    // every root sample row inverse-maps to a regularly spaced parent row.
    // Reuse the existing original-exterior Gaussian row integrator rather
    // than reclassifying the entire curved support at every sample.
    // This is disposable coverage, not a replacement F1 neutral source.
    let shared_row = if signed_spread < 0.0 && a == 1.0 && b == 0.0 && d == 1.0 {
        erosion_boundary
            .as_deref()
            .map(|boundary| OrthogonalExterior::new(boundary, path, aggregate_work))
            .transpose()?
            .flatten()
    } else {
        None
    };
    if let Some(edges) = &erosion_boundary {
        // Admit the simultaneously live copied triangles, exposed union
        // edges, coverage grid and worst-case adaptive cell heap BEFORE
        // allocating the retained coverage. The original semantic graph
        // remains separately retained and admitted by its own owner.
        let temporary_bytes = triangles
            .capacity()
            .checked_mul(std::mem::size_of::<[f64; 2]>())
            .and_then(|n| {
                edges
                    .capacity()
                    .checked_mul(std::mem::size_of::<UnionBoundaryEdge>())
                    .and_then(|v| n.checked_add(v))
            })
            .and_then(|n| {
                area.checked_mul(std::mem::size_of::<f64>())
                    .and_then(|v| n.checked_add(v))
            })
            .and_then(|n| {
                32_768_usize
                    .checked_mul(std::mem::size_of::<SpreadGaussianCell>())
                    .and_then(|v| n.checked_add(v))
            })
            .ok_or_else(|| resource("nested signed Gaussian temporary memory overflow"))?;
        // Cached orthogonal crossing/section scratch remains below 256 KiB
        // at the established 256-edge/16,384-crossing private ceilings.
        let temporary_bytes = temporary_bytes
            .checked_add(if shared_row.is_some() { 256 * 1024 } else { 0 })
            .ok_or_else(|| resource("nested signed Gaussian row-memory overflow"))?;
        if temporary_bytes > 64 * 1024 * 1024 {
            return Err(resource(
                "nested signed Gaussian peak temporary memory exceeds 64 MiB",
            ));
        }
    }
    let mut values = filled(area, 0.0_f64, path)?;
    if let Some(orthogonal) = shared_row {
        for row in 0..height {
            let root_y = f64::from(edges[1]) + as_f64(row) + 0.5;
            let parent_y = root_y - ty * scale;
            let parent_origin_x = f64::from(edges[0]) - tx * scale - c * parent_y;
            let begin = row * width;
            orthogonal.gaussian_row(
                GaussianRow {
                    y: parent_y,
                    origin_x: parent_origin_x,
                    width,
                    radius: -physical_spread,
                    sigma: physical_sigma,
                },
                &mut values[begin..begin + width],
                aggregate_work,
                path,
            )?;
        }
        return Ok(Some(NeutralCoverage {
            origin_x: i64::from(edges[0]),
            origin_y: i64::from(edges[1]),
            width,
            height,
            values,
        }));
    }
    for row in 0..height {
        for column in 0..width {
            let root_x = f64::from(edges[0]) + as_f64(column) + 0.5;
            let root_y = f64::from(edges[1]) + as_f64(row) + 0.5;
            let rx = root_x - tx * scale;
            let ry = root_y - ty * scale;
            let point = [
                (d * rx - c * ry) / determinant,
                (a * ry - b * rx) / determinant,
            ];
            if !point.iter().all(|x| x.is_finite()) {
                return Err(precision("nested Gaussian inverse mapping lost precision"));
            }
            values[row * width + column] = if signed_spread < 0.0 {
                continuous_signed_spread_sample(
                    &triangles,
                    erosion_boundary.as_deref(),
                    point,
                    physical_spread,
                    physical_sigma,
                    aggregate_work,
                    path,
                )?
            } else if signed_spread > 0.0 && sigma == 0.0 {
                disk_area.sample([root_x - 0.5, root_y - 0.5], aggregate_work, path)?
            } else if signed_spread > 0.0 {
                // Membership is evaluated against the exact triangle UNION
                // Euclidean distance. A conservative mass/error certificate
                // bounds continuous disk/Gaussian integration; an uncertified
                // result rejects rather than fabricating coverage.
                continuous_signed_spread_sample(
                    &triangles,
                    None,
                    point,
                    physical_spread,
                    physical_sigma,
                    aggregate_work,
                    path,
                )?
            } else if let Some(rect) = gaussian_rectangle {
                let x = inset_rectangle_axis_mass(rect[0], rect[2], point[0], physical_sigma);
                let y = inset_rectangle_axis_mass(rect[1], rect[3], point[1], physical_sigma);
                (x * y).clamp(0.0, 1.0)
            } else {
                gaussian_union_sample(&triangles, point, physical_sigma, aggregate_work, path)?
            };
        }
    }
    Ok(Some(NeutralCoverage {
        origin_x: i64::from(edges[0]),
        origin_y: i64::from(edges[1]),
        width,
        height,
        values,
    }))
}

/// A deterministic highest-uncertainty-first subdivision cell. `mass`
/// is normalized positive Gaussian measure, never paint or sampled alpha.
#[derive(Clone, Copy, Debug)]
struct SpreadGaussianCell {
    bounds: [f64; 4],
    mass: f64,
    depth: u8,
}

impl PartialEq for SpreadGaussianCell {
    fn eq(&self, other: &Self) -> bool {
        self.mass == other.mass && self.bounds == other.bounds && self.depth == other.depth
    }
}
impl Eq for SpreadGaussianCell {}
impl PartialOrd for SpreadGaussianCell {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SpreadGaussianCell {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.mass
            .total_cmp(&other.mass)
            .then_with(|| self.depth.cmp(&other.depth))
            .then_with(|| self.bounds[0].total_cmp(&other.bounds[0]))
            .then_with(|| self.bounds[1].total_cmp(&other.bounds[1]))
            .then_with(|| self.bounds[2].total_cmp(&other.bounds[2]))
            .then_with(|| self.bounds[3].total_cmp(&other.bounds[3]))
    }
}

/// Euclidean distance to the immutable union of nondegenerate triangles.
/// The distance to a *union* is one-Lipschitz, allowing conservative
/// inside/outside certification over each integration rectangle.
fn neutral_union_distance(triangles: &[[f64; 2]], p: [f64; 2]) -> f64 {
    let mut closest = f64::INFINITY;
    for tri in triangles.as_chunks::<3>().0 {
        if orient(tri[0], tri[1], tri[2]) == 0.0 {
            continue;
        }
        let ab = orient(tri[0], tri[1], p);
        let bc = orient(tri[1], tri[2], p);
        let ca = orient(tri[2], tri[0], p);
        if (ab >= 0.0 && bc >= 0.0 && ca >= 0.0) || (ab <= 0.0 && bc <= 0.0 && ca <= 0.0) {
            return 0.0;
        }
        closest = closest.min(distance_squared_to_segment(p, tri[0], tri[1]));
        closest = closest.min(distance_squared_to_segment(p, tri[1], tri[2]));
        closest = closest.min(distance_squared_to_segment(p, tri[2], tri[0]));
    }
    closest.sqrt()
}

/// A rectangle is wholly in the dilation of one convex triangle if
/// every corner is no farther than the radius from that triangle. Convexity
/// then proves its *entire interior* is admitted, not just four samples.
/// This is sufficient (not necessary) for the union of all triangle dilations.
fn one_dilated_triangle_contains_cell(
    triangles: &[[f64; 2]],
    rectangle: [f64; 4],
    radius: f64,
) -> bool {
    let [x0, y0, x1, y1] = rectangle;
    let corners = [[x0, y0], [x1, y0], [x0, y1], [x1, y1]];
    triangles.as_chunks::<3>().0.iter().any(|tri| {
        orient(tri[0], tri[1], tri[2]) != 0.0
            && corners
                .iter()
                .all(|&p| neutral_union_distance(tri, p) <= radius)
    })
}

/// Bounding rectangles of all convex triangles are a conservative superset.
/// If the cell-to-each-bounds Euclidean gap is >radius, *no* point of the
/// integration cell can lie inside the real disk dilation. This tighter box
/// test avoids the half-diagonal Lipschitz overestimate near long edges.
fn dilated_triangles_miss_cell(triangles: &[[f64; 2]], rectangle: [f64; 4], radius: f64) -> bool {
    let [x0, y0, x1, y1] = rectangle;
    triangles.as_chunks::<3>().0.iter().all(|tri| {
        let min_x = tri.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        let max_x = tri.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
        let min_y = tri.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        let max_y = tri.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
        let dx = (min_x - x1).max(x0 - max_x).max(0.0);
        let dy = (min_y - y1).max(y0 - max_y).max(0.0);
        dx.hypot(dy) > radius
    })
}

/// One interval of the triangle-union *exterior boundary*, not the
/// original tessellation edges. Interior diagonals and overlapping strokes
/// are removed before computing continuous negative Euclidean spread.
type UnionBoundaryEdge = [[f64; 2]; 2];

fn oriented_triangle_contains_outward(tri: &[[f64; 2]; 3], p: [f64; 2], outward: [f64; 2]) -> bool {
    let area = orient(tri[0], tri[1], tri[2]);
    if area == 0.0 {
        return false;
    }
    let sign = area.signum();
    for side in 0..3 {
        let a = tri[side];
        let b = tri[(side + 1) % 3];
        let edge = [b[0] - a[0], b[1] - a[1]];
        let half_plane = sign * orient(a, b, p);
        if half_plane < 0.0 {
            return false;
        }
        if half_plane == 0.0 {
            let derivative = sign * edge[0].mul_add(outward[1], -(edge[1] * outward[0]));
            if derivative <= 0.0 {
                return false;
            }
        }
    }
    true
}

/// Erosion owns bounded fallible buffers. Reallocation can copy the entire
/// old allocation, including spare capacity, so admit that relocation first.
/// The element types here contain only fixed-size scalar words.
fn reserve_erosion_buffer<T: Copy>(
    buffer: &mut Vec<T>,
    required: usize,
    maximum: usize,
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<(), crate::execution_2d::Render2dExecutionError> {
    let resource = |detail| {
        mask_failure(
            path,
            crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            detail,
        )
    };
    if required > maximum || buffer.capacity() > maximum {
        return Err(resource(
            "continuous erosion buffer exceeds admitted capacity",
        ));
    }
    if required <= buffer.capacity() {
        return Ok(());
    }
    let words = std::mem::size_of::<T>().div_ceil(std::mem::size_of::<f64>());
    let relocation = buffer
        .capacity()
        .checked_mul(words)
        .and_then(|n| n.checked_mul(2))
        .and_then(|n| n.checked_add(64))
        .ok_or_else(|| resource("continuous erosion buffer relocation work overflow"))?;
    charge_shadow_preparation_work(work, relocation, path)?;
    let desired = buffer
        .capacity()
        .saturating_mul(2)
        .max(4)
        .min(maximum)
        .max(required);
    buffer
        .try_reserve_exact(desired - buffer.len())
        .map_err(|_| resource("continuous erosion buffer allocation failed"))?;
    if buffer.capacity() > desired {
        return Err(resource(
            "continuous erosion allocated capacity exceeds admission",
        ));
    }
    Ok(())
}

fn push_erosion_buffer<T: Copy>(
    buffer: &mut Vec<T>,
    value: T,
    maximum: usize,
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<(), crate::execution_2d::Render2dExecutionError> {
    // Checked extent/capacity, scalar copies, store and bounded control.
    charge_shadow_preparation_work(
        work,
        16 + 2 * std::mem::size_of::<T>().div_ceil(std::mem::size_of::<f64>()),
        path,
    )?;
    let required = buffer.len().checked_add(1).ok_or_else(|| {
        mask_failure(
            path,
            crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            "continuous erosion buffer extent overflow",
        )
    })?;
    if required > maximum {
        return Err(mask_failure(
            path,
            crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
            "continuous erosion push exceeds admitted inventory",
        ));
    }
    if required > buffer.capacity() {
        reserve_erosion_buffer(buffer, required, maximum, work, path)?;
    }
    buffer.push(value);
    Ok(())
}

/// In-place heapsort gives this owner an explicit comparison/move bound and
/// needs no hidden sorting allocation. Numeric order and dedup are unchanged.
fn sort_erosion_intervals(
    values: &mut Vec<f64>,
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<(), crate::execution_2d::Render2dExecutionError> {
    fn sift(values: &mut [f64], mut root: usize) {
        while root < values.len() / 2 {
            let mut child = 2 * root + 1;
            if child + 1 < values.len() && values[child].total_cmp(&values[child + 1]).is_lt() {
                child += 1;
            }
            if !values[root].total_cmp(&values[child]).is_lt() {
                break;
            }
            values.swap(root, child);
            root = child;
        }
    }
    let levels = (usize::BITS - values.len().leading_zeros()) as usize;
    // Fewer than 2n sifts, each <=levels iterations. 128 primitives cover
    // two total_cmp operations, indexed loads/swaps and control per step;
    // another 64n cover extraction swaps and the final in-place dedup.
    let cost = values
        .len()
        .checked_mul(256 * levels + 64)
        .and_then(|n| n.checked_add(64))
        .ok_or_else(|| {
            mask_failure(
                path,
                crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                "continuous erosion sorting work overflow",
            )
        })?;
    charge_shadow_preparation_work(work, cost, path)?;
    for root in (0..values.len() / 2).rev() {
        sift(values, root);
    }
    for end in (1..values.len()).rev() {
        values.swap(0, end);
        sift(&mut values[..end], 0);
    }
    values.dedup();
    Ok(())
}

/// Extract the exact exposed line segments of a bounded triangle union.
/// Every source edge is split at triangle-edge crossings, including collinear
/// overlaps. A segment is retained precisely when infinitesimal motion to
/// its source triangle's outward side is NOT contained in another triangle.
/// This preserves genuine outer and hole boundaries but removes tessellation
/// diagonals, coincident interior edges and overlapping stroke triangles.
/// Unsupported floating-point crossings reject rather than guessing.
fn union_exterior_boundary(
    triangles: &[[f64; 2]],
    path: &[usize],
    aggregate_work: Option<&mut dyn PreparationWork>,
) -> Result<Vec<UnionBoundaryEdge>, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    if !triangles.len().is_multiple_of(3) {
        return Err(precision("continuous erosion triangle payload invalid"));
    }
    const MAX_BOUNDARY_OPERATIONS: usize = 16_777_216;
    let pairs = triangles
        .len()
        .checked_mul(triangles.len())
        .ok_or_else(|| resource("continuous erosion boundary intersection admission overflow"))?;
    if pairs > MAX_BOUNDARY_OPERATIONS {
        return Err(resource(
            "continuous erosion boundary arrangement exceeds bounded work",
        ));
    }
    let mut standalone_work = 0;
    let aggregate_work = aggregate_work.unwrap_or(&mut standalone_work);
    // Visit admission above remains independent of executed primitive work.
    // Finite scan, setup and all following arithmetic pay the live counter.
    charge_shadow_preparation_work(aggregate_work, 64 + 8 * triangles.len(), path)?;
    if !triangles.iter().flatten().all(|c| c.is_finite()) {
        return Err(precision("continuous erosion triangle payload invalid"));
    }
    let tris = triangles.as_chunks::<3>().0;
    let mut boundary = Vec::new();
    let interval_limit = 2 + 2 * triangles.len();
    let boundary_limit = if tris.is_empty() {
        0
    } else {
        (triangles.len() * (interval_limit - 1)).min((MAX_BOUNDARY_OPERATIONS - pairs) / tris.len())
    };
    // Reuse this one edge-event buffer. Each other source edge emits at
    // most two collinear endpoints; every push still checks the bound.
    let mut intervals = Vec::new();
    let mut work = pairs;
    for (i, tri) in tris.iter().enumerate() {
        charge_shadow_preparation_work(aggregate_work, 64, path)?;
        let area = orient(tri[0], tri[1], tri[2]);
        if !area.is_finite() {
            return Err(precision(
                "continuous erosion triangle area is not representable",
            ));
        }
        if area == 0.0 {
            continue;
        }
        for e in 0..3 {
            charge_shadow_preparation_work(aggregate_work, 64, path)?;
            let a = tri[e];
            let b = tri[(e + 1) % 3];
            let edge = [b[0] - a[0], b[1] - a[1]];
            let length_squared = edge[0].mul_add(edge[0], edge[1] * edge[1]);
            if !length_squared.is_finite() || length_squared == 0.0 {
                return Err(precision("continuous erosion has collapsed boundary edge"));
            }
            intervals.clear();
            push_erosion_buffer(&mut intervals, 0.0, interval_limit, aggregate_work, path)?;
            push_erosion_buffer(&mut intervals, 1.0, interval_limit, aggregate_work, path)?;
            for (j, other) in tris.iter().enumerate() {
                // Pair control, source orientation and indexed vertex loads.
                charge_shadow_preparation_work(aggregate_work, 64, path)?;
                if i == j || orient(other[0], other[1], other[2]) == 0.0 {
                    continue;
                }
                for k in 0..3 {
                    // Cross products/ratios, finite/range guards, and both
                    // possible collinear projections. Pushes pay separately.
                    charge_shadow_preparation_work(aggregate_work, 192, path)?;
                    let c = other[k];
                    let d = other[(k + 1) % 3];
                    let v = [d[0] - c[0], d[1] - c[1]];
                    let divisor = edge[0].mul_add(v[1], -(edge[1] * v[0]));
                    let difference = [c[0] - a[0], c[1] - a[1]];
                    if divisor != 0.0 {
                        let t = difference[0].mul_add(v[1], -(difference[1] * v[0])) / divisor;
                        let u =
                            difference[0].mul_add(edge[1], -(difference[1] * edge[0])) / divisor;
                        if !t.is_finite() || !u.is_finite() {
                            return Err(precision(
                                "continuous erosion edge crossing lost precision",
                            ));
                        }
                        if t > 0.0 && t < 1.0 && (0.0..=1.0).contains(&u) {
                            push_erosion_buffer(
                                &mut intervals,
                                t,
                                interval_limit,
                                aggregate_work,
                                path,
                            )?;
                        }
                    } else if orient(a, b, c) == 0.0 && orient(a, b, d) == 0.0 {
                        for p in [c, d] {
                            let t = (p[0] - a[0]).mul_add(edge[0], (p[1] - a[1]) * edge[1])
                                / length_squared;
                            if !t.is_finite() {
                                return Err(precision(
                                    "continuous erosion collinear crossing lost precision",
                                ));
                            }
                            if t > 0.0 && t < 1.0 {
                                push_erosion_buffer(
                                    &mut intervals,
                                    t,
                                    interval_limit,
                                    aggregate_work,
                                    path,
                                )?;
                            }
                        }
                    }
                }
            }
            sort_erosion_intervals(&mut intervals, aggregate_work, path)?;
            let outward = if area > 0.0 {
                [edge[1], -edge[0]]
            } else {
                [-edge[1], edge[0]]
            };
            for interval in intervals.windows(2) {
                if interval[0] >= interval[1] {
                    continue;
                }
                work = work
                    .checked_add(tris.len())
                    .ok_or_else(|| resource("continuous erosion exposed-edge work overflow"))?;
                if work > MAX_BOUNDARY_OPERATIONS {
                    return Err(resource(
                        "continuous erosion exposed-edge classification exceeds budget",
                    ));
                }
                // At most one area and three side/derivative orientations
                // per source, including indexed loads and short-circuit tests.
                charge_shadow_preparation_work(aggregate_work, 64 + 192 * tris.len(), path)?;
                let t = (interval[0] + interval[1]) * 0.5;
                let p = [edge[0].mul_add(t, a[0]), edge[1].mul_add(t, a[1])];
                if tris.iter().enumerate().any(|(j, other)| {
                    j != i && oriented_triangle_contains_outward(other, p, outward)
                }) {
                    continue;
                }
                charge_shadow_preparation_work(aggregate_work, 64, path)?;
                let endpoints = [
                    [
                        edge[0].mul_add(interval[0], a[0]),
                        edge[1].mul_add(interval[0], a[1]),
                    ],
                    [
                        edge[0].mul_add(interval[1], a[0]),
                        edge[1].mul_add(interval[1], a[1]),
                    ],
                ];
                if !endpoints.iter().flatten().all(|v| v.is_finite()) {
                    return Err(precision("continuous erosion boundary endpoint invalid"));
                }
                push_erosion_buffer(
                    &mut boundary,
                    endpoints,
                    boundary_limit,
                    aggregate_work,
                    path,
                )?;
            }
        }
    }
    if !triangles.is_empty() && boundary.is_empty() {
        return Err(precision(
            "continuous erosion cannot certify exterior boundary",
        ));
    }
    Ok(boundary)
}

/// A conservative separation certificate for the exact union exterior.
/// The shortest distance between a boundary edge's containing AABB and a
/// connected integration cell is a lower bound on their true separation.
/// When every such bound exceeds the erosion radius, the cell cannot meet
/// the union exterior nor its radius-r boundary strip. Membership at the
/// midpoint is therefore constant across the full cell, even for concave
/// unions, overlapping tessellation, and interior holes.
fn eroded_cell_far_from_exterior(
    exterior: &[UnionBoundaryEdge],
    cell: [f64; 4],
    radius: f64,
) -> bool {
    exterior.iter().all(|edge| {
        let lo_x = edge[0][0].min(edge[1][0]);
        let hi_x = edge[0][0].max(edge[1][0]);
        let lo_y = edge[0][1].min(edge[1][1]);
        let hi_y = edge[0][1].max(edge[1][1]);
        let dx = (lo_x - cell[2]).max(cell[0] - hi_x).max(0.0);
        let dy = (lo_y - cell[3]).max(cell[1] - hi_y).max(0.0);
        dx.hypot(dy) > radius
    })
}

fn union_signed_interior_distance(
    triangles: &[[f64; 2]],
    boundary: &[UnionBoundaryEdge],
    p: [f64; 2],
) -> f64 {
    let inside = neutral_union_distance(triangles, p) == 0.0;
    let nearest = boundary
        .iter()
        .map(|&[a, b]| distance_squared_to_segment(p, a, b))
        .fold(f64::INFINITY, f64::min)
        .sqrt();
    if inside { nearest } else { -nearest }
}

/// Certified convex triangle-union erosion. Every exposed union-boundary
/// segment must be a supporting line of ALL original triangle vertices;
/// otherwise the source is nonconvex or has a hole and the general signed-
/// distance path remains authoritative. This is a geometric certificate,
/// never an AABB or sampled-alpha membership shortcut.
///
/// A convex polygon eroded by a Euclidean disk is precisely the intersection
/// of its inward-offset supporting halfplanes. Intersect each ORIGINAL
/// triangle with those planes and take the resulting triangle union; this
/// retains the source geometry and works under arbitrary affine shear and
/// anisotropic scaling without interpreting the transform as a box kernel.
fn certified_convex_eroded_union(
    triangles: &[[f64; 2]],
    exterior: &[UnionBoundaryEdge],
    radius: f64,
    path: &[usize],
    aggregate_work: Option<&mut dyn PreparationWork>,
) -> Result<Option<Vec<[f64; 2]>>, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    if !radius.is_finite() || radius <= 0.0 {
        return Err(precision("convex erosion radius is not representable"));
    }
    if triangles.is_empty() || exterior.is_empty() || !triangles.len().is_multiple_of(3) {
        return Ok(None);
    }
    // Preserve the established independent visit and output class gates.
    // Actual orientations, clipping, copies and allocations pay separately
    // below. Checked pushes enforce the 3+k polygon inventory even if
    // rounded intersection coordinates fail to preserve convexity.
    let m = exterior.len();
    let triangle_count = triangles.len() / 3;
    let clipping_edges = m
        .checked_mul(3)
        .and_then(|n| {
            m.checked_mul(m.saturating_sub(1))
                .and_then(|q| q.checked_div(2))
                .and_then(|q| n.checked_add(q))
        })
        .ok_or_else(|| resource("convex erosion clipping work overflow"))?;
    let work = m
        .checked_mul(triangles.len())
        .and_then(|n| {
            clipping_edges
                .checked_mul(triangle_count)
                .and_then(|q| n.checked_add(q))
        })
        .ok_or_else(|| resource("convex erosion supporting-plane work overflow"))?;
    let retained_vertices = m
        .checked_add(1)
        .and_then(|n| n.checked_mul(triangles.len()))
        .ok_or_else(|| resource("convex erosion output size overflow"))?;
    if work > 16_777_216 || retained_vertices > MAX_NEUTRAL_MASK_SAMPLES {
        return Err(resource(
            "convex erosion supporting-plane work exceeds admission",
        ));
    }
    let mut standalone_work = 0;
    let aggregate_work = aggregate_work.unwrap_or(&mut standalone_work);
    charge_shadow_preparation_work(aggregate_work, 64, path)?;
    let mut planes = Vec::new();
    for &[a, b] in exterior {
        // Length/finite tests, supporting-side state and plane setup.
        charge_shadow_preparation_work(aggregate_work, 64, path)?;
        let length = (b[0] - a[0]).hypot(b[1] - a[1]);
        if !length.is_finite() || length == 0.0 {
            return Err(precision("convex erosion exposed boundary is degenerate"));
        }
        let mut side = 0.0_f64;
        for &point in triangles {
            // One six-arithmetic orientation, finite/sign/side tests,
            // indexed loads, state writes and loop progression.
            charge_shadow_preparation_work(aggregate_work, 32, path)?;
            let cross = orient(a, b, point);
            if !cross.is_finite() {
                return Err(precision("convex erosion supporting line lost precision"));
            }
            if cross == 0.0 {
                continue;
            }
            if side == 0.0 {
                side = cross.signum();
            } else if cross.signum() != side {
                // An exposed edge with triangle vertices on both sides
                // cannot define a supporting halfplane: reject certificate.
                return Ok(None);
            }
        }
        if side == 0.0 {
            return Ok(None);
        }
        push_erosion_buffer(&mut planes, (a, b, side, length), m, aggregate_work, path)?;
    }
    let mut output = Vec::new();
    let polygon_limit = m + 3;
    let mut polygon = Vec::new();
    let mut next = Vec::new();
    // Two bounded scratch buffers survive every triangle and plane. Neither
    // triangle copies nor clipping pushes may allocate infallibly.
    reserve_erosion_buffer(
        &mut polygon,
        polygon_limit,
        polygon_limit,
        aggregate_work,
        path,
    )?;
    reserve_erosion_buffer(
        &mut next,
        polygon_limit,
        polygon_limit,
        aggregate_work,
        path,
    )?;
    for triangle in triangles.as_chunks::<3>().0 {
        charge_shadow_preparation_work(aggregate_work, 64, path)?;
        if orient(triangle[0], triangle[1], triangle[2]) == 0.0 {
            continue;
        }
        polygon.clear();
        for &point in triangle {
            push_erosion_buffer(&mut polygon, point, polygon_limit, aggregate_work, path)?;
        }
        for &(a, b, side, length) in &planes {
            charge_shadow_preparation_work(aggregate_work, 64, path)?;
            if polygon.len() < 3 {
                break;
            }
            next.clear();
            let next_limit = polygon.len() + 1;
            for j in 0..polygon.len() {
                // Two orientations/distances and their finite tests, up to
                // one intersection, range/finite guards and indexed control.
                // Checked output pushes admit their own copies separately.
                charge_shadow_preparation_work(aggregate_work, 192, path)?;
                let p = polygon[j];
                let q = polygon[(j + 1) % polygon.len()];
                let dp = side * orient(a, b, p) / length - radius;
                let dq = side * orient(a, b, q) / length - radius;
                if !dp.is_finite() || !dq.is_finite() {
                    return Err(precision("convex erosion clip distance lost precision"));
                }
                if (dp >= 0.0) != (dq >= 0.0) {
                    let denominator = dp - dq;
                    if denominator == 0.0 {
                        return Err(precision("convex erosion clip intersection is singular"));
                    }
                    let t = dp / denominator;
                    if !t.is_finite() || !(0.0..=1.0).contains(&t) {
                        return Err(precision("convex erosion clip ratio is unrepresentable"));
                    }
                    let intersection = [
                        (q[0] - p[0]).mul_add(t, p[0]),
                        (q[1] - p[1]).mul_add(t, p[1]),
                    ];
                    if !intersection.iter().all(|v| v.is_finite()) {
                        return Err(precision("convex erosion intersection is not finite"));
                    }
                    // A next polygon exceeding n+1 vertices invalidates the
                    // admitted convex clipping inventory. Reject before the
                    // push, even when an old scratch allocation is larger.
                    push_erosion_buffer(&mut next, intersection, next_limit, aggregate_work, path)?;
                }
                if dq >= 0.0 {
                    push_erosion_buffer(&mut next, q, next_limit, aggregate_work, path)?;
                }
            }
            std::mem::swap(&mut polygon, &mut next);
        }
        if polygon.len() < 3 {
            continue;
        }
        for i in 1..polygon.len() - 1 {
            // Output orientation, finite/degenerate tests, extent admission,
            // three point copies and bounded loop progression.
            charge_shadow_preparation_work(aggregate_work, 64, path)?;
            let tri = [polygon[0], polygon[i], polygon[i + 1]];
            let area = orient(tri[0], tri[1], tri[2]);
            if !area.is_finite() {
                return Err(precision(
                    "convex erosion output triangle area lost precision",
                ));
            }
            if area == 0.0 {
                continue;
            }
            let size = output
                .len()
                .checked_add(3)
                .ok_or_else(|| resource("convex erosion triangle count overflow"))?;
            if size > MAX_NEUTRAL_MASK_SAMPLES {
                return Err(resource(
                    "convex erosion triangle retention exceeds admission",
                ));
            }
            reserve_erosion_buffer(&mut output, size, retained_vertices, aggregate_work, path)?;
            output.extend(tri);
        }
    }
    Ok(Some(output))
}

/// Certify a renderer-private nested *negative* Euclidean spread as exact
/// disposable triangles. Only a geometrically certified convex source union
/// admits polygonal disk erosion; nonconvex, holed or otherwise uncertified
/// unions return None and MUST NOT be treated as empty or approximated.
/// Some(empty mesh) is an independently certified fully erased region.
///
/// All coordinates are converted into the effect owner's immediate-parent
/// logical frame. This does not establish a new semantic representation or
/// reuse sampled alpha as geometry.
pub(super) fn certified_convex_erosion_mesh(
    source: &NeutralMesh,
    radius: f64,
    path: &[usize],
    aggregate_work: &mut dyn PreparationWork,
) -> Result<Option<NeutralMesh>, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    if !radius.is_finite() || radius <= 0.0 {
        return Err(precision(
            "nested convex erosion radius is not representable",
        ));
    }
    if !source.units_per_parent_logical_unit.is_finite()
        || source.units_per_parent_logical_unit <= 0.0
        || !source.triangles.len().is_multiple_of(3)
    {
        return Err(precision("nested convex erosion source is unrepresentable"));
    }
    let mut result = NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: Vec::new(),
        bounds: [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
    };
    if source.triangles.is_empty() {
        return Ok(Some(result));
    }
    // Charge contribution-wide inspection and normalization BEFORE another
    // potentially large temporary triangle payload is materialized.
    // Per vertex: zero initialization, two divisions, finite checks, two
    // stores and the later two abs/max reductions, including loop control.
    let normalization = source
        .triangles
        .len()
        .checked_mul(32)
        .and_then(|cost| cost.checked_add(64))
        .ok_or_else(|| resource("nested convex normalization work overflow"))?;
    charge_shadow_preparation_work(aggregate_work, normalization, path)?;
    let mut triangles = filled(source.triangles.len(), [0.0_f64; 2], path)?;
    for (out, &[x, y]) in triangles.iter_mut().zip(&source.triangles) {
        *out = [
            x / source.units_per_parent_logical_unit,
            y / source.units_per_parent_logical_unit,
        ];
        if !out.iter().all(|coord| coord.is_finite()) {
            return Err(precision(
                "nested convex erosion normalization lost precision",
            ));
        }
    }
    // A nonzero radius smaller than the owner-frame coordinate ULP cannot
    // reliably shift supporting planes. Reject such an invocation rather
    // than accidentally reproducing the un-eroded source as exact support.
    let largest_coordinate = triangles
        .iter()
        .flat_map(|point| point.iter())
        .fold(0.0_f64, |largest, value| largest.max(value.abs()));
    if radius <= largest_coordinate * (16.0 * f64::EPSILON) {
        return Err(precision(
            "nested convex erosion radius is below parent-coordinate precision",
        ));
    }
    // A rectangle is certified from the WHOLE triangle union, not its AABB.
    // Its Euclidean disk erosion is exactly an inset rectangle. This common
    // class avoids an otherwise redundant quadratic exposed-edge arrangement.
    let derived =
        if let Some(rect) = charge_and_certify_rectangle(&triangles, aggregate_work, path)? {
            // Inset arithmetic, finite/empty guards and six generated vertices.
            charge_shadow_preparation_work(aggregate_work, 160, path)?;
            let [x0, y0, x1, y1] = rect;
            let bounds = [x0 + radius, y0 + radius, x1 - radius, y1 - radius];
            if !bounds.iter().all(|coordinate| coordinate.is_finite()) {
                return Err(precision(
                    "nested certified rectangle erosion lost precision",
                ));
            }
            if bounds[0] >= bounds[2] || bounds[1] >= bounds[3] {
                Vec::new()
            } else {
                let [x0, y0, x1, y1] = bounds;
                vec![[x0, y0], [x1, y0], [x0, y1], [x1, y0], [x1, y1], [x0, y1]]
            }
        } else {
            let boundary = union_exterior_boundary(&triangles, path, Some(aggregate_work))?;
            let Some(derived) = certified_convex_eroded_union(
                &triangles,
                &boundary,
                radius,
                path,
                Some(aggregate_work),
            )?
            else {
                return Ok(None);
            };
            derived
        };
    if derived.len() > MAX_NEUTRAL_MASK_SAMPLES {
        return Err(resource(
            "nested convex erosion mesh exceeds bounded vertices",
        ));
    }
    // Each triangle orientation and each retained point's four bounds updates,
    // coordinate copies and push are paid before output allocation or scans.
    let output_work = derived
        .len()
        .checked_mul(24)
        .and_then(|cost| cost.checked_add(64))
        .ok_or_else(|| resource("nested convex output work overflow"))?;
    charge_shadow_preparation_work(aggregate_work, output_work, path)?;
    result
        .triangles
        .try_reserve_exact(derived.len())
        .map_err(|_| resource("nested convex erosion output allocation failed"))?;
    if result.triangles.capacity() > derived.len() {
        return Err(resource("nested convex output capacity exceeds admission"));
    }
    for triangle in derived.as_chunks::<3>().0 {
        if orient(triangle[0], triangle[1], triangle[2]) == 0.0 {
            continue;
        }
        for &point in triangle {
            result.bounds[0] = result.bounds[0].min(point[0]);
            result.bounds[1] = result.bounds[1].min(point[1]);
            result.bounds[2] = result.bounds[2].max(point[0]);
            result.bounds[3] = result.bounds[3].max(point[1]);
            result.triangles.push(point);
        }
    }
    Ok(Some(result))
}

/// Qualify an exact signed Euclidean erosion of multiple disconnected,
/// original-F1-proven axis-aligned rectangles. The caller MUST first prove
/// that every physical source rectangle corresponds to its original F1 item.
///
/// A Euclidean disk is connected: if closed rectangles are separated by a
/// strictly positive gap, any disk contained in their union lies entirely in
/// ONE rectangle. Therefore erosion distributes across these components,
/// and the exact eroded regions are inset rectangles. Touching/overlapping
/// rectangles, facets, foreign transforms and uncertain separation reject.
/// This is disposable geometry, never a source-neutral semantic authority.
pub(super) fn certified_separated_rectangles_erosion_mesh(
    source: &NeutralMesh,
    radius: f64,
    path: &[usize],
    aggregate_work: &mut dyn PreparationWork,
) -> Result<Option<NeutralMesh>, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    if !radius.is_finite()
        || radius <= 0.0
        || !source.units_per_parent_logical_unit.is_finite()
        || source.units_per_parent_logical_unit <= 0.0
    {
        return Err(precision(
            "separated F1 rectangle erosion parameters invalid",
        ));
    }
    if source.triangles.len() < 12 || !source.triangles.len().is_multiple_of(6) {
        return Ok(None);
    }
    let count = source.triangles.len() / 6;
    // Reject unbounded candidate inventories WITHOUT penalizing a legacy
    // convex mesh with a speculative quadratic work charge. The existing
    // convex proof retains authority for ineligible physical layouts.
    // An admitted candidate has at most 4,096 rectangles, <256 KiB of
    // rectangles and <=384 KiB of generated six-vertex output storage.
    if count
        .checked_mul(count)
        .is_none_or(|cost| cost > MAX_CERTIFIED_RECTANGLE_AXIS_WORK)
    {
        return Ok(None);
    }
    // A failed qualification still executed work. Debit the live counter
    // before each phase, retaining every paid debit on all return paths.
    charge_shadow_preparation_work(aggregate_work, 64, path)?;
    let mut rectangles = Vec::<[f64; 4]>::new();
    rectangles
        .try_reserve_exact(count)
        .map_err(|_| resource("separated rectangle proof allocation failed"))?;
    if rectangles.capacity() > count {
        return Err(resource(
            "separated rectangle proof capacity exceeds admission",
        ));
    }
    let mut largest = 0.0_f64;
    for triangles in source.triangles.as_chunks::<6>().0 {
        let Some(bounds) = charge_and_certify_rectangle(triangles, aggregate_work, path)? else {
            return Ok(None);
        };
        // Four divisions, finite checks, four abs/max updates and one copy.
        charge_shadow_preparation_work(aggregate_work, 64, path)?;
        let rect = bounds.map(|x| x / source.units_per_parent_logical_unit);
        if !rect.iter().all(|value| value.is_finite()) {
            return Err(precision(
                "separated rectangle normalization lost precision",
            ));
        }
        largest = rect.iter().fold(largest, |max, x| max.max(x.abs()));
        rectangles.push(rect);
    }
    // ONLY proven rectangle chunks may enter quadratic separation.
    // Previously accepted generic convex meshes must not encounter this
    // private 16M candidate budget before their usual convex fallback.
    let separation_work = count
        .checked_mul(count)
        .ok_or_else(|| resource("separated rectangle pairwise work overflow"))?;
    let next = aggregate_work
        .used()
        .checked_add(separation_work)
        .ok_or_else(|| resource("separated rectangle scene work overflow"))?;
    if next > MAX_CERTIFIED_RECTANGLE_AXIS_WORK {
        return Err(resource(
            "separated rectangle erosion exceeds 16M scene work",
        ));
    }
    // Preserve the established candidate/pair class gate independently of
    // actual arithmetic admission. At most four adds/comparisons and three
    // short-circuit tests per visited pair, plus indices and loop control.
    charge_shadow_preparation_work(aggregate_work, 64, path)?;
    if radius <= largest * (16.0 * f64::EPSILON) {
        return Err(precision(
            "separated rectangle erosion radius below coordinate precision",
        ));
    }
    // Strict outward margin ensures near-touching floating geometries do
    // not masquerade as topologically disconnected original rectangles.
    let separation_margin = largest.max(1.0) * (16.0 * f64::EPSILON);
    if !separation_margin.is_finite() {
        return Err(precision("separated rectangle distance lost precision"));
    }
    for (index, a) in rectangles.iter().enumerate() {
        for b in rectangles.iter().skip(index + 1) {
            aggregate_work.charge(
                32,
                MAX_CERTIFIED_RECTANGLE_AXIS_WORK,
                path,
                "separated rectangle erosion exceeds 16M scene work",
            )?;
            if !(a[2] + separation_margin < b[0]
                || b[2] + separation_margin < a[0]
                || a[3] + separation_margin < b[1]
                || b[3] + separation_margin < a[1])
            {
                return Ok(None);
            }
        }
    }
    let mut result = NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: Vec::new(),
        bounds: [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ],
    };
    let maximum_vertices = count
        .checked_mul(6)
        .ok_or_else(|| resource("separated rectangle output count overflow"))?;
    if maximum_vertices > MAX_NEUTRAL_MASK_SAMPLES {
        return Err(resource("separated rectangle output exceeds vertex limit"));
    }
    aggregate_work.charge(
        64,
        MAX_CERTIFIED_RECTANGLE_AXIS_WORK,
        path,
        "separated rectangle erosion exceeds 16M scene work",
    )?;
    result
        .triangles
        .try_reserve_exact(maximum_vertices)
        .map_err(|_| resource("separated rectangle output allocation failed"))?;
    if result.triangles.capacity() > maximum_vertices {
        return Err(resource(
            "separated rectangle output capacity exceeds admission",
        ));
    }
    for [x0, y0, x1, y1] in rectangles {
        // Inset/finite/empty guards and up to six points. Each point pays
        // four bounds updates, coordinate copies, push and loop progression.
        aggregate_work.charge(
            64 + 6 * 16,
            MAX_CERTIFIED_RECTANGLE_AXIS_WORK,
            path,
            "separated rectangle erosion exceeds 16M scene work",
        )?;
        let inset = [x0 + radius, y0 + radius, x1 - radius, y1 - radius];
        if !inset.iter().all(|value| value.is_finite()) {
            return Err(precision("separated rectangle inset lost precision"));
        }
        if inset[0] >= inset[2] || inset[1] >= inset[3] {
            continue;
        }
        let [left, top, right, bottom] = inset;
        for point in [
            [left, top],
            [right, top],
            [left, bottom],
            [right, top],
            [right, bottom],
            [left, bottom],
        ] {
            result.bounds[0] = result.bounds[0].min(point[0]);
            result.bounds[1] = result.bounds[1].min(point[1]);
            result.bounds[2] = result.bounds[2].max(point[0]);
            result.bounds[3] = result.bounds[3].max(point[1]);
            result.triangles.push(point);
        }
    }
    Ok(Some(result))
}

/// Certify that the complete triangle UNION is exactly one axis-aligned
/// rectangle. Each nondegenerate triangle must use three distinct rectangle
/// corners; two triangles missing opposite corners cover the whole rectangle.
/// This admits equivalent diagonals and duplicate tessellation without ever
/// treating the union AABB alone as geometry. All other meshes take the
/// general continuous signed-distance route.
pub(super) fn certified_axis_aligned_rectangle(triangles: &[[f64; 2]]) -> Option<[f64; 4]> {
    if triangles.len() < 6 || !triangles.len().is_multiple_of(3) {
        return None;
    }
    let min_x = triangles.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    let max_x = triangles
        .iter()
        .map(|p| p[0])
        .fold(f64::NEG_INFINITY, f64::max);
    let min_y = triangles.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let max_y = triangles
        .iter()
        .map(|p| p[1])
        .fold(f64::NEG_INFINITY, f64::max);
    if ![min_x, max_x, min_y, max_y].iter().all(|x| x.is_finite())
        || min_x >= max_x
        || min_y >= max_y
    {
        return None;
    }
    let corners = [
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
    ];
    let mut missing_mask = 0_u8;
    for tri in triangles.as_chunks::<3>().0 {
        let mut present = 0_u8;
        for point in tri {
            let index = corners.iter().position(|corner| corner == point)?;
            let bit = 1_u8 << index;
            if present & bit != 0 {
                return None;
            }
            present |= bit;
        }
        if present.count_ones() != 3 {
            return None;
        }
        missing_mask |= !present & 0b1111;
    }
    ((missing_mask & 0b0101 == 0b0101) || (missing_mask & 0b1010 == 0b1010))
        .then_some([min_x, min_y, max_x, max_y])
}

/// Pay for the four extrema scans and complete triangle/corner proof before
/// executing it. The caller may reuse this completed geometric certificate.
fn charge_and_certify_rectangle(
    triangles: &[[f64; 2]],
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<Option<[f64; 4]>, crate::execution_2d::Render2dExecutionError> {
    let cost = triangles
        .len()
        .checked_mul(48)
        .and_then(|cost| cost.checked_add(64))
        .ok_or_else(|| {
            mask_failure(
                path,
                crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                "rectangle certificate work overflow",
            )
        })?;
    charge_shadow_preparation_work(work, cost, path)?;
    Ok(certified_axis_aligned_rectangle(triangles))
}

/// Exact area of a certified axis-aligned inset rectangle under either
/// the identity blur (one physical sample cell) or the normalized finite
/// 3sigma Gaussian. A rectangular footprint never requires adaptive
/// triangulation or an approximated discrete source mask.
/// Exact clipped one-axis intersection (identity blur) or the same
/// normalized continuous 16-node Gaussian used by the single-site reference.
/// A certified rectangle is separable; do not recompute an identical x-axis
/// integral for every y row or introduce a sampled-alpha approximation.
fn inset_rectangle_axis_mass(start: f64, end: f64, center: f64, sigma: f64) -> f64 {
    let reach = if sigma > 0.0 { 3.0 * sigma } else { 0.5 };
    let a = start.max(center - reach);
    let b = end.min(center + reach);
    if a >= b {
        return 0.0;
    }
    if sigma > 0.0 {
        gaussian_axis_mass(a, b, center, sigma)
    } else {
        b - a
    }
}

/// Charge both axes and per-site multiplications before reserving private
/// x-axis or retained coverage storage. Failure leaves sibling work unchanged.
fn charge_certified_rectangle_axis_work(
    width: usize,
    height: usize,
    area: usize,
    sigma: f64,
    aggregate_work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<(), crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    let axis_cost = if sigma > 0.0 {
        GAUSSIAN_AXIS_WORK
    } else {
        IDENTITY_AXIS_WORK
    };
    let axis_work = width
        .checked_add(height)
        .and_then(|axes| axes.checked_mul(axis_cost))
        .and_then(|work| {
            area.checked_mul(COVERAGE_PRODUCT_WORK)
                .and_then(|cells| work.checked_add(cells))
        })
        .ok_or_else(|| resource("certified rectangle axis work overflow"))?;
    if axis_work > MAX_CERTIFIED_RECTANGLE_AXIS_WORK {
        return Err(resource(
            "certified rectangle axis work exceeds bounded admission",
        ));
    }
    charge_shadow_preparation_work(aggregate_work, axis_work, path)
}

/// Independent per-site comparator for the production shared-axis evaluator.
#[cfg(test)]
fn inset_rectangle_sample(rect: [f64; 4], center: [f64; 2], sigma: f64) -> f64 {
    let x_mass = inset_rectangle_axis_mass(rect[0], rect[2], center[0], sigma);
    let y_mass = inset_rectangle_axis_mass(rect[1], rect[3], center[1], sigma);
    (x_mass * y_mass).clamp(0.0, 1.0)
}

/// Deterministic high-order integration of one clipped Gaussian axis.
/// All intervals lie inside the finite 3sigma support; 16-point quadrature
/// on this smooth interval can be checked independently against erf.
fn gaussian_axis_mass(a: f64, b: f64, center: f64, sigma: f64) -> f64 {
    let half = (b - a) * 0.5;
    let midpoint = a + half;
    let integral = GAUSS_NODES
        .iter()
        .zip(GAUSS_WEIGHTS)
        .map(|(&node, weight)| {
            let z = (node.mul_add(half, midpoint) - center) / sigma;
            weight * (-0.5 * z * z).exp()
        })
        .sum::<f64>();
    integral * half / (sigma * GAUSS_3SIGMA_NORMALIZER)
}

/// Continuous positive Euclidean spread at one correlated sample. The
/// positive-sigma finite Gaussian uses its normalized probability footprint;
/// sigma=0 uses exact uniform unit-cell integration of the identity blur.
///
/// A distance-to-union query certifies an entire cell inside if d(center)+
/// half_diagonal <= radius, or outside if d(center)-half_diagonal > radius.
/// Unclassified Gaussian *probability mass* is bisected in both axes, always
/// refining the largest uncertain cell first. The final midpoint estimate is
/// therefore bracketed by independently computed lower/upper geometry
/// membership with absolute error <= 1/4096, otherwise this path rejects
/// with a typed precision/resource failure. No discrete source-mask
/// inference, box dilation or unknown-as-zero approximation is admitted.
fn continuous_signed_spread_sample(
    triangles: &[[f64; 2]],
    boundary: Option<&[UnionBoundaryEdge]>,
    center: [f64; 2],
    signed_spread: f64,
    sigma: f64,
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<f64, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    if !signed_spread.is_finite()
        || signed_spread == 0.0
        || !sigma.is_finite()
        || sigma < 0.0
        || (signed_spread < 0.0 && boundary.is_none())
    {
        return Err(precision(
            "continuous signed spread integration parameters are invalid",
        ));
    }
    let radius = signed_spread.abs();
    if !radius.is_finite() || radius == 0.0 {
        return Err(precision(
            "continuous positive spread integration parameters are invalid",
        ));
    }
    const MAX_GLOBAL_SPREAD_WORK: usize = 16_777_216;
    const MAX_SAMPLE_CELLS: usize = 32_768;
    const MAX_DEPTH: u8 = 22;
    const UNCERTAIN_MASS_TARGET: f64 = 1.0 / 2048.0;
    let cutoff = if sigma > 0.0 { 3.0 * sigma } else { 0.5 };
    let bounds = [
        center[0] - cutoff,
        center[1] - cutoff,
        center[0] + cutoff,
        center[1] + cutoff,
    ];
    if !bounds.iter().all(|v| v.is_finite()) || bounds[0] >= bounds[2] || bounds[1] >= bounds[3] {
        return Err(precision(
            "continuous spread Gaussian domain cannot be resolved",
        ));
    }
    let mut pending = std::collections::BinaryHeap::new();
    pending.push(SpreadGaussianCell {
        bounds,
        mass: 1.0,
        depth: 0,
    });
    let mut unknown = 1.0_f64;
    let mut covered = 0.0_f64;
    let mut visited = 0_usize;
    while unknown > UNCERTAIN_MASS_TARGET {
        let cell = pending.pop().ok_or_else(|| {
            precision("continuous positive spread Gaussian uncertainty lost its queue")
        })?;
        unknown -= cell.mass;
        visited += 1;
        // The conservative exterior-box pass may be followed by a signed
        // distance pass over the SAME boundary, so charge both traversals.
        // A geometry-cache optimization never weakens aggregate admission.
        let boundary_visits = boundary
            .map_or(0, <[UnionBoundaryEdge]>::len)
            .checked_mul(2)
            .ok_or_else(|| resource("continuous signed spread boundary-work overflow"))?;
        let classification_cost = (triangles.len() / 3)
            .checked_add(boundary_visits)
            .ok_or_else(|| resource("continuous signed spread classification work overflow"))?;
        if visited > MAX_SAMPLE_CELLS {
            return Err(resource(
                "continuous positive spread Gaussian exceeds bounded work",
            ));
        }
        work.charge(
            classification_cost,
            MAX_GLOBAL_SPREAD_WORK,
            path,
            "continuous positive spread Gaussian exceeds bounded work",
        )?;
        let [x0, y0, x1, y1] = cell.bounds;
        if signed_spread < 0.0
            && let Some(edges) = boundary
            && eroded_cell_far_from_exterior(edges, cell.bounds, radius)
        {
            // No exterior edge is within r of the cell. The polygon-union
            // interior predicate therefore holds throughout this connected
            // cell or nowhere, without any signed-distance subdivision.
            let point = [(x0 + x1) * 0.5, (y0 + y1) * 0.5];
            if neutral_union_distance(triangles, point) == 0.0 {
                covered += cell.mass;
            }
            continue;
        }
        if signed_spread > 0.0 && dilated_triangles_miss_cell(triangles, cell.bounds, radius) {
            continue;
        }
        if signed_spread > 0.0 && one_dilated_triangle_contains_cell(triangles, cell.bounds, radius)
        {
            covered += cell.mass;
            continue;
        }
        let midpoint = [(x0 + x1) * 0.5, (y0 + y1) * 0.5];
        let half_diagonal = ((x1 - x0) * 0.5).hypot((y1 - y0) * 0.5);
        // Signed Euclidean distance to the actual union exterior, not to
        // individual triangle edges: internal seams are NOT erosion holes.
        let distance = if let Some(edges) = boundary {
            union_signed_interior_distance(triangles, edges, midpoint)
        } else {
            neutral_union_distance(triangles, midpoint)
        };
        if !distance.is_finite() {
            return Err(precision("continuous spread distance is not representable"));
        }
        if signed_spread > 0.0 {
            if distance + half_diagonal <= radius {
                covered += cell.mass;
                continue;
            }
            if distance > radius + half_diagonal {
                continue;
            }
        } else {
            if distance - half_diagonal >= radius {
                covered += cell.mass;
                continue;
            }
            if distance + half_diagonal < radius {
                continue;
            }
        }
        if cell.depth >= MAX_DEPTH
            || midpoint[0] <= x0
            || midpoint[0] >= x1
            || midpoint[1] <= y0
            || midpoint[1] >= y1
        {
            return Err(precision(
                "continuous spread Gaussian cannot certify the required coverage tolerance",
            ));
        }
        let quadrants = [
            [x0, y0, midpoint[0], midpoint[1]],
            [midpoint[0], y0, x1, midpoint[1]],
            [x0, midpoint[1], midpoint[0], y1],
            [midpoint[0], midpoint[1], x1, y1],
        ];
        // Four quadrants have only two unique x and two unique y
        // integrals. Reuse each one instead of evaluating the same smooth
        // Gaussian axis four times at every subdivision.
        let (left, right, top, bottom) = if sigma > 0.0 {
            (
                gaussian_axis_mass(x0, midpoint[0], center[0], sigma),
                gaussian_axis_mass(midpoint[0], x1, center[0], sigma),
                gaussian_axis_mass(y0, midpoint[1], center[1], sigma),
                gaussian_axis_mass(midpoint[1], y1, center[1], sigma),
            )
        } else {
            // The sigma-zero kernel is the identity. Physical 4x coverage
            // integrates its exact geometric indicator over the associated
            // unit sample cell, as for the established zero-spread area path.
            (
                midpoint[0] - x0,
                x1 - midpoint[0],
                midpoint[1] - y0,
                y1 - midpoint[1],
            )
        };
        let weights = [left * top, right * top, left * bottom, right * bottom];
        let total = weights.iter().sum::<f64>();
        if !total.is_finite() || total <= 0.0 {
            return Err(precision(
                "continuous spread Gaussian subdivision mass vanished",
            ));
        }
        // Preserve the exact parent's normalized mass under floating-point
        // quadrature; child partition cannot invent or lose Gaussian mass.
        for (region, weight) in quadrants.into_iter().zip(weights) {
            let mass = cell.mass * (weight / total);
            if mass > 0.0 {
                pending.push(SpreadGaussianCell {
                    bounds: region,
                    mass,
                    depth: cell.depth + 1,
                });
                unknown += mass;
            }
        }
    }
    let alpha = covered + unknown * 0.5;
    if !alpha.is_finite() || !(-1.0e-9..=1.0 + 1.0e-9).contains(&alpha) {
        return Err(precision(
            "continuous spread Gaussian mass is not normalized",
        ));
    }
    Ok(alpha.clamp(0.0, 1.0))
}

/// Reconstruct neutral continuous coverage on the correlated physical grid:
/// sigma>0 integrates the finite Gaussian over geometric support, while
/// positive spread at sigma=0 integrates its exact Euclidean indicator in
/// the corresponding unit sample cell. Ancestor affines/opacity remain
/// separate; no paint alpha or sampled geometry defines F1 support.
fn rasterize_continuous_shadow_coverage(
    mesh: &NeutralMesh,
    signed_spread: f64,
    sigma: f64,
    scale: f64,
    path: &[usize],
    aggregate_work: &mut dyn PreparationWork,
    remaining_coverage_samples: usize,
) -> Result<Option<NeutralCoverage>, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    let Some(grid) = rasterize_mesh_with_positive_spread(
        mesh,
        scale,
        signed_spread.max(0.0),
        path,
        Some(aggregate_work),
    )?
    else {
        return Ok(None);
    };
    let physical_spread = signed_spread * scale;
    let physical_sigma = sigma * scale;
    let cutoff = physical_sigma * 3.0;
    if !physical_sigma.is_finite()
        || physical_sigma < 0.0
        || (physical_sigma == 0.0 && signed_spread == 0.0)
        || !cutoff.is_finite()
        || cutoff.ceil() > f64::from(MAX_GAUSSIAN_RADIUS_SAMPLES)
    {
        return Err(resource(
            "continuous Gaussian window exceeds bounded sample halo",
        ));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let pad = cutoff.ceil() as usize;
    let width = grid
        .width
        .checked_add(
            pad.checked_mul(2)
                .ok_or_else(|| resource("continuous Gaussian width halo overflow"))?,
        )
        .ok_or_else(|| resource("continuous Gaussian width overflow"))?;
    let height = grid
        .height
        .checked_add(
            pad.checked_mul(2)
                .ok_or_else(|| resource("continuous Gaussian height halo overflow"))?,
        )
        .ok_or_else(|| resource("continuous Gaussian height overflow"))?;
    let area = width
        .checked_mul(height)
        .ok_or_else(|| resource("continuous Gaussian area overflow"))?;
    if area > MAX_NEUTRAL_MASK_SAMPLES {
        return Err(resource(
            "continuous Gaussian coverage exceeds bounded sample area",
        ));
    }
    // The original source-lattice budget does not include the expanded
    // 3sigma halo. Every halo cell scans the triangle inventory before local
    // active-set selection, so preflight total traversal work before new
    // coverage storage or caller-owned GPU work.
    let scan_work = area
        .checked_mul(mesh.triangles.len() / 3)
        .ok_or_else(|| resource("continuous Gaussian halo scan work overflow"))?;
    if scan_work > 16_777_216 {
        return Err(resource(
            "continuous Gaussian halo scan exceeds bounded work",
        ));
    }
    charge_shadow_preparation_work(aggregate_work, scan_work, path)?;
    let offset = i64::try_from(pad).map_err(|_| resource("continuous Gaussian halo overflow"))?;
    let origin_x = grid
        .origin_x
        .checked_sub(offset)
        .ok_or_else(|| precision("continuous Gaussian x sample phase overflow"))?;
    let origin_y = grid
        .origin_y
        .checked_sub(offset)
        .ok_or_else(|| precision("continuous Gaussian y sample phase overflow"))?;
    let to_samples = scale / mesh.units_per_parent_logical_unit;
    if !to_samples.is_finite() || to_samples <= 0.0 {
        return Err(precision(
            "continuous Gaussian triangle projection is invalid",
        ));
    }
    let mut triangles = filled(mesh.triangles.len(), [0.0; 2], path)?;
    for (dst, &[x, y]) in triangles.iter_mut().zip(&mesh.triangles) {
        *dst = [x * to_samples, y * to_samples];
        if !dst.iter().all(|v| v.is_finite()) {
            return Err(precision(
                "continuous Gaussian triangle projection lost precision",
            ));
        }
    }
    // A certified rectangular triangle UNION admits an exact inset for
    // negative disk erosion. This avoids a long adaptive search along
    // straight edges, while retaining the same continuous geometry law.
    // An AABB without the full two-triangle coverage proof never qualifies.
    let inset = if signed_spread <= 0.0 {
        charge_and_certify_rectangle(&triangles, aggregate_work, path)?.map(|rect| {
            [
                rect[0] - physical_spread,
                rect[1] - physical_spread,
                rect[2] + physical_spread,
                rect[3] + physical_spread,
            ]
        })
    } else {
        None
    };
    if let Some(rect) = inset {
        if !rect.iter().all(|v| v.is_finite()) {
            return Err(precision(
                "certified rectangular erosion lost finite coordinates",
            ));
        }
        if rect[0] >= rect[2] || rect[1] >= rect[3] {
            // Empty or measure-zero continuous support is a geometrically
            // certified empty paint source, not inferred from absent pixels.
            return Ok(None);
        }
    }
    // Nonrectangular erosion still requires the actual exposed union
    // boundary. Positive dilation and certified rectangular erosion do not.
    let erosion_boundary = if signed_spread < 0.0 && inset.is_none() {
        Some(union_exterior_boundary(
            &triangles,
            path,
            Some(aggregate_work),
        )?)
    } else {
        None
    };
    // A certified convex union admits an EXACT Euclidean inward offset as
    // another triangle union. Integrate that derived geometry under the same
    // continuous area/Gaussian law; never spend adaptive subdivision work on
    // known straight supporting planes or substitute an enclosing rectangle.
    let convex_inset = if let Some(exterior) = &erosion_boundary {
        certified_convex_eroded_union(
            &triangles,
            exterior,
            -physical_spread,
            path,
            Some(aggregate_work),
        )?
    } else {
        None
    };
    if convex_inset.as_ref().is_some_and(Vec::is_empty) {
        return Ok(None);
    }
    // For a certified orthogonal TRUE exterior, sigma-zero erosion uses
    // provable per-cell area enclosures, while finite Gaussian reuses exact
    // exterior/capsule y sections across the physical x samples of each row.
    // Oblique cases retain the general integrator and typed failure.
    let orthogonal = if signed_spread < 0.0 && convex_inset.is_none() {
        erosion_boundary
            .as_deref()
            .map(|boundary| OrthogonalExterior::new(boundary, path, aggregate_work))
            .transpose()?
            .flatten()
    } else {
        None
    };
    // Complete-erosion certificates above can return without retained
    // coverage even when the scene's prior shadows exhausted its allowance.
    admit_retained_coverage_samples(area, remaining_coverage_samples, path)?;
    // The certified rectangular inset is a separable continuous integral.
    // Reuse ONE horizontal axis over all rows; each vertical axis is evaluated
    // once per row. Admit *actual* quadrature and multiplication work before
    // even allocating the bounded horizontal f64 workspace or float coverage.
    // The path is strictly private to an already-certified axis rectangle:
    // other F1 source shapes retain their existing exact union evaluator.
    let x_axis = if let Some(rect) = inset {
        charge_certified_rectangle_axis_work(
            width,
            height,
            area,
            physical_sigma,
            aggregate_work,
            path,
        )?;
        // width <= admitted area <= MAX_NEUTRAL_MASK_SAMPLES, so the
        // temporary x-axis buffer is at most 8 MiB with checked allocation.
        let mut xs = filled(width, 0.0_f64, path)?;
        for (column, x_mass) in xs.iter_mut().enumerate() {
            let center_x = (origin_x as f64 + as_f64(column)) + 0.5;
            *x_mass = inset_rectangle_axis_mass(rect[0], rect[2], center_x, physical_sigma);
        }
        Some(xs)
    } else {
        None
    };
    let mut values = filled(area, 0.0_f64, path)?;
    for row in 0..height {
        let y_mass = inset.map(|rect| {
            let center_y = (origin_y as f64 + as_f64(row)) + 0.5;
            inset_rectangle_axis_mass(rect[1], rect[3], center_y, physical_sigma)
        });
        if let Some(orthogonal) = &orthogonal
            && physical_sigma > 0.0
        {
            let center_y = (origin_y as f64 + as_f64(row)) + 0.5;
            orthogonal.gaussian_row(
                GaussianRow {
                    y: center_y,
                    origin_x: origin_x as f64,
                    width,
                    radius: -physical_spread,
                    sigma: physical_sigma,
                },
                &mut values[row * width..(row + 1) * width],
                aggregate_work,
                path,
            )?;
            continue;
        }
        for column in 0..width {
            let center = [
                (origin_x as f64 + as_f64(column)) + 0.5,
                (origin_y as f64 + as_f64(row)) + 0.5,
            ];
            values[row * width + column] = if let (Some(xs), Some(y_mass)) = (&x_axis, y_mass) {
                (xs[column] * y_mass).clamp(0.0, 1.0)
            } else if let Some(polygon) = &convex_inset {
                if physical_sigma > 0.0 {
                    gaussian_union_sample(polygon, center, physical_sigma, aggregate_work, path)?
                } else {
                    area_sample(
                        polygon,
                        center[0] - 0.5,
                        center[1] - 0.5,
                        aggregate_work,
                        path,
                    )?
                }
            } else if let Some(orthogonal) = &orthogonal {
                orthogonal.area(
                    [
                        center[0] - 0.5,
                        center[1] - 0.5,
                        center[0] + 0.5,
                        center[1] + 0.5,
                    ],
                    -physical_spread,
                    aggregate_work,
                    path,
                )?
            } else if signed_spread != 0.0 {
                continuous_signed_spread_sample(
                    &triangles,
                    erosion_boundary.as_deref(),
                    center,
                    physical_spread,
                    physical_sigma,
                    aggregate_work,
                    path,
                )?
            } else {
                gaussian_union_sample(&triangles, center, physical_sigma, aggregate_work, path)?
            };
        }
    }
    Ok(Some(NeutralCoverage {
        origin_x,
        origin_y,
        width,
        height,
        values,
    }))
}

/// Separable finite 3σ Gaussian-style convolution over the spread mask.
/// The kernel is the normalized discrete approximation of the accepted F1
/// continuous truncated reference, with an explicit finite, complete halo.
/// Every allocation and worst-case sample tap is admitted before execution.
#[cfg(test)]
fn blur_neutral_coverage(
    input: &NeutralCoverage,
    kernel: &GaussianKernel,
    path: &[usize],
) -> Result<NeutralCoverage, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    let source_count = input
        .width
        .checked_mul(input.height)
        .ok_or_else(|| resource("neutral blur source extent overflow"))?;
    if source_count == 0
        || source_count > MAX_NEUTRAL_MASK_SAMPLES
        || input.values.len() != source_count
    {
        return Err(resource("neutral blur source coverage exceeds bounds"));
    }
    if !input
        .values
        .iter()
        .all(|v| v.is_finite() && (0.0..=1.0).contains(v))
    {
        return Err(precision("neutral blur input has invalid coverage"));
    }
    if kernel.radius
        > usize::try_from(MAX_GAUSSIAN_RADIUS_SAMPLES).expect("fixed Gaussian maximum fits usize")
        || kernel.weights.len() != kernel.radius + 1
        || !kernel
            .weights
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
    {
        return Err(precision("neutral blur kernel is malformed"));
    }
    let weight_sum = kernel.weights[0] + 2.0 * kernel.weights.iter().skip(1).sum::<f64>();
    if !weight_sum.is_finite() || (weight_sum - 1.0).abs() > 1.0e-9 {
        return Err(precision("neutral blur kernel is not normalized"));
    }
    let pad = kernel.radius;
    let width = input
        .width
        .checked_add(
            pad.checked_mul(2)
                .ok_or_else(|| resource("neutral blur padding overflow"))?,
        )
        .ok_or_else(|| resource("neutral blur width overflow"))?;
    let height = input
        .height
        .checked_add(
            pad.checked_mul(2)
                .ok_or_else(|| resource("neutral blur padding overflow"))?,
        )
        .ok_or_else(|| resource("neutral blur height overflow"))?;
    let area = width
        .checked_mul(height)
        .ok_or_else(|| resource("neutral blur sample area overflow"))?;
    if area > MAX_NEUTRAL_MASK_SAMPLES {
        return Err(resource("neutral blur halo exceeds bounded sample area"));
    }
    let taps = pad
        .checked_mul(2)
        .and_then(|v| v.checked_add(1))
        .ok_or_else(|| resource("neutral blur tap count overflow"))?;
    let work = area
        .checked_mul(taps)
        .and_then(|v| v.checked_mul(2))
        .ok_or_else(|| resource("neutral blur work count overflow"))?;
    if work > 16_777_216 {
        return Err(resource("neutral blur exceeds bounded sample-tap work"));
    }
    let pad_i64 =
        i64::try_from(pad).map_err(|_| resource("neutral blur origin padding overflow"))?;
    let origin_x = input
        .origin_x
        .checked_sub(pad_i64)
        .ok_or_else(|| precision("neutral blur x origin overflow"))?;
    let origin_y = input
        .origin_y
        .checked_sub(pad_i64)
        .ok_or_else(|| precision("neutral blur y origin overflow"))?;

    let mut source = filled(area, 0.0_f64, path)?;
    for y in 0..input.height {
        for x in 0..input.width {
            source[(y + pad) * width + x + pad] = input.values[y * input.width + x];
        }
    }
    let mut horizontal = filled(area, 0.0_f64, path)?;
    let mut values = filled(area, 0.0_f64, path)?;
    for y in 0..height {
        for x in 0..width {
            let mut coverage = kernel.weights[0] * source[y * width + x];
            for offset in 1..=pad {
                let weight = kernel.weights[offset];
                if let Some(left) = x.checked_sub(offset) {
                    coverage = weight.mul_add(source[y * width + left], coverage);
                }
                if let Some(right) = x.checked_add(offset)
                    && right < width
                {
                    coverage = weight.mul_add(source[y * width + right], coverage);
                }
            }
            horizontal[y * width + x] = coverage;
        }
    }
    for y in 0..height {
        for x in 0..width {
            let mut coverage = kernel.weights[0] * horizontal[y * width + x];
            for offset in 1..=pad {
                let weight = kernel.weights[offset];
                if let Some(top) = y.checked_sub(offset) {
                    coverage = weight.mul_add(horizontal[top * width + x], coverage);
                }
                if let Some(bottom) = y.checked_add(offset)
                    && bottom < height
                {
                    coverage = weight.mul_add(horizontal[bottom * width + x], coverage);
                }
            }
            values[y * width + x] = coverage.clamp(0.0, 1.0);
        }
    }
    Ok(NeutralCoverage {
        origin_x,
        origin_y,
        width,
        height,
        values,
    })
}

/// Test-only historical sampled-mask comparison. The production F1 shadow
/// path reconstructs continuous neutral geometry and never samples cached
/// binary alpha as effect authority.
#[cfg(test)]
pub(super) fn blur_neutral_mask(
    input: &NeutralMask,
    kernel: &GaussianKernel,
    path: &[usize],
) -> Result<NeutralCoverage, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let count = input.width.checked_mul(input.height).ok_or_else(|| {
        mask_failure(
            path,
            Render2dSampleSpaceError::ResourceLimit,
            "neutral mask extent overflow",
        )
    })?;
    if count == 0 || count > MAX_NEUTRAL_MASK_SAMPLES || count != input.samples.len() {
        return Err(mask_failure(
            path,
            Render2dSampleSpaceError::ResourceLimit,
            "neutral blur source mask exceeds bounds",
        ));
    }
    let mut values = filled(count, 0.0_f64, path)?;
    for (dst, src) in values.iter_mut().zip(&input.samples) {
        *dst = f64::from(*src) / f64::from(u8::MAX);
    }
    blur_neutral_coverage(
        &NeutralCoverage {
            origin_x: input.origin_x,
            origin_y: input.origin_y,
            width: input.width,
            height: input.height,
            values,
        },
        kernel,
        path,
    )
}

/// Prepares one immutable neutral support source through signed Euclidean
/// spread and finite Gaussian coverage without deriving shape from paint alpha.
///
/// This handles only the **untranslated** coverage in the attached group's
/// immediate-parent frame. F3F group composition owns shadow offset, clips,
/// ordered colors, once-only opacity and ancestor transforms; applying those
/// here in a flattened final-target frame would change F1 meaning.
#[allow(dead_code, reason = "awaiting F3F group composition")]
pub(super) fn prepare_untranslated_shadow_coverage(
    mesh: &NeutralMesh,
    spread: f64,
    sigma: f64,
    samples_per_logical_unit: f64,
    path: &[usize],
) -> Result<Option<NeutralCoverage>, crate::execution_2d::Render2dExecutionError> {
    let mut aggregate_work = 0;
    prepare_shadow_coverage_admitted(
        mesh,
        spread,
        sigma,
        samples_per_logical_unit,
        path,
        &mut aggregate_work,
        MAX_NEUTRAL_MASK_SAMPLES,
    )
}

/// Same continuous neutral geometry law for production multi-effect scenes.
/// The caller retains this work counter across every sibling and nested owner.
pub(super) fn prepare_shadow_coverage_admitted(
    mesh: &NeutralMesh,
    spread: f64,
    sigma: f64,
    samples_per_logical_unit: f64,
    path: &[usize],
    aggregate_work: &mut dyn PreparationWork,
    remaining_coverage_samples: usize,
) -> Result<Option<NeutralCoverage>, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    if !spread.is_finite() || !sigma.is_finite() || sigma < 0.0 {
        return Err(mask_failure(
            path,
            Render2dSampleSpaceError::PrecisionLimit,
            "shadow spread or blur sigma is not representable",
        ));
    }
    // The caster lies inside its measured semantic-geometry bounding box.
    // A Euclidean disk with radius strictly greater than half the box's
    // smallest width cannot fit inside it. Equality can leave boundary-only
    // support, so it is not an emptiness certificate. Certify complete erosion
    // before bounded halo work, including oversized finite radii.
    // This is a sufficient empty-set proof, never a substitute for nonempty
    // continuous morphology.
    if spread < 0.0 && !mesh.triangles.is_empty() {
        let smallest_extent = (mesh.bounds[2] - mesh.bounds[0])
            .min(mesh.bounds[3] - mesh.bounds[1])
            / mesh.units_per_parent_logical_unit;
        if smallest_extent.is_finite() && smallest_extent >= 0.0 && -spread > smallest_extent / 2.0
        {
            return Ok(None);
        }
    }
    // Integrate the continuous geometric union even for sigma == 0:
    // a real caster must not vanish merely because its area misses all
    // correlated lattice centers. Gaussian convolution preserves that mass.
    if spread == 0.0 {
        if sigma > 0.0 {
            return rasterize_continuous_shadow_coverage(
                mesh,
                0.0,
                sigma,
                samples_per_logical_unit,
                path,
                aggregate_work,
                remaining_coverage_samples,
            );
        }
        return rasterize_area_coverage(
            mesh,
            samples_per_logical_unit,
            path,
            aggregate_work,
            remaining_coverage_samples,
        );
    }
    if spread > 0.0 {
        return rasterize_continuous_shadow_coverage(
            mesh,
            spread,
            sigma,
            samples_per_logical_unit,
            path,
            aggregate_work,
            remaining_coverage_samples,
        );
    }
    // Continuous negative Euclidean erosion uses the boundary-certified
    // signed distance and the same deterministic mass/area integration as
    // positive spread. An all-zero sampled result is NOT proof that the true
    // eroded support is empty; only the bounding extent certificate above
    // can return None, otherwise reject with a truthful precision outcome.
    let coverage = rasterize_continuous_shadow_coverage(
        mesh,
        spread,
        sigma,
        samples_per_logical_unit,
        path,
        aggregate_work,
        remaining_coverage_samples,
    )?;
    if coverage
        .as_ref()
        .is_some_and(|c| c.values.iter().all(|v| *v == 0.0))
    {
        return Err(mask_failure(
            path,
            Render2dSampleSpaceError::PrecisionLimit,
            "continuous erosion support cannot be proved empty from zero output-cell coverage",
        ));
    }
    Ok(coverage)
}

/// Signed Euclidean disk morphology on a *disposable* aligned binary grid.
/// Positive radii dilate; negative radii erode. The padded exterior is empty,
/// so narrow support can erode away completely. Dilation retains offscreen
/// support beyond the input mask's bounds. All work/extent admission precedes
/// allocation and uses exact occurrence paths in typed errors.
#[allow(dead_code, reason = "F3F signed spread awaits group-effect lowering")]
pub(super) fn signed_euclidean_spread(
    input: &NeutralMask,
    radius: f64,
    path: &[usize],
) -> Result<NeutralMask, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let precision = |detail| mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail);
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    if !radius.is_finite() {
        return Err(precision("neutral signed radius is not finite"));
    }
    if input.width == 0 || input.height == 0 {
        return Err(precision("neutral mask extent is empty"));
    }
    let count = input
        .width
        .checked_mul(input.height)
        .ok_or_else(|| resource("neutral source mask extent overflow"))?;
    if count > MAX_NEUTRAL_MASK_SAMPLES || input.samples.len() != count {
        return Err(resource(
            "neutral source mask exceeds the admitted sample budget",
        ));
    }
    if radius == 0.0 || input.samples.iter().all(|value| *value == 0) {
        return Ok(input.clone());
    }
    let magnitude = radius.abs();
    let pad_f64 = magnitude.ceil() + 1.0;
    if !pad_f64.is_finite() || pad_f64 > f64::from(MAX_SPREAD_RADIUS_SAMPLES) {
        return Err(resource(
            "Euclidean spread halo exceeds the bounded sample radius",
        ));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite nonnegative halo is proven bounded by 512"
    )]
    let pad = pad_f64 as usize;
    let width = input
        .width
        .checked_add(
            pad.checked_mul(2)
                .ok_or_else(|| resource("neutral mask padding overflow"))?,
        )
        .ok_or_else(|| resource("neutral mask width overflow"))?;
    let height = input
        .height
        .checked_add(
            pad.checked_mul(2)
                .ok_or_else(|| resource("neutral mask padding overflow"))?,
        )
        .ok_or_else(|| resource("neutral mask height overflow"))?;
    let cells = width
        .checked_mul(height)
        .ok_or_else(|| resource("neutral mask sample count overflow"))?;
    if cells > MAX_NEUTRAL_MASK_SAMPLES {
        return Err(resource(
            "padded neutral mask exceeds bounded sample budget",
        ));
    }
    let pad_i64 = i64::try_from(pad).map_err(|_| resource("neutral mask origin overflow"))?;
    let origin_x = input
        .origin_x
        .checked_sub(pad_i64)
        .ok_or_else(|| precision("signed neutral mask x origin overflow"))?;
    let origin_y = input
        .origin_y
        .checked_sub(pad_i64)
        .ok_or_else(|| precision("signed neutral mask y origin overflow"))?;
    let mut padded = filled(cells, 0_u8, path)?;
    for y in 0..input.height {
        let from = y * input.width;
        let to = (y + pad) * width + pad;
        padded[to..to + input.width].copy_from_slice(&input.samples[from..from + input.width]);
    }
    let squared = squared_distance_2d(&padded, width, height, radius > 0.0, path)?;
    let threshold = magnitude * magnitude;
    if !threshold.is_finite() {
        return Err(precision("Euclidean spread radius cannot be squared"));
    }
    if radius > 0.0 {
        for (value, distance) in padded.iter_mut().zip(squared.iter()) {
            *value = if *distance <= threshold { u8::MAX } else { 0 };
        }
        return Ok(NeutralMask {
            origin_x,
            origin_y,
            width,
            height,
            samples: padded,
        });
    }
    let mut eroded = filled(count, 0_u8, path)?;
    for y in 0..input.height {
        for x in 0..input.width {
            // The padded exterior guarantees the nearest empty sample exists.
            let squared_distance = squared[(y + pad) * width + x + pad];
            if input.samples[y * input.width + x] != 0 && squared_distance > threshold {
                eroded[y * input.width + x] = u8::MAX;
            }
        }
    }
    Ok(NeutralMask {
        origin_x: input.origin_x,
        origin_y: input.origin_y,
        width: input.width,
        height: input.height,
        samples: eroded,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_positive_disk_area_is_euclidean_and_preserves_sheared_quarter_pixel() {
        let square = [
            [2.0, 2.0],
            [4.0, 2.0],
            [2.0, 4.0],
            [4.0, 2.0],
            [4.0, 4.0],
            [2.0, 4.0],
        ];
        let identity = AffinePositiveDiskArea {
            triangles: &square,
            certified_rectangle: Some([2.0, 2.0, 4.0, 4.0]),
            inverse: [1.0, 0.0, 0.0, 1.0],
            root_translation: [0.0, 0.0],
            radius: 1.0,
        };
        let mut work = 0;
        let corner = identity.sample([1.0, 1.0], &mut work, &[0]).unwrap();
        // The positive disk outside a convex rectangular corner is one
        // quarter circle. A square-expanded AABB would incorrectly give 1.
        assert!((corner - std::f64::consts::FRAC_PI_4).abs() < 0.0004);
        assert_eq!(identity.sample([0.0, 0.0], &mut work, &[0]).unwrap(), 0.0);
        assert_eq!(identity.sample([3.0, 3.0], &mut work, &[0]).unwrap(), 1.0);
        assert!(work > 0);
        let mut exhausted = 16_777_216_usize;
        assert!(matches!(
            identity.sample([1.0, 1.0], &mut exhausted, &[3, 2]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [3, 2]
        ));

        let parent_square = [
            [10.0, 10.0],
            [20.0, 10.0],
            [10.0, 20.0],
            [20.0, 10.0],
            [20.0, 20.0],
            [10.0, 20.0],
        ];
        // Root x=2*parent_x +.5*parent_y+4, root y=parent_y.
        let sheared = AffinePositiveDiskArea {
            triangles: &parent_square,
            certified_rectangle: Some([10.0, 10.0, 20.0, 20.0]),
            inverse: [0.5, -0.25, 0.0, 1.0],
            root_translation: [4.0, 0.0],
            radius: 1.0,
        };
        let area = sheared.sample([29.0, 15.0], &mut work, &[0]).unwrap();
        assert!(
            (area - 0.25).abs() < 0.0004,
            "affine parallelogram cell is not a parent-axis square: {area}"
        );
    }

    #[test]
    fn affine_positive_disk_interior_uses_convex_cell_certificate_for_tiny_radius() {
        let large_triangle = [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0]];
        let area = AffinePositiveDiskArea {
            triangles: &large_triangle,
            certified_rectangle: None,
            inverse: [1.0, 0.0, 0.0, 1.0],
            root_translation: [0.0, 0.0],
            radius: 1.0e-7,
        };
        let mut work = 0_usize;
        assert_eq!(area.sample([20.0, 10.0], &mut work, &[1]).unwrap(), 1.0);
        assert!(
            work <= 8,
            "certified triangle interior must not require deep disk subdivision"
        );
    }

    #[test]
    fn certified_rounded_rectangle_admits_large_provable_interiors_and_exteriors() {
        let square = [
            [10.0, 10.0],
            [110.0, 10.0],
            [10.0, 110.0],
            [110.0, 10.0],
            [110.0, 110.0],
            [10.0, 110.0],
        ];
        let area = AffinePositiveDiskArea {
            triangles: &square,
            certified_rectangle: Some([10.0, 10.0, 110.0, 110.0]),
            inverse: [1.0, 0.0, 0.0, 1.0],
            root_translation: [0.0, 0.0],
            radius: 1.0,
        };
        let mut work = 0;
        for y in 20..100 {
            for x in 20..100 {
                assert_eq!(
                    area.sample([f64::from(x), f64::from(y)], &mut work, &[0])
                        .unwrap(),
                    1.0
                );
            }
        }
        for y in 120..200 {
            for x in 120..200 {
                assert_eq!(
                    area.sample([f64::from(x), f64::from(y)], &mut work, &[0])
                        .unwrap(),
                    0.0
                );
            }
        }
        assert_eq!(work, 80 * 80 * 2 * 8);
        // Even this highly amortized path respects the cumulative budget.
        let mut exhausted = 16_777_216;
        assert!(matches!(
            area.sample([50.0, 50.0], &mut exhausted, &[7, 4]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [7, 4]
        ));
    }

    #[test]
    fn affine_disk_area_work_exhaustion_is_typed_and_path_bearing() {
        let triangles = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let area = AffinePositiveDiskArea {
            triangles: &triangles,
            certified_rectangle: None,
            inverse: [1.0, 0.0, 0.0, 1.0],
            root_translation: [0.0, 0.0],
            radius: 0.5,
        };
        let mut work = 16_777_216;
        assert!(matches!(
            area.sample([0.0, 0.0], &mut work, &[2, 1]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [2, 1]
        ));
    }

    #[test]
    fn certified_nested_convex_erosion_uses_true_sheared_halfplanes_not_aabb() {
        // The caster is a sheared parallelogram, NOT its [10,25]x[10,20]
        // enclosing rectangle. Its exact inward Euclidean disk offset
        // remains an actual convex polygon under any later ancestor affine.
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [10.0, 10.0],
                [20.0, 10.0],
                [25.0, 20.0],
                [10.0, 10.0],
                [25.0, 20.0],
                [15.0, 20.0],
            ],
            bounds: [10.0, 10.0, 25.0, 20.0],
        };
        let mut work = 0_usize;
        let region = certified_convex_erosion_mesh(&source, 1.0, &[0, 0], &mut work)
            .unwrap()
            .expect("a sheared convex union has a certified exact erosion");
        assert!(work > 0);
        assert!(!region.triangles.is_empty());
        assert_eq!(neutral_union_distance(&region.triangles, [17.0, 15.0]), 0.0);
        assert!(
            neutral_union_distance(&region.triangles, [15.0, 10.5]) > 0.0,
            "a point outside the true disk-eroded parallelogram is not admitted"
        );
    }

    #[test]
    fn nested_convex_inset_rejects_unresolvable_owner_frame_radius() {
        let large = 1.0e12;
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [large, large],
                [large + 100.0, large],
                [large, large + 100.0],
                [large + 100.0, large],
                [large + 100.0, large + 100.0],
                [large, large + 100.0],
            ],
            bounds: [large, large, large + 100.0, large + 100.0],
        };
        let mut work = 0;
        assert!(matches!(
            certified_convex_erosion_mesh(&source, 1.0e-6, &[7], &mut work),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [7]
        ));
    }

    #[test]
    fn shared_erosion_work_preserves_both_limits_and_rejected_debits() {
        let mut local = 100;
        let mut graph = MAX_COMPOSITIONAL_SHADOW_WORK - 10;
        {
            let mut work = SharedPreparationWork::new(&mut local, &mut graph);
            assert_eq!(work.remaining(MAX_AGGREGATE_SHADOW_PREPARATION_WORK), 10);
            charge_shadow_preparation_work(&mut work, 7, &[4, 2]).unwrap();
            let rejected = charge_shadow_preparation_work(&mut work, 4, &[4, 2]);
            assert!(
                matches!(rejected, Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [4, 2])
            );
            assert_eq!(work.used(), 107);
            assert_eq!(work.remaining(MAX_AGGREGATE_SHADOW_PREPARATION_WORK), 3);
            charge_shadow_preparation_work(&mut work, 3, &[4, 2]).unwrap();
        }
        assert_eq!(local, 110);
        assert_eq!(graph, MAX_COMPOSITIONAL_SHADOW_WORK);

        let mut local = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 2;
        let mut graph = 20;
        {
            let mut work = SharedPreparationWork::new(&mut local, &mut graph);
            charge_shadow_preparation_work(&mut work, 2, &[4, 3]).unwrap();
            assert!(charge_shadow_preparation_work(&mut work, 1, &[4, 3]).is_err());
        }
        assert_eq!(local, MAX_AGGREGATE_SHADOW_PREPARATION_WORK);
        assert_eq!(graph, 22);

        let mut local = MAX_CERTIFIED_RECTANGLE_AXIS_WORK - 1;
        let mut graph = 0;
        {
            let mut work = SharedPreparationWork::new(&mut local, &mut graph);
            assert!(
                work.charge(2, MAX_CERTIFIED_RECTANGLE_AXIS_WORK, &[4, 4], "class cap")
                    .is_err()
            );
            work.charge(1, MAX_CERTIFIED_RECTANGLE_AXIS_WORK, &[4, 4], "class cap")
                .unwrap();
        }
        assert_eq!(local, MAX_CERTIFIED_RECTANGLE_AXIS_WORK);
        assert_eq!(
            graph, 1,
            "the evaluator limit remains independent of graph fuel"
        );
    }

    #[test]
    fn shared_erosion_failure_keeps_paid_geometry_work() {
        let mut triangles = Vec::new();
        for x in [10.0, 22.0] {
            triangles.extend([
                [x, 10.0],
                [x + 4.0, 10.0],
                [x, 14.0],
                [x + 4.0, 10.0],
                [x + 4.0, 14.0],
                [x, 14.0],
            ]);
        }
        let mesh = NeutralMesh {
            // The rectangle proof succeeds, then normalization overflows.
            units_per_parent_logical_unit: f64::MIN_POSITIVE,
            triangles,
            bounds: [10.0, 10.0, 26.0, 14.0],
        };
        let mut local = 0;
        let mut graph = 700;
        {
            let mut work = SharedPreparationWork::new(&mut local, &mut graph);
            assert!(matches!(
                certified_separated_rectangles_erosion_mesh(&mesh, 0.5, &[9, 2], &mut work),
                Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                    kind: crate::execution_2d::Render2dSampleSpaceError::PrecisionLimit,
                    path: Some(path), ..
                }) if path == [9, 2]
            ));
        }
        assert!(
            local > 0,
            "executed failed normalization cannot refund work"
        );
        assert_eq!(graph, 700 + local);
        let paid = local;
        graph = MAX_COMPOSITIONAL_SHADOW_WORK - 1;
        {
            let mut work = SharedPreparationWork::new(&mut local, &mut graph);
            assert!(matches!(
                certified_separated_rectangles_erosion_mesh(&mesh, 0.5, &[9, 2], &mut work),
                Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                    kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                    path: Some(path), ..
                }) if path == [9, 2]
            ));
        }
        assert_eq!(local, paid, "rejected admission executes no geometry phase");
        assert_eq!(graph, MAX_COMPOSITIONAL_SHADOW_WORK - 1);
    }

    #[test]
    fn disjoint_original_rectangles_have_exact_nonabsorbed_erosion() {
        // Original-F1 rectangle provenance is established by the caller.
        // The physical representation is exactly two separated rectangles.
        let one = |x: f64| -> Vec<[f64; 2]> {
            vec![
                [x, 10.0],
                [x + 4.0, 10.0],
                [x, 14.0],
                [x + 4.0, 10.0],
                [x + 4.0, 14.0],
                [x, 14.0],
            ]
        };
        let mut triangles = one(10.0);
        triangles.extend(one(22.0));
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles,
            bounds: [10.0, 10.0, 26.0, 14.0],
        };
        let mut work = 0;
        let exact = certified_separated_rectangles_erosion_mesh(&mesh, 0.5, &[0, 0], &mut work)
            .unwrap()
            .expect("separated original rectangles admit exact erosion");
        assert!(work > 0);
        assert_eq!(exact.triangles.len(), 12);
        assert_eq!(neutral_union_distance(&exact.triangles, [11.0, 11.0]), 0.0);
        assert_eq!(neutral_union_distance(&exact.triangles, [23.0, 11.0]), 0.0);
        assert!(neutral_union_distance(&exact.triangles, [18.0, 11.0]) > 0.0);
        assert!(neutral_union_distance(&exact.triangles, [10.25, 11.0]) > 0.0);

        // A source gap lost at the geometry precision boundary cannot
        // assert disconnectedness merely because it was triangulated.
        let mut touching = one(10.0);
        touching.extend(one(14.0));
        let touching = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: touching,
            bounds: [10.0, 10.0, 18.0, 14.0],
        };
        let mut budget = 0;
        assert!(
            certified_separated_rectangles_erosion_mesh(&touching, 0.5, &[0, 1], &mut budget)
                .unwrap()
                .is_none()
        );

        // The 16M scene-work limit is retained with atomic rejection.
        let mut exhausted = PreparationBudget::for_contribution(true);
        exhausted.record(MAX_CERTIFIED_RECTANGLE_AXIS_WORK);
        assert!(matches!(
            certified_separated_rectangles_erosion_mesh(&mesh, 0.5, &[5, 4], &mut exhausted),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [5, 4]
        ));
        assert_eq!(exhausted.used(), MAX_CERTIFIED_RECTANGLE_AXIS_WORK);
    }

    #[test]
    fn unqualified_large_convex_mesh_skips_disconnected_quadratic_charge() {
        // A large generic triangle inventory with six-vertex alignment
        // MUST remain eligible for the pre-existing convex fallback, even
        // when treating the chunks as rectangles would cost >16M work.
        // Repeating one convex triangle does not create F1 rectangles.
        let count = 4_097_usize;
        let mut triangles = Vec::new();
        triangles.try_reserve_exact(count * 6).unwrap();
        for _ in 0..count {
            triangles.extend([
                [0.0, 0.0],
                [4.0, 0.0],
                [0.0, 4.0],
                [0.0, 0.0],
                [4.0, 0.0],
                [0.0, 4.0],
            ]);
        }
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles,
            bounds: [0.0, 0.0, 4.0, 4.0],
        };
        let mut work = 0;
        assert!(
            certified_separated_rectangles_erosion_mesh(&mesh, 0.5, &[7, 2], &mut work)
                .unwrap()
                .is_none()
        );
        assert_eq!(work, 0, "a noncandidate must not pay pairwise admission");
    }

    #[test]
    fn nested_erosion_cannot_certify_concave_union_as_convex_hull() {
        // Union of two rectangles forms a concave L. Neither its AABB nor
        // convex hull is the neutral caster for signed disk erosion.
        let source = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [10.0, 10.0],
                [14.0, 10.0],
                [10.0, 12.0],
                [14.0, 10.0],
                [14.0, 12.0],
                [10.0, 12.0],
                [10.0, 12.0],
                [12.0, 12.0],
                [10.0, 14.0],
                [12.0, 12.0],
                [12.0, 14.0],
                [10.0, 14.0],
            ],
            bounds: [10.0, 10.0, 14.0, 14.0],
        };
        assert!(
            certified_convex_erosion_mesh(&source, 0.25, &[0, 0], &mut 0)
                .unwrap()
                .is_none(),
            "nonconvex union must remain unsupported, never treated as empty"
        );
    }

    #[test]
    fn rejected_aggregate_shadow_work_is_atomic_and_path_specific() {
        use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};

        let limit = MAX_AGGREGATE_SHADOW_PREPARATION_WORK;
        let mut used = limit - 2;
        charge_shadow_preparation_work(&mut used, 2, &[1, 0]).unwrap();
        assert_eq!(used, limit, "an exact-limit charge must be legal");

        for cost in [1, usize::MAX] {
            let error = charge_shadow_preparation_work(&mut used, cost, &[4, 3])
                .expect_err("an excess or integer overflow must fail closed");
            assert!(matches!(
                error,
                Render2dExecutionError::SampleSpace {
                    kind: Render2dSampleSpaceError::ResourceLimit,
                    path: Some(path),
                    ..
                } if path == [4, 3]
            ));
            assert_eq!(used, limit, "failed work admission cannot consume budget");
        }

        let mut sibling_budget = limit - 2;
        assert!(matches!(
            charge_shadow_preparation_work(&mut sibling_budget, 3, &[5, 2]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [5, 2]
        ));
        assert_eq!(sibling_budget, limit - 2);
        charge_shadow_preparation_work(&mut sibling_budget, 2, &[5, 3]).unwrap();
        assert_eq!(sibling_budget, limit);
    }

    #[test]
    fn sibling_shadow_masks_share_preparation_work_without_changing_coverage() {
        let mesh = NeutralMesh {
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
        let mut shared = 0_usize;
        let first = prepare_shadow_coverage_admitted(
            &mesh,
            0.0,
            0.0,
            4.0,
            &[0],
            &mut shared,
            MAX_NEUTRAL_MASK_SAMPLES,
        )
        .unwrap()
        .unwrap();
        assert!(
            shared > 0,
            "geometry and area integration must both charge work"
        );
        let after_first = shared;
        let second = prepare_shadow_coverage_admitted(
            &mesh,
            0.0,
            0.0,
            4.0,
            &[1],
            &mut shared,
            MAX_NEUTRAL_MASK_SAMPLES,
        )
        .unwrap()
        .unwrap();
        assert!(
            shared > after_first,
            "sibling preparation cannot reset the budget"
        );
        assert_eq!(first.origin_x, second.origin_x);
        assert_eq!(first.origin_y, second.origin_y);
        assert_eq!(first.values, second.values);

        // A late sibling must reject BEFORE its own expensive grid materialization,
        // preserving the actual offending F1 owner path. No per-effect reset.
        let mut nearly_exhausted = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 1;
        assert!(matches!(
            prepare_shadow_coverage_admitted(
                &mesh,
                0.0,
                0.0,
                4.0,
                &[5, 2],
                &mut nearly_exhausted,
                MAX_NEUTRAL_MASK_SAMPLES,
            ),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [5, 2]
        ));
    }

    #[test]
    fn retained_shadow_coverage_preallocation_rejects_with_exact_path() {
        use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};
        let mesh = NeutralMesh {
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
        for (spread, sigma) in [(0.0, 0.0), (0.0, 0.25), (0.2, 0.25)] {
            let mut used = 0;
            let error =
                prepare_shadow_coverage_admitted(&mesh, spread, sigma, 4.0, &[7, 2], &mut used, 0)
                    .expect_err("zero remaining coverage must reject before materialization");
            assert!(matches!(error, Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            } if path == [7, 2]));
            assert!(
                used > 0,
                "temporary geometry proof work is charged before coverage"
            );
        }
        let mut used = 0;
        let error = affine_parent_effect_coverage_admitted(
            &mesh,
            ParentEffectKernel {
                signed_spread: 0.0,
                sigma: 0.25,
            },
            ParentEffectFrame {
                to_root: crate::composition_2d::Render2dAffineTransform::IDENTITY,
                root_envelope: [-1.0, -1.0, 3.0, 3.0],
                sample_scale: 4.0,
            },
            &[8, 5],
            &mut used,
            0,
        )
        .expect_err("nested root-sample grid must reject before allocation");
        assert!(matches!(error, Render2dExecutionError::SampleSpace {
            kind: Render2dSampleSpaceError::ResourceLimit,
            path: Some(path), ..
        } if path == [8, 5]));
        assert_eq!(used, 0);

        let mut used = 0;
        let coverage =
            prepare_shadow_coverage_admitted(&mesh, 0.0, 0.0, 4.0, &[7, 3], &mut used, 64)
                .unwrap()
                .expect("8x8 coverage uses exactly 64 admitted f64 samples");
        assert_eq!(coverage.values.len(), 64);

        // The 2x2 right triangle has true inradius 2-sqrt(2) < 0.8,
        // while its bounding box's half-minimum extent is 1. AABB reach
        // alone cannot prove complete erosion: the exposed supporting
        // planes must certify this valid no-work outcome first.
        let thin = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let mut used = 0;
        let empty = prepare_shadow_coverage_admitted(&thin, -0.8, 0.25, 4.0, &[7, 4], &mut used, 0)
            .unwrap();
        assert!(
            empty.is_none(),
            "certified complete erosion needs no coverage buffer"
        );
    }

    #[test]
    fn cumulative_continuous_halo_work_is_admitted_before_allocation() {
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        // The neutral lattice's source traversal fits in the residual
        // budget, but expansion by the finite 3σ blur halo does not.
        // Reject at the originating occurrence before coverage allocation.
        let mut used = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 100;
        assert!(matches!(
            prepare_shadow_coverage_admitted(
                &mesh, 0.0, 0.25, 4.0, &[4, 3], &mut used, MAX_NEUTRAL_MASK_SAMPLES,
            ),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [4, 3]
        ));
    }

    #[test]
    fn repeated_erosion_boundary_and_convex_work_preserve_path_admission() {
        let triangle = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
        let mut shared = 0_usize;
        let exterior = union_exterior_boundary(&triangle, &[0], Some(&mut shared))
            .expect("small triangle exposes three edges");
        assert!(!exterior.is_empty());
        assert!(shared > 0, "boundary arrangement work must be charged");

        let mut nearly_exhausted = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 1;
        assert!(matches!(
            union_exterior_boundary(&triangle, &[2, 4], Some(&mut nearly_exhausted)),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [2, 4]
        ));

        let mut nearly_exhausted = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 1;
        assert!(matches!(
            certified_convex_eroded_union(
                &triangle,
                &exterior,
                0.25,
                &[3, 1],
                Some(&mut nearly_exhausted),
            ),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [3, 1]
        ));
    }

    #[test]
    fn erosion_geometry_pays_primitive_arithmetic_before_admitted_visits() {
        let triangle = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
        let mut boundary_work = 0;
        let exterior =
            union_exterior_boundary(&triangle, &[5, 1], Some(&mut boundary_work)).unwrap();
        assert_eq!(exterior.len(), 3);
        // A single six-arithmetic orientation, three edge-vector/squared-
        // length calculations and twelve endpoint FMAs already require 33
        // arithmetic primitives, before loads, branches, sorting or copies.
        assert!(
            boundary_work >= 33,
            "the former 12 visit units do not pay arithmetic"
        );
        let mut convex_work = 0;
        let inset = certified_convex_eroded_union(
            &triangle,
            &exterior,
            0.25,
            &[5, 2],
            Some(&mut convex_work),
        )
        .unwrap()
        .unwrap();
        assert!(!inset.is_empty());
        // Three supporting lines inspect all three vertices: nine orient
        // calls alone execute at least 54 arithmetic primitives.
        assert!(
            convex_work >= 54,
            "the former 21 visit units do not pay orientations"
        );

        for (convex, remaining) in [(false, 12), (true, 21)] {
            let mut work = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - remaining;
            let initial = work;
            let result = if convex {
                certified_convex_eroded_union(&triangle, &exterior, 0.25, &[5, 7], Some(&mut work))
                    .map(|_| ())
            } else {
                union_exterior_boundary(&triangle, &[5, 7], Some(&mut work)).map(|_| ())
            };
            assert!(matches!(result,
                Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                    kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                    path: Some(path), ..
                }) if path == [5, 7]));
            assert_eq!(
                work, initial,
                "unadmitted first geometry phase cannot execute"
            );
        }
    }

    #[test]
    fn erosion_buffer_growth_prepays_relocation_and_push_checks_logical_inventory() {
        let mut buffer = vec![1.0_f64, 2.0, 3.0, 4.0];
        let pointer = buffer.as_ptr();
        let capacity = buffer.capacity();
        let mut local = 0;
        // Relocating four allocated scalar words requires eight transfers,
        // in addition to the fixed 64-unit allocation/guard allowance.
        let mut graph = MAX_COMPOSITIONAL_SHADOW_WORK - 71;
        {
            let mut work = SharedPreparationWork::new(&mut local, &mut graph);
            assert!(
                matches!(reserve_erosion_buffer(&mut buffer, 5, 8, &mut work, &[6, 3]),
                Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                    kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                    path: Some(path), ..
                }) if path == [6, 3])
            );
        }
        assert_eq!(buffer.as_ptr(), pointer);
        assert_eq!(buffer.capacity(), capacity);
        assert_eq!(buffer, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(local, 0);
        assert_eq!(graph, MAX_COMPOSITIONAL_SHADOW_WORK - 71);

        reserve_erosion_buffer(&mut buffer, 8, 8, &mut 0, &[6, 3]).unwrap();
        assert!(buffer.capacity() <= 8);
        let capacity = buffer.capacity();
        let mut paid = 0;
        assert!(push_erosion_buffer(&mut buffer, 5.0, 4, &mut paid, &[6, 4]).is_err());
        assert_eq!(
            buffer.len(),
            4,
            "spare physical capacity cannot widen a logical limit"
        );
        assert_eq!(buffer.capacity(), capacity);
        assert!(paid > 0, "executed failed push guards remain paid");
        push_erosion_buffer(&mut buffer, 5.0, 5, &mut paid, &[6, 5]).unwrap();
        assert_eq!(buffer, [1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(buffer.capacity(), capacity);
    }

    #[test]
    fn erosion_event_sort_matches_total_order_without_allocating_or_unfunded_mutation() {
        let original = vec![4.0, -0.0, 3.0, -8.0, 0.0, 4.0, f64::MAX, -f64::MAX, 2.0];
        let mut expected = original.clone();
        expected.sort_by(f64::total_cmp);
        expected.dedup();
        let mut actual = original.clone();
        let pointer = actual.as_ptr();
        let capacity = actual.capacity();
        let mut work = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 1;
        assert!(sort_erosion_intervals(&mut actual, &mut work, &[6, 6]).is_err());
        assert_eq!(actual, original);
        assert_eq!(work, MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 1);
        let mut work = 0;
        sort_erosion_intervals(&mut actual, &mut work, &[6, 6]).unwrap();
        assert_eq!(
            actual.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            expected.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(actual.as_ptr(), pointer);
        assert_eq!(actual.capacity(), capacity);
        assert!(work > 0);
    }

    #[test]
    fn nonconvex_erosion_rejection_keeps_paid_work_for_following_geometry() {
        let source = [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 1.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 4.0],
            [1.0, 1.0],
            [1.0, 4.0],
            [0.0, 4.0],
        ];
        let exterior = union_exterior_boundary(&source, &[7, 1], None).unwrap();
        let triangle = [[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]];
        let triangle_boundary = union_exterior_boundary(&triangle, &[7, 2], None).unwrap();
        let mut local = 0;
        let mut graph = 700;
        {
            let mut work = SharedPreparationWork::new(&mut local, &mut graph);
            assert!(
                certified_convex_eroded_union(&source, &exterior, 0.25, &[7, 1], Some(&mut work),)
                    .unwrap()
                    .is_none()
            );
            let failed = work.used();
            assert!(failed > 0);
            assert!(
                certified_convex_eroded_union(
                    &triangle,
                    &triangle_boundary,
                    0.25,
                    &[7, 2],
                    Some(&mut work),
                )
                .unwrap()
                .is_some()
            );
            assert!(work.used() > failed);
        }
        assert_eq!(
            graph,
            700 + local,
            "failed qualification cannot refund graph fuel"
        );
    }

    #[test]
    fn continuous_positive_spread_gaussian_matches_independent_rectangle_cdf() {
        // A 0.01-wide neutral caster is dilated by 0.015 parent units;
        // the output sample lies outside that dilation, but its finite
        // Gaussian footprint includes the full dilated vertical stripe.
        // The old area-cell/discrete-kernel implementation returned 0.04.
        let rectangle = [
            [0.7, 0.0],
            [0.74, 0.0],
            [0.74, 4.0],
            [0.7, 0.0],
            [0.74, 4.0],
            [0.7, 4.0],
        ];
        let mut work = 0_usize;
        let actual = continuous_signed_spread_sample(
            &rectangle,
            None,
            [0.5, 0.5],
            0.06,
            0.1,
            &mut work,
            &[3, 2],
        )
        .expect("bounded continuous Euclidean Gaussian integration");
        let independent_cdf = 0.079_621_723_618_115_15_f64;
        assert!((actual - independent_cdf).abs() <= 1.0 / 2048.0);
        assert!((actual - 0.04).abs() > 0.03);
        assert!(work > 0);
    }

    #[test]
    fn certified_sheared_convex_erosion_uses_exposed_halfplanes_not_enclosing_box() {
        // The two triangles cover a full sheared parallelogram, and have a
        // shared internal diagonal. The exposed exterior owns the Euclidean
        // erosion support, not that diagonal or the shape's AABB.
        let source = [
            [0.0, 0.0],
            [4.0, 0.0],
            [6.0, 4.0],
            [0.0, 0.0],
            [6.0, 4.0],
            [2.0, 4.0],
        ];
        let exterior = union_exterior_boundary(&source, &[6], None).unwrap();
        let eroded = certified_convex_eroded_union(&source, &exterior, 0.5, &[6], None)
            .unwrap()
            .expect("complete sheared convex union is certified");
        assert!(!eroded.is_empty());
        assert_eq!(neutral_union_distance(&eroded, [3.0, 2.0]), 0.0);
        assert!(neutral_union_distance(&eroded, [1.0, 2.0]) > 0.0);
        let mut total = 0.0;
        let mut work = 0;
        for y in 0..4 {
            for x in 0..6 {
                total += area_sample(&eroded, f64::from(x), f64::from(y), &mut work, &[6])
                    .expect("exposed-plane inset has bounded exact cell area");
            }
        }
        // Convex Euclidean erosion narrows each slanted edge by r*sqrt(1.25)
        // in the x-.5*y coordinates and each horizontal edge by r.
        let independent_area = (4.0 - 1.25_f64.sqrt()) * 3.0;
        assert!(
            (total - independent_area).abs() < 1.0e-8,
            "{total} vs {independent_area}"
        );

        // A concave L has an exposed notch boundary inside its convex hull.
        // Its inner boundary cannot be promoted to a global supporting line.
        let l_shape = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 1.0],
            [0.0, 0.0],
            [4.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 4.0],
            [0.0, 1.0],
            [1.0, 4.0],
            [0.0, 4.0],
        ];
        let concave_boundary = union_exterior_boundary(&l_shape, &[6], None).unwrap();
        assert!(
            certified_convex_eroded_union(&l_shape, &concave_boundary, 0.5, &[6], None)
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn certified_rectangle_erosion_does_not_accept_unfilled_bounds_or_overlapping_halves() {
        let a = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
        ];
        assert_eq!(
            certified_axis_aligned_rectangle(&a),
            Some([0.0, 0.0, 4.0, 4.0])
        );
        let duplicate = [a, a].concat();
        assert_eq!(
            certified_axis_aligned_rectangle(&duplicate),
            Some([0.0, 0.0, 4.0, 4.0])
        );
        let partial = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
        ];
        assert_eq!(certified_axis_aligned_rectangle(&partial), None);
        let triangle_only = [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0]];
        assert_eq!(certified_axis_aligned_rectangle(&triangle_only), None);
        let displaced = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 0.0],
            [4.0, 4.0],
            [2.0, 4.0],
        ];
        assert_eq!(certified_axis_aligned_rectangle(&displaced), None);
    }

    #[test]
    fn actual_transparent_vector_rectangle_proves_complete_neutral_union_certificate() {
        use crate::composition_2d::{
            Render2dAffineTransform, Render2dBrush, Render2dColorRgba8, Render2dItem,
            Render2dOpacity, Render2dPrimitive, Render2dRect, Render2dShape,
        };
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(10.0, 10.0, 16.0, 16.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let mesh = super::super::vector::neutral_support(&item, 0, 4.0, 16_384)
            .unwrap()
            .expect("transparent rectangle retains immutable source-neutral triangles");
        assert_eq!(
            certified_axis_aligned_rectangle(&mesh.triangles),
            Some([40.0, 40.0, 104.0, 104.0]),
            "real F3F vector tessellation must meet the strict full-rectangle certificate"
        );
        let coverage = prepare_untranslated_shadow_coverage(&mesh, -2.0, 0.0, 4.0, &[0])
            .unwrap()
            .expect("ordinary 16x16 negative rectangle must admit bounded exact erosion");
        let index = |x: i64, y: i64| {
            usize::try_from(y - coverage.origin_y).unwrap() * coverage.width
                + usize::try_from(x - coverage.origin_x).unwrap()
        };
        assert_eq!(coverage.values[index(72, 72)], 1.0);
        assert_eq!(coverage.values[index(45, 72)], 0.0);
    }

    #[test]
    fn certified_inset_rectangle_has_independent_identity_and_gaussian_coverage() {
        let inset = [0.8, 0.8, 3.2, 3.2];
        assert!((inset_rectangle_sample(inset, [0.5, 0.5], 0.0) - 0.04).abs() < 1e-12);
        assert_eq!(inset_rectangle_sample(inset, [1.5, 1.5], 0.0), 1.0);
        assert_eq!(inset_rectangle_sample(inset, [0.5, 0.5], 0.05), 0.0);
        assert!((inset_rectangle_sample(inset, [1.5, 1.5], 0.05) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn negative_continuous_erosion_of_one_square_preserves_exact_subcell_area() {
        // A [0,1]^2 continuous square eroded by a disk of radius .2 is
        // exactly [.2,.8]^2. At 4x sample density, four x/y sample cells
        // have fractions [.2,1,1,.2]. The caller pixel has area .36,
        // NOT the sampled-center distance-transform's opaque 1.0.
        let rect = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
        ];
        let boundary = union_exterior_boundary(&rect, &[4, 2], None).unwrap();
        let mut work = 0_usize;
        let a = continuous_signed_spread_sample(
            &rect,
            Some(&boundary),
            [0.5, 0.5],
            -0.8,
            0.0,
            &mut work,
            &[4, 2],
        )
        .unwrap();
        // Both axes meet the lower eroded boundary in this corner cell:
        // intersected widths are .2 × .2, hence exact area .04.
        assert!((a - 0.04).abs() < 1.0 / 2048.0);
        let interior = continuous_signed_spread_sample(
            &rect,
            Some(&boundary),
            [1.5, 1.5],
            -0.8,
            0.0,
            &mut work,
            &[4, 2],
        )
        .unwrap();
        assert!((interior - 1.0).abs() < 1.0 / 2048.0);
    }

    #[test]
    fn erosion_union_boundary_removes_internal_diagonals_and_shared_seams() {
        let left = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ];
        let right = [
            [1.0, 0.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [1.0, 0.0],
            [2.0, 1.0],
            [1.0, 1.0],
        ];
        let triangles = [left, right].concat();
        let boundary = union_exterior_boundary(&triangles, &[8, 3], None).unwrap();
        // The center seam at x=1 is internal; the nearest true union
        // boundary at (1,.5) is .5 away, not zero.
        assert!(
            (union_signed_interior_distance(&triangles, &boundary, [1.0, 0.5]) - 0.5).abs() < 1e-9
        );
        let duplicate = [triangles.clone(), triangles.clone()].concat();
        let repeated = union_exterior_boundary(&duplicate, &[8, 3], None).unwrap();
        assert!(
            (union_signed_interior_distance(&duplicate, &repeated, [1.0, 0.5]) - 0.5).abs() < 1e-9
        );
    }

    #[test]
    fn generic_erosion_far_from_exterior_certificate_never_uses_aabb_as_source() {
        let rectangle = [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
        ];
        let boundary = union_exterior_boundary(&rectangle, &[7], None).unwrap();
        assert!(eroded_cell_far_from_exterior(
            &boundary,
            [1.0, 1.0, 2.0, 2.0],
            0.5
        ));
        assert!(!eroded_cell_far_from_exterior(
            &boundary,
            [0.0, 1.0, 1.0, 2.0],
            0.5
        ));
        // The certificate only proves distance from exposed edges. A cell
        // outside the shape can be far too, so its midpoint still decides
        // its membership; this must never infer filled bounding-box area.
        assert!(eroded_cell_far_from_exterior(
            &boundary,
            [6.0, 6.0, 7.0, 7.0],
            0.5
        ));
        assert!(neutral_union_distance(&rectangle, [6.5, 6.5]) > 0.0);
    }

    #[test]
    fn continuous_erosion_preserves_hole_boundaries_and_signed_exterior() {
        // Four neutral rectangular regions leave the open central hole
        // (1,2)x(1,2). A distance query from its center must have NEGATIVE
        // sign, even though the union bounding rectangle is [0,3]^2.
        let rect = |x: f64, y: f64, w: f64, h: f64| {
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y],
                [x + w, y + h],
                [x, y + h],
            ]
        };
        let triangles = [
            rect(0.0, 0.0, 3.0, 1.0),
            rect(0.0, 2.0, 3.0, 1.0),
            rect(0.0, 1.0, 1.0, 1.0),
            rect(2.0, 1.0, 1.0, 1.0),
        ]
        .concat();
        let boundary = union_exterior_boundary(&triangles, &[9, 1], None).unwrap();
        let hole = union_signed_interior_distance(&triangles, &boundary, [1.5, 1.5]);
        let interior = union_signed_interior_distance(&triangles, &boundary, [0.5, 1.5]);
        assert!((hole + 0.5).abs() < 1e-9, "{hole}");
        assert!((interior - 0.5).abs() < 1e-9, "{interior}");
        let mut work = 0_usize;
        let alpha = continuous_signed_spread_sample(
            &triangles,
            Some(&boundary),
            [1.5, 1.5],
            -0.1,
            0.0,
            &mut work,
            &[9, 1],
        )
        .unwrap();
        // Remaining uncertain mass is bounded, not silently interpreted
        // as exactly zero. A near-zero representative is acceptable here.
        assert!(alpha <= 1.0 / 2048.0, "{alpha}");
    }

    #[test]
    fn continuous_negative_spread_with_gaussian_preserves_fully_covered_interior() {
        // The sigma=.025 truncated domain around (.5,.5) stays entirely
        // inside the .2-eroded unit square. Its alpha must normalize to 1,
        // independent of how the square is triangulated or duplicated.
        let rect = [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ];
        let triangles = [rect, rect].concat();
        let boundary = union_exterior_boundary(&triangles, &[7, 1], None).unwrap();
        let mut work = 0_usize;
        let value = continuous_signed_spread_sample(
            &triangles,
            Some(&boundary),
            [0.5, 0.5],
            -0.2,
            0.025,
            &mut work,
            &[7, 1],
        )
        .unwrap();
        assert!((value - 1.0).abs() <= 1.0 / 2048.0, "{value}");
    }

    #[test]
    fn positive_disk_identity_blur_matches_independent_circular_corner_area() {
        // A 16x16 rectangle expanded by a radius-two Euclidean disk has
        // a true circular corner at (26,26). Cell [27,28]^2 receives the
        // exact area integral int_{1}^{sqrt(3)} (sqrt(4-x*x)-1) dx.
        // Closed-form area: 0.31514674362772044 (unit cell).
        let rectangle = [
            [10.0, 10.0],
            [26.0, 10.0],
            [26.0, 26.0],
            [10.0, 10.0],
            [26.0, 26.0],
            [10.0, 26.0],
        ];
        let mut work = 0_usize;
        let actual = continuous_signed_spread_sample(
            &rectangle,
            None,
            [27.5, 27.5],
            2.0,
            0.0,
            &mut work,
            &[6, 2],
        )
        .expect("bounded continuous Euclidean identity-blur coverage");
        assert!((actual - 0.315_146_743_627_720_44).abs() <= 1.0 / 2048.0);
        assert!((actual - 0.375).abs() > 0.04);
    }

    #[test]
    fn continuous_spread_gaussian_rejects_invalid_params_without_alpha_fabrication() {
        let tri = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mut work = 0_usize;
        assert!(matches!(
            continuous_signed_spread_sample(&tri, None, [0.5, 0.5], -0.01, 0.2, &mut work, &[9, 1]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [9, 1]
        ));
    }

    #[test]
    fn continuous_gaussian_reference_retains_full_interior_when_sample_cell_is_partial() {
        let mut rect = Vec::new();
        let points = [[0.03, 0.03], [0.23, 0.03], [0.23, 0.23], [0.03, 0.23]];
        rect.extend([
            points[0], points[1], points[2], points[0], points[2], points[3],
        ]);
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: rect,
            bounds: [0.03, 0.03, 0.23, 0.23],
        };
        let result = prepare_untranslated_shadow_coverage(&mesh, 0.0, 0.025, 4.0, &[3, 9])
            .unwrap()
            .unwrap();
        let col = usize::try_from(-result.origin_x).unwrap();
        let row = usize::try_from(-result.origin_y).unwrap();
        // At logical (.125,.125) the entire closed 3sigma window
        // [.05,.20]^2 is inside the union. Exact normalized reference is 1.
        let sample = result.values[row * result.width + col];
        assert!((sample - 1.0).abs() < 1.0e-8, "F3F-GAUSS-01 {sample}");
        // An area-cell-first discrete kernel would return (.2/.25)^2 = .64.
        assert!((sample - 0.64).abs() > 0.3);
    }

    #[test]
    fn continuous_gaussian_exact_half_plane_symmetry_and_overlap_union() {
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [-1.0, -1.0],
                [0.125, -1.0],
                [0.125, 1.0],
                [-1.0, -1.0],
                [0.125, 1.0],
                [-1.0, 1.0],
            ],
            bounds: [-1.0, -1.0, 0.125, 1.0],
        };
        let once = prepare_untranslated_shadow_coverage(&mesh, 0.0, 0.025, 4.0, &[4])
            .unwrap()
            .unwrap();
        let col = usize::try_from(-once.origin_x).unwrap();
        let row = usize::try_from(-once.origin_y).unwrap();
        let sample = once.values[row * once.width + col];
        assert!(
            (sample - 0.5).abs() < 1.0e-8,
            "half-plane Gaussian {sample}"
        );
        let mut duplicate = mesh;
        duplicate.triangles.extend_from_within(..);
        let twice = prepare_untranslated_shadow_coverage(&duplicate, 0.0, 0.025, 4.0, &[4])
            .unwrap()
            .unwrap();
        assert!((twice.values[row * twice.width + col] - sample).abs() < 1.0e-8);
    }

    #[test]
    fn continuous_gaussian_partial_rectangle_matches_independent_erf_reference() {
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [-0.25, -0.5],
                [0.175, -0.5],
                [0.175, 0.75],
                [-0.25, -0.5],
                [0.175, 0.75],
                [-0.25, 0.75],
            ],
            bounds: [-0.25, -0.5, 0.175, 0.75],
        };
        let coverage = prepare_untranslated_shadow_coverage(&mesh, 0.0, 0.2, 4.0, &[7, 1])
            .unwrap()
            .unwrap();
        let x = usize::try_from(-coverage.origin_x).unwrap();
        let y = usize::try_from(-coverage.origin_y).unwrap();
        // Independently computed exact erf ratio for one clipped axis:
        // ([left,right]=[-1,.7], sample center=.5, physical sigma=.8)
        // [erf(.25/sqrt(2)) - erf(-1.875/sqrt(2))] / [2 erf(3/sqrt(2))].
        let analytic = 0.569_848_438_488_548_7;
        let actual = coverage.values[y * coverage.width + x];
        assert!((actual - analytic).abs() < 1.0e-8, "{actual}");
    }

    #[test]
    fn oblique_triangle_integral_matches_independent_adaptive_reference() {
        let triangle = [
            [-3.308_716_152_350_335_4, 2.973_430_464_862_749_6],
            [-0.651_481_506_714_978_8, -3.001_405_350_983_398],
            [1.624_245_650_140_160_5, -1.490_622_465_767_189_4],
        ];
        let mut work = 0_usize;
        let value = gaussian_union_sample(&triangle, [0.0, 0.0], 1.0, &mut work, &[5]).unwrap();
        // Independent adaptive y-quadrature + analytic Gaussian x-erf
        // evaluation, partitioned at triangle vertices (not 16x16 GL).
        let reference = 0.449_875_362_999_087_37;
        assert!((value - reference).abs() < 1.0e-8, "{value}");
    }

    #[test]
    fn area_integral_unions_duplicate_and_overlapping_triangles() {
        let mut work = 0;
        let a = [[0.1, 0.1], [0.9, 0.1], [0.1, 0.9]];
        let mut geometry = Vec::from(a);
        let single = area_sample(&geometry, 0.0, 0.0, &mut work, &[2]).unwrap();
        assert!((single - 0.32).abs() < 1.0e-12);
        geometry.extend(a);
        let double = area_sample(&geometry, 0.0, 0.0, &mut work, &[2]).unwrap();
        assert!((double - single).abs() < 1.0e-12);
        let rect = [
            [0.01, 0.01],
            [0.02, 0.01],
            [0.02, 0.02],
            [0.01, 0.01],
            [0.02, 0.02],
            [0.01, 0.02],
        ];
        let sliver = area_sample(&rect, 0.0, 0.0, &mut work, &[2]).unwrap();
        assert!((sliver - 0.0001).abs() < 1.0e-12);
    }

    #[test]
    fn expanded_gaussian_halo_scans_are_admitted_before_coverage_allocation() {
        let triangle = [[0.0, 0.0], [0.25, 0.0], [0.0, 0.25]];
        let mut mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: Vec::new(),
            bounds: [0.0, 0.0, 0.25, 0.25],
        };
        for _ in 0..256 {
            mesh.triangles.extend(triangle);
        }
        assert!(matches!(
            prepare_untranslated_shadow_coverage(&mesh, 0.0, 20.0, 4.0, &[8, 2]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [8, 2]
        ));
    }

    #[test]
    fn area_reconstruction_rejects_unbounded_overlap_before_target_work() {
        let mut mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: Vec::new(),
            bounds: [0.0, 0.0, 1.0, 1.0],
        };
        for _ in 0..257 {
            mesh.triangles.extend([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
        }
        assert!(matches!(
            rasterize_area_coverage(&mesh, 4.0, &[9, 7], &mut 0, MAX_NEUTRAL_MASK_SAMPLES),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [9, 7]
        ));
    }

    #[test]
    fn logically_identical_meshes_at_different_preparation_scales_cast_same_shadow() {
        use crate::composition_2d::{
            Render2dAffineTransform, Render2dBrush, Render2dColorRgba8, Render2dItem,
            Render2dOpacity, Render2dPrimitive, Render2dRect, Render2dShape,
        };
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 2.0, 1.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let low = super::super::vector::neutral_support(&item, 0, 1.0, 16_384)
            .unwrap()
            .unwrap();
        let high = super::super::vector::neutral_support(&item, 0, 2.0, 16_384)
            .unwrap()
            .unwrap();
        assert_eq!(low.units_per_parent_logical_unit, 1.0);
        assert_eq!(high.units_per_parent_logical_unit, 2.0);
        let a = prepare_untranslated_shadow_coverage(&low, 0.5, 0.25, 4.0, &[1, 0])
            .unwrap()
            .unwrap();
        let b = prepare_untranslated_shadow_coverage(&high, 0.5, 0.25, 4.0, &[1, 0])
            .unwrap()
            .unwrap();
        assert_eq!((a.origin_x, a.origin_y), (b.origin_x, b.origin_y));
        assert_eq!((a.width, a.height), (b.width, b.height));
        assert_eq!(a.values, b.values);
    }

    #[test]
    fn off_phase_geometry_still_casts_a_positive_euclidean_spread() {
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [0.01, 0.01],
                [0.02, 0.01],
                [0.02, 0.02],
                [0.01, 0.01],
                [0.02, 0.02],
                [0.01, 0.02],
            ],
            bounds: [0.01, 0.01, 0.02, 0.02],
        };
        let phase_only = rasterize_neutral_mesh(&mesh, 4.0, &[2, 5])
            .unwrap()
            .unwrap();
        assert!(phase_only.samples.iter().all(|value| *value == 0));
        let spread = prepare_untranslated_shadow_coverage(&mesh, 0.5, 0.0, 4.0, &[2, 5])
            .unwrap()
            .expect("positive Euclidean spread of real geometry cannot disappear");
        assert!(spread.values.iter().any(|value| *value > 0.0));
        assert!(spread.origin_x < 0 && spread.origin_y < 0);
        let index = |x: i64, y: i64| {
            usize::try_from(y - spread.origin_y).unwrap() * spread.width
                + usize::try_from(x - spread.origin_x).unwrap()
        };
        assert!(spread.values[index(0, 0)] > 0.0);
        // The disk intersects the [-2,-1]^2 sample cell even though its
        // center (-1.5,-1.5) is outside the dilation. Uniform cell-area
        // reconstruction must retain this fractional Euclidean intersection,
        // unlike the old binary-center path.
        assert!(spread.values[index(-2, -2)] > 0.0);
        assert!(spread.values[index(-2, -2)] < 1.0);
        // The same caster lies completely between centers but has an
        // exact nonzero area of 0.04 x 0.04 physical sample units.
        let area = prepare_untranslated_shadow_coverage(&mesh, 0.0, 0.5, 4.0, &[6, 7])
            .unwrap()
            .unwrap();
        let total = area.values.iter().sum::<f64>();
        // Independently evaluated separable truncated-Gaussian integral:
        // erf((edge-center)/(sqrt(2)*sigma)) per axis, summed on the
        // correlated integer sample lattice. Convolution is not forced
        // to conserve area exactly under point sampling.
        let analytic_reference = 0.001_600_812_911_921_821_8;
        assert!((total - analytic_reference).abs() < 1.0e-8, "{total}");
        let off_phase_spread =
            prepare_untranslated_shadow_coverage(&mesh, 0.001, 0.5, 4.0, &[6, 8])
                .unwrap()
                .expect("tiny positive disk and Gaussian retain a continuous caster");
        assert!(off_phase_spread.values.iter().sum::<f64>() >= 0.0016 - 1.0e-10);
        let exact_identity = prepare_untranslated_shadow_coverage(&mesh, 0.0, 0.0, 4.0, &[6, 9])
            .unwrap()
            .expect("sigma-zero caster retains its exact integrated area");
        assert!((exact_identity.values.iter().sum::<f64>() - 0.0016).abs() < 1.0e-10);
    }

    #[test]
    fn parent_rectangle_gaussian_retains_independent_affine_pixels_under_scene_budget() {
        use crate::composition_2d::Render2dAffineTransform;
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [10.0, 10.0],
                [20.0, 10.0],
                [20.0, 20.0],
                [10.0, 10.0],
                [20.0, 20.0],
                [10.0, 20.0],
            ],
            bounds: [10.0, 10.0, 20.0, 20.0],
        };
        let mut budget = PreparationBudget::for_contribution(true);
        let coverage = affine_parent_effect_coverage_admitted(
            &mesh,
            ParentEffectKernel {
                signed_spread: 0.0,
                sigma: 0.5,
            },
            ParentEffectFrame {
                to_root: Render2dAffineTransform::new(2.0, 0.0, 0.5, 1.0, 4.0, 0.0).unwrap(),
                root_envelope: [25.25, 8.5, 57.75, 21.5],
                sample_scale: 4.0,
            },
            &[0, 0],
            &mut budget,
            MAX_NEUTRAL_MASK_SAMPLES,
        )
        .unwrap()
        .unwrap();
        // Independent normalized erf-CDF values, also asserted by the Vulkan
        // fixture. Average all globally phased points after parent inversion.
        for (pixel_x, expected) in [(30, 0.405_534), (31, 0.763_495), (32, 0.953_792)] {
            let mut sum = 0.0;
            for y in 0..4 {
                for x in 0..4 {
                    let column = usize::try_from(pixel_x * 4 + x - coverage.origin_x).unwrap();
                    let row = usize::try_from(13 * 4 + y - coverage.origin_y).unwrap();
                    sum += coverage.values[row * coverage.width + column];
                }
            }
            assert!((sum / 16.0 - expected).abs() < 1.0e-6);
        }
        assert!(budget.used() < 5_000_000);
    }

    #[test]
    fn full_window_triangle_shortcut_still_pays_source_scan_before_work() {
        use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};
        let triangle = [[-10.0, -10.0], [10.0, -10.0], [0.0, 10.0]];
        let mut work = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 100;
        assert!(matches!(
            gaussian_union_sample(&triangle, [0.0, 0.0], 1.0, &mut work, &[5, 8]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [5, 8]
        ));
        // The fixed setup was executed; no triangle scan/row work ran.
        assert_eq!(work, MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 100 + 64);
        let mut admitted = 0;
        assert_eq!(
            gaussian_union_sample(&triangle, [0.0, 0.0], 1.0, &mut admitted, &[5, 8]).unwrap(),
            1.0
        );
        assert!(admitted > 0);
    }

    #[test]
    fn continuous_area_rejects_unfunded_slabs_before_allocating_rows() {
        use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};
        let triangle = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let initial = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 524;
        let mut construction_work = initial;
        NeutralUnionSweep::new(
            &triangle,
            [0.0, 0.0, 1.0, 1.0],
            8,
            &mut construction_work,
            &[5, 9],
        )
        .unwrap();
        let mut work = initial;
        assert!(
            matches!(area_sample(&triangle, 0.0, 0.0, &mut work, &[5, 9]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [5, 9])
        );
        assert_eq!(
            work, construction_work,
            "unfunded rows must execute no slab work"
        );
        assert!((area_sample(&triangle, 0.0, 0.0, &mut 0, &[5, 9]).unwrap() - 0.5).abs() < 1.0e-14);
    }

    #[test]
    fn gaussian_union_rejects_unfunded_quadrature_before_allocating_rows() {
        use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};
        let triangle = [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]];
        // The old n³ * 256 arrangement charge fits, but cannot also pay for
        // all executed x/y quadrature arithmetic. Reject before row work.
        let mut work = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 384 - 256 - 32 - 1_000;
        let mut arrangement_work = work;
        NeutralUnionSweep::new(
            &triangle,
            [-3.0, -3.0, 3.0, 3.0],
            256,
            &mut arrangement_work,
            &[5, 7],
        )
        .unwrap();
        assert!(matches!(
            gaussian_union_sample(&triangle, [0.0, 0.0], 1.0, &mut work, &[5, 7]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [5, 7]
        ));
        assert_eq!(
            work, arrangement_work,
            "unfunded quadrature must execute no rows"
        );
    }

    #[test]
    fn certified_rectangle_axes_share_continuous_integrals_without_changing_samples() {
        // The production evaluator must return exactly the independently
        // composed per-site x*y integrals for identity and finite Gaussians.
        // No sample-alpha raster mask or Gaussian tap approximation is used.
        let mesh = NeutralMesh {
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
        for sigma in [0.0, 0.125, 0.25, 0.5] {
            let mut work = 0;
            let actual = prepare_shadow_coverage_admitted(
                &mesh,
                -0.25,
                sigma,
                4.0,
                &[9, 2],
                &mut work,
                MAX_NEUTRAL_MASK_SAMPLES,
            )
            .unwrap()
            .unwrap();
            let rectangle = [1.0, 1.0, 7.0, 7.0];
            for row in 0..actual.height {
                for column in 0..actual.width {
                    let center = [
                        actual.origin_x as f64 + as_f64(column) + 0.5,
                        actual.origin_y as f64 + as_f64(row) + 0.5,
                    ];
                    let independent = inset_rectangle_sample(rectangle, center, sigma * 4.0);
                    let shared = actual.values[row * actual.width + column];
                    assert_eq!(shared, independent, "sigma={sigma}, x={column}, y={row}");
                }
            }
        }
    }

    #[test]
    fn certified_rectangle_axis_work_is_charged_before_float_allocation() {
        use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};
        // Charge each node's executed arithmetic, not just its existence:
        // 28 axes * 288 operations plus 196 products * 16 = 11,200.
        // Rejection is atomic before x-axis or retained coverage allocation.
        let axis_work = 11_200;
        let mut charged = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - axis_work + 1;
        assert!(matches!(
            charge_certified_rectangle_axis_work(14, 14, 14 * 14, 1.0, &mut charged, &[9, 4]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [9, 4]
        ));
        assert_eq!(
            charged,
            MAX_AGGREGATE_SHADOW_PREPARATION_WORK - axis_work + 1
        );
        let mut exact = MAX_AGGREGATE_SHADOW_PREPARATION_WORK - axis_work;
        charge_certified_rectangle_axis_work(14, 14, 14 * 14, 1.0, &mut exact, &[9, 5]).unwrap();
        assert_eq!(exact, MAX_AGGREGATE_SHADOW_PREPARATION_WORK);
        assert!(matches!(
            charge_certified_rectangle_axis_work(
                usize::MAX, 14, 14 * 14, 1.0, &mut 0, &[9, 6],
            ),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [9, 6]
        ));
    }

    #[test]
    fn finite_gaussian_stage_one_complete_scenes_keep_primitive_work_admission() {
        let rectangle = |[x0, y0, x1, y1]: [f64; 4]| {
            [[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]]
        };
        let scenes = [
            (
                "concave L",
                vec![[10.0, 10.0, 20.0, 15.0], [10.0, 15.0, 15.0, 20.0]],
                [14, 14],
                [12, 12],
                [18, 18],
            ),
            (
                "holed ring",
                vec![
                    [10.0, 10.0, 30.0, 16.0],
                    [10.0, 16.0, 16.0, 24.0],
                    [24.0, 16.0, 30.0, 24.0],
                    [10.0, 24.0, 30.0, 30.0],
                ],
                [15, 15],
                [13, 20],
                [20, 20],
            ),
        ];
        for (name, rectangles, corner, interior, exterior) in scenes {
            let triangles = rectangles
                .into_iter()
                .flat_map(rectangle)
                .collect::<Vec<_>>();
            let bounds = triangles.iter().fold(
                [
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::NEG_INFINITY,
                    f64::NEG_INFINITY,
                ],
                |b, &[x, y]| [b[0].min(x), b[1].min(y), b[2].max(x), b[3].max(y)],
            );
            let mesh = NeutralMesh {
                units_per_parent_logical_unit: 1.0,
                triangles,
                bounds,
            };
            let mut work = PreparationBudget::for_contribution(true);
            let result = prepare_shadow_coverage_admitted(
                &mesh,
                -1.0,
                0.1,
                4.0,
                &[9, 1],
                &mut work,
                MAX_NEUTRAL_MASK_SAMPLES,
            );
            eprintln!(
                "{name}: charged primitive work={}, maximum={}",
                work.used(),
                work.maximum()
            );
            let coverage = result
                .expect("complete Stage-1 geometry must fit unchanged scene work")
                .unwrap();
            let pixel = |[x, y]: [i64; 2]| {
                let mut alpha = 0.0;
                for row in 0..4 {
                    for column in 0..4 {
                        let sx = usize::try_from(x * 4 + column - coverage.origin_x).unwrap();
                        let sy = usize::try_from(y * 4 + row - coverage.origin_y).unwrap();
                        alpha += coverage.values[sy * coverage.width + sx];
                    }
                }
                alpha / 16.0
            };
            assert!((pixel(corner) - 0.227_529_040_847_66).abs() <= 1.0 / 2048.0);
            assert!((pixel(interior) - 1.0).abs() <= 1.0 / 2048.0);
            assert_eq!(pixel(exterior), 0.0);
        }
    }

    #[test]
    fn negative_subsample_erosion_cannot_disappear_without_geometric_proof() {
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [0.01, 0.01],
                [0.02, 0.01],
                [0.02, 0.02],
                [0.01, 0.01],
                [0.02, 0.02],
                [0.01, 0.02],
            ],
            bounds: [0.01, 0.01, 0.02, 0.02],
        };
        // Eroding the real [.01,.02]^2 caster by .001 leaves the
        // nonempty [.011,.019]^2 support, despite the empty center grid.
        // Once its exact exterior is known, the old fail-closed fallback
        // is no longer the correct conformance expectation.
        let tiny = prepare_untranslated_shadow_coverage(&mesh, -0.001, 0.5, 4.0, &[5, 1])
            .unwrap()
            .expect("bounded continuous erosion retains the tiny caster");
        assert!(
            tiny.values.iter().any(|value| *value > 0.0),
            "real eroded geometry must retain measurable blurred support"
        );
        assert!(
            prepare_untranslated_shadow_coverage(&mesh, -0.006, 0.5, 4.0, &[5, 2])
                .unwrap()
                .is_none(),
            "a disk larger than the caster's smallest enclosing half-width cannot fit"
        );
        assert!(
            prepare_untranslated_shadow_coverage(&mesh, -1000.0, 0.5, 4.0, &[5, 3])
                .unwrap()
                .is_none(),
            "complete erosion is proven geometrically before admitting any 4x halo"
        );
    }

    #[test]
    fn real_transparent_offscreen_fill_flows_through_neutral_shadow_coverage() {
        use crate::composition_2d::{
            Render2dAffineTransform, Render2dBrush, Render2dColorRgba8, Render2dItem,
            Render2dOpacity, Render2dPrimitive, Render2dRect, Render2dShape,
        };
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(-10.0, 0.0, 9.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let mesh = super::super::vector::neutral_support(&item, 0, 1.0, 16_384)
            .unwrap()
            .unwrap();
        let prepared = prepare_untranslated_shadow_coverage(&mesh, 0.0, 0.0, 4.0, &[3, 1])
            .unwrap()
            .expect("transparent offscreen color must not erase geometric support");
        assert_eq!(prepared.origin_x, -40);
        assert_eq!(prepared.origin_y, 0);
        assert_eq!((prepared.width, prepared.height), (36, 8));
        // The signed rectangle has exactly 9 x 2 logical units of neutral
        // area, or 288 covered cells at 4 samples per logical axis.
        // Polygon edge arithmetic is floating point, so physical area
        // conservation—not bit-exact per-cell equality—is the oracle.
        let area = prepared.values.iter().sum::<f64>();
        assert!((area - 288.0).abs() < 1.0e-9);
        assert!(prepared.values.iter().all(|&v| (0.0..=1.0).contains(&v)));
        // Offset (+12,0) remains a separate parent-frame painter translation.
        // Its shift is +48 samples: [-40,-4] -> [8,44], even though the
        // current unshifted source lies entirely off the caller's canvas.
        assert_eq!(prepared.origin_x + 12 * 4, 8);
        assert!(
            prepare_untranslated_shadow_coverage(&mesh, -10.0, 0.0, 4.0, &[3, 1])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn signed_spread_can_erode_real_neutral_geometry_completely() {
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            bounds: [0.0, 0.0, 1.0, 1.0],
        };
        assert!(
            prepare_untranslated_shadow_coverage(&mesh, -1.0, 0.0, 4.0, &[0],)
                .unwrap()
                .is_none()
        );
        let visible = prepare_untranslated_shadow_coverage(&mesh, 0.0, 0.25, 4.0, &[0])
            .unwrap()
            .unwrap();
        assert!(visible.values.iter().any(|value| *value > 0.0));
    }

    #[test]
    fn gaussian_neutral_coverage_is_normalized_and_has_finite_halo() {
        let source = NeutralMask {
            origin_x: -3,
            origin_y: 6,
            width: 1,
            height: 1,
            samples: vec![u8::MAX],
        };
        let kernel = gaussian_kernel(1.0, 1.0, &[0, 3]).unwrap();
        let blurred = blur_neutral_mask(&source, &kernel, &[0, 3]).unwrap();
        assert_eq!((blurred.origin_x, blurred.origin_y), (-6, 3));
        assert_eq!((blurred.width, blurred.height), (7, 7));
        let sum = blurred.values.iter().sum::<f64>();
        assert!((sum - 1.0).abs() < 1.0e-10);
        let center = blurred.values[3 * 7 + 3];
        assert!((center - kernel.weights[0].powi(2)).abs() < 1.0e-10);
        assert!((blurred.values[2 * 7 + 3] - blurred.values[4 * 7 + 3]).abs() < 1.0e-12);
        assert!((blurred.values[3 * 7 + 2] - blurred.values[3 * 7 + 4]).abs() < 1.0e-12);
    }

    #[test]
    fn gaussian_opaque_interior_and_identity_are_preserved() {
        let source = NeutralMask {
            origin_x: -1,
            origin_y: -1,
            width: 9,
            height: 9,
            samples: vec![u8::MAX; 81],
        };
        let kernel = gaussian_kernel(1.0, 1.0, &[0]).unwrap();
        let blurred = blur_neutral_mask(&source, &kernel, &[0]).unwrap();
        assert!((blurred.values[7 * blurred.width + 7] - 1.0).abs() < 1.0e-10);
        let identity = gaussian_kernel(0.0, 1.0, &[0]).unwrap();
        let unchanged = blur_neutral_mask(&source, &identity, &[0]).unwrap();
        assert_eq!((unchanged.origin_x, unchanged.origin_y), (-1, -1));
        assert_eq!((unchanged.width, unchanged.height), (9, 9));
        assert!(unchanged.values.iter().all(|v| *v == 1.0));
    }

    #[test]
    fn gaussian_workspace_limit_rejects_before_allocation_with_path() {
        let source = NeutralMask {
            origin_x: 0,
            origin_y: 0,
            width: 1024,
            height: 1024,
            samples: vec![u8::MAX; 1024 * 1024],
        };
        let kernel = gaussian_kernel(1.0, 1.0, &[7, 8]).unwrap();
        assert!(matches!(
            blur_neutral_mask(&source, &kernel, &[7, 8]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [7, 8]
        ));
    }

    #[test]
    fn rasterized_neutral_rect_is_phase_aligned_and_not_alpha_dependent() {
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [-1.0, -1.0],
                [1.0, -1.0],
                [1.0, 1.0],
                [-1.0, -1.0],
                [1.0, 1.0],
                [-1.0, 1.0],
            ],
            bounds: [-1.0, -1.0, 1.0, 1.0],
        };
        let mask = rasterize_neutral_mesh(&mesh, 4.0, &[1, 2])
            .unwrap()
            .unwrap();
        assert_eq!((mask.origin_x, mask.origin_y), (-4, -4));
        assert_eq!((mask.width, mask.height), (8, 8));
        assert!(mask.samples.iter().all(|value| *value == u8::MAX));
        let expanded = signed_euclidean_spread(&mask, 1.0, &[1, 2]).unwrap();
        assert_eq!((expanded.origin_x, expanded.origin_y), (-6, -6));
        let at = |x: usize, y: usize| expanded.samples[y * expanded.width + x];
        assert_eq!(at(1, 4), u8::MAX);
        assert_eq!(at(0, 4), 0);
        assert_eq!(at(1, 1), 0);
    }

    #[test]
    fn neutral_rasterization_rejects_preallocation_work_excess_with_exact_path() {
        let mesh = NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [1024.0, 0.0], [0.0, 1024.0]],
            bounds: [0.0, 0.0, 1024.0, 1024.0],
        };
        assert!(matches!(
            rasterize_neutral_mesh(&mesh, 4.0, &[3, 4]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [3, 4]
        ));
    }

    #[test]
    fn euclidean_spread_is_disk_not_square_and_retains_signed_halo() {
        let source = NeutralMask {
            origin_x: -5,
            origin_y: 7,
            width: 1,
            height: 1,
            samples: vec![u8::MAX],
        };
        let dilated = signed_euclidean_spread(&source, 1.0, &[1, 2]).unwrap();
        assert_eq!((dilated.origin_x, dilated.origin_y), (-7, 5));
        assert_eq!((dilated.width, dilated.height), (5, 5));
        let at = |x: usize, y: usize| dilated.samples[y * dilated.width + x];
        assert_eq!(at(2, 2), u8::MAX);
        assert_eq!(at(1, 2), u8::MAX);
        assert_eq!(at(3, 2), u8::MAX);
        assert_eq!(at(2, 1), u8::MAX);
        assert_eq!(at(2, 3), u8::MAX);
        assert_eq!(at(1, 1), 0); // diagonal distance sqrt(2) > one
        assert_eq!(at(3, 3), 0);
    }

    #[test]
    fn euclidean_erosion_respects_exterior_and_full_disappearance() {
        let source = NeutralMask {
            origin_x: -1,
            origin_y: -1,
            width: 3,
            height: 3,
            samples: vec![u8::MAX; 9],
        };
        let retained = signed_euclidean_spread(&source, -1.0, &[2]).unwrap();
        assert_eq!(retained.samples, vec![0, 0, 0, 0, u8::MAX, 0, 0, 0, 0]);
        assert_eq!((retained.origin_x, retained.origin_y), (-1, -1));
        let empty = signed_euclidean_spread(&source, -2.0, &[2]).unwrap();
        assert!(empty.samples.iter().all(|value| *value == 0));
        let identity = signed_euclidean_spread(&source, 0.0, &[2]).unwrap();
        assert_eq!(identity, source);
    }

    #[test]
    fn neutral_spread_rejects_oversized_halos_and_tracks_nested_paths() {
        let source = NeutralMask {
            origin_x: 0,
            origin_y: 0,
            width: 2,
            height: 2,
            samples: vec![u8::MAX; 4],
        };
        assert!(matches!(
            signed_euclidean_spread(&source, 513.0, &[4, 9]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [4, 9]
        ));
        assert!(matches!(
            signed_euclidean_spread(&source, f64::NAN, &[7]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [7]
        ));
    }

    #[test]
    fn finite_truncated_kernel_is_normalized_symmetric_and_zero_outside_cutoff() {
        let kernel = gaussian_kernel(1.0, 1.0, &[2, 3]).unwrap();
        assert_eq!(kernel.radius, 3);
        assert_eq!(kernel.weights.len(), 4);
        let sum = kernel.weights[0] + 2.0 * kernel.weights.iter().skip(1).sum::<f64>();
        assert!((sum - 1.0).abs() <= 1.0e-12);
        assert!(kernel.weights.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(
            kernel
                .weights
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0)
        );
        let fractional = gaussian_kernel(0.26, 1.0, &[4]).unwrap();
        assert_eq!(fractional.radius, 1);
        assert_eq!(fractional.weights[1], 0.0);
    }

    /// F3F-NS02-R2 physical candidate falsification: even with exact
    /// original Euclidean-eroded L membership, a binary 4x sample mask
    /// blurred with the current separable discrete kernel misses the
    /// independently accepted 64x64 Vulkan fractional-corner oracle.
    /// This rules out ONLY a naive binary 4x mask, not fractional coverage,
    /// higher-resolution physical masks or bounded analytic fast paths.
    #[test]
    fn naive_binary_four_x_gaussian_fails_accepted_concave_l_corner_quality() {
        let kernel = gaussian_kernel(0.1, 4.0, &[0]).unwrap();
        assert_eq!(kernel.radius, 2);
        assert_eq!(kernel.weights[2], 0.0);

        // The accepted neutral L is [10,20]x[10,15] U [10,15]x[15,20],
        // Euclidean eroded by radius 1. Every queried point around pixel
        // [14,15]^2 lies >1 from the far exterior boundaries. Distance
        // to the exact missing upper-right quadrant is a quarter-disk
        // below-left of (15,15), with adjacent straight exposed edges.
        let eroded_l_membership = |x: f64, y: f64| {
            let distance = if x < 15.0 && y < 15.0 {
                (15.0 - x).hypot(15.0 - y)
            } else if x >= 15.0 && y < 15.0 {
                15.0 - y
            } else if x < 15.0 && y >= 15.0 {
                15.0 - x
            } else {
                0.0
            };
            if distance >= 1.0 { 1.0 } else { 0.0 }
        };
        let mut sum = 0.0;
        for sy in 0..4 {
            for sx in 0..4 {
                let x = 14.0 + (f64::from(sx) + 0.5) / 4.0;
                let y = 14.0 + (f64::from(sy) + 0.5) / 4.0;
                for dy in -1_i32..=1 {
                    for dx in -1_i32..=1 {
                        let wx = kernel.weights[dx.unsigned_abs() as usize];
                        let wy = kernel.weights[dy.unsigned_abs() as usize];
                        let sample =
                            eroded_l_membership(x + f64::from(dx) * 0.25, y + f64::from(dy) * 0.25);
                        sum += wx * wy * sample;
                    }
                }
            }
        }
        let binary_mask_alpha = sum / 16.0;
        // Independent continuous finite-3sigma Euclidean-disk integral;
        // see tests/execution_2d/shadow.rs L/ring real-Vulkan oracles.
        let continuous_alpha = 0.227_529_040_847_66_f64;
        assert!((binary_mask_alpha - 0.197_596_973_313_322_35_f64).abs() < 1.0e-12);
        assert!(
            (binary_mask_alpha - continuous_alpha).abs() * 255.0 > 4.0,
            "a binary 4x mask must not be promoted without quality evidence"
        );
    }

    #[test]
    fn identity_and_bounded_failure_have_exact_nested_path() {
        let identity = gaussian_kernel(0.0, 4.0, &[0]).unwrap();
        assert_eq!(identity.radius, 0);
        assert_eq!(identity.weights, vec![1.0]);
        assert!(matches!(
            gaussian_kernel(100.0, 4.0, &[1, 9]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [1, 9]
        ));
    }
}
