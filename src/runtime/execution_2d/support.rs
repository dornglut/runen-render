//! Disposable neutral primitive support geometry for ordinary F3F effects.
//!
//! These triangles and bounds are physical preparations of F1's one immutable
//! geometry authority. They are never public renderer semantics, paint alpha,
//! an MSDF/image sampled footprint, or a second scene representation.

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

/// Private finite truncated-Gaussian *physical* kernel. These weights approximate
/// the F1 continuous convolution in a chosen group-parent sample frame; the
/// caller must independently bound complete sample-work and halo allocations.
#[derive(Debug)]
#[allow(dead_code, reason = "awaiting F3F group lowering")]
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
#[allow(dead_code, reason = "awaiting F3F group lowering")]
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
    rasterize_mesh_with_positive_spread(mesh, samples_per_logical_unit, 0.0, path)
}

/// Tests the true tessellated geometric distance at each parent-frame sample,
/// rather than dilating only previously occupied sample centers. A geometric
/// sliver with no original sample-center hit can still cast a spread shadow.
fn rasterize_mesh_with_positive_spread(
    mesh: &NeutralMesh,
    samples_per_logical_unit: f64,
    positive_spread: f64,
    path: &[usize],
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
fn area_sample(
    triangles: &[[f64; 2]],
    x0: f64,
    y0: f64,
    work: &mut usize,
    path: &[usize],
) -> Result<f64, crate::execution_2d::Render2dExecutionError> {
    use crate::execution_2d::Render2dSampleSpaceError;
    let resource = |detail| mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail);
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;
    let mut active = Vec::<[[f64; 2]; 3]>::new();
    for tri in triangles.as_chunks::<3>().0 {
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
        active
            .try_reserve(1)
            .map_err(|_| resource("neutral area triangle allocation failed"))?;
        active.push(*tri);
    }
    if active.is_empty() {
        return Ok(0.0);
    }
    // Admission bounds the cubic arrangement sweep even for maximally
    // overlapping tessellated strokes.
    let work_units = active
        .len()
        .checked_pow(3)
        .and_then(|v| v.checked_mul(8))
        .ok_or_else(|| resource("neutral area sweep cost overflow"))?;
    *work = work
        .checked_add(work_units)
        .ok_or_else(|| resource("neutral area sweep work overflow"))?;
    if *work > 134_217_728 {
        return Err(resource("neutral area sweep exceeds bounded work"));
    }
    let mut edges = Vec::new();
    let mut events = vec![y0, y1];
    for triangle in &active {
        // Fully covered unit cells have exact coverage one, no sweep needed.
        if [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            .iter()
            .all(|&p| {
                let a = orient(triangle[0], triangle[1], p);
                let b = orient(triangle[1], triangle[2], p);
                let c = orient(triangle[2], triangle[0], p);
                (a >= 0.0 && b >= 0.0 && c >= 0.0)
                    || (a <= 0.0 && b <= 0.0 && c <= 0.0)
            })
        {
            return Ok(1.0);
        }
        for i in 0..3 {
            let a = triangle[i];
            let b = triangle[(i + 1) % 3];
            edges.push((a, b));
            if a[1] > y0 && a[1] < y1 {
                events.push(a[1]);
            }
            if a[0] != b[0] {
                for x in [x0, x1] {
                    let t = (x - a[0]) / (b[0] - a[0]);
                    if t > 0.0 && t < 1.0 {
                        let y = (b[1] - a[1]).mul_add(t, a[1]);
                        if y > y0 && y < y1 {
                            events.push(y);
                        }
                    }
                }
            }
        }
    }
    for i in 0..edges.len() {
        let (a, b) = edges[i];
        let r = [b[0] - a[0], b[1] - a[1]];
        for &(c, d) in edges.iter().skip(i + 1) {
            let s = [d[0] - c[0], d[1] - c[1]];
            let denominator = r[0].mul_add(s[1], -r[1] * s[0]);
            if denominator == 0.0 {
                continue;
            }
            let q = [c[0] - a[0], c[1] - a[1]];
            let t = q[0].mul_add(s[1], -q[1] * s[0]) / denominator;
            let u = q[0].mul_add(r[1], -q[1] * r[0]) / denominator;
            if (0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u) {
                let y = r[1].mul_add(t, a[1]);
                if y > y0 && y < y1 {
                    events.push(y);
                }
            }
        }
    }
    events.sort_by(|a, b| a.total_cmp(b));
    events.dedup();
    let mut area = 0.0;
    for interval in events.windows(2) {
        let height = interval[1] - interval[0];
        if height <= 0.0 {
            continue;
        }
        let y = interval[0] + height * 0.5;
        let mut segments = Vec::<[f64; 2]>::new();
        for triangle in &active {
            let mut min_x = f64::INFINITY;
            let mut max_x = f64::NEG_INFINITY;
            for i in 0..3 {
                let a = triangle[i];
                let b = triangle[(i + 1) % 3];
                if (a[1] <= y && y < b[1]) || (b[1] <= y && y < a[1]) {
                    let t = (y - a[1]) / (b[1] - a[1]);
                    let x = (b[0] - a[0]).mul_add(t, a[0]);
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                }
            }
            let left = min_x.max(x0);
            let right = max_x.min(x1);
            if right > left {
                segments.push([left, right]);
            }
        }
        segments.sort_by(|a, b| a[0].total_cmp(&b[0]));
        let mut left = x0;
        let mut width = 0.0;
        for [start, end] in segments {
            let next = end.max(left);
            width += (next - left.max(start)).max(0.0);
            left = next;
        }
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
) -> Result<Option<NeutralCoverage>, crate::execution_2d::Render2dExecutionError> {
    let Some(grid) = rasterize_neutral_mesh(mesh, scale, path)? else {
        return Ok(None);
    };
    let transform = scale / mesh.units_per_parent_logical_unit;
    let mut triangles = filled(mesh.triangles.len(), [0.0; 2], path)?;
    for (out, &[x, y]) in triangles.iter_mut().zip(&mesh.triangles) {
        *out = [x * transform, y * transform];
    }
    let mut values = filled(grid.samples.len(), 0.0_f64, path)?;
    let mut work = 0;
    for y in 0..grid.height {
        for x in 0..grid.width {
            let x0 = f64::from(i32::try_from(grid.origin_x).expect("bounded lattice origin")) + as_f64(x);
            let y0 = f64::from(i32::try_from(grid.origin_y).expect("bounded lattice origin")) + as_f64(y);
            values[y * grid.width + x] = area_sample(&triangles, x0, y0, &mut work, path)?;
        }
    }
    Ok(Some(NeutralCoverage {
        origin_x: grid.origin_x,
        origin_y: grid.origin_y,
        width: grid.width,
        height: grid.height,
        values,
    }))
}

/// Separable finite 3σ Gaussian-style convolution over the spread mask.
/// The kernel is the normalized discrete approximation of the accepted F1
/// continuous truncated reference, with an explicit finite, complete halo.
/// Every allocation and worst-case sample tap is admitted before execution.
#[allow(dead_code, reason = "awaiting unified F3F painter integration")]
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
    if !input.values.iter().all(|v| v.is_finite() && (0.0..=1.0).contains(v)) {
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
            source[(y + pad) * width + x + pad] =
                input.values[y * input.width + x];
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

/// Retains the discrete binary-mask convolution for physical morphology and
/// legacy discrete conformance. The area-aware source path shares the same
/// bounded separable kernel and sample phase without u8 quantization.
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
    use crate::execution_2d::Render2dSampleSpaceError;
    if !spread.is_finite() || !sigma.is_finite() || sigma < 0.0 {
        return Err(mask_failure(
            path,
            Render2dSampleSpaceError::PrecisionLimit,
            "shadow spread or blur sigma is not representable",
        ));
    }
    // At zero signed spread, integrate the true triangle UNION over each
    // parent-frame sample cell before the finite Gaussian convolution.
    // A sliver between all sample centers must retain its positive measure.
    if spread == 0.0 && sigma > 0.0 {
        let Some(coverage) = rasterize_area_coverage(mesh, samples_per_logical_unit, path)? else {
            return Ok(None);
        };
        let kernel = gaussian_kernel(sigma, samples_per_logical_unit, path)?;
        return blur_neutral_coverage(&coverage, &kernel, path).map(Some);
    }
    // Positive spread evaluates distance from the original geometry at each
    // sample. Seed-then-dilate would falsely erase thin off-phase casters.
    // Negative spread retains the bounded binary distance-transform path.
    let Some(source) =
        rasterize_mesh_with_positive_spread(mesh, samples_per_logical_unit, spread.max(0.0), path)?
    else {
        return Ok(None);
    };
    let spread_mask = if spread > 0.0 {
        source
    } else {
        let physical_spread = spread * samples_per_logical_unit;
        if !physical_spread.is_finite() {
            return Err(mask_failure(
                path,
                Render2dSampleSpaceError::PrecisionLimit,
                "shadow spread exceeds finite sample coordinates",
            ));
        }
        signed_euclidean_spread(&source, physical_spread, path)?
    };
    if spread_mask.samples.iter().all(|sample| *sample == 0) {
        if spread >= 0.0 && sigma > 0.0 {
            // A nonempty, entirely off-phase source has a nonzero continuous
            // Gaussian integral. Do not misreport absent neutral geometry as
            // a valid zero-radiance effect until area-aware sampling exists.
            return Err(mask_failure(
                path,
                Render2dSampleSpaceError::PrecisionLimit,
                "subsample geometric support cannot be resolved for Gaussian blur",
            ));
        }
        return Ok(None);
    }
    let kernel = gaussian_kernel(sigma, samples_per_logical_unit, path)?;
    blur_neutral_mask(&spread_mask, &kernel, path).map(Some)
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
            rasterize_area_coverage(&mesh, 4.0, &[9, 7]),
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
        assert_eq!(spread.values[index(-2, -2)], 0.0);
        // The same caster lies completely between centers but has an
        // exact nonzero area of 0.04 x 0.04 physical sample units.
        let area = prepare_untranslated_shadow_coverage(&mesh, 0.0, 0.5, 4.0, &[6, 7])
            .unwrap()
            .unwrap();
        let total = area.values.iter().sum::<f64>();
        assert!((total - 0.0016).abs() < 1.0e-10);
        assert!(matches!(
            prepare_untranslated_shadow_coverage(&mesh, 0.001, 0.5, 4.0, &[6, 8]),
            Err(crate::execution_2d::Render2dExecutionError::SampleSpace {
                kind: crate::execution_2d::Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [6, 8]
        ));
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
        assert!(prepared.values.iter().all(|alpha| *alpha == 1.0));
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
