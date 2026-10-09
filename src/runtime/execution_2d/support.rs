//! Disposable neutral primitive support geometry for ordinary F3F effects.
//!
//! These triangles and bounds are physical preparations of F1's one immutable
//! geometry authority. They are never public renderer semantics, paint alpha,
//! an MSDF/image sampled footprint, or a second scene representation.

#[derive(Debug)]
#[allow(dead_code, reason = "awaiting F3F group lowering")]
pub(super) struct NeutralMesh {
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
        mask_failure(path, Render2dSampleSpaceError::ResourceLimit, "neutral mask allocation failed")
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
            intersection = ((values[q] + qf * qf) - (values[p] + pf * pf))
                / (2.0 * (qf - pf));
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
    let count = input.width.checked_mul(input.height)
        .ok_or_else(|| resource("neutral source mask extent overflow"))?;
    if count > MAX_NEUTRAL_MASK_SAMPLES || input.samples.len() != count {
        return Err(resource("neutral source mask exceeds the admitted sample budget"));
    }
    if radius == 0.0 || input.samples.iter().all(|value| *value == 0) {
        return Ok(input.clone());
    }
    let magnitude = radius.abs();
    let pad_f64 = magnitude.ceil() + 1.0;
    if !pad_f64.is_finite() || pad_f64 > f64::from(MAX_SPREAD_RADIUS_SAMPLES) {
        return Err(resource("Euclidean spread halo exceeds the bounded sample radius"));
    }
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "finite nonnegative halo is proven bounded by 512"
    )]
    let pad = pad_f64 as usize;
    let width = input.width.checked_add(pad.checked_mul(2)
        .ok_or_else(|| resource("neutral mask padding overflow"))?)
        .ok_or_else(|| resource("neutral mask width overflow"))?;
    let height = input.height.checked_add(pad.checked_mul(2)
        .ok_or_else(|| resource("neutral mask padding overflow"))?)
        .ok_or_else(|| resource("neutral mask height overflow"))?;
    let cells = width.checked_mul(height)
        .ok_or_else(|| resource("neutral mask sample count overflow"))?;
    if cells > MAX_NEUTRAL_MASK_SAMPLES {
        return Err(resource("padded neutral mask exceeds bounded sample budget"));
    }
    let pad_i64 = i64::try_from(pad).map_err(|_| resource("neutral mask origin overflow"))?;
    let origin_x = input.origin_x.checked_sub(pad_i64)
        .ok_or_else(|| precision("signed neutral mask x origin overflow"))?;
    let origin_y = input.origin_y.checked_sub(pad_i64)
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
