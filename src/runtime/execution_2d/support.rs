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

const MAX_GAUSSIAN_RADIUS_SAMPLES: usize = 512;

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
    use crate::execution_2d::{
        Render2dExecutionError, Render2dSampleSpaceError,
    };
    let fail = |kind, detail: &'static str| Render2dExecutionError::SampleSpace {
        kind,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    };
    if !sigma.is_finite()
        || sigma < 0.0
        || !sample_scale.is_finite()
        || sample_scale <= 0.0
    {
        return Err(fail(Render2dSampleSpaceError::PrecisionLimit, "unrepresentable group-parent shadow kernel"));
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
        return Err(fail(Render2dSampleSpaceError::PrecisionLimit, "shadow blur sample scale is not representable"));
    }
    if cutoff.ceil() > MAX_GAUSSIAN_RADIUS_SAMPLES as f64 {
        return Err(fail(Render2dSampleSpaceError::ResourceLimit, "shadow blur radius exceeds bounded sample kernel"));
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss, reason = "finite nonnegative radius was bounded to 512 samples")]
    let radius = cutoff.ceil() as usize;
    let mut weights = Vec::new();
    weights.try_reserve_exact(radius + 1).map_err(|_| {
        fail(Render2dSampleSpaceError::ResourceLimit, "shadow blur kernel allocation failed")
    })?;
    for offset in 0..=radius {
        let distance = offset as f64;
        let weight = if distance <= cutoff {
            (-0.5 * (distance / physical_sigma).powi(2)).exp()
        } else {
            0.0
        };
        weights.push(weight);
    }
    let normalizer = weights[0] + 2.0 * weights.iter().skip(1).sum::<f64>();
    if !normalizer.is_finite() || normalizer <= 0.0 {
        return Err(fail(Render2dSampleSpaceError::PrecisionLimit, "shadow blur kernel normalization failed"));
    }
    for weight in &mut weights {
        *weight /= normalizer;
    }
    Ok(GaussianKernel { radius, weights })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finite_truncated_kernel_is_normalized_symmetric_and_zero_outside_cutoff() {
        let kernel = gaussian_kernel(1.0, 1.0, &[2, 3]).unwrap();
        assert_eq!(kernel.radius, 3);
        assert_eq!(kernel.weights.len(), 4);
        let sum = kernel.weights[0] + 2.0 * kernel.weights.iter().skip(1).sum::<f64>();
        assert!((sum - 1.0).abs() <= 1.0e-12);
        assert!(kernel.weights.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(kernel.weights.iter().all(|value| value.is_finite() && *value >= 0.0));
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
