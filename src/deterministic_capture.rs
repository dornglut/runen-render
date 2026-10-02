//! Public RunenRender correlation and interpretation for maintained radiance captures.
//!
//! RunenGPU owns readback transport, bytes, and retained-resource continuity.  This module only
//! binds one fresh post-result readback correlation to one exact maintained radiance destination
//! and interprets the returned carrier through RunenRender's private maintained mapping.

use super::admission::RenderOutputDestination;
use super::deterministic_carrier::{WORD_BYTES, decode_word, maintained_evaluation_value};
use super::deterministic_execution::SubmittedDeterministicRender;
use super::request::{RenderOutputValue, RenderRadiometricRepresentation, RenderResultTopology};
use runen_gpu::{
    GpuContext, GpuContextAffinity, GpuDataLayout, GpuOpaqueContentContinuity, GpuReadbackId,
    GpuReadbackStatus, GpuResourceLifetime, GpuResourceRef, GpuSubmission,
    GpuSubmissionFailureKind, GpuSubmissionId, GpuSubmissionStatus, GpuTextureFormat,
    GpuTextureUsage, GpuTransferRegion,
};
use std::error::Error;
use std::fmt;

const CARRIER_FORMAT: GpuTextureFormat = GpuTextureFormat::R32Uint;

/// One exact, one-shot correlation witness for a product-owned readback of a formed radiance
/// output.
///
/// The source and fresh ID are the only physical evidence exposed to the product.  The exact
/// submitted-render binding remains private and is checked when the request is consumed.
pub struct RenderDeterministicRadianceCaptureRequest {
    output_index: usize,
    source: GpuTransferRegion,
    readback_id: GpuReadbackId,
    submission_affinity: GpuContextAffinity,
    submission_id: GpuSubmissionId,
}

impl RenderDeterministicRadianceCaptureRequest {
    pub const fn output_index(&self) -> usize {
        self.output_index
    }

    /// Exact whole-base-mip source to use for the ordinary public RunenGPU readback operation.
    pub fn source(&self) -> &GpuTransferRegion {
        &self.source
    }

    /// Fresh process-local correlation ID allocated after verified result formation.
    pub const fn readback_id(&self) -> GpuReadbackId {
        self.readback_id
    }
}

/// Captured finite maintained evaluation data for one requested spectral-radiance lattice.
///
/// This is an observation of a maintained physical realization, not a `RenderResult`, semantic
/// certification, image, color value, persistence record, or artifact schema.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderCapturedDeterministicRadiance {
    output_index: usize,
    topology: RenderResultTopology,
    representation: RenderRadiometricRepresentation,
    samples: Vec<f32>,
}

impl RenderCapturedDeterministicRadiance {
    pub const fn output_index(&self) -> usize {
        self.output_index
    }

    pub const fn topology(&self) -> RenderResultTopology {
        self.topology
    }

    pub const fn representation(&self) -> RenderRadiometricRepresentation {
        self.representation
    }

    /// Row-major finite evaluation samples matching the retained 2D lattice topology.
    pub fn samples(&self) -> &[f32] {
        &self.samples
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderDeterministicRadianceCaptureRequestError {
    VerificationNotFormed,
    OutputIndexOutOfRange,
    OutputNotRadiance,
    OutputTopologyUnsupported,
    OutputDestinationUnsupported,
    DestinationNotRetained,
    DestinationNotCopySource,
    CarrierFormatUnsupported,
    SourceUnavailable,
    ReadbackIdAllocationExhausted,
}

impl fmt::Display for RenderDeterministicRadianceCaptureRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = match self {
            Self::VerificationNotFormed => "verified result formation has not succeeded",
            Self::OutputIndexOutOfRange => "requested output index is not admitted",
            Self::OutputNotRadiance => "requested output is not spectral radiance",
            Self::OutputTopologyUnsupported => {
                "requested output is not a maintained 2D sample lattice"
            }
            Self::OutputDestinationUnsupported => "admitted output is not a sample-lattice texture",
            Self::DestinationNotRetained => "radiance capture requires a retained destination",
            Self::DestinationNotCopySource => "radiance capture destination lacks CopySource usage",
            Self::CarrierFormatUnsupported => {
                "admitted destination is not the maintained R32Uint carrier"
            }
            Self::SourceUnavailable => {
                "the exact whole-base-mip readback source could not be formed"
            }
            Self::ReadbackIdAllocationExhausted => {
                "RunenGPU readback correlation identity allocation was exhausted"
            }
        };
        formatter.write_str(detail)
    }
}

impl Error for RenderDeterministicRadianceCaptureRequestError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderDeterministicRadianceCaptureError {
    VerificationNotFormed,
    RequestCorrelationMismatch,
    ContextAffinityMismatch,
    RetainedContinuityUnavailable,
    RetainedContinuityAffinityMismatch,
    RetainedContinuityResourceMismatch,
    RetainedContinuityNotEstablished,
    RendererWriteNoLongerCurrent,
    ProductSubmissionAffinityMismatch,
    ProductSubmissionPending,
    ProductSubmissionFailed { kind: GpuSubmissionFailureKind },
    ReadbackCorrelationMissing,
    ReadbackSourceMismatch,
    ReadbackPending,
    ReadbackFailed { kind: GpuSubmissionFailureKind },
    ReadbackFormatMismatch,
    ReadbackLayoutMismatch,
    ReadbackByteLengthMismatch,
    NonFiniteSample,
    HostAllocation,
}

impl fmt::Display for RenderDeterministicRadianceCaptureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = match self {
            Self::VerificationNotFormed => "verified result formation has not succeeded",
            Self::RequestCorrelationMismatch => {
                "capture request does not belong to this submitted render/output"
            }
            Self::ContextAffinityMismatch => {
                "capture context does not match the renderer submission affinity"
            }
            Self::RetainedContinuityUnavailable => {
                "current retained-resource continuity is unavailable"
            }
            Self::RetainedContinuityAffinityMismatch => {
                "retained continuity has a different context affinity"
            }
            Self::RetainedContinuityResourceMismatch => {
                "retained continuity names a different resource"
            }
            Self::RetainedContinuityNotEstablished => {
                "retained opaque content continuity is not established"
            }
            Self::RendererWriteNoLongerCurrent => {
                "the verified renderer write is no longer the current retained writer"
            }
            Self::ProductSubmissionAffinityMismatch => {
                "product readback submission has a different context affinity"
            }
            Self::ProductSubmissionPending => "product readback submission is still pending",
            Self::ProductSubmissionFailed { .. } => "product readback submission failed",
            Self::ReadbackCorrelationMissing => {
                "capture readback ID is absent from the product submission"
            }
            Self::ReadbackSourceMismatch => {
                "returned readback source differs from the exact capture source"
            }
            Self::ReadbackPending => "capture readback is still pending",
            Self::ReadbackFailed { .. } => "capture readback failed",
            Self::ReadbackFormatMismatch => {
                "capture readback format is not the maintained R32Uint carrier"
            }
            Self::ReadbackLayoutMismatch => {
                "capture readback layout does not match the retained lattice"
            }
            Self::ReadbackByteLengthMismatch => {
                "capture readback byte length does not match the retained lattice"
            }
            Self::NonFiniteSample => "capture contains a non-finite maintained evaluation value",
            Self::HostAllocation => "capture sample storage allocation failed",
        };
        formatter.write_str(detail)
    }
}

impl Error for RenderDeterministicRadianceCaptureError {}

pub(super) fn mint_request(
    submitted: &SubmittedDeterministicRender,
    output_index: usize,
) -> Result<RenderDeterministicRadianceCaptureRequest, RenderDeterministicRadianceCaptureRequestError>
{
    if !submitted.result_is_formed() {
        return Err(RenderDeterministicRadianceCaptureRequestError::VerificationNotFormed);
    }

    let admitted = submitted.admitted().admitted();
    let output = admitted
        .outputs()
        .iter()
        .find(|output| output.output_index() == output_index)
        .ok_or(RenderDeterministicRadianceCaptureRequestError::OutputIndexOutOfRange)?;
    let requested = admitted
        .plan()
        .request()
        .outputs()
        .get(output_index)
        .copied()
        .ok_or(RenderDeterministicRadianceCaptureRequestError::OutputIndexOutOfRange)?;
    let RenderOutputValue::Radiance { .. } = requested.spec().value() else {
        return Err(RenderDeterministicRadianceCaptureRequestError::OutputNotRadiance);
    };
    if requested
        .spec()
        .topology()
        .sample_lattice_dimensions()
        .is_none()
    {
        return Err(RenderDeterministicRadianceCaptureRequestError::OutputTopologyUnsupported);
    }

    let RenderOutputDestination::SampleLatticeTexture(destination) = output.binding().destination()
    else {
        return Err(RenderDeterministicRadianceCaptureRequestError::OutputDestinationUnsupported);
    };
    let descriptor = destination.descriptor();
    if descriptor.format() != CARRIER_FORMAT {
        return Err(RenderDeterministicRadianceCaptureRequestError::CarrierFormatUnsupported);
    }
    if descriptor.common().lifetime() != GpuResourceLifetime::Retained {
        return Err(RenderDeterministicRadianceCaptureRequestError::DestinationNotRetained);
    }
    if !descriptor.usages().contains(GpuTextureUsage::CopySource) {
        return Err(RenderDeterministicRadianceCaptureRequestError::DestinationNotCopySource);
    }
    let source = runen_gpu::GpuTextureCopyRegion::whole_base_mip(destination)
        .map_err(|_| RenderDeterministicRadianceCaptureRequestError::SourceUnavailable)?
        .into();
    let readback_id = GpuReadbackId::allocate().map_err(|_| {
        RenderDeterministicRadianceCaptureRequestError::ReadbackIdAllocationExhausted
    })?;
    Ok(RenderDeterministicRadianceCaptureRequest {
        output_index,
        source,
        readback_id,
        submission_affinity: submitted.submission().affinity(),
        submission_id: submitted.submission().id(),
    })
}

/// Decode the already-normalized physical tail of one maintained radiance readback.
///
/// Correlation and ownership are checked by [`capture`] before calling this helper. The helper is
/// intentionally pure so the defensive metadata branches remain directly provable without
/// weakening RunenGPU's normalized-readback encapsulation. Native carrier meaning remains solely
/// in `deterministic_carrier`.
fn decode_radiance_samples(
    width: u32,
    height: u32,
    format: Option<GpuTextureFormat>,
    layout: GpuDataLayout,
    bytes: &[u8],
) -> Result<Vec<f32>, RenderDeterministicRadianceCaptureError> {
    if format != Some(CARRIER_FORMAT) {
        return Err(RenderDeterministicRadianceCaptureError::ReadbackFormatMismatch);
    }
    let expected_byte_len = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|samples| samples.checked_mul(WORD_BYTES as u64))
        .ok_or(RenderDeterministicRadianceCaptureError::ReadbackLayoutMismatch)?;
    let expected_row_bytes = u64::from(width)
        .checked_mul(WORD_BYTES as u64)
        .ok_or(RenderDeterministicRadianceCaptureError::ReadbackLayoutMismatch)?;
    if layout.byte_len() != expected_byte_len
        || layout.alignment() != WORD_BYTES as u64
        || layout.stride() != expected_row_bytes
        || layout.element_count() != u64::from(height)
    {
        return Err(RenderDeterministicRadianceCaptureError::ReadbackLayoutMismatch);
    }
    if u64::try_from(bytes.len()).ok() != Some(expected_byte_len) {
        return Err(RenderDeterministicRadianceCaptureError::ReadbackByteLengthMismatch);
    }

    let sample_count = usize::try_from(u64::from(width) * u64::from(height))
        .map_err(|_| RenderDeterministicRadianceCaptureError::ReadbackLayoutMismatch)?;
    let mut samples = Vec::new();
    samples
        .try_reserve_exact(sample_count)
        .map_err(|_| RenderDeterministicRadianceCaptureError::HostAllocation)?;
    for word_bytes in bytes.as_chunks::<WORD_BYTES>().0 {
        let word = decode_word(word_bytes);
        let value = maintained_evaluation_value(word)
            .ok_or(RenderDeterministicRadianceCaptureError::NonFiniteSample)?;
        samples.push(value);
    }
    if samples.len() != sample_count {
        return Err(RenderDeterministicRadianceCaptureError::ReadbackByteLengthMismatch);
    }
    Ok(samples)
}

pub(super) fn capture(
    submitted: &SubmittedDeterministicRender,
    request: RenderDeterministicRadianceCaptureRequest,
    context: &GpuContext,
    product_submission: &GpuSubmission,
) -> Result<RenderCapturedDeterministicRadiance, RenderDeterministicRadianceCaptureError> {
    if !submitted.result_is_formed() {
        return Err(RenderDeterministicRadianceCaptureError::VerificationNotFormed);
    }
    if request.submission_affinity != submitted.submission().affinity()
        || request.submission_id != submitted.submission().id()
    {
        return Err(RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch);
    }
    if context.affinity() != submitted.submission().affinity() {
        return Err(RenderDeterministicRadianceCaptureError::ContextAffinityMismatch);
    }

    let admitted = submitted.admitted().admitted();
    let output = admitted
        .outputs()
        .iter()
        .find(|output| output.output_index() == request.output_index)
        .ok_or(RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch)?;
    let requested = admitted
        .plan()
        .request()
        .outputs()
        .get(request.output_index)
        .copied()
        .ok_or(RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch)?;
    let RenderOutputValue::Radiance { representation } = requested.spec().value() else {
        return Err(RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch);
    };
    let topology = requested.spec().topology();
    let Some((width, height)) = topology.sample_lattice_dimensions() else {
        return Err(RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch);
    };
    let RenderOutputDestination::SampleLatticeTexture(destination) = output.binding().destination()
    else {
        return Err(RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch);
    };
    let expected_source = runen_gpu::GpuTextureCopyRegion::whole_base_mip(destination)
        .map_err(|_| RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch)?
        .into();
    if request.source != expected_source || destination.descriptor().format() != CARRIER_FORMAT {
        return Err(RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch);
    }

    let continuity = context
        .retained_resource_continuity(destination.diagnostic_identity())
        .ok_or(RenderDeterministicRadianceCaptureError::RetainedContinuityUnavailable)?;
    if continuity.affinity() != submitted.submission().affinity() {
        return Err(RenderDeterministicRadianceCaptureError::RetainedContinuityAffinityMismatch);
    }
    if continuity.resource() != &GpuResourceRef::from(destination.clone()) {
        return Err(RenderDeterministicRadianceCaptureError::RetainedContinuityResourceMismatch);
    }
    match continuity.opaque_content() {
        GpuOpaqueContentContinuity::Established {
            last_completed_write,
        } if last_completed_write == submitted.submission().id() => {}
        GpuOpaqueContentContinuity::Established { .. } => {
            return Err(RenderDeterministicRadianceCaptureError::RendererWriteNoLongerCurrent);
        }
        GpuOpaqueContentContinuity::Unestablished | GpuOpaqueContentContinuity::Unknown => {
            return Err(RenderDeterministicRadianceCaptureError::RetainedContinuityNotEstablished);
        }
    }

    if product_submission.affinity() != submitted.submission().affinity() {
        return Err(RenderDeterministicRadianceCaptureError::ProductSubmissionAffinityMismatch);
    }
    match product_submission.status() {
        GpuSubmissionStatus::Accepted => {
            return Err(RenderDeterministicRadianceCaptureError::ProductSubmissionPending);
        }
        GpuSubmissionStatus::Failed(failure) => {
            return Err(
                RenderDeterministicRadianceCaptureError::ProductSubmissionFailed {
                    kind: failure.kind(),
                },
            );
        }
        GpuSubmissionStatus::Completed => {}
    }
    let readback = product_submission
        .readback(request.readback_id)
        .ok_or(RenderDeterministicRadianceCaptureError::ReadbackCorrelationMissing)?;
    if readback.source() != &request.source {
        return Err(RenderDeterministicRadianceCaptureError::ReadbackSourceMismatch);
    }
    let bytes = match readback.status() {
        GpuReadbackStatus::Pending => {
            return Err(RenderDeterministicRadianceCaptureError::ReadbackPending);
        }
        GpuReadbackStatus::Failed(failure) => {
            return Err(RenderDeterministicRadianceCaptureError::ReadbackFailed {
                kind: failure.kind(),
            });
        }
        GpuReadbackStatus::Ready(bytes) => bytes,
    };

    let samples = decode_radiance_samples(
        width,
        height,
        bytes.texture_format(),
        bytes.layout(),
        bytes.as_bytes(),
    )?;
    Ok(RenderCapturedDeterministicRadiance {
        output_index: request.output_index,
        topology,
        representation,
        samples,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout(byte_len: u64, stride: u64, element_count: u64) -> GpuDataLayout {
        GpuDataLayout::new(
            "radiance metadata proof",
            byte_len,
            WORD_BYTES as u64,
            stride,
            element_count,
        )
        .expect("test metadata must be a valid RunenGPU layout")
    }

    #[test]
    fn physical_metadata_and_finite_carrier_validation_fail_closed() {
        let valid_layout = layout(16, 8, 2);
        let valid_bytes = [
            0.0_f32.to_bits().to_ne_bytes(),
            1.25_f32.to_bits().to_ne_bytes(),
            2.5_f32.to_bits().to_ne_bytes(),
            3.75_f32.to_bits().to_ne_bytes(),
        ]
        .concat();
        assert_eq!(
            decode_radiance_samples(
                2,
                2,
                Some(GpuTextureFormat::Rgba8Unorm),
                valid_layout,
                &valid_bytes
            ),
            Err(RenderDeterministicRadianceCaptureError::ReadbackFormatMismatch)
        );
        assert_eq!(
            decode_radiance_samples(2, 2, Some(CARRIER_FORMAT), layout(16, 16, 1), &valid_bytes),
            Err(RenderDeterministicRadianceCaptureError::ReadbackLayoutMismatch)
        );
        assert_eq!(
            decode_radiance_samples(2, 2, Some(CARRIER_FORMAT), valid_layout, &valid_bytes[..12]),
            Err(RenderDeterministicRadianceCaptureError::ReadbackByteLengthMismatch)
        );
        let non_finite = [f32::NAN.to_bits().to_ne_bytes()].concat();
        assert_eq!(
            decode_radiance_samples(1, 1, Some(CARRIER_FORMAT), layout(4, 4, 1), &non_finite),
            Err(RenderDeterministicRadianceCaptureError::NonFiniteSample)
        );
        assert_eq!(
            decode_radiance_samples(2, 2, Some(CARRIER_FORMAT), valid_layout, &valid_bytes,)
                .expect("valid row-major carrier must decode"),
            vec![0.0, 1.25, 2.5, 3.75]
        );
    }
}
