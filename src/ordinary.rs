//! Curated ordinary public surface for the maintained RunenRender path.
//!
//! This module gives ordinary consumers progressive disclosure over the existing semantic
//! scene/request -> planning -> admission -> lowering -> RunenGPU authority. It does not introduce
//! a second planner, renderer, submission lifecycle, or output-value authority. The maintained
//! deterministic implementation remains an internal method realization behind these names.

use super::admission::{
    AdmittedRenderPlan, RenderOutputBinding, RenderRepresentationAvailabilityFact,
};
use super::field_input::{RenderFieldSemanticInputBinding, RenderFieldSemanticInputGeneration};
use super::lowering::RenderWorkSet;
use super::render_result::RenderResult;
use super::representation::RenderRepresentationId;
use super::request::{RenderRadiometricRepresentation, RenderRequest, RenderResultTopology};
use super::runtime::admission::{
    AdmittedDeterministicRender, RenderDeterministicAdmissionFailure,
    admit_deterministic_render_with_semantic_inputs,
};
use super::runtime::capture::{
    RenderCapturedDeterministicRadiance, RenderDeterministicRadianceCaptureError,
    RenderDeterministicRadianceCaptureRequest, RenderDeterministicRadianceCaptureRequestError,
};
pub use super::runtime::execution::RenderObjectIdentityDecoder;
use super::runtime::execution::{
    DeterministicResourceCache, PreparedDeterministicRadianceOutput, PreparedDeterministicRender,
    RenderDeterministicExecutionError, RenderDeterministicResultFormationError,
    RenderTemporalExecutionEvidence as DeterministicTemporalExecutionEvidence,
    SubmittedDeterministicRender, prepare_deterministic_render,
    prepare_deterministic_render_with_cache_in_scope_and_evaluation, submit_deterministic_render,
    submit_deterministic_render_for_verified_result,
};
use super::runtime::verification::{
    RenderDeterministicVerificationEligibilityError, RenderDeterministicVerificationError,
    RenderDeterministicVerifiedSubmissionError,
};
use super::scene::{RenderObjectId, RenderSceneSnapshot};
use super::surface_input::{
    RenderSurfaceSemanticInputBinding, RenderSurfaceSemanticInputGeneration,
};
use runen_gpu::{
    GpuContext, GpuExportRelationship, GpuReadbackId, GpuResourceProvenance, GpuResourceRef,
    GpuSubmission, GpuSubmissionFailureKind, GpuSubmissionStatus, GpuTextureHandle,
    GpuTransferRegion, GpuWorkImport, GpuWorkSubmissionError,
};
use std::error::Error;
use std::fmt;

/// High-level category for failure before maintained execution is prepared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderAdmissionErrorKind {
    Planning,
    Admission,
    Compatibility,
}

/// Failure while planning, semantically admitting, or checking maintained-method compatibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderAdmissionError {
    inner: RenderDeterministicAdmissionFailure,
}

impl RenderAdmissionError {
    /// Owner-oriented failure category without implementation-specific error names.
    pub fn kind(&self) -> RenderAdmissionErrorKind {
        match &self.inner {
            RenderDeterministicAdmissionFailure::Planning(_) => RenderAdmissionErrorKind::Planning,
            RenderDeterministicAdmissionFailure::Admission(_) => {
                RenderAdmissionErrorKind::Admission
            }
            RenderDeterministicAdmissionFailure::Compatibility(_) => {
                RenderAdmissionErrorKind::Compatibility
            }
        }
    }
}

impl fmt::Display for RenderAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            RenderDeterministicAdmissionFailure::Planning(error) => {
                write!(formatter, "render planning failed: {error}")
            }
            RenderDeterministicAdmissionFailure::Admission(error) => {
                write!(formatter, "render admission failed: {error}")
            }
            RenderDeterministicAdmissionFailure::Compatibility(error) => {
                write!(
                    formatter,
                    "maintained renderer compatibility failed: {error}"
                )
            }
        }
    }
}

impl Error for RenderAdmissionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner {
            RenderDeterministicAdmissionFailure::Planning(error) => Some(error),
            RenderDeterministicAdmissionFailure::Admission(error) => Some(error),
            RenderDeterministicAdmissionFailure::Compatibility(error) => Some(error),
        }
    }
}

/// High-level category for ordinary maintained execution failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderExecutionErrorKind {
    Lowering,
    RunenShaderCompilation,
    RunenGpuPreparation,
    Submission,
}

/// Failure while lowering admitted renderer meaning or submitting it through public RunenGPU.
#[derive(Debug)]
pub struct RenderExecutionError {
    inner: RenderDeterministicExecutionError,
}

impl RenderExecutionError {
    /// Owner-oriented failure category.
    pub fn kind(&self) -> RenderExecutionErrorKind {
        match &self.inner {
            RenderDeterministicExecutionError::Lowering(
                super::runtime::execution::RenderDeterministicLoweringError::RunenShaderCompilation(
                    _,
                ),
            ) => RenderExecutionErrorKind::RunenShaderCompilation,
            RenderDeterministicExecutionError::Lowering(
                super::runtime::execution::RenderDeterministicLoweringError::RunenGpuPreparation(_),
            ) => RenderExecutionErrorKind::RunenGpuPreparation,
            RenderDeterministicExecutionError::Lowering(_) => RenderExecutionErrorKind::Lowering,
            RenderDeterministicExecutionError::Submission(_) => {
                RenderExecutionErrorKind::Submission
            }
        }
    }

    /// RunenShader-owned compilation failure when maintained renderer source did not compile.
    ///
    /// The concrete renderer wrapper stays private; callers receive the exact error chain as an
    /// ordinary error source without RunenRender mirroring RunenShader's diagnostic taxonomy.
    pub fn runen_shader_compilation_source(&self) -> Option<&(dyn Error + 'static)> {
        let RenderDeterministicExecutionError::Lowering(
            super::runtime::execution::RenderDeterministicLoweringError::RunenShaderCompilation(
                error,
            ),
        ) = &self.inner
        else {
            return None;
        };
        Some(error)
    }

    /// Exact public RunenGPU owner error when failure happened during GPU preparation.
    ///
    /// The returned error remains owned by RunenGPU. Callers that need a concrete category can
    /// downcast this source to the public RunenGPU error type they understand without RunenRender
    /// mirroring RunenGPU's error taxonomy.
    pub fn runen_gpu_preparation_source(&self) -> Option<&(dyn Error + 'static)> {
        let RenderDeterministicExecutionError::Lowering(
            super::runtime::execution::RenderDeterministicLoweringError::RunenGpuPreparation(error),
        ) = &self.inner
        else {
            return None;
        };
        Error::source(error)
    }

    /// Stable public RunenGPU submission failure when execution reached physical submission.
    pub const fn submission_error(&self) -> Option<&GpuWorkSubmissionError> {
        match &self.inner {
            RenderDeterministicExecutionError::Submission(error) => Some(error),
            RenderDeterministicExecutionError::Lowering(_) => None,
        }
    }
}

impl fmt::Display for RenderExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            RenderDeterministicExecutionError::Lowering(error) => {
                write!(formatter, "render lowering failed: {error}")
            }
            RenderDeterministicExecutionError::Submission(error) => {
                write!(formatter, "RunenGPU submission failed: {error}")
            }
        }
    }
}

impl Error for RenderExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self.inner {
            RenderDeterministicExecutionError::Lowering(error) => Some(error),
            RenderDeterministicExecutionError::Submission(error) => Some(error),
        }
    }
}

/// Stable category for verifier-domain eligibility failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderVerificationEligibilityErrorKind {
    SelectedObservationMissing,
    PerspectiveFieldOfViewUnsupported,
    SamplingSupportUnsupported,
    ObservationLinearBasisUnsupported,
    SelectedObjectStateMissing,
    ObjectLocalScaleUnsupported,
    ObjectHandednessUnsupported,
    ObjectLinearBasisUnsupported,
}

/// High-level category for a submission that requested semantic result formation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderResultSubmissionErrorKind {
    Eligibility,
    Execution,
    ReadbackCardinality,
    Correlation,
}

/// Failure while selecting result-verification intent or authoring its exact submission.
#[derive(Debug)]
pub struct RenderResultSubmissionError {
    inner: RenderDeterministicVerifiedSubmissionError,
}

impl RenderResultSubmissionError {
    /// Owner-oriented failure category.
    pub fn kind(&self) -> RenderResultSubmissionErrorKind {
        match &self.inner {
            RenderDeterministicVerifiedSubmissionError::Eligibility(_) => {
                RenderResultSubmissionErrorKind::Eligibility
            }
            RenderDeterministicVerifiedSubmissionError::Execution(_) => {
                RenderResultSubmissionErrorKind::Execution
            }
            RenderDeterministicVerifiedSubmissionError::ReadbackCardinality { .. } => {
                RenderResultSubmissionErrorKind::ReadbackCardinality
            }
            RenderDeterministicVerifiedSubmissionError::OutputCorrelationChanged { .. }
            | RenderDeterministicVerifiedSubmissionError::DuplicateReadbackCorrelation { .. }
            | RenderDeterministicVerifiedSubmissionError::MissingSubmissionReadback { .. } => {
                RenderResultSubmissionErrorKind::Correlation
            }
        }
    }

    /// Verifier-domain reason when submission was rejected before maintained execution.
    pub fn verification_eligibility_kind(&self) -> Option<RenderVerificationEligibilityErrorKind> {
        let RenderDeterministicVerifiedSubmissionError::Eligibility(error) = &self.inner else {
            return None;
        };
        Some(verification_eligibility_kind(error))
    }

    /// Referenced observation index when eligibility failure is observation-scoped.
    pub const fn observation_index(&self) -> Option<usize> {
        match &self.inner {
            RenderDeterministicVerifiedSubmissionError::Eligibility(
                RenderDeterministicVerificationEligibilityError::SelectedObservationMissing {
                    observation_index,
                }
                | RenderDeterministicVerificationEligibilityError::PerspectiveFieldOfViewUnsupported {
                    observation_index,
                }
                | RenderDeterministicVerificationEligibilityError::SamplingSupportUnsupported {
                    observation_index,
                }
                | RenderDeterministicVerificationEligibilityError::ObservationLinearBasisUnsupported {
                    observation_index,
                },
            ) => Some(*observation_index),
            _ => None,
        }
    }

    /// Referenced renderer object when eligibility failure is object-scoped.
    pub const fn object_id(&self) -> Option<RenderObjectId> {
        match &self.inner {
            RenderDeterministicVerifiedSubmissionError::Eligibility(
                RenderDeterministicVerificationEligibilityError::SelectedObjectStateMissing {
                    object_id,
                }
                | RenderDeterministicVerificationEligibilityError::ObjectLocalScaleUnsupported {
                    object_id,
                }
                | RenderDeterministicVerificationEligibilityError::ObjectHandednessUnsupported {
                    object_id,
                }
                | RenderDeterministicVerificationEligibilityError::ObjectLinearBasisUnsupported {
                    object_id,
                },
            ) => Some(*object_id),
            _ => None,
        }
    }

    /// Expected and actual readback counts when exact-submission cardinality changed.
    pub const fn readback_cardinality(&self) -> Option<(usize, usize)> {
        match &self.inner {
            RenderDeterministicVerifiedSubmissionError::ReadbackCardinality {
                expected,
                actual,
            } => Some((*expected, *actual)),
            _ => None,
        }
    }

    /// Expected and actual output indices when readback/output correlation changed.
    pub const fn output_correlation(&self) -> Option<(usize, usize)> {
        match &self.inner {
            RenderDeterministicVerifiedSubmissionError::OutputCorrelationChanged {
                expected_output_index,
                actual_output_index,
            } => Some((*expected_output_index, *actual_output_index)),
            _ => None,
        }
    }

    /// Output index for channel-scoped exact-submission correlation failures.
    pub const fn correlation_output_index(&self) -> Option<usize> {
        match &self.inner {
            RenderDeterministicVerifiedSubmissionError::DuplicateReadbackCorrelation {
                output_index,
                ..
            }
            | RenderDeterministicVerifiedSubmissionError::MissingSubmissionReadback {
                output_index,
                ..
            } => Some(*output_index),
            _ => None,
        }
    }

    /// Verification channel for channel-scoped exact-submission correlation failures.
    pub const fn correlation_channel(&self) -> Option<&'static str> {
        match &self.inner {
            RenderDeterministicVerifiedSubmissionError::DuplicateReadbackCorrelation {
                channel,
                ..
            }
            | RenderDeterministicVerifiedSubmissionError::MissingSubmissionReadback {
                channel,
                ..
            } => Some(*channel),
            _ => None,
        }
    }
}

fn verification_eligibility_kind(
    error: &RenderDeterministicVerificationEligibilityError,
) -> RenderVerificationEligibilityErrorKind {
    match error {
        RenderDeterministicVerificationEligibilityError::SelectedObservationMissing { .. } => {
            RenderVerificationEligibilityErrorKind::SelectedObservationMissing
        }
        RenderDeterministicVerificationEligibilityError::PerspectiveFieldOfViewUnsupported {
            ..
        } => RenderVerificationEligibilityErrorKind::PerspectiveFieldOfViewUnsupported,
        RenderDeterministicVerificationEligibilityError::SamplingSupportUnsupported { .. } => {
            RenderVerificationEligibilityErrorKind::SamplingSupportUnsupported
        }
        RenderDeterministicVerificationEligibilityError::ObservationLinearBasisUnsupported {
            ..
        } => RenderVerificationEligibilityErrorKind::ObservationLinearBasisUnsupported,
        RenderDeterministicVerificationEligibilityError::SelectedObjectStateMissing { .. } => {
            RenderVerificationEligibilityErrorKind::SelectedObjectStateMissing
        }
        RenderDeterministicVerificationEligibilityError::ObjectLocalScaleUnsupported { .. } => {
            RenderVerificationEligibilityErrorKind::ObjectLocalScaleUnsupported
        }
        RenderDeterministicVerificationEligibilityError::ObjectHandednessUnsupported { .. } => {
            RenderVerificationEligibilityErrorKind::ObjectHandednessUnsupported
        }
        RenderDeterministicVerificationEligibilityError::ObjectLinearBasisUnsupported {
            ..
        } => RenderVerificationEligibilityErrorKind::ObjectLinearBasisUnsupported,
    }
}

impl fmt::Display for RenderResultSubmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            RenderDeterministicVerifiedSubmissionError::Eligibility(error) => {
                write!(
                    formatter,
                    "render result verification eligibility failed: {error}"
                )
            }
            RenderDeterministicVerifiedSubmissionError::Execution(error) => error.fmt(formatter),
            RenderDeterministicVerifiedSubmissionError::ReadbackCardinality {
                expected,
                actual,
            } => write!(
                formatter,
                "render result verification retained {actual} readback sets for {expected} admitted outputs"
            ),
            RenderDeterministicVerifiedSubmissionError::OutputCorrelationChanged {
                expected_output_index,
                actual_output_index,
            } => write!(
                formatter,
                "render result verification output correlation changed from {expected_output_index} to {actual_output_index}"
            ),
            RenderDeterministicVerifiedSubmissionError::DuplicateReadbackCorrelation {
                output_index,
                channel,
            } => write!(
                formatter,
                "output {output_index} {channel} verification reused a readback correlation"
            ),
            RenderDeterministicVerifiedSubmissionError::MissingSubmissionReadback {
                output_index,
                channel,
            } => write!(
                formatter,
                "output {output_index} {channel} verification readback is not owned by the exact RunenGPU submission"
            ),
        }
    }
}

impl Error for RenderResultSubmissionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.inner)
    }
}

/// High-level category while polling semantic result formation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderResultFormationErrorKind {
    VerificationNotRequested,
    ResultAlreadyFormed,
    SubmissionFailed,
    ReadbackCorrelationLost,
    ReadbackFailed,
    ResultEvidenceRejected,
    VerificationEligibility,
    ObservationNormalization,
    VerificationCorrelation,
    VerificationInconclusive,
    PhysicalMismatch,
    ToleranceMismatch,
}

/// Failure while forming semantic provenance from a result-capable submission.
#[derive(Debug)]
pub struct RenderResultFormationError {
    inner: RenderDeterministicResultFormationError,
}

impl RenderResultFormationError {
    /// Owner-oriented failure category.
    pub fn kind(&self) -> RenderResultFormationErrorKind {
        match &self.inner {
            RenderDeterministicResultFormationError::VerificationNotRequested => {
                RenderResultFormationErrorKind::VerificationNotRequested
            }
            RenderDeterministicResultFormationError::ResultAlreadyFormed => {
                RenderResultFormationErrorKind::ResultAlreadyFormed
            }
            RenderDeterministicResultFormationError::SubmissionFailed { .. } => {
                RenderResultFormationErrorKind::SubmissionFailed
            }
            RenderDeterministicResultFormationError::ReadbackCorrelationLost { .. } => {
                RenderResultFormationErrorKind::ReadbackCorrelationLost
            }
            RenderDeterministicResultFormationError::ReadbackFailed { .. } => {
                RenderResultFormationErrorKind::ReadbackFailed
            }
            RenderDeterministicResultFormationError::Verification(error) => match error {
                RenderDeterministicVerificationError::ResultFormation(_) => {
                    RenderResultFormationErrorKind::ResultEvidenceRejected
                }
                RenderDeterministicVerificationError::Eligibility(_) => {
                    RenderResultFormationErrorKind::VerificationEligibility
                }
                RenderDeterministicVerificationError::ObservationNormalization(_) => {
                    RenderResultFormationErrorKind::ObservationNormalization
                }
                RenderDeterministicVerificationError::Correlation { .. } => {
                    RenderResultFormationErrorKind::VerificationCorrelation
                }
                RenderDeterministicVerificationError::Inconclusive { .. } => {
                    RenderResultFormationErrorKind::VerificationInconclusive
                }
                RenderDeterministicVerificationError::PhysicalMismatch { .. } => {
                    RenderResultFormationErrorKind::PhysicalMismatch
                }
                RenderDeterministicVerificationError::ToleranceMismatch { .. } => {
                    RenderResultFormationErrorKind::ToleranceMismatch
                }
            },
        }
    }

    /// Verifier-domain eligibility reason when completion-time validation fails closed.
    pub fn verification_eligibility_kind(&self) -> Option<RenderVerificationEligibilityErrorKind> {
        let RenderDeterministicResultFormationError::Verification(
            RenderDeterministicVerificationError::Eligibility(error),
        ) = &self.inner
        else {
            return None;
        };
        Some(verification_eligibility_kind(error))
    }

    /// Output index associated with readback or semantic-verification failure.
    pub const fn output_index(&self) -> Option<usize> {
        match &self.inner {
            RenderDeterministicResultFormationError::ReadbackCorrelationLost {
                output_index,
                ..
            }
            | RenderDeterministicResultFormationError::ReadbackFailed { output_index, .. } => {
                Some(*output_index)
            }
            RenderDeterministicResultFormationError::Verification(error) => error.output_index(),
            RenderDeterministicResultFormationError::VerificationNotRequested
            | RenderDeterministicResultFormationError::ResultAlreadyFormed
            | RenderDeterministicResultFormationError::SubmissionFailed { .. } => None,
        }
    }

    /// Sample index associated with semantic-verification failure when one exists.
    pub const fn sample_index(&self) -> Option<usize> {
        match &self.inner {
            RenderDeterministicResultFormationError::Verification(error) => error.sample_index(),
            _ => None,
        }
    }

    /// Physical observation channel associated with a readback/normalization failure.
    pub const fn channel(&self) -> Option<&'static str> {
        match &self.inner {
            RenderDeterministicResultFormationError::ReadbackCorrelationLost {
                channel, ..
            }
            | RenderDeterministicResultFormationError::ReadbackFailed { channel, .. } => {
                Some(*channel)
            }
            RenderDeterministicResultFormationError::Verification(error) => error.channel(),
            _ => None,
        }
    }

    /// Public RunenGPU lifecycle failure kind when submission/readback failed physically.
    pub const fn gpu_failure_kind(&self) -> Option<GpuSubmissionFailureKind> {
        match &self.inner {
            RenderDeterministicResultFormationError::SubmissionFailed { kind }
            | RenderDeterministicResultFormationError::ReadbackFailed { kind, .. } => Some(*kind),
            RenderDeterministicResultFormationError::Verification(error) => {
                error.gpu_failure_kind()
            }
            _ => None,
        }
    }
}

impl fmt::Display for RenderResultFormationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.inner {
            RenderDeterministicResultFormationError::VerificationNotRequested => formatter
                .write_str("this render was submitted without semantic-result verification"),
            RenderDeterministicResultFormationError::ResultAlreadyFormed => {
                formatter.write_str("semantic result evidence was already formed from this render")
            }
            RenderDeterministicResultFormationError::SubmissionFailed { kind } => {
                write!(
                    formatter,
                    "RunenGPU submission failed before result formation: {kind:?}"
                )
            }
            RenderDeterministicResultFormationError::ReadbackCorrelationLost {
                output_index,
                channel,
            } => write!(
                formatter,
                "output {output_index} {channel} verification readback lost exact-submission correlation"
            ),
            RenderDeterministicResultFormationError::ReadbackFailed {
                output_index,
                channel,
                kind,
            } => write!(
                formatter,
                "output {output_index} {channel} verification readback failed: {kind:?}"
            ),
            RenderDeterministicResultFormationError::Verification(error) => {
                write!(formatter, "finite-evaluation verification failed: {error}")
            }
        }
    }
}

impl Error for RenderResultFormationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.inner)
    }
}

/// Stable category for failure to mint one product-owned radiance readback correlation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderRadianceCaptureRequestErrorKind {
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

/// Failure to mint a product-owned readback correlation for one formed radiance output.
#[derive(Debug)]
pub struct RenderRadianceCaptureRequestError {
    inner: RenderDeterministicRadianceCaptureRequestError,
}

impl RenderRadianceCaptureRequestError {
    pub const fn kind(&self) -> RenderRadianceCaptureRequestErrorKind {
        match &self.inner {
            RenderDeterministicRadianceCaptureRequestError::VerificationNotFormed => {
                RenderRadianceCaptureRequestErrorKind::VerificationNotFormed
            }
            RenderDeterministicRadianceCaptureRequestError::OutputIndexOutOfRange => {
                RenderRadianceCaptureRequestErrorKind::OutputIndexOutOfRange
            }
            RenderDeterministicRadianceCaptureRequestError::OutputNotRadiance => {
                RenderRadianceCaptureRequestErrorKind::OutputNotRadiance
            }
            RenderDeterministicRadianceCaptureRequestError::OutputTopologyUnsupported => {
                RenderRadianceCaptureRequestErrorKind::OutputTopologyUnsupported
            }
            RenderDeterministicRadianceCaptureRequestError::OutputDestinationUnsupported => {
                RenderRadianceCaptureRequestErrorKind::OutputDestinationUnsupported
            }
            RenderDeterministicRadianceCaptureRequestError::DestinationNotRetained => {
                RenderRadianceCaptureRequestErrorKind::DestinationNotRetained
            }
            RenderDeterministicRadianceCaptureRequestError::DestinationNotCopySource => {
                RenderRadianceCaptureRequestErrorKind::DestinationNotCopySource
            }
            RenderDeterministicRadianceCaptureRequestError::CarrierFormatUnsupported => {
                RenderRadianceCaptureRequestErrorKind::CarrierFormatUnsupported
            }
            RenderDeterministicRadianceCaptureRequestError::SourceUnavailable => {
                RenderRadianceCaptureRequestErrorKind::SourceUnavailable
            }
            RenderDeterministicRadianceCaptureRequestError::ReadbackIdAllocationExhausted => {
                RenderRadianceCaptureRequestErrorKind::ReadbackIdAllocationExhausted
            }
        }
    }
}

impl fmt::Display for RenderRadianceCaptureRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(formatter)
    }
}

impl Error for RenderRadianceCaptureRequestError {}

/// Stable owner-oriented category for radiance readback correlation/interpretation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderRadianceCaptureErrorKind {
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
    ProductSubmissionFailed,
    ReadbackCorrelationMissing,
    ReadbackSourceMismatch,
    ReadbackPending,
    ReadbackFailed,
    ReadbackFormatMismatch,
    ReadbackLayoutMismatch,
    ReadbackByteLengthMismatch,
    NonFiniteSample,
    HostAllocation,
}

/// Failure to correlate or interpret one completed product-owned radiance readback.
#[derive(Debug)]
pub struct RenderRadianceCaptureError {
    inner: RenderDeterministicRadianceCaptureError,
}

impl RenderRadianceCaptureError {
    pub const fn kind(&self) -> RenderRadianceCaptureErrorKind {
        match &self.inner {
            RenderDeterministicRadianceCaptureError::VerificationNotFormed => {
                RenderRadianceCaptureErrorKind::VerificationNotFormed
            }
            RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch => {
                RenderRadianceCaptureErrorKind::RequestCorrelationMismatch
            }
            RenderDeterministicRadianceCaptureError::ContextAffinityMismatch => {
                RenderRadianceCaptureErrorKind::ContextAffinityMismatch
            }
            RenderDeterministicRadianceCaptureError::RetainedContinuityUnavailable => {
                RenderRadianceCaptureErrorKind::RetainedContinuityUnavailable
            }
            RenderDeterministicRadianceCaptureError::RetainedContinuityAffinityMismatch => {
                RenderRadianceCaptureErrorKind::RetainedContinuityAffinityMismatch
            }
            RenderDeterministicRadianceCaptureError::RetainedContinuityResourceMismatch => {
                RenderRadianceCaptureErrorKind::RetainedContinuityResourceMismatch
            }
            RenderDeterministicRadianceCaptureError::RetainedContinuityNotEstablished => {
                RenderRadianceCaptureErrorKind::RetainedContinuityNotEstablished
            }
            RenderDeterministicRadianceCaptureError::RendererWriteNoLongerCurrent => {
                RenderRadianceCaptureErrorKind::RendererWriteNoLongerCurrent
            }
            RenderDeterministicRadianceCaptureError::ProductSubmissionAffinityMismatch => {
                RenderRadianceCaptureErrorKind::ProductSubmissionAffinityMismatch
            }
            RenderDeterministicRadianceCaptureError::ProductSubmissionPending => {
                RenderRadianceCaptureErrorKind::ProductSubmissionPending
            }
            RenderDeterministicRadianceCaptureError::ProductSubmissionFailed { .. } => {
                RenderRadianceCaptureErrorKind::ProductSubmissionFailed
            }
            RenderDeterministicRadianceCaptureError::ReadbackCorrelationMissing => {
                RenderRadianceCaptureErrorKind::ReadbackCorrelationMissing
            }
            RenderDeterministicRadianceCaptureError::ReadbackSourceMismatch => {
                RenderRadianceCaptureErrorKind::ReadbackSourceMismatch
            }
            RenderDeterministicRadianceCaptureError::ReadbackPending => {
                RenderRadianceCaptureErrorKind::ReadbackPending
            }
            RenderDeterministicRadianceCaptureError::ReadbackFailed { .. } => {
                RenderRadianceCaptureErrorKind::ReadbackFailed
            }
            RenderDeterministicRadianceCaptureError::ReadbackFormatMismatch => {
                RenderRadianceCaptureErrorKind::ReadbackFormatMismatch
            }
            RenderDeterministicRadianceCaptureError::ReadbackLayoutMismatch => {
                RenderRadianceCaptureErrorKind::ReadbackLayoutMismatch
            }
            RenderDeterministicRadianceCaptureError::ReadbackByteLengthMismatch => {
                RenderRadianceCaptureErrorKind::ReadbackByteLengthMismatch
            }
            RenderDeterministicRadianceCaptureError::NonFiniteSample => {
                RenderRadianceCaptureErrorKind::NonFiniteSample
            }
            RenderDeterministicRadianceCaptureError::HostAllocation => {
                RenderRadianceCaptureErrorKind::HostAllocation
            }
        }
    }

    /// RunenGPU lifecycle failure when the product submission or readback itself failed.
    pub const fn gpu_failure_kind(&self) -> Option<GpuSubmissionFailureKind> {
        match &self.inner {
            RenderDeterministicRadianceCaptureError::ProductSubmissionFailed { kind }
            | RenderDeterministicRadianceCaptureError::ReadbackFailed { kind } => Some(*kind),
            _ => None,
        }
    }
}

impl fmt::Display for RenderRadianceCaptureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(formatter)
    }
}

impl Error for RenderRadianceCaptureError {}

/// Stable renderer-owned namespace for retained execution state.
///
/// This scopes reusable renderer resources and temporal history. It is not a scene, object,
/// RunenGPU resource, or submission identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderExecutionScope(u64);

impl RenderExecutionScope {
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }
}

/// Optional finite evaluation selection for one requested output.
///
/// This changes bounded physical work only; it does not change the semantic request topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderEvaluationSelection {
    output_index: usize,
    extent: (u32, u32),
}

impl RenderEvaluationSelection {
    pub fn new(output_index: usize, width: u32, height: u32) -> Option<Self> {
        (width > 0 && height > 0).then_some(Self {
            output_index,
            extent: (width, height),
        })
    }

    pub const fn output_index(self) -> usize {
        self.output_index
    }

    pub const fn extent(self) -> (u32, u32) {
        self.extent
    }
}

/// Public renderer-semantic temporal execution evidence.
///
/// This deliberately excludes renderer-private requested-lattice coverage preparation. Coverage is
/// an internal reconstruction prerequisite, not semantic depth/output authority and not part of the
/// transferable ordinary API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderTemporalExecutionEvidence {
    pub requested_extent: (u32, u32),
    pub evaluation_extent: (u32, u32),
    pub semantic_input_generations:
        Vec<(RenderRepresentationId, RenderSurfaceSemanticInputGeneration)>,
    pub field_semantic_input_generations:
        Vec<(RenderRepresentationId, RenderFieldSemanticInputGeneration)>,
    pub sequence_revision: u32,
    pub reconstruction_revision: u32,
    pub phase: u32,
    pub history_generation: u64,
    pub history_age: u32,
    pub history_reset: bool,
    pub camera_reprojection_eligible: bool,
    pub previous_observation_available: bool,
    pub camera_pose_changed: bool,
    pub camera_same_pose_completed_frames: Option<u32>,
    pub camera_reprojection_revision: Option<u32>,
    pub depth_policy_revision: Option<u32>,
}

impl RenderTemporalExecutionEvidence {
    fn from_deterministic(evidence: &DeterministicTemporalExecutionEvidence) -> Self {
        Self {
            requested_extent: evidence.requested_extent,
            evaluation_extent: evidence.evaluation_extent,
            semantic_input_generations: evidence.semantic_input_generations.clone(),
            field_semantic_input_generations: evidence.field_semantic_input_generations.clone(),
            sequence_revision: evidence.sequence_revision,
            reconstruction_revision: evidence.reconstruction_revision,
            phase: evidence.phase,
            history_generation: evidence.history_generation,
            history_age: evidence.history_age,
            history_reset: evidence.history_reset,
            camera_reprojection_eligible: evidence.camera_reprojection_eligible,
            previous_observation_available: evidence.previous_observation_available,
            camera_pose_changed: evidence.camera_pose_changed,
            camera_same_pose_completed_frames: evidence.camera_same_pose_completed_frames,
            camera_reprojection_revision: evidence.camera_reprojection_revision,
            depth_policy_revision: evidence.depth_policy_revision,
        }
    }
}

/// Stateful ordinary integration for hosts that compose renderer-authored work into a larger
/// public RunenGPU submission.
///
/// It retains only renderer-derived reusable resources and temporal history. Planning, admission,
/// compatibility, and lowering are the same authority used by the one-shot ordinary path.
#[derive(Debug, Default)]
pub struct RenderExecutionState {
    inner: DeterministicResourceCache,
}

impl RenderExecutionState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn has_in_flight_scopes(
        &self,
        scopes: impl IntoIterator<Item = RenderExecutionScope>,
    ) -> bool {
        self.inner
            .any_producer_submission_in_flight(scopes.into_iter().map(RenderExecutionScope::raw))
    }

    pub fn retain_in_flight_submissions(&mut self) {
        self.inner.retain_in_flight_submissions();
    }

    pub fn record_submission(&mut self, scope: RenderExecutionScope, submission: &GpuSubmission) {
        self.inner
            .record_producer_submission(scope.raw(), 0, submission);
    }

    pub fn prepare(
        &mut self,
        admitted: AdmittedRender,
        context: &GpuContext,
        scope: RenderExecutionScope,
        evaluation: Option<RenderEvaluationSelection>,
    ) -> Result<PreparedRender, RenderExecutionError> {
        let finite_evaluation =
            evaluation.map(|selection| (selection.output_index(), selection.extent()));
        prepare_deterministic_render_with_cache_in_scope_and_evaluation(
            admitted.inner,
            context,
            &mut self.inner,
            scope.raw(),
            finite_evaluation,
            false,
        )
        .map(|inner| PreparedRender { inner })
        .map_err(|inner| RenderExecutionError { inner })
    }
}

/// Maintained invocation after semantic planning, binding, and execution admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedRender {
    inner: AdmittedDeterministicRender,
}

impl AdmittedRender {
    /// Exact admitted semantic plan retained by this invocation.
    pub const fn admitted_plan(&self) -> &AdmittedRenderPlan {
        self.inner.admitted()
    }
}

/// Renderer-authored public RunenGPU work prepared from one admitted invocation.
#[derive(Debug, Clone)]
pub struct PreparedRender {
    inner: PreparedDeterministicRender,
}

impl PreparedRender {
    /// Exact admitted semantic plan from which this work was lowered.
    pub const fn admitted_plan(&self) -> &AdmittedRenderPlan {
        self.inner.admitted().admitted()
    }

    /// Backend-neutral public RunenGPU work authored by the renderer.
    pub const fn work_set(&self) -> &RenderWorkSet {
        self.inner.work_set()
    }

    /// Prepared radiance outputs available for product composition.
    pub fn radiance_outputs(
        &self,
    ) -> impl ExactSizeIterator<Item = PreparedRadianceOutput<'_>> + '_ {
        self.inner
            .radiance_outputs()
            .iter()
            .map(|inner| PreparedRadianceOutput { inner })
    }

    /// Prepared radiance output for one requested output index, when applicable.
    pub fn radiance_output(&self, output_index: usize) -> Option<PreparedRadianceOutput<'_>> {
        self.inner
            .radiance_output(output_index)
            .map(|inner| PreparedRadianceOutput { inner })
    }
}

/// Borrowed correlation for one prepared radiance destination and its public RunenGPU export.
#[derive(Debug, Clone, Copy)]
pub struct PreparedRadianceOutput<'a> {
    inner: &'a PreparedDeterministicRadianceOutput,
}

impl PreparedRadianceOutput<'_> {
    /// Requested output index correlated to this prepared destination.
    pub const fn output_index(&self) -> usize {
        self.inner.output_index()
    }

    /// Public RunenGPU resource receiving the renderer output.
    pub fn resource(&self) -> &GpuResourceRef {
        self.inner.resource()
    }

    /// Texture handle when this destination is texture-backed.
    pub fn texture(&self) -> Option<&GpuTextureHandle> {
        self.inner.texture()
    }

    /// Exact producer/consumer relationship authored for composition.
    pub fn export_relationship(&self) -> &GpuExportRelationship {
        self.inner.export_relationship()
    }

    /// Renderer-semantic temporal execution evidence for this prepared output, when present.
    pub fn temporal_execution_evidence(&self) -> Option<RenderTemporalExecutionEvidence> {
        self.inner
            .temporal_execution_evidence()
            .map(RenderTemporalExecutionEvidence::from_deterministic)
    }

    /// Form a public RunenGPU import of this renderer-authored output.
    pub fn import(&self, provenance: GpuResourceProvenance) -> GpuWorkImport {
        self.inner.import(provenance)
    }
}

/// One-shot correlation for a product-owned public RunenGPU radiance readback.
pub struct RenderRadianceCaptureRequest {
    inner: RenderDeterministicRadianceCaptureRequest,
}

impl RenderRadianceCaptureRequest {
    /// Requested output index correlated to this capture.
    pub const fn output_index(&self) -> usize {
        self.inner.output_index()
    }

    /// Exact public RunenGPU transfer source the product should read.
    pub fn source(&self) -> &GpuTransferRegion {
        self.inner.source()
    }

    /// Fresh readback correlation identity for the product submission.
    pub const fn readback_id(&self) -> GpuReadbackId {
        self.inner.readback_id()
    }
}

/// Finite maintained radiance samples interpreted from a product-owned readback.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderCapturedRadiance {
    inner: RenderCapturedDeterministicRadiance,
}

impl RenderCapturedRadiance {
    /// Requested output index whose retained destination was observed.
    pub const fn output_index(&self) -> usize {
        self.inner.output_index()
    }

    /// Semantic sample topology of the captured values.
    pub const fn topology(&self) -> RenderResultTopology {
        self.inner.topology()
    }

    /// Semantic radiometric representation of the captured values.
    pub const fn representation(&self) -> RenderRadiometricRepresentation {
        self.inner.representation()
    }

    /// Row-major finite maintained radiance samples.
    pub fn samples(&self) -> &[f32] {
        self.inner.samples()
    }
}

/// One ordinary maintained render already submitted to RunenGPU.
///
/// This state exposes submission and provenance inspection only. Ordinary submission deliberately
/// authors no semantic-result verification readbacks.
#[derive(Debug)]
pub struct SubmittedRender {
    inner: SubmittedDeterministicRender,
}

impl SubmittedRender {
    /// Exact semantic admission that produced this submission.
    pub const fn admitted_plan(&self) -> &AdmittedRenderPlan {
        self.inner.admitted().admitted()
    }

    /// Current public RunenGPU lifecycle status.
    pub fn submission_status(&self) -> GpuSubmissionStatus {
        self.inner.submission_status()
    }

    /// Decoder for execution-local object-identity carrier values, when requested.
    pub const fn object_identity_decoder(&self) -> &RenderObjectIdentityDecoder {
        self.inner.object_identity_decoder()
    }
}

/// One maintained render submitted with semantic-result verification enabled.
///
/// Callers poll result formation after progressing the public RunenGPU context. Product readback
/// remains a separate optional submission even after semantic result formation succeeds.
#[derive(Debug)]
pub struct SubmittedRenderForResult {
    inner: SubmittedDeterministicRender,
}

impl SubmittedRenderForResult {
    /// Exact semantic admission that produced this submission.
    pub const fn admitted_plan(&self) -> &AdmittedRenderPlan {
        self.inner.admitted().admitted()
    }

    /// Current public RunenGPU lifecycle status.
    pub fn submission_status(&self) -> GpuSubmissionStatus {
        self.inner.submission_status()
    }

    /// Decoder for execution-local object-identity carrier values, when requested.
    pub const fn object_identity_decoder(&self) -> &RenderObjectIdentityDecoder {
        self.inner.object_identity_decoder()
    }

    /// Poll semantic result formation without blocking or driving RunenGPU progress.
    pub fn try_form_result(&mut self) -> Result<Option<RenderResult>, RenderResultFormationError> {
        self.inner
            .try_form_verified_result()
            .map_err(|inner| RenderResultFormationError { inner })
    }

    /// Mint one fresh product-owned readback correlation for a formed radiance output.
    pub fn request_radiance_capture(
        &self,
        output_index: usize,
    ) -> Result<RenderRadianceCaptureRequest, RenderRadianceCaptureRequestError> {
        self.inner
            .request_deterministic_radiance_capture(output_index)
            .map(|inner| RenderRadianceCaptureRequest { inner })
            .map_err(|inner| RenderRadianceCaptureRequestError { inner })
    }

    /// Interpret one completed product-owned RunenGPU readback through the maintained carrier.
    pub fn capture_radiance(
        &self,
        request: RenderRadianceCaptureRequest,
        context: &GpuContext,
        product_submission: &GpuSubmission,
    ) -> Result<RenderCapturedRadiance, RenderRadianceCaptureError> {
        self.inner
            .capture_deterministic_radiance(request.inner, context, product_submission)
            .map(|inner| RenderCapturedRadiance { inner })
            .map_err(|inner| RenderRadianceCaptureError { inner })
    }
}

/// Plan and admit one ordinary invocation of the maintained renderer.
///
/// Callers provide semantic scene/request/input facts, physical output destinations, and a public
/// RunenGPU context. Method selection, planning, binding admission, and maintained compatibility
/// remain inside RunenRender.
pub fn admit_render(
    scene: &RenderSceneSnapshot,
    request: &RenderRequest,
    surface_inputs: &[RenderSurfaceSemanticInputBinding],
    field_inputs: &[RenderFieldSemanticInputBinding],
    availability: &[RenderRepresentationAvailabilityFact],
    output_bindings: &[RenderOutputBinding],
    context: &GpuContext,
) -> Result<AdmittedRender, RenderAdmissionError> {
    admit_deterministic_render_with_semantic_inputs(
        scene,
        request,
        surface_inputs,
        field_inputs,
        availability,
        output_bindings,
        context,
    )
    .map(|inner| AdmittedRender { inner })
    .map_err(|inner| RenderAdmissionError { inner })
}

/// Lower one admitted ordinary render into composable public RunenGPU work without submitting it.
///
/// No CPU readback is authored by this path.
pub fn prepare_render(
    admitted: AdmittedRender,
    context: &GpuContext,
) -> Result<PreparedRender, RenderExecutionError> {
    prepare_deterministic_render(admitted.inner, context)
        .map(|inner| PreparedRender { inner })
        .map_err(|inner| RenderExecutionError { inner })
}

/// Lower and submit one ordinary maintained render through public RunenGPU.
///
/// No semantic-result verification or CPU readback is requested.
pub async fn submit_render(
    admitted: AdmittedRender,
    context: &GpuContext,
) -> Result<SubmittedRender, RenderExecutionError> {
    submit_deterministic_render(admitted.inner, context)
        .await
        .map(|inner| SubmittedRender { inner })
        .map_err(|inner| RenderExecutionError { inner })
}

/// Lower and submit one maintained render with private semantic-result verification enabled.
///
/// The returned type is distinct from SubmittedRender so result formation cannot be requested
/// accidentally from an ordinary readback-free submission.
pub async fn submit_render_for_result(
    admitted: AdmittedRender,
    context: &GpuContext,
) -> Result<SubmittedRenderForResult, RenderResultSubmissionError> {
    submit_deterministic_render_for_verified_result(admitted.inner, context)
        .await
        .map(|inner| SubmittedRenderForResult { inner })
        .map_err(|inner| RenderResultSubmissionError { inner })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_error_preserves_runenshader_owner_category() {
        let error = RenderExecutionError {
            inner: RenderDeterministicExecutionError::Lowering(
                crate::runtime::execution::RenderDeterministicLoweringError::RunenShaderCompilation(
                    crate::runtime::program::RenderRunenShaderCompilationError::CanonicalBytesChanged {
                        program: "test program",
                    },
                ),
            ),
        };
        assert_eq!(
            error.kind(),
            RenderExecutionErrorKind::RunenShaderCompilation
        );
        assert!(error.runen_shader_compilation_source().is_some());
        assert!(error.runen_gpu_preparation_source().is_none());
        assert!(Error::source(&error).is_some());
    }

    #[test]
    fn result_submission_preserves_structured_eligibility_and_correlation() {
        let eligibility = RenderResultSubmissionError {
            inner: RenderDeterministicVerifiedSubmissionError::Eligibility(
                RenderDeterministicVerificationEligibilityError::SamplingSupportUnsupported {
                    observation_index: 3,
                },
            ),
        };
        assert_eq!(
            eligibility.kind(),
            RenderResultSubmissionErrorKind::Eligibility
        );
        assert_eq!(
            eligibility.verification_eligibility_kind(),
            Some(RenderVerificationEligibilityErrorKind::SamplingSupportUnsupported)
        );
        assert_eq!(eligibility.observation_index(), Some(3));
        assert!(Error::source(&eligibility).is_some());

        let correlation = RenderResultSubmissionError {
            inner: RenderDeterministicVerifiedSubmissionError::MissingSubmissionReadback {
                output_index: 2,
                channel: "canonical-output",
            },
        };
        assert_eq!(
            correlation.kind(),
            RenderResultSubmissionErrorKind::Correlation
        );
        assert_eq!(correlation.correlation_output_index(), Some(2));
        assert_eq!(correlation.correlation_channel(), Some("canonical-output"));
    }

    #[test]
    fn result_formation_preserves_semantic_verification_location() {
        let correlation = RenderResultFormationError {
            inner: RenderDeterministicResultFormationError::Verification(
                RenderDeterministicVerificationError::Correlation {
                    output_index: 2,
                    sample_index: Some(7),
                    detail: "semantic sample correlation changed",
                },
            ),
        };
        assert_eq!(
            correlation.kind(),
            RenderResultFormationErrorKind::VerificationCorrelation
        );
        assert_eq!(correlation.output_index(), Some(2));
        assert_eq!(correlation.sample_index(), Some(7));
        assert!(Error::source(&correlation).is_some());

        let tolerance = RenderResultFormationError {
            inner: RenderDeterministicResultFormationError::Verification(
                RenderDeterministicVerificationError::ToleranceMismatch {
                    output_index: 1,
                    sample_index: 5,
                },
            ),
        };
        assert_eq!(
            tolerance.kind(),
            RenderResultFormationErrorKind::ToleranceMismatch
        );
        assert_eq!(tolerance.output_index(), Some(1));
        assert_eq!(tolerance.sample_index(), Some(5));

        let inconclusive = RenderResultFormationError {
            inner: RenderDeterministicResultFormationError::Verification(
                RenderDeterministicVerificationError::Inconclusive {
                    output_index: 3,
                    sample_index: None,
                    detail: "conservative interval did not select one semantic branch",
                },
            ),
        };
        assert_eq!(
            inconclusive.kind(),
            RenderResultFormationErrorKind::VerificationInconclusive
        );
        assert_eq!(inconclusive.output_index(), Some(3));
        assert_eq!(inconclusive.sample_index(), None);

        let physical = RenderResultFormationError {
            inner: RenderDeterministicResultFormationError::Verification(
                RenderDeterministicVerificationError::PhysicalMismatch {
                    output_index: 4,
                    sample_index: Some(2),
                    detail: "physical observation contradicted the semantic result",
                },
            ),
        };
        assert_eq!(
            physical.kind(),
            RenderResultFormationErrorKind::PhysicalMismatch
        );
        assert_eq!(physical.output_index(), Some(4));
        assert_eq!(physical.sample_index(), Some(2));
    }
}
