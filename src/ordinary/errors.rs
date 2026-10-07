use super::*;

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
    pub(super) inner: RenderDeterministicAdmissionFailure,
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
    pub(super) inner: RenderDeterministicExecutionError,
}

impl RenderExecutionError {
    /// Owner-oriented failure category.
    pub fn kind(&self) -> RenderExecutionErrorKind {
        match &self.inner {
            RenderDeterministicExecutionError::Lowering(
                crate::runtime::execution::RenderDeterministicLoweringError::RunenShaderCompilation(
                    _,
                ),
            ) => RenderExecutionErrorKind::RunenShaderCompilation,
            RenderDeterministicExecutionError::Lowering(
                crate::runtime::execution::RenderDeterministicLoweringError::RunenGpuPreparation(_),
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
            crate::runtime::execution::RenderDeterministicLoweringError::RunenShaderCompilation(
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
            crate::runtime::execution::RenderDeterministicLoweringError::RunenGpuPreparation(error),
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

/// Retained ordinary execution lifecycle failure.
#[derive(Debug)]
pub enum RenderExecutionSessionError {
    /// A previously prepared occurrence is still alive and must be associated or dropped first.
    PreparedOccurrenceOutstanding,
    /// The exact associated submission for this continuity has not terminalized.
    SubmissionInFlight,
    /// The supplied occurrence was not prepared by this session or is no longer current.
    OccurrenceNotCurrent,
    /// The supplied submission belongs to a different RunenGPU context/device generation.
    SubmissionAffinityMismatch {
        expected: GpuContextAffinity,
        actual: GpuContextAffinity,
    },
    /// The supplied submission does not contain every renderer-authored node from the occurrence.
    SubmissionMissingRendererWork,
    /// Maintained lowering/preparation failed before an occurrence could be established.
    Execution(RenderExecutionError),
}

impl fmt::Display for RenderExecutionSessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PreparedOccurrenceOutstanding => formatter.write_str(
                "this retained render session still owns a live prepared occurrence",
            ),
            Self::SubmissionInFlight => formatter.write_str(
                "this retained render session still has an associated RunenGPU submission in flight",
            ),
            Self::OccurrenceNotCurrent => formatter.write_str(
                "prepared render occurrence does not belong to the current retained session state",
            ),
            Self::SubmissionAffinityMismatch { expected, actual } => write!(
                formatter,
                "prepared render occurrence requires RunenGPU affinity {expected:?}, got {actual:?}"
            ),
            Self::SubmissionMissingRendererWork => formatter.write_str(
                "RunenGPU submission does not contain every node from the exact prepared render occurrence",
            ),
            Self::Execution(error) => error.fmt(formatter),
        }
    }
}

impl Error for RenderExecutionSessionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Execution(error) => Some(error),
            Self::PreparedOccurrenceOutstanding
            | Self::SubmissionInFlight
            | Self::OccurrenceNotCurrent
            | Self::SubmissionAffinityMismatch { .. }
            | Self::SubmissionMissingRendererWork => None,
        }
    }
}

impl From<RenderExecutionError> for RenderExecutionSessionError {
    fn from(error: RenderExecutionError) -> Self {
        Self::Execution(error)
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
    pub(super) inner: RenderDeterministicVerifiedSubmissionError,
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
    pub(super) inner: RenderDeterministicResultFormationError,
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
    RendererSubmissionPending,
    RendererSubmissionFailed,
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
    pub(super) inner: RenderDeterministicRadianceCaptureRequestError,
}

impl RenderRadianceCaptureRequestError {
    pub const fn kind(&self) -> RenderRadianceCaptureRequestErrorKind {
        match &self.inner {
            RenderDeterministicRadianceCaptureRequestError::VerificationNotFormed => {
                RenderRadianceCaptureRequestErrorKind::VerificationNotFormed
            }
            RenderDeterministicRadianceCaptureRequestError::RendererSubmissionPending => {
                RenderRadianceCaptureRequestErrorKind::RendererSubmissionPending
            }
            RenderDeterministicRadianceCaptureRequestError::RendererSubmissionFailed { .. } => {
                RenderRadianceCaptureRequestErrorKind::RendererSubmissionFailed
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

    /// RunenGPU lifecycle failure when the associated renderer submission failed.
    pub const fn gpu_failure_kind(&self) -> Option<GpuSubmissionFailureKind> {
        match &self.inner {
            RenderDeterministicRadianceCaptureRequestError::RendererSubmissionFailed { kind } => {
                Some(*kind)
            }
            _ => None,
        }
    }
}

impl fmt::Display for RenderRadianceCaptureRequestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.inner.fmt(formatter)
    }
}

impl Error for RenderRadianceCaptureRequestError {}

/// Stable owner-oriented category for retained object-identity decoder lookup failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderObjectIdentityDecoderErrorKind {
    RendererSubmissionPending,
    RendererSubmissionFailed,
    OutputIndexOutOfRange,
    OutputNotObjectIdentity,
}

/// Failure to obtain an execution-local object-identity decoder from one associated occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderObjectIdentityDecoderError {
    kind: RenderObjectIdentityDecoderErrorKind,
    gpu_failure_kind: Option<GpuSubmissionFailureKind>,
}

impl RenderObjectIdentityDecoderError {
    pub(super) const fn submission_pending() -> Self {
        Self {
            kind: RenderObjectIdentityDecoderErrorKind::RendererSubmissionPending,
            gpu_failure_kind: None,
        }
    }

    pub(super) const fn submission_failed(kind: GpuSubmissionFailureKind) -> Self {
        Self {
            kind: RenderObjectIdentityDecoderErrorKind::RendererSubmissionFailed,
            gpu_failure_kind: Some(kind),
        }
    }

    pub(super) const fn output_index_out_of_range() -> Self {
        Self {
            kind: RenderObjectIdentityDecoderErrorKind::OutputIndexOutOfRange,
            gpu_failure_kind: None,
        }
    }

    pub(super) const fn output_not_object_identity() -> Self {
        Self {
            kind: RenderObjectIdentityDecoderErrorKind::OutputNotObjectIdentity,
            gpu_failure_kind: None,
        }
    }

    pub const fn kind(&self) -> RenderObjectIdentityDecoderErrorKind {
        self.kind
    }

    pub const fn gpu_failure_kind(&self) -> Option<GpuSubmissionFailureKind> {
        self.gpu_failure_kind
    }
}

impl fmt::Display for RenderObjectIdentityDecoderError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let detail = match self.kind {
            RenderObjectIdentityDecoderErrorKind::RendererSubmissionPending => {
                "the associated renderer submission has not completed"
            }
            RenderObjectIdentityDecoderErrorKind::RendererSubmissionFailed => {
                "the associated renderer submission failed"
            }
            RenderObjectIdentityDecoderErrorKind::OutputIndexOutOfRange => {
                "requested object-identity output index is not admitted"
            }
            RenderObjectIdentityDecoderErrorKind::OutputNotObjectIdentity => {
                "requested output is not object identity"
            }
        };
        formatter.write_str(detail)
    }
}

impl Error for RenderObjectIdentityDecoderError {}

/// Stable owner-oriented category for radiance readback correlation/interpretation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderRadianceCaptureErrorKind {
    VerificationNotFormed,
    RendererSubmissionPending,
    RendererSubmissionFailed,
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
    pub(super) inner: RenderDeterministicRadianceCaptureError,
}

impl RenderRadianceCaptureError {
    pub const fn kind(&self) -> RenderRadianceCaptureErrorKind {
        match &self.inner {
            RenderDeterministicRadianceCaptureError::VerificationNotFormed => {
                RenderRadianceCaptureErrorKind::VerificationNotFormed
            }
            RenderDeterministicRadianceCaptureError::RendererSubmissionPending => {
                RenderRadianceCaptureErrorKind::RendererSubmissionPending
            }
            RenderDeterministicRadianceCaptureError::RendererSubmissionFailed { .. } => {
                RenderRadianceCaptureErrorKind::RendererSubmissionFailed
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
            RenderDeterministicRadianceCaptureError::RendererSubmissionFailed { kind }
            | RenderDeterministicRadianceCaptureError::ProductSubmissionFailed { kind }
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
