use super::*;

/// Owner-oriented classification of an ordinary admission failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderAdmissionErrorKind {
    InvalidInvocation,
    UnsupportedSemantics,
    NoExecutableCandidate,
    OutputTargetMismatch,
    MaintainedRealizationUnsupported,
}

/// Actionable reason for one rejected candidate. A rejection set can contain different reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderCandidateFailureKind {
    UnsupportedSemantics,
    RepresentationUnavailable,
    CapabilityUnsupported,
    CapabilityNotEnabled,
    ExecutionLifecycle,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderCandidateFailure {
    pub candidate_index: usize,
    pub kind: RenderCandidateFailureKind,
    pub output: Option<crate::request::RenderOutputHandle>,
}

/// Map a single candidate's rejection without flattening mixed-candidate outcomes.
pub(super) fn classify_candidate_rejection(
    reason: &crate::admission::RenderCandidateAdmissionRejectionReason,
) -> (RenderCandidateFailureKind, Option<usize>) {
    use crate::admission::RenderCandidateAdmissionRejectionReason as Reason;
    match reason {
        Reason::ExecutionLifecycle { .. } => (RenderCandidateFailureKind::ExecutionLifecycle, None),
        Reason::RequiredCapabilityUnsupported { .. } => {
            (RenderCandidateFailureKind::CapabilityUnsupported, None)
        }
        Reason::RequiredCapabilityNotEnabled { .. } => {
            (RenderCandidateFailureKind::CapabilityNotEnabled, None)
        }
        Reason::NoSemanticallyAdmissibleRepresentation { output_index, .. } => (
            RenderCandidateFailureKind::UnsupportedSemantics,
            Some(*output_index),
        ),
        Reason::AvailabilityUnknown { output_index, .. }
        | Reason::NoAvailableRepresentation { output_index, .. } => (
            RenderCandidateFailureKind::RepresentationUnavailable,
            Some(*output_index),
        ),
    }
}

/// Failure while planning, semantically admitting, or checking maintained-method compatibility.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderAdmissionError {
    pub(super) inner: RenderDeterministicAdmissionFailure,
    pub(super) request: RenderRequest,
}

impl RenderAdmissionError {
    pub const fn request(&self) -> &RenderRequest {
        &self.request
    }

    pub fn kind(&self) -> RenderAdmissionErrorKind {
        use crate::admission::{
            RenderAdmissionInputError as Input, RenderExecutionAdmissionFailure as Admission,
        };
        use crate::runtime::admission::{
            RenderDeterministicAdmissionFailure as Failure,
            RenderDeterministicCompatibilityError as Compatibility,
        };
        use crate::semantic_plan::RenderPlanningFailure as Planning;
        match &self.inner {
            Failure::Planning(Planning::NoSemanticSolution { .. }) => {
                RenderAdmissionErrorKind::UnsupportedSemantics
            }
            Failure::Planning(_) => RenderAdmissionErrorKind::InvalidInvocation,
            Failure::Admission(Admission::NoExecutableCandidate { .. }) => {
                RenderAdmissionErrorKind::NoExecutableCandidate
            }
            Failure::Admission(Admission::InvalidInput(input)) => match input {
                Input::OutputDestinationKind { .. }
                | Input::ScalarBufferNotWritable { .. }
                | Input::LatticeTextureSurfaceAcquired { .. }
                | Input::LatticeTextureDimension { .. }
                | Input::LatticeTextureExtent { .. }
                | Input::LatticeTextureSampleCount { .. }
                | Input::LatticeTextureNotWritable { .. } => {
                    RenderAdmissionErrorKind::OutputTargetMismatch
                }
                _ => RenderAdmissionErrorKind::InvalidInvocation,
            },
            Failure::Compatibility(compatibility) => match compatibility {
                Compatibility::LatticeFormatUnsupported { .. }
                | Compatibility::LatticeCopyDestinationUnsupported { .. }
                | Compatibility::ScalarDestinationSize { .. }
                | Compatibility::ScalarDestinationNotCopyDestination { .. }
                | Compatibility::LatticeDestinationFormat { .. }
                | Compatibility::LatticeDestinationNotCopyDestination { .. } => {
                    RenderAdmissionErrorKind::OutputTargetMismatch
                }
                Compatibility::ObservationShutterNotInstant { .. }
                | Compatibility::ObservationSamplingSupportUnsupported { .. } => {
                    RenderAdmissionErrorKind::UnsupportedSemantics
                }
                _ => RenderAdmissionErrorKind::MaintainedRealizationUnsupported,
            },
        }
    }

    /// Exact request-owned output implicated by an ordinary admission error, when singular.
    /// Candidate-set failures use `candidate_failures` instead.
    pub fn output(&self) -> Option<crate::request::RenderOutputHandle> {
        use crate::admission::{
            RenderAdmissionInputError as Input, RenderExecutionAdmissionFailure as Admission,
        };
        use crate::runtime::admission::{
            RenderDeterministicAdmissionFailure as Failure,
            RenderDeterministicCompatibilityError as Compatibility,
        };
        let position = match &self.inner {
            Failure::Admission(Admission::InvalidInput(input)) => match input {
                Input::DuplicateOutputBinding { output_index }
                | Input::OutputBindingOutOfRange { output_index, .. }
                | Input::MissingOutputBinding { output_index }
                | Input::OutputDestinationKind { output_index }
                | Input::ScalarBufferNotWritable { output_index }
                | Input::LatticeTextureSurfaceAcquired { output_index }
                | Input::LatticeTextureDimension { output_index, .. }
                | Input::LatticeTextureExtent { output_index, .. }
                | Input::LatticeTextureSampleCount { output_index, .. }
                | Input::LatticeTextureNotWritable { output_index } => Some(*output_index),
                Input::ForeignOutputBinding { .. }
                | Input::SemanticBinding(_)
                | Input::DuplicateAvailabilityFact { .. } => None,
            },
            Failure::Compatibility(compatibility) => match compatibility {
                Compatibility::SelectedRepresentationSurfaceInputUnsupported {
                    output_index,
                    ..
                }
                | Compatibility::SelectedRepresentationFieldInputUnsupported {
                    output_index, ..
                }
                | Compatibility::SelectedObjectFieldTransformNotSimilarity {
                    output_index, ..
                }
                | Compatibility::SelectedObjectStateMissing { output_index, .. }
                | Compatibility::SelectedObjectTransformNonInvertible { output_index, .. }
                | Compatibility::LatticeFormatUnsupported { output_index, .. }
                | Compatibility::LatticeCopyDestinationUnsupported { output_index, .. }
                | Compatibility::ScalarDestinationSize { output_index, .. }
                | Compatibility::ScalarDestinationNotCopyDestination { output_index }
                | Compatibility::LatticeDestinationFormat { output_index, .. }
                | Compatibility::LatticeDestinationNotCopyDestination { output_index } => {
                    Some(*output_index)
                }
                Compatibility::ObservationShutterNotInstant { .. }
                | Compatibility::ObservationSamplingSupportUnsupported { .. } => None,
            },
            _ => None,
        };
        position.and_then(|index| self.request.output_handle(index))
    }

    /// Exact request-owned observation implicated by a maintained compatibility failure.
    pub fn observation(&self) -> Option<crate::request::RenderObservationHandle> {
        use crate::runtime::admission::{
            RenderDeterministicAdmissionFailure as Failure,
            RenderDeterministicCompatibilityError as Compatibility,
        };
        let position = match &self.inner {
            Failure::Compatibility(
                Compatibility::ObservationShutterNotInstant { observation_index }
                | Compatibility::ObservationSamplingSupportUnsupported { observation_index },
            ) => Some(*observation_index),
            _ => None,
        };
        position.and_then(|index| self.request.observation_handle(index))
    }

    /// Every rejected candidate remains inspectable; mixed reasons are never collapsed to a
    /// fictitious single availability/capability/semantic cause.
    pub fn candidate_failures(&self) -> Vec<RenderCandidateFailure> {
        use crate::admission::RenderExecutionAdmissionFailure as Admission;
        use crate::runtime::admission::RenderDeterministicAdmissionFailure as Failure;
        let Failure::Admission(Admission::NoExecutableCandidate { rejections }) = &self.inner
        else {
            return Vec::new();
        };
        rejections
            .iter()
            .map(|rejection| {
                let (kind, index) = classify_candidate_rejection(rejection.reason());
                RenderCandidateFailure {
                    candidate_index: rejection.candidate_index(),
                    kind,
                    output: index.and_then(|position| self.request.output_handle(position)),
                }
            })
            .collect()
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
    /// Finite evaluation selected an output belonging to another request lineage.
    ForeignEvaluationOutput {
        output: crate::request::RenderOutputHandle,
    },
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
            Self::ForeignEvaluationOutput { output } => write!(
                formatter,
                "finite evaluation output {} belongs to another request",
                output.position(),
            ),
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
            Self::ForeignEvaluationOutput { .. }
            | Self::PreparedOccurrenceOutstanding
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
    pub(super) request: RenderRequest,
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
    pub fn observation(&self) -> Option<crate::request::RenderObservationHandle> {
        let position = match &self.inner {
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
        };
        position.and_then(|index| self.request.observation_handle(index))
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
    pub fn output_correlation(
        &self,
    ) -> Option<(
        crate::request::RenderOutputHandle,
        Option<crate::request::RenderOutputHandle>,
    )> {
        match &self.inner {
            RenderDeterministicVerifiedSubmissionError::OutputCorrelationChanged {
                expected_output_index,
                actual_output_index,
            } => self
                .request
                .output_handle(*expected_output_index)
                .map(|expected| (expected, self.request.output_handle(*actual_output_index))),
            _ => None,
        }
    }

    /// Output index for channel-scoped exact-submission correlation failures.
    pub fn correlation_output(&self) -> Option<crate::request::RenderOutputHandle> {
        let position = match &self.inner {
            RenderDeterministicVerifiedSubmissionError::DuplicateReadbackCorrelation {
                output_index,
                ..
            }
            | RenderDeterministicVerifiedSubmissionError::MissingSubmissionReadback {
                output_index,
                ..
            } => Some(*output_index),
            _ => None,
        };
        position.and_then(|index| self.request.output_handle(index))
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
    pub(super) request: RenderRequest,
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
    pub fn output(&self) -> Option<crate::request::RenderOutputHandle> {
        let position = match &self.inner {
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
        };
        position.and_then(|index| self.request.output_handle(index))
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
    OutputNotAdmitted,
    OutputNotRadiance,
    OutputTopologyUnsupported,
    OutputDestinationUnsupported,
    DestinationNotRetained,
    DestinationNotCopySource,
    CarrierFormatUnsupported,
    SourceUnavailable,
    ReadbackIdAllocationExhausted,
}

/// Failure to mint a product-owned readback correlation for one eligible radiance output.
#[derive(Debug)]
pub struct RenderRadianceCaptureRequestError {
    pub(super) inner: RenderDeterministicRadianceCaptureRequestError,
    pub(super) output: crate::request::RenderOutputHandle,
}

impl RenderRadianceCaptureRequestError {
    pub const fn output(&self) -> &crate::request::RenderOutputHandle {
        &self.output
    }

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
                RenderRadianceCaptureRequestErrorKind::OutputNotAdmitted
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
        if matches!(
            self.inner,
            RenderDeterministicRadianceCaptureRequestError::OutputIndexOutOfRange
        ) {
            formatter.write_str("requested output is not admitted for this invocation")
        } else {
            self.inner.fmt(formatter)
        }
    }
}

impl Error for RenderRadianceCaptureRequestError {}

/// Stable owner-oriented category for retained object-identity decoder lookup failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderObjectIdentityDecoderErrorKind {
    RendererSubmissionPending,
    RendererSubmissionFailed,
    OutputNotAdmitted,
    OutputNotObjectIdentity,
}

/// Failure to obtain an execution-local object-identity decoder from one associated occurrence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderObjectIdentityDecoderError {
    kind: RenderObjectIdentityDecoderErrorKind,
    gpu_failure_kind: Option<GpuSubmissionFailureKind>,
    output: crate::request::RenderOutputHandle,
}

impl RenderObjectIdentityDecoderError {
    pub(super) fn submission_pending(output: &crate::request::RenderOutputHandle) -> Self {
        Self {
            kind: RenderObjectIdentityDecoderErrorKind::RendererSubmissionPending,
            gpu_failure_kind: None,
            output: output.clone(),
        }
    }

    pub(super) fn submission_failed(
        kind: GpuSubmissionFailureKind,
        output: &crate::request::RenderOutputHandle,
    ) -> Self {
        Self {
            kind: RenderObjectIdentityDecoderErrorKind::RendererSubmissionFailed,
            gpu_failure_kind: Some(kind),
            output: output.clone(),
        }
    }

    pub(super) fn output_not_admitted(output: &crate::request::RenderOutputHandle) -> Self {
        Self {
            kind: RenderObjectIdentityDecoderErrorKind::OutputNotAdmitted,
            gpu_failure_kind: None,
            output: output.clone(),
        }
    }

    pub(super) fn output_not_object_identity(output: &crate::request::RenderOutputHandle) -> Self {
        Self {
            kind: RenderObjectIdentityDecoderErrorKind::OutputNotObjectIdentity,
            gpu_failure_kind: None,
            output: output.clone(),
        }
    }

    pub const fn kind(&self) -> RenderObjectIdentityDecoderErrorKind {
        self.kind
    }

    pub const fn output(&self) -> &crate::request::RenderOutputHandle {
        &self.output
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
            RenderObjectIdentityDecoderErrorKind::OutputNotAdmitted => {
                "requested object-identity output is not admitted for this invocation"
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
    pub(super) output: crate::request::RenderOutputHandle,
}

impl RenderRadianceCaptureError {
    pub const fn output(&self) -> &crate::request::RenderOutputHandle {
        &self.output
    }

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
