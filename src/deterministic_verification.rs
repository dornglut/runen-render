//! Private static eligibility, same-submission correlation, physical observation normalization, and
//! conservative semantic verification for RR566-EVAL-001 verified-result formation.
//!
//! This module deliberately does not narrow maintained deterministic execution. The ordinary
//! evaluator accepts every request already proven by `AdmittedDeterministicRender`; this gate asks a
//! separate question: whether the exact admitted semantics lie inside the first bounded domain for
//! which RunenRender is allowed to establish requested finite-evaluation fidelity. Verified
//! submission proves renderer-private readbacks are correlated to one exact accepted RunenGPU
//! submission; observation normalization then removes physical padding; the private semantic
//! verifier finally consumes that exact submission and fails closed on unsupported/ambiguous
//! arithmetic or branch evidence. FORM-001 result formation remains a later owner-controlled step.

mod numeric;
mod observation;
mod semantic;

use observation::{
    RenderDeterministicVerificationObservationError, observe_completed_deterministic_verification,
};

use super::deterministic_admission::AdmittedDeterministicRender;
use super::deterministic_execution::{
    self, DeterministicVerificationSubmission, RenderDeterministicExecutionError,
};
use super::render_result::{RenderResultFormationError, RenderResultFormationEvidence};
use super::request::RenderObservationSpec;
use super::scene::RenderObjectId;
use super::space_time::{RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState};
use runen_gpu::GpuContext;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

/// Canonical f64 threshold selected by the accepted first verification domain.
///
/// This is only a static domain gate. The dynamic semantic verifier must not treat this value,
/// `std::f64::consts::PI`, or host transcendental functions as exact mathematical reference
/// authority.
const CERTIFIED_MAX_FULL_FOV_RADIANS: f64 = std::f64::consts::FRAC_PI_2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RenderDeterministicVerificationEligibilityError {
    SelectedObservationMissing { observation_index: usize },
    PerspectiveFieldOfViewUnsupported { observation_index: usize },
    SamplingSupportUnsupported { observation_index: usize },
    ObservationLinearBasisUnsupported { observation_index: usize },
    SelectedObjectStateMissing { object_id: RenderObjectId },
    ObjectLocalScaleUnsupported { object_id: RenderObjectId },
    ObjectHandednessUnsupported { object_id: RenderObjectId },
    ObjectLinearBasisUnsupported { object_id: RenderObjectId },
}

impl fmt::Display for RenderDeterministicVerificationEligibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SelectedObservationMissing { observation_index } => write!(
                formatter,
                "admitted output references missing observation {observation_index} during verified result formation"
            ),
            Self::PerspectiveFieldOfViewUnsupported { observation_index } => write!(
                formatter,
                "observation {observation_index} lies outside the certified perspective field-of-view domain"
            ),
            Self::SamplingSupportUnsupported { observation_index } => write!(
                formatter,
                "observation {observation_index} requires ideal-ray sampling support for verified result formation"
            ),
            Self::ObservationLinearBasisUnsupported { observation_index } => write!(
                formatter,
                "observation {observation_index} requires identity linear basis for verified result formation"
            ),
            Self::SelectedObjectStateMissing { object_id } => write!(
                formatter,
                "selected object {object_id:?} has no retained state for verified result formation"
            ),
            Self::ObjectLocalScaleUnsupported { object_id } => write!(
                formatter,
                "selected object {object_id:?} requires unit local-space scale for verified result formation"
            ),
            Self::ObjectHandednessUnsupported { object_id } => write!(
                formatter,
                "selected object {object_id:?} requires right-handed local space for verified result formation"
            ),
            Self::ObjectLinearBasisUnsupported { object_id } => write!(
                formatter,
                "selected object {object_id:?} requires identity linear basis with translation only for verified result formation"
            ),
        }
    }
}

impl Error for RenderDeterministicVerificationEligibilityError {}

#[derive(Debug)]
pub(super) enum RenderDeterministicVerifiedSubmissionError {
    Eligibility(RenderDeterministicVerificationEligibilityError),
    Execution(RenderDeterministicExecutionError),
    ReadbackCardinality {
        expected: usize,
        actual: usize,
    },
    OutputCorrelationChanged {
        expected_output_index: usize,
        actual_output_index: usize,
    },
    DuplicateReadbackCorrelation {
        output_index: usize,
        channel: &'static str,
    },
    MissingSubmissionReadback {
        output_index: usize,
        channel: &'static str,
    },
}

impl fmt::Display for RenderDeterministicVerifiedSubmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eligibility(error) => {
                write!(formatter, "verification eligibility failed: {error}")
            }
            Self::Execution(error) => write!(
                formatter,
                "verified deterministic submission failed: {error}"
            ),
            Self::ReadbackCardinality { expected, actual } => write!(
                formatter,
                "verified deterministic submission retained {actual} output readback correlations for {expected} admitted outputs"
            ),
            Self::OutputCorrelationChanged {
                expected_output_index,
                actual_output_index,
            } => write!(
                formatter,
                "verified deterministic readback correlation changed from output {expected_output_index} to {actual_output_index}"
            ),
            Self::DuplicateReadbackCorrelation {
                output_index,
                channel,
            } => write!(
                formatter,
                "output {output_index} {channel} readback reused a correlation identity"
            ),
            Self::MissingSubmissionReadback {
                output_index,
                channel,
            } => write!(
                formatter,
                "output {output_index} {channel} readback is not owned by the exact returned RunenGPU submission"
            ),
        }
    }
}

impl Error for RenderDeterministicVerifiedSubmissionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Eligibility(error) => Some(error),
            Self::Execution(error) => Some(error),
            Self::ReadbackCardinality { .. }
            | Self::OutputCorrelationChanged { .. }
            | Self::DuplicateReadbackCorrelation { .. }
            | Self::MissingSubmissionReadback { .. } => None,
        }
    }
}

impl From<RenderDeterministicVerificationEligibilityError>
    for RenderDeterministicVerifiedSubmissionError
{
    fn from(value: RenderDeterministicVerificationEligibilityError) -> Self {
        Self::Eligibility(value)
    }
}

impl From<RenderDeterministicExecutionError> for RenderDeterministicVerifiedSubmissionError {
    fn from(value: RenderDeterministicExecutionError) -> Self {
        Self::Execution(value)
    }
}

/// Dynamic RR566-EVAL-001 failure after one exact verified submission has been authored.
///
/// Static-domain rejection remains the eligibility error above. `Inconclusive` is reserved for a
/// semantic branch the conservative interval proof cannot uniquely establish. Physical mismatch and
/// tolerance mismatch mean the completed observation contradicts the certified semantic result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RenderDeterministicVerificationError {
    ResultFormation(RenderResultFormationError),
    Eligibility(RenderDeterministicVerificationEligibilityError),
    ObservationNormalization(RenderDeterministicVerificationObservationError),
    Correlation {
        output_index: usize,
        sample_index: Option<usize>,
        detail: &'static str,
    },
    Inconclusive {
        output_index: usize,
        sample_index: Option<usize>,
        detail: &'static str,
    },
    PhysicalMismatch {
        output_index: usize,
        sample_index: Option<usize>,
        detail: &'static str,
    },
    ToleranceMismatch {
        output_index: usize,
        sample_index: usize,
    },
}

impl fmt::Display for RenderDeterministicVerificationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Eligibility(error) => {
                write!(formatter, "verification eligibility failed: {error}")
            }
            Self::ResultFormation(error) => write!(
                formatter,
                "generic render result-formation evidence rejected verified execution: {error}"
            ),
            Self::ObservationNormalization(error) => {
                write!(
                    formatter,
                    "same-submission observation normalization failed: {error}"
                )
            }
            Self::Correlation {
                output_index,
                sample_index,
                detail,
            } => format_verification_location(
                formatter,
                "correlation failed",
                *output_index,
                *sample_index,
                detail,
            ),
            Self::Inconclusive {
                output_index,
                sample_index,
                detail,
            } => format_verification_location(
                formatter,
                "verification is inconclusive",
                *output_index,
                *sample_index,
                detail,
            ),
            Self::PhysicalMismatch {
                output_index,
                sample_index,
                detail,
            } => format_verification_location(
                formatter,
                "physical observation mismatch",
                *output_index,
                *sample_index,
                detail,
            ),
            Self::ToleranceMismatch {
                output_index,
                sample_index,
            } => write!(
                formatter,
                "output {output_index} sample {sample_index} does not satisfy the requested semantic tolerance"
            ),
        }
    }
}

impl Error for RenderDeterministicVerificationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::ResultFormation(error) => Some(error),
            Self::Eligibility(error) => Some(error),
            Self::ObservationNormalization(error) => Some(error),
            Self::Correlation { .. }
            | Self::Inconclusive { .. }
            | Self::PhysicalMismatch { .. }
            | Self::ToleranceMismatch { .. } => None,
        }
    }
}

impl RenderDeterministicVerificationError {
    pub(super) const fn output_index(&self) -> Option<usize> {
        match self {
            Self::ObservationNormalization(error) => error.output_index(),
            Self::Correlation { output_index, .. }
            | Self::Inconclusive { output_index, .. }
            | Self::PhysicalMismatch { output_index, .. }
            | Self::ToleranceMismatch { output_index, .. } => Some(*output_index),
            Self::ResultFormation(_) | Self::Eligibility(_) => None,
        }
    }

    pub(super) const fn sample_index(&self) -> Option<usize> {
        match self {
            Self::Correlation { sample_index, .. }
            | Self::Inconclusive { sample_index, .. }
            | Self::PhysicalMismatch { sample_index, .. } => *sample_index,
            Self::ToleranceMismatch { sample_index, .. } => Some(*sample_index),
            Self::ResultFormation(_) | Self::Eligibility(_) | Self::ObservationNormalization(_) => {
                None
            }
        }
    }

    pub(super) const fn channel(&self) -> Option<&'static str> {
        match self {
            Self::ObservationNormalization(error) => error.channel(),
            _ => None,
        }
    }

    pub(super) const fn gpu_failure_kind(&self) -> Option<runen_gpu::GpuSubmissionFailureKind> {
        match self {
            Self::ObservationNormalization(error) => error.gpu_failure_kind(),
            _ => None,
        }
    }
}

fn format_verification_location(
    formatter: &mut fmt::Formatter<'_>,
    category: &'static str,
    output_index: usize,
    sample_index: Option<usize>,
    detail: &'static str,
) -> fmt::Result {
    if let Some(sample_index) = sample_index {
        write!(
            formatter,
            "output {output_index} sample {sample_index} {category}: {detail}"
        )
    } else {
        write!(formatter, "output {output_index} {category}: {detail}")
    }
}

/// Prove only the static RR566-EVAL-001 subset that is knowable before one verified submission.
///
/// Instant shutters and ideal-ray sampling are already invariants of `AdmittedDeterministicRender`.
/// This gate therefore adds only verifier-specific restrictions; failure here must never make the
/// same admitted invocation illegal for ordinary maintained execution.
pub(super) fn ensure_deterministic_verification_eligible(
    maintained: &AdmittedDeterministicRender,
) -> Result<(), RenderDeterministicVerificationEligibilityError> {
    let admitted = maintained.admitted();
    let mut selected_observations = BTreeSet::new();
    let mut selected_objects = BTreeSet::new();
    for output in admitted.outputs() {
        selected_observations.insert(output.observation_index());
        for object in output.object_representations() {
            selected_objects.insert(object.object_id());
        }
    }

    validate_selected_observations(
        admitted.plan().request().observations(),
        selected_observations,
    )?;

    for object_id in selected_objects {
        let state = admitted.plan().scene().object_state(object_id).ok_or(
            RenderDeterministicVerificationEligibilityError::SelectedObjectStateMissing {
                object_id,
            },
        )?;
        validate_object_spatial_state(object_id, state.spatial())?;
    }

    Ok(())
}

fn validate_selected_observations(
    observations: &[RenderObservationSpec],
    selected_observations: impl IntoIterator<Item = usize>,
) -> Result<(), RenderDeterministicVerificationEligibilityError> {
    for observation_index in selected_observations {
        let observation = observations.get(observation_index).copied().ok_or(
            RenderDeterministicVerificationEligibilityError::SelectedObservationMissing {
                observation_index,
            },
        )?;
        validate_observation(observation_index, observation)?;
    }
    Ok(())
}

/// Select verified-result intent before submission, then prove that every private observation
/// correlation resolves back into that exact returned `GpuSubmission`.
///
/// This function intentionally stops before readback readiness or semantic comparison. A successful
/// return proves only the static EVAL-001 domain gate plus same-submission correlation for canonical
/// payload, semantic-definedness, and evaluator-status observations.
pub(super) async fn submit_deterministic_render_for_verified_formation(
    maintained: AdmittedDeterministicRender,
    context: &GpuContext,
) -> Result<DeterministicVerificationSubmission, RenderDeterministicVerifiedSubmissionError> {
    ensure_deterministic_verification_eligible(&maintained)?;
    let verification =
        deterministic_execution::submit_deterministic_render_for_verification(maintained, context)
            .await?;

    let submitted = verification.submitted();
    let expected_outputs = submitted.admitted().admitted().outputs();
    let readbacks = verification.readbacks();
    if readbacks.len() != expected_outputs.len() {
        return Err(
            RenderDeterministicVerifiedSubmissionError::ReadbackCardinality {
                expected: expected_outputs.len(),
                actual: readbacks.len(),
            },
        );
    }

    let mut correlation_ids = BTreeSet::new();
    for (expected_output, readbacks) in expected_outputs.iter().zip(readbacks) {
        if readbacks.output_index() != expected_output.output_index() {
            return Err(
                RenderDeterministicVerifiedSubmissionError::OutputCorrelationChanged {
                    expected_output_index: expected_output.output_index(),
                    actual_output_index: readbacks.output_index(),
                },
            );
        }

        for (channel, id) in [
            ("canonical-output", readbacks.canonical_output()),
            ("definedness", readbacks.definedness()),
            ("evaluator-status", readbacks.status()),
        ] {
            if !correlation_ids.insert(id) {
                return Err(
                    RenderDeterministicVerifiedSubmissionError::DuplicateReadbackCorrelation {
                        output_index: readbacks.output_index(),
                        channel,
                    },
                );
            }
            if submitted.submission().readback(id).is_none() {
                return Err(
                    RenderDeterministicVerifiedSubmissionError::MissingSubmissionReadback {
                        output_index: readbacks.output_index(),
                        channel,
                    },
                );
            }
        }
    }

    Ok(verification)
}

/// Consume one completed same-submission observation set and establish the first conservative
/// deterministic finite-evaluation witness.
///
/// Success does not yet form `RenderResult`; it returns an execution-scoped owner-controlled witness
/// that later FORM-001 wiring can consume without accepting detached output-index assertions.
pub(super) fn verify_completed_deterministic_render(
    verification: DeterministicVerificationSubmission,
) -> Result<RenderResultFormationEvidence, RenderDeterministicVerificationError> {
    ensure_deterministic_verification_eligible(verification.submitted().admitted())
        .map_err(RenderDeterministicVerificationError::Eligibility)?;
    let observations = observe_completed_deterministic_verification(&verification)
        .map_err(RenderDeterministicVerificationError::ObservationNormalization)?;
    semantic::verify_completed_semantics(&verification, &observations)?;
    let admitted = verification.submitted().admitted().admitted();
    RenderResultFormationEvidence::complete(
        admitted,
        admitted
            .outputs()
            .iter()
            .map(|output| output.output_index()),
    )
    .map_err(RenderDeterministicVerificationError::ResultFormation)
}

fn validate_observation(
    observation_index: usize,
    observation: RenderObservationSpec,
) -> Result<(), RenderDeterministicVerificationEligibilityError> {
    let sampling_support = match observation {
        RenderObservationSpec::Perspective(perspective) => perspective.sampling_support(),
        RenderObservationSpec::Probe(probe) => probe.sampling_support(),
    };
    if !sampling_support.is_ideal_ray() {
        return Err(
            RenderDeterministicVerificationEligibilityError::SamplingSupportUnsupported {
                observation_index,
            },
        );
    }

    let transform = match observation {
        RenderObservationSpec::Perspective(perspective) => {
            if perspective.vertical_field_of_view_radians() > CERTIFIED_MAX_FULL_FOV_RADIANS {
                return Err(
                    RenderDeterministicVerificationEligibilityError::PerspectiveFieldOfViewUnsupported {
                        observation_index,
                    },
                );
            }
            perspective.observation_to_scene()
        }
        RenderObservationSpec::Probe(probe) => probe.observation_to_scene(),
    };

    if !has_identity_linear_basis(transform) {
        return Err(
            RenderDeterministicVerificationEligibilityError::ObservationLinearBasisUnsupported {
                observation_index,
            },
        );
    }
    Ok(())
}

fn validate_object_spatial_state(
    object_id: RenderObjectId,
    spatial: &RenderObjectSpatialState,
) -> Result<(), RenderDeterministicVerificationEligibilityError> {
    let local_space = spatial.local_space();
    if local_space.meters_per_unit() != 1.0 {
        return Err(
            RenderDeterministicVerificationEligibilityError::ObjectLocalScaleUnsupported {
                object_id,
            },
        );
    }
    if local_space.handedness() != RenderHandedness::Right {
        return Err(
            RenderDeterministicVerificationEligibilityError::ObjectHandednessUnsupported {
                object_id,
            },
        );
    }
    if !has_identity_linear_basis(spatial.local_to_scene()) {
        return Err(
            RenderDeterministicVerificationEligibilityError::ObjectLinearBasisUnsupported {
                object_id,
            },
        );
    }
    Ok(())
}

fn has_identity_linear_basis(transform: RenderAffineTransform3) -> bool {
    let matrix = transform.row_major_3x4();
    [
        matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
        matrix[10],
    ] == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{
        RenderPerspectiveObservation, RenderProbeObservation, RenderSamplingSupport,
    };
    use crate::space_time::{
        RenderSemanticValueError, RenderSpaceSpec, RenderSpatialCoverage, RenderTimeInterval,
        RenderTimePoint,
    };

    fn instant() -> RenderTimeInterval {
        RenderTimeInterval::instant(RenderTimePoint::from_seconds(0.0).expect("time"))
    }

    fn translated_identity(x: f64, y: f64, z: f64) -> RenderAffineTransform3 {
        RenderAffineTransform3::from_row_major_3x4([
            1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z,
        ])
        .expect("translated identity")
    }

    fn rotated_basis() -> RenderAffineTransform3 {
        RenderAffineTransform3::from_row_major_3x4([
            0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ])
        .expect("rotation")
    }

    #[test]
    fn observation_gate_accepts_translation_but_not_non_identity_linear_basis() {
        let perspective = RenderObservationSpec::Perspective(
            RenderPerspectiveObservation::new(
                translated_identity(1.0, 2.0, 3.0),
                std::f64::consts::FRAC_PI_4,
                1.0,
                instant(),
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("perspective"),
        );
        assert_eq!(validate_observation(0, perspective), Ok(()));

        let probe = RenderObservationSpec::Probe(
            RenderProbeObservation::new(
                rotated_basis(),
                instant(),
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("probe"),
        );
        assert_eq!(
            validate_observation(2, probe),
            Err(
                RenderDeterministicVerificationEligibilityError::ObservationLinearBasisUnsupported {
                    observation_index: 2,
                }
            )
        );
    }

    #[test]
    fn observation_gate_rejects_non_ideal_sampling_support() {
        let perspective = RenderObservationSpec::Perspective(
            RenderPerspectiveObservation::new(
                RenderAffineTransform3::identity(),
                std::f64::consts::FRAC_PI_4,
                1.0,
                instant(),
                RenderSamplingSupport::perspective_lattice_cell(),
            )
            .expect("semantically valid footprint perspective"),
        );
        assert_eq!(
            validate_observation(3, perspective),
            Err(
                RenderDeterministicVerificationEligibilityError::SamplingSupportUnsupported {
                    observation_index: 3,
                }
            )
        );
    }

    #[test]
    fn selected_observation_gate_ignores_unreferenced_unsupported_observations() {
        let supported = RenderObservationSpec::Perspective(
            RenderPerspectiveObservation::new(
                RenderAffineTransform3::identity(),
                std::f64::consts::FRAC_PI_4,
                1.0,
                instant(),
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("supported perspective"),
        );
        let unsupported = RenderObservationSpec::Perspective(
            RenderPerspectiveObservation::new(
                RenderAffineTransform3::identity(),
                std::f64::consts::FRAC_PI_2 * 1.5,
                1.0,
                instant(),
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("semantically legal wider perspective"),
        );
        let observations = [supported, unsupported];

        assert_eq!(validate_selected_observations(&observations, [0]), Ok(()));
        assert_eq!(
            validate_selected_observations(&observations, [1]),
            Err(
                RenderDeterministicVerificationEligibilityError::PerspectiveFieldOfViewUnsupported {
                    observation_index: 1,
                }
            )
        );
    }

    #[test]
    fn selected_observation_gate_fails_closed_on_missing_admitted_correlation() {
        let observation = RenderObservationSpec::Probe(
            RenderProbeObservation::new(
                RenderAffineTransform3::identity(),
                instant(),
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("probe"),
        );
        assert_eq!(
            validate_selected_observations(&[observation], [1]),
            Err(
                RenderDeterministicVerificationEligibilityError::SelectedObservationMissing {
                    observation_index: 1,
                }
            )
        );
    }

    #[test]
    fn perspective_gate_rejects_fov_outside_first_certified_domain() {
        let observation = RenderObservationSpec::Perspective(
            RenderPerspectiveObservation::new(
                RenderAffineTransform3::identity(),
                std::f64::consts::FRAC_PI_2 * 1.5,
                1.0,
                instant(),
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("semantically legal wider perspective"),
        );
        assert_eq!(
            validate_observation(1, observation),
            Err(
                RenderDeterministicVerificationEligibilityError::PerspectiveFieldOfViewUnsupported {
                    observation_index: 1,
                }
            )
        );
    }

    #[test]
    fn object_gate_accepts_unit_right_handed_translation_only_state() {
        let spatial = RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("space"),
            translated_identity(4.0, 5.0, 6.0),
            RenderSpatialCoverage::unbounded(),
        );
        let mut store = crate::scene::RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object id");
        assert_eq!(validate_object_spatial_state(object_id, &spatial), Ok(()));
    }

    #[test]
    fn object_gate_rejects_scale_handedness_and_linear_basis_independently() {
        let mut store = crate::scene::RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object id");

        let scaled = RenderObjectSpatialState::new(
            RenderSpaceSpec::new(2.0, RenderHandedness::Right).expect("space"),
            RenderAffineTransform3::identity(),
            RenderSpatialCoverage::unbounded(),
        );
        assert_eq!(
            validate_object_spatial_state(object_id, &scaled),
            Err(
                RenderDeterministicVerificationEligibilityError::ObjectLocalScaleUnsupported {
                    object_id,
                }
            )
        );

        let left_handed = RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Left).expect("space"),
            RenderAffineTransform3::identity(),
            RenderSpatialCoverage::unbounded(),
        );
        assert_eq!(
            validate_object_spatial_state(object_id, &left_handed),
            Err(
                RenderDeterministicVerificationEligibilityError::ObjectHandednessUnsupported {
                    object_id,
                }
            )
        );

        let rotated = RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("space"),
            rotated_basis(),
            RenderSpatialCoverage::unbounded(),
        );
        assert_eq!(
            validate_object_spatial_state(object_id, &rotated),
            Err(
                RenderDeterministicVerificationEligibilityError::ObjectLinearBasisUnsupported {
                    object_id,
                }
            )
        );
    }

    #[test]
    fn semantic_transform_constructors_remain_the_finiteness_authority() {
        assert_eq!(
            RenderAffineTransform3::from_row_major_3x4([
                1.0,
                0.0,
                0.0,
                f64::INFINITY,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ]),
            Err(RenderSemanticValueError::NonFinite {
                field: "affine_transform"
            })
        );
    }
    #[test]
    fn observation_normalization_error_preserves_structured_location() {
        let error = RenderDeterministicVerificationError::ObservationNormalization(
            RenderDeterministicVerificationObservationError::InvalidPhysicalLayout {
                output_index: 4,
                channel: "canonical-output",
                byte_len: 13,
            },
        );

        assert_eq!(error.output_index(), Some(4));
        assert_eq!(error.sample_index(), None);
        assert_eq!(error.channel(), Some("canonical-output"));
        assert!(Error::source(&error).is_some());
    }
}
