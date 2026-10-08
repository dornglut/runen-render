use super::*;

fn error_request() -> RenderRequest {
    use crate::request::{
        RenderObservationSpec, RenderOutputSpec, RenderOutputValue, RenderProbeObservation,
        RenderRequestBuilder, RenderResultTopology, RenderSamplingSupport, RenderSemanticTolerance,
    };
    use crate::space_time::{RenderAffineTransform3, RenderTimeInterval, RenderTimePoint};
    let shutter =
        RenderTimeInterval::instant(RenderTimePoint::from_seconds(0.0).expect("finite test time"));
    let observation = RenderObservationSpec::Probe(
        RenderProbeObservation::new(
            RenderAffineTransform3::identity(),
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("valid test observation"),
    );
    let output = RenderOutputSpec::new(
        RenderOutputValue::ObjectIdentity,
        RenderResultTopology::scalar(),
        RenderSemanticTolerance::exact(),
    )
    .expect("valid test output");
    let mut builder = RenderRequestBuilder::new(shutter);
    let observations = (0..4)
        .map(|_| builder.add_observation(observation))
        .collect::<Vec<_>>();
    for index in 0..5 {
        builder
            .add_output(&observations[index.min(3)], output)
            .expect("own observation");
    }
    builder.finish().expect("five valid test outputs")
}

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
        request: error_request(),
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
    assert_eq!(
        eligibility.observation().map(|handle| handle.position()),
        Some(3)
    );
    assert!(Error::source(&eligibility).is_some());

    let correlation = RenderResultSubmissionError {
        request: error_request(),
        inner: RenderDeterministicVerifiedSubmissionError::MissingSubmissionReadback {
            output_index: 2,
            channel: "canonical-output",
        },
    };
    assert_eq!(
        correlation.kind(),
        RenderResultSubmissionErrorKind::Correlation
    );
    assert_eq!(
        correlation
            .correlation_output()
            .map(|handle| handle.position()),
        Some(2)
    );
    assert_eq!(correlation.correlation_channel(), Some("canonical-output"));
}

#[test]
fn result_formation_preserves_semantic_verification_location() {
    let correlation = RenderResultFormationError {
        request: error_request(),
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
    assert_eq!(
        correlation.output().map(|handle| handle.position()),
        Some(2)
    );
    assert_eq!(correlation.sample_index(), Some(7));
    assert!(Error::source(&correlation).is_some());

    let tolerance = RenderResultFormationError {
        request: error_request(),
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
    assert_eq!(tolerance.output().map(|handle| handle.position()), Some(1));
    assert_eq!(tolerance.sample_index(), Some(5));

    let inconclusive = RenderResultFormationError {
        request: error_request(),
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
    assert_eq!(
        inconclusive.output().map(|handle| handle.position()),
        Some(3)
    );
    assert_eq!(inconclusive.sample_index(), None);

    let physical = RenderResultFormationError {
        request: error_request(),
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
    assert_eq!(physical.output().map(|handle| handle.position()), Some(4));
    assert_eq!(physical.sample_index(), Some(2));
}
