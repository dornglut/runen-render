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
