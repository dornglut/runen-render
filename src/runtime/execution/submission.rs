use super::*;

pub(crate) async fn submit_deterministic_render_for_verification(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
) -> Result<DeterministicVerificationSubmission, RenderDeterministicExecutionError> {
    let lowered = lower_deterministic_render(
        &admitted,
        context,
        DeterministicObservationIntent::Verify,
        &mut DeterministicResourceCache::default(),
        DeterministicRenderExecutionSelection {
            scope: 0,
            finite_evaluation: None,
            produce_requested_coverage: false,
        },
    )?;
    let verification_readbacks = lowered.verification_readbacks;
    let submitted = submit_lowered_deterministic_render(
        admitted,
        context,
        lowered.work_set,
        lowered.object_identity_decoder,
        DeterministicVerificationState::Requested(verification_readbacks),
    )
    .await?;
    Ok(DeterministicVerificationSubmission { submitted })
}

pub(super) async fn submit_lowered_deterministic_render(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
    work_set: RenderWorkSet,
    object_identity_decoder: RenderObjectIdentityDecoder,
    verification: DeterministicVerificationState,
) -> Result<SubmittedDeterministicRender, RenderDeterministicExecutionError> {
    let submission = context
        .submit_work(
            "RunenRender maintained deterministic execution",
            work_set.fragments().iter().cloned(),
        )
        .await
        .map_err(RenderDeterministicExecutionError::Submission)?;
    Ok(SubmittedDeterministicRender {
        admitted,
        submission,
        object_identity_decoder,
        verification,
    })
}

pub(super) async fn submit_prepared_deterministic_render(
    prepared: PreparedDeterministicRender,
    context: &GpuContext,
) -> Result<SubmittedDeterministicRender, RenderDeterministicExecutionError> {
    let submission = context
        .submit_work(
            "RunenRender maintained deterministic execution",
            prepared.work_set.fragments().iter().cloned(),
        )
        .await
        .map_err(RenderDeterministicExecutionError::Submission)?;
    Ok(SubmittedDeterministicRender {
        admitted: prepared.admitted,
        submission,
        object_identity_decoder: prepared.object_identity_decoder,
        verification: DeterministicVerificationState::NotRequested,
    })
}

