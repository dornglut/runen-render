use super::*;

/// Plan and admit one ordinary invocation of the maintained renderer.
///
/// Callers provide semantic scene/request/input facts, physical output destinations, and a public
/// RunenGPU context. Method selection, planning, binding admission, and maintained compatibility
/// remain inside RunenRender.
pub fn admit_render(
    invocation: &RenderInvocation,
    context: &GpuContext,
) -> Result<AdmittedRender, RenderAdmissionError> {
    admit_deterministic_render_with_semantic_inputs(
        invocation.scene(),
        invocation.request(),
        invocation.surface_inputs(),
        invocation.field_inputs(),
        invocation.availability(),
        invocation.output_bindings(),
        context,
    )
    .map(|inner| AdmittedRender { inner })
    .map_err(|inner| RenderAdmissionError {
        inner,
        request: invocation.request().clone(),
    })
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
    let request = admitted.admitted_plan().plan().request().clone();
    submit_deterministic_render_for_verified_result(admitted.inner, context)
        .await
        .map(|inner| SubmittedRenderForResult { inner })
        .map_err(|inner| RenderResultSubmissionError { inner, request })
}
