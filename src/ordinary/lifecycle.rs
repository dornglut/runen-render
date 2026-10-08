use super::*;
use crate::request::{RenderOutputHandle, RenderOutputValue};

fn checked_identity_decoder<'a>(
    admitted: &AdmittedRenderPlan,
    output: &RenderOutputHandle,
    status: GpuSubmissionStatus,
    decoder: &'a RenderObjectIdentityDecoder,
) -> Result<&'a RenderObjectIdentityDecoder, RenderObjectIdentityDecoderError> {
    match status {
        GpuSubmissionStatus::Accepted => {
            return Err(RenderObjectIdentityDecoderError::submission_pending(output));
        }
        GpuSubmissionStatus::Failed(failure) => {
            return Err(RenderObjectIdentityDecoderError::submission_failed(
                failure.kind(),
                output,
            ));
        }
        GpuSubmissionStatus::Completed => {}
    }
    let request = admitted.plan().request();
    if !request.contains_output(output)
        || !admitted
            .outputs()
            .iter()
            .any(|candidate| candidate.output_index() == output.position())
    {
        return Err(RenderObjectIdentityDecoderError::output_not_admitted(
            output,
        ));
    }
    if !matches!(
        request.outputs()[output.position()].spec().value(),
        RenderOutputValue::ObjectIdentity
    ) {
        return Err(RenderObjectIdentityDecoderError::output_not_object_identity(output));
    }
    Ok(decoder)
}

/// Maintained invocation after semantic planning, binding, and execution admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedRender {
    pub(super) inner: AdmittedDeterministicRender,
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
    pub(super) inner: PreparedDeterministicRender,
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
        let request = self.admitted_plan().plan().request();
        self.inner
            .radiance_outputs()
            .iter()
            .map(move |inner| PreparedRadianceOutput {
                inner,
                output: request
                    .output_handle(inner.output_index())
                    .expect("admitted output handle"),
            })
    }

    /// Prepared radiance output for an exact request-owned output handle, when applicable.
    pub fn radiance_output(
        &self,
        output: &RenderOutputHandle,
    ) -> Option<PreparedRadianceOutput<'_>> {
        if !self
            .admitted_plan()
            .plan()
            .request()
            .contains_output(output)
        {
            return None;
        }
        self.inner
            .radiance_output(output.position())
            .map(|inner| PreparedRadianceOutput {
                inner,
                output: output.clone(),
            })
    }
}

/// One exact retained renderer occurrence prepared for caller-owned RunenGPU composition.
///
/// This value is intentionally not cloneable. It belongs to the exact RenderExecutionSession that
/// prepared it and is consumed by submission association. Dropping it abandons the provisional
/// retained transition; preparation itself is never execution evidence.
#[derive(Debug)]
pub struct PreparedRenderOccurrence {
    pub(super) inner: PreparedDeterministicRender,
    pub(super) session_identity: std::sync::Arc<()>,
    pub(super) occurrence_identity: std::sync::Arc<()>,
    pub(super) affinity: GpuContextAffinity,
}

impl PreparedRenderOccurrence {
    pub const fn admitted_plan(&self) -> &AdmittedRenderPlan {
        self.inner.admitted().admitted()
    }

    pub const fn work_set(&self) -> &RenderWorkSet {
        self.inner.work_set()
    }

    pub fn radiance_outputs(
        &self,
    ) -> impl ExactSizeIterator<Item = PreparedRadianceOutput<'_>> + '_ {
        let request = self.admitted_plan().plan().request();
        self.inner
            .radiance_outputs()
            .iter()
            .map(move |inner| PreparedRadianceOutput {
                inner,
                output: request
                    .output_handle(inner.output_index())
                    .expect("admitted output handle"),
            })
    }

    pub fn radiance_output(
        &self,
        output: &RenderOutputHandle,
    ) -> Option<PreparedRadianceOutput<'_>> {
        if !self
            .admitted_plan()
            .plan()
            .request()
            .contains_output(output)
        {
            return None;
        }
        self.inner
            .radiance_output(output.position())
            .map(|inner| PreparedRadianceOutput {
                inner,
                output: output.clone(),
            })
    }
}

/// Exact output-correlation witness for one retained occurrence associated with RunenGPU.
///
/// The retained session remains the sole owner of temporal-history reconciliation. This witness
/// owns only the exact prepared output interpretation metadata and a clone of the caller's exact
/// associated submission. It survives session reconciliation or session destruction, but output
/// interpretation still fails closed when the submission did not complete successfully, affinity
/// or retained-resource continuity is lost, or a newer completed renderer write supersedes the
/// observed destination.
#[derive(Debug)]
pub struct AssociatedRenderOccurrence {
    pub(super) inner: AssociatedDeterministicRender,
    pub(super) submission: GpuSubmission,
    pub(super) occurrence_identity: std::sync::Arc<()>,
}

impl AssociatedRenderOccurrence {
    /// Exact admitted semantic plan for this associated occurrence.
    pub const fn admitted_plan(&self) -> &AdmittedRenderPlan {
        self.inner.admitted().admitted()
    }

    /// Current physical lifecycle status of the exact associated RunenGPU submission.
    pub fn submission_status(&self) -> GpuSubmissionStatus {
        self.submission.status()
    }

    /// Mint one fresh caller-owned readback correlation for a completed retained radiance output.
    ///
    /// This interprets maintained physical output only. It does not form or certify a RenderResult.
    pub fn request_radiance_capture(
        &self,
        output: &RenderOutputHandle,
    ) -> Result<RenderRadianceCaptureRequest, RenderRadianceCaptureRequestError> {
        if !self
            .admitted_plan()
            .plan()
            .request()
            .contains_output(output)
        {
            return Err(RenderRadianceCaptureRequestError {
                inner: RenderDeterministicRadianceCaptureRequestError::OutputIndexOutOfRange,
                output: output.clone(),
            });
        }
        crate::runtime::capture::mint_retained_request(
            &self.inner,
            &self.submission,
            output.position(),
        )
        .map(|inner| RenderRadianceCaptureRequest {
            inner,
            output: output.clone(),
            retained_occurrence_identity: Some(std::sync::Arc::clone(&self.occurrence_identity)),
        })
        .map_err(|inner| RenderRadianceCaptureRequestError {
            inner,
            output: output.clone(),
        })
    }

    /// Interpret one completed caller-owned RunenGPU readback through this exact occurrence.
    pub fn capture_radiance(
        &self,
        request: RenderRadianceCaptureRequest,
        context: &GpuContext,
        product_submission: &GpuSubmission,
    ) -> Result<RenderCapturedRadiance, RenderRadianceCaptureError> {
        let Some(identity) = request.retained_occurrence_identity.as_ref() else {
            return Err(RenderRadianceCaptureError {
                inner: RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch,
                output: request.output.clone(),
            });
        };
        if !std::sync::Arc::ptr_eq(identity, &self.occurrence_identity) {
            return Err(RenderRadianceCaptureError {
                inner: RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch,
                output: request.output.clone(),
            });
        }
        let output = request.output.clone();
        crate::runtime::capture::capture_retained(
            &self.inner,
            &self.submission,
            request.inner,
            context,
            product_submission,
        )
        .map(|inner| RenderCapturedRadiance {
            inner,
            output: output.clone(),
        })
        .map_err(|inner| RenderRadianceCaptureError { inner, output })
    }

    /// Execution-local physical object-identity decoder for one completed identity output.
    ///
    /// The returned codebook maps physical words to optional renderer object IDs only. It is not
    /// per-pixel definedness, miss/background evidence, or persistent identity.
    pub fn object_identity_decoder(
        &self,
        output: &RenderOutputHandle,
    ) -> Result<&RenderObjectIdentityDecoder, RenderObjectIdentityDecoderError> {
        checked_identity_decoder(
            self.admitted_plan(),
            output,
            self.submission.status(),
            self.inner.object_identity_decoder(),
        )
    }
}

/// Borrowed correlation for one prepared radiance destination and its public RunenGPU export.
#[derive(Debug, Clone)]
pub struct PreparedRadianceOutput<'a> {
    pub(super) inner: &'a PreparedDeterministicRadianceOutput,
    pub(super) output: RenderOutputHandle,
}

impl PreparedRadianceOutput<'_> {
    /// Exact request-owned output correlated to this prepared destination.
    pub const fn output(&self) -> &RenderOutputHandle {
        &self.output
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

/// Exact correlation for one product-owned public RunenGPU radiance readback.
pub struct RenderRadianceCaptureRequest {
    pub(super) inner: RenderDeterministicRadianceCaptureRequest,
    pub(super) output: RenderOutputHandle,
    pub(super) retained_occurrence_identity: Option<std::sync::Arc<()>>,
}

impl RenderRadianceCaptureRequest {
    /// Exact request-owned output correlated to this capture.
    pub const fn output(&self) -> &RenderOutputHandle {
        &self.output
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
#[derive(Debug, Clone)]
pub struct RenderCapturedRadiance {
    pub(super) inner: RenderCapturedDeterministicRadiance,
    pub(super) output: RenderOutputHandle,
}

// Observed value equality remains semantic; request allocation identity is only for correlation.
impl PartialEq for RenderCapturedRadiance {
    fn eq(&self, other: &Self) -> bool {
        self.inner == other.inner
    }
}

impl RenderCapturedRadiance {
    /// Exact request-owned output whose retained destination was observed.
    pub const fn output(&self) -> &RenderOutputHandle {
        &self.output
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
    pub(super) inner: SubmittedDeterministicRender,
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
    pub fn object_identity_decoder(
        &self,
        output: &RenderOutputHandle,
    ) -> Result<&RenderObjectIdentityDecoder, RenderObjectIdentityDecoderError> {
        checked_identity_decoder(
            self.admitted_plan(),
            output,
            self.inner.submission_status(),
            self.inner.object_identity_decoder(),
        )
    }
}

/// One maintained render submitted with semantic-result verification enabled.
///
/// Callers poll result formation after progressing the public RunenGPU context. Product readback
/// remains a separate optional submission even after semantic result formation succeeds.
#[derive(Debug)]
pub struct SubmittedRenderForResult {
    pub(super) inner: SubmittedDeterministicRender,
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
    pub fn object_identity_decoder(
        &self,
        output: &RenderOutputHandle,
    ) -> Result<&RenderObjectIdentityDecoder, RenderObjectIdentityDecoderError> {
        checked_identity_decoder(
            self.admitted_plan(),
            output,
            self.inner.submission_status(),
            self.inner.object_identity_decoder(),
        )
    }

    /// Poll semantic result formation without blocking or driving RunenGPU progress.
    pub fn try_form_result(&mut self) -> Result<Option<RenderResult>, RenderResultFormationError> {
        let request = self.admitted_plan().plan().request().clone();
        self.inner
            .try_form_verified_result()
            .map_err(|inner| RenderResultFormationError { inner, request })
    }

    /// Mint one fresh product-owned readback correlation for a formed radiance output.
    pub fn request_radiance_capture(
        &self,
        output: &RenderOutputHandle,
    ) -> Result<RenderRadianceCaptureRequest, RenderRadianceCaptureRequestError> {
        if !self
            .admitted_plan()
            .plan()
            .request()
            .contains_output(output)
        {
            return Err(RenderRadianceCaptureRequestError {
                inner: RenderDeterministicRadianceCaptureRequestError::OutputIndexOutOfRange,
                output: output.clone(),
            });
        }
        self.inner
            .request_deterministic_radiance_capture(output.position())
            .map(|inner| RenderRadianceCaptureRequest {
                inner,
                output: output.clone(),
                retained_occurrence_identity: None,
            })
            .map_err(|inner| RenderRadianceCaptureRequestError {
                inner,
                output: output.clone(),
            })
    }

    /// Interpret one completed product-owned RunenGPU readback through the maintained carrier.
    pub fn capture_radiance(
        &self,
        request: RenderRadianceCaptureRequest,
        context: &GpuContext,
        product_submission: &GpuSubmission,
    ) -> Result<RenderCapturedRadiance, RenderRadianceCaptureError> {
        if request.retained_occurrence_identity.is_some() {
            return Err(RenderRadianceCaptureError {
                inner: RenderDeterministicRadianceCaptureError::RequestCorrelationMismatch,
                output: request.output.clone(),
            });
        }
        let output = request.output.clone();
        self.inner
            .capture_deterministic_radiance(request.inner, context, product_submission)
            .map(|inner| RenderCapturedRadiance {
                inner,
                output: output.clone(),
            })
            .map_err(|inner| RenderRadianceCaptureError { inner, output })
    }
}
