use super::*;

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
        self.inner
            .radiance_outputs()
            .iter()
            .map(|inner| PreparedRadianceOutput { inner })
    }

    pub fn radiance_output(&self, output_index: usize) -> Option<PreparedRadianceOutput<'_>> {
        self.inner
            .radiance_output(output_index)
            .map(|inner| PreparedRadianceOutput { inner })
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
    pub(super) inner: PreparedDeterministicRender,
    pub(super) submission: GpuSubmission,
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
        output_index: usize,
    ) -> Result<RenderRadianceCaptureRequest, RenderRadianceCaptureRequestError> {
        crate::runtime::capture::mint_retained_request(&self.inner, &self.submission, output_index)
            .map(|inner| RenderRadianceCaptureRequest { inner })
            .map_err(|inner| RenderRadianceCaptureRequestError { inner })
    }

    /// Interpret one completed caller-owned RunenGPU readback through this exact occurrence.
    pub fn capture_radiance(
        &self,
        request: RenderRadianceCaptureRequest,
        context: &GpuContext,
        product_submission: &GpuSubmission,
    ) -> Result<RenderCapturedRadiance, RenderRadianceCaptureError> {
        crate::runtime::capture::capture_retained(
            &self.inner,
            &self.submission,
            request.inner,
            context,
            product_submission,
        )
        .map(|inner| RenderCapturedRadiance { inner })
        .map_err(|inner| RenderRadianceCaptureError { inner })
    }

    /// Execution-local physical object-identity decoder for one completed identity output.
    ///
    /// The returned codebook maps physical words to optional renderer object IDs only. It is not
    /// per-pixel definedness, miss/background evidence, or persistent identity.
    pub fn object_identity_decoder(
        &self,
        output_index: usize,
    ) -> Result<&RenderObjectIdentityDecoder, RenderObjectIdentityDecoderError> {
        match self.submission.status() {
            GpuSubmissionStatus::Accepted => {
                return Err(RenderObjectIdentityDecoderError::submission_pending());
            }
            GpuSubmissionStatus::Failed(failure) => {
                return Err(RenderObjectIdentityDecoderError::submission_failed(
                    failure.kind(),
                ));
            }
            GpuSubmissionStatus::Completed => {}
        }

        let admitted = self.inner.admitted().admitted();
        if !admitted
            .outputs()
            .iter()
            .any(|output| output.output_index() == output_index)
        {
            return Err(RenderObjectIdentityDecoderError::output_index_out_of_range());
        }
        let requested = admitted
            .plan()
            .request()
            .outputs()
            .get(output_index)
            .ok_or_else(RenderObjectIdentityDecoderError::output_index_out_of_range)?;
        if !matches!(
            requested.spec().value(),
            crate::request::RenderOutputValue::ObjectIdentity
        ) {
            return Err(RenderObjectIdentityDecoderError::output_not_object_identity());
        }
        Ok(self.inner.object_identity_decoder())
    }
}

/// Borrowed correlation for one prepared radiance destination and its public RunenGPU export.
#[derive(Debug, Clone, Copy)]
pub struct PreparedRadianceOutput<'a> {
    pub(super) inner: &'a PreparedDeterministicRadianceOutput,
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
    pub(super) inner: RenderDeterministicRadianceCaptureRequest,
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
    pub(super) inner: RenderCapturedDeterministicRadiance,
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
