use super::*;

/// Physical object-identity decoder owned by one exact maintained execution.
///
/// Compact GPU words are not renderer-semantic object identities. Code zero and every unknown code
/// decode to `None`; a non-zero code is meaningful only through this exact immutable codebook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderObjectIdentityDecoder {
    pub(super) objects_by_code: Vec<RenderObjectId>,
}

impl RenderObjectIdentityDecoder {
    pub fn decode(&self, code: u32) -> Option<RenderObjectId> {
        let index = usize::try_from(code.checked_sub(1)?).ok()?;
        self.objects_by_code.get(index).copied()
    }
}

/// One renderer-owned, ordinary maintained execution prepared for composition.
///
/// Construction remains private to the maintained lowerer. The prepared fragments are the same
/// backend-neutral work that ordinary submission uses; the only additional public correlation is
/// for an admitted composable R32Float radiance destination.
#[derive(Debug, Clone)]
pub struct PreparedDeterministicRender {
    pub(super) admitted: AdmittedDeterministicRender,
    pub(super) work_set: RenderWorkSet,
    pub(super) object_identity_decoder: RenderObjectIdentityDecoder,
    pub(super) radiance_outputs: Vec<PreparedDeterministicRadianceOutput>,
}

impl PreparedDeterministicRender {
    pub const fn admitted(&self) -> &AdmittedDeterministicRender {
        &self.admitted
    }

    pub const fn work_set(&self) -> &RenderWorkSet {
        &self.work_set
    }

    pub fn radiance_outputs(&self) -> &[PreparedDeterministicRadianceOutput] {
        &self.radiance_outputs
    }

    pub fn radiance_output(
        &self,
        output_index: usize,
    ) -> Option<&PreparedDeterministicRadianceOutput> {
        self.radiance_outputs
            .iter()
            .find(|output| output.output_index() == output_index)
    }

    pub(crate) fn into_associated(self) -> AssociatedDeterministicRender {
        AssociatedDeterministicRender {
            admitted: self.admitted,
            object_identity_decoder: self.object_identity_decoder,
        }
    }
}

/// Renderer-private post-association metadata needed to interpret one exact execution's outputs.
///
/// Prepared work and export relationships are deliberately dropped at association. The caller
/// already owns the RunenGPU submission; retained observation needs only the admitted semantic
/// contract and the execution-local physical identity codebook.
#[derive(Debug)]
pub(crate) struct AssociatedDeterministicRender {
    admitted: AdmittedDeterministicRender,
    object_identity_decoder: RenderObjectIdentityDecoder,
}

impl AssociatedDeterministicRender {
    pub(crate) const fn admitted(&self) -> &AdmittedDeterministicRender {
        &self.admitted
    }

    pub(crate) const fn object_identity_decoder(&self) -> &RenderObjectIdentityDecoder {
        &self.object_identity_decoder
    }
}

/// Prepared current coverage work, correlated with the enclosing temporal contribution's inputs.
///
/// This records preparation, not GPU completion or CPU-observed coverage availability. The carrier
/// becomes current execution evidence only after its owning producer submission completes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderRequestedCoveragePreparation {
    pub extent: (u32, u32),
    pub policy_revision: u32,
    pub evaluator_revision: u64,
}

/// Bounded renderer-owned evidence for one footprint-reconstruction preparation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderTemporalExecutionEvidence {
    pub requested_extent: (u32, u32),
    pub evaluation_extent: (u32, u32),
    pub current_coverage: Option<RenderRequestedCoveragePreparation>,
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

/// Renderer-owned correlation for one ordinary composable radiance output.
///
/// The correlation carries the exact admitted destination and typed RunenGPU export relationship.
/// Consumers import it through [`Self::import`] and never derive an export key or inspect the
/// maintained evaluator's private carrier.
#[derive(Debug, Clone)]
pub struct PreparedDeterministicRadianceOutput {
    pub(super) output_index: usize,
    pub(super) relationship: GpuExportRelationship,
    pub(super) temporal_evidence: Option<RenderTemporalExecutionEvidence>,
}

impl PreparedDeterministicRadianceOutput {
    pub const fn output_index(&self) -> usize {
        self.output_index
    }

    pub fn resource(&self) -> &GpuResourceRef {
        self.relationship.resource()
    }

    pub fn texture(&self) -> Option<&GpuTextureHandle> {
        match self.resource() {
            GpuResourceRef::Texture(texture) => Some(texture),
            _ => None,
        }
    }

    pub fn export_relationship(&self) -> &GpuExportRelationship {
        &self.relationship
    }

    pub fn temporal_execution_evidence(&self) -> Option<&RenderTemporalExecutionEvidence> {
        self.temporal_evidence.as_ref()
    }

    pub fn import(&self, provenance: GpuResourceProvenance) -> GpuWorkImport {
        GpuWorkImport::new(
            self.relationship.resource().clone(),
            self.relationship.export_key().clone(),
            GpuResourceAccessIntent::Read,
            provenance,
        )
    }
}

/// Renderer-private correlation between one admitted output and the three observations required by
/// RR566-EVAL-001. The IDs are process-local RunenGPU correlation values bound to one exact
/// `GpuSubmission`; they are not semantic identity and are never exposed as public renderer state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeterministicVerificationReadbacks {
    pub(super) output_index: usize,
    pub(super) canonical_output: GpuReadbackId,
    pub(super) definedness: GpuReadbackId,
    pub(super) status: GpuReadbackId,
}

impl DeterministicVerificationReadbacks {
    pub(crate) const fn output_index(self) -> usize {
        self.output_index
    }

    pub(crate) const fn canonical_output(self) -> GpuReadbackId {
        self.canonical_output
    }

    pub(crate) const fn definedness(self) -> GpuReadbackId {
        self.definedness
    }

    pub(crate) const fn status(self) -> GpuReadbackId {
        self.status
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeterministicObservationIntent {
    Ordinary,
    Verify,
}

impl DeterministicObservationIntent {
    pub(super) const fn requires_private_readback(self) -> bool {
        matches!(self, Self::Verify)
    }
}

#[derive(Debug)]
pub(super) enum DeterministicVerificationState {
    NotRequested,
    Requested(Vec<DeterministicVerificationReadbacks>),
    Formed,
}

/// One exact maintained execution after RunenGPU accepted its authored work.
///
/// This is not a second submission lifecycle or a render session. Physical completion/failure stays
/// entirely on [`GpuSubmission`]. Ordinary execution retains no verification correlation and cannot
/// be upgraded after submission. When verified-result intent was selected before submission, the
/// same public submitted-render abstraction privately retains only the same-submission correlation
/// needed by RR566-EVAL-001 and FORM-001.
#[derive(Debug)]
pub struct SubmittedDeterministicRender {
    pub(super) admitted: AdmittedDeterministicRender,
    pub(super) submission: GpuSubmission,
    pub(super) object_identity_decoder: RenderObjectIdentityDecoder,
    pub(super) verification: DeterministicVerificationState,
}

impl SubmittedDeterministicRender {
    pub const fn admitted(&self) -> &AdmittedDeterministicRender {
        &self.admitted
    }

    /// Current physical status of the exact RunenGPU submission.
    ///
    /// The underlying submission and its readback collection remain renderer-private so verified
    /// result formation does not leak private observation identities or bytes through RunenRender.
    pub fn submission_status(&self) -> GpuSubmissionStatus {
        self.submission.status()
    }

    pub(crate) const fn submission(&self) -> &GpuSubmission {
        &self.submission
    }

    pub(crate) const fn result_is_formed(&self) -> bool {
        matches!(self.verification, DeterministicVerificationState::Formed)
    }

    /// Mint one exact public readback correlation for a formed maintained radiance lattice.
    pub fn request_deterministic_radiance_capture(
        &self,
        output_index: usize,
    ) -> Result<
        RenderDeterministicRadianceCaptureRequest,
        RenderDeterministicRadianceCaptureRequestError,
    > {
        crate::runtime::capture::mint_request(self, output_index)
    }

    /// Consume one capture request and interpret its completed product-owned public readback.
    pub fn capture_deterministic_radiance(
        &self,
        request: RenderDeterministicRadianceCaptureRequest,
        context: &GpuContext,
        product_submission: &GpuSubmission,
    ) -> Result<RenderCapturedDeterministicRadiance, RenderDeterministicRadianceCaptureError> {
        crate::runtime::capture::capture(self, request, context, product_submission)
    }

    pub const fn object_identity_decoder(&self) -> &RenderObjectIdentityDecoder {
        &self.object_identity_decoder
    }

    /// Try to form semantic result evidence from this exact verified submission.
    ///
    /// This method never drives RunenGPU progress and never blocks waiting for readback. The caller
    /// retains product/runtime policy for progressing the public RunenGPU context, then polls this
    /// owner-controlled boundary. `Ok(None)` means the exact submission or one of its private
    /// same-submission observations is still pending. Successful formation consumes the private
    /// verification authority exactly once while leaving physical submission status available.
    pub fn try_form_verified_result(
        &mut self,
    ) -> Result<Option<RenderResult>, RenderDeterministicResultFormationError> {
        let verification_readbacks = match &self.verification {
            DeterministicVerificationState::NotRequested => {
                return Err(RenderDeterministicResultFormationError::VerificationNotRequested);
            }
            DeterministicVerificationState::Formed => {
                return Err(RenderDeterministicResultFormationError::ResultAlreadyFormed);
            }
            DeterministicVerificationState::Requested(readbacks) => readbacks.clone(),
        };

        match self.submission.status() {
            GpuSubmissionStatus::Accepted => return Ok(None),
            GpuSubmissionStatus::Failed(failure) => {
                return Err(RenderDeterministicResultFormationError::SubmissionFailed {
                    kind: failure.kind(),
                });
            }
            GpuSubmissionStatus::Completed => {}
        }

        for correlation in &verification_readbacks {
            for (channel, id) in [
                ("canonical-output", correlation.canonical_output()),
                ("definedness", correlation.definedness()),
                ("evaluator-status", correlation.status()),
            ] {
                let readback = self.submission.readback(id).ok_or(
                    RenderDeterministicResultFormationError::ReadbackCorrelationLost {
                        output_index: correlation.output_index(),
                        channel,
                    },
                )?;
                match readback.status() {
                    GpuReadbackStatus::Pending => return Ok(None),
                    GpuReadbackStatus::Failed(failure) => {
                        return Err(RenderDeterministicResultFormationError::ReadbackFailed {
                            output_index: correlation.output_index(),
                            channel,
                            kind: failure.kind(),
                        });
                    }
                    GpuReadbackStatus::Ready(_) => {}
                }
            }
        }

        let verification = DeterministicVerificationSubmission {
            submitted: SubmittedDeterministicRender {
                admitted: self.admitted.clone(),
                submission: self.submission.clone(),
                object_identity_decoder: self.object_identity_decoder.clone(),
                verification: DeterministicVerificationState::Requested(verification_readbacks),
            },
        };
        let formation_evidence =
            crate::runtime::verification::verify_completed_deterministic_render(verification)
                .map_err(RenderDeterministicResultFormationError::Verification)?;
        let result = RenderResult::from_formation_evidence(formation_evidence);
        self.verification = DeterministicVerificationState::Formed;
        Ok(Some(result))
    }
}

/// Private proof witness for a submission whose verification intent was selected before lowering.
///
/// This wrapper exists only so the private verifier can consume the exact owner-controlled submitted
/// value. The public submitted render itself privately retains the correlation; there is no second
/// submission lifecycle and no public verification token.
#[derive(Debug)]
pub(crate) struct DeterministicVerificationSubmission {
    pub(super) submitted: SubmittedDeterministicRender,
}

impl DeterministicVerificationSubmission {
    pub(crate) const fn submitted(&self) -> &SubmittedDeterministicRender {
        &self.submitted
    }

    pub(crate) fn readbacks(&self) -> &[DeterministicVerificationReadbacks] {
        match &self.submitted.verification {
            DeterministicVerificationState::Requested(readbacks) => readbacks,
            DeterministicVerificationState::NotRequested
            | DeterministicVerificationState::Formed => &[],
        }
    }

    pub(crate) fn into_submitted(self) -> SubmittedDeterministicRender {
        self.submitted
    }
}
