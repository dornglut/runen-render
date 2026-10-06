use super::*;
use std::sync::{
    Arc, Weak,
    atomic::{AtomicU64, Ordering},
};

const FIRST_RENDER_EXECUTION_SESSION_SCOPE: u64 = 1;
static NEXT_RENDER_EXECUTION_SESSION_SCOPE: AtomicU64 =
    AtomicU64::new(FIRST_RENDER_EXECUTION_SESSION_SCOPE);

fn allocate_render_execution_session_scope() -> Option<u64> {
    NEXT_RENDER_EXECUTION_SESSION_SCOPE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            (current != 0).then_some(if current == u64::MAX { 0 } else { current + 1 })
        })
        .ok()
}
/// Optional finite evaluation selection for one requested output.
///
/// This changes bounded physical work only; it does not change the semantic request topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderEvaluationSelection {
    output_index: usize,
    extent: (u32, u32),
}

impl RenderEvaluationSelection {
    pub fn new(output_index: usize, width: u32, height: u32) -> Option<Self> {
        (width > 0 && height > 0).then_some(Self {
            output_index,
            extent: (width, height),
        })
    }

    pub const fn output_index(self) -> usize {
        self.output_index
    }

    pub const fn extent(self) -> (u32, u32) {
        self.extent
    }
}

/// Public renderer-semantic temporal execution evidence.
///
/// This deliberately excludes renderer-private requested-lattice coverage preparation. Coverage is
/// an internal reconstruction prerequisite, not semantic depth/output authority and not part of the
/// transferable ordinary API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderTemporalExecutionEvidence {
    pub requested_extent: (u32, u32),
    pub evaluation_extent: (u32, u32),
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

impl RenderTemporalExecutionEvidence {
    pub(super) fn from_deterministic(evidence: &DeterministicTemporalExecutionEvidence) -> Self {
        Self {
            requested_extent: evidence.requested_extent,
            evaluation_extent: evidence.evaluation_extent,
            semantic_input_generations: evidence.semantic_input_generations.clone(),
            field_semantic_input_generations: evidence.field_semantic_input_generations.clone(),
            sequence_revision: evidence.sequence_revision,
            reconstruction_revision: evidence.reconstruction_revision,
            phase: evidence.phase,
            history_generation: evidence.history_generation,
            history_age: evidence.history_age,
            history_reset: evidence.history_reset,
            camera_reprojection_eligible: evidence.camera_reprojection_eligible,
            previous_observation_available: evidence.previous_observation_available,
            camera_pose_changed: evidence.camera_pose_changed,
            camera_same_pose_completed_frames: evidence.camera_same_pose_completed_frames,
            camera_reprojection_revision: evidence.camera_reprojection_revision,
            depth_policy_revision: evidence.depth_policy_revision,
        }
    }
}
/// Retained execution owner for one independent renderer continuity.
///
/// One session owns renderer-derived reusable resources and temporal history for one logical
/// continuity. Product producer, surface, view, ECS, window, and presentation identities remain
/// outside RunenRender and may map to this owner only in downstream integration.
///
/// At most one prepared occurrence or accepted in-flight submission may exist at a time. Dropping
/// an unaccepted occurrence abandons its provisional temporal transition; the next reconciliation
/// or preparation discards that provisional state before continuing.
#[derive(Debug)]
pub struct RenderExecutionSession {
    identity: Arc<()>,
    scope: u64,
    resources: DeterministicResourceCache,
    affinity: Option<GpuContextAffinity>,
    prepared_occurrence: Option<Weak<()>>,
    submission: Option<GpuSubmission>,
}

impl RenderExecutionSession {
    /// Creates one independent retained renderer continuity owner.
    pub fn new() -> Result<Self, RenderExecutionSessionError> {
        let scope = allocate_render_execution_session_scope()
            .ok_or(RenderExecutionSessionError::IdentityExhausted)?;
        Ok(Self {
            identity: Arc::new(()),
            scope,
            resources: DeterministicResourceCache::default(),
            affinity: None,
            prepared_occurrence: None,
            submission: None,
        })
    }

    /// Returns whether this continuity currently has accepted GPU work still in flight.
    #[must_use]
    pub fn is_in_flight(&self) -> bool {
        self.submission
            .as_ref()
            .is_some_and(|submission| matches!(submission.status(), GpuSubmissionStatus::Accepted))
    }

    /// Reconciles terminal accepted execution and abandoned preparation without driving RunenGPU.
    ///
    /// Callers remain responsible for progressing their public GpuContext.
    pub fn reconcile(&mut self) {
        self.reconcile_terminal_submission();
        self.discard_abandoned_occurrence();
    }

    /// Prepares one exact retained occurrence for caller-owned RunenGPU composition.
    pub fn prepare(
        &mut self,
        admitted: AdmittedRender,
        context: &GpuContext,
        evaluation: Option<RenderEvaluationSelection>,
    ) -> Result<PreparedRenderOccurrence, RenderExecutionSessionError> {
        self.reconcile();

        if self.is_in_flight() {
            return Err(RenderExecutionSessionError::SubmissionInFlight);
        }
        if self
            .prepared_occurrence
            .as_ref()
            .and_then(Weak::upgrade)
            .is_some()
        {
            return Err(RenderExecutionSessionError::PreparedOccurrenceOutstanding);
        }

        let affinity = context.affinity();
        match self.affinity {
            None => self.affinity = Some(affinity),
            Some(previous) if previous.context() != affinity.context() => {
                return Err(RenderExecutionSessionError::ForeignContext {
                    expected: previous,
                    actual: affinity,
                });
            }
            Some(previous) if previous.generation() != affinity.generation() => {
                self.resources.reset_for_context_change();
                self.affinity = Some(affinity);
            }
            Some(_) => {}
        }

        let finite_evaluation =
            evaluation.map(|selection| (selection.output_index(), selection.extent()));
        let inner = match prepare_deterministic_render_with_cache_in_scope_and_evaluation(
            admitted.inner,
            context,
            &mut self.resources,
            self.scope,
            finite_evaluation,
            false,
        ) {
            Ok(inner) => inner,
            Err(inner) => {
                self.resources.discard_prepared_temporal_outputs(self.scope);
                return Err(RenderExecutionSessionError::Execution(
                    RenderExecutionError { inner },
                ));
            }
        };

        let occurrence_identity = Arc::new(());
        self.prepared_occurrence = Some(Arc::downgrade(&occurrence_identity));
        Ok(PreparedRenderOccurrence {
            inner,
            session_identity: Arc::clone(&self.identity),
            occurrence_identity,
            affinity,
        })
    }

    /// Associates one exact prepared occurrence with a caller-owned RunenGPU submission.
    ///
    /// The occurrence is consumed so association is single-use. The submission may contain
    /// additional caller-owned work, but it must use the same RunenGPU affinity and contain every
    /// renderer-authored work node from this occurrence.
    pub fn accept(
        &mut self,
        occurrence: PreparedRenderOccurrence,
        submission: &GpuSubmission,
    ) -> Result<(), RenderExecutionSessionError> {
        self.reconcile();

        if self.is_in_flight() {
            return Err(RenderExecutionSessionError::SubmissionInFlight);
        }

        let Some(current_occurrence) = self.prepared_occurrence.as_ref().and_then(Weak::upgrade)
        else {
            return Err(RenderExecutionSessionError::OccurrenceNotCurrent);
        };
        if !Arc::ptr_eq(&self.identity, &occurrence.session_identity)
            || !Arc::ptr_eq(&current_occurrence, &occurrence.occurrence_identity)
        {
            return Err(RenderExecutionSessionError::OccurrenceNotCurrent);
        }

        if submission.affinity() != occurrence.affinity {
            return Err(RenderExecutionSessionError::SubmissionAffinityMismatch {
                expected: occurrence.affinity,
                actual: submission.affinity(),
            });
        }

        let mut required_work = 0usize;
        for fragment in occurrence.inner.work_set().fragments() {
            for node in fragment.nodes() {
                required_work = required_work.saturating_add(1);
                if !submission.contains_work_node(node.id()) {
                    return Err(RenderExecutionSessionError::SubmissionMissingRendererWork);
                }
            }
        }
        if required_work == 0 {
            return Err(RenderExecutionSessionError::SubmissionMissingRendererWork);
        }

        self.prepared_occurrence = None;
        self.submission = Some(submission.clone());
        self.reconcile_terminal_submission();
        Ok(())
    }

    fn discard_abandoned_occurrence(&mut self) {
        let abandoned = self
            .prepared_occurrence
            .as_ref()
            .is_some_and(|identity| identity.upgrade().is_none());
        if abandoned {
            self.resources.discard_prepared_temporal_outputs(self.scope);
            self.prepared_occurrence = None;
        }
    }

    fn reconcile_terminal_submission(&mut self) {
        let completed = match self.submission.as_ref().map(GpuSubmission::status) {
            None | Some(GpuSubmissionStatus::Accepted) => return,
            Some(GpuSubmissionStatus::Completed) => true,
            Some(GpuSubmissionStatus::Failed(_)) => false,
        };
        self.resources
            .reconcile_temporal_outputs(self.scope, completed);
        self.submission = None;
    }
}
