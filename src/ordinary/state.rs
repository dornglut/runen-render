use super::*;

/// Stable renderer-owned namespace for retained execution state.
///
/// This scopes reusable renderer resources and temporal history. It is not a scene, object,
/// RunenGPU resource, or submission identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderExecutionScope(u64);

impl RenderExecutionScope {
    pub const fn new(raw: u64) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u64 {
        self.0
    }
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

/// Stateful ordinary integration for hosts that compose renderer-authored work into a larger
/// public RunenGPU submission.
///
/// It retains only renderer-derived reusable resources and temporal history. Planning, admission,
/// compatibility, and lowering are the same authority used by the one-shot ordinary path.
#[derive(Debug, Default)]
pub struct RenderExecutionState {
    inner: DeterministicResourceCache,
}

impl RenderExecutionState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn has_in_flight_scopes(
        &self,
        scopes: impl IntoIterator<Item = RenderExecutionScope>,
    ) -> bool {
        self.inner
            .any_producer_submission_in_flight(scopes.into_iter().map(RenderExecutionScope::raw))
    }

    pub fn retain_in_flight_submissions(&mut self) {
        self.inner.retain_in_flight_submissions();
    }

    pub fn record_submission(&mut self, scope: RenderExecutionScope, submission: &GpuSubmission) {
        self.inner
            .record_producer_submission(scope.raw(), 0, submission);
    }

    pub fn prepare(
        &mut self,
        admitted: AdmittedRender,
        context: &GpuContext,
        scope: RenderExecutionScope,
        evaluation: Option<RenderEvaluationSelection>,
    ) -> Result<PreparedRender, RenderExecutionError> {
        let finite_evaluation =
            evaluation.map(|selection| (selection.output_index(), selection.extent()));
        prepare_deterministic_render_with_cache_in_scope_and_evaluation(
            admitted.inner,
            context,
            &mut self.inner,
            scope.raw(),
            finite_evaluation,
            false,
        )
        .map(|inner| PreparedRender { inner })
        .map_err(|inner| RenderExecutionError { inner })
    }
}
