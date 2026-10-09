use super::super::program::{
    CAMERA_REPROJECTION_REVISION, TEMPORAL_RECONSTRUCTION_REVISION,
    abi::{camera, temporal},
};
use super::errors::RenderDeterministicLoweringError;
use super::packing::OutputTemporalPackingFacts;
use super::state::{
    DeterministicOutputExecutionSelection, DeterministicResourceCache,
    DeterministicTemporalHistorySelection, DeterministicTemporalHistoryUse,
    DeterministicTemporalHistoryUseStorage, DeterministicTemporalSignature,
    temporal_evaluation_extent_supported, temporal_observation_compatibility,
    temporal_phase_mapping_is_injective,
};
use crate::admission::AdmittedRenderPlan;
use crate::request::{RenderObservationSpec, RenderOutputValue};
use runen_gpu::GpuContext;
use std::num::NonZeroU64;

#[derive(Clone, Copy)]
pub(super) struct ResolvedOutputContext<'a> {
    pub(super) output_index: usize,
    pub(super) admitted_output: &'a crate::admission::RenderAdmittedOutput,
    pub(super) requested: crate::request::RenderRequestedOutput,
    pub(super) observation: RenderObservationSpec,
    pub(super) graph_wiring_namespace: NonZeroU64,
    pub(super) finite_evaluation_extent: Option<(u32, u32)>,
    pub(super) produce_requested_coverage: bool,
    pub(super) bytes_per_row_alignment: Option<u64>,
    pub(super) max_compute_workgroups_per_dimension: u32,
    pub(super) max_storage_buffer_binding_size: u64,
    pub(super) max_buffer_size: u64,
}

pub(super) struct PreparedTemporalState {
    pub(super) history: Option<DeterministicTemporalHistoryUse>,
}

impl PreparedTemporalState {
    pub(super) fn packing_facts(&self) -> Option<OutputTemporalPackingFacts> {
        self.history
            .as_ref()
            .map(|history| OutputTemporalPackingFacts {
                phase: history.phase,
                age: history.age,
                static_history_row_stride_words: match &history.storage {
                    DeterministicTemporalHistoryUseStorage::Static {
                        row_stride_words, ..
                    } => Some(*row_stride_words),
                    DeterministicTemporalHistoryUseStorage::Camera { .. } => None,
                },
            })
    }
}

pub(super) fn resolve_output_context<'a>(
    admitted: &'a AdmittedRenderPlan,
    output_index: usize,
    context: &GpuContext,
    execution: DeterministicOutputExecutionSelection,
) -> Result<ResolvedOutputContext<'a>, RenderDeterministicLoweringError> {
    let DeterministicOutputExecutionSelection {
        graph_wiring_namespace,
        finite_evaluation_extent,
        produce_requested_coverage,
    } = execution;
    let admitted_output = admitted
        .outputs()
        .iter()
        .find(|output| output.output_index() == output_index)
        .ok_or(RenderDeterministicLoweringError::OutputCorrelationChanged { output_index })?;
    let requested = admitted
        .plan()
        .request()
        .outputs()
        .get(output_index)
        .copied()
        .ok_or(RenderDeterministicLoweringError::OutputCorrelationChanged { output_index })?;
    if requested.observation_index() != admitted_output.observation_index() {
        return Err(RenderDeterministicLoweringError::OutputCorrelationChanged { output_index });
    }
    let observation = admitted
        .plan()
        .request()
        .observations()
        .get(requested.observation_index())
        .copied()
        .ok_or(RenderDeterministicLoweringError::OutputCorrelationChanged { output_index })?;

    Ok(ResolvedOutputContext {
        output_index,
        admitted_output,
        requested,
        observation,
        graph_wiring_namespace,
        finite_evaluation_extent,
        produce_requested_coverage,
        bytes_per_row_alignment: context
            .device_facts()
            .device_limits()
            .alignments()
            .bytes_per_row,
        max_compute_workgroups_per_dimension: context
            .device_facts()
            .workload_budget()
            .limits()
            .max_compute_workgroups_per_dimension(),
        max_storage_buffer_binding_size: context
            .device_facts()
            .workload_budget()
            .limits()
            .max_storage_buffer_binding_size()
            .min(context.device_facts().device_limits().values().max_storage_buffer_binding_size()),
        max_buffer_size: context
            .device_facts()
            .workload_budget()
            .limits()
            .max_buffer_size()
            .min(context.device_facts().device_limits().values().max_buffer_size()),
    })
}

pub(super) fn prepare_temporal_state(
    admitted: &AdmittedRenderPlan,
    resolved: ResolvedOutputContext<'_>,
    resources: &mut DeterministicResourceCache,
) -> Result<PreparedTemporalState, RenderDeterministicLoweringError> {
    let Some(evaluation_extent) = resolved.finite_evaluation_extent else {
        return Ok(PreparedTemporalState { history: None });
    };
    let RenderObservationSpec::Perspective(perspective) = resolved.observation else {
        return Err(RenderDeterministicLoweringError::UnsupportedOutput {
            output_index: resolved.output_index,
        });
    };
    if !perspective.sampling_support().is_perspective_lattice_cell()
        || !matches!(
            resolved.requested.spec().value(),
            RenderOutputValue::Radiance { .. }
        )
    {
        return Err(RenderDeterministicLoweringError::UnsupportedOutput {
            output_index: resolved.output_index,
        });
    }
    let requested_extent = resolved
        .requested
        .spec()
        .topology()
        .sample_lattice_dimensions()
        .ok_or(RenderDeterministicLoweringError::UnsupportedOutput {
            output_index: resolved.output_index,
        })?;
    if !temporal_evaluation_extent_supported(requested_extent, evaluation_extent) {
        return Err(
            RenderDeterministicLoweringError::UnsupportedTemporalEvaluationExtent {
                output_index: resolved.output_index,
                requested_extent,
                evaluation_extent,
            },
        );
    }
    if requested_extent != evaluation_extent
        && (!temporal_phase_mapping_is_injective(requested_extent, evaluation_extent)
            || !temporal_phase_mapping_is_injective(requested_extent, requested_extent))
    {
        return Err(RenderDeterministicLoweringError::NonInjectiveTemporalPhaseMapping {
            output_index: resolved.output_index,
            requested_extent,
            evaluation_extent,
        });
    }
    let alignment = resolved
        .bytes_per_row_alignment
        .ok_or(RenderDeterministicLoweringError::MissingBytesPerRowAlignment)?;
    for binding in admitted.surface_semantic_inputs() {
        if binding.generation().is_none() {
            return Err(
                RenderDeterministicLoweringError::MissingTemporalSurfaceInputGeneration {
                    output_index: resolved.output_index,
                    representation_id: binding.representation_id(),
                },
            );
        }
    }
    for binding in admitted.field_semantic_inputs() {
        if binding.generation().is_none() {
            return Err(
                RenderDeterministicLoweringError::MissingTemporalFieldInputGeneration {
                    output_index: resolved.output_index,
                    representation_id: binding.representation_id(),
                },
            );
        }
    }
    let camera_capable = evaluation_extent == requested_extent;
    let signature = DeterministicTemporalSignature {
        scene_revision: admitted.scene_revision(),
        observation: temporal_observation_compatibility(resolved.observation, camera_capable),
        output: resolved.requested.spec(),
        semantic_inputs: admitted.surface_semantic_inputs().to_vec(),
        field_semantic_inputs: admitted.field_semantic_inputs().to_vec(),
        evaluation_extent,
        sequence_revision: temporal::SEQUENCE_REVISION,
        reconstruction_revision: TEMPORAL_RECONSTRUCTION_REVISION,
        camera_reprojection_revision: camera_capable.then_some(CAMERA_REPROJECTION_REVISION),
        depth_policy_revision: camera_capable.then_some(camera::DEPTH_POLICY_REVISION),
    };
    let history = resources.temporal_history(
        resolved.output_index,
        signature,
        requested_extent,
        alignment,
        DeterministicTemporalHistorySelection {
            current_observation: perspective,
            camera_capable,
        },
    )?;
    Ok(PreparedTemporalState {
        history: Some(history),
    })
}
