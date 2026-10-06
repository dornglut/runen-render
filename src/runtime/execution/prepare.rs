use super::super::admission::AdmittedDeterministicRender;
use super::errors::{RenderDeterministicExecutionError, RenderDeterministicLoweringError};
use super::finalize::{LoweredDeterministicOutput, finalize_output};
use super::lifecycle::{
    DeterministicObservationIntent, DeterministicVerificationReadbacks,
    DeterministicVerificationSubmission, PreparedDeterministicRadianceOutput,
    PreparedDeterministicRender, RenderObjectIdentityDecoder, SubmittedDeterministicRender,
};
use super::output_context::{prepare_temporal_state, resolve_output_context};
use super::packing::{MaintainedExecutionKind, OutputPackingInput, pack_output};
use super::passes::{
    PreparedOutputPasses, prepare_primary_pass, prepare_requested_coverage, prepare_temporal_pass,
};
use super::state::{
    DeterministicOutputExecutionSelection, DeterministicRenderExecutionSelection,
    DeterministicResourceCache,
};
use super::submission::submit_prepared_deterministic_render;
use crate::admission::AdmittedRenderPlan;
use crate::lowering::RenderWorkSet;
use crate::scene::RenderObjectId;
use runen_gpu::GpuContext;
use std::collections::{BTreeMap, BTreeSet};

pub(super) struct LoweredDeterministicRender {
    pub(super) work_set: RenderWorkSet,
    pub(super) object_identity_decoder: RenderObjectIdentityDecoder,
    pub(super) verification_readbacks: Vec<DeterministicVerificationReadbacks>,
    pub(super) composable_radiance_outputs: Vec<PreparedDeterministicRadianceOutput>,
}

/// Submit one ordinary maintained deterministic render invocation.
///
/// Outputs are lowered independently from their exact semantic output/observation correlation, but
/// every resulting fragment is submitted together through exactly one `GpuContext::submit_work`.
/// Ordinary execution authors no CPU readback operations.
pub async fn submit_deterministic_render(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
) -> Result<SubmittedDeterministicRender, RenderDeterministicExecutionError> {
    let prepared = prepare_deterministic_render(admitted, context)?;
    submit_prepared_deterministic_render(prepared, context).await
}

/// Prepare one ordinary maintained deterministic render without submitting it.
///
/// This is the owner-controlled composition seam. It performs the same maintained lowering as
/// [`submit_deterministic_render`], authors no CPU readbacks, and exposes only renderer-authored
/// fragments plus typed correlation for ordinary R32Float radiance outputs.
pub fn prepare_deterministic_render(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
) -> Result<PreparedDeterministicRender, RenderDeterministicExecutionError> {
    let mut resources = DeterministicResourceCache::default();
    prepare_deterministic_render_with_cache(admitted, context, &mut resources)
}

/// Prepare one ordinary maintained deterministic render using continuity-local reusable resources.
pub(crate) fn prepare_deterministic_render_with_cache(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
    resources: &mut DeterministicResourceCache,
) -> Result<PreparedDeterministicRender, RenderDeterministicExecutionError> {
    prepare_deterministic_render_with_cache_and_evaluation(
        admitted, context, resources, None, false,
    )
}

pub(crate) fn prepare_deterministic_render_with_cache_and_evaluation(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
    resources: &mut DeterministicResourceCache,
    finite_evaluation: Option<(usize, (u32, u32))>,
    produce_requested_coverage: bool,
) -> Result<PreparedDeterministicRender, RenderDeterministicExecutionError> {
    let lowered = lower_deterministic_render(
        &admitted,
        context,
        DeterministicObservationIntent::Ordinary,
        resources,
        DeterministicRenderExecutionSelection {
            finite_evaluation,
            produce_requested_coverage,
        },
    )?;
    debug_assert!(lowered.verification_readbacks.is_empty());
    Ok(PreparedDeterministicRender {
        admitted,
        work_set: lowered.work_set,
        object_identity_decoder: lowered.object_identity_decoder,
        radiance_outputs: lowered.composable_radiance_outputs,
    })
}

/// Submit one maintained deterministic invocation with explicit verified-result intent.
///
/// This is the product-accessible counterpart to [`submit_deterministic_render`]. It runs the same
/// maintained evaluator/lowering and creates exactly one RunenGPU submission, adding only the
/// renderer-private readback operations required by RR566-EVAL-001. The returned type is the same
/// owner-controlled [`SubmittedDeterministicRender`]; private readback identities never become
/// public API. The caller drives RunenGPU progress and polls
/// [`SubmittedDeterministicRender::try_form_verified_result`] when semantic result evidence is
/// required.
pub async fn submit_deterministic_render_for_verified_result(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
) -> Result<
    SubmittedDeterministicRender,
    crate::runtime::verification::RenderDeterministicVerifiedSubmissionError,
> {
    crate::runtime::verification::submit_deterministic_render_for_verified_formation(
        admitted, context,
    )
    .await
    .map(DeterministicVerificationSubmission::into_submitted)
}

/// Submit the exact maintained deterministic path with renderer-private same-submission readbacks.
///
/// Static verifier eligibility is owned by `deterministic_verification` and must be established
/// before this function is called. The returned private witness wraps the same ordinary submitted
/// execution plus only the readback correlation authored before that submission.
pub(super) fn lower_deterministic_render(
    maintained: &AdmittedDeterministicRender,
    context: &GpuContext,
    intent: DeterministicObservationIntent,
    resources: &mut DeterministicResourceCache,
    execution: DeterministicRenderExecutionSelection,
) -> Result<LoweredDeterministicRender, RenderDeterministicLoweringError> {
    let DeterministicRenderExecutionSelection {
        finite_evaluation,
        produce_requested_coverage,
    } = execution;
    let admitted = maintained.admitted();
    if admitted.environment().affinity() != context.affinity() {
        return Err(RenderDeterministicLoweringError::ContextAffinityChanged {
            admitted: admitted.environment().affinity(),
            actual: context.affinity(),
        });
    }

    let object_identity_decoder = build_object_identity_decoder(admitted)?;
    let object_codes = object_identity_decoder
        .objects_by_code
        .iter()
        .copied()
        .enumerate()
        .map(|(index, object_id)| {
            let code = u32::try_from(index + 1).map_err(|_| {
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "object identity codebook",
                }
            })?;
            Ok((object_id, code))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;

    resources.discard_prepared_temporal_outputs();

    let mut fragments = Vec::new();
    fragments
        .try_reserve_exact(admitted.outputs().len())
        .map_err(|_| RenderDeterministicLoweringError::HostAllocation {
            field: "maintained output fragments",
        })?;
    let mut verification_readbacks = Vec::new();
    let mut composable_radiance_outputs = Vec::new();
    if intent.requires_private_readback() {
        verification_readbacks
            .try_reserve_exact(admitted.outputs().len())
            .map_err(|_| RenderDeterministicLoweringError::HostAllocation {
                field: "verification readback correlation",
            })?;
    }
    for output in admitted.outputs() {
        let lowered = lower_output(
            admitted,
            output.output_index(),
            &object_codes,
            context,
            resources,
            intent,
            DeterministicOutputExecutionSelection {
                finite_evaluation_extent: finite_evaluation.and_then(
                    |(selected_output, extent)| {
                        (selected_output == output.output_index()).then_some(extent)
                    },
                ),
                produce_requested_coverage,
            },
        )?;
        fragments.push(lowered.fragment);
        if let Some(readbacks) = lowered.verification_readbacks {
            verification_readbacks.push(readbacks);
        }
        if let Some(output) = lowered.composable_radiance_output {
            composable_radiance_outputs.push(output);
        }
    }

    Ok(LoweredDeterministicRender {
        work_set: RenderWorkSet::from_lowering(admitted, fragments),
        object_identity_decoder,
        verification_readbacks,
        composable_radiance_outputs,
    })
}

pub(super) fn build_object_identity_decoder(
    admitted: &AdmittedRenderPlan,
) -> Result<RenderObjectIdentityDecoder, RenderDeterministicLoweringError> {
    let objects = admitted
        .outputs()
        .iter()
        .flat_map(|output| output.object_representations())
        .map(|object| object.object_id())
        .collect::<BTreeSet<_>>();
    let object_count = u64::try_from(objects.len()).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "object identity codebook",
        }
    })?;
    if object_count > u64::from(u32::MAX) {
        return Err(RenderDeterministicLoweringError::SizeOverflow {
            field: "object identity codebook",
        });
    }
    Ok(RenderObjectIdentityDecoder {
        objects_by_code: objects.into_iter().collect(),
    })
}

pub(super) fn lower_output(
    admitted: &AdmittedRenderPlan,
    output_index: usize,
    object_codes: &BTreeMap<RenderObjectId, u32>,
    context: &GpuContext,
    resources: &mut DeterministicResourceCache,
    intent: DeterministicObservationIntent,
    execution: DeterministicOutputExecutionSelection,
) -> Result<LoweredDeterministicOutput, RenderDeterministicLoweringError> {
    let resolved = resolve_output_context(admitted, output_index, context, execution)?;
    let temporal_state = prepare_temporal_state(admitted, resolved, resources)?;
    let packed = pack_output(
        admitted,
        resolved.admitted_output,
        MaintainedExecutionKind::Semantic(resolved.requested.spec().value()),
        resolved.observation,
        object_codes,
        OutputPackingInput {
            finite_evaluation_extent: resolved.finite_evaluation_extent,
            bytes_per_row_alignment: resolved.bytes_per_row_alignment,
            temporal: temporal_state.packing_facts(),
        },
    )?;
    let requested_extent = resolved
        .requested
        .spec()
        .topology()
        .sample_lattice_dimensions();
    let requested_coverage = if resolved.produce_requested_coverage
        && resolved
            .finite_evaluation_extent
            .is_some_and(|extent| Some(extent) != requested_extent)
    {
        let coverage_packed = pack_output(
            admitted,
            resolved.admitted_output,
            MaintainedExecutionKind::RequestedCoverage,
            resolved.observation,
            object_codes,
            OutputPackingInput {
                finite_evaluation_extent: None,
                bytes_per_row_alignment: resolved.bytes_per_row_alignment,
                temporal: temporal_state.packing_facts(),
            },
        )?;
        Some(prepare_requested_coverage(
            coverage_packed,
            resolved.max_compute_workgroups_per_dimension,
            resources,
            resolved.output_index,
        )?)
    } else {
        None
    };

    let primary = prepare_primary_pass(&packed, resolved, resources)?;
    let temporal = prepare_temporal_pass(&packed, resolved, &temporal_state, &primary, resources)?;
    finalize_output(
        admitted,
        resolved,
        &packed,
        &temporal_state,
        PreparedOutputPasses {
            requested_coverage,
            primary,
            temporal,
        },
        intent,
    )
}
