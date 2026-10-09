use super::super::program::{
    CAMERA_REPROJECTION_REVISION, MAINTAINED_EVALUATOR_REVISION, TEMPORAL_RECONSTRUCTION_REVISION,
    abi::{camera, requested_coverage, temporal},
};
use super::WORD_BYTES;
use super::errors::{
    RenderDeterministicLoweringError, gpu_readback_request, gpu_resource_descriptor,
    gpu_work_authoring, gpu_work_operation,
};
use super::lifecycle::{
    DeterministicObservationIntent, DeterministicVerificationReadbacks,
    PreparedDeterministicRadianceOutput, RenderRequestedCoveragePreparation,
    RenderTemporalExecutionEvidence,
};
use super::output_context::{PreparedTemporalState, ResolvedOutputContext};
use super::packing::PackedOutput;
use super::passes::{
    PreparedOutputPasses, PreparedPrimaryPass, PreparedRequestedCoverage, PreparedTemporalPass,
};
use super::state::DeterministicTemporalHistoryUseStorage;
use crate::admission::{AdmittedRenderPlan, RenderOutputDestination};
use crate::request::RenderOutputValue;
use runen_gpu::{
    GpuBufferCoverage, GpuBufferRegion, GpuBufferTextureLayout, GpuCopyOperation, GpuExportKey,
    GpuExportRelationship, GpuInitialCoverage, GpuReadbackOperation, GpuResourceAccessIntent,
    GpuResourceProvenance, GpuResourceRef, GpuTextureAccessResource, GpuTextureCopyRegion,
    GpuTextureFormat, GpuWorkFragment, GpuWorkOutput,
};

pub(super) struct LoweredDeterministicOutput {
    pub(super) fragment: GpuWorkFragment,
    pub(super) verification_readbacks: Option<DeterministicVerificationReadbacks>,
    pub(super) composable_radiance_output: Option<PreparedDeterministicRadianceOutput>,
}

struct VerificationReadbackOperations {
    correlation: DeterministicVerificationReadbacks,
    canonical_output: GpuReadbackOperation,
    definedness: GpuReadbackOperation,
    status: GpuReadbackOperation,
}

struct PreparedDestination {
    copy: GpuCopyOperation,
    gpu_output: Option<GpuWorkOutput>,
    availability_output: Option<GpuWorkOutput>,
    radiance_output: Option<PreparedDeterministicRadianceOutput>,
}

fn temporal_execution_evidence(
    admitted: &AdmittedRenderPlan,
    resolved: ResolvedOutputContext<'_>,
    temporal_state: &PreparedTemporalState,
    requested_coverage: Option<&PreparedRequestedCoverage>,
) -> Option<RenderTemporalExecutionEvidence> {
    let history = temporal_state.history.as_ref()?;
    Some(RenderTemporalExecutionEvidence {
        requested_extent: resolved
            .requested
            .spec()
            .topology()
            .sample_lattice_dimensions()
            .expect("temporal radiance output is a sample lattice"),
        evaluation_extent: resolved
            .finite_evaluation_extent
            .expect("temporal history requires finite evaluation"),
        current_coverage: requested_coverage.map(|_| RenderRequestedCoveragePreparation {
            extent: resolved
                .requested
                .spec()
                .topology()
                .sample_lattice_dimensions()
                .expect("coverage lattice"),
            policy_revision: requested_coverage::POLICY_REVISION,
            evaluator_revision: MAINTAINED_EVALUATOR_REVISION,
        }),
        semantic_input_generations: admitted
            .surface_semantic_inputs()
            .iter()
            .map(|binding| {
                (
                    binding.representation_id(),
                    binding
                        .generation()
                        .expect("temporal lowering required surface source generation"),
                )
            })
            .collect(),
        field_semantic_input_generations: admitted
            .field_semantic_inputs()
            .iter()
            .map(|binding| {
                (
                    binding.representation_id(),
                    binding
                        .generation()
                        .expect("temporal lowering required field source generation"),
                )
            })
            .collect(),
        sequence_revision: temporal::SEQUENCE_REVISION,
        reconstruction_revision: TEMPORAL_RECONSTRUCTION_REVISION,
        phase: history.phase,
        history_generation: history.generation,
        history_age: history.age,
        history_reset: history.reset,
        camera_reprojection_eligible: matches!(
            &history.storage,
            DeterministicTemporalHistoryUseStorage::Camera { .. }
        ),
        previous_observation_available: matches!(
            &history.storage,
            DeterministicTemporalHistoryUseStorage::Camera {
                previous_observation: Some(_),
                ..
            }
        ),
        camera_pose_changed: matches!(
            &history.storage,
            DeterministicTemporalHistoryUseStorage::Camera {
                pose_changed: true,
                ..
            }
        ),
        camera_same_pose_completed_frames: match &history.storage {
            DeterministicTemporalHistoryUseStorage::Camera {
                same_pose_completed_frames,
                ..
            } => Some(*same_pose_completed_frames),
            DeterministicTemporalHistoryUseStorage::Static { .. } => None,
        },
        camera_reprojection_revision: matches!(
            &history.storage,
            DeterministicTemporalHistoryUseStorage::Camera { .. }
        )
        .then_some(CAMERA_REPROJECTION_REVISION),
        depth_policy_revision: matches!(
            &history.storage,
            DeterministicTemporalHistoryUseStorage::Camera { .. }
        )
        .then_some(camera::DEPTH_POLICY_REVISION),
    })
}

/// Correlated physical producer inputs for one finalized output occurrence.
struct PreparedDestinationSources<'a> {
    primary: &'a PreparedPrimaryPass,
    temporal: &'a PreparedTemporalPass,
}

fn prepare_destination(
    admitted: &AdmittedRenderPlan,
    resolved: ResolvedOutputContext<'_>,
    packed: &PackedOutput,
    temporal_state: &PreparedTemporalState,
    requested_coverage: Option<&PreparedRequestedCoverage>,
    sources: PreparedDestinationSources<'_>,
    intent: DeterministicObservationIntent,
) -> Result<PreparedDestination, RenderDeterministicLoweringError> {
    match resolved.admitted_output.binding().destination() {
        RenderOutputDestination::ScalarBuffer(destination) => {
            let copy = GpuCopyOperation::buffer_to_buffer(
                GpuBufferRegion::whole(&sources.primary.canonical_output)
                    .map_err(|error| gpu_work_operation("scalar source region", error))?,
                GpuBufferRegion::whole(destination)
                    .map_err(|error| gpu_work_operation("scalar destination region", error))?,
            )
            .map_err(|error| gpu_work_operation("scalar destination copy", error))?;
            Ok(PreparedDestination {
                copy,
                gpu_output: None,
                availability_output: None,
                radiance_output: None,
            })
        }
        RenderOutputDestination::SampleLatticeTexture(destination) => {
            let (copy_source, row_bytes) = if let Some(history) = temporal_state.history.as_ref() {
                match &history.storage {
                    DeterministicTemporalHistoryUseStorage::Static {
                        row_stride_words, ..
                    } => {
                        let fallback = sources.temporal.static_fallback.as_ref().ok_or(
                            RenderDeterministicLoweringError::OutputCorrelationChanged {
                                output_index: resolved.output_index,
                            },
                        )?;
                        let row_bytes = row_stride_words
                            .checked_mul(u32::try_from(WORD_BYTES).expect("word bytes fit u32"))
                            .ok_or(RenderDeterministicLoweringError::SizeOverflow {
                                field: "temporal history row bytes",
                            })?;
                        (&fallback.resolved, row_bytes)
                    }
                    DeterministicTemporalHistoryUseStorage::Camera { .. } => {
                        let row_bytes = packed.texture_row_bytes.ok_or(
                            RenderDeterministicLoweringError::OutputCorrelationChanged {
                                output_index: resolved.output_index,
                            },
                        )?;
                        (&sources.primary.canonical_output, row_bytes)
                    }
                }
            } else {
                let row_bytes = packed.texture_row_bytes.ok_or(
                    RenderDeterministicLoweringError::OutputCorrelationChanged {
                        output_index: resolved.output_index,
                    },
                )?;
                (&sources.primary.canonical_output, row_bytes)
            };
            let source = GpuBufferTextureLayout::new(copy_source, 0, row_bytes, 0)
                .map_err(|error| gpu_work_operation("lattice source layout", error))?;
            let destination_region = GpuTextureCopyRegion::whole_base_mip(destination)
                .map_err(|error| gpu_work_operation("lattice destination region", error))?;
            let copy = GpuCopyOperation::buffer_to_texture(source, destination_region.clone())
                .map_err(|error| gpu_work_operation("lattice destination copy", error))?;
            let composable = if matches!(intent, DeterministicObservationIntent::Ordinary)
                && matches!(
                    resolved.requested.spec().value(),
                    RenderOutputValue::Radiance { .. }
                )
                && destination.descriptor().format() == GpuTextureFormat::R32Float
            {
                let relationship = GpuExportRelationship::new(
                    GpuResourceRef::Texture(destination.clone()),
                    GpuExportKey::new(format!(
                        "runenrender.maintained.radiance.graph.{}.output.{}",
                        resolved.graph_wiring_namespace, resolved.output_index
                    ))
                    .map_err(|error| gpu_resource_descriptor("radiance export key", error))?,
                    GpuResourceAccessIntent::Write,
                    GpuResourceProvenance::new(
                        destination.descriptor().common().label().clone(),
                        None,
                        None,
                    ),
                );
                let coverage = GpuInitialCoverage::texture_subresources(
                    &GpuTextureAccessResource::Texture(destination.clone()),
                    [destination_region.subresources()],
                )
                .map_err(|error| gpu_work_authoring("radiance output coverage", error))?;
                let output = GpuWorkOutput::new(relationship.clone(), coverage)
                    .map_err(|error| gpu_work_authoring("radiance output relationship", error))?;
                let availability = sources
                    .temporal
                    .static_fallback
                    .as_ref()
                    .map(|fallback| {
                        let relationship = GpuExportRelationship::new(
                            GpuResourceRef::Buffer(fallback.availability.clone()),
                            GpuExportKey::new(format!(
                                "runenrender.maintained.radiance.graph.{}.output.{}.availability",
                                resolved.graph_wiring_namespace, resolved.output_index
                            ))
                            .map_err(|error| {
                                gpu_resource_descriptor("radiance availability export key", error)
                            })?,
                            GpuResourceAccessIntent::ReadWrite,
                            GpuResourceProvenance::new(
                                fallback.availability.descriptor().common().label().clone(),
                                None,
                                None,
                            ),
                        );
                        let coverage = GpuInitialCoverage::buffer(
                            &fallback.availability,
                            [GpuBufferCoverage::dense(
                                GpuBufferRegion::whole(&fallback.availability)
                                    .map_err(|error| {
                                        gpu_work_operation("availability buffer range", error)
                                    })?
                                    .range(),
                            )],
                        )
                        .map_err(|error| {
                            gpu_work_authoring("availability buffer coverage", error)
                        })?;
                        let gpu_output = GpuWorkOutput::new(relationship.clone(), coverage)
                            .map_err(|error| {
                                gpu_work_authoring("availability output relationship", error)
                            })?;
                        Ok::<_, RenderDeterministicLoweringError>((gpu_output, relationship))
                    })
                    .transpose()?;
                let (availability_output, availability_relationship) = match availability {
                    Some((output, relationship)) => (Some(output), Some(relationship)),
                    None => (None, None),
                };
                Some((
                    output,
                    availability_output,
                    PreparedDeterministicRadianceOutput {
                        output_index: resolved.output_index,
                        relationship,
                        availability_relationship,
                        temporal_evidence: temporal_execution_evidence(
                            admitted,
                            resolved,
                            temporal_state,
                            requested_coverage,
                        ),
                    },
                ))
            } else {
                None
            };
            let (gpu_output, availability_output, radiance_output) = match composable {
                Some((output, availability, correlation)) => {
                    (Some(output), availability, Some(correlation))
                }
                None => (None, None, None),
            };
            Ok(PreparedDestination {
                copy,
                gpu_output,
                availability_output,
                radiance_output,
            })
        }
    }
}

fn prepare_verification_readbacks(
    intent: DeterministicObservationIntent,
    output_index: usize,
    primary: &PreparedPrimaryPass,
) -> Result<Option<VerificationReadbackOperations>, RenderDeterministicLoweringError> {
    if !intent.requires_private_readback() {
        return Ok(None);
    }
    let canonical_readback = GpuReadbackOperation::ordinary(
        GpuBufferRegion::whole(&primary.canonical_output)
            .map_err(|error| gpu_work_operation("canonical-output readback region", error))?
            .into(),
    )
    .map_err(|error| gpu_readback_request("canonical-output readback", error))?;
    let definedness_readback = GpuReadbackOperation::ordinary(
        GpuBufferRegion::whole(&primary.definedness)
            .map_err(|error| gpu_work_operation("definedness readback region", error))?
            .into(),
    )
    .map_err(|error| gpu_readback_request("definedness readback", error))?;
    let status_readback = GpuReadbackOperation::ordinary(
        GpuBufferRegion::whole(&primary.status)
            .map_err(|error| gpu_work_operation("status readback region", error))?
            .into(),
    )
    .map_err(|error| gpu_readback_request("status readback", error))?;
    Ok(Some(VerificationReadbackOperations {
        correlation: DeterministicVerificationReadbacks {
            output_index,
            canonical_output: canonical_readback.id(),
            definedness: definedness_readback.id(),
            status: status_readback.id(),
        },
        canonical_output: canonical_readback,
        definedness: definedness_readback,
        status: status_readback,
    }))
}

fn build_output_fragment(
    output_index: usize,
    primary: PreparedPrimaryPass,
    requested_coverage: Option<PreparedRequestedCoverage>,
    temporal_pass: PreparedTemporalPass,
    destination: PreparedDestination,
    verification: Option<VerificationReadbackOperations>,
) -> Result<GpuWorkFragment, RenderDeterministicLoweringError> {
    GpuWorkFragment::build(
        format!("RunenRender maintained output {output_index}"),
        |work| {
            work.operation("upload deterministic semantic input", primary.input_upload)?;
            work.operation("clear canonical output", primary.output_clear)?;
            work.operation("clear semantic definedness", primary.definedness_clear)?;
            work.operation("clear evaluator status", primary.status_clear)?;
            work.operation("clear current hit depth", primary.current_depth_clear)?;
            work.operation("clear current hit validity", primary.current_hit_clear)?;
            work.compute("evaluate deterministic output", primary.compute)?;
            if let Some(coverage) = requested_coverage {
                work.operation(
                    "upload current requested coverage input",
                    coverage.input_upload,
                )?;
                for (index, clear) in coverage.clears.into_iter().enumerate() {
                    work.operation(format!("clear requested coverage carrier {index}"), clear)?;
                }
                work.compute(
                    "classify current requested lattice coverage",
                    coverage.compute,
                )?;
            }
            let fallback_compute = if let Some(fallback) = temporal_pass.static_fallback {
                for (index, clear) in fallback.clears.into_iter().enumerate() {
                    work.operation(format!("clear temporal fallback carrier {index}"), clear)?;
                }
                Some(fallback.compute)
            } else {
                None
            };
            if let Some(upload) = temporal_pass.camera_parameter_upload {
                work.operation("upload camera reprojection parameters", upload)?;
            }
            if let Some(reconstruction) = temporal_pass.reconstruction_compute {
                work.compute("reconstruct deterministic footprint output", reconstruction)?;
            }
            if let Some(fallback) = fallback_compute {
                work.compute(
                    "resolve phase-aligned current radiance and availability",
                    fallback,
                )?;
            }
            work.operation(
                "copy reconstructed output to admitted destination",
                destination.copy,
            )?;
            if let Some(output) = destination.gpu_output {
                work.add_output(output)?;
            }
            if let Some(output) = destination.availability_output {
                work.add_output(output)?;
            }
            if let Some(readbacks) = verification {
                work.operation(
                    "read back private canonical output",
                    readbacks.canonical_output,
                )?;
                work.operation(
                    "read back private semantic definedness",
                    readbacks.definedness,
                )?;
                work.operation("read back private evaluator status", readbacks.status)?;
            }
            Ok(())
        },
    )
    .map_err(|error| gpu_work_authoring("work-fragment construction", error))
}

pub(super) fn finalize_output(
    admitted: &AdmittedRenderPlan,
    resolved: ResolvedOutputContext<'_>,
    packed: &PackedOutput,
    temporal_state: &PreparedTemporalState,
    passes: PreparedOutputPasses,
    intent: DeterministicObservationIntent,
) -> Result<LoweredDeterministicOutput, RenderDeterministicLoweringError> {
    let PreparedOutputPasses {
        requested_coverage,
        primary,
        temporal,
    } = passes;
    let destination = prepare_destination(
        admitted,
        resolved,
        packed,
        temporal_state,
        requested_coverage.as_ref(),
        PreparedDestinationSources {
            primary: &primary,
            temporal: &temporal,
        },
        intent,
    )?;
    let verification = prepare_verification_readbacks(intent, resolved.output_index, &primary)?;
    let verification_readbacks = verification.as_ref().map(|readbacks| readbacks.correlation);
    let composable_radiance_output = destination.radiance_output.clone();
    let fragment = build_output_fragment(
        resolved.output_index,
        primary,
        requested_coverage,
        temporal,
        destination,
        verification,
    )?;
    Ok(LoweredDeterministicOutput {
        fragment,
        verification_readbacks,
        composable_radiance_output,
    })
}
