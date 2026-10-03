use super::*;

pub(super) struct LoweredDeterministicRender {
    pub(super) work_set: RenderWorkSet,
    pub(super) object_identity_decoder: RenderObjectIdentityDecoder,
    pub(super) verification_readbacks: Vec<DeterministicVerificationReadbacks>,
    pub(super) composable_radiance_outputs: Vec<PreparedDeterministicRadianceOutput>,
}

pub(super) struct LoweredDeterministicOutput {
    pub(super) fragment: GpuWorkFragment,
    pub(super) verification_readbacks: Option<DeterministicVerificationReadbacks>,
    pub(super) composable_radiance_output: Option<PreparedDeterministicRadianceOutput>,
}

pub(super) struct VerificationReadbackOperations {
    pub(super) correlation: DeterministicVerificationReadbacks,
    pub(super) canonical_output: GpuReadbackOperation,
    pub(super) definedness: GpuReadbackOperation,
    pub(super) status: GpuReadbackOperation,
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

/// Prepare one ordinary maintained deterministic render using renderer-owned reusable resources.
pub(crate) fn prepare_deterministic_render_with_cache(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
    resources: &mut DeterministicResourceCache,
) -> Result<PreparedDeterministicRender, RenderDeterministicExecutionError> {
    prepare_deterministic_render_with_cache_in_scope(admitted, context, resources, 0)
}

/// Prepare one composition using a producer-scoped resource cache namespace.
pub(crate) fn prepare_deterministic_render_with_cache_in_scope(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
    resources: &mut DeterministicResourceCache,
    scope: u64,
) -> Result<PreparedDeterministicRender, RenderDeterministicExecutionError> {
    prepare_deterministic_render_with_cache_in_scope_and_evaluation(
        admitted, context, resources, scope, None, false,
    )
}

pub(crate) fn prepare_deterministic_render_with_cache_in_scope_and_evaluation(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
    resources: &mut DeterministicResourceCache,
    scope: u64,
    finite_evaluation: Option<(usize, (u32, u32))>,
    produce_requested_coverage: bool,
) -> Result<PreparedDeterministicRender, RenderDeterministicExecutionError> {
    let lowered = lower_deterministic_render(
        &admitted,
        context,
        DeterministicObservationIntent::Ordinary,
        resources,
        DeterministicRenderExecutionSelection {
            scope,
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
        scope,
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

    resources.prepared_temporal_outputs.remove(&scope);

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
                scope,
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


#[derive(Clone, Copy)]
struct ResolvedOutputContext<'a> {
    output_index: usize,
    admitted_output: &'a crate::admission::RenderAdmittedOutput,
    requested: crate::request::RenderRequestedOutput,
    observation: RenderObservationSpec,
    scope: u64,
    finite_evaluation_extent: Option<(u32, u32)>,
    produce_requested_coverage: bool,
    bytes_per_row_alignment: Option<u64>,
    max_compute_workgroups_per_dimension: u32,
}

struct PreparedTemporalState {
    history: Option<DeterministicTemporalHistoryUse>,
}

impl PreparedTemporalState {
    fn packing_facts(&self) -> Option<OutputTemporalPackingFacts> {
        self.history.as_ref().map(|history| OutputTemporalPackingFacts {
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

struct PreparedPrimaryPass {
    input: GpuBufferHandle,
    canonical_output: GpuBufferHandle,
    definedness: GpuBufferHandle,
    status: GpuBufferHandle,
    current_depth: GpuBufferHandle,
    current_hit: GpuBufferHandle,
    input_upload: GpuUploadOperation,
    output_clear: GpuClearOperation,
    definedness_clear: GpuClearOperation,
    status_clear: GpuClearOperation,
    current_depth_clear: GpuClearOperation,
    current_hit_clear: GpuClearOperation,
    compute: GpuComputeOperation,
}

struct PreparedTemporalPass {
    camera_parameter_upload: Option<GpuUploadOperation>,
    reconstruction_compute: Option<GpuComputeOperation>,
}

struct PreparedDestination {
    copy: GpuCopyOperation,
    gpu_output: Option<GpuWorkOutput>,
    radiance_output: Option<PreparedDeterministicRadianceOutput>,
}

fn resolve_output_context<'a>(
    admitted: &'a AdmittedRenderPlan,
    output_index: usize,
    context: &GpuContext,
    execution: DeterministicOutputExecutionSelection,
) -> Result<ResolvedOutputContext<'a>, RenderDeterministicLoweringError> {
    let DeterministicOutputExecutionSelection {
        scope,
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
        scope,
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
    })
}

fn prepare_temporal_state(
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
        resolved.scope,
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

fn prepare_primary_pass(
    packed: &PackedOutput,
    resolved: ResolvedOutputContext<'_>,
    resources: &mut DeterministicResourceCache,
) -> Result<PreparedPrimaryPass, RenderDeterministicLoweringError> {
    let output_index = resolved.output_index;
    let sample_byte_len = u64::from(packed.sample_count)
        .checked_mul(WORD_BYTES)
        .ok_or(RenderDeterministicLoweringError::SizeOverflow {
            field: "definedness/status byte length",
        })?;
    let input_payload = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        format!("RunenRender output {output_index} packed semantic input"),
        &packed.input_words,
    )
    .map_err(|error| gpu_transfer_preparation("semantic-input preparation", error))?;

    let input = resources.buffer(
        resolved.scope,
        output_index,
        DeterministicBufferKind::Input,
        GpuBufferDescriptor::ordinary_owned(
            format!("RunenRender output {output_index} packed input"),
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            input_payload.layout().byte_len(),
            [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
            GpuBufferInitialization::Uninitialized,
        )
        .map_err(|error| gpu_resource_descriptor("input-buffer descriptor", error))?,
    )?;
    let canonical_output = resources.buffer(
        resolved.scope,
        output_index,
        DeterministicBufferKind::CanonicalOutput,
        GpuBufferDescriptor::ordinary_owned(
            format!("RunenRender output {output_index} canonical words"),
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            packed.output_byte_len,
            [
                GpuBufferUsage::Storage,
                GpuBufferUsage::CopySource,
                GpuBufferUsage::CopyDestination,
            ],
            GpuBufferInitialization::Uninitialized,
        )
        .map_err(|error| gpu_resource_descriptor("canonical-output descriptor", error))?,
    )?;
    let definedness = resources.buffer(
        resolved.scope,
        output_index,
        DeterministicBufferKind::Definedness,
        GpuBufferDescriptor::ordinary_owned(
            format!("RunenRender output {output_index} definedness"),
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            sample_byte_len,
            [
                GpuBufferUsage::Storage,
                GpuBufferUsage::CopySource,
                GpuBufferUsage::CopyDestination,
            ],
            GpuBufferInitialization::Uninitialized,
        )
        .map_err(|error| gpu_resource_descriptor("definedness descriptor", error))?,
    )?;
    let status = resources.buffer(
        resolved.scope,
        output_index,
        DeterministicBufferKind::Status,
        GpuBufferDescriptor::ordinary_owned(
            format!("RunenRender output {output_index} evaluator status"),
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            sample_byte_len,
            [
                GpuBufferUsage::Storage,
                GpuBufferUsage::CopySource,
                GpuBufferUsage::CopyDestination,
            ],
            GpuBufferInitialization::Uninitialized,
        )
        .map_err(|error| gpu_resource_descriptor("status descriptor", error))?,
    )?;
    let current_depth = resources.buffer(
        resolved.scope,
        output_index,
        DeterministicBufferKind::CurrentDepth,
        GpuBufferDescriptor::ordinary_owned(
            format!("RunenRender output {output_index} current hit depth"),
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            packed.output_byte_len,
            [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
            GpuBufferInitialization::Uninitialized,
        )
        .map_err(|error| gpu_resource_descriptor("current-depth descriptor", error))?,
    )?;
    let current_hit = resources.buffer(
        resolved.scope,
        output_index,
        DeterministicBufferKind::CurrentHit,
        GpuBufferDescriptor::ordinary_owned(
            format!("RunenRender output {output_index} current coherent hit point"),
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            packed
                .output_byte_len
                .checked_mul(camera::history::CURRENT_HIT_WORDS_PER_SAMPLE)
                .ok_or(RenderDeterministicLoweringError::SizeOverflow {
                    field: "current coherent hit point byte length",
                })?,
            [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
            GpuBufferInitialization::Uninitialized,
        )
        .map_err(|error| gpu_resource_descriptor("current-hit descriptor", error))?,
    )?;

    let input_upload = GpuUploadOperation::whole_buffer(&input, input_payload)
        .map_err(|error| gpu_work_operation("input upload", error))?;
    let output_clear = GpuClearOperation::buffer_zero(
        GpuBufferRegion::whole(&canonical_output)
            .map_err(|error| gpu_work_operation("canonical-output clear region", error))?,
    )
    .map_err(|error| gpu_work_operation("canonical-output clear", error))?;
    let definedness_clear = GpuClearOperation::buffer_zero(
        GpuBufferRegion::whole(&definedness)
            .map_err(|error| gpu_work_operation("definedness clear region", error))?,
    )
    .map_err(|error| gpu_work_operation("definedness clear", error))?;
    let status_clear = GpuClearOperation::buffer_zero(
        GpuBufferRegion::whole(&status)
            .map_err(|error| gpu_work_operation("status clear region", error))?,
    )
    .map_err(|error| gpu_work_operation("status clear", error))?;
    let current_depth_clear = GpuClearOperation::buffer_zero(
        GpuBufferRegion::whole(&current_depth)
            .map_err(|error| gpu_work_operation("current-depth clear region", error))?,
    )
    .map_err(|error| gpu_work_operation("current-depth clear", error))?;
    let current_hit_clear = GpuClearOperation::buffer_zero(
        GpuBufferRegion::whole(&current_hit)
            .map_err(|error| gpu_work_operation("current-hit clear region", error))?,
    )
    .map_err(|error| gpu_work_operation("current-hit clear", error))?;

    let source =
        retained_maintained_evaluator_source().map_err(map_maintained_program_build_error)?;
    let pipeline = GpuComputePipelineDescriptor::ordinary(source, "main")
        .map_err(|error| gpu_program_contract("compute-pipeline descriptor", error))?;
    let runtime_bindings = pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, &input),
            GpuRuntimeBindingValue::whole_buffer(0, 1, &canonical_output),
            GpuRuntimeBindingValue::whole_buffer(0, 2, &definedness),
            GpuRuntimeBindingValue::whole_buffer(0, 3, &status),
            GpuRuntimeBindingValue::whole_buffer(0, 4, &current_depth),
            GpuRuntimeBindingValue::whole_buffer(0, 5, &current_hit),
        ])
        .map_err(|error| gpu_program_contract("compute runtime bindings", error))?;
    let dispatch_size = deterministic_dispatch_size(
        packed.sample_count,
        resolved.max_compute_workgroups_per_dimension,
    )?;
    let compute = GpuComputeOperation::new(
        pipeline,
        runtime_bindings,
        GpuDispatchIntent::direct(dispatch_size),
    )
    .map_err(|error| gpu_work_operation("compute operation", error))?;

    Ok(PreparedPrimaryPass {
        input,
        canonical_output,
        definedness,
        status,
        current_depth,
        current_hit,
        input_upload,
        output_clear,
        definedness_clear,
        status_clear,
        current_depth_clear,
        current_hit_clear,
        compute,
    })
}

fn prepare_temporal_pass(
    packed: &PackedOutput,
    resolved: ResolvedOutputContext<'_>,
    temporal_state: &PreparedTemporalState,
    primary: &PreparedPrimaryPass,
    resources: &mut DeterministicResourceCache,
) -> Result<PreparedTemporalPass, RenderDeterministicLoweringError> {
    let Some(history) = temporal_state.history.as_ref() else {
        return Ok(PreparedTemporalPass {
            camera_parameter_upload: None,
            reconstruction_compute: None,
        });
    };

    match &history.storage {
        DeterministicTemporalHistoryUseStorage::Static {
            handle,
            sample_counts,
            ..
        } => {
            let source = retained_temporal_reconstruction_source()
                .map_err(map_maintained_program_build_error)?;
            let pipeline = GpuComputePipelineDescriptor::ordinary(source, "main")
                .map_err(|error| gpu_program_contract("temporal reconstruction pipeline", error))?;
            let runtime_bindings = pipeline
                .runtime_bindings([
                    GpuRuntimeBindingValue::whole_buffer(0, 0, &primary.input),
                    GpuRuntimeBindingValue::whole_buffer(0, 1, &primary.canonical_output),
                    GpuRuntimeBindingValue::whole_buffer(0, 2, &primary.definedness),
                    GpuRuntimeBindingValue::whole_buffer(0, 3, handle),
                    GpuRuntimeBindingValue::whole_buffer(0, 4, sample_counts),
                ])
                .map_err(|error| {
                    gpu_program_contract("temporal reconstruction runtime bindings", error)
                })?;
            let dispatch_size = deterministic_dispatch_size(
                packed.sample_count,
                resolved.max_compute_workgroups_per_dimension,
            )?;
            let compute = GpuComputeOperation::new(
                pipeline,
                runtime_bindings,
                GpuDispatchIntent::direct(dispatch_size),
            )
            .map_err(|error| gpu_work_operation("temporal reconstruction operation", error))?;
            Ok(PreparedTemporalPass {
                camera_parameter_upload: None,
                reconstruction_compute: Some(compute),
            })
        }
        DeterministicTemporalHistoryUseStorage::Camera {
            previous_history,
            current_history,
            previous_observation,
            pose_changed,
            same_pose_completed_frames,
        } => {
            let RenderObservationSpec::Perspective(perspective) = resolved.observation else {
                return Err(RenderDeterministicLoweringError::UnsupportedOutput {
                    output_index: resolved.output_index,
                });
            };
            let parameter_words = camera_reprojection_parameter_words(
                perspective,
                *previous_observation,
                *pose_changed,
                *same_pose_completed_frames,
            )?;
            let payload = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
                format!(
                    "RunenRender output {} camera reprojection parameters",
                    resolved.output_index
                ),
                &parameter_words,
            )
            .map_err(|error| {
                gpu_transfer_preparation("camera-reprojection parameter preparation", error)
            })?;
            let parameters = resources.buffer(
                resolved.scope,
                resolved.output_index,
                DeterministicBufferKind::CameraParameters,
                GpuBufferDescriptor::ordinary_owned(
                    format!(
                        "RunenRender output {} camera reprojection parameters",
                        resolved.output_index
                    ),
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    payload.layout().byte_len(),
                    [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                    GpuBufferInitialization::Uninitialized,
                )
                .map_err(|error| {
                    gpu_resource_descriptor("camera-reprojection parameter descriptor", error)
                })?,
            )?;
            let parameter_upload =
                GpuUploadOperation::whole_buffer(&parameters, payload).map_err(|error| {
                    gpu_work_operation("camera-reprojection parameter upload", error)
                })?;
            let source =
                retained_camera_reprojection_source().map_err(map_maintained_program_build_error)?;
            let pipeline = GpuComputePipelineDescriptor::ordinary(source, "main")
                .map_err(|error| gpu_program_contract("camera-reprojection pipeline", error))?;
            let runtime_bindings = pipeline
                .runtime_bindings([
                    GpuRuntimeBindingValue::whole_buffer(0, 0, &primary.input),
                    GpuRuntimeBindingValue::whole_buffer(0, 1, &primary.canonical_output),
                    GpuRuntimeBindingValue::whole_buffer(0, 2, &primary.definedness),
                    GpuRuntimeBindingValue::whole_buffer(0, 3, &primary.current_depth),
                    GpuRuntimeBindingValue::whole_buffer(0, 4, &primary.current_hit),
                    GpuRuntimeBindingValue::whole_buffer(0, 5, previous_history),
                    GpuRuntimeBindingValue::whole_buffer(0, 6, current_history),
                    GpuRuntimeBindingValue::whole_buffer(0, 7, &parameters),
                ])
                .map_err(|error| {
                    gpu_program_contract("camera-reprojection runtime bindings", error)
                })?;
            let dispatch_size = deterministic_dispatch_size(
                packed.sample_count,
                resolved.max_compute_workgroups_per_dimension,
            )?;
            let compute = GpuComputeOperation::new(
                pipeline,
                runtime_bindings,
                GpuDispatchIntent::direct(dispatch_size),
            )
            .map_err(|error| gpu_work_operation("camera-reprojection operation", error))?;
            Ok(PreparedTemporalPass {
                camera_parameter_upload: Some(parameter_upload),
                reconstruction_compute: Some(compute),
            })
        }
    }
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

fn prepare_destination(
    admitted: &AdmittedRenderPlan,
    resolved: ResolvedOutputContext<'_>,
    packed: &PackedOutput,
    temporal_state: &PreparedTemporalState,
    requested_coverage: Option<&PreparedRequestedCoverage>,
    primary: &PreparedPrimaryPass,
    intent: DeterministicObservationIntent,
) -> Result<PreparedDestination, RenderDeterministicLoweringError> {
    match resolved.admitted_output.binding().destination() {
        RenderOutputDestination::ScalarBuffer(destination) => {
            let copy = GpuCopyOperation::buffer_to_buffer(
                GpuBufferRegion::whole(&primary.canonical_output)
                    .map_err(|error| gpu_work_operation("scalar source region", error))?,
                GpuBufferRegion::whole(destination)
                    .map_err(|error| gpu_work_operation("scalar destination region", error))?,
            )
            .map_err(|error| gpu_work_operation("scalar destination copy", error))?;
            Ok(PreparedDestination {
                copy,
                gpu_output: None,
                radiance_output: None,
            })
        }
        RenderOutputDestination::SampleLatticeTexture(destination) => {
            let (copy_source, row_bytes) = if let Some(history) = temporal_state.history.as_ref() {
                match &history.storage {
                    DeterministicTemporalHistoryUseStorage::Static {
                        handle,
                        row_stride_words,
                        ..
                    } => {
                        let row_bytes = row_stride_words
                            .checked_mul(u32::try_from(WORD_BYTES).expect("word bytes fit u32"))
                            .ok_or(RenderDeterministicLoweringError::SizeOverflow {
                                field: "temporal history row bytes",
                            })?;
                        (handle, row_bytes)
                    }
                    DeterministicTemporalHistoryUseStorage::Camera { .. } => {
                        let row_bytes = packed.texture_row_bytes.ok_or(
                            RenderDeterministicLoweringError::OutputCorrelationChanged {
                                output_index: resolved.output_index,
                            },
                        )?;
                        (&primary.canonical_output, row_bytes)
                    }
                }
            } else {
                let row_bytes = packed.texture_row_bytes.ok_or(
                    RenderDeterministicLoweringError::OutputCorrelationChanged {
                        output_index: resolved.output_index,
                    },
                )?;
                (&primary.canonical_output, row_bytes)
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
                        "runenrender.maintained.radiance.scope.{}.output.{}",
                        resolved.scope, resolved.output_index
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
                Some((
                    output,
                    PreparedDeterministicRadianceOutput {
                        output_index: resolved.output_index,
                        relationship,
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
            let (gpu_output, radiance_output) = match composable {
                Some((output, correlation)) => (Some(output), Some(correlation)),
                None => (None, None),
            };
            Ok(PreparedDestination {
                copy,
                gpu_output,
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
            if let Some(upload) = temporal_pass.camera_parameter_upload {
                work.operation("upload camera reprojection parameters", upload)?;
            }
            if let Some(reconstruction) = temporal_pass.reconstruction_compute {
                work.compute("reconstruct deterministic footprint output", reconstruction)?;
            }
            work.operation(
                "copy reconstructed output to admitted destination",
                destination.copy,
            )?;
            if let Some(output) = destination.gpu_output {
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
            context,
            resources,
            resolved.scope,
            resolved.output_index,
        )?)
    } else {
        None
    };

    let primary = prepare_primary_pass(&packed, resolved, resources)?;
    let temporal_pass =
        prepare_temporal_pass(&packed, resolved, &temporal_state, &primary, resources)?;
    let destination = prepare_destination(
        admitted,
        resolved,
        &packed,
        &temporal_state,
        requested_coverage.as_ref(),
        &primary,
        intent,
    )?;
    let verification = prepare_verification_readbacks(intent, resolved.output_index, &primary)?;
    let verification_readbacks = verification.as_ref().map(|readbacks| readbacks.correlation);
    let radiance_output = destination.radiance_output.clone();
    let fragment = build_output_fragment(
        resolved.output_index,
        primary,
        requested_coverage,
        temporal_pass,
        destination,
        verification,
    )?;

    Ok(LoweredDeterministicOutput {
        fragment,
        verification_readbacks,
        composable_radiance_output: radiance_output,
    })
}

fn camera_reprojection_parameter_words(
    current: RenderPerspectiveObservation,
    previous: Option<RenderPerspectiveObservation>,
    pose_changed: bool,
    same_pose_completed_frames: u32,
) -> Result<[u32; camera::parameters::WORDS], RenderDeterministicLoweringError> {
    let mut words = [0_u32; camera::parameters::WORDS];
    let matrix = current.observation_to_scene().row_major_3x4();
    let inverse = invert_matrix3(
        [
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ],
        "current observation linear transform",
    )?;
    pack_matrix3(
        &mut words,
        camera::parameters::CURRENT_SCENE_TO_OBSERVATION,
        inverse,
    )?;
    words[camera::parameters::PREVIOUS_AVAILABLE] = if previous.is_some() { 1 } else { 0 };
    words[camera::parameters::POSE_CHANGED] = if pose_changed { 1 } else { 0 };
    words[camera::parameters::DEPTH_POLICY_REVISION] = camera::DEPTH_POLICY_REVISION;
    words[camera::parameters::REPROJECTION_REVISION] = CAMERA_REPROJECTION_REVISION;
    words[camera::parameters::DEPTH_ABSOLUTE_EPSILON] = camera::DEPTH_ABSOLUTE_EPSILON.to_bits();
    words[camera::parameters::DEPTH_RELATIVE_EPSILON] = camera::DEPTH_RELATIVE_EPSILON.to_bits();
    words[camera::parameters::MOTION_DIAGNOSTICS_ENABLED] = 0;
    words[camera::parameters::CURRENT_ONLY_FIRST_MOTION] = 0;
    words[camera::parameters::SAME_POSE_COMPLETED_FRAMES] =
        same_pose_completed_frames.min(temporal::PHASE_COUNT);
    if let Some(previous) = previous {
        let matrix = previous.observation_to_scene().row_major_3x4();
        pack_vec3(
            &mut words,
            camera::parameters::PREVIOUS_ORIGIN,
            [matrix[3], matrix[7], matrix[11]],
        )?;
        let inverse = invert_matrix3(
            [
                matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8],
                matrix[9], matrix[10],
            ],
            "previous observation linear transform",
        )?;
        pack_matrix3(
            &mut words,
            camera::parameters::PREVIOUS_SCENE_TO_OBSERVATION,
            inverse,
        )?;
        let forward = normalize_private_vec3(
            [-matrix[2], -matrix[6], -matrix[10]],
            "previous observation forward",
        )?;
        pack_vec3(&mut words, camera::parameters::PREVIOUS_FORWARD, forward)?;
        words[camera::parameters::PREVIOUS_TAN_HALF_FOV] = positive_f32_bits(
            (previous.vertical_field_of_view_radians() * 0.5).tan(),
            "previous perspective tangent half field of view",
        )?;
        words[camera::parameters::PREVIOUS_ASPECT_RATIO] =
            positive_f32_bits(previous.aspect_ratio(), "previous perspective aspect ratio")?;
    }
    Ok(words)
}

fn normalize_private_vec3(
    values: [f64; 3],
    field: &'static str,
) -> Result<[f64; 3], RenderDeterministicLoweringError> {
    let magnitude_squared = values.iter().map(|value| value * value).sum::<f64>();
    if !magnitude_squared.is_finite() || magnitude_squared <= 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    let reciprocal = magnitude_squared.sqrt().recip();
    let normalized = values.map(|value| value * reciprocal);
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    Ok(normalized)
}

fn invert_matrix3(
    values: [f64; 9],
    field: &'static str,
) -> Result<[f64; 9], RenderDeterministicLoweringError> {
    let [a, b, c, d, e, f, g, h, i] = values;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    let reciprocal = determinant.recip();
    let inverse = [
        (e * i - f * h) * reciprocal,
        (c * h - b * i) * reciprocal,
        (b * f - c * e) * reciprocal,
        (f * g - d * i) * reciprocal,
        (a * i - c * g) * reciprocal,
        (c * d - a * f) * reciprocal,
        (d * h - e * g) * reciprocal,
        (b * g - a * h) * reciprocal,
        (a * e - b * d) * reciprocal,
    ];
    if inverse.iter().any(|value| !value.is_finite()) {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    Ok(inverse)
}

pub(super) fn deterministic_dispatch_size(
    sample_count: u32,
    max_workgroups_per_dimension: u32,
) -> Result<GpuDispatchSize, RenderDeterministicLoweringError> {
    let sample_count_u64 = u64::from(sample_count);
    let workgroup_size = u64::from(WORKGROUP_SIZE);
    let admitted_max = u64::from(max_workgroups_per_dimension);
    let required_groups = sample_count_u64.div_ceil(workgroup_size);
    let capacity = admitted_max * admitted_max;

    if required_groups == 0 || required_groups > capacity {
        return Err(RenderDeterministicLoweringError::DispatchCapacityExceeded {
            sample_count,
            workgroup_size: WORKGROUP_SIZE,
            required_workgroups: required_groups,
            max_workgroups_per_dimension,
            capacity_workgroups: capacity,
        });
    }

    let groups_x = required_groups.min(admitted_max);
    let groups_y = required_groups.div_ceil(groups_x);
    if groups_x > admitted_max || groups_y > admitted_max {
        return Err(RenderDeterministicLoweringError::DispatchCapacityExceeded {
            sample_count,
            workgroup_size: WORKGROUP_SIZE,
            required_workgroups: required_groups,
            max_workgroups_per_dimension,
            capacity_workgroups: capacity,
        });
    }

    Ok(GpuDispatchSize::new(
        u32::try_from(groups_x).expect("admitted dispatch x dimension must fit u32"),
        u32::try_from(groups_y).expect("admitted dispatch y dimension must fit u32"),
        1,
    ))
}

pub(super) struct PreparedRequestedCoverage {
    pub(super) input_upload: GpuUploadOperation,
    pub(super) clears: Vec<GpuClearOperation>,
    pub(super) compute: GpuComputeOperation,
}

/// One current requested-extent dispatch, with no history storage or runtime readback. Bindings
/// 1/2 carry padded depth and tightly packed Invalid=0 / Background=1 / Hit=2 states. The ordinary
/// evaluator's other bindings are distinct one-word scratch resources: coverage returns before
/// any access to them, including ordinary invalidation. No phase-sparse resource is reused.
pub(super) fn prepare_requested_coverage(
    packed: PackedOutput,
    context: &GpuContext,
    resources: &mut DeterministicResourceCache,
    scope: u64,
    output_index: usize,
) -> Result<PreparedRequestedCoverage, RenderDeterministicLoweringError> {
    let payload = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "current requested coverage input",
        &packed.input_words,
    )
    .map_err(|error| gpu_transfer_preparation("coverage input preparation", error))?;
    let state_bytes = u64::from(packed.sample_count)
        .checked_mul(WORD_BYTES)
        .ok_or(RenderDeterministicLoweringError::SizeOverflow {
            field: "requested coverage state bytes",
        })?;
    let descriptions = [
        (
            DeterministicBufferKind::CoverageInput,
            payload.layout().byte_len(),
        ),
        (
            DeterministicBufferKind::CoverageDepth,
            packed.output_byte_len,
        ),
        (DeterministicBufferKind::CoverageState, state_bytes),
        (DeterministicBufferKind::CoverageStatusScratch, WORD_BYTES),
        (DeterministicBufferKind::CoverageDepthScratch, WORD_BYTES),
        (DeterministicBufferKind::CoverageHitScratch, WORD_BYTES),
    ];
    let handles = descriptions
        .into_iter()
        .map(|(kind, bytes)| {
            resources.buffer(
                scope,
                output_index,
                kind,
                GpuBufferDescriptor::ordinary_owned(
                    format!("RunenRender output {output_index} {kind:?}"),
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    bytes,
                    [
                        GpuBufferUsage::Storage,
                        GpuBufferUsage::CopyDestination,
                        GpuBufferUsage::CopySource,
                    ],
                    GpuBufferInitialization::Uninitialized,
                )
                .map_err(|error| gpu_resource_descriptor("coverage buffer descriptor", error))?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let input_upload = GpuUploadOperation::whole_buffer(&handles[0], payload)
        .map_err(|error| gpu_work_operation("coverage input upload", error))?;
    let clears = handles[1..]
        .iter()
        .map(|handle| {
            GpuClearOperation::buffer_zero(
                GpuBufferRegion::whole(handle)
                    .map_err(|error| gpu_work_operation("coverage clear region", error))?,
            )
            .map_err(|error| gpu_work_operation("coverage clear", error))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let source =
        retained_maintained_evaluator_source().map_err(map_maintained_program_build_error)?;
    let pipeline = GpuComputePipelineDescriptor::ordinary(source, "main")
        .map_err(|error| gpu_program_contract("coverage pipeline", error))?;
    let bindings = pipeline
        .runtime_bindings(handles.iter().enumerate().map(|(binding, handle)| {
            GpuRuntimeBindingValue::whole_buffer(0, binding as u32, handle)
        }))
        .map_err(|error| gpu_program_contract("coverage runtime bindings", error))?;
    let dispatch = deterministic_dispatch_size(
        packed.sample_count,
        context
            .device_facts()
            .workload_budget()
            .limits()
            .max_compute_workgroups_per_dimension(),
    )?;
    let compute = GpuComputeOperation::new(pipeline, bindings, GpuDispatchIntent::direct(dispatch))
        .map_err(|error| gpu_work_operation("coverage compute", error))?;
    Ok(PreparedRequestedCoverage {
        input_upload,
        clears,
        compute,
    })
}
