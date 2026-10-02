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

pub(super) struct PackedOutput {
    pub(super) input_words: Vec<u32>,
    pub(super) sample_count: u32,
    pub(super) output_byte_len: u64,
    pub(super) texture_row_bytes: Option<u32>,
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
    super::verification::RenderDeterministicVerifiedSubmissionError,
> {
    super::verification::submit_deterministic_render_for_verified_formation(admitted, context)
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

pub(super) fn lower_output(
    admitted: &AdmittedRenderPlan,
    output_index: usize,
    object_codes: &BTreeMap<RenderObjectId, u32>,
    context: &GpuContext,
    resources: &mut DeterministicResourceCache,
    intent: DeterministicObservationIntent,
    execution: DeterministicOutputExecutionSelection,
) -> Result<LoweredDeterministicOutput, RenderDeterministicLoweringError> {
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

    let temporal_history = if let Some(evaluation_extent) = finite_evaluation_extent {
        let RenderObservationSpec::Perspective(perspective) = observation else {
            return Err(RenderDeterministicLoweringError::UnsupportedOutput { output_index });
        };
        if !perspective.sampling_support().is_perspective_lattice_cell()
            || !matches!(requested.spec().value(), RenderOutputValue::Radiance { .. })
        {
            return Err(RenderDeterministicLoweringError::UnsupportedOutput { output_index });
        }
        let requested_extent = requested
            .spec()
            .topology()
            .sample_lattice_dimensions()
            .ok_or(RenderDeterministicLoweringError::UnsupportedOutput { output_index })?;
        if !temporal_evaluation_extent_supported(requested_extent, evaluation_extent) {
            return Err(
                RenderDeterministicLoweringError::UnsupportedTemporalEvaluationExtent {
                    output_index,
                    requested_extent,
                    evaluation_extent,
                },
            );
        }
        let alignment = context
            .device_facts()
            .device_limits()
            .alignments()
            .bytes_per_row
            .ok_or(RenderDeterministicLoweringError::MissingBytesPerRowAlignment)?;
        for binding in admitted.surface_semantic_inputs() {
            if binding.generation().is_none() {
                return Err(
                    RenderDeterministicLoweringError::MissingTemporalSurfaceInputGeneration {
                        output_index,
                        representation_id: binding.representation_id(),
                    },
                );
            }
        }
        for binding in admitted.field_semantic_inputs() {
            if binding.generation().is_none() {
                return Err(
                    RenderDeterministicLoweringError::MissingTemporalFieldInputGeneration {
                        output_index,
                        representation_id: binding.representation_id(),
                    },
                );
            }
        }
        let camera_capable = evaluation_extent == requested_extent;
        let signature = DeterministicTemporalSignature {
            scene_revision: admitted.scene_revision(),
            observation: temporal_observation_compatibility(observation, camera_capable),
            output: requested.spec(),
            semantic_inputs: admitted.surface_semantic_inputs().to_vec(),
            field_semantic_inputs: admitted.field_semantic_inputs().to_vec(),
            evaluation_extent,
            sequence_revision: TEMPORAL_SEQUENCE_REVISION,
            reconstruction_revision: TEMPORAL_RECONSTRUCTION_REVISION,
            camera_reprojection_revision: camera_capable.then_some(CAMERA_REPROJECTION_REVISION),
            depth_policy_revision: camera_capable.then_some(CAMERA_DEPTH_POLICY_REVISION),
        };
        Some(resources.temporal_history(
            scope,
            output_index,
            signature,
            requested_extent,
            alignment,
            DeterministicTemporalHistorySelection {
                current_observation: perspective,
                camera_capable,
            },
        )?)
    } else {
        None
    };

    let packed = pack_output(
        admitted,
        admitted_output,
        MaintainedExecutionKind::Semantic(requested.spec().value()),
        observation,
        object_codes,
        context,
        DeterministicOutputPackingState {
            finite_evaluation_extent,
            temporal_history: temporal_history.as_ref(),
        },
    )?;
    let requested_extent = requested.spec().topology().sample_lattice_dimensions();
    let requested_coverage = if produce_requested_coverage
        && finite_evaluation_extent.is_some_and(|extent| Some(extent) != requested_extent)
    {
        let coverage_packed = pack_output(
            admitted,
            admitted_output,
            MaintainedExecutionKind::RequestedCoverage,
            observation,
            object_codes,
            context,
            DeterministicOutputPackingState {
                finite_evaluation_extent: None,
                temporal_history: temporal_history.as_ref(),
            },
        )?;
        Some(prepare_requested_coverage(
            coverage_packed,
            context,
            resources,
            scope,
            output_index,
        )?)
    } else {
        None
    };
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
        scope,
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
        scope,
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
        scope,
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
        scope,
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
        scope,
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
        scope,
        output_index,
        DeterministicBufferKind::CurrentHit,
        GpuBufferDescriptor::ordinary_owned(
            format!("RunenRender output {output_index} current coherent hit point"),
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            packed
                .output_byte_len
                .checked_mul(CURRENT_HIT_WORDS_PER_SAMPLE)
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

    let source = resources.maintained_source()?;
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
        context
            .device_facts()
            .workload_budget()
            .limits()
            .max_compute_workgroups_per_dimension(),
    )?;
    let compute = GpuComputeOperation::new(
        pipeline,
        runtime_bindings,
        GpuDispatchIntent::direct(dispatch_size),
    )
    .map_err(|error| gpu_work_operation("compute operation", error))?;

    let mut camera_parameter_upload = None;
    let reconstruction_compute = if let Some(history) = temporal_history.as_ref() {
        match &history.storage {
            DeterministicTemporalHistoryUseStorage::Static {
                handle,
                sample_counts,
                ..
            } => {
                let source = resources.reconstruction_source()?;
                let pipeline =
                    GpuComputePipelineDescriptor::ordinary(source, "main").map_err(|error| {
                        gpu_program_contract("temporal reconstruction pipeline", error)
                    })?;
                let runtime_bindings = pipeline
                    .runtime_bindings([
                        GpuRuntimeBindingValue::whole_buffer(0, 0, &input),
                        GpuRuntimeBindingValue::whole_buffer(0, 1, &canonical_output),
                        GpuRuntimeBindingValue::whole_buffer(0, 2, &definedness),
                        GpuRuntimeBindingValue::whole_buffer(0, 3, handle),
                        GpuRuntimeBindingValue::whole_buffer(0, 4, sample_counts),
                    ])
                    .map_err(|error| {
                        gpu_program_contract("temporal reconstruction runtime bindings", error)
                    })?;
                let dispatch_size = deterministic_dispatch_size(
                    packed.sample_count,
                    context
                        .device_facts()
                        .workload_budget()
                        .limits()
                        .max_compute_workgroups_per_dimension(),
                )?;
                Some(
                    GpuComputeOperation::new(
                        pipeline,
                        runtime_bindings,
                        GpuDispatchIntent::direct(dispatch_size),
                    )
                    .map_err(|error| {
                        gpu_work_operation("temporal reconstruction operation", error)
                    })?,
                )
            }
            DeterministicTemporalHistoryUseStorage::Camera {
                previous_history,
                current_history,
                previous_observation,
                pose_changed,
                same_pose_completed_frames,
            } => {
                let parameter_words = camera_reprojection_parameter_words(
                    match observation {
                        RenderObservationSpec::Perspective(perspective) => perspective,
                        _ => {
                            return Err(RenderDeterministicLoweringError::UnsupportedOutput {
                                output_index,
                            });
                        }
                    },
                    *previous_observation,
                    *pose_changed,
                    *same_pose_completed_frames,
                )?;
                let payload = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
                    format!("RunenRender output {output_index} camera reprojection parameters"),
                    &parameter_words,
                )
                .map_err(|error| {
                    gpu_transfer_preparation("camera-reprojection parameter preparation", error)
                })?;
                let parameters = resources.buffer(
                    scope,
                    output_index,
                    DeterministicBufferKind::CameraParameters,
                    GpuBufferDescriptor::ordinary_owned(
                        format!("RunenRender output {output_index} camera reprojection parameters"),
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
                camera_parameter_upload = Some(
                    GpuUploadOperation::whole_buffer(&parameters, payload).map_err(|error| {
                        gpu_work_operation("camera-reprojection parameter upload", error)
                    })?,
                );
                let source = resources.camera_reprojection_source()?;
                let pipeline = GpuComputePipelineDescriptor::ordinary(source, "main")
                    .map_err(|error| gpu_program_contract("camera-reprojection pipeline", error))?;
                let runtime_bindings = pipeline
                    .runtime_bindings([
                        GpuRuntimeBindingValue::whole_buffer(0, 0, &input),
                        GpuRuntimeBindingValue::whole_buffer(0, 1, &canonical_output),
                        GpuRuntimeBindingValue::whole_buffer(0, 2, &definedness),
                        GpuRuntimeBindingValue::whole_buffer(0, 3, &current_depth),
                        GpuRuntimeBindingValue::whole_buffer(0, 4, &current_hit),
                        GpuRuntimeBindingValue::whole_buffer(0, 5, previous_history),
                        GpuRuntimeBindingValue::whole_buffer(0, 6, current_history),
                        GpuRuntimeBindingValue::whole_buffer(0, 7, &parameters),
                    ])
                    .map_err(|error| {
                        gpu_program_contract("camera-reprojection runtime bindings", error)
                    })?;
                let dispatch_size = deterministic_dispatch_size(
                    packed.sample_count,
                    context
                        .device_facts()
                        .workload_budget()
                        .limits()
                        .max_compute_workgroups_per_dimension(),
                )?;
                Some(
                    GpuComputeOperation::new(
                        pipeline,
                        runtime_bindings,
                        GpuDispatchIntent::direct(dispatch_size),
                    )
                    .map_err(|error| gpu_work_operation("camera-reprojection operation", error))?,
                )
            }
        }
    } else {
        None
    };

    let (destination_copy, composable_gpu_output, composable_radiance_output) =
        match admitted_output.binding().destination() {
            RenderOutputDestination::ScalarBuffer(destination) => {
                GpuCopyOperation::buffer_to_buffer(
                    GpuBufferRegion::whole(&canonical_output)
                        .map_err(|error| gpu_work_operation("scalar source region", error))?,
                    GpuBufferRegion::whole(destination)
                        .map_err(|error| gpu_work_operation("scalar destination region", error))?,
                )
                .map(|copy| (copy, None, None))
                .map_err(|error| gpu_work_operation("scalar destination copy", error))?
            }
            RenderOutputDestination::SampleLatticeTexture(destination) => {
                let (copy_source, row_bytes) = if let Some(history) = temporal_history.as_ref() {
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
                                    output_index,
                                },
                            )?;
                            (&canonical_output, row_bytes)
                        }
                    }
                } else {
                    let row_bytes = packed.texture_row_bytes.ok_or(
                        RenderDeterministicLoweringError::OutputCorrelationChanged { output_index },
                    )?;
                    (&canonical_output, row_bytes)
                };
                let source = GpuBufferTextureLayout::new(copy_source, 0, row_bytes, 0)
                    .map_err(|error| gpu_work_operation("lattice source layout", error))?;
                let destination_region = GpuTextureCopyRegion::whole_base_mip(destination)
                    .map_err(|error| gpu_work_operation("lattice destination region", error))?;
                let destination_copy =
                    GpuCopyOperation::buffer_to_texture(source, destination_region.clone())
                        .map_err(|error| gpu_work_operation("lattice destination copy", error))?;
                let composable = if matches!(intent, DeterministicObservationIntent::Ordinary)
                    && matches!(requested.spec().value(), RenderOutputValue::Radiance { .. })
                    && destination.descriptor().format() == GpuTextureFormat::R32Float
                {
                    let relationship = GpuExportRelationship::new(
                        GpuResourceRef::Texture(destination.clone()),
                        GpuExportKey::new(format!(
                            "runenrender.maintained.radiance.scope.{scope}.output.{output_index}"
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
                    let output =
                        GpuWorkOutput::new(relationship.clone(), coverage).map_err(|error| {
                            gpu_work_authoring("radiance output relationship", error)
                        })?;
                    Some((
                        output,
                        PreparedDeterministicRadianceOutput {
                            output_index,
                            relationship,
                            temporal_evidence: temporal_history.as_ref().map(|history| {
                                RenderTemporalExecutionEvidence {
                                    requested_extent: requested
                                        .spec()
                                        .topology()
                                        .sample_lattice_dimensions()
                                        .expect("temporal radiance output is a sample lattice"),
                                    evaluation_extent: finite_evaluation_extent
                                        .expect("temporal history requires finite evaluation"),
                                    current_coverage: requested_coverage.as_ref().map(|_| {
                                        RenderRequestedCoveragePreparation {
                                            extent: requested_extent.expect("coverage lattice"),
                                            policy_revision: REQUESTED_COVERAGE_POLICY_REVISION,
                                            evaluator_revision: MAINTAINED_EVALUATOR_REVISION,
                                        }
                                    }),
                                    semantic_input_generations: admitted
                                        .surface_semantic_inputs()
                                        .iter()
                                        .map(|binding| {
                                            (
                                                binding.representation_id(),
                                                binding.generation().expect(
                                                    "temporal lowering required surface source generation",
                                                ),
                                            )
                                        })
                                        .collect(),
                                    field_semantic_input_generations: admitted
                                        .field_semantic_inputs()
                                        .iter()
                                        .map(|binding| {
                                            (
                                                binding.representation_id(),
                                                binding.generation().expect(
                                                    "temporal lowering required field source generation",
                                                ),
                                            )
                                        })
                                        .collect(),
                                    sequence_revision: TEMPORAL_SEQUENCE_REVISION,
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
                                        DeterministicTemporalHistoryUseStorage::Static { .. } => {
                                            None
                                        }
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
                                    .then_some(CAMERA_DEPTH_POLICY_REVISION),
                                }
                            }),
                        },
                    ))
                } else {
                    None
                };
                let (gpu_output, correlation) = match composable {
                    Some((output, correlation)) => (Some(output), Some(correlation)),
                    None => (None, None),
                };
                (destination_copy, gpu_output, correlation)
            }
        };

    let verification = if intent.requires_private_readback() {
        let canonical_readback = GpuReadbackOperation::ordinary(
            GpuBufferRegion::whole(&canonical_output)
                .map_err(|error| gpu_work_operation("canonical-output readback region", error))?
                .into(),
        )
        .map_err(|error| gpu_readback_request("canonical-output readback", error))?;
        let definedness_readback = GpuReadbackOperation::ordinary(
            GpuBufferRegion::whole(&definedness)
                .map_err(|error| gpu_work_operation("definedness readback region", error))?
                .into(),
        )
        .map_err(|error| gpu_readback_request("definedness readback", error))?;
        let status_readback = GpuReadbackOperation::ordinary(
            GpuBufferRegion::whole(&status)
                .map_err(|error| gpu_work_operation("status readback region", error))?
                .into(),
        )
        .map_err(|error| gpu_readback_request("status readback", error))?;
        Some(VerificationReadbackOperations {
            correlation: DeterministicVerificationReadbacks {
                output_index,
                canonical_output: canonical_readback.id(),
                definedness: definedness_readback.id(),
                status: status_readback.id(),
            },
            canonical_output: canonical_readback,
            definedness: definedness_readback,
            status: status_readback,
        })
    } else {
        None
    };
    let verification_readbacks = verification.as_ref().map(|readbacks| readbacks.correlation);

    let fragment = GpuWorkFragment::build(
        format!("RunenRender maintained output {output_index}"),
        |work| {
            work.operation("upload deterministic semantic input", input_upload)?;
            work.operation("clear canonical output", output_clear)?;
            work.operation("clear semantic definedness", definedness_clear)?;
            work.operation("clear evaluator status", status_clear)?;
            work.operation("clear current hit depth", current_depth_clear)?;
            work.operation("clear current hit validity", current_hit_clear)?;
            work.compute("evaluate deterministic output", compute)?;
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
            if let Some(upload) = camera_parameter_upload {
                work.operation("upload camera reprojection parameters", upload)?;
            }
            if let Some(reconstruction) = reconstruction_compute {
                work.compute("reconstruct deterministic footprint output", reconstruction)?;
            }
            work.operation(
                "copy reconstructed output to admitted destination",
                destination_copy,
            )?;
            if let Some(output) = composable_gpu_output {
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
    .map_err(|error| gpu_work_authoring("work-fragment construction", error))?;

    Ok(LoweredDeterministicOutput {
        fragment,
        verification_readbacks,
        composable_radiance_output,
    })
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
    let pipeline = GpuComputePipelineDescriptor::ordinary(resources.maintained_source()?, "main")
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
