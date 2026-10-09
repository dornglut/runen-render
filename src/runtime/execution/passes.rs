use super::super::program::{
    CAMERA_REPROJECTION_REVISION,
    abi::{WORKGROUP_SIZE, camera, header, temporal, temporal_fallback},
    retained_camera_reprojection_source, retained_maintained_evaluator_source,
    retained_temporal_fallback_source, retained_temporal_reconstruction_source,
};
use super::WORD_BYTES;
use super::errors::{
    RenderDeterministicLoweringError, gpu_program_contract, gpu_resource_descriptor,
    gpu_transfer_preparation, gpu_work_operation, map_maintained_program_build_error,
};
use super::output_context::{PreparedTemporalState, ResolvedOutputContext};
use super::packing::{PackedOutput, pack_matrix3, pack_vec3, positive_f32_bits};
use super::state::{
    DeterministicBufferKind, DeterministicResourceCache, DeterministicTemporalHistoryUseStorage,
};
use crate::request::{RenderObservationSpec, RenderPerspectiveObservation};
use runen_gpu::{
    GpuBufferDescriptor, GpuBufferHandle, GpuBufferInitialization, GpuBufferRegion, GpuBufferUsage,
    GpuClearOperation, GpuComputeOperation, GpuComputePipelineDescriptor, GpuDispatchIntent,
    GpuDispatchSize, GpuReconstruction, GpuResourceLifetime, GpuRuntimeBindingValue,
    GpuUploadOperation, PreparedGpuData, TransferData,
};

pub(super) struct PreparedPrimaryPass {
    pub(super) input: GpuBufferHandle,
    pub(super) canonical_output: GpuBufferHandle,
    pub(super) definedness: GpuBufferHandle,
    pub(super) status: GpuBufferHandle,
    pub(super) current_depth: GpuBufferHandle,
    pub(super) current_hit: GpuBufferHandle,
    pub(super) input_upload: GpuUploadOperation,
    pub(super) output_clear: GpuClearOperation,
    pub(super) definedness_clear: GpuClearOperation,
    pub(super) status_clear: GpuClearOperation,
    pub(super) current_depth_clear: GpuClearOperation,
    pub(super) current_hit_clear: GpuClearOperation,
    pub(super) compute: GpuComputeOperation,
}

pub(super) struct PreparedTemporalPass {
    pub(super) camera_parameter_upload: Option<GpuUploadOperation>,
    pub(super) reconstruction_compute: Option<GpuComputeOperation>,
    pub(super) static_fallback: Option<PreparedStaticFallback>,
}

/// A private phase-aligned output and correlated dense availability carrier.
///
/// These resources are fully rewritten for the exact output occurrence and are
/// never part of retained history. A value 0 means unresolved, 1 a compatible
/// static estimator, and 2 a provisional current-phase sample.
pub(super) struct PreparedStaticFallback {
    pub(super) resolved: GpuBufferHandle,
    pub(super) availability: GpuBufferHandle,
    pub(super) phase_presence: GpuBufferHandle,
    pub(super) clears: Vec<GpuClearOperation>,
    pub(super) compute: GpuComputeOperation,
}

fn prepare_static_fallback(
    packed: &PackedOutput,
    resolved: ResolvedOutputContext<'_>,
    primary: &PreparedPrimaryPass,
    history: &GpuBufferHandle,
    sample_counts: &GpuBufferHandle,
    resources: &mut DeterministicResourceCache,
) -> Result<PreparedStaticFallback, RenderDeterministicLoweringError> {
    let width = packed.input_words[header::REQUESTED_WIDTH];
    let height = packed.input_words[header::REQUESTED_HEIGHT];
    let count = width.checked_mul(height).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "temporal fallback requested cell count",
        },
    )?;
    let availability_bytes = u64::from(count).checked_mul(WORD_BYTES).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "temporal fallback cell state",
        },
    )?;
    let resolved_bytes = history.descriptor().size_bytes();
    let peak_scratch_bytes = resolved_bytes
        .checked_add(availability_bytes)
        .and_then(|bytes| bytes.checked_add(availability_bytes))
        .ok_or(RenderDeterministicLoweringError::SizeOverflow {
            field: "temporal fallback aggregate scratch",
        })?;
    if peak_scratch_bytes > temporal_fallback::MAX_PER_OUTPUT_SCRATCH_BYTES {
        return Err(RenderDeterministicLoweringError::TemporalFallbackScratchBudgetExceeded {
            output_index: resolved.output_index,
            required_bytes: peak_scratch_bytes,
            budget_bytes: temporal_fallback::MAX_PER_OUTPUT_SCRATCH_BYTES,
        });
    }
    // Resource and storage-binding limits are checked on the actual admitted
    // RunenGPU device/workload profile before allocating or dispatching.
    for (carrier, bytes) in [
        ("resolved radiance", resolved_bytes),
        ("phase presence", availability_bytes),
        ("cell availability", availability_bytes),
    ] {
        let limit_bytes = resolved
            .max_storage_buffer_binding_size
            .min(resolved.max_buffer_size);
        if bytes > limit_bytes {
            return Err(RenderDeterministicLoweringError::TemporalFallbackGpuLimitExceeded {
                output_index: resolved.output_index,
                carrier,
                required_bytes: bytes,
                limit_bytes,
            });
        }
    }
    // Reuse continuity-local logical identities, not per-frame GPU handles.
    // The retained history buffers remain disjoint from all three scratch carriers.
    let descriptions = [
        (DeterministicBufferKind::TemporalProvisional, resolved_bytes),
        (DeterministicBufferKind::TemporalPhasePresence, availability_bytes),
        (DeterministicBufferKind::TemporalAvailability, availability_bytes),
    ];
    let handles = descriptions
        .into_iter()
        .map(|(kind, bytes)| {
            resources.buffer(
                resolved.output_index,
                kind,
                GpuBufferDescriptor::ordinary_owned(
                    format!("RunenRender output {} {kind:?}", resolved.output_index),
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
                .map_err(|error| {
                    gpu_resource_descriptor("temporal fallback buffer descriptor", error)
                })?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let clears = handles
        .iter()
        .map(|handle| {
            GpuClearOperation::buffer_zero(
                GpuBufferRegion::whole(handle)
                    .map_err(|error| gpu_work_operation("temporal fallback clear region", error))?,
            )
            .map_err(|error| gpu_work_operation("temporal fallback clear", error))
        })
        .collect::<Result<Vec<_>, _>>()?;

    let source = retained_temporal_fallback_source().map_err(map_maintained_program_build_error)?;
    let pipeline = GpuComputePipelineDescriptor::ordinary(source, "main")
        .map_err(|error| gpu_program_contract("temporal fallback pipeline", error))?;
    let bindings = pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, &primary.input),
            GpuRuntimeBindingValue::whole_buffer(0, 1, history),
            GpuRuntimeBindingValue::whole_buffer(0, 2, sample_counts),
            GpuRuntimeBindingValue::whole_buffer(0, 3, &handles[0]),
            GpuRuntimeBindingValue::whole_buffer(0, 4, &handles[1]),
            GpuRuntimeBindingValue::whole_buffer(0, 5, &handles[2]),
        ])
        .map_err(|error| gpu_program_contract("temporal fallback bindings", error))?;
    let dispatch =
        deterministic_dispatch_size(count, resolved.max_compute_workgroups_per_dimension)?;
    let compute = GpuComputeOperation::new(pipeline, bindings, GpuDispatchIntent::direct(dispatch))
        .map_err(|error| gpu_work_operation("temporal fallback compute", error))?;
    Ok(PreparedStaticFallback {
        resolved: handles[0].clone(),
        phase_presence: handles[1].clone(),
        availability: handles[2].clone(),
        clears,
        compute,
    })
}

pub(super) struct PreparedOutputPasses {
    pub(super) requested_coverage: Option<PreparedRequestedCoverage>,
    pub(super) primary: PreparedPrimaryPass,
    pub(super) temporal: PreparedTemporalPass,
}

pub(super) fn prepare_primary_pass(
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

pub(super) fn prepare_temporal_pass(
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
            static_fallback: None,
        });
    };

    match &history.storage {
        DeterministicTemporalHistoryUseStorage::Static {
            handle,
            sample_counts,
            ..
        } => {
            let fallback = prepare_static_fallback(
                packed,
                resolved,
                primary,
                handle,
                sample_counts,
                resources,
            )?;
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
                    GpuRuntimeBindingValue::whole_buffer(0, 5, &fallback.resolved),
                    GpuRuntimeBindingValue::whole_buffer(0, 6, &fallback.phase_presence),
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
                static_fallback: Some(fallback),
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
            let source = retained_camera_reprojection_source()
                .map_err(map_maintained_program_build_error)?;
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
                static_fallback: None,
            })
        }
    }
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
    max_compute_workgroups_per_dimension: u32,
    resources: &mut DeterministicResourceCache,
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
    let dispatch =
        deterministic_dispatch_size(packed.sample_count, max_compute_workgroups_per_dimension)?;
    let compute = GpuComputeOperation::new(pipeline, bindings, GpuDispatchIntent::direct(dispatch))
        .map_err(|error| gpu_work_operation("coverage compute", error))?;
    Ok(PreparedRequestedCoverage {
        input_upload,
        clears,
        compute,
    })
}
