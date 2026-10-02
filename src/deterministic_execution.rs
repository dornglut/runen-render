//! Owner-controlled ordinary execution for the maintained deterministic RunenRender evaluator.
//!
//! This module lowers one exact [`AdmittedDeterministicRender`] into backend-neutral public
//! RunenGPU work and submits every admitted output through one exact `GpuSubmission`. It does not
//! re-plan renderer semantics, expose caller-authored GPU fragments, or make physical encoding part
//! of renderer-semantic result identity.
//!
//! The maintained carrier is one 32-bit word per semantic sample. Radiance and forward depth use
//! `f32` bits. Object identity uses an execution-local non-zero `u32` code decoded only through the
//! returned [`RenderObjectIdentityDecoder`]. Renderer-private definedness and evaluator-status words
//! remain distinct from payload bits. For this maintained direct/no-environment method only, a
//! primary radiance miss is the defined value zero; generic R2 radiance-miss semantics remain wider.

use super::admission::{AdmittedRenderPlan, RenderOutputDestination};
use super::derived_transform::{
    RenderCompiledMetricSimilarityTransform, RenderCompiledMetricSimilarityTransformError,
    RenderCompiledObjectTransform, RenderCompiledObjectTransformError,
};
use super::deterministic_admission::AdmittedDeterministicRender;
pub use super::deterministic_capture::{
    RenderCapturedDeterministicRadiance, RenderDeterministicRadianceCaptureError,
    RenderDeterministicRadianceCaptureRequest, RenderDeterministicRadianceCaptureRequestError,
};
use super::deterministic_carrier;
use super::field_input::{
    RenderFieldSemanticInput, RenderFieldSemanticInputBinding, RenderFieldSemanticInputGeneration,
};
use super::lowering::RenderWorkSet;
use super::render_result::RenderResult;
use super::representation::{RenderRepresentationId, RenderRepresentationProtocol};
use super::request::{
    RenderDistanceConvention, RenderObservationSpec, RenderOutputSpec, RenderOutputValue,
    RenderPerspectiveObservation, RenderSamplingSupport,
};
use super::scene::{RenderObjectId, RenderSceneRevision};
#[cfg(test)]
use super::shader_bridge::build_maintained_program_sources;
use super::shader_bridge::{
    RenderMaintainedProgramBuildError, RenderRunenShaderCompilationError,
    retained_camera_reprojection_source, retained_maintained_evaluator_source,
    retained_temporal_reconstruction_source,
};
use super::space_time::RenderTimeInterval;
use super::surface_input::{
    RenderSurfaceSemanticInputBinding, RenderSurfaceSemanticInputGeneration,
    RenderSurfaceSemanticInputView,
};
use runen_gpu::{
    GpuAdmittedProgramSource, GpuBufferDescriptor, GpuBufferHandle, GpuBufferInitialization,
    GpuBufferRegion, GpuBufferTextureLayout, GpuBufferUsage, GpuClearOperation,
    GpuComputeOperation, GpuComputePipelineDescriptor, GpuContext, GpuContextAffinity,
    GpuCopyOperation, GpuDispatchIntent, GpuDispatchSize, GpuExportKey, GpuExportRelationship,
    GpuInitialCoverage, GpuOrdinaryTransferPreparationError, GpuProgramContractError,
    GpuProgramSourceError, GpuReadbackId, GpuReadbackOperation, GpuReadbackRequestError,
    GpuReadbackStatus, GpuReconstruction, GpuResourceAccessIntent, GpuResourceDescriptorError,
    GpuResourceLifetime, GpuResourceProvenance, GpuResourceRef, GpuRuntimeBindingValue,
    GpuSubmission, GpuSubmissionFailureKind, GpuSubmissionStatus, GpuTextureAccessResource,
    GpuTextureCopyRegion, GpuTextureFormat, GpuTextureHandle, GpuUploadOperation,
    GpuWorkAuthoringError, GpuWorkFragment, GpuWorkImport, GpuWorkOperationError, GpuWorkOutput,
    GpuWorkResourceIdAllocationError, GpuWorkResourceIdAllocator, GpuWorkSubmissionError,
    PreparedGpuData, TransferData,
};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::sync::LazyLock;

#[cfg(test)]
#[path = "deterministic_camera_history_proof.rs"]
mod camera_history_proof;

#[cfg(test)]
#[path = "deterministic_requested_coverage_proof.rs"]
mod deterministic_execution_r7_proof_coverage;

const WORD_BYTES: u64 = deterministic_carrier::WORD_BYTES as u64;
const HEADER_WORDS: usize = 30;
const GEOMETRY_WORDS: usize = 40;
const EMITTER_WORDS: usize = 4;
const WORKGROUP_SIZE: u32 = 64;
const OUTPUT_RADIANCE: u32 = 1;
const OUTPUT_FORWARD_DEPTH: u32 = 2;
const OUTPUT_OBJECT_IDENTITY: u32 = 3;
const EXECUTION_REQUESTED_COVERAGE: u32 = 4;
const REQUESTED_COVERAGE_POLICY_REVISION: u32 = 1;
const OBSERVATION_PERSPECTIVE: u32 = 1;
const OBSERVATION_PROBE: u32 = 2;
const OBSERVATION_PERSPECTIVE_FOOTPRINT: u32 = 3;
const TEMPORAL_SEQUENCE_REVISION: u32 = 1;
const TEMPORAL_RECONSTRUCTION_REVISION: u32 = 2;
const CAMERA_REPROJECTION_REVISION: u32 = 3;
const CAMERA_DEPTH_POLICY_REVISION: u32 = 1;
const CAMERA_HISTORY_WORDS_PER_SAMPLE: u64 = 8;
const CURRENT_HIT_WORDS_PER_SAMPLE: u64 = 4;
const CAMERA_DIAGNOSTIC_WORDS: u64 = 32 * 32;
const MAINTAINED_EVALUATOR_REVISION: u64 = 3;
const CAMERA_DEPTH_ABSOLUTE_EPSILON: f32 = 0.001;

const CAMERA_DEPTH_RELATIVE_EPSILON: f32 = 0.001;
const TEMPORAL_PHASE_COUNT: u32 = 4;
const SHAPE_SPHERE: u32 = 1;
const SHAPE_PLANE: u32 = 2;
const SHAPE_FIELD: u32 = 3;
const SCENE_QUERY_WGSL: &str = include_str!("deterministic_scene_query.wgsl");
static MAINTAINED_WGSL: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{SCENE_QUERY_WGSL}\n{}",
        include_str!("deterministic_execution.wgsl")
    )
});
const TEMPORAL_RECONSTRUCTION_WGSL: &str =
    include_str!("deterministic_temporal_reconstruction.wgsl");
static CAMERA_REPROJECTION_WGSL: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{SCENE_QUERY_WGSL}\n{}",
        include_str!("deterministic_camera_reprojection.wgsl")
    )
});

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum DeterministicBufferKind {
    Input,
    CanonicalOutput,
    Definedness,
    Status,
    CurrentDepth,
    CurrentHit,
    CameraParameters,
    CoverageInput,
    CoverageDepth,
    CoverageState,
    CoverageStatusScratch,
    CoverageDepthScratch,
    CoverageHitScratch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DeterministicTemporalObservationCompatibility {
    Exact(RenderObservationSpec),
    CameraPerspective {
        vertical_field_of_view_bits: u64,
        aspect_ratio_bits: u64,
        shutter: RenderTimeInterval,
        sampling_support: RenderSamplingSupport,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DeterministicTemporalSignature {
    scene_revision: RenderSceneRevision,
    observation: DeterministicTemporalObservationCompatibility,
    output: RenderOutputSpec,
    semantic_inputs: Vec<RenderSurfaceSemanticInputBinding>,
    field_semantic_inputs: Vec<RenderFieldSemanticInputBinding>,
    evaluation_extent: (u32, u32),
    sequence_revision: u32,
    reconstruction_revision: u32,
    camera_reprojection_revision: Option<u32>,
    depth_policy_revision: Option<u32>,
}

#[derive(Debug)]
struct DeterministicCameraTemporalStorage {
    slots: [GpuBufferHandle; 2],
    completed_slot: usize,
    completed_observation: Option<RenderPerspectiveObservation>,
    same_pose_completed_frames: u32,
    pending_slot: Option<usize>,
    pending_observation: Option<RenderPerspectiveObservation>,
}

#[derive(Debug)]
enum DeterministicTemporalStorage {
    Static {
        handle: GpuBufferHandle,
        sample_counts: GpuBufferHandle,
        row_stride_words: u32,
    },
    Camera(Box<DeterministicCameraTemporalStorage>),
}

#[derive(Debug)]
struct DeterministicTemporalHistory {
    signature: DeterministicTemporalSignature,
    storage: DeterministicTemporalStorage,
    generation: u64,
    phase: u32,
    age: u32,
}

#[derive(Debug, Clone)]
enum DeterministicTemporalHistoryUseStorage {
    Static {
        handle: GpuBufferHandle,
        sample_counts: GpuBufferHandle,
        row_stride_words: u32,
    },
    Camera {
        previous_history: GpuBufferHandle,
        current_history: GpuBufferHandle,
        previous_observation: Option<RenderPerspectiveObservation>,
        pose_changed: bool,
        same_pose_completed_frames: u32,
    },
}

#[derive(Debug, Clone)]
struct DeterministicTemporalHistoryUse {
    storage: DeterministicTemporalHistoryUseStorage,
    generation: u64,
    reset: bool,
    phase: u32,
    age: u32,
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRenderExecutionSelection {
    scope: u64,
    finite_evaluation: Option<(usize, (u32, u32))>,
    produce_requested_coverage: bool,
}

#[derive(Debug, Clone, Copy)]
struct DeterministicOutputExecutionSelection {
    scope: u64,
    finite_evaluation_extent: Option<(u32, u32)>,
    produce_requested_coverage: bool,
}

/// Physical execution selection; private coverage is not a requested semantic depth output.
#[derive(Debug, Clone, Copy)]
enum MaintainedExecutionKind {
    Semantic(RenderOutputValue),
    RequestedCoverage,
}

#[derive(Debug, Clone, Copy)]
struct DeterministicOutputPackingState<'a> {
    finite_evaluation_extent: Option<(u32, u32)>,
    temporal_history: Option<&'a DeterministicTemporalHistoryUse>,
}

#[derive(Debug, Clone, Copy)]
struct DeterministicTemporalHistorySelection {
    current_observation: RenderPerspectiveObservation,
    camera_capable: bool,
}

/// Renderer-owned logical buffer identities reused by ordinary composed frames.
///
/// RunenGPU's bind-group realization retains the resource dependencies of each realized binding.
/// Rebuilding these identities for every interactive frame would therefore grow the authoritative
/// realization registry without bound. The cache is scoped to one renderer/context generation and
/// one deterministic producer; a descriptor change, such as a resize, deliberately allocates a
/// replacement identity.
#[derive(Debug, Default)]
pub(crate) struct DeterministicResourceCache {
    identities: GpuWorkResourceIdAllocator,
    buffers: BTreeMap<(u64, usize, DeterministicBufferKind), GpuBufferHandle>,
    temporal_histories: BTreeMap<(u64, usize), DeterministicTemporalHistory>,
    next_temporal_generation: u64,
    prepared_temporal_outputs: BTreeMap<u64, BTreeSet<usize>>,
    // Keep the latest accepted graph correlated with every producer namespace whose mutable
    // intermediates it used. A peer surface's submission must not stall this producer's cache.
    producer_submissions: BTreeMap<u64, GpuSubmission>,
}

impl DeterministicResourceCache {
    pub(crate) fn any_producer_submission_in_flight(
        &self,
        producer_scopes: impl IntoIterator<Item = u64>,
    ) -> bool {
        any_producer_scope_in_flight(producer_scopes, |producer_scope| {
            self.producer_submissions
                .get(&producer_scope)
                .is_some_and(|submission| {
                    matches!(submission.status(), GpuSubmissionStatus::Accepted)
                })
        })
    }

    pub(crate) fn record_producer_submission(
        &mut self,
        producer_scope: u64,
        _frame_index: u64,
        submission: &GpuSubmission,
    ) {
        self.producer_submissions
            .insert(producer_scope, submission.clone());
    }

    pub(crate) fn retain_in_flight_submissions(&mut self) {
        let terminal = self
            .producer_submissions
            .iter()
            .filter_map(|(scope, submission)| match submission.status() {
                GpuSubmissionStatus::Accepted => None,
                GpuSubmissionStatus::Completed => Some((*scope, true)),
                GpuSubmissionStatus::Failed(_) => Some((*scope, false)),
            })
            .collect::<Vec<_>>();

        for (scope, completed) in terminal {
            self.reconcile_temporal_outputs(scope, completed);
        }

        // Completed and failed submissions are terminal; only an Accepted handle can still be
        // using a producer's reusable intermediates.
        self.producer_submissions
            .retain(|_, submission| matches!(submission.status(), GpuSubmissionStatus::Accepted));
    }

    fn reconcile_temporal_outputs(&mut self, scope: u64, completed: bool) {
        let outputs = self
            .prepared_temporal_outputs
            .remove(&scope)
            .unwrap_or_default();
        if completed {
            for output_index in outputs {
                if let Some(history) = self.temporal_histories.get_mut(&(scope, output_index)) {
                    if let DeterministicTemporalStorage::Camera(camera) = &mut history.storage {
                        if let Some(slot) = camera.pending_slot.take() {
                            camera.completed_slot = slot;
                        }
                        if let Some(observation) = camera.pending_observation.take() {
                            camera.same_pose_completed_frames = match camera.completed_observation {
                                None => 1,
                                Some(previous)
                                    if previous.observation_to_scene()
                                        != observation.observation_to_scene() =>
                                {
                                    0
                                }
                                Some(_) => camera
                                    .same_pose_completed_frames
                                    .saturating_add(1)
                                    .min(TEMPORAL_PHASE_COUNT),
                            };
                            camera.completed_observation = Some(observation);
                        }
                    }
                    history.phase = (history.phase + 1) % TEMPORAL_PHASE_COUNT;
                    history.age = history.age.saturating_add(1);
                }
            }
        } else {
            self.temporal_histories
                .retain(|(history_scope, _), _| *history_scope != scope);
        }
    }

    fn maintained_source(
        &mut self,
    ) -> Result<GpuAdmittedProgramSource, RenderDeterministicLoweringError> {
        retained_maintained_evaluator_source(
            MAINTAINED_EVALUATOR_REVISION,
            MAINTAINED_WGSL.as_str(),
        )
        .map_err(map_maintained_program_build_error)
    }

    fn reconstruction_source(
        &mut self,
    ) -> Result<GpuAdmittedProgramSource, RenderDeterministicLoweringError> {
        retained_temporal_reconstruction_source(
            u64::from(TEMPORAL_RECONSTRUCTION_REVISION),
            TEMPORAL_RECONSTRUCTION_WGSL,
        )
        .map_err(map_maintained_program_build_error)
    }

    fn camera_reprojection_source(
        &mut self,
    ) -> Result<GpuAdmittedProgramSource, RenderDeterministicLoweringError> {
        retained_camera_reprojection_source(
            u64::from(CAMERA_REPROJECTION_REVISION),
            CAMERA_REPROJECTION_WGSL.as_str(),
        )
        .map_err(map_maintained_program_build_error)
    }

    fn temporal_history(
        &mut self,
        scope: u64,
        output_index: usize,
        signature: DeterministicTemporalSignature,
        requested_extent: (u32, u32),
        bytes_per_row_alignment: u64,
        selection: DeterministicTemporalHistorySelection,
    ) -> Result<DeterministicTemporalHistoryUse, RenderDeterministicLoweringError> {
        let DeterministicTemporalHistorySelection {
            current_observation,
            camera_capable,
        } = selection;
        let logical_row_bytes = u64::from(requested_extent.0)
            .checked_mul(WORD_BYTES)
            .ok_or(RenderDeterministicLoweringError::SizeOverflow {
                field: "temporal history logical row bytes",
            })?;
        let row_bytes = align_up(logical_row_bytes, bytes_per_row_alignment)?;
        if row_bytes % WORD_BYTES != 0 {
            return Err(
                RenderDeterministicLoweringError::InvalidBytesPerRowAlignment {
                    alignment: bytes_per_row_alignment,
                },
            );
        }
        let row_stride_words = u32::try_from(row_bytes / WORD_BYTES).map_err(|_| {
            RenderDeterministicLoweringError::SizeOverflow {
                field: "temporal history row stride",
            }
        })?;
        let words = row_stride_words.checked_mul(requested_extent.1).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "temporal history word count",
            },
        )?;
        let byte_len = u64::from(words).checked_mul(WORD_BYTES).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "temporal history byte length",
            },
        )?;
        let camera_sample_count = u64::from(requested_extent.0)
            .checked_mul(u64::from(requested_extent.1))
            .ok_or(RenderDeterministicLoweringError::SizeOverflow {
                field: "camera history sample count",
            })?;
        let camera_byte_len = camera_sample_count
            .checked_mul(CAMERA_HISTORY_WORDS_PER_SAMPLE)
            .and_then(|words| words.checked_add(CAMERA_DIAGNOSTIC_WORDS))
            .and_then(|words| words.checked_mul(WORD_BYTES))
            .ok_or(RenderDeterministicLoweringError::SizeOverflow {
                field: "camera history byte length",
            })?;

        let key = (scope, output_index);
        let recreate = self.temporal_histories.get(&key).is_none_or(|history| {
            if history.signature != signature {
                return true;
            }
            match (&history.storage, camera_capable) {
                (DeterministicTemporalStorage::Camera(_), true) => false,
                (
                    DeterministicTemporalStorage::Static {
                        row_stride_words: retained,
                        ..
                    },
                    false,
                ) => *retained != row_stride_words,
                _ => true,
            }
        });
        if recreate {
            let storage = if camera_capable {
                let descriptor = |slot: usize| {
                    GpuBufferDescriptor::ordinary_owned(
                        format!(
                            "RunenRender output {output_index} camera temporal history slot {slot}"
                        ),
                        GpuResourceLifetime::Retained,
                        GpuReconstruction::SourceBacked,
                        camera_byte_len,
                        [GpuBufferUsage::Storage, GpuBufferUsage::CopySource],
                        GpuBufferInitialization::Zeroed,
                    )
                    .map_err(|error| {
                        gpu_resource_descriptor("camera temporal-history descriptor", error)
                    })
                };
                let first = self
                    .identities
                    .allocate_buffer_handle(descriptor(0)?)
                    .map_err(|error| {
                        gpu_resource_allocation("camera temporal-history allocation", error)
                    })?;
                let second = self
                    .identities
                    .allocate_buffer_handle(descriptor(1)?)
                    .map_err(|error| {
                        gpu_resource_allocation("camera temporal-history allocation", error)
                    })?;
                DeterministicTemporalStorage::Camera(Box::new(DeterministicCameraTemporalStorage {
                    slots: [first, second],
                    completed_slot: 0,
                    completed_observation: None,
                    same_pose_completed_frames: 0,
                    pending_slot: None,
                    pending_observation: None,
                }))
            } else {
                let descriptor = GpuBufferDescriptor::ordinary_owned(
                    format!("RunenRender output {output_index} temporal history"),
                    GpuResourceLifetime::Retained,
                    GpuReconstruction::SourceBacked,
                    byte_len,
                    [GpuBufferUsage::Storage, GpuBufferUsage::CopySource],
                    GpuBufferInitialization::Zeroed,
                )
                .map_err(|error| gpu_resource_descriptor("temporal-history descriptor", error))?;
                let handle =
                    self.identities
                        .allocate_buffer_handle(descriptor)
                        .map_err(|error| {
                            gpu_resource_allocation("temporal-history allocation", error)
                        })?;
                let count_descriptor = GpuBufferDescriptor::ordinary_owned(
                    format!("RunenRender output {output_index} temporal sample counts"),
                    GpuResourceLifetime::Retained,
                    GpuReconstruction::SourceBacked,
                    byte_len,
                    [GpuBufferUsage::Storage],
                    GpuBufferInitialization::Zeroed,
                )
                .map_err(|error| {
                    gpu_resource_descriptor("temporal sample-count descriptor", error)
                })?;
                let sample_counts = self
                    .identities
                    .allocate_buffer_handle(count_descriptor)
                    .map_err(|error| {
                        gpu_resource_allocation("temporal sample-count allocation", error)
                    })?;
                DeterministicTemporalStorage::Static {
                    handle,
                    sample_counts,
                    row_stride_words,
                }
            };
            self.next_temporal_generation = self.next_temporal_generation.saturating_add(1);
            self.temporal_histories.insert(
                key,
                DeterministicTemporalHistory {
                    signature,
                    storage,
                    generation: self.next_temporal_generation,
                    phase: 0,
                    age: 0,
                },
            );
        }
        self.prepared_temporal_outputs
            .entry(scope)
            .or_default()
            .insert(output_index);
        let history = self
            .temporal_histories
            .get_mut(&key)
            .expect("temporal history inserted before use");
        let storage = match &mut history.storage {
            DeterministicTemporalStorage::Static {
                handle,
                sample_counts,
                row_stride_words,
            } => DeterministicTemporalHistoryUseStorage::Static {
                handle: handle.clone(),
                sample_counts: sample_counts.clone(),
                row_stride_words: *row_stride_words,
            },
            DeterministicTemporalStorage::Camera(camera) => {
                let write_slot = 1 - camera.completed_slot;
                let previous_observation = camera.completed_observation;
                let pose_changed = previous_observation.is_some_and(|previous| {
                    previous.observation_to_scene() != current_observation.observation_to_scene()
                });
                camera.pending_slot = Some(write_slot);
                camera.pending_observation = Some(current_observation);
                DeterministicTemporalHistoryUseStorage::Camera {
                    previous_history: camera.slots[camera.completed_slot].clone(),
                    current_history: camera.slots[write_slot].clone(),
                    previous_observation,
                    pose_changed,
                    same_pose_completed_frames: camera.same_pose_completed_frames,
                }
            }
        };
        Ok(DeterministicTemporalHistoryUse {
            storage,
            generation: history.generation,
            // A freshly allocated history remains bootstrap/current-only until one compatible
            // submission has completed. Preparation or submission rejection before acceptance must
            // not turn a zeroed, never-completed generation into apparent compatible reuse.
            reset: recreate || history.age == 0,
            phase: history.phase,
            age: history.age,
        })
    }

    fn buffer(
        &mut self,
        scope: u64,
        output_index: usize,
        kind: DeterministicBufferKind,
        descriptor: GpuBufferDescriptor,
    ) -> Result<GpuBufferHandle, RenderDeterministicLoweringError> {
        let key = (scope, output_index, kind);
        if let Some(existing) = self.buffers.get(&key)
            && existing.descriptor() == &descriptor
        {
            return Ok(existing.clone());
        }
        let handle = self
            .identities
            .allocate_buffer_handle(descriptor)
            .map_err(|error| gpu_resource_allocation("deterministic buffer allocation", error))?;
        self.buffers.insert(key, handle.clone());
        Ok(handle)
    }
}

fn temporal_evaluation_extent_supported(
    requested_extent: (u32, u32),
    evaluation_extent: (u32, u32),
) -> bool {
    evaluation_extent.0 <= requested_extent.0
        && evaluation_extent.1 <= requested_extent.1
        && u64::from(evaluation_extent.0) * 2 >= u64::from(requested_extent.0)
        && u64::from(evaluation_extent.1) * 2 >= u64::from(requested_extent.1)
}

fn temporal_observation_compatibility(
    observation: RenderObservationSpec,
    camera_capable: bool,
) -> DeterministicTemporalObservationCompatibility {
    match (observation, camera_capable) {
        (RenderObservationSpec::Perspective(perspective), true) => {
            DeterministicTemporalObservationCompatibility::CameraPerspective {
                vertical_field_of_view_bits: perspective.vertical_field_of_view_radians().to_bits(),
                aspect_ratio_bits: perspective.aspect_ratio().to_bits(),
                shutter: perspective.shutter(),
                sampling_support: perspective.sampling_support(),
            }
        }
        (observation, false) => DeterministicTemporalObservationCompatibility::Exact(observation),
        (RenderObservationSpec::Probe(_), true) => {
            DeterministicTemporalObservationCompatibility::Exact(observation)
        }
    }
}

fn any_producer_scope_in_flight(
    producer_scopes: impl IntoIterator<Item = u64>,
    mut producer_is_in_flight: impl FnMut(u64) -> bool,
) -> bool {
    producer_scopes.into_iter().any(&mut producer_is_in_flight)
}

/// Physical object-identity decoder owned by one exact maintained execution.
///
/// Compact GPU words are not renderer-semantic object identities. Code zero and every unknown code
/// decode to `None`; a non-zero code is meaningful only through this exact immutable codebook.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderObjectIdentityDecoder {
    objects_by_code: Vec<RenderObjectId>,
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
    admitted: AdmittedDeterministicRender,
    work_set: RenderWorkSet,
    object_identity_decoder: RenderObjectIdentityDecoder,
    radiance_outputs: Vec<PreparedDeterministicRadianceOutput>,
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
    output_index: usize,
    relationship: GpuExportRelationship,
    temporal_evidence: Option<RenderTemporalExecutionEvidence>,
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
pub(super) struct DeterministicVerificationReadbacks {
    output_index: usize,
    canonical_output: GpuReadbackId,
    definedness: GpuReadbackId,
    status: GpuReadbackId,
}

impl DeterministicVerificationReadbacks {
    pub(super) const fn output_index(self) -> usize {
        self.output_index
    }

    pub(super) const fn canonical_output(self) -> GpuReadbackId {
        self.canonical_output
    }

    pub(super) const fn definedness(self) -> GpuReadbackId {
        self.definedness
    }

    pub(super) const fn status(self) -> GpuReadbackId {
        self.status
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeterministicObservationIntent {
    Ordinary,
    Verify,
}

impl DeterministicObservationIntent {
    const fn requires_private_readback(self) -> bool {
        matches!(self, Self::Verify)
    }
}

#[derive(Debug)]
enum DeterministicVerificationState {
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
    admitted: AdmittedDeterministicRender,
    submission: GpuSubmission,
    object_identity_decoder: RenderObjectIdentityDecoder,
    verification: DeterministicVerificationState,
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

    pub(super) const fn submission(&self) -> &GpuSubmission {
        &self.submission
    }

    pub(super) const fn result_is_formed(&self) -> bool {
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
        super::deterministic_capture::mint_request(self, output_index)
    }

    /// Consume one capture request and interpret its completed product-owned public readback.
    pub fn capture_deterministic_radiance(
        &self,
        request: RenderDeterministicRadianceCaptureRequest,
        context: &GpuContext,
        product_submission: &GpuSubmission,
    ) -> Result<RenderCapturedDeterministicRadiance, RenderDeterministicRadianceCaptureError> {
        super::deterministic_capture::capture(self, request, context, product_submission)
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
            super::deterministic_verification::verify_completed_deterministic_render(verification)
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
pub(super) struct DeterministicVerificationSubmission {
    submitted: SubmittedDeterministicRender,
}

impl DeterministicVerificationSubmission {
    pub(super) const fn submitted(&self) -> &SubmittedDeterministicRender {
        &self.submitted
    }

    pub(super) fn readbacks(&self) -> &[DeterministicVerificationReadbacks] {
        match &self.submitted.verification {
            DeterministicVerificationState::Requested(readbacks) => readbacks,
            DeterministicVerificationState::NotRequested
            | DeterministicVerificationState::Formed => &[],
        }
    }

    pub(super) fn into_submitted(self) -> SubmittedDeterministicRender {
        self.submitted
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RenderRunenGpuPreparationError {
    ProgramSource {
        stage: &'static str,
        source: GpuProgramSourceError,
    },
    ResourceDescriptor {
        stage: &'static str,
        source: GpuResourceDescriptorError,
    },
    ResourceAllocation {
        stage: &'static str,
        source: GpuWorkResourceIdAllocationError,
    },
    TransferPreparation {
        stage: &'static str,
        source: GpuOrdinaryTransferPreparationError,
    },
    ProgramContract {
        stage: &'static str,
        source: GpuProgramContractError,
    },
    WorkOperation {
        stage: &'static str,
        source: GpuWorkOperationError,
    },
    ReadbackRequest {
        stage: &'static str,
        source: GpuReadbackRequestError,
    },
    WorkAuthoring {
        stage: &'static str,
        source: GpuWorkAuthoringError,
    },
}

impl RenderRunenGpuPreparationError {
    pub const fn stage(&self) -> &'static str {
        match self {
            Self::ProgramSource { stage, .. }
            | Self::ResourceDescriptor { stage, .. }
            | Self::ResourceAllocation { stage, .. }
            | Self::TransferPreparation { stage, .. }
            | Self::ProgramContract { stage, .. }
            | Self::WorkOperation { stage, .. }
            | Self::ReadbackRequest { stage, .. }
            | Self::WorkAuthoring { stage, .. } => stage,
        }
    }

    fn owner_source(&self) -> &(dyn Error + 'static) {
        match self {
            Self::ProgramSource { source, .. } => source,
            Self::ResourceDescriptor { source, .. } => source,
            Self::ResourceAllocation { source, .. } => source,
            Self::TransferPreparation { source, .. } => source,
            Self::ProgramContract { source, .. } => source,
            Self::WorkOperation { source, .. } => source,
            Self::ReadbackRequest { source, .. } => source,
            Self::WorkAuthoring { source, .. } => source,
        }
    }
}

impl fmt::Display for RenderRunenGpuPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "RunenGPU {} failed: {}",
            self.stage(),
            self.owner_source()
        )
    }
}

impl Error for RenderRunenGpuPreparationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.owner_source())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RenderDeterministicLoweringError {
    ContextAffinityChanged {
        admitted: GpuContextAffinity,
        actual: GpuContextAffinity,
    },
    OutputCorrelationChanged {
        output_index: usize,
    },
    MissingObjectState {
        output_index: usize,
        object_id: RenderObjectId,
    },
    MissingSurfaceInput {
        output_index: usize,
        object_id: RenderObjectId,
        representation_id: RenderRepresentationId,
    },
    MissingFieldInput {
        output_index: usize,
        object_id: RenderObjectId,
        representation_id: RenderRepresentationId,
    },
    MissingTemporalSurfaceInputGeneration {
        output_index: usize,
        representation_id: RenderRepresentationId,
    },
    MissingTemporalFieldInputGeneration {
        output_index: usize,
        representation_id: RenderRepresentationId,
    },
    UnsupportedTemporalEvaluationExtent {
        output_index: usize,
        requested_extent: (u32, u32),
        evaluation_extent: (u32, u32),
    },
    NonInvertibleObjectTransform {
        output_index: usize,
        object_id: RenderObjectId,
    },
    NonSimilarityFieldTransform {
        output_index: usize,
        object_id: RenderObjectId,
    },
    MissingMaterial {
        output_index: usize,
        object_id: RenderObjectId,
    },
    UnsupportedOutput {
        output_index: usize,
    },
    MissingBytesPerRowAlignment,
    InvalidBytesPerRowAlignment {
        alignment: u64,
    },
    SizeOverflow {
        field: &'static str,
    },
    HostAllocation {
        field: &'static str,
    },
    NumericRealization {
        field: &'static str,
    },
    DispatchCapacityExceeded {
        sample_count: u32,
        workgroup_size: u32,
        required_workgroups: u64,
        max_workgroups_per_dimension: u32,
        capacity_workgroups: u64,
    },
    RunenShaderCompilation(RenderRunenShaderCompilationError),
    RunenGpuPreparation(RenderRunenGpuPreparationError),
}

impl fmt::Display for RenderDeterministicLoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextAffinityChanged { .. } => formatter.write_str(
                "maintained deterministic execution context differs from admitted RunenGPU context",
            ),
            Self::OutputCorrelationChanged { output_index } => write!(
                formatter,
                "maintained deterministic output correlation changed at output {output_index}"
            ),
            Self::MissingObjectState {
                output_index,
                object_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} has no retained spatial state"
            ),
            Self::MissingSurfaceInput {
                output_index,
                object_id,
                representation_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} representation {representation_id:?} has no maintained surface input"
            ),
            Self::MissingFieldInput {
                output_index,
                object_id,
                representation_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} representation {representation_id:?} has no maintained field input"
            ),
            Self::MissingTemporalSurfaceInputGeneration {
                output_index,
                representation_id,
            } => write!(
                formatter,
                "output {output_index} surface representation {representation_id:?} has no source generation required for retained temporal history"
            ),
            Self::MissingTemporalFieldInputGeneration {
                output_index,
                representation_id,
            } => write!(
                formatter,
                "output {output_index} field representation {representation_id:?} has no source generation required for retained temporal history"
            ),
            Self::UnsupportedTemporalEvaluationExtent {
                output_index,
                requested_extent,
                evaluation_extent,
            } => write!(
                formatter,
                "output {output_index} temporal evaluation extent {}x{} cannot cover requested lattice {}x{} with the maintained four-phase footprint sequence",
                evaluation_extent.0, evaluation_extent.1, requested_extent.0, requested_extent.1
            ),
            Self::NonInvertibleObjectTransform {
                output_index,
                object_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} has no finite invertible evaluator transform"
            ),
            Self::NonSimilarityFieldTransform {
                output_index,
                object_id,
            } => write!(
                formatter,
                "output {output_index} field object {object_id:?} has no accepted positive similarity transform"
            ),
            Self::MissingMaterial {
                output_index,
                object_id,
            } => write!(
                formatter,
                "radiance output {output_index} object {object_id:?} has no admitted material"
            ),
            Self::UnsupportedOutput { output_index } => write!(
                formatter,
                "output {output_index} is outside the maintained deterministic evaluator contract"
            ),
            Self::MissingBytesPerRowAlignment => {
                formatter.write_str("RunenGPU did not expose a texture bytes-per-row alignment")
            }
            Self::InvalidBytesPerRowAlignment { alignment } => write!(
                formatter,
                "RunenGPU exposed unusable texture bytes-per-row alignment {alignment}"
            ),
            Self::SizeOverflow { field } => {
                write!(
                    formatter,
                    "{field} exceeds maintained physical indexing limits"
                )
            }
            Self::HostAllocation { field } => {
                write!(formatter, "host allocation failed for {field}")
            }
            Self::NumericRealization { field } => write!(
                formatter,
                "{field} cannot be represented by the maintained finite f32 evaluator"
            ),
            Self::DispatchCapacityExceeded {
                sample_count,
                workgroup_size,
                required_workgroups,
                max_workgroups_per_dimension,
                capacity_workgroups,
            } => write!(
                formatter,
                "sample count {sample_count} with workgroup size {workgroup_size} requires {required_workgroups} workgroups, but the admitted maximum per dimension is {max_workgroups_per_dimension} and the 2D dispatch capacity is {capacity_workgroups} workgroups"
            ),
            Self::RunenShaderCompilation(error) => error.fmt(formatter),
            Self::RunenGpuPreparation(error) => error.fmt(formatter),
        }
    }
}

impl Error for RenderDeterministicLoweringError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RunenShaderCompilation(error) => Some(error),
            Self::RunenGpuPreparation(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub(super) enum RenderDeterministicExecutionError {
    Lowering(RenderDeterministicLoweringError),
    Submission(GpuWorkSubmissionError),
}

impl fmt::Display for RenderDeterministicExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lowering(error) => write!(formatter, "deterministic lowering failed: {error}"),
            Self::Submission(error) => write!(formatter, "RunenGPU submission failed: {error}"),
        }
    }
}

impl Error for RenderDeterministicExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Lowering(error) => Some(error),
            Self::Submission(error) => Some(error),
        }
    }
}

impl From<RenderDeterministicLoweringError> for RenderDeterministicExecutionError {
    fn from(value: RenderDeterministicLoweringError) -> Self {
        Self::Lowering(value)
    }
}

/// Failure while polling one exact verified submission for semantic result formation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum RenderDeterministicResultFormationError {
    VerificationNotRequested,
    ResultAlreadyFormed,
    SubmissionFailed {
        kind: GpuSubmissionFailureKind,
    },
    ReadbackCorrelationLost {
        output_index: usize,
        channel: &'static str,
    },
    ReadbackFailed {
        output_index: usize,
        channel: &'static str,
        kind: GpuSubmissionFailureKind,
    },
    Verification(super::deterministic_verification::RenderDeterministicVerificationError),
}

impl fmt::Display for RenderDeterministicResultFormationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VerificationNotRequested => formatter.write_str(
                "this deterministic submission was authored without verified-result observations",
            ),
            Self::ResultAlreadyFormed => formatter.write_str(
                "semantic result evidence was already formed from this deterministic submission",
            ),
            Self::SubmissionFailed { kind } => write!(
                formatter,
                "RunenGPU submission failed before verified result formation: {kind:?}"
            ),
            Self::ReadbackCorrelationLost {
                output_index,
                channel,
            } => write!(
                formatter,
                "output {output_index} {channel} verification readback is no longer correlated to the exact submission"
            ),
            Self::ReadbackFailed {
                output_index,
                channel,
                kind,
            } => write!(
                formatter,
                "output {output_index} {channel} verification readback failed: {kind:?}"
            ),
            Self::Verification(error) => write!(
                formatter,
                "deterministic finite-evaluation verification rejected result formation: {error}"
            ),
        }
    }
}

impl Error for RenderDeterministicResultFormationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Verification(error) => Some(error),
            _ => None,
        }
    }
}

struct LoweredDeterministicRender {
    work_set: RenderWorkSet,
    object_identity_decoder: RenderObjectIdentityDecoder,
    verification_readbacks: Vec<DeterministicVerificationReadbacks>,
    composable_radiance_outputs: Vec<PreparedDeterministicRadianceOutput>,
}

struct LoweredDeterministicOutput {
    fragment: GpuWorkFragment,
    verification_readbacks: Option<DeterministicVerificationReadbacks>,
    composable_radiance_output: Option<PreparedDeterministicRadianceOutput>,
}

struct PackedOutput {
    input_words: Vec<u32>,
    sample_count: u32,
    output_byte_len: u64,
    texture_row_bytes: Option<u32>,
}

struct VerificationReadbackOperations {
    correlation: DeterministicVerificationReadbacks,
    canonical_output: GpuReadbackOperation,
    definedness: GpuReadbackOperation,
    status: GpuReadbackOperation,
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
    super::deterministic_verification::RenderDeterministicVerifiedSubmissionError,
> {
    super::deterministic_verification::submit_deterministic_render_for_verified_formation(
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
pub(super) async fn submit_deterministic_render_for_verification(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
) -> Result<DeterministicVerificationSubmission, RenderDeterministicExecutionError> {
    let lowered = lower_deterministic_render(
        &admitted,
        context,
        DeterministicObservationIntent::Verify,
        &mut DeterministicResourceCache::default(),
        DeterministicRenderExecutionSelection {
            scope: 0,
            finite_evaluation: None,
            produce_requested_coverage: false,
        },
    )?;
    let verification_readbacks = lowered.verification_readbacks;
    let submitted = submit_lowered_deterministic_render(
        admitted,
        context,
        lowered.work_set,
        lowered.object_identity_decoder,
        DeterministicVerificationState::Requested(verification_readbacks),
    )
    .await?;
    Ok(DeterministicVerificationSubmission { submitted })
}

async fn submit_lowered_deterministic_render(
    admitted: AdmittedDeterministicRender,
    context: &GpuContext,
    work_set: RenderWorkSet,
    object_identity_decoder: RenderObjectIdentityDecoder,
    verification: DeterministicVerificationState,
) -> Result<SubmittedDeterministicRender, RenderDeterministicExecutionError> {
    let submission = context
        .submit_work(
            "RunenRender maintained deterministic execution",
            work_set.fragments().iter().cloned(),
        )
        .await
        .map_err(RenderDeterministicExecutionError::Submission)?;
    Ok(SubmittedDeterministicRender {
        admitted,
        submission,
        object_identity_decoder,
        verification,
    })
}

async fn submit_prepared_deterministic_render(
    prepared: PreparedDeterministicRender,
    context: &GpuContext,
) -> Result<SubmittedDeterministicRender, RenderDeterministicExecutionError> {
    let submission = context
        .submit_work(
            "RunenRender maintained deterministic execution",
            prepared.work_set.fragments().iter().cloned(),
        )
        .await
        .map_err(RenderDeterministicExecutionError::Submission)?;
    Ok(SubmittedDeterministicRender {
        admitted: prepared.admitted,
        submission,
        object_identity_decoder: prepared.object_identity_decoder,
        verification: DeterministicVerificationState::NotRequested,
    })
}

fn lower_deterministic_render(
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

fn build_object_identity_decoder(
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

fn lower_output(
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

struct PreparedRequestedCoverage {
    input_upload: GpuUploadOperation,
    clears: Vec<GpuClearOperation>,
    compute: GpuComputeOperation,
}

/// One current requested-extent dispatch, with no history storage or runtime readback. Bindings
/// 1/2 carry padded depth and tightly packed Invalid=0 / Background=1 / Hit=2 states. The ordinary
/// evaluator's other bindings are distinct one-word scratch resources: coverage returns before
/// any access to them, including ordinary invalidation. No phase-sparse resource is reused.
fn prepare_requested_coverage(
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

fn pack_output(
    admitted: &AdmittedRenderPlan,
    admitted_output: &super::admission::RenderAdmittedOutput,
    execution_kind: MaintainedExecutionKind,
    observation: RenderObservationSpec,
    object_codes: &BTreeMap<RenderObjectId, u32>,
    context: &GpuContext,
    packing: DeterministicOutputPackingState<'_>,
) -> Result<PackedOutput, RenderDeterministicLoweringError> {
    let DeterministicOutputPackingState {
        finite_evaluation_extent,
        temporal_history,
    } = packing;
    let output_index = admitted_output.output_index();
    let topology = admitted.plan().request().outputs()[output_index]
        .spec()
        .topology();
    let requested_extent = topology.sample_lattice_dimensions();
    let physical_extent = match (requested_extent, finite_evaluation_extent) {
        (Some(_), Some(extent)) => extent,
        (Some(extent), None) => extent,
        (None, Some(_)) => {
            return Err(RenderDeterministicLoweringError::UnsupportedOutput { output_index });
        }
        (None, None) => (1, 1),
    };
    let (sample_count, width, height, row_stride_words, output_byte_len, texture_row_bytes) =
        if requested_extent.is_some() {
            let (width, height) = physical_extent;
            let sample_count = width.checked_mul(height).ok_or(
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "lattice sample count",
                },
            )?;
            let logical_row_bytes = u64::from(width).checked_mul(WORD_BYTES).ok_or(
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "lattice logical row bytes",
                },
            )?;
            let alignment = context
                .device_facts()
                .device_limits()
                .alignments()
                .bytes_per_row
                .ok_or(RenderDeterministicLoweringError::MissingBytesPerRowAlignment)?;
            let row_bytes = align_up(logical_row_bytes, alignment)?;
            if row_bytes % WORD_BYTES != 0 {
                return Err(
                    RenderDeterministicLoweringError::InvalidBytesPerRowAlignment { alignment },
                );
            }
            let row_stride_words = u32::try_from(row_bytes / WORD_BYTES).map_err(|_| {
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "lattice row stride",
                }
            })?;
            let output_words = row_stride_words.checked_mul(height).ok_or(
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "canonical lattice word count",
                },
            )?;
            let output_byte_len = u64::from(output_words).checked_mul(WORD_BYTES).ok_or(
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "canonical lattice byte length",
                },
            )?;
            let row_bytes = u32::try_from(row_bytes).map_err(|_| {
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "lattice row bytes",
                }
            })?;
            (
                sample_count,
                width,
                height,
                row_stride_words,
                output_byte_len,
                Some(row_bytes),
            )
        } else {
            (1, 1, 1, 1, WORD_BYTES, None)
        };

    let (execution_mode, wavelength) = match execution_kind {
        MaintainedExecutionKind::RequestedCoverage => (EXECUTION_REQUESTED_COVERAGE, None),
        MaintainedExecutionKind::Semantic(value) => match value {
            RenderOutputValue::Radiance { representation } => {
                (OUTPUT_RADIANCE, Some(representation.wavelength_meters()))
            }
            RenderOutputValue::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth,
            } => (OUTPUT_FORWARD_DEPTH, None),
            RenderOutputValue::ObjectIdentity => (OUTPUT_OBJECT_IDENTITY, None),
            RenderOutputValue::Distance { .. } => {
                return Err(RenderDeterministicLoweringError::UnsupportedOutput { output_index });
            }
        },
    };

    let (observation_kind, transform, tan_half_fov, aspect_ratio) = match observation {
        RenderObservationSpec::Perspective(observation) => (
            if observation.sampling_support().is_perspective_lattice_cell() {
                OBSERVATION_PERSPECTIVE_FOOTPRINT
            } else {
                OBSERVATION_PERSPECTIVE
            },
            observation.observation_to_scene(),
            Some((observation.vertical_field_of_view_radians() * 0.5).tan()),
            Some(observation.aspect_ratio()),
        ),
        RenderObservationSpec::Probe(observation) => (
            OBSERVATION_PROBE,
            observation.observation_to_scene(),
            None,
            None,
        ),
    };

    let mut geometry = Vec::new();
    geometry
        .try_reserve_exact(admitted_output.object_representations().len())
        .map_err(|_| RenderDeterministicLoweringError::HostAllocation {
            field: "maintained geometry records",
        })?;
    for object in admitted_output.object_representations() {
        let representation_id = object.representation().representation_id();
        let protocol = object.representation().requirement().protocol().protocol();
        match protocol {
            RenderRepresentationProtocol::SurfaceQuery
            | RenderRepresentationProtocol::OrientedSurfaceQuery => {
                admitted.surface_semantic_input(representation_id).ok_or(
                    RenderDeterministicLoweringError::MissingSurfaceInput {
                        output_index,
                        object_id: object.object_id(),
                        representation_id,
                    },
                )?;
            }
            RenderRepresentationProtocol::FieldDistance => {
                admitted.field_semantic_input(representation_id).ok_or(
                    RenderDeterministicLoweringError::MissingFieldInput {
                        output_index,
                        object_id: object.object_id(),
                        representation_id,
                    },
                )?;
            }
        }
        geometry.push((object.object_id(), representation_id, protocol));
    }
    geometry.sort_by_key(|(object_id, representation_id, _)| (*object_id, *representation_id));

    let emitters = if let Some(wavelength) = wavelength {
        matching_emitters(admitted, wavelength)?
    } else {
        Vec::new()
    };
    let emitter_offset = HEADER_WORDS
        .checked_add(geometry.len().checked_mul(GEOMETRY_WORDS).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "geometry input words",
            },
        )?)
        .ok_or(RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter input offset",
        })?;
    let emitter_words = emitters.len().checked_mul(EMITTER_WORDS).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter input words",
        },
    )?;
    let field_sample_offset = emitter_offset.checked_add(emitter_words).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "field sample input offset",
        },
    )?;
    let field_sample_words = geometry.iter().try_fold(
        0_usize,
        |count, (object_id, representation_id, protocol)| {
            if !matches!(protocol, RenderRepresentationProtocol::FieldDistance) {
                return Ok(count);
            }
            let input = admitted.field_semantic_input(*representation_id).ok_or(
                RenderDeterministicLoweringError::MissingFieldInput {
                    output_index,
                    object_id: *object_id,
                    representation_id: *representation_id,
                },
            )?;
            count.checked_add(input.sample_count()).ok_or(
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "field sample input words",
                },
            )
        },
    )?;
    let total_words = field_sample_offset.checked_add(field_sample_words).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "packed input words",
        },
    )?;
    let mut words = Vec::new();
    words.try_reserve_exact(total_words).map_err(|_| {
        RenderDeterministicLoweringError::HostAllocation {
            field: "packed maintained semantic input",
        }
    })?;
    words.resize(total_words, 0_u32);

    words[0] = sample_count;
    words[1] = width;
    words[2] = height;
    words[3] = row_stride_words;
    words[4] = u32::try_from(geometry.len()).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "geometry count",
        }
    })?;
    words[5] = u32::try_from(emitters.len()).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter count",
        }
    })?;
    words[6] = execution_mode;
    words[7] = observation_kind;
    pack_observation(&mut words, transform, tan_half_fov, aspect_ratio)?;
    let requested_extent = requested_extent.unwrap_or((1, 1));
    words[22] = requested_extent.0;
    words[23] = requested_extent.1;
    words[24] = temporal_history.map_or(0, |history| history.phase);
    words[25] = TEMPORAL_SEQUENCE_REVISION;
    words[26] = temporal_history.map_or(0, |history| history.age);
    words[27] = temporal_history.map_or(row_stride_words, |history| match &history.storage {
        DeterministicTemporalHistoryUseStorage::Static {
            row_stride_words, ..
        } => *row_stride_words,
        DeterministicTemporalHistoryUseStorage::Camera { .. } => row_stride_words,
    });
    words[28] = TEMPORAL_RECONSTRUCTION_REVISION;
    words[29] = u32::try_from(emitter_offset).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter input offset",
        }
    })?;

    let mut field_sample_cursor = field_sample_offset;
    for (index, (object_id, representation_id, protocol)) in geometry.into_iter().enumerate() {
        let base = HEADER_WORDS + index * GEOMETRY_WORDS;
        let state = admitted.plan().scene().object_state(object_id).ok_or(
            RenderDeterministicLoweringError::MissingObjectState {
                output_index,
                object_id,
            },
        )?;
        words[base + 1] = *object_codes
            .get(&object_id)
            .ok_or(RenderDeterministicLoweringError::OutputCorrelationChanged { output_index })?;
        words[base + 2] = if execution_mode == OUTPUT_RADIANCE {
            let material = admitted
                .plan()
                .scene()
                .object_participation(object_id)
                .and_then(|participation| participation.material_assignment())
                .ok_or(RenderDeterministicLoweringError::MissingMaterial {
                    output_index,
                    object_id,
                })?;
            f32_bits(material.material().reflectance(), "diffuse reflectance")?
        } else {
            0
        };

        match protocol {
            RenderRepresentationProtocol::SurfaceQuery
            | RenderRepresentationProtocol::OrientedSurfaceQuery => {
                let transform = RenderCompiledObjectTransform::compile(state.spatial()).map_err(
                    |RenderCompiledObjectTransformError::NonInvertibleObjectTransform| {
                        RenderDeterministicLoweringError::NonInvertibleObjectTransform {
                            output_index,
                            object_id,
                        }
                    },
                )?;
                pack_invertible_matrix3(
                    &mut words,
                    base + 4,
                    transform.scene_to_local_units_row_major(),
                    "object scene-to-local transform",
                )?;
                pack_vec3(&mut words, base + 13, transform.translation_scene())?;
                pack_matrix3(
                    &mut words,
                    base + 16,
                    transform.normal_local_to_scene_row_major(),
                )?;

                let input = admitted.surface_semantic_input(representation_id).ok_or(
                    RenderDeterministicLoweringError::MissingSurfaceInput {
                        output_index,
                        object_id,
                        representation_id,
                    },
                )?;
                match input.execution_view() {
                    RenderSurfaceSemanticInputView::Sphere {
                        center_local_units,
                        radius_local_units,
                    } => {
                        words[base] = SHAPE_SPHERE;
                        pack_vec3(&mut words, base + 25, center_local_units)?;
                        words[base + 28] =
                            positive_f32_bits(radius_local_units, "surface-input sphere radius")?;
                    }
                    RenderSurfaceSemanticInputView::Plane {
                        point_local_units,
                        normal_local,
                    } => {
                        words[base] = SHAPE_PLANE;
                        pack_vec3(&mut words, base + 25, point_local_units)?;
                        pack_vec3(&mut words, base + 28, normal_local)?;
                    }
                }
            }
            RenderRepresentationProtocol::FieldDistance => {
                let transform = RenderCompiledMetricSimilarityTransform::compile(state.spatial())
                    .map_err(
                    |RenderCompiledMetricSimilarityTransformError::NotPositiveSimilarity| {
                        RenderDeterministicLoweringError::NonSimilarityFieldTransform {
                            output_index,
                            object_id,
                        }
                    },
                )?;
                pack_invertible_matrix3(
                    &mut words,
                    base + 4,
                    transform.scene_to_local_meters_row_major(),
                    "field scene-to-local metric transform",
                )?;
                pack_vec3(&mut words, base + 13, transform.translation_scene())?;
                pack_matrix3(
                    &mut words,
                    base + 16,
                    transform.normal_local_to_scene_row_major(),
                )?;

                let input: &RenderFieldSemanticInput = admitted
                    .field_semantic_input(representation_id)
                    .ok_or(RenderDeterministicLoweringError::MissingFieldInput {
                        output_index,
                        object_id,
                        representation_id,
                    })?;
                words[base] = SHAPE_FIELD;
                words[base + 3] = conservative_positive_f32_bits(
                    transform.scene_meters_per_local_meter(),
                    "field scene metres per local metre",
                )?;
                pack_vec3(&mut words, base + 25, input.origin_local_meters())?;
                pack_vec3(&mut words, base + 28, input.sample_spacing_meters())?;
                let dimensions = input.dimensions();
                words[base + 31] = dimensions[0];
                words[base + 32] = dimensions[1];
                words[base + 33] = dimensions[2];
                words[base + 34] = u32::try_from(field_sample_cursor).map_err(|_| {
                    RenderDeterministicLoweringError::SizeOverflow {
                        field: "field sample input offset",
                    }
                })?;
                words[base + 35] = conservative_nonnegative_f32_bits(
                    input.max_absolute_query_error_local_meters(),
                    "field maximum absolute query error",
                )?;

                for sample_index in 0..input.sample_count() {
                    let sample = input.signed_distance_sample_meters(sample_index).ok_or(
                        RenderDeterministicLoweringError::OutputCorrelationChanged { output_index },
                    )?;
                    words[field_sample_cursor] = f32_bits(sample, "field signed-distance sample")?;
                    field_sample_cursor += 1;
                }
            }
        }
    }
    debug_assert_eq!(field_sample_cursor, total_words);

    for (index, emitter) in emitters.into_iter().enumerate() {
        let base = emitter_offset + index * EMITTER_WORDS;
        pack_vec3(&mut words, base, emitter.direction_to_source_scene())?;
        words[base + 3] = f32_bits(
            emitter.spectral_irradiance_w_m3(),
            "directional-emitter spectral irradiance",
        )?;
    }

    Ok(PackedOutput {
        input_words: words,
        sample_count,
        output_byte_len,
        texture_row_bytes,
    })
}

fn matching_emitters(
    admitted: &AdmittedRenderPlan,
    wavelength_meters: f64,
) -> Result<Vec<super::appearance::RenderDirectionalEmitter>, RenderDeterministicLoweringError> {
    let mut emitters = Vec::new();
    emitters
        .try_reserve_exact(admitted.plan().scene().len())
        .map_err(|_| RenderDeterministicLoweringError::HostAllocation {
            field: "directional-emitter realization",
        })?;
    for object_id in admitted.plan().scene().object_ids() {
        let Some(emitter) = admitted
            .plan()
            .scene()
            .object_participation(object_id)
            .and_then(|participation| participation.emitter())
        else {
            continue;
        };
        if emitter.wavelength_meters() == wavelength_meters {
            emitters.push(emitter);
        }
    }
    emitters.sort_by(|left, right| {
        left.direction_to_source_scene()
            .map(f64::to_bits)
            .cmp(&right.direction_to_source_scene().map(f64::to_bits))
            .then_with(|| {
                left.spectral_irradiance_w_m3()
                    .to_bits()
                    .cmp(&right.spectral_irradiance_w_m3().to_bits())
            })
    });

    let total = emitters.iter().try_fold(0.0_f64, |total, emitter| {
        let next = total + emitter.spectral_irradiance_w_m3();
        next.is_finite().then_some(next).ok_or(
            RenderDeterministicLoweringError::NumericRealization {
                field: "summed directional-emitter irradiance",
            },
        )
    })?;
    let _ = f32_bits(total, "summed directional-emitter irradiance")?;
    Ok(emitters)
}

fn camera_reprojection_parameter_words(
    current: RenderPerspectiveObservation,
    previous: Option<RenderPerspectiveObservation>,
    pose_changed: bool,
    same_pose_completed_frames: u32,
) -> Result<[u32; 35], RenderDeterministicLoweringError> {
    let mut words = [0_u32; 35];
    let matrix = current.observation_to_scene().row_major_3x4();
    let inverse = invert_matrix3(
        [
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ],
        "current observation linear transform",
    )?;
    pack_matrix3(&mut words, 24, inverse)?;
    words[0] = if previous.is_some() { 1 } else { 0 };
    words[1] = if pose_changed { 1 } else { 0 };
    words[2] = CAMERA_DEPTH_POLICY_REVISION;
    words[3] = CAMERA_REPROJECTION_REVISION;
    words[4] = CAMERA_DEPTH_ABSOLUTE_EPSILON.to_bits();
    words[5] = CAMERA_DEPTH_RELATIVE_EPSILON.to_bits();
    words[23] = same_pose_completed_frames.min(TEMPORAL_PHASE_COUNT);
    if let Some(previous) = previous {
        let matrix = previous.observation_to_scene().row_major_3x4();
        pack_vec3(&mut words, 6, [matrix[3], matrix[7], matrix[11]])?;
        let inverse = invert_matrix3(
            [
                matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8],
                matrix[9], matrix[10],
            ],
            "previous observation linear transform",
        )?;
        pack_matrix3(&mut words, 9, inverse)?;
        let forward = normalize_private_vec3(
            [-matrix[2], -matrix[6], -matrix[10]],
            "previous observation forward",
        )?;
        pack_vec3(&mut words, 18, forward)?;
        words[21] = positive_f32_bits(
            (previous.vertical_field_of_view_radians() * 0.5).tan(),
            "previous perspective tangent half field of view",
        )?;
        words[22] =
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

fn pack_observation(
    words: &mut [u32],
    transform: super::space_time::RenderAffineTransform3,
    tan_half_fov: Option<f64>,
    aspect_ratio: Option<f64>,
) -> Result<(), RenderDeterministicLoweringError> {
    let matrix = transform.row_major_3x4();
    pack_vec3(words, 8, [matrix[3], matrix[7], matrix[11]])?;
    pack_invertible_matrix3(
        words,
        11,
        [
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ],
        "observation linear transform",
    )?;
    words[20] = match tan_half_fov {
        Some(value) => positive_f32_bits(value, "perspective tangent half field of view")?,
        None => 0,
    };
    words[21] = match aspect_ratio {
        Some(value) => positive_f32_bits(value, "perspective aspect ratio")?,
        None => 1.0_f32.to_bits(),
    };
    Ok(())
}

fn pack_vec3(
    words: &mut [u32],
    base: usize,
    values: [f64; 3],
) -> Result<(), RenderDeterministicLoweringError> {
    words[base] = f32_bits(values[0], "packed vector component")?;
    words[base + 1] = f32_bits(values[1], "packed vector component")?;
    words[base + 2] = f32_bits(values[2], "packed vector component")?;
    Ok(())
}

fn pack_matrix3(
    words: &mut [u32],
    base: usize,
    values: [f64; 9],
) -> Result<(), RenderDeterministicLoweringError> {
    for (offset, value) in values.into_iter().enumerate() {
        words[base + offset] = f32_bits(value, "packed transform component")?;
    }
    Ok(())
}

fn pack_invertible_matrix3(
    words: &mut [u32],
    base: usize,
    values: [f64; 9],
    field: &'static str,
) -> Result<(), RenderDeterministicLoweringError> {
    let mut physical = [0.0_f32; 9];
    for (slot, value) in physical.iter_mut().zip(values) {
        let narrowed = value as f32;
        if !value.is_finite() || !narrowed.is_finite() {
            return Err(RenderDeterministicLoweringError::NumericRealization { field });
        }
        *slot = narrowed;
    }
    let determinant = physical[0] * (physical[4] * physical[8] - physical[5] * physical[7])
        - physical[1] * (physical[3] * physical[8] - physical[5] * physical[6])
        + physical[2] * (physical[3] * physical[7] - physical[4] * physical[6]);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    for (offset, value) in physical.into_iter().enumerate() {
        words[base + offset] = value.to_bits();
    }
    Ok(())
}

fn f32_bits(value: f64, field: &'static str) -> Result<u32, RenderDeterministicLoweringError> {
    let physical = value as f32;
    if !value.is_finite() || !physical.is_finite() {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    Ok(physical.to_bits())
}

fn positive_f32_bits(
    value: f64,
    field: &'static str,
) -> Result<u32, RenderDeterministicLoweringError> {
    let physical = value as f32;
    if !value.is_finite() || !physical.is_finite() || physical <= 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    Ok(physical.to_bits())
}

fn conservative_positive_f32_bits(
    value: f64,
    field: &'static str,
) -> Result<u32, RenderDeterministicLoweringError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    conservative_nonnegative_f32_bits(value, field)
}

fn conservative_nonnegative_f32_bits(
    value: f64,
    field: &'static str,
) -> Result<u32, RenderDeterministicLoweringError> {
    if !value.is_finite() || value < 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    let mut physical = value as f32;
    if !physical.is_finite() || physical < 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    if f64::from(physical) < value {
        physical = f32::from_bits(
            physical
                .to_bits()
                .checked_add(1)
                .ok_or(RenderDeterministicLoweringError::NumericRealization { field })?,
        );
        if !physical.is_finite() {
            return Err(RenderDeterministicLoweringError::NumericRealization { field });
        }
    }
    Ok(physical.to_bits())
}

fn align_up(value: u64, alignment: u64) -> Result<u64, RenderDeterministicLoweringError> {
    if alignment == 0 {
        return Err(RenderDeterministicLoweringError::InvalidBytesPerRowAlignment { alignment });
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value.checked_add(alignment - remainder).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "aligned lattice row bytes",
            },
        )
    }
}

fn deterministic_dispatch_size(
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

fn map_maintained_program_build_error(
    error: RenderMaintainedProgramBuildError,
) -> RenderDeterministicLoweringError {
    match error {
        RenderMaintainedProgramBuildError::RunenShader(error) => {
            RenderDeterministicLoweringError::RunenShaderCompilation(error)
        }
        RenderMaintainedProgramBuildError::RunenGpu { stage, source } => {
            gpu_program_source(stage, source)
        }
    }
}

fn gpu_program_source(
    stage: &'static str,
    source: GpuProgramSourceError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::ProgramSource { stage, source },
    )
}

fn gpu_resource_descriptor(
    stage: &'static str,
    source: GpuResourceDescriptorError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::ResourceDescriptor { stage, source },
    )
}

fn gpu_resource_allocation(
    stage: &'static str,
    source: GpuWorkResourceIdAllocationError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::ResourceAllocation { stage, source },
    )
}

fn gpu_transfer_preparation(
    stage: &'static str,
    source: GpuOrdinaryTransferPreparationError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::TransferPreparation { stage, source },
    )
}

fn gpu_program_contract(
    stage: &'static str,
    source: GpuProgramContractError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::ProgramContract { stage, source },
    )
}

fn gpu_work_operation(
    stage: &'static str,
    source: GpuWorkOperationError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::WorkOperation { stage, source },
    )
}

fn gpu_readback_request(
    stage: &'static str,
    source: GpuReadbackRequestError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::ReadbackRequest { stage, source },
    )
}

fn gpu_work_authoring(
    stage: &'static str,
    source: GpuWorkAuthoringError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::WorkAuthoring { stage, source },
    )
}

#[cfg(test)]
mod tests {
    use super::super::space_time::RenderAffineTransform3;
    use super::*;
    use runen_gpu::GpuBufferRange;

    #[test]
    fn maintained_programs_compile_through_runenshader_and_preserve_exact_gpu_source_bytes() {
        let programs = build_maintained_program_sources(
            MAINTAINED_EVALUATOR_REVISION,
            MAINTAINED_WGSL.as_str(),
            u64::from(TEMPORAL_RECONSTRUCTION_REVISION),
            TEMPORAL_RECONSTRUCTION_WGSL,
            u64::from(CAMERA_REPROJECTION_REVISION),
            CAMERA_REPROJECTION_WGSL.as_str(),
        )
        .expect(
            "all maintained programs must compile through RunenShader and admit through RunenGPU",
        );

        assert_eq!(
            programs.evaluator_artifact().canonical_wgsl().as_bytes(),
            MAINTAINED_WGSL.as_bytes()
        );
        assert_eq!(
            programs.evaluator().canonical_wgsl().as_bytes(),
            MAINTAINED_WGSL.as_bytes()
        );
        assert_eq!(
            programs
                .temporal_reconstruction_artifact()
                .canonical_wgsl()
                .as_bytes(),
            TEMPORAL_RECONSTRUCTION_WGSL.as_bytes()
        );
        assert_eq!(
            programs
                .temporal_reconstruction()
                .canonical_wgsl()
                .as_bytes(),
            TEMPORAL_RECONSTRUCTION_WGSL.as_bytes()
        );
        assert_eq!(
            programs
                .camera_reprojection_artifact()
                .canonical_wgsl()
                .as_bytes(),
            CAMERA_REPROJECTION_WGSL.as_bytes()
        );
        assert_eq!(
            programs.camera_reprojection().canonical_wgsl().as_bytes(),
            CAMERA_REPROJECTION_WGSL.as_bytes()
        );
    }

    fn assert_dispatch(sample_count: u32, maximum: u32, expected: [u32; 3]) {
        assert_eq!(
            deterministic_dispatch_size(sample_count, maximum)
                .expect("dispatch should fit the admitted 2D capacity")
                .as_array(),
            expected
        );
    }

    fn cache_descriptor(byte_len: u64) -> GpuBufferDescriptor {
        GpuBufferDescriptor::ordinary_owned(
            "deterministic cache test buffer",
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            byte_len,
            [GpuBufferUsage::Storage],
            GpuBufferInitialization::Uninitialized,
        )
        .expect("deterministic cache test descriptor should be valid")
    }

    #[test]
    fn conservative_nonnegative_f32_packing_never_rounds_a_bound_down() {
        let source = 0.7_f64;
        let ordinary = source as f32;
        assert!(
            f64::from(ordinary) < source,
            "proof value must exercise a downward ordinary f32 rounding"
        );

        let conservative = f32::from_bits(
            conservative_nonnegative_f32_bits(source, "conservative bound")
                .expect("finite non-negative bound must pack"),
        );
        assert!(f64::from(conservative) >= source);
        assert!(conservative > ordinary);

        assert_eq!(
            f32::from_bits(
                conservative_nonnegative_f32_bits(0.5, "exact conservative bound")
                    .expect("exact f32 bound must pack"),
            ),
            0.5
        );
    }

    #[test]
    fn conservative_field_error_packing_never_narrows_scene_space_bound() {
        let local_error = 0.5_f64;
        let scene_scale = 0.7_f64;
        let ordinary_scale = scene_scale as f32;
        assert!(
            f64::from(ordinary_scale) < scene_scale,
            "proof scale must exercise downward ordinary f32 rounding"
        );

        let packed_error = f32::from_bits(
            conservative_nonnegative_f32_bits(local_error, "field local query error")
                .expect("finite local error must pack"),
        );
        let packed_scale = f32::from_bits(
            conservative_positive_f32_bits(scene_scale, "field scene distance scale")
                .expect("finite positive scale must pack"),
        );
        let admitted_scene_error = local_error * scene_scale;
        let physical_scene_error = f64::from(packed_error) * f64::from(packed_scale);
        assert!(
            physical_scene_error >= admitted_scene_error,
            "physical conservative error margin must not narrow the admitted scene-space bound"
        );

        assert_eq!(
            f32::from_bits(
                conservative_positive_f32_bits(0.5, "exact field scene distance scale")
                    .expect("exact positive f32 scale must pack"),
            ),
            0.5
        );
    }

    #[test]
    fn deterministic_cache_reuses_matching_descriptors_and_replaces_resizes() {
        let mut cache = DeterministicResourceCache::default();
        let first = cache
            .buffer(0, 0, DeterministicBufferKind::Input, cache_descriptor(16))
            .expect("first deterministic buffer should allocate");
        let same = cache
            .buffer(0, 0, DeterministicBufferKind::Input, cache_descriptor(16))
            .expect("matching deterministic buffer should reuse");
        assert_eq!(first.diagnostic_identity(), same.diagnostic_identity());

        let resized = cache
            .buffer(0, 0, DeterministicBufferKind::Input, cache_descriptor(32))
            .expect("changed descriptor should allocate a replacement");
        assert_ne!(first.diagnostic_identity(), resized.diagnostic_identity());
    }

    #[test]
    fn deterministic_cache_scopes_equal_descriptors_by_producer() {
        let mut cache = DeterministicResourceCache::default();
        let first = cache
            .buffer(11, 0, DeterministicBufferKind::Input, cache_descriptor(16))
            .expect("first producer buffer should allocate");
        let same_producer = cache
            .buffer(11, 0, DeterministicBufferKind::Input, cache_descriptor(16))
            .expect("same producer should reuse its buffer");
        let other_producer = cache
            .buffer(12, 0, DeterministicBufferKind::Input, cache_descriptor(16))
            .expect("other producer should allocate an independent buffer");

        assert_eq!(
            first.diagnostic_identity(),
            same_producer.diagnostic_identity()
        );
        assert_ne!(
            first.diagnostic_identity(),
            other_producer.diagnostic_identity()
        );
    }

    #[test]
    fn deterministic_cache_stays_bounded_across_frames_and_resize() {
        let mut cache = DeterministicResourceCache::default();
        let mut identities = BTreeSet::new();
        for frame in 0..120 {
            let byte_len = if frame < 60 { 16 } else { 32 };
            let handle = cache
                .buffer(
                    11,
                    0,
                    DeterministicBufferKind::Input,
                    cache_descriptor(byte_len),
                )
                .expect("sustained deterministic frame should prepare");
            identities.insert(handle.diagnostic_identity());
        }

        assert_eq!(
            identities.len(),
            2,
            "one replacement is expected for the resize"
        );
        assert_eq!(cache.buffers.len(), 1, "the cache retains one live slot");
    }

    fn bounded_cycle_mean(samples: &[Option<f32>]) -> Option<f32> {
        if samples.len() < TEMPORAL_PHASE_COUNT as usize {
            return None;
        }
        let mut sum = 0.0_f32;
        for sample in samples.iter().take(TEMPORAL_PHASE_COUNT as usize) {
            sum += (*sample)?;
        }
        Some(sum / TEMPORAL_PHASE_COUNT as f32)
    }

    fn requested_cell_sample_counts(
        requested_extent: (u32, u32),
        evaluation_extent: (u32, u32),
    ) -> Vec<u32> {
        let mut counts = vec![0_u32; (requested_extent.0 * requested_extent.1) as usize];
        for phase in 0..TEMPORAL_PHASE_COUNT {
            let phase_x = if phase == 1 || phase == 3 {
                0.75_f32
            } else {
                0.25_f32
            };
            let phase_y = if phase >= 2 { 0.75_f32 } else { 0.25_f32 };
            for evaluation_y in 0..evaluation_extent.1 {
                for evaluation_x in 0..evaluation_extent.0 {
                    let requested_x = (((evaluation_x as f32 + phase_x) * requested_extent.0 as f32
                        / evaluation_extent.0 as f32)
                        .floor() as u32)
                        .min(requested_extent.0 - 1);
                    let requested_y = (((evaluation_y as f32 + phase_y) * requested_extent.1 as f32
                        / evaluation_extent.1 as f32)
                        .floor() as u32)
                        .min(requested_extent.1 - 1);
                    counts[(requested_y * requested_extent.0 + requested_x) as usize] += 1;
                }
            }
        }
        counts
    }

    #[test]
    fn bounded_cycle_forms_exact_first_cycle_mean_and_defined_miss_is_zero() {
        assert_eq!(
            bounded_cycle_mean(&[Some(1.0), Some(3.0), Some(5.0), Some(7.0)]),
            Some(4.0)
        );
        assert_eq!(
            bounded_cycle_mean(&[Some(8.0), Some(0.0), Some(4.0), Some(0.0)]),
            Some(3.0),
            "defined background phases contribute radiance zero to the finite estimate"
        );
        assert_eq!(
            bounded_cycle_mean(&[Some(8.0), None, Some(4.0), Some(0.0)]),
            None,
            "undefined evaluation must remain distinct from a defined background miss"
        );
        assert_eq!(
            bounded_cycle_mean(&[
                Some(1.0),
                Some(3.0),
                Some(5.0),
                Some(7.0),
                Some(100.0),
                Some(-100.0),
                Some(50.0),
                Some(-50.0),
            ]),
            Some(4.0),
            "later cycles must not mutate the already formed bounded estimate"
        );
    }

    #[test]
    fn finite_phase_mapping_has_truthful_p75_p67_and_p50_per_cell_divisors() {
        let p100 = requested_cell_sample_counts((4, 4), (4, 4));
        assert!(p100.iter().all(|count| *count == 4));

        let p75 = requested_cell_sample_counts((4, 4), (3, 3));
        assert!(p75.contains(&1));
        assert!(p75.contains(&2));
        assert!(p75.contains(&4));

        let p67 = requested_cell_sample_counts((6, 6), (4, 4));
        assert!(p67.contains(&1));
        assert!(p67.contains(&2));
        assert!(p67.contains(&4));

        let p50 = requested_cell_sample_counts((4, 4), (2, 2));
        assert!(
            p50.iter().all(|count| *count == 1),
            "P50 cells settle after the global sequence even though each receives one sample"
        );
    }

    fn temporal_test_observation(
        transform: super::super::space_time::RenderAffineTransform3,
    ) -> RenderPerspectiveObservation {
        use super::super::request::{RenderPerspectiveObservation, RenderSamplingSupport};
        use super::super::space_time::{RenderTimeInterval, RenderTimePoint};

        let shutter = RenderTimeInterval::instant(
            RenderTimePoint::from_seconds(0.0).expect("finite test time"),
        );
        RenderPerspectiveObservation::new(
            transform,
            std::f64::consts::FRAC_PI_3,
            1.0,
            shutter,
            RenderSamplingSupport::perspective_lattice_cell(),
        )
        .expect("valid temporal test observation")
    }

    fn temporal_signature(source_generation: u64) -> DeterministicTemporalSignature {
        use super::super::request::{
            RenderOutputSpec, RenderOutputValue, RenderRadiometricRepresentation,
            RenderResultTopology, RenderSemanticTolerance,
        };
        use super::super::space_time::{RenderAffineTransform3, RenderTemporalSupport};
        use super::super::surface_input::{
            RenderSurfaceSemanticInput, RenderSurfaceSemanticInputBinding,
            RenderSurfaceSemanticInputGeneration,
        };

        let observation = RenderObservationSpec::Perspective(temporal_test_observation(
            RenderAffineTransform3::identity(),
        ));
        let output = RenderOutputSpec::new(
            RenderOutputValue::Radiance {
                representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                    550.0e-9,
                )
                .expect("valid wavelength"),
            },
            RenderResultTopology::sample_lattice_2d(4, 4).expect("valid requested lattice"),
            RenderSemanticTolerance::absolute(0.001).expect("valid tolerance"),
        )
        .expect("valid temporal test output");
        let input = RenderSurfaceSemanticInput::sphere(
            [0.0, 0.0, -3.0],
            1.0,
            RenderTemporalSupport::unbounded(),
        )
        .expect("valid temporal test input");
        let binding = RenderSurfaceSemanticInputBinding::new(
            RenderRepresentationId::from_raw(1).expect("non-zero representation id"),
            input,
        )
        .with_generation(RenderSurfaceSemanticInputGeneration::new(source_generation));

        DeterministicTemporalSignature {
            scene_revision: RenderSceneRevision::INITIAL,
            observation: temporal_observation_compatibility(observation, false),
            output,
            semantic_inputs: vec![binding],
            field_semantic_inputs: Vec::new(),
            evaluation_extent: (2, 2),
            sequence_revision: TEMPORAL_SEQUENCE_REVISION,
            reconstruction_revision: TEMPORAL_RECONSTRUCTION_REVISION,
            camera_reprojection_revision: None,
            depth_policy_revision: None,
        }
    }

    fn temporal_signature_with_field_generation(
        surface_generation: u64,
        field_generation: u64,
    ) -> DeterministicTemporalSignature {
        use super::super::field_input::{
            RenderFieldSemanticInput, RenderFieldSemanticInputBinding,
            RenderFieldSemanticInputGeneration,
        };
        use super::super::space_time::RenderTemporalSupport;

        let mut signature = temporal_signature(surface_generation);
        let input = RenderFieldSemanticInput::dense(
            [-1.0; 3],
            [1.0; 3],
            [2, 2, 2],
            vec![0.0; 8],
            0.0,
            RenderTemporalSupport::unbounded(),
        )
        .expect("valid temporal field input");
        signature.field_semantic_inputs = vec![
            RenderFieldSemanticInputBinding::new(
                RenderRepresentationId::from_raw(2).expect("non-zero field representation id"),
                input,
            )
            .with_generation(RenderFieldSemanticInputGeneration::new(field_generation)),
        ];
        signature
    }

    fn camera_temporal_signature(
        source_generation: u64,
        observation: RenderPerspectiveObservation,
        evaluation_extent: (u32, u32),
    ) -> DeterministicTemporalSignature {
        let mut signature = temporal_signature(source_generation);
        signature.observation = temporal_observation_compatibility(
            RenderObservationSpec::Perspective(observation),
            true,
        );
        signature.evaluation_extent = evaluation_extent;
        signature.camera_reprojection_revision = Some(CAMERA_REPROJECTION_REVISION);
        signature.depth_policy_revision = Some(CAMERA_DEPTH_POLICY_REVISION);
        signature
    }

    fn temporal_test_observation_with(
        vertical_fov: f64,
        aspect_ratio: f64,
        shutter: super::super::space_time::RenderTimeInterval,
        sampling_support: super::super::request::RenderSamplingSupport,
    ) -> RenderPerspectiveObservation {
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            vertical_fov,
            aspect_ratio,
            shutter,
            sampling_support,
        )
        .expect("valid varied temporal test observation")
    }

    fn assert_camera_signature_recreates(
        changed_signature: DeterministicTemporalSignature,
        changed_observation: RenderPerspectiveObservation,
    ) {
        let baseline_observation = temporal_test_observation(RenderAffineTransform3::identity());
        let baseline_signature = camera_temporal_signature(7, baseline_observation, (4, 4));
        let mut cache = DeterministicResourceCache::default();
        let baseline = cache
            .temporal_history(
                41,
                0,
                baseline_signature,
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: baseline_observation,
                    camera_capable: true,
                },
            )
            .expect("baseline camera history");
        let changed = cache
            .temporal_history(
                41,
                0,
                changed_signature,
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: changed_observation,
                    camera_capable: true,
                },
            )
            .expect("changed camera history");
        assert!(changed.reset);
        assert_ne!(changed.generation, baseline.generation);
    }

    #[test]
    fn temporal_history_reuses_compatible_generation_and_resets_on_source_generation_change() {
        let mut cache = DeterministicResourceCache::default();
        let first = cache
            .temporal_history(
                11,
                0,
                temporal_signature(7),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: temporal_test_observation(
                        RenderAffineTransform3::identity(),
                    ),
                    camera_capable: false,
                },
            )
            .expect("initial temporal history should allocate");
        assert!(first.reset);
        assert_eq!(first.phase, 0);
        assert_eq!(first.age, 0);

        let retry_before_completion = cache
            .temporal_history(
                11,
                0,
                temporal_signature(7),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: temporal_test_observation(
                        RenderAffineTransform3::identity(),
                    ),
                    camera_capable: false,
                },
            )
            .expect("uncompleted temporal history retry should remain bootstrap");
        assert!(retry_before_completion.reset);
        assert_eq!(retry_before_completion.generation, first.generation);
        assert_eq!(retry_before_completion.phase, 0);
        assert_eq!(retry_before_completion.age, 0);

        let state = cache
            .temporal_histories
            .get_mut(&(11, 0))
            .expect("initial temporal history should be retained");
        state.phase = 1;
        state.age = 1;

        let reused = cache
            .temporal_history(
                11,
                0,
                temporal_signature(7),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: temporal_test_observation(
                        RenderAffineTransform3::identity(),
                    ),
                    camera_capable: false,
                },
            )
            .expect("compatible temporal history should reuse");
        assert!(!reused.reset);
        assert_eq!(reused.generation, first.generation);
        assert_eq!(reused.phase, 1);
        assert_eq!(reused.age, 1);

        let reset = cache
            .temporal_history(
                11,
                0,
                temporal_signature(8),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: temporal_test_observation(
                        RenderAffineTransform3::identity(),
                    ),
                    camera_capable: false,
                },
            )
            .expect("changed source generation should recreate history");
        assert!(reset.reset);
        assert_ne!(reset.generation, first.generation);
        assert_eq!(reset.phase, 0);
        assert_eq!(reset.age, 0);
    }

    #[test]
    fn temporal_history_resets_when_field_source_generation_changes() {
        let observation = temporal_test_observation(RenderAffineTransform3::identity());
        let mut cache = DeterministicResourceCache::default();

        let first = cache
            .temporal_history(
                17,
                0,
                temporal_signature_with_field_generation(7, 11),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: observation,
                    camera_capable: false,
                },
            )
            .expect("initial mixed-input history should allocate");
        assert!(first.reset);

        let state = cache
            .temporal_histories
            .get_mut(&(17, 0))
            .expect("mixed-input temporal history should be retained");
        state.phase = 1;
        state.age = 1;

        let reused = cache
            .temporal_history(
                17,
                0,
                temporal_signature_with_field_generation(7, 11),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: observation,
                    camera_capable: false,
                },
            )
            .expect("unchanged field generation should reuse history");
        assert!(!reused.reset);
        assert_eq!(reused.generation, first.generation);

        let reset = cache
            .temporal_history(
                17,
                0,
                temporal_signature_with_field_generation(7, 12),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: observation,
                    camera_capable: false,
                },
            )
            .expect("changed field generation should recreate history");
        assert!(reset.reset);
        assert_ne!(reset.generation, first.generation);
        assert_eq!(reset.phase, 0);
        assert_eq!(reset.age, 0);
    }

    #[test]
    fn p100_camera_compatibility_excludes_pose_but_retains_projection_semantics() {
        use super::super::space_time::RenderAffineTransform3;

        let first = temporal_test_observation(RenderAffineTransform3::identity());
        let moved = temporal_test_observation(
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .expect("valid moved observation"),
        );
        assert_eq!(
            temporal_observation_compatibility(RenderObservationSpec::Perspective(first), true),
            temporal_observation_compatibility(RenderObservationSpec::Perspective(moved), true),
        );
        assert_ne!(
            temporal_observation_compatibility(RenderObservationSpec::Perspective(first), false),
            temporal_observation_compatibility(RenderObservationSpec::Perspective(moved), false),
        );
    }

    #[test]
    fn camera_compatibility_key_retains_every_non_pose_dependency() {
        use super::super::request::{
            RenderOutputSpec, RenderOutputValue, RenderRadiometricRepresentation,
            RenderResultTopology, RenderSamplingSupport, RenderSemanticTolerance,
        };
        use super::super::scene::{RenderSceneStore, RenderSceneUpdate};
        use super::super::space_time::{RenderTimeInterval, RenderTimePoint};

        let baseline_observation = temporal_test_observation(RenderAffineTransform3::identity());
        let baseline = camera_temporal_signature(7, baseline_observation, (4, 4));

        let changed_fov = temporal_test_observation_with(
            std::f64::consts::FRAC_PI_4,
            1.0,
            baseline_observation.shutter(),
            RenderSamplingSupport::perspective_lattice_cell(),
        );
        let changed_aspect = temporal_test_observation_with(
            std::f64::consts::FRAC_PI_3,
            1.25,
            baseline_observation.shutter(),
            RenderSamplingSupport::perspective_lattice_cell(),
        );
        let changed_shutter = temporal_test_observation_with(
            std::f64::consts::FRAC_PI_3,
            1.0,
            RenderTimeInterval::instant(
                RenderTimePoint::from_seconds(1.0).expect("finite changed shutter"),
            ),
            RenderSamplingSupport::perspective_lattice_cell(),
        );
        let changed_support = temporal_test_observation_with(
            std::f64::consts::FRAC_PI_3,
            1.0,
            baseline_observation.shutter(),
            RenderSamplingSupport::ideal_ray(),
        );

        for changed_observation in [
            changed_fov,
            changed_aspect,
            changed_shutter,
            changed_support,
        ] {
            assert_ne!(
                camera_temporal_signature(7, changed_observation, (4, 4)),
                baseline,
                "projection/shutter/support changes must remain in the camera compatibility key"
            );
        }

        assert_ne!(
            camera_temporal_signature(8, baseline_observation, (4, 4)),
            baseline,
            "source generation must remain in the camera compatibility key"
        );
        assert_ne!(
            camera_temporal_signature(7, baseline_observation, (3, 4)),
            baseline,
            "finite evaluation extent must remain in the camera compatibility key"
        );

        let mut changed_topology = baseline.clone();
        changed_topology.output = RenderOutputSpec::new(
            RenderOutputValue::Radiance {
                representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                    550.0e-9,
                )
                .expect("valid wavelength"),
            },
            RenderResultTopology::sample_lattice_2d(8, 4).expect("changed topology"),
            RenderSemanticTolerance::absolute(0.001).expect("valid tolerance"),
        )
        .expect("valid changed output");
        assert_ne!(changed_topology, baseline);

        let mut store = RenderSceneStore::new();
        let object = store.allocate_object_id().expect("test object id");
        let mut update = RenderSceneUpdate::new();
        update.insert(object);
        store.commit(update).expect("advance test scene revision");
        let mut changed_scene = baseline.clone();
        changed_scene.scene_revision = store.snapshot().revision();
        assert_ne!(changed_scene, baseline);
    }

    #[test]
    fn camera_non_pose_signature_change_recreates_history_generation() {
        let observation = temporal_test_observation(RenderAffineTransform3::identity());
        assert_camera_signature_recreates(
            camera_temporal_signature(8, observation, (4, 4)),
            observation,
        );

        let changed_fov = temporal_test_observation_with(
            std::f64::consts::FRAC_PI_4,
            1.0,
            observation.shutter(),
            super::super::request::RenderSamplingSupport::perspective_lattice_cell(),
        );
        assert_camera_signature_recreates(
            camera_temporal_signature(7, changed_fov, (4, 4)),
            changed_fov,
        );
    }

    #[test]
    fn sub_native_camera_pose_change_recreates_temporal_history() {
        let mut cache = DeterministicResourceCache::default();
        let first_observation = temporal_test_observation(RenderAffineTransform3::identity());
        let moved_observation = temporal_test_observation(
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .expect("valid moved observation"),
        );

        let mut first_signature = temporal_signature(7);
        first_signature.observation = temporal_observation_compatibility(
            RenderObservationSpec::Perspective(first_observation),
            false,
        );
        let first = cache
            .temporal_history(
                12,
                0,
                first_signature,
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: first_observation,
                    camera_capable: false,
                },
            )
            .expect("sub-native history should allocate");

        let mut moved_signature = temporal_signature(7);
        moved_signature.observation = temporal_observation_compatibility(
            RenderObservationSpec::Perspective(moved_observation),
            false,
        );
        let moved = cache
            .temporal_history(
                12,
                0,
                moved_signature,
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: moved_observation,
                    camera_capable: false,
                },
            )
            .expect("sub-native moved history should recreate");
        assert!(moved.reset);
        assert_ne!(moved.generation, first.generation);
        assert!(matches!(
            moved.storage,
            DeterministicTemporalHistoryUseStorage::Static { .. }
        ));
    }

    #[test]
    fn camera_history_completion_retry_failure_and_ping_pong_are_fail_closed() {
        use super::super::space_time::RenderAffineTransform3;

        let mut cache = DeterministicResourceCache::default();
        let observation = temporal_test_observation(RenderAffineTransform3::identity());
        let mut signature = temporal_signature(7);
        signature.observation = temporal_observation_compatibility(
            RenderObservationSpec::Perspective(observation),
            true,
        );
        signature.evaluation_extent = (4, 4);
        signature.camera_reprojection_revision = Some(CAMERA_REPROJECTION_REVISION);
        signature.depth_policy_revision = Some(CAMERA_DEPTH_POLICY_REVISION);

        let first = cache
            .temporal_history(
                11,
                0,
                signature.clone(),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: observation,
                    camera_capable: true,
                },
            )
            .expect("camera history should allocate");
        assert!(first.reset);
        let (first_previous_identity, first_current_identity, first_previous_observation) =
            match &first.storage {
                DeterministicTemporalHistoryUseStorage::Camera {
                    previous_history,
                    current_history,
                    previous_observation,
                    ..
                } => (
                    previous_history.diagnostic_identity(),
                    current_history.diagnostic_identity(),
                    *previous_observation,
                ),
                _ => panic!("P100 history must use camera storage"),
            };
        assert_ne!(
            first_previous_identity, first_current_identity,
            "camera reprojection must never read and write one retained slot in place"
        );
        assert_eq!(first_previous_observation, None);

        let retry = cache
            .temporal_history(
                11,
                0,
                signature.clone(),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: observation,
                    camera_capable: true,
                },
            )
            .expect("pre-acceptance retry should preserve bootstrap state");
        assert!(retry.reset);
        assert_eq!(retry.generation, first.generation);
        assert_eq!(retry.age, 0);
        assert!(matches!(
            retry.storage,
            DeterministicTemporalHistoryUseStorage::Camera {
                previous_observation: None,
                ..
            }
        ));

        cache.reconcile_temporal_outputs(11, true);
        let retained = cache
            .temporal_histories
            .get(&(11, 0))
            .expect("completed camera history retained");
        assert_eq!(retained.age, 1);
        assert_eq!(retained.phase, 1);
        let DeterministicTemporalStorage::Camera(camera) = &retained.storage else {
            panic!("P100 history must remain camera storage");
        };
        assert_eq!(camera.completed_slot, 1);
        assert_eq!(camera.completed_observation, Some(observation));
        assert_eq!(camera.pending_slot, None);
        assert_eq!(camera.pending_observation, None);

        let moved = temporal_test_observation(
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .expect("valid moved observation"),
        );
        let reused = cache
            .temporal_history(
                11,
                0,
                signature,
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: moved,
                    camera_capable: true,
                },
            )
            .expect("pose-only motion should reuse camera history generation");
        assert_eq!(reused.generation, first.generation);
        assert!(!reused.reset);
        assert!(matches!(
            reused.storage,
            DeterministicTemporalHistoryUseStorage::Camera {
                previous_observation: Some(previous),
                pose_changed: true,
                ..
            } if previous == observation
        ));

        cache.reconcile_temporal_outputs(11, false);
        assert!(
            !cache.temporal_histories.contains_key(&(11, 0)),
            "failed accepted execution must discard affected camera history"
        );
    }

    #[test]
    fn camera_same_pose_convergence_advances_only_on_completed_frames_and_resets_after_motion() {
        let mut cache = DeterministicResourceCache::default();
        let observation = temporal_test_observation(RenderAffineTransform3::identity());
        let signature = camera_temporal_signature(7, observation, (4, 4));

        let bootstrap = cache
            .temporal_history(
                23,
                0,
                signature.clone(),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: observation,
                    camera_capable: true,
                },
            )
            .expect("bootstrap camera history");
        assert!(matches!(
            bootstrap.storage,
            DeterministicTemporalHistoryUseStorage::Camera {
                same_pose_completed_frames: 0,
                ..
            }
        ));

        let retry = cache
            .temporal_history(
                23,
                0,
                signature.clone(),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: observation,
                    camera_capable: true,
                },
            )
            .expect("uncompleted bootstrap retry");
        assert!(matches!(
            retry.storage,
            DeterministicTemporalHistoryUseStorage::Camera {
                same_pose_completed_frames: 0,
                ..
            }
        ));

        cache.reconcile_temporal_outputs(23, true);
        for expected_completed in 1..=TEMPORAL_PHASE_COUNT {
            let use_state = cache
                .temporal_history(
                    23,
                    0,
                    signature.clone(),
                    (4, 4),
                    4,
                    DeterministicTemporalHistorySelection {
                        current_observation: observation,
                        camera_capable: true,
                    },
                )
                .expect("same-pose camera history");
            assert!(matches!(
                use_state.storage,
                DeterministicTemporalHistoryUseStorage::Camera {
                    pose_changed: false,
                    same_pose_completed_frames,
                    ..
                } if same_pose_completed_frames == expected_completed
            ));
            cache.reconcile_temporal_outputs(23, true);
        }

        let moved = temporal_test_observation(
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.25, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .expect("valid moved observation"),
        );
        let moving = cache
            .temporal_history(
                23,
                0,
                signature,
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: moved,
                    camera_capable: true,
                },
            )
            .expect("pose-only motion should reuse camera history");
        assert!(matches!(
            moving.storage,
            DeterministicTemporalHistoryUseStorage::Camera {
                pose_changed: true,
                same_pose_completed_frames: 4,
                ..
            }
        ));

        cache.reconcile_temporal_outputs(23, true);
        let after_motion = cache
            .temporal_histories
            .get(&(23, 0))
            .expect("completed moving history retained");
        let DeterministicTemporalStorage::Camera(camera) = &after_motion.storage else {
            panic!("P100 history must remain camera storage");
        };
        assert_eq!(camera.same_pose_completed_frames, 0);
        assert_eq!(camera.completed_observation, Some(moved));

        let stopped = cache
            .temporal_history(
                23,
                0,
                camera_temporal_signature(7, moved, (4, 4)),
                (4, 4),
                4,
                DeterministicTemporalHistorySelection {
                    current_observation: moved,
                    camera_capable: true,
                },
            )
            .expect("same pose after motion should begin a fresh bounded estimate");
        assert!(matches!(
            stopped.storage,
            DeterministicTemporalHistoryUseStorage::Camera {
                pose_changed: false,
                same_pose_completed_frames: 0,
                ..
            }
        ));
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum CameraReferenceDecision {
        Accept,
        CurrentBackground,
        BehindPreviousCamera,
        OutOfBounds,
        MissingPreviousHistory,
        DepthInconsistent,
        GeometryInconsistent,
    }

    struct CameraReferenceSample {
        current_hit: bool,
        previous_hit: bool,
        previous_local: [f32; 3],
        tan_half_fov: f32,
        aspect: f32,
        projected_depth: f32,
        previous_depth: f32,
        visible_point: [f32; 3],
        retained_anchor: [f32; 3],
    }

    fn matching_world_point(current: [f32; 3], previous: [f32; 3]) -> bool {
        if current
            .iter()
            .chain(previous.iter())
            .any(|value| !value.is_finite())
        {
            return false;
        }
        current == previous
    }

    fn camera_reference_decision(sample: CameraReferenceSample) -> CameraReferenceDecision {
        let CameraReferenceSample {
            current_hit,
            previous_hit,
            previous_local,
            tan_half_fov,
            aspect,
            projected_depth,
            previous_depth,
            visible_point,
            retained_anchor,
        } = sample;
        if !current_hit {
            return CameraReferenceDecision::CurrentBackground;
        }
        if previous_local.iter().any(|value| !value.is_finite()) || previous_local[2] >= 0.0 {
            return CameraReferenceDecision::BehindPreviousCamera;
        }
        let projected_x = previous_local[0] / (-previous_local[2] * tan_half_fov * aspect);
        let projected_y = previous_local[1] / (-previous_local[2] * tan_half_fov);
        let u = projected_x * 0.5 + 0.5;
        let v = 0.5 - projected_y * 0.5;
        if !u.is_finite() || !v.is_finite() || !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v)
        {
            return CameraReferenceDecision::OutOfBounds;
        }
        if !previous_hit {
            return CameraReferenceDecision::MissingPreviousHistory;
        }
        let tolerance = 0.001_f32 + 0.001_f32 * projected_depth.abs().max(previous_depth.abs());
        if !projected_depth.is_finite()
            || !previous_depth.is_finite()
            || (projected_depth - previous_depth).abs() > tolerance
        {
            return CameraReferenceDecision::DepthInconsistent;
        }
        if !matching_world_point(visible_point, retained_anchor) {
            return CameraReferenceDecision::GeometryInconsistent;
        }
        CameraReferenceDecision::Accept
    }

    #[test]
    fn camera_reprojection_reference_rejects_background_bounds_missing_and_depth_mismatch() {
        assert_eq!(
            camera_reference_decision(CameraReferenceSample {
                current_hit: false,
                previous_hit: true,
                previous_local: [0.0, 0.0, -2.0],
                tan_half_fov: 1.0,
                aspect: 1.0,
                projected_depth: 2.0,
                previous_depth: 2.0,
                visible_point: [0.0; 3],
                retained_anchor: [0.0; 3],
            }),
            CameraReferenceDecision::CurrentBackground
        );
        assert_eq!(
            camera_reference_decision(CameraReferenceSample {
                current_hit: true,
                previous_hit: true,
                previous_local: [0.0, 0.0, 0.1],
                tan_half_fov: 1.0,
                aspect: 1.0,
                projected_depth: 2.0,
                previous_depth: 2.0,
                visible_point: [0.0; 3],
                retained_anchor: [0.0; 3],
            }),
            CameraReferenceDecision::BehindPreviousCamera
        );
        assert_eq!(
            camera_reference_decision(CameraReferenceSample {
                current_hit: true,
                previous_hit: true,
                previous_local: [3.0, 0.0, -1.0],
                tan_half_fov: 1.0,
                aspect: 1.0,
                projected_depth: 2.0,
                previous_depth: 2.0,
                visible_point: [0.0; 3],
                retained_anchor: [0.0; 3],
            }),
            CameraReferenceDecision::OutOfBounds
        );
        assert_eq!(
            camera_reference_decision(CameraReferenceSample {
                current_hit: true,
                previous_hit: false,
                previous_local: [0.0, 0.0, -2.0],
                tan_half_fov: 1.0,
                aspect: 1.0,
                projected_depth: 2.0,
                previous_depth: 2.0,
                visible_point: [0.0; 3],
                retained_anchor: [0.0; 3],
            }),
            CameraReferenceDecision::MissingPreviousHistory
        );
        assert_eq!(
            camera_reference_decision(CameraReferenceSample {
                current_hit: true,
                previous_hit: true,
                previous_local: [0.0, 0.0, -2.0],
                tan_half_fov: 1.0,
                aspect: 1.0,
                projected_depth: 2.02,
                previous_depth: 2.0,
                visible_point: [0.0; 3],
                retained_anchor: [0.0; 3],
            }),
            CameraReferenceDecision::DepthInconsistent
        );
        assert_eq!(
            camera_reference_decision(CameraReferenceSample {
                current_hit: true,
                previous_hit: true,
                previous_local: [0.0, 0.0, -2.0],
                tan_half_fov: 1.0,
                aspect: 1.0,
                projected_depth: 2.001,
                previous_depth: 2.0,
                visible_point: [1.0, 2.0, 3.0],
                retained_anchor: [1.0, 2.0, 3.0],
            }),
            CameraReferenceDecision::Accept
        );
        assert_eq!(
            camera_reference_decision(CameraReferenceSample {
                current_hit: true,
                previous_hit: true,
                previous_local: [0.0, 0.0, -2.0],
                tan_half_fov: 1.0,
                aspect: 1.0,
                projected_depth: 2.001,
                previous_depth: 2.0,
                visible_point: [1.0, 2.0, 3.0],
                retained_anchor: [1.002, 2.0, 3.0],
            }),
            CameraReferenceDecision::GeometryInconsistent,
            "depth tolerance alone cannot validate a retained anchor against a different visible point"
        );
    }

    #[test]
    fn camera_reprojection_shader_matches_the_reference_rejection_contract() {
        for source_law in [
            "if !hit {",
            "local.z >= 0.0",
            "u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0",
            "previous_history_words[base + 3u] == 0u",
            "abs(result.projected_depth - previous_depth) > result.tolerance",
            "motion_candidate(sample_index, current_point.xyz, current_hit, false)",
            "matching_world_point(anchor, result.visible_point)",
            "bitcast<u32>(validated.value) != bitcast<u32>(radiance)",
            "previous_history_words[previous + 4u]",
            "return index * 8u",
        ] {
            assert!(
                CAMERA_REPROJECTION_WGSL.contains(source_law),
                "camera reprojection shader is missing reference law: {source_law}"
            );
        }
    }

    #[test]
    fn camera_reprojection_wgsl_forms_a_canonical_compute_pipeline() {
        let programs = build_maintained_program_sources(
            MAINTAINED_EVALUATOR_REVISION,
            MAINTAINED_WGSL.as_str(),
            u64::from(TEMPORAL_RECONSTRUCTION_REVISION),
            TEMPORAL_RECONSTRUCTION_WGSL,
            u64::from(CAMERA_REPROJECTION_REVISION),
            CAMERA_REPROJECTION_WGSL.as_str(),
        )
        .expect("maintained programs must compile through RunenShader and admit through RunenGPU");
        GpuComputePipelineDescriptor::ordinary(programs.camera_reprojection().clone(), "main")
            .expect("camera reprojection must form a canonical compute pipeline");
    }

    #[test]
    fn camera_reprojection_shader_carries_versioned_depth_policy() {
        assert!(CAMERA_REPROJECTION_WGSL.contains("CAMERA_DEPTH_ABSOLUTE_EPSILON: f32 = 0.001"));
        assert!(CAMERA_REPROJECTION_WGSL.contains("CAMERA_DEPTH_RELATIVE_EPSILON: f32 = 0.001"));
        assert_eq!(CAMERA_DEPTH_POLICY_REVISION, 1);
        assert_eq!(CAMERA_REPROJECTION_REVISION, 3);
    }

    #[test]
    fn temporal_reconstruction_shaders_keep_undefined_samples_fail_closed() {
        assert!(TEMPORAL_RECONSTRUCTION_WGSL.contains("INVALID_HISTORY_SAMPLE_COUNT"));
        assert!(
            TEMPORAL_RECONSTRUCTION_WGSL
                .contains("history_sample_counts[history_index] = INVALID_HISTORY_SAMPLE_COUNT")
        );
        assert!(
            TEMPORAL_RECONSTRUCTION_WGSL
                .contains("retained_sample_count == INVALID_HISTORY_SAMPLE_COUNT")
        );

        assert!(CAMERA_REPROJECTION_WGSL.contains("INVALID_HISTORY_SAMPLE_COUNT"));
        assert!(
            CAMERA_REPROJECTION_WGSL.contains(
                "write_current(sample_index, 0.0, 0.0, INVALID_HISTORY_SAMPLE_COUNT, 0u)"
            )
        );
        assert!(CAMERA_REPROJECTION_WGSL.contains("count == INVALID_HISTORY_SAMPLE_COUNT"));
    }

    #[test]
    fn temporal_four_phase_extent_requires_half_to_native_coverage() {
        let requested = (1920, 1080);
        for supported in [(1920, 1080), (1440, 810), (1280, 720), (960, 540)] {
            assert!(temporal_evaluation_extent_supported(requested, supported));
        }
        assert!(!temporal_evaluation_extent_supported(requested, (959, 540)));
        assert!(!temporal_evaluation_extent_supported(requested, (960, 539)));
        assert!(!temporal_evaluation_extent_supported(
            requested,
            (1921, 1080)
        ));
        assert!(!temporal_evaluation_extent_supported(
            requested,
            (1920, 1081)
        ));
    }

    #[test]
    fn maintained_source_is_retained_across_fresh_renderer_resource_caches() {
        let first = DeterministicResourceCache::default()
            .maintained_source()
            .expect("maintained source should admit");
        for _ in 0..120 {
            let next = DeterministicResourceCache::default()
                .maintained_source()
                .expect("maintained source should remain available");
            assert!(first.is_same_record(&next));
        }
    }

    #[test]
    fn maintained_wgsl_forms_a_canonical_compute_pipeline() {
        let programs = build_maintained_program_sources(
            MAINTAINED_EVALUATOR_REVISION,
            MAINTAINED_WGSL.as_str(),
            u64::from(TEMPORAL_RECONSTRUCTION_REVISION),
            TEMPORAL_RECONSTRUCTION_WGSL,
            u64::from(CAMERA_REPROJECTION_REVISION),
            CAMERA_REPROJECTION_WGSL.as_str(),
        )
        .expect("maintained programs must compile through RunenShader and admit through RunenGPU");
        let source = programs.evaluator().clone();
        assert_eq!(
            source.identity().revision().get(),
            MAINTAINED_EVALUATOR_REVISION
        );
        let pipeline = GpuComputePipelineDescriptor::ordinary(source, "main")
            .expect("maintained deterministic WGSL must form a canonical compute pipeline");
        assert_eq!(pipeline.entry_point().as_str(), "main");
    }

    #[test]
    fn physical_identity_decoder_reserves_zero_and_unknown_codes() {
        let decoder = RenderObjectIdentityDecoder {
            objects_by_code: Vec::new(),
        };
        assert_eq!(decoder.decode(0), None);
        assert_eq!(decoder.decode(1), None);
    }

    #[test]
    fn ordinary_observation_intent_never_requests_private_readback() {
        assert!(!DeterministicObservationIntent::Ordinary.requires_private_readback());
        assert!(DeterministicObservationIntent::Verify.requires_private_readback());
    }

    #[test]
    fn row_alignment_is_checked_without_embedding_device_policy() {
        assert_eq!(align_up(12, 4), Ok(12));
        assert_eq!(align_up(12, 8), Ok(16));
        assert_eq!(
            align_up(12, 0),
            Err(RenderDeterministicLoweringError::InvalidBytesPerRowAlignment { alignment: 0 })
        );
    }

    #[test]
    fn deterministic_dispatch_tiles_samples_within_the_admitted_dimension_limit() {
        assert_dispatch(512, 8, [8, 1, 1]);
        assert_dispatch(513, 8, [8, 2, 1]);
        assert_dispatch(4096, 8, [8, 8, 1]);
        assert_dispatch(65_535 * 64, 65_535, [65_535, 1, 1]);
        assert_dispatch(65_535 * 64 + 1, 65_535, [65_535, 2, 1]);
    }

    #[test]
    fn deterministic_dispatch_rejects_work_beyond_two_dimensional_capacity() {
        let error = deterministic_dispatch_size(4097, 8).expect_err("dispatch must reject");
        assert_eq!(
            error,
            RenderDeterministicLoweringError::DispatchCapacityExceeded {
                sample_count: 4097,
                workgroup_size: 64,
                required_workgroups: 65,
                max_workgroups_per_dimension: 8,
                capacity_workgroups: 64,
            }
        );
    }
    #[test]
    fn every_runengpu_preparation_owner_remains_a_typed_source() {
        fn assert_owner<Owner>(
            error: RenderRunenGpuPreparationError,
            expected: &Owner,
            expected_stage: &'static str,
        ) where
            Owner: Error + PartialEq + fmt::Debug + 'static,
        {
            assert_eq!(error.stage(), expected_stage);
            let source = Error::source(&error).expect("typed RunenGPU owner source");
            assert_eq!(source.downcast_ref::<Owner>(), Some(expected));
        }

        let mut resources = GpuWorkResourceIdAllocator::new();
        let buffer = resources
            .allocate_buffer_handle(
                GpuBufferDescriptor::ordinary_owned(
                    "typed error owner proof",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    16,
                    [GpuBufferUsage::Storage],
                    GpuBufferInitialization::Uninitialized,
                )
                .expect("proof buffer descriptor"),
            )
            .expect("proof buffer allocation");

        let access = GpuBufferRange::new(&buffer, 0, 0)
            .expect_err("zero range must fail in RunenGPU access authority");
        let program_source = runen_gpu::GpuProgramSourceKey::new("")
            .expect_err("empty source key must fail in RunenGPU source authority");
        assert_owner(
            RenderRunenGpuPreparationError::ProgramSource {
                stage: "program-source",
                source: program_source.clone(),
            },
            &program_source,
            "program-source",
        );

        let resource_descriptor = GpuBufferDescriptor::ordinary_owned(
            "",
            GpuResourceLifetime::Transient,
            GpuReconstruction::SourceBacked,
            16,
            [GpuBufferUsage::Storage],
            GpuBufferInitialization::Uninitialized,
        )
        .expect_err("empty label must fail in RunenGPU descriptor authority");
        assert_owner(
            RenderRunenGpuPreparationError::ResourceDescriptor {
                stage: "resource-descriptor",
                source: resource_descriptor.clone(),
            },
            &resource_descriptor,
            "resource-descriptor",
        );

        let resource_allocation = GpuWorkResourceIdAllocationError::Exhausted;
        assert_owner(
            RenderRunenGpuPreparationError::ResourceAllocation {
                stage: "resource-allocation",
                source: resource_allocation,
            },
            &resource_allocation,
            "resource-allocation",
        );

        let transfer_preparation =
            PreparedGpuData::<TransferData>::ordinary_pod_transfer("", &[1_u32])
                .expect_err("empty label must fail in RunenGPU transfer preparation");
        assert_owner(
            RenderRunenGpuPreparationError::TransferPreparation {
                stage: "transfer-preparation",
                source: transfer_preparation.clone(),
            },
            &transfer_preparation,
            "transfer-preparation",
        );

        let programs = build_maintained_program_sources(
            MAINTAINED_EVALUATOR_REVISION,
            MAINTAINED_WGSL.as_str(),
            u64::from(TEMPORAL_RECONSTRUCTION_REVISION),
            TEMPORAL_RECONSTRUCTION_WGSL,
            u64::from(CAMERA_REPROJECTION_REVISION),
            CAMERA_REPROJECTION_WGSL.as_str(),
        )
        .expect("maintained programs must compile through RunenShader and admit through RunenGPU");
        let program_contract =
            GpuComputePipelineDescriptor::ordinary(programs.evaluator().clone(), "")
                .expect_err("empty entry point must fail in RunenGPU program authority");
        assert_owner(
            RenderRunenGpuPreparationError::ProgramContract {
                stage: "program-contract",
                source: program_contract.clone(),
            },
            &program_contract,
            "program-contract",
        );

        let region = GpuBufferRegion::whole(&buffer).expect("whole proof buffer region");
        let work_operation = GpuClearOperation::buffer_zero(region.clone())
            .expect_err("buffer without copy-destination usage must reject clear");
        assert_owner(
            RenderRunenGpuPreparationError::WorkOperation {
                stage: "work-operation",
                source: work_operation.clone(),
            },
            &work_operation,
            "work-operation",
        );

        let readback_request = GpuReadbackOperation::ordinary(region.into())
            .expect_err("buffer without copy-source usage must reject readback");
        assert_owner(
            RenderRunenGpuPreparationError::ReadbackRequest {
                stage: "readback-request",
                source: readback_request.clone(),
            },
            &readback_request,
            "readback-request",
        );

        let work_authoring = GpuWorkAuthoringError::from(access);
        assert_owner(
            RenderRunenGpuPreparationError::WorkAuthoring {
                stage: "work-authoring",
                source: work_authoring.clone(),
            },
            &work_authoring,
            "work-authoring",
        );
    }

    #[test]
    fn runengpu_preparation_preserves_typed_source_chain() {
        let source = runen_gpu::GpuProgramSourceKey::new("")
            .expect_err("empty RunenGPU source key must be rejected");
        let error = gpu_program_source("typed source proof", source.clone());

        let preparation =
            Error::source(&error).expect("lowering must expose RunenGPU preparation source");
        let owner = preparation
            .source()
            .expect("RunenGPU preparation must expose the exact owner error");
        assert_eq!(owner.downcast_ref::<GpuProgramSourceError>(), Some(&source));
    }

    #[test]
    fn renderer_local_lowering_does_not_claim_a_runengpu_source() {
        let error = RenderDeterministicLoweringError::DispatchCapacityExceeded {
            sample_count: 65,
            workgroup_size: 64,
            required_workgroups: 2,
            max_workgroups_per_dimension: 1,
            capacity_workgroups: 1,
        };
        assert!(Error::source(&error).is_none());
    }

    #[test]
    fn maintained_execution_has_no_string_flattening_gpu_authoring_bucket() {
        let source = include_str!("deterministic_execution.rs");
        assert!(!source.contains(concat!("gpu_", "authoring(")));
        assert!(!source.contains(concat!("RunenGpu", "Authoring")));
    }
}
