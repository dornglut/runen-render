use super::super::program::abi::{camera, temporal};
use super::WORD_BYTES;
use super::errors::{
    RenderDeterministicLoweringError, gpu_resource_allocation, gpu_resource_descriptor,
};
use crate::field_input::RenderFieldSemanticInputBinding;
use crate::request::{
    RenderObservationSpec, RenderOutputSpec, RenderPerspectiveObservation, RenderSamplingSupport,
};
use crate::scene::RenderSceneRevision;
use crate::space_time::RenderTimeInterval;
use crate::surface_input::RenderSurfaceSemanticInputBinding;
use runen_gpu::{
    GpuBufferDescriptor, GpuBufferHandle, GpuBufferInitialization, GpuBufferUsage,
    GpuReconstruction, GpuResourceLifetime, GpuWorkResourceIdAllocator,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    num::NonZeroU64,
    sync::atomic::{AtomicU64, Ordering},
};

static NEXT_GRAPH_WIRING_NAMESPACE: AtomicU64 = AtomicU64::new(1);

fn allocate_graph_wiring_namespace() -> Result<NonZeroU64, RenderDeterministicLoweringError> {
    let raw = NEXT_GRAPH_WIRING_NAMESPACE
        .try_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            (current != 0).then_some(if current == u64::MAX { 0 } else { current + 1 })
        })
        .map_err(|_| RenderDeterministicLoweringError::GraphWiringIdentityExhausted)?;
    Ok(NonZeroU64::new(raw).expect("graph-wiring allocator never returns zero"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum DeterministicBufferKind {
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
    TemporalProvisional,
    TemporalPhasePresence,
    TemporalAvailability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum DeterministicTemporalObservationCompatibility {
    Exact(RenderObservationSpec),
    CameraPerspective {
        vertical_field_of_view_bits: u64,
        aspect_ratio_bits: u64,
        shutter: RenderTimeInterval,
        sampling_support: RenderSamplingSupport,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct DeterministicTemporalSignature {
    pub(super) scene_revision: RenderSceneRevision,
    pub(super) observation: DeterministicTemporalObservationCompatibility,
    pub(super) output: RenderOutputSpec,
    pub(super) semantic_inputs: Vec<RenderSurfaceSemanticInputBinding>,
    pub(super) field_semantic_inputs: Vec<RenderFieldSemanticInputBinding>,
    pub(super) evaluation_extent: (u32, u32),
    pub(super) sequence_revision: u32,
    pub(super) reconstruction_revision: u32,
    pub(super) camera_reprojection_revision: Option<u32>,
    pub(super) depth_policy_revision: Option<u32>,
}

#[derive(Debug)]
pub(super) struct DeterministicCameraTemporalStorage {
    pub(super) slots: [GpuBufferHandle; 2],
    pub(super) completed_slot: usize,
    pub(super) completed_observation: Option<RenderPerspectiveObservation>,
    pub(super) same_pose_completed_frames: u32,
    pub(super) pending_slot: Option<usize>,
    pub(super) pending_observation: Option<RenderPerspectiveObservation>,
}

#[derive(Debug)]
pub(super) enum DeterministicTemporalStorage {
    Static {
        handle: GpuBufferHandle,
        sample_counts: GpuBufferHandle,
        row_stride_words: u32,
    },
    Camera(Box<DeterministicCameraTemporalStorage>),
}

#[derive(Debug)]
pub(super) struct DeterministicTemporalHistory {
    pub(super) signature: DeterministicTemporalSignature,
    pub(super) storage: DeterministicTemporalStorage,
    pub(super) generation: u64,
    pub(super) phase: u32,
    pub(super) age: u32,
}

#[derive(Debug, Clone)]
pub(super) enum DeterministicTemporalHistoryUseStorage {
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
pub(super) struct DeterministicTemporalHistoryUse {
    pub(super) storage: DeterministicTemporalHistoryUseStorage,
    pub(super) generation: u64,
    pub(super) reset: bool,
    pub(super) phase: u32,
    pub(super) age: u32,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DeterministicRenderExecutionSelection {
    pub(super) finite_evaluation: Option<(usize, (u32, u32))>,
    pub(super) produce_requested_coverage: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DeterministicOutputExecutionSelection {
    pub(super) graph_wiring_namespace: NonZeroU64,
    pub(super) finite_evaluation_extent: Option<(u32, u32)>,
    pub(super) produce_requested_coverage: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DeterministicTemporalHistorySelection {
    pub(super) current_observation: RenderPerspectiveObservation,
    pub(super) camera_capable: bool,
}

/// Renderer-owned logical buffer identities reused by one retained execution continuity.
///
/// RunenGPU's bind-group realization retains the resource dependencies of each realized binding.
/// Rebuilding these identities for every interactive frame would therefore grow the authoritative
/// realization registry without bound. One retained ordinary session owns one cache. Descriptor
/// changes, such as resize, deliberately allocate replacement identities.
#[derive(Debug, Default)]
pub(crate) struct DeterministicResourceCache {
    graph_wiring_namespace: Option<NonZeroU64>,
    pub(super) identities: GpuWorkResourceIdAllocator,
    pub(super) buffers: BTreeMap<(usize, DeterministicBufferKind), GpuBufferHandle>,
    pub(super) temporal_histories: BTreeMap<usize, DeterministicTemporalHistory>,
    pub(super) next_temporal_generation: u64,
    pub(super) prepared_temporal_outputs: BTreeSet<usize>,
}

impl DeterministicResourceCache {
    #[cfg(test)]
    pub(crate) fn has_temporal_history(&self, output_index: usize) -> bool {
        self.temporal_histories.contains_key(&output_index)
    }

    pub(crate) fn graph_wiring_namespace(
        &mut self,
    ) -> Result<NonZeroU64, RenderDeterministicLoweringError> {
        if let Some(namespace) = self.graph_wiring_namespace {
            return Ok(namespace);
        }
        let namespace = allocate_graph_wiring_namespace()?;
        self.graph_wiring_namespace = Some(namespace);
        Ok(namespace)
    }

    pub(crate) fn discard_prepared_temporal_outputs(&mut self) {
        let outputs = std::mem::take(&mut self.prepared_temporal_outputs);
        for output_index in outputs {
            if let Some(history) = self.temporal_histories.get_mut(&output_index)
                && let DeterministicTemporalStorage::Camera(camera) = &mut history.storage
            {
                camera.pending_slot = None;
                camera.pending_observation = None;
            }
        }
    }

    pub(crate) fn reset_for_context_change(&mut self) {
        let graph_wiring_namespace = self.graph_wiring_namespace;
        let next_temporal_generation = self.next_temporal_generation;
        *self = Self::default();
        self.graph_wiring_namespace = graph_wiring_namespace;
        self.next_temporal_generation = next_temporal_generation;
    }

    /// Discard renderer-derived GPU resources and history when prepared work could have
    /// escaped without exact accepted-submission correlation. The caller may still own or
    /// submit old fragments, so graph export wiring also requires a fresh namespace.
    /// Keep temporal generations monotonic so subsequent evidence shows the reset.
    pub(crate) fn invalidate_unverified_occurrence(&mut self) {
        let next_temporal_generation = self.next_temporal_generation;
        *self = Self::default();
        self.next_temporal_generation = next_temporal_generation;
    }

    pub(crate) fn reconcile_temporal_outputs(&mut self, completed: bool) {
        let outputs = std::mem::take(&mut self.prepared_temporal_outputs);
        if completed {
            for output_index in outputs {
                if let Some(history) = self.temporal_histories.get_mut(&output_index) {
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
                                    .min(temporal::PHASE_COUNT),
                            };
                            camera.completed_observation = Some(observation);
                        }
                    }
                    history.phase = (history.phase + 1) % temporal::PHASE_COUNT;
                    history.age = history.age.saturating_add(1);
                }
            }
        } else {
            for output_index in outputs {
                self.temporal_histories.remove(&output_index);
            }
        }
    }
    pub(super) fn temporal_history(
        &mut self,
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
        let row_bytes = super::layout::align_up(logical_row_bytes, bytes_per_row_alignment)?;
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
            .checked_mul(camera::history::WORDS_PER_SAMPLE)
            .and_then(|words| words.checked_add(camera::history::DIAGNOSTIC_WORDS))
            .and_then(|words| words.checked_mul(WORD_BYTES))
            .ok_or(RenderDeterministicLoweringError::SizeOverflow {
                field: "camera history byte length",
            })?;

        let key = output_index;
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
                    [GpuBufferUsage::Storage, GpuBufferUsage::CopySource],
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
        self.prepared_temporal_outputs.insert(output_index);
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

    pub(super) fn buffer(
        &mut self,
        output_index: usize,
        kind: DeterministicBufferKind,
        descriptor: GpuBufferDescriptor,
    ) -> Result<GpuBufferHandle, RenderDeterministicLoweringError> {
        let key = (output_index, kind);
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

pub(super) fn temporal_evaluation_extent_supported(
    requested_extent: (u32, u32),
    evaluation_extent: (u32, u32),
) -> bool {
    evaluation_extent.0 <= requested_extent.0
        && evaluation_extent.1 <= requested_extent.1
        && u64::from(evaluation_extent.0) * 2 >= u64::from(requested_extent.0)
        && u64::from(evaluation_extent.1) * 2 >= u64::from(requested_extent.1)
}

/// The maintained WGSL uses f32 (not rational) coordinates to scatter each
/// evaluation sample to a requested cell. An injective mapping is a prerequisite
/// for race-free scatter and for bounding current-only fallback by N - M.
///
/// Validate the *same* expression/rounding stages as the shader on both axes
/// for both phase offsets. Any alias fails admission, rather than relying on an
/// invalid mathematical real-number assumption for large near-native extents.
/// Bound the host-side scan independently of the potentially huge lattice area.
pub(super) fn temporal_phase_mapping_is_injective(
    requested_extent: (u32, u32),
    evaluation_extent: (u32, u32),
) -> bool {
    fn axis(requested: u32, evaluation: u32) -> bool {
        if requested == 0 || evaluation == 0 || requested > 65_536 {
            return false;
        }
        for offset in [0.25_f32, 0.75_f32] {
            let mut prior = None;
            for index in 0..evaluation {
                let mapped = (((index as f32 + offset) * requested as f32) / evaluation as f32)
                    .floor() as u32;
                let mapped = mapped.min(requested - 1);
                if prior.is_some_and(|prior| mapped <= prior) {
                    return false;
                }
                prior = Some(mapped);
            }
        }
        true
    }
    axis(requested_extent.0, evaluation_extent.0) && axis(requested_extent.1, evaluation_extent.1)
}

pub(super) fn temporal_observation_compatibility(
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
