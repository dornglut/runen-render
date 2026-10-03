use super::*;

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
    pub(super) scope: u64,
    pub(super) finite_evaluation: Option<(usize, (u32, u32))>,
    pub(super) produce_requested_coverage: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DeterministicOutputExecutionSelection {
    pub(super) scope: u64,
    pub(super) finite_evaluation_extent: Option<(u32, u32)>,
    pub(super) produce_requested_coverage: bool,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DeterministicTemporalHistorySelection {
    pub(super) current_observation: RenderPerspectiveObservation,
    pub(super) camera_capable: bool,
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
    pub(super) identities: GpuWorkResourceIdAllocator,
    pub(super) buffers: BTreeMap<(u64, usize, DeterministicBufferKind), GpuBufferHandle>,
    pub(super) temporal_histories: BTreeMap<(u64, usize), DeterministicTemporalHistory>,
    pub(super) next_temporal_generation: u64,
    pub(super) prepared_temporal_outputs: BTreeMap<u64, BTreeSet<usize>>,
    // Keep the latest accepted graph correlated with every producer namespace whose mutable
    // intermediates it used. A peer surface's submission must not stall this producer's cache.
    pub(super) producer_submissions: BTreeMap<u64, GpuSubmission>,
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

    pub(super) fn reconcile_temporal_outputs(&mut self, scope: u64, completed: bool) {
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
            self.temporal_histories
                .retain(|(history_scope, _), _| *history_scope != scope);
        }
    }

    pub(super) fn temporal_history(
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

    pub(super) fn buffer(
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

pub(super) fn temporal_evaluation_extent_supported(
    requested_extent: (u32, u32),
    evaluation_extent: (u32, u32),
) -> bool {
    evaluation_extent.0 <= requested_extent.0
        && evaluation_extent.1 <= requested_extent.1
        && u64::from(evaluation_extent.0) * 2 >= u64::from(requested_extent.0)
        && u64::from(evaluation_extent.1) * 2 >= u64::from(requested_extent.1)
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

pub(super) fn any_producer_scope_in_flight(
    producer_scopes: impl IntoIterator<Item = u64>,
    mut producer_is_in_flight: impl FnMut(u64) -> bool,
) -> bool {
    producer_scopes.into_iter().any(&mut producer_is_in_flight)
}
