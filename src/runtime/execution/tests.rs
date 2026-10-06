use super::*;
use crate::space_time::RenderAffineTransform3;
use runen_gpu::GpuBufferRange;

#[test]
fn maintained_programs_compile_through_runenshader_and_preserve_exact_gpu_source_bytes() {
    let programs = build_maintained_program_sources().expect(
        "all maintained programs must compile through RunenShader and admit through RunenGPU",
    );

    assert_eq!(
        programs.evaluator_artifact().canonical_wgsl().as_bytes(),
        EVALUATOR_WGSL.as_bytes()
    );
    assert_eq!(
        programs.evaluator().canonical_wgsl().as_bytes(),
        EVALUATOR_WGSL.as_bytes()
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
        super::passes::deterministic_dispatch_size(sample_count, maximum)
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
    if samples.len() < temporal::PHASE_COUNT as usize {
        return None;
    }
    let mut sum = 0.0_f32;
    for sample in samples.iter().take(temporal::PHASE_COUNT as usize) {
        sum += (*sample)?;
    }
    Some(sum / temporal::PHASE_COUNT as f32)
}

fn requested_cell_sample_counts(
    requested_extent: (u32, u32),
    evaluation_extent: (u32, u32),
) -> Vec<u32> {
    let mut counts = vec![0_u32; (requested_extent.0 * requested_extent.1) as usize];
    for phase in 0..temporal::PHASE_COUNT {
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
    transform: crate::space_time::RenderAffineTransform3,
) -> RenderPerspectiveObservation {
    use crate::request::{RenderPerspectiveObservation, RenderSamplingSupport};
    use crate::space_time::{RenderTimeInterval, RenderTimePoint};

    let shutter =
        RenderTimeInterval::instant(RenderTimePoint::from_seconds(0.0).expect("finite test time"));
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
    use crate::request::{
        RenderOutputSpec, RenderOutputValue, RenderRadiometricRepresentation, RenderResultTopology,
        RenderSemanticTolerance,
    };
    use crate::space_time::{RenderAffineTransform3, RenderTemporalSupport};
    use crate::surface_input::{
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
        sequence_revision: temporal::SEQUENCE_REVISION,
        reconstruction_revision: TEMPORAL_RECONSTRUCTION_REVISION,
        camera_reprojection_revision: None,
        depth_policy_revision: None,
    }
}

fn temporal_signature_with_field_generation(
    surface_generation: u64,
    field_generation: u64,
) -> DeterministicTemporalSignature {
    use crate::field_input::{
        RenderFieldSemanticInput, RenderFieldSemanticInputBinding,
        RenderFieldSemanticInputGeneration,
    };
    use crate::space_time::RenderTemporalSupport;

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
    signature.observation =
        temporal_observation_compatibility(RenderObservationSpec::Perspective(observation), true);
    signature.evaluation_extent = evaluation_extent;
    signature.camera_reprojection_revision = Some(CAMERA_REPROJECTION_REVISION);
    signature.depth_policy_revision = Some(camera::DEPTH_POLICY_REVISION);
    signature
}

fn temporal_test_observation_with(
    vertical_fov: f64,
    aspect_ratio: f64,
    shutter: crate::space_time::RenderTimeInterval,
    sampling_support: crate::request::RenderSamplingSupport,
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
                current_observation: temporal_test_observation(RenderAffineTransform3::identity()),
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
                current_observation: temporal_test_observation(RenderAffineTransform3::identity()),
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
                current_observation: temporal_test_observation(RenderAffineTransform3::identity()),
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
                current_observation: temporal_test_observation(RenderAffineTransform3::identity()),
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
    use crate::space_time::RenderAffineTransform3;

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
    use crate::request::{
        RenderOutputSpec, RenderOutputValue, RenderRadiometricRepresentation, RenderResultTopology,
        RenderSamplingSupport, RenderSemanticTolerance,
    };
    use crate::scene::{RenderSceneStore, RenderSceneUpdate};
    use crate::space_time::{RenderTimeInterval, RenderTimePoint};

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
        crate::request::RenderSamplingSupport::perspective_lattice_cell(),
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
    use crate::space_time::RenderAffineTransform3;

    let mut cache = DeterministicResourceCache::default();
    let observation = temporal_test_observation(RenderAffineTransform3::identity());
    let mut signature = temporal_signature(7);
    signature.observation =
        temporal_observation_compatibility(RenderObservationSpec::Perspective(observation), true);
    signature.evaluation_extent = (4, 4);
    signature.camera_reprojection_revision = Some(CAMERA_REPROJECTION_REVISION);
    signature.depth_policy_revision = Some(camera::DEPTH_POLICY_REVISION);

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
    for expected_completed in 1..=temporal::PHASE_COUNT {
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
    if !u.is_finite() || !v.is_finite() || !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
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
    let programs = build_maintained_program_sources()
        .expect("maintained programs must compile through RunenShader and admit through RunenGPU");
    GpuComputePipelineDescriptor::ordinary(programs.camera_reprojection().clone(), "main")
        .expect("camera reprojection must form a canonical compute pipeline");
}

#[test]
fn camera_reprojection_shader_carries_versioned_depth_policy() {
    assert!(CAMERA_REPROJECTION_WGSL.contains("CAMERA_DEPTH_ABSOLUTE_EPSILON: f32 = 0.001"));
    assert!(CAMERA_REPROJECTION_WGSL.contains("CAMERA_DEPTH_RELATIVE_EPSILON: f32 = 0.001"));
    assert_eq!(camera::DEPTH_POLICY_REVISION, 1);
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
        CAMERA_REPROJECTION_WGSL
            .contains("write_current(sample_index, 0.0, 0.0, INVALID_HISTORY_SAMPLE_COUNT, 0u)")
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
fn maintained_program_sources_are_retained_by_the_program_owner() {
    let first_evaluator =
        retained_maintained_evaluator_source().expect("maintained evaluator should admit");
    let first_temporal =
        retained_temporal_reconstruction_source().expect("temporal reconstruction should admit");
    let first_camera =
        retained_camera_reprojection_source().expect("camera reprojection should admit");

    for _ in 0..120 {
        let next_evaluator = retained_maintained_evaluator_source()
            .expect("maintained evaluator should remain available");
        let next_temporal = retained_temporal_reconstruction_source()
            .expect("temporal reconstruction should remain available");
        let next_camera = retained_camera_reprojection_source()
            .expect("camera reprojection should remain available");

        assert!(first_evaluator.is_same_record(&next_evaluator));
        assert!(first_temporal.is_same_record(&next_temporal));
        assert!(first_camera.is_same_record(&next_camera));
    }
}

#[test]
fn maintained_wgsl_forms_a_canonical_compute_pipeline() {
    let programs = build_maintained_program_sources()
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
    assert_eq!(super::layout::align_up(12, 4), Ok(12));
    assert_eq!(super::layout::align_up(12, 8), Ok(16));
    assert_eq!(
        super::layout::align_up(12, 0),
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
    let error =
        super::passes::deterministic_dispatch_size(4097, 8).expect_err("dispatch must reject");
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

    let transfer_preparation = PreparedGpuData::<TransferData>::ordinary_pod_transfer("", &[1_u32])
        .expect_err("empty label must fail in RunenGPU transfer preparation");
    assert_owner(
        RenderRunenGpuPreparationError::TransferPreparation {
            stage: "transfer-preparation",
            source: transfer_preparation.clone(),
        },
        &transfer_preparation,
        "transfer-preparation",
    );

    let programs = build_maintained_program_sources()
        .expect("maintained programs must compile through RunenShader and admit through RunenGPU");
    let program_contract = GpuComputePipelineDescriptor::ordinary(programs.evaluator().clone(), "")
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
    let source = concat!(
        include_str!("mod.rs"),
        include_str!("state.rs"),
        include_str!("lifecycle.rs"),
        include_str!("errors.rs"),
        include_str!("prepare.rs"),
        include_str!("submission.rs"),
        include_str!("packing.rs"),
    );
    assert!(!source.contains(concat!("gpu_", "authoring(")));
    assert!(!source.contains(concat!("RunenGpu", "Authoring")));
}

#[test]
fn failed_retained_occurrence_discards_committed_temporal_history() {
    let mut resources = DeterministicResourceCache::default();
    let scope = 91;
    let output_index = 0;
    let shutter = crate::space_time::RenderTimeInterval::instant(
        crate::space_time::RenderTimePoint::from_seconds(0.0).expect("finite temporal test time"),
    );
    let perspective = RenderPerspectiveObservation::new(
        RenderAffineTransform3::identity(),
        std::f64::consts::FRAC_PI_3,
        1.0,
        shutter,
        RenderSamplingSupport::perspective_lattice_cell(),
    )
    .expect("valid temporal test observation");
    let signature = DeterministicTemporalSignature {
        scene_revision: crate::scene::RenderSceneStore::new().revision(),
        observation: temporal_observation_compatibility(
            RenderObservationSpec::Perspective(perspective),
            true,
        ),
        output: RenderOutputSpec::new(
            RenderOutputValue::ObjectIdentity,
            crate::request::RenderResultTopology::sample_lattice_2d(2, 2)
                .expect("temporal test lattice"),
            crate::request::RenderSemanticTolerance::exact(),
        )
        .expect("temporal test output"),
        semantic_inputs: Vec::new(),
        field_semantic_inputs: Vec::new(),
        evaluation_extent: (2, 2),
        sequence_revision: 1,
        reconstruction_revision: 1,
        camera_reprojection_revision: Some(1),
        depth_policy_revision: Some(1),
    };
    let selection = DeterministicTemporalHistorySelection {
        current_observation: perspective,
        camera_capable: true,
    };

    resources
        .temporal_history(
            scope,
            output_index,
            signature.clone(),
            (2, 2),
            256,
            selection,
        )
        .expect("first temporal history preparation");
    resources.reconcile_temporal_outputs(scope, true);
    let committed = resources
        .temporal_histories
        .get(&(scope, output_index))
        .expect("completed temporal history");
    assert_eq!(committed.age, 1);

    resources
        .temporal_history(scope, output_index, signature, (2, 2), 256, selection)
        .expect("second temporal history preparation");
    resources.reconcile_temporal_outputs(scope, false);
    assert!(
        !resources
            .temporal_histories
            .contains_key(&(scope, output_index)),
        "failed accepted occurrence must not preserve successful temporal history"
    );
}
