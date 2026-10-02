//! Focused Vulkan-capable proof for sampled field-distance execution.
//!
//! These tests deliberately exercise the ordinary maintained evaluator and its renderer-private
//! same-submission readbacks. The older RR566 analytic-surface verifier remains unchanged; this
//! module does not claim that its sphere/plane reference model certifies sampled fields.

use crate::admission::{
    RenderOutputBinding, RenderOutputDestination, RenderRepresentationAvailabilityFact,
    RenderRepresentationAvailabilityState,
};
use crate::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
use crate::field_input::{
    RenderFieldSemanticInput, RenderFieldSemanticInputBinding, RenderFieldSemanticInputRequirement,
};
use crate::participation::{RenderMaterialAssignment, RenderObjectParticipation};
use crate::representation::{
    RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
    RENDER_SURFACE_QUERY_PROTOCOL_REVISION, RenderFieldDistanceGuarantee,
    RenderFieldDistanceProtocolEvidence, RenderOrientedSurfaceProtocolEvidence,
    RenderRefinementEvidence, RenderRepresentationRecord, RenderSurfaceProtocolEvidence,
};
use crate::request::{
    RenderDistanceConvention, RenderObservationSpec, RenderOutputSpec, RenderOutputValue,
    RenderPerspectiveObservation, RenderRadiometricRepresentation, RenderRequest,
    RenderRequestedOutput, RenderResultTopology, RenderSamplingSupport, RenderSemanticTolerance,
};
use crate::runtime::admission::{
    AdmittedDeterministicRender, admit_deterministic_render_with_semantic_inputs,
};
use crate::runtime::execution::{
    DeterministicVerificationSubmission, prepare_deterministic_render,
    submit_deterministic_render_for_verification,
};
use crate::scene::{
    RenderObjectId, RenderObjectState, RenderSceneSnapshot, RenderSceneStore, RenderSceneUpdate,
};
use crate::space_time::{
    RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState, RenderObjectTemporalState,
    RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport, RenderTimeInterval,
    RenderTimePoint,
};
use crate::surface_input::{
    RenderSurfaceSemanticInput, RenderSurfaceSemanticInputBinding,
    RenderSurfaceSemanticInputRequirement,
};
use runen_gpu::{
    GpuContext, GpuContextDescriptor, GpuContextRequestErrorCategory, GpuFormatRole,
    GpuReadbackStatus, GpuReconstruction, GpuResourceLifetime, GpuSubmissionStatus,
    GpuTextureDescriptor, GpuTextureFormat, GpuTextureInitialization, GpuTextureUsage,
    GpuWorkResourceIdAllocator,
};
use std::time::{Duration, Instant};

const WAVELENGTH_METERS: f64 = 550.0e-9;

fn instant() -> RenderTimeInterval {
    RenderTimeInterval::instant(
        RenderTimePoint::from_seconds(0.0).expect("finite sampled-field proof time"),
    )
}

fn object_state(local_to_scene: RenderAffineTransform3) -> RenderObjectState {
    RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right)
                .expect("sampled-field proof local space"),
            local_to_scene,
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    )
}

fn translated(z: f64) -> RenderAffineTransform3 {
    RenderAffineTransform3::from_row_major_3x4([
        1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, z,
    ])
    .expect("finite sampled-field translation")
}

fn translated_xyz(x: f64, y: f64, z: f64) -> RenderAffineTransform3 {
    RenderAffineTransform3::from_row_major_3x4([
        1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, y, 0.0, 0.0, 1.0, z,
    ])
    .expect("finite sampled-field translation")
}

fn rotated_uniform_scale(scale: f64, translation_z: f64) -> RenderAffineTransform3 {
    RenderAffineTransform3::from_row_major_3x4([
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        scale,
        0.0,
        0.0,
        -scale,
        0.0,
        0.0,
        translation_z,
    ])
    .expect("finite rotated uniform-scale transform")
}

fn field_evidence(max_error_meters: f64) -> RenderFieldDistanceProtocolEvidence {
    RenderFieldDistanceProtocolEvidence::new(
        RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
        RenderFieldDistanceGuarantee::conservative(max_error_meters)
            .expect("finite sampled-field guarantee"),
    )
    .expect("sampled-field protocol")
    .with_semantic_input_requirement(RenderFieldSemanticInputRequirement::current())
}

fn surface_evidence() -> RenderSurfaceProtocolEvidence {
    RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
        .expect("surface protocol")
        .with_oriented_surface(
            RenderOrientedSurfaceProtocolEvidence::exact(
                RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            )
            .expect("oriented surface protocol"),
        )
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current())
}

fn sampled_sphere_field(
    radius_meters: f64,
    max_query_error_meters: f64,
) -> RenderFieldSemanticInput {
    sampled_sphere_field_at([0.0; 3], radius_meters, max_query_error_meters)
}

fn sampled_sphere_field_at(
    center_local_meters: [f64; 3],
    radius_meters: f64,
    max_query_error_meters: f64,
) -> RenderFieldSemanticInput {
    let dimensions = [17_u32; 3];
    let origin = [-1.5_f64; 3];
    let spacing = [3.0 / 16.0; 3];
    let mut samples = Vec::with_capacity(17 * 17 * 17);
    for z in 0..dimensions[2] {
        for y in 0..dimensions[1] {
            for x in 0..dimensions[0] {
                let point = [
                    origin[0] + spacing[0] * f64::from(x),
                    origin[1] + spacing[1] * f64::from(y),
                    origin[2] + spacing[2] * f64::from(z),
                ];
                let relative = [
                    point[0] - center_local_meters[0],
                    point[1] - center_local_meters[1],
                    point[2] - center_local_meters[2],
                ];
                samples.push(
                    (relative[0] * relative[0]
                        + relative[1] * relative[1]
                        + relative[2] * relative[2])
                        .sqrt()
                        - radius_meters,
                );
            }
        }
    }
    RenderFieldSemanticInput::dense(
        origin,
        spacing,
        dimensions,
        samples,
        max_query_error_meters,
        RenderTemporalSupport::unbounded(),
    )
    .expect("synthetic sampled sphere field")
}

fn long_parallel_plane_field() -> RenderFieldSemanticInput {
    let dimensions = [2_u32, 2, 2];
    let origin = [0.001_f64, -1.0, -200.0];
    let spacing = [1.0_f64, 2.0, 200.0];
    let mut samples = Vec::with_capacity(8);
    for _z in 0..2 {
        for _y in 0..2 {
            for x in 0..2 {
                samples.push(origin[0] + spacing[0] * f64::from(x));
            }
        }
    }
    RenderFieldSemanticInput::dense(
        origin,
        spacing,
        dimensions,
        samples,
        0.0,
        RenderTemporalSupport::unbounded(),
    )
    .expect("exact parallel-plane field")
}

fn perspective_request(values: &[RenderOutputValue]) -> RenderRequest {
    let shutter = instant();
    let observation = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_3,
            1.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("sampled-field proof perspective"),
    );
    let outputs = values
        .iter()
        .copied()
        .enumerate()
        .map(|(output_index, value)| {
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    value,
                    RenderResultTopology::sample_lattice_2d(1, 1)
                        .expect("sampled-field proof 1x1 lattice"),
                    RenderSemanticTolerance::exact(),
                )
                .unwrap_or_else(|error| {
                    panic!("sampled-field proof output {output_index} must be valid: {error}")
                }),
            )
        })
        .collect();
    RenderRequest::new(shutter, vec![observation], outputs).expect("sampled-field proof request")
}

fn request_execution_context() -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(runen_gpu::GpuCapabilityProfile::ComputeBaseline.requirements())
            .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopyDestination)
            .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopySource)
            .with_label("RunenRender GP1B2 sampled-field execution proof");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "permanent GP1B2 execution CI requires a public RunenGPU adapter"
            );
            None
        }
        Err(error) => panic!("unexpected GP1B2 RunenGPU context failure: {error}"),
    }
}

fn lattice_bindings(output_count: usize) -> Vec<RenderOutputBinding> {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    (0..output_count)
        .map(|output_index| {
            let destination = allocator
                .allocate_texture_handle(
                    GpuTextureDescriptor::ordinary_owned_2d(
                        format!("GP1B2 sampled-field output {output_index}"),
                        GpuResourceLifetime::Transient,
                        GpuReconstruction::SourceBacked,
                        1,
                        1,
                        GpuTextureFormat::R32Uint,
                        [GpuTextureUsage::CopyDestination],
                        GpuTextureInitialization::Uninitialized,
                    )
                    .expect("sampled-field destination descriptor"),
                )
                .expect("sampled-field destination");
            RenderOutputBinding::new(
                output_index,
                RenderOutputDestination::SampleLatticeTexture(destination),
            )
        })
        .collect()
}

fn add_field_object(
    store: &mut RenderSceneStore,
    transform: RenderAffineTransform3,
    coverage_min_local_meters: [f64; 3],
    coverage_max_local_meters: [f64; 3],
    guarantee_error_meters: f64,
    reflectance: Option<f64>,
) -> (
    RenderObjectId,
    crate::representation::RenderRepresentationId,
) {
    let object_id = store.allocate_object_id().expect("field object id");
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, object_state(transform));
    store.commit(insert).expect("insert field object");

    let representation_id = store
        .allocate_representation_id(object_id)
        .expect("field representation id");
    let representation = RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::axis_aligned_bounds(
            coverage_min_local_meters,
            coverage_max_local_meters,
        )
        .expect("finite field representation coverage"),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        None,
        Some(field_evidence(guarantee_error_meters)),
    )
    .expect("field representation");
    let material = reflectance.map(|value| {
        RenderMaterialAssignment::new(
            RenderDiffuseMaterial::new(value).expect("field diffuse material"),
        )
    });
    let participation = RenderObjectParticipation::new(vec![representation], material, None)
        .expect("field participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    store.commit(attach).expect("attach field participation");
    (object_id, representation_id)
}

fn add_surface_sphere(
    store: &mut RenderSceneStore,
    transform: RenderAffineTransform3,
) -> (
    RenderObjectId,
    crate::representation::RenderRepresentationId,
) {
    let object_id = store.allocate_object_id().expect("surface object id");
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, object_state(transform));
    store.commit(insert).expect("insert surface object");

    let representation_id = store
        .allocate_representation_id(object_id)
        .expect("surface representation id");
    let representation = RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(surface_evidence()),
        None,
    )
    .expect("surface representation");
    let participation = RenderObjectParticipation::new(vec![representation], None, None)
        .expect("surface participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    store.commit(attach).expect("attach surface participation");
    (object_id, representation_id)
}

fn add_emitter(store: &mut RenderSceneStore) {
    let object_id = store.allocate_object_id().expect("emitter object id");
    let mut insert = RenderSceneUpdate::new();
    insert.insert(object_id);
    store.commit(insert).expect("insert emitter object");
    let emitter = RenderDirectionalEmitter::new([0.0, 0.0, 1.0], WAVELENGTH_METERS, 12.0)
        .expect("sampled-field directional emitter");
    let participation = RenderObjectParticipation::new(Vec::new(), None, Some(emitter))
        .expect("emitter participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    store.commit(attach).expect("attach emitter participation");
}

fn admit(
    scene: &RenderSceneSnapshot,
    request: &RenderRequest,
    surface_inputs: &[RenderSurfaceSemanticInputBinding],
    field_inputs: &[RenderFieldSemanticInputBinding],
    availability: &[RenderRepresentationAvailabilityFact],
    context: &GpuContext,
) -> AdmittedDeterministicRender {
    admit_deterministic_render_with_semantic_inputs(
        scene,
        request,
        surface_inputs,
        field_inputs,
        availability,
        &lattice_bindings(request.outputs().len()),
        context,
    )
    .expect("sampled-field work must reach maintained deterministic admission")
}

fn wait_for_private_readbacks(
    context: &GpuContext,
    verification: &DeterministicVerificationSubmission,
) {
    let submission = verification.submitted().submission();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        context.progress();
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("sampled-field submission failed: {failure:?}");
        }
        let all_ready = verification.readbacks().iter().all(|correlation| {
            [
                correlation.canonical_output(),
                correlation.definedness(),
                correlation.status(),
            ]
            .into_iter()
            .all(|id| {
                matches!(
                    submission
                        .readback(id)
                        .expect("private readback correlation")
                        .status(),
                    GpuReadbackStatus::Ready(_)
                )
            })
        });
        if all_ready && matches!(submission.status(), GpuSubmissionStatus::Completed) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "sampled-field private readbacks did not complete before timeout"
        );
        std::thread::yield_now();
    }
}

fn first_word(
    verification: &DeterministicVerificationSubmission,
    output_index: usize,
    channel: &'static str,
) -> u32 {
    let correlation = verification
        .readbacks()
        .iter()
        .find(|correlation| correlation.output_index() == output_index)
        .expect("output readback correlation");
    let id = match channel {
        "canonical" => correlation.canonical_output(),
        "defined" => correlation.definedness(),
        "status" => correlation.status(),
        _ => panic!("unknown private readback channel"),
    };
    let readback = verification
        .submitted()
        .submission()
        .readback(id)
        .expect("private readback");
    let GpuReadbackStatus::Ready(bytes) = readback.status() else {
        panic!("private readback must be ready");
    };
    let raw = bytes.as_bytes();
    assert!(raw.len() >= 4, "private readback must contain one word");
    u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]])
}

#[test]
fn sampled_field_executes_identity_depth_and_radiance_on_public_runengpu() {
    let Some(context) = request_execution_context() else {
        return;
    };

    let mut store = RenderSceneStore::new();
    let (_, representation_id) = add_field_object(
        &mut store,
        translated(-3.0),
        [-1.5; 3],
        [1.5; 3],
        0.05,
        Some(0.5),
    );
    add_emitter(&mut store);

    let request = perspective_request(&[
        RenderOutputValue::ObjectIdentity,
        RenderOutputValue::Distance {
            convention: RenderDistanceConvention::ObservationForwardDepth,
        },
        RenderOutputValue::Radiance {
            representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                WAVELENGTH_METERS,
            )
            .expect("sampled-field radiance representation"),
        },
    ]);
    let field_input =
        RenderFieldSemanticInputBinding::new(representation_id, sampled_sphere_field(1.0, 0.05));
    let availability = [RenderRepresentationAvailabilityFact::new(
        representation_id,
        RenderRepresentationAvailabilityState::Available,
    )];
    let admitted = admit(
        &store.snapshot(),
        &request,
        &[],
        &[field_input],
        &availability,
        &context,
    );
    let verification = pollster::block_on(submit_deterministic_render_for_verification(
        admitted, &context,
    ))
    .expect("sampled-field execution must submit");
    wait_for_private_readbacks(&context, &verification);

    for output_index in 0..3 {
        assert_eq!(
            first_word(&verification, output_index, "status"),
            0,
            "sampled-field evaluator must remain valid"
        );
        assert_eq!(
            first_word(&verification, output_index, "defined"),
            1,
            "central sampled-field ray must define every requested output"
        );
    }
    assert_ne!(
        first_word(&verification, 0, "canonical"),
        0,
        "sampled-field object identity must produce a non-zero execution-local code"
    );
    let depth = f32::from_bits(first_word(&verification, 1, "canonical"));
    assert!(
        (1.8..=2.0001).contains(&depth),
        "conservative sampled-field stepping must not advance past the analytic sphere front at two scene metres; got {depth}"
    );
    let radiance = f32::from_bits(first_word(&verification, 2, "canonical"));
    assert!(
        radiance.is_finite() && radiance > 0.0,
        "finite-difference field normal and material must produce positive direct radiance, got {radiance}"
    );
}

#[test]
fn sampled_field_executes_through_rotation_and_uniform_scale() {
    let Some(context) = request_execution_context() else {
        return;
    };

    let mut store = RenderSceneStore::new();
    let (_, representation_id) = add_field_object(
        &mut store,
        rotated_uniform_scale(2.0, -4.0),
        [-1.5; 3],
        [1.5; 3],
        0.10,
        None,
    );
    let request = perspective_request(&[RenderOutputValue::Distance {
        convention: RenderDistanceConvention::ObservationForwardDepth,
    }]);
    let field_inputs = [RenderFieldSemanticInputBinding::new(
        representation_id,
        sampled_sphere_field_at([0.5, 0.0, 0.0], 1.0, 0.05),
    )];
    let availability = [RenderRepresentationAvailabilityFact::new(
        representation_id,
        RenderRepresentationAvailabilityState::Available,
    )];
    let admitted = admit(
        &store.snapshot(),
        &request,
        &[],
        &field_inputs,
        &availability,
        &context,
    );
    let verification = pollster::block_on(submit_deterministic_render_for_verification(
        admitted, &context,
    ))
    .expect("similarity-transformed field execution must submit");
    wait_for_private_readbacks(&context, &verification);

    assert_eq!(first_word(&verification, 0, "status"), 0);
    assert_eq!(first_word(&verification, 0, "defined"), 1);
    let depth = f32::from_bits(first_word(&verification, 0, "canonical"));
    assert!(
        (2.8..=3.0002).contains(&depth),
        "conservative sampled-field stepping through the 2x similarity transform must not advance past the analytic front at three scene metres; got {depth}"
    );
}

#[test]
fn sampled_field_and_analytic_surface_share_one_nearest_hit_loop() {
    let Some(context) = request_execution_context() else {
        return;
    };

    let mut store = RenderSceneStore::new();
    let (_, field_representation_id) = add_field_object(
        &mut store,
        translated(-3.0),
        [-1.5; 3],
        [1.5; 3],
        0.05,
        None,
    );
    let (_, surface_representation_id) = add_surface_sphere(&mut store, translated(-5.0));

    let request = perspective_request(&[RenderOutputValue::Distance {
        convention: RenderDistanceConvention::ObservationForwardDepth,
    }]);
    let field_inputs = [RenderFieldSemanticInputBinding::new(
        field_representation_id,
        sampled_sphere_field(1.0, 0.05),
    )];
    let surface_inputs = [RenderSurfaceSemanticInputBinding::new(
        surface_representation_id,
        RenderSurfaceSemanticInput::sphere([0.0; 3], 1.0, RenderTemporalSupport::unbounded())
            .expect("analytic sphere input"),
    )];
    let availability = [
        RenderRepresentationAvailabilityFact::new(
            field_representation_id,
            RenderRepresentationAvailabilityState::Available,
        ),
        RenderRepresentationAvailabilityFact::new(
            surface_representation_id,
            RenderRepresentationAvailabilityState::Available,
        ),
    ];
    let admitted = admit(
        &store.snapshot(),
        &request,
        &surface_inputs,
        &field_inputs,
        &availability,
        &context,
    );
    let verification = pollster::block_on(submit_deterministic_render_for_verification(
        admitted, &context,
    ))
    .expect("mixed geometry execution must submit");
    wait_for_private_readbacks(&context, &verification);

    assert_eq!(first_word(&verification, 0, "status"), 0);
    assert_eq!(first_word(&verification, 0, "defined"), 1);
    let depth = f32::from_bits(first_word(&verification, 0, "canonical"));
    assert!(
        (1.8..=2.1).contains(&depth),
        "the nearer sampled field must beat the analytic sphere behind it, got depth {depth}"
    );
}

#[test]
fn adding_field_objects_does_not_add_evaluator_passes() {
    let Some(context) = request_execution_context() else {
        return;
    };

    let request = perspective_request(&[RenderOutputValue::ObjectIdentity]);

    let mut one_store = RenderSceneStore::new();
    let (_, one_representation) = add_field_object(
        &mut one_store,
        translated(-3.0),
        [-1.5; 3],
        [1.5; 3],
        0.05,
        None,
    );
    let one_inputs = [RenderFieldSemanticInputBinding::new(
        one_representation,
        sampled_sphere_field(1.0, 0.05),
    )];
    let one_availability = [RenderRepresentationAvailabilityFact::new(
        one_representation,
        RenderRepresentationAvailabilityState::Available,
    )];
    let one_prepared = prepare_deterministic_render(
        admit(
            &one_store.snapshot(),
            &request,
            &[],
            &one_inputs,
            &one_availability,
            &context,
        ),
        &context,
    )
    .expect("one-field preparation");

    let mut two_store = RenderSceneStore::new();
    let (_, first_representation) = add_field_object(
        &mut two_store,
        translated(-3.0),
        [-1.5; 3],
        [1.5; 3],
        0.05,
        None,
    );
    let (_, second_representation) = add_field_object(
        &mut two_store,
        translated_xyz(4.0, 0.0, -3.0),
        [-1.5; 3],
        [1.5; 3],
        0.05,
        None,
    );
    let two_inputs = [
        RenderFieldSemanticInputBinding::new(first_representation, sampled_sphere_field(1.0, 0.05)),
        RenderFieldSemanticInputBinding::new(
            second_representation,
            sampled_sphere_field(1.0, 0.05),
        ),
    ];
    let two_availability = [
        RenderRepresentationAvailabilityFact::new(
            first_representation,
            RenderRepresentationAvailabilityState::Available,
        ),
        RenderRepresentationAvailabilityFact::new(
            second_representation,
            RenderRepresentationAvailabilityState::Available,
        ),
    ];
    let two_prepared = prepare_deterministic_render(
        admit(
            &two_store.snapshot(),
            &request,
            &[],
            &two_inputs,
            &two_availability,
            &context,
        ),
        &context,
    )
    .expect("two-field preparation");

    assert_eq!(
        two_prepared.work_set().fragments().len(),
        one_prepared.work_set().fragments().len(),
        "adding a field object must not add an evaluator fragment"
    );
    let one_nodes = one_prepared
        .work_set()
        .fragments()
        .iter()
        .map(|fragment| fragment.nodes().len())
        .sum::<usize>();
    let two_nodes = two_prepared
        .work_set()
        .fragments()
        .iter()
        .map(|fragment| fragment.nodes().len())
        .sum::<usize>();
    assert_eq!(
        two_nodes, one_nodes,
        "adding a field object must not add a per-entity GPU pass"
    );
}

#[test]
fn sampled_field_step_budget_exhaustion_fails_closed() {
    let Some(context) = request_execution_context() else {
        return;
    };

    let mut store = RenderSceneStore::new();
    let (_, representation_id) = add_field_object(
        &mut store,
        translated_xyz(-0.001, 0.0, -1.0),
        [0.001, -1.0, -200.0],
        [1.001, 1.0, 0.0],
        0.0,
        None,
    );
    let request = perspective_request(&[RenderOutputValue::ObjectIdentity]);
    let field_inputs = [RenderFieldSemanticInputBinding::new(
        representation_id,
        long_parallel_plane_field(),
    )];
    let availability = [RenderRepresentationAvailabilityFact::new(
        representation_id,
        RenderRepresentationAvailabilityState::Available,
    )];
    let admitted = admit(
        &store.snapshot(),
        &request,
        &[],
        &field_inputs,
        &availability,
        &context,
    );
    let verification = pollster::block_on(submit_deterministic_render_for_verification(
        admitted, &context,
    ))
    .expect("bounded-step sampled field must submit");
    wait_for_private_readbacks(&context, &verification);

    assert_eq!(
        first_word(&verification, 0, "status"),
        1,
        "step-budget exhaustion must become evaluator-invalid evidence"
    );
    assert_eq!(
        first_word(&verification, 0, "defined"),
        0,
        "step-budget exhaustion must not be relabeled as a defined miss"
    );
}
