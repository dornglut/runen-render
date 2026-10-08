//! Vulkan-capable edge proofs for the maintained deterministic execution seam.
//!
//! These fixtures own no evaluator or method authority. They exercise the maintained public-RunenGPU
//! path to prove emitter cardinality and geometric-miss physical encodings, then require the private
//! RR566-EVAL-001 verifier to certify the same exact submissions.

use crate::admission::{
    RenderOutputBinding, RenderOutputDestination, RenderRepresentationAvailabilityFact,
    RenderRepresentationAvailabilityState,
};
use crate::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
use crate::participation::{RenderMaterialAssignment, RenderObjectParticipation};
use crate::representation::{
    RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION, RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
    RenderOrientedSurfaceProtocolEvidence, RenderRefinementEvidence, RenderRepresentationRecord,
    RenderSurfaceProtocolEvidence,
};
use crate::request::{
    RenderDistanceConvention, RenderObservationSpec, RenderOutputSpec, RenderOutputValue,
    RenderPerspectiveObservation, RenderProbeObservation, RenderRadiometricRepresentation,
    RenderRequest, RenderRequestedOutput, RenderResultTopology, RenderSamplingSupport,
    RenderSemanticTolerance,
};
use crate::runtime::admission::admit_deterministic_render_with_semantic_inputs;
use crate::runtime::carrier::{decode_word, maintained_evaluation_value};
use crate::runtime::execution::DeterministicVerificationSubmission;
use crate::runtime::verification::{
    submit_deterministic_render_for_verified_formation, verify_completed_deterministic_render,
};
use crate::scene::{RenderObjectState, RenderSceneSnapshot, RenderSceneStore, RenderSceneUpdate};
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
    GpuBufferDescriptor, GpuBufferInitialization, GpuBufferUsage, GpuCapabilityProfile, GpuContext,
    GpuContextDescriptor, GpuContextRequestErrorCategory, GpuFormatRole, GpuReadbackId,
    GpuReadbackStatus, GpuReconstruction, GpuResourceLifetime, GpuSubmissionStatus,
    GpuTextureDescriptor, GpuTextureFormat, GpuTextureInitialization, GpuTextureUsage,
    GpuWorkResourceIdAllocator,
};
use std::time::{Duration, Instant};

const TEST_WAVELENGTH_METERS: f64 = 550.0e-9;
const NUMERIC_TOLERANCE: f64 = 1.0e-4;

struct EdgeFixture {
    scene: RenderSceneSnapshot,
    request: RenderRequest,
    semantic_inputs: Vec<RenderSurfaceSemanticInputBinding>,
    availability: Vec<RenderRepresentationAvailabilityFact>,
}

fn instant() -> RenderTimeInterval {
    RenderTimeInterval::instant(
        RenderTimePoint::from_seconds(0.0).expect("finite R7 edge-proof time"),
    )
}

fn object_state(translation_scene: [f64; 3]) -> RenderObjectState {
    RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("R7 edge-proof local space"),
            RenderAffineTransform3::from_row_major_3x4([
                1.0,
                0.0,
                0.0,
                translation_scene[0],
                0.0,
                1.0,
                0.0,
                translation_scene[1],
                0.0,
                0.0,
                1.0,
                translation_scene[2],
            ])
            .expect("finite translation-only R7 edge-proof transform"),
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    )
}

fn oriented_surface_evidence() -> RenderSurfaceProtocolEvidence {
    RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
        .expect("R7 edge-proof surface protocol")
        .with_oriented_surface(
            RenderOrientedSurfaceProtocolEvidence::exact(
                RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            )
            .expect("R7 edge-proof oriented-surface protocol"),
        )
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current())
}

fn insert_geometry(
    store: &mut RenderSceneStore,
    translation_scene: [f64; 3],
) -> crate::representation::RenderRepresentationId {
    let object_id = store.allocate_object_id().expect("R7 edge-proof object id");
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, object_state(translation_scene));
    store.commit(insert).expect("insert R7 edge-proof object");

    let representation_id = store
        .allocate_representation_id(object_id)
        .expect("R7 edge-proof representation id");
    let representation = RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(oriented_surface_evidence()),
        None,
    )
    .expect("R7 edge-proof representation");
    let material = RenderMaterialAssignment::new(
        RenderDiffuseMaterial::new(0.5).expect("R7 edge-proof diffuse material"),
    );
    let participation = RenderObjectParticipation::new(vec![representation], Some(material), None)
        .expect("R7 edge-proof geometry participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    store
        .commit(attach)
        .expect("attach R7 edge-proof geometry participation");
    representation_id
}

fn insert_emitter(store: &mut RenderSceneStore, wavelength_meters: f64, irradiance: f64) {
    let object_id = store
        .allocate_object_id()
        .expect("R7 edge-proof emitter id");
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, object_state([0.0, 0.0, 0.0]));
    store.commit(insert).expect("insert R7 edge-proof emitter");

    let emitter = RenderDirectionalEmitter::new([0.0, 0.0, 1.0], wavelength_meters, irradiance)
        .expect("R7 edge-proof directional emitter");
    let participation = RenderObjectParticipation::new(Vec::new(), None, Some(emitter))
        .expect("R7 edge-proof emitter participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    store
        .commit(attach)
        .expect("attach R7 edge-proof emitter participation");
}

fn edge_fixture(
    translation_scene: [f64; 3],
    request: RenderRequest,
    emitters: &[(f64, f64)],
) -> EdgeFixture {
    let mut store = RenderSceneStore::new();
    let representation_id = insert_geometry(&mut store, translation_scene);
    for &(wavelength_meters, irradiance) in emitters {
        insert_emitter(&mut store, wavelength_meters, irradiance);
    }
    let semantic_input = RenderSurfaceSemanticInput::sphere(
        [0.0, 0.0, 0.0],
        1.0,
        RenderTemporalSupport::unbounded(),
    )
    .expect("R7 edge-proof sphere semantic input");
    EdgeFixture {
        scene: store.snapshot(),
        request,
        semantic_inputs: vec![RenderSurfaceSemanticInputBinding::new(
            representation_id,
            semantic_input,
        )],
        availability: vec![RenderRepresentationAvailabilityFact::new(
            representation_id,
            RenderRepresentationAvailabilityState::Available,
        )],
    }
}

fn radiometric() -> RenderRadiometricRepresentation {
    RenderRadiometricRepresentation::spectral_at_wavelength_meters(TEST_WAVELENGTH_METERS)
        .expect("R7 edge-proof radiometric representation")
}

fn numeric_tolerance() -> RenderSemanticTolerance {
    RenderSemanticTolerance::absolute(NUMERIC_TOLERANCE).expect("R7 edge-proof numeric tolerance")
}

fn probe_radiance_request() -> RenderRequest {
    let shutter = instant();
    let observation = RenderObservationSpec::Probe(
        RenderProbeObservation::new(
            RenderAffineTransform3::identity(),
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("R7 edge-proof probe"),
    );
    RenderRequest::new(
        shutter,
        vec![observation],
        vec![RenderRequestedOutput::new(
            0,
            RenderOutputSpec::new(
                RenderOutputValue::Radiance {
                    representation: radiometric(),
                },
                RenderResultTopology::scalar(),
                numeric_tolerance(),
            )
            .expect("R7 edge-proof probe radiance output"),
        )],
    )
    .expect("R7 edge-proof probe request")
}

fn perspective_miss_request() -> RenderRequest {
    let shutter = instant();
    let observation = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_3,
            1.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("R7 edge-proof perspective"),
    );
    let lattice =
        || RenderResultTopology::sample_lattice_2d(1, 1).expect("R7 edge-proof one-sample lattice");
    RenderRequest::new(
        shutter,
        vec![observation],
        vec![
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::Radiance {
                        representation: radiometric(),
                    },
                    lattice(),
                    numeric_tolerance(),
                )
                .expect("R7 edge-proof miss radiance output"),
            ),
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::Distance {
                        convention: RenderDistanceConvention::ObservationForwardDepth,
                    },
                    lattice(),
                    numeric_tolerance(),
                )
                .expect("R7 edge-proof miss depth output"),
            ),
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::ObjectIdentity,
                    lattice(),
                    RenderSemanticTolerance::exact(),
                )
                .expect("R7 edge-proof miss identity output"),
            ),
        ],
    )
    .expect("R7 edge-proof perspective miss request")
}

fn request_execution_context() -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopyDestination)
            .with_label("RunenRender R7 edge execution proof");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "permanent R7 edge proof requires a public RunenGPU adapter"
            );
            None
        }
        Err(error) => panic!("unexpected R7 edge-proof RunenGPU context failure: {error}"),
    }
}

fn scalar_bindings(request: &RenderRequest) -> Vec<RenderOutputBinding> {
    let output_count = request.outputs().len();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    (0..output_count)
        .map(|output_index| {
            let destination = allocator
                .allocate_buffer_handle(
                    GpuBufferDescriptor::ordinary_owned(
                        format!("R7 edge scalar output {output_index}"),
                        GpuResourceLifetime::Transient,
                        GpuReconstruction::SourceBacked,
                        4,
                        [GpuBufferUsage::CopyDestination],
                        GpuBufferInitialization::Uninitialized,
                    )
                    .expect("R7 edge scalar descriptor"),
                )
                .expect("R7 edge scalar handle");
            RenderOutputBinding::new(
                request.output_handle(output_index).expect("output handle"),
                RenderOutputDestination::ScalarBuffer(destination),
            )
        })
        .collect()
}

fn lattice_bindings(request: &RenderRequest) -> Vec<RenderOutputBinding> {
    let output_count = request.outputs().len();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    (0..output_count)
        .map(|output_index| {
            let destination = allocator
                .allocate_texture_handle(
                    GpuTextureDescriptor::ordinary_owned_2d(
                        format!("R7 edge lattice output {output_index}"),
                        GpuResourceLifetime::Transient,
                        GpuReconstruction::SourceBacked,
                        1,
                        1,
                        GpuTextureFormat::R32Uint,
                        [GpuTextureUsage::CopyDestination],
                        GpuTextureInitialization::Uninitialized,
                    )
                    .expect("R7 edge lattice descriptor"),
                )
                .expect("R7 edge lattice handle");
            RenderOutputBinding::new(
                request.output_handle(output_index).expect("output handle"),
                RenderOutputDestination::SampleLatticeTexture(destination),
            )
        })
        .collect()
}

fn submit_verified(
    fixture: &EdgeFixture,
    output_bindings: &[RenderOutputBinding],
    context: &GpuContext,
) -> DeterministicVerificationSubmission {
    let admitted = admit_deterministic_render_with_semantic_inputs(
        &fixture.scene,
        &fixture.request,
        &fixture.semantic_inputs,
        &[],
        &fixture.availability,
        output_bindings,
        context,
    )
    .expect("R7 edge fixture must reach maintained deterministic admission");
    pollster::block_on(submit_deterministic_render_for_verified_formation(
        admitted, context,
    ))
    .expect("R7 edge fixture must reach verified same-submission execution")
}

fn wait_for_verification(context: &GpuContext, verification: &DeterministicVerificationSubmission) {
    let submission = verification.submitted().submission();
    let readback_ids = verification
        .readbacks()
        .iter()
        .flat_map(|correlation| {
            [
                correlation.canonical_output(),
                correlation.definedness(),
                correlation.status(),
            ]
        })
        .collect::<Vec<GpuReadbackId>>();
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        context.progress();
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("R7 edge verified submission failed: {failure:?}");
        }
        let mut ready = true;
        for readback_id in &readback_ids {
            match submission
                .readback(*readback_id)
                .expect("R7 edge correlation must remain on exact submission")
                .status()
            {
                GpuReadbackStatus::Ready(_) => {}
                GpuReadbackStatus::Pending => ready = false,
                GpuReadbackStatus::Failed(failure) => {
                    panic!("R7 edge private readback failed: {failure:?}")
                }
            }
        }
        if ready && matches!(submission.status(), GpuSubmissionStatus::Completed) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "R7 edge private readbacks did not become ready before timeout"
        );
        std::thread::yield_now();
    }
}

fn ready_first_word(verification: &DeterministicVerificationSubmission, id: GpuReadbackId) -> u32 {
    let bytes = match verification
        .submitted()
        .submission()
        .readback(id)
        .expect("R7 edge readback must belong to exact submission")
        .status()
    {
        GpuReadbackStatus::Ready(bytes) => bytes,
        status => panic!("R7 edge readback must be ready after wait: {status:?}"),
    };
    let word = bytes
        .as_bytes()
        .get(..4)
        .expect("R7 edge readback must contain at least one word");
    decode_word(word.try_into().expect("R7 edge word has four bytes"))
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= NUMERIC_TOLERANCE,
        "R7 edge numeric mismatch: actual={actual}, expected={expected}"
    );
}

fn verified_probe_radiance(context: &GpuContext, emitters: &[(f64, f64)]) -> f64 {
    let fixture = edge_fixture([0.0, 0.0, -3.0], probe_radiance_request(), emitters);
    let bindings = scalar_bindings(&fixture.request);
    let verification = submit_verified(&fixture, &bindings, context);
    wait_for_verification(context, &verification);
    let correlation = verification.readbacks()[0];
    assert_eq!(correlation.output_index(), 0);
    assert_eq!(
        ready_first_word(&verification, correlation.definedness()),
        1
    );
    assert_eq!(ready_first_word(&verification, correlation.status()), 0);
    let value = f64::from(
        maintained_evaluation_value(ready_first_word(
            &verification,
            correlation.canonical_output(),
        ))
        .expect("R7 edge maintained radiance must be finite"),
    );
    verify_completed_deterministic_render(verification)
        .expect("R7 edge radiance must satisfy EVAL-001");
    value
}

#[test]
fn maintained_radiance_truthfully_handles_zero_one_and_many_matching_emitters() {
    let Some(context) = request_execution_context() else {
        return;
    };

    let zero = verified_probe_radiance(&context, &[(600.0e-9, 50.0)]);
    assert_eq!(
        zero, 0.0,
        "non-matching emitters must contribute no radiance"
    );

    let one = verified_probe_radiance(&context, &[(TEST_WAVELENGTH_METERS, 1.0)]);
    assert_close(one, 0.5 / std::f64::consts::PI);

    let many = verified_probe_radiance(
        &context,
        &[
            (TEST_WAVELENGTH_METERS, 1.0),
            (TEST_WAVELENGTH_METERS, 2.0),
            (600.0e-9, 50.0),
        ],
    );
    assert_close(many, 0.5 * 3.0 / std::f64::consts::PI);
    assert!(
        many > one,
        "multiple matching emitters must add their contributions"
    );
}

#[test]
fn maintained_miss_encodings_are_truthful_and_verifier_certified() {
    let Some(context) = request_execution_context() else {
        return;
    };
    let fixture = edge_fixture(
        [10.0, 0.0, -3.0],
        perspective_miss_request(),
        &[(TEST_WAVELENGTH_METERS, 1.0)],
    );
    let bindings = lattice_bindings(&fixture.request);
    let verification = submit_verified(&fixture, &bindings, &context);
    wait_for_verification(&context, &verification);
    assert_eq!(verification.readbacks().len(), 3);

    let output = |output_index| {
        verification
            .readbacks()
            .iter()
            .find(|correlation| correlation.output_index() == output_index)
            .copied()
            .expect("R7 edge output correlation")
    };

    let radiance = output(0);
    assert_eq!(
        ready_first_word(&verification, radiance.canonical_output()),
        0
    );
    assert_eq!(ready_first_word(&verification, radiance.definedness()), 1);
    assert_eq!(ready_first_word(&verification, radiance.status()), 0);

    let depth = output(1);
    assert_eq!(ready_first_word(&verification, depth.canonical_output()), 0);
    assert_eq!(ready_first_word(&verification, depth.definedness()), 0);
    assert_eq!(ready_first_word(&verification, depth.status()), 0);

    let identity = output(2);
    assert_eq!(
        ready_first_word(&verification, identity.canonical_output()),
        0
    );
    assert_eq!(ready_first_word(&verification, identity.definedness()), 0);
    assert_eq!(ready_first_word(&verification, identity.status()), 0);
    assert_eq!(
        verification.submitted().object_identity_decoder().decode(0),
        None,
        "undefined identity code must remain outside semantic object identity"
    );

    verify_completed_deterministic_render(verification)
        .expect("R7 edge miss semantics must satisfy EVAL-001");
}
