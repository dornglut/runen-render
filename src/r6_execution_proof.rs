// Permanent public-RunenGPU execution proof for the bounded R6 founding renderer.
//
// The proof keeps an independent CPU semantic oracle, but owns no GPU evaluator, WGSL, packing,
// lowering, or submission authority. Execution flows only through the maintained deterministic R7
// path. Direct inspection of renderer-private same-submission bytes is proof evidence only; EVAL-001
// independently normalizes and certifies the same exact submission before semantic formation.

use super::super::deterministic_admission::{AdmittedDeterministicRender, admit_deterministic_render_with_semantic_inputs};
use super::super::deterministic_execution::DeterministicVerificationSubmission;
use super::super::deterministic_verification::{
    submit_deterministic_render_for_verified_formation, verify_completed_deterministic_render,
};
use super::super::r6_proof::FoundingRepresentationRealization;
use super::super::r6_reference_proof::{direct_lighting_radiance, observation_forward_depth};
use super::super::render_result::RenderResult;
use super::super::representation::{RenderSurfaceQuery, RenderSurfaceProtocolEvidence};
use super::super::scene::RenderSceneSnapshot;
use super::super::surface_input::{
    RenderSurfaceSemanticInput, RenderSurfaceSemanticInputBinding,
    RenderSurfaceSemanticInputRequirement,
};
use super::super::surface_result::RenderOrientedSurfaceHit;
use super::*;
use runen_gpu::{
    GpuBufferDescriptor, GpuBufferInitialization, GpuBufferUsage, GpuCapabilityProfile, GpuContext,
    GpuContextDescriptor, GpuContextRequestErrorCategory, GpuFormatRole, GpuReadbackBytes,
    GpuReadbackId, GpuReadbackStatus, GpuReconstruction, GpuResourceLifetime, GpuSubmissionStatus,
    GpuTextureDescriptor, GpuTextureFormat, GpuTextureInitialization, GpuTextureUsage,
    GpuWorkResourceIdAllocator,
};
use std::time::{Duration, Instant};

const NUMERIC_TOLERANCE: f64 = 1.0e-4;
const WORD_BYTES: usize = 4;
const SPHERE_RADIUS: f64 = 1.25;
const PLANE_POINT_LOCAL: [f64; 3] = [0.0, 1.0, -6.0];
const PLANE_NORMAL_LOCAL: [f64; 3] = [0.0, 0.0, 1.0];

struct ExecutionFixture {
    scene: RenderSceneSnapshot,
    request: RenderRequest,
    semantic_inputs: Vec<RenderSurfaceSemanticInputBinding>,
    availability: Vec<RenderRepresentationAvailabilityFact>,
    geometry_object_ids: [RenderObjectId; 3],
    representation_ids: [RenderRepresentationId; 3],
    emitter: RenderDirectionalEmitter,
}

#[derive(Debug, Clone)]
struct ReferenceRealization {
    object_id: RenderObjectId,
    realization: FoundingRepresentationRealization,
}

#[derive(Debug, Clone, Copy)]
struct ReferenceHit {
    object_id: RenderObjectId,
    hit: RenderOrientedSurfaceHit,
}

#[derive(Debug, Clone, Copy)]
struct ReferencePixel {
    radiance: f64,
    depth: f64,
    object_id: RenderObjectId,
}

struct ProofObservation {
    output_index: usize,
    canonical_words: Vec<u32>,
    definedness_words: Vec<u32>,
    status_words: Vec<u32>,
}

fn execution_tolerance() -> RenderSemanticTolerance {
    RenderSemanticTolerance::absolute(NUMERIC_TOLERANCE).expect("positive R6 execution tolerance")
}

fn maintained_surface_evidence() -> RenderSurfaceProtocolEvidence {
    RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
        .expect("R6 maintained surface protocol")
        .with_oriented_surface(
            RenderOrientedSurfaceProtocolEvidence::exact(
                RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            )
            .expect("R6 maintained oriented-surface protocol"),
        )
        .with_semantic_input_requirement(RenderSurfaceSemanticInputRequirement::current())
}

fn insert_execution_geometry(
    store: &mut RenderSceneStore,
    translation_scene: [f64; 3],
    with_field_distance: bool,
) -> (RenderObjectId, RenderRepresentationId) {
    let object_id = store.allocate_object_id().expect("R6 geometry object id");
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(object_id, object_state(translation_scene));
    store.commit(insert).expect("insert R6 geometry object");

    let representation_id = store
        .allocate_representation_id(object_id)
        .expect("R6 representation id");
    let field_distance = with_field_distance.then(|| {
        RenderFieldDistanceProtocolEvidence::new(
            RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
            RenderFieldDistanceGuarantee::exact(),
        )
        .expect("R6 exact field-distance protocol")
    });
    let representation = RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(maintained_surface_evidence()),
        field_distance,
    )
    .expect("R6 maintained representation");
    let material = RenderMaterialAssignment::new(
        RenderDiffuseMaterial::new(0.5).expect("R6 diffuse material"),
    );
    let participation = RenderObjectParticipation::new(vec![representation], Some(material), None)
        .expect("R6 geometry participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    store
        .commit(attach)
        .expect("attach R6 geometry participation");
    (object_id, representation_id)
}

fn execution_request() -> RenderRequest {
    let shutter = instant();
    let perspective = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_4,
            1.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("R6 execution perspective"),
    );
    let probe = RenderObservationSpec::Probe(
        RenderProbeObservation::new(
            RenderAffineTransform3::identity(),
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("R6 execution probe"),
    );
    let radiometric =
        RenderRadiometricRepresentation::spectral_at_wavelength_meters(PROOF_WAVELENGTH_METERS)
            .expect("R6 execution spectral domain");
    let lattice = || {
        RenderResultTopology::sample_lattice_2d(PROOF_WIDTH, PROOF_HEIGHT)
            .expect("R6 execution lattice")
    };
    let radiance = |topology| {
        RenderOutputSpec::new(
            RenderOutputValue::Radiance {
                representation: radiometric,
            },
            topology,
            execution_tolerance(),
        )
        .expect("R6 radiance output")
    };

    RenderRequest::new(
        shutter,
        vec![perspective, probe],
        vec![
            RenderRequestedOutput::new(0, radiance(lattice())),
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::Distance {
                        convention: RenderDistanceConvention::ObservationForwardDepth,
                    },
                    lattice(),
                    execution_tolerance(),
                )
                .expect("R6 depth output"),
            ),
            RenderRequestedOutput::new(
                0,
                RenderOutputSpec::new(
                    RenderOutputValue::ObjectIdentity,
                    lattice(),
                    RenderSemanticTolerance::exact(),
                )
                .expect("R6 identity output"),
            ),
            RenderRequestedOutput::new(1, radiance(RenderResultTopology::scalar())),
        ],
    )
    .expect("R6 execution request")
}

fn execution_fixture() -> ExecutionFixture {
    let mut store = RenderSceneStore::new();
    let (sphere_object, sphere_representation) =
        insert_execution_geometry(&mut store, [-1.5, 0.0, -4.0], false);
    let (plane_object, plane_representation) =
        insert_execution_geometry(&mut store, [0.0, -1.0, 0.0], false);
    let (field_object, field_representation) =
        insert_execution_geometry(&mut store, [1.5, 0.0, -4.0], true);
    let emitter_object = insert_directional_emitter(&mut store);
    let scene = store.snapshot();
    let emitter = scene
        .object_participation(emitter_object)
        .and_then(|participation| participation.emitter())
        .expect("R6 founding directional emitter");
    let semantic_inputs = vec![
        RenderSurfaceSemanticInputBinding::new(
            sphere_representation,
            RenderSurfaceSemanticInput::sphere(
                [0.0; 3],
                SPHERE_RADIUS,
                RenderTemporalSupport::unbounded(),
            )
            .expect("R6 sphere semantic input"),
        ),
        RenderSurfaceSemanticInputBinding::new(
            plane_representation,
            RenderSurfaceSemanticInput::plane(
                PLANE_POINT_LOCAL,
                PLANE_NORMAL_LOCAL,
                RenderTemporalSupport::unbounded(),
            )
            .expect("R6 plane semantic input"),
        ),
        RenderSurfaceSemanticInputBinding::new(
            field_representation,
            RenderSurfaceSemanticInput::sphere(
                [0.0; 3],
                SPHERE_RADIUS,
                RenderTemporalSupport::unbounded(),
            )
            .expect("R6 field-backed surface semantic input"),
        ),
    ];
    let representation_ids = [
        sphere_representation,
        plane_representation,
        field_representation,
    ];
    let availability = representation_ids
        .iter()
        .copied()
        .map(|representation_id| {
            RenderRepresentationAvailabilityFact::new(
                representation_id,
                RenderRepresentationAvailabilityState::Available,
            )
        })
        .collect();

    ExecutionFixture {
        scene,
        request: execution_request(),
        semantic_inputs,
        availability,
        geometry_object_ids: [sphere_object, plane_object, field_object],
        representation_ids,
        emitter,
    }
}

fn request_execution_context() -> Option<GpuContext> {
    let descriptor = GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
        .require_format_role(GpuTextureFormat::R32Uint, GpuFormatRole::CopyDestination)
        .with_label("RunenRender R6 maintained execution proof");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "permanent R6 execution CI requires a public RunenGPU adapter"
            );
            None
        }
        Err(error) => panic!("unexpected R6 RunenGPU context failure: {error}"),
    }
}

fn output_bindings() -> Vec<RenderOutputBinding> {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let mut bindings = Vec::with_capacity(4);
    for (output_index, label) in [
        "R6 maintained perspective radiance",
        "R6 maintained perspective depth",
        "R6 maintained perspective identity",
    ]
    .into_iter()
    .enumerate()
    {
        let texture = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::ordinary_owned_2d(
                    label,
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    PROOF_WIDTH,
                    PROOF_HEIGHT,
                    GpuTextureFormat::R32Uint,
                    [GpuTextureUsage::CopyDestination],
                    GpuTextureInitialization::Uninitialized,
                )
                .expect("R6 maintained texture descriptor"),
            )
            .expect("R6 maintained texture handle");
        bindings.push(RenderOutputBinding::new(
            output_index,
            RenderOutputDestination::SampleLatticeTexture(texture),
        ));
    }

    let scalar = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::ordinary_owned(
                "R6 maintained scalar radiance",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                4,
                [GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Uninitialized,
            )
            .expect("R6 maintained scalar descriptor"),
        )
        .expect("R6 maintained scalar handle");
    bindings.push(RenderOutputBinding::new(
        3,
        RenderOutputDestination::ScalarBuffer(scalar),
    ));
    bindings
}

fn admit_execution(
    fixture: &ExecutionFixture,
    context: &GpuContext,
) -> AdmittedDeterministicRender {
    let bindings = output_bindings();
    admit_deterministic_render_with_semantic_inputs(
        &fixture.scene,
        &fixture.request,
        &fixture.semantic_inputs,
        &[],
        &fixture.availability,
        &bindings,
        context,
    )
    .expect("R6 founding semantics must reach maintained deterministic admission")
}

fn reference_realizations(fixture: &ExecutionFixture) -> [ReferenceRealization; 3] {
    [
        ReferenceRealization {
            object_id: fixture.geometry_object_ids[0],
            realization: FoundingRepresentationRealization::analytic_sphere(
                fixture.representation_ids[0],
                [0.0; 3],
                SPHERE_RADIUS,
            )
            .expect("R6 reference sphere"),
        },
        ReferenceRealization {
            object_id: fixture.geometry_object_ids[1],
            realization: FoundingRepresentationRealization::analytic_plane(
                fixture.representation_ids[1],
                PLANE_POINT_LOCAL,
                PLANE_NORMAL_LOCAL,
            )
            .expect("R6 reference plane"),
        },
        ReferenceRealization {
            object_id: fixture.geometry_object_ids[2],
            realization: FoundingRepresentationRealization::field_sphere(
                fixture.representation_ids[2],
                [0.0; 3],
                SPHERE_RADIUS,
            )
            .expect("R6 reference field sphere"),
        },
    ]
}

fn nearest_reference_hit(
    admitted: &super::super::admission::AdmittedRenderPlan,
    realizations: &[ReferenceRealization],
    origin: [f64; 3],
    direction: [f64; 3],
    ignored: Option<RenderObjectId>,
) -> Result<Option<ReferenceHit>, String> {
    let mut nearest: Option<ReferenceHit> = None;
    for input in realizations {
        if ignored == Some(input.object_id) {
            continue;
        }
        let state = admitted
            .plan()
            .scene()
            .object_state(input.object_id)
            .ok_or_else(|| "R6 reference object state missing".to_string())?;
        let query = RenderSurfaceQuery::new(
            origin,
            direction,
            RenderTimePoint::from_seconds(0.0).expect("finite R6 reference time"),
        )
        .map_err(|error| format!("R6 reference query rejected: {error:?}"))?;
        let result = input
            .realization
            .oriented_surface_query(state, query)
            .map_err(|error| format!("R6 reference realization query failed: {error:?}"))?;
        let Some(hit) = result.hit() else {
            continue;
        };
        if nearest.is_none_or(|current| {
            hit.surface_hit().distance_meters() < current.hit.surface_hit().distance_meters()
        }) {
            nearest = Some(ReferenceHit {
                object_id: input.object_id,
                hit,
            });
        }
    }
    Ok(nearest)
}

fn normalize3(value: [f64; 3]) -> Result<[f64; 3], String> {
    let length = (value[0] * value[0] + value[1] * value[1] + value[2] * value[2]).sqrt();
    if length == 0.0 || !length.is_finite() {
        return Err("R6 zero/non-finite direction".to_string());
    }
    Ok(value.map(|component| component / length))
}

fn perspective_ray(
    observation: RenderPerspectiveObservation,
    x: u32,
    y: u32,
) -> Result<([f64; 3], [f64; 3]), String> {
    let transform = observation.observation_to_scene().row_major_3x4();
    let origin = [transform[3], transform[7], transform[11]];
    let u = (f64::from(x) + 0.5) / f64::from(PROOF_WIDTH);
    let v = (f64::from(y) + 0.5) / f64::from(PROOF_HEIGHT);
    let tan_half = (observation.vertical_field_of_view_radians() * 0.5).tan();
    let local = [
        (2.0 * u - 1.0) * tan_half * observation.aspect_ratio(),
        (1.0 - 2.0 * v) * tan_half,
        -1.0,
    ];
    let scene = [
        transform[0] * local[0] + transform[1] * local[1] + transform[2] * local[2],
        transform[4] * local[0] + transform[5] * local[1] + transform[6] * local[2],
        transform[8] * local[0] + transform[9] * local[1] + transform[10] * local[2],
    ];
    Ok((origin, normalize3(scene)?))
}

fn observation_forward(transform: RenderAffineTransform3) -> Result<[f64; 3], String> {
    let matrix = transform.row_major_3x4();
    normalize3([-matrix[2], -matrix[6], -matrix[10]])
}

fn reference_outputs(
    admitted: &super::super::admission::AdmittedRenderPlan,
    realizations: &[ReferenceRealization],
    emitter: RenderDirectionalEmitter,
) -> Result<(Vec<ReferencePixel>, f64), String> {
    let observations = admitted.plan().request().observations();
    let RenderObservationSpec::Perspective(perspective) = observations[0] else {
        return Err("R6 reference perspective changed".to_string());
    };
    let RenderObservationSpec::Probe(probe) = observations[1] else {
        return Err("R6 reference probe changed".to_string());
    };
    let requested =
        RenderRadiometricRepresentation::spectral_at_wavelength_meters(PROOF_WAVELENGTH_METERS)
            .expect("R6 reference radiometric representation");
    let mut pixels = Vec::new();
    for y in 0..PROOF_HEIGHT {
        for x in 0..PROOF_WIDTH {
            let (origin, direction) = perspective_ray(perspective, x, y)?;
            let hit = nearest_reference_hit(admitted, realizations, origin, direction, None)?
                .ok_or_else(|| format!("unexpected R6 CPU primary miss at ({x}, {y})"))?;
            let material = admitted
                .plan()
                .scene()
                .object_participation(hit.object_id)
                .and_then(|participation| participation.material_assignment())
                .expect("R6 reference material")
                .material();
            let occluded = nearest_reference_hit(
                admitted,
                realizations,
                hit.hit.surface_hit().position_scene_meters(),
                emitter.direction_to_source_scene(),
                Some(hit.object_id),
            )?
            .is_some();
            let radiance = direct_lighting_radiance(
                material,
                emitter,
                requested,
                hit.hit,
                occluded,
            )
            .map_err(|error| format!("R6 CPU radiance oracle failed: {error:?}"))?;
            pixels.push(ReferencePixel {
                radiance,
                depth: observation_forward_depth(perspective, hit.hit),
                object_id: hit.object_id,
            });
        }
    }

    let transform = probe.observation_to_scene().row_major_3x4();
    let origin = [transform[3], transform[7], transform[11]];
    let direction = observation_forward(probe.observation_to_scene())?;
    let hit = nearest_reference_hit(admitted, realizations, origin, direction, None)?
        .ok_or_else(|| "unexpected R6 CPU probe miss".to_string())?;
    let material = admitted
        .plan()
        .scene()
        .object_participation(hit.object_id)
        .and_then(|participation| participation.material_assignment())
        .expect("R6 probe material")
        .material();
    let occluded = nearest_reference_hit(
        admitted,
        realizations,
        hit.hit.surface_hit().position_scene_meters(),
        emitter.direction_to_source_scene(),
        Some(hit.object_id),
    )?
    .is_some();
    let probe_radiance = direct_lighting_radiance(
        material,
        emitter,
        requested,
        hit.hit,
        occluded,
    )
    .map_err(|error| format!("R6 CPU probe oracle failed: {error:?}"))?;
    Ok((pixels, probe_radiance))
}

fn wait_for_verification_readbacks(
    context: &GpuContext,
    verification: &DeterministicVerificationSubmission,
) {
    let submission = verification.submitted().submission();
    let ids = verification
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
            panic!("R6 maintained submission failed: {failure:?}");
        }
        let mut ready = true;
        for id in &ids {
            let readback = submission
                .readback(*id)
                .expect("R6 private readback must belong to the exact submission");
            match readback.status() {
                GpuReadbackStatus::Ready(_) => {}
                GpuReadbackStatus::Pending => ready = false,
                GpuReadbackStatus::Failed(failure) => {
                    panic!("R6 private readback failed: {failure:?}")
                }
            }
        }
        if ready && matches!(submission.status(), GpuSubmissionStatus::Completed) {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "R6 maintained readbacks timed out"
        );
        std::thread::yield_now();
    }
}

fn ready_bytes(
    verification: &DeterministicVerificationSubmission,
    id: GpuReadbackId,
) -> GpuReadbackBytes {
    match verification
        .submitted()
        .submission()
        .readback(id)
        .expect("R6 readback correlation must remain on exact submission")
        .status()
    {
        GpuReadbackStatus::Ready(bytes) => bytes,
        status => panic!("R6 proof readback must be ready after wait: {status:?}"),
    }
}

fn decode_words(bytes: &[u8]) -> Vec<u32> {
    let (chunks, remainder) = bytes.as_chunks::<WORD_BYTES>();
    assert!(remainder.is_empty(), "R6 proof readback must contain whole words");
    chunks.iter().map(|chunk| u32::from_ne_bytes(*chunk)).collect()
}

fn decode_canonical_words(bytes: &[u8], topology: RenderResultTopology) -> Vec<u32> {
    let Some((width, height)) = topology.sample_lattice_dimensions() else {
        assert_eq!(bytes.len(), WORD_BYTES);
        return decode_words(bytes);
    };
    let height = usize::try_from(height).expect("R6 lattice height fits usize");
    let logical_row = usize::try_from(width)
        .expect("R6 lattice width fits usize")
        .checked_mul(WORD_BYTES)
        .expect("R6 logical row byte length");
    assert_eq!(bytes.len() % height, 0);
    let stride = bytes.len() / height;
    assert!(stride >= logical_row && stride.is_multiple_of(WORD_BYTES));
    let mut words = Vec::new();
    for row in 0..height {
        let start = row * stride;
        words.extend(decode_words(&bytes[start..start + logical_row]));
    }
    words
}

fn proof_observations(verification: &DeterministicVerificationSubmission) -> Vec<ProofObservation> {
    let request = verification.submitted().admitted().admitted().plan().request();
    verification
        .readbacks()
        .iter()
        .map(|correlation| {
            let output_index = correlation.output_index();
            let topology = request.outputs()[output_index].spec().topology();
            let sample_count = topology
                .sample_lattice_dimensions()
                .map(|(width, height)| usize::try_from(width * height).unwrap())
                .unwrap_or(1);
            let canonical = ready_bytes(verification, correlation.canonical_output());
            let definedness = ready_bytes(verification, correlation.definedness());
            let status = ready_bytes(verification, correlation.status());
            let canonical_words = decode_canonical_words(canonical.as_bytes(), topology);
            let definedness_words = decode_words(definedness.as_bytes());
            let status_words = decode_words(status.as_bytes());
            assert_eq!(canonical_words.len(), sample_count);
            assert_eq!(definedness_words.len(), sample_count);
            assert_eq!(status_words.len(), sample_count);
            ProofObservation {
                output_index,
                canonical_words,
                definedness_words,
                status_words,
            }
        })
        .collect()
}

fn assert_close(label: &str, actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= NUMERIC_TOLERANCE,
        "{label}: actual={actual}, expected={expected}, tolerance={NUMERIC_TOLERANCE}"
    );
}

#[test]
fn founding_renderer_executes_through_maintained_path_and_matches_cpu_reference() {
    let Some(context) = request_execution_context() else {
        return;
    };
    let fixture = execution_fixture();
    let admitted = admit_execution(&fixture, &context);
    let admitted_plan = admitted.admitted().clone();
    let realizations = reference_realizations(&fixture);
    let (expected_pixels, expected_probe) =
        reference_outputs(&admitted_plan, &realizations, fixture.emitter)
            .expect("R6 independent CPU semantic reference");
    assert!(
        expected_pixels
            .iter()
            .any(|sample| sample.object_id == fixture.geometry_object_ids[2]),
        "R6 CPU reference must exercise the field-backed founding surface"
    );

    let verification = pollster::block_on(submit_deterministic_render_for_verified_formation(
        admitted,
        &context,
    ))
    .expect("R6 founding semantics must submit through the maintained verified path");
    wait_for_verification_readbacks(&context, &verification);
    let observations = proof_observations(&verification);
    assert_eq!(observations.len(), 4);
    for observation in &observations {
        assert!(
            observation.status_words.iter().all(|word| *word == 0),
            "R6 maintained evaluator status must remain valid"
        );
        assert!(
            observation.definedness_words.iter().all(|word| *word == 1),
            "the bounded R6 founding fixture intentionally requires every sample to be defined"
        );
    }

    let output = |index| {
        observations
            .iter()
            .find(|observation| observation.output_index == index)
            .expect("R6 proof observation must retain output correlation")
    };
    let radiance = &output(0).canonical_words;
    let depth = &output(1).canonical_words;
    let identity = &output(2).canonical_words;
    let probe = &output(3).canonical_words;
    assert_eq!(radiance.len(), expected_pixels.len());
    assert_eq!(depth.len(), expected_pixels.len());
    assert_eq!(identity.len(), expected_pixels.len());
    assert_eq!(probe.len(), 1);

    let decoder = verification.submitted().object_identity_decoder();
    let decoded_identity = identity
        .iter()
        .map(|word| decoder.decode(*word).expect("defined R6 identity must decode"))
        .collect::<Vec<_>>();
    assert!(
        decoded_identity.contains(&fixture.geometry_object_ids[2]),
        "maintained execution must actually render the field-backed founding surface"
    );
    for (index, expected) in expected_pixels.iter().enumerate() {
        assert_close(
            "R6 perspective radiance",
            f64::from(f32::from_bits(radiance[index])),
            expected.radiance,
        );
        assert_close(
            "R6 perspective depth",
            f64::from(f32::from_bits(depth[index])),
            expected.depth,
        );
        assert_eq!(
            decoded_identity[index], expected.object_id,
            "R6 semantic object identity must remain exact through the execution-local codebook"
        );
    }
    assert_close(
        "R6 scalar radiance probe",
        f64::from(f32::from_bits(probe[0])),
        expected_probe,
    );

    assert_eq!(
        verification.submitted().admitted().admitted(),
        &admitted_plan,
        "R6 verifier must remain bound to the exact maintained admission"
    );
    let formation = verify_completed_deterministic_render(verification)
        .expect("R6 founding finite evaluation must satisfy EVAL-001");
    let result = RenderResult::from_formation_evidence(formation);
    assert_eq!(result.scene_revision(), admitted_plan.scene_revision());
    assert_eq!(result.scene(), admitted_plan.plan().scene());
    assert_eq!(result.request(), admitted_plan.plan().request());
    assert_eq!(
        result.surface_semantic_inputs(),
        admitted_plan.surface_semantic_inputs(),
        "semantic result must retain the exact admitted surface-input provenance"
    );
    assert_eq!(
        result.method_id(),
        admitted_plan.selected_candidate().method_id(),
        "semantic result must retain the exact selected method"
    );
    assert_eq!(result.outputs().len(), admitted_plan.outputs().len());
    for (result_output, admitted_output) in result.outputs().iter().zip(admitted_plan.outputs()) {
        assert_eq!(result_output.output_index(), admitted_output.output_index());
        assert_eq!(
            result.request().outputs()[result_output.output_index()].observation_index(),
            admitted_output.observation_index(),
            "result output-to-observation correlation must derive from the retained request"
        );
        assert_eq!(result_output.approximation(), admitted_output.approximation());
        assert_eq!(
            result_output.object_representations().len(),
            admitted_output.object_representations().len()
        );
        for (result_object, admitted_object) in result_output
            .object_representations()
            .iter()
            .zip(admitted_output.object_representations())
        {
            assert_eq!(result_object.object_id(), admitted_object.object_id());
            assert_eq!(
                result_object.representation(),
                admitted_object.representation()
            );
        }
    }
}
