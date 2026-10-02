//! End-to-end R1-R5 semantic-spine proof for the bounded R6 founding renderer.
//!
//! This remains test-only. It proves that the exact founding semantic scene/request/method shape can
//! use the permanent R1-R5 contracts before R6 method lowering exists. Concrete primitive evaluation
//! remains in `r6_proof`; this module does not create a second realization or planning authority.

use super::admission::{
    RenderOutputBinding, RenderOutputDestination, RenderRepresentationAvailabilityFact,
    RenderRepresentationAvailabilityState, admit_render_plan,
};
use super::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
use super::method::{
    RenderAbstractExecutionRequirement, RenderMethodContract, RenderMethodId,
    RenderMethodOutputContract, RenderMethodOutputGuarantee, RenderMethodOutputKind,
    RenderMethodRepresentationRequirement, RenderObservationKind,
    RenderRepresentationProtocolRequirement, RenderSpectralRadianceSupport,
};
use super::participation::{RenderMaterialAssignment, RenderObjectParticipation};
use super::representation::{
    RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
    RENDER_SURFACE_QUERY_PROTOCOL_REVISION, RenderFieldDistanceGuarantee,
    RenderFieldDistanceProtocolEvidence, RenderOrientedSurfaceProtocolEvidence,
    RenderRefinementEvidence, RenderRepresentationId, RenderRepresentationProtocol,
    RenderRepresentationRecord, RenderSurfaceProtocolEvidence,
};
use super::request::{
    RenderDistanceConvention, RenderObservationSpec, RenderOutputSpec, RenderOutputValue,
    RenderPerspectiveObservation, RenderProbeObservation, RenderRadiometricRepresentation,
    RenderRequest, RenderRequestedOutput, RenderResultTopology, RenderSamplingSupport,
    RenderSemanticTolerance,
};
use super::scene::{RenderObjectId, RenderObjectState, RenderSceneStore, RenderSceneUpdate};
use super::semantic_plan::{RenderPlan, plan_render};
use super::space_time::{
    RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState, RenderObjectTemporalState,
    RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport, RenderTimeInterval,
    RenderTimePoint,
};
use runen_gpu::{
    GpuBufferDescriptor, GpuBufferInitialization, GpuBufferUsage, GpuBufferUsages,
    GpuCapabilityProfile, GpuContext, GpuContextDescriptor, GpuContextRequestErrorCategory,
    GpuMemoryIntent, GpuReconstruction, GpuResourceCommon, GpuResourceLabel, GpuResourceLifetime,
    GpuResourceProvenance, GpuTextureDescriptor, GpuTextureDimension, GpuTextureExtent,
    GpuTextureFormat, GpuTextureInitialization, GpuTextureUsage, GpuTextureUsages,
    GpuWorkResourceIdAllocator,
};
use std::collections::BTreeSet;

const PROOF_WAVELENGTH_METERS: f64 = 550e-9;
const PROOF_WIDTH: u32 = 2;
const PROOF_HEIGHT: u32 = 2;

struct FoundingSpineFixture {
    plan: RenderPlan,
    representation_ids: [RenderRepresentationId; 3],
    geometry_object_ids: [RenderObjectId; 3],
    emitter_object_id: RenderObjectId,
}

fn instant() -> RenderTimeInterval {
    RenderTimeInterval::instant(RenderTimePoint::from_seconds(0.0).expect("finite R6 proof time"))
}

fn object_state(translation_scene: [f64; 3]) -> RenderObjectState {
    RenderObjectState::new(
        RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("R6 proof space"),
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
            .expect("finite R6 proof transform"),
            RenderSpatialCoverage::unbounded(),
        ),
        RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
    )
}

fn oriented_surface_evidence() -> RenderSurfaceProtocolEvidence {
    RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
        .expect("surface protocol")
        .with_oriented_surface(
            RenderOrientedSurfaceProtocolEvidence::exact(
                RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
            )
            .expect("oriented surface protocol"),
        )
}

fn represented_geometry(
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
        .expect("exact field-distance protocol")
    });
    let representation = RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(oriented_surface_evidence()),
        field_distance,
    )
    .expect("R6 geometry representation");
    let material = RenderMaterialAssignment::new(
        RenderDiffuseMaterial::new(0.5).expect("minimum diffuse material"),
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

fn insert_directional_emitter(store: &mut RenderSceneStore) -> RenderObjectId {
    let object_id = store.allocate_object_id().expect("R6 emitter object id");
    let mut insert = RenderSceneUpdate::new();
    insert.insert(object_id);
    store.commit(insert).expect("insert R6 emitter object");

    let emitter = RenderDirectionalEmitter::new([0.0, 1.0, 1.0], PROOF_WAVELENGTH_METERS, 12.0)
        .expect("directional emitter");
    let participation = RenderObjectParticipation::new(vec![], None, Some(emitter))
        .expect("R6 emitter participation");
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object_id, participation);
    store
        .commit(attach)
        .expect("attach R6 emitter participation");
    object_id
}

fn oriented_requirement() -> RenderMethodRepresentationRequirement {
    RenderMethodRepresentationRequirement::new(
        RenderRepresentationProtocolRequirement::OrientedSurfaceQuery {
            revision: RENDER_ORIENTED_SURFACE_QUERY_PROTOCOL_REVISION,
        },
        None,
    )
    .expect("oriented surface requirement")
}

fn surface_requirement() -> RenderMethodRepresentationRequirement {
    RenderMethodRepresentationRequirement::new(
        RenderRepresentationProtocolRequirement::SurfaceQuery {
            revision: RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
        },
        None,
    )
    .expect("surface requirement")
}

fn founding_method() -> RenderMethodContract {
    let spectral =
        RenderSpectralRadianceSupport::new(400e-9, 700e-9).expect("spectral method support");
    let outputs = vec![
        RenderMethodOutputContract::new(
            RenderObservationKind::Perspective,
            RenderMethodOutputKind::Radiance { spectral },
            RenderMethodOutputGuarantee::Exact,
            vec![oriented_requirement()],
            true,
        )
        .expect("perspective radiance contract"),
        RenderMethodOutputContract::new(
            RenderObservationKind::Perspective,
            RenderMethodOutputKind::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth,
            },
            RenderMethodOutputGuarantee::Exact,
            vec![surface_requirement()],
            false,
        )
        .expect("perspective depth contract"),
        RenderMethodOutputContract::new(
            RenderObservationKind::Perspective,
            RenderMethodOutputKind::ObjectIdentity,
            RenderMethodOutputGuarantee::Exact,
            vec![surface_requirement()],
            false,
        )
        .expect("perspective identity contract"),
        RenderMethodOutputContract::new(
            RenderObservationKind::Probe,
            RenderMethodOutputKind::Radiance { spectral },
            RenderMethodOutputGuarantee::Exact,
            vec![oriented_requirement()],
            true,
        )
        .expect("probe radiance contract"),
    ];
    RenderMethodContract::new(
        RenderMethodId::new(1).expect("R6 method id"),
        outputs,
        vec![RenderAbstractExecutionRequirement::GeneralParallelWork],
    )
    .expect("founding direct-lighting method contract")
}

fn founding_request() -> RenderRequest {
    let shutter = instant();
    let perspective = RenderObservationSpec::Perspective(
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_2,
            1.0,
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("R6 perspective observation"),
    );
    let probe = RenderObservationSpec::Probe(
        RenderProbeObservation::new(
            RenderAffineTransform3::identity(),
            shutter,
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("R6 scalar probe"),
    );
    let radiometric =
        RenderRadiometricRepresentation::spectral_at_wavelength_meters(PROOF_WAVELENGTH_METERS)
            .expect("R6 spectral radiance domain");
    let lattice = || {
        RenderResultTopology::sample_lattice_2d(PROOF_WIDTH, PROOF_HEIGHT)
            .expect("R6 perspective lattice")
    };
    let radiance = |topology| {
        RenderOutputSpec::new(
            RenderOutputValue::Radiance {
                representation: radiometric,
            },
            topology,
            RenderSemanticTolerance::exact(),
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
                    RenderSemanticTolerance::exact(),
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
    .expect("R6 founding request")
}

fn founding_fixture() -> FoundingSpineFixture {
    let mut store = RenderSceneStore::new();
    let (sphere_object_id, sphere_representation_id) =
        represented_geometry(&mut store, [-1.5, 0.0, -4.0], false);
    let (plane_object_id, plane_representation_id) =
        represented_geometry(&mut store, [0.0, -1.0, 0.0], false);
    let (field_object_id, field_representation_id) =
        represented_geometry(&mut store, [1.5, 0.0, -4.0], true);
    let emitter_object_id = insert_directional_emitter(&mut store);
    let snapshot = store.snapshot();
    let method = founding_method();
    let request = founding_request();
    let plan = plan_render(&snapshot, &request, std::slice::from_ref(&method))
        .expect("founding scene must use the ordinary R4 planner");
    FoundingSpineFixture {
        plan,
        representation_ids: [
            sphere_representation_id,
            plane_representation_id,
            field_representation_id,
        ],
        geometry_object_ids: [sphere_object_id, plane_object_id, field_object_id],
        emitter_object_id,
    }
}

fn common_resource(label_text: &str) -> GpuResourceCommon {
    let label = GpuResourceLabel::new(label_text).expect("R6 proof resource label");
    let provenance = GpuResourceProvenance::new(label.clone(), None, None);
    GpuResourceCommon::owned(
        label,
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance,
    )
    .expect("R6 proof resource common")
}

fn founding_output_bindings() -> Vec<RenderOutputBinding> {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let mut bindings = Vec::with_capacity(4);
    for (output_index, label_text) in [
        "R6 perspective radiance",
        "R6 perspective depth",
        "R6 perspective identity",
    ]
    .into_iter()
    .enumerate()
    {
        let label = GpuResourceLabel::new(label_text).expect("R6 texture label");
        let usages = GpuTextureUsages::new(&label, [GpuTextureUsage::CopyDestination])
            .expect("writable R6 texture usage");
        let extent = GpuTextureExtent::new(
            &label,
            GpuTextureDimension::D2,
            PROOF_WIDTH,
            PROOF_HEIGHT,
            1,
        )
        .expect("R6 texture extent");
        let descriptor = GpuTextureDescriptor::new(
            common_resource(label_text),
            GpuTextureDimension::D2,
            extent,
            1,
            1,
            GpuTextureFormat::R8Unorm,
            usages,
            GpuTextureInitialization::Uninitialized,
        )
        .expect("R6 texture descriptor");
        let texture = allocator
            .allocate_texture_handle(descriptor)
            .expect("R6 texture handle");
        bindings.push(RenderOutputBinding::new(
            output_index,
            RenderOutputDestination::SampleLatticeTexture(texture),
        ));
    }

    let label_text = "R6 scalar radiance probe";
    let label = GpuResourceLabel::new(label_text).expect("R6 buffer label");
    let usages =
        GpuBufferUsages::new(&label, [GpuBufferUsage::Storage]).expect("writable R6 buffer usage");
    let descriptor = GpuBufferDescriptor::new(
        common_resource(label_text),
        16,
        usages,
        GpuBufferInitialization::Uninitialized,
    )
    .expect("R6 buffer descriptor");
    let buffer = allocator
        .allocate_buffer_handle(descriptor)
        .expect("R6 buffer handle");
    bindings.push(RenderOutputBinding::new(
        3,
        RenderOutputDestination::ScalarBuffer(buffer),
    ));
    bindings
}

#[test]
fn founding_scene_uses_exact_r1_r4_semantic_spine_and_independent_field_capability() {
    let fixture = founding_fixture();
    let scene = fixture.plan.scene();
    assert_eq!(scene.len(), 4, "three geometry objects plus one emitter");
    assert_eq!(fixture.plan.candidates().len(), 1);
    assert_eq!(
        fixture.plan.candidates()[0].method_id(),
        RenderMethodId::new(1).unwrap()
    );
    assert_eq!(fixture.plan.candidates()[0].outputs().len(), 4);

    let expected_objects = BTreeSet::from(fixture.geometry_object_ids);
    let expected_representations = BTreeSet::from(fixture.representation_ids);
    for (output_index, expected_protocol) in [
        RenderRepresentationProtocol::OrientedSurfaceQuery,
        RenderRepresentationProtocol::SurfaceQuery,
        RenderRepresentationProtocol::SurfaceQuery,
        RenderRepresentationProtocol::OrientedSurfaceQuery,
    ]
    .into_iter()
    .enumerate()
    {
        let output = &fixture.plan.candidates()[0].outputs()[output_index];
        let actual_objects = output
            .object_representations()
            .iter()
            .map(|object| object.object_id())
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_objects, expected_objects);
        let actual_representations = output
            .object_representations()
            .iter()
            .map(|object| {
                assert_eq!(object.uses().len(), 1);
                assert_eq!(
                    object.uses()[0].requirement().protocol().protocol(),
                    expected_protocol
                );
                object.uses()[0].representation_id()
            })
            .collect::<BTreeSet<_>>();
        assert_eq!(actual_representations, expected_representations);
    }

    for object_id in fixture.geometry_object_ids {
        assert!(
            scene
                .object_participation(object_id)
                .expect("geometry participation")
                .material_assignment()
                .is_some()
        );
    }
    let emitter = scene
        .object_participation(fixture.emitter_object_id)
        .expect("emitter participation");
    assert!(emitter.representations().is_empty());
    assert!(emitter.emitter().is_some());

    let field = scene
        .object_participation(fixture.geometry_object_ids[2])
        .expect("field object participation")
        .representation(fixture.representation_ids[2])
        .expect("field representation");
    assert!(
        field
            .field_distance_protocol(RENDER_FIELD_DISTANCE_PROTOCOL_REVISION)
            .is_ok(),
        "field-distance meaning remains independently declared even though direct-lighting planning selects surface protocols"
    );
}

#[test]
fn founding_scene_reaches_public_r5_admission_without_synthetic_semantic_bindings() {
    let fixture = founding_fixture();
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .with_label("RunenRender R6 founding admission proof");
    let context = match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => context,
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            return;
        }
        Err(error) => panic!("unexpected R6 headless RunenGPU context failure: {error}"),
    };

    let availability = fixture
        .representation_ids
        .iter()
        .copied()
        .map(|representation_id| {
            RenderRepresentationAvailabilityFact::new(
                representation_id,
                RenderRepresentationAvailabilityState::Available,
            )
        })
        .collect::<Vec<_>>();
    let bindings = founding_output_bindings();
    let admitted = admit_render_plan(&fixture.plan, &availability, &bindings, &context)
        .expect("founding R6 plan should reach ordinary public R5 admission");

    assert_eq!(admitted.plan(), &fixture.plan);
    assert_eq!(
        admitted.selected_candidate().method_id(),
        RenderMethodId::new(1).unwrap()
    );
    assert_eq!(admitted.outputs().len(), 4);
    let expected_representations = BTreeSet::from(fixture.representation_ids);
    for (output_index, output) in admitted.outputs().iter().enumerate() {
        assert_eq!(output.output_index(), output_index);
        assert_eq!(output.binding(), &bindings[output_index]);
        assert_eq!(
            output
                .object_representations()
                .iter()
                .map(|object| object.representation().representation_id())
                .collect::<BTreeSet<_>>(),
            expected_representations
        );
    }
}

mod execution {
    include!("r6_execution_proof.rs");
}
