//! Focused R7 proof that the retained compiled transform ignores same-object non-spatial facets.
//!
//! The compiled transform depends exactly on `ObjectSpatialState`; this proof exercises that exact
//! retained artifact through accepted temporal and R3 participation changes on the same object.

use super::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
use super::derived_transform::RenderRetainedObjectTransform;
use super::participation::{RenderMaterialAssignment, RenderObjectParticipation};
use super::representation::{
    RENDER_SURFACE_QUERY_PROTOCOL_REVISION, RenderRefinementEvidence, RenderRepresentationRecord,
    RenderSurfaceProtocolEvidence,
};
use super::scene::{RenderObjectId, RenderObjectState, RenderSceneStore, RenderSceneUpdate};
use super::space_time::{
    RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState, RenderObjectTemporalState,
    RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport, RenderTimeInterval,
    RenderTimePoint,
};

fn spatial() -> RenderObjectSpatialState {
    RenderObjectSpatialState::new(
        RenderSpaceSpec::new(0.5, RenderHandedness::Right).expect("proof local space"),
        RenderAffineTransform3::from_row_major_3x4([
            2.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0, 3.0,
        ])
        .expect("proof object transform"),
        RenderSpatialCoverage::unbounded(),
    )
}

fn state(validity: RenderTemporalSupport) -> RenderObjectState {
    RenderObjectState::new(spatial(), RenderObjectTemporalState::new(validity))
}

fn surface_representation(
    store: &mut RenderSceneStore,
    object_id: RenderObjectId,
) -> RenderRepresentationRecord {
    let representation_id = store
        .allocate_representation_id(object_id)
        .expect("proof representation id");
    RenderRepresentationRecord::new(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
        RenderRefinementEvidence::none(),
        Some(
            RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
                .expect("proof surface protocol"),
        ),
        None,
    )
    .expect("proof representation")
}

#[test]
fn retained_spatial_transform_survives_same_object_non_spatial_changes() {
    let mut store = RenderSceneStore::new();
    let target = store.allocate_object_id().expect("target id");
    let mut insert = RenderSceneUpdate::new();
    insert.insert_with_state(target, state(RenderTemporalSupport::unbounded()));
    let initial = store.commit(insert).expect("initial target state");
    let retained = RenderRetainedObjectTransform::from_snapshot(initial.snapshot(), target)
        .expect("retained target transform");
    let expected = retained.compiled();

    let interval = RenderTimeInterval::new(
        RenderTimePoint::from_seconds(0.0).expect("proof start"),
        RenderTimePoint::from_seconds(2.0).expect("proof end"),
    )
    .expect("proof interval");
    let mut temporal = RenderSceneUpdate::new();
    temporal.replace_state(target, state(RenderTemporalSupport::interval(interval)));
    let temporal = store.commit(temporal).expect("temporal-only change");
    let retained = retained
        .advance(&temporal)
        .expect("temporal state is outside the transform dependency");
    assert_eq!(retained.compiled(), expected);

    let representation = surface_representation(&mut store, target);
    let representation_only =
        RenderObjectParticipation::new(vec![representation.clone()], None, None)
            .expect("representation-only participation");
    let mut representation_update = RenderSceneUpdate::new();
    representation_update.replace_participation(target, representation_only);
    let representation_commit = store
        .commit(representation_update)
        .expect("representation-only change");
    let retained = retained
        .advance(&representation_commit)
        .expect("representation state is outside the transform dependency");
    assert_eq!(retained.compiled(), expected);

    let material =
        RenderMaterialAssignment::new(RenderDiffuseMaterial::new(0.5).expect("proof material"));
    let with_material =
        RenderObjectParticipation::new(vec![representation.clone()], Some(material), None)
            .expect("material participation");
    let mut material_update = RenderSceneUpdate::new();
    material_update.replace_participation(target, with_material);
    let material_commit = store.commit(material_update).expect("material-only change");
    let retained = retained
        .advance(&material_commit)
        .expect("material assignment is outside the transform dependency");
    assert_eq!(retained.compiled(), expected);

    let emitter =
        RenderDirectionalEmitter::new([0.0, 1.0, 0.0], 550e-9, 2.0).expect("proof emitter");
    let with_emitter =
        RenderObjectParticipation::new(vec![representation], Some(material), Some(emitter))
            .expect("emitter participation");
    let mut emitter_update = RenderSceneUpdate::new();
    emitter_update.replace_participation(target, with_emitter);
    let emitter_commit = store.commit(emitter_update).expect("emitter-only change");
    let retained = retained
        .advance(&emitter_commit)
        .expect("emitter state is outside the transform dependency");
    assert_eq!(retained.compiled(), expected);
}
