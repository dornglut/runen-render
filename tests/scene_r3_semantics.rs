use runen_render::appearance::RenderDiffuseMaterial;
use runen_render::participation::{RenderMaterialAssignment, RenderObjectParticipation};
use runen_render::representation::{
    RENDER_SURFACE_QUERY_PROTOCOL_REVISION, RenderRepresentationId, RenderRepresentationRecord,
    RenderSurfaceProtocolEvidence,
};
use runen_render::scene::{
    RenderObjectId, RenderSceneCommitError, RenderSceneStore, RenderSceneUpdate,
};
use runen_render::space_time::{RenderSpatialCoverage, RenderTemporalSupport};

fn insert_object(store: &mut RenderSceneStore, object_id: RenderObjectId) {
    let mut update = RenderSceneUpdate::new();
    update.insert(object_id);
    store.commit(update).expect("object insert should commit");
}

fn surface_representation(
    store: &mut RenderSceneStore,
    owner: RenderObjectId,
) -> RenderRepresentationRecord {
    let representation_id = store
        .allocate_representation_id(owner)
        .expect("representation ID should allocate");
    surface_representation_with_id(representation_id)
}

fn surface_representation_with_id(
    representation_id: RenderRepresentationId,
) -> RenderRepresentationRecord {
    RenderRepresentationRecord::builder(
        representation_id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
    )
    .surface_query(Some(
        RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
            .expect("valid surface protocol"),
    ))
    .build()
    .expect("valid surface representation")
}

fn material(reflectance: f64) -> RenderMaterialAssignment {
    RenderMaterialAssignment::new(
        RenderDiffuseMaterial::new(reflectance).expect("valid diffuse material"),
    )
}

#[test]
fn invalid_second_object_rejects_valid_first_object_r3_change_atomically() {
    let mut store = RenderSceneStore::new();
    let first = store.allocate_object_id().expect("first object ID");
    let second = store.allocate_object_id().expect("second object ID");
    insert_object(&mut store, first);
    insert_object(&mut store, second);

    let first_representation = surface_representation(&mut store, first);
    let representation_id = first_representation.id();
    let valid_first =
        RenderObjectParticipation::from_representations(vec![first_representation.clone()])
            .expect("valid first participation")
            .with_material_assignment(Some(material(0.25)));
    let invalid_second =
        RenderObjectParticipation::from_representations(vec![first_representation])
            .expect("structurally valid second participation");

    let before = store.snapshot();
    let revision = store.revision();
    let mut update = RenderSceneUpdate::new();
    update
        .replace_participation(first, valid_first)
        .replace_participation(second, invalid_second);

    assert_eq!(
        store.commit(update),
        Err(RenderSceneCommitError::RepresentationOwnerMismatch {
            representation_id,
            allocated_owner: first,
            requested_owner: second,
        })
    );
    assert_eq!(store.revision(), revision);
    assert_eq!(store.snapshot(), before);
    assert!(store.snapshot().object_participation(first).is_none());
    assert!(store.snapshot().object_participation(second).is_none());
}

#[test]
fn full_and_incremental_r3_construction_have_identical_semantic_participation() {
    let mut full = RenderSceneStore::new();
    let full_object = full.allocate_object_id().expect("full object ID");
    insert_object(&mut full, full_object);
    let full_first = surface_representation(&mut full, full_object);
    let full_second = surface_representation(&mut full, full_object);
    let expected = RenderObjectParticipation::from_representations(vec![full_second, full_first])
        .expect("valid full participation")
        .with_material_assignment(Some(material(0.5)));
    let mut full_update = RenderSceneUpdate::new();
    full_update.replace_participation(full_object, expected.clone());
    full.commit(full_update)
        .expect("full participation should commit");

    let mut incremental = RenderSceneStore::new();
    let incremental_object = incremental
        .allocate_object_id()
        .expect("incremental object ID");
    insert_object(&mut incremental, incremental_object);
    let incremental_first = surface_representation(&mut incremental, incremental_object);
    let incremental_second = surface_representation(&mut incremental, incremental_object);
    let first_step =
        RenderObjectParticipation::from_representations(vec![incremental_first.clone()])
            .expect("valid first step")
            .with_material_assignment(Some(material(0.5)));
    let mut first_update = RenderSceneUpdate::new();
    first_update.replace_participation(incremental_object, first_step);
    incremental
        .commit(first_update)
        .expect("first incremental participation should commit");
    let final_step = RenderObjectParticipation::from_representations(vec![
        incremental_second,
        incremental_first,
    ])
    .expect("valid final step")
    .with_material_assignment(Some(material(0.5)));
    let mut final_update = RenderSceneUpdate::new();
    final_update.replace_participation(incremental_object, final_step.clone());
    incremental
        .commit(final_update)
        .expect("final incremental participation should commit");

    assert_eq!(
        full.snapshot().object_participation(full_object),
        Some(&expected)
    );
    assert_eq!(
        incremental
            .snapshot()
            .object_participation(incremental_object),
        Some(&final_step)
    );
    assert_eq!(expected, final_step);
}

#[test]
fn material_assignment_removal_and_missing_endpoint_are_deterministic() {
    let mut store = RenderSceneStore::new();
    let present = store.allocate_object_id().expect("present object ID");
    let missing = store.allocate_object_id().expect("missing object ID");
    insert_object(&mut store, present);

    let assignment = RenderObjectParticipation::from_representations([])
        .expect("valid material assignment")
        .with_material_assignment(Some(material(0.75)));
    let mut assign = RenderSceneUpdate::new();
    assign.replace_participation(present, assignment);
    store
        .commit(assign)
        .expect("material assignment should commit");

    let mut remove = RenderSceneUpdate::new();
    remove.clear_participation(present);
    let removal = store
        .commit(remove)
        .expect("material assignment removal should commit");
    assert_eq!(
        removal.change_set().material_assignment_changed(),
        Some(&[present][..])
    );
    assert_eq!(removal.change_set().representation_changed(), Some(&[][..]));
    assert_eq!(removal.change_set().emitter_changed(), Some(&[][..]));
    assert!(removal.snapshot().object_participation(present).is_none());

    let before = store.snapshot();
    let revision = store.revision();
    let missing_assignment = RenderObjectParticipation::from_representations([])
        .expect("valid missing assignment value")
        .with_material_assignment(Some(material(0.5)));
    let mut invalid = RenderSceneUpdate::new();
    invalid.replace_participation(missing, missing_assignment);
    assert_eq!(
        store.commit(invalid),
        Err(RenderSceneCommitError::ObjectMissing { object_id: missing })
    );
    assert_eq!(store.revision(), revision);
    assert_eq!(store.snapshot(), before);
}

#[test]
fn material_assignment_replacement_is_deterministic_and_precise() {
    let mut store = RenderSceneStore::new();
    let object_id = store.allocate_object_id().expect("object ID");
    insert_object(&mut store, object_id);

    let initial = RenderObjectParticipation::from_representations([])
        .expect("valid initial material assignment")
        .with_material_assignment(Some(material(0.25)));
    let mut assign = RenderSceneUpdate::new();
    assign.replace_participation(object_id, initial);
    store
        .commit(assign)
        .expect("initial material assignment should commit");
    let revision = store.revision();

    let replacement = RenderObjectParticipation::from_representations([])
        .expect("valid replacement material assignment")
        .with_material_assignment(Some(material(0.75)));
    let mut replace = RenderSceneUpdate::new();
    replace.replace_participation(object_id, replacement.clone());
    let commit = store
        .commit(replace)
        .expect("material assignment replacement should commit");

    assert_ne!(commit.revision(), revision);
    assert_eq!(commit.snapshot().object_ids(), vec![object_id]);
    assert_eq!(
        commit.snapshot().object_participation(object_id),
        Some(&replacement)
    );
    assert_eq!(
        commit.change_set().material_assignment_changed(),
        Some(&[object_id][..])
    );
    assert_eq!(commit.change_set().representation_changed(), Some(&[][..]));
    assert_eq!(commit.change_set().emitter_changed(), Some(&[][..]));
    assert_eq!(commit.change_set().inserted(), Some(&[][..]));
    assert_eq!(commit.change_set().removed(), Some(&[][..]));
}

#[test]
fn object_identity_is_stable_across_representation_add_replace_and_removal() {
    let mut store = RenderSceneStore::new();
    let object_id = store.allocate_object_id().expect("object ID");
    insert_object(&mut store, object_id);

    let first = surface_representation(&mut store, object_id);
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(
        object_id,
        RenderObjectParticipation::from_representations(vec![first]).expect("first participation"),
    );
    let first_commit = store
        .commit(attach)
        .expect("first representation should commit");
    assert_eq!(first_commit.snapshot().object_ids(), vec![object_id]);
    assert_eq!(first_commit.change_set().inserted(), Some(&[][..]));
    assert_eq!(first_commit.change_set().removed(), Some(&[][..]));

    let second = surface_representation(&mut store, object_id);
    let mut replace = RenderSceneUpdate::new();
    replace.replace_participation(
        object_id,
        RenderObjectParticipation::from_representations(vec![second])
            .expect("second participation"),
    );
    let second_commit = store
        .commit(replace)
        .expect("representation replacement should commit");
    assert_eq!(second_commit.snapshot().object_ids(), vec![object_id]);
    assert_eq!(second_commit.change_set().inserted(), Some(&[][..]));
    assert_eq!(second_commit.change_set().removed(), Some(&[][..]));

    let mut clear = RenderSceneUpdate::new();
    clear.clear_participation(object_id);
    let clear_commit = store
        .commit(clear)
        .expect("representation removal should commit");
    assert_eq!(clear_commit.snapshot().object_ids(), vec![object_id]);
    assert!(clear_commit.snapshot().contains(object_id));
    assert!(
        clear_commit
            .snapshot()
            .object_participation(object_id)
            .is_none()
    );
}

#[test]
fn unknown_representation_identity_is_rejected_without_publication() {
    let mut store = RenderSceneStore::new();
    let object_id = store.allocate_object_id().expect("object ID");
    insert_object(&mut store, object_id);
    let mut foreign_store = RenderSceneStore::new();
    let foreign_owner = foreign_store
        .allocate_object_id()
        .expect("foreign object ID");
    let unknown = foreign_store
        .allocate_representation_id(foreign_owner)
        .expect("foreign representation ID");
    let participation =
        RenderObjectParticipation::from_representations(vec![surface_representation_with_id(
            unknown,
        )])
        .expect("structurally valid participation");
    let before = store.snapshot();

    let mut update = RenderSceneUpdate::new();
    update.replace_participation(object_id, participation);
    assert_eq!(
        store.commit(update),
        Err(RenderSceneCommitError::UnknownRepresentationId {
            representation_id: unknown,
        })
    );
    assert_eq!(store.snapshot(), before);
}

#[test]
fn mixed_same_object_structural_and_r3_operations_reject_atomically() {
    let mut store = RenderSceneStore::new();
    let object_id = store.allocate_object_id().expect("object ID");
    insert_object(&mut store, object_id);
    let representation = surface_representation(&mut store, object_id);
    let participation = RenderObjectParticipation::from_representations(vec![representation])
        .expect("valid participation");
    let before = store.snapshot();
    let revision = store.revision();

    let mut update = RenderSceneUpdate::new();
    update
        .remove(object_id)
        .replace_participation(object_id, participation);

    assert_eq!(
        store.commit(update),
        Err(RenderSceneCommitError::ConflictingOperations { object_id })
    );
    assert_eq!(store.revision(), revision);
    assert_eq!(store.snapshot(), before);
}
