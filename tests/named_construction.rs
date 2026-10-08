//! External consumer acceptance proofs for canonical named construction and scene normalization.
use runen_render::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
use runen_render::participation::{
    RenderMaterialAssignment, RenderObjectParticipation, RenderParticipationValidationError,
};
use runen_render::representation::{
    RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
    RenderFieldDistanceGuarantee, RenderFieldDistanceProtocolEvidence,
    RenderProtocolCompatibilityError, RenderRefinementEvidence, RenderRepresentationId,
    RenderRepresentationProtocol, RenderRepresentationRecord, RenderRepresentationValidationError,
    RenderSurfaceProtocolEvidence,
};
use runen_render::scene::{RenderSceneStore, RenderSceneUpdate};
use runen_render::space_time::{
    RenderSpatialCoverage, RenderTemporalSupport, RenderTimeInterval, RenderTimePoint,
};

fn surface() -> RenderSurfaceProtocolEvidence {
    RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION).unwrap()
}

fn field() -> RenderFieldDistanceProtocolEvidence {
    RenderFieldDistanceProtocolEvidence::new(
        RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
        RenderFieldDistanceGuarantee::conservative(0.25).unwrap(),
    )
    .unwrap()
}

fn material() -> RenderMaterialAssignment {
    RenderMaterialAssignment::new(RenderDiffuseMaterial::new(0.5).unwrap())
}

fn emitter() -> RenderDirectionalEmitter {
    RenderDirectionalEmitter::new([0.0, 0.0, 1.0], 550e-9, 12.0).unwrap()
}

fn record(id: RenderRepresentationId) -> RenderRepresentationRecord {
    RenderRepresentationRecord::builder(
        id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
    )
    .surface_query(Some(surface()))
    .build()
    .unwrap()
}

#[test]
fn dynamic_protocol_channels_preserve_facts_compatibility_and_no_protocol_error() {
    let mut store = RenderSceneStore::new();
    let object = store.allocate_object_id().unwrap();
    let id = store.allocate_representation_id(object).unwrap();
    let coverage = RenderSpatialCoverage::axis_aligned_bounds([-1.0; 3], [1.0; 3]).unwrap();
    let interval = RenderTimeInterval::instant(RenderTimePoint::from_seconds(2.0).unwrap());
    let temporal = RenderTemporalSupport::interval(interval);
    let refinement = RenderRefinementEvidence::bounded(0.01).unwrap();
    for (surface, field) in [
        (None, None),
        (Some(surface()), None),
        (None, Some(field())),
        (Some(surface()), Some(field())),
    ] {
        let candidate = RenderRepresentationRecord::builder(id, coverage.clone(), temporal)
            .surface_query(surface)
            .field_distance(field)
            .refinement(refinement)
            .build();
        if surface.is_none() && field.is_none() {
            assert_eq!(
                candidate,
                Err(RenderRepresentationValidationError::NoProtocols)
            );
            continue;
        }
        let candidate = candidate.unwrap();
        assert_eq!(candidate.id(), id);
        assert_eq!(candidate.spatial_coverage(), &coverage);
        assert_eq!(candidate.temporal_support(), temporal);
        assert_eq!(candidate.refinement(), refinement);
        assert_eq!(candidate.supports_field_distance(), field.is_some());
        assert_eq!(
            candidate.surface_query_protocol(RENDER_SURFACE_QUERY_PROTOCOL_REVISION),
            surface.ok_or(RenderProtocolCompatibilityError::Unsupported {
                protocol: RenderRepresentationProtocol::SurfaceQuery
            })
        );
        assert_eq!(
            candidate.field_distance_protocol(RENDER_FIELD_DISTANCE_PROTOCOL_REVISION),
            field.ok_or(RenderProtocolCompatibilityError::Unsupported {
                protocol: RenderRepresentationProtocol::FieldDistance
            })
        );
        if let Some(surface) = surface {
            assert_eq!(
                candidate.surface_query_protocol(surface.revision() + 1),
                Err(RenderProtocolCompatibilityError::VersionMismatch {
                    protocol: RenderRepresentationProtocol::SurfaceQuery,
                    requested_revision: surface.revision() + 1,
                    supported_revision: surface.revision(),
                })
            );
        }
        if let Some(field) = field {
            assert_eq!(
                candidate.field_distance_protocol(field.revision() + 1),
                Err(RenderProtocolCompatibilityError::VersionMismatch {
                    protocol: RenderRepresentationProtocol::FieldDistance,
                    requested_revision: field.revision() + 1,
                    supported_revision: field.revision(),
                })
            );
        }
        assert_eq!(
            candidate,
            RenderRepresentationRecord::builder(id, coverage.clone(), temporal)
                .refinement(refinement)
                .field_distance(field)
                .surface_query(surface)
                .build()
                .unwrap()
        );
    }
}

#[test]
fn default_refinement_is_absence_and_named_channels_replace_or_clear() {
    let mut store = RenderSceneStore::new();
    let object = store.allocate_object_id().unwrap();
    let id = store.allocate_representation_id(object).unwrap();
    assert_eq!(record(id).refinement(), RenderRefinementEvidence::none());
    assert_eq!(record(id).refinement().finest_absolute_error_meters(), None);
    let field_only = RenderRepresentationRecord::builder(
        id,
        RenderSpatialCoverage::unbounded(),
        RenderTemporalSupport::unbounded(),
    )
    .surface_query(Some(surface()))
    .field_distance(Some(field()))
    .surface_query(None)
    .build()
    .unwrap();
    assert_eq!(
        field_only.field_distance_protocol(RENDER_FIELD_DISTANCE_PROTOCOL_REVISION),
        Ok(field())
    );
    assert_eq!(
        field_only.surface_query_protocol(RENDER_SURFACE_QUERY_PROTOCOL_REVISION),
        Err(RenderProtocolCompatibilityError::Unsupported {
            protocol: RenderRepresentationProtocol::SurfaceQuery
        })
    );
    assert_eq!(
        RenderRepresentationRecord::builder(
            id,
            RenderSpatialCoverage::unbounded(),
            RenderTemporalSupport::unbounded()
        )
        .surface_query(Some(surface()))
        .field_distance(Some(field()))
        .surface_query(None)
        .field_distance(None)
        .build(),
        Err(RenderRepresentationValidationError::NoProtocols)
    );
    assert_eq!(
        RenderSurfaceProtocolEvidence::exact(0),
        Err(RenderRepresentationValidationError::InvalidProtocolRevision)
    );
    assert_eq!(
        RenderFieldDistanceProtocolEvidence::new(0, RenderFieldDistanceGuarantee::exact()),
        Err(RenderRepresentationValidationError::InvalidProtocolRevision)
    );
    assert_eq!(
        RenderRefinementEvidence::bounded(-0.01),
        Err(RenderRepresentationValidationError::NegativeErrorBound)
    );
}

#[test]
fn participation_collection_iterator_and_incremental_assembly_have_identical_canonical_meaning() {
    let mut store = RenderSceneStore::new();
    let object = store.allocate_object_id().unwrap();
    let first = record(store.allocate_representation_id(object).unwrap());
    let second = record(store.allocate_representation_id(object).unwrap());
    let mut accumulated = Vec::new();
    accumulated.push(second.clone());
    accumulated.extend([first.clone()]);
    let from_collection = RenderObjectParticipation::from_representations(accumulated)
        .unwrap()
        .with_material_assignment(Some(material()))
        .with_emitter(Some(emitter()));
    let from_iterator = RenderObjectParticipation::from_representations(
        std::iter::once(first.clone()).chain([second.clone()]),
    )
    .unwrap()
    .with_emitter(Some(emitter()))
    .with_material_assignment(Some(material()));
    assert_eq!(from_collection, from_iterator);
    assert_eq!(
        from_collection.representations(),
        &[first.clone(), second.clone()]
    );
    assert_eq!(from_collection.representation(first.id()), Some(&first));
    assert_eq!(from_collection.representation(second.id()), Some(&second));
    for representations in [
        vec![first.clone(), second.clone(), first.clone()],
        vec![first.clone(), first.clone(), second],
    ] {
        assert_eq!(
            RenderObjectParticipation::from_representations(representations),
            Err(
                RenderParticipationValidationError::DuplicateRepresentationId {
                    representation_id: first.id()
                }
            )
        );
    }
}

#[test]
fn dynamic_optional_material_and_emitter_preserve_empty_and_partial_participation() {
    for (material, emitter) in [
        (None, None),
        (Some(material()), None),
        (None, Some(emitter())),
        (Some(material()), Some(emitter())),
    ] {
        let participation = RenderObjectParticipation::from_representations([])
            .unwrap()
            .with_material_assignment(material)
            .with_emitter(emitter);
        assert_eq!(participation.material_assignment(), material);
        assert_eq!(participation.emitter(), emitter);
        assert!(participation.representations().is_empty());
        assert_eq!(
            participation.is_empty(),
            material.is_none() && emitter.is_none()
        );
        assert_eq!(
            participation
                .clone()
                .with_material_assignment(None)
                .with_emitter(None),
            RenderObjectParticipation::from_representations([]).unwrap()
        );
    }
}

#[test]
fn empty_participation_normalizes_to_clear_with_exact_changes_and_repeated_noop() {
    let mut store = RenderSceneStore::new();
    let object = store.allocate_object_id().unwrap();
    let mut insert = RenderSceneUpdate::new();
    insert.insert(object);
    store.commit(insert).unwrap();
    let representation = record(store.allocate_representation_id(object).unwrap());
    let full = RenderObjectParticipation::from_representations([representation])
        .unwrap()
        .with_material_assignment(Some(material()))
        .with_emitter(Some(emitter()));
    let mut attach = RenderSceneUpdate::new();
    attach.replace_participation(object, full.clone());
    let before = store.commit(attach).unwrap().snapshot().clone();

    let mut empty = RenderSceneUpdate::new();
    empty.replace_participation(
        object,
        RenderObjectParticipation::from_representations([]).unwrap(),
    );
    let mut clear = RenderSceneUpdate::new();
    clear.clear_participation(object);
    assert_eq!(empty, clear);
    let commit = store.commit(empty.clone()).unwrap();
    assert_ne!(commit.revision(), before.revision());
    assert!(commit.snapshot().contains(object));
    assert_eq!(commit.snapshot().object_participation(object), None);
    assert_eq!(before.object_participation(object), Some(&full));
    assert_eq!(
        commit.change_set().representation_changed(),
        Some(&[object][..])
    );
    assert_eq!(
        commit.change_set().material_assignment_changed(),
        Some(&[object][..])
    );
    assert_eq!(commit.change_set().emitter_changed(), Some(&[object][..]));
    assert_eq!(commit.change_set().inserted(), Some(&[][..]));
    assert_eq!(commit.change_set().removed(), Some(&[][..]));
    assert_eq!(commit.change_set().spatial_changed(), Some(&[][..]));
    assert_eq!(commit.change_set().temporal_changed(), Some(&[][..]));
    for update in [empty, clear] {
        let noop = store.commit(update).unwrap();
        assert_eq!(noop.snapshot(), commit.snapshot());
        assert_eq!(noop.revision(), commit.revision());
        assert!(noop.change_set().is_empty_incremental());
    }
}
