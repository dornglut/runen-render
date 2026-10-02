//! R7 renderer-derived scene dependency and invalidation semantics.
//!
//! Derived state is non-authoritative. This module records only which accepted renderer-scene facts
//! one retained derived artifact depends on and whether accepted scene-change evidence invalidates
//! those facts. It deliberately does not own cached payloads, retention/eviction policy, memory
//! budgets, reconstruction recipes, GPU realization, histories, sessions, readback, or presentation.

use super::scene::{RenderObjectId, RenderSceneChangeSet};

/// One exact renderer-scene fact on which retained derived state depends.
///
/// These dependencies use only renderer-owned semantic identity and accepted scene-change families.
/// Source/ECS/product identities and RunenGPU identities remain separate authorities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderDerivedSceneDependency {
    /// Whether this renderer object is present in the semantic scene.
    ObjectPresence(RenderObjectId),
    /// The object's R2 spatial state.
    ObjectSpatialState(RenderObjectId),
    /// The object's R2 temporal state.
    ObjectTemporalState(RenderObjectId),
    /// The object's R3 renderer-visible representation set/evidence.
    ObjectRepresentations(RenderObjectId),
    /// The object's R3 material assignment.
    ObjectMaterialAssignment(RenderObjectId),
    /// The object's R3 emitter semantics.
    ObjectEmitter(RenderObjectId),
}

impl RenderDerivedSceneDependency {
    const fn object_id(self) -> RenderObjectId {
        match self {
            Self::ObjectPresence(object_id)
            | Self::ObjectSpatialState(object_id)
            | Self::ObjectTemporalState(object_id)
            | Self::ObjectRepresentations(object_id)
            | Self::ObjectMaterialAssignment(object_id)
            | Self::ObjectEmitter(object_id) => object_id,
        }
    }

    const fn facet_rank(self) -> u8 {
        match self {
            Self::ObjectPresence(_) => 0,
            Self::ObjectSpatialState(_) => 1,
            Self::ObjectTemporalState(_) => 2,
            Self::ObjectRepresentations(_) => 3,
            Self::ObjectMaterialAssignment(_) => 4,
            Self::ObjectEmitter(_) => 5,
        }
    }

    fn invalidated_by(self, changes: &RenderSceneChangeSet) -> bool {
        let object_id = self.object_id();
        let structural_change = changes
            .inserted()
            .is_some_and(|objects| objects.contains(&object_id))
            || changes
                .removed()
                .is_some_and(|objects| objects.contains(&object_id));
        if structural_change {
            return true;
        }

        match self {
            Self::ObjectPresence(_) => false,
            Self::ObjectSpatialState(_) => changes
                .spatial_changed()
                .is_some_and(|objects| objects.contains(&object_id)),
            Self::ObjectTemporalState(_) => changes
                .temporal_changed()
                .is_some_and(|objects| objects.contains(&object_id)),
            Self::ObjectRepresentations(_) => changes
                .representation_changed()
                .is_some_and(|objects| objects.contains(&object_id)),
            Self::ObjectMaterialAssignment(_) => changes
                .material_assignment_changed()
                .is_some_and(|objects| objects.contains(&object_id)),
            Self::ObjectEmitter(_) => changes
                .emitter_changed()
                .is_some_and(|objects| objects.contains(&object_id)),
        }
    }
}

/// Canonical renderer-scene dependency evidence for one derived artifact.
///
/// Caller order and duplicate declarations are non-semantic: construction sorts and deduplicates the
/// dependency set. No scene revision is stored because `RenderSceneRevision` is not a universal
/// derived-state generation.
///
/// Reuse across multiple scene commits is sound only when the consumer evaluates every accepted
/// `RenderSceneChangeSet` since the artifact was derived. If that incremental evidence is incomplete,
/// lost, or otherwise untrusted, the consumer must conservatively invalidate or consume explicit
/// full-resynchronization evidence rather than infer validity from a later unrelated delta.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RenderDerivedSceneDependencies {
    dependencies: Vec<RenderDerivedSceneDependency>,
}

impl RenderDerivedSceneDependencies {
    pub fn new(dependencies: impl IntoIterator<Item = RenderDerivedSceneDependency>) -> Self {
        let mut dependencies = dependencies.into_iter().collect::<Vec<_>>();
        dependencies
            .sort_unstable_by_key(|dependency| (dependency.object_id(), dependency.facet_rank()));
        dependencies.dedup();
        Self { dependencies }
    }

    /// Returns dependencies in canonical deterministic inspection order.
    ///
    /// The order carries no semantic priority or execution meaning.
    pub fn dependencies(&self) -> &[RenderDerivedSceneDependency] {
        &self.dependencies
    }

    pub fn is_empty(&self) -> bool {
        self.dependencies.is_empty()
    }

    /// Returns whether this one accepted scene-change delta invalidates this dependency set.
    ///
    /// Full resynchronization conservatively invalidates any scene-dependent artifact because precise
    /// incremental evidence is unavailable. An empty dependency set is scene-independent and remains
    /// valid. Incremental invalidation is object/facet-specific and never falls back to comparing a
    /// global scene revision.
    pub fn is_invalidated_by(&self, changes: &RenderSceneChangeSet) -> bool {
        if self.dependencies.is_empty() {
            return false;
        }
        if changes.is_full_resync() {
            return true;
        }
        self.dependencies
            .iter()
            .copied()
            .any(|dependency| dependency.invalidated_by(changes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::render::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
    use crate::plugins::render::participation::{
        RenderMaterialAssignment, RenderObjectParticipation,
    };
    use crate::plugins::render::representation::{
        RENDER_SURFACE_QUERY_PROTOCOL_REVISION, RenderRefinementEvidence,
        RenderRepresentationRecord, RenderSurfaceProtocolEvidence,
    };
    use crate::plugins::render::scene::{
        RenderObjectState, RenderSceneSnapshot, RenderSceneStore, RenderSceneUpdate,
    };
    use crate::plugins::render::space_time::{
        RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState,
        RenderObjectTemporalState, RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport,
        RenderTimeInterval, RenderTimePoint,
    };

    fn object_state(translation_x: f64, validity: RenderTemporalSupport) -> RenderObjectState {
        RenderObjectState::new(
            RenderObjectSpatialState::new(
                RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("proof local space"),
                RenderAffineTransform3::from_row_major_3x4([
                    1.0,
                    0.0,
                    0.0,
                    translation_x,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                ])
                .expect("proof transform"),
                RenderSpatialCoverage::unbounded(),
            ),
            RenderObjectTemporalState::new(validity),
        )
    }

    fn bounded_validity(start: f64, end: f64) -> RenderTemporalSupport {
        RenderTemporalSupport::interval(
            RenderTimeInterval::new(
                RenderTimePoint::from_seconds(start).expect("finite start"),
                RenderTimePoint::from_seconds(end).expect("finite end"),
            )
            .expect("ordered proof interval"),
        )
    }

    fn representation(
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
                    .expect("surface protocol"),
            ),
            None,
        )
        .expect("proof representation")
    }

    fn participation(
        representation: Option<RenderRepresentationRecord>,
        reflectance: Option<f64>,
        irradiance: Option<f64>,
    ) -> RenderObjectParticipation {
        let representations = representation.into_iter().collect();
        let material_assignment = reflectance.map(|value| {
            RenderMaterialAssignment::new(
                RenderDiffuseMaterial::new(value).expect("proof diffuse material"),
            )
        });
        let emitter = irradiance.map(|value| {
            RenderDirectionalEmitter::new([0.0, 1.0, 0.0], 550e-9, value).expect("proof emitter")
        });
        RenderObjectParticipation::new(representations, material_assignment, emitter)
            .expect("proof participation")
    }

    fn dependencies_for_all_facets(object_id: RenderObjectId) -> [RenderDerivedSceneDependency; 6] {
        [
            RenderDerivedSceneDependency::ObjectPresence(object_id),
            RenderDerivedSceneDependency::ObjectSpatialState(object_id),
            RenderDerivedSceneDependency::ObjectTemporalState(object_id),
            RenderDerivedSceneDependency::ObjectRepresentations(object_id),
            RenderDerivedSceneDependency::ObjectMaterialAssignment(object_id),
            RenderDerivedSceneDependency::ObjectEmitter(object_id),
        ]
    }

    fn derived_translation_x(snapshot: &RenderSceneSnapshot, object_id: RenderObjectId) -> f64 {
        snapshot
            .object_state(object_id)
            .expect("derived proof object state")
            .spatial()
            .local_to_scene()
            .row_major_3x4()[3]
    }

    #[test]
    fn dependency_normalization_is_deterministic_and_duplicate_free() {
        let mut store = RenderSceneStore::new();
        let first = store.allocate_object_id().expect("first object id");
        let second = store.allocate_object_id().expect("second object id");
        let first_spatial = RenderDerivedSceneDependency::ObjectSpatialState(first);
        let second_material = RenderDerivedSceneDependency::ObjectMaterialAssignment(second);

        let left = RenderDerivedSceneDependencies::new([
            second_material,
            first_spatial,
            second_material,
            first_spatial,
        ]);
        let right = RenderDerivedSceneDependencies::new([first_spatial, second_material]);

        assert_eq!(left, right);
        assert_eq!(left.dependencies().len(), 2);
    }

    #[test]
    fn structural_change_invalidates_every_dependency_on_the_same_object_only() {
        let mut store = RenderSceneStore::new();
        let target = store.allocate_object_id().expect("target id");
        let other = store.allocate_object_id().expect("other id");
        let target_dependencies = dependencies_for_all_facets(target)
            .map(|dependency| RenderDerivedSceneDependencies::new([dependency]));
        let other_dependencies = RenderDerivedSceneDependencies::new([
            RenderDerivedSceneDependency::ObjectSpatialState(other),
        ]);

        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(
            target,
            object_state(1.0, RenderTemporalSupport::unbounded()),
        );
        let inserted = store.commit(insert).expect("insert target");
        for dependencies in &target_dependencies {
            assert!(dependencies.is_invalidated_by(inserted.change_set()));
        }
        assert!(!other_dependencies.is_invalidated_by(inserted.change_set()));

        let mut insert_other = RenderSceneUpdate::new();
        insert_other
            .insert_with_state(other, object_state(2.0, RenderTemporalSupport::unbounded()));
        let inserted_other = store.commit(insert_other).expect("insert other");
        for dependencies in &target_dependencies {
            assert!(!dependencies.is_invalidated_by(inserted_other.change_set()));
        }
        assert!(other_dependencies.is_invalidated_by(inserted_other.change_set()));

        let mut remove = RenderSceneUpdate::new();
        remove.remove(target);
        let removed = store.commit(remove).expect("remove target");
        for dependencies in &target_dependencies {
            assert!(dependencies.is_invalidated_by(removed.change_set()));
        }
        assert!(!other_dependencies.is_invalidated_by(removed.change_set()));
    }

    #[test]
    fn incremental_facet_changes_invalidate_only_matching_dependencies() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object id");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(
            object_id,
            object_state(1.0, RenderTemporalSupport::unbounded()),
        );
        store.commit(insert).expect("insert object");

        let presence =
            RenderDerivedSceneDependencies::new([RenderDerivedSceneDependency::ObjectPresence(
                object_id,
            )]);
        let spatial = RenderDerivedSceneDependencies::new([
            RenderDerivedSceneDependency::ObjectSpatialState(object_id),
        ]);
        let temporal = RenderDerivedSceneDependencies::new([
            RenderDerivedSceneDependency::ObjectTemporalState(object_id),
        ]);
        let representations = RenderDerivedSceneDependencies::new([
            RenderDerivedSceneDependency::ObjectRepresentations(object_id),
        ]);
        let material = RenderDerivedSceneDependencies::new([
            RenderDerivedSceneDependency::ObjectMaterialAssignment(object_id),
        ]);
        let emitter =
            RenderDerivedSceneDependencies::new([RenderDerivedSceneDependency::ObjectEmitter(
                object_id,
            )]);

        let bounded = bounded_validity(1.0, 2.0);
        let mut temporal_update = RenderSceneUpdate::new();
        temporal_update.replace_state(object_id, object_state(1.0, bounded));
        let temporal_change = store.commit(temporal_update).expect("temporal change");
        assert!(!presence.is_invalidated_by(temporal_change.change_set()));
        assert!(!spatial.is_invalidated_by(temporal_change.change_set()));
        assert!(temporal.is_invalidated_by(temporal_change.change_set()));
        assert!(!representations.is_invalidated_by(temporal_change.change_set()));
        assert!(!material.is_invalidated_by(temporal_change.change_set()));
        assert!(!emitter.is_invalidated_by(temporal_change.change_set()));

        let mut spatial_update = RenderSceneUpdate::new();
        spatial_update.replace_state(object_id, object_state(3.0, bounded));
        let spatial_change = store.commit(spatial_update).expect("spatial change");
        assert!(!presence.is_invalidated_by(spatial_change.change_set()));
        assert!(spatial.is_invalidated_by(spatial_change.change_set()));
        assert!(!temporal.is_invalidated_by(spatial_change.change_set()));
        assert!(!representations.is_invalidated_by(spatial_change.change_set()));
        assert!(!material.is_invalidated_by(spatial_change.change_set()));
        assert!(!emitter.is_invalidated_by(spatial_change.change_set()));

        let representation = representation(&mut store, object_id);
        let mut representation_update = RenderSceneUpdate::new();
        representation_update.replace_participation(
            object_id,
            participation(Some(representation.clone()), None, None),
        );
        let representation_change = store
            .commit(representation_update)
            .expect("representation change");
        assert!(representations.is_invalidated_by(representation_change.change_set()));
        assert!(!material.is_invalidated_by(representation_change.change_set()));
        assert!(!emitter.is_invalidated_by(representation_change.change_set()));

        let mut material_update = RenderSceneUpdate::new();
        material_update.replace_participation(
            object_id,
            participation(Some(representation.clone()), Some(0.5), None),
        );
        let material_change = store.commit(material_update).expect("material change");
        assert!(!representations.is_invalidated_by(material_change.change_set()));
        assert!(material.is_invalidated_by(material_change.change_set()));
        assert!(!emitter.is_invalidated_by(material_change.change_set()));

        let mut emitter_update = RenderSceneUpdate::new();
        emitter_update.replace_participation(
            object_id,
            participation(Some(representation), Some(0.5), Some(2.0)),
        );
        let emitter_change = store.commit(emitter_update).expect("emitter change");
        assert!(!representations.is_invalidated_by(emitter_change.change_set()));
        assert!(!material.is_invalidated_by(emitter_change.change_set()));
        assert!(emitter.is_invalidated_by(emitter_change.change_set()));
    }

    #[test]
    fn full_resync_invalidates_scene_dependent_state_but_not_scene_independent_state() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object id");
        let dependent =
            RenderDerivedSceneDependencies::new([RenderDerivedSceneDependency::ObjectPresence(
                object_id,
            )]);
        let independent = RenderDerivedSceneDependencies::default();
        let resync = store.full_resync();

        assert!(dependent.is_invalidated_by(resync.change_set()));
        assert!(!independent.is_invalidated_by(resync.change_set()));
    }

    #[test]
    fn unrelated_revision_advance_preserves_cache_hit_semantics_and_relevant_change_invalidates() {
        let mut store = RenderSceneStore::new();
        let target = store.allocate_object_id().expect("target id");
        let unrelated = store.allocate_object_id().expect("unrelated id");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(
            target,
            object_state(1.0, RenderTemporalSupport::unbounded()),
        );
        insert.insert_with_state(
            unrelated,
            object_state(10.0, RenderTemporalSupport::unbounded()),
        );
        let initial = store.commit(insert).expect("initial scene");
        let cached_value = derived_translation_x(initial.snapshot(), target);
        let cached_dependencies = RenderDerivedSceneDependencies::new([
            RenderDerivedSceneDependency::ObjectPresence(target),
            RenderDerivedSceneDependency::ObjectSpatialState(target),
        ]);
        let initial_revision = initial.revision();

        let mut unrelated_update = RenderSceneUpdate::new();
        unrelated_update.replace_state(
            unrelated,
            object_state(20.0, RenderTemporalSupport::unbounded()),
        );
        let unrelated_commit = store
            .commit(unrelated_update)
            .expect("unrelated spatial change");
        assert_ne!(unrelated_commit.revision(), initial_revision);
        assert!(!cached_dependencies.is_invalidated_by(unrelated_commit.change_set()));
        assert_eq!(
            cached_value,
            derived_translation_x(unrelated_commit.snapshot(), target),
            "cache hit must preserve the same renderer-semantic value as clean recomputation"
        );

        let mut target_update = RenderSceneUpdate::new();
        target_update.replace_state(
            target,
            object_state(4.0, RenderTemporalSupport::unbounded()),
        );
        let target_commit = store.commit(target_update).expect("target spatial change");
        assert!(cached_dependencies.is_invalidated_by(target_commit.change_set()));
        let recomputed = derived_translation_x(target_commit.snapshot(), target);
        assert_ne!(cached_value, recomputed);
        assert_eq!(recomputed, 4.0);
    }
}
