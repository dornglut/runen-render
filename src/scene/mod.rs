use super::participation::RenderObjectParticipation;
use super::representation::{RenderRepresentationId, classify_field_distance_transform};
use super::space_time::{RenderObjectSpatialState, RenderObjectTemporalState};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::num::NonZeroU64;
use std::sync::Arc;

mod storage;
use storage::SceneObjects;
#[cfg(test)]
use storage::{RADIX_DEPTH, SceneStorageContinuity};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderObjectId(NonZeroU64);

impl RenderObjectId {
    fn from_raw(raw: u64) -> Option<Self> {
        NonZeroU64::new(raw).map(Self)
    }

    const fn raw(self) -> u64 {
        self.0.get()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct RenderSceneRevision(u64);

impl RenderSceneRevision {
    pub const INITIAL: Self = Self(0);

    fn checked_next(self) -> Option<Self> {
        self.0.checked_add(1).map(Self)
    }
}

/// The concrete R2-owned semantic state of one renderer object.
///
/// Presence-only R1 objects remain valid and therefore have no `RenderObjectState`. This record
/// begins same-identity replacement only for the spatial and temporal semantics R2 actually owns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderObjectState {
    spatial: RenderObjectSpatialState,
    temporal: RenderObjectTemporalState,
}

impl RenderObjectState {
    pub fn new(spatial: RenderObjectSpatialState, temporal: RenderObjectTemporalState) -> Self {
        Self { spatial, temporal }
    }

    pub const fn spatial(&self) -> &RenderObjectSpatialState {
        &self.spatial
    }

    pub const fn temporal(&self) -> &RenderObjectTemporalState {
        &self.temporal
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RenderSceneOperationKind {
    Insert {
        state: Option<RenderObjectState>,
    },
    Remove,
    ReplaceState {
        state: RenderObjectState,
    },
    ReplaceParticipation {
        participation: Option<RenderObjectParticipation>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RenderSceneOperation {
    object_id: RenderObjectId,
    kind: RenderSceneOperationKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RenderSceneStructuralMutation {
    Insert { state: Option<RenderObjectState> },
    Remove,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RenderSceneObjectMutation {
    structural: Option<RenderSceneStructuralMutation>,
    state: Option<RenderObjectState>,
    participation: Option<Option<RenderObjectParticipation>>,
}

impl RenderSceneObjectMutation {
    fn absorb(&mut self, kind: &RenderSceneOperationKind) -> bool {
        match kind {
            RenderSceneOperationKind::Insert { state } => {
                if self.structural.is_some() || self.state.is_some() || self.participation.is_some()
                {
                    return false;
                }
                self.structural = Some(RenderSceneStructuralMutation::Insert {
                    state: state.clone(),
                });
            }
            RenderSceneOperationKind::Remove => {
                if self.structural.is_some() || self.state.is_some() || self.participation.is_some()
                {
                    return false;
                }
                self.structural = Some(RenderSceneStructuralMutation::Remove);
            }
            RenderSceneOperationKind::ReplaceState { state } => {
                if self.structural.is_some() || self.state.is_some() {
                    return false;
                }
                self.state = Some(state.clone());
            }
            RenderSceneOperationKind::ReplaceParticipation { participation } => {
                if self.structural.is_some() || self.participation.is_some() {
                    return false;
                }
                self.participation = Some(participation.clone());
            }
        }
        true
    }
}

/// One atomic renderer-scene transaction.
///
/// Replacements of independent facets on the same existing object may be combined in one update;
/// in particular, state and participation are validated against their proposed final combination.
/// Structural insertion/removal remains exclusive with any other same-object operation, and
/// duplicate replacement of the same facet is conflicting.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RenderSceneUpdate {
    operations: Vec<RenderSceneOperation>,
}

impl RenderSceneUpdate {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, object_id: RenderObjectId) -> &mut Self {
        self.operations.push(RenderSceneOperation {
            object_id,
            kind: RenderSceneOperationKind::Insert { state: None },
        });
        self
    }

    pub fn insert_with_state(
        &mut self,
        object_id: RenderObjectId,
        state: RenderObjectState,
    ) -> &mut Self {
        self.operations.push(RenderSceneOperation {
            object_id,
            kind: RenderSceneOperationKind::Insert { state: Some(state) },
        });
        self
    }

    pub fn replace_state(
        &mut self,
        object_id: RenderObjectId,
        state: RenderObjectState,
    ) -> &mut Self {
        self.operations.push(RenderSceneOperation {
            object_id,
            kind: RenderSceneOperationKind::ReplaceState { state },
        });
        self
    }

    pub fn replace_participation(
        &mut self,
        object_id: RenderObjectId,
        participation: RenderObjectParticipation,
    ) -> &mut Self {
        let participation = (!participation.is_empty()).then_some(participation);
        self.operations.push(RenderSceneOperation {
            object_id,
            kind: RenderSceneOperationKind::ReplaceParticipation { participation },
        });
        self
    }

    pub fn clear_participation(&mut self, object_id: RenderObjectId) -> &mut Self {
        self.operations.push(RenderSceneOperation {
            object_id,
            kind: RenderSceneOperationKind::ReplaceParticipation {
                participation: None,
            },
        });
        self
    }

    pub fn remove(&mut self, object_id: RenderObjectId) -> &mut Self {
        self.operations.push(RenderSceneOperation {
            object_id,
            kind: RenderSceneOperationKind::Remove,
        });
        self
    }

    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    pub fn len(&self) -> usize {
        self.operations.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderSceneChangeSet {
    kind: RenderSceneChangeKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum RenderSceneChangeKind {
    Incremental {
        inserted: Arc<[RenderObjectId]>,
        removed: Arc<[RenderObjectId]>,
        spatial_changed: Arc<[RenderObjectId]>,
        temporal_changed: Arc<[RenderObjectId]>,
        representation_changed: Arc<[RenderObjectId]>,
        material_assignment_changed: Arc<[RenderObjectId]>,
        emitter_changed: Arc<[RenderObjectId]>,
    },
    FullResync,
}

impl RenderSceneChangeSet {
    fn incremental(
        inserted: Vec<RenderObjectId>,
        removed: Vec<RenderObjectId>,
        spatial_changed: Vec<RenderObjectId>,
        temporal_changed: Vec<RenderObjectId>,
        representation_changed: Vec<RenderObjectId>,
        material_assignment_changed: Vec<RenderObjectId>,
        emitter_changed: Vec<RenderObjectId>,
    ) -> Self {
        Self {
            kind: RenderSceneChangeKind::Incremental {
                inserted: Arc::from(inserted),
                removed: Arc::from(removed),
                spatial_changed: Arc::from(spatial_changed),
                temporal_changed: Arc::from(temporal_changed),
                representation_changed: Arc::from(representation_changed),
                material_assignment_changed: Arc::from(material_assignment_changed),
                emitter_changed: Arc::from(emitter_changed),
            },
        }
    }

    fn full_resync() -> Self {
        Self {
            kind: RenderSceneChangeKind::FullResync,
        }
    }

    pub fn inserted(&self) -> Option<&[RenderObjectId]> {
        match &self.kind {
            RenderSceneChangeKind::Incremental { inserted, .. } => Some(inserted),
            RenderSceneChangeKind::FullResync => None,
        }
    }

    pub fn removed(&self) -> Option<&[RenderObjectId]> {
        match &self.kind {
            RenderSceneChangeKind::Incremental { removed, .. } => Some(removed),
            RenderSceneChangeKind::FullResync => None,
        }
    }

    /// Objects whose existing R2 spatial state changed through same-identity replacement.
    /// Stateful insertions are reported by `inserted()` rather than duplicated here.
    pub fn spatial_changed(&self) -> Option<&[RenderObjectId]> {
        match &self.kind {
            RenderSceneChangeKind::Incremental {
                spatial_changed, ..
            } => Some(spatial_changed),
            RenderSceneChangeKind::FullResync => None,
        }
    }

    /// Objects whose existing R2 temporal state changed through same-identity replacement.
    /// Stateful insertions are reported by `inserted()` rather than duplicated here.
    pub fn temporal_changed(&self) -> Option<&[RenderObjectId]> {
        match &self.kind {
            RenderSceneChangeKind::Incremental {
                temporal_changed, ..
            } => Some(temporal_changed),
            RenderSceneChangeKind::FullResync => None,
        }
    }

    pub fn representation_changed(&self) -> Option<&[RenderObjectId]> {
        match &self.kind {
            RenderSceneChangeKind::Incremental {
                representation_changed,
                ..
            } => Some(representation_changed),
            RenderSceneChangeKind::FullResync => None,
        }
    }

    pub fn material_assignment_changed(&self) -> Option<&[RenderObjectId]> {
        match &self.kind {
            RenderSceneChangeKind::Incremental {
                material_assignment_changed,
                ..
            } => Some(material_assignment_changed),
            RenderSceneChangeKind::FullResync => None,
        }
    }

    pub fn emitter_changed(&self) -> Option<&[RenderObjectId]> {
        match &self.kind {
            RenderSceneChangeKind::Incremental {
                emitter_changed, ..
            } => Some(emitter_changed),
            RenderSceneChangeKind::FullResync => None,
        }
    }

    pub fn is_full_resync(&self) -> bool {
        matches!(self.kind, RenderSceneChangeKind::FullResync)
    }

    pub fn is_empty_incremental(&self) -> bool {
        matches!(
            &self.kind,
            RenderSceneChangeKind::Incremental {
                inserted,
                removed,
                spatial_changed,
                temporal_changed,
                representation_changed,
                material_assignment_changed,
                emitter_changed,
            } if inserted.is_empty()
                && removed.is_empty()
                && spatial_changed.is_empty()
                && temporal_changed.is_empty()
                && representation_changed.is_empty()
                && material_assignment_changed.is_empty()
                && emitter_changed.is_empty()
        )
    }
}

/// Opaque renderer-local continuity evidence for one immutable scene position.
///
/// The weak root identity is provenance only: it does not retain old scene contents and is not part
/// of public semantic snapshot equality or a persisted/global scene identity.
#[cfg(test)]
#[derive(Debug, Clone)]
pub(crate) struct RenderSceneContinuity {
    revision: RenderSceneRevision,
    position: SceneStorageContinuity,
}

#[cfg(test)]
impl RenderSceneContinuity {
    pub(crate) const fn revision(&self) -> RenderSceneRevision {
        self.revision
    }

    fn same_position(&self, other: &Self) -> bool {
        self.revision == other.revision && self.position.same_position(&other.position)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderSceneSnapshot {
    revision: RenderSceneRevision,
    objects: SceneObjects,
}

impl RenderSceneSnapshot {
    pub const fn revision(&self) -> RenderSceneRevision {
        self.revision
    }

    pub fn len(&self) -> usize {
        self.objects.len()
    }

    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    pub fn contains(&self, object_id: RenderObjectId) -> bool {
        self.objects.contains(object_id)
    }

    pub fn object_state(&self, object_id: RenderObjectId) -> Option<&RenderObjectState> {
        self.objects.object_state(object_id)
    }

    pub fn object_participation(
        &self,
        object_id: RenderObjectId,
    ) -> Option<&RenderObjectParticipation> {
        self.objects.object_participation(object_id)
    }

    pub fn object_ids(&self) -> Vec<RenderObjectId> {
        self.objects.object_ids()
    }

    #[cfg(test)]
    pub(crate) fn continuity(&self) -> RenderSceneContinuity {
        RenderSceneContinuity {
            revision: self.revision,
            position: self.objects.continuity(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RenderSceneCommit {
    #[cfg(test)]
    previous: RenderSceneContinuity,
    snapshot: RenderSceneSnapshot,
    change_set: RenderSceneChangeSet,
}

impl PartialEq for RenderSceneCommit {
    fn eq(&self, other: &Self) -> bool {
        self.snapshot == other.snapshot && self.change_set == other.change_set
    }
}

impl Eq for RenderSceneCommit {}

impl RenderSceneCommit {
    pub const fn revision(&self) -> RenderSceneRevision {
        self.snapshot.revision()
    }

    pub const fn snapshot(&self) -> &RenderSceneSnapshot {
        &self.snapshot
    }

    pub const fn change_set(&self) -> &RenderSceneChangeSet {
        &self.change_set
    }

    #[cfg(test)]
    pub(crate) const fn previous_revision(&self) -> RenderSceneRevision {
        self.previous.revision()
    }

    #[cfg(test)]
    pub(crate) fn directly_follows(&self, previous: &RenderSceneContinuity) -> bool {
        self.previous.same_position(previous)
    }

    #[cfg(test)]
    pub(crate) fn continuity(&self) -> RenderSceneContinuity {
        self.snapshot.continuity()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderSceneResync {
    snapshot: RenderSceneSnapshot,
    change_set: RenderSceneChangeSet,
}

impl RenderSceneResync {
    pub const fn snapshot(&self) -> &RenderSceneSnapshot {
        &self.snapshot
    }

    pub const fn change_set(&self) -> &RenderSceneChangeSet {
        &self.change_set
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderObjectIdAllocationError {
    Exhausted,
}

impl fmt::Display for RenderObjectIdAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted => write!(f, "RenderObjectId allocator exhausted"),
        }
    }
}

impl Error for RenderObjectIdAllocationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderRepresentationIdAllocationError {
    Exhausted,
}

impl fmt::Display for RenderRepresentationIdAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exhausted => write!(f, "RenderRepresentationId allocator exhausted"),
        }
    }
}

impl Error for RenderRepresentationIdAllocationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderSceneCommitError {
    ConflictingOperations {
        object_id: RenderObjectId,
    },
    ObjectAlreadyPresent {
        object_id: RenderObjectId,
    },
    ObjectMissing {
        object_id: RenderObjectId,
    },
    UnknownRepresentationId {
        representation_id: RenderRepresentationId,
    },
    RepresentationOwnerMismatch {
        representation_id: RenderRepresentationId,
        allocated_owner: RenderObjectId,
        requested_owner: RenderObjectId,
    },
    FieldRepresentationRequiresSpatialState {
        object_id: RenderObjectId,
        representation_id: RenderRepresentationId,
    },
    InvalidFieldTransform {
        object_id: RenderObjectId,
        representation_id: RenderRepresentationId,
    },
    RevisionExhausted,
}

impl fmt::Display for RenderSceneCommitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConflictingOperations { object_id } => write!(
                f,
                "RenderSceneUpdate contains conflicting operations for {object_id:?}"
            ),
            Self::ObjectAlreadyPresent { object_id } => {
                write!(f, "RenderObjectId {object_id:?} is already present")
            }
            Self::ObjectMissing { object_id } => {
                write!(f, "RenderObjectId {object_id:?} is not present")
            }
            Self::UnknownRepresentationId { representation_id } => write!(
                f,
                "RenderRepresentationId {representation_id:?} was not allocated by this scene store"
            ),
            Self::RepresentationOwnerMismatch {
                representation_id,
                allocated_owner,
                requested_owner,
            } => write!(
                f,
                "RenderRepresentationId {representation_id:?} belongs to {allocated_owner:?}, not {requested_owner:?}"
            ),
            Self::FieldRepresentationRequiresSpatialState {
                object_id,
                representation_id,
            } => write!(
                f,
                "field representation {representation_id:?} on {object_id:?} requires R2 spatial state"
            ),
            Self::InvalidFieldTransform {
                object_id,
                representation_id,
            } => write!(
                f,
                "field representation {representation_id:?} on {object_id:?} has invalid scene transform semantics"
            ),
            Self::RevisionExhausted => write!(f, "RenderSceneRevision exhausted"),
        }
    }
}

impl Error for RenderSceneCommitError {}

#[derive(Debug)]
pub struct RenderSceneStore {
    revision: RenderSceneRevision,
    objects: SceneObjects,
    next_object_raw: u64,
    next_representation_raw: u64,
    representation_owners: BTreeMap<RenderRepresentationId, RenderObjectId>,
}

impl Default for RenderSceneStore {
    fn default() -> Self {
        Self {
            revision: RenderSceneRevision::INITIAL,
            objects: SceneObjects::default(),
            next_object_raw: 1,
            next_representation_raw: 1,
            representation_owners: BTreeMap::new(),
        }
    }
}

impl RenderSceneStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn revision(&self) -> RenderSceneRevision {
        self.revision
    }

    pub fn snapshot(&self) -> RenderSceneSnapshot {
        RenderSceneSnapshot {
            revision: self.revision,
            objects: self.objects.clone(),
        }
    }

    pub fn allocate_object_id(&mut self) -> Result<RenderObjectId, RenderObjectIdAllocationError> {
        if self.next_object_raw == u64::MAX {
            return Err(RenderObjectIdAllocationError::Exhausted);
        }

        let raw = self.next_object_raw;
        self.next_object_raw += 1;
        Ok(RenderObjectId::from_raw(raw).expect("renderer allocator never issues zero"))
    }

    pub fn allocate_representation_id(
        &mut self,
        owner: RenderObjectId,
    ) -> Result<RenderRepresentationId, RenderRepresentationIdAllocationError> {
        if self.next_representation_raw == u64::MAX {
            return Err(RenderRepresentationIdAllocationError::Exhausted);
        }
        let raw = self.next_representation_raw;
        self.next_representation_raw += 1;
        let representation_id = RenderRepresentationId::from_raw(raw)
            .expect("renderer representation allocator never issues zero");
        self.representation_owners.insert(representation_id, owner);
        Ok(representation_id)
    }

    pub fn commit(
        &mut self,
        update: RenderSceneUpdate,
    ) -> Result<RenderSceneCommit, RenderSceneCommitError> {
        let validated = self.validate_update(&update)?;
        if validated.is_noop() {
            #[cfg(test)]
            let previous = self.snapshot().continuity();
            return Ok(RenderSceneCommit {
                #[cfg(test)]
                previous,
                snapshot: self.snapshot(),
                change_set: RenderSceneChangeSet::incremental(
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                ),
            });
        }

        let next_revision = self
            .revision
            .checked_next()
            .ok_or(RenderSceneCommitError::RevisionExhausted)?;
        #[cfg(test)]
        let previous = self.snapshot().continuity();

        let ValidatedRenderSceneUpdate {
            inserted,
            removed,
            replaced,
            spatial_changed,
            temporal_changed,
            representation_changed,
            material_assignment_changed,
            emitter_changed,
        } = validated;
        let inserted_ids = inserted.iter().map(|(object_id, _)| *object_id).collect();

        let mut next_objects = self.objects.clone();
        for (object_id, state) in inserted {
            next_objects = next_objects.inserted(object_id, state).0;
        }
        for replacement in replaced {
            next_objects = next_objects
                .replaced_facets(
                    replacement.object_id,
                    replacement.state,
                    replacement.participation,
                )
                .0;
        }
        for object_id in &removed {
            next_objects = next_objects.removed(*object_id).0;
        }

        self.objects = next_objects;
        self.revision = next_revision;

        Ok(RenderSceneCommit {
            #[cfg(test)]
            previous,
            snapshot: self.snapshot(),
            change_set: RenderSceneChangeSet::incremental(
                inserted_ids,
                removed,
                spatial_changed,
                temporal_changed,
                representation_changed,
                material_assignment_changed,
                emitter_changed,
            ),
        })
    }

    pub fn full_resync(&self) -> RenderSceneResync {
        RenderSceneResync {
            snapshot: self.snapshot(),
            change_set: RenderSceneChangeSet::full_resync(),
        }
    }

    fn validate_update(
        &self,
        update: &RenderSceneUpdate,
    ) -> Result<ValidatedRenderSceneUpdate, RenderSceneCommitError> {
        let mut normalized = BTreeMap::<RenderObjectId, RenderSceneObjectMutation>::new();
        let mut conflicts = BTreeSet::<RenderObjectId>::new();

        for operation in &update.operations {
            let mutation = normalized.entry(operation.object_id).or_default();
            if !mutation.absorb(&operation.kind) {
                conflicts.insert(operation.object_id);
            }
        }

        if let Some(object_id) = conflicts.first().copied() {
            return Err(RenderSceneCommitError::ConflictingOperations { object_id });
        }

        let mut inserted = Vec::new();
        let mut removed = Vec::new();
        let mut replaced = Vec::new();
        let mut spatial_changed = Vec::new();
        let mut temporal_changed = Vec::new();
        let mut representation_changed = Vec::new();
        let mut material_assignment_changed = Vec::new();
        let mut emitter_changed = Vec::new();

        for (object_id, mutation) in normalized {
            if let Some(structural) = mutation.structural {
                match structural {
                    RenderSceneStructuralMutation::Insert { state } => {
                        if self.objects.contains(object_id) {
                            return Err(RenderSceneCommitError::ObjectAlreadyPresent { object_id });
                        }
                        inserted.push((object_id, state));
                    }
                    RenderSceneStructuralMutation::Remove => {
                        if !self.objects.contains(object_id) {
                            return Err(RenderSceneCommitError::ObjectMissing { object_id });
                        }
                        removed.push(object_id);
                    }
                }
                continue;
            }

            if !self.objects.contains(object_id) {
                return Err(RenderSceneCommitError::ObjectMissing { object_id });
            }

            let current_state = self.objects.object_state(object_id);
            let current_participation = self.objects.object_participation(object_id);
            let proposed_state = mutation.state.as_ref().or(current_state);
            let proposed_participation = match mutation.participation.as_ref() {
                Some(participation) => participation.as_ref(),
                None => current_participation,
            };

            if let Some(Some(participation)) = mutation.participation.as_ref() {
                self.validate_representation_ownership(object_id, participation)?;
            }
            self.validate_field_participation(object_id, proposed_state, proposed_participation)?;

            let mut state_replacement = None;
            if let Some(state) = mutation.state {
                let spatial_differs =
                    current_state.is_none_or(|current| current.spatial() != state.spatial());
                let temporal_differs =
                    current_state.is_none_or(|current| current.temporal() != state.temporal());
                if spatial_differs || temporal_differs {
                    if spatial_differs {
                        spatial_changed.push(object_id);
                    }
                    if temporal_differs {
                        temporal_changed.push(object_id);
                    }
                    state_replacement = Some(state);
                }
            }

            let mut participation_replacement = None;
            if let Some(participation) = mutation.participation {
                let next = participation.as_ref();
                let current_representations = current_participation
                    .map(RenderObjectParticipation::representations)
                    .unwrap_or(&[]);
                let next_representations = next
                    .map(RenderObjectParticipation::representations)
                    .unwrap_or(&[]);
                let representations_differ = current_representations != next_representations;
                let material_differs = current_participation
                    .and_then(RenderObjectParticipation::material_assignment)
                    != next.and_then(RenderObjectParticipation::material_assignment);
                let emitter_differs = current_participation
                    .and_then(RenderObjectParticipation::emitter)
                    != next.and_then(RenderObjectParticipation::emitter);

                if representations_differ || material_differs || emitter_differs {
                    if representations_differ {
                        representation_changed.push(object_id);
                    }
                    if material_differs {
                        material_assignment_changed.push(object_id);
                    }
                    if emitter_differs {
                        emitter_changed.push(object_id);
                    }
                    participation_replacement = Some(participation);
                }
            }

            if state_replacement.is_some() || participation_replacement.is_some() {
                replaced.push(ValidatedRenderSceneObjectReplacement {
                    object_id,
                    state: state_replacement,
                    participation: participation_replacement,
                });
            }
        }

        Ok(ValidatedRenderSceneUpdate {
            inserted,
            removed,
            replaced,
            spatial_changed,
            temporal_changed,
            representation_changed,
            material_assignment_changed,
            emitter_changed,
        })
    }

    fn validate_representation_ownership(
        &self,
        object_id: RenderObjectId,
        participation: &RenderObjectParticipation,
    ) -> Result<(), RenderSceneCommitError> {
        for representation in participation.representations() {
            let representation_id = representation.id();
            let Some(allocated_owner) = self.representation_owners.get(&representation_id).copied()
            else {
                return Err(RenderSceneCommitError::UnknownRepresentationId { representation_id });
            };
            if allocated_owner != object_id {
                return Err(RenderSceneCommitError::RepresentationOwnerMismatch {
                    representation_id,
                    allocated_owner,
                    requested_owner: object_id,
                });
            }
        }
        Ok(())
    }

    fn validate_field_participation(
        &self,
        object_id: RenderObjectId,
        state: Option<&RenderObjectState>,
        participation: Option<&RenderObjectParticipation>,
    ) -> Result<(), RenderSceneCommitError> {
        let Some(participation) = participation else {
            return Ok(());
        };
        for representation in participation.representations() {
            if !representation.supports_field_distance() {
                continue;
            }
            let Some(state) = state else {
                return Err(
                    RenderSceneCommitError::FieldRepresentationRequiresSpatialState {
                        object_id,
                        representation_id: representation.id(),
                    },
                );
            };
            let classification =
                classify_field_distance_transform(state.spatial().local_to_scene());
            if classification.is_invalid() {
                return Err(RenderSceneCommitError::InvalidFieldTransform {
                    object_id,
                    representation_id: representation.id(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
struct ValidatedRenderSceneObjectReplacement {
    object_id: RenderObjectId,
    state: Option<RenderObjectState>,
    participation: Option<Option<RenderObjectParticipation>>,
}

#[derive(Debug)]
struct ValidatedRenderSceneUpdate {
    inserted: Vec<(RenderObjectId, Option<RenderObjectState>)>,
    removed: Vec<RenderObjectId>,
    replaced: Vec<ValidatedRenderSceneObjectReplacement>,
    spatial_changed: Vec<RenderObjectId>,
    temporal_changed: Vec<RenderObjectId>,
    representation_changed: Vec<RenderObjectId>,
    material_assignment_changed: Vec<RenderObjectId>,
    emitter_changed: Vec<RenderObjectId>,
}

impl ValidatedRenderSceneUpdate {
    fn is_noop(&self) -> bool {
        self.inserted.is_empty() && self.removed.is_empty() && self.replaced.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
    use crate::participation::{RenderMaterialAssignment, RenderObjectParticipation};
    use crate::representation::{
        RENDER_FIELD_DISTANCE_PROTOCOL_REVISION, RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
        RenderFieldDistanceGuarantee, RenderFieldDistanceProtocolEvidence,
        RenderRefinementEvidence, RenderRepresentationRecord, RenderSurfaceProtocolEvidence,
    };
    use crate::space_time::{
        RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState,
        RenderObjectTemporalState, RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport,
        RenderTimeInterval, RenderTimePoint,
    };

    fn insert_one(store: &mut RenderSceneStore, object_id: RenderObjectId) -> RenderSceneCommit {
        let mut update = RenderSceneUpdate::new();
        update.insert(object_id);
        store.commit(update).expect("single insert should commit")
    }

    fn object_state(translation_x: f64, temporal_end: f64) -> RenderObjectState {
        object_state_with_transform(
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
            .expect("finite transform"),
            translation_x,
            temporal_end,
        )
    }

    fn object_state_with_transform(
        transform: RenderAffineTransform3,
        translation_x: f64,
        temporal_end: f64,
    ) -> RenderObjectState {
        let spatial = RenderObjectSpatialState::new(
            RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("valid space"),
            transform,
            RenderSpatialCoverage::axis_aligned_bounds(
                [translation_x - 1.0, -1.0, -1.0],
                [translation_x + 1.0, 1.0, 1.0],
            )
            .expect("valid coverage"),
        );
        let validity = RenderTimeInterval::new(
            RenderTimePoint::from_seconds(0.0).expect("finite time"),
            RenderTimePoint::from_seconds(temporal_end).expect("finite time"),
        )
        .expect("ordered interval");
        let temporal = RenderObjectTemporalState::new(RenderTemporalSupport::interval(validity));
        RenderObjectState::new(spatial, temporal)
    }

    fn surface_representation(
        store: &mut RenderSceneStore,
        object_id: RenderObjectId,
    ) -> RenderRepresentationRecord {
        let representation_id = store
            .allocate_representation_id(object_id)
            .expect("representation ID should allocate");
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

    fn field_representation(
        store: &mut RenderSceneStore,
        object_id: RenderObjectId,
    ) -> RenderRepresentationRecord {
        let representation_id = store
            .allocate_representation_id(object_id)
            .expect("representation ID should allocate");
        RenderRepresentationRecord::builder(
            representation_id,
            RenderSpatialCoverage::unbounded(),
            RenderTemporalSupport::unbounded(),
        )
        .refinement(RenderRefinementEvidence::bounded(0.01).expect("valid refinement"))
        .field_distance(Some(
            RenderFieldDistanceProtocolEvidence::new(
                RENDER_FIELD_DISTANCE_PROTOCOL_REVISION,
                RenderFieldDistanceGuarantee::conservative(0.01).expect("valid field guarantee"),
            )
            .expect("valid field protocol"),
        ))
        .build()
        .expect("valid field representation")
    }

    #[test]
    fn empty_scene_has_defined_initial_revision() {
        let store = RenderSceneStore::new();
        let snapshot = store.snapshot();
        assert_eq!(snapshot.revision(), RenderSceneRevision::INITIAL);
        assert!(snapshot.is_empty());
        assert_eq!(snapshot.object_ids(), Vec::<RenderObjectId>::new());
    }

    #[test]
    fn allocation_is_monotonic_non_reusing_and_does_not_advance_scene_revision() {
        let mut store = RenderSceneStore::new();
        let first = store
            .allocate_object_id()
            .expect("first ID should allocate");
        let second = store
            .allocate_object_id()
            .expect("second ID should allocate");
        assert_eq!(first.raw(), 1);
        assert_eq!(second.raw(), 2);
        assert_eq!(store.revision(), RenderSceneRevision::INITIAL);

        insert_one(&mut store, first);
        let mut remove = RenderSceneUpdate::new();
        remove.remove(first);
        store.commit(remove).expect("remove should commit");
        let third = store
            .allocate_object_id()
            .expect("third ID should allocate");
        assert_eq!(third.raw(), 3);
        assert_ne!(third, first);
    }

    #[test]
    fn insert_and_remove_publish_precise_structural_changes() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("ID should allocate");
        let insert = insert_one(&mut store, object_id);
        assert_eq!(insert.revision(), RenderSceneRevision(1));
        assert_eq!(insert.previous_revision(), RenderSceneRevision::INITIAL);
        assert!(insert.snapshot().contains(object_id));
        assert_eq!(insert.change_set().inserted(), Some(&[object_id][..]));
        assert_eq!(insert.change_set().removed(), Some(&[][..]));
        assert_eq!(insert.change_set().spatial_changed(), Some(&[][..]));
        assert_eq!(insert.change_set().temporal_changed(), Some(&[][..]));
        assert_eq!(insert.change_set().representation_changed(), Some(&[][..]));
        assert_eq!(
            insert.change_set().material_assignment_changed(),
            Some(&[][..])
        );
        assert_eq!(insert.change_set().emitter_changed(), Some(&[][..]));

        let mut remove_update = RenderSceneUpdate::new();
        remove_update.remove(object_id);
        let remove = store.commit(remove_update).expect("remove should commit");
        assert_eq!(remove.previous_revision(), RenderSceneRevision(1));
        assert_eq!(remove.revision(), RenderSceneRevision(2));
        assert!(!remove.snapshot().contains(object_id));
        assert_eq!(remove.change_set().removed(), Some(&[object_id][..]));
    }

    #[test]
    fn duplicate_insert_and_duplicate_same_kind_reject_without_publication() {
        let mut store = RenderSceneStore::new();
        let present = store.allocate_object_id().expect("ID should allocate");
        let absent = store.allocate_object_id().expect("ID should allocate");
        insert_one(&mut store, present);
        let before = store.snapshot();

        let mut update = RenderSceneUpdate::new();
        update.insert(absent).insert(present);
        assert_eq!(
            store.commit(update),
            Err(RenderSceneCommitError::ObjectAlreadyPresent { object_id: present })
        );
        assert_eq!(store.snapshot(), before);

        let mut duplicate = RenderSceneUpdate::new();
        duplicate.insert(absent).insert(absent);
        assert_eq!(
            store.commit(duplicate),
            Err(RenderSceneCommitError::ConflictingOperations { object_id: absent })
        );
        assert_eq!(store.snapshot(), before);
    }

    #[test]
    fn missing_remove_and_same_object_mixed_operation_reject_atomically() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("ID should allocate");
        let before = store.snapshot();
        let mut missing_remove = RenderSceneUpdate::new();
        missing_remove.remove(object_id);
        assert_eq!(
            store.commit(missing_remove),
            Err(RenderSceneCommitError::ObjectMissing { object_id })
        );
        assert_eq!(store.snapshot(), before);

        let mut mixed = RenderSceneUpdate::new();
        mixed.insert(object_id).remove(object_id);
        assert_eq!(
            store.commit(mixed),
            Err(RenderSceneCommitError::ConflictingOperations { object_id })
        );
        assert_eq!(store.snapshot(), before);
    }

    #[test]
    fn empty_update_is_accepted_no_op() {
        let mut store = RenderSceneStore::new();
        let before = store.snapshot();
        let continuity = before.continuity();
        let commit = store
            .commit(RenderSceneUpdate::new())
            .expect("empty update should be accepted");
        assert_eq!(commit.snapshot(), &before);
        assert_eq!(commit.previous_revision(), RenderSceneRevision::INITIAL);
        assert_eq!(commit.revision(), RenderSceneRevision::INITIAL);
        assert!(commit.directly_follows(&continuity));
        assert!(commit.change_set().is_empty_incremental());
        assert_eq!(store.snapshot(), before);
    }

    #[test]
    fn multi_operation_commit_advances_revision_once() {
        let mut store = RenderSceneStore::new();
        let first = store.allocate_object_id().expect("ID should allocate");
        let second = store.allocate_object_id().expect("ID should allocate");
        let third = store.allocate_object_id().expect("ID should allocate");
        insert_one(&mut store, first);
        let previous = store.snapshot().continuity();
        let mut update = RenderSceneUpdate::new();
        update.remove(first).insert(second).insert(third);
        let commit = store
            .commit(update)
            .expect("multi-operation update should commit");
        assert_eq!(commit.previous_revision(), RenderSceneRevision(1));
        assert_eq!(commit.revision(), RenderSceneRevision(2));
        assert!(commit.directly_follows(&previous));
        assert_eq!(commit.snapshot().object_ids(), vec![second, third]);
        assert_eq!(commit.change_set().inserted(), Some(&[second, third][..]));
        assert_eq!(commit.change_set().removed(), Some(&[first][..]));
    }

    #[test]
    fn retained_snapshot_remains_immutable_after_later_commits() {
        let mut store = RenderSceneStore::new();
        let first = store.allocate_object_id().expect("ID should allocate");
        let second = store.allocate_object_id().expect("ID should allocate");
        let retained = insert_one(&mut store, first).snapshot().clone();
        insert_one(&mut store, second);
        assert_eq!(retained.revision(), RenderSceneRevision(1));
        assert_eq!(retained.object_ids(), vec![first]);
        assert_eq!(store.snapshot().object_ids(), vec![first, second]);
    }

    #[test]
    fn full_and_incremental_presence_construction_are_equivalent() {
        let mut full = RenderSceneStore::new();
        let full_ids = [
            full.allocate_object_id().expect("ID should allocate"),
            full.allocate_object_id().expect("ID should allocate"),
            full.allocate_object_id().expect("ID should allocate"),
        ];
        let mut full_update = RenderSceneUpdate::new();
        for object_id in full_ids {
            full_update.insert(object_id);
        }
        full.commit(full_update)
            .expect("full construction should commit");

        let mut incremental = RenderSceneStore::new();
        let incremental_ids = [
            incremental
                .allocate_object_id()
                .expect("ID should allocate"),
            incremental
                .allocate_object_id()
                .expect("ID should allocate"),
            incremental
                .allocate_object_id()
                .expect("ID should allocate"),
        ];
        for object_id in incremental_ids {
            insert_one(&mut incremental, object_id);
        }
        assert_eq!(
            full.snapshot().object_ids(),
            incremental.snapshot().object_ids()
        );
        assert_ne!(full.revision(), incremental.revision());
    }

    #[test]
    fn full_resync_is_explicit_and_does_not_advance_revision() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("ID should allocate");
        insert_one(&mut store, object_id);
        let revision = store.revision();
        let resync = store.full_resync();
        assert!(resync.change_set().is_full_resync());
        assert_eq!(resync.snapshot().revision(), revision);
        assert_eq!(resync.snapshot().object_ids(), vec![object_id]);
        assert_eq!(store.revision(), revision);
    }

    #[test]
    fn small_presence_change_path_copy_is_bounded_by_id_width() {
        let mut store = RenderSceneStore::new();
        for _ in 0..4096 {
            let object_id = store.allocate_object_id().expect("ID should allocate");
            insert_one(&mut store, object_id);
        }
        let next = store.allocate_object_id().expect("ID should allocate");
        let (_, insert_copies) = store.objects.inserted(next, None);
        assert_eq!(insert_copies, RADIX_DEPTH + 1);
        let existing = store.snapshot().object_ids()[2048];
        let (_, remove_copies) = store.objects.removed(existing);
        assert_eq!(remove_copies, RADIX_DEPTH + 1);
    }

    #[test]
    fn allocation_and_revision_exhaustion_are_explicit() {
        let mut allocation = RenderSceneStore {
            next_object_raw: u64::MAX,
            ..RenderSceneStore::default()
        };
        assert_eq!(
            allocation.allocate_object_id(),
            Err(RenderObjectIdAllocationError::Exhausted)
        );

        let mut revision = RenderSceneStore {
            revision: RenderSceneRevision(u64::MAX),
            ..RenderSceneStore::default()
        };
        let object_id = revision.allocate_object_id().expect("ID should allocate");
        let before = revision.snapshot();
        let mut update = RenderSceneUpdate::new();
        update.insert(object_id);
        assert_eq!(
            revision.commit(update),
            Err(RenderSceneCommitError::RevisionExhausted)
        );
        assert_eq!(revision.snapshot(), before);
    }

    #[test]
    fn stateful_insert_and_changed_replacement_preserve_identity_and_publish_r2_evidence() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("ID should allocate");
        let initial = object_state(0.0, 1.0);
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, initial.clone());
        let insert_commit = store.commit(insert).expect("stateful insert should commit");
        assert_eq!(
            insert_commit.change_set().inserted(),
            Some(&[object_id][..])
        );
        assert_eq!(insert_commit.change_set().spatial_changed(), Some(&[][..]));
        assert_eq!(insert_commit.change_set().temporal_changed(), Some(&[][..]));
        assert_eq!(store.snapshot().object_state(object_id), Some(&initial));

        let replacement = object_state(2.0, 2.0);
        let mut replace = RenderSceneUpdate::new();
        replace.replace_state(object_id, replacement.clone());
        let commit = store
            .commit(replace)
            .expect("state replacement should commit");
        assert_eq!(commit.revision(), RenderSceneRevision(2));
        assert_eq!(commit.snapshot().object_ids(), vec![object_id]);
        assert_eq!(
            commit.snapshot().object_state(object_id),
            Some(&replacement)
        );
        assert_eq!(
            commit.change_set().spatial_changed(),
            Some(&[object_id][..])
        );
        assert_eq!(
            commit.change_set().temporal_changed(),
            Some(&[object_id][..])
        );
    }

    #[test]
    fn equal_state_replacement_is_noop_and_single_axis_change_is_precise() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("ID should allocate");
        let initial = object_state(0.0, 1.0);
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, initial.clone());
        store.commit(insert).expect("stateful insert should commit");
        let revision = store.revision();

        let mut equal = RenderSceneUpdate::new();
        equal.replace_state(object_id, initial.clone());
        let equal_commit = store
            .commit(equal)
            .expect("equal replacement should be accepted");
        assert_eq!(equal_commit.previous_revision(), revision);
        assert_eq!(equal_commit.revision(), revision);
        assert!(equal_commit.change_set().is_empty_incremental());

        let spatial_only = RenderObjectState::new(
            object_state(3.0, 1.0).spatial().clone(),
            *initial.temporal(),
        );
        let mut changed = RenderSceneUpdate::new();
        changed.replace_state(object_id, spatial_only);
        let commit = store
            .commit(changed)
            .expect("spatial replacement should commit");
        assert_eq!(
            commit.change_set().spatial_changed(),
            Some(&[object_id][..])
        );
        assert_eq!(commit.change_set().temporal_changed(), Some(&[][..]));
    }

    #[test]
    fn missing_and_mixed_state_replacement_reject_without_partial_publication() {
        let mut store = RenderSceneStore::new();
        let missing = store.allocate_object_id().expect("ID should allocate");
        let before = store.snapshot();
        let mut replace = RenderSceneUpdate::new();
        replace.replace_state(missing, object_state(0.0, 1.0));
        assert_eq!(
            store.commit(replace),
            Err(RenderSceneCommitError::ObjectMissing { object_id: missing })
        );
        assert_eq!(store.snapshot(), before);

        let mut mixed = RenderSceneUpdate::new();
        mixed
            .insert(missing)
            .replace_state(missing, object_state(0.0, 1.0));
        assert_eq!(
            store.commit(mixed),
            Err(RenderSceneCommitError::ConflictingOperations { object_id: missing })
        );
        assert_eq!(store.snapshot(), before);
    }

    #[test]
    fn invalid_operation_in_multi_object_r2_update_rejects_without_partial_replacement() {
        let mut store = RenderSceneStore::new();
        let present = store.allocate_object_id().expect("ID should allocate");
        let missing = store.allocate_object_id().expect("ID should allocate");
        let initial = object_state(0.0, 1.0);
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(present, initial.clone());
        store.commit(insert).expect("stateful insert should commit");
        let before = store.snapshot();
        let revision = store.revision();

        let mut update = RenderSceneUpdate::new();
        update
            .replace_state(present, object_state(4.0, 2.0))
            .remove(missing);
        assert_eq!(
            store.commit(update),
            Err(RenderSceneCommitError::ObjectMissing { object_id: missing })
        );
        assert_eq!(store.revision(), revision);
        assert_eq!(store.snapshot(), before);
        assert_eq!(store.snapshot().object_state(present), Some(&initial));
    }

    #[test]
    fn retained_snapshot_preserves_prior_r2_state() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("ID should allocate");
        let initial = object_state(0.0, 1.0);
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, initial.clone());
        let retained = store
            .commit(insert)
            .expect("insert should commit")
            .snapshot()
            .clone();
        let mut replace = RenderSceneUpdate::new();
        replace.replace_state(object_id, object_state(4.0, 2.0));
        store.commit(replace).expect("replace should commit");
        assert_eq!(retained.object_state(object_id), Some(&initial));
        assert_ne!(
            retained.object_state(object_id),
            store.snapshot().object_state(object_id)
        );
    }

    #[test]
    fn full_and_incremental_r2_state_construction_are_semantically_equivalent() {
        let state = object_state(1.0, 2.0);
        let mut full = RenderSceneStore::new();
        let full_id = full.allocate_object_id().expect("ID should allocate");
        let mut full_update = RenderSceneUpdate::new();
        full_update.insert_with_state(full_id, state.clone());
        full.commit(full_update)
            .expect("stateful insert should commit");

        let mut incremental = RenderSceneStore::new();
        let incremental_id = incremental
            .allocate_object_id()
            .expect("ID should allocate");
        insert_one(&mut incremental, incremental_id);
        let mut replace = RenderSceneUpdate::new();
        replace.replace_state(incremental_id, state.clone());
        incremental
            .commit(replace)
            .expect("replacement should commit");

        assert_eq!(full.snapshot().object_ids(), vec![full_id]);
        assert_eq!(incremental.snapshot().object_ids(), vec![incremental_id]);
        assert_eq!(full.snapshot().object_state(full_id), Some(&state));
        assert_eq!(
            incremental.snapshot().object_state(incremental_id),
            Some(&state)
        );
    }

    #[test]
    fn small_state_replacement_path_copy_is_bounded_by_id_width() {
        let mut store = RenderSceneStore::new();
        for index in 0..4096 {
            let object_id = store.allocate_object_id().expect("ID should allocate");
            let mut update = RenderSceneUpdate::new();
            update.insert_with_state(object_id, object_state(index as f64, 1.0));
            store.commit(update).expect("insert should commit");
        }
        let existing = store.snapshot().object_ids()[2048];
        let (_, copies) =
            store
                .objects
                .replaced_facets(existing, Some(object_state(9999.0, 2.0)), None);
        assert_eq!(copies, RADIX_DEPTH + 1);
    }

    #[test]
    fn representation_allocator_is_renderer_owned_non_reusing_and_revision_neutral() {
        let mut store = RenderSceneStore::new();
        let first_object = store.allocate_object_id().expect("object ID");
        let second_object = store.allocate_object_id().expect("object ID");
        let first = store
            .allocate_representation_id(first_object)
            .expect("representation ID");
        let second = store
            .allocate_representation_id(second_object)
            .expect("representation ID");
        assert_eq!(first.raw(), 1);
        assert_eq!(second.raw(), 2);
        assert_ne!(first, second);
        assert_eq!(store.revision(), RenderSceneRevision::INITIAL);
    }

    #[test]
    fn representation_allocator_exhaustion_is_explicit_and_revision_neutral() {
        let mut store = RenderSceneStore::new();
        let owner = store.allocate_object_id().expect("object ID");
        let revision = store.revision();
        store.next_representation_raw = u64::MAX;

        assert_eq!(
            store.allocate_representation_id(owner),
            Err(RenderRepresentationIdAllocationError::Exhausted)
        );
        assert_eq!(store.revision(), revision);
        assert!(store.representation_owners.is_empty());
    }

    #[test]
    fn r3_participation_supports_multiple_representations_and_precise_change_evidence() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object ID");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, object_state(0.0, 1.0));
        store.commit(insert).expect("insert should commit");

        let first = surface_representation(&mut store, object_id);
        let second = field_representation(&mut store, object_id);
        let participation = RenderObjectParticipation::from_representations(vec![second, first])
            .expect("valid participation");
        let mut update = RenderSceneUpdate::new();
        update.replace_participation(object_id, participation.clone());
        let commit = store
            .commit(update)
            .expect("R3 participation should commit");

        assert_eq!(commit.revision(), RenderSceneRevision(2));
        assert_eq!(
            commit.change_set().representation_changed(),
            Some(&[object_id][..])
        );
        assert_eq!(
            commit.change_set().material_assignment_changed(),
            Some(&[][..])
        );
        assert_eq!(commit.change_set().emitter_changed(), Some(&[][..]));
        assert_eq!(
            commit.snapshot().object_participation(object_id),
            Some(&participation)
        );
        assert_eq!(commit.snapshot().object_ids(), vec![object_id]);
    }

    #[test]
    fn representation_identity_cannot_move_between_objects() {
        let mut store = RenderSceneStore::new();
        let first = store.allocate_object_id().expect("object ID");
        let second = store.allocate_object_id().expect("object ID");
        insert_one(&mut store, first);
        insert_one(&mut store, second);
        let representation = surface_representation(&mut store, first);
        let representation_id = representation.id();
        let participation = RenderObjectParticipation::from_representations(vec![representation])
            .expect("valid participation");
        let before = store.snapshot();

        let mut update = RenderSceneUpdate::new();
        update.replace_participation(second, participation);
        assert_eq!(
            store.commit(update),
            Err(RenderSceneCommitError::RepresentationOwnerMismatch {
                representation_id,
                allocated_owner: first,
                requested_owner: second,
            })
        );
        assert_eq!(store.snapshot(), before);
    }

    #[test]
    fn equal_r3_replacement_is_noop_and_r2_replacement_preserves_participation() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object ID");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, object_state(0.0, 1.0));
        store.commit(insert).expect("insert should commit");
        let representation = surface_representation(&mut store, object_id);
        let participation = RenderObjectParticipation::from_representations(vec![representation])
            .expect("valid participation");
        let mut attach = RenderSceneUpdate::new();
        attach.replace_participation(object_id, participation.clone());
        store.commit(attach).expect("participation should commit");
        let revision = store.revision();

        let mut equal = RenderSceneUpdate::new();
        equal.replace_participation(object_id, participation.clone());
        let equal_commit = store
            .commit(equal)
            .expect("equal R3 replacement should commit");
        assert_eq!(equal_commit.revision(), revision);
        assert!(equal_commit.change_set().is_empty_incremental());

        let mut spatial = RenderSceneUpdate::new();
        spatial.replace_state(object_id, object_state(2.0, 2.0));
        store.commit(spatial).expect("R2 replacement should commit");
        assert_eq!(
            store.snapshot().object_participation(object_id),
            Some(&participation)
        );
    }

    #[test]
    fn material_assignment_and_emitter_publish_only_their_truthful_r3_evidence() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object ID");
        insert_one(&mut store, object_id);
        let material =
            RenderMaterialAssignment::new(RenderDiffuseMaterial::new(0.5).expect("valid material"));
        let material_only = RenderObjectParticipation::from_representations([])
            .expect("valid state")
            .with_material_assignment(Some(material));
        let mut material_update = RenderSceneUpdate::new();
        material_update.replace_participation(object_id, material_only);
        let material_commit = store
            .commit(material_update)
            .expect("material assignment should commit");
        assert_eq!(
            material_commit.change_set().material_assignment_changed(),
            Some(&[object_id][..])
        );
        assert_eq!(
            material_commit.change_set().representation_changed(),
            Some(&[][..])
        );
        assert_eq!(
            material_commit.change_set().emitter_changed(),
            Some(&[][..])
        );

        let emitter =
            RenderDirectionalEmitter::new([0.0, 1.0, 0.0], 550e-9, 2.0).expect("valid emitter");
        let with_emitter = RenderObjectParticipation::from_representations([])
            .expect("valid state")
            .with_material_assignment(Some(material))
            .with_emitter(Some(emitter));
        let mut emitter_update = RenderSceneUpdate::new();
        emitter_update.replace_participation(object_id, with_emitter);
        let emitter_commit = store
            .commit(emitter_update)
            .expect("emitter semantics should commit");
        assert_eq!(
            emitter_commit.change_set().emitter_changed(),
            Some(&[object_id][..])
        );
        assert_eq!(
            emitter_commit.change_set().material_assignment_changed(),
            Some(&[][..])
        );
    }

    #[test]
    fn atomic_multi_facet_replacement_commits_once_with_precise_evidence() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object ID");
        let initial_state = object_state(0.0, 1.0);
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, initial_state.clone());
        store.commit(insert).expect("stateful insert should commit");

        let first_representation = surface_representation(&mut store, object_id);
        let initial_participation =
            RenderObjectParticipation::from_representations(vec![first_representation])
                .expect("valid initial participation");
        let mut attach = RenderSceneUpdate::new();
        attach.replace_participation(object_id, initial_participation.clone());
        store
            .commit(attach)
            .expect("initial participation should commit");

        let retained = store.snapshot();
        let retained_revision = retained.revision();
        let second_representation = surface_representation(&mut store, object_id);
        let material =
            RenderMaterialAssignment::new(RenderDiffuseMaterial::new(0.5).expect("material"));
        let next_participation =
            RenderObjectParticipation::from_representations(vec![second_representation])
                .expect("valid next participation")
                .with_material_assignment(Some(material));
        let next_state = object_state(2.0, 2.0);

        let mut update = RenderSceneUpdate::new();
        update
            .replace_state(object_id, next_state.clone())
            .replace_participation(object_id, next_participation.clone());
        let commit = store
            .commit(update)
            .expect("joint state/participation replacement should commit");

        assert_eq!(commit.previous_revision(), retained_revision);
        assert_eq!(
            commit.revision(),
            retained_revision
                .checked_next()
                .expect("revision should advance")
        );
        assert_eq!(
            commit.change_set().spatial_changed(),
            Some(&[object_id][..])
        );
        assert_eq!(
            commit.change_set().temporal_changed(),
            Some(&[object_id][..])
        );
        assert_eq!(
            commit.change_set().representation_changed(),
            Some(&[object_id][..])
        );
        assert_eq!(
            commit.change_set().material_assignment_changed(),
            Some(&[object_id][..])
        );
        assert_eq!(commit.change_set().emitter_changed(), Some(&[][..]));
        assert_eq!(commit.snapshot().object_state(object_id), Some(&next_state));
        assert_eq!(
            commit.snapshot().object_participation(object_id),
            Some(&next_participation)
        );
        assert_eq!(retained.object_state(object_id), Some(&initial_state));
        assert_eq!(
            retained.object_participation(object_id),
            Some(&initial_participation)
        );
    }

    #[test]
    fn atomic_multi_facet_validation_uses_the_proposed_final_object() {
        let mut add = RenderSceneStore::new();
        let add_id = add.allocate_object_id().expect("object ID");
        insert_one(&mut add, add_id);
        let field = field_representation(&mut add, add_id);
        let field_participation = RenderObjectParticipation::from_representations(vec![field])
            .expect("field participation");
        let mut add_both = RenderSceneUpdate::new();
        add_both
            .replace_state(add_id, object_state(0.0, 1.0))
            .replace_participation(add_id, field_participation.clone());
        add.commit(add_both)
            .expect("joint compatible state/field participation should commit");
        assert_eq!(
            add.snapshot().object_participation(add_id),
            Some(&field_participation)
        );

        let mut clear = RenderSceneStore::new();
        let clear_id = clear.allocate_object_id().expect("object ID");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(clear_id, object_state(0.0, 1.0));
        clear.commit(insert).expect("stateful insert");
        let field = field_representation(&mut clear, clear_id);
        let field_participation = RenderObjectParticipation::from_representations(vec![field])
            .expect("field participation");
        let mut attach = RenderSceneUpdate::new();
        attach.replace_participation(clear_id, field_participation);
        clear
            .commit(attach)
            .expect("field participation should commit");

        let singular = RenderAffineTransform3::from_row_major_3x4([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ])
        .expect("finite singular transform");
        let mut clear_and_replace = RenderSceneUpdate::new();
        clear_and_replace
            .replace_state(clear_id, object_state_with_transform(singular, 0.0, 2.0))
            .clear_participation(clear_id);
        clear
            .commit(clear_and_replace)
            .expect("clearing field participation makes final singular state valid");
        assert!(clear.snapshot().object_participation(clear_id).is_none());

        let mut invalid = RenderSceneStore::new();
        let invalid_id = invalid.allocate_object_id().expect("object ID");
        insert_one(&mut invalid, invalid_id);
        let field = field_representation(&mut invalid, invalid_id);
        let representation_id = field.id();
        let participation = RenderObjectParticipation::from_representations(vec![field])
            .expect("field participation");
        let before = invalid.snapshot();
        let revision = invalid.revision();
        let mut invalid_joint = RenderSceneUpdate::new();
        invalid_joint
            .replace_state(invalid_id, object_state_with_transform(singular, 0.0, 2.0))
            .replace_participation(invalid_id, participation);
        assert_eq!(
            invalid.commit(invalid_joint),
            Err(RenderSceneCommitError::InvalidFieldTransform {
                object_id: invalid_id,
                representation_id,
            })
        );
        assert_eq!(invalid.revision(), revision);
        assert_eq!(invalid.snapshot(), before);
    }

    #[test]
    fn multi_facet_noop_preserves_precise_evidence_for_changed_peer_facet() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object ID");
        let state = object_state(0.0, 1.0);
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, state.clone());
        store.commit(insert).expect("stateful insert");

        let first_representation = surface_representation(&mut store, object_id);
        let first_participation =
            RenderObjectParticipation::from_representations(vec![first_representation])
                .expect("initial participation");
        let mut attach = RenderSceneUpdate::new();
        attach.replace_participation(object_id, first_participation.clone());
        store.commit(attach).expect("initial participation");

        let next_representation = surface_representation(&mut store, object_id);
        let next_participation =
            RenderObjectParticipation::from_representations(vec![next_representation])
                .expect("next participation");
        let previous_revision = store.revision();
        let mut update = RenderSceneUpdate::new();
        update
            .replace_state(object_id, state)
            .replace_participation(object_id, next_participation.clone());
        let commit = store
            .commit(update)
            .expect("equal state plus changed participation should commit");

        assert_eq!(commit.previous_revision(), previous_revision);
        assert_eq!(
            commit.revision(),
            previous_revision
                .checked_next()
                .expect("revision should advance")
        );
        assert_eq!(commit.change_set().spatial_changed(), Some(&[][..]));
        assert_eq!(commit.change_set().temporal_changed(), Some(&[][..]));
        assert_eq!(
            commit.change_set().representation_changed(),
            Some(&[object_id][..])
        );
        assert_eq!(
            commit.snapshot().object_participation(object_id),
            Some(&next_participation)
        );
    }

    #[test]
    fn duplicate_same_facet_and_structural_mixes_still_reject() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object ID");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, object_state(0.0, 1.0));
        store.commit(insert).expect("stateful insert");
        let before = store.snapshot();

        let mut duplicate_state = RenderSceneUpdate::new();
        duplicate_state
            .replace_state(object_id, object_state(1.0, 2.0))
            .replace_state(object_id, object_state(2.0, 3.0));
        assert_eq!(
            store.commit(duplicate_state),
            Err(RenderSceneCommitError::ConflictingOperations { object_id })
        );
        assert_eq!(store.snapshot(), before);

        let mut remove_and_replace = RenderSceneUpdate::new();
        remove_and_replace
            .remove(object_id)
            .replace_state(object_id, object_state(1.0, 2.0));
        assert_eq!(
            store.commit(remove_and_replace),
            Err(RenderSceneCommitError::ConflictingOperations { object_id })
        );
        assert_eq!(store.snapshot(), before);
    }

    #[test]
    fn duplicate_same_object_r3_replacement_rejects_without_publication() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object ID");
        insert_one(&mut store, object_id);
        let representation = surface_representation(&mut store, object_id);
        let participation = RenderObjectParticipation::from_representations(vec![representation])
            .expect("valid state");
        let before = store.snapshot();

        let mut update = RenderSceneUpdate::new();
        update
            .replace_participation(object_id, participation.clone())
            .replace_participation(object_id, participation);
        assert_eq!(
            store.commit(update),
            Err(RenderSceneCommitError::ConflictingOperations { object_id })
        );
        assert_eq!(store.snapshot(), before);
    }
    #[test]
    fn retained_snapshot_preserves_prior_r3_participation() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object ID");
        insert_one(&mut store, object_id);
        let first = surface_representation(&mut store, object_id);
        let first_state = RenderObjectParticipation::from_representations(vec![first])
            .expect("valid participation");
        let mut attach = RenderSceneUpdate::new();
        attach.replace_participation(object_id, first_state.clone());
        let retained = store
            .commit(attach)
            .expect("participation should commit")
            .snapshot()
            .clone();

        let second = surface_representation(&mut store, object_id);
        let second_state = RenderObjectParticipation::from_representations(vec![second])
            .expect("valid participation");
        let mut replace = RenderSceneUpdate::new();
        replace.replace_participation(object_id, second_state.clone());
        store.commit(replace).expect("replacement should commit");

        assert_eq!(retained.object_participation(object_id), Some(&first_state));
        assert_eq!(
            store.snapshot().object_participation(object_id),
            Some(&second_state)
        );
    }

    #[test]
    fn full_and_incremental_r3_construction_are_semantically_equivalent() {
        let mut full = RenderSceneStore::new();
        let full_id = full.allocate_object_id().expect("object ID");
        insert_one(&mut full, full_id);
        let full_first = surface_representation(&mut full, full_id);
        let full_second = field_representation(&mut full, full_id);
        let full_state =
            RenderObjectParticipation::from_representations(vec![full_first, full_second])
                .expect("valid participation");
        let mut full_update = RenderSceneUpdate::new();
        full_update.replace_state(full_id, object_state(0.0, 1.0));
        full.commit(full_update).expect("R2 state should commit");
        let mut full_participation = RenderSceneUpdate::new();
        full_participation.replace_participation(full_id, full_state.clone());
        full.commit(full_participation)
            .expect("full participation should commit");

        let mut incremental = RenderSceneStore::new();
        let incremental_id = incremental.allocate_object_id().expect("object ID");
        insert_one(&mut incremental, incremental_id);
        let incremental_first = surface_representation(&mut incremental, incremental_id);
        let incremental_second = field_representation(&mut incremental, incremental_id);
        let mut incremental_state = RenderSceneUpdate::new();
        incremental_state.replace_state(incremental_id, object_state(0.0, 1.0));
        incremental
            .commit(incremental_state)
            .expect("R2 state should commit");
        let first_only = RenderObjectParticipation::from_representations(vec![incremental_first])
            .expect("valid participation");
        let mut first_update = RenderSceneUpdate::new();
        first_update.replace_participation(incremental_id, first_only);
        incremental
            .commit(first_update)
            .expect("first representation should commit");
        let incremental_final = RenderObjectParticipation::from_representations(vec![
            incremental
                .snapshot()
                .object_participation(incremental_id)
                .expect("participation")
                .representations()[0]
                .clone(),
            incremental_second,
        ])
        .expect("valid participation");
        let mut final_update = RenderSceneUpdate::new();
        final_update.replace_participation(incremental_id, incremental_final.clone());
        incremental
            .commit(final_update)
            .expect("second representation should commit");

        assert_eq!(
            full_state.representations().len(),
            incremental_final.representations().len()
        );
        assert_eq!(
            full_state
                .representations()
                .iter()
                .map(|representation| representation
                    .surface_query_protocol(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
                    .is_ok())
                .collect::<Vec<_>>(),
            incremental_final
                .representations()
                .iter()
                .map(|representation| representation
                    .surface_query_protocol(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
                    .is_ok())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn field_participation_rejects_missing_or_singular_spatial_state_atomically() {
        let mut missing_state = RenderSceneStore::new();
        let object_id = missing_state.allocate_object_id().expect("object ID");
        insert_one(&mut missing_state, object_id);
        let field = field_representation(&mut missing_state, object_id);
        let representation_id = field.id();
        let participation = RenderObjectParticipation::from_representations(vec![field])
            .expect("valid participation");
        let before = missing_state.snapshot();
        let mut attach = RenderSceneUpdate::new();
        attach.replace_participation(object_id, participation);
        assert_eq!(
            missing_state.commit(attach),
            Err(
                RenderSceneCommitError::FieldRepresentationRequiresSpatialState {
                    object_id,
                    representation_id,
                }
            )
        );
        assert_eq!(missing_state.snapshot(), before);

        let mut singular = RenderSceneStore::new();
        let singular_id = singular.allocate_object_id().expect("object ID");
        let singular_transform = RenderAffineTransform3::from_row_major_3x4([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ])
        .expect("finite singular transform");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(
            singular_id,
            object_state_with_transform(singular_transform, 0.0, 1.0),
        );
        singular
            .commit(insert)
            .expect("R2 insert accepts degenerate object transform");
        let field = field_representation(&mut singular, singular_id);
        let representation_id = field.id();
        let participation = RenderObjectParticipation::from_representations(vec![field])
            .expect("valid participation");
        let before = singular.snapshot();
        let mut attach = RenderSceneUpdate::new();
        attach.replace_participation(singular_id, participation);
        assert_eq!(
            singular.commit(attach),
            Err(RenderSceneCommitError::InvalidFieldTransform {
                object_id: singular_id,
                representation_id,
            })
        );
        assert_eq!(singular.snapshot(), before);
    }

    #[test]
    fn spatial_replacement_cannot_invalidate_existing_field_representation() {
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object ID");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, object_state(0.0, 1.0));
        store.commit(insert).expect("insert should commit");
        let field = field_representation(&mut store, object_id);
        let representation_id = field.id();
        let participation = RenderObjectParticipation::from_representations(vec![field])
            .expect("valid participation");
        let mut attach = RenderSceneUpdate::new();
        attach.replace_participation(object_id, participation.clone());
        store
            .commit(attach)
            .expect("field participation should commit");
        let before = store.snapshot();
        let revision = store.revision();

        let singular = RenderAffineTransform3::from_row_major_3x4([
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ])
        .expect("finite singular transform");
        let mut replace = RenderSceneUpdate::new();
        replace.replace_state(object_id, object_state_with_transform(singular, 0.0, 2.0));
        assert_eq!(
            store.commit(replace),
            Err(RenderSceneCommitError::InvalidFieldTransform {
                object_id,
                representation_id,
            })
        );
        assert_eq!(store.revision(), revision);
        assert_eq!(store.snapshot(), before);
        assert_eq!(
            store.snapshot().object_participation(object_id),
            Some(&participation)
        );
    }

    #[test]
    fn small_multi_facet_replacement_path_copy_is_bounded_by_id_width() {
        let mut store = RenderSceneStore::new();
        let mut target = None;
        for index in 0..4096 {
            let object_id = store.allocate_object_id().expect("object ID");
            insert_one(&mut store, object_id);
            if index == 2048 {
                target = Some(object_id);
            }
        }
        let object_id = target.expect("target object");
        let representation = surface_representation(&mut store, object_id);
        let participation = RenderObjectParticipation::from_representations(vec![representation])
            .expect("valid participation");
        let (_, copies) = store.objects.replaced_facets(
            object_id,
            Some(object_state(1.0, 2.0)),
            Some(Some(participation)),
        );
        assert_eq!(copies, RADIX_DEPTH + 1);
    }
}
