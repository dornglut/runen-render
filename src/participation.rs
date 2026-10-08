//! R3 object-local renderer participation semantics.
//!
//! Participation is distinct from the R2 spatial/temporal object state so that either layer can be
//! replaced without reconstructing or erasing the other.

use super::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
use super::representation::{RenderRepresentationId, RenderRepresentationRecord};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderMaterialAssignment {
    material: RenderDiffuseMaterial,
}

impl RenderMaterialAssignment {
    pub const fn new(material: RenderDiffuseMaterial) -> Self {
        Self { material }
    }

    pub const fn material(self) -> RenderDiffuseMaterial {
        self.material
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderParticipationValidationError {
    DuplicateRepresentationId {
        representation_id: RenderRepresentationId,
    },
}

impl fmt::Display for RenderParticipationValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateRepresentationId { representation_id } => write!(
                f,
                "object participation contains duplicate representation {representation_id:?}"
            ),
        }
    }
}

impl Error for RenderParticipationValidationError {}

/// The complete R3-owned participation state of one renderer object.
///
/// Representations are canonicalized by `RenderRepresentationId`, so equality and change evidence
/// do not depend on caller insertion order. Material assignment is the founding typed relationship:
/// the owning `RenderObjectId` endpoint is supplied by the scene leaf, while this value carries the
/// typed renderer-semantic material target. No independent material identity is invented in R3.
///
/// Construct the representation set with [`Self::from_representations`], then optionally name
/// material assignment and emitter facts. Empty, material-only, and emitter-only values are legal.
/// Publishing an empty value through [`crate::scene::RenderSceneUpdate::replace_participation`]
/// clears the existing participation facet; it does not attach a persistent empty value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderObjectParticipation {
    representations: Vec<RenderRepresentationRecord>,
    material_assignment: Option<RenderMaterialAssignment>,
    emitter: Option<RenderDirectionalEmitter>,
}

impl RenderObjectParticipation {
    /// Canonicalize a caller-assembled representation collection (or iterator).
    ///
    /// Callers may accumulate entries in a collection before construction, or supply an iterator.
    /// An empty iterator creates legitimate empty participation without material or emitter facts.
    ///
    /// # Errors
    /// Returns [`RenderParticipationValidationError::DuplicateRepresentationId`] for duplicate IDs.
    pub fn from_representations(
        representations: impl IntoIterator<Item = RenderRepresentationRecord>,
    ) -> Result<Self, RenderParticipationValidationError> {
        let mut representations = representations.into_iter().collect::<Vec<_>>();
        representations.sort_by_key(RenderRepresentationRecord::id);
        if let Some(pair) = representations
            .windows(2)
            .find(|pair| pair[0].id() == pair[1].id())
        {
            return Err(
                RenderParticipationValidationError::DuplicateRepresentationId {
                    representation_id: pair[0].id(),
                },
            );
        }
        Ok(Self {
            representations,
            material_assignment: None,
            emitter: None,
        })
    }

    /// Replace or clear the named material-assignment fact without rebuilding representations.
    #[must_use]
    pub fn with_material_assignment(
        mut self,
        assignment: Option<RenderMaterialAssignment>,
    ) -> Self {
        self.material_assignment = assignment;
        self
    }

    /// Replace or clear the named emitter fact without rebuilding representations.
    #[must_use]
    pub fn with_emitter(mut self, emitter: Option<RenderDirectionalEmitter>) -> Self {
        self.emitter = emitter;
        self
    }

    pub fn representations(&self) -> &[RenderRepresentationRecord] {
        &self.representations
    }

    pub const fn material_assignment(&self) -> Option<RenderMaterialAssignment> {
        self.material_assignment
    }

    pub const fn emitter(&self) -> Option<RenderDirectionalEmitter> {
        self.emitter
    }

    pub fn is_empty(&self) -> bool {
        self.representations.is_empty()
            && self.material_assignment.is_none()
            && self.emitter.is_none()
    }

    pub fn representation(
        &self,
        representation_id: RenderRepresentationId,
    ) -> Option<&RenderRepresentationRecord> {
        self.representations
            .binary_search_by_key(&representation_id, RenderRepresentationRecord::id)
            .ok()
            .map(|index| &self.representations[index])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::representation::{
        RENDER_SURFACE_QUERY_PROTOCOL_REVISION, RenderSurfaceProtocolEvidence,
    };
    use crate::space_time::{RenderSpatialCoverage, RenderTemporalSupport};

    fn representation(raw: u64) -> RenderRepresentationRecord {
        RenderRepresentationRecord::builder(
            RenderRepresentationId::from_raw(raw).expect("non-zero representation id"),
            RenderSpatialCoverage::unbounded(),
            RenderTemporalSupport::unbounded(),
        )
        .surface_query(Some(
            RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
                .expect("valid protocol"),
        ))
        .build()
        .expect("valid representation")
    }

    #[test]
    fn representation_order_is_semantic_not_caller_order() {
        let participation = RenderObjectParticipation::from_representations(vec![
            representation(3),
            representation(1),
            representation(2),
        ])
        .expect("unique representations");
        let ids = participation
            .representations()
            .iter()
            .map(RenderRepresentationRecord::id)
            .collect::<Vec<_>>();
        assert_eq!(
            ids,
            vec![
                RenderRepresentationId::from_raw(1).expect("id"),
                RenderRepresentationId::from_raw(2).expect("id"),
                RenderRepresentationId::from_raw(3).expect("id"),
            ]
        );
    }

    #[test]
    fn duplicate_representation_identity_rejects_deterministically() {
        let duplicate = RenderRepresentationId::from_raw(2).expect("id");
        assert_eq!(
            RenderObjectParticipation::from_representations(vec![
                representation(2),
                representation(1),
                representation(2)
            ]),
            Err(
                RenderParticipationValidationError::DuplicateRepresentationId {
                    representation_id: duplicate,
                }
            )
        );
    }
}
