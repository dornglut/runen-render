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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderObjectParticipation {
    representations: Vec<RenderRepresentationRecord>,
    material_assignment: Option<RenderMaterialAssignment>,
    emitter: Option<RenderDirectionalEmitter>,
}

impl RenderObjectParticipation {
    pub fn new(
        mut representations: Vec<RenderRepresentationRecord>,
        material_assignment: Option<RenderMaterialAssignment>,
        emitter: Option<RenderDirectionalEmitter>,
    ) -> Result<Self, RenderParticipationValidationError> {
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
            material_assignment,
            emitter,
        })
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
    use crate::plugins::render::representation::{
        RENDER_SURFACE_QUERY_PROTOCOL_REVISION, RenderRefinementEvidence,
        RenderSurfaceProtocolEvidence,
    };
    use crate::plugins::render::space_time::{RenderSpatialCoverage, RenderTemporalSupport};

    fn representation(raw: u64) -> RenderRepresentationRecord {
        RenderRepresentationRecord::new(
            RenderRepresentationId::from_raw(raw).expect("non-zero representation id"),
            RenderSpatialCoverage::unbounded(),
            RenderTemporalSupport::unbounded(),
            RenderRefinementEvidence::none(),
            Some(
                RenderSurfaceProtocolEvidence::exact(RENDER_SURFACE_QUERY_PROTOCOL_REVISION)
                    .expect("valid protocol"),
            ),
            None,
        )
        .expect("valid representation")
    }

    #[test]
    fn representation_order_is_semantic_not_caller_order() {
        let participation = RenderObjectParticipation::new(
            vec![representation(3), representation(1), representation(2)],
            None,
            None,
        )
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
            RenderObjectParticipation::new(
                vec![representation(2), representation(1), representation(2)],
                None,
                None,
            ),
            Err(
                RenderParticipationValidationError::DuplicateRepresentationId {
                    representation_id: duplicate,
                }
            )
        );
    }
}
