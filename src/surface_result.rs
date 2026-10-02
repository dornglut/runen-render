//! Query-relative result semantics for the R3 surface protocols.
//!
//! Plain surface results own only hit/miss meaning. An oriented-surface result adds one
//! representation-defined geometric front-side normal in canonical scene coordinates. Hit position
//! remains derived from the canonicalized query ray and non-negative distance, so neither protocol
//! can publish a hit whose position contradicts the query. Orientation is never implicitly flipped
//! toward the query ray. Dispatch/provider topology remains outside R3.

use super::representation::{RenderRepresentationValidationError, RenderSurfaceQuery};
use super::space_time::CanonicalF64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderSurfaceHit {
    distance_meters: CanonicalF64,
    position_scene_meters: [CanonicalF64; 3],
}

impl RenderSurfaceHit {
    fn from_query_distance(
        query: RenderSurfaceQuery,
        distance_meters: f64,
    ) -> Result<Self, RenderRepresentationValidationError> {
        let distance_meters = CanonicalF64::new(distance_meters, "surface_hit_distance_meters")?;
        if distance_meters.get() < 0.0 {
            return Err(RenderRepresentationValidationError::NegativeSurfaceHitDistance);
        }

        let origin = query.origin_scene_meters();
        let direction = query.direction_scene();
        let distance = distance_meters.get();
        let position = [
            origin[0] + direction[0] * distance,
            origin[1] + direction[1] * distance,
            origin[2] + direction[2] * distance,
        ];

        Ok(Self {
            distance_meters,
            position_scene_meters: [
                CanonicalF64::new(position[0], "surface_hit_position_scene_meters")?,
                CanonicalF64::new(position[1], "surface_hit_position_scene_meters")?,
                CanonicalF64::new(position[2], "surface_hit_position_scene_meters")?,
            ],
        })
    }

    pub fn distance_meters(self) -> f64 {
        self.distance_meters.get()
    }

    pub fn position_scene_meters(self) -> [f64; 3] {
        self.position_scene_meters.map(CanonicalF64::get)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderSurfaceQueryResult {
    kind: RenderSurfaceQueryResultKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RenderSurfaceQueryResultKind {
    Miss,
    Hit(RenderSurfaceHit),
}

impl RenderSurfaceQueryResult {
    pub const fn miss() -> Self {
        Self {
            kind: RenderSurfaceQueryResultKind::Miss,
        }
    }

    pub fn hit_at_distance(
        query: RenderSurfaceQuery,
        distance_meters: f64,
    ) -> Result<Self, RenderRepresentationValidationError> {
        let hit = RenderSurfaceHit::from_query_distance(query, distance_meters)?;
        Ok(Self {
            kind: RenderSurfaceQueryResultKind::Hit(hit),
        })
    }

    pub const fn is_miss(self) -> bool {
        matches!(self.kind, RenderSurfaceQueryResultKind::Miss)
    }

    pub const fn hit(self) -> Option<RenderSurfaceHit> {
        match self.kind {
            RenderSurfaceQueryResultKind::Miss => None,
            RenderSurfaceQueryResultKind::Hit(hit) => Some(hit),
        }
    }
}

/// Exact surface hit plus representation-defined geometric front-side orientation.
///
/// The normal is a canonical unit vector in renderer scene coordinates. Its sign is stable
/// representation-semantic meaning: it is not automatically made outward, viewer-facing,
/// ray-facing, two-sided, or material/shading dependent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderOrientedSurfaceHit {
    surface_hit: RenderSurfaceHit,
    geometric_normal_scene: [CanonicalF64; 3],
}

impl RenderOrientedSurfaceHit {
    pub const fn surface_hit(self) -> RenderSurfaceHit {
        self.surface_hit
    }

    pub fn geometric_normal_scene(self) -> [f64; 3] {
        self.geometric_normal_scene.map(CanonicalF64::get)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderOrientedSurfaceQueryResult {
    kind: RenderOrientedSurfaceQueryResultKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RenderOrientedSurfaceQueryResultKind {
    Miss,
    Hit(RenderOrientedSurfaceHit),
}

impl RenderOrientedSurfaceQueryResult {
    pub const fn miss() -> Self {
        Self {
            kind: RenderOrientedSurfaceQueryResultKind::Miss,
        }
    }

    pub fn hit_at_distance(
        query: RenderSurfaceQuery,
        distance_meters: f64,
        geometric_normal_scene: [f64; 3],
    ) -> Result<Self, RenderRepresentationValidationError> {
        let surface_hit = RenderSurfaceHit::from_query_distance(query, distance_meters)?;
        let geometric_normal_scene = canonical_unit_geometric_normal(geometric_normal_scene)?;
        Ok(Self {
            kind: RenderOrientedSurfaceQueryResultKind::Hit(RenderOrientedSurfaceHit {
                surface_hit,
                geometric_normal_scene,
            }),
        })
    }

    pub const fn is_miss(self) -> bool {
        matches!(self.kind, RenderOrientedSurfaceQueryResultKind::Miss)
    }

    pub const fn hit(self) -> Option<RenderOrientedSurfaceHit> {
        match self.kind {
            RenderOrientedSurfaceQueryResultKind::Miss => None,
            RenderOrientedSurfaceQueryResultKind::Hit(hit) => Some(hit),
        }
    }
}

fn canonical_unit_geometric_normal(
    normal: [f64; 3],
) -> Result<[CanonicalF64; 3], RenderRepresentationValidationError> {
    let values = [
        CanonicalF64::new(normal[0], "surface_geometric_normal_scene")?.get(),
        CanonicalF64::new(normal[1], "surface_geometric_normal_scene")?.get(),
        CanonicalF64::new(normal[2], "surface_geometric_normal_scene")?.get(),
    ];
    let scale = values.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(RenderRepresentationValidationError::ZeroSurfaceNormal);
    }
    let scaled = values.map(|value| value / scale);
    let length = scaled
        .iter()
        .copied()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let normalized = scaled.map(|value| value / length);
    Ok([
        CanonicalF64::new(normalized[0], "surface_geometric_normal_scene")?,
        CanonicalF64::new(normalized[1], "surface_geometric_normal_scene")?,
        CanonicalF64::new(normalized[2], "surface_geometric_normal_scene")?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plugins::render::space_time::{RenderSemanticValueError, RenderTimePoint};

    fn query() -> RenderSurfaceQuery {
        RenderSurfaceQuery::new(
            [0.0, 0.0, 3.0],
            [0.0, 0.0, -2.0],
            RenderTimePoint::from_seconds(0.0).expect("finite time"),
        )
        .expect("valid query")
    }

    #[test]
    fn surface_result_has_explicit_miss_semantics() {
        let result = RenderSurfaceQueryResult::miss();
        assert!(result.is_miss());
        assert_eq!(result.hit(), None);
    }

    #[test]
    fn surface_hit_position_is_derived_from_query_ray() {
        let result = RenderSurfaceQueryResult::hit_at_distance(query(), 2.0)
            .expect("non-negative hit distance");
        let hit = result.hit().expect("hit result");
        assert_eq!(hit.distance_meters(), 2.0);
        assert_eq!(hit.position_scene_meters(), [0.0, 0.0, 1.0]);
    }

    #[test]
    fn surface_result_rejects_negative_hit_distance() {
        assert_eq!(
            RenderSurfaceQueryResult::hit_at_distance(query(), -1.0),
            Err(RenderRepresentationValidationError::NegativeSurfaceHitDistance)
        );
    }

    #[test]
    fn oriented_surface_result_reuses_query_consistent_hit_and_canonicalizes_normal() {
        let result =
            RenderOrientedSurfaceQueryResult::hit_at_distance(query(), 2.0, [0.0, 3.0, 4.0])
                .expect("valid oriented hit");
        let hit = result.hit().expect("oriented hit");
        assert_eq!(hit.surface_hit().distance_meters(), 2.0);
        assert_eq!(hit.surface_hit().position_scene_meters(), [0.0, 0.0, 1.0]);
        assert_eq!(hit.geometric_normal_scene(), [0.0, 0.6, 0.8]);
    }

    #[test]
    fn oriented_surface_normal_is_not_implicitly_flipped_against_query_ray() {
        let result =
            RenderOrientedSurfaceQueryResult::hit_at_distance(query(), 2.0, [0.0, 0.0, -5.0])
                .expect("valid representation-defined orientation");
        assert_eq!(
            result.hit().expect("oriented hit").geometric_normal_scene(),
            [0.0, 0.0, -1.0]
        );
    }

    #[test]
    fn oriented_surface_result_rejects_zero_and_non_finite_normals() {
        assert_eq!(
            RenderOrientedSurfaceQueryResult::hit_at_distance(query(), 2.0, [0.0; 3]),
            Err(RenderRepresentationValidationError::ZeroSurfaceNormal)
        );
        assert!(matches!(
            RenderOrientedSurfaceQueryResult::hit_at_distance(query(), 2.0, [f64::NAN, 0.0, 1.0]),
            Err(RenderRepresentationValidationError::SemanticValue(
                RenderSemanticValueError::NonFinite { .. }
            ))
        ));
    }

    #[test]
    fn oriented_surface_miss_requires_no_fabricated_normal() {
        let result = RenderOrientedSurfaceQueryResult::miss();
        assert!(result.is_miss());
        assert_eq!(result.hit(), None);
    }
}
