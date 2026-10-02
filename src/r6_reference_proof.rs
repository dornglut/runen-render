//! CPU semantic oracle for the bounded R6 founding direct-lighting renderer.
//!
//! This module is test-only. It composes accepted R2/R3 observation, surface-result, material,
//! emitter, and renderer-local identity semantics. It does not realize geometry, define a new render
//! method contract, choose GPU formats, or authorize production API surface.

use super::appearance::{RenderDiffuseMaterial, RenderDirectionalEmitter};
use super::representation::RenderSurfaceQuery;
use super::request::{
    RenderPerspectiveObservation, RenderRadiometricRepresentation, RenderSamplingSupport,
};
use super::scene::{RenderObjectId, RenderSceneStore, RenderSceneUpdate};
use super::space_time::{RenderAffineTransform3, RenderTimeInterval, RenderTimePoint};
use super::surface_result::{RenderOrientedSurfaceHit, RenderOrientedSurfaceQueryResult};

#[derive(Debug, Clone, Copy, PartialEq)]
struct FoundingReferenceSample {
    spectral_radiance_w_sr_m3: f64,
    observation_forward_depth_meters: f64,
    object_id: RenderObjectId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FoundingReferenceError {
    SurfaceMiss,
    WavelengthMismatch,
}

pub(super) fn direct_lighting_radiance(
    material: RenderDiffuseMaterial,
    emitter: RenderDirectionalEmitter,
    requested: RenderRadiometricRepresentation,
    hit: RenderOrientedSurfaceHit,
    occluded: bool,
) -> Result<f64, FoundingReferenceError> {
    if emitter.wavelength_meters() != requested.wavelength_meters() {
        return Err(FoundingReferenceError::WavelengthMismatch);
    }
    if occluded {
        return Ok(0.0);
    }

    let normal = hit.geometric_normal_scene();
    let direction_to_source = emitter.direction_to_source_scene();
    let cosine = dot(normal, direction_to_source).max(0.0);
    Ok(material.reflectance() * emitter.spectral_irradiance_w_m3() * cosine / std::f64::consts::PI)
}

pub(super) fn observation_forward_depth(
    observation: RenderPerspectiveObservation,
    hit: RenderOrientedSurfaceHit,
) -> f64 {
    let observation_to_scene = observation.observation_to_scene().row_major_3x4();
    let origin_scene = [
        observation_to_scene[3],
        observation_to_scene[7],
        observation_to_scene[11],
    ];
    let forward_scene = normalize([
        -observation_to_scene[2],
        -observation_to_scene[6],
        -observation_to_scene[10],
    ]);
    dot(
        sub(hit.surface_hit().position_scene_meters(), origin_scene),
        forward_scene,
    )
}

fn reference_hit_sample(
    object_id: RenderObjectId,
    observation: RenderPerspectiveObservation,
    surface: RenderOrientedSurfaceQueryResult,
    material: RenderDiffuseMaterial,
    emitter: RenderDirectionalEmitter,
    requested: RenderRadiometricRepresentation,
    occluded: bool,
) -> Result<FoundingReferenceSample, FoundingReferenceError> {
    let hit = surface.hit().ok_or(FoundingReferenceError::SurfaceMiss)?;
    Ok(FoundingReferenceSample {
        spectral_radiance_w_sr_m3: direct_lighting_radiance(
            material, emitter, requested, hit, occluded,
        )?,
        observation_forward_depth_meters: observation_forward_depth(observation, hit),
        object_id,
    })
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn sub(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let length = dot(vector, vector).sqrt();
    debug_assert!(length > 0.0 && length.is_finite());
    vector.map(|component| component / length)
}

#[cfg(test)]
mod tests {
    use super::*;

    const WAVELENGTH_METERS: f64 = 550e-9;

    fn instant() -> RenderTimePoint {
        RenderTimePoint::from_seconds(0.0).expect("finite R6 reference time")
    }

    fn observation() -> RenderPerspectiveObservation {
        RenderPerspectiveObservation::new(
            RenderAffineTransform3::from_row_major_3x4([
                0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0,
            ])
            .expect("finite observation transform"),
            std::f64::consts::FRAC_PI_2,
            1.0,
            RenderTimeInterval::instant(instant()),
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("valid R6 reference observation")
    }

    fn represented_object_id() -> RenderObjectId {
        let mut store = RenderSceneStore::new();
        let object_id = store
            .allocate_object_id()
            .expect("R6 reference object identity");
        let mut insert = RenderSceneUpdate::new();
        insert.insert(object_id);
        store.commit(insert).expect("insert R6 reference object");
        object_id
    }

    fn requested_radiance() -> RenderRadiometricRepresentation {
        RenderRadiometricRepresentation::spectral_at_wavelength_meters(WAVELENGTH_METERS)
            .expect("valid R6 reference wavelength")
    }

    fn material() -> RenderDiffuseMaterial {
        RenderDiffuseMaterial::new(0.5).expect("valid R6 reference diffuse material")
    }

    fn emitter(direction_to_source_scene: [f64; 3]) -> RenderDirectionalEmitter {
        RenderDirectionalEmitter::new(direction_to_source_scene, WAVELENGTH_METERS, 12.0)
            .expect("valid R6 reference emitter")
    }

    fn hit(normal_scene: [f64; 3]) -> RenderOrientedSurfaceQueryResult {
        let query = RenderSurfaceQuery::new([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], instant())
            .expect("valid R6 reference query");
        RenderOrientedSurfaceQueryResult::hit_at_distance(query, 2.0, normal_scene)
            .expect("valid R6 reference hit")
    }

    #[test]
    fn cpu_reference_preserves_lambertian_units_forward_depth_and_renderer_identity() {
        let object_id = represented_object_id();
        let sample = reference_hit_sample(
            object_id,
            observation(),
            hit([1.0, 0.0, 0.0]),
            material(),
            emitter([1.0, 0.0, 0.0]),
            requested_radiance(),
            false,
        )
        .expect("lit R6 reference sample");

        let expected_radiance = 6.0 / std::f64::consts::PI;
        assert!((sample.spectral_radiance_w_sr_m3 - expected_radiance).abs() <= f64::EPSILON);
        assert_eq!(sample.observation_forward_depth_meters, 2.0);
        assert_eq!(sample.object_id, object_id);
    }

    #[test]
    fn cpu_reference_back_facing_and_occluded_samples_are_unlit() {
        let object_id = represented_object_id();
        let back_facing = reference_hit_sample(
            object_id,
            observation(),
            hit([-1.0, 0.0, 0.0]),
            material(),
            emitter([1.0, 0.0, 0.0]),
            requested_radiance(),
            false,
        )
        .expect("back-facing R6 reference sample");
        assert_eq!(back_facing.spectral_radiance_w_sr_m3, 0.0);

        let occluded = reference_hit_sample(
            object_id,
            observation(),
            hit([1.0, 0.0, 0.0]),
            material(),
            emitter([1.0, 0.0, 0.0]),
            requested_radiance(),
            true,
        )
        .expect("occluded R6 reference sample");
        assert_eq!(occluded.spectral_radiance_w_sr_m3, 0.0);
    }

    #[test]
    fn cpu_reference_rejects_surface_miss_and_unproven_wavelength_substitution() {
        let object_id = represented_object_id();
        let miss = reference_hit_sample(
            object_id,
            observation(),
            RenderOrientedSurfaceQueryResult::miss(),
            material(),
            emitter([1.0, 0.0, 0.0]),
            requested_radiance(),
            false,
        );
        assert_eq!(miss, Err(FoundingReferenceError::SurfaceMiss));

        let mismatched_emitter = RenderDirectionalEmitter::new([1.0, 0.0, 0.0], 540e-9, 12.0)
            .expect("valid different reference wavelength");
        let mismatch = reference_hit_sample(
            object_id,
            observation(),
            hit([1.0, 0.0, 0.0]),
            material(),
            mismatched_emitter,
            requested_radiance(),
            false,
        );
        assert_eq!(mismatch, Err(FoundingReferenceError::WavelengthMismatch));
    }
}
