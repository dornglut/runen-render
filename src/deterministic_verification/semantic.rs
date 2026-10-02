//! Conservative semantic verification for RR566-EVAL-001.
//!
//! This module is deliberately narrower than maintained execution. Static eligibility has already
//! proven identity observation linear bases plus unit, right-handed, translation-only object frames.
//! The verifier therefore evaluates the exact retained semantic scene/request/input facts directly
//! in scene space instead of duplicating the general execution transform compiler. Every uncertain
//! branch fails closed; no physical packing, WGSL behavior, or test-only CPU oracle becomes semantic
//! authority.

use super::super::admission::{AdmittedRenderPlan, RenderAdmittedOutput};
use super::super::appearance::RenderDirectionalEmitter;
use super::super::deterministic_carrier::maintained_evaluation_value;
use super::super::deterministic_execution::{
    DeterministicVerificationSubmission, RenderObjectIdentityDecoder,
};
use super::super::request::{
    RenderDistanceConvention, RenderObservationSpec, RenderOutputValue, RenderRequestedOutput,
    RenderResultTopology,
};
use super::super::scene::RenderObjectId;
use super::super::surface_input::RenderSurfaceSemanticInputView;
use super::RenderDeterministicVerificationError;
use super::numeric::{
    VerificationInterval, certified_tan_half_fov, mathematical_pi_interval,
    numeric_value_satisfies_tolerance,
};
use super::observation::DeterministicVerificationObservation;

#[derive(Debug, Clone, Copy)]
struct CertifiedRay {
    origin_scene: [VerificationInterval; 3],
    direction_scene: [VerificationInterval; 3],
}

#[derive(Debug, Clone, Copy)]
struct VerificationGeometry {
    object_id: RenderObjectId,
    translation_scene: [f64; 3],
    surface: RenderSurfaceSemanticInputView,
    reflectance: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct CertifiedHit {
    object_id: RenderObjectId,
    distance: VerificationInterval,
    position_scene: [VerificationInterval; 3],
    normal_scene: [VerificationInterval; 3],
    reflectance: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
enum CertifiedIntersection {
    Miss,
    Hit(CertifiedHit),
}

#[derive(Debug, Clone, Copy)]
struct ObservedSample {
    output_index: usize,
    sample_index: usize,
    canonical_word: u32,
    definedness_word: u32,
}

pub(super) fn verify_completed_semantics(
    verification: &DeterministicVerificationSubmission,
    observations: &[DeterministicVerificationObservation],
) -> Result<(), RenderDeterministicVerificationError> {
    let submitted = verification.submitted();
    let admitted = submitted.admitted().admitted();
    if observations.len() != admitted.outputs().len() {
        return Err(correlation_error(
            0,
            None,
            "normalized observation count changed after verified submission correlation",
        ));
    }

    for observed in observations {
        verify_output(admitted, submitted.object_identity_decoder(), observed)?;
    }
    Ok(())
}

fn verify_output(
    admitted: &AdmittedRenderPlan,
    identity_decoder: &RenderObjectIdentityDecoder,
    observed: &DeterministicVerificationObservation,
) -> Result<(), RenderDeterministicVerificationError> {
    let output_index = observed.output_index();
    let admitted_output = admitted
        .outputs()
        .iter()
        .find(|output| output.output_index() == output_index)
        .ok_or(correlation_error(
            output_index,
            None,
            "normalized observation no longer resolves to an admitted output",
        ))?;
    let requested = admitted
        .plan()
        .request()
        .outputs()
        .get(output_index)
        .copied()
        .ok_or(correlation_error(
            output_index,
            None,
            "admitted output no longer resolves to the retained request",
        ))?;
    let observation = admitted
        .plan()
        .request()
        .observations()
        .get(requested.observation_index())
        .copied()
        .ok_or(correlation_error(
            output_index,
            None,
            "requested observation correlation is outside the retained request",
        ))?;
    let sample_count = sample_count(requested.spec().topology()).ok_or(inconclusive_error(
        output_index,
        None,
        "semantic topology exceeds verifier host indexing limits",
    ))?;
    if observed.canonical_words().len() != sample_count
        || observed.definedness_words().len() != sample_count
        || observed.status_words().len() != sample_count
    {
        return Err(correlation_error(
            output_index,
            None,
            "normalized observation cardinality differs from retained semantic topology",
        ));
    }

    let needs_material = matches!(requested.spec().value(), RenderOutputValue::Radiance { .. });
    let geometry = collect_geometry(admitted, admitted_output, needs_material)?;
    for sample_index in 0..sample_count {
        let status = observed.status_words()[sample_index];
        if status != 0 {
            return Err(physical_mismatch_error(
                output_index,
                Some(sample_index),
                "maintained evaluator reported an internal-invalid sample",
            ));
        }
        let ray = certified_sample_ray(observation, requested.spec().topology(), sample_index)
            .ok_or(inconclusive_error(
                output_index,
                Some(sample_index),
                "sample ray could not be conservatively certified",
            ))?;
        let intersection =
            nearest_primary_intersection(ray, &geometry).ok_or(inconclusive_error(
                output_index,
                Some(sample_index),
                "primary surface branch or nearest-hit choice is ambiguous",
            ))?;
        verify_sample(
            admitted,
            identity_decoder,
            requested,
            ray,
            intersection,
            &geometry,
            ObservedSample {
                output_index,
                sample_index,
                canonical_word: observed.canonical_words()[sample_index],
                definedness_word: observed.definedness_words()[sample_index],
            },
        )?;
    }
    Ok(())
}

fn verify_sample(
    admitted: &AdmittedRenderPlan,
    identity_decoder: &RenderObjectIdentityDecoder,
    requested: RenderRequestedOutput,
    ray: CertifiedRay,
    intersection: CertifiedIntersection,
    geometry: &[VerificationGeometry],
    observed: ObservedSample,
) -> Result<(), RenderDeterministicVerificationError> {
    let ObservedSample {
        output_index,
        sample_index,
        canonical_word,
        definedness_word,
    } = observed;
    match requested.spec().value() {
        RenderOutputValue::Radiance { representation } => {
            require_definedness(output_index, sample_index, definedness_word, 1)?;
            let reference = match intersection {
                CertifiedIntersection::Miss => singleton(0.0).ok_or(inconclusive_error(
                    output_index,
                    Some(sample_index),
                    "zero radiance reference could not be represented",
                ))?,
                CertifiedIntersection::Hit(hit) => {
                    direct_radiance(admitted, geometry, hit, representation.wavelength_meters())
                        .ok_or(inconclusive_error(
                            output_index,
                            Some(sample_index),
                            "radiance orientation, visibility, or arithmetic branch is ambiguous",
                        ))?
                }
            };
            verify_numeric_value(
                output_index,
                sample_index,
                canonical_word,
                reference,
                requested.spec().tolerance(),
            )
        }
        RenderOutputValue::Distance {
            convention: RenderDistanceConvention::ObservationForwardDepth,
        } => match intersection {
            CertifiedIntersection::Miss => require_undefined_zero(
                output_index,
                sample_index,
                canonical_word,
                definedness_word,
                "forward-depth miss",
            ),
            CertifiedIntersection::Hit(hit) => {
                require_definedness(output_index, sample_index, definedness_word, 1)?;
                let displacement =
                    vec_sub(hit.position_scene, ray.origin_scene).ok_or(inconclusive_error(
                        output_index,
                        Some(sample_index),
                        "forward-depth displacement arithmetic is unsupported",
                    ))?;
                let forward = interval_vec3([0.0, 0.0, -1.0]).ok_or(inconclusive_error(
                    output_index,
                    Some(sample_index),
                    "observation forward vector is not representable",
                ))?;
                let reference = dot(displacement, forward).ok_or(inconclusive_error(
                    output_index,
                    Some(sample_index),
                    "forward-depth arithmetic is unsupported",
                ))?;
                verify_numeric_value(
                    output_index,
                    sample_index,
                    canonical_word,
                    reference,
                    requested.spec().tolerance(),
                )
            }
        },
        RenderOutputValue::ObjectIdentity => match intersection {
            CertifiedIntersection::Miss => {
                require_undefined_zero(
                    output_index,
                    sample_index,
                    canonical_word,
                    definedness_word,
                    "object-identity miss",
                )?;
                if identity_decoder.decode(canonical_word).is_some() {
                    return Err(physical_mismatch_error(
                        output_index,
                        Some(sample_index),
                        "undefined object-identity code unexpectedly decodes to an object",
                    ));
                }
                Ok(())
            }
            CertifiedIntersection::Hit(hit) => {
                require_definedness(output_index, sample_index, definedness_word, 1)?;
                if canonical_word == 0
                    || identity_decoder.decode(canonical_word) != Some(hit.object_id)
                {
                    return Err(physical_mismatch_error(
                        output_index,
                        Some(sample_index),
                        "physical object-identity code does not decode to the uniquely certified semantic hit",
                    ));
                }
                Ok(())
            }
        },
        RenderOutputValue::Distance { .. } => Err(correlation_error(
            output_index,
            Some(sample_index),
            "unsupported distance convention escaped maintained method admission",
        )),
    }
}

fn verify_numeric_value(
    output_index: usize,
    sample_index: usize,
    canonical_word: u32,
    reference: VerificationInterval,
    tolerance: super::super::request::RenderSemanticTolerance,
) -> Result<(), RenderDeterministicVerificationError> {
    let Some(observed) = maintained_evaluation_value(canonical_word).map(f64::from) else {
        return Err(physical_mismatch_error(
            output_index,
            Some(sample_index),
            "maintained numeric carrier is not finite",
        ));
    };
    if !numeric_value_satisfies_tolerance(observed, reference, tolerance) {
        return Err(RenderDeterministicVerificationError::ToleranceMismatch {
            output_index,
            sample_index,
        });
    }
    Ok(())
}

fn require_definedness(
    output_index: usize,
    sample_index: usize,
    actual: u32,
    expected: u32,
) -> Result<(), RenderDeterministicVerificationError> {
    if actual != expected {
        return Err(physical_mismatch_error(
            output_index,
            Some(sample_index),
            "physical definedness does not match certified semantic definedness",
        ));
    }
    Ok(())
}

fn require_undefined_zero(
    output_index: usize,
    sample_index: usize,
    canonical_word: u32,
    definedness_word: u32,
    detail: &'static str,
) -> Result<(), RenderDeterministicVerificationError> {
    require_definedness(output_index, sample_index, definedness_word, 0)?;
    if canonical_word != 0 {
        return Err(physical_mismatch_error(
            output_index,
            Some(sample_index),
            detail,
        ));
    }
    Ok(())
}

fn collect_geometry(
    admitted: &AdmittedRenderPlan,
    output: &RenderAdmittedOutput,
    needs_material: bool,
) -> Result<Vec<VerificationGeometry>, RenderDeterministicVerificationError> {
    let output_index = output.output_index();
    let mut geometry = Vec::new();
    geometry
        .try_reserve_exact(output.object_representations().len())
        .map_err(|_| {
            inconclusive_error(
                output_index,
                None,
                "host allocation failed while collecting verification geometry",
            )
        })?;
    for object in output.object_representations() {
        let object_id = object.object_id();
        let representation_id = object.representation().representation_id();
        let state = admitted
            .plan()
            .scene()
            .object_state(object_id)
            .ok_or(correlation_error(
                output_index,
                None,
                "selected verification object no longer has retained spatial state",
            ))?;
        let matrix = state.spatial().local_to_scene().row_major_3x4();
        let input = admitted
            .surface_semantic_input(representation_id)
            .ok_or(correlation_error(
                output_index,
                None,
                "selected verification representation lost its admitted semantic input",
            ))?
            .execution_view();
        let reflectance = if needs_material {
            Some(
                admitted
                    .plan()
                    .scene()
                    .object_participation(object_id)
                    .and_then(|participation| participation.material_assignment())
                    .ok_or(correlation_error(
                        output_index,
                        None,
                        "radiance verification object has no retained admitted material",
                    ))?
                    .material()
                    .reflectance(),
            )
        } else {
            None
        };
        geometry.push(VerificationGeometry {
            object_id,
            translation_scene: [matrix[3], matrix[7], matrix[11]],
            surface: input,
            reflectance,
        });
    }
    Ok(geometry)
}

fn certified_sample_ray(
    observation: RenderObservationSpec,
    topology: RenderResultTopology,
    sample_index: usize,
) -> Option<CertifiedRay> {
    let transform = match observation {
        RenderObservationSpec::Perspective(observation) => observation.observation_to_scene(),
        RenderObservationSpec::Probe(observation) => observation.observation_to_scene(),
    };
    let matrix = transform.row_major_3x4();
    let origin_scene = interval_vec3([matrix[3], matrix[7], matrix[11]])?;
    let direction_scene = match observation {
        RenderObservationSpec::Probe(_) => {
            if !topology.is_scalar() || sample_index != 0 {
                return None;
            }
            interval_vec3([0.0, 0.0, -1.0])?
        }
        RenderObservationSpec::Perspective(observation) => {
            let (width, height) = topology.sample_lattice_dimensions()?;
            let width_usize = usize::try_from(width).ok()?;
            let height_usize = usize::try_from(height).ok()?;
            let total = width_usize.checked_mul(height_usize)?;
            if sample_index >= total {
                return None;
            }
            let x = sample_index % width_usize;
            let y = sample_index / width_usize;
            perspective_direction(
                u32::try_from(x).ok()?,
                u32::try_from(y).ok()?,
                width,
                height,
                observation.vertical_field_of_view_radians(),
                observation.aspect_ratio(),
            )?
        }
    };
    Some(CertifiedRay {
        origin_scene,
        direction_scene,
    })
}

fn perspective_direction(
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    full_fov_radians: f64,
    aspect_ratio: f64,
) -> Option<[VerificationInterval; 3]> {
    let half = singleton(0.5)?;
    let one = singleton(1.0)?;
    let two = singleton(2.0)?;
    let u = singleton(f64::from(x))?
        .add(half)?
        .div(singleton(f64::from(width))?)?;
    let v = singleton(f64::from(y))?
        .add(half)?
        .div(singleton(f64::from(height))?)?;
    let tangent = certified_tan_half_fov(full_fov_radians)?;
    let x_component = u
        .mul(two)?
        .sub(one)?
        .mul(tangent)?
        .mul(singleton(aspect_ratio)?)?;
    let y_component = one.sub(v.mul(two)?)?.mul(tangent)?;
    normalize([x_component, y_component, singleton(-1.0)?])
}

fn nearest_primary_intersection(
    ray: CertifiedRay,
    geometry: &[VerificationGeometry],
) -> Option<CertifiedIntersection> {
    let mut hits = Vec::new();
    hits.try_reserve_exact(geometry.len()).ok()?;
    for geometry in geometry {
        match intersect(ray, *geometry)? {
            CertifiedIntersection::Miss => {}
            CertifiedIntersection::Hit(hit) => hits.push(hit),
        }
    }
    if hits.is_empty() {
        return Some(CertifiedIntersection::Miss);
    }
    if hits.len() == 1 {
        return Some(CertifiedIntersection::Hit(hits[0]));
    }

    let mut winner = None;
    for (index, candidate) in hits.iter().copied().enumerate() {
        let uniquely_nearest = hits.iter().enumerate().all(|(other_index, other)| {
            index == other_index || candidate.distance.upper() < other.distance.lower()
        });
        if uniquely_nearest {
            if winner.is_some() {
                return None;
            }
            winner = Some(candidate);
        }
    }
    winner.map(CertifiedIntersection::Hit)
}

fn intersect(ray: CertifiedRay, geometry: VerificationGeometry) -> Option<CertifiedIntersection> {
    match geometry.surface {
        RenderSurfaceSemanticInputView::Sphere {
            center_local_units,
            radius_local_units,
        } => intersect_sphere(ray, geometry, center_local_units, radius_local_units),
        RenderSurfaceSemanticInputView::Plane {
            point_local_units,
            normal_local,
        } => intersect_plane(ray, geometry, point_local_units, normal_local),
    }
}

fn intersect_sphere(
    ray: CertifiedRay,
    geometry: VerificationGeometry,
    center_local_units: [f64; 3],
    radius_local_units: f64,
) -> Option<CertifiedIntersection> {
    let center_scene = translated_point(center_local_units, geometry.translation_scene)?;
    let relative = vec_sub(ray.origin_scene, center_scene)?;
    let a = dot(ray.direction_scene, ray.direction_scene)?;
    if a.lower() <= 0.0 {
        return None;
    }
    let two = singleton(2.0)?;
    let four = singleton(4.0)?;
    let b = dot(relative, ray.direction_scene)?.mul(two)?;
    let radius = singleton(radius_local_units)?;
    let c = dot(relative, relative)?.sub(radius.mul(radius)?)?;
    let discriminant = b.mul(b)?.sub(four.mul(a)?.mul(c)?)?;
    if discriminant.upper() < 0.0 {
        return Some(CertifiedIntersection::Miss);
    }
    if discriminant.lower() <= 0.0 {
        return None;
    }
    let root = discriminant.sqrt()?;
    let denominator = a.mul(two)?;
    if denominator.lower() <= 0.0 {
        return None;
    }
    let negative_b = b.neg();
    let first = negative_b.sub(root)?.div(denominator)?;
    let second = negative_b.add(root)?.div(denominator)?;
    let distance = if first.lower() >= 0.0 {
        first
    } else if first.upper() < 0.0 {
        if second.lower() >= 0.0 {
            second
        } else if second.upper() < 0.0 {
            return Some(CertifiedIntersection::Miss);
        } else {
            return None;
        }
    } else {
        return None;
    };
    certified_hit(ray, geometry, distance, center_scene)
}

fn intersect_plane(
    ray: CertifiedRay,
    geometry: VerificationGeometry,
    point_local_units: [f64; 3],
    normal_local: [f64; 3],
) -> Option<CertifiedIntersection> {
    let point_scene = translated_point(point_local_units, geometry.translation_scene)?;
    let normal_scene = normalize(interval_vec3(normal_local)?)?;
    let denominator = dot(normal_scene, ray.direction_scene)?;
    if denominator.lower() <= 0.0 && denominator.upper() >= 0.0 {
        return None;
    }
    let numerator = dot(vec_sub(point_scene, ray.origin_scene)?, normal_scene)?;
    let distance = numerator.div(denominator)?;
    if distance.upper() < 0.0 {
        return Some(CertifiedIntersection::Miss);
    }
    if distance.lower() < 0.0 {
        return None;
    }
    let position_scene = vec_add(ray.origin_scene, vec_scale(ray.direction_scene, distance)?)?;
    Some(CertifiedIntersection::Hit(CertifiedHit {
        object_id: geometry.object_id,
        distance,
        position_scene,
        normal_scene,
        reflectance: geometry.reflectance,
    }))
}

fn certified_hit(
    ray: CertifiedRay,
    geometry: VerificationGeometry,
    distance: VerificationInterval,
    center_scene: [VerificationInterval; 3],
) -> Option<CertifiedIntersection> {
    let position_scene = vec_add(ray.origin_scene, vec_scale(ray.direction_scene, distance)?)?;
    let normal_scene = normalize(vec_sub(position_scene, center_scene)?)?;
    Some(CertifiedIntersection::Hit(CertifiedHit {
        object_id: geometry.object_id,
        distance,
        position_scene,
        normal_scene,
        reflectance: geometry.reflectance,
    }))
}

fn direct_radiance(
    admitted: &AdmittedRenderPlan,
    geometry: &[VerificationGeometry],
    hit: CertifiedHit,
    wavelength_meters: f64,
) -> Option<VerificationInterval> {
    let reflectance = hit.reflectance?;
    let mut total = singleton(0.0)?;
    for object_id in admitted.plan().scene().object_ids() {
        let Some(emitter) = admitted
            .plan()
            .scene()
            .object_participation(object_id)
            .and_then(|participation| participation.emitter())
        else {
            continue;
        };
        if emitter.wavelength_meters() != wavelength_meters {
            continue;
        }
        total = total.add(directional_contribution(
            geometry,
            hit,
            reflectance,
            emitter,
        )?)?;
    }
    Some(total)
}

fn directional_contribution(
    geometry: &[VerificationGeometry],
    hit: CertifiedHit,
    reflectance: f64,
    emitter: RenderDirectionalEmitter,
) -> Option<VerificationInterval> {
    if reflectance == 0.0 || emitter.spectral_irradiance_w_m3() == 0.0 {
        return singleton(0.0);
    }
    let direction = normalize(interval_vec3(emitter.direction_to_source_scene())?)?;
    let cosine = dot(hit.normal_scene, direction)?;
    if cosine.upper() <= 0.0 {
        return singleton(0.0);
    }
    if cosine.lower() <= 0.0 {
        return None;
    }
    if visibility_blocked(hit.position_scene, direction, hit.object_id, geometry)? {
        return singleton(0.0);
    }
    singleton(reflectance)?
        .mul(singleton(emitter.spectral_irradiance_w_m3())?)?
        .mul(cosine)?
        .div(mathematical_pi_interval())
}

fn visibility_blocked(
    origin_scene: [VerificationInterval; 3],
    direction_scene: [VerificationInterval; 3],
    ignored_object: RenderObjectId,
    geometry: &[VerificationGeometry],
) -> Option<bool> {
    let ray = CertifiedRay {
        origin_scene,
        direction_scene,
    };
    let mut ambiguous = false;
    for geometry in geometry {
        if geometry.object_id == ignored_object {
            continue;
        }
        match intersect(ray, *geometry) {
            Some(CertifiedIntersection::Hit(_)) => return Some(true),
            Some(CertifiedIntersection::Miss) => {}
            None => ambiguous = true,
        }
    }
    if ambiguous { None } else { Some(false) }
}

fn sample_count(topology: RenderResultTopology) -> Option<usize> {
    match topology.sample_lattice_dimensions() {
        Some((width, height)) => usize::try_from(width)
            .ok()?
            .checked_mul(usize::try_from(height).ok()?),
        None => Some(1),
    }
}

fn translated_point(point: [f64; 3], translation: [f64; 3]) -> Option<[VerificationInterval; 3]> {
    vec_add(interval_vec3(point)?, interval_vec3(translation)?)
}

fn singleton(value: f64) -> Option<VerificationInterval> {
    VerificationInterval::singleton(value)
}

fn interval_vec3(values: [f64; 3]) -> Option<[VerificationInterval; 3]> {
    Some([
        singleton(values[0])?,
        singleton(values[1])?,
        singleton(values[2])?,
    ])
}

fn vec_add(
    left: [VerificationInterval; 3],
    right: [VerificationInterval; 3],
) -> Option<[VerificationInterval; 3]> {
    Some([
        left[0].add(right[0])?,
        left[1].add(right[1])?,
        left[2].add(right[2])?,
    ])
}

fn vec_sub(
    left: [VerificationInterval; 3],
    right: [VerificationInterval; 3],
) -> Option<[VerificationInterval; 3]> {
    Some([
        left[0].sub(right[0])?,
        left[1].sub(right[1])?,
        left[2].sub(right[2])?,
    ])
}

fn vec_scale(
    vector: [VerificationInterval; 3],
    factor: VerificationInterval,
) -> Option<[VerificationInterval; 3]> {
    Some([
        vector[0].mul(factor)?,
        vector[1].mul(factor)?,
        vector[2].mul(factor)?,
    ])
}

fn dot(
    left: [VerificationInterval; 3],
    right: [VerificationInterval; 3],
) -> Option<VerificationInterval> {
    left[0]
        .mul(right[0])?
        .add(left[1].mul(right[1])?)?
        .add(left[2].mul(right[2])?)
}

fn normalize(vector: [VerificationInterval; 3]) -> Option<[VerificationInterval; 3]> {
    let magnitude_squared = dot(vector, vector)?;
    if magnitude_squared.lower() <= 0.0 {
        return None;
    }
    let magnitude = magnitude_squared.sqrt()?;
    Some([
        vector[0].div(magnitude)?,
        vector[1].div(magnitude)?,
        vector[2].div(magnitude)?,
    ])
}

fn correlation_error(
    output_index: usize,
    sample_index: Option<usize>,
    detail: &'static str,
) -> RenderDeterministicVerificationError {
    RenderDeterministicVerificationError::Correlation {
        output_index,
        sample_index,
        detail,
    }
}

fn inconclusive_error(
    output_index: usize,
    sample_index: Option<usize>,
    detail: &'static str,
) -> RenderDeterministicVerificationError {
    RenderDeterministicVerificationError::Inconclusive {
        output_index,
        sample_index,
        detail,
    }
}

fn physical_mismatch_error(
    output_index: usize,
    sample_index: Option<usize>,
    detail: &'static str,
) -> RenderDeterministicVerificationError {
    RenderDeterministicVerificationError::PhysicalMismatch {
        output_index,
        sample_index,
        detail,
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::scene::RenderSceneStore;
    use super::*;

    fn object_id() -> RenderObjectId {
        let mut store = RenderSceneStore::new();
        store.allocate_object_id().expect("verification object id")
    }

    fn ray(origin: [f64; 3], direction: [f64; 3]) -> CertifiedRay {
        CertifiedRay {
            origin_scene: interval_vec3(origin).expect("finite origin"),
            direction_scene: normalize(interval_vec3(direction).expect("finite direction"))
                .expect("nonzero direction"),
        }
    }

    fn sphere(object_id: RenderObjectId, z: f64, radius: f64) -> VerificationGeometry {
        VerificationGeometry {
            object_id,
            translation_scene: [0.0, 0.0, z],
            surface: RenderSurfaceSemanticInputView::Sphere {
                center_local_units: [0.0, 0.0, 0.0],
                radius_local_units: radius,
            },
            reflectance: None,
        }
    }

    #[test]
    fn sphere_hit_and_miss_are_conservatively_distinguished() {
        let geometry = sphere(object_id(), -3.0, 1.0);
        let hit = intersect(ray([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]), geometry)
            .expect("axis-aligned sphere branch must certify");
        let CertifiedIntersection::Hit(hit) = hit else {
            panic!("axis-aligned ray must hit sphere");
        };
        assert!(hit.distance.contains(2.0));

        assert!(matches!(
            intersect(ray([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]), geometry),
            Some(CertifiedIntersection::Miss)
        ));
    }

    #[test]
    fn exact_parallel_plane_and_overlapping_nearest_hits_fail_closed() {
        let plane = VerificationGeometry {
            object_id: object_id(),
            translation_scene: [0.0, 0.0, -3.0],
            surface: RenderSurfaceSemanticInputView::Plane {
                point_local_units: [0.0, 0.0, 0.0],
                normal_local: [0.0, 1.0, 0.0],
            },
            reflectance: None,
        };
        assert!(
            intersect(ray([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]), plane).is_none(),
            "parallel plane branch must be inconclusive rather than inventing a hit/miss epsilon"
        );

        let first = sphere(object_id(), -3.0, 1.0);
        let second = sphere(object_id(), -3.0, 1.0);
        assert!(
            nearest_primary_intersection(ray([0.0, 0.0, 0.0], [0.0, 0.0, -1.0]), &[first, second],)
                .is_none(),
            "overlapping nearest surfaces must fail closed"
        );
    }
}
