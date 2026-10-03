use super::super::program::{
    CAMERA_REPROJECTION_REVISION, TEMPORAL_RECONSTRUCTION_REVISION,
    abi::{
        WORKGROUP_SIZE, camera, emitter, execution_mode, geometry, header, observation_kind, shape,
        temporal,
    },
};
use super::super::transform::{
    RenderCompiledMetricSimilarityTransform, RenderCompiledMetricSimilarityTransformError,
    RenderCompiledObjectTransform, RenderCompiledObjectTransformError,
};
use super::errors::RenderDeterministicLoweringError;
use crate::admission::AdmittedRenderPlan;
use crate::field_input::RenderFieldSemanticInput;
use crate::representation::RenderRepresentationProtocol;
use crate::request::{
    RenderDistanceConvention, RenderObservationSpec, RenderOutputValue,
    RenderPerspectiveObservation,
};
use crate::scene::RenderObjectId;
use crate::surface_input::RenderSurfaceSemanticInputView;
use runen_gpu::GpuDispatchSize;
use std::collections::BTreeMap;

use super::WORD_BYTES;

pub(super) struct PackedOutput {
    pub(super) input_words: Vec<u32>,
    pub(super) sample_count: u32,
    pub(super) output_byte_len: u64,
    pub(super) texture_row_bytes: Option<u32>,
}

/// Physical execution selection; private coverage is not a requested semantic depth output.
#[derive(Debug, Clone, Copy)]
pub(super) enum MaintainedExecutionKind {
    Semantic(RenderOutputValue),
    RequestedCoverage,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct OutputTemporalPackingFacts {
    pub(super) phase: u32,
    pub(super) age: u32,
    /// Static histories retain their own physical row stride. Camera histories use the
    /// current output layout, so `None` deliberately means "use current row stride".
    pub(super) static_history_row_stride_words: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct OutputPackingInput {
    pub(super) finite_evaluation_extent: Option<(u32, u32)>,
    pub(super) bytes_per_row_alignment: Option<u64>,
    pub(super) temporal: Option<OutputTemporalPackingFacts>,
}

#[derive(Debug, Clone, Copy)]
struct OutputPhysicalLayout {
    requested_extent: (u32, u32),
    sample_count: u32,
    width: u32,
    height: u32,
    row_stride_words: u32,
    output_byte_len: u64,
    texture_row_bytes: Option<u32>,
}

#[derive(Debug, Clone, Copy)]
struct PhysicalObservation {
    kind: u32,
    transform: crate::space_time::RenderAffineTransform3,
    tan_half_fov: Option<f64>,
    aspect_ratio: Option<f64>,
}

enum GeometryPackingEntry<'a> {
    Surface {
        object_code: u32,
        reflectance: Option<f64>,
        transform: RenderCompiledObjectTransform,
        input: RenderSurfaceSemanticInputView,
    },
    Field {
        object_code: u32,
        reflectance: Option<f64>,
        transform: RenderCompiledMetricSimilarityTransform,
        input: &'a RenderFieldSemanticInput,
        sample_offset: usize,
    },
}

struct OutputPackingPlan<'a> {
    output_index: usize,
    layout: OutputPhysicalLayout,
    execution_mode: u32,
    observation: PhysicalObservation,
    geometry: Vec<GeometryPackingEntry<'a>>,
    emitters: Vec<crate::appearance::RenderDirectionalEmitter>,
    emitter_offset: usize,
    total_words: usize,
    temporal: Option<OutputTemporalPackingFacts>,
}

pub(super) fn pack_output(
    admitted: &AdmittedRenderPlan,
    admitted_output: &crate::admission::RenderAdmittedOutput,
    execution_kind: MaintainedExecutionKind,
    observation: RenderObservationSpec,
    object_codes: &BTreeMap<RenderObjectId, u32>,
    packing: OutputPackingInput,
) -> Result<PackedOutput, RenderDeterministicLoweringError> {
    let plan = plan_output_packing(
        admitted,
        admitted_output,
        execution_kind,
        observation,
        object_codes,
        packing,
    )?;
    encode_output_packing(plan)
}

fn derive_output_physical_layout(
    output_index: usize,
    requested_extent: Option<(u32, u32)>,
    finite_evaluation_extent: Option<(u32, u32)>,
    bytes_per_row_alignment: Option<u64>,
) -> Result<OutputPhysicalLayout, RenderDeterministicLoweringError> {
    let is_lattice = requested_extent.is_some();
    let physical_extent = match (requested_extent, finite_evaluation_extent) {
        (Some(_), Some(extent)) => extent,
        (Some(extent), None) => extent,
        (None, Some(_)) => {
            return Err(RenderDeterministicLoweringError::UnsupportedOutput { output_index });
        }
        (None, None) => (1, 1),
    };
    let requested_extent = requested_extent.unwrap_or((1, 1));
    if is_lattice {
        let (width, height) = physical_extent;
        let sample_count =
            width
                .checked_mul(height)
                .ok_or(RenderDeterministicLoweringError::SizeOverflow {
                    field: "lattice sample count",
                })?;
        let logical_row_bytes = u64::from(width).checked_mul(WORD_BYTES).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "lattice logical row bytes",
            },
        )?;
        let alignment = bytes_per_row_alignment
            .ok_or(RenderDeterministicLoweringError::MissingBytesPerRowAlignment)?;
        let row_bytes = super::layout::align_up(logical_row_bytes, alignment)?;
        if row_bytes % WORD_BYTES != 0 {
            return Err(
                RenderDeterministicLoweringError::InvalidBytesPerRowAlignment { alignment },
            );
        }
        let row_stride_words = u32::try_from(row_bytes / WORD_BYTES).map_err(|_| {
            RenderDeterministicLoweringError::SizeOverflow {
                field: "lattice row stride",
            }
        })?;
        let output_words = row_stride_words.checked_mul(height).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "canonical lattice word count",
            },
        )?;
        let output_byte_len = u64::from(output_words).checked_mul(WORD_BYTES).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "canonical lattice byte length",
            },
        )?;
        let texture_row_bytes = u32::try_from(row_bytes).map_err(|_| {
            RenderDeterministicLoweringError::SizeOverflow {
                field: "lattice row bytes",
            }
        })?;
        Ok(OutputPhysicalLayout {
            requested_extent,
            sample_count,
            width,
            height,
            row_stride_words,
            output_byte_len,
            texture_row_bytes: Some(texture_row_bytes),
        })
    } else {
        Ok(OutputPhysicalLayout {
            requested_extent,
            sample_count: 1,
            width: 1,
            height: 1,
            row_stride_words: 1,
            output_byte_len: WORD_BYTES,
            texture_row_bytes: None,
        })
    }
}

fn execution_realization(
    output_index: usize,
    execution_kind: MaintainedExecutionKind,
) -> Result<(u32, Option<f64>), RenderDeterministicLoweringError> {
    match execution_kind {
        MaintainedExecutionKind::RequestedCoverage => {
            Ok((execution_mode::REQUESTED_COVERAGE, None))
        }
        MaintainedExecutionKind::Semantic(value) => match value {
            RenderOutputValue::Radiance { representation } => Ok((
                execution_mode::RADIANCE,
                Some(representation.wavelength_meters()),
            )),
            RenderOutputValue::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth,
            } => Ok((execution_mode::FORWARD_DEPTH, None)),
            RenderOutputValue::ObjectIdentity => Ok((execution_mode::OBJECT_IDENTITY, None)),
            RenderOutputValue::Distance { .. } => {
                Err(RenderDeterministicLoweringError::UnsupportedOutput { output_index })
            }
        },
    }
}

fn physical_observation(observation: RenderObservationSpec) -> PhysicalObservation {
    match observation {
        RenderObservationSpec::Perspective(observation) => PhysicalObservation {
            kind: if observation.sampling_support().is_perspective_lattice_cell() {
                observation_kind::PERSPECTIVE_FOOTPRINT
            } else {
                observation_kind::PERSPECTIVE
            },
            transform: observation.observation_to_scene(),
            tan_half_fov: Some((observation.vertical_field_of_view_radians() * 0.5).tan()),
            aspect_ratio: Some(observation.aspect_ratio()),
        },
        RenderObservationSpec::Probe(observation) => PhysicalObservation {
            kind: observation_kind::PROBE,
            transform: observation.observation_to_scene(),
            tan_half_fov: None,
            aspect_ratio: None,
        },
    }
}

fn plan_output_packing<'a>(
    admitted: &'a AdmittedRenderPlan,
    admitted_output: &crate::admission::RenderAdmittedOutput,
    execution_kind: MaintainedExecutionKind,
    observation: RenderObservationSpec,
    object_codes: &BTreeMap<RenderObjectId, u32>,
    packing: OutputPackingInput,
) -> Result<OutputPackingPlan<'a>, RenderDeterministicLoweringError> {
    let output_index = admitted_output.output_index();
    let requested_extent = admitted.plan().request().outputs()[output_index]
        .spec()
        .topology()
        .sample_lattice_dimensions();
    let layout = derive_output_physical_layout(
        output_index,
        requested_extent,
        packing.finite_evaluation_extent,
        packing.bytes_per_row_alignment,
    )?;
    let (execution_mode, wavelength) = execution_realization(output_index, execution_kind)?;
    let observation = physical_observation(observation);

    let mut geometry_keys = Vec::new();
    geometry_keys
        .try_reserve_exact(admitted_output.object_representations().len())
        .map_err(|_| RenderDeterministicLoweringError::HostAllocation {
            field: "maintained geometry records",
        })?;
    for object in admitted_output.object_representations() {
        let representation_id = object.representation().representation_id();
        let protocol = object.representation().requirement().protocol().protocol();
        match protocol {
            RenderRepresentationProtocol::SurfaceQuery
            | RenderRepresentationProtocol::OrientedSurfaceQuery => {
                admitted.surface_semantic_input(representation_id).ok_or(
                    RenderDeterministicLoweringError::MissingSurfaceInput {
                        output_index,
                        object_id: object.object_id(),
                        representation_id,
                    },
                )?;
            }
            RenderRepresentationProtocol::FieldDistance => {
                admitted.field_semantic_input(representation_id).ok_or(
                    RenderDeterministicLoweringError::MissingFieldInput {
                        output_index,
                        object_id: object.object_id(),
                        representation_id,
                    },
                )?;
            }
        }
        geometry_keys.push((object.object_id(), representation_id, protocol));
    }
    geometry_keys.sort_by_key(|(object_id, representation_id, _)| (*object_id, *representation_id));

    let emitters = if let Some(wavelength) = wavelength {
        matching_emitters(admitted, wavelength)?
    } else {
        Vec::new()
    };
    let emitter_offset = header::WORDS
        .checked_add(geometry_keys.len().checked_mul(geometry::WORDS).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "geometry input words",
            },
        )?)
        .ok_or(RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter input offset",
        })?;
    let emitter_words = emitters.len().checked_mul(emitter::WORDS).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter input words",
        },
    )?;
    let mut field_sample_cursor = emitter_offset.checked_add(emitter_words).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "field sample input offset",
        },
    )?;

    let mut geometry = Vec::new();
    geometry
        .try_reserve_exact(geometry_keys.len())
        .map_err(|_| RenderDeterministicLoweringError::HostAllocation {
            field: "maintained geometry packing plan",
        })?;

    for (object_id, representation_id, protocol) in geometry_keys {
        let state = admitted.plan().scene().object_state(object_id).ok_or(
            RenderDeterministicLoweringError::MissingObjectState {
                output_index,
                object_id,
            },
        )?;
        let object_code = *object_codes
            .get(&object_id)
            .ok_or(RenderDeterministicLoweringError::OutputCorrelationChanged { output_index })?;
        let reflectance = if execution_mode == execution_mode::RADIANCE {
            Some(
                admitted
                    .plan()
                    .scene()
                    .object_participation(object_id)
                    .and_then(|participation| participation.material_assignment())
                    .ok_or(RenderDeterministicLoweringError::MissingMaterial {
                        output_index,
                        object_id,
                    })?
                    .material()
                    .reflectance(),
            )
        } else {
            None
        };

        match protocol {
            RenderRepresentationProtocol::SurfaceQuery
            | RenderRepresentationProtocol::OrientedSurfaceQuery => {
                let transform = RenderCompiledObjectTransform::compile(state.spatial()).map_err(
                    |RenderCompiledObjectTransformError::NonInvertibleObjectTransform| {
                        RenderDeterministicLoweringError::NonInvertibleObjectTransform {
                            output_index,
                            object_id,
                        }
                    },
                )?;
                let input = admitted
                    .surface_semantic_input(representation_id)
                    .ok_or(RenderDeterministicLoweringError::MissingSurfaceInput {
                        output_index,
                        object_id,
                        representation_id,
                    })?
                    .execution_view();
                geometry.push(GeometryPackingEntry::Surface {
                    object_code,
                    reflectance,
                    transform,
                    input,
                });
            }
            RenderRepresentationProtocol::FieldDistance => {
                let transform = RenderCompiledMetricSimilarityTransform::compile(state.spatial())
                    .map_err(
                    |RenderCompiledMetricSimilarityTransformError::NotPositiveSimilarity| {
                        RenderDeterministicLoweringError::NonSimilarityFieldTransform {
                            output_index,
                            object_id,
                        }
                    },
                )?;
                let input = admitted.field_semantic_input(representation_id).ok_or(
                    RenderDeterministicLoweringError::MissingFieldInput {
                        output_index,
                        object_id,
                        representation_id,
                    },
                )?;
                let sample_offset = field_sample_cursor;
                field_sample_cursor = field_sample_cursor
                    .checked_add(input.sample_count())
                    .ok_or(RenderDeterministicLoweringError::SizeOverflow {
                        field: "field sample input words",
                    })?;
                geometry.push(GeometryPackingEntry::Field {
                    object_code,
                    reflectance,
                    transform,
                    input,
                    sample_offset,
                });
            }
        }
    }

    Ok(OutputPackingPlan {
        output_index,
        layout,
        execution_mode,
        observation,
        geometry,
        emitters,
        emitter_offset,
        total_words: field_sample_cursor,
        temporal: packing.temporal,
    })
}

fn encode_output_packing(
    plan: OutputPackingPlan<'_>,
) -> Result<PackedOutput, RenderDeterministicLoweringError> {
    let OutputPackingPlan {
        output_index,
        layout,
        execution_mode,
        observation,
        geometry,
        emitters,
        emitter_offset,
        total_words,
        temporal,
    } = plan;
    let mut words = Vec::new();
    words.try_reserve_exact(total_words).map_err(|_| {
        RenderDeterministicLoweringError::HostAllocation {
            field: "packed maintained semantic input",
        }
    })?;
    words.resize(total_words, 0_u32);

    words[header::SAMPLE_COUNT] = layout.sample_count;
    words[header::EVALUATION_WIDTH] = layout.width;
    words[header::EVALUATION_HEIGHT] = layout.height;
    words[header::ROW_STRIDE_WORDS] = layout.row_stride_words;
    words[header::GEOMETRY_COUNT] = u32::try_from(geometry.len()).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "geometry count",
        }
    })?;
    words[header::EMITTER_COUNT] = u32::try_from(emitters.len()).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter count",
        }
    })?;
    words[header::EXECUTION_MODE] = execution_mode;
    words[header::OBSERVATION_KIND] = observation.kind;
    pack_observation(
        &mut words,
        observation.transform,
        observation.tan_half_fov,
        observation.aspect_ratio,
    )?;
    words[header::REQUESTED_WIDTH] = layout.requested_extent.0;
    words[header::REQUESTED_HEIGHT] = layout.requested_extent.1;
    words[header::TEMPORAL_PHASE] = temporal.map_or(0, |history| history.phase);
    words[header::TEMPORAL_SEQUENCE_REVISION] = temporal::SEQUENCE_REVISION;
    words[header::TEMPORAL_HISTORY_AGE] = temporal.map_or(0, |history| history.age);
    words[header::TEMPORAL_HISTORY_ROW_STRIDE] = temporal
        .and_then(|history| history.static_history_row_stride_words)
        .unwrap_or(layout.row_stride_words);
    words[header::TEMPORAL_RECONSTRUCTION_REVISION] = TEMPORAL_RECONSTRUCTION_REVISION;
    words[header::EMITTER_OFFSET] = u32::try_from(emitter_offset).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter input offset",
        }
    })?;

    for (index, entry) in geometry.into_iter().enumerate() {
        let base = header::WORDS + index * geometry::WORDS;
        let (object_code, reflectance) = match &entry {
            GeometryPackingEntry::Surface {
                object_code,
                reflectance,
                ..
            }
            | GeometryPackingEntry::Field {
                object_code,
                reflectance,
                ..
            } => (*object_code, *reflectance),
        };
        words[base + geometry::OBJECT_CODE] = object_code;
        words[base + geometry::RADIANCE_REFLECTANCE] = match reflectance {
            Some(value) => f32_bits(value, "diffuse reflectance")?,
            None => 0,
        };

        match entry {
            GeometryPackingEntry::Surface {
                transform, input, ..
            } => {
                pack_invertible_matrix3(
                    &mut words,
                    base + geometry::SCENE_TO_LOCAL,
                    transform.scene_to_local_units_row_major(),
                    "object scene-to-local transform",
                )?;
                pack_vec3(
                    &mut words,
                    base + geometry::TRANSLATION,
                    transform.translation_scene(),
                )?;
                pack_matrix3(
                    &mut words,
                    base + geometry::NORMAL_LOCAL_TO_SCENE,
                    transform.normal_local_to_scene_row_major(),
                )?;
                match input {
                    RenderSurfaceSemanticInputView::Sphere {
                        center_local_units,
                        radius_local_units,
                    } => {
                        words[base + geometry::SHAPE] = shape::SPHERE;
                        pack_vec3(&mut words, base + geometry::SHAPE_DATA, center_local_units)?;
                        words[base + geometry::SPHERE_RADIUS] =
                            positive_f32_bits(radius_local_units, "surface-input sphere radius")?;
                    }
                    RenderSurfaceSemanticInputView::Plane {
                        point_local_units,
                        normal_local,
                    } => {
                        words[base + geometry::SHAPE] = shape::PLANE;
                        pack_vec3(&mut words, base + geometry::SHAPE_DATA, point_local_units)?;
                        pack_vec3(&mut words, base + geometry::PLANE_NORMAL, normal_local)?;
                    }
                }
            }
            GeometryPackingEntry::Field {
                transform,
                input,
                sample_offset,
                ..
            } => {
                pack_invertible_matrix3(
                    &mut words,
                    base + geometry::SCENE_TO_LOCAL,
                    transform.scene_to_local_meters_row_major(),
                    "field scene-to-local metric transform",
                )?;
                pack_vec3(
                    &mut words,
                    base + geometry::TRANSLATION,
                    transform.translation_scene(),
                )?;
                pack_matrix3(
                    &mut words,
                    base + geometry::NORMAL_LOCAL_TO_SCENE,
                    transform.normal_local_to_scene_row_major(),
                )?;
                words[base + geometry::SHAPE] = shape::FIELD;
                words[base + geometry::FIELD_SCENE_SCALE] = conservative_positive_f32_bits(
                    transform.scene_meters_per_local_meter(),
                    "field scene metres per local metre",
                )?;
                pack_vec3(
                    &mut words,
                    base + geometry::FIELD_ORIGIN,
                    input.origin_local_meters(),
                )?;
                pack_vec3(
                    &mut words,
                    base + geometry::FIELD_SAMPLE_SPACING,
                    input.sample_spacing_meters(),
                )?;
                let dimensions = input.dimensions();
                words[base + geometry::FIELD_DIMENSION_X] = dimensions[0];
                words[base + geometry::FIELD_DIMENSION_Y] = dimensions[1];
                words[base + geometry::FIELD_DIMENSION_Z] = dimensions[2];
                words[base + geometry::FIELD_SAMPLE_OFFSET] = u32::try_from(sample_offset)
                    .map_err(|_| RenderDeterministicLoweringError::SizeOverflow {
                        field: "field sample input offset",
                    })?;
                words[base + geometry::FIELD_MAX_QUERY_ERROR] = conservative_nonnegative_f32_bits(
                    input.max_absolute_query_error_local_meters(),
                    "field maximum absolute query error",
                )?;

                for sample_index in 0..input.sample_count() {
                    let sample = input.signed_distance_sample_meters(sample_index).ok_or(
                        RenderDeterministicLoweringError::OutputCorrelationChanged { output_index },
                    )?;
                    words[sample_offset + sample_index] =
                        f32_bits(sample, "field signed-distance sample")?;
                }
            }
        }
    }

    for (index, emitter) in emitters.into_iter().enumerate() {
        let base = emitter_offset + index * emitter::WORDS;
        pack_vec3(
            &mut words,
            base + emitter::DIRECTION,
            emitter.direction_to_source_scene(),
        )?;
        words[base + emitter::SPECTRAL_IRRADIANCE] = f32_bits(
            emitter.spectral_irradiance_w_m3(),
            "directional-emitter spectral irradiance",
        )?;
    }

    Ok(PackedOutput {
        input_words: words,
        sample_count: layout.sample_count,
        output_byte_len: layout.output_byte_len,
        texture_row_bytes: layout.texture_row_bytes,
    })
}

pub(super) fn matching_emitters(
    admitted: &AdmittedRenderPlan,
    wavelength_meters: f64,
) -> Result<Vec<crate::appearance::RenderDirectionalEmitter>, RenderDeterministicLoweringError> {
    let mut emitters = Vec::new();
    emitters
        .try_reserve_exact(admitted.plan().scene().len())
        .map_err(|_| RenderDeterministicLoweringError::HostAllocation {
            field: "directional-emitter realization",
        })?;
    for object_id in admitted.plan().scene().object_ids() {
        let Some(emitter) = admitted
            .plan()
            .scene()
            .object_participation(object_id)
            .and_then(|participation| participation.emitter())
        else {
            continue;
        };
        if emitter.wavelength_meters() == wavelength_meters {
            emitters.push(emitter);
        }
    }
    emitters.sort_by(|left, right| {
        left.direction_to_source_scene()
            .map(f64::to_bits)
            .cmp(&right.direction_to_source_scene().map(f64::to_bits))
            .then_with(|| {
                left.spectral_irradiance_w_m3()
                    .to_bits()
                    .cmp(&right.spectral_irradiance_w_m3().to_bits())
            })
    });

    let total = emitters.iter().try_fold(0.0_f64, |total, emitter| {
        let next = total + emitter.spectral_irradiance_w_m3();
        next.is_finite().then_some(next).ok_or(
            RenderDeterministicLoweringError::NumericRealization {
                field: "summed directional-emitter irradiance",
            },
        )
    })?;
    let _ = f32_bits(total, "summed directional-emitter irradiance")?;
    Ok(emitters)
}

pub(super) fn pack_observation(
    words: &mut [u32],
    transform: crate::space_time::RenderAffineTransform3,
    tan_half_fov: Option<f64>,
    aspect_ratio: Option<f64>,
) -> Result<(), RenderDeterministicLoweringError> {
    let matrix = transform.row_major_3x4();
    pack_vec3(
        words,
        header::OBSERVATION_TRANSLATION,
        [matrix[3], matrix[7], matrix[11]],
    )?;
    pack_invertible_matrix3(
        words,
        header::OBSERVATION_SCENE_TO_LOCAL,
        [
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ],
        "observation linear transform",
    )?;
    words[header::PERSPECTIVE_TAN_HALF_FOV] = match tan_half_fov {
        Some(value) => positive_f32_bits(value, "perspective tangent half field of view")?,
        None => 0,
    };
    words[header::PERSPECTIVE_ASPECT_RATIO] = match aspect_ratio {
        Some(value) => positive_f32_bits(value, "perspective aspect ratio")?,
        None => 1.0_f32.to_bits(),
    };
    Ok(())
}

pub(super) fn pack_vec3(
    words: &mut [u32],
    base: usize,
    values: [f64; 3],
) -> Result<(), RenderDeterministicLoweringError> {
    words[base] = f32_bits(values[0], "packed vector component")?;
    words[base + 1] = f32_bits(values[1], "packed vector component")?;
    words[base + 2] = f32_bits(values[2], "packed vector component")?;
    Ok(())
}

pub(super) fn pack_matrix3(
    words: &mut [u32],
    base: usize,
    values: [f64; 9],
) -> Result<(), RenderDeterministicLoweringError> {
    for (offset, value) in values.into_iter().enumerate() {
        words[base + offset] = f32_bits(value, "packed transform component")?;
    }
    Ok(())
}

pub(super) fn pack_invertible_matrix3(
    words: &mut [u32],
    base: usize,
    values: [f64; 9],
    field: &'static str,
) -> Result<(), RenderDeterministicLoweringError> {
    let mut physical = [0.0_f32; 9];
    for (slot, value) in physical.iter_mut().zip(values) {
        let narrowed = value as f32;
        if !value.is_finite() || !narrowed.is_finite() {
            return Err(RenderDeterministicLoweringError::NumericRealization { field });
        }
        *slot = narrowed;
    }
    let determinant = physical[0] * (physical[4] * physical[8] - physical[5] * physical[7])
        - physical[1] * (physical[3] * physical[8] - physical[5] * physical[6])
        + physical[2] * (physical[3] * physical[7] - physical[4] * physical[6]);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    for (offset, value) in physical.into_iter().enumerate() {
        words[base + offset] = value.to_bits();
    }
    Ok(())
}

pub(super) fn f32_bits(
    value: f64,
    field: &'static str,
) -> Result<u32, RenderDeterministicLoweringError> {
    let physical = value as f32;
    if !value.is_finite() || !physical.is_finite() {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    Ok(physical.to_bits())
}

pub(super) fn positive_f32_bits(
    value: f64,
    field: &'static str,
) -> Result<u32, RenderDeterministicLoweringError> {
    let physical = value as f32;
    if !value.is_finite() || !physical.is_finite() || physical <= 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    Ok(physical.to_bits())
}

pub(super) fn conservative_positive_f32_bits(
    value: f64,
    field: &'static str,
) -> Result<u32, RenderDeterministicLoweringError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    conservative_nonnegative_f32_bits(value, field)
}

pub(super) fn conservative_nonnegative_f32_bits(
    value: f64,
    field: &'static str,
) -> Result<u32, RenderDeterministicLoweringError> {
    if !value.is_finite() || value < 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    let mut physical = value as f32;
    if !physical.is_finite() || physical < 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    if f64::from(physical) < value {
        physical = f32::from_bits(
            physical
                .to_bits()
                .checked_add(1)
                .ok_or(RenderDeterministicLoweringError::NumericRealization { field })?,
        );
        if !physical.is_finite() {
            return Err(RenderDeterministicLoweringError::NumericRealization { field });
        }
    }
    Ok(physical.to_bits())
}

#[cfg(test)]
mod packing_layout_tests {
    use super::*;

    #[test]
    fn scalar_layout_does_not_require_row_alignment() {
        let layout =
            derive_output_physical_layout(0, None, None, None).expect("scalar physical layout");
        assert_eq!(layout.requested_extent, (1, 1));
        assert_eq!(layout.sample_count, 1);
        assert_eq!(layout.row_stride_words, 1);
        assert_eq!(layout.output_byte_len, WORD_BYTES);
        assert_eq!(layout.texture_row_bytes, None);
    }

    #[test]
    fn one_by_one_lattice_keeps_texture_row_layout() {
        let layout = derive_output_physical_layout(0, Some((1, 1)), None, Some(256))
            .expect("lattice physical layout");
        assert_eq!(layout.requested_extent, (1, 1));
        assert_eq!(layout.sample_count, 1);
        assert_eq!(layout.row_stride_words, 64);
        assert_eq!(layout.output_byte_len, 256);
        assert_eq!(layout.texture_row_bytes, Some(256));
    }

    #[test]
    fn lattice_layout_requires_row_alignment() {
        assert!(matches!(
            derive_output_physical_layout(0, Some((2, 2)), None, None),
            Err(RenderDeterministicLoweringError::MissingBytesPerRowAlignment)
        ));
    }
}
