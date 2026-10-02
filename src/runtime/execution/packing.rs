use super::*;

pub(super) fn pack_output(
    admitted: &AdmittedRenderPlan,
    admitted_output: &crate::admission::RenderAdmittedOutput,
    execution_kind: MaintainedExecutionKind,
    observation: RenderObservationSpec,
    object_codes: &BTreeMap<RenderObjectId, u32>,
    context: &GpuContext,
    packing: DeterministicOutputPackingState<'_>,
) -> Result<PackedOutput, RenderDeterministicLoweringError> {
    let DeterministicOutputPackingState {
        finite_evaluation_extent,
        temporal_history,
    } = packing;
    let output_index = admitted_output.output_index();
    let topology = admitted.plan().request().outputs()[output_index]
        .spec()
        .topology();
    let requested_extent = topology.sample_lattice_dimensions();
    let physical_extent = match (requested_extent, finite_evaluation_extent) {
        (Some(_), Some(extent)) => extent,
        (Some(extent), None) => extent,
        (None, Some(_)) => {
            return Err(RenderDeterministicLoweringError::UnsupportedOutput { output_index });
        }
        (None, None) => (1, 1),
    };
    let (sample_count, width, height, row_stride_words, output_byte_len, texture_row_bytes) =
        if requested_extent.is_some() {
            let (width, height) = physical_extent;
            let sample_count = width.checked_mul(height).ok_or(
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "lattice sample count",
                },
            )?;
            let logical_row_bytes = u64::from(width).checked_mul(WORD_BYTES).ok_or(
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "lattice logical row bytes",
                },
            )?;
            let alignment = context
                .device_facts()
                .device_limits()
                .alignments()
                .bytes_per_row
                .ok_or(RenderDeterministicLoweringError::MissingBytesPerRowAlignment)?;
            let row_bytes = align_up(logical_row_bytes, alignment)?;
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
            let row_bytes = u32::try_from(row_bytes).map_err(|_| {
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "lattice row bytes",
                }
            })?;
            (
                sample_count,
                width,
                height,
                row_stride_words,
                output_byte_len,
                Some(row_bytes),
            )
        } else {
            (1, 1, 1, 1, WORD_BYTES, None)
        };

    let (execution_mode, wavelength) = match execution_kind {
        MaintainedExecutionKind::RequestedCoverage => (EXECUTION_REQUESTED_COVERAGE, None),
        MaintainedExecutionKind::Semantic(value) => match value {
            RenderOutputValue::Radiance { representation } => {
                (OUTPUT_RADIANCE, Some(representation.wavelength_meters()))
            }
            RenderOutputValue::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth,
            } => (OUTPUT_FORWARD_DEPTH, None),
            RenderOutputValue::ObjectIdentity => (OUTPUT_OBJECT_IDENTITY, None),
            RenderOutputValue::Distance { .. } => {
                return Err(RenderDeterministicLoweringError::UnsupportedOutput { output_index });
            }
        },
    };

    let (observation_kind, transform, tan_half_fov, aspect_ratio) = match observation {
        RenderObservationSpec::Perspective(observation) => (
            if observation.sampling_support().is_perspective_lattice_cell() {
                OBSERVATION_PERSPECTIVE_FOOTPRINT
            } else {
                OBSERVATION_PERSPECTIVE
            },
            observation.observation_to_scene(),
            Some((observation.vertical_field_of_view_radians() * 0.5).tan()),
            Some(observation.aspect_ratio()),
        ),
        RenderObservationSpec::Probe(observation) => (
            OBSERVATION_PROBE,
            observation.observation_to_scene(),
            None,
            None,
        ),
    };

    let mut geometry = Vec::new();
    geometry
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
        geometry.push((object.object_id(), representation_id, protocol));
    }
    geometry.sort_by_key(|(object_id, representation_id, _)| (*object_id, *representation_id));

    let emitters = if let Some(wavelength) = wavelength {
        matching_emitters(admitted, wavelength)?
    } else {
        Vec::new()
    };
    let emitter_offset = HEADER_WORDS
        .checked_add(geometry.len().checked_mul(GEOMETRY_WORDS).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "geometry input words",
            },
        )?)
        .ok_or(RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter input offset",
        })?;
    let emitter_words = emitters.len().checked_mul(EMITTER_WORDS).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter input words",
        },
    )?;
    let field_sample_offset = emitter_offset.checked_add(emitter_words).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "field sample input offset",
        },
    )?;
    let field_sample_words = geometry.iter().try_fold(
        0_usize,
        |count, (object_id, representation_id, protocol)| {
            if !matches!(protocol, RenderRepresentationProtocol::FieldDistance) {
                return Ok(count);
            }
            let input = admitted.field_semantic_input(*representation_id).ok_or(
                RenderDeterministicLoweringError::MissingFieldInput {
                    output_index,
                    object_id: *object_id,
                    representation_id: *representation_id,
                },
            )?;
            count.checked_add(input.sample_count()).ok_or(
                RenderDeterministicLoweringError::SizeOverflow {
                    field: "field sample input words",
                },
            )
        },
    )?;
    let total_words = field_sample_offset.checked_add(field_sample_words).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "packed input words",
        },
    )?;
    let mut words = Vec::new();
    words.try_reserve_exact(total_words).map_err(|_| {
        RenderDeterministicLoweringError::HostAllocation {
            field: "packed maintained semantic input",
        }
    })?;
    words.resize(total_words, 0_u32);

    words[0] = sample_count;
    words[1] = width;
    words[2] = height;
    words[3] = row_stride_words;
    words[4] = u32::try_from(geometry.len()).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "geometry count",
        }
    })?;
    words[5] = u32::try_from(emitters.len()).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter count",
        }
    })?;
    words[6] = execution_mode;
    words[7] = observation_kind;
    pack_observation(&mut words, transform, tan_half_fov, aspect_ratio)?;
    let requested_extent = requested_extent.unwrap_or((1, 1));
    words[22] = requested_extent.0;
    words[23] = requested_extent.1;
    words[24] = temporal_history.map_or(0, |history| history.phase);
    words[25] = TEMPORAL_SEQUENCE_REVISION;
    words[26] = temporal_history.map_or(0, |history| history.age);
    words[27] = temporal_history.map_or(row_stride_words, |history| match &history.storage {
        DeterministicTemporalHistoryUseStorage::Static {
            row_stride_words, ..
        } => *row_stride_words,
        DeterministicTemporalHistoryUseStorage::Camera { .. } => row_stride_words,
    });
    words[28] = TEMPORAL_RECONSTRUCTION_REVISION;
    words[29] = u32::try_from(emitter_offset).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter input offset",
        }
    })?;

    let mut field_sample_cursor = field_sample_offset;
    for (index, (object_id, representation_id, protocol)) in geometry.into_iter().enumerate() {
        let base = HEADER_WORDS + index * GEOMETRY_WORDS;
        let state = admitted.plan().scene().object_state(object_id).ok_or(
            RenderDeterministicLoweringError::MissingObjectState {
                output_index,
                object_id,
            },
        )?;
        words[base + 1] = *object_codes
            .get(&object_id)
            .ok_or(RenderDeterministicLoweringError::OutputCorrelationChanged { output_index })?;
        words[base + 2] = if execution_mode == OUTPUT_RADIANCE {
            let material = admitted
                .plan()
                .scene()
                .object_participation(object_id)
                .and_then(|participation| participation.material_assignment())
                .ok_or(RenderDeterministicLoweringError::MissingMaterial {
                    output_index,
                    object_id,
                })?;
            f32_bits(material.material().reflectance(), "diffuse reflectance")?
        } else {
            0
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
                pack_invertible_matrix3(
                    &mut words,
                    base + 4,
                    transform.scene_to_local_units_row_major(),
                    "object scene-to-local transform",
                )?;
                pack_vec3(&mut words, base + 13, transform.translation_scene())?;
                pack_matrix3(
                    &mut words,
                    base + 16,
                    transform.normal_local_to_scene_row_major(),
                )?;

                let input = admitted.surface_semantic_input(representation_id).ok_or(
                    RenderDeterministicLoweringError::MissingSurfaceInput {
                        output_index,
                        object_id,
                        representation_id,
                    },
                )?;
                match input.execution_view() {
                    RenderSurfaceSemanticInputView::Sphere {
                        center_local_units,
                        radius_local_units,
                    } => {
                        words[base] = SHAPE_SPHERE;
                        pack_vec3(&mut words, base + 25, center_local_units)?;
                        words[base + 28] =
                            positive_f32_bits(radius_local_units, "surface-input sphere radius")?;
                    }
                    RenderSurfaceSemanticInputView::Plane {
                        point_local_units,
                        normal_local,
                    } => {
                        words[base] = SHAPE_PLANE;
                        pack_vec3(&mut words, base + 25, point_local_units)?;
                        pack_vec3(&mut words, base + 28, normal_local)?;
                    }
                }
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
                pack_invertible_matrix3(
                    &mut words,
                    base + 4,
                    transform.scene_to_local_meters_row_major(),
                    "field scene-to-local metric transform",
                )?;
                pack_vec3(&mut words, base + 13, transform.translation_scene())?;
                pack_matrix3(
                    &mut words,
                    base + 16,
                    transform.normal_local_to_scene_row_major(),
                )?;

                let input: &RenderFieldSemanticInput = admitted
                    .field_semantic_input(representation_id)
                    .ok_or(RenderDeterministicLoweringError::MissingFieldInput {
                        output_index,
                        object_id,
                        representation_id,
                    })?;
                words[base] = SHAPE_FIELD;
                words[base + 3] = conservative_positive_f32_bits(
                    transform.scene_meters_per_local_meter(),
                    "field scene metres per local metre",
                )?;
                pack_vec3(&mut words, base + 25, input.origin_local_meters())?;
                pack_vec3(&mut words, base + 28, input.sample_spacing_meters())?;
                let dimensions = input.dimensions();
                words[base + 31] = dimensions[0];
                words[base + 32] = dimensions[1];
                words[base + 33] = dimensions[2];
                words[base + 34] = u32::try_from(field_sample_cursor).map_err(|_| {
                    RenderDeterministicLoweringError::SizeOverflow {
                        field: "field sample input offset",
                    }
                })?;
                words[base + 35] = conservative_nonnegative_f32_bits(
                    input.max_absolute_query_error_local_meters(),
                    "field maximum absolute query error",
                )?;

                for sample_index in 0..input.sample_count() {
                    let sample = input.signed_distance_sample_meters(sample_index).ok_or(
                        RenderDeterministicLoweringError::OutputCorrelationChanged { output_index },
                    )?;
                    words[field_sample_cursor] = f32_bits(sample, "field signed-distance sample")?;
                    field_sample_cursor += 1;
                }
            }
        }
    }
    debug_assert_eq!(field_sample_cursor, total_words);

    for (index, emitter) in emitters.into_iter().enumerate() {
        let base = emitter_offset + index * EMITTER_WORDS;
        pack_vec3(&mut words, base, emitter.direction_to_source_scene())?;
        words[base + 3] = f32_bits(
            emitter.spectral_irradiance_w_m3(),
            "directional-emitter spectral irradiance",
        )?;
    }

    Ok(PackedOutput {
        input_words: words,
        sample_count,
        output_byte_len,
        texture_row_bytes,
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

pub(super) fn camera_reprojection_parameter_words(
    current: RenderPerspectiveObservation,
    previous: Option<RenderPerspectiveObservation>,
    pose_changed: bool,
    same_pose_completed_frames: u32,
) -> Result<[u32; 35], RenderDeterministicLoweringError> {
    let mut words = [0_u32; 35];
    let matrix = current.observation_to_scene().row_major_3x4();
    let inverse = invert_matrix3(
        [
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ],
        "current observation linear transform",
    )?;
    pack_matrix3(&mut words, 24, inverse)?;
    words[0] = if previous.is_some() { 1 } else { 0 };
    words[1] = if pose_changed { 1 } else { 0 };
    words[2] = CAMERA_DEPTH_POLICY_REVISION;
    words[3] = CAMERA_REPROJECTION_REVISION;
    words[4] = CAMERA_DEPTH_ABSOLUTE_EPSILON.to_bits();
    words[5] = CAMERA_DEPTH_RELATIVE_EPSILON.to_bits();
    words[23] = same_pose_completed_frames.min(TEMPORAL_PHASE_COUNT);
    if let Some(previous) = previous {
        let matrix = previous.observation_to_scene().row_major_3x4();
        pack_vec3(&mut words, 6, [matrix[3], matrix[7], matrix[11]])?;
        let inverse = invert_matrix3(
            [
                matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8],
                matrix[9], matrix[10],
            ],
            "previous observation linear transform",
        )?;
        pack_matrix3(&mut words, 9, inverse)?;
        let forward = normalize_private_vec3(
            [-matrix[2], -matrix[6], -matrix[10]],
            "previous observation forward",
        )?;
        pack_vec3(&mut words, 18, forward)?;
        words[21] = positive_f32_bits(
            (previous.vertical_field_of_view_radians() * 0.5).tan(),
            "previous perspective tangent half field of view",
        )?;
        words[22] =
            positive_f32_bits(previous.aspect_ratio(), "previous perspective aspect ratio")?;
    }
    Ok(words)
}

pub(super) fn normalize_private_vec3(
    values: [f64; 3],
    field: &'static str,
) -> Result<[f64; 3], RenderDeterministicLoweringError> {
    let magnitude_squared = values.iter().map(|value| value * value).sum::<f64>();
    if !magnitude_squared.is_finite() || magnitude_squared <= 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    let reciprocal = magnitude_squared.sqrt().recip();
    let normalized = values.map(|value| value * reciprocal);
    if normalized.iter().any(|value| !value.is_finite()) {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    Ok(normalized)
}

pub(super) fn invert_matrix3(
    values: [f64; 9],
    field: &'static str,
) -> Result<[f64; 9], RenderDeterministicLoweringError> {
    let [a, b, c, d, e, f, g, h, i] = values;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if !determinant.is_finite() || determinant == 0.0 {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    let reciprocal = determinant.recip();
    let inverse = [
        (e * i - f * h) * reciprocal,
        (c * h - b * i) * reciprocal,
        (b * f - c * e) * reciprocal,
        (f * g - d * i) * reciprocal,
        (a * i - c * g) * reciprocal,
        (c * d - a * f) * reciprocal,
        (d * h - e * g) * reciprocal,
        (b * g - a * h) * reciprocal,
        (a * e - b * d) * reciprocal,
    ];
    if inverse.iter().any(|value| !value.is_finite()) {
        return Err(RenderDeterministicLoweringError::NumericRealization { field });
    }
    Ok(inverse)
}

pub(super) fn pack_observation(
    words: &mut [u32],
    transform: crate::space_time::RenderAffineTransform3,
    tan_half_fov: Option<f64>,
    aspect_ratio: Option<f64>,
) -> Result<(), RenderDeterministicLoweringError> {
    let matrix = transform.row_major_3x4();
    pack_vec3(words, 8, [matrix[3], matrix[7], matrix[11]])?;
    pack_invertible_matrix3(
        words,
        11,
        [
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ],
        "observation linear transform",
    )?;
    words[20] = match tan_half_fov {
        Some(value) => positive_f32_bits(value, "perspective tangent half field of view")?,
        None => 0,
    };
    words[21] = match aspect_ratio {
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

pub(super) fn align_up(
    value: u64,
    alignment: u64,
) -> Result<u64, RenderDeterministicLoweringError> {
    if alignment == 0 {
        return Err(RenderDeterministicLoweringError::InvalidBytesPerRowAlignment { alignment });
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value.checked_add(alignment - remainder).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "aligned lattice row bytes",
            },
        )
    }
}

pub(super) fn deterministic_dispatch_size(
    sample_count: u32,
    max_workgroups_per_dimension: u32,
) -> Result<GpuDispatchSize, RenderDeterministicLoweringError> {
    let sample_count_u64 = u64::from(sample_count);
    let workgroup_size = u64::from(WORKGROUP_SIZE);
    let admitted_max = u64::from(max_workgroups_per_dimension);
    let required_groups = sample_count_u64.div_ceil(workgroup_size);
    let capacity = admitted_max * admitted_max;

    if required_groups == 0 || required_groups > capacity {
        return Err(RenderDeterministicLoweringError::DispatchCapacityExceeded {
            sample_count,
            workgroup_size: WORKGROUP_SIZE,
            required_workgroups: required_groups,
            max_workgroups_per_dimension,
            capacity_workgroups: capacity,
        });
    }

    let groups_x = required_groups.min(admitted_max);
    let groups_y = required_groups.div_ceil(groups_x);
    if groups_x > admitted_max || groups_y > admitted_max {
        return Err(RenderDeterministicLoweringError::DispatchCapacityExceeded {
            sample_count,
            workgroup_size: WORKGROUP_SIZE,
            required_workgroups: required_groups,
            max_workgroups_per_dimension,
            capacity_workgroups: capacity,
        });
    }

    Ok(GpuDispatchSize::new(
        u32::try_from(groups_x).expect("admitted dispatch x dimension must fit u32"),
        u32::try_from(groups_y).expect("admitted dispatch y dimension must fit u32"),
        1,
    ))
}

