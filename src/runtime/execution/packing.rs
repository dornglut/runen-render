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
        MaintainedExecutionKind::RequestedCoverage => (execution_mode::REQUESTED_COVERAGE, None),
        MaintainedExecutionKind::Semantic(value) => match value {
            RenderOutputValue::Radiance { representation } => (
                execution_mode::RADIANCE,
                Some(representation.wavelength_meters()),
            )
            RenderOutputValue::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth,
            } => (execution_mode::FORWARD_DEPTH, None),
            RenderOutputValue::ObjectIdentity => (execution_mode::OBJECT_IDENTITY, None),
            RenderOutputValue::Distance { .. } => {
                return Err(RenderDeterministicLoweringError::UnsupportedOutput { output_index });
            }
        },
    };

    let (observation_kind, transform, tan_half_fov, aspect_ratio) = match observation {
        RenderObservationSpec::Perspective(observation) => (
            if observation.sampling_support().is_perspective_lattice_cell() {
                observation_kind::PERSPECTIVE_FOOTPRINT
            } else {
                observation_kind::PERSPECTIVE
            },
            observation.observation_to_scene(),
            Some((observation.vertical_field_of_view_radians() * 0.5).tan()),
            Some(observation.aspect_ratio()),
        ),
        RenderObservationSpec::Probe(observation) => (
            observation_kind::PROBE,
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
    let emitter_offset = header::WORDS
        .checked_add(geometry.len().checked_mul(geometry::WORDS).ok_or(
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

    words[header::SAMPLE_COUNT] = sample_count;
    words[header::EVALUATION_WIDTH] = width;
    words[header::EVALUATION_HEIGHT] = height;
    words[header::ROW_STRIDE_WORDS] = row_stride_words;
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
    words[header::OBSERVATION_KIND] = observation_kind;
    pack_observation(&mut words, transform, tan_half_fov, aspect_ratio)?;
    let requested_extent = requested_extent.unwrap_or((1, 1));
    words[header::REQUESTED_WIDTH] = requested_extent.0;
    words[header::REQUESTED_HEIGHT] = requested_extent.1;
    words[header::TEMPORAL_PHASE] = temporal_history.map_or(0, |history| history.phase);
    words[header::TEMPORAL_SEQUENCE_REVISION] = temporal::SEQUENCE_REVISION;
    words[header::TEMPORAL_HISTORY_AGE] = temporal_history.map_or(0, |history| history.age);
    words[header::TEMPORAL_HISTORY_ROW_STRIDE] =
        temporal_history.map_or(row_stride_words, |history| match &history.storage {
            DeterministicTemporalHistoryUseStorage::Static {
                row_stride_words, ..
            } => *row_stride_words,
            DeterministicTemporalHistoryUseStorage::Camera { .. } => row_stride_words,
        });
    words[header::TEMPORAL_RECONSTRUCTION_REVISION] = TEMPORAL_RECONSTRUCTION_REVISION;
    words[header::EMITTER_OFFSET] = u32::try_from(emitter_offset).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "emitter input offset",
        }
    })?;

    let mut field_sample_cursor = field_sample_offset;
    for (index, (object_id, representation_id, protocol)) in geometry.into_iter().enumerate() {
        let base = header::WORDS + index * geometry::WORDS;
        let state = admitted.plan().scene().object_state(object_id).ok_or(
            RenderDeterministicLoweringError::MissingObjectState {
                output_index,
                object_id,
            },
        )?;
        words[base + geometry::OBJECT_CODE] = *object_codes
            .get(&object_id)
            .ok_or(RenderDeterministicLoweringError::OutputCorrelationChanged { output_index })?;
        words[base + geometry::RADIANCE_REFLECTANCE] = if execution_mode == execution_mode::RADIANCE
        {
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

                let input: &RenderFieldSemanticInput = admitted
                    .field_semantic_input(representation_id)
                    .ok_or(RenderDeterministicLoweringError::MissingFieldInput {
                        output_index,
                        object_id,
                        representation_id,
                    })?;
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
                words[base + geometry::FIELD_SAMPLE_OFFSET] = u32::try_from(field_sample_cursor)
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
                    words[field_sample_cursor] = f32_bits(sample, "field signed-distance sample")?;
                    field_sample_cursor += 1;
                }
            }
        }
    }
    debug_assert_eq!(field_sample_cursor, total_words);

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
) -> Result<[u32; camera::parameters::WORDS], RenderDeterministicLoweringError> {
    let mut words = [0_u32; camera::parameters::WORDS];
    let matrix = current.observation_to_scene().row_major_3x4();
    let inverse = invert_matrix3(
        [
            matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
            matrix[10],
        ],
        "current observation linear transform",
    )?;
    pack_matrix3(
        &mut words,
        camera::parameters::CURRENT_SCENE_TO_OBSERVATION,
        inverse,
    )?;
    words[camera::parameters::PREVIOUS_AVAILABLE] = if previous.is_some() { 1 } else { 0 };
    words[camera::parameters::POSE_CHANGED] = if pose_changed { 1 } else { 0 };
    words[camera::parameters::DEPTH_POLICY_REVISION] = camera::DEPTH_POLICY_REVISION;
    words[camera::parameters::REPROJECTION_REVISION] = CAMERA_REPROJECTION_REVISION;
    words[camera::parameters::DEPTH_ABSOLUTE_EPSILON] = camera::DEPTH_ABSOLUTE_EPSILON.to_bits();
    words[camera::parameters::DEPTH_RELATIVE_EPSILON] = camera::DEPTH_RELATIVE_EPSILON.to_bits();
    // These former proof controls remain part of the retained shader parameter layout but are
    // deliberately disabled in ordinary production execution.
    words[camera::parameters::MOTION_DIAGNOSTICS_ENABLED] = 0;
    words[camera::parameters::CURRENT_ONLY_FIRST_MOTION] = 0;
    words[camera::parameters::SAME_POSE_COMPLETED_FRAMES] =
        same_pose_completed_frames.min(temporal::PHASE_COUNT);
    if let Some(previous) = previous {
        let matrix = previous.observation_to_scene().row_major_3x4();
        pack_vec3(
            &mut words,
            camera::parameters::PREVIOUS_ORIGIN,
            [matrix[3], matrix[7], matrix[11]],
        )?;
        let inverse = invert_matrix3(
            [
                matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8],
                matrix[9], matrix[10],
            ],
            "previous observation linear transform",
        )?;
        pack_matrix3(
            &mut words,
            camera::parameters::PREVIOUS_SCENE_TO_OBSERVATION,
            inverse,
        )?;
        let forward = normalize_private_vec3(
            [-matrix[2], -matrix[6], -matrix[10]],
            "previous observation forward",
        )?;
        pack_vec3(&mut words, camera::parameters::PREVIOUS_FORWARD, forward)?;
        words[camera::parameters::PREVIOUS_TAN_HALF_FOV] = positive_f32_bits(
            (previous.vertical_field_of_view_radians() * 0.5).tan(),
            "previous perspective tangent half field of view",
        )?;
        words[camera::parameters::PREVIOUS_ASPECT_RATIO] =
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
