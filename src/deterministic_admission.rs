//! Owner-controlled semantic planning and R5 admission for the maintained deterministic renderer.
//!
//! A caller-created `RenderMethodContract` cannot become maintained execution authority merely by
//! reusing the renderer-local method ID. RunenRender selects the exact method contract and admits
//! the resulting ordinary plan before applying evaluator/physical-realization compatibility.

use super::admission::{
    AdmittedRenderPlan, RenderExecutionAdmissionFailure, RenderOutputBinding,
    RenderOutputDestination, RenderRepresentationAvailabilityFact,
    admit_render_plan_with_semantic_inputs,
};
use super::derived_transform::{
    RenderCompiledMetricSimilarityTransform, RenderCompiledMetricSimilarityTransformError,
    RenderCompiledObjectTransform, RenderCompiledObjectTransformError,
};
use super::field_input::{RenderFieldSemanticInput, RenderFieldSemanticInputBinding};
use super::maintained_method::maintained_deterministic_method;
use super::representation::{RenderRepresentationId, RenderRepresentationProtocol};
use super::request::{RenderObservationSpec, RenderOutputValue, RenderRequest};
use super::scene::{RenderObjectId, RenderObjectState, RenderSceneSnapshot};
use super::semantic_plan::{RenderPlanningFailure, plan_render};
use super::surface_input::{RenderSurfaceSemanticInput, RenderSurfaceSemanticInputBinding};
use runen_gpu::{GpuBufferUsage, GpuContext, GpuTextureFormat, GpuTextureUsage};
use std::error::Error;
use std::fmt;

const SCALAR_CARRIER_BYTES: u64 = 4;
const LEGACY_LATTICE_FORMAT: GpuTextureFormat = GpuTextureFormat::R32Uint;
const COMPOSABLE_RADIANCE_LATTICE_FORMAT: GpuTextureFormat = GpuTextureFormat::R32Float;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderDeterministicCompatibilityError {
    ObservationShutterNotInstant {
        observation_index: usize,
    },
    ObservationSamplingSupportUnsupported {
        observation_index: usize,
    },
    SelectedRepresentationSurfaceInputUnsupported {
        output_index: usize,
        object_id: RenderObjectId,
        representation_id: RenderRepresentationId,
    },
    SelectedRepresentationFieldInputUnsupported {
        output_index: usize,
        object_id: RenderObjectId,
        representation_id: RenderRepresentationId,
    },
    SelectedObjectFieldTransformNotSimilarity {
        output_index: usize,
        object_id: RenderObjectId,
    },
    SelectedObjectStateMissing {
        output_index: usize,
        object_id: RenderObjectId,
    },
    SelectedObjectTransformNonInvertible {
        output_index: usize,
        object_id: RenderObjectId,
    },
    LatticeFormatUnsupported {
        output_index: usize,
        format: GpuTextureFormat,
    },
    LatticeCopyDestinationUnsupported {
        output_index: usize,
        format: GpuTextureFormat,
    },
    ScalarDestinationSize {
        output_index: usize,
        expected_size_bytes: u64,
        actual_size_bytes: u64,
    },
    ScalarDestinationNotCopyDestination {
        output_index: usize,
    },
    LatticeDestinationFormat {
        output_index: usize,
        actual: GpuTextureFormat,
    },
    LatticeDestinationNotCopyDestination {
        output_index: usize,
    },
}

impl fmt::Display for RenderDeterministicCompatibilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ObservationShutterNotInstant { observation_index } => write!(
                formatter,
                "observation {observation_index} requires an instant shutter"
            ),
            Self::ObservationSamplingSupportUnsupported { observation_index } => write!(
                formatter,
                "observation {observation_index} requires ideal-ray or perspective lattice-cell sampling support"
            ),
            Self::SelectedRepresentationSurfaceInputUnsupported {
                output_index,
                object_id,
                representation_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} representation {representation_id:?} has no surface semantic input supported by the maintained evaluator"
            ),
            Self::SelectedRepresentationFieldInputUnsupported {
                output_index,
                object_id,
                representation_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} representation {representation_id:?} has no sampled field semantic input supported by the maintained evaluator"
            ),
            Self::SelectedObjectFieldTransformNotSimilarity {
                output_index,
                object_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} field representation requires an exact positive similarity transform"
            ),
            Self::SelectedObjectStateMissing {
                output_index,
                object_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} has no retained object state"
            ),
            Self::SelectedObjectTransformNonInvertible {
                output_index,
                object_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} has a non-invertible semantic transform unsupported by the maintained evaluator"
            ),
            Self::LatticeFormatUnsupported {
                output_index,
                format,
            } => write!(
                formatter,
                "lattice output {output_index} requires admitted {format:?} format capabilities"
            ),
            Self::LatticeCopyDestinationUnsupported {
                output_index,
                format,
            } => write!(
                formatter,
                "lattice output {output_index} destination format {format:?} lacks CopyDestination support"
            ),
            Self::ScalarDestinationSize {
                output_index,
                expected_size_bytes,
                actual_size_bytes,
            } => write!(
                formatter,
                "scalar output {output_index} requires {expected_size_bytes} destination bytes, got {actual_size_bytes}"
            ),
            Self::ScalarDestinationNotCopyDestination { output_index } => write!(
                formatter,
                "scalar output {output_index} destination lacks CopyDestination usage"
            ),
            Self::LatticeDestinationFormat {
                output_index,
                actual,
            } => write!(
                formatter,
                "lattice output {output_index} does not admit destination format {actual:?} for its semantic value"
            ),
            Self::LatticeDestinationNotCopyDestination { output_index } => write!(
                formatter,
                "lattice output {output_index} destination lacks CopyDestination usage"
            ),
        }
    }
}

impl Error for RenderDeterministicCompatibilityError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderDeterministicAdmissionFailure {
    Planning(RenderPlanningFailure),
    Admission(RenderExecutionAdmissionFailure),
    Compatibility(RenderDeterministicCompatibilityError),
}

impl fmt::Display for RenderDeterministicAdmissionFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Planning(error) => {
                write!(formatter, "deterministic render planning failed: {error}")
            }
            Self::Admission(error) => {
                write!(formatter, "deterministic render admission failed: {error}")
            }
            Self::Compatibility(error) => {
                write!(
                    formatter,
                    "deterministic renderer compatibility failed: {error}"
                )
            }
        }
    }
}

impl Error for RenderDeterministicAdmissionFailure {}

/// R5-admitted work for the exact RunenRender-owned maintained deterministic method.
///
/// The wrapper carries no submission/session identity. Construction additionally proves that the
/// selected representation inputs, semantic object transforms, request, and physical destinations
/// fit the currently maintained evaluator/carrier contract; RunenGPU still owns graph preparation,
/// realization, submission, and execution failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmittedDeterministicRender {
    admitted: AdmittedRenderPlan,
}

impl AdmittedDeterministicRender {
    pub const fn admitted(&self) -> &AdmittedRenderPlan {
        &self.admitted
    }
}

pub fn admit_deterministic_render_with_semantic_inputs(
    scene: &RenderSceneSnapshot,
    request: &RenderRequest,
    surface_inputs: &[RenderSurfaceSemanticInputBinding],
    field_inputs: &[RenderFieldSemanticInputBinding],
    availability: &[RenderRepresentationAvailabilityFact],
    output_bindings: &[RenderOutputBinding],
    context: &GpuContext,
) -> Result<AdmittedDeterministicRender, RenderDeterministicAdmissionFailure> {
    let method = maintained_deterministic_method();
    let plan = plan_render(scene, request, std::slice::from_ref(&method))
        .map_err(RenderDeterministicAdmissionFailure::Planning)?;
    let admitted = admit_render_plan_with_semantic_inputs(
        &plan,
        surface_inputs,
        field_inputs,
        availability,
        output_bindings,
        context,
    )
    .map_err(RenderDeterministicAdmissionFailure::Admission)?;
    validate_maintained_compatibility(&admitted, context)
        .map_err(RenderDeterministicAdmissionFailure::Compatibility)?;
    Ok(AdmittedDeterministicRender { admitted })
}

fn validate_maintained_compatibility(
    admitted: &AdmittedRenderPlan,
    context: &GpuContext,
) -> Result<(), RenderDeterministicCompatibilityError> {
    for (observation_index, observation) in
        admitted.plan().request().observations().iter().enumerate()
    {
        validate_observation(observation_index, *observation)?;
    }

    validate_selected_evaluator_inputs(admitted)?;

    for output in admitted.outputs() {
        let value = admitted.plan().request().outputs()[output.output_index()]
            .spec()
            .value();
        validate_destination(
            output.output_index(),
            value,
            output.binding().destination(),
            Some(context),
        )?;
    }
    Ok(())
}

fn validate_selected_evaluator_inputs(
    admitted: &AdmittedRenderPlan,
) -> Result<(), RenderDeterministicCompatibilityError> {
    for output in admitted.outputs() {
        for object in output.object_representations() {
            let object_id = object.object_id();
            let representation_id = object.representation().representation_id();
            validate_selected_evaluator_object(
                output.output_index(),
                object_id,
                representation_id,
                object.representation().requirement().protocol().protocol(),
                admitted.surface_semantic_input(representation_id),
                admitted.field_semantic_input(representation_id),
                admitted.plan().scene().object_state(object_id),
            )?;
        }
    }
    Ok(())
}

fn validate_selected_evaluator_object(
    output_index: usize,
    object_id: RenderObjectId,
    representation_id: RenderRepresentationId,
    protocol: RenderRepresentationProtocol,
    surface_input: Option<&RenderSurfaceSemanticInput>,
    field_input: Option<&RenderFieldSemanticInput>,
    state: Option<&RenderObjectState>,
) -> Result<(), RenderDeterministicCompatibilityError> {
    let state = state.ok_or(
        RenderDeterministicCompatibilityError::SelectedObjectStateMissing {
            output_index,
            object_id,
        },
    )?;

    match protocol {
        RenderRepresentationProtocol::SurfaceQuery
        | RenderRepresentationProtocol::OrientedSurfaceQuery => {
            if surface_input.is_none() {
                return Err(
                    RenderDeterministicCompatibilityError::SelectedRepresentationSurfaceInputUnsupported {
                        output_index,
                        object_id,
                        representation_id,
                    },
                );
            }
            RenderCompiledObjectTransform::compile(state.spatial()).map_err(
                |RenderCompiledObjectTransformError::NonInvertibleObjectTransform| {
                    RenderDeterministicCompatibilityError::SelectedObjectTransformNonInvertible {
                        output_index,
                        object_id,
                    }
                },
            )?;
        }
        RenderRepresentationProtocol::FieldDistance => {
            if field_input.is_none() {
                return Err(
                    RenderDeterministicCompatibilityError::SelectedRepresentationFieldInputUnsupported {
                        output_index,
                        object_id,
                        representation_id,
                    },
                );
            }
            RenderCompiledMetricSimilarityTransform::compile(state.spatial()).map_err(
                |RenderCompiledMetricSimilarityTransformError::NotPositiveSimilarity| {
                    RenderDeterministicCompatibilityError::SelectedObjectFieldTransformNotSimilarity {
                        output_index,
                        object_id,
                    }
                },
            )?;
        }
    }
    Ok(())
}

fn validate_observation(
    observation_index: usize,
    observation: RenderObservationSpec,
) -> Result<(), RenderDeterministicCompatibilityError> {
    let shutter = observation.shutter();
    if shutter.start() != shutter.end() {
        return Err(
            RenderDeterministicCompatibilityError::ObservationShutterNotInstant {
                observation_index,
            },
        );
    }

    let sampling_support = match observation {
        RenderObservationSpec::Perspective(observation) => observation.sampling_support(),
        RenderObservationSpec::Probe(observation) => observation.sampling_support(),
    };
    if !sampling_support.is_ideal_ray() && !sampling_support.is_perspective_lattice_cell() {
        return Err(
            RenderDeterministicCompatibilityError::ObservationSamplingSupportUnsupported {
                observation_index,
            },
        );
    }
    Ok(())
}

fn validate_destination(
    output_index: usize,
    value: RenderOutputValue,
    destination: &RenderOutputDestination,
    context: Option<&GpuContext>,
) -> Result<(), RenderDeterministicCompatibilityError> {
    match destination {
        RenderOutputDestination::ScalarBuffer(buffer) => {
            let descriptor = buffer.descriptor();
            if descriptor.size_bytes() != SCALAR_CARRIER_BYTES {
                return Err(
                    RenderDeterministicCompatibilityError::ScalarDestinationSize {
                        output_index,
                        expected_size_bytes: SCALAR_CARRIER_BYTES,
                        actual_size_bytes: descriptor.size_bytes(),
                    },
                );
            }
            if !descriptor
                .usages()
                .contains(GpuBufferUsage::CopyDestination)
            {
                return Err(
                    RenderDeterministicCompatibilityError::ScalarDestinationNotCopyDestination {
                        output_index,
                    },
                );
            }
        }
        RenderOutputDestination::SampleLatticeTexture(texture) => {
            let descriptor = texture.descriptor();
            let format = descriptor.format();
            let format_admitted = match value {
                RenderOutputValue::Radiance { .. } => {
                    matches!(
                        format,
                        LEGACY_LATTICE_FORMAT | COMPOSABLE_RADIANCE_LATTICE_FORMAT
                    )
                }
                RenderOutputValue::Distance { .. } | RenderOutputValue::ObjectIdentity => {
                    format == LEGACY_LATTICE_FORMAT
                }
            };
            if !format_admitted {
                return Err(
                    RenderDeterministicCompatibilityError::LatticeDestinationFormat {
                        output_index,
                        actual: format,
                    },
                );
            }
            if !descriptor
                .usages()
                .contains(GpuTextureUsage::CopyDestination)
            {
                return Err(
                    RenderDeterministicCompatibilityError::LatticeDestinationNotCopyDestination {
                        output_index,
                    },
                );
            }
            if let Some(context) = context {
                let Some(format_capabilities) = context.adapter_facts().supported().format(format)
                else {
                    return Err(
                        RenderDeterministicCompatibilityError::LatticeFormatUnsupported {
                            output_index,
                            format,
                        },
                    );
                };
                if !format_capabilities.copy_destination {
                    return Err(
                        RenderDeterministicCompatibilityError::LatticeCopyDestinationUnsupported {
                            output_index,
                            format,
                        },
                    );
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::method::{
        RenderAbstractExecutionRequirement, RenderMethodContract, RenderMethodId,
        RenderMethodOutputContract, RenderMethodOutputGuarantee, RenderMethodOutputKind,
        RenderObservationKind, RenderSpectralRadianceSupport,
    };
    use crate::request::{
        RenderDistanceConvention, RenderPerspectiveObservation, RenderProbeObservation,
        RenderSamplingSupport,
    };
    use crate::scene::{RenderSceneStore, RenderSceneUpdate};
    use crate::space_time::{
        RenderAffineTransform3, RenderHandedness, RenderObjectSpatialState,
        RenderObjectTemporalState, RenderSpaceSpec, RenderSpatialCoverage, RenderTemporalSupport,
        RenderTimeInterval, RenderTimePoint,
    };
    use runen_gpu::{
        GpuBufferDescriptor, GpuBufferInitialization, GpuBufferUsage, GpuReconstruction,
        GpuResourceLifetime, GpuTextureDescriptor, GpuTextureFormat, GpuTextureInitialization,
        GpuTextureUsage, GpuWorkResourceIdAllocator,
    };

    fn time(seconds: f64) -> RenderTimePoint {
        RenderTimePoint::from_seconds(seconds).expect("finite deterministic-admission test time")
    }

    fn instant() -> RenderTimeInterval {
        RenderTimeInterval::instant(time(0.0))
    }

    fn object_state(local_to_scene: RenderAffineTransform3) -> RenderObjectState {
        RenderObjectState::new(
            RenderObjectSpatialState::new(
                RenderSpaceSpec::new(1.0, RenderHandedness::Right).expect("test space"),
                local_to_scene,
                RenderSpatialCoverage::unbounded(),
            ),
            RenderObjectTemporalState::new(RenderTemporalSupport::unbounded()),
        )
    }

    #[test]
    fn same_method_id_is_not_the_maintained_method_authority() {
        let spectral = RenderSpectralRadianceSupport::new(500e-9, 600e-9).expect("range");
        let impostor = RenderMethodContract::new(
            RenderMethodId::new(1).expect("id"),
            vec![
                RenderMethodOutputContract::new(
                    RenderObservationKind::Probe,
                    RenderMethodOutputKind::Radiance { spectral },
                    RenderMethodOutputGuarantee::Exact,
                    vec![],
                    false,
                )
                .expect("output"),
            ],
            vec![RenderAbstractExecutionRequirement::GeneralParallelWork],
        )
        .expect("impostor contract");
        let maintained = maintained_deterministic_method();
        assert_eq!(impostor.id(), maintained.id());
        assert_ne!(impostor, maintained);
        assert!(
            maintained
                .output_contracts()
                .iter()
                .any(|contract| matches!(
                    contract.output_kind(),
                    RenderMethodOutputKind::Distance {
                        convention: RenderDistanceConvention::ObservationForwardDepth
                    }
                ))
        );
    }

    #[test]
    fn maintained_evaluator_rejects_noninstant_and_nonideal_observations() {
        let interval =
            RenderTimeInterval::new(time(0.0), time(1.0)).expect("ordered test interval");
        let noninstant = RenderObservationSpec::Probe(
            RenderProbeObservation::new(
                RenderAffineTransform3::identity(),
                interval,
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("valid probe"),
        );
        assert_eq!(
            validate_observation(3, noninstant),
            Err(
                RenderDeterministicCompatibilityError::ObservationShutterNotInstant {
                    observation_index: 3
                }
            )
        );

        let cone = RenderObservationSpec::Perspective(
            RenderPerspectiveObservation::new(
                RenderAffineTransform3::identity(),
                std::f64::consts::FRAC_PI_4,
                1.0,
                instant(),
                RenderSamplingSupport::cone(0.01).expect("valid cone"),
            )
            .expect("valid perspective"),
        );
        assert_eq!(
            validate_observation(4, cone),
            Err(
                RenderDeterministicCompatibilityError::ObservationSamplingSupportUnsupported {
                    observation_index: 4
                }
            )
        );
    }

    #[test]
    fn maintained_evaluator_requires_supported_surface_input_and_invertible_transform() {
        let valid_state = object_state(RenderAffineTransform3::identity());
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object id");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, valid_state.clone());
        store.commit(insert).expect("insert object");
        let representation_id = store
            .allocate_representation_id(object_id)
            .expect("representation id");

        assert_eq!(
            validate_selected_evaluator_object(
                5,
                object_id,
                representation_id,
                RenderRepresentationProtocol::SurfaceQuery,
                None,
                None,
                Some(&valid_state),
            ),
            Err(
                RenderDeterministicCompatibilityError::SelectedRepresentationSurfaceInputUnsupported {
                    output_index: 5,
                    object_id,
                    representation_id,
                }
            )
        );

        let input = RenderSurfaceSemanticInput::sphere(
            [0.0, 0.0, 0.0],
            1.0,
            RenderTemporalSupport::unbounded(),
        )
        .expect("surface input");
        let singular_state = object_state(
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .expect("finite singular transform"),
        );
        assert_eq!(
            validate_selected_evaluator_object(
                6,
                object_id,
                representation_id,
                RenderRepresentationProtocol::SurfaceQuery,
                Some(&input),
                None,
                Some(&singular_state),
            ),
            Err(
                RenderDeterministicCompatibilityError::SelectedObjectTransformNonInvertible {
                    output_index: 6,
                    object_id,
                }
            )
        );
        assert_eq!(
            validate_selected_evaluator_object(
                7,
                object_id,
                representation_id,
                RenderRepresentationProtocol::SurfaceQuery,
                Some(&input),
                None,
                Some(&valid_state),
            ),
            Ok(())
        );
    }

    #[test]
    fn maintained_evaluator_requires_field_input_and_exact_similarity_transform() {
        let valid_state = object_state(RenderAffineTransform3::identity());
        let mut store = RenderSceneStore::new();
        let object_id = store.allocate_object_id().expect("object id");
        let mut insert = RenderSceneUpdate::new();
        insert.insert_with_state(object_id, valid_state.clone());
        store.commit(insert).expect("insert object");
        let representation_id = store
            .allocate_representation_id(object_id)
            .expect("representation id");

        assert_eq!(
            validate_selected_evaluator_object(
                8,
                object_id,
                representation_id,
                RenderRepresentationProtocol::FieldDistance,
                None,
                None,
                Some(&valid_state),
            ),
            Err(
                RenderDeterministicCompatibilityError::SelectedRepresentationFieldInputUnsupported {
                    output_index: 8,
                    object_id,
                    representation_id,
                }
            )
        );

        let field = RenderFieldSemanticInput::dense(
            [-1.0; 3],
            [1.0; 3],
            [3, 3, 3],
            vec![0.0; 27],
            0.0,
            RenderTemporalSupport::unbounded(),
        )
        .expect("finite exact sampled field");

        assert_eq!(
            validate_selected_evaluator_object(
                9,
                object_id,
                representation_id,
                RenderRepresentationProtocol::FieldDistance,
                None,
                Some(&field),
                Some(&valid_state),
            ),
            Ok(())
        );

        let non_uniform = object_state(
            RenderAffineTransform3::from_row_major_3x4([
                1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            ])
            .expect("finite non-uniform transform"),
        );
        assert_eq!(
            validate_selected_evaluator_object(
                10,
                object_id,
                representation_id,
                RenderRepresentationProtocol::FieldDistance,
                None,
                Some(&field),
                Some(&non_uniform),
            ),
            Err(
                RenderDeterministicCompatibilityError::SelectedObjectFieldTransformNotSimilarity {
                    output_index: 10,
                    object_id,
                }
            )
        );
    }

    #[test]
    fn maintained_destination_contract_is_one_word_and_copy_destination() {
        let mut allocator = GpuWorkResourceIdAllocator::new();

        let valid_scalar = allocator
            .allocate_buffer_handle(
                GpuBufferDescriptor::ordinary_owned(
                    "valid maintained scalar",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    4,
                    [GpuBufferUsage::CopyDestination],
                    GpuBufferInitialization::Uninitialized,
                )
                .expect("valid scalar descriptor"),
            )
            .expect("scalar handle");
        assert_eq!(
            validate_destination(
                0,
                RenderOutputValue::ObjectIdentity,
                &RenderOutputDestination::ScalarBuffer(valid_scalar),
                None,
            ),
            Ok(())
        );

        let oversized_scalar = allocator
            .allocate_buffer_handle(
                GpuBufferDescriptor::ordinary_owned(
                    "oversized maintained scalar",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    8,
                    [GpuBufferUsage::CopyDestination],
                    GpuBufferInitialization::Uninitialized,
                )
                .expect("valid oversized descriptor"),
            )
            .expect("oversized scalar handle");
        assert_eq!(
            validate_destination(
                1,
                RenderOutputValue::ObjectIdentity,
                &RenderOutputDestination::ScalarBuffer(oversized_scalar),
                None,
            ),
            Err(
                RenderDeterministicCompatibilityError::ScalarDestinationSize {
                    output_index: 1,
                    expected_size_bytes: 4,
                    actual_size_bytes: 8,
                }
            )
        );

        let storage_only_scalar = allocator
            .allocate_buffer_handle(
                GpuBufferDescriptor::ordinary_owned(
                    "storage-only maintained scalar",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    4,
                    [GpuBufferUsage::Storage],
                    GpuBufferInitialization::Uninitialized,
                )
                .expect("valid storage-only scalar descriptor"),
            )
            .expect("storage-only scalar handle");
        assert_eq!(
            validate_destination(
                2,
                RenderOutputValue::ObjectIdentity,
                &RenderOutputDestination::ScalarBuffer(storage_only_scalar),
                None,
            ),
            Err(
                RenderDeterministicCompatibilityError::ScalarDestinationNotCopyDestination {
                    output_index: 2
                }
            )
        );

        let wrong_format = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "wrong maintained lattice format",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    2,
                    2,
                    GpuTextureFormat::Rgba8Unorm,
                    [GpuTextureUsage::CopyDestination],
                    GpuTextureInitialization::Uninitialized,
                )
                .expect("valid wrong-format descriptor"),
            )
            .expect("wrong-format texture handle");
        assert_eq!(
            validate_destination(
                3,
                RenderOutputValue::ObjectIdentity,
                &RenderOutputDestination::SampleLatticeTexture(wrong_format),
                None,
            ),
            Err(
                RenderDeterministicCompatibilityError::LatticeDestinationFormat {
                    output_index: 3,
                    actual: GpuTextureFormat::Rgba8Unorm,
                }
            )
        );

        let radiance_format = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "composable radiance lattice format",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    2,
                    2,
                    GpuTextureFormat::R32Float,
                    [GpuTextureUsage::CopyDestination],
                    GpuTextureInitialization::Uninitialized,
                )
                .expect("valid composable radiance descriptor"),
            )
            .expect("composable radiance texture handle");
        let radiance_value = RenderOutputValue::Radiance {
            representation: super::super::request::RenderRadiometricRepresentation::
                spectral_at_wavelength_meters(550.0e-9)
                .expect("radiance representation"),
        };
        assert_eq!(
            validate_destination(
                5,
                radiance_value,
                &RenderOutputDestination::SampleLatticeTexture(radiance_format.clone()),
                None,
            ),
            Ok(())
        );

        let radiance_without_copy_destination = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "radiance without copy destination",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    2,
                    2,
                    GpuTextureFormat::R32Float,
                    [GpuTextureUsage::StorageWrite],
                    GpuTextureInitialization::Uninitialized,
                )
                .expect("valid radiance storage-only descriptor"),
            )
            .expect("radiance storage-only texture handle");
        assert_eq!(
            validate_destination(
                6,
                radiance_value,
                &RenderOutputDestination::SampleLatticeTexture(radiance_without_copy_destination,),
                None,
            ),
            Err(
                RenderDeterministicCompatibilityError::LatticeDestinationNotCopyDestination {
                    output_index: 6,
                }
            )
        );
        assert_eq!(
            validate_destination(
                7,
                RenderOutputValue::ObjectIdentity,
                &RenderOutputDestination::SampleLatticeTexture(radiance_format),
                None,
            ),
            Err(
                RenderDeterministicCompatibilityError::LatticeDestinationFormat {
                    output_index: 7,
                    actual: GpuTextureFormat::R32Float,
                }
            )
        );

        let storage_only_lattice = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "storage-only maintained lattice",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    2,
                    2,
                    GpuTextureFormat::R32Uint,
                    [GpuTextureUsage::StorageWrite],
                    GpuTextureInitialization::Uninitialized,
                )
                .expect("valid storage-only lattice descriptor"),
            )
            .expect("storage-only lattice handle");
        assert_eq!(
            validate_destination(
                4,
                RenderOutputValue::ObjectIdentity,
                &RenderOutputDestination::SampleLatticeTexture(storage_only_lattice),
                None,
            ),
            Err(
                RenderDeterministicCompatibilityError::LatticeDestinationNotCopyDestination {
                    output_index: 4,
                }
            )
        );
    }
}
