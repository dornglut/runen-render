use super::*;

pub(crate) enum RenderRunenGpuPreparationError {
    ProgramSource {
        stage: &'static str,
        source: GpuProgramSourceError,
    },
    ResourceDescriptor {
        stage: &'static str,
        source: GpuResourceDescriptorError,
    },
    ResourceAllocation {
        stage: &'static str,
        source: GpuWorkResourceIdAllocationError,
    },
    TransferPreparation {
        stage: &'static str,
        source: GpuOrdinaryTransferPreparationError,
    },
    ProgramContract {
        stage: &'static str,
        source: GpuProgramContractError,
    },
    WorkOperation {
        stage: &'static str,
        source: GpuWorkOperationError,
    },
    ReadbackRequest {
        stage: &'static str,
        source: GpuReadbackRequestError,
    },
    WorkAuthoring {
        stage: &'static str,
        source: GpuWorkAuthoringError,
    },
}

impl RenderRunenGpuPreparationError {
    pub const fn stage(&self) -> &'static str {
        match self {
            Self::ProgramSource { stage, .. }
            | Self::ResourceDescriptor { stage, .. }
            | Self::ResourceAllocation { stage, .. }
            | Self::TransferPreparation { stage, .. }
            | Self::ProgramContract { stage, .. }
            | Self::WorkOperation { stage, .. }
            | Self::ReadbackRequest { stage, .. }
            | Self::WorkAuthoring { stage, .. } => stage,
        }
    }

    fn owner_source(&self) -> &(dyn Error + 'static) {
        match self {
            Self::ProgramSource { source, .. } => source,
            Self::ResourceDescriptor { source, .. } => source,
            Self::ResourceAllocation { source, .. } => source,
            Self::TransferPreparation { source, .. } => source,
            Self::ProgramContract { source, .. } => source,
            Self::WorkOperation { source, .. } => source,
            Self::ReadbackRequest { source, .. } => source,
            Self::WorkAuthoring { source, .. } => source,
        }
    }
}

impl fmt::Display for RenderRunenGpuPreparationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "RunenGPU {} failed: {}",
            self.stage(),
            self.owner_source()
        )
    }
}

impl Error for RenderRunenGpuPreparationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.owner_source())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RenderDeterministicLoweringError {
    ContextAffinityChanged {
        admitted: GpuContextAffinity,
        actual: GpuContextAffinity,
    },
    OutputCorrelationChanged {
        output_index: usize,
    },
    MissingObjectState {
        output_index: usize,
        object_id: RenderObjectId,
    },
    MissingSurfaceInput {
        output_index: usize,
        object_id: RenderObjectId,
        representation_id: RenderRepresentationId,
    },
    MissingFieldInput {
        output_index: usize,
        object_id: RenderObjectId,
        representation_id: RenderRepresentationId,
    },
    MissingTemporalSurfaceInputGeneration {
        output_index: usize,
        representation_id: RenderRepresentationId,
    },
    MissingTemporalFieldInputGeneration {
        output_index: usize,
        representation_id: RenderRepresentationId,
    },
    UnsupportedTemporalEvaluationExtent {
        output_index: usize,
        requested_extent: (u32, u32),
        evaluation_extent: (u32, u32),
    },
    NonInvertibleObjectTransform {
        output_index: usize,
        object_id: RenderObjectId,
    },
    NonSimilarityFieldTransform {
        output_index: usize,
        object_id: RenderObjectId,
    },
    MissingMaterial {
        output_index: usize,
        object_id: RenderObjectId,
    },
    UnsupportedOutput {
        output_index: usize,
    },
    MissingBytesPerRowAlignment,
    InvalidBytesPerRowAlignment {
        alignment: u64,
    },
    SizeOverflow {
        field: &'static str,
    },
    HostAllocation {
        field: &'static str,
    },
    NumericRealization {
        field: &'static str,
    },
    DispatchCapacityExceeded {
        sample_count: u32,
        workgroup_size: u32,
        required_workgroups: u64,
        max_workgroups_per_dimension: u32,
        capacity_workgroups: u64,
    },
    RunenShaderCompilation(RenderRunenShaderCompilationError),
    RunenGpuPreparation(RenderRunenGpuPreparationError),
}

impl fmt::Display for RenderDeterministicLoweringError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ContextAffinityChanged { .. } => formatter.write_str(
                "maintained deterministic execution context differs from admitted RunenGPU context",
            ),
            Self::OutputCorrelationChanged { output_index } => write!(
                formatter,
                "maintained deterministic output correlation changed at output {output_index}"
            ),
            Self::MissingObjectState {
                output_index,
                object_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} has no retained spatial state"
            ),
            Self::MissingSurfaceInput {
                output_index,
                object_id,
                representation_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} representation {representation_id:?} has no maintained surface input"
            ),
            Self::MissingFieldInput {
                output_index,
                object_id,
                representation_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} representation {representation_id:?} has no maintained field input"
            ),
            Self::MissingTemporalSurfaceInputGeneration {
                output_index,
                representation_id,
            } => write!(
                formatter,
                "output {output_index} surface representation {representation_id:?} has no source generation required for retained temporal history"
            ),
            Self::MissingTemporalFieldInputGeneration {
                output_index,
                representation_id,
            } => write!(
                formatter,
                "output {output_index} field representation {representation_id:?} has no source generation required for retained temporal history"
            ),
            Self::UnsupportedTemporalEvaluationExtent {
                output_index,
                requested_extent,
                evaluation_extent,
            } => write!(
                formatter,
                "output {output_index} temporal evaluation extent {}x{} cannot cover requested lattice {}x{} with the maintained four-phase footprint sequence",
                evaluation_extent.0, evaluation_extent.1, requested_extent.0, requested_extent.1
            ),
            Self::NonInvertibleObjectTransform {
                output_index,
                object_id,
            } => write!(
                formatter,
                "output {output_index} object {object_id:?} has no finite invertible evaluator transform"
            ),
            Self::NonSimilarityFieldTransform {
                output_index,
                object_id,
            } => write!(
                formatter,
                "output {output_index} field object {object_id:?} has no accepted positive similarity transform"
            ),
            Self::MissingMaterial {
                output_index,
                object_id,
            } => write!(
                formatter,
                "radiance output {output_index} object {object_id:?} has no admitted material"
            ),
            Self::UnsupportedOutput { output_index } => write!(
                formatter,
                "output {output_index} is outside the maintained deterministic evaluator contract"
            ),
            Self::MissingBytesPerRowAlignment => {
                formatter.write_str("RunenGPU did not expose a texture bytes-per-row alignment")
            }
            Self::InvalidBytesPerRowAlignment { alignment } => write!(
                formatter,
                "RunenGPU exposed unusable texture bytes-per-row alignment {alignment}"
            ),
            Self::SizeOverflow { field } => {
                write!(
                    formatter,
                    "{field} exceeds maintained physical indexing limits"
                )
            }
            Self::HostAllocation { field } => {
                write!(formatter, "host allocation failed for {field}")
            }
            Self::NumericRealization { field } => write!(
                formatter,
                "{field} cannot be represented by the maintained finite f32 evaluator"
            ),
            Self::DispatchCapacityExceeded {
                sample_count,
                workgroup_size,
                required_workgroups,
                max_workgroups_per_dimension,
                capacity_workgroups,
            } => write!(
                formatter,
                "sample count {sample_count} with workgroup size {workgroup_size} requires {required_workgroups} workgroups, but the admitted maximum per dimension is {max_workgroups_per_dimension} and the 2D dispatch capacity is {capacity_workgroups} workgroups"
            ),
            Self::RunenShaderCompilation(error) => error.fmt(formatter),
            Self::RunenGpuPreparation(error) => error.fmt(formatter),
        }
    }
}

impl Error for RenderDeterministicLoweringError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::RunenShaderCompilation(error) => Some(error),
            Self::RunenGpuPreparation(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub(crate) enum RenderDeterministicExecutionError {
    Lowering(RenderDeterministicLoweringError),
    Submission(GpuWorkSubmissionError),
}

impl fmt::Display for RenderDeterministicExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lowering(error) => write!(formatter, "deterministic lowering failed: {error}"),
            Self::Submission(error) => write!(formatter, "RunenGPU submission failed: {error}"),
        }
    }
}

impl Error for RenderDeterministicExecutionError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Lowering(error) => Some(error),
            Self::Submission(error) => Some(error),
        }
    }
}

impl From<RenderDeterministicLoweringError> for RenderDeterministicExecutionError {
    fn from(value: RenderDeterministicLoweringError) -> Self {
        Self::Lowering(value)
    }
}

/// Failure while polling one exact verified submission for semantic result formation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RenderDeterministicResultFormationError {
    VerificationNotRequested,
    ResultAlreadyFormed,
    SubmissionFailed {
        kind: GpuSubmissionFailureKind,
    },
    ReadbackCorrelationLost {
        output_index: usize,
        channel: &'static str,
    },
    ReadbackFailed {
        output_index: usize,
        channel: &'static str,
        kind: GpuSubmissionFailureKind,
    },
    Verification(super::verification::RenderDeterministicVerificationError),
}

impl fmt::Display for RenderDeterministicResultFormationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VerificationNotRequested => formatter.write_str(
                "this deterministic submission was authored without verified-result observations",
            ),
            Self::ResultAlreadyFormed => formatter.write_str(
                "semantic result evidence was already formed from this deterministic submission",
            ),
            Self::SubmissionFailed { kind } => write!(
                formatter,
                "RunenGPU submission failed before verified result formation: {kind:?}"
            ),
            Self::ReadbackCorrelationLost {
                output_index,
                channel,
            } => write!(
                formatter,
                "output {output_index} {channel} verification readback is no longer correlated to the exact submission"
            ),
            Self::ReadbackFailed {
                output_index,
                channel,
                kind,
            } => write!(
                formatter,
                "output {output_index} {channel} verification readback failed: {kind:?}"
            ),
            Self::Verification(error) => write!(
                formatter,
                "deterministic finite-evaluation verification rejected result formation: {error}"
            ),
        }
    }
}

impl Error for RenderDeterministicResultFormationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Verification(error) => Some(error),
            _ => None,
        }
    }
}


pub(super) fn map_maintained_program_build_error(
    error: RenderMaintainedProgramBuildError,
) -> RenderDeterministicLoweringError {
    match error {
        RenderMaintainedProgramBuildError::RunenShader(error) => {
            RenderDeterministicLoweringError::RunenShaderCompilation(error)
        }
        RenderMaintainedProgramBuildError::RunenGpu { stage, source } => {
            gpu_program_source(stage, source)
        }
    }
}

pub(super) fn gpu_program_source(
    stage: &'static str,
    source: GpuProgramSourceError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::ProgramSource { stage, source },
    )
}

pub(super) fn gpu_resource_descriptor(
    stage: &'static str,
    source: GpuResourceDescriptorError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::ResourceDescriptor { stage, source },
    )
}

pub(super) fn gpu_resource_allocation(
    stage: &'static str,
    source: GpuWorkResourceIdAllocationError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::ResourceAllocation { stage, source },
    )
}

pub(super) fn gpu_transfer_preparation(
    stage: &'static str,
    source: GpuOrdinaryTransferPreparationError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::TransferPreparation { stage, source },
    )
}

pub(super) fn gpu_program_contract(
    stage: &'static str,
    source: GpuProgramContractError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::ProgramContract { stage, source },
    )
}

pub(super) fn gpu_work_operation(
    stage: &'static str,
    source: GpuWorkOperationError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::WorkOperation { stage, source },
    )
}

pub(super) fn gpu_readback_request(
    stage: &'static str,
    source: GpuReadbackRequestError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::ReadbackRequest { stage, source },
    )
}

pub(super) fn gpu_work_authoring(
    stage: &'static str,
    source: GpuWorkAuthoringError,
) -> RenderDeterministicLoweringError {
    RenderDeterministicLoweringError::RunenGpuPreparation(
        RenderRunenGpuPreparationError::WorkAuthoring { stage, source },
    )
}

