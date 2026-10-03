//! Owner-controlled ordinary execution for the maintained deterministic RunenRender evaluator.
//!
//! This module lowers one exact [`AdmittedDeterministicRender`] into backend-neutral public
//! RunenGPU work and submits every admitted output through one exact `GpuSubmission`. It does not
//! re-plan renderer semantics, expose caller-authored GPU fragments, or make physical encoding part
//! of renderer-semantic result identity.
//!
//! The maintained carrier is one 32-bit word per semantic sample. Radiance and forward depth use
//! `f32` bits. Object identity uses an execution-local non-zero `u32` code decoded only through the
//! returned [`RenderObjectIdentityDecoder`]. Renderer-private definedness and evaluator-status words
//! remain distinct from payload bits. For this maintained direct/no-environment method only, a
//! primary radiance miss is the defined value zero; generic R2 radiance-miss semantics remain wider.

use super::admission::AdmittedDeterministicRender;
pub use super::capture::{
    RenderCapturedDeterministicRadiance, RenderDeterministicRadianceCaptureError,
    RenderDeterministicRadianceCaptureRequest, RenderDeterministicRadianceCaptureRequestError,
};
use super::carrier;
use super::program::{
    CAMERA_REPROJECTION_REVISION, MAINTAINED_EVALUATOR_REVISION, RenderMaintainedProgramBuildError,
    RenderRunenShaderCompilationError, TEMPORAL_RECONSTRUCTION_REVISION,
    abi::{camera, temporal},
    retained_camera_reprojection_source, retained_maintained_evaluator_source,
    retained_temporal_reconstruction_source,
};
#[cfg(test)]
use super::program::{
    CAMERA_REPROJECTION_WGSL, EVALUATOR_WGSL, TEMPORAL_RECONSTRUCTION_WGSL,
    build_maintained_program_sources,
};
use crate::admission::AdmittedRenderPlan;
use crate::field_input::{RenderFieldSemanticInputBinding, RenderFieldSemanticInputGeneration};
use crate::lowering::RenderWorkSet;
use crate::render_result::RenderResult;
use crate::representation::RenderRepresentationId;
use crate::request::{
    RenderObservationSpec, RenderOutputSpec, RenderOutputValue, RenderPerspectiveObservation,
    RenderSamplingSupport,
};
use crate::scene::{RenderObjectId, RenderSceneRevision};
use crate::space_time::RenderTimeInterval;
use crate::surface_input::{
    RenderSurfaceSemanticInputBinding, RenderSurfaceSemanticInputGeneration,
};
use runen_gpu::{
    GpuBufferDescriptor, GpuBufferHandle, GpuBufferInitialization, GpuBufferRegion, GpuBufferUsage,
    GpuClearOperation, GpuComputeOperation, GpuComputePipelineDescriptor, GpuContext,
    GpuContextAffinity, GpuDispatchIntent, GpuExportRelationship,
    GpuOrdinaryTransferPreparationError, GpuProgramContractError, GpuProgramSourceError,
    GpuReadbackId, GpuReadbackOperation, GpuReadbackRequestError, GpuReadbackStatus,
    GpuReconstruction, GpuResourceAccessIntent, GpuResourceDescriptorError, GpuResourceLifetime,
    GpuResourceProvenance, GpuResourceRef, GpuRuntimeBindingValue, GpuSubmission,
    GpuSubmissionFailureKind, GpuSubmissionStatus, GpuTextureFormat, GpuTextureHandle,
    GpuUploadOperation, GpuWorkAuthoringError, GpuWorkFragment, GpuWorkImport,
    GpuWorkOperationError, GpuWorkResourceIdAllocationError, GpuWorkResourceIdAllocator,
    GpuWorkSubmissionError, PreparedGpuData, TransferData,
};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

#[cfg(test)]
#[path = "../../proofs/camera_history.rs"]
mod camera_history_proof;

#[cfg(test)]
#[path = "../../proofs/requested_coverage.rs"]
mod requested_coverage_proof;

const WORD_BYTES: u64 = carrier::WORD_BYTES as u64;

mod errors;
mod finalize;
mod layout;
mod lifecycle;
mod output_context;
mod packing;
mod passes;
mod prepare;
mod state;
mod submission;

pub use lifecycle::{
    PreparedDeterministicRadianceOutput, PreparedDeterministicRender, RenderObjectIdentityDecoder,
    RenderTemporalExecutionEvidence, SubmittedDeterministicRender,
};
pub use prepare::{
    prepare_deterministic_render, submit_deterministic_render,
    submit_deterministic_render_for_verified_result,
};

pub(crate) use errors::{
    RenderDeterministicExecutionError, RenderDeterministicLoweringError,
    RenderDeterministicResultFormationError,
};
#[cfg(test)]
use errors::{RenderRunenGpuPreparationError, gpu_program_source};
pub(crate) use lifecycle::{
    DeterministicVerificationReadbacks, DeterministicVerificationSubmission,
};
pub(crate) use prepare::prepare_deterministic_render_with_cache_in_scope_and_evaluation;
pub(crate) use state::DeterministicResourceCache;
pub(crate) use submission::submit_deterministic_render_for_verification;

use errors::{gpu_resource_allocation, gpu_resource_descriptor};
use lifecycle::{DeterministicObservationIntent, DeterministicVerificationState};
use packing::*;
use prepare::lower_deterministic_render;
#[cfg(test)]
use state::DeterministicTemporalStorage;
use state::{
    DeterministicBufferKind, DeterministicOutputExecutionSelection,
    DeterministicRenderExecutionSelection, DeterministicTemporalHistorySelection,
    DeterministicTemporalHistoryUseStorage, DeterministicTemporalSignature,
    temporal_evaluation_extent_supported, temporal_observation_compatibility,
};
use submission::submit_prepared_deterministic_render;

#[cfg(test)]
mod tests;
