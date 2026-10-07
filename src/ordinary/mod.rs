//! Curated ordinary public surface for the maintained RunenRender path.
//!
//! This module gives ordinary consumers progressive disclosure over the existing semantic
//! scene/request -> planning -> admission -> lowering -> RunenGPU authority. It does not introduce
//! a second planner, renderer, submission lifecycle, or output-value authority. The maintained
//! deterministic implementation remains an internal method realization behind these names.

use super::admission::{
    AdmittedRenderPlan, RenderOutputBinding, RenderRepresentationAvailabilityFact,
};
use super::field_input::{RenderFieldSemanticInputBinding, RenderFieldSemanticInputGeneration};
use super::lowering::RenderWorkSet;
use super::render_result::RenderResult;
use super::representation::RenderRepresentationId;
use super::request::{RenderRadiometricRepresentation, RenderRequest, RenderResultTopology};
use super::runtime::admission::{
    AdmittedDeterministicRender, RenderDeterministicAdmissionFailure,
    admit_deterministic_render_with_semantic_inputs,
};
use super::runtime::capture::{
    RenderCapturedDeterministicRadiance, RenderDeterministicRadianceCaptureError,
    RenderDeterministicRadianceCaptureRequest, RenderDeterministicRadianceCaptureRequestError,
};
pub use super::runtime::execution::RenderObjectIdentityDecoder;
use super::runtime::execution::{
    AssociatedDeterministicRender, DeterministicResourceCache, PreparedDeterministicRadianceOutput,
    PreparedDeterministicRender, RenderDeterministicExecutionError,
    RenderDeterministicResultFormationError,
    RenderTemporalExecutionEvidence as DeterministicTemporalExecutionEvidence,
    SubmittedDeterministicRender, prepare_deterministic_render,
    prepare_deterministic_render_with_cache_and_evaluation, submit_deterministic_render,
    submit_deterministic_render_for_verified_result,
};
use super::runtime::verification::{
    RenderDeterministicVerificationEligibilityError, RenderDeterministicVerificationError,
    RenderDeterministicVerifiedSubmissionError,
};
use super::scene::{RenderObjectId, RenderSceneSnapshot};
use super::surface_input::{
    RenderSurfaceSemanticInputBinding, RenderSurfaceSemanticInputGeneration,
};
use runen_gpu::{
    GpuContext, GpuContextAffinity, GpuExportRelationship, GpuReadbackId, GpuResourceProvenance,
    GpuResourceRef, GpuSubmission, GpuSubmissionFailureKind, GpuSubmissionStatus, GpuTextureHandle,
    GpuTransferRegion, GpuWorkImport, GpuWorkSubmissionError,
};
use std::error::Error;
use std::fmt;

mod errors;
mod lifecycle;
mod operations;
mod state;

pub use errors::*;
pub use lifecycle::*;
pub use operations::*;
pub use state::*;

#[cfg(test)]
mod tests;
