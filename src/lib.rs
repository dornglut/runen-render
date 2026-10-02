//! RunenRender's reusable semantic rendering and maintained image-formation contract.
//!
//! This crate contains the transferred standalone renderer implementation. Under Engineering
//! ADR 0008, physical source presence does not determine semantic authority: Runenwerk remains the
//! accepted authority before the first accepted standalone successor revision, and the exact
//! accepted `runen-render/main` revision becomes authority at that switch.

pub mod admission;
pub mod appearance;
pub mod derived_state;
pub mod field_input;
pub mod lowering;
pub mod method;
mod ordinary;
pub mod output_result;
pub mod participation;
mod render_result;
mod runtime;
pub mod representation;
pub mod request;
pub mod scene;
mod semantic_binding;
pub mod semantic_plan;
pub mod space_time;
pub mod surface_input;
pub mod surface_result;

pub use ordinary::*;
pub use render_result::{
    RenderResult, RenderResultObjectRepresentation, RenderResultOutputEvidence,
};
pub use semantic_binding::RenderSemanticBindingInputError;

#[cfg(test)]
mod proofs;
