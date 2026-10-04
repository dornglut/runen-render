//! RunenRender's reusable semantic rendering and maintained image-formation contract.
//!
//! This crate is the accepted standalone authority for reusable renderer semantics and maintained
//! image formation. Public semantic contracts are intentionally separated from the private
//! maintained runtime realization. The completed Engineering ADR 0008 transfer is provenance;
//! see `BOOTSTRAP.md` for the historical authority transition.

pub mod admission;
pub mod appearance;
pub mod composition_2d;
pub mod derived_state;
pub mod field_input;
pub mod lowering;
pub mod method;
mod ordinary;
pub mod output_result;
pub mod participation;
mod render_result;
pub mod representation;
pub mod request;
mod runtime;
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
