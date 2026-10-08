//! RunenRender's reusable semantic rendering and maintained image-formation contract.
//!
//! This crate is the accepted standalone authority for reusable renderer semantics and maintained
//! image formation. Public semantic contracts are intentionally separated from the private
//! maintained runtime realization. The completed Engineering ADR 0008 transfer is provenance;
//! see `BOOTSTRAP.md` for the historical authority transition.
//!
//! # Ordinary integration
//!
//! A source adapter projects immutable world/asset facts into renderer semantics. Standalone
//! consumers construct the same contracts directly:
//!
//! 1. Commit objects to [`scene::RenderSceneStore`], then attach named
//!    [`representation::RenderRepresentationRecord`] and [`participation::RenderObjectParticipation`]
//!    facts. Retain the immutable [`scene::RenderSceneSnapshot`]. Insertion and attachment are
//!    separate transactions; multi-facet replacement is atomic for existing objects.
//! 2. Use [`request::RenderRequestBuilder`] to describe observations and output meanings. Keep
//!    the minted [`request::RenderObservationHandle`] and [`request::RenderOutputHandle`] values;
//!    numeric positions are diagnostic order, not identity across requests.
//! 3. When physical destinations are available, construct [`RenderInvocation`] with that scene,
//!    request, typed [`surface_input`] / [`field_input`] bindings, current representation
//!    availability, and [`admission::RenderOutputBinding`] values using the original handles.
//! 4. Call [`admit_render`] with the caller's RunenGPU context. Admission preserves the canonical
//!    planning, binding, capability and destination-validation authorities.
//! 5. For retained execution, [`RenderExecutionSession::prepare`] yields composable RunenGPU work.
//!    The caller submits/composes that work, then consumes the exact prepared occurrence through
//!    [`RenderExecutionSession::associate_submission`]. Progress RunenGPU to terminal completion
//!    and [`RenderExecutionSession::reconcile`] before continuing that session.
//! 6. The [`AssociatedRenderOccurrence`] witnesses exact execution and output correlation.
//!    Optional caller-owned readbacks use its handle-based capture/decoder APIs. Captured physical
//!    radiance and decoded identity words do not certify semantic definedness or a [`RenderResult`].
//!
//! The independently runnable [ordinary example](https://github.com/dornglut/runen-render/blob/main/examples/ordinary_render.rs)
//! follows this complete path with spectral radiance and object identity, retained original
//! handles, reversed physical binding order, explicit submission, and bounded completion:
//! `cargo run --example ordinary_render --locked`. It requires a GPU adapter and propagates failures.
//! The [headless scene inspector](https://github.com/dornglut/runen-render/blob/main/examples/headless_scene.rs)
//! adds two-frame retained execution, temporal evidence, and optional diagnostic artifacts.
//!
//! # Named semantic construction (no GPU required)
//!
//! Optional named channels take typed `Option` values so source adapters can pass dynamic facts
//! without alternate constructors. Absent refinement evidence makes no error-bound promise.
//!
//! ```
//! use runen_render::participation::RenderObjectParticipation;
//! use runen_render::representation::{
//!     RenderRepresentationRecord, RenderSurfaceProtocolEvidence,
//!     RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
//! };
//! use runen_render::scene::{RenderSceneStore, RenderSceneUpdate};
//! use runen_render::space_time::{RenderSpatialCoverage, RenderTemporalSupport};
//! use runen_render::request::*;
//! use runen_render::space_time::{RenderAffineTransform3, RenderTimeInterval, RenderTimePoint};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut scene = RenderSceneStore::new();
//! let object = scene.allocate_object_id()?;
//! let mut insert = RenderSceneUpdate::new();
//! insert.insert(object);
//! scene.commit(insert)?;
//! let id = scene.allocate_representation_id(object)?;
//! let record = RenderRepresentationRecord::builder(
//!     id, RenderSpatialCoverage::unbounded(), RenderTemporalSupport::unbounded(),
//! )
//! .surface_query(Some(RenderSurfaceProtocolEvidence::exact(
//!     RENDER_SURFACE_QUERY_PROTOCOL_REVISION,
//! )?))
//! .build()?;
//! let mut attach = RenderSceneUpdate::new();
//! attach.replace_participation(object, RenderObjectParticipation::from_representations([record])?);
//! let snapshot = scene.commit(attach)?.snapshot().clone();
//!
//! let shutter = RenderTimeInterval::instant(RenderTimePoint::from_seconds(0.0)?);
//! let mut builder = RenderRequestBuilder::new(shutter);
//! let observation = builder.add_observation(RenderObservationSpec::Probe(
//!     RenderProbeObservation::new(RenderAffineTransform3::identity(), shutter,
//!         RenderSamplingSupport::ideal_ray())?,
//! ));
//! let output = builder.add_output(&observation, RenderOutputSpec::new(
//!     RenderOutputValue::ObjectIdentity, RenderResultTopology::scalar(),
//!     RenderSemanticTolerance::exact(),
//! )?)?;
//! let request = builder.finish()?;
//! assert!(snapshot.contains(object));
//! assert!(request.contains_output(&output));
//! assert_eq!(output.observation(), observation);
//! # Ok(())
//! # }
//! ```
//!
//! # One-shot, retained, and verified results
//!
//! [`prepare_render`] prepares one-shot composable work; [`submit_render`] offers ordinary
//! one-shot submission without CPU readback. Neither requires a retained session.
//! [`submit_render_for_result`] selects semantic-result verification before submission; after
//! driving RunenGPU progress, [`SubmittedRenderForResult::try_form_result`] may form a
//! [`RenderResult`] bound to the exact admitted scene, request, method and evidence.
//! This verification workflow is distinct from retained physical capture.
//!
//! Advanced consumers can inspect [`semantic_plan::plan_render`],
//! [`admission::admit_render_plan`] and [`lowering`] separately. The ordinary façade uses those
//! same authorities. [`composition_2d`] and [`execution_2d`] describe the separate accepted 2D path.

pub mod admission;
pub mod appearance;
pub mod composition_2d;
pub mod derived_state;
pub mod execution_2d;
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
