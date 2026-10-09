//! Disposable neutral primitive support geometry for ordinary F3F effects.
//!
//! These triangles and bounds are physical preparations of F1's one immutable
//! geometry authority. They are never public renderer semantics, paint alpha,
//! an MSDF/image sampled footprint, or a second scene representation.

#[derive(Debug)]
#[allow(dead_code, reason = "F3F prepared neutral support awaits group-effect lowering")]
pub(super) struct NeutralMesh {
    pub(super) triangles: Vec<[f64; 2]>,
    pub(super) bounds: [f64; 4],
}
