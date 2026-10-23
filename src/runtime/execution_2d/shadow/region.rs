//! Immutable renderer-private neutral SUPPORT expressions for ordinary F3F.
//!
//! These are prepared from the ONE borrowed F1 painter, not a second semantic
//! scene. Curved effects remain symbolic across parent frames. Polygonal
//! realization is merely one *certified* disposable query path; an unsupported
//! curved consumer fails before target mutation instead of inventing triangles
//! from bounds, alpha or a sampled distance field.
use crate::composition_2d::{
    Render2dAffineTransform, Render2dClip, Render2dDropShadow, Render2dItem,
};
use crate::execution_2d::{
    Render2dExecutionError, Render2dSampleSpaceError, Render2dUnsupportedContent,
};
use crate::runtime::execution_2d::{shadow, support};
use std::sync::{Arc, OnceLock};

mod gaussian_consumer;
pub(in crate::runtime::execution_2d) mod interval;
mod source_envelope;
pub(super) use certified_cells::scan_area::gaussian::GaussianPlaneRequest;
pub(in crate::runtime::execution_2d) use certified_cells::scan_area::gaussian::PreparedPlane;

const MAX_REGION_NODES: usize = 65_536;
const MAX_REGION_DEPTH: usize = 512;
const MAX_REGION_PATH_UNITS: usize = 1_048_576;
const MAX_REGION_GRAPH_WORK: usize = 16_777_216;
// Primitive source-level arithmetic/comparisons, loads/stores and bounded
// control steps, not machine instructions. Proof receipts still pay lookup.
const PROVENANCE_LOOKUP_WORK: usize = 16;
const PROVENANCE_NODE_WORK: usize = 64;
// Authored rectangle/path/triangle checks, finite transforms and corner
// correspondence are bounded independently of the physical vertex count.
const PROVENANCE_SOURCE_WORK: usize = 1_024;
// Four extrema scans plus triangle/corner membership, bit tests and loops.
const RECTANGLE_VERTEX_WORK: usize = 48;
// ID scan, stable retain/drop, depth/envelope scans and propagated flags.
const UNION_CHILD_WORK: usize = 64;
const UNION_FIXED_WORK: usize = 128;
const SOURCE_VERTEX_WORK: usize = 16;
const MATERIALIZATION_LOOKUP_WORK: usize = 16;
const MATERIALIZATION_NODE_WORK: usize = 192;
// Fixed node/path bookkeeping and nonleaf affine/effect interval envelopes.
const REGION_NODE_WORK: usize = 1_024;
const AFFINE_REACH_VERTEX_WORK: usize = 32;
// A control visits both item and ancestor frames: at most eight interval
// products and eight sums, plus scalar/validity/loop/extrema work.
const SOURCE_REACH_CONTROL_WORK: usize = 2_048;
// Stroke radius and transformed cap/miter bounds have additional interval
// work independent of the number of source controls.
const SOURCE_REACH_STROKE_WORK: usize = 2_048;

/// Fixed query inventory reserved by the physical consumer, excluding its
/// separately admitted width-dependent rows and output/conversion buffers.
pub(in crate::runtime::execution_2d) const MAX_AREA_QUERY_FIXED_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
pub(in crate::runtime::execution_2d) enum AreaQueryError {
    Unsupported,
    Precision,
    Resource,
}

pub(in crate::runtime::execution_2d) struct NeutralRegion<'a> {
    kind: Kind<'a>,
    // Borrow the original F1 item. A disposable source tessellation is NOT
    // a certificate of semantic curves, clips, strokes or affine geometry.
    #[allow(dead_code, reason = "reserved for certified F1 source-region queries")]
    semantic_item: Option<&'a Render2dItem>,
    path: Vec<usize>,
    id: usize,
    depth: usize,
    /// Conservative logical bounds in this node's owning frame. Bounds
    /// certify reach/finite preflight only; they are NEVER membership.
    envelope: Option<[f64; 4]>,
    /// Only true when every original F1 source's authored reach is enclosed
    /// by the graph envelope, independent of disposable tessellation.
    /// A physical glyph mesh bound cannot authorize complete signed erosion;
    /// text requires the separately paid original-outline control hull.
    certified_source_reach: bool,
    /// One immutable DAG-local eligibility bit. Positive Euclidean disks
    /// cannot be reinterpreted as the newly transported SIGNED caster;
    /// retain their earlier typed rejection until their own exact
    /// continuous ancestor evaluator is independently qualified.
    pub(super) contains_positive_disk: bool,
    /// Original-only consumer selection skips the unchanged plain-source
    /// Gaussian family in constant time. This is morphology presence, not
    /// membership or proof that the numerical consumer can admit the source.
    pub(super) contains_negative_disk: bool,
}

/// An invocation-local optional physical realization, never source meaning.
/// Once installed, every polygon consumer borrows this same immutable mesh.
struct SourceGeometry {
    mesh: OnceLock<Arc<support::NeutralMesh>>,
    glyphs: Option<GlyphSources>,
}

/// Frozen original F2 controls and placement, independent of field bitmaps
/// and resource/cache lifetimes. Per-glyph winding precedes set union.
struct GlyphSource {
    outline: crate::runtime::execution_2d::field::GlyphOutline,
    position: [f64; 2],
}

struct GlyphSources {
    instances: Vec<GlyphSource>,
    font_size: f64,
    /// Original control-hull reach in the item's parent frame; absence only.
    reach: [f64; 4],
    /// Conservative per-occurrence shared-body inventory. Repeated handles
    /// may count one allocation more than once; they never duplicate it.
    outline_bytes: usize,
}

impl GlyphSources {
    fn storage_bytes(&self) -> Option<usize> {
        self.instances
            .capacity()
            .checked_mul(size_of::<GlyphSource>())?
            .checked_add(self.outline_bytes)
    }
}

impl SourceGeometry {
    fn deferred() -> Self {
        Self {
            mesh: OnceLock::new(),
            glyphs: None,
        }
    }

    fn prepared(mesh: Arc<support::NeutralMesh>) -> Self {
        Self {
            mesh: OnceLock::from(mesh),
            glyphs: None,
        }
    }

    fn get(&self) -> Option<&Arc<support::NeutralMesh>> {
        self.mesh.get()
    }
}

enum Kind<'a> {
    Union(Vec<Arc<NeutralRegion<'a>>>),
    Source(SourceGeometry),
    Affine {
        source: Arc<NeutralRegion<'a>>,
        to_parent: Render2dAffineTransform,
    },
    Clip {
        source: Arc<NeutralRegion<'a>>,
        clips: &'a [Render2dClip],
    },
    Effect {
        source: Arc<NeutralRegion<'a>>,
        shadow: Render2dDropShadow,
    },
}

/// Constant-work original F1 facts that establish no positive-area source.
/// This receipt is independent of disposable tessellation and paint alpha.
fn structurally_empty_fill_shape(shape: &crate::composition_2d::Render2dShape) -> bool {
    shape.as_rect().is_some_and(|rect| rect.is_empty())
        || shape.as_ellipse().is_some_and(|rect| rect.is_empty())
        || shape
            .as_rounded_rect()
            .is_some_and(|(rect, _)| rect.is_empty())
        || shape
            .as_path()
            .is_some_and(|path| path.commands().is_empty())
}

pub(super) fn trivially_empty_original_source(item: &Render2dItem) -> bool {
    let [a, b, c, d, _, _] = item.local_to_parent().components();
    if interval::exact_products_equal(a, d, b, c) == Some(true) {
        return true;
    }
    match item.primitive() {
        crate::composition_2d::Render2dPrimitive::Fill { shape, .. } => {
            structurally_empty_fill_shape(shape)
        }
        crate::composition_2d::Render2dPrimitive::Stroke { style, .. } => style.width() == 0.0,
        crate::composition_2d::Render2dPrimitive::Image(image) => image.patches().is_empty(),
        crate::composition_2d::Render2dPrimitive::ShapedText(_) => false,
    }
}

fn resource(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::ResourceLimit,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    }
}

fn precision(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::PrecisionLimit,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    }
}

fn unsupported(path: &[usize]) -> Render2dExecutionError {
    crate::runtime::execution_2d::unsupported_at(
        path,
        Render2dUnsupportedContent::Shadows {
            root_index: path[0],
        },
    )
}

/// A direct mesh leaf can be projected more precisely than the corners of
/// its AABB. The latter can overflow at a wholly unoccupied corner under
/// extreme shear even if every original triangle vertex is representable.
/// This remains a conservative reach envelope, NEVER a membership oracle.
fn affine_leaf_envelope(
    mesh: &support::NeutralMesh,
    affine: Render2dAffineTransform,
    path: &[usize],
) -> Result<Option<[f64; 4]>, Render2dExecutionError> {
    if mesh.triangles.is_empty() {
        return Ok(None);
    }
    let scale = mesh.units_per_parent_logical_unit;
    let [a, b, c, d, tx, ty] = affine.components();
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for &[x, y] in &mesh.triangles {
        let x = x / scale;
        let y = y / scale;
        // Match the real polygonal materialization arithmetic exactly.
        let point = [
            a.mul_add(x, c.mul_add(y, tx)),
            b.mul_add(x, d.mul_add(y, ty)),
        ];
        if !point.iter().all(|v| v.is_finite()) {
            return Err(precision(
                path,
                "neutral-region affine leaf loses vertex representability",
            ));
        }
        bounds[0] = bounds[0].min(point[0]);
        bounds[1] = bounds[1].min(point[1]);
        bounds[2] = bounds[2].max(point[0]);
        bounds[3] = bounds[3].max(point[1]);
    }
    Ok(Some(bounds))
}

/// Conservative support reach derived from exact symbolic operators.
/// An axis-aligned extent must NEVER be interpreted as region membership or
/// Gaussian coverage. In particular, Clip may shrink but cannot enlarge its
/// source, and negative Euclidean erosion may shrink or erase but cannot grow.
/// Retaining the larger original extent avoids guessing curved clip geometry.
/// Certify the axis-aligned reach of a non-leaf affine without treating
/// the enclosing rectangle as the exact source geometry. The source-leaf
/// vertex-tight path above remains separate: projecting unused AABB corners
/// can overflow even when every actual triangle vertex is representable.
fn nested_affine_envelope(
    bounds: [f64; 4],
    transform: Render2dAffineTransform,
    path: &[usize],
) -> Result<[f64; 4], Render2dExecutionError> {
    if transform == Render2dAffineTransform::IDENTITY {
        return Ok(bounds);
    }
    use interval::Interval;
    let x = Interval::new(bounds[0], bounds[2])
        .ok_or_else(|| precision(path, "nested affine x bounds are invalid"))?;
    let y = Interval::new(bounds[1], bounds[3])
        .ok_or_else(|| precision(path, "nested affine y bounds are invalid"))?;
    let [a, b, c, d, tx, ty] = transform.components().map(Interval::scalar);
    let [Some(a), Some(b), Some(c), Some(d), Some(tx), Some(ty)] = [a, b, c, d, tx, ty] else {
        return Err(precision(path, "nested affine coefficients are not finite"));
    };
    let interval_x = a
        .mul(x)
        .and_then(|term| term.add(c.mul(y)?))
        .and_then(|sum| sum.add(tx))
        .ok_or_else(|| precision(path, "nested affine x reach cannot be enclosed"))?;
    let interval_y = b
        .mul(x)
        .and_then(|term| term.add(d.mul(y)?))
        .and_then(|sum| sum.add(ty))
        .ok_or_else(|| precision(path, "nested affine y reach cannot be enclosed"))?;
    Ok([interval_x.lo, interval_y.lo, interval_x.hi, interval_y.hi])
}

/// A signed Euclidean spread can enlarge support by at most max(spread, 0)
/// on each axis; a Gaussian has a separate finite square support halo.
/// Conservative bounds are not membership or Gaussian coverage certificates.
fn nested_effect_envelope(
    bounds: [f64; 4],
    effect: Render2dDropShadow,
    path: &[usize],
) -> Result<[f64; 4], Render2dExecutionError> {
    use interval::Interval;
    let x = Interval::new(bounds[0], bounds[2])
        .ok_or_else(|| precision(path, "nested effect x bounds are invalid"))?;
    let y = Interval::new(bounds[1], bounds[3])
        .ok_or_else(|| precision(path, "nested effect y bounds are invalid"))?;
    if effect.offset_x() == 0.0
        && effect.offset_y() == 0.0
        && effect.sigma() == 0.0
        && effect.spread() <= 0.0
    {
        // Identity or erosion-only cannot enlarge support; this also avoids
        // a false precision failure from rounding a finite f64::MAX endpoint.
        return Ok(bounds);
    }
    let margin = Interval::scalar(effect.spread().max(0.0))
        .and_then(|spread| {
            spread.add(Interval::scalar(3.0)?.mul(Interval::scalar(effect.sigma())?)?)
        })
        .ok_or_else(|| precision(path, "nested effect halo is not representable"))?
        .hi;
    let pad = Interval::new(-margin, margin)
        .ok_or_else(|| precision(path, "nested effect padding is not finite"))?;
    let offset_x = Interval::scalar(effect.offset_x())
        .ok_or_else(|| precision(path, "nested effect horizontal offset is not finite"))?;
    let offset_y = Interval::scalar(effect.offset_y())
        .ok_or_else(|| precision(path, "nested effect vertical offset is not finite"))?;
    let horizontal = x
        .add(offset_x)
        .and_then(|shifted| shifted.add(pad))
        .ok_or_else(|| precision(path, "nested effect horizontal reach cannot be enclosed"))?;
    let vertical = y
        .add(offset_y)
        .and_then(|shifted| shifted.add(pad))
        .ok_or_else(|| precision(path, "nested effect vertical reach cannot be enclosed"))?;
    Ok([horizontal.lo, vertical.lo, horizontal.hi, vertical.hi])
}

fn region_envelope(
    kind: &Kind<'_>,
    semantic_item: Option<&Render2dItem>,
    path: &[usize],
) -> Result<Option<[f64; 4]>, Render2dExecutionError> {
    let result = match kind {
        Kind::Source(mesh) => {
            // This DAG's reach belongs to original F1 geometry. Disposable
            // realization vertices may neither enlarge nor erase it.
            if let Some(glyphs) = &mesh.glyphs {
                Some(glyphs.reach)
            } else if let Some(item) = semantic_item
                && let Some(authored) = source_envelope::bounds(item, None, path)?
            {
                Some(authored)
            } else if let Some(mesh) = mesh.get() {
                if mesh.triangles.is_empty() {
                    return Ok(None);
                }
                let scale = mesh.units_per_parent_logical_unit;
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(precision(
                        path,
                        "neutral-region source scale is unrepresentable",
                    ));
                }
                // Missing authored reach remains a physical fallback; glyph
                // provenance does not certify this as original support.
                Some(mesh.bounds.map(|coordinate| coordinate / scale))
            } else {
                None
            }
        }
        Kind::Union(children) => {
            let mut combined: Option<[f64; 4]> = None;
            for node in children {
                if let Some(bounds) = node.envelope {
                    combined = Some(match combined {
                        Some(prev) => [
                            prev[0].min(bounds[0]),
                            prev[1].min(bounds[1]),
                            prev[2].max(bounds[2]),
                            prev[3].max(bounds[3]),
                        ],
                        None => bounds,
                    });
                }
            }
            combined
        }
        Kind::Affine { source, to_parent } => {
            if let Kind::Source(mesh) = &source.kind {
                if let Some(glyphs) = &mesh.glyphs {
                    // The original hull already includes glyph placement and
                    // the item affine. Its interval image encloses all source
                    // controls without revisiting a font or physical mesh.
                    Some(nested_affine_envelope(glyphs.reach, *to_parent, path)?)
                } else if let Some(item) = source.semantic_item
                    && let Some(authored) = source_envelope::bounds(item, Some(*to_parent), path)?
                {
                    Some(authored)
                } else {
                    mesh.get()
                        .map(|mesh| affine_leaf_envelope(mesh, *to_parent, path))
                        .transpose()?
                        .flatten()
                }
            } else {
                source
                    .envelope
                    .map(|bounds| nested_affine_envelope(bounds, *to_parent, path))
                    .transpose()?
            }
        }
        Kind::Clip { source, .. } => source.envelope,
        Kind::Effect {
            source,
            shadow: effect,
        } => source
            .envelope
            .map(|bounds| nested_effect_envelope(bounds, *effect, path))
            .transpose()?,
    };
    if let Some(bounds) = result
        && (!bounds.iter().all(|value| value.is_finite())
            || bounds[0] > bounds[2]
            || bounds[1] > bounds[3])
    {
        return Err(precision(
            path,
            "neutral-region envelope is unrepresentable",
        ));
    }
    Ok(result)
}

/// Certify that NO translated/blurred shadow support can survive signed
/// Euclidean erosion by a disk. Any disk of radius r contained in C requires
/// both the horizontal and vertical span of C to be at least 2r.
///
/// The enclosing box is deliberately used only to prove ABSENCE: it never
/// supplies inside membership, a manufactured erosion polygon, an alpha mask,
/// or a Gaussian weight. Original borrowed-F1 controls, affine ancestors and
/// true positive-effect envelopes are already included in source.envelope.
///
/// Outward-rounded differences and halves avoid deleting a thin surviving
/// shape at a floating-point boundary. Unrepresentable spans return Unknown.
fn certified_complete_disk_erosion(bounds: Option<[f64; 4]>, spread: f64) -> bool {
    use interval::Interval;
    if !spread.is_finite() || spread >= 0.0 {
        return false;
    }
    let Some([left, top, right, bottom]) = bounds else {
        return false;
    };
    let Some(radius) = Interval::scalar(-spread) else {
        return false;
    };
    let Some(two) = Interval::scalar(2.0) else {
        return false;
    };
    for (lo, hi) in [(left, right), (top, bottom)] {
        let half_span = Interval::scalar(hi)
            .and_then(|end| end.sub(Interval::scalar(lo)?))
            .and_then(|width| width.div(two));
        if half_span.is_some_and(|half| radius.lo > half.hi) {
            return true;
        }
    }
    false
}

impl NeutralRegion<'_> {
    pub(super) fn owner_path(&self) -> &[usize] {
        &self.path
    }

    pub(super) fn is_negative_effect(&self) -> bool {
        matches!(&self.kind, Kind::Effect { shadow, .. } if shadow.spread() < 0.0)
    }

    /// Caller prepays the bounded scalar interval envelope operation. This
    /// reuses original reach and the sole owner-frame morphology formula.
    pub(super) fn effect_envelope(
        &self,
        effect: Render2dDropShadow,
    ) -> Result<Option<[f64; 4]>, Render2dExecutionError> {
        if !self.certified_source_reach {
            return Err(unsupported(&self.path));
        }
        self.envelope
            .map(|bounds| nested_effect_envelope(bounds, effect, &self.path))
            .transpose()
    }

    pub(super) fn certified_logical_envelope(&self) -> Option<[f64; 4]> {
        self.certified_source_reach
            .then_some(self.envelope)
            .flatten()
    }

    /// Classify shared sample spans before the expensive uncertain boundary
    /// cells. This exposes a bounded geometric *interval grid*, never a
    /// physical alpha mask; F3E must separately certify final-color error.
    pub(super) fn continuous_grid_enclosures(
        &self,
        root_grid: [u32; 4],
        samples_per_logical_unit: f64,
        owner_to_root: Option<Render2dAffineTransform>,
        max_cell_depth: u8,
        max_cells: usize,
        fuel: &mut usize,
    ) -> Result<Vec<[f64; 2]>, AreaQueryError> {
        certified_cells::certified_grid_area_query(
            self,
            root_grid,
            samples_per_logical_unit,
            owner_to_root,
            max_cell_depth,
            max_cells,
            fuel,
        )
    }

    pub(super) fn all_shadows_provably_empty(&self, shadows: &[Render2dDropShadow]) -> bool {
        !shadows.is_empty()
            && self.certified_source_reach
            && shadows
                .iter()
                .all(|shadow| certified_complete_disk_erosion(self.envelope, shadow.spread()))
    }

    /// A root's complete authored shadow set can be discarded only when
    /// original-F1 reach, including nested curved effects, proves that EVERY
    /// shadow is strictly outside the requested canvas. An enclosing box is
    /// used exclusively to certify absence, never as positive coverage.
    /// Canvas division and all effect halos are rounded outward.
    pub(super) fn all_root_shadows_provably_offscreen(
        &self,
        shadows: &[Render2dDropShadow],
        canvas: [f64; 2],
        raster_scale: f64,
    ) -> Result<bool, Render2dExecutionError> {
        if shadows.is_empty() || !self.certified_source_reach {
            return Ok(false);
        }
        let Some(scale) = interval::Interval::scalar(raster_scale) else {
            return Ok(false);
        };
        if scale.lo <= 0.0 {
            return Ok(false);
        }
        let Some(width) =
            interval::Interval::scalar(canvas[0]).and_then(|extent| extent.div(scale))
        else {
            return Ok(false);
        };
        let Some(height) =
            interval::Interval::scalar(canvas[1]).and_then(|extent| extent.div(scale))
        else {
            return Ok(false);
        };
        if width.lo <= 0.0 || height.lo <= 0.0 {
            return Ok(false);
        }
        let Some(bounds) = self.envelope else {
            // No source envelope is NOT proof of an empty original-F1 region:
            // a disposable empty mesh may still miss authored geometry.
            // Only a certified finite outer reach can authorize this cull.
            return Ok(false);
        };
        for &shadow in shadows {
            let outer = nested_effect_envelope(bounds, shadow, &self.path)?;
            if !(outer[2] < 0.0 || outer[3] < 0.0 || outer[0] > width.hi || outer[1] > height.hi) {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// One invocation-owned bounded graph and memoized polygonal realization.
/// Memo keys are monotonic graph IDs, NEVER F1 semantic or cache IDs.
pub(super) struct NeutralRegionBuilder {
    pub(super) host: support::PreparationHostBudget,
    nodes: usize,
    constructed_node_bytes: usize,
    retained_path_units: usize,
    max_owner_path: usize,
    retained_vertices: usize,
    /// One invocation-wide original reach AND stable neutral-union scan budget.
    pub(super) graph_work: usize,
    pub(super) clip_work: usize,
    erosion_work: usize,
    // Dense, fallibly allocated by monotonic node ID; no implicit per-entry
    // tree allocation and no aliasing if a prior node is dropped.
    materialized: Vec<Option<Arc<support::NeutralMesh>>>,
    // One bounded transient stamp table for idempotent support union.
    // Stable first-source order is preserved for owner-path diagnostics.
    union_seen: Vec<usize>,
    union_epoch: usize,
    // Completed original-F1 proofs belong to this immutable invocation DAG.
    // A cached false result is a paid rejection, never physical authority.
    polygonal_proofs: Vec<Option<bool>>,
    scales: shadow::NeutralPreparationScale,
    max_buffer_bytes: u64,
}

/// One transient physical-consumer dependency set, indexed by invocation DAG
/// IDs and registered caster positions. It carries no geometric authority.
pub(super) struct PolygonDependencies<'scene> {
    nodes: Vec<bool>,
    consumers: Vec<bool>,
    frames: Vec<(Arc<NeutralRegion<'scene>>, usize)>,
    metadata_bytes: usize,
}

impl PolygonDependencies<'_> {
    pub(super) fn metadata_bytes(&self) -> usize {
        self.metadata_bytes
    }

    pub(super) fn select_consumer(&mut self, position: usize) {
        self.consumers[position] = true;
    }

    pub(super) fn consumer(&self, position: usize) -> bool {
        self.consumers[position]
    }

    pub(super) fn source(&self, source: &NeutralRegion<'_>) -> bool {
        self.nodes[source.id]
    }
}

/// Bounded private physical realization of one proven F1 rounded rectangle.
/// The original child path remains authoritative for unsupported extensions.
pub(in crate::runtime::execution_2d) struct CertifiedRoundedRectangle {
    pub(in crate::runtime::execution_2d) rectangle: Arc<support::NeutralMesh>,
    pub(in crate::runtime::execution_2d) radius: f64,
    pub(in crate::runtime::execution_2d) curved_owner: Vec<usize>,
}

impl NeutralRegionBuilder {
    pub(super) fn new(scales: shadow::NeutralPreparationScale, max_buffer_bytes: u64) -> Self {
        let mut host = support::PreparationHostBudget::new();
        host.reserve(size_of::<Self>(), &[])
            .expect("fixed source owner fits host admission");
        Self {
            host,
            nodes: 0,
            constructed_node_bytes: 0,
            retained_path_units: 0,
            max_owner_path: 0,
            retained_vertices: 0,
            graph_work: 0,
            clip_work: 0,
            erosion_work: 0,
            materialized: Vec::new(),
            union_seen: Vec::new(),
            union_epoch: 0,
            polygonal_proofs: Vec::new(),
            scales,
            max_buffer_bytes,
        }
    }

    fn node<'a>(
        &mut self,
        kind: Kind<'a>,
        path: &[usize],
        depth: usize,
        semantic_item: Option<&'a Render2dItem>,
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        self.node_with_source_admission(kind, path, depth, semantic_item, false)
    }

    fn node_with_source_admission<'a>(
        &mut self,
        kind: Kind<'a>,
        path: &[usize],
        depth: usize,
        semantic_item: Option<&'a Render2dItem>,
        source_prepaid: bool,
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        let next = self
            .nodes
            .checked_add(1)
            .ok_or_else(|| resource(path, "neutral support node count overflow"))?;
        let paths = self
            .retained_path_units
            .checked_add(path.len())
            .ok_or_else(|| resource(path, "neutral support graph owner-path budget overflow"))?;
        if next > MAX_REGION_NODES || depth > MAX_REGION_DEPTH || paths > MAX_REGION_PATH_UNITS {
            return Err(resource(
                path,
                "neutral support graph exceeds node/depth/path admission",
            ));
        }
        // A tight direct-leaf projection visits its ACTUAL mesh vertices;
        // admit that work invocation-wide rather than hiding an unbounded
        // traversal behind a cached semantic-looking rectangle.
        let mesh_visits = match &kind {
            Kind::Affine { source, .. } => match &source.kind {
                Kind::Source(mesh) if mesh.glyphs.is_none() => {
                    mesh.get().map_or(0, |mesh| mesh.triangles.len())
                }
                _ => 0,
            },
            _ => 0,
        };
        let original_visits = match &kind {
            Kind::Source(_) if !source_prepaid => {
                semantic_item.map_or(0, source_envelope::control_visits)
            }
            Kind::Affine { source, .. } => source
                .semantic_item
                .map_or(0, source_envelope::control_visits),
            _ => 0,
        };
        let stroke = (!source_prepaid)
            .then_some(semantic_item)
            .flatten()
            .or_else(|| match &kind {
                Kind::Affine { source, .. } => source.semantic_item,
                _ => None,
            })
            .is_some_and(|item| {
                matches!(
                    item.primitive(),
                    crate::composition_2d::Render2dPrimitive::Stroke { .. }
                )
            });
        let work = mesh_visits
            .checked_mul(AFFINE_REACH_VERTEX_WORK)
            .and_then(|cost| {
                original_visits
                    .checked_mul(SOURCE_REACH_CONTROL_WORK)
                    .and_then(|controls| cost.checked_add(controls))
            })
            .and_then(|cost| {
                path.len()
                    .checked_mul(2)
                    .and_then(|copy| cost.checked_add(copy))
            })
            .and_then(|cost| cost.checked_add(if source_prepaid { 0 } else { REGION_NODE_WORK }))
            .and_then(|cost| cost.checked_add(if stroke { SOURCE_REACH_STROKE_WORK } else { 0 }))
            .ok_or_else(|| resource(path, "original F1 source-reach work overflow"))?;
        let affine_work = self
            .graph_work
            .checked_add(work)
            .ok_or_else(|| resource(path, "neutral affine reach work overflow"))?;
        if affine_work > MAX_REGION_GRAPH_WORK {
            return Err(resource(
                path,
                "neutral affine reach exceeds bounded whole-invocation work",
            ));
        }
        // Pay before traversing controls/vertices or computing an envelope.
        // A later Precision failure did spend this work, but cannot commit a
        // node ID or retained owner path. Curved bounds remain reach-only.
        self.graph_work = affine_work;
        let envelope = region_envelope(&kind, semantic_item, path)?;
        // Carry original-source reach provenance through the same immutable
        // DAG. A glyph outline reconstructed into physical triangles is NOT
        // a conservative authored Bézier control-hull certificate.
        let certified_source_reach = match &kind {
            Kind::Source(source) => {
                source.glyphs.is_some()
                    || semantic_item.is_none_or(|item| {
                        !matches!(
                            item.primitive(),
                            crate::composition_2d::Render2dPrimitive::ShapedText(_)
                        )
                    })
            }
            Kind::Union(children) => children.iter().all(|child| child.certified_source_reach),
            Kind::Affine { source, .. }
            | Kind::Clip { source, .. }
            | Kind::Effect { source, .. } => source.certified_source_reach,
        };
        let contains_positive_disk = match &kind {
            Kind::Source(_) => false,
            Kind::Union(children) => children.iter().any(|child| child.contains_positive_disk),
            Kind::Affine { source, .. } | Kind::Clip { source, .. } => {
                source.contains_positive_disk
            }
            Kind::Effect { source, shadow } => {
                shadow.spread() > 0.0 || source.contains_positive_disk
            }
        };
        let contains_negative_disk = match &kind {
            Kind::Source(_) => false,
            Kind::Union(children) => children.iter().any(|child| child.contains_negative_disk),
            Kind::Affine { source, .. } | Kind::Clip { source, .. } => {
                source.contains_negative_disk
            }
            Kind::Effect { source, shadow } => {
                shadow.spread() < 0.0 || source.contains_negative_disk
            }
        };
        let mut owner_path = Vec::new();
        self.host
            .reserve_vector(&mut owner_path, path.len(), path)?;
        let node_bytes = size_of::<NeutralRegion<'_>>() + 2 * size_of::<usize>();
        if let Err(error) = self.host.reserve(node_bytes, path) {
            self.host
                .release(owner_path.capacity() * size_of::<usize>());
            return Err(error);
        }
        owner_path.extend_from_slice(path);
        self.constructed_node_bytes = self
            .constructed_node_bytes
            .checked_add(size_of::<NeutralRegion<'_>>() + 2 * size_of::<usize>())
            .and_then(|bytes| {
                bytes.checked_add(owner_path.capacity().checked_mul(size_of::<usize>())?)
            })
            .and_then(|bytes| match &kind {
                Kind::Union(children) => bytes.checked_add(
                    children
                        .capacity()
                        .checked_mul(size_of::<Arc<NeutralRegion<'_>>>())?,
                ),
                Kind::Source(source) => match source.get() {
                    Some(mesh) => bytes.checked_add(Self::mesh_bytes(mesh)?),
                    None => Some(bytes),
                }
                .and_then(|bytes| {
                    bytes.checked_add(
                        source
                            .glyphs
                            .as_ref()
                            .map_or(Some(0), GlyphSources::storage_bytes)?,
                    )
                }),
                _ => Some(bytes),
            })
            .ok_or_else(|| resource(path, "neutral node storage extent overflow"))?;
        self.nodes = next;
        self.retained_path_units = paths;
        self.max_owner_path = self.max_owner_path.max(path.len());
        Ok(Arc::new(NeutralRegion {
            kind,
            semantic_item,
            path: owner_path,
            id: next,
            depth,
            envelope,
            certified_source_reach,
            contains_positive_disk,
            contains_negative_disk,
        }))
    }

    pub(super) fn storage_limit(&self) -> u64 {
        self.max_buffer_bytes
    }

    pub(super) fn into_host(self) -> support::PreparationHostBudget {
        self.host
    }

    /// A normal capability result may transport one admitted node's owner
    /// path. Reserve the observed maximum BEFORE materialization can produce
    /// it; the caller either adopts its actual capacity or releases the lease.
    pub(super) fn reserve_diagnostic_path(
        &mut self,
        path: &[usize],
    ) -> Result<usize, Render2dExecutionError> {
        let work = self
            .max_owner_path
            .checked_mul(2)
            .and_then(|copy| copy.checked_add(64))
            .ok_or_else(|| resource(path, "source diagnostic path work overflow"))?;
        self.charge_graph_work(work, path)?;
        let bytes = self
            .max_owner_path
            .checked_mul(size_of::<usize>())
            .ok_or_else(|| resource(path, "source diagnostic path extent overflow"))?;
        self.host.reserve(bytes, path)?;
        Ok(bytes)
    }

    pub(super) fn charge_preparation(
        &mut self,
        cost: usize,
        path: &[usize],
    ) -> Result<(), Render2dExecutionError> {
        self.charge_graph_work(cost, path)
    }

    /// Admit the complete source-selection phase before allocation. The
    /// caller's retained/construction receipt already includes the existing
    /// 512 * 128-byte traversal allowance; explicit DFS frames fit within it.
    /// The tables and their headers are additional coexisting storage.
    pub(super) fn polygon_dependencies<'scene>(
        &mut self,
        consumers: usize,
        source_phase_bytes: usize,
        max_host_bytes: u64,
    ) -> Result<PolygonDependencies<'scene>, Render2dExecutionError> {
        // Scalar overflow/admission checks are work even on a rejected call.
        self.charge_graph_work(128, &[])?;
        let slots = self
            .nodes
            .checked_add(1)
            .ok_or_else(|| resource(&[], "polygon dependency index overflow"))?;
        let entries = slots
            .checked_add(consumers)
            .ok_or_else(|| resource(&[], "polygon dependency extent overflow"))?;
        let metadata_bytes = entries
            .checked_add(size_of::<PolygonDependencies<'_>>())
            .ok_or_else(|| resource(&[], "polygon dependency bytes overflow"))?;
        let phase_bytes = source_phase_bytes
            .checked_add(metadata_bytes)
            .ok_or_else(|| resource(&[], "polygon dependency phase bytes overflow"))?;
        if u64::try_from(phase_bytes)
            .ok()
            .is_none_or(|bytes| bytes > max_host_bytes.min(self.storage_limit()))
        {
            return Err(resource(
                &[],
                "polygon dependency source phase exceeds host admission",
            ));
        }
        self.charge_graph_work(
            entries
                .checked_mul(8)
                .ok_or_else(|| resource(&[], "polygon dependency initialization work overflow"))?,
            &[],
        )?;
        self.charge_graph_work(256, &[])?;
        self.host
            .reserve(size_of::<PolygonDependencies<'_>>(), &[])?;
        fn table(
            count: usize,
            host: &mut support::PreparationHostBudget,
        ) -> Result<Vec<bool>, Render2dExecutionError> {
            let mut values = Vec::new();
            host.reserve_vector(&mut values, count, &[])?;
            values.resize(count, false);
            Ok(values)
        }
        // No initialized frame array or compiler-dependent recursion stack.
        // Arc + next-child index is at most 128 bytes on the supported host;
        // at most MAX_REGION_DEPTH frames can coexist by the DAG invariant.
        const { assert!(size_of::<(Arc<NeutralRegion<'_>>, usize)>() <= 128) };
        let mut frames = Vec::new();
        self.host
            .reserve_vector(&mut frames, MAX_REGION_DEPTH, &[])?;
        Ok(PolygonDependencies {
            nodes: table(slots, &mut self.host)?,
            consumers: table(consumers, &mut self.host)?,
            frames,
            metadata_bytes,
        })
    }

    /// Mark physical dependencies with one bounded DFS frame vector reused by
    /// every selected root. Each node visit (including cache hits) prepays its
    /// flag, Arc/frame operations and the parent's next-child bookkeeping.
    pub(super) fn require_polygon_dependencies<'scene>(
        &mut self,
        source: &Arc<NeutralRegion<'scene>>,
        dependencies: &mut PolygonDependencies<'scene>,
    ) -> Result<(), Render2dExecutionError> {
        self.charge_graph_work(64, &source.path)?;
        debug_assert!(dependencies.frames.is_empty());
        dependencies.frames.push((Arc::clone(source), 0));
        while let Some((node, next)) = dependencies.frames.last_mut() {
            if *next == 0 {
                let needed = dependencies.nodes.get_mut(node.id).ok_or_else(|| {
                    precision(
                        &node.path,
                        "polygon dependency ID exceeds the invocation inventory",
                    )
                })?;
                if *needed {
                    dependencies.frames.pop();
                    continue;
                }
                *needed = true;
            }
            let child = match &node.kind {
                Kind::Source(_) => None,
                Kind::Union(children) => children.get(*next),
                Kind::Affine { source, .. }
                | Kind::Clip { source, .. }
                | Kind::Effect { source, .. } => (*next == 0).then_some(source),
            };
            if let Some(child) = child {
                self.charge_graph_work(64, &child.path)?;
                *next += 1;
                let child = Arc::clone(child);
                if dependencies.frames.len() == MAX_REGION_DEPTH {
                    return Err(precision(
                        &child.path,
                        "polygon dependency depth exceeds inventory",
                    ));
                }
                dependencies.frames.push((child, 0));
            } else {
                dependencies.frames.pop();
            }
        }
        Ok(())
    }

    fn mesh_bytes(mesh: &support::NeutralMesh) -> Option<usize> {
        mesh.triangles
            .capacity()
            .checked_mul(size_of::<[f64; 2]>())?
            .checked_add(size_of::<support::NeutralMesh>() + 2 * size_of::<usize>())
    }

    /// Allocation-capacity inventory, not a membership or backend-memory claim.
    /// IDs deduplicate shared DAG nodes. Cached mesh entries are counted
    /// conservatively; a source seed shared with its own cache slot pays once.
    pub(super) fn storage_bounds<'node, 'borrow>(
        &mut self,
        roots: impl IntoIterator<Item = &'borrow Arc<NeutralRegion<'node>>>,
    ) -> Result<(usize, usize), Render2dExecutionError>
    where
        'node: 'borrow,
    {
        fn visit(
            node: &NeutralRegion<'_>,
            seen: &mut [bool],
            cached: &[Option<Arc<support::NeutralMesh>>],
            fuel: &mut usize,
        ) -> Option<usize> {
            *fuel = fuel.checked_sub(64)?;
            if *seen.get(node.id)? {
                return Some(0);
            }
            seen[node.id] = true;
            let mut bytes = size_of::<NeutralRegion<'_>>() + 2 * size_of::<usize>();
            bytes = bytes.checked_add(node.path.capacity().checked_mul(size_of::<usize>())?)?;
            match &node.kind {
                Kind::Source(source) => {
                    bytes = bytes.checked_add(
                        source
                            .glyphs
                            .as_ref()
                            .map_or(Some(0), GlyphSources::storage_bytes)?,
                    )?;
                    if cached.get(node.id).is_none_or(Option::is_none)
                        && let Some(mesh) = source.get()
                    {
                        bytes = bytes.checked_add(NeutralRegionBuilder::mesh_bytes(mesh)?)?;
                    }
                }
                Kind::Union(children) => {
                    bytes = bytes.checked_add(
                        children
                            .capacity()
                            .checked_mul(size_of::<Arc<NeutralRegion<'_>>>())?,
                    )?;
                    for child in children {
                        bytes = bytes.checked_add(visit(child, seen, cached, fuel)?)?;
                    }
                }
                Kind::Affine { source, .. }
                | Kind::Clip { source, .. }
                | Kind::Effect { source, .. } => {
                    bytes = bytes.checked_add(visit(source, seen, cached, fuel)?)?;
                }
            }
            Some(bytes)
        }
        let slots = self
            .nodes
            .checked_add(1)
            .ok_or_else(|| resource(&[], "source storage index overflow"))?;
        self.charge_graph_work(
            slots
                .checked_mul(8)
                .and_then(|v| v.checked_add(128))
                .ok_or_else(|| resource(&[], "source storage work overflow"))?,
            &[],
        )?;
        let mut seen = Vec::new();
        self.host.reserve_vector(&mut seen, slots, &[])?;
        seen.resize(slots, false);
        // Preserve the existing structural traversal allowance. This is not
        // a measurement or proof of compiler-generated Rust stack frames.
        self.host.reserve(MAX_REGION_DEPTH * 128, &[])?;
        let mut fuel = MAX_REGION_GRAPH_WORK - self.graph_work;
        let result = (|| {
            let mut bytes = size_of::<Self>()
                .checked_add(
                    self.materialized
                        .capacity()
                        .checked_mul(size_of::<Option<Arc<support::NeutralMesh>>>())?,
                )?
                .checked_add(self.union_seen.capacity().checked_mul(size_of::<usize>())?)?
                .checked_add(
                    self.polygonal_proofs
                        .capacity()
                        .checked_mul(size_of::<Option<bool>>())?,
                )?;
            let headers = bytes;
            let mut cached_bytes = 0_usize;
            for mesh in self.materialized.iter().flatten() {
                fuel = fuel.checked_sub(32)?;
                cached_bytes = cached_bytes.checked_add(Self::mesh_bytes(mesh)?)?;
            }
            bytes = bytes.checked_add(cached_bytes)?;
            for root in roots {
                bytes =
                    bytes.checked_add(visit(root, &mut seen, &self.materialized, &mut fuel)?)?;
            }
            // Construction may retain nodes later discarded by normalization.
            // The temporary seen table and bounded traversal frames coexist.
            let construction = self
                .constructed_node_bytes
                // At most one table is relocating at a time. Twice the
                // final capacities covers the old and new allocations.
                .checked_add(headers.checked_mul(2)?)?
                .checked_add(cached_bytes)?
                .checked_add(seen.capacity())?
                .checked_add(MAX_REGION_DEPTH * 128)?;
            Some((bytes, construction))
        })();
        self.graph_work = MAX_REGION_GRAPH_WORK - fuel;
        self.host.release(seen.capacity() + MAX_REGION_DEPTH * 128);
        result.ok_or_else(|| resource(&[], "source storage inventory exceeds admission"))
    }

    /// Synthetic test fixtures have no original F1 item to borrow. They may
    /// certify physical mesh geometry only, not arbitrary authoring semantics.
    #[cfg(test)]
    pub(super) fn source<'a>(
        &mut self,
        mesh: support::NeutralMesh,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        self.source_impl(mesh, None, path, false, 0)
    }

    /// Preserve the original borrowed F1 authoring item beside disposable
    /// tessellation without cloning it or constructing a second semantic tree.
    #[cfg(test)]
    pub(super) fn source_from_item<'a>(
        &mut self,
        mesh: support::NeutralMesh,
        item: &'a Render2dItem,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        self.source_impl(mesh, Some(item), path, false, 0)
    }

    /// Admit the existing original-source reach and node allowance BEFORE
    /// entering disposable physical reconstruction. Its amount and purpose
    /// are unchanged: this is not a bound on Lyon's internal primitive work.
    /// Keep prepayment, the live leaf counters and consumption inside this
    /// builder call, so no receipt can be reused or moved to another budget.
    pub(super) fn prepare_source_from_item<'a>(
        &mut self,
        item: &'a Render2dItem,
        path: &[usize],
        prepare: impl FnOnce(
            &mut shadow::NeutralPreparationWork<'_>,
        ) -> Result<Option<support::NeutralMesh>, Render2dExecutionError>,
    ) -> Result<Option<Arc<NeutralRegion<'a>>>, Render2dExecutionError> {
        let controls = source_envelope::control_visits(item);
        let stroke = matches!(
            item.primitive(),
            crate::composition_2d::Render2dPrimitive::Stroke { .. }
        );
        let cost = controls
            .checked_mul(SOURCE_REACH_CONTROL_WORK)
            .and_then(|cost| cost.checked_add(if stroke { SOURCE_REACH_STROKE_WORK } else { 0 }))
            .and_then(|cost| cost.checked_add(REGION_NODE_WORK))
            .ok_or_else(|| resource(path, "neutral source preparation admission overflow"))?;
        self.charge_graph_work(cost, path)?;
        let host_before = self.host.live();
        let result = (|| {
            let mesh = prepare(&mut shadow::NeutralPreparationWork::new(
                &mut self.graph_work,
                &mut self.clip_work,
                &mut self.host,
            ))?;
            let Some(mesh) = mesh else {
                return Ok(None);
            };
            let factory_bytes = self.host.live() - host_before;
            self.source_impl(mesh, Some(item), path, true, factory_bytes)
                .map(Some)
        })();
        if result.is_err() || matches!(result, Ok(None)) {
            self.host.release(self.host.live() - host_before);
        }
        result
    }

    /// Register original F1 meaning without asking a polygon consumer to
    /// reconstruct it. Absence is established only by original source facts.
    pub(super) fn register_source<'a>(
        &mut self,
        item: &'a Render2dItem,
        path: &[usize],
    ) -> Result<Option<Arc<NeutralRegion<'a>>>, Render2dExecutionError> {
        let controls = source_envelope::control_visits(item);
        let stroke = matches!(
            item.primitive(),
            crate::composition_2d::Render2dPrimitive::Stroke { .. }
        );
        let cost = controls
            .checked_mul(SOURCE_REACH_CONTROL_WORK)
            .and_then(|v| v.checked_add(if stroke { SOURCE_REACH_STROKE_WORK } else { 0 }))
            .and_then(|v| v.checked_add(REGION_NODE_WORK))
            .ok_or_else(|| resource(path, "neutral source registration work overflow"))?;
        self.charge_graph_work(cost, path)?;
        if trivially_empty_original_source(item) {
            return Ok(None);
        }
        self.node_with_source_admission(
            Kind::Source(SourceGeometry::deferred()),
            path,
            1,
            Some(item),
            true,
        )
        .map(Some)
    }

    /// Resolve original glyph reach and shared controls from already validated
    /// F2 facts before any polygon factory. The hull remains absence-only;
    /// membership uses original winding and separately retained placement.
    pub(super) fn register_shaped_source<'a>(
        &mut self,
        item: &'a Render2dItem,
        resource: &crate::composition_2d::Render2dShapedTextResource,
        fields: &crate::runtime::execution_2d::field::ResourceFields,
        path: &[usize],
    ) -> Result<Option<Arc<NeutralRegion<'a>>>, Render2dExecutionError> {
        self.charge_graph_work(REGION_NODE_WORK, path)?;
        let glyphs = source_envelope::shaped_source(item, resource, fields, path, self)?;
        let Some(glyphs) = glyphs else {
            // Every admitted glyph was empty. No disposable factory or mesh
            // is needed to establish absence of original outline controls.
            return Ok(None);
        };
        if trivially_empty_original_source(item) {
            return Ok(None);
        }
        self.node_with_source_admission(
            Kind::Source(SourceGeometry {
                mesh: OnceLock::new(),
                glyphs: Some(glyphs),
            }),
            path,
            1,
            Some(item),
            true,
        )
        .map(Some)
    }

    /// Explicit polygon demand. The factory is invocation-scoped, not retained
    /// by a source leaf; failed reconstruction installs no physical receipt.
    pub(super) fn prepare_registered_source(
        &mut self,
        region: &NeutralRegion<'_>,
        prepare: impl FnOnce(
            &Render2dItem,
            &[usize],
            &mut shadow::NeutralPreparationWork<'_>,
        ) -> Result<Option<support::NeutralMesh>, Render2dExecutionError>,
    ) -> Result<(), Render2dExecutionError> {
        let Kind::Source(source) = &region.kind else {
            return Err(unsupported(&region.path));
        };
        if source.get().is_some() {
            self.charge_graph_work(MATERIALIZATION_LOOKUP_WORK, &region.path)?;
            return Ok(());
        }
        let item = region
            .semantic_item
            .ok_or_else(|| unsupported(&region.path))?;
        let host_before = self.host.live();
        let result = (|| {
            let mut mesh = prepare(
                item,
                &region.path,
                &mut shadow::NeutralPreparationWork::new(
                    &mut self.graph_work,
                    &mut self.clip_work,
                    &mut self.host,
                ),
            )?
            .ok_or_else(|| {
                precision(
                    &region.path,
                    "registered nonempty source lost its original support",
                )
            })?;
            let factory_bytes = self.host.live() - host_before;
            let retained = self.admit_source_mesh(&mut mesh, &region.path, factory_bytes)?;
            source
                .mesh
                .set(Arc::new(mesh))
                .expect("one invocation owns source reconstruction");
            self.retained_vertices = retained;
            Ok(())
        })();
        if result.is_err() {
            self.host.release(self.host.live() - host_before);
        }
        result
    }

    /// Shared physical admission for seeded and explicitly requested meshes.
    /// Bounds are disposable metadata; retained counters commit only after the
    /// owning node or once-only realization succeeds.
    fn admit_source_mesh(
        &mut self,
        mesh: &mut support::NeutralMesh,
        path: &[usize],
        factory_bytes: usize,
    ) -> Result<usize, Render2dExecutionError> {
        // The cached tessellation AABB is disposable physical metadata, not
        // F1 semantic authority. Certify the support envelope from the
        // ACTUAL retained vertices once, and normalize only after ensuring
        // scale/vertex representability. Inconsistent stored bounds cannot
        // cause a curved ancestor effect to be culled prematurely.
        if !mesh.triangles.len().is_multiple_of(3)
            || !mesh.units_per_parent_logical_unit.is_finite()
            || mesh.units_per_parent_logical_unit <= 0.0
        {
            return Err(precision(
                path,
                "neutral-region mesh scale or topology is invalid",
            ));
        }
        let scan_work = mesh
            .triangles
            .len()
            .checked_mul(SOURCE_VERTEX_WORK)
            .and_then(|cost| cost.checked_add(64))
            .ok_or_else(|| resource(path, "neutral source scan work overflow"))?;
        self.charge_graph_work(scan_work, path)?;
        // Transfer factory-owned capacity into the retained mesh, without
        // billing the same allocation twice. Legacy vector factories still
        // expose only their returned mesh: this does not certify Lyon scratch.
        let bytes = Self::mesh_bytes(mesh)
            .ok_or_else(|| resource(path, "source mesh retained storage overflow"))?;
        if bytes >= factory_bytes {
            self.host.reserve(bytes - factory_bytes, path)?;
        } else {
            self.host.release(factory_bytes - bytes);
        }
        let mut bounds = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for &[x, y] in &mesh.triangles {
            if !x.is_finite() || !y.is_finite() {
                return Err(precision(path, "neutral-region source vertex is nonfinite"));
            }
            bounds[0] = bounds[0].min(x);
            bounds[1] = bounds[1].min(y);
            bounds[2] = bounds[2].max(x);
            bounds[3] = bounds[3].max(y);
        }
        mesh.bounds = bounds;
        // A rejected graph node MUST NOT consume retained geometry budget.
        let mut retained = self.retained_vertices;
        shadow::charge_vertices(&mut retained, mesh.triangles.len(), path)?;
        Ok(retained)
    }

    fn source_impl<'a>(
        &mut self,
        mut mesh: support::NeutralMesh,
        semantic_item: Option<&'a Render2dItem>,
        path: &[usize],
        source_prepaid: bool,
        factory_bytes: usize,
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        let retained = self.admit_source_mesh(&mut mesh, path, factory_bytes)?;
        let node = self.node_with_source_admission(
            Kind::Source(SourceGeometry::prepared(Arc::new(mesh))),
            path,
            1,
            semantic_item,
            source_prepaid,
        )?;
        self.retained_vertices = retained;
        Ok(node)
    }

    pub(super) fn union<'a>(
        &mut self,
        mut children: Vec<Arc<NeutralRegion<'a>>>,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        let scan_work = children
            .len()
            .checked_mul(UNION_CHILD_WORK)
            .and_then(|cost| cost.checked_add(UNION_FIXED_WORK))
            .ok_or_else(|| resource(path, "neutral union scan work overflow"))?;
        self.charge_graph_work(scan_work, path)?;
        if children.len() == 1 {
            return Ok(children.pop().expect("single union member"));
        }
        if children.len() > 1 {
            // Exact set-union idempotence, not paint-order deduplication.
            // Keep the FIRST occurrence to preserve authored error paths.
            // A bounded stamp table avoids the O(n^2) duplicate search and
            // O(total DAG nodes) repeated allocation per group.
            let epoch = self
                .union_epoch
                .checked_add(1)
                .ok_or_else(|| resource(path, "neutral union stamp overflow"))?;
            let needed = children
                .iter()
                .map(|node| node.id)
                .max()
                .unwrap_or(0)
                .checked_add(1)
                .ok_or_else(|| resource(path, "neutral union stamp size overflow"))?;
            let bytes = needed
                .checked_mul(std::mem::size_of::<usize>())
                .ok_or_else(|| resource(path, "neutral union stamp bytes overflow"))?;
            if needed > MAX_REGION_NODES + 1
                || u64::try_from(bytes)
                    .ok()
                    .is_none_or(|bytes| bytes > self.max_buffer_bytes)
            {
                return Err(resource(path, "neutral union stamp exceeds bounded memory"));
            }
            if needed > self.union_seen.len() {
                let old_len = self.union_seen.len();
                let grows = needed > self.union_seen.capacity();
                let relocation = if grows { old_len } else { 0 };
                let work = (needed - old_len)
                    .checked_add(relocation)
                    .and_then(|slots| slots.checked_mul(2))
                    .and_then(|cost| cost.checked_add(64))
                    .ok_or_else(|| resource(path, "neutral union stamp work overflow"))?;
                self.charge_graph_work(work, path)?;
                // Bounded geometric capacity prevents an exact one-slot
                // reservation at every new union from repeatedly relocating
                // the whole invocation's stamp table. Initialize only live IDs.
                let capacity_limit = usize::try_from(
                    (self.max_buffer_bytes / std::mem::size_of::<usize>() as u64)
                        .min((MAX_REGION_NODES + 1) as u64),
                )
                .map_err(|_| resource(path, "neutral union capacity limit overflow"))?;
                let desired = if grows {
                    self.union_seen
                        .capacity()
                        .saturating_mul(2)
                        .min(capacity_limit)
                        .max(needed)
                } else {
                    self.union_seen.capacity()
                };
                let capacity_bytes = desired
                    .checked_mul(std::mem::size_of::<usize>())
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .ok_or_else(|| resource(path, "neutral union capacity bytes overflow"))?;
                if capacity_bytes > self.max_buffer_bytes {
                    return Err(resource(
                        path,
                        "neutral union capacity exceeds bounded memory",
                    ));
                }
                self.host
                    .reserve_vector(&mut self.union_seen, desired, path)?;
                self.union_seen.resize(needed, 0);
            }
            // Even when node admission fails after normalization, advance
            // the transient stamp generation so a subsequent safe retry can
            // never mistake the rejected attempt's IDs for current marks.
            self.union_epoch = epoch;
            children.retain(|child| {
                let old = std::mem::replace(&mut self.union_seen[child.id], epoch);
                old != epoch
            });
            let result = if children.len() == 1 {
                children.pop().expect("deduplicated single union member")
            } else {
                let depth = children.iter().map(|node| node.depth).max().unwrap_or(0) + 1;
                self.node(Kind::Union(children), path, depth, None)?
            };
            return Ok(result);
        }
        self.node(Kind::Union(children), path, 1, None)
    }

    fn charge_graph_work(
        &mut self,
        work: usize,
        path: &[usize],
    ) -> Result<(), Render2dExecutionError> {
        let next = self
            .graph_work
            .checked_add(work)
            .ok_or_else(|| resource(path, "neutral graph work overflow"))?;
        if next > MAX_REGION_GRAPH_WORK {
            return Err(resource(
                path,
                "neutral graph exceeds whole-invocation work",
            ));
        }
        self.graph_work = next;
        Ok(())
    }

    pub(super) fn affine<'a>(
        &mut self,
        source: Arc<NeutralRegion<'a>>,
        to_parent: Render2dAffineTransform,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        // An identity map does not introduce a geometric frame change. Keep
        // the original borrowed source and owner path instead of rescanning
        // its controls/vertices and retaining a redundant support node.
        // Group clips and effects are separate operators in the caller.
        self.charge_graph_work(64, path)?;
        if to_parent.components() == Render2dAffineTransform::IDENTITY.components() {
            return Ok(source);
        }
        let depth = source.depth + 1;
        self.node(Kind::Affine { source, to_parent }, path, depth, None)
    }

    pub(super) fn clip<'a>(
        &mut self,
        source: Arc<NeutralRegion<'a>>,
        clips: &'a [Render2dClip],
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        if clips.is_empty() {
            return Ok(source);
        }
        let depth = source.depth + 1;
        self.node(Kind::Clip { source, clips }, path, depth, None)
    }

    pub(super) fn effect<'a>(
        &mut self,
        source: Arc<NeutralRegion<'a>>,
        shadow: Render2dDropShadow,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        let depth = source.depth + 1;
        self.node(Kind::Effect { source, shadow }, path, depth, None)
    }

    /// An authored rectangle classification does not certify that a
    /// disposable tessellation contains the SAME support. A forged, stale or
    /// rounded physical rectangle may not silently replace the original F1
    /// extent in a curved Minkowski specialization. Prove each source's four
    /// transformed corners exactly in its own immediate-parent physical frame.
    ///
    /// This is intentionally a narrow *exact* shortcut. Multi-patch images,
    /// any clips and coordinates requiring tessellation rounding must use
    /// the future general original-F1 integration path instead.
    /// A single structural F1 path can be proven rectangular only from
    /// its ORIGINAL straight commands, never from an approximately similar
    /// Lyon triangle union. Both fill rules agree on one simple contour.
    /// Open fill contours use F1's specified implicit closing edge.
    /// Any extra contour, Bézier, duplicated corner or crossed diagonal
    /// is unproved and must retain the existing typed fallback.
    fn original_rectangle_path_corners(
        path: &crate::composition_2d::Render2dPath,
    ) -> Option<[[f64; 2]; 4]> {
        use crate::composition_2d::Render2dPathCommand as Command;
        let [
            Command::MoveTo(a),
            Command::LineTo(b),
            Command::LineTo(c),
            Command::LineTo(d),
            rest @ ..,
        ] = path.commands()
        else {
            return None;
        };
        if !(rest.is_empty()
            || matches!(rest, [Command::Close])
            || matches!(rest, [Command::LineTo(end)] if end == a)
            || matches!(rest, [Command::LineTo(end), Command::Close] if end == a))
        {
            return None;
        }
        let points = [
            [a.x(), a.y()],
            [b.x(), b.y()],
            [c.x(), c.y()],
            [d.x(), d.y()],
        ];
        let x0 = points.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
        let x1 = points
            .iter()
            .map(|p| p[0])
            .fold(f64::NEG_INFINITY, f64::max);
        let y0 = points.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
        let y1 = points
            .iter()
            .map(|p| p[1])
            .fold(f64::NEG_INFINITY, f64::max);
        if !(x0 < x1 && y0 < y1) {
            return None;
        }
        let corners = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
        let mut seen = 0_u8;
        for i in 0..4 {
            let p = points[i];
            let next = points[(i + 1) % 4];
            // Exactly ONE coordinate changes along each boundary edge.
            if (p[0] == next[0]) == (p[1] == next[1]) {
                return None;
            }
            let index = corners.iter().position(|corner| *corner == p)?;
            let bit = 1_u8 << index;
            if seen & bit != 0 {
                return None;
            }
            seen |= bit;
        }
        (seen == 0b1111).then_some(points)
    }

    fn exact_original_rectangle_source(mesh: &support::NeutralMesh, item: &Render2dItem) -> bool {
        let (rect, path) = match item.primitive() {
            crate::composition_2d::Render2dPrimitive::Fill { shape, .. } => {
                (shape.as_rect(), shape.as_path())
            }
            crate::composition_2d::Render2dPrimitive::Image(image) => {
                if image.patches().len() != 1 {
                    return false;
                }
                (Some(image.patches()[0].destination()), None)
            }
            crate::composition_2d::Render2dPrimitive::Stroke { .. }
            | crate::composition_2d::Render2dPrimitive::ShapedText(_) => (None, None),
        };
        if !item.clips().is_empty() {
            return false;
        }
        let authored = if let Some(rect) = rect {
            if rect.is_empty() {
                return false;
            }
            let x0 = rect.x();
            let x1 = rect.x() + rect.width();
            let y0 = rect.y();
            let y1 = rect.y() + rect.height();
            [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
        } else if let Some(path) = path {
            let Some(corners) = Self::original_rectangle_path_corners(path) else {
                return false;
            };
            corners
        } else {
            return false;
        };
        let Some(actual) = support::certified_axis_aligned_rectangle(&mesh.triangles) else {
            return false;
        };
        let [a, b, c, d, tx, ty] = item.local_to_parent().components();
        let mut seen = 0_u8;
        for [x, y] in authored {
            let px = a.mul_add(x, c.mul_add(y, tx)) * mesh.units_per_parent_logical_unit;
            let py = b.mul_add(x, d.mul_add(y, ty)) * mesh.units_per_parent_logical_unit;
            if !px.is_finite() || !py.is_finite() {
                return false;
            }
            let corners = [
                [actual[0], actual[1]],
                [actual[2], actual[1]],
                [actual[2], actual[3]],
                [actual[0], actual[3]],
            ];
            let Some(index) = corners.iter().position(|corner| *corner == [px, py]) else {
                return false;
            };
            seen |= 1 << index;
        }
        seen == 0b1111
    }

    /// Original command grammar shared with physical triangle correspondence.
    fn original_triangle_path_points(
        path: &crate::composition_2d::Render2dPath,
    ) -> Option<[[f64; 2]; 3]> {
        use crate::composition_2d::Render2dPathCommand as Command;
        let [
            Command::MoveTo(a),
            Command::LineTo(b),
            Command::LineTo(c),
            rest @ ..,
        ] = path.commands()
        else {
            return None;
        };
        if !(rest.is_empty()
            || matches!(rest, [Command::Close])
            || matches!(rest, [Command::LineTo(end)] if end == a)
            || matches!(rest, [Command::LineTo(end), Command::Close] if end == a))
        {
            return None;
        }
        Some([[a.x(), a.y()], [b.x(), b.y()], [c.x(), c.y()]])
    }

    /// Necessary original-shape class for the existing physical correspondence
    /// proof. False is conclusive without asking for a disposable mesh; true
    /// still requires that unchanged proof, including all transformed vertices.
    fn original_polygon_source_possible(item: &Render2dItem) -> bool {
        if !item.clips().is_empty() {
            return false;
        }
        match item.primitive() {
            crate::composition_2d::Render2dPrimitive::Fill { shape, .. } => {
                shape.as_rect().is_some()
                    || shape.as_path().is_some_and(|path| {
                        Self::original_rectangle_path_corners(path).is_some()
                            || Self::original_triangle_path_points(path).is_some()
                    })
            }
            crate::composition_2d::Render2dPrimitive::Image(image) => image.patches().len() == 1,
            _ => false,
        }
    }

    /// A structural F1 triangle admits exact signed Euclidean support only
    /// if its ONLY filled contour is three original straight vertices and
    /// the disposable physical tessellation is precisely the same triangle.
    /// Neither its AABB nor three physical corners alone can certify a
    /// curved, crossed, multi-contour, clipped or subdivided authored source.
    fn exact_original_triangle_source(mesh: &support::NeutralMesh, item: &Render2dItem) -> bool {
        if !item.clips().is_empty() {
            return false;
        }
        let crate::composition_2d::Render2dPrimitive::Fill { shape, .. } = item.primitive() else {
            return false;
        };
        let Some(path) = shape.as_path() else {
            return false;
        };
        if mesh.triangles.len() != 3 {
            return false;
        }
        let Some(authored) = Self::original_triangle_path_points(path) else {
            return false;
        };
        let orientation = |points: [[f64; 2]; 3]| {
            let [a, b, c] = points;
            (b[0] - a[0]).mul_add(c[1] - a[1], -((b[1] - a[1]) * (c[0] - a[0])))
        };
        if orientation(authored) == 0.0 || !orientation(authored).is_finite() {
            return false;
        }
        let actual = [mesh.triangles[0], mesh.triangles[1], mesh.triangles[2]];
        if orientation(actual) == 0.0 || !orientation(actual).is_finite() {
            return false;
        }
        let [a, b, c, d, tx, ty] = item.local_to_parent().components();
        let mut seen = 0_u8;
        for [x, y] in authored {
            let px = a.mul_add(x, c.mul_add(y, tx)) * mesh.units_per_parent_logical_unit;
            let py = b.mul_add(x, d.mul_add(y, ty)) * mesh.units_per_parent_logical_unit;
            if !px.is_finite() || !py.is_finite() {
                return false;
            }
            let Some(index) = actual.iter().position(|vertex| *vertex == [px, py]) else {
                return false;
            };
            let flag = 1_u8 << index;
            if seen & flag != 0 {
                return false;
            }
            seen |= flag;
        }
        seen == 0b111
    }

    fn provenance_charge(
        remaining: &mut usize,
        cost: usize,
        path: &[usize],
    ) -> Result<(), Render2dExecutionError> {
        *remaining = remaining
            .checked_sub(cost)
            .ok_or_else(|| resource(path, "original F1 provenance exceeds remaining graph work"))?;
        Ok(())
    }

    fn rectangle_scan_work(
        mesh: &support::NeutralMesh,
        path: &[usize],
    ) -> Result<usize, Render2dExecutionError> {
        mesh.triangles
            .len()
            .checked_mul(RECTANGLE_VERTEX_WORK)
            .and_then(|cost| cost.checked_add(PROVENANCE_NODE_WORK))
            .ok_or_else(|| resource(path, "original rectangle proof work overflow"))
    }

    fn provenance_node_work(region: &NeutralRegion<'_>) -> Result<usize, Render2dExecutionError> {
        let extra = match &region.kind {
            Kind::Source(mesh) if region.semantic_item.is_some() => mesh
                .get()
                .map(|mesh| Self::rectangle_scan_work(mesh, &region.path))
                .transpose()?
                .unwrap_or(0)
                .checked_add(PROVENANCE_SOURCE_WORK),
            // Pay the complete child-edge inventory BEFORE visiting it.
            // Child receipt lookups and uncached bodies pay separately.
            Kind::Union(children) => children.len().checked_mul(PROVENANCE_LOOKUP_WORK),
            _ => Some(0),
        };
        extra
            .and_then(|cost| cost.checked_add(PROVENANCE_NODE_WORK))
            .ok_or_else(|| resource(&region.path, "original source proof work overflow"))
    }

    fn polygonal_source_proven(
        region: &NeutralRegion<'_>,
        remaining: &mut usize,
        memo: &mut [Option<bool>],
    ) -> Result<bool, Render2dExecutionError> {
        Self::provenance_charge(remaining, PROVENANCE_LOOKUP_WORK, &region.path)?;
        if let Some(cached) = memo.get(region.id).copied().flatten() {
            return Ok(cached);
        }
        Self::provenance_charge(remaining, Self::provenance_node_work(region)?, &region.path)?;
        Self::polygonal_source_uncached(region, remaining, memo)
    }

    /// The node body and its entire possible leaf/fanout work were admitted
    /// before entry. Only COMPLETED results become reusable receipts.
    fn polygonal_source_uncached(
        region: &NeutralRegion<'_>,
        remaining: &mut usize,
        memo: &mut [Option<bool>],
    ) -> Result<bool, Render2dExecutionError> {
        let result = match &region.kind {
            Kind::Source(mesh) => {
                // Original shape ineligibility is conclusive independently
                // of a mesh. A potentially eligible unrequested realization
                // still cannot cache a failed physical correspondence.
                if region
                    .semantic_item
                    .is_some_and(|item| !Self::original_polygon_source_possible(item))
                {
                    false
                } else {
                    let mesh = mesh.get().ok_or_else(|| unsupported(&region.path))?;
                    region.semantic_item.is_some_and(|item| {
                        Self::exact_original_rectangle_source(mesh, item)
                            || Self::exact_original_triangle_source(mesh, item)
                    })
                }
            }
            Kind::Union(children) => {
                let mut proven = true;
                for child in children {
                    if !Self::polygonal_source_proven(child, remaining, memo)? {
                        proven = false;
                        break;
                    }
                }
                proven
            }
            Kind::Affine { source, .. } => Self::polygonal_source_proven(source, remaining, memo)?,
            Kind::Clip { source, clips } => {
                // Only the existing no-op clip wrapper qualifies. Curved or
                // transformed clip geometry never inherits a mesh-only proof.
                clips.is_empty() && Self::polygonal_source_proven(source, remaining, memo)?
            }
            Kind::Effect { .. } => false,
        };
        let Some(slot) = memo.get_mut(region.id) else {
            return Err(resource(
                &region.path,
                "original F1 provenance cache index exceeds admitted graph",
            ));
        };
        *slot = Some(result);
        Ok(result)
    }

    fn prepare_polygonal_proof(
        root: &NeutralRegion<'_>,
        remaining: &mut usize,
        memo: &mut Vec<Option<bool>>,
        max_buffer_bytes: u64,
        host: &mut support::PreparationHostBudget,
    ) -> Result<bool, Render2dExecutionError> {
        Self::provenance_charge(remaining, PROVENANCE_LOOKUP_WORK, &root.path)?;
        if let Some(cached) = memo.get(root.id).copied().flatten() {
            return Ok(cached);
        }
        let entries = root
            .id
            .checked_add(1)
            .ok_or_else(|| resource(&root.path, "original F1 provenance index overflow"))?;
        let bytes = entries
            .checked_mul(std::mem::size_of::<Option<bool>>())
            .ok_or_else(|| resource(&root.path, "original F1 provenance bytes overflow"))?;
        if entries > MAX_REGION_NODES + 1
            || u64::try_from(bytes)
                .ok()
                .is_none_or(|bytes| bytes > max_buffer_bytes)
        {
            return Err(resource(
                &root.path,
                "original F1 provenance exceeds bounded memory",
            ));
        }
        let growth = entries.saturating_sub(memo.len());
        let node_work = Self::provenance_node_work(root)?;
        // Exact reservation may relocate every retained receipt even for a
        // single new slot. Pay both its read and write before allocation.
        let relocation = if entries > memo.capacity() {
            memo.len()
        } else {
            0
        };
        let work = growth
            .checked_add(relocation)
            .and_then(|slots| slots.checked_mul(2))
            .and_then(|cost| cost.checked_add(PROVENANCE_NODE_WORK))
            .and_then(|cost| cost.checked_add(node_work))
            .and_then(|cost| cost.checked_add(if entries > memo.capacity() { 64 } else { 0 }))
            .ok_or_else(|| {
                resource(
                    &root.path,
                    "original F1 provenance preparation work overflow",
                )
            })?;
        // Both a large root leaf and a large fanout reject here, before any
        // proof-cache allocation or source/child traversal can begin.
        Self::provenance_charge(remaining, work, &root.path)?;
        if growth != 0 {
            host.reserve_vector(memo, entries, &root.path)?;
            if memo.capacity() > MAX_REGION_NODES + 1
                || memo
                    .capacity()
                    .checked_mul(std::mem::size_of::<Option<bool>>())
                    .and_then(|bytes| u64::try_from(bytes).ok())
                    .is_none_or(|bytes| bytes > max_buffer_bytes)
            {
                return Err(resource(
                    &root.path,
                    "original F1 provenance allocated capacity exceeds admission",
                ));
            }
            memo.resize(entries, None);
        }
        Self::polygonal_source_uncached(root, remaining, memo)
    }

    /// Certify authored rectangles/straight triangles and their immutable
    /// unions using invocation-wide, source-derived proof receipts. Physical
    /// convexity never proves an authored curve, stroke, glyph or clip.
    pub(super) fn proven_original_straight_polygon_union(
        &mut self,
        root: &Arc<NeutralRegion<'_>>,
    ) -> Result<bool, Render2dExecutionError> {
        let mut remaining = MAX_REGION_GRAPH_WORK
            .checked_sub(self.graph_work)
            .ok_or_else(|| resource(&root.path, "original F1 graph work exceeds admission"))?;
        let proof = Self::prepare_polygonal_proof(
            root,
            &mut remaining,
            &mut self.polygonal_proofs,
            self.max_buffer_bytes,
            &mut self.host,
        );
        // Failed/false queries keep every actually admitted charge; errors
        // cannot fall through to a free second proof or fresh allowance.
        self.graph_work = MAX_REGION_GRAPH_WORK - remaining;
        proof
    }

    /// Recognize only the exact existing F1 DAG: C union (C ⊕ disk(r)).
    /// The completed child source must be a whole, certified axis-aligned rectangle,
    /// with identity frames, no clip, zero blur and no effect translation.
    /// All other curved DAGs remain unsupported rather than faceted.
    pub(super) fn qualified_rounded_rectangle<'a>(
        &mut self,
        root: &Arc<NeutralRegion<'a>>,
    ) -> Result<Option<CertifiedRoundedRectangle>, Render2dExecutionError> {
        fn identity<'a, 'b>(
            mut node: &'b Arc<NeutralRegion<'a>>,
            remaining: &mut usize,
        ) -> Result<&'b Arc<NeutralRegion<'a>>, Render2dExecutionError> {
            loop {
                NeutralRegionBuilder::provenance_charge(
                    remaining,
                    PROVENANCE_NODE_WORK,
                    &node.path,
                )?;
                match &node.kind {
                    Kind::Affine { source, to_parent }
                        if to_parent.components()
                            == Render2dAffineTransform::IDENTITY.components() =>
                    {
                        node = source;
                    }
                    Kind::Clip { source, clips: [] } => node = source,
                    Kind::Union(children) if children.len() == 1 => node = &children[0],
                    _ => return Ok(node),
                }
            }
        }
        let mut remaining = MAX_REGION_GRAPH_WORK
            .checked_sub(self.graph_work)
            .ok_or_else(|| resource(&root.path, "original F1 graph work exceeds admission"))?;
        let unwrapped = identity(root, &mut remaining);
        self.graph_work = MAX_REGION_GRAPH_WORK - remaining;
        let Kind::Union(children) = &unwrapped?.kind else {
            return Ok(None);
        };
        if children.len() != 2 {
            return Ok(None);
        }
        let mut remaining = MAX_REGION_GRAPH_WORK - self.graph_work;
        let curved = identity(&children[1], &mut remaining);
        self.graph_work = MAX_REGION_GRAPH_WORK - remaining;
        let (original, curved) = (&children[0], curved?);
        let Kind::Effect { source, shadow } = &curved.kind else {
            return Ok(None);
        };
        if !Arc::ptr_eq(original, source)
            || shadow.spread() <= 0.0
            || shadow.sigma() != 0.0
            || shadow.offset_x() != 0.0
            || shadow.offset_y() != 0.0
        {
            return Ok(None);
        }
        if !self.proven_original_straight_polygon_union(original)? {
            return Ok(None);
        }
        let rectangle = self.materialize(original)?;
        let mut remaining = MAX_REGION_GRAPH_WORK
            .checked_sub(self.graph_work)
            .ok_or_else(|| resource(&original.path, "original F1 graph work exceeds admission"))?;
        let scan = Self::rectangle_scan_work(&rectangle, &original.path)?;
        let charge = Self::provenance_charge(&mut remaining, scan, &original.path);
        self.graph_work = MAX_REGION_GRAPH_WORK - remaining;
        charge?;
        if support::certified_axis_aligned_rectangle(&rectangle.triangles).is_none() {
            return Ok(None);
        }
        self.charge_graph_work(curved.path.len() * 2 + 64, &curved.path)?;
        let mut curved_owner = Vec::new();
        self.host
            .reserve_vector(&mut curved_owner, curved.path.len(), &curved.path)?;
        curved_owner.extend_from_slice(&curved.path);
        Ok(Some(CertifiedRoundedRectangle {
            rectangle,
            radius: shadow.spread(),
            curved_owner,
        }))
    }

    /// Cached polygons share the one original C. IDs remain unique even when
    /// a completed graph subtree is discarded before its siblings are built.
    /// Every new cached materialization is charged once, including copies.
    pub(super) fn materialize<'a>(
        &mut self,
        region: &Arc<NeutralRegion<'a>>,
    ) -> Result<Arc<support::NeutralMesh>, Render2dExecutionError> {
        let path = region.path.as_slice();
        self.charge_graph_work(MATERIALIZATION_LOOKUP_WORK, path)?;
        if let Some(Some(existing)) = self.materialized.get(region.id) {
            return Ok(Arc::clone(existing));
        }
        // Reserve the bounded receipt before recursively materializing any
        // geometry. A failed parent retains completed child receipts and all
        // paid work; its own slot stays None until successful completion.
        let needed = region
            .id
            .checked_add(1)
            .ok_or_else(|| resource(path, "neutral materialization cache size overflow"))?;
        let capacity_limit = usize::try_from(
            (self.max_buffer_bytes
                / std::mem::size_of::<Option<Arc<support::NeutralMesh>>>() as u64)
                .min((MAX_REGION_NODES + 1) as u64),
        )
        .map_err(|_| resource(path, "neutral materialization capacity limit overflow"))?;
        if needed > capacity_limit {
            return Err(resource(
                path,
                "neutral materialization cache exceeds admission",
            ));
        }
        let old_len = self.materialized.len();
        let grows = needed > self.materialized.capacity();
        let relocation = if grows { old_len } else { 0 };
        let preparation = needed
            .saturating_sub(old_len)
            .checked_add(relocation)
            .and_then(|slots| slots.checked_mul(2))
            .and_then(|cost| cost.checked_add(MATERIALIZATION_NODE_WORK))
            .ok_or_else(|| resource(path, "neutral materialization receipt work overflow"))?;
        self.charge_graph_work(preparation, path)?;
        if needed > old_len {
            let desired = if grows {
                self.materialized
                    .capacity()
                    .saturating_mul(2)
                    .min(capacity_limit)
                    .max(needed)
            } else {
                self.materialized.capacity()
            };
            self.host
                .reserve_vector(&mut self.materialized, desired, path)?;
            self.materialized.resize_with(needed, || None);
        }
        let new_mesh = !matches!(&region.kind, Kind::Source(_));
        let mesh_header = size_of::<support::NeutralMesh>() + 2 * size_of::<usize>();
        if new_mesh {
            self.host.reserve(mesh_header, path)?;
        }
        // Failed attempts retain completed child receipts and paid CPU work,
        // but must release this unpublished parent's mesh-header lease.
        let result = (|| -> Result<Arc<support::NeutralMesh>, Render2dExecutionError> {
            let mut output_admitted = false;
            let mesh = match &region.kind {
                Kind::Source(mesh) => Arc::clone(mesh.get().ok_or_else(|| unsupported(path))?),
                Kind::Union(children) => {
                    let mut output = shadow::empty_mesh();
                    let assembled = (|| -> Result<(), Render2dExecutionError> {
                        for child in children {
                            let mesh = self.materialize(child)?;
                            shadow::charge_vertices(
                                &mut self.retained_vertices,
                                mesh.triangles.len(),
                                path,
                            )?;
                            self.charge_graph_work(
                                shadow::append_work(&output, &mesh, path)?,
                                path,
                            )?;
                            shadow::append_admitted(
                                &mut output,
                                &mesh,
                                Render2dAffineTransform::IDENTITY,
                                path,
                                &mut self.host,
                            )?;
                        }
                        Ok(())
                    })();
                    if let Err(error) = assembled {
                        // Incomplete union geometry is dropped; refund only its
                        // own result and header, not child receipts or paid work.
                        self.host
                            .release(output.triangles.capacity() * size_of::<[f64; 2]>());
                        return Err(error);
                    }
                    output_admitted = true;
                    Arc::new(output)
                }
                Kind::Affine { source, to_parent } => {
                    let mesh = self.materialize(source)?;
                    shadow::charge_vertices(
                        &mut self.retained_vertices,
                        mesh.triangles.len(),
                        path,
                    )?;
                    let mut output = shadow::empty_mesh();
                    self.charge_graph_work(shadow::append_work(&output, &mesh, path)?, path)?;
                    shadow::append_admitted(&mut output, &mesh, *to_parent, path, &mut self.host)?;
                    output_admitted = true;
                    Arc::new(output)
                }
                Kind::Clip { source, clips } => {
                    let mesh = self.materialize(source)?;
                    let clipped = shadow::intersect_clips(
                        &mesh,
                        clips,
                        path[0],
                        path,
                        self.scales.geometry_sample_scale,
                        self.max_buffer_bytes,
                        &mut shadow::NeutralPreparationWork {
                            graph: &mut self.graph_work,
                            clip_pairs: &mut self.clip_work,
                            host: &mut self.host,
                        },
                    )?;
                    shadow::charge_vertices(
                        &mut self.retained_vertices,
                        clipped.triangles.len(),
                        path,
                    )?;
                    Arc::new(clipped)
                }
                Kind::Effect {
                    source,
                    shadow: effect,
                } => {
                    // Prove truly complete erosion from original F1 *outer reach*
                    // BEFORE touching child polygons (which may be a curved
                    // symbolic effect). Blur of the empty eroded set remains
                    // empty; a near-threshold radius is never guessed empty.
                    // This is independent of disposable Lyon mesh fidelity.
                    // A leaf without original reach admission cannot infer
                    // emptiness OR signed erosion from sampled physical bounds.
                    if effect.spread() < 0.0 && !source.certified_source_reach {
                        return Err(unsupported(path));
                    }
                    if certified_complete_disk_erosion(source.envelope, effect.spread()) {
                        Arc::new(shadow::empty_mesh())
                    } else {
                        // Physical convexity cannot certify F1 curved geometry.
                        // Proven complete erosion above is independently exact,
                        // but nonempty erosion must prove F1/mesh correspondence.
                        if effect.spread() < 0.0
                            && !self.proven_original_straight_polygon_union(source)?
                        {
                            return Err(unsupported(path));
                        }
                        let mesh = self.materialize(source)?;
                        // A positive disk is retained as a precise expression but
                        // cannot be *polygonalized* as finite triangles. A consuming
                        // ancestor needs certified curved-region evaluation.
                        if effect.spread() > 0.0 {
                            return Err(unsupported(path));
                        }
                        let erased = if effect.spread() < 0.0 {
                            // Erosion retains its evaluator-wide local ceiling,
                            // but every executed preparation phase also spends
                            // this invocation's graph fuel. Keep one live borrow
                            // through qualification AND fallback: failed attempts
                            // cannot refund work or start a fresh allowance.
                            let mut erosion_work = support::SharedPreparationWork::new(
                                &mut self.erosion_work,
                                &mut self.graph_work,
                            );
                            // Source-proven, strictly disconnected rectangles
                            // admit exact per-component disk erosion without
                            // spending the general O(triangles^2) arrangement.
                            let separated = support::certified_separated_rectangles_erosion_mesh(
                                &mesh,
                                -effect.spread(),
                                path,
                                &mut erosion_work,
                            )?;
                            if let Some(separated) = separated {
                                separated
                            } else {
                                // Touching, overlapping, rotated or otherwise
                                // unqualified sources retain the existing exact
                                // convex proof and its typed fallback.
                                support::certified_convex_erosion_mesh(
                                    &mesh,
                                    -effect.spread(),
                                    path,
                                    &mut erosion_work,
                                )?
                                .ok_or_else(|| unsupported(path))?
                            }
                        } else {
                            shadow::empty_mesh()
                        };
                        let caster: &support::NeutralMesh = if effect.spread() < 0.0 {
                            &erased
                        } else {
                            mesh.as_ref()
                        };
                        let offset = Render2dAffineTransform::translation(
                            effect.offset_x(),
                            effect.offset_y(),
                        )
                        .map_err(|_| {
                            precision(path, "neutral effect offset cannot be represented")
                        })?;
                        shadow::charge_vertices(
                            &mut self.retained_vertices,
                            caster.triangles.len(),
                            path,
                        )?;
                        let mut shifted = shadow::empty_mesh();
                        self.charge_graph_work(shadow::append_work(&shifted, caster, path)?, path)?;
                        shadow::append(&mut shifted, caster, offset, path)?;
                        if effect.sigma() > 0.0 && !shifted.triangles.is_empty() {
                            // Twelve translated vertices; at most 12² comparison/
                            // movement steps in the small sort; deduplication;
                            // at most 48 orientation steps across both monotone
                            // hull passes; at most thirty emitted vertices. The
                            // 8192 bound includes all primitive guards and copies.
                            let hull_work = (shifted.triangles.len() / 3)
                                .checked_mul(8_192)
                                .and_then(|cost| cost.checked_add(256))
                                .ok_or_else(|| {
                                    resource(path, "neutral Gaussian hull work overflow")
                                })?;
                            self.charge_graph_work(hull_work, path)?;
                            let square = shadow::gaussian_square_support(
                                &shifted,
                                effect.sigma() * 3.0,
                                path,
                                &mut self.host,
                            )?;
                            output_admitted = true;
                            shadow::charge_vertices(
                                &mut self.retained_vertices,
                                square.triangles.len(),
                                path,
                            )?;
                            Arc::new(square)
                        } else {
                            Arc::new(shifted)
                        }
                    }
                }
            };
            if new_mesh && !output_admitted {
                // Other materializers still transfer their returned capacities
                // here. Affine and Gaussian results were reserved in their factories.
                self.host.reserve(
                    mesh.triangles
                        .capacity()
                        .checked_mul(size_of::<[f64; 2]>())
                        .ok_or_else(|| resource(path, "materialized mesh storage overflow"))?,
                    path,
                )?;
            }
            if mesh.triangles.is_empty()
                && (matches!(&region.kind, Kind::Clip { .. }) || region.semantic_item.is_some())
            {
                let mut fuel = MAX_REGION_GRAPH_WORK
                    .checked_sub(self.graph_work)
                    .ok_or_else(|| resource(path, "original absence proof work overflow"))?;
                let empty = region.original_empty_certificate(&mut fuel);
                self.graph_work = MAX_REGION_GRAPH_WORK - fuel;
                if fuel == 0 {
                    return Err(resource(path, "original absence proof exhausted work"));
                }
                if !empty {
                    return Err(unsupported(path));
                }
            }
            self.materialized[region.id] = Some(Arc::clone(&mesh));
            Ok(mesh)
        })();
        if result.is_err() && new_mesh {
            self.host.release(mesh_header);
        }
        result
    }
}

/// A proof-only point classifier for the *borrowed F1* graph.
/// Unknown is mandatory for unproven geometry or uncertain boundaries.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PointCertificate {
    Inside,
    Outside,
    Unknown,
}

#[cfg(test)]
impl NeutralRegion<'_> {
    fn rectangle(&self) -> Option<[f64; 4]> {
        match &self.kind {
            Kind::Source(mesh) => {
                let mesh = mesh.get()?;
                let rect = support::certified_axis_aligned_rectangle(&mesh.triangles)?;
                let scale = mesh.units_per_parent_logical_unit;
                (scale.is_finite() && scale > 0.0).then(|| rect.map(|v| v / scale))
            }
            Kind::Affine { source, to_parent }
                if to_parent.components() == Render2dAffineTransform::IDENTITY.components() =>
            {
                source.rectangle()
            }
            Kind::Clip { source, clips: [] } => source.rectangle(),
            Kind::Union(children) if children.len() == 1 => children[0].rectangle(),
            _ => None,
        }
    }

    fn point_certificate(&self, point: [f64; 2]) -> PointCertificate {
        if !point.iter().all(|v| v.is_finite()) {
            return PointCertificate::Unknown;
        }
        match &self.kind {
            Kind::Source(_) => {
                let Some([x0, y0, x1, y1]) = self.rectangle() else {
                    return PointCertificate::Unknown;
                };
                if point[0] > x0 && point[0] < x1 && point[1] > y0 && point[1] < y1 {
                    PointCertificate::Inside
                } else if point[0] < x0 || point[0] > x1 || point[1] < y0 || point[1] > y1 {
                    PointCertificate::Outside
                } else {
                    PointCertificate::Unknown
                }
            }
            Kind::Union(children) => {
                let mut unknown = false;
                for child in children {
                    match child.point_certificate(point) {
                        PointCertificate::Inside => return PointCertificate::Inside,
                        PointCertificate::Unknown => unknown = true,
                        PointCertificate::Outside => {}
                    }
                }
                if unknown {
                    PointCertificate::Unknown
                } else {
                    PointCertificate::Outside
                }
            }
            Kind::Affine { source, to_parent }
                if to_parent.components() == Render2dAffineTransform::IDENTITY.components() =>
            {
                source.point_certificate(point)
            }
            Kind::Clip { source, clips: [] } => source.point_certificate(point),
            Kind::Effect { source, shadow } if shadow.spread() > 0.0 && shadow.sigma() == 0.0 => {
                let Some([x0, y0, x1, y1]) = source.rectangle() else {
                    return PointCertificate::Unknown;
                };
                let x = point[0] - shadow.offset_x();
                let y = point[1] - shadow.offset_y();
                let dx = x - x.clamp(x0, x1);
                let dy = y - y.clamp(y0, y1);
                let squared = dx.mul_add(dx, dy * dy);
                let radius = shadow.spread();
                let limit = radius * radius;
                if !squared.is_finite() || !limit.is_finite() {
                    return PointCertificate::Unknown;
                }
                let margin = 128.0 * f64::EPSILON * limit.abs().max(squared.abs()).max(1.0);
                if squared + margin < limit {
                    PointCertificate::Inside
                } else if squared > limit + margin {
                    PointCertificate::Outside
                } else {
                    PointCertificate::Unknown
                }
            }
            // Nonidentity affine, real clips, finite Gaussian envelopes and
            // negative/nonrectangular disks require independent certificates.
            _ => PointCertificate::Unknown,
        }
    }
}

mod certified_cells;

#[cfg(test)]
mod tests {
    use super::support::PreparationWork;
    use super::*;
    use crate::composition_2d::Render2dColorRgba8;

    fn triangle() -> support::NeutralMesh {
        support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]],
            bounds: [0.0, 0.0, 4.0, 4.0],
        }
    }

    fn builder() -> NeutralRegionBuilder {
        NeutralRegionBuilder::new(
            shadow::NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            1_048_576,
        )
    }

    pub(super) fn glyph_source_fixture(
        glyph_id: u32,
        count: usize,
    ) -> (
        crate::composition_2d::Render2dShapedTextResource,
        crate::runtime::execution_2d::field::ResourceFields,
        Render2dItem,
    ) {
        use crate::composition_2d::*;
        use crate::runtime::execution_2d::field;
        let id = Render2dResourceId::new(94).unwrap();
        let resource = Render2dShapedTextResource::new(
            Render2dFontBinding::new(
                include_bytes!("../../../../tests/fixtures/f2_outline.ttf").to_vec(),
                0,
                Vec::new(),
                false,
                None,
            )
            .unwrap(),
            24.0,
            (0..count)
                .map(|_| Render2dGlyph::new(glyph_id, 0.25, -0.5, 24.0).unwrap())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let fields = field::realize(
            id,
            &resource,
            field::QualityTier::P24,
            4096,
            &mut field::FieldBudget::default(),
        )
        .unwrap();
        let item = Render2dItem::new(
            Render2dPrimitive::ShapedText(Render2dShapedTextPrimitive::new(
                id,
                Render2dPoint::new(1.0, 2.0).unwrap(),
                Render2dColorRgba8::TRANSPARENT,
            )),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        (resource, fields, item)
    }

    #[test]
    fn source_metadata_host_rejection_precedes_node_path_and_arc_allocation() {
        let mut graph = builder();
        let remaining = support::MAX_SHADOW_HOST_BYTES - graph.host.live();
        graph.host.reserve(remaining, &[]).unwrap();
        let before = graph.host.live();
        let item = provenance_rectangle_item();
        assert!(
            matches!(graph.register_source(&item, &[7, 1]), Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), .. }) if path == [7, 1])
        );
        assert_eq!(graph.nodes, 0);
        assert_eq!(graph.retained_path_units, 0);
        assert_eq!(graph.constructed_node_bytes, 0);
        assert_eq!(graph.host.live(), before);
        assert!(
            graph.graph_work > 0,
            "attempted source work cannot be refunded with storage"
        );
        graph.host.release(remaining);
        assert!(graph.register_source(&item, &[7, 1]).unwrap().is_some());
    }

    #[test]
    fn original_glyph_metadata_prepays_shared_handles_and_combined_capacity() {
        let (resource, fields, item) = glyph_source_fixture(2, 2);
        let owners = fields.glyph(2).unwrap().outline_owners();
        let mut rejected = builder();
        let fill = support::MAX_SHADOW_HOST_BYTES - rejected.host.live();
        rejected.host.reserve(fill, &[]).unwrap();
        assert!(matches!(
            rejected.register_shaped_source(&item, &resource, &fields, &[6, 2]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
        assert_eq!(
            fields.glyph(2).unwrap().outline_owners(),
            owners,
            "a failed host debit must precede handle retention"
        );
        assert_eq!(rejected.nodes, 0);
        assert_eq!(rejected.constructed_node_bytes, 0);

        let mut measured = builder();
        let base = measured.host.live();
        let source = measured
            .register_shaped_source(&item, &resource, &fields, &[6, 2])
            .unwrap()
            .unwrap();
        let needed = measured.host.peak() - base;
        let Kind::Source(geometry) = &source.kind else {
            panic!("glyph source")
        };
        let glyphs = geometry.glyphs.as_ref().unwrap();
        assert_eq!(glyphs.instances.len(), 2);
        assert_eq!(
            glyphs.outline_bytes,
            2 * fields.glyph(2).unwrap().outline_storage_bytes().unwrap(),
            "repeated shared bodies are conservatively inventoried per occurrence"
        );
        assert_eq!(fields.glyph(2).unwrap().outline_owners(), owners + 2);
        let (retained, construction) = measured.storage_bounds([&source]).unwrap();
        eprintln!(
            "GLYPH_SOURCE_METADATA instance_bytes={} node_bytes={} outline_body_per_occurrence={} source_work={} source_host_growth={} retained_bound={} construction_bound={}",
            size_of::<GlyphSource>(),
            size_of::<NeutralRegion<'_>>(),
            fields.glyph(2).unwrap().outline_storage_bytes().unwrap(),
            measured.graph_work,
            needed,
            retained,
            construction
        );
        assert!(retained >= glyphs.storage_bytes().unwrap());
        assert!(construction >= retained);
        drop(source);
        assert_eq!(fields.glyph(2).unwrap().outline_owners(), owners);

        // A late unadmitted glyph is NOT a validated empty outline. Already
        // shared earlier controls must drop without installing a source node.
        use crate::composition_2d::{Render2dGlyph, Render2dShapedTextResource};
        let missing_resource = Render2dShapedTextResource::new(
            resource.font().clone(),
            resource.font_size(),
            vec![
                Render2dGlyph::new(2, 0.0, 0.0, 24.0).unwrap(),
                Render2dGlyph::new(1, 0.0, 0.0, 24.0).unwrap(),
            ],
        )
        .unwrap();
        let mut missing = builder();
        assert!(matches!(
            missing.register_shaped_source(&item, &missing_resource, &fields, &[6, 3]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                ..
            })
        ));
        assert_eq!(missing.nodes, 0);
        assert_eq!(fields.glyph(2).unwrap().outline_owners(), owners);

        let mut exact = builder();
        exact
            .host
            .reserve(
                support::MAX_SHADOW_HOST_BYTES - exact.host.live() - needed,
                &[],
            )
            .unwrap();
        assert!(
            exact
                .register_shaped_source(&item, &resource, &fields, &[6, 2])
                .unwrap()
                .is_some()
        );
        assert_eq!(exact.host.peak(), support::MAX_SHADOW_HOST_BYTES);
        let mut short = builder();
        short
            .host
            .reserve(
                support::MAX_SHADOW_HOST_BYTES - short.host.live() - needed + 1,
                &[],
            )
            .unwrap();
        assert!(matches!(
            short.register_shaped_source(&item, &resource, &fields, &[6, 2]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
        assert_eq!(short.nodes, 0);
        assert_eq!(fields.glyph(2).unwrap().outline_owners(), owners);
    }

    #[test]
    fn original_glyph_reach_survives_empty_mesh_and_affine_without_inventing_membership() {
        let (resource, fields, item) = glyph_source_fixture(2, 1);
        let mut graph = builder();
        let source = graph
            .register_shaped_source(&item, &resource, &fields, &[3, 0])
            .unwrap()
            .unwrap();
        let Kind::Source(geometry) = &source.kind else {
            panic!("original source")
        };
        assert!(geometry.get().is_none(), "original reach precedes Lyon");
        let before = source.certified_logical_envelope().unwrap();
        // The existing F2 size-one unhinted pen resolves this font's box to
        // [0,-51/64,52/64,0], including its original bearing adjustment and
        // outline coordinate representation. Preserve THAT accepted source,
        // independently of Lyon's bounds and the MSDF field pixels.
        for (actual, expected) in before.iter().zip([1.25, -17.625, 20.75, 1.5]) {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        for [x, y] in [[1.25, -17.625], [20.75, -17.625], [1.25, 1.5], [20.75, 1.5]] {
            assert!(before[0] <= x && before[2] >= x);
            assert!(before[1] <= y && before[3] >= y);
        }
        // An empty disposable reconstruction cannot erase original text.
        graph
            .prepare_registered_source(&source, |_, _, _| Ok(Some(shadow::empty_mesh())))
            .unwrap();
        assert_eq!(source.certified_logical_envelope(), Some(before));
        assert!(
            matches!(graph.materialize(&source), Err(Render2dExecutionError::UnsupportedEntry { path, .. }) if path == [3, 0])
        );
        let affine = Render2dAffineTransform::new(-2.0, 0.5, 0.75, 1.0, 8.0, -2.0).unwrap();
        let mapped = graph.affine(Arc::clone(&source), affine, &[3]).unwrap();
        let bounds = mapped.certified_logical_envelope().unwrap();
        for [x, y] in [
            [before[0], before[1]],
            [before[0], before[3]],
            [before[2], before[1]],
            [before[2], before[3]],
        ] {
            let [a, b, c, d, tx, ty] = affine.components();
            let px = a * x + c * y + tx;
            let py = b * x + d * y + ty;
            assert!(bounds[0] <= px && px <= bounds[2]);
            assert!(bounds[1] <= py && py <= bounds[3]);
        }
        // Strict complete erosion is an absence proof, independent of any
        // physical mesh or font-field alpha. The neighboring nonempty case
        // must retain its unsupported result until original winding is ready.
        let eroded = graph
            .effect(
                Arc::clone(&mapped),
                Render2dDropShadow::new(2.0, -1.0, 0.2, -1000.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[3, 1],
            )
            .unwrap();
        assert!(graph.materialize(&eroded).unwrap().triangles.is_empty());
        let nonempty = graph
            .effect(
                mapped,
                Render2dDropShadow::new(0.0, 0.0, 0.0, -0.1, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[3, 2],
            )
            .unwrap();
        assert!(
            matches!(graph.materialize(&nonempty), Err(Render2dExecutionError::UnsupportedEntry { path, .. }) if path == [3, 2])
        );
    }

    #[test]
    fn original_glyph_reach_pays_empty_repeated_missing_and_exact_work_before_node_commit() {
        let (resource, fields, item) = glyph_source_fixture(1, 3);
        let mut graph = builder();
        assert!(
            graph
                .register_shaped_source(&item, &resource, &fields, &[4, 0])
                .unwrap()
                .is_none()
        );
        assert_eq!(graph.nodes, 0);
        assert_eq!(
            graph.graph_work,
            REGION_NODE_WORK + 512 + 3 * fields.outline_lookup_work()
        );

        let (resource, fields, item) = glyph_source_fixture(2, 2);
        let mut measured = builder();
        measured
            .register_shaped_source(&item, &resource, &fields, &[4, 1])
            .unwrap();
        let cost = measured.graph_work;
        let mut exact = builder();
        exact.graph_work = MAX_REGION_GRAPH_WORK - cost;
        assert!(
            exact
                .register_shaped_source(&item, &resource, &fields, &[4, 1])
                .unwrap()
                .is_some()
        );
        assert_eq!(exact.graph_work, MAX_REGION_GRAPH_WORK);
        let mut short = builder();
        short.graph_work = MAX_REGION_GRAPH_WORK - cost + 1;
        assert!(
            matches!(short.register_shaped_source(&item, &resource, &fields, &[4, 1]), Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), .. }) if path == [4, 1])
        );
        assert_eq!(short.nodes, 0);
        assert_eq!(short.retained_vertices, 0);
        assert_eq!(short.constructed_node_bytes, 0);

        let (_, empty_fields, _) = glyph_source_fixture(1, 1);
        let mut missing = builder();
        assert!(
            matches!(missing.register_shaped_source(&item, &resource, &empty_fields, &[4, 2]), Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::PrecisionLimit, path: Some(path), .. }) if path == [4, 2])
        );
        assert_eq!(missing.nodes, 0, "missing is not an admitted empty glyph");
    }

    #[test]
    fn original_glyph_reach_encloses_placement_cancellation_and_rejects_overflow() {
        use crate::composition_2d::*;
        let (resource, fields, _) = glyph_source_fixture(2, 1);
        let item = Render2dItem::new(
            Render2dPrimitive::ShapedText(Render2dShapedTextPrimitive::new(
                Render2dResourceId::new(94).unwrap(),
                Render2dPoint::new(1.0e16, 0.0).unwrap(),
                Render2dColorRgba8::TRANSPARENT,
            )),
            Render2dAffineTransform::translation(-1.0e16, 0.0).unwrap(),
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = builder();
        let bounds = graph
            .register_shaped_source(&item, &resource, &fields, &[5, 0])
            .unwrap()
            .unwrap()
            .certified_logical_envelope()
            .unwrap();
        assert!(
            bounds[0] <= 0.25 && bounds[2] >= 19.75,
            "rounded combined glyph origin must not become source truth"
        );
        let overflow = Render2dItem::new(
            item.primitive().clone(),
            Render2dAffineTransform::new(f64::MAX, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap(),
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        assert!(
            matches!(graph.register_shaped_source(&overflow, &resource, &fields, &[5, 1]), Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::PrecisionLimit, path: Some(path), .. }) if path == [5, 1])
        );
    }

    #[test]
    fn original_cubic_control_hull_survives_a_coarser_physical_tessellation() {
        use crate::composition_2d::{
            Render2dBrush, Render2dFillRule, Render2dItem, Render2dOpacity, Render2dPath,
            Render2dPathCommand, Render2dPoint, Render2dPrimitive, Render2dShape,
        };
        use crate::runtime::execution_2d::vector;
        let point = |x, y| Render2dPoint::new(x, y).unwrap();
        let path = Render2dPath::new(
            Render2dFillRule::NonZero,
            vec![
                Render2dPathCommand::MoveTo(point(0.0, 0.0)),
                Render2dPathCommand::CubicTo {
                    control1: point(2.0, 7.0),
                    control2: point(8.0, 7.0),
                    to: point(10.0, 0.0),
                },
                Render2dPathCommand::LineTo(point(0.0, 0.0)),
                Render2dPathCommand::Close,
            ],
        )
        .unwrap();
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::path(path),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let physical = vector::neutral_support(&item, 0, 4.0, 1_048_576)
            .unwrap()
            .unwrap();
        let physical_top = physical.bounds[3] / physical.units_per_parent_logical_unit;
        let mut graph = builder();
        let leaf = graph.source_from_item(physical, &item, &[0, 0]).unwrap();
        assert!(
            physical_top < 7.0,
            "a physical curve is below its authored control hull"
        );
        assert!(
            leaf.envelope.unwrap()[3] >= 7.0,
            "original F1 control hull encloses the curve"
        );
        let shear = Render2dAffineTransform::new(1.0, 0.0, 0.5, 1.0, 0.0, 0.0).unwrap();
        let transformed = graph.affine(leaf, shear, &[0]).unwrap();
        assert!(
            transformed.envelope.unwrap()[3] >= 7.0,
            "parent-affine reach must preserve exact authored control hull"
        );
    }

    #[test]
    fn semantic_source_bounds_are_still_not_membership_certificates() {
        use crate::composition_2d::{
            Render2dBrush, Render2dItem, Render2dOpacity, Render2dPrimitive, Render2dRect,
            Render2dShape,
        };
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::ellipse(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let bounds = source_envelope::bounds(&item, None, &[0]).unwrap().unwrap();
        assert!(bounds[0] <= 0.0 && bounds[1] <= 0.0);
        assert!(bounds[2] >= 2.0 && bounds[3] >= 2.0);
    }

    #[test]
    fn original_stroke_caps_miters_and_shear_increase_conservative_reach() {
        use crate::composition_2d::{
            Render2dBrush, Render2dItem, Render2dOpacity, Render2dPrimitive, Render2dRect,
            Render2dShape, Render2dStrokeCap, Render2dStrokeJoin, Render2dStrokeStyle,
        };
        let item = Render2dItem::new(
            Render2dPrimitive::Stroke {
                shape: Render2dShape::rect(Render2dRect::new(10.0, 10.0, 4.0, 4.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
                style: Render2dStrokeStyle::new(
                    2.0,
                    Render2dStrokeCap::Square,
                    Render2dStrokeJoin::Miter,
                    5.0,
                )
                .unwrap(),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let bound = source_envelope::bounds(&item, None, &[0]).unwrap().unwrap();
        assert!(bound[0] <= 5.0 && bound[1] <= 5.0);
        assert!(bound[2] >= 19.0 && bound[3] >= 19.0);
        let shear = Render2dAffineTransform::new(1.0, 0.0, 1.0, 1.0, 0.0, 0.0).unwrap();
        let transformed = source_envelope::bounds(&item, Some(shear), &[0])
            .unwrap()
            .unwrap();
        assert!(transformed[0] <= 10.0 && transformed[2] >= 33.0);
    }

    #[test]
    fn borrowed_f1_group_events_build_exact_transparent_disk_expression() {
        use crate::composition_2d::{
            Render2dBrush, Render2dComposition, Render2dEntry, Render2dGroup, Render2dItem,
            Render2dOpacity, Render2dPrimitive, Render2dRect, Render2dResourceBindings,
            Render2dShape,
        };
        let transparent = Render2dColorRgba8::TRANSPARENT;
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(10.0, 10.0, 10.0, 10.0).unwrap()),
                brush: Render2dBrush::solid(transparent),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let positive = Render2dDropShadow::new(0.0, 0.0, 0.0, 1.0, transparent).unwrap();
        let child = Render2dGroup::new(
            vec![Render2dEntry::item(item)],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            vec![positive],
        );
        let root = Render2dGroup::new(
            vec![Render2dEntry::group(child)],
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
            vec![
                Render2dDropShadow::new(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    Render2dColorRgba8::new(255, 0, 0, 255),
                )
                .unwrap(),
            ],
        );
        let composition = Render2dComposition::new(vec![Render2dEntry::group(root)]).unwrap();
        let plan = shadow::scene::analyze(&composition, &mut ()).unwrap();
        let bindings = Render2dResourceBindings::default();
        // The same original source graph is a complete continuous-query
        // input before any disposable vector geometry has been requested.
        let fields = std::collections::BTreeMap::new();
        let dag = shadow::build_group_sources(
            plan.composition(),
            plan.retained_bytes(),
            &bindings,
            &fields,
            shadow::SourcePreparation::new(
                shadow::NeutralPreparationScale {
                    field_raster_scale: 1.0,
                    geometry_sample_scale: 4.0,
                },
                1_048_576,
            ),
        )
        .unwrap();
        assert_eq!(dag.graph.retained_vertices, 0);
        assert!(dag.graph.materialized.is_empty());
        let raw = dag.source(0).unwrap();
        let mut fuel = 1_000_000;
        assert_eq!(
            certified_cells::certified_area_enclosure(
                &raw,
                [15.0, 15.0, 16.0, 16.0],
                None,
                4,
                &mut fuel,
            ),
            Some([1.0, 1.0]),
        );
        assert!(fuel < 1_000_000);
        assert_eq!(dag.graph.retained_vertices, 0);
        assert!(dag.storage().retained_bytes > 0);
        assert!(dag.storage().construction_peak_bytes >= dag.storage().retained_bytes);
        drop(dag);
        let mut inspected_root = false;
        let result = shadow::group_child_sources_with_probe(
            &plan,
            &bindings,
            &std::collections::BTreeMap::new(),
            shadow::NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            1_048_576,
            None,
            |region, path| {
                if path == [0] {
                    inspected_root = true;
                    assert_eq!(
                        region.point_certificate([9.5, 9.5]),
                        PointCertificate::Inside
                    );
                    assert_eq!(
                        region.point_certificate([8.5, 8.5]),
                        PointCertificate::Outside
                    );
                    assert_eq!(
                        region.point_certificate([15.0, 15.0]),
                        PointCertificate::Inside
                    );
                    assert_eq!(
                        region.point_certificate([9.0, 9.0]),
                        PointCertificate::Outside
                    );
                    assert_eq!(
                        region.point_certificate([9.0, 15.0]),
                        PointCertificate::Unknown
                    );
                }
            },
        );
        assert!(
            inspected_root,
            "the actual root caster must be inspected before materialization"
        );
        let sources = result.expect("qualified C union disk must reach the ancestor");
        let Some(shadow::NeutralCaster::RoundedRectangle(qualified)) = sources.caster(0) else {
            panic!("the root must carry one exact qualified rounded caster");
        };
        assert_eq!(qualified.radius, 1.0);
        assert_eq!(qualified.curved_owner, &[0, 0]);
        let rect = support::certified_axis_aligned_rectangle(&qualified.rectangle.triangles)
            .expect("the child C is still a complete physical rectangle");
        let scale = qualified.rectangle.units_per_parent_logical_unit;
        assert_eq!(rect.map(|v| v / scale), [10.0, 10.0, 20.0, 20.0]);
    }

    #[test]
    fn original_f1_straight_rectangle_path_proof_rejects_crossings_and_curves() {
        use crate::composition_2d::{
            Render2dBrush, Render2dFillRule, Render2dItem, Render2dOpacity, Render2dPath,
            Render2dPathCommand as Command, Render2dPoint, Render2dPrimitive, Render2dShape,
        };
        let point = |x, y| Render2dPoint::new(x, y).unwrap();
        let p0 = point(0.0, 0.0);
        let p1 = point(2.0, 0.0);
        let p2 = point(2.0, 2.0);
        let p3 = point(0.0, 2.0);
        let physical = || support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let make_item = |path: Render2dPath| {
            Render2dItem::new(
                Render2dPrimitive::Fill {
                    shape: Render2dShape::path(path),
                    brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
                },
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
            )
        };
        let mut builder = builder();
        for (i, (fill, commands)) in [
            (
                Render2dFillRule::NonZero,
                vec![
                    Command::MoveTo(p0),
                    Command::LineTo(p1),
                    Command::LineTo(p2),
                    Command::LineTo(p3),
                    Command::Close,
                ],
            ),
            (
                Render2dFillRule::EvenOdd,
                vec![
                    Command::MoveTo(p2),
                    Command::LineTo(p1),
                    Command::LineTo(p0),
                    Command::LineTo(p3),
                ],
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let authored = make_item(Render2dPath::new(fill, commands).unwrap());
            let leaf = builder
                .source_from_item(physical(), &authored, &[i, 0])
                .unwrap();
            assert!(
                builder
                    .proven_original_straight_polygon_union(&leaf)
                    .unwrap()
            );
            let disk =
                Render2dDropShadow::new(0.0, 0.0, 0.0, -0.25, Render2dColorRgba8::TRANSPARENT)
                    .unwrap();
            let erosion = builder.effect(leaf, disk, &[i, 1]).unwrap();
            assert!(!builder.materialize(&erosion).unwrap().triangles.is_empty());
        }
        // A crossed four-corner path has the same bounding rectangle and
        // may be forged with the same physical tessellation, but is NOT
        // the same F1 filled region. Nor is a curved F1 path.
        for commands in [
            vec![
                Command::MoveTo(p0),
                Command::LineTo(p2),
                Command::LineTo(p1),
                Command::LineTo(p3),
                Command::Close,
            ],
            vec![
                Command::MoveTo(p0),
                Command::LineTo(p1),
                Command::QuadraticTo {
                    control: p2,
                    to: p3,
                },
                Command::LineTo(p0),
                Command::Close,
            ],
        ] {
            let authored =
                make_item(Render2dPath::new(Render2dFillRule::EvenOdd, commands).unwrap());
            let leaf = builder
                .source_from_item(physical(), &authored, &[7, 0])
                .unwrap();
            assert!(
                !builder
                    .proven_original_straight_polygon_union(&leaf)
                    .unwrap()
            );
        }
        // An exact rectangular PATH with a forged mismatched 3x2 tessellation
        // also cannot be certified merely because both shapes are rectangles.
        let authored = make_item(
            Render2dPath::new(
                Render2dFillRule::NonZero,
                vec![
                    Command::MoveTo(p0),
                    Command::LineTo(p1),
                    Command::LineTo(p2),
                    Command::LineTo(p3),
                ],
            )
            .unwrap(),
        );
        let mut stale = physical();
        stale.triangles[1][0] = 3.0;
        stale.triangles[2][0] = 3.0;
        stale.triangles[4][0] = 3.0;
        let forged = builder.source_from_item(stale, &authored, &[8, 0]).unwrap();
        assert!(
            !builder
                .proven_original_straight_polygon_union(&forged)
                .unwrap()
        );
    }

    #[test]
    fn original_straight_triangle_proof_matches_only_exact_authored_fill() {
        use crate::composition_2d::{
            Render2dBrush, Render2dFillRule, Render2dItem, Render2dOpacity, Render2dPath,
            Render2dPathCommand as Command, Render2dPoint, Render2dPrimitive, Render2dShape,
        };
        let p = |x, y| Render2dPoint::new(x, y).unwrap();
        let a = p(10.0, 10.0);
        let b = p(26.0, 10.0);
        let c = p(10.0, 26.0);
        let mesh = || support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[10.0, 10.0], [26.0, 10.0], [10.0, 26.0]],
            bounds: [10.0, 10.0, 26.0, 26.0],
        };
        let item = |commands: Vec<Command>| {
            Render2dItem::new(
                Render2dPrimitive::Fill {
                    shape: Render2dShape::path(
                        Render2dPath::new(Render2dFillRule::EvenOdd, commands).unwrap(),
                    ),
                    brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
                },
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
            )
        };
        let mut graph = builder();
        for (index, commands) in [
            vec![Command::MoveTo(a), Command::LineTo(b), Command::LineTo(c)],
            vec![
                Command::MoveTo(c),
                Command::LineTo(b),
                Command::LineTo(a),
                Command::LineTo(c),
                Command::Close,
            ],
        ]
        .into_iter()
        .enumerate()
        {
            let authored = item(commands);
            let leaf = graph.source_from_item(mesh(), &authored, &[index]).unwrap();
            assert!(graph.proven_original_straight_polygon_union(&leaf).unwrap());
            let shadow =
                Render2dDropShadow::new(6.0, 0.0, 0.1, -1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap();
            let eroded = graph.effect(leaf, shadow, &[index]).unwrap();
            assert!(!graph.materialize(&eroded).unwrap().triangles.is_empty());
        }
        // Same physical triangle as a different F1 geometry cannot
        // certify either signed erosion or original source support.
        for (index, commands) in [
            vec![
                Command::MoveTo(a),
                Command::LineTo(b),
                Command::QuadraticTo { control: c, to: a },
            ],
            vec![
                Command::MoveTo(a),
                Command::LineTo(b),
                Command::LineTo(c),
                Command::MoveTo(p(30.0, 30.0)),
                Command::LineTo(p(32.0, 30.0)),
                Command::LineTo(p(30.0, 32.0)),
            ],
            vec![
                Command::MoveTo(a),
                Command::LineTo(p(28.0, 10.0)),
                Command::LineTo(c),
            ],
        ]
        .into_iter()
        .enumerate()
        {
            let authored = item(commands);
            let leaf = graph
                .source_from_item(mesh(), &authored, &[index + 10])
                .unwrap();
            assert!(!graph.proven_original_straight_polygon_union(&leaf).unwrap());
        }
        let valid = item(vec![
            Command::MoveTo(a),
            Command::LineTo(b),
            Command::LineTo(c),
        ]);
        let mut forged = mesh();
        forged.triangles[1] = [28.0, 10.0];
        let leaf = graph.source_from_item(forged, &valid, &[20]).unwrap();
        assert!(!graph.proven_original_straight_polygon_union(&leaf).unwrap());
    }

    #[test]
    fn rounded_rectangle_specialization_requires_original_f1_source_proof() {
        use crate::composition_2d::{
            Render2dBrush, Render2dItem, Render2dOpacity, Render2dPrimitive, Render2dRect,
            Render2dShape,
        };
        let rect = Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap();
        // Deliberately forge identical disposable tessellations for two
        // DIFFERENT original F1 shapes. The mesh is not a semantic witness.
        let physical = || support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let valid = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(rect),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let curved = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::ellipse(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let disk =
            Render2dDropShadow::new(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let mut graph = builder();
        let plain_source = graph.source_from_item(physical(), &valid, &[0, 0]).unwrap();
        let plain_disk = graph
            .effect(Arc::clone(&plain_source), disk, &[0, 1])
            .unwrap();
        let plain = graph.union(vec![plain_source, plain_disk], &[0]).unwrap();
        assert!(graph.qualified_rounded_rectangle(&plain).unwrap().is_some());

        let ellipse_source = graph
            .source_from_item(physical(), &curved, &[1, 0])
            .unwrap();
        let ellipse_disk = graph
            .effect(Arc::clone(&ellipse_source), disk, &[1, 1])
            .unwrap();
        let ellipse = graph
            .union(vec![ellipse_source, ellipse_disk], &[1])
            .unwrap();
        assert!(
            graph
                .qualified_rounded_rectangle(&ellipse)
                .unwrap()
                .is_none(),
            "an F1 ellipse may not be promoted from a rectangular Lyon mesh"
        );
        assert!(matches!(
            graph.materialize(&ellipse),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 1 },
            }) if path == [1, 1]
        ));

        // Even the exact same F1 primitive *type* must not authorize an
        // unrelated physical rectangle with different source extent.
        let forged = support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [0.0, 0.0],
                [3.0, 0.0],
                [3.0, 2.0],
                [0.0, 0.0],
                [3.0, 2.0],
                [0.0, 2.0],
            ],
            bounds: [0.0, 0.0, 3.0, 2.0],
        };
        let forged_source = graph.source_from_item(forged, &valid, &[2, 0]).unwrap();
        let forged_disk = graph
            .effect(Arc::clone(&forged_source), disk, &[2, 1])
            .unwrap();
        let forged_union = graph.union(vec![forged_source, forged_disk], &[2]).unwrap();
        assert!(
            graph
                .qualified_rounded_rectangle(&forged_union)
                .unwrap()
                .is_none(),
            "a 2x2 authored F1 rect cannot acquire a 3x2 semantic support"
        );
        assert!(matches!(
            graph.materialize(&forged_union),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 2 },
            }) if path == [2, 1]
        ));

        // A group clip carries its OWN affine. A raw rectangular clip type
        // cannot certify correspondence of the authored clipped source to a
        // disposable triangle intersection, even when the clip is a no-op
        // for this particular fixture.
        let group_clip = crate::composition_2d::Render2dClip::new(
            Render2dShape::rect(rect),
            Render2dAffineTransform::IDENTITY,
        );
        let clip_source = graph.source_from_item(physical(), &valid, &[3, 0]).unwrap();
        let clip_source = graph
            .clip(clip_source, std::slice::from_ref(&group_clip), &[3, 0])
            .unwrap();
        let clip_disk = graph
            .effect(Arc::clone(&clip_source), disk, &[3, 1])
            .unwrap();
        let clipped = graph.union(vec![clip_source, clip_disk], &[3]).unwrap();
        assert!(
            graph
                .qualified_rounded_rectangle(&clipped)
                .unwrap()
                .is_none(),
            "the clip's authored affine/intersection needs its own exact F1 proof"
        );
        assert!(matches!(
            graph.materialize(&clipped),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 3 },
            }) if path == [3, 1]
        ));
    }

    #[test]
    fn neutral_union_is_idempotent_without_reordering_authored_first_occurrences() {
        let mut graph = builder();
        let first = graph.source(triangle(), &[4, 0]).unwrap();
        let second = graph.source(triangle(), &[4, 1]).unwrap();
        let combined = graph
            .union(
                vec![
                    Arc::clone(&first),
                    Arc::clone(&second),
                    Arc::clone(&first),
                    Arc::clone(&second),
                ],
                &[4],
            )
            .unwrap();
        let Kind::Union(children) = &combined.kind else {
            panic!("two distinct authored sources must remain one union");
        };
        assert_eq!(children.len(), 2);
        assert!(Arc::ptr_eq(&children[0], &first));
        assert!(Arc::ptr_eq(&children[1], &second));
        let mesh = graph.materialize(&combined).unwrap();
        assert_eq!(mesh.triangles.len(), triangle().triangles.len() * 2);

        let mut repeated = Arc::clone(&first);
        let nodes = graph.nodes;
        let before_repeated = graph.graph_work;
        for _ in 0..12 {
            repeated = graph
                .union(vec![Arc::clone(&repeated), Arc::clone(&repeated)], &[4, 2])
                .unwrap();
        }
        assert!(Arc::ptr_eq(&repeated, &first));
        assert_eq!(
            graph.nodes, nodes,
            "union idempotence must not mint fake nodes"
        );
        // Idempotence avoids nodes, not the work of proving identity.
        assert_eq!(
            graph.graph_work - before_repeated,
            12 * (UNION_FIXED_WORK + 2 * UNION_CHILD_WORK)
        );

        // An admitted stamp scan is not an authored node commitment.
        // Trigger a late owner-path rejection, restore that test-only
        // counter and ensure the very next identical union remains intact.
        let committed = graph.graph_work;
        let paths = graph.retained_path_units;
        graph.retained_path_units = MAX_REGION_PATH_UNITS;
        assert!(matches!(
            graph.union(
                vec![Arc::clone(&first), Arc::clone(&second)],
                &[4, 3],
            ),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [4, 3]
        ));
        assert_eq!(
            graph.graph_work,
            committed + UNION_FIXED_WORK + 2 * UNION_CHILD_WORK
        );
        graph.retained_path_units = paths;
        let retried = graph
            .union(vec![Arc::clone(&first), Arc::clone(&second)], &[4, 3])
            .unwrap();
        let Kind::Union(children) = &retried.kind else {
            panic!("late rejection cannot poison the next union epoch");
        };
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn union_stamp_initialization_rejects_before_unfunded_allocation() {
        let mut graph = builder();
        let mut last = graph.source(shadow::empty_mesh(), &[7, 0]).unwrap();
        for _ in 1..4_095 {
            last = graph.source(shadow::empty_mesh(), &[7, 0]).unwrap();
        }
        let next = graph.source(shadow::empty_mesh(), &[7, 1]).unwrap();
        let scan = UNION_FIXED_WORK + 2 * UNION_CHILD_WORK;
        graph.graph_work = MAX_REGION_GRAPH_WORK - scan - 2;
        let nodes = graph.nodes;
        assert!(matches!(graph.union(vec![last, next], &[7, 3]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [7, 3]));
        assert_eq!(graph.graph_work, MAX_REGION_GRAPH_WORK - 2);
        assert_eq!(graph.union_seen.len(), 0);
        assert_eq!(graph.union_seen.capacity(), 0);
        assert_eq!(graph.union_epoch, 0);
        assert_eq!(graph.nodes, nodes);
    }

    #[test]
    fn union_stamp_growth_pays_relocation_before_reservation() {
        let mut graph = builder();
        graph.union_seen = vec![0; 4_096];
        let capacity = graph.union_seen.capacity();
        graph.nodes = capacity - 1;
        let left = graph.source(shadow::empty_mesh(), &[8, 0]).unwrap();
        let right = graph.source(shadow::empty_mesh(), &[8, 1]).unwrap();
        let scan = UNION_FIXED_WORK + 2 * UNION_CHILD_WORK;
        // Initializing two new slots would fit; relocating all old slots
        // would not. Reject before reserve can move the retained table.
        graph.graph_work = MAX_REGION_GRAPH_WORK - scan - capacity;
        assert!(matches!(graph.union(vec![left, right], &[8, 3]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [8, 3]));
        assert_eq!(graph.graph_work, MAX_REGION_GRAPH_WORK - capacity);
        assert_eq!(graph.union_seen.len(), 4_096);
        assert_eq!(graph.union_seen.capacity(), capacity);
        assert_eq!(graph.union_epoch, 0);
        assert!(graph.union_seen.iter().all(|mark| *mark == 0));
    }

    #[test]
    fn source_vertex_scan_rejects_before_unfunded_traversal() {
        let mut graph = builder();
        graph.graph_work = MAX_REGION_GRAPH_WORK - 64;
        let before = graph.graph_work;
        assert!(matches!(graph.source(triangle(), &[5, 2]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [5, 2]));
        assert_eq!(graph.graph_work, before);
        assert_eq!(graph.nodes, 0);
        assert_eq!(graph.retained_vertices, 0);
    }

    #[test]
    fn materialization_receipt_requires_fuel_before_cache_allocation() {
        let mut graph = builder();
        graph.nodes = 4_095;
        let source = graph.source(triangle(), &[6, 2]).unwrap();
        graph.graph_work = MAX_REGION_GRAPH_WORK - MATERIALIZATION_LOOKUP_WORK;
        assert!(matches!(graph.materialize(&source),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [6, 2]));
        assert_eq!(graph.graph_work, MAX_REGION_GRAPH_WORK);
        assert_eq!(graph.materialized.len(), 0);
        assert_eq!(graph.materialized.capacity(), 0);
    }

    #[test]
    fn materialization_growth_pays_relocation_and_cached_queries_pay_lookup() {
        let mut graph = builder();
        graph.materialized = vec![None; 4_096];
        let capacity = graph.materialized.capacity();
        graph.nodes = capacity - 1;
        let source = graph.source(triangle(), &[9, 2]).unwrap();
        graph.graph_work = MAX_REGION_GRAPH_WORK - capacity;
        assert!(matches!(graph.materialize(&source),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [9, 2]));
        assert_eq!(graph.materialized.len(), 4_096);
        assert_eq!(graph.materialized.capacity(), capacity);
        assert_eq!(
            graph.graph_work,
            MAX_REGION_GRAPH_WORK - capacity + MATERIALIZATION_LOOKUP_WORK
        );

        graph.graph_work = 0;
        let first = graph.materialize(&source).unwrap();
        let paid = graph.graph_work;
        assert!(Arc::ptr_eq(&first, &graph.materialize(&source).unwrap()));
        assert_eq!(graph.graph_work, paid + MATERIALIZATION_LOOKUP_WORK);
        graph.graph_work = MAX_REGION_GRAPH_WORK;
        assert!(graph.materialize(&source).is_err());
        assert!(graph.materialized[source.id].is_some());
    }

    #[test]
    fn identity_affine_preserves_source_owner_and_reuses_geometry_without_a_node() {
        let mut graph = builder();
        let source = graph.source(triangle(), &[5, 1]).unwrap();
        let mesh = graph.materialize(&source).unwrap();
        let nodes = graph.nodes;
        let paths = graph.retained_path_units;
        let before = graph.graph_work;
        let projected = graph
            .affine(Arc::clone(&source), Render2dAffineTransform::IDENTITY, &[5])
            .unwrap();
        assert!(Arc::ptr_eq(&source, &projected));
        assert_eq!(projected.path, [5, 1]);
        assert_eq!(graph.nodes, nodes);
        assert_eq!(graph.retained_path_units, paths);
        assert_eq!(graph.graph_work - before, 64);
        assert!(Arc::ptr_eq(&mesh, &graph.materialize(&projected).unwrap()));
    }

    #[test]
    fn union_materialization_reserves_growth_and_transfers_once() {
        let mut graph = builder();
        graph
            .host
            .reserve_vector(&mut graph.materialized, 8, &[8])
            .unwrap();
        let left = graph.source(triangle(), &[8, 0]).unwrap();
        let right = graph.source(triangle(), &[8, 1]).unwrap();
        graph.materialize(&left).unwrap();
        graph.materialize(&right).unwrap();
        let union = graph.union(vec![left, right], &[8]).unwrap();
        let before = graph.host.live();
        let result = graph.materialize(&union).unwrap();
        assert_eq!(result.triangles.len(), 6);
        let header = size_of::<support::NeutralMesh>() + 2 * size_of::<usize>();
        let bytes = header + result.triangles.capacity() * size_of::<[f64; 2]>();
        assert_eq!(graph.host.live() - before, bytes);
        // The old 3-vertex vector and the new 6-vertex vector must both
        // fit at relocation, not merely the final retained capacity.
        assert!(graph.host.peak() >= graph.host.live() + 3 * size_of::<[f64; 2]>());
        assert!(Arc::ptr_eq(&result, &graph.materialize(&union).unwrap()));
        assert_eq!(graph.host.live() - before, bytes);
    }

    #[test]
    fn union_destination_growth_is_amortized_and_peak_admitted() {
        let mut graph = builder();
        graph
            .host
            .reserve_vector(&mut graph.materialized, 16, &[7])
            .unwrap();
        let mut children = Vec::new();
        for index in 0..5 {
            let child = graph.source(triangle(), &[7, index]).unwrap();
            graph.materialize(&child).unwrap();
            children.push(child);
        }
        let union = graph.union(children, &[7]).unwrap();
        let before = graph.host.live();
        let result = graph.materialize(&union).unwrap();
        // 3, 6, 12, 12, 24 capacity, rather than five exact relocations.
        assert_eq!(result.triangles.len(), 15);
        assert_eq!(result.triangles.capacity(), 24);
        let header = size_of::<support::NeutralMesh>() + 2 * size_of::<usize>();
        assert_eq!(
            graph.host.live() - before,
            header + 24 * size_of::<[f64; 2]>()
        );
        assert!(graph.host.peak() >= graph.host.live() + 12 * size_of::<[f64; 2]>());
    }

    #[test]
    fn union_growth_one_short_refunds_partial_mesh_and_header() {
        let mut graph = builder();
        graph
            .host
            .reserve_vector(&mut graph.materialized, 8, &[9])
            .unwrap();
        let left = graph.source(triangle(), &[9, 0]).unwrap();
        let right = graph.source(triangle(), &[9, 1]).unwrap();
        graph.materialize(&left).unwrap();
        graph.materialize(&right).unwrap();
        let union = graph.union(vec![left, right], &[9]).unwrap();
        let header = size_of::<support::NeutralMesh>() + 2 * size_of::<usize>();
        // After the first triangle, relocation temporarily owns BOTH the
        // 3-vertex old buffer and the 6-vertex replacement allocation.
        let peak_added = header + 9 * size_of::<[f64; 2]>();
        graph
            .host
            .reserve(
                support::MAX_SHADOW_HOST_BYTES - graph.host.live() - peak_added + 1,
                &[9],
            )
            .unwrap();
        let before = graph.host.live();
        assert!(matches!(
            graph.materialize(&union),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [9]
        ));
        assert_eq!(graph.host.live(), before);
        assert!(graph.materialized[union.id].is_none());
    }

    #[test]
    fn rejected_positive_disk_materialization_refunds_unpublished_header() {
        let mut graph = builder();
        graph
            .host
            .reserve_vector(&mut graph.materialized, 8, &[2])
            .unwrap();
        let source = graph.source(triangle(), &[2, 0]).unwrap();
        graph.materialize(&source).unwrap();
        let effect = graph
            .effect(
                source,
                Render2dDropShadow::new(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[2, 1],
            )
            .unwrap();
        let before = graph.host.live();
        for _ in 0..2 {
            assert!(matches!(
                graph.materialize(&effect),
                Err(Render2dExecutionError::UnsupportedEntry {
                    path,
                    kind: Render2dUnsupportedContent::Shadows { root_index: 2 },
                }) if path == [2, 1]
            ));
            assert_eq!(graph.host.live(), before);
            assert!(graph.materialized[effect.id].is_none());
        }
    }

    #[test]
    fn affine_materialization_transfers_its_vector_lease_exactly_once() {
        let mut graph = builder();
        // Prevent unrelated receipt-vector relocation from obscuring the
        // affine output's actual retained allocation charge.
        graph
            .host
            .reserve_vector(&mut graph.materialized, 8, &[4, 1])
            .unwrap();
        let source = graph.source(triangle(), &[4, 0]).unwrap();
        graph.materialize(&source).unwrap();
        let affine = graph
            .affine(
                Arc::clone(&source),
                Render2dAffineTransform::translation(2.0, 0.0).unwrap(),
                &[4, 1],
            )
            .unwrap();
        let before = graph.host.live();
        let output = graph.materialize(&affine).unwrap();
        let bytes = size_of::<support::NeutralMesh>()
            + 2 * size_of::<usize>()
            + output.triangles.capacity() * size_of::<[f64; 2]>();
        assert_eq!(output.triangles.len(), 3);
        assert_eq!(graph.host.live() - before, bytes);
        assert!(Arc::ptr_eq(&output, &graph.materialize(&affine).unwrap()));
        assert_eq!(graph.host.live() - before, bytes);
    }

    #[test]
    fn affine_destination_one_short_does_not_commit_or_leak_header() {
        let mut graph = builder();
        graph
            .host
            .reserve_vector(&mut graph.materialized, 8, &[6, 1])
            .unwrap();
        let source = graph.source(triangle(), &[6, 0]).unwrap();
        graph.materialize(&source).unwrap();
        let affine = graph
            .affine(
                Arc::clone(&source),
                Render2dAffineTransform::translation(2.0, 0.0).unwrap(),
                &[6, 1],
            )
            .unwrap();
        let header = size_of::<support::NeutralMesh>() + 2 * size_of::<usize>();
        let vertex_bytes = 3 * size_of::<[f64; 2]>();
        let remaining = header + vertex_bytes - 1;
        graph
            .host
            .reserve(
                support::MAX_SHADOW_HOST_BYTES - graph.host.live() - remaining,
                &[6, 1],
            )
            .unwrap();
        let before = graph.host.live();
        assert!(matches!(
            graph.materialize(&affine),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [6, 1]
        ));
        assert_eq!(graph.host.live(), before);
        assert!(graph.materialized[affine.id].is_none());
    }

    #[test]
    fn materialized_transform_rejects_before_unfunded_vertex_copy() {
        let mut graph = builder();
        let source = graph.source(triangle(), &[4, 0]).unwrap();
        graph.materialize(&source).unwrap();
        let parent = graph
            .affine(
                Arc::clone(&source),
                Render2dAffineTransform::translation(1.0, 0.0).unwrap(),
                &[4, 1],
            )
            .unwrap();
        // Parent lookup/receipt, three initialized/copied cache slots and
        // child lookup fit. The subsequent vertex transform/copy does not.
        let prefix = 2 * MATERIALIZATION_LOOKUP_WORK + MATERIALIZATION_NODE_WORK + 6;
        let copy = shadow::append_work(&shadow::empty_mesh(), &triangle(), &[4, 1]).unwrap();
        graph.graph_work = MAX_REGION_GRAPH_WORK - prefix - copy + 1;
        assert!(matches!(graph.materialize(&parent),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [4, 1]));
        assert_eq!(graph.graph_work, MAX_REGION_GRAPH_WORK - copy + 1);
        assert!(graph.materialized[source.id].is_some());
        assert!(graph.materialized[parent.id].is_none());
    }

    #[test]
    fn finite_square_hull_rejects_before_unfunded_construction() {
        let mut graph = builder();
        let source = graph.source(triangle(), &[3, 0]).unwrap();
        graph.materialize(&source).unwrap();
        let parent = graph
            .effect(
                source,
                Render2dDropShadow::new(0.0, 0.0, 0.5, 0.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[3, 1],
            )
            .unwrap();
        let prefix = 2 * MATERIALIZATION_LOOKUP_WORK + MATERIALIZATION_NODE_WORK + 6;
        let copy = shadow::append_work(&shadow::empty_mesh(), &triangle(), &[3, 1]).unwrap();
        let hull = 8_192 + 256;
        graph.graph_work = MAX_REGION_GRAPH_WORK - prefix - copy - hull + 1;
        assert!(matches!(graph.materialize(&parent),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [3, 1]));
        assert_eq!(graph.graph_work, MAX_REGION_GRAPH_WORK - hull + 1);
        assert!(graph.materialized[parent.id].is_none());
    }

    #[test]
    fn symbolic_node_needs_fuel_even_without_source_vertices() {
        let mut graph = builder();
        let source = graph.source(shadow::empty_mesh(), &[5, 0]).unwrap();
        let nodes = graph.nodes;
        let paths = graph.retained_path_units;
        let work = REGION_NODE_WORK + 4;
        graph.graph_work = MAX_REGION_GRAPH_WORK - work + 1;
        let before = graph.graph_work;
        let effect =
            Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        assert!(matches!(graph.effect(source, effect, &[5, 1]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [5, 1]));
        assert_eq!(graph.graph_work, before);
        assert_eq!(graph.nodes, nodes);
        assert_eq!(graph.retained_path_units, paths);
    }

    #[test]
    fn authored_controls_are_admitted_before_source_reach_traversal() {
        let item = provenance_rectangle_item();
        let mesh = provenance_rectangle_mesh(1);
        let scan = 64 + SOURCE_VERTEX_WORK * mesh.triangles.len();
        let mut graph = builder();
        // The physical scan and fixed node work fit, but four original
        // controls require the interval transforms' full operation allowance.
        graph.graph_work = MAX_REGION_GRAPH_WORK - scan - REGION_NODE_WORK - 4;
        let before = graph.graph_work;
        assert!(matches!(graph.source_from_item(mesh, &item, &[8, 2]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [8, 2]));
        assert_eq!(graph.graph_work, before + scan);
        assert_eq!(graph.nodes, 0);
        assert_eq!(graph.retained_path_units, 0);
    }

    #[test]
    fn failed_envelope_keeps_paid_work_without_committing_node_or_path() {
        let mut graph = builder();
        let left = graph.source(triangle(), &[2, 0]).unwrap();
        let right = graph.source(triangle(), &[2, 1]).unwrap();
        let source = graph.union(vec![left, right], &[2]).unwrap();
        let nodes = graph.nodes;
        let paths = graph.retained_path_units;
        let before = graph.graph_work;
        let extreme = Render2dAffineTransform::new(f64::MAX, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap();
        assert!(matches!(graph.affine(source, extreme, &[2, 3]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path), ..
            }) if path == [2, 3]));
        assert_eq!(graph.graph_work, before + 64 + REGION_NODE_WORK + 4);
        assert_eq!(graph.nodes, nodes);
        assert_eq!(graph.retained_path_units, paths);
    }

    #[test]
    fn original_f1_provenance_reuses_shared_dag_proofs_with_atomic_budget() {
        use crate::composition_2d::{
            Render2dBrush, Render2dItem, Render2dOpacity, Render2dPrimitive, Render2dRect,
            Render2dShape,
        };
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let physical = support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let mut builder = builder();
        let mut shared = builder.source_from_item(physical, &item, &[2, 0]).unwrap();
        // Twelve union nodes create more than 4,000 naive repeated visits,
        // although the borrowed F1 source graph itself has only 13 nodes.
        for _ in 0..12 {
            // Construct a deliberately unnormalized historical DAG to test
            // the memo itself. The production union builder now deduplicates
            // these identical Arc children before materialization.
            shared = builder
                .node(
                    Kind::Union(vec![Arc::clone(&shared), Arc::clone(&shared)]),
                    &[2, 9],
                    shared.depth + 1,
                    None,
                )
                .unwrap();
        }
        let before = builder.graph_work;
        assert!(
            builder
                .proven_original_straight_polygon_union(&shared)
                .unwrap()
        );
        let paid = builder.graph_work - before;
        assert!(
            paid > 13,
            "node visits must include their actual source work"
        );
        assert!(paid < 20_000, "shared branches reuse completed receipts");
        assert!(
            builder.polygonal_proofs[1..14]
                .iter()
                .all(|entry| *entry == Some(true))
        );
        assert!(
            builder
                .proven_original_straight_polygon_union(&shared)
                .unwrap()
        );
        assert_eq!(builder.graph_work - before, paid + PROVENANCE_LOOKUP_WORK);
        assert!(builder.materialized.is_empty());

        // Cached truth still pays bounded lookup; it cannot mint extra fuel.
        builder.graph_work = MAX_REGION_GRAPH_WORK - PROVENANCE_LOOKUP_WORK;
        assert!(
            builder
                .proven_original_straight_polygon_union(&shared)
                .unwrap()
        );
        assert_eq!(builder.graph_work, MAX_REGION_GRAPH_WORK);
        assert!(
            matches!(builder.proven_original_straight_polygon_union(&shared),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [2, 9])
        );
        assert_eq!(builder.graph_work, MAX_REGION_GRAPH_WORK);
    }

    fn provenance_rectangle_item() -> Render2dItem {
        use crate::composition_2d::{
            Render2dBrush, Render2dOpacity, Render2dPrimitive, Render2dRect, Render2dShape,
        };
        Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        )
    }

    fn provenance_rectangle_mesh(repeats: usize) -> support::NeutralMesh {
        support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ]
            .repeat(repeats),
            bounds: [0.0, 0.0, 2.0, 2.0],
        }
    }

    #[test]
    fn source_preparation_prepays_once_on_the_same_live_graph_counter() {
        let item = provenance_rectangle_item();
        let mut direct = builder();
        let expected = direct
            .source_from_item(provenance_rectangle_mesh(1), &item, &[3, 2])
            .unwrap();
        let mut prepared = builder();
        let source = prepared
            .prepare_source_from_item(&item, &[3, 2], |work| {
                assert_eq!(
                    *work.graph,
                    REGION_NODE_WORK + 4 * SOURCE_REACH_CONTROL_WORK
                );
                work.charge(7, &[3, 2])?;
                Ok(Some(provenance_rectangle_mesh(1)))
            })
            .unwrap()
            .unwrap();
        assert_eq!(prepared.graph_work, direct.graph_work + 7);
        assert_eq!(prepared.retained_vertices, direct.retained_vertices);
        assert_eq!(source.envelope, expected.envelope);
        assert_eq!(source.path, expected.path);
        // An ancestor's distinct control traversal still pays its own work.
        direct
            .affine(
                expected,
                Render2dAffineTransform::translation(1.0, 0.0).unwrap(),
                &[3],
            )
            .unwrap();
        prepared
            .affine(
                source,
                Render2dAffineTransform::translation(1.0, 0.0).unwrap(),
                &[3],
            )
            .unwrap();
        assert_eq!(prepared.graph_work, direct.graph_work + 7);
    }

    #[test]
    fn source_preparation_rejects_before_callback_and_retains_failed_preparation_debits() {
        let item = provenance_rectangle_item();
        let cost = REGION_NODE_WORK + 4 * SOURCE_REACH_CONTROL_WORK;
        let mut graph = builder();
        graph.graph_work = MAX_REGION_GRAPH_WORK - cost + 1;
        let initial = graph.graph_work;
        let mut entered = false;
        assert!(
            matches!(graph.prepare_source_from_item(&item, &[8, 1], |_| {
            entered = true;
            Ok(None)
        }), Err(Render2dExecutionError::SampleSpace {
            kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
        }) if path == [8, 1])
        );
        assert!(!entered);
        assert_eq!(graph.graph_work, initial);
        assert_eq!(graph.nodes, 0);

        let mut graph = builder();
        assert!(
            matches!(graph.prepare_source_from_item(&item, &[8, 2], |work| {
            work.charge(13, &[8, 2])?;
            Err(precision(&[8, 2], "test physical source failure"))
        }), Err(Render2dExecutionError::SampleSpace {
            kind: Render2dSampleSpaceError::PrecisionLimit, path: Some(path), ..
        }) if path == [8, 2])
        );
        assert_eq!(graph.graph_work, cost + 13);
        assert_eq!(graph.nodes, 0);
        assert_eq!(graph.retained_vertices, 0);
        assert!(
            graph
                .prepare_source_from_item(&item, &[8, 3], |_| Ok(None))
                .unwrap()
                .is_none()
        );
        assert_eq!(
            graph.graph_work,
            2 * cost + 13,
            "a completed empty preparation cannot refund work"
        );
        assert_eq!(graph.nodes, 0);
    }

    #[test]
    fn deferred_source_failures_do_not_poison_once_only_realization_or_correspondence() {
        let item = provenance_rectangle_item();
        let mut graph = builder();
        let source = graph.register_source(&item, &[6, 2]).unwrap().unwrap();
        let Kind::Source(geometry) = &source.kind else {
            panic!("registered leaf must preserve original-source ownership");
        };
        assert!(geometry.get().is_none());
        assert!(matches!(
            graph.proven_original_straight_polygon_union(&source),
            Err(Render2dExecutionError::UnsupportedEntry { .. })
        ));
        let before_failure = graph.graph_work;
        assert!(matches!(
            graph.prepare_registered_source(&source, |_, path, work| {
                work.charge(13, path)?;
                Err(precision(path, "test reconstruction failure"))
            }),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit, path: Some(path), ..
            }) if path == [6, 2]
        ));
        assert_eq!(graph.graph_work, before_failure + 13);
        assert!(geometry.get().is_none());
        assert_eq!(graph.retained_vertices, 0);
        graph
            .prepare_registered_source(&source, |borrowed, path, work| {
                assert!(std::ptr::eq(borrowed, &item));
                assert_eq!(path, &[6, 2]);
                work.charge(7, path)?;
                Ok(Some(provenance_rectangle_mesh(1)))
            })
            .unwrap();
        let prepared_work = graph.graph_work;
        graph
            .prepare_registered_source(&source, |_, _, _| {
                panic!("a completed source must never reconstruct twice")
            })
            .unwrap();
        assert_eq!(
            graph.graph_work,
            prepared_work + MATERIALIZATION_LOOKUP_WORK
        );
        assert_eq!(graph.retained_vertices, 6);
        assert!(
            graph
                .proven_original_straight_polygon_union(&source)
                .unwrap()
        );
        let physical = graph.materialize(&source).unwrap();
        assert!(Arc::ptr_eq(&physical, geometry.get().unwrap()));
        assert!(Arc::ptr_eq(&physical, &graph.materialize(&source).unwrap()));
    }

    fn factory_image_item(second_x: f64) -> Render2dItem {
        use crate::composition_2d::*;
        let image = Render2dImagePrimitive::new(
            Render2dResourceId::new(1).unwrap(),
            Render2dPixelExtent::new(1, 1).unwrap(),
            [0.0, second_x]
                .map(|x| {
                    Render2dImagePatch::new(
                        Render2dImageSourceRect::new(0.0, 0.0, 0.0, 0.0).unwrap(),
                        Render2dRect::new(x, 0.0, 2.0, 2.0).unwrap(),
                    )
                })
                .to_vec(),
        )
        .unwrap();
        Render2dItem::new(
            Render2dPrimitive::Image(image),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        )
    }

    #[test]
    fn image_factory_transfers_retained_capacity_and_refunds_only_failed_storage() {
        let item = factory_image_item(4.0);
        let mut graph = builder();
        let path = [7, 1];
        let source = graph.register_source(&item, &path).unwrap().unwrap();
        let Kind::Source(geometry) = &source.kind else {
            unreachable!()
        };
        let before = graph.host.live();
        let work_before = graph.graph_work;
        let prepare =
            |item: &Render2dItem, path: &[usize], work: &mut shadow::NeutralPreparationWork<'_>| {
                shadow::neutral_item(
                    item,
                    path,
                    &crate::composition_2d::Render2dResourceBindings::new(Vec::new()).unwrap(),
                    &std::collections::BTreeMap::new(),
                    shadow::NeutralPreparationScale {
                        field_raster_scale: 1.0,
                        geometry_sample_scale: 4.0,
                    },
                    1_048_576,
                    work,
                )
            };
        let bytes = 12 * size_of::<[f64; 2]>();
        assert!(
            graph
                .prepare_registered_source(&source, |item, path, work| {
                    let mesh = prepare(item, path, work)?.unwrap();
                    assert_eq!(work.host.live(), before + bytes);
                    drop(mesh);
                    Err(precision(path, "independent late factory rejection"))
                })
                .is_err()
        );
        assert!(geometry.get().is_none());
        assert_eq!(graph.host.live(), before);
        assert_eq!(graph.host.peak(), before + bytes);
        let paid = graph.graph_work - work_before;
        assert!(paid > 0);
        assert_eq!(graph.retained_vertices, 0);

        graph.prepare_registered_source(&source, prepare).unwrap();
        let mesh = geometry.get().unwrap();
        assert_eq!(mesh.triangles.len(), 12);
        assert_eq!(mesh.units_per_parent_logical_unit, 4.0);
        let retained = NeutralRegionBuilder::mesh_bytes(mesh).unwrap();
        assert_eq!(graph.host.live(), before + retained);
        assert_eq!(graph.host.peak(), before + retained);
        assert!(graph.graph_work >= work_before + 2 * paid);
        assert_eq!(graph.retained_vertices, 12);
        let completed = graph.graph_work;
        graph
            .prepare_registered_source(&source, |_, _, _| panic!("already prepared"))
            .unwrap();
        assert_eq!(graph.graph_work, completed + MATERIALIZATION_LOOKUP_WORK);
        assert_eq!(graph.host.live(), before + retained);
    }

    #[test]
    fn late_image_precision_failure_releases_factory_storage_without_installing_source() {
        let item = factory_image_item(2.0_f64.powi(44));
        let mut graph = builder();
        let path = [7, 2];
        let source = graph.register_source(&item, &path).unwrap().unwrap();
        let Kind::Source(geometry) = &source.kind else {
            unreachable!()
        };
        let before = graph.host.live();
        let work_before = graph.graph_work;
        let nodes_before = graph.nodes;
        assert!(
            matches!(graph.prepare_registered_source(&source, |item, path, work| {
            shadow::neutral_item(
                item, path, &crate::composition_2d::Render2dResourceBindings::new(Vec::new()).unwrap(),
                &std::collections::BTreeMap::new(),
                shadow::NeutralPreparationScale {
                    field_raster_scale: 1.0,
                    geometry_sample_scale: 4.0,
                },
                1_048_576, work,
            )
        }), Err(Render2dExecutionError::SampleSpace {
            kind: Render2dSampleSpaceError::PrecisionLimit, path: Some(owner), ..
        }) if owner == path)
        );
        assert!(geometry.get().is_none());
        assert_eq!(graph.host.live(), before);
        assert_eq!(graph.host.peak(), before + 12 * size_of::<[f64; 2]>());
        assert!(graph.graph_work > work_before);
        assert_eq!(graph.retained_vertices, 0);
        assert_eq!(graph.nodes, nodes_before);
    }

    #[test]
    fn source_inventory_counts_spare_mesh_capacity_and_discarded_nodes() {
        let mut graph = builder();
        let mut mesh = provenance_rectangle_mesh(1);
        mesh.triangles.try_reserve_exact(256).unwrap();
        let triangle_bytes = mesh.triangles.capacity() * size_of::<[f64; 2]>();
        let source = graph.source(mesh, &[4, 3, 2]).unwrap();
        let dropped = graph.source(triangle(), &[8, 7, 6, 5]).unwrap();
        drop(dropped);
        let work_before = graph.graph_work;
        let (retained, construction) = graph.storage_bounds([&source]).unwrap();
        assert!(retained > triangle_bytes);
        assert!(construction > retained);
        assert!(graph.graph_work > work_before);
    }

    fn consumer_group(
        children: Vec<crate::composition_2d::Render2dEntry>,
        effect: Render2dDropShadow,
    ) -> crate::composition_2d::Render2dEntry {
        use crate::composition_2d::{Render2dEntry, Render2dGroup, Render2dOpacity};
        Render2dEntry::group(Render2dGroup::new(
            children,
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::OPAQUE,
            vec![effect],
        ))
    }

    #[test]
    fn selected_polygon_consumers_leave_unselected_sources_and_effects_unrealized() {
        use crate::composition_2d::{
            Render2dBrush, Render2dComposition, Render2dEntry, Render2dOpacity, Render2dPrimitive,
            Render2dRect, Render2dResourceBindings, Render2dShape,
        };
        let zero = Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::WHITE).unwrap();
        let ellipse = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::ellipse(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let composition = Render2dComposition::new(vec![
            consumer_group(vec![Render2dEntry::item(provenance_rectangle_item())], zero),
            consumer_group(
                vec![consumer_group(vec![Render2dEntry::item(ellipse)], zero)],
                zero,
            ),
        ])
        .unwrap();
        let plan = shadow::scene::analyze(&composition, &mut ()).unwrap();
        let bindings = Render2dResourceBindings::default();
        let fields = std::collections::BTreeMap::new();
        let build = || {
            shadow::build_group_sources(
                plan.composition(),
                plan.retained_bytes(),
                &bindings,
                &fields,
                shadow::SourcePreparation::new(
                    shadow::NeutralPreparationScale {
                        field_raster_scale: 1.0,
                        geometry_sample_scale: 4.0,
                    },
                    1_048_576,
                ),
            )
            .unwrap()
        };
        // Each table fits this lease, but the retained source phase and
        // tables coexist. Reject before the first physical source factory.
        let mut dag = build();
        let original = dag.source(0).unwrap();
        let before = dag.graph_work();
        let storage = dag.storage();
        let source_phase = storage.input_bytes + storage.construction_peak_bytes;
        assert!(source_phase > 65_536);
        assert!(matches!(
            dag.selected_dependencies(&[0], source_phase as u64),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
        assert_eq!(dag.graph_work(), before + 32 + 128);
        assert_eq!(
            dag.storage().construction_peak_bytes,
            storage.construction_peak_bytes
        );
        let Kind::Source(geometry) = &original.kind else {
            panic!("original leaf")
        };
        assert!(geometry.get().is_none());

        // The exact combined lease admits both immutable tables. Their
        // capacity is reported in addition to the existing traversal reserve.
        let mut dag = build();
        let metadata =
            dag.graph.nodes + 1 + dag.sources.len() + size_of::<PolygonDependencies<'_>>();
        let lease = (source_phase + metadata) as u64;
        let dependencies = dag.selected_dependencies(&[0], lease).unwrap();
        assert_eq!(dependencies.metadata_bytes(), metadata);
        assert_eq!(dependencies.frames.capacity(), MAX_REGION_DEPTH);
        assert!(dependencies.frames.is_empty());
        assert_eq!(
            dag.storage().construction_peak_bytes,
            storage.construction_peak_bytes + metadata
        );
        assert_eq!(
            dag.storage().input_bytes + dag.storage().construction_peak_bytes,
            lease as usize
        );
        drop(dependencies);

        let dag = build();
        let rectangle = dag.source(0).unwrap();
        let ellipse = dag.source(4).unwrap();
        assert!(
            dag.demands
                .iter()
                .any(|d| matches!(d, shadow::SourceDemand::Effect(_)))
        );
        let prepared = dag
            .prepare_polygonal_consumers(Some(&[0]), None, 1_048_576)
            .unwrap();
        assert_eq!(
            prepared
                .casters
                .iter()
                .map(|entry| entry.0)
                .collect::<Vec<_>>(),
            vec![0]
        );
        let Kind::Source(rectangle_geometry) = &rectangle.kind else {
            panic!("rectangle leaf")
        };
        let Kind::Source(ellipse_geometry) = &ellipse.kind else {
            panic!("ellipse leaf")
        };
        assert!(rectangle_geometry.get().is_some());
        assert!(ellipse_geometry.get().is_none());
        let mut fuel = 1_000_000;
        assert_eq!(
            certified_cells::certified_area_enclosure(
                &ellipse,
                [0.9, 0.9, 1.1, 1.1],
                None,
                4,
                &mut fuel
            ),
            Some([1.0, 1.0]),
        );

        // An explicit original-only request is different from the default
        // physical consumer. It must also skip descendant effect warmups.
        let dag = build();
        let original = dag.source(0).unwrap();
        let before = dag.graph_work();
        let prepared = dag
            .prepare_polygonal_consumers(Some(&[]), None, 1_048_576)
            .unwrap();
        assert!(prepared.casters.is_empty());
        assert!(
            prepared.work.used() > before,
            "dependency metadata is actual work"
        );
        let Kind::Source(geometry) = &original.kind else {
            panic!("original leaf")
        };
        assert!(geometry.get().is_none());

        // Validate ALL selection indices before realizing an earlier valid
        // request. Invalid caller keys must not install a physical receipt.
        let dag = build();
        let original = dag.source(0).unwrap();
        assert!(matches!(
            dag.prepare_polygonal_consumers(Some(&[0, usize::MAX]), None, 1_048_576),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                ..
            })
        ));
        let Kind::Source(geometry) = &original.kind else {
            panic!("original leaf")
        };
        assert!(geometry.get().is_none());
    }

    #[test]
    fn lexical_sources_preserve_active_plan_indices_and_explicit_live_input() {
        use crate::composition_2d::{Render2dComposition, Render2dEntry, Render2dResourceBindings};
        let zero = Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::WHITE).unwrap();
        let inert = crate::composition_2d::Render2dGroup::new(
            Vec::<Render2dEntry>::new(),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            crate::composition_2d::Render2dOpacity::TRANSPARENT,
            Vec::new(),
        );
        let child = consumer_group(vec![Render2dEntry::item(provenance_rectangle_item())], zero);
        let composition = Render2dComposition::new(vec![
            Render2dEntry::group(inert.clone()),
            consumer_group(
                vec![Render2dEntry::group(inert), child.clone(), child],
                zero,
            ),
        ])
        .unwrap();
        let bindings = Render2dResourceBindings::default();
        let fields = std::collections::BTreeMap::new();
        let build = |input_bytes| {
            shadow::build_group_sources(
                &composition,
                input_bytes,
                &bindings,
                &fields,
                shadow::SourcePreparation::new(
                    shadow::NeutralPreparationScale {
                        field_raster_scale: 1.0,
                        geometry_sample_scale: 4.0,
                    },
                    1_048_576,
                ),
            )
            .unwrap()
        };
        // There is no Plan or cumulative affine dependency at this point.
        let original = build(0);
        assert_eq!(original.storage().input_bytes, 0);
        let plan = shadow::scene::analyze(&composition, &mut ()).unwrap();
        let planned = build(plan.retained_bytes());
        assert_eq!(planned.storage().input_bytes, plan.retained_bytes());
        assert_eq!(original.graph_work(), planned.graph_work());
        assert_eq!(
            original.storage().retained_bytes,
            planned.storage().retained_bytes
        );
        let mut owners = Vec::new();
        for (index, event) in plan.events.iter().enumerate() {
            if let shadow::scene::Event::BeginGroup { group, path, .. } = event
                && !group.shadows().is_empty()
            {
                owners.push(index);
                let source = original.source(index).unwrap();
                assert_eq!(source.envelope, planned.source(index).unwrap().envelope);
                assert_eq!(
                    source.owner_path(),
                    planned.source(index).unwrap().owner_path()
                );
                assert_eq!(path[0], 1);
            }
        }
        assert_eq!(owners, [0, 1, 4]);
        assert_eq!(
            original
                .sources
                .iter()
                .map(|entry| entry.0)
                .collect::<Vec<_>>(),
            owners
        );
        assert!(original.graph.materialized.is_empty());
        assert_eq!(original.graph.retained_vertices, 0);
    }

    #[test]
    fn host_only_source_cursor_does_not_require_native_buffer_capacity() {
        use crate::composition_2d::{Render2dComposition, Render2dEntry, Render2dResourceBindings};
        let composition = Render2dComposition::new(Vec::<Render2dEntry>::new()).unwrap();
        let bindings = Render2dResourceBindings::default();
        let fields = std::collections::BTreeMap::new();
        // No scene Plan, source node, upload or GPU buffer is needed. The
        // cursor's fixed host storage must not inherit a device buffer limit.
        let dag = shadow::build_group_sources(
            &composition,
            0,
            &bindings,
            &fields,
            shadow::SourcePreparation::new(
                shadow::NeutralPreparationScale {
                    field_raster_scale: 1.0,
                    geometry_sample_scale: 4.0,
                },
                32,
            ),
        )
        .unwrap();
        assert!(dag.sources.is_empty());
        assert!(dag.demands.is_empty());
        assert_eq!(dag.graph.nodes, 0);
        assert_eq!(dag.storage().input_bytes, 0);
        assert!(
            dag.storage().construction_peak_bytes
                >= shadow::scene::OccurrenceCursor::storage_bytes()
        );
        assert_eq!(dag.graph.storage_limit(), 32);
    }

    #[test]
    fn lexical_source_work_includes_inert_occurrences_before_any_plan() {
        use crate::composition_2d::{
            Render2dComposition, Render2dEntry, Render2dGroup, Render2dOpacity,
            Render2dResourceBindings,
        };
        let inert = Render2dEntry::group(Render2dGroup::new(
            Vec::<Render2dEntry>::new(),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
            Vec::new(),
        ));
        let composition = Render2dComposition::new(vec![inert; 140_000]).unwrap();
        let bindings = Render2dResourceBindings::default();
        let fields = std::collections::BTreeMap::new();
        let mut preparation = shadow::SourcePreparation::new(
            shadow::NeutralPreparationScale {
                field_raster_scale: 1.0,
                geometry_sample_scale: 4.0,
            },
            1_048_576,
        );
        // An earlier phase already spent this scalar debit. Besides proving
        // ownership continuity, it places exhaustion at a leaf's closing step.
        preparation.graph.charge_preparation(64, &[]).unwrap();
        let result = shadow::build_group_sources(&composition, 0, &bindings, &fields, preparation);
        // Source-owner setup (64), host/input transfer (128), builder dispatch
        // (32), cursor setup (128) and initial next (32) pay first. Each inert
        // root then needs a visit and a closing-frame step.
        // Rejection pays no closing pop, so its known owner is that leaf.
        let owner = (MAX_REGION_GRAPH_WORK - 64 - 64 - 128 - 32 - 128 - 32) / (2 * 64);
        let error = match result {
            Err(error) => error,
            Ok(_) => panic!("inert occurrences must consume source fuel"),
        };
        assert!(
            matches!(&error, Render2dExecutionError::SampleSpace {
            kind: Render2dSampleSpaceError::ResourceLimit,
            path: Some(path),
            ..
        } if path == &[owner]),
            "expected owner {owner}, observed {error:?}"
        );
        // The ordinary non-shadow scene policy still permits this bounded
        // authored tree, whose active affine plan is genuinely empty.
        assert!(
            shadow::scene::analyze(&composition, &mut ())
                .unwrap()
                .events
                .is_empty()
        );
    }

    #[test]
    fn selected_ancestor_and_child_share_the_same_original_and_physical_source() {
        use crate::composition_2d::{Render2dComposition, Render2dEntry, Render2dResourceBindings};
        let zero = Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::WHITE).unwrap();
        let eroded =
            Render2dDropShadow::new(0.0, 0.0, 0.0, -1.0, Render2dColorRgba8::WHITE).unwrap();
        // The child erosion is a subset of C and is absorbed only for
        // ancestor support. Its own signed physical consumer still exists.
        let composition = Render2dComposition::new(vec![consumer_group(
            vec![consumer_group(
                vec![Render2dEntry::item(provenance_rectangle_item())],
                eroded,
            )],
            zero,
        )])
        .unwrap();
        let plan = shadow::scene::analyze(&composition, &mut ()).unwrap();
        let bindings = Render2dResourceBindings::default();
        let fields = std::collections::BTreeMap::new();
        let dag = shadow::build_group_sources(
            plan.composition(),
            plan.retained_bytes(),
            &bindings,
            &fields,
            shadow::SourcePreparation::new(
                shadow::NeutralPreparationScale {
                    field_raster_scale: 1.0,
                    geometry_sample_scale: 4.0,
                },
                1_048_576,
            ),
        )
        .unwrap();
        let parent = dag.source(0).unwrap();
        let child = dag.source(1).unwrap();
        assert!(Arc::ptr_eq(&parent, &child));
        // Unsorted/repeated indices describe the same set of consumers.
        let prepared = dag
            .prepare_polygonal_consumers(Some(&[1, 0, 1]), None, 1_048_576)
            .unwrap();
        assert_eq!(prepared.casters.len(), 2);
        let shadow::NeutralCaster::Mesh(parent_mesh) = prepared.caster(0).unwrap() else {
            panic!("parent mesh")
        };
        let shadow::NeutralCaster::OriginalStraightPolygonUnion(child_mesh) =
            prepared.caster(1).unwrap()
        else {
            panic!("original signed polygon proof")
        };
        let Kind::Source(geometry) = &parent.kind else {
            panic!("shared original leaf")
        };
        assert!(Arc::ptr_eq(parent_mesh, child_mesh));
        assert!(Arc::ptr_eq(parent_mesh, geometry.get().unwrap()));
        assert_eq!(geometry.get().unwrap().triangles.len(), 6);
        assert_eq!(parent.owner_path(), &[0, 0, 0]);
    }

    #[test]
    fn polygon_dependency_metadata_and_each_visit_require_prepaid_admission() {
        let item = provenance_rectangle_item();
        let mut graph = builder();
        let source = graph.register_source(&item, &[9, 2]).unwrap().unwrap();
        graph.graph_work = MAX_REGION_GRAPH_WORK - 1;
        assert!(matches!(
            graph.polygon_dependencies(1, MAX_REGION_DEPTH * 128, u64::MAX),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
        let Kind::Source(geometry) = &source.kind else {
            panic!("source leaf")
        };
        assert!(geometry.get().is_none());
        // Funding only the scalar preflight still cannot allocate/initialize
        // the tables: the per-entry debit must succeed first.
        graph.graph_work = MAX_REGION_GRAPH_WORK - 128;
        assert!(
            graph
                .polygon_dependencies(1, MAX_REGION_DEPTH * 128, u64::MAX)
                .is_err()
        );
        assert_eq!(graph.graph_work, MAX_REGION_GRAPH_WORK);
        assert!(geometry.get().is_none());
        let mut tiny = builder();
        tiny.max_buffer_bytes = 1;
        assert!(matches!(
            tiny.polygon_dependencies(1, MAX_REGION_DEPTH * 128, u64::MAX),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                ..
            })
        ));
        assert_eq!(tiny.graph_work, 128);
        graph.graph_work = 0;
        let mut dependencies = graph
            .polygon_dependencies(1, MAX_REGION_DEPTH * 128, u64::MAX)
            .unwrap();
        graph.graph_work = MAX_REGION_GRAPH_WORK - 63;
        assert!(
            matches!(graph.require_polygon_dependencies(&source, &mut dependencies), Err(Render2dExecutionError::SampleSpace {
            kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
        }) if path == [9, 2])
        );
        assert!(!dependencies.source(&source));
        graph.graph_work = 0;
        graph
            .require_polygon_dependencies(&source, &mut dependencies)
            .unwrap();
        assert_eq!(graph.graph_work, 64);
        assert!(dependencies.source(&source));
        graph
            .require_polygon_dependencies(&source, &mut dependencies)
            .unwrap();
        assert_eq!(graph.graph_work, 128, "shared dependency lookups still pay");
        assert!(
            geometry.get().is_none(),
            "marking is never physical realization"
        );
    }

    #[test]
    fn polygon_dependency_frames_cover_maximum_admitted_depth_without_recursion() {
        let item = provenance_rectangle_item();
        let mut graph = builder();
        let leaf = graph.register_source(&item, &[9, 2]).unwrap().unwrap();
        let mut root = Arc::clone(&leaf);
        for _ in 1..MAX_REGION_DEPTH {
            root = graph
                .affine(
                    root,
                    Render2dAffineTransform::translation(1.0, 0.0).unwrap(),
                    &[9],
                )
                .unwrap();
        }
        assert_eq!(root.depth, MAX_REGION_DEPTH);
        let mut dependencies = graph
            .polygon_dependencies(1, MAX_REGION_DEPTH * 128, u64::MAX)
            .unwrap();
        let before = graph.graph_work;
        graph
            .require_polygon_dependencies(&root, &mut dependencies)
            .unwrap();
        assert_eq!(graph.graph_work - before, MAX_REGION_DEPTH * 64);
        assert!(dependencies.source(&root));
        assert!(dependencies.source(&leaf));
        assert!(dependencies.frames.is_empty());
        assert!(
            dependencies.frames.capacity() * size_of::<(Arc<NeutralRegion<'_>>, usize)>()
                <= MAX_REGION_DEPTH * 128
        );
        // The same frame capacity serves another selected, already-marked root.
        graph
            .require_polygon_dependencies(&leaf, &mut dependencies)
            .unwrap();
        assert_eq!(graph.graph_work - before, MAX_REGION_DEPTH * 64 + 64);
        let Kind::Source(geometry) = &leaf.kind else {
            panic!("leaf")
        };
        assert!(geometry.get().is_none());
    }

    #[test]
    fn nested_erosion_requires_graph_fuel_before_local_geometry() {
        let item = provenance_rectangle_item();
        let mut graph = builder();
        let source = graph
            .source_from_item(provenance_rectangle_mesh(1), &item, &[7, 0])
            .unwrap();
        assert!(
            graph
                .proven_original_straight_polygon_union(&source)
                .unwrap()
        );
        let original = graph.materialize(&source).unwrap();
        let erosion = graph
            .effect(
                Arc::clone(&source),
                Render2dDropShadow::new(0.0, 0.0, 0.0, -0.25, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[7, 1],
            )
            .unwrap();
        // Pre-existing receipt storage isolates the remaining numerical
        // fuel: effect body, provenance lookup and child receipt lookup fit;
        // the very first erosion geometry phase has only one unit left.
        graph.materialized.resize_with(erosion.id + 1, || None);
        let prefix =
            MATERIALIZATION_NODE_WORK + 2 * MATERIALIZATION_LOOKUP_WORK + PROVENANCE_LOOKUP_WORK;
        graph.graph_work = MAX_REGION_GRAPH_WORK - prefix - 1;
        let vertices = graph.retained_vertices;
        assert!(matches!(graph.materialize(&erosion),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path == [7, 1]));
        assert_eq!(graph.graph_work, MAX_REGION_GRAPH_WORK - 1);
        assert_eq!(
            graph.erosion_work, 0,
            "unadmitted geometry must not execute"
        );
        assert_eq!(graph.retained_vertices, vertices);
        assert!(graph.materialized[erosion.id].is_none());
        assert!(Arc::ptr_eq(
            graph.materialized[source.id].as_ref().unwrap(),
            &original
        ));
    }

    #[test]
    fn failed_erosion_candidate_and_convex_fallback_keep_the_same_graph_debits() {
        let item = provenance_rectangle_item();
        // Duplicate exact rectangle triangles qualify as original support,
        // but are not disconnected components. Separation must fail after
        // doing work, then the convex whole-union proof must run normally.
        let mesh = provenance_rectangle_mesh(2);
        let mut expected_local = 0;
        assert!(
            support::certified_separated_rectangles_erosion_mesh(
                &mesh,
                0.25,
                &[8, 1],
                &mut expected_local,
            )
            .unwrap()
            .is_none()
        );
        let failed_candidate = expected_local;
        assert!(failed_candidate > 0);
        let expected =
            support::certified_convex_erosion_mesh(&mesh, 0.25, &[8, 1], &mut expected_local)
                .unwrap()
                .unwrap();
        assert!(expected_local > failed_candidate);
        let append = shadow::append_work(&shadow::empty_mesh(), &expected, &[8, 1]).unwrap();

        for reject_fallback in [false, true] {
            let mut graph = builder();
            let source = graph
                .source_from_item(provenance_rectangle_mesh(2), &item, &[8, 0])
                .unwrap();
            assert!(
                graph
                    .proven_original_straight_polygon_union(&source)
                    .unwrap()
            );
            graph.materialize(&source).unwrap();
            let erosion = graph
                .effect(
                    source,
                    Render2dDropShadow::new(0.0, 0.0, 0.0, -0.25, Render2dColorRgba8::TRANSPARENT)
                        .unwrap(),
                    &[8, 1],
                )
                .unwrap();
            graph.materialized.resize_with(erosion.id + 1, || None);
            let prefix = MATERIALIZATION_NODE_WORK
                + 2 * MATERIALIZATION_LOOKUP_WORK
                + PROVENANCE_LOOKUP_WORK;
            if reject_fallback {
                graph.graph_work = MAX_REGION_GRAPH_WORK - prefix - failed_candidate - 1;
                assert!(matches!(graph.materialize(&erosion),
                    Err(Render2dExecutionError::SampleSpace {
                        kind: Render2dSampleSpaceError::ResourceLimit,
                        path: Some(path), ..
                    }) if path == [8, 1]));
                assert_eq!(graph.erosion_work, failed_candidate);
                assert_eq!(graph.graph_work, MAX_REGION_GRAPH_WORK - 1);
                assert!(graph.materialized[erosion.id].is_none());
            } else {
                let before = graph.graph_work;
                let actual = graph.materialize(&erosion).unwrap();
                assert_eq!(actual.triangles, expected.triangles);
                assert_eq!(actual.bounds, expected.bounds);
                assert_eq!(graph.erosion_work, expected_local);
                assert_eq!(graph.graph_work - before, prefix + expected_local + append);
            }
        }
    }

    #[test]
    fn sibling_erosions_accumulate_graph_fuel_once_and_cached_meshes_pay_only_lookup() {
        let item = provenance_rectangle_item();
        let mut graph = builder();
        let source = graph
            .source_from_item(provenance_rectangle_mesh(1), &item, &[6, 0])
            .unwrap();
        assert!(
            graph
                .proven_original_straight_polygon_union(&source)
                .unwrap()
        );
        let mesh = graph.materialize(&source).unwrap();
        let mut siblings = Vec::new();
        for (index, radius) in [(1, 0.25), (2, 0.5)] {
            let erosion = graph
                .effect(
                    Arc::clone(&source),
                    Render2dDropShadow::new(
                        0.0,
                        0.0,
                        0.0,
                        -radius,
                        Render2dColorRgba8::TRANSPARENT,
                    )
                    .unwrap(),
                    &[6, index],
                )
                .unwrap();
            siblings.push((erosion, radius));
        }
        graph
            .materialized
            .resize_with(siblings.last().unwrap().0.id + 1, || None);
        let before = graph.graph_work;
        let prefix =
            MATERIALIZATION_NODE_WORK + 2 * MATERIALIZATION_LOOKUP_WORK + PROVENANCE_LOOKUP_WORK;
        let mut expected_local = 0;
        let mut expected_graph = before;
        for (erosion, radius) in siblings {
            let paid = expected_local;
            let expected = support::certified_convex_erosion_mesh(
                &mesh,
                radius,
                &erosion.path,
                &mut expected_local,
            )
            .unwrap()
            .unwrap();
            let append =
                shadow::append_work(&shadow::empty_mesh(), &expected, &erosion.path).unwrap();
            expected_graph += prefix + expected_local - paid + append;
            let actual = graph.materialize(&erosion).unwrap();
            assert_eq!(actual.triangles, expected.triangles);
            assert_eq!(graph.erosion_work, expected_local);
            assert_eq!(graph.graph_work, expected_graph);

            assert!(Arc::ptr_eq(&actual, &graph.materialize(&erosion).unwrap()));
            expected_graph += MATERIALIZATION_LOOKUP_WORK;
            assert_eq!(graph.graph_work, expected_graph);
            assert_eq!(
                graph.erosion_work, expected_local,
                "no erosion replay on a hit"
            );
        }
    }

    #[test]
    fn provenance_large_leaf_and_fanout_reject_before_proof_cache_allocation() {
        let item = provenance_rectangle_item();
        let mut graph = builder();
        let leaf = graph
            .source_from_item(provenance_rectangle_mesh(4_096), &item, &[6, 2])
            .unwrap();
        graph.graph_work = MAX_REGION_GRAPH_WORK - 512;
        assert!(
            matches!(graph.proven_original_straight_polygon_union(&leaf),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [6, 2])
        );
        assert_eq!(
            graph.graph_work,
            MAX_REGION_GRAPH_WORK - 512 + PROVENANCE_LOOKUP_WORK
        );
        assert_eq!(graph.polygonal_proofs.capacity(), 0);
        assert!(graph.materialized.is_empty());

        let mut graph = builder();
        let leaf = graph
            .source_from_item(provenance_rectangle_mesh(1), &item, &[7, 2])
            .unwrap();
        let fanout = graph
            .node(Kind::Union(vec![leaf; 2_048]), &[7, 3], 2, None)
            .unwrap();
        graph.graph_work = MAX_REGION_GRAPH_WORK - 512;
        assert!(
            matches!(graph.proven_original_straight_polygon_union(&fanout),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [7, 3])
        );
        assert_eq!(
            graph.graph_work,
            MAX_REGION_GRAPH_WORK - 512 + PROVENANCE_LOOKUP_WORK
        );
        assert_eq!(graph.polygonal_proofs.capacity(), 0);
        assert!(graph.materialized.is_empty());
    }

    #[test]
    fn provenance_growth_cannot_relocate_unfunded_existing_receipts() {
        let item = provenance_rectangle_item();
        let mut graph = builder();
        graph.polygonal_proofs = vec![None; 4_096];
        let previous_capacity = graph.polygonal_proofs.capacity();
        // Simulate the valid monotonic ID inventory after earlier completed
        // queries. One new root needs only one initialized slot, but may copy
        // all old receipts. The old initialization-only charge would fit.
        graph.nodes = previous_capacity - 1;
        let root = graph
            .source_from_item(provenance_rectangle_mesh(1), &item, &[7, 8])
            .unwrap();
        assert_eq!(root.id, previous_capacity);
        graph.graph_work = MAX_REGION_GRAPH_WORK - 4_096;
        assert!(
            matches!(graph.proven_original_straight_polygon_union(&root),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [7, 8])
        );
        assert_eq!(graph.polygonal_proofs.capacity(), previous_capacity);
        assert_eq!(graph.polygonal_proofs.len(), 4_096);
        assert!(graph.polygonal_proofs.iter().all(Option::is_none));
        assert_eq!(
            graph.graph_work,
            MAX_REGION_GRAPH_WORK - 4_096 + PROVENANCE_LOOKUP_WORK
        );
    }

    #[test]
    fn provenance_failed_child_keeps_paid_work_and_completed_receipts_only() {
        let item = provenance_rectangle_item();
        let mut graph = builder();
        let small = graph
            .source_from_item(provenance_rectangle_mesh(1), &item, &[6, 1])
            .unwrap();
        let large = graph
            .source_from_item(provenance_rectangle_mesh(4_096), &item, &[6, 2])
            .unwrap();
        let root = graph
            .union(vec![Arc::clone(&small), Arc::clone(&large)], &[6, 9])
            .unwrap();
        graph.graph_work = MAX_REGION_GRAPH_WORK - 10_000;
        let before = graph.graph_work;
        assert!(
            matches!(graph.proven_original_straight_polygon_union(&root),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [6, 2])
        );
        assert!(graph.graph_work > before + PROVENANCE_LOOKUP_WORK);
        assert!(graph.graph_work < MAX_REGION_GRAPH_WORK);
        assert_eq!(graph.polygonal_proofs[small.id], Some(true));
        assert_eq!(graph.polygonal_proofs[large.id], None);
        assert_eq!(graph.polygonal_proofs[root.id], None);
        let paid = graph.graph_work;
        assert!(
            graph
                .proven_original_straight_polygon_union(&small)
                .unwrap()
        );
        assert_eq!(graph.graph_work, paid + PROVENANCE_LOOKUP_WORK);
        assert!(graph.materialized.is_empty());
    }

    #[test]
    fn provenance_cache_memory_rejects_before_allocation_with_owner_path() {
        let item = provenance_rectangle_item();
        let mut graph = builder();
        let leaf = graph
            .source_from_item(provenance_rectangle_mesh(1), &item, &[7, 4])
            .unwrap();
        graph.max_buffer_bytes = 1;
        let before = graph.graph_work;
        assert!(
            matches!(graph.proven_original_straight_polygon_union(&leaf),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [7, 4])
        );
        assert_eq!(graph.graph_work, before + PROVENANCE_LOOKUP_WORK);
        assert_eq!(graph.polygonal_proofs.capacity(), 0);
    }

    #[test]
    fn rounded_caster_qualification_reuses_canonical_source_receipt_and_pays_work() {
        let item = provenance_rectangle_item();
        let mut graph = builder();
        let leaf = graph
            .source_from_item(provenance_rectangle_mesh(1), &item, &[8, 0])
            .unwrap();
        let disk = graph
            .effect(
                Arc::clone(&leaf),
                Render2dDropShadow::new(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[8, 1],
            )
            .unwrap();
        let root = graph.union(vec![Arc::clone(&leaf), disk], &[8]).unwrap();
        let before = graph.graph_work;
        assert!(graph.qualified_rounded_rectangle(&root).unwrap().is_some());
        assert!(graph.graph_work > before);
        assert_eq!(graph.polygonal_proofs[leaf.id], Some(true));
        let paid = graph.graph_work;
        assert!(graph.proven_original_straight_polygon_union(&leaf).unwrap());
        assert_eq!(graph.graph_work, paid + PROVENANCE_LOOKUP_WORK);

        graph.graph_work = MAX_REGION_GRAPH_WORK;
        assert!(matches!(graph.qualified_rounded_rectangle(&root),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [8]));
        assert_eq!(graph.graph_work, MAX_REGION_GRAPH_WORK);
    }

    #[test]
    fn glyph_mesh_bounds_cannot_certify_original_f1_signed_erosion() {
        use crate::composition_2d::{
            Render2dItem, Render2dOpacity, Render2dPoint, Render2dPrimitive, Render2dResourceId,
            Render2dShapedTextPrimitive,
        };
        // Deliberately attach a small disposable physical triangle to a
        // borrowed shaped-text item. The glyph's actual original font-curve
        // extent is NOT derivable from these vertices or their AABB.
        let text = Render2dItem::new(
            Render2dPrimitive::ShapedText(Render2dShapedTextPrimitive::new(
                Render2dResourceId::new(17).unwrap(),
                Render2dPoint::new(0.0, 0.0).unwrap(),
                Render2dColorRgba8::TRANSPARENT,
            )),
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = builder();
        let leaf = graph.source_from_item(triangle(), &text, &[10, 0]).unwrap();
        assert!(!leaf.certified_source_reach);
        let eroded = graph
            .effect(
                Arc::clone(&leaf),
                Render2dDropShadow::new(0.0, 0.0, 0.2, -10.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[10, 1],
            )
            .unwrap();
        // The physical 4x4 box would appear smaller than the disk, but
        // original font outlines can escape it. This cannot be empty proof.
        assert!(certified_complete_disk_erosion(leaf.envelope, -10.0));
        assert!(!eroded.certified_source_reach);
        assert!(matches!(
            graph.materialize(&eroded),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 10 },
            }) if path == [10, 1]
        ));
        assert_eq!(graph.erosion_work, 0);
        assert!(graph.materialized.get(leaf.id).is_none_or(Option::is_none));
    }

    #[test]
    fn original_region_root_offscreen_proof_requires_every_effect_and_exact_canvas_extent() {
        let mut graph = builder();
        let source = graph.source(triangle(), &[6, 0]).unwrap();
        let descendant =
            Render2dDropShadow::new(0.0, 0.0, 0.1, 1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let curved = graph
            .effect(Arc::clone(&source), descendant, &[6, 1])
            .unwrap();
        let root = graph.union(vec![source, curved], &[6]).unwrap();
        let outside =
            Render2dDropShadow::new(100.0, 0.0, 0.25, -0.5, Render2dColorRgba8::TRANSPARENT)
                .unwrap();
        let reaches =
            Render2dDropShadow::new(0.0, 0.0, 0.25, -0.5, Render2dColorRgba8::TRANSPARENT).unwrap();
        assert!(
            root.all_root_shadows_provably_offscreen(&[outside, outside], [64.0, 64.0], 1.0)
                .unwrap()
        );
        assert!(
            !root
                .all_root_shadows_provably_offscreen(&[outside, reaches], [64.0, 64.0], 1.0)
                .unwrap()
        );
        assert!(
            !root
                .all_root_shadows_provably_offscreen(&[reaches, outside], [64.0, 64.0], 1.0)
                .unwrap()
        );
        assert!(
            !root
                .all_root_shadows_provably_offscreen(&[], [64.0, 64.0], 1.0)
                .unwrap()
        );
        assert!(
            !root
                .all_root_shadows_provably_offscreen(&[outside], [64.0, 64.0], 0.0)
                .unwrap()
        );
        assert!(graph.materialize(&root).is_err());

        // A missing PHYSICAL source cannot erase original F1 reach.
        // This adversarial item retains its positive-area rectangle; authored
        // reach can prove the offset effect offscreen independently of mesh.
        use crate::composition_2d::{
            Render2dAffineTransform, Render2dBrush, Render2dItem, Render2dOpacity,
            Render2dPrimitive, Render2dRect, Render2dShape,
        };
        let original = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(1.0, 1.0, 4.0, 4.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let unrealized = graph
            .source_from_item(shadow::empty_mesh(), &original, &[6, 2])
            .unwrap();
        let original_reach = unrealized
            .envelope
            .expect("original F1 retains positive reach even without a physical realization");
        assert!(
            original_reach[0] <= 1.0
                && original_reach[1] <= 1.0
                && original_reach[2] >= 5.0
                && original_reach[3] >= 5.0
        );
        assert!(
            unrealized
                .all_root_shadows_provably_offscreen(&[outside], [64.0, 64.0], 1.0)
                .unwrap()
        );
    }

    #[test]
    fn whole_root_shadow_set_requires_independent_complete_erosion_proofs() {
        let mut graph = builder();
        let pristine = graph.source(triangle(), &[6, 0]).unwrap();
        let positive =
            Render2dDropShadow::new(0.0, 0.0, 0.2, 1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let symbolic = graph
            .effect(Arc::clone(&pristine), positive, &[6, 1])
            .unwrap();
        let complete = graph.union(vec![pristine, symbolic], &[6]).unwrap();
        let erased =
            Render2dDropShadow::new(2.0, -3.0, 0.3, -8.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let survivor =
            Render2dDropShadow::new(0.0, 0.0, 0.0, -0.5, Render2dColorRgba8::TRANSPARENT).unwrap();
        assert!(complete.all_shadows_provably_empty(&[erased, erased]));
        assert!(!complete.all_shadows_provably_empty(&[erased, survivor]));
        assert!(!complete.all_shadows_provably_empty(&[survivor, erased]));
        assert!(!complete.all_shadows_provably_empty(&[]));
        assert!(
            graph.materialize(&complete).is_err(),
            "the child positive disk remains symbolic, not silently polygonalized"
        );
    }

    #[test]
    fn authored_curve_nested_disk_erosion_can_prove_exact_empty_without_faceting() {
        use crate::composition_2d::{
            Render2dBrush, Render2dItem, Render2dOpacity, Render2dPrimitive, Render2dRect,
            Render2dShape,
        };
        // The borrowed F1 ELLIPSE is not a polygonal rectangle. Its physical
        // mesh is only an auxiliary reach witness. A positive disk remains
        // symbolic and unsupported for materialization; a sufficiently large
        // subsequent negative disk has provably EMPTY original support.
        let ellipse = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::ellipse(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            vec![],
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = builder();
        let mesh = support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let source = graph.source_from_item(mesh, &ellipse, &[7, 0]).unwrap();
        let positive =
            Render2dDropShadow::new(0.0, 0.0, 0.0, 0.25, Render2dColorRgba8::TRANSPARENT).unwrap();
        let expanded = graph
            .effect(Arc::clone(&source), positive, &[7, 1])
            .unwrap();
        assert!(expanded.certified_source_reach);
        let shrink =
            Render2dDropShadow::new(3.0, -2.0, 0.5, -2.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let empty = graph
            .effect(Arc::clone(&expanded), shrink, &[7, 2])
            .unwrap();
        // Source width 2 plus 2*0.25 outward spread = 2.5. An erosion
        // disk r=2 cannot fit in its original-F1 reach in EITHER axis.
        assert!(certified_complete_disk_erosion(expanded.envelope, -2.0));
        let vertices = graph.retained_vertices;
        let erosion_work = graph.erosion_work;
        let result = graph.materialize(&empty).unwrap();
        assert!(
            result.triangles.is_empty(),
            "empty support stays empty under finite blur"
        );
        assert_eq!(graph.retained_vertices, vertices);
        assert_eq!(graph.erosion_work, erosion_work);
        assert!(
            graph
                .materialized
                .get(expanded.id)
                .is_none_or(Option::is_none),
            "the unsupported positive-disk child must not be materialized"
        );
        assert!(
            graph
                .materialized
                .get(source.id)
                .is_none_or(Option::is_none),
            "the disposable F1 mesh is not needed for the exact absence proof"
        );
        // Prove that the success is the geometric erasure, not a broad
        // fallback quietly replacing unsupported curved morphology.
        assert!(matches!(
            graph.materialize(&expanded),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 7 },
            }) if path == [7, 1]
        ));
    }

    #[test]
    fn complete_erosion_certificates_fail_closed_at_thresholds_and_overflow() {
        // A 2x2 region with r=1 has at most a measure-zero survivor.
        // Keep equality and adjacent float uncertainty unresolved.
        assert!(!certified_complete_disk_erosion(
            Some([0.0, 0.0, 2.0, 2.0]),
            -1.0
        ));
        assert!(!certified_complete_disk_erosion(
            Some([0.0, 0.0, 2.0, 2.0]),
            -1.0_f64.next_up(),
        ));
        assert!(certified_complete_disk_erosion(
            Some([0.0, 0.0, 2.0, 2.0]),
            -1.25,
        ));
        assert!(!certified_complete_disk_erosion(
            Some([-f64::MAX, 0.0, f64::MAX, 2.0]),
            -1.0,
        ));
        assert!(!certified_complete_disk_erosion(None, -100.0));
        assert!(!certified_complete_disk_erosion(
            Some([0.0, 0.0, 2.0, 2.0]),
            1.25
        ));

        let mut graph = builder();
        let source = graph.source(triangle(), &[8, 0]).unwrap();
        let borderline = graph
            .effect(
                source,
                Render2dDropShadow::new(0.0, 0.0, 0.0, -0.5, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[8, 1],
            )
            .unwrap();
        assert!(
            !certified_complete_disk_erosion(borderline.envelope, -0.5,),
            "a prior erosion reach cannot be interpreted as a new exact membership claim"
        );
        assert!(
            matches!(
                graph.materialize(&borderline),
                Err(Render2dExecutionError::UnsupportedEntry {
                    path,
                    kind: Render2dUnsupportedContent::Shadows { root_index: 8 },
                }) if path == [8, 1]
            ),
            "a synthetic triangle is not an authored F1 erosion certificate"
        );
    }

    #[test]
    fn nonempty_erosion_requires_original_f1_geometry_not_convex_facets() {
        use crate::composition_2d::{
            Render2dBrush, Render2dItem, Render2dOpacity, Render2dPrimitive, Render2dRect,
            Render2dShape,
        };
        let rect = Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap();
        let original = |shape| {
            Render2dItem::new(
                Render2dPrimitive::Fill {
                    shape,
                    brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
                },
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
            )
        };
        // Both F1 objects receive the SAME convex physical rectangle.
        // The original ellipse is not a license to erode these fake facets.
        let physical = || support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 2.0],
                [0.0, 0.0],
                [2.0, 2.0],
                [0.0, 2.0],
            ],
            bounds: [0.0, 0.0, 2.0, 2.0],
        };
        let rectangle = original(Render2dShape::rect(rect));
        let ellipse = original(Render2dShape::ellipse(rect));
        let erode =
            Render2dDropShadow::new(0.0, 0.0, 0.0, -0.25, Render2dColorRgba8::TRANSPARENT).unwrap();
        let mut graph = builder();
        let curved_source = graph
            .source_from_item(physical(), &ellipse, &[7, 0])
            .unwrap();
        let curved_effect = graph.effect(curved_source, erode, &[7, 1]).unwrap();
        let before = graph.retained_vertices;
        assert!(matches!(
            graph.materialize(&curved_effect),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 7 },
            }) if path == [7, 1]
        ));
        assert_eq!(graph.retained_vertices, before);
        let valid_source = graph
            .source_from_item(physical(), &rectangle, &[8, 0])
            .unwrap();
        let valid_effect = graph.effect(valid_source, erode, &[8, 1]).unwrap();
        assert!(
            !graph
                .materialize(&valid_effect)
                .unwrap()
                .triangles
                .is_empty()
        );
    }

    #[test]
    fn original_f1_curved_affine_and_clip_reach_can_certify_empty_nested_erosion() {
        use crate::composition_2d::{
            Render2dBrush, Render2dClip, Render2dItem, Render2dOpacity, Render2dPrimitive,
            Render2dRect, Render2dShape,
        };
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::ellipse(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = builder();
        let leaf = graph.source_from_item(triangle(), &item, &[9, 0]).unwrap();
        let shear = Render2dAffineTransform::new(1.0, 0.5, 0.25, 1.0, 3.0, -1.0).unwrap();
        let affined = graph.affine(leaf, shear, &[9, 1]).unwrap();
        let clip = [Render2dClip::new(
            Render2dShape::rect(Render2dRect::new(-100.0, -100.0, 200.0, 200.0).unwrap()),
            Render2dAffineTransform::IDENTITY,
        )];
        let clipped = graph.clip(affined, &clip, &[9, 2]).unwrap();
        let effect = graph
            .effect(
                Arc::clone(&clipped),
                Render2dDropShadow::new(1.5, -0.75, 0.1, -6.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[9, 3],
            )
            .unwrap();
        assert!(
            certified_complete_disk_erosion(clipped.envelope, -6.0),
            "the sheared original-F1 reach conservatively proves complete erosion"
        );
        assert!(graph.materialize(&effect).unwrap().triangles.is_empty());
        assert!(
            graph
                .materialized
                .get(clipped.id)
                .is_none_or(Option::is_none),
            "exact complete erosion cannot require or guess the clipped tessellation"
        );
    }

    #[test]
    fn independently_authored_curved_effect_remains_symbolic_and_preserves_pristine_c() {
        let mut builder = builder();
        let caster = builder.source(triangle(), &[0, 0]).unwrap();
        let spread =
            Render2dDropShadow::new(0.0, 0.0, 0.1, 1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let effect = builder
            .effect(Arc::clone(&caster), spread, &[0, 2])
            .unwrap();
        let complete = builder
            .union(vec![Arc::clone(&caster), effect], &[0])
            .unwrap();
        let pristine = builder.materialize(&caster).unwrap();
        assert_eq!(pristine.triangles, triangle().triangles);
        assert!(matches!(
            builder.materialize(&complete),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 0 },
            }) if path == [0, 2]
        ));
        assert_eq!(
            builder.materialize(&caster).unwrap().triangles,
            pristine.triangles
        );
    }

    #[test]
    fn neutral_support_dag_shares_pristine_source_between_distinct_effects() {
        let mut builder = builder();
        let caster = builder.source(triangle(), &[1, 0]).unwrap();
        let effect =
            Render2dDropShadow::new(2.0, 0.0, 0.0, 0.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let e1 = builder
            .effect(Arc::clone(&caster), effect, &[1, 1])
            .unwrap();
        let e2 = builder
            .effect(Arc::clone(&caster), effect, &[1, 2])
            .unwrap();
        let complete = builder
            .union(vec![Arc::clone(&caster), e1, e2], &[1])
            .unwrap();
        let mesh = builder.materialize(&complete).unwrap();
        assert_eq!(mesh.triangles.len(), triangle().triangles.len() * 3);
        assert_eq!(
            builder
                .materialized
                .iter()
                .filter(|entry| entry.is_some())
                .count(),
            4
        );
        assert_eq!(
            builder.materialize(&caster).unwrap().triangles,
            triangle().triangles
        );
    }

    #[test]
    fn sparse_cache_keys_cannot_alias_after_unmaterialized_curved_nodes() {
        let mut builder = builder();
        let caster = builder.source(triangle(), &[2, 0]).unwrap();
        let effect =
            Render2dDropShadow::new(0.0, 0.0, 0.0, 1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        // Symbolic node 2 is never materialized. Node 3 must never inherit
        // an allocation identity or cached geometry from that node.
        let curved = builder
            .effect(Arc::clone(&caster), effect, &[2, 1])
            .unwrap();
        let affine = builder
            .affine(
                Arc::clone(&caster),
                Render2dAffineTransform::translation(1.0, 0.0).unwrap(),
                &[2, 2],
            )
            .unwrap();
        let realized = builder.materialize(&affine).unwrap();
        assert!(Arc::ptr_eq(
            &realized,
            &builder.materialize(&affine).unwrap()
        ));
        assert_eq!(builder.materialized.len(), affine.id + 1);
        assert!(builder.materialized[curved.id].is_none());
        assert_eq!(
            builder
                .materialized
                .iter()
                .filter(|entry| entry.is_some())
                .count(),
            2
        );
    }

    #[test]
    fn leaf_reach_is_certified_from_real_vertices_not_cached_metadata() {
        let mut builder = builder();
        let mut stale = triangle();
        stale.bounds = [0.0; 4];
        let leaf = builder.source(stale, &[7, 0]).unwrap();
        assert_eq!(leaf.envelope, Some([0.0, 0.0, 4.0, 4.0]));
        assert_eq!(
            builder.materialize(&leaf).unwrap().bounds,
            [0.0, 0.0, 4.0, 4.0]
        );
    }

    #[test]
    fn nonfinite_leaf_rejects_without_mutating_graph_admission() {
        let mut builder = builder();
        let mut invalid = triangle();
        invalid.triangles[1][0] = f64::NAN;
        assert!(matches!(
            builder.source(invalid, &[6, 2]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [6, 2]
        ));
        assert_eq!(builder.nodes, 0);
        assert_eq!(builder.retained_vertices, 0);
    }

    #[test]
    fn symbolic_effect_reach_uses_immediate_parent_frame_and_finite_cutoff() {
        let mut builder = builder();
        let caster = builder.source(triangle(), &[5, 0]).unwrap();
        let scale_and_shear = Render2dAffineTransform::new(2.0, 0.0, 0.5, 1.0, 0.0, 0.0).unwrap();
        let in_parent = builder.affine(caster, scale_and_shear, &[5, 1]).unwrap();
        // Projecting the actual triangle vertices avoids the unused AABB
        // corner: its x reach is 8 rather than a box-derived 10.
        assert_eq!(in_parent.envelope, Some([0.0, 0.0, 8.0, 4.0]));
        let effect =
            Render2dDropShadow::new(1.0, -2.0, 0.5, 1.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let curved = builder.effect(in_parent, effect, &[5, 2]).unwrap();
        let extent = curved.envelope.unwrap();
        assert!(extent.iter().all(|v| v.is_finite()));
        assert!(extent[0] <= -1.5 && extent[1] <= -4.5);
        assert!(extent[2] >= 11.5 && extent[3] >= 4.5);
        assert!(matches!(
            builder.materialize(&curved),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 5 },
            }) if path == [5, 2]
        ));
    }

    #[test]
    fn nested_affine_encloses_sub_ulp_parent_translation() {
        let mut builder = builder();
        let first = builder.source(triangle(), &[6, 0]).unwrap();
        let second = builder.source(triangle(), &[6, 3]).unwrap();
        let combined = builder.union(vec![first, second], &[6, 1]).unwrap();
        assert!(matches!(&combined.kind, Kind::Union(_)));
        let tiny = 2.0_f64.powi(-54);
        let motion = Render2dAffineTransform::new(1.0, 0.0, 0.0, 1.0, tiny, 0.0).unwrap();
        let moved = builder.affine(combined, motion, &[6, 2]).unwrap();
        let bounds = moved.envelope.unwrap();
        assert!(
            bounds[2] > 4.0,
            "nearest-rounded 4 + tiny would incorrectly remain 4"
        );
        assert!(bounds[0] <= tiny);
        assert!(bounds[1] <= 0.0 && bounds[3] >= 4.0);
    }

    #[test]
    fn nested_effect_encloses_sub_ulp_offset_and_halo() {
        let mut builder = builder();
        let src = builder.source(triangle(), &[7, 0]).unwrap();
        let tiny = 2.0_f64.powi(-54);
        let offset =
            Render2dDropShadow::new(tiny, 0.0, 0.0, 0.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let moved = builder.effect(src, offset, &[7, 1]).unwrap();
        let bounds = moved.envelope.unwrap();
        assert!(
            bounds[2] > 4.0,
            "outward effect reach must include sub-ULP translation"
        );
        assert!(bounds[0] <= tiny);
        let blur =
            Render2dDropShadow::new(0.0, 0.0, tiny, 0.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        let widened = builder.effect(moved, blur, &[7, 2]).unwrap();
        let reach = widened.envelope.unwrap();
        assert!(reach[0] < bounds[0] && reach[2] > bounds[2]);
    }

    #[test]
    fn nested_axis_reflection_preserves_extreme_finite_reach() {
        let bounds = [0.0, 0.0, f64::MAX, 1.0];
        let reflected = Render2dAffineTransform::new(-1.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap();
        assert_eq!(
            nested_affine_envelope(bounds, reflected, &[8]).unwrap(),
            [-f64::MAX, 0.0, 0.0, 1.0]
        );
        let reflected_y = Render2dAffineTransform::new(1.0, 0.0, 0.0, -1.0, 0.0, 0.0).unwrap();
        assert_eq!(
            nested_affine_envelope(bounds, reflected_y, &[8]).unwrap(),
            [0.0, -1.0, f64::MAX, 0.0]
        );
    }

    #[test]
    fn nested_identity_retains_representable_extreme_extent() {
        let bounds = [0.0, 0.0, f64::MAX, 1.0];
        assert_eq!(
            nested_affine_envelope(bounds, Render2dAffineTransform::IDENTITY, &[8]).unwrap(),
            bounds
        );
        let zero =
            Render2dDropShadow::new(0.0, 0.0, 0.0, 0.0, Render2dColorRgba8::TRANSPARENT).unwrap();
        assert_eq!(nested_effect_envelope(bounds, zero, &[8]).unwrap(), bounds);
    }

    #[test]
    fn real_triangle_affine_reach_does_not_overflow_at_unoccupied_box_corner() {
        let mut builder = builder();
        let leaf = builder
            .source(
                support::NeutralMesh {
                    units_per_parent_logical_unit: 1.0,
                    triangles: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
                    bounds: [0.0, 0.0, 1.0, 1.0],
                },
                &[9, 0],
            )
            .unwrap();
        // The artificial AABB corner (1,1) gives MAX+MAX = infinity.
        // None of the triangle's three REAL vertices has that coordinate.
        let affine = Render2dAffineTransform::new(f64::MAX, 0.0, f64::MAX, 1.0, 0.0, 0.0).unwrap();
        let before_affine = builder.graph_work;
        let projected = builder.affine(leaf, affine, &[9, 1]).unwrap();
        assert_eq!(projected.envelope, Some([0.0, 0.0, f64::MAX, 1.0]));
        assert_eq!(
            builder.graph_work - before_affine,
            64 + REGION_NODE_WORK + 4 + 3 * AFFINE_REACH_VERTEX_WORK
        );
        let realized = builder.materialize(&projected).unwrap();
        assert_eq!(realized.bounds, [0.0, 0.0, f64::MAX, 1.0]);
        assert_eq!(realized.triangles.len(), 3);
    }

    #[test]
    fn affine_vertex_budget_rejects_without_mutating_invocation_admission() {
        let mut builder = builder();
        let leaf = builder.source(triangle(), &[4, 0]).unwrap();
        builder.graph_work = MAX_REGION_GRAPH_WORK - 2;
        let before_nodes = builder.nodes;
        assert!(matches!(
            builder.affine(leaf, Render2dAffineTransform::IDENTITY, &[4, 1]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [4, 1]
        ));
        assert_eq!(builder.nodes, before_nodes);
        assert_eq!(builder.graph_work, MAX_REGION_GRAPH_WORK - 2);
    }

    #[test]
    fn composite_reach_rejects_nonfinite_transform_before_materializing() {
        let mut builder = builder();
        let caster = builder.source(triangle(), &[8, 0]).unwrap();
        let extreme = Render2dAffineTransform::new(f64::MAX, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap();
        assert!(matches!(
            builder.affine(caster, extreme, &[8, 4]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path),
                ..
            }) if path == [8, 4]
        ));
        assert!(builder.materialized.is_empty());
    }

    #[test]
    fn neutral_region_depth_rejects_with_authored_path_before_materialization() {
        let mut builder = builder();
        let mut region = builder.source(triangle(), &[3]).unwrap();
        let mut result = None;
        for _ in 0..MAX_REGION_DEPTH {
            match builder.affine(
                Arc::clone(&region),
                Render2dAffineTransform::translation(1.0, 0.0).unwrap(),
                &[3, 5],
            ) {
                Ok(next) => region = next,
                Err(error) => {
                    result = Some(error);
                    break;
                }
            }
        }
        assert!(matches!(
            result,
            Some(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path),
                ..
            }) if path == [3, 5]
        ));
        assert!(builder.materialized.is_empty());
    }
}
