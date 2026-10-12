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
use std::sync::Arc;

mod interval;
mod source_envelope;

const MAX_REGION_NODES: usize = 65_536;
const MAX_REGION_DEPTH: usize = 512;
const MAX_REGION_PATH_UNITS: usize = 1_048_576;
const MAX_REGION_GRAPH_WORK: usize = 16_777_216;

pub(super) struct NeutralRegion<'a> {
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
}

enum Kind<'a> {
    Union(Vec<Arc<NeutralRegion<'a>>>),
    Source(Arc<support::NeutralMesh>),
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
            if mesh.triangles.is_empty() {
                None
            } else {
                let scale = mesh.units_per_parent_logical_unit;
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(precision(
                        path,
                        "neutral-region source scale is unrepresentable",
                    ));
                }
                let physical = mesh.bounds.map(|coordinate| coordinate / scale);
                if let Some(item) = semantic_item
                    && let Some(authored) = source_envelope::bounds(item, None, path)?
                {
                    Some([
                        physical[0].min(authored[0]),
                        physical[1].min(authored[1]),
                        physical[2].max(authored[2]),
                        physical[3].max(authored[3]),
                    ])
                } else {
                    Some(physical)
                }
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
                let physical = affine_leaf_envelope(mesh, *to_parent, path)?;
                if let Some(item) = source.semantic_item
                    && let Some(authored) = source_envelope::bounds(item, Some(*to_parent), path)?
                {
                    Some(match physical {
                        Some(bounds) => [
                            bounds[0].min(authored[0]),
                            bounds[1].min(authored[1]),
                            bounds[2].max(authored[2]),
                            bounds[3].max(authored[3]),
                        ],
                        None => authored,
                    })
                } else {
                    physical
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
fn certified_complete_disk_erosion(
    bounds: Option<[f64; 4]>,
    spread: f64,
) -> bool {
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

/// One invocation-owned bounded graph and memoized polygonal realization.
/// Memo keys are monotonic graph IDs, NEVER F1 semantic or cache IDs.
pub(super) struct NeutralRegionBuilder {
    nodes: usize,
    retained_path_units: usize,
    retained_vertices: usize,
    /// One invocation-wide original reach AND stable neutral-union scan budget.
    graph_work: usize,
    pub(super) clip_work: usize,
    erosion_work: usize,
    // Dense, fallibly allocated by monotonic node ID; no implicit per-entry
    // tree allocation and no aliasing if a prior node is dropped.
    materialized: Vec<Option<Arc<support::NeutralMesh>>>,
    // One bounded transient stamp table for idempotent support union.
    // Stable first-source order is preserved for owner-path diagnostics.
    union_seen: Vec<usize>,
    union_epoch: usize,
    scales: shadow::NeutralPreparationScale,
    max_buffer_bytes: u64,
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
        Self {
            nodes: 0,
            retained_path_units: 0,
            retained_vertices: 0,
            graph_work: 0,
            clip_work: 0,
            erosion_work: 0,
            materialized: Vec::new(),
            union_seen: Vec::new(),
            union_epoch: 0,
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
                Kind::Source(mesh) => mesh.triangles.len(),
                _ => 0,
            },
            _ => 0,
        };
        let original_visits = match &kind {
            Kind::Source(_) => semantic_item.map_or(0, source_envelope::control_visits),
            Kind::Affine { source, .. } => source
                .semantic_item
                .map_or(0, source_envelope::control_visits),
            _ => 0,
        };
        let visits = mesh_visits
            .checked_add(original_visits)
            .ok_or_else(|| resource(path, "original F1 source-reach work overflow"))?;
        let affine_work = self
            .graph_work
            .checked_add(visits)
            .ok_or_else(|| resource(path, "neutral affine reach work overflow"))?;
        if affine_work > MAX_REGION_GRAPH_WORK {
            return Err(resource(
                path,
                "neutral affine reach exceeds bounded whole-invocation work",
            ));
        }
        // Validate every finite support reach before committing either graph
        // or work admission. Curved region bounds remain reach-only.
        let envelope = region_envelope(&kind, semantic_item, path)?;
        self.nodes = next;
        self.retained_path_units = paths;
        self.graph_work = affine_work;
        Ok(Arc::new(NeutralRegion {
            kind,
            semantic_item,
            path: path.to_vec(),
            id: next,
            depth,
            envelope,
        }))
    }

    /// Synthetic test fixtures have no original F1 item to borrow. They may
    /// certify physical mesh geometry only, not arbitrary authoring semantics.
    #[cfg(test)]
    pub(super) fn source<'a>(
        &mut self,
        mesh: support::NeutralMesh,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        self.source_impl(mesh, None, path)
    }

    /// Preserve the original borrowed F1 authoring item beside disposable
    /// tessellation without cloning it or constructing a second semantic tree.
    pub(super) fn source_from_item<'a>(
        &mut self,
        mesh: support::NeutralMesh,
        item: &'a Render2dItem,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        self.source_impl(mesh, Some(item), path)
    }

    fn source_impl<'a>(
        &mut self,
        mut mesh: support::NeutralMesh,
        semantic_item: Option<&'a Render2dItem>,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
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
        let node = self.node(Kind::Source(Arc::new(mesh)), path, 1, semantic_item)?;
        self.retained_vertices = retained;
        Ok(node)
    }

    pub(super) fn union<'a>(
        &mut self,
        mut children: Vec<Arc<NeutralRegion<'a>>>,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        if children.len() == 1 {
            return Ok(children.pop().expect("single union member"));
        }
        if children.len() > 1 {
            // Exact set-union idempotence, not paint-order deduplication.
            // Keep the FIRST occurrence to preserve authored error paths.
            // A bounded stamp table avoids the O(n^2) duplicate search and
            // O(total DAG nodes) repeated allocation per group.
            let next_work = self
                .graph_work
                .checked_add(children.len())
                .ok_or_else(|| resource(path, "neutral union scan work overflow"))?;
            if next_work > MAX_REGION_GRAPH_WORK {
                return Err(resource(
                    path,
                    "neutral union scan exceeds whole-invocation work",
                ));
            }
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
                self.union_seen
                    .try_reserve_exact(needed - self.union_seen.len())
                    .map_err(|_| resource(path, "neutral union stamp allocation failed"))?;
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
            self.graph_work = next_work;
            return Ok(result);
        }
        self.node(Kind::Union(children), path, 1, None)
    }

    pub(super) fn affine<'a>(
        &mut self,
        source: Arc<NeutralRegion<'a>>,
        to_parent: Render2dAffineTransform,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
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
    fn exact_original_rectangle_source(mesh: &support::NeutralMesh, item: &Render2dItem) -> bool {
        let rect = match item.primitive() {
            crate::composition_2d::Render2dPrimitive::Fill { shape, .. } => shape.as_rect(),
            crate::composition_2d::Render2dPrimitive::Image(image) => {
                if image.patches().len() != 1 {
                    return false;
                }
                Some(image.patches()[0].destination())
            }
            crate::composition_2d::Render2dPrimitive::Stroke { .. }
            | crate::composition_2d::Render2dPrimitive::ShapedText(_) => None,
        };
        let Some(rect) = rect else {
            return false;
        };
        if rect.is_empty() || !item.clips().is_empty() {
            return false;
        }
        let Some(actual) = support::certified_axis_aligned_rectangle(&mesh.triangles) else {
            return false;
        };
        let [a, b, c, d, tx, ty] = item.local_to_parent().components();
        let x0 = rect.x();
        let x1 = rect.x() + rect.width();
        let y0 = rect.y();
        let y1 = rect.y() + rect.height();
        let mut seen = 0_u8;
        for (x, y) in [(x0, y0), (x1, y0), (x1, y1), (x0, y1)] {
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

    /// This narrow polygon-to-rounded-disk specialization may only promote
    /// a rectangle proven from original F1 polygonal geometry. A disposable
    /// Lyon mesh alone cannot certify an ellipse, rounded corner, Bézier,
    /// stroke or curved clip merely because its triangles happen to form a
    /// bounding rectangle. Resolved image destinations and plain rectangular
    /// fills remain exact polygonal sources, independent of color alpha.
    fn polygonal_source_proven<'a>(
        region: &NeutralRegion<'a>,
        remaining_visits: &mut usize,
        memo: &mut [Option<bool>],
    ) -> Result<bool, Render2dExecutionError> {
        // Node IDs are immutable, unique within this builder and bounded.
        // Every retained Arc branch must reuse the same source verdict:
        // the 65_536-node budget cannot permit 2^depth repeated work.
        if let Some(cached) = memo.get(region.id).copied().flatten() {
            return Ok(cached);
        }
        *remaining_visits = remaining_visits.checked_sub(1).ok_or_else(|| {
            resource(
                &region.path,
                "original F1 provenance query exceeds work budget",
            )
        })?;
        let result = match &region.kind {
            Kind::Source(mesh) => region
                .semantic_item
                .is_some_and(|item| Self::exact_original_rectangle_source(mesh, item)),
            Kind::Union(children) => {
                let mut proven = true;
                for child in children {
                    if !Self::polygonal_source_proven(child, remaining_visits, memo)? {
                        proven = false;
                        break;
                    }
                }
                proven
            }
            Kind::Affine { source, .. } => {
                Self::polygonal_source_proven(source, remaining_visits, memo)?
            }
            Kind::Clip { source, clips } => {
                // A rectangular *shape type* is not proof that its separate
                // clip_to_parent affine and physical intersection preserve
                // the original F1 rectangle. Only the identity/no-op clip
                // wrapper qualifies for this exact rounded-caster shortcut.
                clips.is_empty() && Self::polygonal_source_proven(source, remaining_visits, memo)?
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

    /// Recognize only the exact existing F1 DAG: C union (C ⊕ disk(r)).
    /// The child source must be a whole, certified axis-aligned rectangle,
    /// with identity frames, no clip, zero blur and no effect translation.
    /// All other curved DAGs remain unsupported rather than faceted.
    pub(super) fn qualified_rounded_rectangle<'a>(
        &mut self,
        root: &Arc<NeutralRegion<'a>>,
    ) -> Result<Option<CertifiedRoundedRectangle>, Render2dExecutionError> {
        fn identity<'a, 'b>(mut node: &'b Arc<NeutralRegion<'a>>) -> &'b Arc<NeutralRegion<'a>> {
            loop {
                match &node.kind {
                    Kind::Affine { source, to_parent }
                        if to_parent.components()
                            == Render2dAffineTransform::IDENTITY.components() =>
                    {
                        node = source;
                    }
                    Kind::Clip { source, clips: [] } => node = source,
                    Kind::Union(children) if children.len() == 1 => node = &children[0],
                    _ => return node,
                }
            }
        }
        let Kind::Union(children) = &identity(root).kind else {
            return Ok(None);
        };
        if children.len() != 2 {
            return Ok(None);
        }
        let (original, curved) = (&children[0], identity(&children[1]));
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
        // One fallibly admitted dense per-query cache, keyed by unique
        // immutable F1 graph ID. Repeated Arc edges never re-evaluate a
        // shared subtree; no exponential proof walk or global scene cache.
        let cache_entries = original
            .id
            .checked_add(1)
            .ok_or_else(|| resource(&original.path, "original F1 proof cache size overflow"))?;
        let cache_bytes = cache_entries
            .checked_mul(std::mem::size_of::<Option<bool>>())
            .ok_or_else(|| resource(&original.path, "original F1 proof cache bytes overflow"))?;
        if cache_entries > MAX_REGION_NODES + 1
            || u64::try_from(cache_bytes)
                .ok()
                .is_none_or(|bytes| bytes > self.max_buffer_bytes)
        {
            return Err(resource(
                &original.path,
                "original F1 proof cache exceeds bounded memory",
            ));
        }
        let mut memo = Vec::<Option<bool>>::new();
        memo.try_reserve_exact(cache_entries)
            .map_err(|_| resource(&original.path, "original F1 proof cache allocation failed"))?;
        memo.resize(cache_entries, None);
        let mut remaining_visits = MAX_REGION_NODES;
        if !Self::polygonal_source_proven(original, &mut remaining_visits, &mut memo)? {
            return Ok(None);
        }
        let rectangle = self.materialize(original)?;
        if support::certified_axis_aligned_rectangle(&rectangle.triangles).is_none() {
            return Ok(None);
        }
        Ok(Some(CertifiedRoundedRectangle {
            rectangle,
            radius: shadow.spread(),
            curved_owner: curved.path.clone(),
        }))
    }

    /// Cached polygons share the one original C. IDs remain unique even when
    /// a completed graph subtree is discarded before its siblings are built.
    /// Every new cached materialization is charged once, including copies.
    pub(super) fn materialize<'a>(
        &mut self,
        region: &Arc<NeutralRegion<'a>>,
    ) -> Result<Arc<support::NeutralMesh>, Render2dExecutionError> {
        if let Some(Some(existing)) = self.materialized.get(region.id) {
            return Ok(Arc::clone(existing));
        }
        let path = region.path.as_slice();
        let mesh = match &region.kind {
            Kind::Source(mesh) => Arc::clone(mesh),
            Kind::Union(children) => {
                let mut output = shadow::empty_mesh();
                for child in children {
                    let mesh = self.materialize(child)?;
                    shadow::charge_vertices(
                        &mut self.retained_vertices,
                        mesh.triangles.len(),
                        path,
                    )?;
                    shadow::append(&mut output, &mesh, Render2dAffineTransform::IDENTITY, path)?;
                }
                Arc::new(output)
            }
            Kind::Affine { source, to_parent } => {
                let mesh = self.materialize(source)?;
                shadow::charge_vertices(&mut self.retained_vertices, mesh.triangles.len(), path)?;
                let mut output = shadow::empty_mesh();
                shadow::append(&mut output, &mesh, *to_parent, path)?;
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
                    &mut self.clip_work,
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
                if certified_complete_disk_erosion(source.envelope, effect.spread()) {
                    Arc::new(shadow::empty_mesh())
                } else {
                let mesh = self.materialize(source)?;
                // A positive disk is retained as a precise expression but
                // cannot be *polygonalized* as finite triangles. A consuming
                // ancestor needs certified curved-region evaluation.
                if effect.spread() > 0.0 {
                    return Err(unsupported(path));
                }
                let erased = if effect.spread() < 0.0 {
                    support::certified_convex_erosion_mesh(
                        &mesh,
                        -effect.spread(),
                        path,
                        &mut self.erosion_work,
                    )?
                    .ok_or_else(|| unsupported(path))?
                } else {
                    shadow::empty_mesh()
                };
                let caster: &support::NeutralMesh = if effect.spread() < 0.0 {
                    &erased
                } else {
                    mesh.as_ref()
                };
                let offset =
                    Render2dAffineTransform::translation(effect.offset_x(), effect.offset_y())
                        .map_err(|_| {
                            precision(path, "neutral effect offset cannot be represented")
                        })?;
                shadow::charge_vertices(&mut self.retained_vertices, caster.triangles.len(), path)?;
                let mut shifted = shadow::empty_mesh();
                shadow::append(&mut shifted, caster, offset, path)?;
                if effect.sigma() > 0.0 && !shifted.triangles.is_empty() {
                    let square =
                        shadow::gaussian_square_support(&shifted, effect.sigma() * 3.0, path)?;
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
        // Every node ID is monotonic and bounded by MAX_REGION_NODES.
        // Geometric Vec growth amortizes allocation across node inserts;
        // each growth uses fallible try_reserve BEFORE changing cache state.
        // The maximum cache capacity is bounded by at most twice the
        // admitted index range, with no infallible per-entry tree allocation.
        let needed = region
            .id
            .checked_add(1)
            .ok_or_else(|| resource(path, "neutral materialization cache size overflow"))?;
        if needed > MAX_REGION_NODES + 1 {
            return Err(resource(
                path,
                "neutral materialization cache exceeds node admission",
            ));
        }
        if needed > self.materialized.len() {
            self.materialized
                .try_reserve(needed - self.materialized.len())
                .map_err(|_| resource(path, "neutral materialization cache allocation failed"))?;
            self.materialized.resize_with(needed, || None);
        }
        self.materialized[region.id] = Some(Arc::clone(&mesh));
        Ok(mesh)
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

#[cfg(test)]
mod certified_cells;

#[cfg(test)]
mod tests {
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
        let plan = shadow::scene::analyze(&composition).unwrap();
        let bindings = Render2dResourceBindings::default();
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
        let Some(shadow::NeutralCaster::RoundedRectangle(qualified)) = sources.get(&0) else {
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
        // Four real entries plus 24 repeated references, all admitted.
        assert_eq!(graph.graph_work, 28);

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
        assert_eq!(graph.graph_work, committed);
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
        // One unique leaf and 12 unique union nodes: 13 semantic checks,
        // NOT 8191 visits through duplicated immutable Arc branches.
        let mut memo = [None; 14];
        let mut visits = 13;
        assert!(
            NeutralRegionBuilder::polygonal_source_proven(&shared, &mut visits, &mut memo,)
                .unwrap()
        );
        assert_eq!(visits, 0);
        assert!(memo[1..14].iter().all(|entry| *entry == Some(true)));
        // Even a cached query with an exhausted budget remains deterministic.
        assert!(
            NeutralRegionBuilder::polygonal_source_proven(&shared, &mut visits, &mut memo,)
                .unwrap()
        );
        let mut cold = [None; 14];
        let mut insufficient = 12;
        assert!(matches!(
            NeutralRegionBuilder::polygonal_source_proven(
                &shared, &mut insufficient, &mut cold,
            ),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            }) if path.first() == Some(&2)
        ));
        assert_eq!(insufficient, 0);
        assert!(builder.materialized.is_empty());
    }

    #[test]
    fn authored_curve_nested_disk_erosion_can_prove_exact_empty_without_faceting() {
        use crate::composition_2d::{
            Render2dBrush, Render2dItem, Render2dOpacity, Render2dPrimitive,
            Render2dRect, Render2dShape,
        };
        // The borrowed F1 ELLIPSE is not a polygonal rectangle. Its physical
        // mesh is only an auxiliary reach witness. A positive disk remains
        // symbolic and unsupported for materialization; a sufficiently large
        // subsequent negative disk has provably EMPTY original support.
        let ellipse = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::ellipse(
                    Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap(),
                ),
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
        let positive = Render2dDropShadow::new(
            0.0, 0.0, 0.0, 0.25, Render2dColorRgba8::TRANSPARENT,
        )
        .unwrap();
        let expanded = graph.effect(Arc::clone(&source), positive, &[7, 1]).unwrap();
        let shrink = Render2dDropShadow::new(
            3.0, -2.0, 0.5, -2.0, Render2dColorRgba8::TRANSPARENT,
        )
        .unwrap();
        let empty = graph.effect(Arc::clone(&expanded), shrink, &[7, 2]).unwrap();
        // Source width 2 plus 2*0.25 outward spread = 2.5. An erosion
        // disk r=2 cannot fit in its original-F1 reach in EITHER axis.
        assert!(certified_complete_disk_erosion(expanded.envelope, -2.0));
        let vertices = graph.retained_vertices;
        let erosion_work = graph.erosion_work;
        let result = graph.materialize(&empty).unwrap();
        assert!(result.triangles.is_empty(), "empty support stays empty under finite blur");
        assert_eq!(graph.retained_vertices, vertices);
        assert_eq!(graph.erosion_work, erosion_work);
        assert!(
            graph.materialized.get(expanded.id).is_none_or(Option::is_none),
            "the unsupported positive-disk child must not be materialized"
        );
        assert!(
            graph.materialized.get(source.id).is_none_or(Option::is_none),
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
        assert!(!certified_complete_disk_erosion(Some([0.0, 0.0, 2.0, 2.0]), -1.0));
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
        assert!(!certified_complete_disk_erosion(Some([0.0, 0.0, 2.0, 2.0]), 1.25));

        let mut graph = builder();
        let source = graph.source(triangle(), &[8, 0]).unwrap();
        let borderline = graph.effect(
            source,
            Render2dDropShadow::new(
                0.0, 0.0, 0.0, -0.5, Render2dColorRgba8::TRANSPARENT,
            ).unwrap(),
            &[8, 1],
        ).unwrap();
        assert!(
            !certified_complete_disk_erosion(
                borderline.envelope,
                -0.5,
            ),
            "a prior erosion reach cannot be interpreted as a new exact membership claim"
        );
        let nonempty = graph.materialize(&borderline).unwrap();
        assert!(
            !nonempty.triangles.is_empty(),
            "uncertain enclosure must take the existing actual geometric erosion path"
        );
    }

    #[test]
    fn original_f1_curved_affine_and_clip_reach_can_certify_empty_nested_erosion() {
        use crate::composition_2d::{
            Render2dBrush, Render2dClip, Render2dItem, Render2dOpacity,
            Render2dPrimitive, Render2dRect, Render2dShape,
        };
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::ellipse(
                    Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap(),
                ),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = builder();
        let leaf = graph.source_from_item(triangle(), &item, &[9, 0]).unwrap();
        let shear = Render2dAffineTransform::new(
            1.0, 0.5, 0.25, 1.0, 3.0, -1.0,
        ).unwrap();
        let affined = graph.affine(leaf, shear, &[9, 1]).unwrap();
        let clip = [Render2dClip::new(
            Render2dShape::rect(
                Render2dRect::new(-100.0, -100.0, 200.0, 200.0).unwrap(),
            ),
            Render2dAffineTransform::IDENTITY,
        )];
        let clipped = graph.clip(affined, &clip, &[9, 2]).unwrap();
        let effect = graph.effect(
            Arc::clone(&clipped),
            Render2dDropShadow::new(
                1.5, -0.75, 0.1, -6.0, Render2dColorRgba8::TRANSPARENT,
            ).unwrap(),
            &[9, 3],
        ).unwrap();
        assert!(
            certified_complete_disk_erosion(clipped.envelope, -6.0),
            "the sheared original-F1 reach conservatively proves complete erosion"
        );
        assert!(graph.materialize(&effect).unwrap().triangles.is_empty());
        assert!(
            graph.materialized.get(clipped.id).is_none_or(Option::is_none),
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
                Render2dAffineTransform::IDENTITY,
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
        let projected = builder.affine(leaf, affine, &[9, 1]).unwrap();
        assert_eq!(projected.envelope, Some([0.0, 0.0, f64::MAX, 1.0]));
        assert_eq!(builder.graph_work, 3);
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
                Render2dAffineTransform::IDENTITY,
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
