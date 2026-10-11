//! Immutable renderer-private neutral SUPPORT expressions for ordinary F3F.
//!
//! These are prepared from the ONE borrowed F1 painter, not a second semantic
//! scene. Curved effects remain symbolic across parent frames. Polygonal
//! realization is merely one *certified* disposable query path; an unsupported
//! curved consumer fails before target mutation instead of inventing triangles
//! from bounds, alpha or a sampled distance field.
use crate::composition_2d::{Render2dAffineTransform, Render2dClip, Render2dDropShadow};
use crate::execution_2d::{
    Render2dExecutionError, Render2dSampleSpaceError, Render2dUnsupportedContent,
};
use crate::runtime::execution_2d::{shadow, support};
use std::sync::Arc;

const MAX_REGION_NODES: usize = 65_536;
const MAX_REGION_DEPTH: usize = 512;
const MAX_REGION_PATH_UNITS: usize = 1_048_576;
const MAX_REGION_AFFINE_VERTEX_WORK: usize = 16_777_216;

pub(super) struct NeutralRegion<'a> {
    kind: Kind<'a>,
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
fn region_envelope(
    kind: &Kind<'_>,
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
                Some(mesh.bounds.map(|coordinate| coordinate / scale))
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
                affine_leaf_envelope(mesh, *to_parent, path)?
            } else {
                source
                    .envelope
                    .map(|bounds| shadow::transform_envelope(bounds, *to_parent, path))
                    .transpose()?
            }
        }
        Kind::Clip { source, .. } => source.envelope,
        Kind::Effect {
            source,
            shadow: effect,
        } => source.envelope.map(|bounds| {
            // Spread is a Euclidean DISK while sigma>0 has a finite axis-aligned
            // square envelope. The sum is conservative for support reach, not
            // a square substituted as the positive disk's actual membership.
            let pad = effect.spread().max(0.0) + 3.0 * effect.sigma();
            [
                bounds[0] + effect.offset_x() - pad,
                bounds[1] + effect.offset_y() - pad,
                bounds[2] + effect.offset_x() + pad,
                bounds[3] + effect.offset_y() + pad,
            ]
        }),
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

/// One invocation-owned bounded graph and memoized polygonal realization.
/// Memo keys are monotonic graph IDs, NEVER F1 semantic or cache IDs.
pub(super) struct NeutralRegionBuilder {
    nodes: usize,
    retained_path_units: usize,
    retained_vertices: usize,
    affine_vertex_work: usize,
    pub(super) clip_work: usize,
    erosion_work: usize,
    // Dense, fallibly allocated by monotonic node ID; no implicit per-entry
    // tree allocation and no aliasing if a prior node is dropped.
    materialized: Vec<Option<Arc<support::NeutralMesh>>>,
    scales: shadow::NeutralPreparationScale,
    max_buffer_bytes: u64,
}

impl NeutralRegionBuilder {
    pub(super) fn new(scales: shadow::NeutralPreparationScale, max_buffer_bytes: u64) -> Self {
        Self {
            nodes: 0,
            retained_path_units: 0,
            retained_vertices: 0,
            affine_vertex_work: 0,
            clip_work: 0,
            erosion_work: 0,
            materialized: Vec::new(),
            scales,
            max_buffer_bytes,
        }
    }

    fn node<'a>(
        &mut self,
        kind: Kind<'a>,
        path: &[usize],
        depth: usize,
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
        let visits = match &kind {
            Kind::Affine { source, .. } => match &source.kind {
                Kind::Source(mesh) => mesh.triangles.len(),
                _ => 0,
            },
            _ => 0,
        };
        let affine_work = self
            .affine_vertex_work
            .checked_add(visits)
            .ok_or_else(|| resource(path, "neutral affine reach work overflow"))?;
        if affine_work > MAX_REGION_AFFINE_VERTEX_WORK {
            return Err(resource(
                path,
                "neutral affine reach exceeds bounded whole-invocation work",
            ));
        }
        // Validate every finite support reach before committing either graph
        // or work admission. Curved region bounds remain reach-only.
        let envelope = region_envelope(&kind, path)?;
        self.nodes = next;
        self.retained_path_units = paths;
        self.affine_vertex_work = affine_work;
        Ok(Arc::new(NeutralRegion {
            kind,
            path: path.to_vec(),
            id: next,
            depth,
            envelope,
        }))
    }

    pub(super) fn source<'a>(
        &mut self,
        mut mesh: support::NeutralMesh,
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
        let node = self.node(Kind::Source(Arc::new(mesh)), path, 1)?;
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
        let depth = children.iter().map(|node| node.depth).max().unwrap_or(0) + 1;
        self.node(Kind::Union(children), path, depth)
    }

    pub(super) fn affine<'a>(
        &mut self,
        source: Arc<NeutralRegion<'a>>,
        to_parent: Render2dAffineTransform,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        let depth = source.depth + 1;
        self.node(Kind::Affine { source, to_parent }, path, depth)
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
        self.node(Kind::Clip { source, clips }, path, depth)
    }

    pub(super) fn effect<'a>(
        &mut self,
        source: Arc<NeutralRegion<'a>>,
        shadow: Render2dDropShadow,
        path: &[usize],
    ) -> Result<Arc<NeutralRegion<'a>>, Render2dExecutionError> {
        let depth = source.depth + 1;
        self.node(Kind::Effect { source, shadow }, path, depth)
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
                    assert_eq!(region.point_certificate([9.5, 9.5]), PointCertificate::Inside);
                    assert_eq!(region.point_certificate([8.5, 8.5]), PointCertificate::Outside);
                    assert_eq!(region.point_certificate([15.0, 15.0]), PointCertificate::Inside);
                    assert_eq!(region.point_certificate([9.0, 9.0]), PointCertificate::Outside);
                    assert_eq!(region.point_certificate([9.0, 15.0]), PointCertificate::Unknown);
                }
            },
        );
        assert!(inspected_root, "the actual root caster must be inspected before materialization");
        assert!(matches!(
            result,
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 0 },
            }) if path == [0, 0]
        ));
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
        assert_eq!(curved.envelope, Some([-1.5, -4.5, 11.5, 4.5]));
        assert!(matches!(
            builder.materialize(&curved),
            Err(Render2dExecutionError::UnsupportedEntry {
                path,
                kind: Render2dUnsupportedContent::Shadows { root_index: 5 },
            }) if path == [5, 2]
        ));
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
        assert_eq!(builder.affine_vertex_work, 3);
        let realized = builder.materialize(&projected).unwrap();
        assert_eq!(realized.bounds, [0.0, 0.0, f64::MAX, 1.0]);
        assert_eq!(realized.triangles.len(), 3);
    }

    #[test]
    fn affine_vertex_budget_rejects_without_mutating_invocation_admission() {
        let mut builder = builder();
        let leaf = builder.source(triangle(), &[4, 0]).unwrap();
        builder.affine_vertex_work = MAX_REGION_AFFINE_VERTEX_WORK - 2;
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
        assert_eq!(
            builder.affine_vertex_work,
            MAX_REGION_AFFINE_VERTEX_WORK - 2
        );
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
