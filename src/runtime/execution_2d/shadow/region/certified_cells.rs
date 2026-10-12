//! Test-only feasibility of conservative cell classification over the actual
//! F1 neutral-region DAG. This does NOT admit a new production shadow class.
//!
//! Only entire, certified physical rectangles are treated as known sources.
//! Tessellated curves are deliberately Unknown: a physical triangle inventory
//! is not a certificate of exact F1 source geometry. A production evaluator
//! must separately establish source enclosures and Gaussian *coverage* bounds.
//! A certificate concerns SUPPORT membership, not alpha or Gaussian weights.

use super::interval::Interval;
use super::*;
use crate::composition_2d::{
    Render2dClip, Render2dItem, Render2dPath, Render2dPathCommand, Render2dPrimitive,
    Render2dRect, Render2dShape,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CellCertificate {
    Inside,
    Outside,
    Unknown,
}

// All calculated bounds are outward-rounded. Any arithmetic uncertainty,
// nonfinite result or singular transform yields Unknown, never fake geometry.

#[derive(Clone, Copy)]
struct Cell {
    x: Interval,
    y: Interval,
}

impl Cell {
    fn new([x0, y0, x1, y1]: [f64; 4]) -> Option<Self> {
        Some(Self {
            x: Interval::new(x0, x1)?,
            y: Interval::new(y0, y1)?,
        })
    }

    fn shift(self, x: f64, y: f64) -> Option<Self> {
        Some(Self {
            x: self.x.sub(Interval::scalar(x)?)?,
            y: self.y.sub(Interval::scalar(y)?)?,
        })
    }

    fn expanded(self, margin: f64) -> Option<Self> {
        let r = Interval::scalar(margin)?;
        if margin < 0.0 {
            return None;
        }
        Some(Self {
            x: Interval::new(self.x.lo, self.x.hi)?.add(Interval::new(-r.hi, r.hi)?)?,
            y: Interval::new(self.y.lo, self.y.hi)?.add(Interval::new(-r.hi, r.hi)?)?,
        })
    }

    fn preimage(self, affine: Render2dAffineTransform) -> Option<Self> {
        let [a, b, c, d, tx, ty] = affine.components().map(Interval::scalar);
        let (a, b, c, d, tx, ty) = (a?, b?, c?, d?, tx?, ty?);
        let determinant = a.mul(d)?.sub(b.mul(c)?)?;
        let x = self.x.sub(tx)?;
        let y = self.y.sub(ty)?;
        Some(Self {
            x: d.mul(x)?.sub(c.mul(y)?)?.div(determinant)?,
            y: a.mul(y)?.sub(b.mul(x)?)?.div(determinant)?,
        })
    }

    fn classify_rectangle(self, [x0, y0, x1, y1]: [Interval; 4]) -> CellCertificate {
        // A physical vertex divided by its source scale may round in either
        // direction. A definite answer must account for BOTH possibilities.
        if self.x.lo > x0.hi && self.x.hi < x1.lo && self.y.lo > y0.hi && self.y.hi < y1.lo {
            CellCertificate::Inside
        } else if self.x.hi < x0.lo || self.x.lo > x1.hi || self.y.hi < y0.lo || self.y.lo > y1.hi {
            CellCertificate::Outside
        } else {
            CellCertificate::Unknown
        }
    }

    fn classify_ellipse(self, rect: Render2dRect) -> CellCertificate {
        if rect.is_empty() {
            return CellCertificate::Outside;
        }
        // Normalized ellipse equation, without approximating the curve by
        // an inscribed polygon or by its enclosing rectangle.
        let normalized = |coordinate: Interval, origin: f64, extent: f64| {
            coordinate
                .sub(Interval::scalar(origin)?)?
                .div(Interval::scalar(extent)?)?
                .mul(Interval::scalar(2.0)?)?
                .sub(Interval::scalar(1.0)?)
        };
        let Some(x) = normalized(self.x, rect.x(), rect.width()) else {
            return CellCertificate::Unknown;
        };
        let Some(y) = normalized(self.y, rect.y(), rect.height()) else {
            return CellCertificate::Unknown;
        };
        let Some(square_sum) = squared_interval(x).and_then(|xx| xx.add(squared_interval(y)?))
        else {
            return CellCertificate::Unknown;
        };
        if square_sum.hi < 1.0 {
            CellCertificate::Inside
        } else if square_sum.lo > 1.0 {
            CellCertificate::Outside
        } else {
            CellCertificate::Unknown
        }
    }
}

/// Square an interval without the sign-induced underestimation of a naive
/// interval product. The result still encloses every possible square.
fn squared_interval(v: Interval) -> Option<Interval> {
    let a = v.lo * v.lo;
    let b = v.hi * v.hi;
    if !a.is_finite() || !b.is_finite() {
        return None;
    }
    let lo = if v.lo <= 0.0 && v.hi >= 0.0 {
        0.0
    } else {
        a.min(b).next_down().max(0.0)
    };
    Interval::new(lo, a.max(b).next_up())
}

/// A path has exact authored control-point containment, even when its
/// disposable tessellation uses a coarser approximation. Every polynomial
/// Bézier segment lies in the convex hull of its control points. This gives a
/// conservative OUTSIDE result for arbitrary multi-contour paths. A single
/// strictly convex straight-line contour additionally permits exact halfplane
/// membership, independently of the original fill rule and Lyon triangles.
fn certify_path(path: &Render2dPath, cell: Cell, visits: &mut usize) -> CellCertificate {
    let commands = path.commands();
    // Proving strict convexity checks EVERY vertex against EVERY edge,
    // not merely neighboring turns: pentagrams can have consistent turns.
    let Some(cost) = commands
        .len()
        .checked_add(6)
        .and_then(|n| n.checked_mul(commands.len()))
    else {
        return CellCertificate::Unknown;
    };
    if *visits < cost {
        *visits = 0;
        return CellCertificate::Unknown;
    }
    *visits -= cost;

    let mut vertices = Vec::new();
    if vertices.try_reserve_exact(commands.len()).is_err() {
        return CellCertificate::Unknown;
    }
    let mut hull: Option<[f64; 4]> = None;
    let mut contours = 0_usize;
    let mut straight = true;
    let mut segment_count = 0_usize;
    let mut point = |p: crate::composition_2d::Render2dPoint| {
        let (x, y) = (p.x(), p.y());
        hull = Some(match hull {
            Some([x0, y0, x1, y1]) => [x0.min(x), y0.min(y), x1.max(x), y1.max(y)],
            None => [x, y, x, y],
        });
    };
    for command in commands {
        match *command {
            Render2dPathCommand::MoveTo(p) => {
                contours += 1;
                point(p);
                vertices.push([p.x(), p.y()]);
            }
            Render2dPathCommand::LineTo(p) => {
                segment_count += 1;
                point(p);
                vertices.push([p.x(), p.y()]);
            }
            Render2dPathCommand::QuadraticTo { control, to } => {
                straight = false;
                segment_count += 1;
                point(control);
                point(to);
            }
            Render2dPathCommand::CubicTo {
                control1,
                control2,
                to,
            } => {
                straight = false;
                segment_count += 1;
                point(control1);
                point(control2);
                point(to);
            }
            Render2dPathCommand::Close => {}
        }
    }
    if segment_count == 0 {
        return CellCertificate::Outside;
    }
    let Some([x0, y0, x1, y1]) = hull else {
        return CellCertificate::Unknown;
    };
    // Strict separation from the control-point hull is sufficient even for
    // curves, self-overlaps, alternate fill rules and multiple contours.
    if cell.x.hi < x0 || cell.x.lo > x1 || cell.y.hi < y0 || cell.y.lo > y1 {
        return CellCertificate::Outside;
    }
    if !straight || contours != 1 {
        return CellCertificate::Unknown;
    }
    if vertices.len() >= 2 && vertices.first() == vertices.last() {
        vertices.pop();
    }
    if vertices.len() < 3 {
        return CellCertificate::Unknown;
    }

    let orientation = |a: [f64; 2], b: [f64; 2], c: [Interval; 2]| {
        let dx = Interval::scalar(b[0])?.sub(Interval::scalar(a[0])?)?;
        let dy = Interval::scalar(b[1])?.sub(Interval::scalar(a[1])?)?;
        let px = c[0].sub(Interval::scalar(a[0])?)?;
        let py = c[1].sub(Interval::scalar(a[1])?)?;
        dx.mul(py)?.sub(dy.mul(px)?)
    };
    let mut winding = 0_i8;
    for i in 0..vertices.len() {
        let a = vertices[i];
        let b = vertices[(i + 1) % vertices.len()];
        for (j, &c) in vertices.iter().enumerate() {
            if j == i || j == (i + 1) % vertices.len() {
                continue;
            }
            let Some(turn) = orientation(
                a,
                b,
                [Interval::scalar(c[0]).unwrap(), Interval::scalar(c[1]).unwrap()],
            ) else {
                return CellCertificate::Unknown;
            };
            let sign = if turn.lo > 0.0 {
                1
            } else if turn.hi < 0.0 {
                -1
            } else {
                return CellCertificate::Unknown;
            };
            if winding != 0 && winding != sign {
                // A supporting halfplane of a genuine convex polygon
                // contains ALL other vertices. Stars and concave polygons
                // fail here even when their local turns have the same sign.
                return CellCertificate::Unknown;
            }
            winding = sign;
        }
    }
    let mut uncertain = false;
    for i in 0..vertices.len() {
        let Some(side) = orientation(
            vertices[i],
            vertices[(i + 1) % vertices.len()],
            [cell.x, cell.y],
        ) else {
            return CellCertificate::Unknown;
        };
        if (winding > 0 && side.hi < 0.0) || (winding < 0 && side.lo > 0.0) {
            return CellCertificate::Outside;
        }
        if (winding > 0 && side.lo <= 0.0) || (winding < 0 && side.hi >= 0.0) {
            uncertain = true;
        }
    }
    if uncertain {
        CellCertificate::Unknown
    } else {
        CellCertificate::Inside
    }
}

fn certify_shape(shape: &Render2dShape, cell: Cell, visits: &mut usize) -> CellCertificate {
    if let Some(rect) = shape.as_rect() {
        if rect.is_empty() {
            return CellCertificate::Outside;
        }
        let Some(x0) = Interval::scalar(rect.x()) else {
            return CellCertificate::Unknown;
        };
        let Some(y0) = Interval::scalar(rect.y()) else {
            return CellCertificate::Unknown;
        };
        let Some(x1) = x0.add(Interval::scalar(rect.width()).expect("valid finite width")) else {
            return CellCertificate::Unknown;
        };
        let Some(y1) = y0.add(Interval::scalar(rect.height()).expect("valid finite height")) else {
            return CellCertificate::Unknown;
        };
        cell.classify_rectangle([x0, y0, x1, y1])
    } else if let Some(ellipse) = shape.as_ellipse() {
        cell.classify_ellipse(ellipse)
    } else if let Some(path) = shape.as_path() {
        certify_path(path, cell, visits)
    } else {
        // Paths, centered strokes and rounded corners still need their own
        // independently certified source geometry. Never infer from meshes.
        CellCertificate::Unknown
    }
}

/// A clip is authored in the OWNER PARENT frame, not source-local space.
/// This same operation applies to F1 item clips and immutable group clip DAGs.
fn certify_clipped(
    source: CellCertificate,
    parent_cell: Cell,
    clips: &[Render2dClip],
    visits: &mut usize,
) -> CellCertificate {
    if source == CellCertificate::Outside {
        return CellCertificate::Outside;
    }
    let mut uncertain = source == CellCertificate::Unknown;
    for clip in clips {
        if *visits == 0 {
            return CellCertificate::Unknown;
        }
        *visits -= 1;
        let certificate = parent_cell
            .preimage(clip.clip_to_parent())
            .map(|local| certify_shape(clip.shape(), local, visits))
            .unwrap_or(CellCertificate::Unknown);
        match certificate {
            CellCertificate::Outside => return CellCertificate::Outside,
            CellCertificate::Unknown => uncertain = true,
            CellCertificate::Inside => {}
        }
    }
    if uncertain {
        CellCertificate::Unknown
    } else {
        CellCertificate::Inside
    }
}

impl NeutralRegion<'_> {
    fn certify_cell(&self, cell: Cell, visits: &mut usize) -> CellCertificate {
        if *visits == 0 {
            return CellCertificate::Unknown;
        }
        *visits -= 1;
        match &self.kind {
            Kind::Source(mesh) => {
                if let Some(item) = self.semantic_item {
                    return certify_authored_item(item, cell, visits);
                }
                // Synthetic private fixture only. Real renderable F1 leaves
                // always borrow their original item through source_from_item.
                if mesh.triangles.is_empty() {
                    return CellCertificate::Outside;
                }
                let Some(rect) = support::certified_axis_aligned_rectangle(&mesh.triangles) else {
                    return CellCertificate::Unknown;
                };
                let scale = mesh.units_per_parent_logical_unit;
                if !scale.is_finite() || scale <= 0.0 {
                    return CellCertificate::Unknown;
                }
                let Some(scale) = Interval::scalar(scale) else {
                    return CellCertificate::Unknown;
                };
                let intervals = rect.map(|v| Interval::scalar(v).and_then(|v| v.div(scale)));
                let [Some(x0), Some(y0), Some(x1), Some(y1)] = intervals else {
                    return CellCertificate::Unknown;
                };
                cell.classify_rectangle([x0, y0, x1, y1])
            }
            Kind::Union(children) => {
                let mut unknown = false;
                for child in children {
                    match child.certify_cell(cell, visits) {
                        CellCertificate::Inside => return CellCertificate::Inside,
                        CellCertificate::Unknown => unknown = true,
                        CellCertificate::Outside => {}
                    }
                }
                if unknown {
                    CellCertificate::Unknown
                } else {
                    CellCertificate::Outside
                }
            }
            Kind::Affine { source, to_parent } => cell
                .preimage(*to_parent)
                .map(|preimage| source.certify_cell(preimage, visits))
                .unwrap_or(CellCertificate::Unknown),
            Kind::Clip { source, clips } => {
                let child = source.certify_cell(cell, visits);
                certify_clipped(child, cell, clips, visits)
            }
            Kind::Effect { source, shadow } => {
                let Some(translated) = cell.shift(shadow.offset_x(), shadow.offset_y()) else {
                    return CellCertificate::Unknown;
                };
                let spread = shadow.spread();
                let halo = 3.0 * shadow.sigma();
                // A spread region is contained in its own finite-Gaussian
                // support, but this says NOTHING about blurred alpha.
                let definite_inside = if spread < 0.0 {
                    translated
                        .expanded(-spread)
                        .map(|c| source.certify_cell(c, visits))
                } else {
                    Some(source.certify_cell(translated, visits))
                };
                if definite_inside == Some(CellCertificate::Inside) {
                    return CellCertificate::Inside;
                }
                let Some(outer) = translated.expanded(halo + spread.max(0.0)) else {
                    return CellCertificate::Unknown;
                };
                if source.certify_cell(outer, visits) == CellCertificate::Outside {
                    CellCertificate::Outside
                } else {
                    CellCertificate::Unknown
                }
            }
        }
    }
}

/// Certify original F1 fill geometry first; apply item clips independently
/// in their declared immediate-parent space. Painter color/opacity are ignored.
fn certify_authored_item(item: &Render2dItem, cell: Cell, visits: &mut usize) -> CellCertificate {
    let Render2dPrimitive::Fill { shape, .. } = item.primitive() else {
        return CellCertificate::Unknown;
    };
    let source = cell
        .preimage(item.local_to_parent())
        .map(|local| certify_shape(shape, local, visits))
        .unwrap_or(CellCertificate::Unknown);
    certify_clipped(source, cell, item.clips(), visits)
}

fn certify(region: &NeutralRegion<'_>, bounds: [f64; 4]) -> CellCertificate {
    let Some(cell) = Cell::new(bounds) else {
        return CellCertificate::Unknown;
    };
    let mut visits = 4096;
    region.certify_cell(cell, &mut visits)
}

fn rectangle(
    builder: &mut NeutralRegionBuilder,
    rect: [f64; 4],
    path: &[usize],
) -> Arc<NeutralRegion<'static>> {
    let [x0, y0, x1, y1] = rect;
    builder
        .source(
            support::NeutralMesh {
                units_per_parent_logical_unit: 1.0,
                triangles: vec![[x0, y0], [x1, y0], [x1, y1], [x0, y0], [x1, y1], [x0, y1]],
                bounds: rect,
            },
            path,
        )
        .unwrap()
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
fn certified_cells_handle_holed_union_and_signed_morphology_without_alpha_masks() {
    let mut builder = builder();
    let sources = [
        [0.0, 0.0, 3.0, 10.0],
        [7.0, 0.0, 10.0, 10.0],
        [3.0, 0.0, 7.0, 3.0],
        [3.0, 7.0, 7.0, 10.0],
    ]
    .into_iter()
    .enumerate()
    .map(|(i, rect)| rectangle(&mut builder, rect, &[0, i]))
    .collect();
    let ring = builder.union(sources, &[0]).unwrap();
    assert_eq!(
        certify(&ring, [4.5, 4.5, 5.5, 5.5]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&ring, [1.0, 4.0, 2.0, 6.0]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&ring, [2.9, 4.0, 3.1, 6.0]),
        CellCertificate::Unknown
    );
    for spread in [-0.5, 0.5] {
        let effect = Render2dDropShadow::new(
            0.0,
            0.0,
            0.0,
            spread,
            crate::composition_2d::Render2dColorRgba8::TRANSPARENT,
        )
        .unwrap();
        let region = builder.effect(Arc::clone(&ring), effect, &[0, 5]).unwrap();
        assert_eq!(
            certify(&region, [4.5, 4.5, 5.5, 5.5]),
            CellCertificate::Outside
        );
        if spread < 0.0 {
            assert_eq!(
                certify(&region, [1.0, 4.0, 2.0, 6.0]),
                CellCertificate::Inside
            );
        }
    }
}

#[test]
fn affine_cell_preimage_is_outward_bounded_and_rejects_singular_certainty() {
    let mut builder = builder();
    let rect = rectangle(&mut builder, [0.0, 0.0, 2.0, 2.0], &[0, 0]);
    let shear = Render2dAffineTransform::new(1.0, 0.0, 0.5, 1.0, 4.0, 3.0).unwrap();
    let sheared = builder.affine(Arc::clone(&rect), shear, &[0, 1]).unwrap();
    assert_eq!(
        certify(&sheared, [5.25, 3.5, 5.5, 4.0]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&sheared, [0.0, 0.0, 1.0, 1.0]),
        CellCertificate::Outside
    );
    let reflection = Render2dAffineTransform::new(-1.0, 0.0, 0.0, 1.0, 3.0, 0.0).unwrap();
    let reflected = builder.affine(rect, reflection, &[0, 2]).unwrap();
    assert_eq!(
        certify(&reflected, [1.5, 0.5, 2.0, 1.0]),
        CellCertificate::Inside
    );
    let singular = Render2dAffineTransform::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
    let singular_region = builder
        .affine(Arc::clone(&sheared), singular, &[0, 3])
        .unwrap();
    assert_eq!(
        certify(&singular_region, [1.0, 1.0, 2.0, 2.0]),
        CellCertificate::Unknown
    );
}

#[test]
fn actual_f1_rectangle_is_queried_before_disposable_lyon_triangles() {
    use crate::composition_2d::{
        Render2dBrush, Render2dClip, Render2dColorRgba8, Render2dOpacity, Render2dRect,
        Render2dShape,
    };
    use crate::runtime::execution_2d::vector;
    let rectangle = Render2dShape::rect(Render2dRect::new(0.0, 0.0, 2.0, 2.0).unwrap());
    let shear = Render2dAffineTransform::new(1.0, 0.0, 0.5, 1.0, 4.0, 3.0).unwrap();
    let item = Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: rectangle.clone(),
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        shear,
        Vec::new(),
        Render2dOpacity::TRANSPARENT,
    );
    let mesh = vector::neutral_support(&item, 0, 4.0, 1_048_576)
        .unwrap()
        .expect("a real F1 rectangle tessellates");
    let mut graph = builder();
    let region = graph.source_from_item(mesh, &item, &[0, 0]).unwrap();
    assert_eq!(
        certify(&region, [4.5, 3.5, 4.6, 3.6]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&region, [0.0, 0.0, 1.0, 1.0]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&region, [4.0, 3.0, 4.0, 3.0]),
        CellCertificate::Unknown
    );

    let clip = Render2dClip::new(
        Render2dShape::rect(Render2dRect::new(0.0, 0.0, 0.25, 0.25).unwrap()),
        Render2dAffineTransform::IDENTITY,
    );
    let clipped = Render2dItem::new(
        Render2dPrimitive::Fill {
            shape: rectangle,
            brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
        },
        shear,
        vec![clip],
        Render2dOpacity::TRANSPARENT,
    );
    // A deliberately independent physical mesh cannot authorize an Inside
    // result through a real clip without first proving its geometry.
    let physical = support::NeutralMesh {
        units_per_parent_logical_unit: 1.0,
        triangles: vec![
            [4.0, 3.0],
            [6.0, 3.0],
            [6.0, 5.0],
            [4.0, 3.0],
            [6.0, 5.0],
            [4.0, 5.0],
        ],
        bounds: [4.0, 3.0, 6.0, 5.0],
    };
    let clipped_region = graph.source_from_item(physical, &clipped, &[0, 1]).unwrap();
    assert_eq!(
        certify(&clipped_region, [4.5, 3.5, 4.6, 3.6]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&clipped_region, [0.0, 0.0, 1.0, 1.0]),
        CellCertificate::Outside
    );
}

#[test]
fn original_f1_ellipse_uses_exact_semantics_instead_of_rectangle_mesh() {
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dOpacity, Render2dRect, Render2dShape,
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
    let mesh = support::NeutralMesh {
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
    let source = graph.source_from_item(mesh, &item, &[0, 0]).unwrap();
    assert_eq!(
        certify(&source, [0.5, 0.5, 1.5, 1.5]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&source, [-2.0, -2.0, -1.0, -1.0]),
        CellCertificate::Outside
    );
}

#[test]
fn group_ellipse_clip_certifies_inside_outside_and_budget_exhaustion() {
    use crate::composition_2d::{Render2dClip, Render2dRect, Render2dShape};
    let mut graph = builder();
    let source = rectangle(&mut graph, [0.0, 0.0, 6.0, 6.0], &[0, 0]);
    let clips = vec![Render2dClip::new(
        Render2dShape::ellipse(Render2dRect::new(1.0, 1.0, 4.0, 4.0).unwrap()),
        Render2dAffineTransform::IDENTITY,
    )];
    let clipped = graph.clip(source, &clips, &[0]).unwrap();
    assert_eq!(
        certify(&clipped, [2.75, 2.75, 3.25, 3.25]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&clipped, [1.1, 1.1, 1.2, 1.2]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&clipped, [5.0, 3.0, 5.0, 3.0]),
        CellCertificate::Unknown
    );
    let mut work = 2;
    assert_eq!(
        clipped.certify_cell(Cell::new([2.75, 2.75, 3.25, 3.25]).unwrap(), &mut work,),
        CellCertificate::Unknown
    );
    assert_eq!(work, 0);
}

#[test]
fn original_f1_convex_line_path_certifies_both_windings_and_edge_uncertainty() {
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dFillRule, Render2dOpacity,
        Render2dPath, Render2dPoint,
    };
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    let make = |points: [(f64, f64); 3], rule| {
        let shape = Render2dShape::path(
            Render2dPath::new(
                rule,
                vec![
                    Render2dPathCommand::MoveTo(p(points[0].0, points[0].1)),
                    Render2dPathCommand::LineTo(p(points[1].0, points[1].1)),
                    Render2dPathCommand::LineTo(p(points[2].0, points[2].1)),
                    Render2dPathCommand::Close,
                ],
            )
            .unwrap(),
        );
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
    for (points, rule) in [
        ([(0.0, 0.0), (8.0, 0.0), (0.0, 8.0)], Render2dFillRule::NonZero),
        ([(0.0, 8.0), (8.0, 0.0), (0.0, 0.0)], Render2dFillRule::EvenOdd),
    ] {
        let item = make(points, rule);
        let mut graph = builder();
        // An intentionally misleading physical rectangle may not override
        // the original triangle's genuine diagonal boundary.
        let source = graph
            .source_from_item(
                support::NeutralMesh {
                    units_per_parent_logical_unit: 1.0,
                    triangles: vec![
                        [0.0, 0.0], [8.0, 0.0], [8.0, 8.0],
                        [0.0, 0.0], [8.0, 8.0], [0.0, 8.0],
                    ],
                    bounds: [0.0, 0.0, 8.0, 8.0],
                },
                &item,
                &[0],
            )
            .unwrap();
        assert_eq!(certify(&source, [1.0, 1.0, 2.0, 2.0]), CellCertificate::Inside);
        assert_eq!(certify(&source, [6.0, 6.0, 7.0, 7.0]), CellCertificate::Outside);
        assert_eq!(certify(&source, [3.9, 3.9, 4.1, 4.1]), CellCertificate::Unknown);
    }
}

#[test]
fn self_intersecting_star_cannot_be_certified_from_consistent_local_turns() {
    use crate::composition_2d::{Render2dFillRule, Render2dPath, Render2dPoint};
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    let star = Render2dShape::path(
        Render2dPath::new(
            Render2dFillRule::EvenOdd,
            vec![
                Render2dPathCommand::MoveTo(p(0.0, 10.0)),
                Render2dPathCommand::LineTo(p(-5.8778525, -8.0901699)),
                Render2dPathCommand::LineTo(p(9.5105652, 3.0901699)),
                Render2dPathCommand::LineTo(p(-9.5105652, 3.0901699)),
                Render2dPathCommand::LineTo(p(5.8778525, -8.0901699)),
                Render2dPathCommand::Close,
            ],
        )
        .unwrap(),
    );
    let mut visits = 4096;
    assert_eq!(
        certify_shape(
            &star,
            Cell::new([-0.1, -0.1, 0.1, 0.1]).unwrap(),
            &mut visits,
        ),
        CellCertificate::Unknown
    );
    assert_eq!(
        certify_shape(
            &star,
            Cell::new([12.0, 12.0, 13.0, 13.0]).unwrap(),
            &mut visits,
        ),
        CellCertificate::Outside
    );
}

#[test]
fn original_f1_curves_and_multiple_contours_can_only_prove_control_hull_exterior() {
    use crate::composition_2d::{Render2dFillRule, Render2dPath, Render2dPoint};
    let p = |x, y| Render2dPoint::new(x, y).unwrap();
    let cubic = Render2dShape::path(
        Render2dPath::new(
            Render2dFillRule::NonZero,
            vec![
                Render2dPathCommand::MoveTo(p(0.0, 0.0)),
                Render2dPathCommand::CubicTo {
                    control1: p(0.0, 5.0),
                    control2: p(5.0, 5.0),
                    to: p(5.0, 0.0),
                },
                Render2dPathCommand::Close,
            ],
        )
        .unwrap(),
    );
    let mut visits = 100;
    let cell = Cell::new([6.0, 1.0, 7.0, 2.0]).unwrap();
    assert_eq!(certify_shape(&cubic, cell, &mut visits), CellCertificate::Outside);
    assert_eq!(
        certify_shape(&cubic, Cell::new([1.0, 1.0, 2.0, 2.0]).unwrap(), &mut visits),
        CellCertificate::Unknown
    );
    let multi = Render2dShape::path(
        Render2dPath::new(
            Render2dFillRule::EvenOdd,
            vec![
                Render2dPathCommand::MoveTo(p(0.0, 0.0)),
                Render2dPathCommand::LineTo(p(4.0, 0.0)),
                Render2dPathCommand::LineTo(p(4.0, 4.0)),
                Render2dPathCommand::Close,
                Render2dPathCommand::MoveTo(p(1.0, 1.0)),
                Render2dPathCommand::LineTo(p(2.0, 1.0)),
                Render2dPathCommand::LineTo(p(2.0, 2.0)),
                Render2dPathCommand::Close,
            ],
        )
        .unwrap(),
    );
    assert_eq!(
        certify_shape(&multi, Cell::new([1.2, 1.2, 1.4, 1.4]).unwrap(), &mut visits),
        CellCertificate::Unknown
    );
    assert_eq!(
        certify_shape(&multi, Cell::new([7.0, 7.0, 8.0, 8.0]).unwrap(), &mut visits),
        CellCertificate::Outside
    );
    let mut exhausted = 1;
    assert_eq!(
        certify_shape(&multi, Cell::new([7.0, 7.0, 8.0, 8.0]).unwrap(), &mut exhausted),
        CellCertificate::Unknown
    );
    assert_eq!(exhausted, 0);
}

#[test]
fn scaled_source_certificate_uses_outward_rounded_rectangle_division() {
    let mut builder = builder();
    let mesh = support::NeutralMesh {
        units_per_parent_logical_unit: 3.0,
        triangles: vec![
            [0.0, 0.0],
            [9.0, 0.0],
            [9.0, 9.0],
            [0.0, 0.0],
            [9.0, 9.0],
            [0.0, 9.0],
        ],
        bounds: [0.0, 0.0, 9.0, 9.0],
    };
    let source = builder.source(mesh, &[4, 0]).unwrap();
    assert_eq!(
        certify(&source, [1.0, 1.0, 2.0, 2.0]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&source, [-1.0, -1.0, -0.5, -0.5]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&source, [3.0, 1.0, 3.0, 2.0]),
        CellCertificate::Unknown
    );
}

#[test]
fn bounded_cell_query_never_treats_unknown_curve_or_exhaustion_as_empty() {
    let mut builder = builder();
    let rect = rectangle(&mut builder, [0.0, 0.0, 2.0, 2.0], &[0, 0]);
    let effect = Render2dDropShadow::new(
        0.0,
        0.0,
        0.25,
        1.0,
        crate::composition_2d::Render2dColorRgba8::TRANSPARENT,
    )
    .unwrap();
    let curved = builder.effect(rect, effect, &[0, 1]).unwrap();
    assert_eq!(
        certify(&curved, [0.5, 0.5, 1.5, 1.5]),
        CellCertificate::Inside
    );
    assert_eq!(
        certify(&curved, [-4.0, -4.0, -3.0, -3.0]),
        CellCertificate::Outside
    );
    assert_eq!(
        certify(&curved, [-0.9, -0.9, -0.8, -0.8]),
        CellCertificate::Unknown
    );
    let mut exhausted = 0;
    assert_eq!(
        curved.certify_cell(Cell::new([0.5, 0.5, 1.5, 1.5]).unwrap(), &mut exhausted),
        CellCertificate::Unknown,
    );
}
