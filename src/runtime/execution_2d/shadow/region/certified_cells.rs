//! Test-only feasibility of conservative cell classification over the actual
//! F1 neutral-region DAG. This does NOT admit a new production shadow class.
//!
//! Only entire, certified physical rectangles are treated as known sources.
//! Tessellated curves are deliberately Unknown: a physical triangle inventory
//! is not a certificate of exact F1 source geometry. A production evaluator
//! must separately establish source enclosures and Gaussian *coverage* bounds.
//! A certificate concerns SUPPORT membership, not alpha or Gaussian weights.

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CellCertificate {
    Inside,
    Outside,
    Unknown,
}

// All calculated bounds are outward-rounded. Any arithmetic uncertainty,
// nonfinite result or singular transform yields Unknown, never fake geometry.
#[derive(Clone, Copy, Debug)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn new(lo: f64, hi: f64) -> Option<Self> {
        (lo.is_finite() && hi.is_finite() && lo <= hi).then_some(Self { lo, hi })
    }

    fn scalar(value: f64) -> Option<Self> {
        Self::new(value, value)
    }

    fn add(self, other: Self) -> Option<Self> {
        Self::new(
            (self.lo + other.lo).next_down(),
            (self.hi + other.hi).next_up(),
        )
    }

    fn sub(self, other: Self) -> Option<Self> {
        Self::new(
            (self.lo - other.hi).next_down(),
            (self.hi - other.lo).next_up(),
        )
    }

    fn mul(self, other: Self) -> Option<Self> {
        let candidates = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ];
        if !candidates.iter().all(|x| x.is_finite()) {
            return None;
        }
        let lo = candidates.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = candidates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::new(lo.next_down(), hi.next_up())
    }

    fn div(self, other: Self) -> Option<Self> {
        if other.lo <= 0.0 && other.hi >= 0.0 {
            return None;
        }
        let candidates = [
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        ];
        if !candidates.iter().all(|x| x.is_finite()) {
            return None;
        }
        let lo = candidates.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = candidates.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::new(lo.next_down(), hi.next_up())
    }
}

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
            x: Interval::new(self.x.lo, self.x.hi)?
                .add(Interval::new(-r.hi, r.hi)?)?,
            y: Interval::new(self.y.lo, self.y.hi)?
                .add(Interval::new(-r.hi, r.hi)?)?,
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

    fn classify_rectangle(self, [x0, y0, x1, y1]: [f64; 4]) -> CellCertificate {
        if self.x.lo > x0 && self.x.hi < x1 && self.y.lo > y0 && self.y.hi < y1 {
            CellCertificate::Inside
        } else if self.x.hi < x0
            || self.x.lo > x1
            || self.y.hi < y0
            || self.y.lo > y1
        {
            CellCertificate::Outside
        } else {
            CellCertificate::Unknown
        }
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
                let rect = rect.map(|v| v / scale);
                if rect.iter().all(|v| v.is_finite()) {
                    cell.classify_rectangle(rect)
                } else {
                    CellCertificate::Unknown
                }
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
                if clips.is_empty() || child == CellCertificate::Outside {
                    child
                } else {
                    // Actual clip geometry requires a separate certification.
                    CellCertificate::Unknown
                }
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

fn certify(region: &NeutralRegion<'_>, bounds: [f64; 4]) -> CellCertificate {
    let Some(cell) = Cell::new(bounds) else {
        return CellCertificate::Unknown;
    };
    let mut visits = 4096;
    region.certify_cell(cell, &mut visits)
}

fn rectangle(builder: &mut NeutralRegionBuilder, rect: [f64; 4], path: &[usize]) -> Arc<NeutralRegion<'static>> {
    let [x0, y0, x1, y1] = rect;
    builder
        .source(
            support::NeutralMesh {
                units_per_parent_logical_unit: 1.0,
                triangles: vec![
                    [x0, y0], [x1, y0], [x1, y1],
                    [x0, y0], [x1, y1], [x0, y1],
                ],
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
    assert_eq!(certify(&ring, [4.5, 4.5, 5.5, 5.5]), CellCertificate::Outside);
    assert_eq!(certify(&ring, [1.0, 4.0, 2.0, 6.0]), CellCertificate::Inside);
    assert_eq!(certify(&ring, [2.9, 4.0, 3.1, 6.0]), CellCertificate::Unknown);
    for spread in [-0.5, 0.5] {
        let effect = Render2dDropShadow::new(
            0.0, 0.0, 0.0, spread,
            crate::composition_2d::Render2dColorRgba8::TRANSPARENT,
        ).unwrap();
        let region = builder.effect(Arc::clone(&ring), effect, &[0, 5]).unwrap();
        assert_eq!(certify(&region, [4.5, 4.5, 5.5, 5.5]), CellCertificate::Outside);
        if spread < 0.0 {
            assert_eq!(certify(&region, [1.0, 4.0, 2.0, 6.0]), CellCertificate::Inside);
        }
    }
}

#[test]
fn affine_cell_preimage_is_outward_bounded_and_rejects_singular_certainty() {
    let mut builder = builder();
    let rect = rectangle(&mut builder, [0.0, 0.0, 2.0, 2.0], &[0, 0]);
    let shear = Render2dAffineTransform::new(1.0, 0.0, 0.5, 1.0, 4.0, 3.0).unwrap();
    let sheared = builder.affine(Arc::clone(&rect), shear, &[0, 1]).unwrap();
    assert_eq!(certify(&sheared, [5.25, 3.5, 5.5, 4.0]), CellCertificate::Inside);
    assert_eq!(certify(&sheared, [0.0, 0.0, 1.0, 1.0]), CellCertificate::Outside);
    let reflection = Render2dAffineTransform::new(-1.0, 0.0, 0.0, 1.0, 3.0, 0.0).unwrap();
    let reflected = builder.affine(rect, reflection, &[0, 2]).unwrap();
    assert_eq!(certify(&reflected, [1.5, 0.5, 2.0, 1.0]), CellCertificate::Inside);
    let singular = Render2dAffineTransform::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0).unwrap();
    let singular_region = builder.affine(Arc::clone(&sheared), singular, &[0, 3]).unwrap();
    assert_eq!(certify(&singular_region, [1.0, 1.0, 2.0, 2.0]), CellCertificate::Unknown);
}

#[test]
fn bounded_cell_query_never_treats_unknown_curve_or_exhaustion_as_empty() {
    let mut builder = builder();
    let rect = rectangle(&mut builder, [0.0, 0.0, 2.0, 2.0], &[0, 0]);
    let effect = Render2dDropShadow::new(
        0.0, 0.0, 0.25, 1.0,
        crate::composition_2d::Render2dColorRgba8::TRANSPARENT,
    ).unwrap();
    let curved = builder.effect(rect, effect, &[0, 1]).unwrap();
    assert_eq!(certify(&curved, [0.5, 0.5, 1.5, 1.5]), CellCertificate::Inside);
    assert_eq!(certify(&curved, [-4.0, -4.0, -3.0, -3.0]), CellCertificate::Outside);
    assert_eq!(certify(&curved, [-0.9, -0.9, -0.8, -0.8]), CellCertificate::Unknown);
    let mut exhausted = 0;
    assert_eq!(
        curved.certify_cell(Cell::new([0.5, 0.5, 1.5, 1.5]).unwrap(), &mut exhausted),
        CellCertificate::Unknown,
    );
}
