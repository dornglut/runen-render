//! Source-derived exposed boundaries for orthogonal F1 fill arrangements.
//!
//! Event coordinates come only from authored geometry. Original F1 membership
//! determines each open arrangement cell; adjacent unequal cells expose the
//! true boundary, including holes and excluding union seams. This retained
//! geometric certificate is independent of physical meshes and paint alpha.
use super::super::interval::exact_add;
use super::*;

const MAX_EVENTS: usize = 256;
const MAX_CELLS: usize = 16_384;
const MAX_EDGES: usize = 1_024;

fn charge(fuel: &mut usize, amount: usize) -> Option<()> {
    if *fuel < amount {
        *fuel = 0;
        return None;
    }
    *fuel -= amount;
    Some(())
}

#[derive(Default)]
struct Events {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl Events {
    fn point(
        &mut self,
        mut p: [f64; 2],
        transforms: &[Render2dAffineTransform],
        fuel: &mut usize,
    ) -> Option<[f64; 2]> {
        charge(fuel, 32_usize.checked_mul(transforms.len() + 1)?)?;
        for transform in transforms.iter().rev() {
            let [a, b, c, d, tx, ty] = transform.components();
            // Exact signed axis permutations and translations. Oblique or
            // scaled source arrangements require a different boundary proof.
            let sign = |v: f64| v == 1.0 || v == -1.0;
            p = if sign(a) && b == 0.0 && c == 0.0 && sign(d) {
                [exact_add(a * p[0], tx)?, exact_add(d * p[1], ty)?]
            } else if a == 0.0 && sign(b) && sign(c) && d == 0.0 {
                [exact_add(c * p[1], tx)?, exact_add(b * p[0], ty)?]
            } else {
                return None;
            };
        }
        if self.x.len() == MAX_EVENTS {
            return None;
        }
        self.x.try_reserve_exact(1).ok()?;
        self.y.try_reserve_exact(1).ok()?;
        self.x.push(p[0]);
        self.y.push(p[1]);
        Some(p)
    }

    fn shape(
        &mut self,
        shape: &Render2dShape,
        transforms: &[Render2dAffineTransform],
        fuel: &mut usize,
    ) -> Option<()> {
        if let Some(rect) = shape.as_rect() {
            let x1 = exact_add(rect.x(), rect.width())?;
            let y1 = exact_add(rect.y(), rect.height())?;
            self.point([rect.x(), rect.y()], transforms, fuel)?;
            self.point([x1, y1], transforms, fuel)?;
            return Some(());
        }
        let path = shape.as_path()?;
        charge(fuel, path.commands().len().checked_mul(8)?)?;
        let mut first = None;
        let mut previous = None;
        let mut edge = |a: [f64; 2], b: [f64; 2]| -> Option<()> {
            charge(fuel, 4)?;
            (a[0] == b[0] || a[1] == b[1]).then_some(())
        };
        // Every contour closes implicitly, just as original F1 fill does.
        for command in path.commands() {
            match *command {
                Render2dPathCommand::MoveTo(p) => {
                    if let (Some(a), Some(b)) = (previous, first) {
                        edge(a, b)?;
                    }
                    let point = [p.x(), p.y()];
                    first = Some(point);
                    previous = first;
                }
                Render2dPathCommand::LineTo(p) => {
                    let point = [p.x(), p.y()];
                    edge(previous?, point)?;
                    previous = Some(point);
                }
                Render2dPathCommand::Close => {
                    if let (Some(a), Some(b)) = (previous, first) {
                        edge(a, b)?;
                    }
                    previous = first;
                }
                _ => return None,
            }
        }
        if let (Some(a), Some(b)) = (previous, first) {
            edge(a, b)?;
        }
        for command in path.commands() {
            if let Render2dPathCommand::MoveTo(p) | Render2dPathCommand::LineTo(p) = *command {
                self.point([p.x(), p.y()], transforms, fuel)?;
            }
        }
        Some(())
    }

    fn clips(
        &mut self,
        clips: &[Render2dClip],
        transforms: &mut Vec<Render2dAffineTransform>,
        fuel: &mut usize,
    ) -> Option<()> {
        for clip in clips {
            transforms.try_reserve_exact(1).ok()?;
            transforms.push(clip.clip_to_parent());
            self.shape(clip.shape(), transforms, fuel)?;
            transforms.pop();
        }
        Some(())
    }

    fn region(
        &mut self,
        region: &NeutralRegion<'_>,
        transforms: &mut Vec<Render2dAffineTransform>,
        fuel: &mut usize,
    ) -> Option<()> {
        charge(fuel, 1)?;
        match &region.kind {
            Kind::Source(_) => {
                let item = region.semantic_item?;
                let Render2dPrimitive::Fill { shape, .. } = item.primitive() else {
                    return None;
                };
                transforms.try_reserve_exact(1).ok()?;
                transforms.push(item.local_to_parent());
                self.shape(shape, transforms, fuel)?;
                transforms.pop();
                self.clips(item.clips(), transforms, fuel)
            }
            Kind::Union(children) => {
                for child in children {
                    self.region(child, transforms, fuel)?;
                }
                Some(())
            }
            Kind::Affine { source, to_parent } => {
                transforms.try_reserve_exact(1).ok()?;
                transforms.push(*to_parent);
                self.region(source, transforms, fuel)?;
                transforms.pop();
                Some(())
            }
            Kind::Clip { source, clips } => {
                self.region(source, transforms, fuel)?;
                self.clips(clips, transforms, fuel)
            }
            Kind::Effect { .. } => None,
        }
    }
}

pub(super) struct Boundary {
    edges: Vec<[[f64; 2]; 2]>,
}

impl Boundary {
    pub(super) fn edges(&self) -> &[[[f64; 2]; 2]] {
        &self.edges
    }
    pub(super) fn edge_count(&self) -> usize {
        self.edges.len()
    }
    pub(super) fn prepare(source: &NeutralRegion<'_>, fuel: &mut usize) -> Option<Self> {
        let mut events = Events::default();
        // Reserve bounded inventories once. Per-element growth would hide
        // repeated allocator copies behind nominal constant insertion charges.
        events.x.try_reserve_exact(MAX_EVENTS).ok()?;
        events.y.try_reserve_exact(MAX_EVENTS).ok()?;
        let mut transforms = Vec::new();
        transforms
            .try_reserve_exact(source.depth.checked_add(2)?)
            .ok()?;
        events.region(source, &mut transforms, fuel)?;
        sort_admitted(&mut events.x, f64::total_cmp, 4, fuel)?;
        events.x.dedup();
        sort_admitted(&mut events.y, f64::total_cmp, 4, fuel)?;
        events.y.dedup();
        let width = events.x.len().checked_sub(1)?;
        let height = events.y.len().checked_sub(1)?;
        let count = width.checked_mul(height)?;
        if count == 0 || count > MAX_CELLS {
            return None;
        }
        charge(fuel, count.checked_mul(32)?)?;
        let mut occupied = Vec::new();
        occupied.try_reserve_exact(count).ok()?;
        for ys in events.y.windows(2) {
            for xs in events.x.windows(2) {
                let x = xs[0] * 0.5 + xs[1] * 0.5;
                let y = ys[0] * 0.5 + ys[1] * 0.5;
                if !(xs[0] < x && x < xs[1] && ys[0] < y && y < ys[1]) {
                    return None;
                }
                let point = Cell::new([x, y, x, y])?;
                // Original semantics resolve overlaps and fill rules. No
                // source boundary can cross this open event cell, so its
                // certified point membership applies throughout the cell.
                occupied.push(match source.certify_cell(point, fuel) {
                    CellCertificate::Inside => true,
                    CellCertificate::Outside => false,
                    CellCertificate::Unknown => return None,
                });
            }
        }
        let mut edges = Vec::new();
        edges.try_reserve_exact(MAX_EDGES).ok()?;
        for y in 0..height {
            for x in 0..width {
                if !occupied[y * width + x] {
                    continue;
                }
                let [x0, x1] = [events.x[x], events.x[x + 1]];
                let [y0, y1] = [events.y[y], events.y[y + 1]];
                for (exposed, a, b) in [
                    (x == 0 || !occupied[y * width + x - 1], [x0, y0], [x0, y1]),
                    (
                        x + 1 == width || !occupied[y * width + x + 1],
                        [x1, y0],
                        [x1, y1],
                    ),
                    (y == 0 || !occupied[(y - 1) * width + x], [x0, y0], [x1, y0]),
                    (
                        y + 1 == height || !occupied[(y + 1) * width + x],
                        [x0, y1],
                        [x1, y1],
                    ),
                ] {
                    if exposed {
                        if edges.len() == MAX_EDGES {
                            return None;
                        }
                        edges.try_reserve_exact(1).ok()?;
                        edges.push([a, b]);
                    }
                }
            }
        }
        // Exact collinear union: event-grid subdivisions are not additional
        // geometric boundaries. Retain gaps; merge only touching/overlapping
        // segments on the same authored coordinate line.
        charge(fuel, edges.len().checked_mul(256)?)?;
        for [a, b] in &mut edges {
            if a[0] > b[0] || a[1] > b[1] {
                std::mem::swap(a, b);
            }
        }
        sort_admitted(
            &mut edges,
            |[a, b], [c, d]| {
                let vertical = a[0] == b[0];
                let other = c[0] == d[0];
                vertical.cmp(&other).then_with(|| {
                    if vertical {
                        a[0].total_cmp(&c[0])
                            .then(a[1].total_cmp(&c[1]))
                            .then(b[1].total_cmp(&d[1]))
                    } else {
                        a[1].total_cmp(&c[1])
                            .then(a[0].total_cmp(&c[0]))
                            .then(b[0].total_cmp(&d[0]))
                    }
                })
            },
            24,
            fuel,
        )?;
        let mut retained = 0_usize;
        for index in 0..edges.len() {
            let [a, b] = edges[index];
            if retained > 0 {
                let [p, q] = &mut edges[retained - 1];
                let joined = if a[0] == b[0] && p[0] == q[0] && a[0] == p[0] && a[1] <= q[1] {
                    q[1] = q[1].max(b[1]);
                    true
                } else if a[1] == b[1] && p[1] == q[1] && a[1] == p[1] && a[0] <= q[0] {
                    q[0] = q[0].max(b[0]);
                    true
                } else {
                    false
                };
                if joined {
                    continue;
                }
            }
            edges[retained] = [a, b];
            retained += 1;
        }
        edges.truncate(retained);
        Some(Self { edges })
    }

    /// Enclose every point's distance to the TRUE exposed boundary. Unlike
    /// fixed disk witnesses, uncertainty tends to zero with query size.
    pub(super) fn eroded_cell(
        &self,
        source: CellCertificate,
        cell: Cell,
        radius: f64,
        fuel: &mut usize,
    ) -> CellCertificate {
        if source == CellCertificate::Outside {
            return source;
        }
        if charge(fuel, 16).is_none() {
            return CellCertificate::Unknown;
        }
        let Some(r2) = Interval::scalar(radius).and_then(squared_interval) else {
            return CellCertificate::Unknown;
        };
        let mut definitely_far = true;
        for &[a, b] in &self.edges {
            if charge(fuel, 96).is_none() {
                return CellCertificate::Unknown;
            }
            let (perpendicular, along, constant, lo, hi) = if a[0] == b[0] {
                (cell.x, cell.y, a[0], a[1].min(b[1]), a[1].max(b[1]))
            } else {
                (cell.y, cell.x, a[1], a[0].min(b[0]), a[0].max(b[0]))
            };
            let distance = || {
                let normal = squared_interval(perpendicular.sub(Interval::scalar(constant)?)?)?;
                let gap = if along.hi < lo {
                    Interval::scalar(lo)?.sub(along)?
                } else if along.lo > hi {
                    along.sub(Interval::scalar(hi)?)?
                } else {
                    Interval::new(0.0, (lo - along.lo).max(along.hi - hi).max(0.0).next_up())?
                };
                normal.add(squared_interval(gap)?)
            };
            let Some(d2) = distance() else {
                return CellCertificate::Unknown;
            };
            if d2.hi < r2.lo {
                return CellCertificate::Outside;
            }
            definitely_far &= d2.lo > r2.hi;
        }
        let original = if definitely_far && source == CellCertificate::Unknown {
            let mut winding = 0_i64;
            for &[a, b] in &self.edges {
                if charge(fuel, 96).is_none() {
                    return CellCertificate::Unknown;
                }
                let Some(crossing) = certified_line_crossing(cell, a, b) else {
                    return CellCertificate::Unknown;
                };
                let Some(next) = winding.checked_add(crossing) else {
                    return CellCertificate::Unknown;
                };
                winding = next;
            }
            // Arrangement occupancy already resolved original NonZero/EvenOdd
            // semantics. Even parity of its true set boundary is therefore
            // membership, including internal authored union seams.
            if winding % 2 != 0 {
                CellCertificate::Inside
            } else {
                CellCertificate::Outside
            }
        } else {
            source
        };
        if definitely_far && original == CellCertificate::Inside {
            CellCertificate::Inside
        } else if definitely_far && original == CellCertificate::Outside {
            CellCertificate::Outside
        } else {
            CellCertificate::Unknown
        }
    }
}
#[cfg(test)]
mod tests {
    use super::super::interval::exact_add;
    use super::*;
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dFillRule, Render2dOpacity, Render2dPoint,
    };

    fn holed_item(rule: Render2dFillRule) -> Render2dItem {
        let p = |x, y| Render2dPoint::new(x, y).unwrap();
        Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::path(
                    Render2dPath::new(
                        rule,
                        vec![
                            Render2dPathCommand::MoveTo(p(10.0, 10.0)),
                            Render2dPathCommand::LineTo(p(30.0, 10.0)),
                            Render2dPathCommand::LineTo(p(30.0, 30.0)),
                            Render2dPathCommand::LineTo(p(10.0, 30.0)),
                            Render2dPathCommand::Close,
                            Render2dPathCommand::MoveTo(p(18.0, 18.0)),
                            Render2dPathCommand::LineTo(p(22.0, 18.0)),
                            Render2dPathCommand::LineTo(p(22.0, 22.0)),
                            Render2dPathCommand::LineTo(p(18.0, 22.0)),
                            Render2dPathCommand::Close,
                        ],
                    )
                    .unwrap(),
                ),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        )
    }

    fn fake_mesh() -> support::NeutralMesh {
        support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [64.0, 0.0], [0.0, 64.0]],
            bounds: [0.0, 0.0, 64.0, 64.0],
        }
    }

    #[test]
    fn exposed_original_boundary_respects_fill_rule_and_converges_at_curved_erosion() {
        for rule in [Render2dFillRule::EvenOdd, Render2dFillRule::NonZero] {
            let item = holed_item(rule);
            let mut graph = builder();
            let source = graph.source_from_item(fake_mesh(), &item, &[0]).unwrap();
            let mut fuel = 16_777_216;
            let boundary = Boundary::prepare(&source, &mut fuel).unwrap();
            let cell = Cell::new([22.6, 22.6, 22.61, 22.61]).unwrap();
            let membership = source.certify_cell(cell, &mut fuel);
            let expected = if rule == Render2dFillRule::EvenOdd {
                CellCertificate::Outside
            } else {
                CellCertificate::Inside
            };
            assert_eq!(
                boundary.eroded_cell(membership, cell, 1.0, &mut fuel),
                expected
            );
            // The previous .985r witness circle cannot prove this tiny cell
            // just 0.001 inside/outside the true circular erosion boundary.
            let near = 1.0 / 2.0_f64.sqrt();
            for (delta, expected) in [
                (-0.001, CellCertificate::Outside),
                (0.001, CellCertificate::Inside),
            ] {
                let q = 22.0 + near + delta;
                let cell = Cell::new([q, q, q + 0.00001, q + 0.00001]).unwrap();
                let actual = boundary.eroded_cell(
                    source.certify_cell(cell, &mut fuel),
                    cell,
                    1.0,
                    &mut fuel,
                );
                assert_eq!(
                    actual,
                    if rule == Render2dFillRule::EvenOdd {
                        expected
                    } else {
                        CellCertificate::Inside
                    }
                );
            }
        }
    }

    #[test]
    fn source_boundary_admits_continuous_sheared_hole_area_under_existing_fuel() {
        let item = holed_item(Render2dFillRule::EvenOdd);
        let mut graph = builder();
        let source = graph.source_from_item(fake_mesh(), &item, &[0, 0]).unwrap();
        let child = graph
            .effect(
                source,
                Render2dDropShadow::new(8.0, 0.0, 0.0, -1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[0],
            )
            .unwrap();
        let mut fuel = 16_777_216;
        let areas = certified_grid_area_enclosures(
            &child,
            [144, 88, 148, 92],
            4.0,
            Some(Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap()),
            8,
            16,
            &mut fuel,
        )
        .expect("original exposed-boundary refinement must fit existing work admission");
        let low = areas.iter().map(|a| a[0]).sum::<f64>() / 16.0;
        let high = areas.iter().map(|a| a[1]).sum::<f64>() / 16.0;
        let oracle = 0.586_837_842_949_633_9;
        // Independent 110-digit Decimal circle-segment integration. With
        // v=y-22, the boundary is x=35.5+v/4+sqrt(1-v*v). Vertical-edge
        // crossings solve v=4/17*(k +/- sqrt(17-16*k*k)), k=xedge-35.5,
        // with k-v/4>=0. Integrate width using
        // (xright-35.5)*v-v*v/8-(v*sqrt(1-v*v)+asin(v))/2.
        // asin was independently evaluated as 2*atan(v/(1+sqrt(1-v*v)))
        // with 190 reduced alternating terms (remainder below 6e-118).
        // Constants are physical areas, not implementation-owned masks.
        let reference_areas = [
            0.0,
            0.0,
            0.057_316_642_859_978_895,
            0.0625,
            0.0,
            0.000_128_806_734_546_232_42,
            0.057_998_811_660_215_774,
            0.0625,
            0.0,
            0.017_173_013_842_213_225,
            0.0625,
            0.0625,
            0.021_568_309_894_473_65,
            0.057_652_257_958_206_13,
            0.0625,
            0.0625,
        ];
        for (interval, area) in areas.iter().zip(reference_areas) {
            let alpha = area * 16.0;
            assert!(
                interval[0] <= alpha && alpha <= interval[1],
                "{interval:?} vs {alpha}"
            );
        }
        assert!(
            low <= oracle && oracle <= high,
            "[{low},{high}] vs {oracle}"
        );
        assert!(high - low < 0.005, "unresolved area [{low},{high}]");
        assert!(fuel > 0);
    }

    #[test]
    fn source_boundary_declines_unrepresentable_events_and_exhausted_preparation() {
        assert!(exact_add(1.0, 2.0_f64.powi(-54)).is_none());
        let item = holed_item(Render2dFillRule::EvenOdd);
        let mut graph = builder();
        let source = graph.source_from_item(fake_mesh(), &item, &[0]).unwrap();
        let mut fuel = 1;
        assert!(Boundary::prepare(&source, &mut fuel).is_none());
        assert_eq!(fuel, 0);
    }

    #[test]
    fn source_boundary_whole_target_continuous_area_feasibility() {
        let item = holed_item(Render2dFillRule::EvenOdd);
        let mut graph = builder();
        let source = graph.source_from_item(fake_mesh(), &item, &[0, 0]).unwrap();
        let child = graph
            .effect(
                Arc::clone(&source),
                Render2dDropShadow::new(8.0, 0.0, 0.0, -1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[0],
            )
            .unwrap();
        let completed = graph.union(vec![source, child], &[0]).unwrap();
        let projected = graph
            .affine(
                completed,
                Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap(),
                &[0],
            )
            .unwrap();
        let parent = graph
            .effect(
                projected,
                Render2dDropShadow::new(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    Render2dColorRgba8::new(0, 0, 255, 255),
                )
                .unwrap(),
                &[0],
            )
            .unwrap();
        let mut fuel = 16_777_216;
        let areas =
            certified_grid_area_query(&parent, [0, 0, 256, 256], 4.0, None, 8, 65_536, &mut fuel);
        assert!(
            areas.is_ok(),
            "whole-target proof exhausted budget, fuel={fuel}"
        );
        let areas = areas.unwrap();
        assert!(areas.iter().all(|a| a[1] - a[0] <= 1.0 / (2048.0 * 256.0)));
        let mut lower = 0.0;
        let mut upper = 0.0;
        for sy in 88..92 {
            for sx in 144..148 {
                let a = areas[sy * 256 + sx];
                lower += a[0] / 16.0;
                upper += a[1] / 16.0;
            }
        }
        assert!(lower <= 0.586_837_842_949_633_9 && upper >= 0.586_837_842_949_633_9);
    }

    #[test]
    fn original_union_seam_is_not_an_exposed_erosion_boundary() {
        let item = |x| {
            Render2dItem::new(
                Render2dPrimitive::Fill {
                    shape: Render2dShape::rect(Render2dRect::new(x, 10.0, 10.0, 20.0).unwrap()),
                    brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
                },
                Render2dAffineTransform::IDENTITY,
                Vec::new(),
                Render2dOpacity::TRANSPARENT,
            )
        };
        let a = item(10.0);
        let b = item(20.0);
        let mut graph = builder();
        let first = graph.source_from_item(fake_mesh(), &a, &[0, 0]).unwrap();
        let second = graph.source_from_item(fake_mesh(), &b, &[0, 1]).unwrap();
        let source = graph.union(vec![first, second], &[0]).unwrap();
        let child = graph
            .effect(
                Arc::clone(&source),
                Render2dDropShadow::new(30.0, 0.0, 0.0, -1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[0],
            )
            .unwrap();
        let completed = graph.union(vec![source, child], &[0]).unwrap();
        let mut fuel = 16_777_216;
        let areas = certified_grid_area_enclosures(
            &completed,
            [78, 78, 82, 82],
            4.0,
            None,
            8,
            16,
            &mut fuel,
        )
        .expect("pristine union seam must remain fully occupied without shifted E");
        assert!(areas.iter().all(|a| a[0] > 0.999_999 && a[1] == 1.0));
        let areas = certified_grid_area_enclosures(
            &completed,
            [198, 78, 202, 82],
            4.0,
            None,
            8,
            16,
            &mut fuel,
        )
        .expect("distance to internal authored seam cannot erase shifted E");
        assert!(areas.iter().all(|a| a[0] > 0.999_999 && a[1] == 1.0));
        assert!(
            certified_grid_area_enclosures(
                &completed,
                [198, 78, 202, 82],
                3.0,
                None,
                8,
                16,
                &mut fuel,
            )
            .is_none(),
            "unrepresentable lattice division is not an exact event certificate"
        );
    }
}
