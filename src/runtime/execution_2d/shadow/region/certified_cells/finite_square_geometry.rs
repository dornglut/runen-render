//! Test-only complete candidate boundary cover for orthogonal disk erosion
//! followed by a finite square halo. Candidates never supply membership.
//! Original exposed source edges supply capsule branches. All their possible
//! vertices and axis extrema supply the square's missing extrusion segments.
//!
//! At a boundary point of E+square, a source witness cannot pair with the
//! square's interior. A square corner gives a translated source branch. On
//! the interior of a square edge, a regular nonparallel source tangent would
//! span an open neighbourhood, contradicting boundary status. Remaining
//! witnesses are axis-normal extrema, vertices, or parallel straight branches;
//! their extrusion is covered by connectors and corner-translated branches.
//! False branches from conservative intersections and merged domains are
//! harmless only because they partition original-source membership queries.
use super::super::interval::{exact_add, exact_mul};
use super::conic_integral::sqrt_enclosure;
use super::orthogonal_boundary::Boundary;
use super::*;

const MAX_LINES: usize = 1_024;
const MAX_CIRCLES: usize = 128;
const MAX_VERTICES: usize = 4_096;

#[derive(Clone, Copy, Debug)]
pub(super) struct Line {
    /// Constant local-coordinate axis supplied by construction provenance.
    pub(super) axis: usize,
    pub(super) ends: [[Interval; 2]; 2],
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Circle {
    pub(super) center: [Interval; 2],
    pub(super) radius: f64,
    /// None is a full source capsule circle. Translated candidates keep this
    /// original radial quadrant, derived from the eroded set's outward normal.
    pub(super) quadrant: Option<[f64; 2]>,
}

pub(super) struct Inventory {
    pub(super) lines: Vec<Line>,
    pub(super) circles: Vec<Circle>,
    pub(super) vertices: Vec<[Interval; 2]>,
}

fn charge(fuel: &mut usize, work: usize) -> Option<()> {
    if *fuel < work {
        *fuel = 0;
        None
    } else {
        *fuel -= work;
        Some(())
    }
}

// Exact scalar identities reduce duplicate event geometry. These wrappers
// delegate to the canonical numerical authority; other cases stay intervals.
fn add(a: Interval, b: Interval) -> Option<Interval> {
    if a.lo == a.hi
        && b.lo == b.hi
        && let Some(value) = exact_add(a.lo, b.lo)
    {
        return Interval::scalar(value);
    }
    a.add(b)
}
fn neg(a: Interval) -> Option<Interval> {
    Interval::new(-a.hi, -a.lo)
}
fn sub(a: Interval, b: Interval) -> Option<Interval> {
    add(a, neg(b)?)
}
fn mul(a: Interval, b: Interval) -> Option<Interval> {
    if a.lo == a.hi
        && b.lo == b.hi
        && let Some(value) = exact_mul(a.lo, b.lo)
    {
        return Interval::scalar(value);
    }
    a.mul(b)
}
fn scalar(a: f64) -> Option<Interval> {
    Interval::scalar(a)
}
fn square(a: Interval) -> Option<Interval> {
    if a.lo == a.hi {
        mul(a, a)
    } else {
        squared_interval(a)
    }
}
fn radical(a: Interval) -> Option<Interval> {
    if a.hi < 0.0 {
        return None;
    }
    let positive = Interval::new(a.lo.max(0.0), a.hi)?;
    if positive.lo == positive.hi {
        let value = positive.lo.sqrt();
        if exact_mul(value, value) == Some(positive.lo) {
            return scalar(value);
        }
    }
    sqrt_enclosure(positive)
}

impl Inventory {
    fn away_from_eroded_boundary(
        boundary: &Boundary,
        cell: Cell,
        radius: f64,
        fuel: &mut usize,
    ) -> Option<bool> {
        let original = boundary.classify_original(cell, fuel);
        let result = boundary.eroded_cell(original, cell, radius, fuel);
        if *fuel == 0 {
            return None;
        }
        // Both strict interior and strict exterior contain no E boundary.
        // eroded_cell uses strict distance separation; equality is Unknown.
        Some(result != CellCertificate::Unknown)
    }

    /// Drop a full circle only after a complete interval cover proves every
    /// branch point is strictly inside or outside the ORIGINAL eroded set.
    /// A missed or uncertain piece retains the circle. This removes no part
    /// of its boundary and never infers absence from sampled membership.
    fn circle_away_from_eroded_boundary(
        boundary: &Boundary,
        circle: Circle,
        radius: f64,
        fuel: &mut usize,
    ) -> Option<bool> {
        const PARTS: usize = 16;
        let mut low = -circle.radius;
        for index in 0..PARTS {
            charge(fuel, 512)?;
            let high = if index + 1 == PARTS {
                circle.radius
            } else {
                -circle.radius + (2.0 * circle.radius / PARTS as f64) * (index + 1) as f64
            };
            if !high.is_finite() || low >= high || high > circle.radius {
                return Some(false);
            }
            let y = Interval::new(low, high)?;
            let Some(x) = sub(square(scalar(circle.radius)?)?, square(y)?).and_then(radical) else {
                return Some(false);
            };
            for sign in [-1.0, 1.0] {
                charge(fuel, 256)?;
                let cell = Cell {
                    x: add(circle.center[0], mul(scalar(sign)?, x)?)?,
                    y: add(circle.center[1], y)?,
                };
                if !Self::away_from_eroded_boundary(boundary, cell, radius, fuel)? {
                    return Some(false);
                }
            }
            low = high;
        }
        Some(true)
    }

    fn retain_possible_boundary(
        &mut self,
        boundary: &Boundary,
        radius: f64,
        fuel: &mut usize,
    ) -> Option<()> {
        let mut retained = 0;
        for index in 0..self.vertices.len() {
            charge(fuel, 64)?;
            let point = self.vertices[index];
            if !Self::away_from_eroded_boundary(
                boundary,
                Cell {
                    x: point[0],
                    y: point[1],
                },
                radius,
                fuel,
            )? {
                self.vertices[retained] = point;
                retained += 1;
            }
        }
        self.vertices.truncate(retained);
        let mut retained = 0;
        for index in 0..self.lines.len() {
            charge(fuel, 128)?;
            let line = self.lines[index];
            let cell = Cell {
                x: Interval::new(
                    line.ends[0][0].lo.min(line.ends[1][0].lo),
                    line.ends[0][0].hi.max(line.ends[1][0].hi),
                )?,
                y: Interval::new(
                    line.ends[0][1].lo.min(line.ends[1][1].lo),
                    line.ends[0][1].hi.max(line.ends[1][1].hi),
                )?,
            };
            if !Self::away_from_eroded_boundary(boundary, cell, radius, fuel)? {
                self.lines[retained] = line;
                retained += 1;
            }
        }
        self.lines.truncate(retained);
        let mut retained = 0;
        for index in 0..self.circles.len() {
            charge(fuel, 64)?;
            let circle = self.circles[index];
            if !Self::circle_away_from_eroded_boundary(boundary, circle, radius, fuel)? {
                self.circles[retained] = circle;
                retained += 1;
            }
        }
        self.circles.truncate(retained);
        Some(())
    }

    /// Both source and expanded candidate inventories coexist during build.
    /// No per-pair temporary vector or library sort allocation is present.
    pub(super) fn preparation_scratch_bound() -> Option<usize> {
        MAX_LINES
            .checked_mul(std::mem::size_of::<Line>())?
            .checked_add(MAX_CIRCLES.checked_mul(std::mem::size_of::<Circle>())?)?
            .checked_add(MAX_VERTICES.checked_mul(std::mem::size_of::<[Interval; 2]>())?)?
            .checked_add(std::mem::size_of::<Self>())?
            .checked_mul(2)
    }
    fn empty() -> Option<Self> {
        Some(Self {
            lines: bounded_vector(MAX_LINES)?,
            circles: bounded_vector(MAX_CIRCLES)?,
            vertices: bounded_vector(MAX_VERTICES)?,
        })
    }
    fn line(&mut self, axis: usize, ends: [[Interval; 2]; 2], fuel: &mut usize) -> Option<()> {
        charge(fuel, 128 + self.lines.len().checked_mul(32)?)?;
        // An exact coordinate establishes actual line identity. Combining
        // domains may introduce false branches across gaps, which is safe:
        // membership remains original-source-derived and resolves every strip.
        if axis > 1 {
            return None;
        }
        if ends[0][axis].lo == ends[0][axis].hi {
            for old in &mut self.lines {
                if old.axis == axis
                    && old.ends[0][axis].lo == ends[0][axis].lo
                    && old.ends[0][axis].hi == ends[0][axis].lo
                    && old.ends[1][axis].lo == ends[0][axis].lo
                    && old.ends[1][axis].hi == ends[0][axis].lo
                {
                    let other = 1 - axis;
                    old.ends[0][other] = Interval::new(
                        old.ends[0][other].lo.min(ends[0][other].lo),
                        old.ends[0][other].hi.min(ends[0][other].hi),
                    )?;
                    old.ends[1][other] = Interval::new(
                        old.ends[1][other].lo.max(ends[1][other].lo),
                        old.ends[1][other].hi.max(ends[1][other].hi),
                    )?;
                    return Some(());
                }
            }
        }
        if self.lines.len() == MAX_LINES {
            return None;
        }
        self.lines.push(Line { axis, ends });
        Some(())
    }
    fn circle(
        &mut self,
        center: [Interval; 2],
        radius: f64,
        quadrant: Option<[f64; 2]>,
        fuel: &mut usize,
    ) -> Option<()> {
        charge(fuel, 128 + self.circles.len().checked_mul(32)?)?;
        if center.iter().all(|v| v.lo == v.hi)
            && self.circles.iter().any(|old| {
                old.radius == radius
                    && old.quadrant == quadrant
                    && (0..2).all(|a| {
                        old.center[a].lo == center[a].lo && old.center[a].hi == center[a].lo
                    })
            })
        {
            return Some(());
        }
        if self.circles.len() == MAX_CIRCLES {
            return None;
        }
        self.circles.push(Circle {
            center,
            radius,
            quadrant,
        });
        Some(())
    }
    fn vertex(&mut self, point: [Interval; 2], fuel: &mut usize) -> Option<()> {
        charge(fuel, 64 + self.vertices.len().checked_mul(16)?)?;
        if point.iter().all(|v| v.lo == v.hi)
            && self
                .vertices
                .iter()
                .any(|old| (0..2).all(|a| old[a].lo == point[a].lo && old[a].hi == point[a].lo))
        {
            return Some(());
        }
        if self.vertices.len() == MAX_VERTICES {
            return None;
        }
        self.vertices.push(point);
        Some(())
    }
    fn line_circle_vertices(&mut self, line: Line, circle: Circle, fuel: &mut usize) -> Option<()> {
        charge(fuel, 512)?;
        let axis = line.axis;
        let other = 1 - axis;
        let coordinate = line.ends[0][axis];
        let d = sub(coordinate, circle.center[axis])?;
        let discriminant = sub(square(scalar(circle.radius)?)?, square(d)?)?;
        if discriminant.hi < 0.0 {
            return Some(());
        }
        let height = radical(discriminant)?;
        for sign in [-1.0, 1.0] {
            let along = add(circle.center[other], mul(scalar(sign)?, height)?)?;
            if along.hi < line.ends[0][other].lo || along.lo > line.ends[1][other].hi {
                continue;
            }
            let mut p = circle.center;
            p[axis] = coordinate;
            p[other] = along;
            self.vertex(p, fuel)?;
        }
        Some(())
    }
    fn circle_vertices(&mut self, left: Circle, right: Circle, fuel: &mut usize) -> Option<()> {
        charge(fuel, 1_536)?;
        // The initial inventory has one common scalar disk radius. Distinct
        // original centers are scalar and cannot create concentric ambiguity.
        if left.radius != right.radius {
            return None;
        }
        let dx = sub(right.center[0], left.center[0])?;
        let dy = sub(right.center[1], left.center[1])?;
        let d2 = add(square(dx)?, square(dy)?)?;
        if d2.lo <= 0.0 {
            return None;
        }
        let height2 = sub(square(scalar(left.radius)?)?, mul(scalar(0.25)?, d2)?)?;
        if height2.hi < 0.0 {
            return Some(());
        }
        let factor = radical(height2)?.div(radical(d2)?)?;
        let middle = [
            mul(add(left.center[0], right.center[0])?, scalar(0.5)?)?,
            mul(add(left.center[1], right.center[1])?, scalar(0.5)?)?,
        ];
        for sign in [-1.0, 1.0] {
            let scale = mul(scalar(sign)?, factor)?;
            self.vertex(
                [
                    sub(middle[0], mul(dy, scale)?)?,
                    add(middle[1], mul(dx, scale)?)?,
                ],
                fuel,
            )?;
        }
        Some(())
    }
    fn complete_vertices(&mut self, fuel: &mut usize) -> Option<()> {
        for index in 0..self.lines.len() {
            let line = self.lines[index];
            self.vertex(line.ends[0], fuel)?;
            self.vertex(line.ends[1], fuel)?;
            for j in index + 1..self.lines.len() {
                charge(fuel, 128)?;
                let other = self.lines[j];
                let vertical = line.axis == 0;
                let other_vertical = other.axis == 0;
                if vertical == other_vertical {
                    continue;
                }
                let (v, h) = if vertical {
                    (line, other)
                } else {
                    (other, line)
                };
                let p = [v.ends[0][0], h.ends[0][1]];
                if p[0].hi < h.ends[0][0].lo
                    || p[0].lo > h.ends[1][0].hi
                    || p[1].hi < v.ends[0][1].lo
                    || p[1].lo > v.ends[1][1].hi
                {
                    continue;
                }
                self.vertex(p, fuel)?;
            }
            for j in 0..self.circles.len() {
                self.line_circle_vertices(line, self.circles[j], fuel)?;
            }
        }
        for index in 0..self.circles.len() {
            let circle = self.circles[index];
            for axis in 0..2 {
                for sign in [-1.0, 1.0] {
                    charge(fuel, 64)?;
                    let mut p = circle.center;
                    p[axis] = add(p[axis], scalar(sign * circle.radius)?)?;
                    self.vertex(p, fuel)?;
                }
            }
            for j in index + 1..self.circles.len() {
                self.circle_vertices(circle, self.circles[j], fuel)?;
            }
        }
        Some(())
    }
    pub(super) fn prepare(
        boundary: &Boundary,
        radius: f64,
        halo: Interval,
        fuel: &mut usize,
    ) -> Option<Self> {
        charge(fuel, 512)?;
        if !radius.is_finite() || radius <= 0.0 || halo.lo < 0.0 {
            return None;
        }
        let mut source = Self::empty()?;
        for &[p, q] in boundary.edges() {
            let axis = if p[0] == q[0] { 0 } else { 1 };
            for sign in [-1.0, 1.0] {
                charge(fuel, 128)?;
                let mut a = p.map(|v| scalar(v).unwrap());
                let mut b = q.map(|v| scalar(v).unwrap());
                a[axis] = add(a[axis], scalar(sign * radius)?)?;
                b[axis] = add(b[axis], scalar(sign * radius)?)?;
                source.line(axis, [a, b], fuel)?;
            }
            source.circle(p.map(|v| scalar(v).unwrap()), radius, None, fuel)?;
            source.circle(q.map(|v| scalar(v).unwrap()), radius, None, fuel)?;
        }
        source.complete_vertices(fuel)?;
        source.retain_possible_boundary(boundary, radius, fuel)?;
        let mut result = Self::empty()?;
        for vertex in &source.vertices {
            for sign in [-1.0, 1.0] {
                charge(fuel, 256)?;
                let signed = mul(scalar(sign)?, halo)?;
                let x = add(vertex[0], signed)?;
                let y = add(vertex[1], signed)?;
                result.line(
                    0,
                    [[x, sub(vertex[1], halo)?], [x, add(vertex[1], halo)?]],
                    fuel,
                )?;
                result.line(
                    1,
                    [[sub(vertex[0], halo)?, y], [add(vertex[0], halo)?, y]],
                    fuel,
                )?;
            }
        }
        for xsign in [-1.0, 1.0] {
            for ysign in [-1.0, 1.0] {
                let shift = [mul(scalar(xsign)?, halo)?, mul(scalar(ysign)?, halo)?];
                for line in &source.lines {
                    charge(fuel, 128)?;
                    let point =
                        |p: [Interval; 2]| Some([add(p[0], shift[0])?, add(p[1], shift[1])?]);
                    result.line(
                        line.axis,
                        [point(line.ends[0])?, point(line.ends[1])?],
                        fuel,
                    )?;
                }
                for circle in &source.circles {
                    charge(fuel, 128)?;
                    result.circle(
                        [
                            add(circle.center[0], shift[0])?,
                            add(circle.center[1], shift[1])?,
                        ],
                        radius,
                        // E is outside each capsule endpoint disk. An exposed
                        // arc's outward normal is inward radial, selecting the
                        // opposite radial quadrant for this square corner.
                        // Axis-normal and vertex cases retain all connectors.
                        Some([-xsign, -ysign]),
                        fuel,
                    )?;
                }
            }
        }
        result.vertices = source.vertices;
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::{
        Render2dBrush, Render2dColorRgba8, Render2dFillRule, Render2dOpacity, Render2dPoint,
    };

    fn fixture(two_holes: bool) -> Render2dItem {
        let rectangles = if two_holes {
            vec![
                [-10.0, -10.0, 10.0, 10.0],
                [-1.5, -1.0, -0.5, 0.0],
                [0.5, -1.0, 1.5, 0.0],
            ]
        } else {
            vec![[10.0, 10.0, 30.0, 30.0], [18.0, 18.0, 22.0, 22.0]]
        };
        let mut commands = Vec::new();
        for [x0, y0, x1, y1] in rectangles {
            for (index, [x, y]) in [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
                .into_iter()
                .enumerate()
            {
                let p = Render2dPoint::new(x, y).unwrap();
                commands.push(if index == 0 {
                    Render2dPathCommand::MoveTo(p)
                } else {
                    Render2dPathCommand::LineTo(p)
                });
            }
            commands.push(Render2dPathCommand::Close);
        }
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
    }
    fn mesh() -> support::NeutralMesh {
        support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [64.0, 0.0], [0.0, 64.0]],
            bounds: [0.0, 0.0, 64.0, 64.0],
        }
    }
    #[test]
    fn finite_square_inventory_contains_circle_intersection_connectors() {
        let item = fixture(true);
        let mut graph = builder();
        let source = graph.source_from_item(mesh(), &item, &[0]).unwrap();
        let mut fuel = 16_777_216;
        let boundary = Boundary::prepare(&source, &mut fuel).unwrap();
        let before = fuel;
        let inventory =
            Inventory::prepare(&boundary, 1.0, scalar(0.09375).unwrap(), &mut fuel).unwrap();
        // Independent sqrt(3)/2 enclosure; the actual positive vertex is
        // approximately .86602540378443864676372317075293618.
        let y = 0.866_025_403_784_438_6;
        assert!(
            inventory
                .vertices
                .iter()
                .any(|p| p[0].lo <= 0.0 && 0.0 <= p[0].hi && p[1].lo <= y && y <= p[1].hi)
        );
        let plane = 0.772_275_403_784_438_6;
        assert!(
            inventory.lines.iter().any(|line| {
                let [a, b] = line.ends;
                a[1].lo <= plane
                    && plane <= a[1].hi
                    && b[1].lo <= plane
                    && plane <= b[1].hi
                    && a[0].lo <= -0.09375
                    && b[0].hi >= 0.09375
            }),
            "true noncardinal vertex extrusion must be present"
        );
        eprintln!(
            "two-hole inventory: {}lines {}circles {}vertices; preparationwork{}",
            inventory.lines.len(),
            inventory.circles.len(),
            inventory.vertices.len(),
            before - fuel
        );
    }
    #[test]
    fn finite_square_ring_inventory_keeps_true_hole_circles_and_discards_inactive_outer_arcs() {
        let item = fixture(false);
        let mut graph = builder();
        let source = graph.source_from_item(mesh(), &item, &[0]).unwrap();
        let mut fuel = 16_777_216;
        let boundary = Boundary::prepare(&source, &mut fuel).unwrap();
        let before = fuel;
        let inventory =
            Inventory::prepare(&boundary, 1.0, scalar(0.09375).unwrap(), &mut fuel).unwrap();
        for [x, y] in [[18.0, 18.0], [22.0, 18.0], [18.0, 22.0], [22.0, 22.0]] {
            for dx in [-0.09375, 0.09375] {
                for dy in [-0.09375, 0.09375] {
                    assert!(inventory.circles.iter().any(|c| c.center[0].lo == x + dx
                        && c.center[0].hi == x + dx
                        && c.center[1].lo == y + dy
                        && c.center[1].hi == y + dy
                        && c.radius == 1.0));
                }
            }
        }
        // A unit circle about an outer corner never reaches the eroded
        // rectangle's required x/y margins of one simultaneously. Its entire
        // interval cover is outside E; the sharp offset-line vertex remains.
        assert!(
            inventory
                .circles
                .iter()
                .all(|c| c.center[0].lo > 17.0 && c.center[0].hi < 23.0)
        );
        let branches = inventory.lines.len() + 2 * inventory.circles.len();
        assert!(
            branches * branches <= 65_536,
            "unchanged scanner matrix admission"
        );
        assert!(
            before - fuel < 1_000_000,
            "inventory alone must leave meaningful query fuel"
        );
        eprintln!(
            "ring inventory: {}lines {}circles {}vertices; {}maxbranches; preparationwork{}",
            inventory.lines.len(),
            inventory.circles.len(),
            inventory.vertices.len(),
            branches,
            before - fuel
        );
    }
    #[test]
    fn finite_square_inventory_preserves_halo_uncertainty_and_exhaustion() {
        let item = fixture(false);
        let mut graph = builder();
        let source = graph.source_from_item(mesh(), &item, &[0]).unwrap();
        let mut fuel = 16_777_216;
        let boundary = Boundary::prepare(&source, &mut fuel).unwrap();
        let result = Inventory::prepare(
            &boundary,
            1.0,
            Interval::new(0.09, 0.09375).unwrap(),
            &mut fuel,
        )
        .unwrap();
        assert!(
            result
                .circles
                .iter()
                .all(|c| c.center.iter().any(|v| v.lo < v.hi))
        );
        let mut fuel = 511;
        assert!(Inventory::prepare(&boundary, 1.0, scalar(0.09375).unwrap(), &mut fuel).is_none());
        assert_eq!(fuel, 0);
    }
}
