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
    ) -> Result<[f64; 2], AreaQueryError> {
        let work = transforms
            .len()
            .checked_add(1)
            .and_then(|count| count.checked_mul(64))
            .ok_or(AreaQueryError::Resource)?;
        charge(fuel, work).ok_or(AreaQueryError::Resource)?;
        for transform in transforms.iter().rev() {
            let [a, b, c, d, tx, ty] = transform.components();
            // Exact signed axis permutations and translations. Oblique or
            // scaled source arrangements require a different boundary proof.
            let sign = |v: f64| v == 1.0 || v == -1.0;
            p = if sign(a) && b == 0.0 && c == 0.0 && sign(d) {
                [
                    exact_add(a * p[0], tx).ok_or(AreaQueryError::Precision)?,
                    exact_add(d * p[1], ty).ok_or(AreaQueryError::Precision)?,
                ]
            } else if a == 0.0 && sign(b) && sign(c) && d == 0.0 {
                [
                    exact_add(c * p[1], tx).ok_or(AreaQueryError::Precision)?,
                    exact_add(b * p[0], ty).ok_or(AreaQueryError::Precision)?,
                ]
            } else {
                return Err(AreaQueryError::Unsupported);
            };
        }
        if self.x.len() >= MAX_EVENTS
            || self.x.len() == self.x.capacity()
            || self.y.len() == self.y.capacity()
        {
            return Err(AreaQueryError::Resource);
        }
        self.x.push(p[0]);
        self.y.push(p[1]);
        Ok(p)
    }

    fn shape(
        &mut self,
        shape: &Render2dShape,
        transforms: &[Render2dAffineTransform],
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        if let Some(rect) = shape.as_rect() {
            charge(fuel, 64).ok_or(AreaQueryError::Resource)?;
            let x1 = exact_add(rect.x(), rect.width()).ok_or(AreaQueryError::Precision)?;
            let y1 = exact_add(rect.y(), rect.height()).ok_or(AreaQueryError::Precision)?;
            self.point([rect.x(), rect.y()], transforms, fuel)?;
            self.point([x1, y1], transforms, fuel)?;
            return Ok(());
        }
        let path = shape.as_path().ok_or(AreaQueryError::Unsupported)?;
        charge(
            fuel,
            path.commands()
                .len()
                .checked_mul(8)
                .ok_or(AreaQueryError::Resource)?,
        )
        .ok_or(AreaQueryError::Resource)?;
        let mut first = None;
        let mut previous = None;
        let mut edge = |a: [f64; 2], b: [f64; 2]| -> Result<(), AreaQueryError> {
            charge(fuel, 4).ok_or(AreaQueryError::Resource)?;
            (a[0] == b[0] || a[1] == b[1])
                .then_some(())
                .ok_or(AreaQueryError::Unsupported)
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
                    edge(previous.ok_or(AreaQueryError::Unsupported)?, point)?;
                    previous = Some(point);
                }
                Render2dPathCommand::Close => {
                    if let (Some(a), Some(b)) = (previous, first) {
                        edge(a, b)?;
                    }
                    previous = first;
                }
                _ => return Err(AreaQueryError::Unsupported),
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
        Ok(())
    }

    fn clips(
        &mut self,
        clips: &[Render2dClip],
        transforms: &mut Vec<Render2dAffineTransform>,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        for clip in clips {
            if transforms.len() == transforms.capacity() {
                return Err(AreaQueryError::Resource);
            }
            transforms.push(clip.clip_to_parent());
            self.shape(clip.shape(), transforms, fuel)?;
            transforms.pop();
        }
        Ok(())
    }

    fn region(
        &mut self,
        region: &NeutralRegion<'_>,
        transforms: &mut Vec<Render2dAffineTransform>,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        charge(fuel, 1).ok_or(AreaQueryError::Resource)?;
        match &region.kind {
            Kind::Source(_) => {
                let item = region.semantic_item.ok_or(AreaQueryError::Unsupported)?;
                let Render2dPrimitive::Fill { shape, .. } = item.primitive() else {
                    return Err(AreaQueryError::Unsupported);
                };
                if transforms.len() == transforms.capacity() {
                    return Err(AreaQueryError::Resource);
                }
                transforms.push(item.local_to_parent());
                self.shape(shape, transforms, fuel)?;
                transforms.pop();
                self.clips(item.clips(), transforms, fuel)
            }
            Kind::Union(children) => {
                for child in children {
                    self.region(child, transforms, fuel)?;
                }
                Ok(())
            }
            Kind::Affine { source, to_parent } => {
                if transforms.len() == transforms.capacity() {
                    return Err(AreaQueryError::Resource);
                }
                transforms.push(*to_parent);
                self.region(source, transforms, fuel)?;
                transforms.pop();
                Ok(())
            }
            Kind::Clip { source, clips } => {
                self.region(source, transforms, fuel)?;
                self.clips(clips, transforms, fuel)
            }
            Kind::Effect { .. } => Err(AreaQueryError::Unsupported),
        }
    }
}

pub(super) struct Boundary {
    edges: Vec<[[f64; 2]; 2]>,
}

impl Boundary {
    /// Each fallibly reserved edge vector has capacity at most MAX_EDGES.
    /// Include the retained Boundary and Arc control block once per allocation.
    pub(super) fn retained_allocation_bound() -> Option<usize> {
        MAX_EDGES
            .checked_mul(std::mem::size_of::<[[f64; 2]; 2]>())?
            .checked_add(std::mem::size_of::<Self>())?
            .checked_add(2 * std::mem::size_of::<usize>())
    }

    /// Preparation keeps event coordinates, arrangement occupancy and the
    /// authored transform stack concurrently with its eventual edge vector.
    /// The edge allocation itself belongs to retained_allocation_bound.
    pub(super) fn preparation_scratch_bound(max_depth: usize) -> Option<usize> {
        if max_depth > MAX_REGION_DEPTH {
            return None;
        }
        let events = 2_usize
            .checked_mul(MAX_EVENTS)?
            .checked_mul(std::mem::size_of::<f64>())?;
        let occupied = MAX_CELLS.checked_mul(std::mem::size_of::<bool>())?;
        let transforms = max_depth
            .checked_add(2)?
            .checked_mul(std::mem::size_of::<Render2dAffineTransform>())?;
        events
            .checked_add(occupied)?
            .checked_add(transforms)?
            .checked_add(std::mem::size_of::<Events>())?
            .checked_add(2 * std::mem::size_of::<Vec<usize>>())
    }

    /// Geometric prerequisite for finite Gaussian support. The
    /// prepared boundary, rather than a sampled mask, retains original F1.
    /// A common displacement proves every point of Q belongs to E + square;
    /// only a complete Outside covering of Q + square proves absence.
    pub(super) fn square_dilated_eroded_cell(
        &self,
        cell: Cell,
        radius: f64,
        halo: Interval,
        fuel: &mut usize,
    ) -> CellCertificate {
        const MAX_DEPTH: u8 = 16;
        const MAX_NODES: usize = 4_096;
        const MAX_FRONTIER: usize = 17;
        // Includes validation, the fixed stack initialization and counters.
        if charge(fuel, 256).is_none()
            || !radius.is_finite()
            || radius < 0.0
            || !halo.lo.is_finite()
            || !halo.hi.is_finite()
            || halo.lo < 0.0
            || halo.lo > halo.hi
        {
            return CellCertificate::Unknown;
        }
        // Bulk geometry is usually proved with one zero-displacement witness
        // or one whole expanded-domain exclusion. Do these before the other
        // eight witnesses, and reuse the expanded certificate in the DFS.
        if charge(fuel, 64).is_none() {
            return CellCertificate::Unknown;
        }
        let original = self.classify_original(cell, fuel);
        if self.eroded_cell(original, cell, radius, fuel) == CellCertificate::Inside {
            return CellCertificate::Inside;
        }
        if charge(fuel, 64).is_none() {
            return CellCertificate::Unknown;
        }
        let Some(expanded) = cell.expanded(halo.hi) else {
            return CellCertificate::Unknown;
        };
        let original = self.classify_original(expanded, fuel);
        let expanded_eroded = self.eroded_cell(original, expanded, radius, fuel);
        if expanded_eroded != CellCertificate::Unknown {
            return expanded_eroded;
        }
        let offsets = [0.0, -halo.lo, halo.lo];
        let count = if halo.lo == 0.0 { 1 } else { 3 };
        for &x in &offsets[..count] {
            for &y in &offsets[..count] {
                if x == 0.0 && y == 0.0 {
                    continue;
                }
                if charge(fuel, 64).is_none() {
                    return CellCertificate::Unknown;
                }
                // shift subtracts its arguments. Outward rounding encloses
                // the entire translated Q for this one admissible witness.
                let Some(translated) = cell.shift(-x, -y) else {
                    return CellCertificate::Unknown;
                };
                let original = self.classify_original(translated, fuel);
                if self.eroded_cell(original, translated, radius, fuel) == CellCertificate::Inside {
                    return CellCertificate::Inside;
                }
                if *fuel == 0 {
                    return CellCertificate::Unknown;
                }
            }
        }
        if charge(fuel, 64).is_none() {
            return CellCertificate::Unknown;
        }
        // hi proves absence for every admissible halo; lo above supplies
        // witnesses valid for every admissible halo. Uncertainty is retained.
        let mut frontier = [None; MAX_FRONTIER];
        frontier[0] = Some((expanded, 0_u8));
        let mut length = 1_usize;
        let mut visited = 0_usize;
        let mut complete_outside = true;
        while length > 0 {
            if visited == MAX_NODES || charge(fuel, 32).is_none() {
                return CellCertificate::Unknown;
            }
            visited += 1;
            length -= 1;
            let (part, depth) = frontier[length].take().unwrap();
            let eroded = if visited == 1 {
                expanded_eroded
            } else {
                let original = self.classify_original(part, fuel);
                self.eroded_cell(original, part, radius, fuel)
            };
            match eroded {
                CellCertificate::Outside => continue,
                CellCertificate::Inside | CellCertificate::Unknown => {}
            }
            if charge(fuel, 128).is_none() {
                return CellCertificate::Unknown;
            }
            // A part midpoint selects one displacement, not membership.
            // Clamping keeps this actual f64 witness within every admissible
            // halo. Independently certify the whole translated original Q.
            let displacement = |q: Interval, p: Interval| {
                let value = (p.lo * 0.5 + p.hi * 0.5) - (q.lo * 0.5 + q.hi * 0.5);
                value.clamp(-halo.lo, halo.lo)
            };
            let x = displacement(cell.x, part.x);
            let y = displacement(cell.y, part.y);
            let Some(translated) = cell.shift(-x, -y) else {
                return CellCertificate::Unknown;
            };
            let original = self.classify_original(translated, fuel);
            if self.eroded_cell(original, translated, radius, fuel) == CellCertificate::Inside {
                return CellCertificate::Inside;
            }
            if eroded == CellCertificate::Inside || depth == MAX_DEPTH {
                // Do not abandon remaining parts: another part can supply an
                // Inside witness. This leaf permanently defeats Outside.
                complete_outside = false;
                continue;
            }
            // Covers both complete Cell copies, interval validation, all
            // midpoint operations and the two fixed-stack entry writes.
            if length + 2 > MAX_FRONTIER || charge(fuel, 256).is_none() {
                return CellCertificate::Unknown;
            }
            let width = part.x.hi - part.x.lo;
            let height = part.y.hi - part.y.lo;
            if !width.is_finite() || !height.is_finite() {
                return CellCertificate::Unknown;
            }
            let mut left = part;
            let mut right = part;
            let interval = if width >= height { part.x } else { part.y };
            let midpoint = interval.lo * 0.5 + interval.hi * 0.5;
            if !(interval.lo < midpoint && midpoint < interval.hi) {
                return CellCertificate::Unknown;
            }
            // Shared closed split endpoints leave neither gaps nor discarded
            // uncertainty. Both children must eventually be Outside.
            let lower = Interval::new(interval.lo, midpoint).unwrap();
            let upper = Interval::new(midpoint, interval.hi).unwrap();
            if width >= height {
                left.x = lower;
                right.x = upper;
            } else {
                left.y = lower;
                right.y = upper;
            }
            frontier[length] = Some((right, depth + 1));
            frontier[length + 1] = Some((left, depth + 1));
            length += 2;
        }
        if complete_outside {
            CellCertificate::Outside
        } else {
            CellCertificate::Unknown
        }
    }

    /// Arrangement occupancy already resolved original F1 fill rules, union
    /// and clips. Away from its exposed boundary, exact axis comparisons and
    /// half-open ray parity classify the original set without replaying paths.
    pub(super) fn classify_original(&self, cell: Cell, fuel: &mut usize) -> CellCertificate {
        if charge(fuel, 16).is_none() {
            return CellCertificate::Unknown;
        }
        let x = cell.x.lo * 0.5 + cell.x.hi * 0.5;
        let y = cell.y.lo * 0.5 + cell.y.hi * 0.5;
        if !(cell.x.lo <= x && x <= cell.x.hi && cell.y.lo <= y && y <= cell.y.hi) {
            return CellCertificate::Unknown;
        }
        let mut inside = false;
        for &[a, b] in &self.edges {
            if charge(fuel, 32).is_none() {
                return CellCertificate::Unknown;
            }
            if a[0] == b[0] {
                if cell.x.lo <= a[0] && a[0] <= cell.x.hi && cell.y.lo <= b[1] && a[1] <= cell.y.hi
                {
                    return CellCertificate::Unknown;
                }
                if a[1] <= y && y < b[1] && x < a[0] {
                    inside = !inside;
                }
            } else if cell.y.lo <= a[1]
                && a[1] <= cell.y.hi
                && cell.x.lo <= b[0]
                && a[0] <= cell.x.hi
            {
                return CellCertificate::Unknown;
            }
        }
        if inside {
            CellCertificate::Inside
        } else {
            CellCertificate::Outside
        }
    }

    pub(super) fn edges(&self) -> &[[[f64; 2]; 2]] {
        &self.edges
    }
    pub(super) fn edge_count(&self) -> usize {
        self.edges.len()
    }
    pub(super) fn prepare(
        source: &NeutralRegion<'_>,
        fuel: &mut usize,
    ) -> Result<Self, AreaQueryError> {
        let mut events = Events {
            x: bounded_vector(MAX_EVENTS).ok_or(AreaQueryError::Resource)?,
            y: bounded_vector(MAX_EVENTS).ok_or(AreaQueryError::Resource)?,
        };
        // Reserve bounded inventories once. Per-element growth would hide
        // repeated allocator copies behind nominal constant insertion charges.
        let mut transforms = bounded_vector(
            source
                .depth
                .checked_add(2)
                .ok_or(AreaQueryError::Resource)?,
        )
        .ok_or(AreaQueryError::Resource)?;
        events.region(source, &mut transforms, fuel)?;
        sort_admitted(&mut events.x, f64::total_cmp, 4, fuel).ok_or(AreaQueryError::Resource)?;
        events.x.dedup();
        sort_admitted(&mut events.y, f64::total_cmp, 4, fuel).ok_or(AreaQueryError::Resource)?;
        events.y.dedup();
        let width = events
            .x
            .len()
            .checked_sub(1)
            .ok_or(AreaQueryError::Unsupported)?;
        let height = events
            .y
            .len()
            .checked_sub(1)
            .ok_or(AreaQueryError::Unsupported)?;
        let count = width.checked_mul(height).ok_or(AreaQueryError::Resource)?;
        if count == 0 {
            return Err(AreaQueryError::Unsupported);
        }
        if count > MAX_CELLS {
            return Err(AreaQueryError::Resource);
        }
        charge(fuel, count.checked_mul(32).ok_or(AreaQueryError::Resource)?)
            .ok_or(AreaQueryError::Resource)?;
        let mut occupied = bounded_vector(count).ok_or(AreaQueryError::Resource)?;
        for ys in events.y.windows(2) {
            for xs in events.x.windows(2) {
                let x = xs[0] * 0.5 + xs[1] * 0.5;
                let y = ys[0] * 0.5 + ys[1] * 0.5;
                if !(xs[0] < x && x < xs[1] && ys[0] < y && y < ys[1]) {
                    return Err(AreaQueryError::Precision);
                }
                let point = Cell::new([x, y, x, y]).ok_or(AreaQueryError::Precision)?;
                // Original semantics resolve overlaps and fill rules. No
                // source boundary can cross this open event cell, so its
                // certified point membership applies throughout the cell.
                occupied.push(match source.certify_cell(point, fuel) {
                    CellCertificate::Inside => true,
                    CellCertificate::Outside => false,
                    CellCertificate::Unknown => {
                        return Err(if *fuel == 0 {
                            AreaQueryError::Resource
                        } else {
                            AreaQueryError::Precision
                        });
                    }
                });
            }
        }
        let mut edges = bounded_vector(MAX_EDGES).ok_or(AreaQueryError::Resource)?;
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
                        if edges.len() >= MAX_EDGES || edges.len() == edges.capacity() {
                            return Err(AreaQueryError::Resource);
                        }
                        edges.push([a, b]);
                    }
                }
            }
        }
        // Exact collinear union: event-grid subdivisions are not additional
        // geometric boundaries. Retain gaps; merge only touching/overlapping
        // segments on the same authored coordinate line.
        charge(
            fuel,
            edges
                .len()
                .checked_mul(256)
                .ok_or(AreaQueryError::Resource)?,
        )
        .ok_or(AreaQueryError::Resource)?;
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
        )
        .ok_or(AreaQueryError::Resource)?;
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
        Ok(Self { edges })
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
        // Radius validation, interval construction/squaring, outward bounds,
        // and loop state precede the separately admitted per-edge work.
        if charge(fuel, 64).is_none() {
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
            self.classify_original(cell, fuel)
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
    fn source_boundary_reports_resource_exhaustion_without_profile_decline() {
        assert!(exact_add(1.0, 2.0_f64.powi(-54)).is_none());
        let item = holed_item(Render2dFillRule::EvenOdd);
        let mut graph = builder();
        let source = graph.source_from_item(fake_mesh(), &item, &[0]).unwrap();
        let mut fuel = 1;
        assert!(matches!(
            Boundary::prepare(&source, &mut fuel),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(fuel, 0);
    }

    #[test]
    fn source_boundary_distinguishes_precision_from_unsupported_source() {
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                // The original positive width is valid F1, but its far edge
                // cannot be retained as an exact f64 event coordinate.
                shape: Render2dShape::rect(
                    Render2dRect::new(1.0, 1.0, 2.0_f64.powi(-54), 1.0).unwrap(),
                ),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = builder();
        let precise = graph.source_from_item(fake_mesh(), &item, &[0]).unwrap();
        let mut fuel = 16_777_216;
        assert!(matches!(
            Boundary::prepare(&precise, &mut fuel),
            Err(AreaQueryError::Precision)
        ));
        assert!(fuel > 0);

        // A physical source without original F1 is outside this certificate's
        // profile; more work or numeric refinement cannot admit it.
        let physical = graph.source(fake_mesh(), &[1]).unwrap();
        assert!(matches!(
            Boundary::prepare(&physical, &mut fuel),
            Err(AreaQueryError::Unsupported)
        ));
        assert!(fuel > 0);
    }

    #[test]
    fn source_events_distinguish_profile_precision_and_inventory_limits() {
        let mut events = Events {
            x: bounded_vector(MAX_EVENTS).unwrap(),
            y: bounded_vector(MAX_EVENTS).unwrap(),
        };
        let mut fuel = 16_777_216;
        let scale = Render2dAffineTransform::new(2.0, 0.0, 0.0, 1.0, 0.0, 0.0).unwrap();
        assert!(matches!(
            events.point([1.0, 1.0], &[scale], &mut fuel),
            Err(AreaQueryError::Unsupported)
        ));
        let tiny_translation =
            Render2dAffineTransform::new(1.0, 0.0, 0.0, 1.0, 2.0_f64.powi(-54), 0.0).unwrap();
        assert!(matches!(
            events.point([1.0, 1.0], &[tiny_translation], &mut fuel),
            Err(AreaQueryError::Precision)
        ));
        assert!(events.x.is_empty() && events.y.is_empty());
        for index in 0..MAX_EVENTS {
            events.point([index as f64, 0.0], &[], &mut fuel).unwrap();
        }
        assert!(matches!(
            events.point([0.0, 0.0], &[], &mut fuel),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!((events.x.len(), events.y.len()), (MAX_EVENTS, MAX_EVENTS));
        assert_eq!(
            (events.x.capacity(), events.y.capacity()),
            (MAX_EVENTS, MAX_EVENTS)
        );
        assert!(fuel > 0);
    }

    #[test]
    fn source_boundary_reports_arrangement_capacity_without_exhausting_work() {
        // Two event points per rectangle fill the admitted event inventory
        // exactly. All 256 x and y coordinates differ, so the complete open
        // arrangement needs 255^2 cells, exceeding the fixed 16,384 cells.
        let items: Vec<_> = (0..MAX_EVENTS / 2)
            .map(|index| {
                let coordinate = (index * 2) as f64;
                Render2dItem::new(
                    Render2dPrimitive::Fill {
                        shape: Render2dShape::rect(
                            Render2dRect::new(coordinate, coordinate, 1.0, 1.0).unwrap(),
                        ),
                        brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
                    },
                    Render2dAffineTransform::IDENTITY,
                    Vec::new(),
                    Render2dOpacity::TRANSPARENT,
                )
            })
            .collect();
        let mut graph = builder();
        let children = items
            .iter()
            .enumerate()
            .map(|(index, item)| graph.source_from_item(fake_mesh(), item, &[index]).unwrap())
            .collect();
        let source = graph.union(children, &[]).unwrap();
        let mut fuel = 16_777_216;
        assert!(matches!(
            Boundary::prepare(&source, &mut fuel),
            Err(AreaQueryError::Resource)
        ));
        assert!(fuel > 0);
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
    fn finite_square_whole_target_continuous_area_feasibility() {
        let item = holed_item(Render2dFillRule::EvenOdd);
        let mut graph = builder();
        let source = graph.source_from_item(fake_mesh(), &item, &[0, 0]).unwrap();
        let child = graph
            .effect(
                Arc::clone(&source),
                Render2dDropShadow::new(8.0, 0.0, 0.03125, -1.0, Render2dColorRgba8::TRANSPARENT)
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
            "complete finite-square query failed: {:?}, fuel={fuel}",
            areas.as_ref().err()
        );
        let areas = areas.unwrap();
        assert_eq!(areas.len(), 65_536);
        assert!(areas.iter().all(|a| a[1] - a[0] <= 1.0 / (2048.0 * 256.0)));
        // Independent 120-digit Decimal integration. With h=3/32,
        // w=y-22+h and base=35.5-1.25h, the hole boundary is
        // x=base+w/4+sqrt(1-w*w) for h<=w<=1; above w=1 it is full.
        // Circle-segment primitives use an independently reduced 190-term
        // atan series with remainder below 2^-381/381, never Conic calls.
        let reference = [
            0.0,
            0.022_382_673_718_899_372,
            0.0625,
            0.0625,
            0.0,
            0.030_118_830_474_179_53,
            0.0625,
            0.0625,
            0.006_334_288_852_653_319,
            0.056_225_039_450_836_93,
            0.0625,
            0.0625,
            0.056_973_470_109_734_34,
            0.0625,
            0.0625,
            0.0625,
        ];
        for (index, area) in reference.into_iter().enumerate() {
            let actual = areas[(88 + index / 4) * 256 + 144 + index % 4];
            let expected = area * 16.0;
            assert!(
                actual[0] <= expected && expected <= actual[1],
                "cell{index}: {actual:?} vs {expected}"
            );
        }
        assert_eq!(areas[0], [0.0, 0.0]);
        assert_eq!(areas[80 * 256 + 192], [0.0, 0.0]);
        eprintln!("finite whole-target work {}", 16_777_216 - fuel);
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

    #[test]
    fn identical_line_functions_keep_distinct_transformed_domains() {
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(10.0, 10.0, 10.0, 10.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = builder();
        let first = graph.source_from_item(fake_mesh(), &item, &[0, 0]).unwrap();
        let second = graph
            .affine(
                Arc::clone(&first),
                Render2dAffineTransform::translation(0.0, 10.0).unwrap(),
                &[0, 1],
            )
            .unwrap();
        let combined = graph.union(vec![first, second], &[0]).unwrap();
        let mut fuel = 16_777_216;
        let areas =
            certified_grid_area_query(&combined, [0, 0, 256, 256], 4.0, None, 8, 65_536, &mut fuel)
                .unwrap();
        // The translated copy continues the same x(y)=10/20 lines but
        // occupies y=[20,30]. Function identity cannot erase this domain.
        assert!(areas[(25 * 4) * 256 + 15 * 4][0] > 0.999_999);
        assert!(areas[(25 * 4) * 256 + 21 * 4][1] < 0.000_001);
    }

    fn finite_square_item(rule: Render2dFillRule, two_holes: bool) -> Render2dItem {
        let mut commands = Vec::new();
        let mut contour = |[x0, y0, x1, y1]: [f64; 4], hole: bool| {
            let points = if hole {
                [[x0, y0], [x0, y1], [x1, y1], [x1, y0]]
            } else {
                [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
            };
            for (index, [x, y]) in points.into_iter().enumerate() {
                let p = Render2dPoint::new(x, y).unwrap();
                commands.push(if index == 0 {
                    Render2dPathCommand::MoveTo(p)
                } else {
                    Render2dPathCommand::LineTo(p)
                });
            }
            commands.push(Render2dPathCommand::Close);
        };
        if two_holes {
            contour([-10.0, -10.0, 10.0, 10.0], false);
            contour([-1.5, -1.0, -0.5, 0.0], true);
            contour([0.5, -1.0, 1.5, 0.0], true);
        } else {
            contour([10.0, 10.0, 30.0, 30.0], false);
            contour([18.0, 18.0, 22.0, 22.0], true);
        }
        Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::path(Render2dPath::new(rule, commands).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        )
    }

    #[test]
    fn finite_square_integrates_noncardinal_two_hole_vertex_connector() {
        let item = finite_square_item(Render2dFillRule::EvenOdd, true);
        let mut graph = builder();
        let source = graph.source_from_item(fake_mesh(), &item, &[0]).unwrap();
        let effect = graph
            .effect(
                source,
                Render2dDropShadow::new(0.0, 0.0, 0.03125, -1.0, Render2dColorRgba8::TRANSPARENT)
                    .unwrap(),
                &[0],
            )
            .unwrap();
        let mut fuel = 16_777_216;
        let areas = certified_grid_area_query(
            &effect,
            [319, 332, 321, 333],
            16.0,
            Some(Render2dAffineTransform::translation(20.0, 20.0).unwrap()),
            8,
            2,
            &mut fuel,
        )
        .unwrap();
        // The ORIGINAL erosion has a nonsmooth vertex (0,sqrt(3)/2).
        // Its box halo h=3/32 creates a horizontal connector for |x|<=h.
        // Both requested cells lie entirely below its cardinal neighbours,
        // across that connector: area fraction=16*(29/32-sqrt(3)/2).
        let reference = 14.5 - 8.0 * 3.0_f64.sqrt();
        assert_eq!(areas.len(), 2);
        for actual in areas {
            assert!(
                actual[0] <= reference && reference <= actual[1],
                "vertex connector {actual:?} vs {reference}"
            );
        }
    }

    #[test]
    fn finite_square_two_hole_intersection_vertex_membership() {
        // E=C eroded by disk1 has the convex vertex V=(0,sqrt(3)/2).
        // E+[-h,h]^2 has a new horizontal boundary y=sqrt(3)/2-h,
        // |x|<=h. It is absent from corner-shifted circles and cardinal
        // extrema alone. This fixture tests original geometry, not that
        // incomplete candidate inventory or a sampled support mask.
        // Independent inequalities avoid sqrt rounding in the oracle:
        // (0.77+h)^2=0.7460640625 <3/4;
        // (0.78+h)^2=0.7634390625 >3/4.
        let h = 0.09375;
        for rule in [Render2dFillRule::EvenOdd, Render2dFillRule::NonZero] {
            let item = finite_square_item(rule, true);
            let mut graph = builder();
            let source = graph.source_from_item(fake_mesh(), &item, &[0]).unwrap();
            let mut fuel = 16_777_216;
            let boundary = Boundary::prepare(&source, &mut fuel).unwrap();
            for epsilon in [0.0, 0.000001] {
                for (y, expected) in [
                    (0.77, CellCertificate::Outside),
                    (0.78, CellCertificate::Inside),
                ] {
                    let cell = Cell::new([-epsilon, y - epsilon, epsilon, y + epsilon]).unwrap();
                    assert_eq!(
                        boundary.square_dilated_eroded_cell(
                            cell,
                            1.0,
                            Interval::scalar(h).unwrap(),
                            &mut fuel,
                        ),
                        expected,
                    );
                }
            }
            // lo supplies a common Inside witness; hi supplies a complete
            // Outside cover, valid for every halo in this enclosure.
            let halo = Interval::new(0.09, h).unwrap();
            for (y, expected) in [
                (0.77, CellCertificate::Outside),
                (0.78, CellCertificate::Inside),
            ] {
                assert_eq!(
                    boundary.square_dilated_eroded_cell(
                        Cell::new([0.0, y, 0.0, y]).unwrap(),
                        1.0,
                        halo,
                        &mut fuel,
                    ),
                    expected,
                );
            }
            assert_eq!(
                boundary.square_dilated_eroded_cell(
                    Cell::new([0.0, 0.78, 0.0, 0.78]).unwrap(),
                    1.0,
                    Interval::new(0.0, h).unwrap(),
                    &mut fuel,
                ),
                CellCertificate::Unknown,
                "different admissible halos disagree; neither proof is valid",
            );
            // The same new horizontal segment also admits x=.05: choosing
            // t=(-.05,h) reaches (0,.87375), strictly inside E. All nine
            // fixed witnesses miss that interior x displacement. Adaptive
            // source-part midpoints must independently prove a common witness.
            assert_eq!(
                boundary.square_dilated_eroded_cell(
                    Cell::new([0.05, 0.78, 0.05, 0.78]).unwrap(),
                    1.0,
                    Interval::scalar(h).unwrap(),
                    &mut fuel,
                ),
                CellCertificate::Inside,
            );
        }
    }

    #[test]
    fn finite_square_original_ring_common_witness_and_complete_cover() {
        for rule in [Render2dFillRule::EvenOdd, Render2dFillRule::NonZero] {
            let item = finite_square_item(rule, false);
            let mut graph = builder();
            let source = graph.source_from_item(fake_mesh(), &item, &[0]).unwrap();
            let mut fuel = 16_777_216;
            let boundary = Boundary::prepare(&source, &mut fuel).unwrap();
            for ([x, y], expected) in [
                ([15.0, 15.0], CellCertificate::Inside),
                ([20.0, 20.0], CellCertificate::Outside),
                ([10.92, 15.0], CellCertificate::Inside),
                ([10.8, 15.0], CellCertificate::Outside),
                ([22.65, 22.65], CellCertificate::Inside),
                ([22.6, 22.6], CellCertificate::Outside),
            ] {
                assert_eq!(
                    boundary.square_dilated_eroded_cell(
                        Cell::new([x, y, x + 0.000001, y + 0.000001]).unwrap(),
                        1.0,
                        Interval::scalar(0.09375).unwrap(),
                        &mut fuel,
                    ),
                    expected,
                );
            }
        }
    }

    #[test]
    fn finite_square_uncertainty_and_exhaustion_never_prove_absence() {
        let item = finite_square_item(Render2dFillRule::EvenOdd, true);
        let mut graph = builder();
        let source = graph.source_from_item(fake_mesh(), &item, &[0]).unwrap();
        let mut fuel = 16_777_216;
        let boundary = Boundary::prepare(&source, &mut fuel).unwrap();
        // This box straddles the true dilated boundary. No single result is
        // valid over the entire box, even though its center is on one side.
        let ambiguous = Cell::new([-0.000001, 0.77, 0.000001, 0.78]).unwrap();
        assert_eq!(
            boundary.square_dilated_eroded_cell(
                ambiguous,
                1.0,
                Interval::scalar(0.09375).unwrap(),
                &mut fuel,
            ),
            CellCertificate::Unknown,
        );
        for available in [0, 255, 256, 512] {
            let mut limited = available;
            assert_eq!(
                boundary.square_dilated_eroded_cell(
                    Cell::new([0.0, 0.77, 0.0, 0.77]).unwrap(),
                    1.0,
                    Interval::scalar(0.09375).unwrap(),
                    &mut limited,
                ),
                CellCertificate::Unknown,
            );
            assert_eq!(limited, 0);
        }
        for (radius, halo) in [
            (-1.0, Interval::scalar(0.09375).unwrap()),
            (f64::INFINITY, Interval::scalar(0.09375).unwrap()),
            (1.0, Interval::new(-0.01, 0.09375).unwrap()),
        ] {
            assert_eq!(
                boundary.square_dilated_eroded_cell(ambiguous, radius, halo, &mut fuel),
                CellCertificate::Unknown,
            );
        }
    }

    #[test]
    fn finite_square_missed_witnesses_and_tangency_fail_closed() {
        let tiny = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::rect(Render2dRect::new(-0.01, -0.01, 0.02, 0.02).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mut graph = builder();
        let source = graph.source_from_item(fake_mesh(), &tiny, &[0]).unwrap();
        let mut fuel = 16_777_216;
        let boundary = Boundary::prepare(&source, &mut fuel).unwrap();
        let query = Cell::new([0.02, 0.03, 0.02, 0.03]).unwrap();
        // Every one of the nine fixed displacement witnesses misses the
        // tiny region, but t=(-.02,-.03) is an actual Inside witness.
        // Adaptive witnesses are independently certified; sampling their
        // failures never discharges the Outside covering.
        assert_eq!(
            boundary.square_dilated_eroded_cell(
                query,
                0.0,
                Interval::scalar(1.0).unwrap(),
                &mut fuel,
            ),
            CellCertificate::Inside,
        );
        let ring = finite_square_item(Render2dFillRule::EvenOdd, false);
        let source = graph.source_from_item(fake_mesh(), &ring, &[1]).unwrap();
        let boundary = Boundary::prepare(&source, &mut fuel).unwrap();
        // Exact dyadic tangency:10.90625+.09375=11, the disk-eroded
        // outer boundary. Strict separation cannot decide closed equality.
        assert_eq!(
            boundary.square_dilated_eroded_cell(
                Cell::new([10.90625, 15.0, 10.90625, 15.0]).unwrap(),
                1.0,
                Interval::scalar(0.09375).unwrap(),
                &mut fuel,
            ),
            CellCertificate::Unknown,
        );
    }

    #[test]
    fn finite_square_membership_work_profiles() {
        for (two_holes, probes) in [
            (
                false,
                vec![
                    [15.0, 15.0],
                    [0.0, 0.0],
                    [20.0, 20.0],
                    [10.92, 15.0],
                    [22.65, 22.65],
                    [22.6, 22.6],
                ],
            ),
            (true, vec![[0.0, 0.77], [0.0, 0.78], [0.05, 0.78]]),
        ] {
            let item = finite_square_item(Render2dFillRule::EvenOdd, two_holes);
            let mut graph = builder();
            let source = graph.source_from_item(fake_mesh(), &item, &[0]).unwrap();
            let mut fuel = 16_777_216;
            let boundary = Boundary::prepare(&source, &mut fuel).unwrap();
            for [x, y] in probes {
                let mut fuel = 16_777_216;
                let result = boundary.square_dilated_eroded_cell(
                    Cell::new([x, y, x, y]).unwrap(),
                    1.0,
                    Interval::scalar(0.09375).unwrap(),
                    &mut fuel,
                );
                assert_ne!(result, CellCertificate::Unknown);
                eprintln!(
                    "finite member({x},{y})={result:?}, work{}",
                    16_777_216 - fuel
                );
            }
        }
    }
}
