//! Bounded numerical query compilation over certified original-region rows.
//! This module owns neither a diagnostic census nor renderer source meaning.
use super::super::super::gaussian_kernel::fixed::batch;
use super::*;

#[derive(Clone, Copy, Debug)]
pub(super) struct BoundaryRequest {
    pub(super) id: usize,
    pub(super) y: Interval,
    pub(super) center: [f64; 2],
    pub(super) range: Interval,
    pub(super) terms: usize,
    pub(super) sign: f64,
}
const MAX_NODES: usize = 65_536;
const MAX_GEOMETRY: usize = 512;

pub(super) struct IntegralRequest {
    pub(super) map: MappedBoundary,
    pub(super) domain: Interval,
    pub(super) y: Interval,
    pub(super) center: [f64; 2],
    pub(super) terms: usize,
    pub(super) sign: f64,
}

pub(super) struct Compiler<'a> {
    sigma: f64,
    center_bound: [f64; 2],
    kernel: &'a GaussianKernel,
    compiled: CompiledCache,
    factor: HalfRatioFactor,
    proof_index: numeric_index::Index<8>,
    proofs: Vec<Option<f64>>,
    geometry_index: numeric_index::Index<9>,
    coordinates: Vec<scalar_map::RawCoordinates>,
    normalizations: Vec<scalar_map::query::Normalization>,
    mass_index: numeric_index::Index<3>,
    masses: Vec<Interval>,
    cutoff_index: numeric_index::Index<9>,
    cutoff_events: Vec<CutoffEvents>,
    pub(super) nodes: Vec<batch::Node>,
    cuts: Vec<f64>,
    pending: Vec<Interval>,
    #[cfg(test)]
    pub(super) quadrature: usize,
    #[cfg(test)]
    pub(super) slivers: usize,
}

impl<'a> Compiler<'a> {
    /// Every capacity below is fixed and fallible. Include the embedded
    /// transient compiled-map construction record. No diagnostic integrator
    /// or unused CPU coverage/profile cache is constructed by this compiler.
    pub(super) fn reservation_bytes() -> Option<usize> {
        MAX_NODES
            .checked_mul(size_of::<batch::Node>())?
            .checked_add(MAX_GEOMETRY.checked_mul(
                size_of::<Option<f64>>()
                    + 16 * size_of::<scalar_map::RawCoordinates>()
                    + size_of::<scalar_map::query::Normalization>()
                    + size_of::<Interval>(),
            )?)?
            .checked_add(numeric_index::Index::<8>::heap_bound()?)?
            .checked_add(numeric_index::Index::<9>::heap_bound()?)?
            .checked_add(numeric_index::Index::<3>::heap_bound()?)?
            .checked_add(numeric_index::Index::<9>::heap_bound()?)?
            .checked_add(MAX_GEOMETRY.checked_mul(size_of::<CutoffEvents>())?)?
            .checked_add(34 * size_of::<f64>() + 64 * size_of::<Interval>())?
            .checked_add(CompiledCache::peak_bound()?)?
            .checked_add(size_of::<Self>())
    }

    pub(super) fn new(
        kernel: &'a GaussianKernel,
        sigma: f64,
        center_bound: [f64; 2],
        fuel: &mut usize,
    ) -> Result<Self, AreaQueryError> {
        debit(fuel, 1024)?;
        if !center_bound.into_iter().all(|v| v.is_finite() && v >= 0.0)
            || !sigma.is_finite()
            || sigma <= 0.0
        {
            return Err(AreaQueryError::Precision);
        }
        let rule = PositiveRule::new(fuel)?;
        let factor = rule.half_ratio_factor(fuel)?;
        let compiled = CompiledCache::new(fuel)?;
        let proof_index = numeric_index::Index::new(fuel)?;
        let geometry_index = numeric_index::Index::new(fuel)?;
        let mass_index = numeric_index::Index::new(fuel)?;
        let cutoff_index = numeric_index::Index::new(fuel)?;
        let proofs = bounded_vector(MAX_GEOMETRY).ok_or(AreaQueryError::Resource)?;
        let coordinates = bounded_vector(MAX_GEOMETRY * 16).ok_or(AreaQueryError::Resource)?;
        let normalizations = bounded_vector(MAX_GEOMETRY).ok_or(AreaQueryError::Resource)?;
        let masses = bounded_vector(MAX_GEOMETRY).ok_or(AreaQueryError::Resource)?;
        let cutoff_events = bounded_vector(MAX_GEOMETRY).ok_or(AreaQueryError::Resource)?;
        let nodes = bounded_vector(MAX_NODES).ok_or(AreaQueryError::Resource)?;
        let cuts = bounded_vector(34).ok_or(AreaQueryError::Resource)?;
        let pending = bounded_vector(64).ok_or(AreaQueryError::Resource)?;
        if proofs.capacity() != MAX_GEOMETRY
            || coordinates.capacity() != MAX_GEOMETRY * 16
            || normalizations.capacity() != MAX_GEOMETRY
            || masses.capacity() != MAX_GEOMETRY
            || cutoff_events.capacity() != MAX_GEOMETRY
            || nodes.capacity() != MAX_NODES
            || cuts.capacity() != 34
            || pending.capacity() != 64
        {
            return Err(AreaQueryError::Resource);
        }
        Ok(Self {
            sigma,
            center_bound,
            kernel,
            compiled,
            factor,
            proof_index,
            proofs,
            geometry_index,
            coordinates,
            normalizations,
            mass_index,
            masses,
            cutoff_index,
            cutoff_events,
            nodes,
            cuts,
            pending,
            #[cfg(test)]
            quadrature: 0,
            #[cfg(test)]
            slivers: 0,
        })
    }

    fn normalize(
        &self,
        value: Interval,
        center: f64,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        debit(fuel, 160)?;
        checked(checked(value.sub(iv(center)))?.div(iv(self.sigma)))
    }

    pub(super) fn mass(
        &mut self,
        y: Interval,
        cy: f64,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        debit(fuel, 80)?;
        let key = [y.lo.to_bits(), y.hi.to_bits(), cy.to_bits()];
        let (index, inserted) = self.mass_index.observe(key, fuel)?;
        if inserted {
            let lo = self.normalize(iv(y.lo), cy, fuel)?;
            let hi = self.normalize(iv(y.hi), cy, fuel)?;
            let value = self.kernel.mass(lo, hi, fuel)?;
            debit(fuel, 48)?;
            if index != self.masses.len() || self.masses.len() == MAX_GEOMETRY {
                return Err(AreaQueryError::Resource);
            }
            self.masses.push(value);
        }
        Ok(self.masses[index])
    }

    pub(super) fn halfplane_slice(
        &self,
        line: [f64; 2],
        slab: Interval,
        center: [f64; 2],
        inside_left: bool,
        mass: Interval,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        let x = line_range(line[0], line[1], slab, fuel)?;
        let z = self.normalize(x, center[0], fuel)?;
        let mut cdf = self.kernel.cdf(z, fuel)?;
        debit(fuel, 192)?;
        if !inside_left {
            cdf = checked(iv(1.0).sub(cdf))?;
        }
        checked(cdf.mul(mass))
    }

    fn error(
        &mut self,
        map: MappedBoundary,
        panel: Interval,
        fuel: &mut usize,
    ) -> Result<Option<f64>, AreaQueryError> {
        debit(fuel, 256)?;
        let middle = checked(checked(iv(panel.lo).add(iv(panel.hi)))?.mul(iv(0.5)))?;
        let half = checked(checked(iv(panel.hi).sub(iv(panel.lo)))?.mul(iv(0.5)))?;
        let radius = (2.0 * half.hi).next_up();
        let key = match map {
            MappedBoundary::Line { slope, .. } => {
                [0.0, slope.abs(), 0.0, 0.0, 0.0, 0.0, half.hi, radius]
            }
            MappedBoundary::Circle { a, b, r, .. } => [
                1.0,
                a.abs(),
                b.abs(),
                r,
                middle.lo,
                middle.hi,
                half.hi,
                radius,
            ],
        }
        .map(f64::to_bits);
        let (index, inserted) = self.proof_index.observe(key, fuel)?;
        if inserted {
            let value = match map.complex_bound(middle, radius, self.sigma, fuel)? {
                Some(bound) => Some(self.factor.error(half.hi, radius, bound, fuel)?),
                None => None,
            };
            debit(fuel, 48)?;
            if index != self.proofs.len() || self.proofs.len() == MAX_GEOMETRY {
                return Err(AreaQueryError::Resource);
            }
            self.proofs.push(value);
        }
        Ok(self.proofs[index])
    }

    fn append_panel(
        &mut self,
        map: MappedBoundary,
        domain: Interval,
        center: [f64; 2],
        sign: f64,
        fuel: &mut usize,
    ) -> Result<(), AreaQueryError> {
        debit(fuel, 192)?;
        let (index, inserted) = self
            .geometry_index
            .observe(map_key(map, [domain.lo, domain.hi]), fuel)?;
        if inserted {
            let compiled = self.compiled.get(map, fuel)?;
            let entry = compiled.map(|index| &self.compiled.entries[index]);
            let panel = scalar_map::query::QueryPanel::new(
                map,
                entry,
                domain,
                self.center_bound,
                self.sigma,
                fuel,
            )?;
            debit(fuel, 96)?;
            if self.normalizations.len() != index || self.normalizations.len() == MAX_GEOMETRY {
                return Err(AreaQueryError::Resource);
            }
            self.normalizations.push(panel.normalization());
            if index * 16 != self.coordinates.len() {
                return Err(AreaQueryError::Precision);
            }
            for (node, _) in RULE {
                for direction in [-1.0, 1.0] {
                    debit(fuel, 48)?;
                    let raw = panel.raw_node(direction * node, fuel)?;
                    if self.coordinates.len() == MAX_GEOMETRY * 16 {
                        return Err(AreaQueryError::Resource);
                    }
                    self.coordinates.push(raw);
                }
            }
        }
        for (pair, (_, weight)) in RULE.into_iter().enumerate() {
            for side in 0..2 {
                debit(fuel, 64)?;
                let raw = self.coordinates[index * 16 + pair * 2 + side];
                let input = self.normalizations[index].node(raw, weight, center, fuel)?;
                if self.nodes.len() == MAX_NODES {
                    return Err(AreaQueryError::Resource);
                }
                self.nodes.push(batch::Node {
                    z: input.z,
                    z_error: input.z_error,
                    weight: sign * input.weight,
                    weight_error: input.weight_error,
                });
            }
        }
        #[cfg(test)]
        {
            self.quadrature += 1;
        }
        Ok(())
    }

    /// Returns a signed known interval plus a symmetric analytic remainder.
    /// Cutoff root completeness and a strict midpoint branch test are separate
    /// from the reusable complex numerical bound; root slivers are never
    /// asserted analytic. Finite-cutoff endpoint disagreements have zero area.
    pub(super) fn curve(
        &mut self,
        plan: &ScanPlan,
        demand: BoundaryRequest,
        fuel: &mut usize,
    ) -> Result<(Interval, f64), AreaQueryError> {
        debit(fuel, 256)?;
        let halo = checked(iv(self.sigma).mul(iv(3.0)))?;
        let footprint =
            checked(iv(demand.center[0]).add(checked(Interval::new(-halo.hi, halo.hi))?))?;
        if demand.range.hi <= footprint.lo {
            return Ok((iv(0.0), 0.0));
        }
        if demand.range.lo >= footprint.hi {
            let value = self.mass(demand.y, demand.center[1], fuel)?;
            debit(fuel, 96)?;
            return Ok((checked(value.mul(iv(demand.sign)))?, 0.0));
        }
        let (map, domain) = MappedBoundary::domain(plan, demand.id, demand.y, fuel)?;
        self.integral(
            IntegralRequest {
                map,
                domain,
                y: demand.y,
                center: demand.center,
                terms: demand.terms,
                sign: demand.sign,
            },
            fuel,
        )
    }

    pub(super) fn integral(
        &mut self,
        demand: IntegralRequest,
        fuel: &mut usize,
    ) -> Result<(Interval, f64), AreaQueryError> {
        let map = demand.map;
        let domain = demand.domain;
        // For constant X and Y(t)=t, separability factors the entire integral
        // into one certified CDF and one certified Y mass. This also handles
        // exact finite-cutoff equality: subdividing Y cannot make a constant
        // X at +/-3 sigma strictly belong to an analytic interior branch.
        // Require identical parameter/ordinate limits so no endpoint-domain
        // uncertainty is discarded. Other maps retain their existing proof.
        debit(fuel, 256)?;
        if let MappedBoundary::Line {
            intercept,
            slope: 0.0,
        } = map
            && domain.lo == demand.y.lo
            && domain.hi == demand.y.hi
        {
            let z = self.normalize(iv(intercept), demand.center[0], fuel)?;
            let cdf = self.kernel.cdf(z, fuel)?;
            let mass = self.mass(demand.y, demand.center[1], fuel)?;
            debit(fuel, 192)?;
            return Ok((checked(checked(cdf.mul(mass))?.mul(iv(demand.sign)))?, 0.0));
        }
        // A boundary's cutoff roots depend on its coefficients and center,
        // not on which occupied strip/domain references that same boundary.
        // Reuse only these numerical events; source occupancy is never cached
        // under a coefficient key. Every lookup, record and copy is paid.
        let (index, inserted) = self
            .cutoff_index
            .observe(map_key(map, demand.center), fuel)?;
        debit(fuel, 256)?;
        if inserted {
            let events = map
                .cutoff_events(demand.center, self.sigma, fuel)?
                .ok_or(AreaQueryError::Unsupported)?;
            if self.cutoff_events.len() != index || index == MAX_GEOMETRY {
                return Err(AreaQueryError::Resource);
            }
            self.cutoff_events.push(events);
        }
        let events = self.cutoff_events[index];
        debit(fuel, 256)?;
        let allowance = checked(iv(1.0 / 524_288.0).div(iv(8.0 * demand.terms as f64)))?.lo;
        let original_length = checked(iv(domain.hi).sub(iv(domain.lo)))?;
        let endpoints = [
            map.ordinate(iv(domain.lo), fuel)?,
            map.ordinate(iv(domain.hi), fuel)?,
        ];
        debit(fuel, 256)?;
        let bottom = endpoints[0].lo.min(endpoints[1].lo);
        let top = endpoints[0].hi.max(endpoints[1].hi);
        let lower = checked(iv(demand.y.lo).sub(iv(bottom)))?.hi.max(0.0);
        let upper = checked(iv(top).sub(iv(demand.y.hi)))?.hi.max(0.0);
        let endpoint_error = checked(checked(iv(lower).add(iv(upper)))?.div(iv(self.sigma)))?.hi;
        let mut error = endpoint_error;
        let mut known = iv(0.0);
        // One fallibly bounded scratch allocation for the compiler. Clearing
        // lengths does not zero unused slots on every curve; only admitted
        // cut/frontier records are written, under their actual push allowances.
        debit(fuel, 64)?;
        self.cuts.clear();
        self.pending.clear();
        self.cuts.push(domain.lo);
        self.cuts.push(domain.hi);
        for event in events.iter() {
            for value in [event.lo, event.hi] {
                debit(fuel, 24)?;
                if value > domain.lo && value < domain.hi {
                    if self.cuts.len() == 34 {
                        return Err(AreaQueryError::Resource);
                    }
                    self.cuts.push(value);
                }
            }
        }
        debit(fuel, 4 * self.cuts.len() * self.cuts.len())?;
        self.cuts.sort_unstable_by(f64::total_cmp);
        for pair in self.cuts.windows(2) {
            debit(fuel, 32)?;
            if pair[0] < pair[1] {
                if self.pending.len() == 64 {
                    return Err(AreaQueryError::Resource);
                }
                self.pending.push(checked(Interval::new(pair[0], pair[1]))?);
            }
        }
        let mut attempts = 0;
        while let Some(panel) = self.pending.pop() {
            debit(fuel, 384)?;
            if attempts == 4096 {
                return Err(AreaQueryError::Resource);
            }
            attempts += 1;
            let ya = map.ordinate(iv(panel.lo), fuel)?;
            let yb = map.ordinate(iv(panel.hi), fuel)?;
            let y = checked(Interval::new(ya.lo.min(yb.lo), ya.hi.max(yb.hi)))?;
            let sliver = events
                .iter()
                .any(|root| panel.lo < root.hi && panel.hi > root.lo);
            if sliver {
                debit(fuel, 192)?;
                let width = checked(checked(iv(y.hi).sub(iv(y.lo)))?.div(iv(self.sigma)))?.hi;
                let uncertain = checked(Interval::new(0.0, width))?;
                known = checked(known.add(checked(uncertain.mul(iv(demand.sign)))?))?;
                #[cfg(test)]
                {
                    self.slivers += 1;
                }
                continue;
            }
            // A root-free connected panel remains in the same branch as its
            // strictly classified real midpoint. Interval overlap is rejected.
            debit(fuel, 64)?;
            let mid = panel.lo * 0.5 + panel.hi * 0.5;
            let (xmid, ymid) = map.position(iv(mid), fuel)?;
            let x = self.normalize(xmid, demand.center[0], fuel)?;
            let zy = self.normalize(ymid, demand.center[1], fuel)?;
            let y_inside = zy.lo > -3.0 && zy.hi < 3.0;
            if zy.hi < -3.0 || zy.lo > 3.0 || x.hi < -3.0 {
                continue;
            }
            if x.lo > 3.0 && y_inside {
                let mass = self.mass(y, demand.center[1], fuel)?;
                debit(fuel, 384)?;
                // `y` encloses both exact endpoint ordinates, rather than
                // supplying exact limits. Outer-span mass may overestimate
                // the true panel by at most both endpoint widths / sigma.
                let wa = checked(iv(ya.hi).sub(iv(ya.lo)))?;
                let wb = checked(iv(yb.hi).sub(iv(yb.lo)))?;
                let excess = checked(checked(wa.add(wb))?.div(iv(self.sigma)))?.hi;
                error = checked(iv(error).add(iv(excess)))?.hi;
                known = checked(known.add(checked(mass.mul(iv(demand.sign)))?))?;
                continue;
            }
            let analytic = y_inside && x.lo > -3.0 && x.hi < 3.0;
            if !analytic {
                // Exact cutoff equality and the last representable parameter
                // panels need not admit a strict analytic midpoint branch.
                // Keep their COMPLETE possible mass, just as for an enclosed
                // cutoff-root sliver. CDF<=1 and normalized density<=1/sigma
                // bound this by physical ordinate width/sigma. Nothing is
                // dropped; the final combined query radius still must pass.
                debit(fuel, 256)?;
                let width = checked(checked(iv(y.hi).sub(iv(y.lo)))?.div(iv(self.sigma)))?.hi;
                if width <= allowance / 64.0 {
                    let uncertain = checked(Interval::new(0.0, width))?;
                    known = checked(known.add(checked(uncertain.mul(iv(demand.sign)))?))?;
                    continue;
                }
            }
            debit(fuel, 160)?;
            let allocation = checked(iv(allowance).mul(checked(
                checked(iv(panel.hi).sub(iv(panel.lo)))?.div(original_length),
            )?))?
            .lo;
            if analytic
                && let Some(radius) = self.error(map, panel, fuel)?
                && radius <= allocation
            {
                self.append_panel(map, panel, demand.center, demand.sign, fuel)?;
                debit(fuel, 48)?;
                error = checked(iv(error).add(iv(radius)))?.hi;
                continue;
            }
            if !(panel.lo < mid && mid < panel.hi) || self.pending.len() + 2 > 64 {
                return Err(AreaQueryError::Precision);
            }
            self.pending.push(checked(Interval::new(panel.lo, mid))?);
            self.pending.push(checked(Interval::new(mid, panel.hi))?);
        }
        Ok((known, error))
    }
}

pub(super) fn map_key(map: MappedBoundary, tail: [f64; 2]) -> [u64; 9] {
    let raw = match map {
        MappedBoundary::Line { intercept, slope } => {
            [0.0, intercept, slope, 0.0, 0.0, 0.0, 0.0, tail[0], tail[1]]
        }
        MappedBoundary::Circle {
            h,
            a,
            b,
            k,
            r,
            sign,
        } => [1.0, h, a, b, k, r, sign, tail[0], tail[1]],
    };
    raw.map(f64::to_bits)
}

#[test]
fn cutoff_event_reuse_requires_identical_coefficients_and_center_not_domain_or_sign() {
    let mut fuel = 16_777_216;
    let kernel = GaussianKernel::compiled(&mut fuel, MAX_BYTES).unwrap();
    let mut compiler = Compiler::new(&kernel, 0.125, [2.0; 2], &mut fuel).unwrap();
    for (intercept, center, domain, sign) in [
        (0.0, [0.0; 2], [-0.125, 0.0], 1.0),
        (0.0, [0.0; 2], [0.0, 0.125], -1.0),
        (0.0, [0.03125, 0.0], [0.0, 0.125], 1.0),
        (0.03125, [0.03125, 0.0], [0.0, 0.125], 1.0),
    ] {
        let y = Interval::new(domain[0], domain[1]).unwrap();
        compiler
            .integral(
                IntegralRequest {
                    map: MappedBoundary::Line {
                        intercept,
                        slope: 1.0,
                    },
                    domain: y,
                    y,
                    center,
                    terms: 1,
                    sign,
                },
                &mut fuel,
            )
            .unwrap();
    }
    assert_eq!(compiler.cutoff_events.len(), 3);
    assert!(compiler.cutoff_index.maximum_probes > 0);
    assert!(fuel > 0);
}
