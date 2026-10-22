//! Direct CPU integration and diagnostic census retained only for conformance.
#![cfg(test)]
use super::*;

pub(super) struct Integrator<'a> {
    pub(super) rule: PositiveRule,
    pub(super) kernel: &'a GaussianKernel,
    pub(super) sigma: f64,
    // An exact line profile depends only on exact (intercept+slope*cy-cx),
    // slope and sigma. Entries exist only after whole-footprint topology
    // proves this actual halfplane. No source membership comes from a cache.
    pub(super) profiles: Vec<(f64, f64, Interval)>,
    pub(super) compiled: CompiledCache,
    pub(super) panels: usize,
    pub(super) nodes: usize,
    pub(super) last_point: [f64; 2],
    pub(super) completed_points: usize,
    pub(super) class_work: [usize; 5],
    pub(super) class_points: [usize; 5],
    pub(super) last_site_before: usize,
    pub(super) last_class: usize,
    pub(super) diagnostics: &'a mut Diagnostics,
}

#[derive(Clone, Copy)]
pub(super) struct MassCertificate {
    pub(super) y: Interval,
    pub(super) center: f64,
    pub(super) sigma: f64,
    pub(super) value: Interval,
}

impl MassCertificate {
    pub(super) fn get(
        self,
        y: Interval,
        center: f64,
        sigma: f64,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        debit(fuel, 32)?;
        if self.y.lo != y.lo || self.y.hi != y.hi || self.center != center || self.sigma != sigma {
            return Err(AreaQueryError::Precision);
        }
        Ok(self.value)
    }
}

#[derive(Clone, Copy)]
pub(super) struct PanelTask {
    pub(super) domain: Interval,
    pub(super) analytic: bool,
    pub(super) endpoints: [Interval; 2],
}

pub(super) struct Integral {
    pub(super) value: Interval,
    pub(super) endpoints: [Interval; 2],
}

#[derive(Default, Debug)]
pub(super) struct Diagnostics {
    pub(super) last_point: [f64; 2],
    pub(super) completed_points: usize,
    pub(super) class_work: [usize; 5],
    pub(super) class_points: [usize; 5],
    pub(super) last_site_before: usize,
    pub(super) last_class: usize,
    pub(super) panels: usize,
    pub(super) nodes: usize,
    pub(super) profiles: usize,
    pub(super) complete_plane: bool,
    pub(super) compiled: CompiledMetrics,
}

impl Drop for Integrator<'_> {
    fn drop(&mut self) {
        self.diagnostics.last_point = self.last_point;
        self.diagnostics.completed_points = self.completed_points;
        self.diagnostics.class_work = self.class_work;
        self.diagnostics.class_points = self.class_points;
        self.diagnostics.last_site_before = self.last_site_before;
        self.diagnostics.last_class = self.last_class;
        self.diagnostics.panels = self.panels;
        self.diagnostics.nodes = self.nodes;
        self.diagnostics.profiles = self.profiles.len();
        self.diagnostics.compiled = self.compiled.metrics;
        println!(
            "Gaussian integrator last_point={:?}, panels={},nodes={},profiles={}, completed_points={}, class_points={:?}, completed_class_work={:?}, active_class={}, last_site_start_remaining={}",
            self.last_point,
            self.panels,
            self.nodes,
            self.profiles.len(),
            self.completed_points,
            self.class_points,
            self.class_work,
            self.last_class,
            self.last_site_before
        );
        println!(
            "Gaussian compiled map counters: {:?}, peak_reservation={}",
            self.compiled.metrics,
            CompiledCache::peak_bound().unwrap_or(usize::MAX)
        );
    }
}

impl<'a> Integrator<'a> {
    pub(super) fn new(
        kernel: &'a GaussianKernel,
        sigma: f64,
        diagnostics: &'a mut Diagnostics,
        fuel: &mut usize,
    ) -> Result<Self, AreaQueryError> {
        Ok(Self {
            rule: PositiveRule::new(fuel)?,
            kernel,
            sigma,
            profiles: bounded_vector(4096).ok_or(AreaQueryError::Resource)?,
            compiled: CompiledCache::new(fuel)?,
            panels: 0,
            nodes: 0,
            last_point: [0.0, 0.0],
            completed_points: 0,
            class_work: [0; 5],
            class_points: [0; 5],
            last_site_before: 0,
            last_class: 0,
            diagnostics,
        })
    }

    pub(super) fn normalize(
        &self,
        value: Interval,
        center: f64,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        debit(fuel, 160)?;
        checked(checked(value.sub(iv(center)))?.div(iv(self.sigma)))
    }

    pub(super) fn mass(
        &self,
        range: Interval,
        cy: f64,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        let lo = self.normalize(iv(range.lo), cy, fuel)?;
        let hi = self.normalize(iv(range.hi), cy, fuel)?;
        self.kernel.mass(lo, hi, fuel)
    }

    pub(super) fn certify_mass(
        &self,
        y: Interval,
        center: f64,
        fuel: &mut usize,
    ) -> Result<MassCertificate, AreaQueryError> {
        let value = self.mass(y, center, fuel)?;
        debit(fuel, 48)?;
        Ok(MassCertificate {
            y,
            center,
            sigma: self.sigma,
            value,
        })
    }

    /// Evaluate the scalar certified kernel once, then propagate the input
    /// enclosure with a real derivative bound. F is globally 1-Lipschitz
    /// because Z>1. On the STRICT interior, |k'|=|z|exp(-z²/2)/Z<=3.
    /// Density cutoff crossings use its full interval interface instead.
    pub(super) fn point_kernel(
        &self,
        z: Interval,
        density: bool,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        debit(fuel, 256)?;
        if density && (z.lo <= -3.0 || z.hi >= 3.0) {
            return self.kernel.density(z, fuel);
        }
        let middle = z.lo * 0.5 + z.hi * 0.5;
        let distance = checked(z.sub(iv(middle)))?;
        let radius = distance.lo.abs().max(distance.hi.abs());
        let radius = if density {
            (3.0 * radius).next_up()
        } else {
            radius
        };
        let value = if density {
            self.kernel.density(iv(middle), fuel)?
        } else {
            self.kernel.cdf(iv(middle), fuel)?
        };
        let value = checked(value.add(checked(Interval::new(-radius, radius))?))?;
        checked(Interval::new(value.lo.max(0.0), value.hi.min(1.0)))
    }

    pub(super) fn sample(
        &mut self,
        map: MappedBoundary,
        u: Interval,
        center: [f64; 2],
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        debit(fuel, 16)?;
        self.nodes += 1;
        if matches!(map, MappedBoundary::Line { .. }) {
            self.compiled.metrics.line_nodes += 1;
        } else {
            self.compiled.metrics.interval_circle_nodes += 1;
        }
        let (x, y, derivative) = map.at(u, fuel)?;
        self.sample_coordinates(
            Coordinates {
                x,
                y,
                jacobian: derivative,
            },
            center,
            fuel,
        )
    }

    pub(super) fn sample_coordinates(
        &self,
        coordinate: Coordinates,
        center: [f64; 2],
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        let Coordinates {
            x,
            y,
            jacobian: derivative,
        } = coordinate;
        let x = self.normalize(x, center[0], fuel)?;
        let y = self.normalize(y, center[1], fuel)?;
        let cdf = self.point_kernel(x, false, fuel)?;
        let density = self.point_kernel(y, true, fuel)?;
        debit(fuel, 304)?;
        checked(checked(checked(cdf.mul(density))?.mul(derivative))?.div(iv(self.sigma)))
    }

    pub(super) fn panel(
        &mut self,
        map: MappedBoundary,
        compiled: Option<usize>,
        domain: Interval,
        center: [f64; 2],
        fuel: &mut usize,
    ) -> Result<Option<Interval>, AreaQueryError> {
        self.panels += 1;
        debit(fuel, 384)?;
        let middle = checked(checked(iv(domain.lo).add(iv(domain.hi)))?.mul(iv(0.5)))?;
        let half = checked(checked(iv(domain.hi).sub(iv(domain.lo)))?.mul(iv(0.5)))?;
        let radius = (2.0 * half.hi).next_up();
        let Some(bound) = map.complex_bound(middle, radius, self.sigma, fuel)? else {
            return Ok(None);
        };
        let error = self.rule.error(half.hi, radius, bound, fuel)?;
        if error > 1.0e-9 {
            return Ok(None);
        }
        let scalar_panel = if compiled.is_some() {
            let before = *fuel;
            let result = scalar_map::Panel::new(domain, fuel);
            self.compiled.metrics.panel_work += before - *fuel;
            Some(result?)
        } else {
            None
        };
        let mut sum = iv(0.0);
        for (node, weight) in RULE {
            for sign in [-1.0, 1.0] {
                let value = if let (Some(index), Some(panel)) = (compiled, scalar_panel) {
                    // Only weight multiplication/accumulation and expanded
                    // metadata/loop work remain here. ScalarPanel pays its
                    // native node FMA; no interval node coordinate is built.
                    debit(fuel, 192)?;
                    self.nodes += 1;
                    self.compiled.metrics.compiled_nodes += 1;
                    let before = *fuel;
                    let result = self
                        .compiled
                        .entries
                        .get(index)
                        .ok_or(AreaQueryError::Precision)?
                        .map
                        .at_node(panel, sign * node, fuel);
                    self.compiled.metrics.node_work += before - *fuel;
                    self.sample_coordinates(result?, center, fuel)?
                } else {
                    debit(fuel, 336)?;
                    let u = checked(middle.add(checked(half.mul(iv(sign * node)))?))?;
                    self.sample(map, u, center, fuel)?
                };
                sum = checked(sum.add(checked(iv(weight).mul(value))?))?;
            }
        }
        debit(fuel, 176)?;
        let value = checked(checked(half.mul(sum))?.add(checked(Interval::new(-error, error))?))?;
        Ok(Some(value))
    }

    pub(super) fn integral(
        &mut self,
        map: MappedBoundary,
        domain: Interval,
        center: [f64; 2],
        fuel: &mut usize,
    ) -> Result<Integral, AreaQueryError> {
        let compiled = self.compiled.get(map, fuel)?;
        let endpoints = [
            self.ordinate_point(map, compiled, domain.lo, fuel)?,
            self.ordinate_point(map, compiled, domain.hi, fuel)?,
        ];
        let empty = PanelTask {
            domain: iv(0.0),
            analytic: false,
            endpoints: [iv(0.0); 2],
        };
        let mut pending = [empty; 64];
        let mut count = 0;
        if let Some(events) = map.cutoff_events(center, self.sigma, fuel)? {
            let mut cuts = [0.0_f64; 34];
            cuts[0] = domain.lo;
            cuts[1] = domain.hi;
            let mut length = 2;
            for event in events.iter() {
                for value in [event.lo, event.hi] {
                    debit(fuel, 16)?;
                    if value > domain.lo && value < domain.hi {
                        cuts[length] = value;
                        length += 1;
                    }
                }
            }
            debit(fuel, 4 * length * length)?;
            cuts[..length].sort_by(f64::total_cmp);
            let mut ordinates = [iv(0.0); 34];
            for (index, &value) in cuts[..length].iter().enumerate() {
                debit(fuel, 32)?;
                ordinates[index] = if value == domain.lo {
                    endpoints[0]
                } else if value == domain.hi {
                    endpoints[1]
                } else if index > 0 && value == cuts[index - 1] {
                    ordinates[index - 1]
                } else {
                    self.ordinate_point(map, compiled, value, fuel)?
                };
            }
            for (index, pair) in cuts[..length].windows(2).enumerate() {
                if pair[0] == pair[1] {
                    continue;
                }
                let local = checked(Interval::new(pair[0], pair[1]))?;
                let complete = !events
                    .iter()
                    .any(|event| event.lo < local.hi && event.hi > local.lo);
                let middle = local.lo * 0.5 + local.hi * 0.5;
                let (x, y) = self.position_point(map, compiled, middle, fuel)?;
                let x = self.normalize(x, center[0], fuel)?;
                let y = self.normalize(y, center[1], fuel)?;
                let analytic = complete && x.lo > -3.0 && x.hi < 3.0 && y.lo > -3.0 && y.hi < 3.0;
                pending[count] = PanelTask {
                    domain: local,
                    analytic,
                    endpoints: [ordinates[index], ordinates[index + 1]],
                };
                count += 1;
            }
        } else {
            pending[0] = PanelTask {
                domain,
                analytic: false,
                endpoints,
            };
            count = 1;
        }
        let mut result = iv(0.0);
        let mut admitted = 0_usize;
        while count > 0 {
            count -= 1;
            let task = pending[count];
            let panel = task.domain;
            let certified_analytic = task.analytic;
            debit(fuel, 192)?;
            if admitted >= 4096 {
                return Err(AreaQueryError::Resource);
            }
            let x = map.abscissa(panel, fuel)?;
            let normalized_x = self.normalize(x, center[0], fuel)?;
            let [ya, yb] = task.endpoints;
            let (bottom, top) = if matches!(map,MappedBoundary::Circle{sign,..} if sign<0.0) {
                (yb, ya)
            } else {
                (ya, yb)
            };
            let y = checked(Interval::new(bottom.lo, top.hi))?;
            let normalized_y = self.normalize(y, center[1], fuel)?;
            let cdf = self.kernel.cdf(normalized_x, fuel)?;
            let normalized_bottom = self.normalize(bottom, center[1], fuel)?;
            let normalized_top = self.normalize(top, center[1], fuel)?;
            let mass = self.kernel.mass(normalized_bottom, normalized_top, fuel)?;
            debit(fuel, 160)?;
            let range = checked(cdf.mul(mass))?;
            if range.hi - range.lo <= 1.0e-9 {
                result = checked(result.add(range))?;
                admitted += 1;
                continue;
            }
            // A finite-cutoff CDF/density is not analytic at +/-3.
            // Panels straddling those events subdivide or use the positive
            // range enclosure above; no high-order remainder crosses them.
            let analytic = certified_analytic
                || normalized_x.lo >= -3.0
                    && normalized_x.hi <= 3.0
                    && normalized_y.lo >= -3.0
                    && normalized_y.hi <= 3.0;
            if analytic && let Some(value) = self.panel(map, compiled, panel, center, fuel)? {
                debit(fuel, 80)?;
                result = checked(result.add(value))?;
                admitted += 1;
                continue;
            }
            let middle = panel.lo * 0.5 + panel.hi * 0.5;
            if !(panel.lo < middle && middle < panel.hi) {
                return Err(AreaQueryError::Precision);
            }
            if count + 2 > pending.len() {
                return Err(AreaQueryError::Resource);
            }
            let y_middle = self.ordinate_point(map, compiled, middle, fuel)?;
            pending[count] = PanelTask {
                domain: checked(Interval::new(panel.lo, middle))?,
                analytic: certified_analytic,
                endpoints: [ya, y_middle],
            };
            pending[count + 1] = PanelTask {
                domain: checked(Interval::new(middle, panel.hi))?,
                analytic: certified_analytic,
                endpoints: [y_middle, yb],
            };
            count += 2;
        }
        Ok(Integral {
            value: result,
            endpoints,
        })
    }

    pub(super) fn exact_coordinates(
        &mut self,
        index: usize,
        u: f64,
        components: Components,
        fuel: &mut usize,
    ) -> Result<Coordinates, AreaQueryError> {
        let before = *fuel;
        let result = (|| {
            debit(fuel, 16)?;
            self.compiled
                .entries
                .get(index)
                .ok_or(AreaQueryError::Precision)?
                .map
                .at_exact(u, components, fuel)
        })();
        self.compiled.metrics.point_work += before - *fuel;
        self.compiled.metrics.point_queries += 1;
        result
    }

    pub(super) fn ordinate_point(
        &mut self,
        map: MappedBoundary,
        compiled: Option<usize>,
        u: f64,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        if let Some(index) = compiled {
            Ok(self.exact_coordinates(index, u, Components::Y, fuel)?.y)
        } else {
            map.ordinate(iv(u), fuel)
        }
    }

    pub(super) fn position_point(
        &mut self,
        map: MappedBoundary,
        compiled: Option<usize>,
        u: f64,
        fuel: &mut usize,
    ) -> Result<(Interval, Interval), AreaQueryError> {
        if let Some(index) = compiled {
            let value = self.exact_coordinates(index, u, Components::Position, fuel)?;
            Ok((value.x, value.y))
        } else {
            map.position(iv(u), fuel)
        }
    }

    pub(super) fn curve(
        &mut self,
        plan: &ScanPlan,
        id: usize,
        band: &RowBand,
        center: [f64; 2],
        range: Interval,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        let y = band.y;
        // RowRun already certified this identical curve and Y-band range
        // from the immutable plan; do not replay its coordinate query.
        let cdf = self
            .kernel
            .cdf(self.normalize(range, center[0], fuel)?, fuel)?;
        let mass = band
            .mass
            .ok_or(AreaQueryError::Precision)?
            .get(y, center[1], self.sigma, fuel)?;
        debit(fuel, 160)?;
        let coarse = checked(cdf.mul(mass))?;
        if coarse.hi - coarse.lo <= 1.0e-9 {
            return Ok(coarse);
        }
        let (map, domain) = MappedBoundary::domain(plan, id, y, fuel)?;
        let integrated = self.integral(map, domain, center, fuel)?;
        let mut integral = integrated.value;
        if matches!(map, MappedBoundary::Circle { .. }) {
            // Parameter endpoint enclosures expand the desired Y interval.
            // Remove at most that extra positive Y mass, independently of X.
            let [ya, yb] = integrated.endpoints;
            let mapped_y = checked(Interval::new(ya.lo.min(yb.lo), ya.hi.max(yb.hi)))?;
            let mut excess = iv(0.0);
            if mapped_y.lo < y.lo {
                excess = checked(excess.add(self.mass(
                    checked(Interval::new(mapped_y.lo, y.lo))?,
                    center[1],
                    fuel,
                )?))?;
            }
            if mapped_y.hi > y.hi {
                excess = checked(excess.add(self.mass(
                    checked(Interval::new(y.hi, mapped_y.hi))?,
                    center[1],
                    fuel,
                )?))?;
            }
            debit(fuel, 128)?;
            integral = checked(integral.sub(checked(Interval::new(0.0, excess.hi))?))?;
        }
        checked(Interval::new(
            integral.lo.max(0.0),
            integral.hi.min(mass.hi),
        ))
    }

    pub(super) fn halfplane(
        &mut self,
        plan: &ScanPlan,
        id: usize,
        center: [f64; 2],
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        debit(fuel, 128)?;
        let CurveKey::ExactLine(intercept, slope) = plan.curves[id].key else {
            return Err(AreaQueryError::Precision);
        };
        let distance = exact_add(
            exact_add(
                intercept,
                exact_mul(slope, center[1]).ok_or(AreaQueryError::Precision)?,
            )
            .ok_or(AreaQueryError::Precision)?,
            -center[0],
        )
        .ok_or(AreaQueryError::Precision)?;
        self.profile(distance, slope, fuel)
    }

    /// For even normalized k on the symmetric finite support,
    /// H(d,s)=integral k(z)F(d+s*z)dz obeys H(d,-s)=H(d,s) and
    /// H(-d,s)=1-H(d,s). These exact changes of variables permit canonical
    /// absolute keys. Sigma is immutable for this invocation; source geometry
    /// still has to certify the complete halfplane before calling this path.
    pub(super) fn profile(
        &mut self,
        distance: f64,
        slope: f64,
        fuel: &mut usize,
    ) -> Result<Interval, AreaQueryError> {
        debit(fuel, 96)?;
        if !distance.is_finite() || !slope.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        let reflected = distance < 0.0;
        let distance = distance.abs();
        let slope = slope.abs();
        // Native FMA and one outward neighbor enclose the extreme affine X
        // values over the entire finite Y footprint. These tests avoid any
        // transcendental query for exact saturated halfplane responses.
        let halo = (self.sigma * 3.0).next_up();
        let low = (-slope.abs()).mul_add(halo, distance).next_down();
        let high = slope.abs().mul_add(halo, distance).next_up();
        if low >= halo {
            return Ok(iv(if reflected { 0.0 } else { 1.0 }));
        }
        if high <= -halo {
            return Ok(iv(0.0));
        }
        let mut first = 0;
        let mut end = self.profiles.len();
        while first < end {
            debit(fuel, 32)?;
            let middle = first + (end - first) / 2;
            let (d, s, _) = self.profiles[middle];
            if d < distance || d == distance && s < slope {
                first = middle + 1;
            } else {
                end = middle;
            }
        }
        if let Some(&(d, s, value)) = self.profiles.get(first)
            && d == distance
            && s == slope
        {
            debit(fuel, 48)?;
            return if reflected {
                checked(iv(1.0).sub(value))
            } else {
                Ok(value)
            };
        }
        let domain = checked(Interval::new(-halo, halo))?;
        let map = MappedBoundary::Line {
            intercept: distance,
            slope,
        };
        let value = self.integral(map, domain, [0.0, 0.0], fuel)?.value;
        if self.profiles.len() == self.profiles.capacity() {
            return Err(AreaQueryError::Resource);
        }
        debit(fuel, 32 + 16 * (self.profiles.len() - first))?;
        self.profiles.insert(first, (distance, slope, value));
        debit(fuel, 48)?;
        if reflected {
            checked(iv(1.0).sub(value))
        } else {
            Ok(value)
        }
    }
}
