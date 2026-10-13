//! Test-only feasibility: share rigorously bounded, normalized Gaussian
//! line integrals across rows and columns of an authored axis-aligned F1
//! rectangle. No sampled source/alpha mask, SDF, GPU or production path.
//! Arbitrary clips, transforms, strokes, curves and nested regions are NOT
//! certified by separability, and may never silently enter this fast path.

use super::{Interval, gaussian_density_enclosure};
use crate::composition_2d::Render2dRect;

pub(super) const BINS: usize = 16_384;
const BIN_WIDTH: f64 = 6.0 / BINS as f64;
pub(super) const BIN_WORK: usize = 512;
pub(super) const AXIS_WORK: usize = 128;
const SITE_WORK: usize = 48;
pub(super) const STRIP_BIN_WORK: usize = 8;
pub(super) const MAX_SCENE_WORK: usize = 16_777_216;
pub(super) const MAX_PROOF_BYTES: usize = 1_048_576;

pub(super) struct NormalizedGaussianCdf {
    prefix: Vec<Interval>,
    bin_masses: Vec<Interval>,
    normalization: Interval,
}

impl NormalizedGaussianCdf {
    // Grid knots are exactly representable dyadic rational numbers, so binary
    // searches compare exact f64 grid coordinates, with no rounding-derived
    // false floor/ceiling indices.
    fn knot(index: usize) -> f64 {
        -3.0 + index as f64 * BIN_WIDTH
    }

    pub(super) fn new(work: &mut usize) -> Option<Self> {
        let charge = BINS.checked_mul(BIN_WORK)?;
        if *work < charge {
            return None;
        }
        *work -= charge;
        let mut prefix = Vec::new();
        prefix.try_reserve_exact(BINS + 1).ok()?;
        let mut bin_masses = Vec::new();
        bin_masses.try_reserve_exact(BINS).ok()?;
        let mut cumulative = Interval::scalar(0.0)?;
        prefix.push(cumulative);
        let zero = Interval::scalar(0.0)?;
        let width = Interval::scalar(BIN_WIDTH)?;
        for i in 0..BINS {
            // This integrates a CONTINUOUS nonnegative 1D Gaussian, not
            // samples of a rasterized region. Every bin uses a density
            // enclosure over its entire exact coordinate range.
            let coordinates = Interval::new(Self::knot(i), Self::knot(i + 1))?;
            let density = gaussian_density_enclosure(coordinates, zero)?;
            let mass = density.mul(width)?;
            cumulative = cumulative.add(mass)?;
            bin_masses.push(mass);
            prefix.push(cumulative);
        }
        if cumulative.lo <= 0.0 {
            return None;
        }
        Some(Self {
            prefix,
            bin_masses,
            normalization: cumulative,
        })
    }

    fn floor_index(position: f64) -> usize {
        let mut lo = 0;
        let mut hi = BINS;
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if Self::knot(mid) <= position {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        lo
    }

    fn ceil_index(position: f64) -> usize {
        let mut lo = 0;
        let mut hi = BINS;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if Self::knot(mid) < position {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        lo
    }

    fn cumulative(&self, bound: Interval) -> Option<Interval> {
        let lo = bound.lo.clamp(-3.0, 3.0);
        let hi = bound.hi.clamp(-3.0, 3.0);
        Interval::new(
            self.prefix[Self::floor_index(lo)].lo,
            self.prefix[Self::ceil_index(hi)].hi,
        )
    }

    /// Bounded nonnegative normalized Gaussian mass between two possibly
    /// interval-enclosed cut positions, with the original finite 3-sigma law.
    /// This is shared by x-slabs and certified vertical source enclosures.
    pub(super) fn interval_mass(
        &self,
        start: Interval,
        end: Interval,
        center: f64,
        sigma: Interval,
        work: &mut usize,
    ) -> Option<Interval> {
        if *work < AXIS_WORK || sigma.lo <= 0.0 || start.lo > end.hi {
            return None;
        }
        *work -= AXIS_WORK;
        let origin = Interval::scalar(center)?;
        let near = start.sub(origin)?.div(sigma)?;
        let far = end.sub(origin)?.div(sigma)?;
        if near.lo >= 3.0 || far.hi <= -3.0 {
            return Interval::scalar(0.0);
        }
        if near.hi <= -3.0 && far.lo >= 3.0 {
            return Interval::scalar(1.0);
        }
        let numerator = self.cumulative(far)?.sub(self.cumulative(near)?)?;
        let numerator = Interval::new(
            numerator.lo.max(0.0),
            numerator.hi.min(self.normalization.hi),
        )?;
        let mass = numerator.div(self.normalization)?;
        Interval::new(mass.lo.max(0.0), mass.hi.min(1.0))
    }

    /// Avoid dependent-prefix subtraction for an exact CDF grid-aligned
    /// x-slab: sums ORIGINAL cached 1D Gaussian bin enclosures directly.
    /// Separate prefix subtraction would compound uncertainty over every
    /// earlier bin and fail the boundary precision gate.
    pub(super) fn equal_strip_mass(
        &self,
        strip: usize,
        strips: usize,
        work: &mut usize,
    ) -> Option<Interval> {
        if strips == 0
            || !strips.is_power_of_two()
            || strips > BINS
            || !BINS.is_multiple_of(strips)
            || strip >= strips
        {
            return None;
        }
        let per_strip = BINS / strips;
        let charge = per_strip.checked_mul(STRIP_BIN_WORK)?;
        if *work < charge {
            *work = 0;
            return None;
        }
        *work -= charge;
        let first = strip.checked_mul(per_strip)?;
        let mut mass = Interval::scalar(0.0)?;
        for bin in &self.bin_masses[first..first.checked_add(per_strip)?] {
            mass = mass.add(*bin)?;
        }
        let normalized = mass.div(self.normalization)?;
        Interval::new(normalized.lo.max(0.0), normalized.hi.min(1.0))
    }

    pub(super) fn peak_buffer_bytes(&self) -> Option<usize> {
        self.prefix
            .capacity()
            .checked_add(self.bin_masses.capacity())?
            .checked_mul(std::mem::size_of::<Interval>())
    }

    fn axis_coverage(
        &self,
        start: f64,
        extent: f64,
        center: f64,
        sigma: Interval,
        work: &mut usize,
    ) -> Option<Interval> {
        if *work < AXIS_WORK {
            return None;
        }
        *work -= AXIS_WORK;
        if extent <= 0.0 {
            return Interval::scalar(0.0);
        }
        let start = Interval::scalar(start)?;
        let center = Interval::scalar(center)?;
        let near = start.sub(center)?.div(sigma)?;
        let far = start
            .add(Interval::scalar(extent)?)?
            .sub(center)?
            .div(sigma)?;
        // Complete support exclusion and inclusion are exact, independent
        // of how broad the integration's finite numeric enclosure is.
        if near.lo >= 3.0 || far.hi <= -3.0 {
            return Interval::scalar(0.0);
        }
        if near.hi <= -3.0 && far.lo >= 3.0 {
            return Interval::scalar(1.0);
        }
        let bound = self.cumulative(far)?.sub(self.cumulative(near)?)?;
        let numerator = Interval::new(bound.lo.max(0.0), bound.hi.min(self.normalization.hi))?;
        let normalized = numerator.div(self.normalization)?;
        Interval::new(normalized.lo.max(0.0), normalized.hi.min(1.0))
    }
}

#[derive(Debug)]
struct RectangleSceneCertificate {
    sites: usize,
    maximum_width: f64,
    work_charged: usize,
    peak_buffer_bytes: usize,
    // Sample coordinates [18.125,18.125], [18.375,18.375],
    // [17.875,18.125] independently checked against the continuous law.
    probe_intervals: [[f64; 2]; 3],
}

/// All 4x4 correlated subpixel centers of a finite canvas are covered from
/// one reusable 1D normalized Gaussian table and two axis vectors. No
/// per-pixel coverage buffer is allocated and no 2D numerical integration
/// is repeated per center. Only ORIGINAL axis-aligned, unclipped,
/// zero-spread F1 rectangles may use this separability proof.
fn certify_rectangle_scene(
    rect: Render2dRect,
    canvas_side: usize,
    sigma: f64,
    budget: usize,
    max_bytes: usize,
) -> Option<RectangleSceneCertificate> {
    if canvas_side != 64
        || !sigma.is_finite()
        || sigma <= 0.0
        || budget > MAX_SCENE_WORK
        || max_bytes > MAX_PROOF_BYTES
    {
        return None;
    }
    let samples = canvas_side.checked_mul(4)?;
    let mut remaining = budget;
    let table = NormalizedGaussianCdf::new(&mut remaining)?;
    let sigma = Interval::scalar(sigma)?;
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    xs.try_reserve_exact(samples).ok()?;
    ys.try_reserve_exact(samples).ok()?;
    for i in 0..samples {
        // The accepted compositor's globally phased 4x4 centers.
        let center = (i as f64 + 0.5) / 4.0;
        xs.push(table.axis_coverage(rect.x(), rect.width(), center, sigma, &mut remaining)?);
        ys.push(table.axis_coverage(rect.y(), rect.height(), center, sigma, &mut remaining)?);
    }
    let bytes = table.peak_buffer_bytes()?.checked_add(
        xs.capacity()
            .checked_add(ys.capacity())?
            .checked_mul(std::mem::size_of::<Interval>())?,
    )?;
    if bytes > max_bytes {
        return None;
    }
    let sites = samples.checked_mul(samples)?;
    let site_charge = sites.checked_mul(SITE_WORK)?;
    if remaining < site_charge {
        return None;
    }
    remaining -= site_charge;

    let mut maximum_width = 0.0_f64;
    let mut probe_intervals = [[0.0; 2]; 3];
    for (y, &ymass) in ys.iter().enumerate() {
        for (x, &xmass) in xs.iter().enumerate() {
            let result = xmass.mul(ymass)?;
            let interval = Interval::new(result.lo.max(0.0), result.hi.min(1.0))?;
            maximum_width = maximum_width.max(interval.hi - interval.lo);
            for (k, (px, py)) in [(72, 72), (73, 73), (71, 72)].into_iter().enumerate() {
                if (x, y) == (px, py) {
                    probe_intervals[k] = [interval.lo, interval.hi];
                }
            }
        }
    }
    Some(RectangleSceneCertificate {
        sites,
        maximum_width,
        work_charged: budget - remaining,
        peak_buffer_bytes: bytes,
        probe_intervals,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition_2d::Render2dShape;

    fn original_f1_rect() -> Render2dRect {
        let shape = Render2dShape::rect(Render2dRect::new(18.0, 18.0, 28.0, 28.0).unwrap());
        shape.as_rect().expect("F1 retains the authored rectangle")
    }

    #[test]
    fn continuous_gaussian_rectangle_boundary_uses_shared_1d_integrals() {
        let result = certify_rectangle_scene(
            original_f1_rect(),
            64,
            0.25,
            MAX_SCENE_WORK,
            MAX_PROOF_BYTES,
        )
        .unwrap();
        assert_eq!(result.sites, 64 * 64 * 16);
        assert!(
            result.maximum_width < 1.0 / 255.0,
            "not even one 8-bit channel can be certified: {}",
            result.maximum_width
        );
        assert!(result.work_charged < MAX_SCENE_WORK);
        assert!(result.peak_buffer_bytes < MAX_PROOF_BYTES);
        // Independently evaluated continuous truncated-normal references,
        // NOT the renderer's raster pixels or reference implementation.
        for ([lower, upper], reference) in result.probe_intervals.into_iter().zip([
            0.478_837_386_328_834_27,
            0.873_038_881_965_241_6,
            0.213_143_383_872_790_78,
        ]) {
            assert!(
                lower <= reference && reference <= upper,
                "reference coverage {reference} outside certified {lower}..{upper}"
            );
        }
    }

    #[test]
    fn continuous_gaussian_scene_rejects_unadmitted_work_and_memory() {
        let rect = original_f1_rect();
        assert!(certify_rectangle_scene(rect, 64, 0.25, 100, MAX_PROOF_BYTES).is_none());
        assert!(
            certify_rectangle_scene(rect, 64, 0.25, MAX_SCENE_WORK, 1_024).is_none(),
            "table's peak buffer must be part of admission"
        );
        assert!(certify_rectangle_scene(rect, 64, 0.25, MAX_SCENE_WORK, MAX_PROOF_BYTES).is_some());
    }

    #[test]
    fn continuous_gaussian_truncation_does_not_invent_nonzero_outside() {
        let mut work = MAX_SCENE_WORK;
        let table = NormalizedGaussianCdf::new(&mut work).unwrap();
        let sigma = Interval::scalar(0.25).unwrap();
        let zero = table
            .axis_coverage(18.0, 28.0, 0.0, sigma, &mut work)
            .unwrap();
        assert_eq!([zero.lo, zero.hi], [0.0, 0.0]);
        let one = table
            .axis_coverage(18.0, 28.0, 32.0, sigma, &mut work)
            .unwrap();
        assert_eq!([one.lo, one.hi], [1.0, 1.0]);
    }
}
