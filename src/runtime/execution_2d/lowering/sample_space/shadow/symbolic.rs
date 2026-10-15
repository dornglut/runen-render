//! Qualified zero-kernel paint from the retained original-F1 neutral region.
//!
//! The region remains geometric authority. Continuous cell enclosures are
//! disposable numerical evidence, converted to one mask only after bounded
//! precision, scene work and peak-live host storage admission. The existing
//! F3E compositor and RunenGPU upload remain the sole physical consumers.
use super::*;
use geometry_support::region::interval::{Interval, exact_add, exact_mul, exact_products_equal};
use support::PreparationWork;

use support::MAX_COMPOSITIONAL_SHADOW_WORK as MAX_QUERY_WORK;
const MAX_QUERY_CELLS: usize = 1_048_576;
const MAX_CELL_UNCERTAINTY: f64 = 1.0 / (2048.0 * 256.0);
use geometry_support::region::MAX_AREA_QUERY_FIXED_BYTES as QUERY_FIXED_SCRATCH_BYTES;

pub(super) struct Context<'a> {
    pub(super) parent_to_root: Render2dAffineTransform,
    pub(super) paintable: bool,
    pub(super) target: &'a AdmittedTarget,
    pub(super) path: &'a [usize],
    pub(super) aggregate_work: &'a mut support::PreparationBudget,
    pub(super) retained_bytes: &'a mut u64,
}

fn charge(
    work: &mut dyn PreparationWork,
    amount: usize,
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    work.charge(
        amount,
        MAX_QUERY_WORK,
        path,
        "symbolic shadow exceeds bounded whole-scene numerical work",
    )
}

fn unsupported(owner: &[usize]) -> Render2dExecutionError {
    let Some(&root_index) = owner.first() else {
        return shadow_precision(owner, "symbolic shadow has no original occurrence owner");
    };
    crate::runtime::execution_2d::unsupported_at(
        owner,
        crate::execution_2d::Render2dUnsupportedContent::Shadows { root_index },
    )
}

fn admit_effects(
    effects: &[Render2dDropShadow],
    owner: &[usize],
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    if effects.len() > MAX_AUTHORED_SHADOWS {
        return Err(shadow_resource(
            path,
            "symbolic authored shadow count exceeds admission",
        ));
    }
    if effects
        .iter()
        .any(|effect| effect.sigma() != 0.0 || effect.spread() != 0.0)
    {
        return Err(unsupported(owner));
    }
    Ok(())
}

fn scalar(value: f64) -> Option<Interval> {
    Interval::scalar(value)
}

/// The parent offset precedes ancestor projection. A rounded new translation
/// would move the source rather than enclose it, so admit only a proved exact
/// composition until the region query accepts an interval-valued forward map.
fn forward(
    ancestor: Render2dAffineTransform,
    offset: [f64; 2],
    path: &[usize],
) -> Result<Option<Render2dAffineTransform>, Render2dExecutionError> {
    let [a, b, c, d, tx, ty] = ancestor.components();
    if exact_products_equal(a, d, b, c) == Some(true) {
        return Ok(None);
    }
    let compose = |p, q, t| {
        exact_add(
            exact_add(exact_mul(p, offset[0])?, exact_mul(q, offset[1])?)?,
            t,
        )
    };
    let new_x = compose(a, c, tx).ok_or_else(|| {
        shadow_precision(
            path,
            "symbolic parent offset cannot be composed exactly through ancestor",
        )
    })?;
    let new_y = compose(b, d, ty).ok_or_else(|| {
        shadow_precision(
            path,
            "symbolic parent offset cannot be composed exactly through ancestor",
        )
    })?;
    let determinant = scalar(a)
        .and_then(|a| a.mul(scalar(d)?))
        .and_then(|ad| ad.sub(scalar(b)?.mul(scalar(c)?)?))
        .ok_or_else(|| {
            shadow_precision(path, "symbolic ancestor determinant cannot be enclosed")
        })?;
    if determinant.lo <= 0.0 && determinant.hi >= 0.0 {
        return Err(shadow_precision(
            path,
            "symbolic ancestor invertibility is numerically uncertain",
        ));
    }
    Render2dAffineTransform::new(a, b, c, d, new_x, new_y)
        .map(Some)
        .map_err(|_| shadow_precision(path, "symbolic composed ancestor is unrepresentable"))
}

fn projected_envelope(
    bounds: [f64; 4],
    forward: Render2dAffineTransform,
    path: &[usize],
) -> Result<[f64; 4], Render2dExecutionError> {
    let calculate = || {
        let x = Interval::new(bounds[0], bounds[2])?;
        let y = Interval::new(bounds[1], bounds[3])?;
        let [a, b, c, d, tx, ty] = forward.components().map(scalar);
        let root_x = a?.mul(x)?.add(c?.mul(y)?)?.add(tx?)?;
        let root_y = b?.mul(x)?.add(d?.mul(y)?)?.add(ty?)?;
        Some([root_x.lo, root_y.lo, root_x.hi, root_y.hi])
    };
    calculate().ok_or_else(|| {
        shadow_precision(path, "symbolic original-support reach cannot be projected")
    })
}

fn grid(
    bounds: [f64; 4],
    sample_scale: f64,
    sample_extent: [u64; 2],
    path: &[usize],
) -> Result<Option<([u32; 4], usize)>, Render2dExecutionError> {
    let calculate_axis = |lo: f64, hi: f64, extent: u64| -> Option<[f64; 2]> {
        let scale = scalar(sample_scale)?;
        let limit = scalar(extent as f64)?.div(scale)?.hi;
        let left = lo.max(0.0);
        let right = hi.min(limit);
        if right <= left {
            return Some([0.0, 0.0]);
        }
        // Clamp only the disposable physical extent, after completed source
        // support and parent offset have been projected. Original C is intact.
        let left = scalar(left)?.mul(scale)?.lo.floor().max(0.0);
        let right = scalar(right)?.mul(scale)?.hi.ceil().min(extent as f64);
        Some([left, right])
    };
    let [left, right] = calculate_axis(bounds[0], bounds[2], sample_extent[0])
        .ok_or_else(|| shadow_precision(path, "symbolic x sample envelope cannot be enclosed"))?;
    let [top, bottom] = calculate_axis(bounds[1], bounds[3], sample_extent[1])
        .ok_or_else(|| shadow_precision(path, "symbolic y sample envelope cannot be enclosed"))?;
    if left >= right || top >= bottom {
        return Ok(None);
    }
    let limit = f64::from(i32::MAX as u32 / 2);
    if [left, top, right, bottom]
        .into_iter()
        .any(|value| !value.is_finite() || value > limit)
    {
        return Err(shadow_precision(
            path,
            "symbolic globally phased sample indices exceed precision admission",
        ));
    }
    let root = [left as u32, top as u32, right as u32, bottom as u32];
    let count = usize::try_from(root[2] - root[0])
        .ok()
        .and_then(|width| width.checked_mul(usize::try_from(root[3] - root[1]).ok()?))
        .ok_or_else(|| {
            shadow_resource(path, "symbolic sample footprint overflows host indexing")
        })?;
    if count > MAX_QUERY_CELLS {
        return Err(shadow_resource(
            path,
            "symbolic sample footprint exceeds bounded interval cells",
        ));
    }
    Ok(Some((root, count)))
}

fn peak_bytes(
    retained: u64,
    count: usize,
    width: usize,
    metadata: u64,
    path: &[usize],
) -> Result<u64, Render2dExecutionError> {
    let float_bytes = u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(8))
        .ok_or_else(|| shadow_resource(path, "symbolic retained coverage byte extent overflow"))?;
    let row_bytes = u64::try_from(width)
        .ok()
        .and_then(|width| width.checked_add(1))
        .and_then(|width| width.checked_mul(24))
        .ok_or_else(|| shadow_resource(path, "symbolic row and x-event byte extent overflow"))?;
    let query_bytes = float_bytes
        .checked_mul(2)
        .and_then(|intervals| intervals.checked_add(row_bytes))
        .ok_or_else(|| shadow_resource(path, "symbolic query interval and row bytes overflow"))?;
    let conversion_bytes = float_bytes.checked_mul(3).ok_or_else(|| {
        shadow_resource(
            path,
            "symbolic coexisting interval and float bytes overflow",
        )
    })?;
    // The source query holds 16 bytes/cell plus 24 bytes/row column;
    // conversion instead holds 16-byte intervals plus 8-byte float coverage.
    // Only the larger phase is live. Fixed boundary/matrix/frontier scratch
    // is conservatively reserved separately from the upstream source DAG.
    let peak = query_bytes
        .max(conversion_bytes)
        .checked_add(QUERY_FIXED_SCRATCH_BYTES)
        .and_then(|bytes| bytes.checked_add(retained))
        .and_then(|bytes| bytes.checked_add(metadata))
        .ok_or_else(|| {
            shadow_resource(path, "symbolic interval and coverage peak bytes overflow")
        })?;
    if peak > MAX_SHADOW_COVERAGE_BYTES {
        return Err(shadow_resource(
            path,
            "symbolic interval plus coverage peak exceeds bounded host storage",
        ));
    }
    Ok(float_bytes)
}

fn midpoint([lo, hi]: [f64; 2], path: &[usize]) -> Result<f64, Render2dExecutionError> {
    if !lo.is_finite() || !hi.is_finite() || lo < 0.0 || hi > 1.0 || lo > hi {
        return Err(shadow_precision(
            path,
            "symbolic continuous-area enclosure is not normalized",
        ));
    }
    let uncertainty = if lo == hi { 0.0 } else { (hi - lo).next_up() };
    if uncertainty > MAX_CELL_UNCERTAINTY {
        return Err(shadow_precision(
            path,
            "symbolic continuous-area enclosure exceeds composite precision",
        ));
    }
    let value = lo * 0.5 + hi * 0.5;
    // Outward arithmetic can enclose an exact empty area as [0, tiny].
    // Zero is then an admitted value inside the geometry certificate and
    // survives the physical mask. A strictly positive lower bound never
    // acquires this option: its GPU underflow still fails at upload.
    Ok(if lo == 0.0 && value as f32 == 0.0 {
        0.0
    } else {
        value
    })
}

fn occurrence_path(
    work: &mut dyn PreparationWork,
    owner: &[usize],
) -> Result<Vec<usize>, Render2dExecutionError> {
    charge(work, owner.len() + 64, owner)?;
    let mut path = Vec::new();
    path.try_reserve_exact(owner.len())
        .map_err(|_| shadow_resource(owner, "symbolic owner-path allocation failed"))?;
    if path.capacity() > owner.len() {
        return Err(shadow_resource(
            owner,
            "symbolic owner-path capacity exceeds admission",
        ));
    }
    path.extend_from_slice(owner);
    Ok(path)
}

pub(super) fn prepare(
    source: &geometry_support::SymbolicCaster<'_>,
    effects: &[Render2dDropShadow],
    context: Context<'_>,
) -> Result<Vec<PreparedShadow>, Render2dExecutionError> {
    let owner = source.owner_path();
    charge(context.aggregate_work, 128, context.path)?;
    admit_effects(effects, owner, context.path)?;
    charge(context.aggregate_work, effects.len(), context.path)?;
    let source_envelope = source
        .logical_envelope()
        .ok_or_else(|| unsupported(owner))?;
    let sample_scale = context.target.raster_scale() * f64::from(SAMPLES);
    if !sample_scale.is_finite() || sample_scale <= 0.0 {
        return Err(shadow_precision(
            context.path,
            "symbolic target sample scale is invalid",
        ));
    }
    let sample_extent = [
        u64::from(context.target.physical_width) * u64::from(SAMPLES),
        u64::from(context.target.physical_height) * u64::from(SAMPLES),
    ];
    // Persistent headers/control blocks/paths for all groups were reserved
    // by the contribution preflight; only this invocation's transient keys
    // remain additional working metadata.
    let metadata = effects
        .len()
        .checked_mul(std::mem::size_of::<[u64; 6]>())
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| shadow_resource(context.path, "symbolic query-key metadata overflows"))?;
    let mut result = Vec::<PreparedShadow>::new();
    // Invocation-local disposable query reuse. Source and target are fixed
    // here; exact forward coefficients identify the same area request. Paint
    // order and color remain separate authored compositor contributions.
    let mut query_keys = Vec::<[u64; 6]>::new();
    let mut retained = *context.retained_bytes;
    for effect in effects {
        charge(context.aggregate_work, 640, context.path)?;
        let Some(to_root) = forward(
            context.parent_to_root,
            [effect.offset_x(), effect.offset_y()],
            context.path,
        )?
        else {
            continue;
        };
        let logical_envelope = projected_envelope(source_envelope, to_root, context.path)?;
        let color = linear_color(effect.color());
        // Full geometric validity and reach precede nonpainting mask elision.
        // The independent source traversal already retained transparent effects.
        if !context.paintable || color[3] == 0.0 {
            continue;
        }
        let key = to_root.components().map(f64::to_bits);
        charge(
            context.aggregate_work,
            query_keys.len() * 6 + 32,
            context.path,
        )?;
        if let Some(index) = query_keys.iter().position(|previous| *previous == key) {
            let coverage = Arc::clone(&result[index].coverage);
            let bounds = result[index].bounds;
            let path = occurrence_path(context.aggregate_work, context.path)?;
            result.push(PreparedShadow {
                color,
                coverage,
                logical_envelope,
                bounds,
                group_visible: context.paintable,
                path,
                sample_area_coverage: true,
            });
            query_keys.push(key);
            continue;
        }
        let Some((root_grid, count)) =
            grid(logical_envelope, sample_scale, sample_extent, context.path)?
        else {
            continue;
        };
        let width = usize::try_from(root_grid[2] - root_grid[0]).map_err(|_| {
            shadow_resource(context.path, "symbolic sample width exceeds host indexing")
        })?;
        let float_bytes = peak_bytes(retained, count, width, metadata, context.path)?;
        if result.capacity() == 0 {
            result.try_reserve_exact(effects.len()).map_err(|_| {
                shadow_resource(context.path, "symbolic preparation list allocation failed")
            })?;
            query_keys.try_reserve_exact(effects.len()).map_err(|_| {
                shadow_resource(context.path, "symbolic query-key allocation failed")
            })?;
            if result.capacity() > effects.len() || query_keys.capacity() > effects.len() {
                return Err(shadow_resource(
                    context.path,
                    "symbolic occurrence capacity exceeds admission",
                ));
            }
        }
        let available = context.aggregate_work.remaining(MAX_QUERY_WORK);
        let mut fuel = available;
        let enclosures = source.continuous_grid_enclosures(
            root_grid,
            sample_scale,
            Some(to_root),
            8,
            count,
            &mut fuel,
        );
        charge(context.aggregate_work, available - fuel, context.path)?;
        let enclosures = enclosures.map_err(|error| match error {
            geometry_support::region::AreaQueryError::Unsupported => unsupported(owner),
            geometry_support::region::AreaQueryError::Resource => shadow_resource(
                owner,
                "symbolic region query exceeded bounded whole-scene resources",
            ),
            geometry_support::region::AreaQueryError::Precision => shadow_precision(
                owner,
                "symbolic continuous region coverage could not be certified",
            ),
        })?;
        if enclosures.len() != count {
            return Err(shadow_precision(
                owner,
                "symbolic region coverage has inconsistent lattice extent",
            ));
        }
        // Recheck the actual returned interval capacity before allocating its
        // coexisting float coverage. The planned query uses exact reservation.
        let interval_bytes = u64::try_from(enclosures.capacity())
            .ok()
            .and_then(|capacity| capacity.checked_mul(16))
            .ok_or_else(|| {
                shadow_resource(owner, "symbolic interval allocation capacity overflows")
            })?;
        if retained
            .checked_add(metadata)
            .and_then(|bytes| bytes.checked_add(interval_bytes))
            .and_then(|bytes| bytes.checked_add(float_bytes))
            .and_then(|bytes| bytes.checked_add(QUERY_FIXED_SCRATCH_BYTES))
            .is_none_or(|peak| peak > MAX_SHADOW_COVERAGE_BYTES)
        {
            return Err(shadow_resource(
                owner,
                "symbolic actual interval capacity exceeds admitted peak storage",
            ));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| shadow_resource(context.path, "symbolic coverage allocation failed"))?;
        let actual_float_bytes = u64::try_from(values.capacity())
            .ok()
            .and_then(|capacity| capacity.checked_mul(8))
            .ok_or_else(|| {
                shadow_resource(owner, "symbolic float capacity overflows retained storage")
            })?;
        if retained
            .checked_add(metadata)
            .and_then(|bytes| bytes.checked_add(interval_bytes))
            .and_then(|bytes| bytes.checked_add(actual_float_bytes))
            .and_then(|bytes| bytes.checked_add(QUERY_FIXED_SCRATCH_BYTES))
            .is_none_or(|peak| peak > MAX_SHADOW_COVERAGE_BYTES)
        {
            return Err(shadow_resource(
                owner,
                "symbolic actual float capacity exceeds admitted peak storage",
            ));
        }
        for enclosure in enclosures {
            charge(context.aggregate_work, 32, context.path)?;
            values.push(midpoint(enclosure, owner)?);
        }
        let height = usize::try_from(root_grid[3] - root_grid[1]).map_err(|_| {
            shadow_resource(context.path, "symbolic sample height exceeds host indexing")
        })?;
        let coverage = NeutralCoverage {
            origin_x: i64::from(root_grid[0]),
            origin_y: i64::from(root_grid[1]),
            width,
            height,
            values,
        };
        let bounds = output_bounds(&coverage, context.target, context.path)?;
        let path = occurrence_path(context.aggregate_work, context.path)?;
        result.push(PreparedShadow {
            color,
            coverage: Arc::new(coverage),
            logical_envelope,
            bounds,
            group_visible: context.paintable,
            path,
            sample_area_coverage: true,
        });
        query_keys.push(key);
        retained = retained.checked_add(actual_float_bytes).ok_or_else(|| {
            shadow_resource(context.path, "symbolic retained coverage total overflows")
        })?;
    }
    // Failed preflight drops all its local physical preparations. Only successful
    // complete preparation advances retained bytes; actual CPU work is never undone.
    *context.retained_bytes = retained;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shadow(spread: f64, sigma: f64) -> Render2dDropShadow {
        Render2dDropShadow::new(
            0.0,
            0.0,
            sigma,
            spread,
            crate::composition_2d::Render2dColorRgba8::new(0, 0, 255, 255),
        )
        .unwrap()
    }

    #[test]
    fn exact_rank_distinguishes_empty_from_unrepresentable_inverse() {
        let singular = Render2dAffineTransform::new(1.1, 1.1, 1.1, 1.1, 0.0, 0.0).unwrap();
        assert!(forward(singular, [0.0, 0.0], &[2, 1]).unwrap().is_none());
        let step = 2.0_f64.powi(-27);
        let full_rank =
            Render2dAffineTransform::new(1.0, 1.0 - step, 1.0 + step, 1.0, 0.0, 0.0).unwrap();
        assert!(
            matches!(forward(full_rank,[0.0,0.0],&[2,1]), Err(Render2dExecutionError::SampleSpace {
            kind:crate::execution_2d::Render2dSampleSpaceError::PrecisionLimit,path:Some(path),..
        }) if path==[2,1])
        );
    }

    #[test]
    fn unsupported_parent_kernels_keep_original_occurrence_path() {
        for effect in [shadow(1.0, 0.0), shadow(-1.0, 0.0), shadow(0.0, 0.001)] {
            assert!(matches!(admit_effects(&[effect], &[2, 0, 1], &[2]),
                Err(Render2dExecutionError::UnsupportedEntry { path, .. }) if path == [2,0,1]));
        }
        assert!(admit_effects(&[shadow(0.0, 0.0)], &[2, 0, 1], &[2]).is_ok());
    }

    #[test]
    fn offset_precedes_shear_and_uncertain_composition_fails_closed() {
        let ancestor = Render2dAffineTransform::new(1.0, 0.0, 0.25, 1.0, 0.0, 0.0).unwrap();
        let projected = forward(ancestor, [8.0, 2.0], &[0]).unwrap().unwrap();
        assert_eq!(projected.components(), [1.0, 0.0, 0.25, 1.0, 8.5, 2.0]);
        let lossy = Render2dAffineTransform::translation(1.0, 0.0).unwrap();
        assert!(forward(lossy, [f64::from_bits((1023_u64 - 54) << 52), 0.0], &[0]).is_err());
        let collapsed = Render2dAffineTransform::new(1.0, 1.0, 1.0, 1.0, 0.0, 0.0).unwrap();
        assert!(forward(collapsed, [0.0, 0.0], &[0]).unwrap().is_none());
    }

    #[test]
    fn sample_cells_survive_when_their_centers_miss_the_original_reach() {
        let (extent, count) = grid([0.01, 0.01, 0.02, 0.02], 4.0, [256, 256], &[0])
            .unwrap()
            .unwrap();
        assert_eq!(extent, [0, 0, 1, 1]);
        assert_eq!(count, 1);
        assert!(
            grid([-20.0, -20.0, -10.0, -10.0], 4.0, [256, 256], &[0])
                .unwrap()
                .is_none()
        );
        assert!(
            grid([100.0, 100.0, 120.0, 120.0], 4.0, [256, 256], &[0])
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn interval_and_float_peak_are_admitted_before_allocation() {
        assert_eq!(peak_bytes(0, 65_536, 256, 0, &[0]).unwrap(), 524_288);
        assert!(peak_bytes(MAX_SHADOW_COVERAGE_BYTES - 16, 1, 1, 0, &[0]).is_err());
        assert!(peak_bytes(u64::MAX, 1, 1, 0, &[0]).is_err());
        assert!(grid([0.0, 0.0, 1000.0, 1000.0], 4.0, [4000, 4000], &[0]).is_err());
        let mut work = MAX_QUERY_WORK - 2;
        assert!(charge(&mut work, 3, &[0]).is_err());
        assert_eq!(work, MAX_QUERY_WORK - 2);
    }

    #[test]
    fn mixed_siblings_share_one_limit_in_both_orders() {
        use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};

        let mesh = support::NeutralMesh {
            units_per_parent_logical_unit: 1.0,
            triangles: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            bounds: [0.0, 0.0, 1.0, 1.0],
        };
        let legacy = |work: &mut support::PreparationBudget| {
            support::prepare_shadow_coverage_admitted(&mesh, 0.0, 0.0, 4.0, &[3, 1], work, 16)
        };
        let mut probe = support::PreparationBudget::for_contribution(false);
        let expected = legacy(&mut probe).unwrap().unwrap();
        let legacy_cost = probe.used();
        assert!(legacy_cost > 0 && legacy_cost < MAX_QUERY_WORK);
        assert!((expected.values.iter().sum::<f64>() - 8.0).abs() < 1.0e-12);

        for symbolic_first in [false, true] {
            // The complete caster inventory selected the 16M policy before
            // either sibling. Prior invocation work leaves exactly enough
            // room for the legacy coverage, but not the extra symbolic unit.
            let mut work = support::PreparationBudget::for_contribution(true);
            charge(&mut work, MAX_QUERY_WORK - legacy_cost, &[0]).unwrap();
            let error = if symbolic_first {
                charge(&mut work, 1, &[3, 2]).unwrap();
                legacy(&mut work).unwrap_err()
            } else {
                let coverage = legacy(&mut work).unwrap().unwrap();
                assert_eq!(coverage.values, expected.values);
                charge(&mut work, 1, &[3, 2]).unwrap_err()
            };
            let owner = if symbolic_first { [3, 1] } else { [3, 2] };
            assert!(matches!(error, Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), ..
            } if path == owner));
            assert!(work.used() <= MAX_QUERY_WORK);

            // A genuinely admissible mixed contribution succeeds in either
            // order and produces identical continuous legacy coverage.
            let mut work = support::PreparationBudget::for_contribution(true);
            if symbolic_first {
                charge(&mut work, 1, &[3, 2]).unwrap();
            }
            let coverage = legacy(&mut work).unwrap().unwrap();
            if !symbolic_first {
                charge(&mut work, 1, &[3, 2]).unwrap();
            }
            assert_eq!(coverage.values, expected.values);
            assert_eq!(work.used(), legacy_cost + 1);
        }

        // A pure legacy invocation retains its established 134M admission.
        let mut work = support::PreparationBudget::for_contribution(false);
        work.charge(MAX_QUERY_WORK, usize::MAX, &[0], "test prior work")
            .unwrap();
        let coverage = legacy(&mut work).unwrap().unwrap();
        assert_eq!(coverage.values, expected.values);
        assert_eq!(work.used(), MAX_QUERY_WORK + legacy_cost);
    }

    #[test]
    fn physical_midpoints_require_normalized_tight_geometric_evidence() {
        assert!(midpoint([0.0, 1.0], &[0]).is_err());
        assert!(midpoint([f64::NAN, 0.0], &[0]).is_err());
        assert_eq!(midpoint([0.0, 0.0], &[0]).unwrap(), 0.0);
        assert_eq!(midpoint([1.0, 1.0], &[0]).unwrap(), 1.0);
        assert_eq!(midpoint([0.0, f64::MIN_POSITIVE], &[0]).unwrap(), 0.0);
        assert!(midpoint([f64::MIN_POSITIVE, f64::MIN_POSITIVE], &[0]).unwrap() > 0.0);
        let area = 0.057_316_642_859_978_895_f64;
        let alpha = area * 16.0;
        let value = midpoint([alpha.next_down(), alpha.next_up()], &[0]).unwrap();
        assert!((value - alpha).abs() <= f64::EPSILON);
    }
}
