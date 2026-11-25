//! Shared continuous area integration of the existing physical triangle union.
//!
//! Triangle vertices/crossings fix interval topology. Every integer y boundary
//! fixes the output row, and every edge/integer-x crossing fixes clipping to
//! output columns. Between consecutive events each clipped interval width is
//! affine in y, so one midpoint integrates its AREA. No occupancy samples,
//! alpha, new source geometry or Gaussian approximation enter this plan.
use super::{
    NeutralMask, NeutralUnionSweep, PreparationWork, as_f64, charge_shadow_preparation_work,
    filled, mask_failure,
};
use crate::execution_2d::{Render2dExecutionError, Render2dSampleSpaceError};

const MAX_EVENTS: usize = super::MAX_NEUTRAL_UNION_EVENTS;

fn resource(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    mask_failure(path, Render2dSampleSpaceError::ResourceLimit, detail)
}
fn precision(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    mask_failure(path, Render2dSampleSpaceError::PrecisionLimit, detail)
}

fn event(events: &mut Vec<f64>, y: f64, path: &[usize]) -> Result<(), Render2dExecutionError> {
    if !y.is_finite() {
        return Err(precision(path, "shared area event is not finite"));
    }
    if events.len() == MAX_EVENTS {
        return Err(resource(
            path,
            "shared area event inventory exceeds admission",
        ));
    }
    events.push(y);
    Ok(())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "the caller admits integer lattice edges into the safe i32 range; cell indices are clipped to that lattice"
)]
pub(super) fn rasterize(
    triangles: &[[f64; 2]],
    grid: &NeutralMask,
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<Vec<f64>, Render2dExecutionError> {
    // Both schedules integrate the same canonical triangle union. A global
    // arrangement can be much more expensive than local cells for a detailed
    // curved tessellation. Select BEFORE constructing it, without spending a
    // failed query's work or giving the alternative a fresh allowance. This
    // estimate selects a schedule; actual debits still decide admission.
    charge_shadow_preparation_work(work, 128, path)?;
    let n = triangles.len() / 3;
    let estimate = (|| {
        let edges = n.checked_mul(3)?;
        let pairs = edges.checked_mul(edges.saturating_sub(1))?.checked_div(2)?;
        let events = grid
            .width
            .checked_add(4)?
            .checked_mul(edges)?
            .checked_add(pairs)?
            .checked_add(grid.height)?
            .checked_add(2)?;
        if events > MAX_EVENTS {
            return None;
        }
        let row = n
            .checked_mul(128)?
            .checked_add(n.checked_mul(n)?.checked_mul(16)?)?
            .checked_add(64)?;
        // Disjoint intervals write at most width+2n column fragments per row.
        let columns = grid.width.checked_add(n.checked_mul(3)?)?.checked_mul(32)?;
        let sorting = usize::try_from(events.ilog2())
            .ok()?
            .checked_add(2)?
            .checked_mul(64)?;
        let bands = row
            .checked_add(columns)?
            .checked_add(sorting)?
            .checked_mul(events)?;
        n.checked_pow(3)?
            .checked_mul(8)?
            .checked_add(bands)?
            .checked_add(grid.width.checked_mul(grid.height)?.checked_mul(32)?)
    })();
    if estimate.is_some_and(|cost| cost <= work.remaining(usize::MAX)) {
        rasterize_shared(triangles, grid, work, path)
    } else {
        rasterize_cells(triangles, grid, work, path)
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the caller admits safe i32-range integer sample origins"
)]
fn rasterize_cells(
    triangles: &[[f64; 2]],
    grid: &NeutralMask,
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<Vec<f64>, Render2dExecutionError> {
    let area = grid
        .width
        .checked_mul(grid.height)
        .ok_or_else(|| resource(path, "cell area extent overflow"))?;
    // Retained output coexists with a local sweep's constructor. Geometric
    // active growth is bounded by twice the copied input; admitted n³*8 edge
    // construction has at most 256 active triangles, as in the shared path.
    let peak = triangles
        .len()
        .checked_mul(3 * std::mem::size_of::<[f64; 2]>())
        .and_then(|bytes| {
            area.checked_mul(std::mem::size_of::<f64>())
                .and_then(|output| bytes.checked_add(output))
        })
        .and_then(|bytes| bytes.checked_add(grid.samples.capacity()))
        .and_then(|bytes| bytes.checked_add(2 * MAX_EVENTS * 8 + 256 * 3 * 32 + 4_096));
    if peak.is_none_or(|bytes| bytes > 64 * 1024 * 1024) {
        return Err(resource(path, "cell area peak storage exceeds admission"));
    }
    let cost = area
        .checked_mul(32)
        .ok_or_else(|| resource(path, "cell area initialization work overflow"))?;
    charge_shadow_preparation_work(work, cost, path)?;
    let mut values = filled(area, 0.0, path)?;
    for row in 0..grid.height {
        for column in 0..grid.width {
            values[row * grid.width + column] = super::area_sample(
                triangles,
                grid.origin_x as f64 + as_f64(column),
                grid.origin_y as f64 + as_f64(row),
                work,
                path,
            )?;
        }
    }
    Ok(values)
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "the caller admits integer lattice edges into the safe i32 range; cell indices are clipped to that lattice"
)]
fn rasterize_shared(
    triangles: &[[f64; 2]],
    grid: &NeutralMask,
    work: &mut dyn PreparationWork,
    path: &[usize],
) -> Result<Vec<f64>, Render2dExecutionError> {
    let left = grid.origin_x as f64;
    let top = grid.origin_y as f64;
    let right = left + as_f64(grid.width);
    let bottom = top + as_f64(grid.height);
    let area = grid
        .width
        .checked_mul(grid.height)
        .ok_or_else(|| resource(path, "shared area extent overflow"))?;
    // Before constructor allocation: its growing active Vec retains at most
    // twice the source vertices (plus the minimum small allocation). Its
    // n³*8 allowance admits at most 256 active triangles before edge/event
    // construction. Include both phases before their respective allocations.
    let constructor_peak = triangles
        .len()
        .checked_mul(3 * std::mem::size_of::<[f64; 2]>())
        .and_then(|bytes| bytes.checked_add(grid.samples.capacity()))
        .and_then(|bytes| bytes.checked_add(256 * 3 * std::mem::size_of::<([f64; 2], [f64; 2])>()))
        .and_then(|bytes| bytes.checked_add(2 * MAX_EVENTS * std::mem::size_of::<f64>() + 4_096));
    let shortcut_peak = triangles
        .len()
        .checked_mul(std::mem::size_of::<[f64; 2]>())
        .and_then(|bytes| bytes.checked_add(grid.samples.capacity()))
        .and_then(|bytes| {
            area.checked_mul(std::mem::size_of::<f64>())
                .and_then(|output| bytes.checked_add(output))
        })
        .and_then(|bytes| bytes.checked_add(4_096));
    if constructor_peak.is_none_or(|bytes| bytes > 64 * 1024 * 1024)
        || shortcut_peak.is_none_or(|bytes| bytes > 64 * 1024 * 1024)
    {
        return Err(resource(
            path,
            "shared area constructor or shortcut peak exceeds admission",
        ));
    }
    let mut sweep = NeutralUnionSweep::new(triangles, [left, top, right, bottom], 8, work, path)?;
    if sweep.fully_covered || sweep.active.is_empty() {
        let cost = area
            .checked_mul(16)
            .ok_or_else(|| resource(path, "shared area fill work overflow"))?;
        charge_shadow_preparation_work(work, cost, path)?;
        return filled(area, f64::from(u8::from(sweep.fully_covered)), path);
    }
    if sweep.events.len() > MAX_EVENTS || sweep.events.capacity() > MAX_EVENTS {
        return Err(resource(
            path,
            "shared area source event inventory exceeds admission",
        ));
    }
    let allocation_work = sweep
        .events
        .len()
        .checked_mul(2)
        .and_then(|cost| cost.checked_add(64))
        .ok_or_else(|| resource(path, "shared area event relocation work overflow"))?;
    charge_shadow_preparation_work(work, allocation_work, path)?;
    sweep
        .events
        .try_reserve_exact(MAX_EVENTS - sweep.events.len())
        .map_err(|_| resource(path, "shared area event allocation failed"))?;
    if sweep.events.capacity() > MAX_EVENTS {
        return Err(resource(
            path,
            "shared area event capacity exceeds admission",
        ));
    }
    // Include the caller's copied triangles and binary grid, this retained
    // arrangement, old+new event allocations during relocation, output and
    // simultaneous row buffers.
    // The constructor's temporary edges have already been released.
    let peak = triangles
        .len()
        .checked_mul(std::mem::size_of::<[f64; 2]>())
        .and_then(|bytes| bytes.checked_add(grid.samples.capacity()))
        .and_then(|bytes| {
            sweep
                .active
                .capacity()
                .checked_mul(std::mem::size_of::<[[f64; 2]; 3]>())
                .and_then(|active| bytes.checked_add(active))
        })
        .and_then(|bytes| {
            MAX_EVENTS
                .checked_mul(2 * std::mem::size_of::<f64>())
                .and_then(|events| bytes.checked_add(events))
        })
        .and_then(|bytes| {
            area.checked_mul(std::mem::size_of::<f64>())
                .and_then(|output| bytes.checked_add(output))
        })
        .and_then(|bytes| {
            sweep
                .active
                .len()
                .checked_mul(2 * std::mem::size_of::<[f64; 2]>())
                .and_then(|rows| bytes.checked_add(rows))
        })
        .and_then(|bytes| bytes.checked_add(4_096));
    if peak.is_none_or(|bytes| bytes > 64 * 1024 * 1024) {
        return Err(resource(path, "shared area peak storage exceeds admission"));
    }
    for row in 1..grid.height {
        charge_shadow_preparation_work(work, 16, path)?;
        event(&mut sweep.events, top + as_f64(row), path)?;
    }
    for triangle in &sweep.active {
        for edge in 0..3 {
            charge_shadow_preparation_work(work, 64, path)?;
            let a = triangle[edge];
            let b = triangle[(edge + 1) % 3];
            if a[0] == b[0] || a[1] == b[1] {
                continue;
            }
            if !(b[0] - a[0]).is_finite() || !(b[1] - a[1]).is_finite() {
                return Err(precision(path, "shared area edge difference is not finite"));
            }
            let first = a[0].min(b[0]).max(left).ceil() as i64;
            let last = a[0].max(b[0]).min(right).floor() as i64;
            for column in first..=last {
                charge_shadow_preparation_work(work, 64, path)?;
                let t = (column as f64 - a[0]) / (b[0] - a[0]);
                if t > 0.0 && t < 1.0 {
                    let y = (b[1] - a[1]).mul_add(t, a[1]);
                    if y > top && y < bottom {
                        event(&mut sweep.events, y, path)?;
                    }
                }
            }
        }
    }
    let levels =
        usize::try_from(sweep.events.len().ilog2()).expect("usize bit width fits usize") + 2;
    let sorting = sweep
        .events
        .len()
        .checked_mul(levels)
        .and_then(|cost| cost.checked_mul(64))
        .ok_or_else(|| resource(path, "shared area event sort work overflow"))?;
    charge_shadow_preparation_work(work, sorting, path)?;
    sweep.events.sort_unstable_by(f64::total_cmp);
    sweep.events.dedup();
    let row_work = sweep.interval_work(path)?;
    // Initialization plus final finite/range/clamp checks and stores.
    let initialization = area
        .checked_mul(32)
        .ok_or_else(|| resource(path, "shared area output work overflow"))?;
    charge_shadow_preparation_work(work, initialization, path)?;
    let mut values = filled(area, 0.0, path)?;
    for band in sweep.events.windows(2) {
        let height = band[1] - band[0];
        let y = band[0] + height * 0.5;
        // Usually one midpoint suffices. Adjacent representable event heights
        // have no interior float: use the right limit at the lower endpoint
        // and left limit at the upper endpoint. Their trapezoid integrates
        // the same affine clipped width, including horizontal births/deaths.
        let samples = if y > band[0] && y < band[1] {
            [(y, true, height), (y, true, 0.0)]
        } else {
            if height * 0.5 == 0.0 {
                return Err(precision(path, "shared area slab mass underflow"));
            }
            [
                (band[0], true, height * 0.5),
                (band[1], false, height * 0.5),
            ]
        };
        let row = (band[0] - top).floor() as usize;
        if row >= grid.height {
            return Err(precision(path, "shared area row escaped admitted lattice"));
        }
        for (y, from_above, mass) in samples {
            if mass == 0.0 {
                continue;
            }
            charge_shadow_preparation_work(work, row_work, path)?;
            for [begin, end] in sweep.intervals_limit(y, from_above, path)? {
                let first = (begin - left).floor().max(0.0) as usize;
                let last = ((end - left).ceil() as usize).min(grid.width);
                let cost = last
                    .saturating_sub(first)
                    .checked_mul(32)
                    .and_then(|cost| cost.checked_add(32))
                    .ok_or_else(|| resource(path, "shared area column work overflow"))?;
                charge_shadow_preparation_work(work, cost, path)?;
                for column in first..last {
                    let x = left + as_f64(column);
                    let width = end.min(x + 1.0) - begin.max(x);
                    values[row * grid.width + column] += width.max(0.0) * mass;
                }
            }
        }
    }
    for value in &mut values {
        if !value.is_finite() || !(-1.0e-9..=1.0 + 1.0e-9).contains(value) {
            return Err(precision(path, "shared continuous area is not normalized"));
        }
        *value = value.clamp(0.0, 1.0);
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::execution_2d::support::area_sample;

    #[test]
    fn curved_physical_inventory_keeps_identity_area_admission() {
        use crate::composition_2d::*;
        let item = Render2dItem::new(
            Render2dPrimitive::Fill {
                shape: Render2dShape::ellipse(Render2dRect::new(10.0, 10.0, 8.0, 8.0).unwrap()),
                brush: Render2dBrush::solid(Render2dColorRgba8::TRANSPARENT),
            },
            Render2dAffineTransform::IDENTITY,
            Vec::new(),
            Render2dOpacity::TRANSPARENT,
        );
        let mesh =
            crate::runtime::execution_2d::vector::neutral_support(&item, 0, 4.0, 16 * 1024 * 1024)
                .unwrap()
                .unwrap();
        let expected = mesh
            .triangles
            .as_chunks::<3>()
            .0
            .iter()
            .map(|triangle| super::super::orient(triangle[0], triangle[1], triangle[2]).abs() * 0.5)
            .sum::<f64>();
        let mut work = 0;
        let coverage = super::super::prepare_shadow_coverage_admitted(
            &mesh,
            0.0,
            0.0,
            4.0,
            &[7, 8],
            &mut work,
            1_048_576,
        )
        .unwrap()
        .unwrap();
        assert!((coverage.values.iter().sum::<f64>() - expected).abs() < 1.0e-9);
        assert!(coverage.values.contains(&1.0));
    }

    #[test]
    fn shared_bands_match_independent_cell_arrangements_at_crossings_and_seams() {
        let triangles = [
            [-1.75, -1.25],
            [4.25, 1.75],
            [0.25, 3.25],
            [-0.75, 2.75],
            [3.25, -0.75],
            [4.75, 3.75],
            [-1.75, -1.25],
            [4.25, 1.75],
            [0.25, 3.25],
            [1.01, 0.04],
            [1.04, 0.04],
            [1.01, 0.08],
        ];
        let grid = NeutralMask {
            origin_x: -2,
            origin_y: -2,
            width: 8,
            height: 6,
            samples: Vec::new(),
        };
        let mut shared_work = 0;
        let actual = rasterize(&triangles, &grid, &mut shared_work, &[7, 2]).unwrap();
        for row in 0..grid.height {
            for column in 0..grid.width {
                let mut reference_work = 0;
                let expected = area_sample(
                    &triangles,
                    -2.0 + as_f64(column),
                    -2.0 + as_f64(row),
                    &mut reference_work,
                    &[7, 2],
                )
                .unwrap();
                assert!(
                    (actual[row * grid.width + column] - expected).abs() < 1.0e-12,
                    "row={row} column={column}: {} vs {expected}",
                    actual[row * grid.width + column]
                );
            }
        }
        let tiny = [[1.01, 0.04], [1.04, 0.04], [1.01, 0.08]];
        let values = rasterize(&tiny, &grid, &mut 0, &[7, 3]).unwrap();
        assert!((values.iter().sum::<f64>() - 0.0006).abs() < 1.0e-14);
    }

    #[test]
    fn endpoint_limits_preserve_horizontal_triangle_births_and_deaths() {
        let triangles = [[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let sweep =
            NeutralUnionSweep::new(&triangles, [0.0, 0.0, 1.0, 2.0], 8, &mut 0, &[7, 5]).unwrap();
        assert_eq!(
            sweep.intervals_limit(1.0, false, &[7, 5]).unwrap(),
            vec![[0.0, 1.0]]
        );
        assert!(
            sweep
                .intervals_limit(1.0, true, &[7, 5])
                .unwrap()
                .is_empty()
        );
        let reversed = [[0.0, 1.0], [1.0, 0.0], [0.0, 0.0]];
        let sweep =
            NeutralUnionSweep::new(&reversed, [0.0, -1.0, 1.0, 1.0], 8, &mut 0, &[7, 6]).unwrap();
        assert_eq!(
            sweep.intervals_limit(0.0, true, &[7, 6]).unwrap(),
            vec![[0.0, 1.0]]
        );
        assert!(
            sweep
                .intervals_limit(0.0, false, &[7, 6])
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn source_arrangement_itself_cannot_exceed_the_shared_event_cap() {
        // Coincident triangles preserve one geometric union, but their raw
        // edge-pair intersections can otherwise grow before the consumer cap.
        let triangles = [[-1.0, -1.0], [1.0, -1.0], [0.0, 1.0]].repeat(200);
        assert!(matches!(
            NeutralUnionSweep::new(&triangles, [-3.0, -3.0, 3.0, 3.0], 8, &mut 0, &[7, 7]),
            Err(Render2dExecutionError::SampleSpace {
                kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
            }) if path == [7, 7]
        ));
    }

    #[test]
    fn shared_area_fuel_and_event_capacity_fail_at_the_owning_path() {
        let triangles = [[0.0, 0.0], [1.0, 0.0], [0.0, 65_537.0]];
        let grid = NeutralMask {
            origin_x: 0,
            origin_y: 0,
            width: 1,
            height: 65_537,
            samples: Vec::new(),
        };
        for mut fuel in [super::super::MAX_AGGREGATE_SHADOW_PREPARATION_WORK - 1, 0] {
            assert!(
                matches!(rasterize_shared(&triangles, &grid, &mut fuel, &[7, 4]),
                Err(Render2dExecutionError::SampleSpace {
                    kind: Render2dSampleSpaceError::ResourceLimit, path: Some(path), ..
                }) if path == [7, 4])
            );
        }
    }
}
