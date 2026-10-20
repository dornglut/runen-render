//! Ordered, bounded classification of globally phased kernel footprints.
//! Visitors consume borrowed original-region rows, without a demand census
//! or an intermediate per-sample classification plane. Unknown source slabs
//! remain separate: `Outside` alone never proves the complete source empty.
use super::*;

pub(super) struct Span<'a> {
    pub(super) y: u32,
    pub(super) offset: usize,
    pub(super) xs: &'a [f64],
    pub(super) cy: f64,
    pub(super) ys: Interval,
    pub(super) window: Interval,
    pub(super) outside_reach: bool,
    pub(super) class: Class,
    pub(super) bands: &'a [RowBand],
    pub(super) runs: &'a [RowRun],
    pub(super) pieces: &'a [Class],
}

/// Classifier-owned heap data and vector headers only. Source plans, scalar
/// stack, visitor state and output add to an invocation's peak inventory.
pub(super) fn scratch_bytes(width: usize) -> Option<usize> {
    width
        .checked_mul(std::mem::size_of::<f64>())?
        .checked_add(MAX_BANDS * (std::mem::size_of::<RowBand>() + std::mem::size_of::<Class>()))?
        .checked_add(MAX_RUNS * std::mem::size_of::<RowRun>())?
        .checked_add(64 * std::mem::size_of::<(usize, usize)>())?
        .checked_add(5 * std::mem::size_of::<Vec<usize>>())
}

pub(super) fn visit(
    strips: &Strips,
    plan: &ScanPlan,
    region: &NeutralRegion<'_>,
    lattice: Lattice,
    fuel: &mut usize,
    mut visitor: impl FnMut(Span<'_>, &mut usize) -> Result<(), AreaQueryError>,
) -> Result<usize, AreaQueryError> {
    debit(fuel, 512)?;
    let Lattice {
        root: [x0, y0, x1, y1],
        scale,
        sigma,
        offset: translation,
    } = lattice;
    let width = x1.checked_sub(x0).ok_or(AreaQueryError::Precision)? as usize;
    let height = y1.checked_sub(y0).ok_or(AreaQueryError::Precision)? as usize;
    let cells = width.checked_mul(height).ok_or(AreaQueryError::Resource)?;
    if cells == 0
        || cells > 1_048_576
        || !scale.is_finite()
        || scale <= 0.0
        || !sigma.is_finite()
        || sigma <= 0.0
    {
        return Err(AreaQueryError::Precision);
    }
    let scratch = scratch_bytes(width).ok_or(AreaQueryError::Resource)?;
    if scratch > MAX_BYTES {
        return Err(AreaQueryError::Resource);
    }
    let halo = checked(iv(sigma).mul(iv(3.0)))?.hi;
    let reach = region
        .certified_logical_envelope()
        .ok_or(AreaQueryError::Unsupported)?;
    let mut xs = bounded_vector(width).ok_or(AreaQueryError::Resource)?;
    let mut bands = bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?;
    let mut runs = bounded_vector(MAX_RUNS).ok_or(AreaQueryError::Resource)?;
    let mut pieces = bounded_vector(MAX_BANDS).ok_or(AreaQueryError::Resource)?;
    let mut pending = bounded_vector(64).ok_or(AreaQueryError::Resource)?;
    if xs.capacity() != width
        || bands.capacity() != MAX_BANDS
        || runs.capacity() != MAX_RUNS
        || pieces.capacity() != MAX_BANDS
        || pending.capacity() != 64
    {
        return Err(AreaQueryError::Resource);
    }
    for x in x0..x1 {
        debit(fuel, 64)?;
        let cx = exact_div(
            exact_add(f64::from(x), 0.5).ok_or(AreaQueryError::Precision)?,
            scale,
        )
        .ok_or(AreaQueryError::Precision)?;
        xs.push(if translation[0] == 0.0 {
            cx
        } else {
            debit(fuel, 32)?;
            exact_add(cx, -translation[0]).ok_or(AreaQueryError::Precision)?
        });
    }
    let horizontal = checked(Interval::new(-halo, halo))?;
    for y in y0..y1 {
        debit(fuel, 128)?;
        let mut cy = exact_div(
            exact_add(f64::from(y), 0.5).ok_or(AreaQueryError::Precision)?,
            scale,
        )
        .ok_or(AreaQueryError::Precision)?;
        if translation[1] != 0.0 {
            debit(fuel, 32)?;
            cy = exact_add(cy, -translation[1]).ok_or(AreaQueryError::Precision)?;
        }
        let ys = checked(iv(cy).add(horizontal))?;
        let offset = (y - y0) as usize * width;
        bands.clear();
        runs.clear();
        pieces.clear();
        if ys.hi <= reach[1] || ys.lo >= reach[3] {
            let window = checked(checked(Interval::new(xs[0], xs[width - 1]))?.add(horizontal))?;
            visitor(
                Span {
                    y,
                    offset,
                    xs: &xs,
                    cy,
                    ys,
                    window,
                    outside_reach: true,
                    class: Class::Outside,
                    bands: &bands,
                    runs: &runs,
                    pieces: &pieces,
                },
                fuel,
            )?;
            continue;
        }
        if ys.lo < strips.bounds[1] || ys.hi > strips.bounds[3] {
            return Err(AreaQueryError::Precision);
        }
        for band in &strips.bands {
            debit(fuel, 16)?;
            if band.end <= ys.lo || band.start >= ys.hi {
                continue;
            }
            let local = checked(Interval::new(band.start.max(ys.lo), band.end.min(ys.hi)))?;
            let first = runs.len();
            for &ids in &strips.runs[band.first..band.first + band.count] {
                let left = curve_range(plan, strips.bounds, ids[0], local, fuel)?;
                let right = curve_range(plan, strips.bounds, ids[1], local, fuel)?;
                if runs.len() == MAX_RUNS {
                    return Err(AreaQueryError::Resource);
                }
                runs.push(RowRun {
                    ids,
                    x: [left, right],
                });
            }
            if bands.len() == MAX_BANDS {
                return Err(AreaQueryError::Resource);
            }
            bands.push(RowBand {
                y: local,
                first,
                count: runs.len() - first,
                #[cfg(test)]
                mass: None,
            });
        }
        pending.push((0, width));
        while let Some((first, end)) = pending.pop() {
            debit(fuel, 128)?;
            let window = checked(checked(Interval::new(xs[first], xs[end - 1]))?.add(horizontal))?;
            let outside_reach = window.hi <= reach[0] || window.lo >= reach[2];
            pieces.clear();
            let class = if outside_reach {
                Class::Outside
            } else {
                if window.lo < strips.bounds[0] || window.hi > strips.bounds[2] {
                    return Err(AreaQueryError::Precision);
                }
                classify_site(plan, &bands, &runs, window, &mut pieces, fuel)?
            };
            if matches!(class, Class::General) && end - first > 1 {
                if pending.len() + 2 > 64 {
                    return Err(AreaQueryError::Resource);
                }
                let middle = first + (end - first) / 2;
                // LIFO visits the left half first, so visitors append output
                // exactly once in row-major order, including empty regions.
                pending.extend([(middle, end), (first, middle)]);
                continue;
            }
            visitor(
                Span {
                    y,
                    offset: offset + first,
                    xs: &xs[first..end],
                    cy,
                    ys,
                    window,
                    outside_reach,
                    class,
                    bands: &bands,
                    runs: &runs,
                    pieces: &pieces,
                },
                fuel,
            )?;
        }
    }
    Ok(scratch)
}
