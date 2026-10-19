//! Combined renderer-owned numerical storage at distinct preparation phases.
//! Original scene/DAG storage is separately owned by the invoking source plan.
use super::*;

fn sum(terms: impl IntoIterator<Item = Option<usize>>) -> Option<usize> {
    terms
        .into_iter()
        .try_fold(0_usize, |sum, term| sum.checked_add(term?))
}

fn heap<T>(values: &Vec<T>) -> Option<usize> {
    values.capacity().checked_mul(size_of::<T>())
}

pub(super) fn query_plans(prepared: &PreparedBoundaries, plan: &ScanPlan) -> Option<usize> {
    // Boundaries can be shared. Counting every retained entry and the full
    // admitted Boundary allocation/control block is conservative.
    sum([
        heap(&prepared.entries),
        heap(&prepared.sources),
        prepared
            .entries
            .len()
            .checked_mul(orthogonal_boundary::Boundary::retained_allocation_bound()?),
        Some(size_of::<PreparedBoundaries>()),
        heap(&plan.curves),
        heap(&plan.crossings),
        heap(&plan.extra_y_events),
        Some(size_of::<ScanPlan>()),
    ])
}

pub(super) fn strip_preparation(
    prepared: &PreparedBoundaries,
    plan: &ScanPlan,
    fuel: &mut usize,
) -> Result<usize, AreaQueryError> {
    debit(fuel, 512)?;
    let count = plan.curves.len();
    let peak = sum([
        query_plans(prepared, plan),
        // Event cuts, semantic cuts, unresolved intervals.
        MAX_BANDS.checked_mul(2 * size_of::<f64>() + size_of::<Interval>()),
        // Two complete finite-domain crossings per original boundary.
        count.checked_mul(2 * size_of::<Crossings>()),
        // Per-curve sorted crossing intervals: all vectors coexist.
        count.checked_mul(size_of::<Vec<f64>>()),
        count
            .checked_mul(count)
            .and_then(|v| v.checked_mul(4 * size_of::<f64>())),
        Some(2 * size_of::<f64>()), // one membership-query x interval
        MembershipCache::fixed_scratch_bytes(),
        // Result capacities remain live alongside their construction inputs.
        MAX_BANDS.checked_mul(size_of::<Band>() + size_of::<Interval>()),
        MAX_RUNS.checked_mul(size_of::<[usize; 2]>()),
        count
            .checked_add(2)
            .and_then(|v| v.checked_mul(size_of::<(usize, &Curve, Interval, f64)>())),
        Some(size_of::<Strips>() + size_of::<GridQuery<'_, '_>>()),
        // Existing scalar/frame/header reservation, independent of the above
        // actual heap capacities. Sorting is bounded allocation-free heapsort.
        Some(65_536),
    ])
    .ok_or(AreaQueryError::Resource)?;
    if peak > MAX_AREA_QUERY_FIXED_BYTES as usize || peak > MAX_BYTES {
        return Err(AreaQueryError::Resource);
    }
    Ok(peak)
}
