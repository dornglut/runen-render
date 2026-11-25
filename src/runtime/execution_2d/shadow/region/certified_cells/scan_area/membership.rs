//! Bounded reuse of original-region certificates within complete topology bands.
//! Entries are invocation-local proofs, never sampled support or paint alpha.
use super::*;

const MEMBERSHIP_SLOTS: usize = 16_384;

#[derive(Clone, Copy)]
struct MembershipEntry {
    pair: [usize; 2],
    band: Interval,
    value: CellCertificate,
}

pub(super) struct MembershipCache {
    slots: Vec<Option<MembershipEntry>>,
    used: usize,
}

impl MembershipCache {
    pub(super) fn fixed_scratch_bytes() -> Option<usize> {
        MEMBERSHIP_SLOTS.checked_mul(std::mem::size_of::<Option<MembershipEntry>>())
    }

    pub(super) fn new(fuel: &mut usize) -> Result<Self, AreaQueryError> {
        charge(fuel, 6 * MEMBERSHIP_SLOTS).ok_or(AreaQueryError::Resource)?;
        let mut slots = bounded_vector(MEMBERSHIP_SLOTS).ok_or(AreaQueryError::Resource)?;
        slots.resize(MEMBERSHIP_SLOTS, None);
        Ok(Self { slots, used: 0 })
    }

    fn find(
        &self,
        key: [usize; 2],
        y: f64,
        fuel: &mut usize,
    ) -> Result<(usize, Option<CellCertificate>), AreaQueryError> {
        charge(fuel, 32).ok_or(AreaQueryError::Resource)?;
        let mut index = key.into_iter().fold(0x811c_9dc5_usize, |hash, value| {
            hash.wrapping_mul(0x0100_0193).rotate_left(7) ^ value
        }) & (MEMBERSHIP_SLOTS - 1);
        for _ in 0..MEMBERSHIP_SLOTS {
            charge(fuel, 16).ok_or(AreaQueryError::Resource)?;
            match self.slots[index] {
                Some(entry) if entry.pair == key => {
                    return Ok((
                        index,
                        (entry.band.lo < y && y < entry.band.hi).then_some(entry.value),
                    ));
                }
                None => return Ok((index, None)),
                Some(_) => index = (index + 1) & (MEMBERSHIP_SLOTS - 1),
            }
        }
        Err(AreaQueryError::Resource)
    }
}

impl GridQuery<'_, '_> {
    pub(super) fn membership(
        &mut self,
        pair: [usize; 2],
        point: Cell,
        y: f64,
        fuel: &mut usize,
    ) -> Result<CellCertificate, AreaQueryError> {
        charge(fuel, 64).ok_or(AreaQueryError::Resource)?;
        let slot = if let Some(cache) = &self.membership {
            let (slot, value) = cache.find(pair, y, fuel)?;
            if let Some(value) = value {
                return Ok(value);
            }
            Some(slot)
        } else {
            None
        };
        let result = self
            .region
            .certify_prepared(point, fuel, Some(self.prepared));
        if *fuel == 0 {
            return Err(AreaQueryError::Resource);
        }
        if result != CellCertificate::Unknown
            && let (Some(cache), Some(slot)) = (&mut self.membership, slot)
            && cache.used < MEMBERSHIP_SLOTS / 2
        {
            // A strip can gain/lose a boundary only at a branch endpoint or
            // a crossing involving one of its two bordering branches. Other
            // crossings elsewhere do not disconnect this open strip.
            charge(fuel, 256).ok_or(AreaQueryError::Resource)?;
            let cut = self.semantic_cuts.partition_point(|cut| *cut <= y);
            let mut low = cut
                .checked_sub(1)
                .map_or(f64::MIN, |i| self.semantic_cuts[i]);
            let mut high = self.semantic_cuts.get(cut).copied().unwrap_or(f64::MAX);
            for id in pair {
                let Some(cuts) = self.curve_cuts.get(id) else {
                    continue;
                }; // clip-plane events are global
                let cut = cuts.partition_point(|cut| *cut <= y);
                if let Some(index) = cut.checked_sub(1) {
                    low = low.max(cuts[index]);
                }
                if let Some(&end) = cuts.get(cut) {
                    high = high.min(end);
                }
            }
            charge(fuel, 8).ok_or(AreaQueryError::Resource)?;
            if cache.slots[slot].is_none() {
                cache.used += 1;
            }
            cache.slots[slot] = Some(MembershipEntry {
                pair,
                band: Interval::new(low, high).ok_or(AreaQueryError::Precision)?,
                value: result,
            });
        }
        Ok(result)
    }
}
