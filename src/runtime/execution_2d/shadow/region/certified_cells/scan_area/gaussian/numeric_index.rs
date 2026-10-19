//! Bounded exact numeric-key index for disposable numerical certificates.
use super::{AreaQueryError, debit};

const SLOTS: usize = 1024;
const ENTRIES: usize = 512;
const PROBES: usize = 64;

pub(super) struct Index<const N: usize> {
    slots: Vec<usize>,
    pub(super) keys: Vec<[u64; N]>,
    pub(super) maximum_probes: usize,
}

impl<const N: usize> Index<N> {
    pub(super) fn heap_bound() -> Option<usize> {
        SLOTS
            .checked_mul(size_of::<usize>())?
            .checked_add(ENTRIES.checked_mul(size_of::<[u64; N]>())?)
    }

    pub(super) fn new(fuel: &mut usize) -> Result<Self, AreaQueryError> {
        debit(fuel, 256 + SLOTS * 8)?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(SLOTS)
            .map_err(|_| AreaQueryError::Resource)?;
        slots.resize(SLOTS, 0);
        let mut keys = Vec::new();
        keys.try_reserve_exact(ENTRIES)
            .map_err(|_| AreaQueryError::Resource)?;
        if slots.capacity() != SLOTS || keys.capacity() != ENTRIES {
            return Err(AreaQueryError::Resource);
        }
        Ok(Self {
            slots,
            keys,
            maximum_probes: 0,
        })
    }

    pub(super) fn observe(
        &mut self,
        key: [u64; N],
        fuel: &mut usize,
    ) -> Result<(usize, bool), AreaQueryError> {
        debit(fuel, 64)?;
        let mut hash = 0x9e37_79b9_7f4a_7c15_u64;
        for word in key {
            debit(fuel, 16)?;
            hash = (hash ^ word)
                .rotate_left(27)
                .wrapping_mul(0x94d0_49bb_1331_11eb);
        }
        hash = (hash ^ (hash >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        hash = (hash ^ (hash >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        hash ^= hash >> 31;
        for probe in 1..=PROBES {
            debit(fuel, 32 + N * 8)?;
            let slot = (hash as usize).wrapping_add(probe - 1) & (SLOTS - 1);
            let entry = self.slots[slot];
            self.maximum_probes = self.maximum_probes.max(probe);
            if entry == 0 {
                if self.keys.len() == ENTRIES {
                    return Err(AreaQueryError::Resource);
                }
                debit(fuel, 32 + N * 8)?;
                self.keys.push(key);
                self.slots[slot] = self.keys.len();
                return Ok((self.keys.len() - 1, true));
            }
            // Hash equality never supplies numeric or geometric identity.
            if self.keys[entry - 1] == key {
                return Ok((entry - 1, false));
            }
        }
        Err(AreaQueryError::Resource)
    }

    pub(super) fn bytes(&self) -> usize {
        self.slots.capacity() * size_of::<usize>() + self.keys.capacity() * size_of::<[u64; N]>()
    }
}

#[test]
fn charged_index_distinguishes_hash_collisions_and_rejects_exhaustion() {
    let mut fuel = 16_777_216;
    let mut index = Index::<2>::new(&mut fuel).unwrap();
    for value in 0..200 {
        assert_eq!(
            index.observe([value, value + 1], &mut fuel).unwrap(),
            (value as usize, true)
        );
        assert_eq!(
            index.observe([value, value + 1], &mut fuel).unwrap(),
            (value as usize, false)
        );
    }
    assert_eq!(index.keys.len(), 200);
    assert!(index.maximum_probes > 1);
    let mut empty = 0;
    assert!(matches!(
        index.observe([999, 0], &mut empty),
        Err(AreaQueryError::Resource)
    ));
    assert_eq!(index.keys.len(), 200);
}
