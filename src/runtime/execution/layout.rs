use super::RenderDeterministicLoweringError;

pub(super) fn align_up(
    value: u64,
    alignment: u64,
) -> Result<u64, RenderDeterministicLoweringError> {
    if alignment == 0 {
        return Err(RenderDeterministicLoweringError::InvalidBytesPerRowAlignment { alignment });
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value.checked_add(alignment - remainder).ok_or(
            RenderDeterministicLoweringError::SizeOverflow {
                field: "aligned lattice row bytes",
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_alignment_preserves_aligned_values_and_rounds_up() {
        assert_eq!(align_up(256, 256).expect("aligned value"), 256);
        assert_eq!(align_up(257, 256).expect("rounded value"), 512);
    }

    #[test]
    fn row_alignment_rejects_zero_alignment() {
        assert!(matches!(
            align_up(1, 0),
            Err(RenderDeterministicLoweringError::InvalidBytesPerRowAlignment { alignment: 0 })
        ));
    }
}
