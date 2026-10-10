use super::super::program::abi::temporal_fallback;
use super::{RenderDeterministicLoweringError, WORD_BYTES};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct TemporalFallbackBufferLayout {
    pub(super) cell_count: u32,
    pub(super) resolved_bytes: u64,
    pub(super) availability_bytes: u64,
}

/// Admit the complete private fallback scratch before allocating retained
/// temporal identities or preparing primary work. Physical preparation uses
/// this same checked layout, never an independently recomputed budget law.
pub(super) fn temporal_fallback_layout(
    output_index: usize,
    extent: (u32, u32),
    alignment: u64,
    max_storage_buffer_binding_size: u64,
    max_buffer_size: u64,
) -> Result<TemporalFallbackBufferLayout, RenderDeterministicLoweringError> {
    let cell_count =
        extent
            .0
            .checked_mul(extent.1)
            .ok_or(RenderDeterministicLoweringError::SizeOverflow {
                field: "temporal fallback requested cell count",
            })?;
    let logical_row_bytes = u64::from(extent.0).checked_mul(WORD_BYTES).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "temporal fallback row bytes",
        },
    )?;
    let row_bytes = align_up(logical_row_bytes, alignment)?;
    if row_bytes % WORD_BYTES != 0 {
        return Err(RenderDeterministicLoweringError::InvalidBytesPerRowAlignment { alignment });
    }
    u32::try_from(row_bytes / WORD_BYTES).map_err(|_| {
        RenderDeterministicLoweringError::SizeOverflow {
            field: "temporal fallback row stride",
        }
    })?;
    let resolved_bytes = row_bytes.checked_mul(u64::from(extent.1)).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "temporal fallback resolved bytes",
        },
    )?;
    let availability_bytes = u64::from(cell_count).checked_mul(WORD_BYTES).ok_or(
        RenderDeterministicLoweringError::SizeOverflow {
            field: "temporal fallback cell state",
        },
    )?;
    let peak_scratch_bytes = resolved_bytes
        .checked_add(availability_bytes)
        .and_then(|bytes| bytes.checked_add(availability_bytes))
        .ok_or(RenderDeterministicLoweringError::SizeOverflow {
            field: "temporal fallback aggregate scratch",
        })?;
    if peak_scratch_bytes > temporal_fallback::MAX_PER_OUTPUT_SCRATCH_BYTES {
        return Err(
            RenderDeterministicLoweringError::TemporalFallbackScratchBudgetExceeded {
                output_index,
                required_bytes: peak_scratch_bytes,
                budget_bytes: temporal_fallback::MAX_PER_OUTPUT_SCRATCH_BYTES,
            },
        );
    }
    let limit_bytes = max_storage_buffer_binding_size.min(max_buffer_size);
    for (carrier, bytes) in [
        ("resolved radiance", resolved_bytes),
        ("phase presence", availability_bytes),
        ("cell availability", availability_bytes),
    ] {
        if bytes > limit_bytes {
            return Err(
                RenderDeterministicLoweringError::TemporalFallbackGpuLimitExceeded {
                    output_index,
                    carrier,
                    required_bytes: bytes,
                    limit_bytes,
                },
            );
        }
    }
    Ok(TemporalFallbackBufferLayout {
        cell_count,
        resolved_bytes,
        availability_bytes,
    })
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

    #[test]
    fn fallback_layout_uses_actual_padded_rows_and_dense_availability() {
        let layout = temporal_fallback_layout(0, (7, 5), 256, u64::MAX, u64::MAX).unwrap();
        assert_eq!(layout.cell_count, 35);
        assert_eq!(layout.resolved_bytes, 256 * 5);
        assert_eq!(layout.availability_bytes, 35 * WORD_BYTES);
    }

    #[test]
    fn fallback_layout_fails_budget_before_retained_allocation() {
        assert!(matches!(
            temporal_fallback_layout(2, (8192, 8192), 256, u64::MAX, u64::MAX),
            Err(RenderDeterministicLoweringError::TemporalFallbackScratchBudgetExceeded {
                output_index: 2,
                required_bytes,
                budget_bytes,
            }) if required_bytes > budget_bytes
        ));
    }

    #[test]
    fn fallback_layout_fails_device_binding_limit_and_cell_overflow() {
        assert!(matches!(
            temporal_fallback_layout(0, (7, 5), 256, 128, u64::MAX),
            Err(
                RenderDeterministicLoweringError::TemporalFallbackGpuLimitExceeded {
                    output_index: 0,
                    carrier: "resolved radiance",
                    required_bytes: 1280,
                    limit_bytes: 128,
                }
            )
        ));
        assert!(matches!(
            temporal_fallback_layout(0, (u32::MAX, 2), 256, u64::MAX, u64::MAX),
            Err(RenderDeterministicLoweringError::SizeOverflow {
                field: "temporal fallback requested cell count",
            })
        ));
    }
}
