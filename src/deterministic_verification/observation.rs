//! Same-submission physical observation decoding for RR566-EVAL-001.
//!
//! This module consumes only renderer-private readbacks already correlated to one exact maintained
//! `GpuSubmission`. It normalizes the maintained one-word carrier into logical per-sample words
//! using the retained semantic topology plus the exact returned canonical-buffer byte length. It
//! never re-derives physical packing from device facts and it does not perform semantic verification.

use super::super::deterministic_carrier::{WORD_BYTES, decode_word};
use super::super::deterministic_execution::DeterministicVerificationSubmission;
use super::super::request::RenderResultTopology;
use runen_gpu::{
    GpuReadbackBytes, GpuReadbackId, GpuReadbackStatus, GpuSubmission, GpuSubmissionFailureKind,
    GpuSubmissionStatus,
};
use std::error::Error;
use std::fmt;

/// Renderer-private normalized physical observations for one exact admitted output.
///
/// `canonical_words` contains only logical semantic samples. Any canonical-buffer row padding used
/// by maintained lowering is validated and removed here. Definedness and evaluator-status channels
/// are already tightly packed one-word-per-sample buffers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DeterministicVerificationObservation {
    output_index: usize,
    canonical_words: Vec<u32>,
    definedness_words: Vec<u32>,
    status_words: Vec<u32>,
}

impl DeterministicVerificationObservation {
    pub(crate) const fn output_index(&self) -> usize {
        self.output_index
    }

    pub(crate) fn canonical_words(&self) -> &[u32] {
        &self.canonical_words
    }

    pub(crate) fn definedness_words(&self) -> &[u32] {
        &self.definedness_words
    }

    pub(crate) fn status_words(&self) -> &[u32] {
        &self.status_words
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RenderDeterministicVerificationObservationError {
    SubmissionPending,
    SubmissionFailed {
        kind: GpuSubmissionFailureKind,
    },
    CorrelationChanged {
        output_index: usize,
        channel: &'static str,
    },
    ReadbackPending {
        output_index: usize,
        channel: &'static str,
    },
    ReadbackFailed {
        output_index: usize,
        channel: &'static str,
        kind: GpuSubmissionFailureKind,
    },
    InvalidPhysicalLayout {
        output_index: usize,
        channel: &'static str,
        byte_len: u64,
    },
    SizeOverflow {
        output_index: usize,
        field: &'static str,
    },
    HostAllocation {
        output_index: usize,
        channel: &'static str,
    },
}

impl RenderDeterministicVerificationObservationError {
    pub(super) const fn output_index(&self) -> Option<usize> {
        match self {
            Self::CorrelationChanged { output_index, .. }
            | Self::ReadbackPending { output_index, .. }
            | Self::ReadbackFailed { output_index, .. }
            | Self::InvalidPhysicalLayout { output_index, .. }
            | Self::SizeOverflow { output_index, .. }
            | Self::HostAllocation { output_index, .. } => Some(*output_index),
            Self::SubmissionPending | Self::SubmissionFailed { .. } => None,
        }
    }

    pub(super) const fn channel(&self) -> Option<&'static str> {
        match self {
            Self::CorrelationChanged { channel, .. }
            | Self::ReadbackPending { channel, .. }
            | Self::ReadbackFailed { channel, .. }
            | Self::InvalidPhysicalLayout { channel, .. }
            | Self::HostAllocation { channel, .. } => Some(*channel),
            Self::SubmissionPending | Self::SubmissionFailed { .. } | Self::SizeOverflow { .. } => {
                None
            }
        }
    }

    pub(super) const fn gpu_failure_kind(&self) -> Option<GpuSubmissionFailureKind> {
        match self {
            Self::SubmissionFailed { kind } | Self::ReadbackFailed { kind, .. } => Some(*kind),
            _ => None,
        }
    }
}

impl fmt::Display for RenderDeterministicVerificationObservationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SubmissionPending => formatter.write_str("verified submission is still pending"),
            Self::SubmissionFailed { kind } => {
                write!(formatter, "verified submission failed: {kind:?}")
            }
            Self::CorrelationChanged {
                output_index,
                channel,
            } => write!(
                formatter,
                "output {output_index} {channel} observation lost exact-submission correlation"
            ),
            Self::ReadbackPending {
                output_index,
                channel,
            } => write!(
                formatter,
                "output {output_index} {channel} readback is still pending"
            ),
            Self::ReadbackFailed {
                output_index,
                channel,
                kind,
            } => write!(
                formatter,
                "output {output_index} {channel} readback failed: {kind:?}"
            ),
            Self::InvalidPhysicalLayout {
                output_index,
                channel,
                byte_len,
            } => write!(
                formatter,
                "output {output_index} {channel} has invalid physical readback layout with {byte_len} bytes"
            ),
            Self::SizeOverflow {
                output_index,
                field,
            } => write!(
                formatter,
                "output {output_index} {field} exceeds host indexing limits during observation normalization"
            ),
            Self::HostAllocation {
                output_index,
                channel,
            } => write!(
                formatter,
                "host allocation failed while normalizing output {output_index} {channel}"
            ),
        }
    }
}

impl Error for RenderDeterministicVerificationObservationError {}

/// Observe one completed verified submission without yet claiming semantic correctness.
///
/// The exact returned canonical buffer supplies physical byte-length evidence. The retained
/// `RenderRequest` remains the sole authority for scalar versus lattice topology and logical lattice
/// dimensions. Combining those already-bound facts makes the maintained row stride uniquely
/// derivable, so no duplicate device/alignment metadata is retained in renderer state.
pub(crate) fn observe_completed_deterministic_verification(
    verification: &DeterministicVerificationSubmission,
) -> Result<
    Vec<DeterministicVerificationObservation>,
    RenderDeterministicVerificationObservationError,
> {
    let submitted = verification.submitted();
    match submitted.submission().status() {
        GpuSubmissionStatus::Accepted => {
            return Err(RenderDeterministicVerificationObservationError::SubmissionPending);
        }
        GpuSubmissionStatus::Failed(failure) => {
            return Err(
                RenderDeterministicVerificationObservationError::SubmissionFailed {
                    kind: failure.kind(),
                },
            );
        }
        GpuSubmissionStatus::Completed => {}
    }

    let admitted = submitted.admitted().admitted();
    let correlations = verification.readbacks();
    if correlations.len() != admitted.outputs().len() {
        return Err(
            RenderDeterministicVerificationObservationError::CorrelationChanged {
                output_index: 0,
                channel: "observation-set",
            },
        );
    }

    let mut observations = Vec::new();
    observations
        .try_reserve_exact(correlations.len())
        .map_err(
            |_| RenderDeterministicVerificationObservationError::HostAllocation {
                output_index: 0,
                channel: "observation-set",
            },
        )?;

    for correlation in correlations {
        let output_index = correlation.output_index();
        let requested = admitted
            .plan()
            .request()
            .outputs()
            .get(output_index)
            .copied()
            .ok_or(
                RenderDeterministicVerificationObservationError::CorrelationChanged {
                    output_index,
                    channel: "requested-output",
                },
            )?;
        let topology = requested.spec().topology();
        let sample_count = semantic_sample_count(output_index, topology)?;

        let canonical = ready_buffer_readback(
            submitted.submission(),
            output_index,
            "canonical-output",
            correlation.canonical_output(),
        )?;
        let definedness = ready_buffer_readback(
            submitted.submission(),
            output_index,
            "definedness",
            correlation.definedness(),
        )?;
        let status = ready_buffer_readback(
            submitted.submission(),
            output_index,
            "evaluator-status",
            correlation.status(),
        )?;

        let canonical_words =
            normalize_canonical_words(output_index, topology, canonical.as_bytes())?;
        let definedness_words = decode_tightly_packed_words(
            output_index,
            "definedness",
            sample_count,
            definedness.as_bytes(),
        )?;
        let status_words = decode_tightly_packed_words(
            output_index,
            "evaluator-status",
            sample_count,
            status.as_bytes(),
        )?;

        observations.push(DeterministicVerificationObservation {
            output_index,
            canonical_words,
            definedness_words,
            status_words,
        });
    }

    Ok(observations)
}

fn ready_buffer_readback(
    submission: &GpuSubmission,
    output_index: usize,
    channel: &'static str,
    id: GpuReadbackId,
) -> Result<GpuReadbackBytes, RenderDeterministicVerificationObservationError> {
    let readback = submission.readback(id).ok_or(
        RenderDeterministicVerificationObservationError::CorrelationChanged {
            output_index,
            channel,
        },
    )?;
    let bytes = match readback.status() {
        GpuReadbackStatus::Pending => {
            return Err(
                RenderDeterministicVerificationObservationError::ReadbackPending {
                    output_index,
                    channel,
                },
            );
        }
        GpuReadbackStatus::Failed(failure) => {
            return Err(
                RenderDeterministicVerificationObservationError::ReadbackFailed {
                    output_index,
                    channel,
                    kind: failure.kind(),
                },
            );
        }
        GpuReadbackStatus::Ready(bytes) => bytes,
    };
    let actual_byte_len = byte_len(output_index, bytes.as_bytes(), "readback byte length")?;
    if bytes.texture_format().is_some() || bytes.layout().byte_len() != actual_byte_len {
        return Err(
            RenderDeterministicVerificationObservationError::InvalidPhysicalLayout {
                output_index,
                channel,
                byte_len: actual_byte_len,
            },
        );
    }
    Ok(bytes)
}

fn semantic_sample_count(
    output_index: usize,
    topology: RenderResultTopology,
) -> Result<u64, RenderDeterministicVerificationObservationError> {
    match topology.sample_lattice_dimensions() {
        Some((width, height)) => u64::from(width).checked_mul(u64::from(height)).ok_or(
            RenderDeterministicVerificationObservationError::SizeOverflow {
                output_index,
                field: "semantic sample count",
            },
        ),
        None => Ok(1),
    }
}

fn normalize_canonical_words(
    output_index: usize,
    topology: RenderResultTopology,
    bytes: &[u8],
) -> Result<Vec<u32>, RenderDeterministicVerificationObservationError> {
    let actual_byte_len = byte_len(output_index, bytes, "canonical byte length")?;
    let Some((width, height)) = topology.sample_lattice_dimensions() else {
        if bytes.len() != WORD_BYTES {
            return invalid_layout(output_index, "canonical-output", actual_byte_len);
        }
        let word: &[u8; WORD_BYTES] = bytes.try_into().map_err(|_| {
            RenderDeterministicVerificationObservationError::InvalidPhysicalLayout {
                output_index,
                channel: "canonical-output",
                byte_len: actual_byte_len,
            }
        })?;
        return Ok(vec![decode_word(word)]);
    };

    let height_u64 = u64::from(height);
    if actual_byte_len == 0 || actual_byte_len % height_u64 != 0 {
        return invalid_layout(output_index, "canonical-output", actual_byte_len);
    }
    let row_stride_bytes = actual_byte_len / height_u64;
    let logical_row_bytes = u64::from(width).checked_mul(WORD_BYTES as u64).ok_or(
        RenderDeterministicVerificationObservationError::SizeOverflow {
            output_index,
            field: "logical canonical row byte length",
        },
    )?;
    if row_stride_bytes < logical_row_bytes || row_stride_bytes % WORD_BYTES as u64 != 0 {
        return invalid_layout(output_index, "canonical-output", actual_byte_len);
    }

    let sample_count =
        usize::try_from(semantic_sample_count(output_index, topology)?).map_err(|_| {
            RenderDeterministicVerificationObservationError::SizeOverflow {
                output_index,
                field: "logical canonical sample count",
            }
        })?;
    let row_stride_bytes = usize::try_from(row_stride_bytes).map_err(|_| {
        RenderDeterministicVerificationObservationError::SizeOverflow {
            output_index,
            field: "canonical row stride",
        }
    })?;
    let logical_row_bytes = usize::try_from(logical_row_bytes).map_err(|_| {
        RenderDeterministicVerificationObservationError::SizeOverflow {
            output_index,
            field: "logical canonical row byte length",
        }
    })?;
    let height = usize::try_from(height).map_err(|_| {
        RenderDeterministicVerificationObservationError::SizeOverflow {
            output_index,
            field: "canonical row count",
        }
    })?;

    let mut words = Vec::new();
    words.try_reserve_exact(sample_count).map_err(|_| {
        RenderDeterministicVerificationObservationError::HostAllocation {
            output_index,
            channel: "canonical-output",
        }
    })?;
    for row in 0..height {
        let start = row.checked_mul(row_stride_bytes).ok_or(
            RenderDeterministicVerificationObservationError::SizeOverflow {
                output_index,
                field: "canonical row offset",
            },
        )?;
        let end = start.checked_add(logical_row_bytes).ok_or(
            RenderDeterministicVerificationObservationError::SizeOverflow {
                output_index,
                field: "canonical row range",
            },
        )?;
        let row_bytes = bytes.get(start..end).ok_or(
            RenderDeterministicVerificationObservationError::InvalidPhysicalLayout {
                output_index,
                channel: "canonical-output",
                byte_len: actual_byte_len,
            },
        )?;
        for chunk in row_bytes.as_chunks::<WORD_BYTES>().0 {
            words.push(decode_word(chunk));
        }
    }
    Ok(words)
}

fn decode_tightly_packed_words(
    output_index: usize,
    channel: &'static str,
    sample_count: u64,
    bytes: &[u8],
) -> Result<Vec<u32>, RenderDeterministicVerificationObservationError> {
    let expected_byte_len = sample_count.checked_mul(WORD_BYTES as u64).ok_or(
        RenderDeterministicVerificationObservationError::SizeOverflow {
            output_index,
            field: "tightly packed observation byte length",
        },
    )?;
    let actual_byte_len = byte_len(output_index, bytes, "observation byte length")?;
    if actual_byte_len != expected_byte_len {
        return invalid_layout(output_index, channel, actual_byte_len);
    }

    let sample_count = usize::try_from(sample_count).map_err(|_| {
        RenderDeterministicVerificationObservationError::SizeOverflow {
            output_index,
            field: "tightly packed observation word count",
        }
    })?;
    let mut words = Vec::new();
    words.try_reserve_exact(sample_count).map_err(|_| {
        RenderDeterministicVerificationObservationError::HostAllocation {
            output_index,
            channel,
        }
    })?;
    for chunk in bytes.as_chunks::<WORD_BYTES>().0 {
        words.push(decode_word(chunk));
    }
    Ok(words)
}

fn byte_len(
    output_index: usize,
    bytes: &[u8],
    field: &'static str,
) -> Result<u64, RenderDeterministicVerificationObservationError> {
    u64::try_from(bytes.len()).map_err(|_| {
        RenderDeterministicVerificationObservationError::SizeOverflow {
            output_index,
            field,
        }
    })
}

fn invalid_layout<T>(
    output_index: usize,
    channel: &'static str,
    byte_len: u64,
) -> Result<T, RenderDeterministicVerificationObservationError> {
    Err(
        RenderDeterministicVerificationObservationError::InvalidPhysicalLayout {
            output_index,
            channel,
            byte_len,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words_to_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_ne_bytes()).collect()
    }

    #[test]
    fn canonical_lattice_normalization_removes_physical_row_padding() {
        let topology = RenderResultTopology::sample_lattice_2d(2, 2).expect("2x2 topology");
        let bytes = words_to_bytes(&[11, 12, 91, 92, 21, 22, 93, 94]);
        assert_eq!(
            normalize_canonical_words(0, topology, &bytes).expect("padded canonical lattice"),
            vec![11, 12, 21, 22]
        );
    }

    #[test]
    fn canonical_lattice_normalization_rejects_impossible_row_layout() {
        let topology = RenderResultTopology::sample_lattice_2d(2, 2).expect("2x2 topology");
        let bytes = words_to_bytes(&[11, 12, 21]);
        assert_eq!(
            normalize_canonical_words(3, topology, &bytes),
            Err(
                RenderDeterministicVerificationObservationError::InvalidPhysicalLayout {
                    output_index: 3,
                    channel: "canonical-output",
                    byte_len: 12,
                }
            )
        );
    }
}
