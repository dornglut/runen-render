//! Private maintained deterministic carrier interpretation.
//!
//! The carrier is a physical realization detail of the maintained evaluator.  This module keeps
//! its native-byte and finite-evaluation interpretation in one renderer-owned implementation for
//! both private verification and the public captured-observation boundary.

pub(super) const WORD_BYTES: usize = core::mem::size_of::<u32>();

pub(super) fn decode_word(bytes: &[u8; WORD_BYTES]) -> u32 {
    u32::from_ne_bytes(*bytes)
}

pub(super) fn maintained_evaluation_value(word: u32) -> Option<f32> {
    let value = f32::from_bits(word);
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maintained_carrier_preserves_native_word_and_rejects_non_finite_values() {
        let value = 1.25_f32;
        assert_eq!(decode_word(&value.to_bits().to_ne_bytes()), value.to_bits());
        assert_eq!(maintained_evaluation_value(value.to_bits()), Some(value));
        assert_eq!(maintained_evaluation_value(f32::NAN.to_bits()), None);
        assert_eq!(maintained_evaluation_value(f32::INFINITY.to_bits()), None);
    }
}
