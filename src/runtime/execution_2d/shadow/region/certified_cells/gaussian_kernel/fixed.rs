//! Bounded arithmetic experiment owned by the existing Gaussian certificate.
//! No geometry, quadrature, image plane, compositor or production program owner.
//!
//! Signed Q48 represents n*2^-48 with |n|<2^63. The GPU implementation uses
//! only u32 limbs; the independent CPU oracle uses a single exact i128 product.
//! All reduction remains Q48. Conversion to f32 is permitted only for a completed
//! value: clamp its integer to [0,2^48], discard 24 fractional bits, then convert
//! the remaining integer <=2^24 exactly and scale by exact 2^-24. Its one-sided
//! conversion error is [0,2^-24), not an assumed WGSL rounding mode.

use super::{AreaQueryError, Arithmetic, DEGREE, PATCHES};
#[cfg(test)]
use super::{GaussianKernel, HALF_WIDTH, Interval};

pub(in super::super) mod batch;
#[cfg(test)]
mod gpu;
mod precomputed;

const SCALE: f64 = 281_474_976_710_656.0;
const EPSILON: f64 = 1.0 / SCALE;
const ONE: i64 = 1_i64 << 48;
#[cfg(test)]
const HALF: i64 = ONE / 2;
const MAX_CASES: usize = 4096;
#[cfg(test)]
const INPUT_BYTES: usize = 128;
const OUTPUT_BYTES: usize = 16;

#[derive(Clone, Copy)]
struct FixedPolynomial {
    coefficients: [i64; DEGREE + 1],
    error: f64,
}
#[cfg(test)]
const ZERO_POLYNOMIAL: FixedPolynomial = FixedPolynomial {
    coefficients: [0; DEGREE + 1],
    error: 0.0,
};

pub(in super::super) struct FixedKernel {
    polynomials: [[FixedPolynomial; 2]; PATCHES],
}

#[test]
#[ignore = "explicit regeneration of derived canonical fixed-point data"]
fn emit_precomputed_fixed_kernel() {
    let mut fuel = super::MAX_WORK;
    let kernel = GaussianKernel::new(&mut fuel, super::MAX_BYTES).unwrap();
    let fixed = FixedKernel::new(&kernel, &mut fuel, super::MAX_BYTES).unwrap();
    println!(
        "BEGIN_FIXED_DATA\n//! Derived normalized Q48 coefficients; verified against the interval proof.\nuse super::FixedPolynomial;\npub(super) const DATA: [[FixedPolynomial; 2]; 16] = ["
    );
    for patch in fixed.polynomials {
        println!("[");
        for polynomial in patch {
            println!(
                "FixedPolynomial {{ coefficients: {:?}, error: f64::from_bits({}) }},",
                polynomial.coefficients,
                polynomial.error.to_bits()
            );
        }
        println!("],");
    }
    println!("];\nEND_FIXED_DATA");
}

#[cfg(test)]
fn quantize(value: f64, arithmetic: &mut Arithmetic<'_>) -> Result<i64, AreaQueryError> {
    // Guard, exact binary scaling, nearest integer, exact guarded conversion,
    // indexing/bookkeeping. Rounded result has magnitude <=2^52, so both the
    // binary64 integer and i64 conversion are exact. This is CPU arithmetic.
    arithmetic.charge(96)?;
    if !value.is_finite() || value.abs() > 16.0 {
        return Err(AreaQueryError::Precision);
    }
    Ok((value * SCALE).round_ties_even() as i64)
}

impl FixedKernel {
    pub(in super::super) fn compiled(
        fuel: &mut usize,
        max_bytes: usize,
    ) -> Result<Self, AreaQueryError> {
        let bytes = std::mem::size_of::<Self>();
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(1024 + 2 * bytes)?;
        // Include both immutable canonical data and its admitted local copy.
        if 2 * bytes > max_bytes {
            return Err(AreaQueryError::Resource);
        }
        Ok(Self {
            polynomials: precomputed::DATA,
        })
    }
    #[cfg(test)]
    pub(in super::super) fn new(
        kernel: &GaussianKernel,
        fuel: &mut usize,
        max_bytes: usize,
    ) -> Result<Self, AreaQueryError> {
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(1024)?;
        if std::mem::size_of::<Self>() > max_bytes {
            return Err(AreaQueryError::Resource);
        }
        if kernel.normalization.lo <= 1.0 {
            return Err(AreaQueryError::Precision);
        }
        let mut result = Self {
            polynomials: [[ZERO_POLYNOMIAL; 2]; PATCHES],
        };
        let h = arithmetic.scalar(HALF_WIDTH)?;
        for (index, patch) in kernel.patches.iter().enumerate() {
            for (kind, original) in [&patch.cdf, &patch.density].into_iter().enumerate() {
                arithmetic.charge(64)?;
                let mut coefficient_error = arithmetic.scalar(0.0)?;
                for i in (0..=DEGREE).rev() {
                    arithmetic.charge(32)?;
                    let scalar = arithmetic.scalar(original.coefficients[i])?;
                    let normalized = arithmetic.div(scalar, kernel.normalization)?;
                    let midpoint = normalized.lo * 0.5 + normalized.hi * 0.5;
                    let fixed = quantize(midpoint, &mut arithmetic)?;
                    result.polynomials[index][kind].coefficients[i] = fixed;
                    let fixed_scalar = arithmetic.scalar(fixed as f64 / SCALE)?;
                    let difference = arithmetic.sub(normalized, fixed_scalar)?;
                    arithmetic.charge(16)?;
                    let radius = arithmetic.scalar(difference.lo.abs().max(difference.hi.abs()))?;
                    let product = arithmetic.mul(coefficient_error, h)?;
                    coefficient_error = arithmetic.add(radius, product)?;
                }
                let error = arithmetic.scalar(original.error)?;
                let source = arithmetic.div(error, kernel.normalization)?;
                let error = arithmetic.add(source, coefficient_error)?;
                // Independent fixed-point arithmetic proof. For |delta|<=h and
                // coefficients <=16, every true Horner suffix is <16/(1-h)<18.
                // Quantized delta differs by <=epsilon. At each stage:
                // E_i <= (h+epsilon) E_(i+1) +18epsilon+epsilon+epsilon.
                // The last two terms cover coefficient quantization (also
                // retained above, conservatively) and truncation of one exact
                // product. Twelve stages and initial coefficient error give
                // E <= epsilon +20epsilon/(1-h-epsilon)<24epsilon.
                // Thus all computed intermediates are still strictly below18.
                // Original.error retains source Taylor/delta/series authority;
                // normalization and quantized coefficient errors are outward.
                let fixed_error = arithmetic.scalar(24.0 * EPSILON)?;
                result.polynomials[index][kind].error = arithmetic.add(error, fixed_error)?.hi;
            }
        }
        Ok(result)
    }

    #[cfg(test)]
    fn point(
        &self,
        z: f64,
        density: bool,
        fuel: &mut usize,
    ) -> Result<(Case, f64), AreaQueryError> {
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(128)?;
        if !z.is_finite() {
            return Err(AreaQueryError::Precision);
        }
        if z.abs() >= 3.0 || (!density && z == 0.0) {
            let value = if density || z <= -3.0 {
                0
            } else if z >= 3.0 {
                ONE
            } else {
                HALF
            };
            let mut coefficients = [0; DEGREE + 1];
            coefficients[0] = value;
            return Ok((
                Case {
                    opcode: 1,
                    a: 0,
                    b: 0,
                    coefficients,
                },
                0.0,
            ));
        }
        let positive = z.abs();
        let mut index = 0;
        let mut end = PATCHES;
        while index < end {
            // At most five comparisons of exact dyadic endpoints. Rounded
            // z/PATCH_WIDTH followed by an integer cast is not seam authority.
            arithmetic.charge(32)?;
            let middle = index + (end - index) / 2;
            if (middle + 1) as f64 * (2.0 * HALF_WIDTH) <= positive {
                index = middle + 1;
            } else {
                end = middle;
            }
        }
        arithmetic.charge(64)?;
        let center = (2 * index + 1) as f64 * HALF_WIDTH;
        let delta = positive - center;
        if delta.abs() > HALF_WIDTH {
            return Err(AreaQueryError::Precision);
        }
        let polynomial = self.polynomials[index][usize::from(density)];
        let a = quantize(delta, &mut arithmetic)?;
        let opcode = if density {
            1
        } else if z >= 0.0 {
            2
        } else {
            3
        };
        Ok((
            Case {
                opcode,
                a,
                b: 0,
                coefficients: polynomial.coefficients,
            },
            polynomial.error,
        ))
    }
}

#[test]
fn compiled_fixed_kernel_is_bit_identical_to_complete_interval_construction() {
    let mut fuel = super::MAX_WORK;
    let kernel = GaussianKernel::new(&mut fuel, super::MAX_BYTES).unwrap();
    let proof = FixedKernel::new(&kernel, &mut fuel, super::MAX_BYTES).unwrap();
    let compiled = FixedKernel::compiled(&mut fuel, super::MAX_BYTES).unwrap();
    for (a, b) in proof
        .polynomials
        .iter()
        .flatten()
        .zip(compiled.polynomials.iter().flatten())
    {
        assert_eq!(a.coefficients, b.coefficients);
        assert_eq!(a.error.to_bits(), b.error.to_bits());
    }
    assert!(matches!(
        FixedKernel::compiled(&mut 0, super::MAX_BYTES),
        Err(AreaQueryError::Resource)
    ));
    assert!(matches!(
        FixedKernel::compiled(&mut fuel, 0),
        Err(AreaQueryError::Resource)
    ));
}

#[derive(Clone, Copy)]
#[cfg(test)]
struct Case {
    // 0: exact wide multiply/truncate, 1: Horner, 2/3: positive/negative CDF,
    // 4: integer reduction of base plus thirteen signed Q48 terms.
    opcode: u32,
    a: i64,
    b: i64,
    coefficients: [i64; DEGREE + 1],
}
#[cfg(test)]
fn wide_product(a: i64, b: i64) -> i64 {
    // Signed division truncates toward zero, matching GPU magnitude-product
    // truncation followed by sign application. No Rust limb implementation is
    // reused by the GPU comparison oracle.
    i64::try_from((i128::from(a) * i128::from(b)) / i128::from(ONE)).unwrap()
}
#[cfg(test)]
impl Case {
    fn validate(&self, fuel: &mut usize) -> Result<(), AreaQueryError> {
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(256)?;
        match self.opcode {
            0 if self.a.unsigned_abs() <= 18 * ONE as u64
                && self.b.unsigned_abs() <= 18 * ONE as u64 =>
            {
                Ok(())
            }
            1..=3 => {
                if self.a.unsigned_abs() > (3 * ONE / 32) as u64 {
                    return Err(AreaQueryError::Precision);
                }
                for &c in &self.coefficients {
                    arithmetic.charge(16)?;
                    if c.unsigned_abs() > 16 * ONE as u64 {
                        return Err(AreaQueryError::Precision);
                    }
                }
                Ok(())
            }
            4 => {
                let mut absolute = u128::from(self.a.unsigned_abs());
                for &c in &self.coefficients {
                    arithmetic.charge(32)?;
                    absolute += u128::from(c.unsigned_abs());
                }
                if absolute >= 1_u128 << 63 {
                    return Err(AreaQueryError::Precision);
                }
                Ok(())
            }
            _ => Err(AreaQueryError::Precision),
        }
    }
    fn oracle(&self) -> i64 {
        match self.opcode {
            0 => wide_product(self.a, self.b),
            4 => self
                .coefficients
                .iter()
                .fold(self.a, |sum, &c| sum.checked_add(c).unwrap()),
            _ => {
                let mut p = self.coefficients[DEGREE];
                for &c in self.coefficients[..DEGREE].iter().rev() {
                    p = wide_product(p, self.a).checked_add(c).unwrap();
                }
                match self.opcode {
                    2 => HALF + p,
                    3 => HALF - p,
                    _ => p,
                }
            }
        }
    }
    fn words(self) -> [u32; 32] {
        let mut result = [0; 32];
        result[0] = self.opcode;
        for (i, value) in [self.a, self.b]
            .into_iter()
            .chain(self.coefficients)
            .enumerate()
        {
            result[2 + 2 * i] = value as u32;
            result[3 + 2 * i] = (value as u64 >> 32) as u32;
        }
        result
    }
}
#[cfg(test)]
fn q24_bits(value: i64) -> u32 {
    let value = value.clamp(0, ONE) as u64;
    let integer = (value >> 24) as u32;
    (integer as f32 * (1.0_f32 / 16_777_216.0)).to_bits()
}

#[cfg(test)]
struct Payload {
    bytes: Vec<u8>,
    count: u32,
    output_bytes: usize,
    conservative_data_bytes: usize,
}
#[cfg(test)]
impl Payload {
    fn new(cases: &[Case], fuel: &mut usize, max_bytes: usize) -> Result<Self, AreaQueryError> {
        let mut arithmetic = Arithmetic { fuel };
        arithmetic.charge(256)?;
        if cases.is_empty() || cases.len() > MAX_CASES {
            return Err(AreaQueryError::Resource);
        }
        let input = cases
            .len()
            .checked_mul(INPUT_BYTES)
            .ok_or(AreaQueryError::Resource)?;
        let output = cases
            .len()
            .checked_mul(OUTPUT_BYTES)
            .ok_or(AreaQueryError::Resource)?;
        // Data-only inventory: source Vec, Prepared Arc, mapped upload buffer,
        // and device input (four copies, including b11c5c6's upload lifetime),
        // device/staging/readback/decoded output (four copies). No allocation
        // aliases or early source releases are assumed. Scope/graph/compiler
        // metadata and the caller's borrowed case list remain separate.
        let required = input
            .checked_mul(4)
            .and_then(|n| n.checked_add(output * 4))
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(AreaQueryError::Resource)?;
        if required > max_bytes {
            return Err(AreaQueryError::Resource);
        }
        for case in cases {
            case.validate(arithmetic.fuel)?;
        }
        // 64 units per packed u32: scalar extraction/shift, fixed indexing,
        // little-endian byte encoding/copy and both bounded loop controls.
        // This is conservative CPU data preparation, separate from GPU work.
        arithmetic.charge(64 + input * 16)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(input)
            .map_err(|_| AreaQueryError::Resource)?;
        if bytes.capacity() != input {
            return Err(AreaQueryError::Resource);
        }
        for &case in cases {
            for word in case.words() {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
        }
        Ok(Self {
            bytes,
            count: cases.len() as u32,
            output_bytes: output,
            conservative_data_bytes: required,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_normalized_kernel_encloses_independent_decimal_truth() {
        let mut fuel = super::super::MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, super::super::MAX_BYTES).unwrap();
        let before = fuel;
        let fixed =
            FixedKernel::new(&kernel, &mut fuel, std::mem::size_of::<FixedKernel>()).unwrap();
        let construction = before - fuel;
        let mut maximum_error = 0.0_f64;
        let mut max_point_work = 0;
        for i in 0..=32 {
            let seam = i as f64 * HALF_WIDTH;
            for z in [
                seam,
                seam.next_down(),
                seam.next_up(),
                -seam,
                (-seam).next_down(),
                (-seam).next_up(),
            ] {
                for density in [false, true] {
                    let before = fuel;
                    let (case, error) = fixed.point(z, density, &mut fuel).unwrap();
                    max_point_work = max_point_work.max(before - fuel);
                    case.validate(&mut fuel).unwrap();
                    let raw = case.oracle();
                    let value = raw.clamp(0, ONE) as f64 / SCALE;
                    let reference = if density {
                        kernel.density(Interval::scalar(z).unwrap(), &mut fuel)
                    } else {
                        kernel.cdf(Interval::scalar(z).unwrap(), &mut fuel)
                    }
                    .unwrap();
                    assert!(
                        value - error <= reference.hi && reference.lo <= value + error,
                        "z={z} density={density} value={value} error={error} reference={reference:?}"
                    );
                    // Stronger independent check, separate from the interval
                    // overlap above: reuse the owning kernel's single 110-digit
                    // direct-series oracle at exact seams/centers. Neighbouring
                    // binary64 probes use global CDF Lipschitz one / interior
                    // density Lipschitz three to enclose their displacement.
                    let cutoff = density && z.abs() >= 3.0;
                    let truth = if cutoff {
                        0.0
                    } else if density {
                        super::super::tests::DENSITY_REFERENCE[i]
                    } else {
                        let positive = super::super::tests::REFERENCE[i];
                        if z < 0.0 { 1.0 - positive } else { positive }
                    };
                    if error == 0.0 {
                        assert_eq!(value, truth);
                    } else {
                        let movement = if cutoff {
                            0.0
                        } else {
                            (z.abs() - seam).abs() * if density { 3.0 } else { 1.0 }
                        };
                        let radius = (error + movement).next_up();
                        let enclosure = Interval::scalar(value)
                            .unwrap()
                            .add(Interval::new(-radius, radius).unwrap())
                            .unwrap();
                        assert!(
                            enclosure.lo <= truth.next_down() && truth.next_up() <= enclosure.hi,
                            "independent Decimal truth z={z} density={density}: {enclosure:?} excludes {truth}"
                        );
                    }
                    maximum_error = maximum_error.max(error);
                    let narrowed = f64::from(f32::from_bits(q24_bits(raw)));
                    assert!(narrowed <= value && value - narrowed < 2.0_f64.powi(-24));
                }
            }
        }
        assert!(maximum_error < 1e-10);
        eprintln!(
            "Fixed Gaussian: compile_work={construction}, max_prepare_point_work={max_point_work}, bytes={}, max_normalized_error={maximum_error:e}; no geometry/coverage admission",
            std::mem::size_of::<FixedKernel>()
        );
    }
    #[test]
    fn fixed_component_admission_rejects_precision_fuel_and_capacity() {
        let mut fuel = super::super::MAX_WORK;
        let kernel = GaussianKernel::new(&mut fuel, super::super::MAX_BYTES).unwrap();
        assert!(matches!(
            FixedKernel::new(&kernel, &mut fuel, 0),
            Err(AreaQueryError::Resource)
        ));
        assert!(matches!(
            FixedKernel::new(&kernel, &mut 1023, usize::MAX),
            Err(AreaQueryError::Resource)
        ));
        let fixed = FixedKernel::new(&kernel, &mut fuel, usize::MAX).unwrap();
        assert!(matches!(
            fixed.point(f64::NAN, false, &mut fuel),
            Err(AreaQueryError::Precision)
        ));
        assert!(matches!(
            quantize(16.0_f64.next_up(), &mut Arithmetic { fuel: &mut fuel }),
            Err(AreaQueryError::Precision)
        ));
        for z in [
            f64::from_bits(1),
            -f64::from_bits(1),
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
        ] {
            let (case, error) = fixed.point(z, false, &mut fuel).unwrap();
            assert!((case.oracle() as f64 / SCALE - 0.5).abs() <= error);
        }
        let mut bad = Case {
            opcode: 1,
            a: 0,
            b: 0,
            coefficients: [0; 13],
        };
        bad.coefficients[0] = 16 * ONE + 1;
        assert!(matches!(
            bad.validate(&mut fuel),
            Err(AreaQueryError::Precision)
        ));
        bad.coefficients[0] = 0;
        bad.a = 3 * ONE / 32 + 1;
        assert!(matches!(
            bad.validate(&mut fuel),
            Err(AreaQueryError::Precision)
        ));
        bad.opcode = 4;
        bad.a = i64::MAX;
        bad.coefficients[0] = 1;
        assert!(matches!(
            bad.validate(&mut fuel),
            Err(AreaQueryError::Precision)
        ));
        let (valid, _) = fixed.point(0.5, false, &mut fuel).unwrap();
        assert!(matches!(
            Payload::new(&[valid], &mut fuel, 1),
            Err(AreaQueryError::Resource)
        ));
        assert!(matches!(
            Payload::new(&[valid], &mut 1, usize::MAX),
            Err(AreaQueryError::Resource)
        ));
        let payload = Payload::new(&[valid], &mut fuel, usize::MAX).unwrap();
        assert!(matches!(
            Payload::new(&[], &mut fuel, usize::MAX),
            Err(AreaQueryError::Resource)
        ));
        assert!(matches!(
            Payload::new(&vec![valid; MAX_CASES + 1], &mut fuel, usize::MAX),
            Err(AreaQueryError::Resource)
        ));
        assert_eq!(payload.bytes.len(), 128);
        assert_eq!(payload.output_bytes, 16);
    }
}
