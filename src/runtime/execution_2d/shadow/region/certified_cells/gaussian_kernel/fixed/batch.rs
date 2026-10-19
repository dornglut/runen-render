//! Test-only complete signed reductions over a shared certified kernel table.
//! No geometry/cutoff partition, quadrature error or source membership lives here.
use super::*;

const TABLE_WORDS: usize = PATCHES * 2 * (DEGREE + 1) * 2;
const HEADER_WORDS: usize = 4;
const QUERY_WORDS: usize = 5;
const NODE_WORDS: usize = 6;
const MAX_NODES: usize = 65_536;

pub(in super::super::super) struct Node {
    pub(in super::super::super) z: [f64; 2],
    pub(in super::super::super) z_error: [f64; 2],
    pub(in super::super::super) weight: f64,
    pub(in super::super::super) weight_error: f64,
}
pub(in super::super::super) struct Query {
    pub(in super::super::super) base: f64,
    pub(in super::super::super) base_error: f64,
    pub(in super::super::super) run: usize,
    pub(in super::super::super) negate: bool,
}

struct RunCertificate {
    absolute: u128,
    error: f64,
}
pub(in super::super::super) struct Batch {
    payload: Payload,
    pub(in super::super::super) errors: Vec<f64>,
    nodes: usize,
    gpu_node_visits: usize,
}

fn pack_scalar(value: f64, fuel: &mut usize) -> Result<i64, AreaQueryError> {
    // Guard, exact power-of-two scaling, nearest integer, exact i64 cast,
    // result/call bookkeeping. |value|<=16 => scaled integer<=2^52.
    // Thus no uncertain floating conversion or platform tie behavior enters
    // the GPU path. Quantization error is at most 2^-49 including subnormals.
    let mut arithmetic = Arithmetic { fuel };
    arithmetic.charge(48)?;
    if !value.is_finite() || value.abs() > 16.0 {
        return Err(AreaQueryError::Precision);
    }
    Ok((value * SCALE).round_ties_even() as i64)
}

fn push_word(bytes: &mut Vec<u8>, value: u32, fuel: &mut usize) -> Result<(), AreaQueryError> {
    // Capacity is admitted once; fixed four-byte store and loop bookkeeping.
    Arithmetic { fuel }.charge(16)?;
    if bytes.capacity() - bytes.len() < 4 {
        return Err(AreaQueryError::Resource);
    }
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}
fn push_pair(bytes: &mut Vec<u8>, value: i64, fuel: &mut usize) -> Result<(), AreaQueryError> {
    push_word(bytes, value as u32, fuel)?;
    push_word(bytes, (value as u64 >> 32) as u32, fuel)
}
fn up_add(a: f64, b: f64) -> f64 {
    (a + b).next_up()
}
fn up_mul(a: f64, b: f64) -> f64 {
    (a * b).next_up()
}

impl Batch {
    pub(in super::super::super) fn field_preparation_work(
        &self,
        samples: usize,
    ) -> Result<usize, AreaQueryError> {
        // Actual one-time transfer copies: prepared query words and the POD
        // u32 sample map. No intermediate expanded float plane is formed.
        // Also reserve bounded descriptors/handles, two six-vertex F3E draws
        // and the four-word extent uniform. Compiler/backend work has its
        // existing separate sibling authority.
        self.payload
            .bytes
            .len()
            .checked_add(samples.checked_mul(4).ok_or(AreaQueryError::Resource)?)
            // A literal-only field retains one initialized 16-byte binding,
            // rather than inventing a numerical query or compute operation.
            .and_then(|v| v.checked_add(if self.payload.count == 0 { 16 } else { 0 }))
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| v.checked_add(4096 + 1024 + 768 + 32))
            .ok_or(AreaQueryError::Resource)
    }

    pub(in super::super::super) fn field_data(&self) -> (&[u8], u32) {
        (&self.payload.bytes, self.payload.count)
    }

    pub(in super::super::super) fn new(
        fixed: &FixedKernel,
        nodes: &[Node],
        runs: &[std::ops::Range<usize>],
        queries: &[Query],
        fuel: &mut usize,
        max_bytes: usize,
    ) -> Result<Self, AreaQueryError> {
        Arithmetic { fuel }.charge(512)?;
        if queries.is_empty() {
            if !nodes.is_empty() || !runs.is_empty() {
                return Err(AreaQueryError::Precision);
            }
            let required = size_of::<Self>() + 4 * OUTPUT_BYTES;
            if required > max_bytes {
                return Err(AreaQueryError::Resource);
            }
            return Ok(Self {
                payload: Payload {
                    bytes: Vec::new(),
                    count: 0,
                    output_bytes: 0,
                    conservative_data_bytes: required,
                },
                errors: Vec::new(),
                nodes: 0,
                gpu_node_visits: 0,
            });
        }
        if queries.len() > MAX_CASES
            || runs.is_empty()
            || runs.len() > MAX_CASES
            || nodes.len() > MAX_NODES
        {
            return Err(AreaQueryError::Resource);
        }
        // Runs partition one immutable node table, including empty runs.
        // Query references may reuse a run arbitrarily; neither node payload
        // nor its numerical certificate is rebuilt per physical sample.
        let mut cursor = 0;
        for run in runs {
            Arithmetic { fuel }.charge(64)?;
            if run.start != cursor || run.end < run.start || run.end > nodes.len() {
                return Err(AreaQueryError::Precision);
            }
            cursor = run.end;
        }
        if cursor != nodes.len() {
            return Err(AreaQueryError::Precision);
        }
        let mut gpu_node_visits = 0_usize;
        for query in queries {
            Arithmetic { fuel }.charge(80)?;
            if query.run >= runs.len() || !query.base_error.is_finite() || query.base_error < 0.0 {
                return Err(AreaQueryError::Precision);
            }
            // Reusing a node run saves preparation/storage, but the shader
            // still evaluates it for each referencing query. Inventory the
            // expanded GPU loop work, not only unique uploaded nodes.
            gpu_node_visits = gpu_node_visits
                .checked_add(runs[query.run].len())
                .ok_or(AreaQueryError::Resource)?;
        }
        let input = 4
            * (HEADER_WORDS + TABLE_WORDS + QUERY_WORDS * queries.len() + NODE_WORDS * nodes.len());
        let output = queries.len() * OUTPUT_BYTES;
        // Four input and four encoded-output data copies, plus optional float
        // proof decoding, query radii and the temporary once-per-run records.
        // Borrowed nodes/ranges/queries and graph/compiler metadata are extra.
        let required = 4 * input
            + 4 * output
            + queries.len() * (size_of::<f64>() + size_of::<f32>())
            + runs.len() * size_of::<RunCertificate>()
            + size_of::<Self>()
            + size_of::<Vec<RunCertificate>>();
        if required > max_bytes {
            return Err(AreaQueryError::Resource);
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(input)
            .map_err(|_| AreaQueryError::Resource)?;
        let mut errors = Vec::new();
        errors
            .try_reserve_exact(queries.len())
            .map_err(|_| AreaQueryError::Resource)?;
        let mut certificates = Vec::new();
        certificates
            .try_reserve_exact(runs.len())
            .map_err(|_| AreaQueryError::Resource)?;
        if bytes.capacity() != input
            || errors.capacity() != queries.len()
            || certificates.capacity() != runs.len()
        {
            return Err(AreaQueryError::Resource);
        }
        for word in [queries.len() as u32, nodes.len() as u32, 0, 0] {
            push_word(&mut bytes, word, fuel)?;
        }
        let mut kernel_error = 0.0_f64;
        for patch in fixed.polynomials {
            for polynomial in patch {
                Arithmetic { fuel }.charge(16)?;
                kernel_error = kernel_error.max(polynomial.error);
                for coefficient in polynomial.coefficients {
                    push_pair(&mut bytes, coefficient, fuel)?;
                }
            }
        }
        if !kernel_error.is_finite() || kernel_error > 1e-6 {
            return Err(AreaQueryError::Precision);
        }
        for query in queries {
            Arithmetic { fuel }.charge(64)?;
            let run = &runs[query.run];
            let base = pack_scalar(query.base, fuel)?;
            push_word(&mut bytes, run.start as u32, fuel)?;
            push_word(&mut bytes, run.len() as u32, fuel)?;
            push_pair(&mut bytes, base, fuel)?;
            push_word(&mut bytes, u32::from(query.negate), fuel)?;
        }
        for run in runs {
            Arithmetic { fuel }.charge(64)?;
            let mut error = 0.0;
            let mut absolute = 0_u128;
            Arithmetic { fuel }.charge(16 * run.len())?;
            for group in nodes[run.clone()]
                .chunk_by(|a, b| a.z_error == b.z_error && a.weight_error == b.weight_error)
            {
                Arithmetic { fuel }.charge(160)?;
                let uniform = &group[0];
                if !uniform
                    .z_error
                    .into_iter()
                    .all(|v| v.is_finite() && v >= 0.0)
                    || !uniform.weight_error.is_finite()
                    || uniform.weight_error < 0.0
                {
                    return Err(AreaQueryError::Precision);
                }
                let ex = up_add(kernel_error, up_add(uniform.z_error[0], EPSILON * 0.5));
                let ey = up_add(kernel_error, up_add(uniform.z_error[1], EPSILON * 0.5));
                let product_error = up_add(up_add(ex, ey), EPSILON);
                let per_node = up_add(up_add(uniform.weight_error, EPSILON * 0.5), EPSILON);
                let mut group_absolute = 0_u128;
                for node in group {
                    // Finite coordinates, cutoff guards, exact absolute sums
                    // and control. Quantization/stores prepay their own work.
                    Arithmetic { fuel }.charge(64)?;
                    if !node.z.into_iter().all(f64::is_finite) {
                        return Err(AreaQueryError::Precision);
                    }
                    let zy = node.z[1];
                    let outside = (zy.abs() - uniform.z_error[1]).next_down() >= 3.0;
                    let inside = up_add(zy.abs(), uniform.z_error[1]) < 3.0;
                    if !outside && !inside {
                        return Err(AreaQueryError::Precision);
                    }
                    let x = pack_scalar(node.z[0].clamp(-3.0, 3.0), fuel)?;
                    let y = pack_scalar(if outside { 3.0 } else { zy }, fuel)?;
                    if inside && y.unsigned_abs() >= (3 * ONE) as u64 {
                        return Err(AreaQueryError::Precision);
                    }
                    let weight = pack_scalar(node.weight, fuel)?;
                    let magnitude = u128::from(weight.unsigned_abs());
                    absolute += magnitude;
                    group_absolute += magnitude;
                    if absolute >= 1_u128 << 63 {
                        return Err(AreaQueryError::Precision);
                    }
                    push_pair(&mut bytes, x, fuel)?;
                    push_pair(&mut bytes, y, fuel)?;
                    push_pair(&mut bytes, weight, fuel)?;
                }
                Arithmetic { fuel }.charge(160)?;
                // Exact nonnegative integer sum; upward neighbor encloses the
                // f64 conversion beyond 2^53. Binary normalization is exact.
                let weight_sum = ((group_absolute as f64).next_up() / SCALE).next_up();
                let independent = up_mul(group.len() as f64, per_node);
                error = up_add(
                    error,
                    up_add(independent, up_mul(weight_sum, product_error)),
                );
            }
            Arithmetic { fuel }.charge(48)?;
            certificates.push(RunCertificate { absolute, error });
        }
        for query in queries {
            Arithmetic { fuel }.charge(96)?;
            let base = pack_scalar(query.base, fuel)?;
            let certificate = &certificates[query.run];
            if certificate.absolute + u128::from(base.unsigned_abs()) >= 1_u128 << 63 {
                return Err(AreaQueryError::Precision);
            }
            let error = up_add(up_add(query.base_error, EPSILON * 0.5), certificate.error);
            // Clamp is nonexpansive; one completed Q24 truncation, never one
            // per node. Negation preserves the same absolute certificate.
            let error = up_add(error, 1.0 / 16_777_216.0);
            if !error.is_finite() {
                return Err(AreaQueryError::Precision);
            }
            errors.push(error);
        }
        if bytes.len() != input {
            return Err(AreaQueryError::Precision);
        }
        Ok(Self {
            payload: Payload {
                bytes,
                count: queries.len() as u32,
                output_bytes: output,
                conservative_data_bytes: required,
            },
            errors,
            nodes: nodes.len(),
            gpu_node_visits,
        })
    }
    pub(in super::super::super) fn execute_proof(&self) -> Vec<f32> {
        if self.payload.count == 0 {
            return Vec::new();
        }
        println!(
            "GAUSSIAN_DISPATCH queries={} workgroups={} unique_nodes={} GPU_node_visits={} Q48_products_upper={} u32_limb_products_upper={}",
            self.errors.len(),
            self.errors.len().div_ceil(32),
            self.nodes,
            self.gpu_node_visits,
            self.gpu_node_visits as u64 * 26,
            self.gpu_node_visits as u64 * 26 * 16,
        );
        gpu::execute(&self.payload, source())
            .as_chunks::<16>()
            .0
            .iter()
            .map(|bytes| f32::from_bits(u32::from_le_bytes(bytes[8..12].try_into().unwrap())))
            .collect()
    }
}

fn source() -> runen_gpu::GpuAdmittedProgramSource {
    crate::runtime::program::retained_gaussian_coverage_source()
        .expect("qualified Gaussian program must compile through the maintained artifact bridge")
}

fn oracle_kernel(fixed: &FixedKernel, z: i64, density: bool) -> i64 {
    let magnitude = z.unsigned_abs();
    if magnitude >= (3 * ONE) as u64 {
        return if density || z < 0 { 0 } else { ONE };
    }
    let patch = (magnitude / ((3 * ONE / 16) as u64)) as usize;
    let center = (2 * patch + 1) as i64 * (3 * ONE / 32);
    let delta = magnitude as i64 - center;
    let coefficients = fixed.polynomials[patch][usize::from(density)].coefficients;
    let mut value = coefficients[DEGREE];
    for coefficient in coefficients[..DEGREE].iter().rev() {
        value = ((i128::from(value) * i128::from(delta)) / i128::from(ONE)) as i64 + coefficient;
    }
    if !density {
        value = HALF + if z < 0 { -value } else { value };
    }
    value.clamp(0, ONE)
}

#[test]
fn literal_only_gaussian_batch_has_no_query_payload_or_orphaned_runs() {
    let mut fuel = super::super::MAX_WORK;
    let fixed = FixedKernel::compiled(&mut fuel, usize::MAX).unwrap();
    let before = fuel;
    let batch = Batch::new(&fixed, &[], &[], &[], &mut fuel, usize::MAX).unwrap();
    assert_eq!(before - fuel, 512);
    assert_eq!(batch.field_data(), (&[][..], 0));
    assert!(batch.errors.is_empty());
    assert!(batch.execute_proof().is_empty());
    // No allocation or partial retention can precede the caller's host lease
    // or CPU work admission, including the otherwise allocation-free case.
    assert!(matches!(
        Batch::new(&fixed, &[], &[], &[], &mut fuel, 0),
        Err(AreaQueryError::Resource)
    ));
    let mut short = 511;
    assert!(matches!(
        Batch::new(&fixed, &[], &[], &[], &mut short, usize::MAX),
        Err(AreaQueryError::Resource)
    ));
    let orphan_run = 0..0;
    assert!(matches!(
        Batch::new(
            &fixed,
            &[],
            std::slice::from_ref(&orphan_run),
            &[],
            &mut fuel,
            usize::MAX
        ),
        Err(AreaQueryError::Precision)
    ));
    let orphan = Node {
        z: [0.0; 2],
        z_error: [0.0; 2],
        weight: 0.0,
        weight_error: 0.0,
    };
    assert!(matches!(
        Batch::new(&fixed, &[orphan], &[], &[], &mut fuel, usize::MAX),
        Err(AreaQueryError::Precision)
    ));
}

#[test]
fn shared_gaussian_batch_shader_and_admission_guards() {
    let mut fuel = super::super::MAX_WORK;
    let kernel = GaussianKernel::new(&mut fuel, super::super::MAX_BYTES).unwrap();
    let fixed = FixedKernel::new(&kernel, &mut fuel, usize::MAX).unwrap();
    let _ = source();
    let node = Node {
        z: [0.25, 0.5],
        z_error: [1e-12; 2],
        weight: 0.125,
        weight_error: 1e-12,
    };
    let run = 0..1;
    let query = Query {
        base: 0.25,
        base_error: 1e-12,
        run: 0,
        negate: false,
    };
    let batch = Batch::new(
        &fixed,
        std::slice::from_ref(&node),
        std::slice::from_ref(&run),
        std::slice::from_ref(&query),
        &mut fuel,
        4 * 1024 * 1024,
    )
    .unwrap();
    assert_eq!(batch.nodes, 1);
    assert!(batch.errors[0] < 6e-8);
    let unrelated = Query {
        base: 0.0,
        base_error: 0.0,
        run: 1,
        negate: false,
    };
    assert!(matches!(
        Batch::new(
            &fixed,
            std::slice::from_ref(&node),
            std::slice::from_ref(&run),
            &[unrelated],
            &mut fuel,
            usize::MAX
        ),
        Err(AreaQueryError::Precision)
    ));
    let incomplete = 1..1;
    assert!(matches!(
        Batch::new(
            &fixed,
            std::slice::from_ref(&node),
            std::slice::from_ref(&incomplete),
            std::slice::from_ref(&query),
            &mut fuel,
            usize::MAX
        ),
        Err(AreaQueryError::Precision)
    ));
    assert!(matches!(
        Batch::new(
            &fixed,
            std::slice::from_ref(&node),
            std::slice::from_ref(&run),
            &[query],
            &mut fuel,
            1
        ),
        Err(AreaQueryError::Resource)
    ));
    let cutoff = Node {
        z: [0.0, 3.0],
        z_error: [0.0; 2],
        weight: 1.0,
        weight_error: 0.0,
    };
    let crossing = Query {
        base: 0.0,
        base_error: 0.0,
        run: 0,
        negate: false,
    };
    assert!(matches!(
        Batch::new(
            &fixed,
            &[cutoff],
            std::slice::from_ref(&run),
            &[crossing],
            &mut fuel,
            usize::MAX
        ),
        Err(AreaQueryError::Precision)
    ));
}

#[test]
fn shared_gaussian_run_and_base_admit_exact_absolute_integer_sums() {
    let mut fuel = super::super::MAX_WORK;
    let kernel = GaussianKernel::new(&mut fuel, super::super::MAX_BYTES).unwrap();
    let fixed = FixedKernel::new(&kernel, &mut fuel, usize::MAX).unwrap();
    let nodes: Vec<_> = (0..2047)
        .map(|_| Node {
            z: [0.0; 2],
            z_error: [0.0; 2],
            weight: 16.0,
            weight_error: 0.0,
        })
        .collect();
    let run = 0..nodes.len();
    // Run alone is 32752 Q48 units. Base +16 reaches the forbidden 32768
    // absolute-sum limit, independent of any fortunate arithmetic cancellation.
    let rejected = Query {
        base: 16.0,
        base_error: 0.0,
        run: 0,
        negate: true,
    };
    assert!(matches!(
        Batch::new(
            &fixed,
            &nodes,
            std::slice::from_ref(&run),
            &[rejected],
            &mut fuel,
            usize::MAX
        ),
        Err(AreaQueryError::Precision)
    ));
}

#[test]
fn shared_gaussian_packing_matches_exact_scaling_at_ties_and_exhaustion() {
    let mut fuel = super::super::MAX_WORK;
    for integer in [
        -((1_i64 << 52) - 1),
        -65537,
        -2,
        -1,
        0,
        1,
        2,
        65537,
        (1_i64 << 52) - 1,
    ] {
        for half in [-0.5, 0.0, 0.5] {
            let value = (integer as f64 + half) / SCALE;
            // These scaled inputs are exact integers or exact half-integers.
            // Integer parity supplies an independent ties-to-even oracle.
            let expected = integer
                + if half == 0.0 || integer & 1 == 0 {
                    0
                } else if half > 0.0 {
                    1
                } else {
                    -1
                };
            assert_eq!(pack_scalar(value, &mut fuel).unwrap(), expected);
        }
    }
    for value in [
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
    ] {
        assert_eq!(pack_scalar(value, &mut fuel).unwrap(), 0);
    }
    assert!(matches!(
        pack_scalar(16.0_f64.next_up(), &mut fuel),
        Err(AreaQueryError::Precision)
    ));
    let mut short = 47;
    assert!(matches!(
        pack_scalar(0.5, &mut short),
        Err(AreaQueryError::Resource)
    ));
    assert_eq!(short, 0);
}

#[test]
#[ignore = "requires real Vulkan: complete shared Gaussian query reductions"]
fn shared_gaussian_batch_matches_independent_i128_queries_on_vulkan() {
    let mut fuel = super::super::MAX_WORK;
    let kernel = GaussianKernel::new(&mut fuel, super::super::MAX_BYTES).unwrap();
    let fixed = FixedKernel::new(&kernel, &mut fuel, usize::MAX).unwrap();
    let mut nodes = Vec::new();
    for index in 0..97 {
        nodes.push(Node {
            z: [(index as f64 - 48.0) / 16.0, (index as f64 - 48.0) / 17.0],
            z_error: [1e-13; 2],
            weight: if index % 2 == 0 { 0.03125 } else { -0.03125 },
            weight_error: 1e-14,
        });
    }
    nodes.push(Node {
        z: [9.0, -3.5],
        z_error: [0.0; 2],
        weight: 1.0,
        weight_error: 0.0,
    });
    let queries = [
        Query {
            base: 0.5,
            base_error: 1e-14,
            run: 0,
            negate: false,
        },
        Query {
            base: 0.25,
            base_error: 0.0,
            run: 0,
            negate: true,
        },
        Query {
            base: 0.125,
            base_error: 0.0,
            run: 1,
            negate: false,
        },
    ];
    let before = fuel;
    let runs = [0..nodes.len(), nodes.len()..nodes.len()];
    let batch = Batch::new(&fixed, &nodes, &runs, &queries, &mut fuel, 4 * 1024 * 1024).unwrap();
    let work = before - fuel;
    let bytes = gpu::execute(&batch.payload, source());
    for (query, result) in queries.iter().zip(bytes.as_chunks::<16>().0) {
        let mut sum = 0_i64;
        for node in &nodes[runs[query.run].clone()] {
            let x = pack_scalar(node.z[0].clamp(-3.0, 3.0), &mut fuel).unwrap();
            let y = pack_scalar(node.z[1], &mut fuel).unwrap();
            let f = oracle_kernel(&fixed, x, false);
            let g = oracle_kernel(&fixed, y, true);
            let w = pack_scalar(node.weight, &mut fuel).unwrap();
            let product = i128::from(f) * i128::from(g) / i128::from(ONE);
            sum += (product * i128::from(w) / i128::from(ONE)) as i64;
        }
        let expected =
            pack_scalar(query.base, &mut fuel).unwrap() + if query.negate { -sum } else { sum };
        let actual = i64::from_le_bytes(result[..8].try_into().unwrap());
        assert_eq!(actual, expected);
        assert_eq!(
            u32::from_le_bytes(result[8..12].try_into().unwrap()),
            q24_bits(expected)
        );
    }
    let evaluated_nodes: usize = queries.iter().map(|query| runs[query.run].len()).sum();
    println!(
        "ACTUAL VULKAN shared Gaussian queries={} nodes={} preparation_work={work} conservative_data_bytes={} maximum_error={} all_Q48_and_completed_Q24_bits_exact; GPU_upper_Q48_products={} u32_limb_products={}; no geometry/quadrature/scene claim",
        queries.len(),
        batch.nodes,
        batch.payload.conservative_data_bytes,
        batch.errors.iter().copied().fold(0.0, f64::max),
        26 * evaluated_nodes,
        416 * evaluated_nodes
    );
}
