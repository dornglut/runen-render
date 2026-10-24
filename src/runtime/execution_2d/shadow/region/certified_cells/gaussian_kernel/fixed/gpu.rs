//! Actual RunenShader -> public RunenGPU arithmetic conformance.
#![cfg(test)]
use super::*;
use runen_gpu::*;
use runen_shader::*;
use std::time::{Duration, Instant};

static WGSL: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    format!(
        "{}\n{}",
        crate::runtime::program::GAUSSIAN_ARITHMETIC_WGSL,
        include_str!("arithmetic_proof.wgsl")
    )
});

fn source() -> GpuAdmittedProgramSource {
    admit_source(
        "runen-render fixed Gaussian arithmetic investigation",
        &WGSL,
    )
}

pub(super) fn admit_source(key: &str, wgsl: &str) -> GpuAdmittedProgramSource {
    let input = ShaderCompilationInput::exact_wgsl(
        ShaderPackageIdentity::try_from_raw(8191).unwrap(),
        ShaderModuleIdentity::try_from_raw(1).unwrap(),
        ShaderSourceSnapshot::new(
            ShaderSourceUnitIdentity::try_from_raw(1).unwrap(),
            ShaderSourceRevision::try_from_raw(1).unwrap(),
            wgsl,
        ),
    );
    let invocation =
        ShaderCompilationInvocation::new(input, ShaderCompilerRealization::Naga3001ExactWgslGateV1);
    let artifact = match ShaderCompiler::new().compile(&invocation).unwrap() {
        ShaderCompilationOutcome::Accepted(artifact) => artifact,
        other => panic!("fixed Gaussian arithmetic compiler rejected: {other:?}"),
    };
    assert_eq!(artifact.canonical_wgsl(), wgsl);
    let mut registry = GpuProgramSourceRegistry::new(1, wgsl.len()).unwrap();
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new(key).unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    registry
        .admit_wgsl(
            identity,
            artifact.canonical_wgsl(),
            GpuProgramSourceProvenance::new(
                "RunenRender test-only RunenShader artifact bridge",
                None,
            )
            .unwrap(),
        )
        .unwrap()
}

fn cases(fixed: &FixedKernel, fuel: &mut usize) -> Vec<Case> {
    let mut cases = Vec::new();
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
                cases.push(fixed.point(z, density, fuel).unwrap().0);
            }
        }
    }
    for a in [
        0,
        1,
        -1,
        65535,
        -65535,
        ONE - 1,
        ONE + 1,
        16 * ONE - 1,
        -16 * ONE + 1,
    ] {
        for b in [0, 1, -1, 65535, 65536, ONE - 1, -ONE - 1, 16 * ONE - 1] {
            cases.push(Case {
                opcode: 0,
                a,
                b,
                coefficients: [0; 13],
            });
        }
    }
    for z in [
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
    ] {
        for density in [false, true] {
            cases.push(fixed.point(z, density, fuel).unwrap().0);
        }
    }
    for value in [
        -ONE,
        -1,
        0,
        1,
        (1 << 24) - 1,
        1 << 24,
        ONE - 1,
        ONE,
        ONE + 1,
    ] {
        let mut coefficients = [0; 13];
        coefficients[0] = value;
        cases.push(Case {
            opcode: 4,
            a: 0,
            b: 0,
            coefficients,
        });
    }
    let mut coefficients = [0; 13];
    for (i, c) in coefficients.iter_mut().enumerate() {
        *c = if i % 2 == 0 { ONE / 3 } else { -ONE / 3 };
    }
    cases.push(Case {
        opcode: 4,
        a: ONE / 7,
        b: 0,
        coefficients,
    });
    cases
}

pub(super) fn execute(bytes: &[u8], count: u32, program: GpuAdmittedProgramSource) -> Vec<u8> {
    let mut requirements = GpuCapabilityRequirements::new();
    for feature in [GpuCapabilityFeature::Compute, GpuCapabilityFeature::Copy] {
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
    }
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("RunenRender private Q48 Gaussian arithmetic");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("real Vulkan software adapter required");
    let mut resources = GpuResourceScope::new();
    let (output, operations) = prepare(bytes, count, program, &context, &mut resources);
    let readback =
        GpuReadbackOperation::ordinary(GpuBufferRegion::whole(&output).unwrap().into()).unwrap();
    let readback_id = readback.id();
    let fragment = GpuWorkFragment::build("fixed arithmetic exact conformance", |work| {
        for operation in operations {
            work.operation("admitted Q48 preparation", operation)?;
        }
        work.operation("read arithmetic proof output", readback)?;
        Ok(())
    })
    .unwrap();
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("fixed Gaussian arithmetic proof").unwrap(),
        [fragment],
    )
    .unwrap();
    let submission = context
        .submit_prepared(pollster::block_on(context.prepare_submission(graph)).unwrap())
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(45);
    loop {
        context.progress();
        match submission.readback(readback_id).unwrap().status() {
            GpuReadbackStatus::Ready(bytes)
                if submission.status() == GpuSubmissionStatus::Completed =>
            {
                break bytes.as_bytes().to_vec();
            }
            GpuReadbackStatus::Failed(error) => panic!("fixed arithmetic readback: {error:?}"),
            _ => {
                assert!(Instant::now() < deadline, "fixed arithmetic Vulkan timeout");
                std::thread::yield_now();
            }
        }
    }
}

#[test]
fn fixed_gaussian_shader_compiles_through_existing_artifact_bridge() {
    let pipeline = GpuComputePipelineDescriptor::ordinary(source(), "main").unwrap();
    assert_eq!(pipeline.entry_point().as_str(), "main");
}

#[test]
#[ignore = "requires real Vulkan: explicit bounded fixed-Gaussian arithmetic conformance"]
fn fixed_gaussian_q48_matches_exact_integer_oracle_on_vulkan() {
    let mut fuel = super::super::MAX_WORK;
    let kernel = GaussianKernel::new(&mut fuel, super::super::MAX_BYTES).unwrap();
    let fixed = FixedKernel::new(&kernel, &mut fuel, usize::MAX).unwrap();
    let cases = cases(&fixed, &mut fuel);
    let before = fuel;
    let payload = Payload::new(&cases, &mut fuel, 4 * 1024 * 1024).unwrap();
    let payload_work = before - fuel;
    let bytes = execute(&payload.bytes, payload.count, source());
    assert_eq!(bytes.len(), payload.output_bytes);
    for (index, (case, result)) in cases
        .iter()
        .zip(bytes.as_slice().as_chunks::<16>().0)
        .enumerate()
    {
        let words: [u32; 4] = std::array::from_fn(|i| {
            u32::from_le_bytes(result[i * 4..i * 4 + 4].try_into().unwrap())
        });
        let actual = ((u64::from(words[1]) << 32) | u64::from(words[0])) as i64;
        let expected = case.oracle();
        assert_eq!(
            actual, expected,
            "GPU signed Q48 case {index}, opcode{}",
            case.opcode
        );
        assert_eq!(
            words[2],
            q24_bits(expected),
            "GPU completed Q24 conversion case{index}"
        );
    }
    let products: usize = cases
        .iter()
        .map(|case| match case.opcode {
            0 => 1,
            4 => 0,
            _ => 12,
        })
        .sum();
    println!(
        "ACTUAL VULKAN fixed Gaussian arithmetic: cases={}, all_raw_Q48_and_final_Q24_bits_exact=true, payload_work={payload_work}, input_bytes={}, output_bytes={}, conservative_data_bytes={}, Q48_products={products}, exact_u32_limb_products={}, initialization_nodes=1, compute_nodes=1, proof_readback_nodes=1; no coverage or scene admission",
        cases.len(),
        payload.bytes.len(),
        payload.output_bytes,
        payload.conservative_data_bytes,
        16 * products
    );
}

pub(super) fn prepare(
    bytes: &[u8],
    count: u32,
    program: GpuAdmittedProgramSource,
    context: &GpuContext,
    resources: &mut GpuResourceScope,
) -> (GpuBufferHandle, Vec<GpuWorkOperation>) {
    let output_bytes = count as usize * OUTPUT_BYTES;
    let device = context.device_facts();
    let limits = device.workload_budget().limits();
    let native = device.device_limits().values();
    for bytes in [bytes.len(), output_bytes] {
        assert!((bytes as u64) <= limits.max_buffer_size().min(native.max_buffer_size()));
        assert!(
            (bytes as u64)
                <= limits
                    .max_storage_buffer_binding_size()
                    .min(native.max_storage_buffer_binding_size())
        );
    }
    let groups = count.div_ceil(32);
    assert!(
        groups
            <= limits
                .max_compute_workgroups_per_dimension()
                .min(native.max_compute_workgroups_per_dimension())
    );
    let pipeline = GpuComputePipelineDescriptor::ordinary(program, "main").unwrap();

    let prepared =
        PreparedGpuData::<TransferData>::ordinary_pod_transfer("fixed arithmetic input", bytes)
            .unwrap();
    let input = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "fixed arithmetic input",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                bytes.len() as u64,
                [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Prepared(prepared),
            )
            .unwrap(),
        )
        .unwrap();
    let output = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "fixed arithmetic output",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                output_bytes as u64,
                [
                    GpuBufferUsage::Storage,
                    GpuBufferUsage::CopySource,
                    GpuBufferUsage::CopyDestination,
                ],
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let bindings = pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, &input),
            GpuRuntimeBindingValue::whole_buffer(0, 1, &output),
        ])
        .unwrap();
    let compute = GpuComputeOperation::new(
        pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(groups, 1, 1)),
    )
    .unwrap();

    let clear = GpuClearOperation::buffer_zero(GpuBufferRegion::whole(&output).unwrap()).unwrap();
    (output, vec![clear.into(), compute.into()])
}
