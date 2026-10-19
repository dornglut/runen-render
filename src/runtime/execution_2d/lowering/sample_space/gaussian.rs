//! RunenGPU preparation for certified Gaussian query data. Numerical payloads
//! contain no GPU ownership; this lowering uses the maintained program bridge.
use super::*;

fn resource(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::ResourceLimit,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    }
}

pub(super) fn prepare_queries(
    input: &[u8],
    count: u32,
    path: &[usize],
    context: &GpuContext,
    resources: &mut GpuResourceScope,
) -> Result<
    (
        Vec<GpuWorkOperation>,
        GpuBufferHandle,
        GpuRenderPipelineDescriptor,
    ),
    Render2dExecutionError,
> {
    let device = context.device_facts();
    if ![GpuCapabilityFeature::Compute, GpuCapabilityFeature::Copy]
        .into_iter()
        .all(|feature| device.is_enabled(feature))
    {
        let root_index = path.first().copied().ok_or_else(|| {
            precision_failure(path, "Gaussian query has no original occurrence owner")
        })?;
        return Err(crate::runtime::execution_2d::unsupported_at(
            path,
            crate::execution_2d::Render2dUnsupportedContent::Shadows { root_index },
        ));
    }
    if !device.is_enabled(GpuCapabilityFeature::RenderPipeline) {
        return Err(Render2dTargetAdmissionError::RenderPipelineNotAdmitted.into());
    }
    if ![GpuFormatRole::ColorAttachment, GpuFormatRole::Sampled]
        .into_iter()
        .all(|role| {
            device
                .admission_contract()
                .format_roles()
                .any(|fact| fact == (GpuTextureFormat::R32Float, role))
        })
    {
        return Err(Render2dTargetAdmissionError::SamplePlaneFormatUnsupported.into());
    }
    let limits = device.workload_budget().limits();
    let native = device.device_limits().values();
    let output_bytes = u64::from(count) * 16;
    let groups = count.div_ceil(32);
    let maximum = limits.max_buffer_size().min(native.max_buffer_size()).min(
        limits
            .max_storage_buffer_binding_size()
            .min(native.max_storage_buffer_binding_size()),
    );
    if count == 0
        || input.is_empty()
        || !input.len().is_multiple_of(4)
        || u64::try_from(input.len()).map_or(true, |bytes| bytes > maximum)
        || output_bytes > maximum
        || groups
            > limits
                .max_compute_workgroups_per_dimension()
                .min(native.max_compute_workgroups_per_dimension())
    {
        return Err(resource(
            path,
            "Gaussian query buffers or dispatch exceed device admission",
        ));
    }
    // Numerical node ranges and coefficients are certified by Batch. Keep
    // this physical ABI boundary independently safe against count/header
    // drift: shader dispatch, output size and encoded header must agree.
    let encoded_count = input
        .get(..4)
        .and_then(|word| word.try_into().ok())
        .map(u32::from_le_bytes);
    let minimum_bytes = usize::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(5))
        .and_then(|headers| headers.checked_add(836))
        .and_then(|words| words.checked_mul(4));
    if encoded_count != Some(count) || minimum_bytes.is_none_or(|bytes| input.len() < bytes) {
        return Err(precision_failure(
            path,
            "Gaussian encoded query header disagrees with dispatch",
        ));
    }
    let source = crate::runtime::program::retained_gaussian_coverage_source().map_err(|e| {
        Render2dExecutionError::Program {
            stage: "Gaussian maintained coverage source",
            detail: format!("{e:?}"),
        }
    })?;
    let compute_pipeline = GpuComputePipelineDescriptor::ordinary(source.clone(), "main")
        .map_err(|e| gpu("Gaussian query compute pipeline", e))?;
    let vertex = GpuEntryPointName::new("field_vertex")
        .map_err(|e| gpu("Gaussian field vertex entry", e))?;
    let fragment = GpuEntryPointName::new("field_fragment")
        .map_err(|e| gpu("Gaussian field fragment entry", e))?;
    let program = GpuProgramDescriptor::new(source, [vertex.clone(), fragment.clone()], [])
        .map_err(|e| gpu("Gaussian field program", e))?;
    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::R32Float,
        None,
        GpuColorWriteMask::ALL,
    )
    .map_err(|e| gpu("Gaussian field target", e))?;
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([])
            .map_err(|e| gpu("Gaussian field vertex input", e))?,
        Some(GpuFragmentOutputStateDescriptor::new([target])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .map_err(|e| gpu("Gaussian field pipeline state", e))?;
    let field_pipeline = GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .map_err(|e| gpu("Gaussian field pipeline", e))?;
    let mut operations = Vec::new();
    operations
        .try_reserve_exact(2)
        .map_err(|_| resource(path, "Gaussian operation allocation failed"))?;
    if operations.capacity() != 2 {
        return Err(resource(
            path,
            "Gaussian operation capacity exceeds admission",
        ));
    }
    let prepared =
        PreparedGpuData::<TransferData>::ordinary_pod_transfer("Gaussian query input", input)
            .map_err(|e| gpu("Gaussian query transfer", e))?;
    let input = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "Gaussian query input",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                prepared.layout().byte_len(),
                [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Prepared(prepared),
            )
            .map_err(|e| gpu("Gaussian query input descriptor", e))?,
        )
        .map_err(|e| gpu("Gaussian query input", e))?;
    let output = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "Gaussian query output",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                output_bytes,
                [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Uninitialized,
            )
            .map_err(|e| gpu("Gaussian query output descriptor", e))?,
        )
        .map_err(|e| gpu("Gaussian query output", e))?;
    let bindings = compute_pipeline
        .runtime_bindings([
            GpuRuntimeBindingValue::whole_buffer(0, 0, &input),
            GpuRuntimeBindingValue::whole_buffer(0, 1, &output),
        ])
        .map_err(|e| gpu("Gaussian query compute bindings", e))?;
    let compute = GpuComputeOperation::new(
        compute_pipeline,
        bindings,
        GpuDispatchIntent::direct(GpuDispatchSize::new(groups, 1, 1)),
    )
    .map_err(|e| gpu("Gaussian query dispatch", e))?;
    let region = GpuBufferRegion::whole(&output)
        .map_err(|e| gpu("Gaussian query initialization region", e))?;
    let clear = GpuClearOperation::buffer_zero(region)
        .map_err(|e| gpu("Gaussian query initialization", e))?;
    operations.extend([clear.into(), compute.into()]);
    Ok((operations, output, field_pipeline))
}

#[test]
#[ignore = "requires Vulkan: typed query dispatch admission and same-context reuse"]
fn gaussian_query_dispatch_limit_keeps_nested_owner_and_context_reuse_on_vulkan() {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Compute,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::Sampled)
        .permit_limit(GpuLimitKind::MaxComputeWorkgroupsPerDimension, 1);
    let context = pollster::block_on(GpuContext::request(descriptor)).unwrap();
    assert_eq!(
        context
            .device_facts()
            .workload_budget()
            .limits()
            .max_compute_workgroups_per_dimension(),
        1
    );
    // A full table prefix followed by 33 zero-node queries needs two dispatch
    // groups. These queries do not read any coefficient. Reject using the
    // admitted workload ceiling even when the native device can execute more.
    let mut input = vec![0_u8; (836 + 5 * 33) * 4];
    input[..4].copy_from_slice(&33_u32.to_le_bytes());
    let mut resources = GpuResourceScope::new();
    let error = prepare_queries(&input, 33, &[2, 1], &context, &mut resources).unwrap_err();
    assert!(matches!(error, Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::ResourceLimit, path: Some(owner), ..
    } if owner == [2, 1]));
    // The same context and preparation scope remain usable after rejection.
    // No target operation or synthetic contribution token was manufactured.
    input[..4].copy_from_slice(&1_u32.to_le_bytes());
    // A mismatched or truncated ABI cannot authorize a smaller output than
    // the shader will address, even when its physical byte extent fits.
    for malformed in [&input[..4], &input[..input.len()]] {
        let requested_count = if malformed.len() == 4 { 1 } else { 2 };
        assert!(
            matches!(prepare_queries(malformed, requested_count, &[2, 1], &context, &mut resources),
            Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::PrecisionLimit, path: Some(owner), .. })
            if owner == [2, 1])
        );
    }
    let (operations, _, _) = prepare_queries(&input, 1, &[2, 1], &context, &mut resources).unwrap();
    assert_eq!(operations.len(), 2);
}
