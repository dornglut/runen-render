//! RunenGPU preparation for certified Gaussian query data. Numerical payloads
//! contain no GPU ownership; this lowering uses the maintained program bridge.
use super::*;

pub(super) const TILE_UNIFORM_BYTES: u64 = 32;

/// One-time source-backed POD copies plus bounded field/dispatch descriptor
/// preparation. Shader compilation and backend work retain their sibling
/// authorities. Tile uniforms and compositor vertices are paid per use below.
pub(in crate::runtime::execution_2d) fn field_preparation_work(
    samples: usize,
    query_bytes: usize,
    query_count: u32,
) -> Option<usize> {
    samples
        .checked_mul(size_of::<u32>())?
        .checked_add(query_bytes)?
        .checked_add(if query_count == 0 { 16 } else { 0 })?
        .checked_mul(2)?
        .checked_add(4096)
}

/// One lookup uniform, lookup descriptors and one six-vertex F3E fill draw.
/// Every source-backed upload copies into a transfer Vec and then an Arc.
pub(in crate::runtime::execution_2d) fn lookup_preparation_work() -> usize {
    512 + 1024 + 2 * TILE_UNIFORM_BYTES as usize + 2 * 6 * VERTEX_STRIDE as usize
}

/// Convert the existing continuous-canvas center guard to map-local integer
/// cutoffs. All u32 sample indices and their half-unit centers are exactly
/// representable in f64. Binary search therefore preserves equality and its
/// adjacent representable canvas values without a rounded f32 shader bound.
pub(super) fn canvas_extent(
    canvas: [f64; 2],
    root: [u32; 4],
    work: &mut dyn crate::runtime::execution_2d::support::PreparationWork,
    path: &[usize],
) -> Result<[u32; 2], Render2dExecutionError> {
    let charge = |work: &mut dyn crate::runtime::execution_2d::support::PreparationWork, cost| {
        work.charge(
            cost,
            usize::MAX,
            path,
            "Gaussian canvas clipping exceeds invocation work",
        )
    };
    charge(work, 128)?;
    let mut result = [0; 2];
    for axis in 0..2 {
        let limit = canvas[axis] * f64::from(SAMPLES);
        if !limit.is_finite() || limit < 0.0 || root[axis] > root[axis + 2] {
            return Err(precision_failure(path, "Gaussian canvas cutoff is invalid"));
        }
        let mut low = root[axis];
        let mut high = root[axis + 2];
        while low < high {
            charge(work, 64)?;
            let middle = low + (high - low) / 2;
            if f64::from(middle) + 0.5 < limit {
                low = middle + 1;
            } else {
                high = middle;
            }
        }
        result[axis] = low - root[axis];
    }
    Ok(result)
}

fn resource(path: &[usize], detail: &'static str) -> Render2dExecutionError {
    Render2dExecutionError::SampleSpace {
        kind: Render2dSampleSpaceError::ResourceLimit,
        path: Some(path.to_vec()),
        detail: detail.to_owned(),
    }
}

/// Compiler-owned immutable data. The compiler certifies every literal/query
/// index; this physical boundary checks extents without rescanning its samples.
pub(super) struct FieldData<'a> {
    pub(super) samples: &'a [u32],
    pub(super) extent: [u32; 2],
    /// Exclusive canvas cutoff in map coordinates. It clips disposable
    /// samples only; map extent/stride and original support stay unchanged.
    pub(super) visible_extent: [u32; 2],
    pub(super) query_input: &'a [u8],
    pub(super) query_count: u32,
}

/// One immutable map upload and optional query dispatch, shared across tiles.
pub(super) struct PreparedField {
    pub(super) operations: Vec<GpuWorkOperation>,
    mapping: GpuBufferHandle,
    queries: GpuBufferHandle,
    pipeline: GpuRenderPipelineDescriptor,
    extent: [u32; 2],
    visible_extent: [u32; 2],
    max_texture_dimension: u32,
}

/// One renderer-owned R32 lookup destination. Ordered lookup/fill operations
/// may reuse this allocation; uniforms remain immutable per authored lookup.
pub(super) struct FieldTile {
    pub(super) view: GpuTextureViewHandle,
    extent: [u32; 2],
    pub(super) scratch_bytes: u64,
}

struct FieldSlot {
    root: [u32; 2],
    field: Option<PreparedField>,
}

/// Invocation-local physical fields share one ordered R32 lookup tile.
/// Initialization is consumed only when a visible compositor fill uses a field.
pub(super) struct FieldSet {
    slots: Vec<FieldSlot>,
    tile: Option<FieldTile>,
}

pub(super) fn field_metadata_bytes(count: usize) -> Option<usize> {
    count
        .checked_mul(size_of::<FieldSlot>() + 2 * size_of::<GpuWorkOperation>())?
        .checked_add(size_of::<FieldSet>())
}

pub(super) fn field_data(field: &super::shadow::NumericalField) -> FieldData<'_> {
    let (query_input, query_count) = field.plane.query_data();
    FieldData {
        samples: field.plane.samples(),
        extent: field.plane.extent(),
        visible_extent: field.visible_extent,
        query_input,
        query_count,
    }
}

impl FieldSet {
    pub(super) fn prepare(
        fields: &[super::shadow::NumericalField],
        uses: &[u64],
        dimension: u32,
        scratch_lease: u64,
        context: &GpuContext,
        resources: &mut GpuResourceScope,
    ) -> Result<Self, Render2dExecutionError> {
        if fields.len() != uses.len() {
            return Err(precision_failure(
                &[],
                "Gaussian use inventory disagrees with fields",
            ));
        }
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(fields.len())
            .map_err(|_| resource(&[], "Gaussian physical field inventory allocation failed"))?;
        if slots.capacity() > fields.len() {
            return Err(resource(
                &[],
                "Gaussian physical field capacity exceeds admission",
            ));
        }
        let mut tile = None;
        for (source, &uses) in fields.iter().zip(uses) {
            let root = source.plane.root();
            let field = if uses == 0 {
                None
            } else {
                let field = prepare_field(field_data(source), &source.path, context, resources)?;
                if tile.is_none() {
                    let allocated =
                        field.tile([dimension; 2], scratch_lease, &source.path, resources)?;
                    if allocated.scratch_bytes > scratch_lease {
                        return Err(resource(
                            &source.path,
                            "Gaussian scratch receipt exceeds admission",
                        ));
                    }
                    tile = Some(allocated);
                }
                Some(field)
            };
            slots.push(FieldSlot {
                root: [root[0], root[1]],
                field,
            });
        }
        Ok(Self { slots, tile })
    }

    pub(super) fn lookup(
        &mut self,
        index: usize,
        origin: [u32; 2],
        path: &[usize],
        operations: &mut Vec<GpuWorkOperation>,
        resources: &mut GpuResourceScope,
    ) -> Result<GpuTextureViewHandle, Render2dExecutionError> {
        let slot = self
            .slots
            .get_mut(index)
            .ok_or_else(|| precision_failure(path, "Gaussian lookup has no field"))?;
        let field = slot
            .field
            .as_mut()
            .ok_or_else(|| precision_failure(path, "Gaussian lookup was not admitted"))?;
        let tile = self
            .tile
            .as_ref()
            .ok_or_else(|| precision_failure(path, "Gaussian lookup has no scratch tile"))?;
        let offset = |axis| {
            i32::try_from(i64::from(origin[axis]) * i64::from(SAMPLES) - i64::from(slot.root[axis]))
                .map_err(|_| {
                    precision_failure(path, "Gaussian root-to-map offset exceeds signed domain")
                })
        };
        let offset = [offset(0)?, offset(1)?];
        // Every caller immediately emits a fill and marks its group/tile
        // painted. A successful enclosing merge cannot truncate this prefix.
        // Any later failure abandons the complete contribution and this owner.
        for operation in field.operations.drain(..) {
            append(operations, operation)?;
        }
        append(operations, tile.lookup(field, offset, path, resources)?)?;
        Ok(tile.view.clone())
    }
}

fn map_bytes(
    extent: [u32; 2],
    samples: usize,
    maximum: u64,
    path: &[usize],
) -> Result<u64, Render2dExecutionError> {
    if extent.contains(&0) {
        return Err(precision_failure(path, "Gaussian map extent is empty"));
    }
    let count = u64::from(extent[0]) * u64::from(extent[1]);
    // WGSL performs row-major indexing in u32; signed crop coordinates must
    // cover the complete map. These are physical ABI bounds, not geometry.
    if extent.iter().any(|&v| v > i32::MAX as u32)
        || count > u64::from(u32::MAX)
        || count * 4 > maximum
    {
        return Err(resource(
            path,
            "Gaussian map exceeds buffer/index admission",
        ));
    }
    if usize::try_from(count).ok() != Some(samples) {
        return Err(precision_failure(
            path,
            "Gaussian map extent disagrees with samples",
        ));
    }
    Ok(count * 4)
}

fn tile_bytes(
    extent: [u32; 2],
    dimension: u32,
    lease: u64,
    path: &[usize],
) -> Result<u64, Render2dExecutionError> {
    if extent.contains(&0) || extent.iter().any(|&v| v > dimension || v > i32::MAX as u32) {
        return Err(resource(
            path,
            "Gaussian tile exceeds texture extent admission",
        ));
    }
    let bytes = u64::from(extent[0]) * u64::from(extent[1]) * 4;
    if bytes > lease.min(MAX_PRIVATE_SCRATCH_BYTES) {
        return Err(resource(
            path,
            "Gaussian tile exceeds remaining scratch admission",
        ));
    }
    Ok(bytes)
}

fn tile_offset(
    extent: [u32; 2],
    offset: [i32; 2],
    path: &[usize],
) -> Result<(), Render2dExecutionError> {
    for (size, offset) in extent.into_iter().zip(offset) {
        // All fragment-local coordinates are nonnegative. Check the largest
        // signed sum BEFORE uniform allocation; WGSL must never wrap it.
        if size == 0 || i64::from(offset) + i64::from(size) - 1 > i64::from(i32::MAX) {
            return Err(precision_failure(
                path,
                "Gaussian tile-to-map signed index overflow",
            ));
        }
    }
    Ok(())
}

pub(super) fn admit_field(
    data: &FieldData<'_>,
    path: &[usize],
    context: &GpuContext,
) -> Result<(u64, u32), Render2dExecutionError> {
    let facts = context.device_facts();
    let native = facts.device_limits().values();
    let limits = facts.workload_budget().limits();
    let maximum = limits.max_buffer_size().min(native.max_buffer_size()).min(
        limits
            .max_storage_buffer_binding_size()
            .min(native.max_storage_buffer_binding_size()),
    );
    if native
        .max_uniform_buffer_binding_size()
        .min(limits.max_uniform_buffer_binding_size())
        < TILE_UNIFORM_BYTES
    {
        return Err(resource(
            path,
            "Gaussian tile uniform exceeds device admission",
        ));
    }
    let bytes = map_bytes(data.extent, data.samples.len(), maximum, path)?;
    if data
        .visible_extent
        .into_iter()
        .zip(data.extent)
        .any(|(visible, extent)| visible > extent)
    {
        return Err(precision_failure(
            path,
            "Gaussian canvas cutoff exceeds map extent",
        ));
    }
    admit_queries(data.query_input, data.query_count, path, context)?;
    Ok((
        bytes,
        native
            .max_texture_dimension_2d()
            .min(limits.max_texture_dimension_2d()),
    ))
}

pub(super) fn prepare_field(
    data: FieldData<'_>,
    path: &[usize],
    context: &GpuContext,
    resources: &mut GpuResourceScope,
) -> Result<PreparedField, Render2dExecutionError> {
    let (bytes, max_texture_dimension) = admit_field(&data, path, context)?;
    let (operations, queries, pipeline) =
        prepare_queries(data.query_input, data.query_count, path, context, resources)?;
    let payload = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "Gaussian immutable sample map",
        data.samples,
    )
    .map_err(|e| gpu("Gaussian map transfer", e))?;
    let mapping = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "Gaussian immutable sample map",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                bytes,
                [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Prepared(payload),
            )
            .map_err(|e| gpu("Gaussian map descriptor", e))?,
        )
        .map_err(|e| gpu("Gaussian map buffer", e))?;
    Ok(PreparedField {
        operations,
        mapping,
        queries,
        pipeline,
        extent: data.extent,
        visible_extent: data.visible_extent,
        max_texture_dimension,
    })
}

impl PreparedField {
    pub(super) fn tile(
        &self,
        extent: [u32; 2],
        scratch_lease: u64,
        path: &[usize],
        resources: &mut GpuResourceScope,
    ) -> Result<FieldTile, Render2dExecutionError> {
        let scratch_bytes = tile_bytes(extent, self.max_texture_dimension, scratch_lease, path)?;
        let texture = resources
            .texture(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "Gaussian reusable lookup tile",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    extent[0],
                    extent[1],
                    GpuTextureFormat::R32Float,
                    [GpuTextureUsage::ColorAttachment, GpuTextureUsage::Sampled],
                    GpuTextureInitialization::Uninitialized,
                )
                .map_err(|e| gpu("Gaussian tile descriptor", e))?,
            )
            .map_err(|e| gpu("Gaussian tile texture", e))?;
        let view = resources
            .texture_view(
                GpuTextureViewDescriptor::ordinary_full_owned(
                    "Gaussian reusable lookup tile view",
                    &texture,
                )
                .map_err(|e| gpu("Gaussian tile view descriptor", e))?,
            )
            .map_err(|e| gpu("Gaussian tile view", e))?;
        Ok(FieldTile {
            view,
            extent,
            scratch_bytes,
        })
    }
}

impl FieldTile {
    pub(super) fn lookup(
        &self,
        field: &PreparedField,
        offset: [i32; 2],
        path: &[usize],
        resources: &mut GpuResourceScope,
    ) -> Result<GpuRenderOperation, Render2dExecutionError> {
        tile_offset(self.extent, offset, path)?;
        let uniform = [
            field.extent[0],
            field.extent[1],
            offset[0].cast_unsigned(),
            offset[1].cast_unsigned(),
            field.visible_extent[0],
            field.visible_extent[1],
            0,
            0,
        ];
        let payload = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
            "Gaussian tile-to-map origin",
            &uniform,
        )
        .map_err(|e| gpu("Gaussian tile origin transfer", e))?;
        let dimensions = resources
            .buffer(
                GpuBufferDescriptor::ordinary_owned(
                    "Gaussian tile-to-map origin",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    TILE_UNIFORM_BYTES,
                    [GpuBufferUsage::Uniform, GpuBufferUsage::CopyDestination],
                    GpuBufferInitialization::Prepared(payload),
                )
                .map_err(|e| gpu("Gaussian tile origin descriptor", e))?,
            )
            .map_err(|e| gpu("Gaussian tile origin buffer", e))?;
        let bindings = field
            .pipeline
            .runtime_bindings([
                GpuRuntimeBindingValue::whole_buffer(0, 2, &field.mapping),
                GpuRuntimeBindingValue::whole_buffer(0, 3, &field.queries),
                GpuRuntimeBindingValue::whole_buffer(0, 4, &dimensions),
            ])
            .map_err(|e| gpu("Gaussian tile lookup bindings", e))?;
        let draw = GpuRenderDraw::new(
            field.pipeline.clone(),
            bindings,
            [],
            None,
            GpuDrawIntent::direct(
                GpuDrawRange::new(0, 3).map_err(|e| gpu("Gaussian tile vertices", e))?,
                GpuDrawRange::new(0, 1).map_err(|e| gpu("Gaussian tile instances", e))?,
            ),
            GpuViewport::new(
                0.0,
                0.0,
                self.extent[0] as f32,
                self.extent[1] as f32,
                0.0,
                1.0,
            )
            .map_err(|e| gpu("Gaussian tile viewport", e))?,
            GpuScissorRect::new(0, 0, self.extent[0], self.extent[1])
                .map_err(|e| gpu("Gaussian tile scissor", e))?,
            GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0)
                .map_err(|e| gpu("Gaussian tile blend constant", e))?,
            0,
        )
        .map_err(|e| gpu("Gaussian tile draw", e))?;
        operation(&self.view, true, vec![draw])
    }
}

fn admit_queries(
    input: &[u8],
    count: u32,
    path: &[usize],
    context: &GpuContext,
) -> Result<(u64, u32), Render2dExecutionError> {
    let device = context.device_facts();
    if !device.is_enabled(GpuCapabilityFeature::Copy)
        || (count > 0 && !device.is_enabled(GpuCapabilityFeature::Compute))
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
    // The field shader has one query binding even when every sample is a
    // literal. Retain one initialized element without dispatching computation.
    let output_bytes = (u64::from(count) * 16).max(16);
    let groups = count.div_ceil(32);
    let maximum = limits.max_buffer_size().min(native.max_buffer_size()).min(
        limits
            .max_storage_buffer_binding_size()
            .min(native.max_storage_buffer_binding_size()),
    );
    if !input.len().is_multiple_of(4)
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
    if (count == 0 && !input.is_empty())
        || (count > 0
            && (encoded_count != Some(count)
                || minimum_bytes.is_none_or(|bytes| input.len() < bytes)))
    {
        return Err(precision_failure(
            path,
            "Gaussian encoded query header disagrees with dispatch",
        ));
    }
    Ok((output_bytes, groups))
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
    let (output_bytes, groups) = admit_queries(input, count, path, context)?;
    let source = crate::runtime::program::retained_gaussian_coverage_source().map_err(|e| {
        Render2dExecutionError::Program {
            stage: "Gaussian maintained coverage source",
            detail: format!("{e:?}"),
        }
    })?;
    let vertex = GpuEntryPointName::new("field_vertex")
        .map_err(|e| gpu("Gaussian field vertex entry", e))?;
    let fragment = GpuEntryPointName::new("field_fragment")
        .map_err(|e| gpu("Gaussian field fragment entry", e))?;
    let program = GpuProgramDescriptor::new(source.clone(), [vertex.clone(), fragment.clone()], [])
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
    if count == 0 {
        let prepared = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
            "Gaussian literal field query binding",
            &[0_u32; 4],
        )
        .map_err(|e| gpu("Gaussian literal field query transfer", e))?;
        let output = resources
            .buffer(
                GpuBufferDescriptor::ordinary_owned(
                    "Gaussian literal field query binding",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    output_bytes,
                    [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                    GpuBufferInitialization::Prepared(prepared),
                )
                .map_err(|e| gpu("Gaussian literal query descriptor", e))?,
            )
            .map_err(|e| gpu("Gaussian literal query binding", e))?;
        return Ok((operations, output, field_pipeline));
    }
    let compute_pipeline = GpuComputePipelineDescriptor::ordinary(source, "main")
        .map_err(|e| gpu("Gaussian query compute pipeline", e))?;
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

#[test]
fn gaussian_tile_metadata_admission_preserves_limits_and_signed_crop_domain() {
    let owner = &[2, 1];
    assert_eq!(map_bytes([16, 8], 128, 512, owner).unwrap(), 512);
    for (extent, count, maximum) in [([16, 8], 128, 511), ([u32::MAX, 8], 0, u64::MAX)] {
        assert!(matches!(map_bytes(extent, count, maximum, owner),
            Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::ResourceLimit,
                path: Some(path), .. }) if path == *owner));
    }
    for (extent, count) in [([0, 8], 0), ([16, 8], 127)] {
        assert!(matches!(map_bytes(extent, count, 512, owner),
            Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::PrecisionLimit,
                path: Some(path), .. }) if path == *owner));
    }
    assert_eq!(tile_bytes([8, 4], 8, 128, owner).unwrap(), 128);
    assert!(tile_bytes([8, 4], 8, 127, owner).is_err());
    assert!(tile_bytes([8, 4], 4, 128, owner).is_err());
    assert!(tile_bytes([0, 4], 8, 128, owner).is_err());
    for offset in [[-4, -4], [4, 4], [i32::MIN, i32::MAX - 7]] {
        tile_offset([8; 2], offset, owner).unwrap();
    }
    assert!(matches!(tile_offset([8; 2], [0, i32::MAX - 6], owner),
        Err(Render2dExecutionError::SampleSpace { kind: Render2dSampleSpaceError::PrecisionLimit,
            path: Some(path), .. }) if path == *owner));
}

#[test]
fn gaussian_canvas_cutoffs_preserve_half_sample_equality_phase_and_query_fuel() {
    let boundary = 1.125_f64;
    for (canvas, expected) in [
        (boundary.next_down(), 4),
        (boundary, 4),
        (boundary.next_up(), 5),
        (1.2, 5),
    ] {
        assert_eq!(
            canvas_extent([canvas; 2], [0, 0, 8, 8], &mut 0_usize, &[2, 1]).unwrap(),
            [expected; 2]
        );
    }
    assert_eq!(
        canvas_extent([1.2; 2], [3, 3, 11, 11], &mut 0_usize, &[2, 1]).unwrap(),
        [2; 2],
        "map origins retain the global sample phase"
    );
    for canvas in [[f64::NAN, 1.0], [-1.0, 1.0], [f64::MAX, 1.0]] {
        assert!(canvas_extent(canvas, [0, 0, 8, 8], &mut 0_usize, &[2, 1]).is_err());
    }
    assert!(canvas_extent([1.0; 2], [9, 0, 8, 8], &mut 0_usize, &[2, 1]).is_err());
    let mut measured =
        crate::runtime::execution_2d::support::PreparationBudget::for_contribution(true);
    canvas_extent([1.2; 2], [0, 0, 8, 8], &mut measured, &[2, 1]).unwrap();
    use crate::runtime::execution_2d::support::PreparationWork;
    let paid = measured.used();
    for (remaining, succeeds) in [(paid, true), (paid - 1, false)] {
        let mut work =
            crate::runtime::execution_2d::support::PreparationBudget::from_prepared_sources(
                true,
                crate::runtime::execution_2d::support::MAX_COMPOSITIONAL_SHADOW_WORK - remaining,
            );
        assert_eq!(
            canvas_extent([1.2; 2], [0, 0, 8, 8], &mut work, &[2, 1]).is_ok(),
            succeeds
        );
    }
}

#[test]
fn gaussian_physical_preparation_work_counts_shared_field_and_each_lookup() {
    assert_eq!(
        field_preparation_work(64, 0, 0),
        Some(4096 + 2 * (256 + 16))
    );
    assert_eq!(
        field_preparation_work(64, 100, 1),
        Some(4096 + 2 * (256 + 100))
    );
    assert!(field_preparation_work(usize::MAX, 0, 0).is_none());
    assert!(field_preparation_work(1, usize::MAX, 1).is_none());
    assert_eq!(lookup_preparation_work(), 1984);
}
