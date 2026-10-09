//! Ordered private RunenGPU lowering for admitted text and solid vectors.

mod clip;
mod image;
mod vector;

use super::field::GlyphField;
use crate::composition_2d::{Render2dColorRgba8, Render2dResourceId};
use crate::execution_2d::{
    Render2dExecutionError, Render2dTarget, Render2dTargetAdmissionError, Render2dWorkBinding,
};
use crate::runtime::program::retained_shaped_text_source;
use runen_gpu::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    sync::Arc,
};

const FIELD_FORMAT: GpuTextureFormat = GpuTextureFormat::Rgba8Unorm;
const VERTEX_STRIDE: u64 = crate::runtime::program::abi::COMPOSITION_VERTEX_STRIDE;
const FLOATS_PER_VERTEX: usize = 8;
const VERTICES_PER_GLYPH: u32 = 6;
const GLYPH_VERTEX_ARRAY_LEN: usize = 6;

#[derive(Clone, Debug)]
pub(super) struct AdmittedTarget {
    view: GpuTextureViewHandle,
    format: GpuTextureFormat,
    physical_width: u32,
    physical_height: u32,
    continuous_width: f64,
    continuous_height: f64,
    raster_scale: f64,
    max_texture_dimension_2d: u32,
    max_buffer_bytes: u64,
    coverage_format: bool,
    image_format: bool,
    clip_format: bool,
}

impl AdmittedTarget {
    pub(super) const fn canvas(&self) -> [f64; 2] {
        [self.continuous_width, self.continuous_height]
    }
    pub(super) const fn max_buffer_bytes(&self) -> u64 {
        self.max_buffer_bytes
    }

    pub(super) const fn raster_scale(&self) -> f64 {
        self.raster_scale
    }

    pub(super) const fn max_texture_dimension_2d(&self) -> u32 {
        self.max_texture_dimension_2d
    }
}

#[derive(Clone, Debug)]
pub(super) struct GlyphOccurrence {
    pub(super) root_index: usize,
    pub(super) resource_id: Render2dResourceId,
    pub(super) field: Arc<GlyphField>,
    pub(super) logical_x: f64,
    pub(super) logical_y: f64,
    pub(super) logical_width: f64,
    pub(super) logical_height: f64,
    pub(super) color: Render2dColorRgba8,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct FieldTextureKey {
    resource_id: Render2dResourceId,
    glyph_id: u32,
}

impl FieldTextureKey {
    const fn new(resource_id: Render2dResourceId, glyph_id: u32) -> Self {
        Self {
            resource_id,
            glyph_id,
        }
    }
}

#[derive(Clone, Debug)]
struct DrawSpec {
    key: FieldTextureKey,
    field: Arc<GlyphField>,
    first_vertex: u32,
}

pub(super) fn admit_target(
    context: &GpuContext,
    target: &Render2dTarget,
    needs_fields: bool,
) -> Result<AdmittedTarget, Render2dExecutionError> {
    let continuous_width = target.logical_width() * target.raster_scale();
    let continuous_height = target.logical_height() * target.raster_scale();
    if !continuous_width.is_finite()
        || !continuous_height.is_finite()
        || continuous_width <= 0.0
        || continuous_height <= 0.0
        || continuous_width.ceil() > f64::from(u32::MAX)
        || continuous_height.ceil() > f64::from(u32::MAX)
    {
        return Err(Render2dTargetAdmissionError::PhysicalExtentOutOfRange.into());
    }

    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "positive finite ceiling values are explicitly bounded by u32::MAX above"
    )]
    let (physical_width, physical_height) = (
        continuous_width.ceil() as u32,
        continuous_height.ceil() as u32,
    );

    let view = target.view();
    let view_descriptor = view.descriptor();
    let texture = view_descriptor.texture();
    let texture_descriptor = texture.descriptor();
    let subresources = view_descriptor.subresources();

    if view_descriptor.dimension() != GpuTextureViewDimension::D2
        || texture_descriptor.dimension() != GpuTextureDimension::D2
        || texture_descriptor.extent().depth_or_layers() != 1
        || subresources.base_mip_level() != 0
        || subresources.mip_level_count() != 1
        || subresources.base_array_layer() != 0
        || subresources.array_layer_count() != 1
        || !matches!(
            subresources.aspect(),
            GpuTextureAspect::All | GpuTextureAspect::Color
        )
    {
        return Err(Render2dTargetAdmissionError::UnsupportedViewShape.into());
    }

    if texture_descriptor.sample_count() != 1 {
        return Err(Render2dTargetAdmissionError::UnsupportedSampleCount.into());
    }

    let format = view_descriptor
        .format()
        .unwrap_or_else(|| texture_descriptor.format());
    if !matches!(
        format,
        GpuTextureFormat::Rgba8UnormSrgb | GpuTextureFormat::Bgra8UnormSrgb
    ) {
        return Err(Render2dTargetAdmissionError::UnsupportedFormat { format }.into());
    }

    if !texture_descriptor
        .usages()
        .contains(GpuTextureUsage::ColorAttachment)
    {
        return Err(Render2dTargetAdmissionError::MissingColorAttachmentUsage.into());
    }

    let actual_width = texture_descriptor.extent().width();
    let actual_height = texture_descriptor.extent().height();
    if actual_width != physical_width || actual_height != physical_height {
        return Err(Render2dTargetAdmissionError::PhysicalExtentMismatch {
            expected_width: physical_width,
            expected_height: physical_height,
            actual_width,
            actual_height,
        }
        .into());
    }

    if !context
        .device_facts()
        .is_enabled(GpuCapabilityFeature::RenderPipeline)
    {
        return Err(Render2dTargetAdmissionError::RenderPipelineNotAdmitted.into());
    }

    let admitted_roles = context
        .device_facts()
        .admission_contract()
        .format_roles()
        .collect::<BTreeSet<_>>();
    let parent_format = texture_descriptor.format();
    if !admitted_roles.contains(&(parent_format, GpuFormatRole::ColorAttachment))
        || !admitted_roles.contains(&(format, GpuFormatRole::ColorAttachment))
        || !admitted_roles.contains(&(format, GpuFormatRole::Blendable))
    {
        return Err(Render2dTargetAdmissionError::TargetFormatNotBlendable.into());
    }
    for role in [
        GpuFormatRole::Sampled,
        GpuFormatRole::Filterable,
        GpuFormatRole::CopyDestination,
    ] {
        if needs_fields && !admitted_roles.contains(&(FIELD_FORMAT, role)) {
            return Err(Render2dTargetAdmissionError::FieldFormatUnsupported.into());
        }
    }

    let device_limit = context
        .device_facts()
        .device_limits()
        .values()
        .max_texture_dimension_2d();
    let workload_limit = context
        .device_facts()
        .workload_budget()
        .limits()
        .max_texture_dimension_2d();
    let max_texture_dimension_2d = device_limit.min(workload_limit);
    if physical_width > max_texture_dimension_2d || physical_height > max_texture_dimension_2d {
        return Err(Render2dTargetAdmissionError::TextureDimensionLimitExceeded.into());
    }

    Ok(AdmittedTarget {
        view: view.clone(),
        format,
        physical_width,
        physical_height,
        continuous_width,
        continuous_height,
        raster_scale: target.raster_scale(),
        max_texture_dimension_2d,
        coverage_format: [GpuFormatRole::ColorAttachment, GpuFormatRole::Sampled]
            .into_iter()
            .all(|role| admitted_roles.contains(&(FIELD_FORMAT, role))),
        image_format: [
            GpuFormatRole::Sampled,
            GpuFormatRole::Filterable,
            GpuFormatRole::CopyDestination,
        ]
        .into_iter()
        .all(|role| admitted_roles.contains(&(GpuTextureFormat::Rgba8UnormSrgb, role))),
        clip_format: [GpuFormatRole::Sampled, GpuFormatRole::CopyDestination]
            .into_iter()
            .all(|role| admitted_roles.contains(&(FIELD_FORMAT, role))),
        max_buffer_bytes: context
            .device_facts()
            .device_limits()
            .values()
            .max_buffer_size()
            .min(
                context
                    .device_facts()
                    .workload_budget()
                    .limits()
                    .max_buffer_size(),
            ),
    })
}

fn lower(
    target: &AdmittedTarget,
    occurrences: &[GlyphOccurrence],
    clipped: Option<&clip::ClipGpu>,
) -> Result<Option<GpuRenderOperation>, Render2dExecutionError> {
    let mut vertex_values = Vec::<f32>::new();
    let mut specs = Vec::<DrawSpec>::new();

    for occurrence in occurrences {
        let Some(vertices) = glyph_vertices(target, occurrence)? else {
            continue;
        };
        let start = vertex_count(&vertex_values)?;
        for vertex in vertices {
            vertex_values.extend_from_slice(&vertex);
        }
        start
            .checked_add(VERTICES_PER_GLYPH)
            .ok_or_else(|| gpu_text("vertex range", "2D glyph draw range overflow"))?;
        specs.push(DrawSpec {
            key: FieldTextureKey::new(occurrence.resource_id, occurrence.field.glyph_id()),
            field: Arc::clone(&occurrence.field),
            first_vertex: start,
        });
    }

    if specs.is_empty() {
        return Ok(None);
    }

    let pipeline = shaped_text_pipeline(target.format, clipped.is_some())?;
    let mut resources = GpuResourceScope::new();
    let sampler = create_sampler(&mut resources)?;
    let vertex_buffer = create_vertex_buffer(&mut resources, &vertex_values)?;
    let vertex_binding = GpuVertexBufferBinding::new(
        0,
        &vertex_buffer,
        GpuBufferRange::whole(&vertex_buffer).map_err(|error| gpu("vertex range", error))?,
    )
    .map_err(|error| gpu("vertex binding", error))?;

    let mut views = BTreeMap::<FieldTextureKey, GpuTextureViewHandle>::new();
    for spec in &specs {
        if views.contains_key(&spec.key) {
            continue;
        }
        let view = create_field_view(&mut resources, spec.key, &spec.field)?;
        views.insert(spec.key, view);
    }

    let viewport = GpuViewport::new(
        0.0,
        0.0,
        f32_from_u32(target.physical_width),
        f32_from_u32(target.physical_height),
        0.0,
        1.0,
    )
    .map_err(|error| gpu("viewport", error))?;
    let scissor = GpuScissorRect::new(0, 0, target.physical_width, target.physical_height)
        .map_err(|error| gpu("scissor", error))?;
    let blend_constant =
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).map_err(|error| gpu("blend constant", error))?;

    let mut draws = Vec::with_capacity(specs.len());
    for spec in specs {
        let view = views
            .get(&spec.key)
            .expect("every retained draw spec must have one prepared field view");
        let mut values = vec![texture_binding(0, view)?, sampler_binding(1, &sampler)?];
        if let Some(clip) = clipped {
            values.extend(clip::bindings(clip, 2, 3)?);
        }
        let bindings = pipeline
            .runtime_bindings(values)
            .map_err(|error| gpu("runtime binding validation", error))?;
        let draw = GpuRenderDraw::new(
            pipeline.clone(),
            bindings,
            [vertex_binding.clone()],
            None,
            GpuDrawIntent::direct(
                GpuDrawRange::new(spec.first_vertex, VERTICES_PER_GLYPH)
                    .map_err(|error| gpu("direct vertex range", error))?,
                GpuDrawRange::new(0, 1).map_err(|error| gpu("instance range", error))?,
            ),
            viewport,
            scissor,
            blend_constant,
            0,
        )
        .map_err(|error| gpu("render draw", error))?;
        draws.push(draw);
    }

    let attachment = GpuRenderColorAttachment::new(
        target.view.clone(),
        GpuColorAttachmentLoad::Load,
        GpuAttachmentStore::Store,
        None,
    )
    .map_err(|error| gpu("load/store target attachment", error))?;
    let render = GpuRenderOperation::new([attachment], None, draws, None)
        .map_err(|error| gpu("render operation", error))?;

    Ok(Some(render))
}

pub(crate) fn add_target_boundary(
    builder: &mut GpuWorkFragmentBuilder,
    target: &GpuTextureViewHandle,
    work_binding: &Render2dWorkBinding,
) -> Result<(), GpuWorkAuthoringError> {
    let texture = target.descriptor().texture();
    let resource = GpuResourceRef::Texture(texture.clone());
    builder.declare_resource(resource.clone())?;
    let provenance = GpuResourceProvenance::new(
        GpuResourceLabel::new("runen-render 2D target contents").expect("maintained label"),
        None,
        None,
    );
    if let Some(prior) = work_binding.prior() {
        builder.add_import(GpuWorkImport::new(
            resource.clone(),
            prior.clone(),
            GpuResourceAccessIntent::ReadWrite,
            provenance.clone(),
        ))?;
    }
    let coverage = GpuInitialCoverage::texture_subresources(
        &GpuTextureAccessResource::Texture(texture.clone()),
        [target.descriptor().subresources()],
    )?;
    builder.add_output(GpuWorkOutput::new(
        GpuExportRelationship::new(
            resource,
            work_binding.output().clone(),
            GpuResourceAccessIntent::ReadWrite,
            provenance,
        ),
        coverage,
    )?)?;
    Ok(())
}

fn shaped_text_pipeline(
    format: GpuTextureFormat,
    clipped: bool,
) -> Result<GpuRenderPipelineDescriptor, Render2dExecutionError> {
    let source =
        retained_shaped_text_source().map_err(|error| Render2dExecutionError::Program {
            stage: "retained shaped-text source",
            detail: format!("{error:?}"),
        })?;
    let vertex =
        GpuEntryPointName::new("vs_main").map_err(|error| gpu("vertex entry-point name", error))?;
    let fragment = GpuEntryPointName::new(if clipped {
        "fs_main_clipped"
    } else {
        "fs_main"
    })
    .map_err(|error| gpu("fragment entry-point name", error))?;
    let mut refinements = vec![
        GpuBindingLayoutRefinement::new(
            GpuBindingKey::try_new(0, 0).map_err(|error| gpu("field texture layout key", error))?,
        )
        .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
        GpuBindingLayoutRefinement::new(
            GpuBindingKey::try_new(0, 1).map_err(|error| gpu("field sampler layout key", error))?,
        )
        .with_sampler_class(GpuSamplerClass::Filtering),
    ];
    if clipped {
        refinements.push(
            GpuBindingLayoutRefinement::new(
                GpuBindingKey::try_new(0, 2)
                    .map_err(|error| gpu("clip texture layout key", error))?,
            )
            .with_texture_sample_class(GpuTextureSampleClass::FloatUnfilterable),
        );
    }
    let program =
        GpuProgramDescriptor::new(source, [vertex.clone(), fragment.clone()], refinements)
            .map_err(|error| gpu("shaped-text program descriptor", error))?;
    let vertex_layout = GpuVertexBufferLayoutDescriptor::new(
        0,
        VERTEX_STRIDE,
        GpuVertexStepMode::Vertex,
        [
            GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32x2),
            GpuVertexAttribute::new(1, 8, GpuVertexFormat::Float32x2),
            GpuVertexAttribute::new(2, 16, GpuVertexFormat::Float32x4),
        ],
    )
    .map_err(|error| gpu("shaped-text vertex layout", error))?;
    let blend_component = GpuBlendComponent::new(
        GpuBlendFactor::One,
        GpuBlendFactor::OneMinusSrcAlpha,
        GpuBlendOperation::Add,
    )
    .map_err(|error| gpu("source-over blend component", error))?;
    let target = GpuColorTargetStateDescriptor::new(
        format,
        Some(GpuBlendState::new(blend_component, blend_component)),
        GpuColorWriteMask::ALL,
    )
    .map_err(|error| gpu("shaped-text color target", error))?;
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([vertex_layout])
            .map_err(|error| gpu("shaped-text vertex input", error))?,
        Some(GpuFragmentOutputStateDescriptor::new([target])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .map_err(|error| gpu("shaped-text pipeline state", error))?;
    GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .map_err(|error| gpu("shaped-text render pipeline", error))
}

fn create_vertex_buffer(
    resources: &mut GpuResourceScope,
    values: &[f32],
) -> Result<GpuBufferHandle, Render2dExecutionError> {
    let prepared = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "runen-render 2D shaped-text vertices",
        values,
    )
    .map_err(|error| gpu("vertex payload", error))?;
    let byte_len = prepared.layout().byte_len();
    resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                "runen-render 2D shaped-text vertices",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                byte_len,
                [GpuBufferUsage::Vertex, GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Prepared(prepared),
            )
            .map_err(|error| gpu("vertex descriptor", error))?,
        )
        .map_err(|error| gpu("vertex identity", error))
}

fn create_field_view(
    resources: &mut GpuResourceScope,
    key: FieldTextureKey,
    field: &GlyphField,
) -> Result<GpuTextureViewHandle, Render2dExecutionError> {
    let name = format!(
        "runen-render 2D field {} {}",
        key.resource_id.get(),
        key.glyph_id
    );
    let label = GpuResourceLabel::new(&name).map_err(|error| gpu("field label", error))?;
    let data = PreparedGpuData::<TransferData>::ordinary_pod_transfer(&name, field.rgba8())
        .map_err(|error| gpu("field payload", error))?;
    let extent = GpuTextureExtent::new(
        &label,
        GpuTextureDimension::D2,
        field.width(),
        field.height(),
        1,
    )
    .map_err(|error| gpu("field extent", error))?;
    let bytes_per_row = field
        .width()
        .checked_mul(4)
        .ok_or_else(|| gpu_text("field row layout", "2D field row byte count overflow"))?;
    let prepared =
        GpuPreparedTextureData::new(&label, data, FIELD_FORMAT, extent, bytes_per_row, 0)
            .map_err(|error| gpu("field texture payload", error))?;
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                &name,
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                field.width(),
                field.height(),
                FIELD_FORMAT,
                [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
                GpuTextureInitialization::Prepared(prepared),
            )
            .map_err(|error| gpu("field texture descriptor", error))?,
        )
        .map_err(|error| gpu("field texture identity", error))?;
    resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(format!("{name} view"), &texture)
                .map_err(|error| gpu("field texture view descriptor", error))?,
        )
        .map_err(|error| gpu("field texture view identity", error))
}

fn create_sampler(
    resources: &mut GpuResourceScope,
) -> Result<GpuSamplerHandle, Render2dExecutionError> {
    let label = GpuResourceLabel::new("runen-render 2D field sampler")
        .map_err(|error| gpu("sampler label", error))?;
    let provenance = GpuResourceProvenance::new(label.clone(), None, None);
    let common = GpuResourceCommon::owned(
        label,
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance,
    )
    .map_err(|error| gpu("sampler common descriptor", error))?;
    let filters = GpuSamplerFilterState::new(
        GpuFilterMode::Linear,
        GpuFilterMode::Linear,
        GpuFilterMode::Linear,
        1,
    )
    .map_err(|error| gpu("sampler filter state", error))?;
    resources
        .sampler(
            GpuSamplerDescriptor::new(
                common,
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                filters,
                0.0,
                0.0,
                None,
            )
            .map_err(|error| gpu("sampler descriptor", error))?,
        )
        .map_err(|error| gpu("sampler identity", error))
}

fn texture_binding(
    binding: u32,
    view: &GpuTextureViewHandle,
) -> Result<GpuRuntimeBindingValue, Render2dExecutionError> {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, u64::from(binding))
            .map_err(|error| gpu("texture binding key", error))?,
        [GpuRuntimeBindingResource::TextureView(
            GpuRuntimeTextureViewBinding::new(view.clone()),
        )],
    )
    .map_err(|error| gpu("texture runtime binding", error))
}

fn sampler_binding(
    binding: u32,
    sampler: &GpuSamplerHandle,
) -> Result<GpuRuntimeBindingValue, Render2dExecutionError> {
    GpuRuntimeBindingValue::new(
        GpuBindingKey::try_new(0, u64::from(binding))
            .map_err(|error| gpu("sampler binding key", error))?,
        [GpuRuntimeBindingResource::Sampler(sampler.clone())],
    )
    .map_err(|error| gpu("sampler runtime binding", error))
}

fn vertex_count(values: &[f32]) -> Result<u32, Render2dExecutionError> {
    if !values.len().is_multiple_of(FLOATS_PER_VERTEX) {
        return Err(gpu_text(
            "vertex packing",
            "2D vertex payload lost fixed-stride alignment",
        ));
    }
    u32::try_from(values.len() / FLOATS_PER_VERTEX)
        .map_err(|_| gpu_text("vertex packing", "2D vertex count exceeds u32"))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "validated u32 target extents narrow to f32 only for RunenGPU viewport state"
)]
fn f32_from_u32(value: u32) -> f32 {
    value as f32
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "finite clipped logical/physical coordinates narrow only at the private GPU vertex ABI boundary"
)]
fn f32_from_f64(value: f64) -> f32 {
    value as f32
}

fn glyph_vertices(
    target: &AdmittedTarget,
    occurrence: &GlyphOccurrence,
) -> Result<Option<[[f32; FLOATS_PER_VERTEX]; GLYPH_VERTEX_ARRAY_LEN]>, Render2dExecutionError> {
    if occurrence.logical_width <= 0.0 || occurrence.logical_height <= 0.0 {
        return Err(gpu_text("glyph bounds", "2D glyph extent must be positive"));
    }

    let scale = target.raster_scale;
    let x0 = occurrence.logical_x * scale;
    let y0 = occurrence.logical_y * scale;
    let width = occurrence.logical_width * scale;
    let height = occurrence.logical_height * scale;
    let x1 = x0 + width;
    let y1 = y0 + height;
    if ![x0, y0, width, height, x1, y1]
        .into_iter()
        .all(f64::is_finite)
        || width <= 0.0
        || height <= 0.0
    {
        return Err(gpu_text(
            "glyph bounds",
            "2D physical glyph bounds are not representable",
        ));
    }

    let left = x0.max(0.0);
    let top = y0.max(0.0);
    let right = x1.min(target.continuous_width);
    let bottom = y1.min(target.continuous_height);
    if right <= left || bottom <= top {
        return Ok(None);
    }

    // Generated samples lie at texel centers. Geometry spans the texture edges,
    // so normalized UVs preserve the field's exact projection without rescaling it.
    let u0 = f32_from_f64(((left - x0) / width).clamp(0.0, 1.0));
    let u1 = f32_from_f64(((right - x0) / width).clamp(0.0, 1.0));
    let v0 = f32_from_f64(((top - y0) / height).clamp(0.0, 1.0));
    let v1 = f32_from_f64(((bottom - y0) / height).clamp(0.0, 1.0));

    let left_ndc = physical_x_to_ndc(left, target.physical_width);
    let right_ndc = physical_x_to_ndc(right, target.physical_width);
    let top_ndc = physical_y_to_ndc(top, target.physical_height);
    let bottom_ndc = physical_y_to_ndc(bottom, target.physical_height);
    let color = linear_color(occurrence.color);

    let vertex =
        |x: f32, y: f32, u: f32, v: f32| [x, y, u, v, color[0], color[1], color[2], color[3]];
    Ok(Some([
        vertex(left_ndc, top_ndc, u0, v0),
        vertex(right_ndc, top_ndc, u1, v0),
        vertex(left_ndc, bottom_ndc, u0, v1),
        vertex(left_ndc, bottom_ndc, u0, v1),
        vertex(right_ndc, top_ndc, u1, v0),
        vertex(right_ndc, bottom_ndc, u1, v1),
    ]))
}

fn physical_x_to_ndc(value: f64, physical_width: u32) -> f32 {
    f32_from_f64(2.0 * (value / f64::from(physical_width)) - 1.0)
}

fn physical_y_to_ndc(value: f64, physical_height: u32) -> f32 {
    f32_from_f64(1.0 - 2.0 * (value / f64::from(physical_height)))
}

fn linear_color(color: Render2dColorRgba8) -> [f32; 4] {
    let [r, g, b, a] = color.channels();
    [
        srgb8_to_linear(r),
        srgb8_to_linear(g),
        srgb8_to_linear(b),
        f32::from(a) / 255.0,
    ]
}

fn srgb8_to_linear(value: u8) -> f32 {
    let encoded = f64::from(value) / 255.0;
    let linear = if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    };
    f32_from_f64(linear)
}

fn gpu(stage: &'static str, error: impl fmt::Display) -> Render2dExecutionError {
    Render2dExecutionError::Gpu {
        stage,
        detail: error.to_string(),
    }
}

fn gpu_text(stage: &'static str, detail: impl Into<String>) -> Render2dExecutionError {
    Render2dExecutionError::Gpu {
        stage,
        detail: detail.into(),
    }
}

#[derive(Debug)]
pub(super) enum OrderedItem {
    Glyph(GlyphOccurrence),
    Vector(super::vector::VectorMesh),
    Image(super::image::ImagePatchWork),
}

// One contribution-local coverage surface is reused in lexical order. Clearing the
// complete mask before each vector item keeps discarded geometry and painter history
// out of coverage. Overlapping triangles overwrite 1; brush alpha is applied only
// after resolving the 16 binary samples of each destination pixel.
pub(super) fn lower_ordered(
    target: &AdmittedTarget,
    ordered: Vec<OrderedItem>,
    clipped: BTreeMap<usize, crate::runtime::execution_2d::clip::ClipMask>,
) -> Result<Vec<GpuRenderOperation>, Render2dExecutionError> {
    use crate::execution_2d::Render2dVectorError;
    let mut mask_width = 0;
    let mut mask_height = 0;
    for item in &ordered {
        if let OrderedItem::Vector(mesh) = item {
            let fail = |kind| super::vector::error(mesh.root_index, kind);
            if !target.coverage_format {
                return Err(fail(Render2dVectorError::CoverageFormatUnsupported));
            }
            let width = mesh.bounds[2]
                .checked_mul(crate::runtime::program::abi::VECTOR_COVERAGE_AXIS_SAMPLES)
                .ok_or_else(|| fail(Render2dVectorError::ResourceLimit))?;
            let height = mesh.bounds[3]
                .checked_mul(crate::runtime::program::abi::VECTOR_COVERAGE_AXIS_SAMPLES)
                .ok_or_else(|| fail(Render2dVectorError::ResourceLimit))?;
            mask_width = mask_width.max(width);
            mask_height = mask_height.max(height);
            if mask_width > target.max_texture_dimension_2d
                || mask_height > target.max_texture_dimension_2d
                || u64::from(mask_width) * u64::from(mask_height) * 4 > 64 * 1024 * 1024
            {
                return Err(fail(Render2dVectorError::ResourceLimit));
            }
        }
    }
    let mut resources = GpuResourceScope::new();
    let mask = if mask_width > 0 && mask_height > 0 {
        let texture = resources
            .texture(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "runen-render vector union coverage",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    mask_width,
                    mask_height,
                    FIELD_FORMAT,
                    [GpuTextureUsage::ColorAttachment, GpuTextureUsage::Sampled],
                    GpuTextureInitialization::Uninitialized,
                )
                .map_err(|e| gpu("coverage texture descriptor", e))?,
            )
            .map_err(|e| gpu("coverage texture", e))?;
        Some(
            resources
                .texture_view(
                    GpuTextureViewDescriptor::ordinary_full_owned(
                        "runen-render vector union coverage view",
                        &texture,
                    )
                    .map_err(|e| gpu("coverage view descriptor", e))?,
                )
                .map_err(|e| gpu("coverage view", e))?,
        )
    } else {
        None
    };
    let mut clip_views = BTreeMap::new();
    for (root_index, mask) in &clipped {
        clip_views.insert(*root_index, clip::upload(target, mask, &mut resources)?);
    }
    let mut operations = Vec::new();
    let mut glyphs = Vec::new();
    let mut glyph_root: Option<usize> = None;
    let mut image_views = BTreeMap::new();
    let mut image_bytes: u64 = 0;
    for item in ordered {
        match item {
            OrderedItem::Glyph(glyph) => {
                if glyph_root != Some(glyph.root_index) && !glyphs.is_empty() {
                    operations.extend(lower(
                        target,
                        &glyphs,
                        glyph_root.and_then(|root| clip_views.get(&root)),
                    )?);
                    glyphs.clear();
                }
                glyph_root = Some(glyph.root_index);
                glyphs.push(glyph);
            }
            OrderedItem::Vector(mesh) => {
                operations.extend(lower(
                    target,
                    &glyphs,
                    glyph_root.and_then(|root| clip_views.get(&root)),
                )?);
                glyphs.clear();
                glyph_root = None;
                operations.extend(vector::lower(
                    target,
                    &mesh,
                    mask.as_ref().expect("vector mask"),
                    [mask_width, mask_height],
                    clip_views.get(&mesh.root_index),
                    &mut resources,
                )?);
            }
            OrderedItem::Image(patch) => {
                operations.extend(lower(
                    target,
                    &glyphs,
                    glyph_root.and_then(|root| clip_views.get(&root)),
                )?);
                glyphs.clear();
                glyph_root = None;
                if !target.image_format {
                    return Err(super::image::failure(
                        patch.root_index,
                        crate::execution_2d::Render2dImageError::FormatUnsupported,
                    ));
                }
                let vacant = image_views.entry(patch.resource_id);
                if let std::collections::btree_map::Entry::Vacant(entry) = vacant {
                    let bytes = patch.source.rgba8_srgb().len() as u64;
                    image_bytes = image_bytes.checked_add(bytes).ok_or_else(|| {
                        super::image::failure(
                            patch.root_index,
                            crate::execution_2d::Render2dImageError::ResourceLimit,
                        )
                    })?;
                    if image_bytes > 128 * 1024 * 1024 {
                        return Err(super::image::failure(
                            patch.root_index,
                            crate::execution_2d::Render2dImageError::ResourceLimit,
                        ));
                    }
                    let view = image::upload(target, &patch, &mut resources)?;
                    entry.insert(view);
                }
                operations.push(image::lower(
                    target,
                    &patch,
                    image_views.get(&patch.resource_id).expect("uploaded image"),
                    clip_views.get(&patch.root_index),
                    &mut resources,
                )?);
            }
        }
    }
    operations.extend(lower(
        target,
        &glyphs,
        glyph_root.and_then(|root| clip_views.get(&root)),
    )?);
    Ok(operations)
}

#[cfg(test)]
mod sample_space_gpu_proof {
    //! GPU-required hardware admission of the exact F3E sample-plane resolve.
    //! This is a physical prerequisite, not an assertion that group lowering
    //! is already implemented. The reference is independently derived from
    //! sample locations and premultiplied source-over.
    use super::*;
    use runen_gpu::{
        GpuBackendFamily, GpuCapabilityProfile, GpuContextDescriptor, GpuFormatRole,
        GpuReadbackOperation, GpuReadbackStatus, GpuSoftwareFallbackPolicy, GpuSubmissionStatus,
        GpuTextureCopyRegion, GpuWorkFragment,
    };
    use std::time::{Duration, Instant};

    fn pipeline(
        target_format: GpuTextureFormat,
        entry: &str,
        blend: bool,
    ) -> GpuRenderPipelineDescriptor {
        let source = crate::runtime::program::retained_vector_source()
            .expect("maintained vector shader source");
        let vertex = GpuEntryPointName::new("vs_main").unwrap();
        let fragment = GpuEntryPointName::new(entry).unwrap();
        let refinements = if entry == "fs_sample_resolve" {
            vec![
                GpuBindingLayoutRefinement::new(GpuBindingKey::try_new(0, 6).unwrap())
                    .with_texture_sample_class(GpuTextureSampleClass::FloatUnfilterable),
            ]
        } else {
            Vec::new()
        };
        let program =
            GpuProgramDescriptor::new(source, [vertex.clone(), fragment.clone()], refinements)
                .unwrap();
        let vertex_layout = GpuVertexBufferLayoutDescriptor::new(
            0,
            VERTEX_STRIDE,
            GpuVertexStepMode::Vertex,
            [
                GpuVertexAttribute::new(0, 0, GpuVertexFormat::Float32x2),
                GpuVertexAttribute::new(1, 8, GpuVertexFormat::Float32x2),
                GpuVertexAttribute::new(2, 16, GpuVertexFormat::Float32x4),
            ],
        )
        .unwrap();
        let blend = blend.then(|| {
            let component = GpuBlendComponent::new(
                GpuBlendFactor::One,
                GpuBlendFactor::OneMinusSrcAlpha,
                GpuBlendOperation::Add,
            )
            .unwrap();
            GpuBlendState::new(component, component)
        });
        let output =
            GpuColorTargetStateDescriptor::new(target_format, blend, GpuColorWriteMask::ALL)
                .unwrap();
        let state = GpuRenderPipelineStateDescriptor::new(
            GpuVertexInputStateDescriptor::new([vertex_layout]).unwrap(),
            Some(GpuFragmentOutputStateDescriptor::new([output])),
            GpuPrimitiveStateDescriptor::default(),
            None,
            GpuMultisampleStateDescriptor::default(),
        )
        .unwrap();
        GpuRenderPipelineDescriptor::new(
            program,
            GpuRenderEntryPoints::new(vertex, Some(fragment)),
            state,
            GpuPipelineConfiguration::default(),
        )
        .unwrap()
    }

    fn rectangle(vertices: &mut Vec<f32>, bounds: [f32; 4], extent: [f32; 2], color: [f32; 4]) {
        let [left, top, right, bottom] = bounds;
        for [x, y] in [
            [left, top],
            [right, top],
            [left, bottom],
            [left, bottom],
            [right, top],
            [right, bottom],
        ] {
            vertices.extend([
                x / extent[0] * 2.0 - 1.0,
                1.0 - y / extent[1] * 2.0,
                0.0,
                0.0,
                color[0],
                color[1],
                color[2],
                color[3],
            ]);
        }
    }

    #[test]
    #[ignore = "GPU-required F3E proof: explicitly executed by the Vulkan workflow"]
    fn retained_gpu_plane_resolves_correlated_siblings_once_over_prior_target() {
        let mut request = GpuContextDescriptor::new(
            GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements(),
        )
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("F3E float sample-plane and final resolve proof");
        for (format, role) in [
            (
                GpuTextureFormat::Rgba8UnormSrgb,
                GpuFormatRole::ColorAttachment,
            ),
            (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::Blendable),
            (GpuTextureFormat::Rgba8UnormSrgb, GpuFormatRole::CopySource),
            (
                GpuTextureFormat::Rgba16Float,
                GpuFormatRole::ColorAttachment,
            ),
            (GpuTextureFormat::Rgba16Float, GpuFormatRole::Blendable),
            (GpuTextureFormat::Rgba16Float, GpuFormatRole::Sampled),
        ] {
            request = request.require_format_role(format, role);
        }
        let context = pollster::block_on(GpuContext::request(request))
            .expect("F3E GPU-only proof requires admitted Lavapipe Vulkan and Rgba16Float roles");

        let mut resources = GpuResourceScope::new();
        let sample = resources
            .texture(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "F3E sample layer",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    256,
                    256,
                    GpuTextureFormat::Rgba16Float,
                    [GpuTextureUsage::ColorAttachment, GpuTextureUsage::Sampled],
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let sample_view = resources
            .texture_view(
                GpuTextureViewDescriptor::ordinary_full_owned("F3E sample view", &sample).unwrap(),
            )
            .unwrap();
        let output = resources
            .texture(
                GpuTextureDescriptor::ordinary_owned_2d(
                    "F3E one-time resolved output",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    64,
                    64,
                    GpuTextureFormat::Rgba8UnormSrgb,
                    [
                        GpuTextureUsage::ColorAttachment,
                        GpuTextureUsage::CopySource,
                    ],
                    GpuTextureInitialization::Zeroed,
                )
                .unwrap(),
            )
            .unwrap();
        let output_view = resources
            .texture_view(
                GpuTextureViewDescriptor::ordinary_full_owned("F3E output view", &output).unwrap(),
            )
            .unwrap();

        let mut sample_vertices = Vec::new();
        // Red covers precisely 8 of the 16 logical pixel samples, and blue
        // covers the other 8. Per-item pixel averaging would give alpha .75,
        // whereas one coherent sample space has alpha 1.0 before the resolve.
        rectangle(
            &mut sample_vertices,
            [40.0, 80.0, 42.0, 84.0],
            [256.0, 256.0],
            [1.0, 0.0, 0.0, 1.0],
        );
        rectangle(
            &mut sample_vertices,
            [42.0, 80.0, 44.0, 84.0],
            [256.0, 256.0],
            [0.0, 0.0, 1.0, 1.0],
        );
        let sample_pipeline = pipeline(GpuTextureFormat::Rgba16Float, "fs_sample_fill", true);
        let sample_draw = vector::vector_draw(
            sample_pipeline.clone(),
            sample_pipeline.runtime_bindings([]).unwrap(),
            &sample_vertices,
            [256, 256],
            &mut resources,
        )
        .unwrap();
        let samples = GpuRenderOperation::new(
            [GpuRenderColorAttachment::new(
                sample_view.clone(),
                GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 0.0).unwrap()),
                GpuAttachmentStore::Store,
                None,
            )
            .unwrap()],
            None,
            [sample_draw],
            None,
        )
        .unwrap();

        let resolve_pipeline =
            pipeline(GpuTextureFormat::Rgba8UnormSrgb, "fs_sample_resolve", true);
        let mut output_vertices = Vec::new();
        rectangle(
            &mut output_vertices,
            [0.0, 0.0, 64.0, 64.0],
            [64.0, 64.0],
            [1.0, 1.0, 1.0, 1.0],
        );
        let resolve_draw = vector::vector_draw(
            resolve_pipeline.clone(),
            resolve_pipeline
                .runtime_bindings([texture_binding(6, &sample_view).unwrap()])
                .unwrap(),
            &output_vertices,
            [64, 64],
            &mut resources,
        )
        .unwrap();
        let clear = GpuRenderOperation::new(
            [GpuRenderColorAttachment::new(
                output_view.clone(),
                GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 1.0, 1.0).unwrap()),
                GpuAttachmentStore::Store,
                None,
            )
            .unwrap()],
            None,
            [],
            None,
        )
        .unwrap();
        let resolve = GpuRenderOperation::new(
            [GpuRenderColorAttachment::new(
                output_view,
                GpuColorAttachmentLoad::Load,
                GpuAttachmentStore::Store,
                None,
            )
            .unwrap()],
            None,
            [resolve_draw],
            None,
        )
        .unwrap();
        let readback = GpuReadbackOperation::ordinary(
            GpuTextureCopyRegion::whole_base_mip(&output)
                .unwrap()
                .into(),
        )
        .unwrap();
        let readback_id = readback.id();
        let fragment = GpuWorkFragment::build("F3E float sample source and resolve", |builder| {
            builder.operation("caller prior blue", clear)?;
            builder.operation("16 sample premultiplied colors", samples)?;
            builder.operation("single physical pixel resolve", resolve)?;
            builder.operation("terminal output readback", readback)?;
            Ok(())
        })
        .unwrap();
        let graph = GpuPreparedWorkGraph::prepare(
            GpuResourceLabel::new("F3E correlated GPU sample proof").unwrap(),
            [fragment],
        )
        .unwrap();
        let submission = context
            .submit_prepared(pollster::block_on(context.prepare_submission(graph)).unwrap())
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(45);
        let bytes = loop {
            context.progress();
            match submission.readback(readback_id).unwrap().status() {
                GpuReadbackStatus::Ready(bytes)
                    if submission.status() == GpuSubmissionStatus::Completed =>
                {
                    break bytes;
                }
                GpuReadbackStatus::Failed(problem) => panic!("F3E readback: {problem:?}"),
                _ => {
                    assert!(Instant::now() < deadline, "F3E GPU submission timed out");
                    std::thread::yield_now();
                }
            }
        };
        let pixel = |x: usize, y: usize| {
            let offset = (y * 64 + x) * 4;
            &bytes.as_bytes()[offset..offset + 4]
        };
        // Linear 0.5 is sRGB 188. The last 0.5 blue comes from the
        // *composed sample plane*, not from partially obscured caller blue.
        for (actual, expected) in pixel(10, 20).iter().zip([188_u8, 0, 188, 255]) {
            assert!(
                actual.abs_diff(expected) <= 4,
                "correlated color {actual}, expected {expected}"
            );
        }
        assert_eq!(pixel(11, 20), &[0, 0, 255, 255]);
    }
}
