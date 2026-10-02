//! Framework composition proof.
//!
//! This proves only reusable RunenRender -> RunenGPU producer/import semantics. Runenwerk
//! Render-Lab visualization, surface routing, and Present policy remain downstream product tests.

use crate::admission::{RenderOutputBinding, RenderOutputDestination};
use crate::deterministic_admission::admit_deterministic_render_with_semantic_inputs;
use crate::deterministic_execution::prepare_deterministic_render;
use super::execution::{
    MaintainedExecutionFixture, admit_with_retained_radiance_destination, maintained_fixture,
};
use crate::request::{
    RenderOutputSpec, RenderOutputValue, RenderRadiometricRepresentation, RenderRequest,
    RenderRequestedOutput, RenderResultTopology, RenderSemanticTolerance,
};
use runen_gpu::{
    GpuBufferDescriptor, GpuBufferInitialization, GpuBufferTextureLayout, GpuBufferUsage,
    GpuCapabilityProfile, GpuContext, GpuContextDescriptor, GpuContextRequestErrorCategory,
    GpuCopyOperation, GpuDependencyReason, GpuDependencyRegion, GpuFormatRole,
    GpuPreparedWorkGraph, GpuReconstruction, GpuResourceLabel, GpuResourceLifetime,
    GpuResourceProvenance, GpuTextureCopyRegion, GpuTextureDescriptor, GpuTextureFormat,
    GpuTextureInitialization, GpuTextureUsage, GpuWorkFragment, GpuWorkNodeKind,
    GpuWorkResourceIdAllocator,
};

fn context() -> Option<GpuContext> {
    let descriptor =
        GpuContextDescriptor::new(GpuCapabilityProfile::ComputeBaseline.requirements())
            .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopyDestination)
            .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopySource)
            .with_label("RunenRender standalone R7 composition proof");
    match pollster::block_on(GpuContext::request(descriptor)) {
        Ok(context) => Some(context),
        Err(error) if error.category() == GpuContextRequestErrorCategory::NoAdapterAvailable => {
            assert_ne!(
                std::env::var("RUNEN_RENDER_REQUIRE_GPU").ok().as_deref(),
                Some("1"),
                "RunenRender composition CI requires a public RunenGPU adapter"
            );
            None
        }
        Err(error) => panic!("unexpected RunenGPU context failure: {error}"),
    }
}

fn radiance_fixture() -> MaintainedExecutionFixture {
    let mut fixture = maintained_fixture();
    fixture.request = RenderRequest::new(
        fixture.request.render_interval(),
        fixture.request.observations().to_vec(),
        vec![RenderRequestedOutput::new(
            0,
            RenderOutputSpec::new(
                RenderOutputValue::Radiance {
                    representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                        550.0e-9,
                    )
                    .expect("R7 composition radiance representation"),
                },
                RenderResultTopology::sample_lattice_2d(2, 2)
                    .expect("R7 composition radiance topology"),
                RenderSemanticTolerance::exact(),
            )
            .expect("R7 composition radiance output"),
        )],
    )
    .expect("R7 composition radiance request");
    fixture
}

fn r32float_radiance_destination(
    allocator: &mut GpuWorkResourceIdAllocator,
) -> runen_gpu::GpuTextureHandle {
    allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::ordinary_owned_2d(
                "standalone R7 composition radiance destination",
                GpuResourceLifetime::Retained,
                GpuReconstruction::SourceBacked,
                2,
                2,
                GpuTextureFormat::R32Float,
                [
                    GpuTextureUsage::CopyDestination,
                    GpuTextureUsage::CopySource,
                    GpuTextureUsage::Sampled,
                ],
                GpuTextureInitialization::Zeroed,
            )
            .expect("R7 composition destination descriptor"),
        )
        .expect("R7 composition destination handle")
}

fn admit_r32float_radiance(
    fixture: &MaintainedExecutionFixture,
    context: &GpuContext,
) -> crate::deterministic_admission::AdmittedDeterministicRender {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let destination = r32float_radiance_destination(&mut allocator);
    let output_bindings = [RenderOutputBinding::new(
        0,
        RenderOutputDestination::SampleLatticeTexture(destination),
    )];
    admit_deterministic_render_with_semantic_inputs(
        &fixture.scene,
        &fixture.request,
        &fixture.semantic_inputs,
        &[],
        &fixture.availability,
        &output_bindings,
        context,
    )
    .expect("R7 composition radiance must admit")
}

#[test]
fn maintained_radiance_is_a_typed_producer_for_consumer_first_graphs() {
    let Some(context) = context() else {
        return;
    };
    let fixture = radiance_fixture();

    let legacy = prepare_deterministic_render(
        admit_with_retained_radiance_destination(&fixture, &context),
        &context,
    )
    .expect("legacy R32Uint radiance preparation");
    assert!(
        legacy.radiance_outputs().is_empty(),
        "legacy capture-compatible R32Uint radiance is not a composable output"
    );

    let prepared =
        prepare_deterministic_render(admit_r32float_radiance(&fixture, &context), &context)
            .expect("R7 composition preparation");
    assert_eq!(prepared.radiance_outputs().len(), 1);
    assert!(
        prepared
            .work_set()
            .fragments()
            .iter()
            .flat_map(|fragment| fragment.nodes())
            .all(|node| node.kind() != GpuWorkNodeKind::Readback),
        "ordinary composable preparation must remain CPU-readback-free"
    );

    let output = prepared.radiance_output(0).expect("radiance correlation");
    let texture = output.texture().expect("radiance texture").clone();
    assert_eq!(texture.descriptor().format(), GpuTextureFormat::R32Float);

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let consumer_buffer = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::ordinary_owned(
                "standalone R7 composition consumer",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                16,
                [GpuBufferUsage::CopyDestination],
                GpuBufferInitialization::Uninitialized,
            )
            .expect("consumer buffer descriptor"),
        )
        .expect("consumer buffer");
    let copy = GpuCopyOperation::texture_to_buffer(
        GpuTextureCopyRegion::whole_base_mip(&texture).expect("radiance source region"),
        GpuBufferTextureLayout::new(&consumer_buffer, 0, 8, 2).expect("consumer layout"),
    )
    .expect("consumer copy");
    let consumer_provenance = GpuResourceProvenance::new(
        GpuResourceLabel::new("standalone R7 radiance consumer").expect("consumer label"),
        None,
        None,
    );
    let consumer = GpuWorkFragment::build_with_provenance(
        GpuResourceLabel::new("standalone R7 radiance consumer").expect("consumer label"),
        consumer_provenance.clone(),
        |work| {
            work.operation("consume maintained radiance", copy)?;
            work.add_import(output.import(consumer_provenance))?;
            Ok(())
        },
    )
    .expect("consumer fragment");

    let producer = prepared.work_set().fragments()[0].clone();
    let graph = GpuPreparedWorkGraph::prepare(
        GpuResourceLabel::new("standalone R7 consumer-first composition").expect("graph label"),
        [consumer, producer],
    )
    .expect("typed consumer-first composition graph");

    assert!(
        graph
            .dependencies()
            .iter()
            .flat_map(|dependency| dependency.reasons())
            .any(|reason| {
                matches!(
                    reason,
                    GpuDependencyReason::ReadAfterWrite { resource, region }
                        if *resource == output.resource().diagnostic_identity()
                            && *region == GpuDependencyRegion::Texture(
                                GpuTextureCopyRegion::whole_base_mip(&texture)
                                    .expect("dependency region")
                                    .subresources()
                            )
                )
            }),
        "typed import must create the producer -> consumer dependency"
    );
    assert_eq!(
        graph.topological_order()[0].fragment_ordinal(),
        1,
        "consumer-first authoring must still execute the producer first"
    );
}
