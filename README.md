# RunenRender

RunenRender is a standalone Rust framework for reusable rendering semantics,
maintained image formation, semantic result formation, and renderer-owned
composition of RunenShader artifacts into RunenGPU program execution.

## Maturity

RunenRender is now the accepted standalone semantic and conformance authority
for its renderer domain. The Engineering ADR 0008 RX transfer and the
exact-revision Runenwerk predecessor cutover are complete.

The accepted transfer established the initial standalone `0.1.0` semantic
baseline. Ordinary framework evolution now belongs in this repository under
current accepted RunenRender issues; transfer-era issues and predecessor source
remain historical provenance rather than active implementation authority.

Pre-1.0 status does not imply that every current internal decomposition is a
permanent API or architecture commitment. Public semantic changes still require
explicit accepted authority and compatibility review.

## Boundary

The standalone boundary assigns RunenRender ownership of reusable renderer
semantics, semantic scene/request/representation/input contracts, maintained
renderer execution, temporal/camera rendering evidence, semantic result
formation, and the explicit renderer-owned bridge from accepted RunenShader
artifacts into public RunenGPU program admission.

RunenRender does not own:

- RunenShader source/compilation/artifact semantics;
- generic RunenGPU resource/work/submission/backend semantics;
- Runenwerk App/ECS/Winit/native-host lifecycle;
- Runenwerk World/UI/Editor adapters;
- product frame/presentation scheduling and final Present policy;
- Render Lab product/window/oracle policy;
- product filesystem/JSON/image/video artifact persistence.

## Package

```text
package: runen-render
crate: runen_render
version: 0.1.0
edition: 2024
MSRV: 1.97.1
publish: false
```

The supported repository floor is Rust 1.97.1. Source-free bootstrap initially
selected Rust 1.93.0, but executable transfer validation proved that the
accepted RunenShader dependency graph requires Rust 1.97.1 through its exact
WESL 0.5.0 graph.

## 2D composition execution

The public `composition_2d` module expresses one immutable source-neutral
painter tree of geometry, solid and gradient paint, image patches, already-shaped
text, conjunctive parent-space clips and atomic nested groups. The public
`execution_2d` executor admits compatible immutable resource bindings and a
caller-owned color target, then returns composable RunenGPU work and exact
completion evidence. All admitted root and nested content uses the same
correlated 4×4 sample-space color law with once-only item and group opacity.
Shaped text remains limited to the supported monochrome outline/
identity-or-translation class. Ordinary shadows/effects are not yet rendered,
and native presentation remains with the host.

The renderer's private realization uses bounded tiles and reconstructible
field caches; semantic resource identities remain immutable across cache loss.
The separate long-lived identity retirement contract is tracked under #72.

## Start with the ordinary API

The compact public-only [ordinary example](examples/ordinary_render.rs) builds a
scene with named representation/participation construction, retains the
request-owned observation and output handles, and binds two distinguishable
outputs: spectral radiance at 550 nm and object identity. It carries those
original handles through late physical binding, admission, prepared-output
lookup and associated capture/decoding. Reversed physical binding order shows
that output positions are not identity.

```sh
cargo run --example ordinary_render --locked
cargo doc --open --no-deps --locked
```

The example prepares renderer work through a `RenderExecutionSession`, submits
it explicitly through caller-owned RunenGPU, associates the exact occurrence,
waits for bounded terminal completion, reconciles the session, and observes
outputs through a separate caller-owned readback. It requires a GPU adapter;
failures propagate from `main`. It has no file/artifact, ECS or Present policy.
The repository's Vulkan lane executes its actual `main`.

The crate front page includes a GPU-independent compiling construction example
and navigation to the ordinary and advanced APIs. Use
`RenderRepresentationRecord::builder(id, spatial_coverage, temporal_support)`
with named `surface_query`, `field_distance` and optional `refinement`, then
fallible `build()`. Spatial/temporal meaning is always explicit; default
refinement means no evidence, not exactness. Named optional channels accept
typed `Option` values, including dynamic source-adapter facts.

Use `RenderObjectParticipation::from_representations(collection_or_iterator)?`
then `with_material_assignment` / `with_emitter` as needed. Representation order
is canonical and duplicate IDs fail with the existing typed error. Empty,
material-only and emitter-only participation are legal; publishing empty
participation clears the scene facet. Object insertion and participation
attachment remain separate transactions; atomic multi-facet replacement applies
to existing objects.

For one-shot work, use `prepare_render` or `submit_render`. For certified
semantic result formation, select `submit_render_for_result` before submission
and poll `try_form_result` after progressing RunenGPU. Retained physical capture
and execution-local identity decoding are distinct from `RenderResult`
certification and semantic definedness.

## Retained headless scene-inspector example

The [headless scene inspector](examples/headless_scene.rs) exercises retained ordinary rendering across two scene revisions with exact output/submission correlation. Optional diagnostic artifacts are executable-local policy, not framework persistence or Present behavior.

```sh
cargo run --example headless_scene --locked
cargo run --example headless_scene --locked -- --output ./runen-render-inspection
```

See the [walkthrough](docs/examples/headless-scene.md) for output meanings, temporal invalidation, lifecycle requirements and artifact layout.

## Validation

`cargo validate` is the canonical repository-owned baseline. It proves
package/profile integrity, exact sibling dependency policy, public-surface
guards, production-boundary residue, compile/test/Clippy/rustdoc/MSRV, and
clean-tree invariants.

Pull-request and accepted-main CI additionally runs a dedicated headless Vulkan
conformance lane with GPU availability required through Mesa Lavapipe. See
[TESTING.md](TESTING.md).

## Authority and policy

- [Architecture](ARCHITECTURE.md)
- [Testing](TESTING.md)
- [Bootstrap and provenance](BOOTSTRAP.md)
- [Executor guidance](AGENTS.md)
- [Public license](LICENSE)
- [Licensing and historical provenance](LICENSING.md)
- [Organization contribution guidance](https://github.com/dornglut/.github/blob/main/CONTRIBUTING.md)
- [Organization security policy](https://github.com/dornglut/.github/blob/main/SECURITY.md)

## Contribution

Tracked-content contributions are currently `owner-only` until an accepted
inbound mechanism preserves the rights required by the public/commercial
licensing model.
