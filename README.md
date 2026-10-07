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

## Runnable headless scene-inspector example

The public-only [headless scene inspector](examples/headless_scene.rs) constructs a
renderer-owned sphere and plane, one perspective observation, and two correlated
outputs: **spectral radiance at 550 nm** and **object identity**. It does not
need Runenwerk, an ECS, native windows, or Present.

Run without writing files, or opt into locally inspectable diagnostic artifacts:

```sh
cargo run --example headless_scene
cargo run --example headless_scene -- --output ./runen-render-inspection
```

The second command writes:

```text
runen-render-inspection/frame_001/
├── radiance.png
├── object_ids.png
└── evidence.json
```

The radiance PNG applies a fixed 0.25 exposure to a *single* 550 nm
spectral-radiance sample lattice and displays the result in grayscale; **it
is not RGB rendering**. The object-ID PNG assigns fixed diagnostic colors
to this fixture's decoded sphere and plane identities. Black denotes a
sample without a decoded identity, not a semantic "background entity".
Neither PNG is itself a canonical renderer-semantic output.

Decoded physical words alone do not establish per-pixel semantic definedness or
miss reasons. Undefined payload may be arbitrary, so the number of undecoded
words is a diagnostic count rather than a background mask or correctness oracle.

`evidence.json` records semantic output indices, scene revision, topology,
the selected adapter backend, visualization conventions, and verified
execution/readback outcomes. The example explicitly checks admitted output
indices against their physical destinations; the API's residual positional
correlation remains visible.
Image and JSON encoding belong to the executable example, **not** to
RunenRender's production rendering API. Object and representation IDs,
output indices, two-phase scene assembly, and the separate verified readback
remain explicit: this example documents the accepted existing API rather
than fabricating future convenience builders.

The GPU-required Vulkan CI lane also executes the example's public-API
conformance test. Without a suitable GPU, the executable reports the
missing adapter rather than pretending a render succeeded.

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
