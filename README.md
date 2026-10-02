# RunenRender

RunenRender is a standalone Rust framework for reusable rendering semantics,
maintained image formation, semantic result formation, and renderer-owned
composition of RunenShader artifacts into RunenGPU program execution.

## Maturity

This repository is in the RX standalone successor-candidate phase. The current
transfer branch physically contains the frozen RunenRender implementation and
standalone conformance candidate, but that physical copy is **not** accepted
semantic authority yet.

Runenwerk remains the sole accepted RunenRender semantic source authority until
the fully validated successor candidate is accepted on `runen-render/main`
under Engineering ADR 0008. That accepted merge is the authority switch.

## Boundary

RunenRender will own reusable renderer semantics, semantic scene/request/
representation/input contracts, maintained renderer execution, temporal/camera
rendering evidence, semantic result formation, and the explicit renderer-owned
bridge from accepted RunenShader artifacts into public RunenGPU program
admission.

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

The current RX MSRV is the supported repository floor. Source-free bootstrap
initially selected Rust 1.93.0, but executable transfer validation proved that
the accepted RunenShader dependency graph requires Rust 1.97.1 through its exact
WESL 0.5.0 graph. The candidate therefore raises the floor to 1.97.1 rather than
advertising an unsupported lower compiler.

## Validation

`cargo validate` is the canonical repository-owned baseline. It now proves
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
