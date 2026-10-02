# RunenRender

RunenRender is a standalone Rust framework for reusable rendering semantics,
maintained image formation, semantic result formation, and renderer-owned
composition of RunenShader artifacts into RunenGPU program execution.

## Maturity

This repository is in RX bootstrap and extraction preparation. The repository
exists and owns its package/profile/validation surface, but it does **not** yet
contain the transferred RunenRender implementation.

Runenwerk remains the sole RunenRender semantic source authority until a later
validated successor candidate is accepted on `runen-render/main` under
Engineering ADR 0008.

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
MSRV: 1.93.0
publish: false
```

The initial MSRV is the supported repository floor. It does not claim that the
future transferred source cannot compile on an older compiler; a lower support
floor requires separate evidence and acceptance.

## Validation

`cargo validate` is the single repository-owned merge-readiness command. At
bootstrap it proves repository/package identity and profile integrity rather
than renderer implementation conformance.

See [TESTING.md](TESTING.md).

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
