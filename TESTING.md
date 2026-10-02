# Testing and validation

## Canonical baseline

```text
cargo validate
```

Repository-local `xtask` owns the merge-readiness baseline. It proves:

- a clean starting repository and required authority/source files;
- RunenRender package/repository/version/license/MSRV identity;
- complete GPLv3 license text;
- exact immutable RunenGPU and RunenShader dependency revisions;
- no moving sibling branch/tag dependency;
- public semantic modules plus private maintained/deterministic implementation topology;
- no `Deterministic*` vocabulary in the package-level ordinary public consumer;
- no Runenwerk/App/ECS/Winit/World/UI/Editor/product env/fs/JSON/private-WGPU
  coupling in production source;
- presence of the dedicated Vulkan CI enforcement lane;
- rustfmt;
- locked workspace tests;
- strict Clippy;
- rustdoc with warnings denied;
- explicit Rust 1.93.0 workspace/all-targets check;
- Git whitespace checks;
- validation not mutating repository state.

## CI baseline

`.github/workflows/validation.yml` keeps the repository baseline thin. The
`Validate RunenRender / Repository baseline` check pins the accepted
`dornglut/github-workflows` reusable Rust validation workflow to immutable
revision `688f274ec1fdd19acba9bf3577b26b4b7b7f4037`.

The shared workflow proves the exact caller revision and provisions stable plus
Cargo-declared `rust-version` toolchains.

## RunenRender Vulkan conformance

The same workflow also owns a separate exact-revision
`RunenRender Vulkan conformance` job.

That job:

- checks out the exact pull-request head or accepted-main revision;
- installs Mesa's Lavapipe Vulkan implementation on Ubuntu;
- pins WGPU to Vulkan through the discovered Lavapipe ICD;
- proves the software Vulkan adapter with `vulkaninfo --summary`;
- sets `RUNEN_RENDER_REQUIRE_GPU=1`;
- runs the standalone `runen-render` package tests serially.

The environment variable is test-only enforcement. It converts
`NoAdapterAvailable` from an allowed local skip into CI failure; production
RunenRender does not read environment variables.

This lane owns reusable framework evidence only. It deliberately excludes
Runenwerk native-window, Present, Render Lab image/JSON artifact, product
surface-routing, and product diagnostic policy.

## RX acceptance evidence

Before the current successor candidate may be accepted, the unchanged reviewed
head must additionally establish through tests/cold review:

- package-level ordinary public conformance;
- exact three-effective-program RunenShader artifact -> RunenGPU admission;
- renderer-lifetime retained program admission rather than per-frame compilation;
- headless/offscreen maintained execution;
- same-submission semantic result formation and requested tolerance;
- ordinary submission with no implicit CPU readback;
- temporal reconstruction, camera-history, field, and requested-lattice behavior;
- no proof-private camera diagnostic seam promoted into public API;
- exact frozen transfer/adaptation/provenance ledger;
- no predecessor mirror/forwarder/private sibling reach-through.

Accepted-main validation is rechecked after the authority-switch merge. Runenwerk
downstream cutover/deletion evidence remains owned by Runenwerk #1134.
