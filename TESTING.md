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
- public semantic modules plus private runtime/proof ownership boundaries;
- no `Deterministic*` vocabulary in the package-level ordinary public consumer;
- no Runenwerk/App/ECS/Winit/World/UI/Editor/product env/fs/JSON/private-WGPU
  coupling in production source;
- presence of the dedicated Vulkan CI enforcement lane;
- rustfmt;
- locked workspace tests;
- strict Clippy;
- rustdoc with warnings denied;
- explicit Rust 1.97.1 workspace/all-targets check;
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

## Accepted standalone baseline evidence

The completed RX authority transition established the initial standalone
baseline through unchanged reviewed-head and accepted-main evidence. The
retained proof surface establishes:

- package-level ordinary public conformance;
- exact three-effective-program RunenShader artifact -> RunenGPU admission;
- successful maintained-program compilation/admission retained across both
  one-shot and stateful renderer resource-cache lifetimes rather than per-frame
  compilation;
- headless/offscreen maintained execution;
- same-submission semantic result formation and requested tolerance;
- ordinary submission with no implicit CPU readback;
- temporal reconstruction, camera-history, field, and requested-lattice behavior;
- no proof-private camera diagnostic seam promoted into public API;
- exact frozen transfer/adaptation/provenance evidence;
- no predecessor mirror/forwarder/private sibling reach-through.

These are accepted baseline properties, not a claim that one physical
implementation layout is permanently frozen. The post-RX structural program
replaced transfer-era path authority with durable source-boundary guards,
grouped private maintained realization under `runtime`, grouped conformance
proofs under `src/proofs`, made the host/WGSL ABI explicit, decomposed
execution and the ordinary façade by responsibility, and isolated persistent
scene storage behind `scene::storage`.

Future changes must preserve or deliberately replace the relevant evidence
under their owning issue and must be validated on the exact reviewed head.

Runenwerk downstream native/product integration remains downstream consumer
evidence. Its predecessor cutover is complete and is no longer an activation
gate for ordinary standalone RunenRender evolution.
