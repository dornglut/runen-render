# Testing and validation

## Canonical baseline

```text
cargo validate
```

Repository-local `xtask` owns the merge-readiness baseline. It proves:

- a clean starting repository and required authority/source files;
- RunenRender package/repository/version/license/MSRV identity;
- complete GPLv3 license text;
- exact immutable RunenGPU and RunenShader dependency revisions plus exact
  private shaped-text and vector realization dependency declarations;
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
- runs the standalone `runen-render` package tests serially;
- retains the comprehensive `headless_scene` two-frame conformance path;
- executes `cargo +stable run --example ordinary_render --locked`, including its
  actual `main`, two output meanings, original request-owned handles, reversed
  physical bindings, exact occurrence association, completion and readback.
  Adapter unavailability fails the executable; this path never silently skips;
- runs the external F2 2D execution consumer proof with the Vulkan adapter
  required, including downstream graph composition, exact work-node completion
  evidence, shaped-outline readback, cache reconstruction, non-painting glyphs,
  and fail-closed COLR v0/v1, SVG, bitmap, and faux-bold cases.

F2 proofs also cover caller-owned clear -> appended F2 -> readback ordering,
typed prior-target imports independent of fragment array order, rejection of
unrelated/in-flight submission evidence, transactional failed preparation,
fresh-device reconstruction, exact raster scale/translation/continuous-canvas
clipping, and malformed intrinsic tables or glyph IDs failing closed. CPU
field proofs establish deterministic reconstruction and later texture-limit
revalidation without requiring an adapter.

The same external 2D execution suite includes solid and gradient vector output proofs:
structural shapes and curved paths, an independent winding oracle for both fill
rules, caps/joins/miter fallback, an independent distance-to-segment sample-union
oracle for translucent crossing strokes, linear-light alpha and item opacity,
vector/text interleaving, affine and raster-scale mapping, fractional target
coverage, caller clear/import ordering, no-work, unsupported classes, late
preparation rejection and cache/device reconstruction. A private GPU proof
rejects terminal subsets of a required multi-node contribution.

Private vector coverage uses sixteen regular samples per output pixel and a
shared cropped RGBA8 mask cleared for each item. Flattening targets 1/64 physical
pixel; geometry conversion fails closed when precision cannot support it.
Preprocessing/geometry are bounded to 1,048,576 elements; vertex storage respects
device/workload buffer limits; the mask respects texture dimensions and a 64 MiB
private allocation bound. These are replaceable physical policy, not semantic
geometry or public quality controls. Supersampling costs and precision/limit
rejections are supported limitations, not native performance claims.

F3B Vulkan-required tests independently assert premultiplied linear and radial
stop interpolation, hard-stop boundary colors, transformed brush coordinates,
per-sample translucent stroke coverage, mixed text/gradient order, retained
reconstruction and fail-closed resource/precision limits. Gradient stop buffers
are private bounded storage; no source-framework identity or per-frame public
brush state is introduced.

The environment variable enforces GPU-required tests. It converts
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
- exact maintained-program RunenShader artifact -> RunenGPU admission,
  including shaped text and solid vector coverage/composition;
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

The F2 execution consumer uses repository-owned tiny OpenType fixtures. Their
generator fixes FontTools 4.63.0 and asserts exact SHA-256 outputs so proof
content does not depend on host fonts or opaque downstream assets.

Future changes must preserve or deliberately replace the relevant evidence
under their owning issue and must be validated on the exact reviewed head.

Runenwerk downstream native/product integration remains downstream consumer
evidence. Its predecessor cutover is complete and is no longer an activation
gate for ordinary standalone RunenRender evolution.


F3C GPU-required public image tests exercise immutable source-neutral RGBA8
resource binding, exact continuous source/destination patches, private
nearest-texel sampling, affine geometry, four-by-four physical edge coverage,
linear-light premultiplication/source-over, patch and mixed painter ordering,
no-work outcomes, typed format/precision/resource failures and reconstruction
after cache discard. The image path must not depend on direct wgpu rendering.

F3D GPU-required public proof cases exercise cropped packed 16-bit conjunctive
item masks, parent-space clip transforms, rounded corners, radial gradients,
nonzero/even-odd winding paths, ellipses, image/text coverage and reconstruction,
fractional intersections that differ from products
of pixel-averaged clip alphas, typed unadmitted-format failure, empty intersection
no-work and no leakage across unclipped siblings. CPU mask work is deliberately
bounded; no native-GPU performance claim or persistent GPU mask cache follows.
