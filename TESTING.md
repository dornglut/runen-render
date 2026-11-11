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

## Current execution proof coverage

These are **capability checks**, not implementation stages. Accepted standalone
behavior is tested on `main`; additional F3F cases below are **draft evidence**
until [#81](https://github.com/dornglut/runen-render/issues/81) is accepted.
A passing narrow proof or a typed unsupported result does not establish support
for the entire semantic class.

| Boundary | Evidence and source of tests |
| --- | --- |
| Ordinary maintained rendering | Package-level ordinary public API; RunenShader artifacts admitted through RunenGPU; one-shot and retained resource-cache lifetimes; headless/offscreen execution, request-owned results, no implicit CPU readback, same-submission formation, camera history and reconstruction (`src/proofs/`, `tests/ordinary_public_api.rs`, `examples/headless_scene.rs`, `examples/ordinary_render.rs`) |
| 2D vectors and shaped text | Public execution, winding/fill rules, centered strokes/caps/joins, raster scaling and affine coverage; stable tiny OpenType fixtures; immutable glyph identities/outlines, error and cache recovery (`tests/execution_2d_public_api.rs`, `tests/execution_2d/`) |
| Gradients and image patches | Linear-light gradient interpolation and hard stops; immutable RGBA8 source-to-destination patches, nearest texels, affine transforms, translucent painter order, cache reconstruction and invalid inputs (`tests/execution_2d/`) |
| Clips | Conjunctive item/group clips, shape/fill rules, rounded regions, transformed masks, subpixel intersections, empty/invalid masks, no sibling leakage and preallocation rejection (`tests/execution_2d/`) |
| Correlated group compositor | One 4×4 linear-premultiplied sample plane across vectors/images/text, nested group opacity once, exact painter ordering, source-over and final resolve over caller-owned prior color, fractional canvas and multiple tiles, no-work, resource identity, failure atomicity and completed GPU work (`tests/execution_2d_public_api.rs`, `src/runtime/execution_2d/lowering/sample_space.rs`) |
| Ordinary shadows — **draft** | Selected transparent-source, straight-boundary unions/holes, finite signed spread/Gaussian, nested ancestor charts, clips, tiles, color order and cache discard; independent Vulkan pixel readbacks and typed unsupported cases (`tests/execution_2d/shadow.rs`, `src/runtime/execution_2d/shadow/`) |

The completed standalone authority transfer is a provenance fact, **not** an
active migration gate. Preserve ordinary retained-program/cache and temporal
behavior, with no predecessor mirror, private sibling reach-through or
proof-only camera diagnostic promoted into the public API. Exact reviewed-head
and accepted-main validation remain necessary; these proofs do not freeze one
physical implementation layout. The F2 tiny-font generator pins FontTools
4.63.0 and verifies fixture SHA-256 hashes, independent of installed host fonts.

## F3F draft evidence and limitations

[PR #85](https://github.com/dornglut/runen-render/pull/85) extends the
existing compositor rather than introducing another painter or geometric
authority. Its registered Gaussian tests cover the **qualified**
straight-boundary original-source family: transparent holed fills or image
destination unions, negative child effects, consuming ancestor shear,
normalized finite Gaussian, independently checked 16-site/boundary pixels,
authored shadow order, group opacity, clipped and reflected parent charts,
cache reconstruction and tile/cutoff equivalence. Missing compute capability
fails before contribution creation; the executor remains reusable.
Supported parent-chart numerical selection is deliberately limited to proven
row-preserving cases, not arbitrary transforms.

Finite-cutoff tests additionally cover constant-coordinate separability,
tangent/reduced-degree cutoff equations, adjacent floating endpoints,
uncertain mass, empty runs and bounded numerical-record reuse. Root-event
reuse does not reuse original-region occupancy. The same-executor
touching-vs-microscopic-gap test renders the exact contact union and **rejects**
the positive-gap case as unsupported: typed rejection is **not** successful
shadow rendering.

### Resource and precision evidence

- The compiler admits aggregate clip/geometry/tile replay, 4×4 scratch-depth
  and peak allocations, text MSDF fields, image/gradient payloads, work-node
  counts and preflight failures before appending caller-owned RunenGPU work.
  Private budgets are **physical policies**, not F1 source meaning or native
  frame-time guarantees; safe retained-field residency does not imply a
  persistent shadow-mask or native GPU performance promise.
- Draft F3F source construction, original-region numerical work, affine
  planning and physical setup share one CPU invocation budget. Encoded GPU
  lookup visits and bounded Q48 arithmetic use the **separate** aggregate
  tile-work budget; a GPU operation must not be double-counted as a CPU loop.
  Immutable uploads and retained fields coexist in host accounting, and
  numerical tile lookup contributes to compositor peak scratch. Exact/one-short
  work, byte, canvas-phase and Vulkan tests prove their specific paths.
- Neutral image destination factories debit source work/capacity before
  emitting exact triangles and transfer accepted storage without duplicate
  retention. Glyph command conversion separately admits traversal and
  coexisting immutable F1 outline paths, including failure cleanup and
  owner-attributed diagnostics.
- **Open F3F obligations:** full original curved fill/stroke/glyph signed
  morphology and general nested consumption; 256-effect compositional
  accuracy and complete work/peak/in-flight accounting. Lyon's opaque vector/
  clip tessellation sweep, glyph mesh assembly and other unobserved
  construction scratch are **not** certified by current mesh-output receipts.
  A failed wide/nonnumeric chart or unsupported source is safe rejection, not
  a completed source-family proof.

Runenwerk native-window, Present, Render Lab and product policy remain
downstream consumer evidence, not standalone renderer validation. Future
changes must preserve or deliberately supersede the relevant evidence under
its owning issue and validate the exact reviewed revision.
