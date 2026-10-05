# RunenRender architecture

## Dependency direction

```text
Runenwerk product/integration
    -> RunenRender renderer semantics + maintained image formation
        -> RunenShader source/compilation/artifact semantics
        -> RunenGPU generic GPU execution
```

RunenRender composes downward sibling authorities; it does not absorb or
re-export their private implementations.

## Public semantic contract

RunenRender owns reusable renderer-domain meaning:

- semantic scene, request, representation, participation, appearance, and input
  contracts;
- semantic planning, binding, execution admission, and result evidence;
- maintained image-formation behavior and temporal/camera rendering evidence;
- renderer-owned composition of accepted RunenShader artifacts into public
  RunenGPU program admission and execution.

The canonical public surface remains organized around semantic responsibility.
The crate root also re-exports the ordinary progressive-disclosure API and
generic result vocabulary. Proof-era `Deterministic*` vocabulary is private
implementation terminology, not a second public renderer ontology.

## 2D composition extension contract

The 2D composition extension defines durable semantic and ownership authority.
The immutable composition/resource model is implemented, and the initial F2
execution slice realizes the bounded direct-root shaped-text subset described
below. This section does not claim that the complete 2D production subsystem,
other primitive families, groups, clips, effects, or native presentation are
already implemented.

The reusable semantic root is one lifetime-neutral immutable 2D composition
value. It represents resolved renderer meaning such as:

- exact painter and atomic-group order, with group opacity applied once;
- structural rectangle, rounded-rectangle, ellipse, and arbitrary path geometry,
  including explicit fill rule, centered stroke, cap, join, and miter semantics;
- item/group transforms, conjunctive clips, fills, strokes, and opacity;
- straight-alpha sRGB literal color meaning, gradient interpolation in
  premultiplied linear-sRGB, and ordered linear-light source-over composition;
- ordinary-shadow/effect meaning derived from neutral semantic support geometry
  rather than cached sampled source alpha, including effects from transparent
  child color;
- resolved image source/destination mapping plus immutable semantic resource
  identity;
- already-shaped glyph occurrences plus semantic shaped-resource identities
  whose immutable bindings carry exact font/face, variation, synthesis, size,
  and glyph facts required for realization.

This is a sibling renderer-semantic domain to the existing
surface/field/query representation protocols. Ordered 2D composition must not
be encoded as a `SurfaceQuery`, `FieldDistance`, display-list/GPU command
stream, source-framework paint type, or universal vector/field ontology merely
to reuse an existing physical path.

Source adapters project already-resolved renderer facts into RunenRender-owned
values. Source-framework publication/object identities, authoring policy, and
native presentation identity remain outside RunenRender semantic identity.

One execution combines one immutable composition value with one compatible
immutable semantic-resource binding set and target/output fact set. Private
parallel preparation is permitted only when it cannot observe mutable source
state, alter painter/group order, or change semantic results.

### Admission and private realization

The 2D path preserves the same architectural law as the existing renderer
without reusing incompatible 3D semantic types:

```text
immutable 2D composition semantics
    -> semantic validation
    -> invocation-compatible resource and target binding
    -> semantically admissible private realizations
    -> private cost/quality/availability selection
    -> private compilation and retained derived state
    -> RunenGPU work
    -> source-neutral contribution/execution correlation
```

Semantic admissibility and physical policy are separate. Cache residency,
device capability, preprocessing cost, or a preferred implementation cannot
make a semantically incompatible realization admissible. A cache miss may
change cost or select another already-admitted equivalent realization; it must
not rewrite content, ordering, resource identity, color/composition meaning, or
text metrics.

A realization family is replaceable only within its admitted semantic
obligation. When one semantic class has a single initial realization, no second
implementation is required merely to demonstrate replaceability. A later
alternative must consume the same semantic input and prove the same admitted
semantics before policy may select it.

### Resources, text, and derived state

Semantic resource identity is distinct from every prepared or resident form:

```text
semantic resource identity
    -> immutable semantic binding / intrinsic representation
        -> private realization descriptor
            -> prepared/cache realization
                -> atlas / buffer / texture / RunenGPU residency
```

Cache keys may contain private reproducibility facts such as field quality,
range, generator revision, or tessellation parameters. Atlas coordinates,
cache buckets, device generations, and RunenGPU handles are never semantic
identity. Derived state must be reconstructible from immutable semantic facts or
fail with an owner-oriented structural/capability outcome.

Text is shaped before RunenRender. The renderer must not perform fallback,
line breaking, reshaping, font rediscovery, or metric authority. The initial
admitted text class is already-shaped scalable monochrome outline content with
supported exact synthesis. Its physical outline/field/coverage preparation is
renderer-private. Missing compatible derived state is reconstructed from the
immutable shaped resource before execution; another realization family is not
required solely for cache misses.

Intrinsic COLR, SVG, bitmap glyph content and unsupported synthesis remain
distinct capability classes and fail closed until an exact separately accepted
realization exists. They must not be silently flattened to monochrome outline
semantics.

The initial F2 production execution boundary is `execution_2d`. It accepts a
caller-owned RunenGPU color view plus logical canvas extent and raster scale,
admits only the exact bounded direct-root shaped-text subset, and returns
composable RunenGPU work wrapped with source-neutral execution correlation.
Retained semantic identity observations are separate from the private derived
field cache: cache discard may remove derived MSDF fields but cannot make an
already-observed resource identity legal to rebind. F2 retains reconstructible
CPU-side field data by semantic resource and quality tier while GPU
field textures, vertex buffers, sampler state, and draw bindings are
contribution-local disposable realization state. No global atlas or device
residency authority is introduced by this slice.

A painting F2 contribution prepares one private RunenGPU render operation containing
painter-ordered glyph draws. Consuming it appends one renderer-authored node to a
caller-owned RunenGPU fragment, or authors a separate fragment. Both paths use
the same lowering. Caller-owned `Render2dWorkBinding` keys connect
prior target contents and the resulting target output through RunenGPU's typed
import/export relationships. They are graph wiring, not semantic identity or
execution evidence. Non-painting contributions expose no fabricated output;
callers retain the prior target-content relationship when no render work exists.
Successful contribution evidence requires exact
membership of that authored `GpuWorkNodeId` in the downstream-composed
submission and terminal `Completed` status from that same submission. Valid
non-painting content authors no synthetic draw or sentinel and therefore has no
execution token.

The initial admitted image payload class is immutable tightly packed
unpremultiplied RGBA8 sRGB plus exact non-zero intrinsic extent and resolved
source/destination mapping. Sampled textures, uploads, caches, and device
residency remain private.

### Private realization boundary

Concrete implementation-library selection is not part of this semantic
contract. Implementations may privately use backend-neutral tessellation,
analytic coverage, signed-distance fields, direct curves, intrinsic sampled resources,
intermediate masks/targets, or other realization families when they preserve
the admitted semantic obligation and lower through RunenGPU. Concrete library,
algorithm, threshold, atlas, cache, batching, and pass-topology choices remain
replaceable implementation policy.

Adding, replacing, or deleting one private realization family does not require a
public semantic-contract change when the replacement preserves the same
admitted semantics. A new performance-driven selection requires evidence
appropriate to the claim; the architecture itself does not claim native-GPU
superiority for one family.

### Failure, evidence, and evolution

Caller-material failures remain owner-oriented and distinguish at least:

- malformed semantic composition;
- unsupported semantic or intrinsic capability;
- missing, incompatible, or unavailable semantic resource binding;
- valid content with no currently admitted realization;
- private realization/lowering failure;
- RunenShader compilation failure;
- RunenGPU preparation/submission/execution failure.

Native Present and source-publication failures remain downstream-owned.
Diagnostics identify the affected semantic/resource subject without exposing
private cache/backend identity as source truth.

RunenRender execution evidence identifies the admitted/executed renderer
contribution without depending on source-framework publication IDs or native
Present policy. For the initial F2 slice, that evidence is exact authored
RunenGPU work-node membership plus terminal completion, not labels, cache
identity, readback contents, or sentinel inference. The downstream combining
boundary owns the correspondence between that contribution, the source
publication, and terminal presentation.

While RunenRender is pre-1.0 and consumers are revision-pinned, a public 2D
semantic change requires explicit RunenRender authority, a validated accepted
revision, downstream compatibility review, and downstream repin/adaptation.
Do not introduce duplicate versioned types, compatibility aliases, runtime
schema-version fields, or persistence/wire machinery without a real
compatibility obligation.

The 2D semantic/compiler responsibility remains inside the RunenRender crate.
A separate public text/path/image renderer, `runen-render-2d` package, or
backend-specific public namespace would duplicate composition authority without
a demonstrated independent owner.

## Private maintained realization

The maintained physical renderer is private under `runtime`:

```text
runtime
├─ method          maintained method realization
├─ admission       method-specific semantic admission
├─ transform       compiled renderer transforms
├─ carrier         private physical carrier facts
├─ program
│  ├─ abi          named host <-> maintained-WGSL layout/revision authority
│  └─ shaders      maintained shader sources
├─ execution
│  ├─ state          retained resources and temporal/history lifetime
│  ├─ lifecycle      prepared/submitted/result correlation
│  ├─ output_context output/request/observation correlation and temporal preparation
│  ├─ packing        host physical semantic encoding
│  ├─ layout         physical row/layout alignment helpers
│  ├─ passes         primary/coverage/temporal GPU pass preparation and dispatch
│  ├─ finalize       destination/export/evidence/readback and fragment assembly
│  ├─ prepare        render/output orchestration
│  ├─ submission     RunenGPU submission
│  └─ errors         typed owner-preserving error projection
├─ execution_2d
│  ├─ intrinsic      fail-closed font/glyph representation admission
│  ├─ field          retained reconstructible shaped-outline/MSDF field realization
│  └─ lowering       contribution-local RunenGPU resources, ordered draws, and work
├─ capture
└─ verification
```

RunenShader remains the authority for source identity, compilation, artifact
provenance, and typed compilation outcomes. RunenGPU remains the authority for
backend-neutral programs, resources, work, submission, readback, and private
backend realization.

Maintained program source composition, exact revisions, retained RunenShader
artifacts, and RunenGPU admission have one private owner under
`runtime::program`. The host/WGSL contract uses named private ABI constants and
parity proofs; it is not a public renderer ABI.

Maintained output preparation is intentionally an orchestration pipeline rather
than a single execution owner:

```text
resolved output context
    -> temporal state
    -> physical packing
    -> prepared primary / coverage / temporal passes
    -> destination + execution evidence + optional verification readbacks
    -> ordered work-fragment assembly
```

The retained resource cache owns reusable resource identities and temporal
history lifecycle, not maintained program-source retention. Dispatch sizing and
camera-reprojection parameter realization belong with pass preparation rather
than physical output packing. Execution modules use explicit dependencies at
the preparation/state boundary instead of a shared private prelude.

## Ordinary façade

The ordinary API is an ergonomic façade over the canonical semantic and runtime
stages, not a second architecture. Its implementation is private under
`ordinary` and separates state, lifecycle, operations, and error projection
while preserving the crate-root public API.

## Scene storage

`runen_render::scene` owns public object identity, scene revisions, updates,
change sets, immutable snapshots, commits, resync, allocation, and store
semantics.

The persistent structurally shared radix storage is private under
`scene::storage`. Radix nodes, path-copy mechanics, and storage continuity are
implementation details and are not part of public scene identity or persistence
semantics.

## Proof and validation topology

Crate-private conformance evidence is grouped under `src/proofs` by current
responsibility rather than historical delivery phase. External-consumer proofs
remain under `tests`.

Repository validation discovers current maintained Rust/WGSL source
recursively. It enforces public/private boundaries and forbidden
product/backend coupling without freezing transfer-era private filenames.

The dedicated Vulkan lane proves the maintained offscreen execution path on the
exact reviewed revision.

## Product boundary

Runenwerk stays above the framework boundary and owns App/ECS/Winit/native-host
lifecycle, World/UI/Editor adapters, frame/presentation scheduling, product
composition/Present policy, Render Lab product behavior, and product
filesystem/JSON/media artifact policy.

No compatibility façade, source mirror, Git submodule, moving sibling
dependency, mirrored maintained WGSL authority, or private sibling
reach-through is part of this architecture.

## Extension laws

New renderer capabilities should extend an existing semantic domain when it can
express the required meaning losslessly. A new sibling semantic domain is
justified only when forcing the behavior into an existing ontology would erase
or falsify owned semantics; the 2D composition contract defined above is such
a case.

Private physical realization remains behind the owning semantic domain. A
private acceleration structure, cache, atlas, compiled representation, GPU
resource, or backend choice is not public semantic identity merely because it
improves realization.

A public contract change requires explicit authority and consumer evidence. Do
not expose a physical implementation strategy simply to make one feature easier
to implement.

## Authority provenance

Engineering ADR 0008 transferred reusable RunenRender semantic authority from
Runenwerk to this repository. That transfer and the exact-revision downstream
predecessor deletion are complete.

Historical transfer details remain provenance in `BOOTSTRAP.md`; they are not
current execution sequencing or a reason to preserve extraction-era internal
layout.
