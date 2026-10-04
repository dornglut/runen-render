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

## Accepted 2D composition extension contract

The accepted 2D composition extension is a target architecture for the
separately sequenced production implementation. This section defines durable
semantic and ownership authority; it does not claim that the complete 2D
production subsystem is already implemented.

The reusable semantic root is one lifetime-neutral immutable 2D composition
value. It represents resolved renderer meaning such as:

- exact painter and atomic-group order;
- generic 2D geometry, transforms, clips, fills, strokes, and opacity;
- explicit color, gradient, source-over, and ordinary-shadow/effect meaning;
- resolved image source/destination mapping plus immutable semantic resource
  identity;
- already-shaped glyph occurrences plus exact immutable font, variation,
  synthesis, size, and glyph-position facts required for realization.

This is a sibling renderer-semantic domain to the existing
surface/field/query representation protocols. Ordered 2D composition must not
be encoded as a `SurfaceQuery`, `FieldDistance`, display-list/GPU command
stream, source-framework paint type, or universal vector/field ontology merely
to reuse an existing physical path.

Source adapters project already-resolved renderer facts into RunenRender-owned
values. Source-framework publication revisions, widget/mounted identities,
authoring policy, and native presentation identity remain outside RunenRender
semantic identity.

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
supported exact synthesis, privately realized through an MSDF/MTSDF-family
field representation. Missing compatible field state is reconstructed
synchronously from the immutable shaped resource before execution; a second
vector/alpha realization is not required solely for cache misses.

Intrinsic COLR, SVG, bitmap glyph content and unsupported synthesis remain
distinct capability classes and fail closed until an exact separately accepted
realization exists. They must not be silently flattened to monochrome outline
semantics.

The initial image resource contract admits the known-consumer immutable RGBA8
sRGB payload plus exact intrinsic extent and resolved source/destination
mapping. Its sampled texture, upload, cache, and device residency remain
private.

### Initial private realization disposition

The first production implementation is authorized to target this private
portfolio without making the listed implementation families public semantics:

- general fills, paths, strokes, clip/support geometry: Lyon-class
  backend-neutral tessellation lowered into RunenGPU geometry work;
- scalable monochrome outline text: MSDF/MTSDF-family retained field
  realization with synchronous cache-miss reconstruction;
- images: intrinsic sampled-resource realization;
- atomic groups, clips, and ordinary shadows: private intermediate/mask work
  derived from the same ordered semantic composition.

Analytic simple-shape specialization, Sparse Strips, direct-curve rendering,
direct-vector/hinted/alpha text alternatives, and other acceleration families
remain deferred replaceable implementation options. They require new evidence
when a real workload or capability need makes their additional machinery
material; they do not require a public-contract change when they preserve the
same admitted semantics.

No current architectural decision claims native-GPU superiority for one
realization family. Concrete future performance selections require evidence
appropriate to the claim.

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

RunenRender execution evidence identifies the accepted/executed renderer
contribution without depending on source-framework publication IDs or native
Present policy. The downstream combining boundary owns the correspondence
between that contribution, the source publication, and terminal presentation.

RunenRender is pre-1.0 and current consumers use accepted immutable revisions.
A public 2D semantic change therefore requires explicit RunenRender authority,
validated accepted revision, downstream compatibility review, and downstream
repin/adaptation. Do not introduce duplicate versioned types, compatibility
aliases, runtime schema-version fields, or persistence/wire machinery without a
real compatibility obligation.

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
or falsify owned semantics; the accepted 2D composition contract is such a
case.

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
