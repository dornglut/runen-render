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

New renderer capabilities should extend the existing semantic stages or add
private physical realization behind them. A private acceleration structure,
cache, atlas, compiled representation, GPU resource, or backend choice is not
public semantic identity merely because it improves realization.

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
