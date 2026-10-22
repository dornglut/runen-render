# Maintained runtime, ordinary lifecycle and scene storage

This detailed runtime architecture is summarized by [ARCHITECTURE.md](../../ARCHITECTURE.md). It introduces no second API or implementation authority.

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
│  ├─ vector         bounded private local geometry tessellation and target clipping
│  └─ lowering       ordered text/coverage/composition RunenGPU resources and work
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

Ordinary admission SHOULD consume one validated invocation aggregate rather than
requiring callers to repeatedly transport parallel scene/request/input/
availability/output collections as unrelated arguments. The façade MUST use the
same canonical planning, semantic binding, execution admission, and lowering
authorities as advanced consumers. Those canonical stages remain independently
inspectable for advanced consumers; ordinary convenience MUST NOT become the
only path to renderer-owned semantic evidence.

Retained ordinary execution has an explicit continuity owner. One independent
logical renderer continuity MUST map to one independent retained session/state
owner rather than multiplexing unrelated continuities through caller-chosen raw
integer namespaces. Product producer, surface, view, ECS, window, and native
presentation identities may map to that owner in downstream integration, but
they are not renderer continuity identity.

Stateful preparation and caller-owned RunenGPU composition obey this lifecycle:

```text
admitted renderer invocation
    -> prepare one exact renderer occurrence
        -> renderer-authored RunenGPU work
        -> caller composes arbitrary peer RunenGPU work
        -> caller obtains GpuSubmission
        -> exact occurrence/submission association
        -> terminal reconciliation
```

The prepared occurrence MUST carry or identify the exact renderer-authored work
required for its retained-state transition. A raw continuity/session identifier,
label, provenance string, cache identity, or output resource is insufficient
execution evidence. Association MUST fail when the supplied submission does not
contain the occurrence's required RunenGPU work. Only the exact accepted
occurrence may become pending retained history, and only successful terminal
execution may advance successful temporal history. Abandoned or failed
occurrences MUST NOT later become successful history by scope convention.

Successful exact association returns a durable output-correlation witness distinct
from the retained session. Such a witness MUST NOT own or advance temporal
history. It may survive session reconciliation or destruction because its
authority is limited to the exact associated occurrence's admitted outputs and
RunenGPU submission.

Retained output interpretation MUST remain exactly correlated to that occurrence.
Radiance capture is eligible only after the associated renderer submission
completes successfully and while the exact retained destination still names that
submission as its current completed writer. Caller-owned readback transport
remains RunenGPU authority; RunenRender owns carrier interpretation and MUST
reject foreign correlation, lost affinity/continuity, stale writer state,
incompatible layout, or failed/pending execution. Captured maintained samples
are physical output interpretation, not a newly certified `RenderResult`.

An execution-local object-identity decoder MAY be retained with the exact
associated occurrence, but it maps physical carrier words only to optional
renderer object IDs. It MUST NOT be treated as per-pixel definedness,
miss/background evidence, persistent identity, or authority for words produced
by another occurrence.

One continuity may serialize its own pending lifecycle when correctness requires
it. Independent continuities MUST NOT be serialized merely because an
implementation stores them in one global mutable map. A future renderer-wide
derived cache may share immutable/reconstructible work across sessions when
concrete reuse justifies that owner, but renderer-wide cache identity is
distinct from temporal continuity identity.

RunenGPU context affinity includes both logical context and device generation.
When the device generation changes, renderer-derived temporal history and other
current-content state that depends on prior completed renderer execution MUST
reset unless RunenRender has an explicitly proven reconstruction source for that
exact state. This does not forbid RunenGPU from reconstructing retained physical
resources whose owner contract already provides a truthful reconstruction
source. Semantic scene/request/source identity may survive the renderer-history
reset. Reset must remain observable through renderer-owned temporal
history/generation evidence. A new device generation requires fresh
capability-dependent admission.

One-shot ordinary submission and verified-result paths MAY omit a retained
session when no cross-invocation continuity is requested.

### Ordinary execution lifecycle at a glance

The caller constructs an admitted invocation, prepares renderer-authored work, and submits it through RunenGPU. The exact prepared occurrence must be associated with the resulting submission: association proves work membership but not terminal completion. Reconciliation advances retained history only after successful terminal execution. The associated output-correlation witness can survive session reconciliation; physical capture additionally requires the retained destination's current completed writer to match. The separate one-shot `submit_render_for_result` requests verified semantic-result formation before submission. It does not convert ordinary physical capture into a `RenderResult`.

## Scene storage

`runen_render::scene` owns public object identity, scene revisions, updates,
change sets, immutable snapshots, commits, resync, allocation, and store
semantics.

One accepted scene update is an atomic renderer-semantic transaction. An update
MUST be able to change multiple independent owned facets of one existing object
within the same transaction when those changes are jointly valid, for example
spatial/temporal state plus renderer participation. Insert/remove conflicts and
other incompatible mutations remain fail-before-commit validation errors.

One successful transaction establishes one new scene revision and precise change
evidence for every affected facet. Callers MUST NOT need multiple scene
revisions merely because independently owned facets were encoded as separate
operation variants.

The persistent structurally shared radix storage is private under
`scene::storage`. Radix nodes, path-copy mechanics, and storage continuity are
implementation details and are not part of public scene identity or persistence
semantics. Storage remains free to apply one optimized path copy per changed
object or use another persistent representation while preserving the same
transaction/snapshot/change-set contract.

