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

Detailed public integration laws are in the [semantic integration contract](docs/architecture/semantic-integration.md). The code and tests remain the source of actual behavior.

## 2D composition extension contract

The 2D composition extension defines durable semantic and ownership authority.
The immutable composition/resource model is implemented. Maintained execution
realizes one mixed direct-root and nested-group painter tree: solid, linear-gradient
and radial-gradient vector fills/strokes, immutable RGBA8-sRGB image patches,
retained F2 shaped monochrome outline text, and item/group opacity and clips.
All admitted painting uses one correlated 4×4 color sample lattice followed by
one final source-over resolve into the caller-owned target. Group children are
composed in isolated sample-space color before group clips and group opacity
apply exactly once. Ordinary shadows/effects and native presentation remain
outside the admitted realization and reject or stay downstream-owned respectively.

The detailed [2D composition contract](docs/architecture/composition-2d.md) owns shadow semantics, realization and resource guarantees. Accepted immutable shadow meaning does not imply the unmerged F3F physical implementation is complete.

## Private maintained realization

The private `runtime` hierarchy owns renderer-method admission, maintained programs and host/WGSL ABI, execution, 2D realization, capture and verification. The [maintained runtime architecture](docs/architecture/maintained-runtime.md) records detailed owners and lifecycle invariants.

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
or falsify owned semantics; the [2D composition contract](docs/architecture/composition-2d.md) is such
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

