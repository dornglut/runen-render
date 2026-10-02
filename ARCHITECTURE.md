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

## Ownership

RunenRender owns reusable renderer-domain meaning:

- semantic scene, request, representation, participation, appearance, and input
  contracts;
- render planning/admission/lowering into the maintained renderer authority;
- maintained image-formation execution state and temporal/camera evidence;
- generic semantic result-formation evidence;
- the renderer-owned retained bridge from exact RunenShader artifacts to public
  RunenGPU program admission/execution.

RunenShader owns shader source identity, compilation, artifact provenance, and
typed compilation outcomes.

RunenGPU owns backend-neutral program/resource/work/submission/readback
contracts and private backend realization.

Runenwerk stays above the framework boundary and owns App/ECS/Winit/native-host
lifecycle, World/UI/Editor adapters, frame/presentation scheduling, product
composition/Present policy, Render Lab product behavior, and product
filesystem/JSON/media artifact policy.

## Public versus private surface

The RX successor candidate keeps ordinary/public renderer vocabulary separate
from maintained implementation vocabulary. Proof-era `Deterministic*` types
and methods remain private implementation details unless a separately accepted
public-contract change proves otherwise.

No compatibility facade, mirror, source include, Git submodule, moving branch
dependency, mirrored maintained WGSL authority, or private sibling reach-through
is part of this architecture.

## Authority transfer

During the current unmerged successor candidate, Runenwerk remains the sole
accepted RunenRender semantic source authority even though the candidate source
is physically present here.

Under ADR 0008, accepted successor default-branch publication switches semantic
authority to RunenRender. The transferred Runenwerk predecessor boundary freezes
at that moment and is deletion-bound during the exact-revision downstream
cutover.
