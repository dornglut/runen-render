# RunenRender executor contract

Begin with `README.md`, `ARCHITECTURE.md`, `TESTING.md`, `BOOTSTRAP.md`,
the current owning issue, and applicable Dornglut Engineering governance. Confirm
the accepted base, repository state, current semantic authority, dependency
revisions, active writers, and required validation before editing.

## Current authority

RunenRender is the accepted standalone semantic and conformance authority for
reusable renderer semantics and maintained image formation. The RX authority
transfer and the Runenwerk exact-revision predecessor cutover are complete.

Ordinary standalone evolution proceeds only under a current accepted owning
issue. Historical RX issues and predecessor branches are provenance, not current
implementation authority.

## Durable constraints

- Keep exactly one semantic source authority per concern.
- Preserve accepted public semantics unless a separately accepted contract
  change authorizes otherwise.
- Keep proof-era `Deterministic*` vocabulary private unless a separately
  accepted public-contract change proves it belongs in the public API.
- RunenRender owns the renderer-side RunenShader-artifact -> RunenGPU-program
  bridge; RunenShader and RunenGPU retain their separate error, artifact,
  resource, work, submission, and backend authorities.
- Keep Runenwerk App/ECS/Winit/World/UI/Editor/Render-Lab/product policy out of
  reusable production source.
- Do not add compatibility aliases, forwarding modules/crates, mirrors, source
  includes, submodules, moving dependencies, duplicate renderer authority, or
  private sibling reach-through.
- Do not recreate transferred predecessor semantic/method/WGSL authority in
  Runenwerk or mirror maintained WGSL across accepted semantic owners.
- Keep tracked-content contributions `owner-only` until accepted inbound terms
  preserve required relicensing rights.

## Historical RX sequence

Engineering ADR 0008 completed this authority sequence:

```text
accepted Runenwerk R8
    -> predecessor semantic source authority
source-free runen-render bootstrap
    -> repository authority only
accepted runen-render successor on main
    -> semantic authority switched
Runenwerk predecessor transfer boundary
    -> frozen and deletion-bound
accepted Runenwerk exact-revision cutover
    -> predecessor authority removed
```

Do not use this completed sequence as a reason to block current standalone work
or to reopen transfer-era ownership.

## Validation and evidence

Run the canonical command from a clean checkout:

```text
cargo validate
```

CI must validate the exact reviewed feature head through the repository-owned
baseline and dedicated Vulkan conformance job. Report only evidence actually
observed.

Runenwerk native-window/Present/Render-Lab/product evidence remains downstream
consumer evidence and must not be recreated here as reusable framework policy.
