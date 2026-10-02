# RunenRender executor contract

Begin with `README.md`, `ARCHITECTURE.md`, `TESTING.md`, `BOOTSTRAP.md`,
the current owning issue, and the active Engineering RX initiative. Confirm the
accepted base, repository state, current semantic authority, and dependency
revisions before editing.

## Durable constraints

- Keep exactly one semantic source authority per concern.
- Before successor acceptance, Runenwerk remains RunenRender semantic authority.
- Repository bootstrap #2 is complete. Transfer implementation only under
  successor transfer authority #4 and parent acceptance authority #1.
- Preserve the accepted ordinary/public surface; do not expose proof-era
  `Deterministic*` vocabulary by physical-transfer accident.
- RunenRender owns the renderer-side RunenShader-artifact -> RunenGPU-program
  bridge; RunenShader and RunenGPU retain their separate error/identity
  authorities.
- Keep Runenwerk App/ECS/Winit/World/UI/Editor/Render-Lab/product policy out of
  reusable production source.
- Do not add compatibility aliases, forwarding modules/crates, mirrors, source
  includes, submodules, moving dependencies, duplicate renderer authority, or
  private sibling reach-through.
- Do not mirror maintained WGSL in two accepted semantic owners.
- Keep tracked-content contributions `owner-only` until accepted inbound terms
  preserve required relicensing rights.

## ADR-0008 sequence

```text
accepted Runenwerk R8
    -> sole semantic source authority
source-free runen-render bootstrap
    -> repository authority only
unmerged runen-render semantic candidate
    -> candidate only
accepted runen-render successor on main
    -> semantic authority switches
Runenwerk predecessor transfer boundary
    -> frozen and deletion-bound
accepted Runenwerk exact-revision cutover
    -> predecessor authority removed
```

## Validation and evidence

Run the canonical command from a clean checkout:

```text
cargo validate
```

CI must validate the exact reviewed feature head through the repository-owned
baseline and dedicated Vulkan conformance job. Do not claim shader-artifact,
RunenGPU, semantic-result, or downstream evidence until the relevant transfer
work actually runs those proofs. Runenwerk native-window/Present/Render-Lab
evidence stays downstream and must not be recreated here as product policy.
