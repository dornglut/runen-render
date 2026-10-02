# RunenRender bootstrap and provenance

This record captures stable repository bootstrap facts and the boundary before
semantic extraction. It is not a branch, pull-request, workflow-run, or current-
head ledger.

## Accepted template provenance

```text
template repository:
dornglut/rust-framework-template

accepted template commit:
500461d51fe155febc806e288e5bc013e413a785

accepted template tree:
1e1ae24713cd48b5ea2c3fe1da87cf8dd8f8358a

generated RunenRender initial commit:
8f1b9719f41769430ef50f540f4c0e14e7e2c1f9

generated RunenRender initial tree:
1e1ae24713cd48b5ea2c3fe1da87cf8dd8f8358a
```

The generated initial tree exactly matched the accepted template tree.
Generated-repository validation run `36984365701` succeeded on that exact
initial commit through immutable shared workflow revision
`688f274ec1fdd19acba9bf3577b26b4b7b7f4037`.

The historical template material originated under Apache-2.0. That historical
grant remains historical provenance and is not ongoing synchronization,
architecture, product-license, or source authority.

## Product decisions

```text
repository: dornglut/runen-render
package: runen-render
crate: runen_render
version: 0.1.0
edition: 2024
MSRV: 1.97.1
publish: false
features: default=[]
license: GPL-3.0-only
profile: rust-framework
lifecycle: active
contribution: owner-only
```

Version `0.1.0` represents the first standalone consumer-facing RunenRender
framework contract rather than a placeholder identity.

Source-free bootstrap initially selected Rust `1.93.0` because no lower
RunenRender support floor was accepted or proven during R8. That provisional
choice was superseded before semantic authority acceptance: exact-head RX
validation with the accepted RunenShader revision proved that its exact
`wesl 0.5.0` dependency graph requires Rust `1.97.1`. The successor candidate
therefore establishes `1.97.1` as the supported MSRV. A future lower floor
requires an accepted sibling/dependency change plus direct validation evidence.

The generic template `unsafe_code = "forbid"` lint is intentionally removed.
R8 did not accept a new blanket RunenRender unsafe-code policy, and extraction
must not silently add source constraints.

## Repository classification

The accepted bootstrap repository posture is:

```text
visibility: public
default branch: main
merge: squash-only
delete merged branches: enabled
profile: rust-framework
lifecycle: active
contribution: owner-only
```

Native settings/protection were reconciled before bootstrap acceptance. Security
controls remain native platform administration and must not be inferred from
source files.

Accepted source-free bootstrap main:

```text
8c1d72b00e50d5e2bc14fcbf5ffd909dfce4c674
```

## RX transfer provenance

The separately authorized successor transfer is governed by `runen-render#1`
and `runen-render#4` from frozen Runenwerk R8:

```text
predecessor:
dornglut/runenwerk

accepted predecessor revision:
75d793d227235441952167303051334f5b3a1e0f

accepted predecessor tree:
2abd9fe0792969b1d26106cb7f8e1741c6c9d68f

transfer authority:
runenwerk#1129
```

The accepted bootstrap transferred zero RunenRender implementation source.

The current unmerged RX candidate transfers the frozen production/conformance
boundary and uses these exact sibling revisions:

```text
runen-gpu:
789b430fdefeda89bfe59de86d548618b8f8ab9a

runen-shader:
406f8165da92caa2d296b3a2f774ba534279870d
```

Current full-file blob census against the frozen 32-file R8 production set:

```text
byte-identical full files: 14
classified adaptations:   18
total:                     32
```

The 18 adapted files are fully classified:

- test-import relocation only:
  `admission.rs`, `derived_state.rs`, `deterministic_admission.rs`,
  `deterministic_verification.rs`, `field_input.rs`, `method.rs`,
  `output_result.rs`, `participation.rs`, `representation.rs`,
  `request.rs`, `semantic_plan.rs`, and `surface_result.rs`;
- standalone proof-only scoping in addition to test-import relocation:
  `derived_transform.rs` and `scene/mod.rs`;
- standalone proof-only scoping:
  `deterministic_verification/numeric.rs`;
- monolith-relative visibility normalization:
  `deterministic_verification/observation.rs`;
- owner-correct RunenShader ordinary error projection plus deletion of the
  predecessor camera artifact side channel:
  `ordinary.rs`;
- retained RunenShader artifact -> RunenGPU admission bridge, deletion of the
  predecessor camera artifact side channel, test-only import relocation, and
  canonical formatting:
  `deterministic_execution.rs`.

The four maintained WGSL blobs remain byte-identical to the accepted R8 freeze.
Runenwerk remains sole accepted RunenRender semantic authority until the
successor candidate is accepted on `runen-render/main` under ADR 0008.
