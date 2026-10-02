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
MSRV: 1.93.0
publish: false
features: default=[]
license: GPL-3.0-only
profile: rust-framework
lifecycle: active
contribution: owner-only
```

Version `0.1.0` represents the first standalone consumer-facing RunenRender
framework contract rather than a placeholder identity.

Rust `1.93.0` is the initial supported MSRV because no lower RunenRender support
floor was accepted or proven during R8. A later lower MSRV requires independent
evidence; bootstrap does not infer one from sibling packages.

The generic template `unsafe_code = "forbid"` lint is intentionally removed.
R8 did not accept a new blanket RunenRender unsafe-code policy, and extraction
must not silently add source constraints.

## Repository classification

The intended accepted repository posture is:

```text
visibility: public
default branch: main
merge: squash-only
delete merged branches: enabled
profile: rust-framework
lifecycle: active
contribution: owner-only
```

Native settings/protection/security are repository administration, not semantic
source authority. They must be reconciled before bootstrap acceptance.

## Future extraction provenance

The separately authorized semantic transfer is governed by `runen-render#1`
from frozen Runenwerk R8:

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

This bootstrap transfers zero RunenRender implementation source and adds no
RunenGPU or RunenShader product dependency. Runenwerk remains sole RunenRender
semantic authority until the later accepted successor merge under ADR 0008.
