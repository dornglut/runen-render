# RunenRender licensing

The current public RunenRender representation is `GPL-3.0-only`; the complete
public license is in [LICENSE](LICENSE).

A separately negotiated commercial license may be available from copyright
holder(s) with sufficient rights to grant it. Repository ownership, organization
membership, or maintainer status alone does not prove ownership of every
copyright interest. Commercial pricing, warranties, indemnification, service
levels, customer-specific rights, and similar contract terms are outside this
document and require separate agreement and review.

Third-party code and other third-party material retain their own licenses and
notices. This repository does not silently relicense material it does not own.

## F2 shaped-text realization dependencies

The initial private shaped-text realization declares these exact direct
third-party dependencies:

- `bymsdfgen-core 0.1.1` — MIT;
- `skrifa 0.44.0` — MIT OR Apache-2.0 under the Fontations workspace policy.

`bymsdfgen-core` disables its default parallel/Rayon feature and reuses the
accepted `arrayvec 0.7.8` dependency. Skrifa disables default features and enables
only `std`. Its new locked closure is `read-fonts 0.41.0` and `font-types 0.12.6`,
both MIT OR Apache-2.0; it also reuses accepted `bytemuck 1.25.2` and
`once_cell 1.21.4`. No shaping, autohint-shaping, or parallel generation feature
is enabled by this slice.

Their code remains third-party material under those upstream terms. Cargo owns
the generated dependency closure in `Cargo.lock`; generated lockfile state must
come from Cargo rather than manual transcription.

The tiny F2 OpenType files under `tests/fixtures` are owner-created proof
material generated from the repository's geometric/table descriptions in
`generate_f2_fonts.py`. They do not copy a system or downstream font asset.
FontTools is manual fixture-generation tooling and is not vendored into the
RunenRender package.

Until an accepted inbound contribution mechanism preserves the rights required
for commercial relicensing, tracked-content contributions remain `owner-only`.
That includes code, documentation, tests, examples, build scripts, and assets.
Issue reports, discussion, reviews, and reproducible cases may still be accepted
through the repository's public channels according to repository policy.

## Historical framework-template grant

The generated initial repository contained material originating from the
Apache-2.0 `dornglut/rust-framework-template`. Rights already granted on those
historical template-origin revisions and material remain historical rights. The
current RunenRender product representation is established prospectively as
GPL-3.0-only.

## Runenwerk predecessor provenance

The reusable RunenRender implementation to be transferred later originated in
`dornglut/runenwerk`. The current accepted Runenwerk representation is
GPL-3.0-only. Earlier Runenwerk revisions published under the MIT License retain
the rights already granted for those revisions; the current policy does not
revoke or reinterpret them.

The later semantic-transfer authority records the exact predecessor revision,
file/blob census, adaptations, and third-party obligations. This bootstrap
transfers no Runenwerk implementation source.
