# Testing and validation

## Canonical command

```text
cargo validate
```

This command is implemented by repository-local `xtask` and is the merge-
readiness baseline.

## Bootstrap baseline checks

At bootstrap, validation proves repository/package integrity rather than the
future RunenRender execution/conformance portfolio. It covers:

- a clean starting repository and required authority files;
- RunenRender package/repository/version/license/MSRV identity;
- complete GPLv3 license text;
- no stale active framework-template identity/license;
- rustfmt;
- locked workspace tests;
- strict Clippy;
- rustdoc with warnings denied;
- explicit Rust 1.93.0 workspace/all-targets check;
- Git whitespace checks;
- validation not mutating repository state.

## CI

`.github/workflows/validation.yml` is intentionally thin. It pins the accepted
`dornglut/github-workflows` reusable Rust validation workflow to immutable
revision `688f274ec1fdd19acba9bf3577b26b4b7b7f4037` and delegates repository
meaning to `cargo +stable validate`.

The shared workflow proves the exact caller feature head and provisions stable
plus Cargo-declared `rust-version` toolchains.

## Later RX evidence

The semantic-transfer issue separately owns:
- package-level ordinary public conformance;
- exact RunenShader artifact -> RunenGPU admission/execution;
- headless/offscreen maintained execution;
- same-submission semantic result formation;
- no-implicit-readback proof;
- temporal/camera/requested-lattice conformance;
- applicable native/Vulkan proof;
- residue and no-mirror/no-private-reach-through guards.

No such renderer evidence is claimed by this source-free bootstrap.
