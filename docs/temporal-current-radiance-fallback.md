# Phase-aligned current radiance fallback

Status: implementation scope of [issue #83](https://github.com/dornglut/runen-render/issues/83). The corrected semantic decision and counterexamples remain in [issue #8](https://github.com/dornglut/runen-render/issues/8); this note describes the maintained physical implementation and does not replace that authority.

## Ownership and semantics

Sub-native static reconstruction uses the already admitted renderer scene, materials, emitters, perspective observation, immutable semantic-input generations and four-phase lattice sampling law. The existing P100 camera-history shader remains distinct. There is no TAAU, motion-vector API, general camera-motion reuse, new scene evaluator, presentation owner or extra submission lifecycle.

One physical evaluation phase produces M primary rays for N requested cells. The renderer checks the **actual existing f32 phase mapping for injectivity on both axes and both phase offsets before admitting sub-native scatter**. Real-number sampling ratios alone are not proof of unique physical writes. A rejected mapping is a typed pre-execution error.

The temporal reconstruction shader scatters an explicit current-phase visited state for each primary sample (0 = not visited, 1 = visited but invalid, 2 = visited and finite) and preserves the current physical sample independently of retained finite history. Only primary evaluations change retained temporal sample counts. A first current primary sample may seed the static storage count inside that submission, but is still classified as **provisional current** rather than previously certified history. Once at least two genuinely compatible phases contribute, their finite static estimate remains available; fallback itself never advances counts. The current-only fallback pass has one invocation per requested cell:

- For a compatible, finite retained static estimator and a non-invalid current visit, output the estimator (availability 1).
- If current-phase primary evaluation is valid but no usable estimator exists, output that actual primary sample (availability 2).
- If neither is available and this phase did not visit the cell, evaluate the **same phase-aligned requested-cell ray** through the shared RunenShader-admitted nearest-hit/direct-radiance implementation. A valid same-ray no-hit is defined zero under the maintained no-environment policy; a hit requires successful finite direct illumination (availability 2).
- If the primary query was invalid, current-only query is invalid, or radiance cannot be established, output availability 0 and the explicit quiet-NaN physical sentinel `0x7fc00000`. A zero-initialized scratch word is never considered availability evidence.

A provisional current-only sample is neither a finished footprint integral nor a new history contributor. Source generation and scene/observation revisions continue to invalidate retained history. The temporal signature additionally retains and compares the **exact structurally shared immutable scene snapshot**: different scene stores with coincident numeric revisions but different material/emitter facts MUST NOT be accepted as compatible history. Persistent root equality is O(1) for unchanged snapshots. GPU submission identity and terminal status remain with RunenGPU. A prepared output alone is not successful execution evidence.

## Downstream contract

For sub-native static radiance outputs, the renderer authors two exact same-fragment RunenGPU exports: the physical R32Float texture and a **dense unpadded row-major u32 availability buffer**. Import and honor both for composition; the same output/graph relationship and completed producer submission are necessary. The numeric availability ABI is `RenderRadianceCellAvailability`:

| Word | Meaning |
| --- | --- |
| 0 | Unresolved: physical payload is not verified radiance |
| 1 | Compatible finite static history estimate (not motion-coherent) |
| 2 | Provisional finite current-phase primary or fallback sample |

A consumer may elect to display a visual placeholder for unresolved cells as **its own explicit presentation policy**, but MUST NOT report those placeholders or zero-filled/uninitialized texture bytes as resolved renderer radiance. Existing R32Float readback interpretation rejects NaNs instead of fabricating finite captures.

At P100, ordinary camera-history execution and established semantic verification remain unchanged; no sub-native availability mask is exported there.

## Resource and dispatch budget

For one output, private temporal fallback storage consists of a padded R32Float-resolution buffer and two dense one-word-per-cell buffers (visited state and availability). Allocation dimensions and `N` use checked arithmetic. The combined scratch is limited to **256 MiB per output**; each binding is independently bounded by the admitted device and workload storage-buffer and buffer-size limits. Both the primary and the fallback dispatches respect the admitted RunenGPU 2D workgroup limit. Buffer identities reuse the existing `DeterministicResourceCache`; no per-frame GPU identifiers, ordinary readbacks or global GPU-idle wait are introduced.

After phase mapping injectivity, at most `N - M` new current-only scene/radiance queries can occur per output/frame. The shader still touches `N` availability cells to resolve the output: **worst-case cost approaches full-resolution direct radiance**. This is a correctness mechanism, not an optimization or a claimed speedup. The compatible path does not repeat nearest-hit or direct-radiance work for already resolved cells. Resource retirement remains owned by issue #72's later cutover.

## Required proof and acceptance

The source of validation truth is `TESTING.md` and the canonical/native Vulkan workflows. The independent tests in `src/proofs/requested_coverage.rs` compare the actual fallback WGSL and states against the maintained **full-requested-lattice, same-phase evaluator executed on GPU** across P075/P067/P050, odd and narrow extents, field miss/invalid conditions, phase continuity and counter reset. CPU mapping tests in `src/runtime/execution/tests.rs` explicitly falsify naive real-number injectivity. Existing P100 camera and renderer conformance must remain green. Neither preparation metadata nor CPU-only color calculations substitute for terminal actual GPU execution.

Closure requires an unchanged exact feature head, independent critical review, accepted-main validation, and the owner acceptance required by repository governance. No successful CI run alone authorizes merge.
