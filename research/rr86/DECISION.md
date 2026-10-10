# RR86 — Decision proposal: retain replaceable geometry; reject universal SDF

**Status:** proposed negative architecture-migration disposition, subject to independent owner acceptance. Investigation authority: RunenRender #86. This is not a public API amendment, production implementation, or accepted ADR.

**Live reviewed refs (2026-10-10):** accepted main 0dd4f09978578a374d1c91c62254987c659a4f16; F3F production draft #85 head fe6d6ccb3c15e2172214cdb28525049f82f8e866; research draft #87 base still unchanged. Recheck refs before promotion.

## Decision and rationale

**Retain accepted #23 multi-representation private renderer architecture. Reject adopting mandatory SDF-first geometry and reject a literal all-SDF renderer.** No new public SDF ontology, second compositor, backend owner, permanent RenderUI-specific semantics, or RunenSDF dependency is justified.

The accepted F1 and #79 laws already determine all observable semantics. They permit multiple private representations and disposable realization caches. The findings below demonstrate that universal SDF min/max-as-true-metric gives *wrong geometry and wrong pixels* and that even a correctly certified field prepass is sometimes slower than indexed exact-region queries. They do **not** prove that all possible implicit evaluators are wrong; a correct SDF-assisted evaluator remains an optional private optimization candidate, not an architectural requirement.

**Private design guidance for the existing F3F #81 owner:** represent the logical neutral support as a *compositional region capability* with owner-frame-aware operations; dispatch certified exact geometry/coverage and optional signed-field pruning, without forcing all users through a field. This is conceptual architecture guidance, not a mandated public type or production algorithm and not an acceptance of incomplete #81.

**Decision type is negative migration, not positive alternate-backend adoption.** No compared new GPU renderer exists and no GPU performance advantage is claimed. We should not manufacture such a backend merely to reject an unproven replatforming idea. A later alternative implementation requires its own measured workload, bounded successor scope and exact-equivalence conformance; it may still be a good optimization.

## Source and semantic obligations

- Accepted RunenRender ARCHITECTURE.md lines 102–173 and source-neutral F1 #36 govern geometry: immutable composition, exact fills/strokes/paths and clipping, image *destination* rectangles, shaped glyph outlines, and neutral geometry independent of all paint alpha, resource-cache or atlas data.
- A group neutral child-support set C is the union of pre-shadow children in the group's **immediate-parent** frame. Every authored sibling shadow independently consumes that same C, never prior colored/blurred shadow pixels.
- Signed spread is true Euclidean disk dilation for s>0, identity for s=0 and disk erosion for s<0, including exact complete disappearance. Apply offset in the owner frame, then finite normalized separable Gaussian in that frame; the geometric effect envelope adds an owner-axis-aligned square of radius 3 sigma on each axis.
- Group structural clips intersect completed neutral support; group opacity and shadow-color alpha do not erase that support for an ancestor. Only after this owner's effects and clipping do ancestor affine transforms map the completed support. A parent-frame Gaussian cannot be replaced with an unchanged root-frame Gaussian after nonuniform scale or shear.
- Coverage is a normalized integral of spread-region *membership* within the finite Gaussian kernel (identity at sigma=0), not a transfer function of an SDF scalar. Painter order, globally correlated F3E 4×4 sample phase, linear-premultiplied source-over, once-only group opacity and RunenGPU execution remain existing owner authority.
- Exact/uncertain/unsupported must remain different states. Preflight work, recursion, peak-live CPU/GPU/in-flight memory and admissible precision **before caller target mutation**; no unknown-as-zero or speculative alpha.

These are *existing* F1/#79 guarantees; naming an internal region/field interface adds no public contract.

### Critical eligibility correction: empty and rank-degenerate Boolean regions

A primitive 1-Lipschitz field's *composed* zero-set cannot automatically certify its real region boundary. Example: subtracting a rectangle from itself produces the **empty** geometric region, but naïve max(d, -d) = |d| is zero all along the canceled rectangle boundary. A positive spread threshold at that ghost zero falsely admits shadow support. Two positive-area rectangles touching along a single edge produce **rank-degenerate empty filled support** under the accepted F1 law, while max(dA,dB) again has a ghost zero at the touching seam. Field certification must therefore require **independent valid-boundary/nonempty support evidence**; sign plus Lipschitz continuity alone is insufficient. All signed spreads of proven empty support stay empty, and exact-distance queries on an empty region are not numeric zero. The checked-in orthogonal reference now contains runnable negative assertions for both cases. This adds a mandatory capability precondition, **not** an exception allowing false coverage or an SDF-wide redesign.

## Independent evidence (bounded, not a native benchmark)

Corrected reference PR #87 head adfc3175e930acc6e1fe9ba72360caef68dfa428 passed independent hosted **reference push #38077578258** and **reference PR #38077581555**, executing all six Python source files, after correcting a previous unexecutable script (QA-01). Existing renderer canonical and software-Vulkan #38077582081 passed on the same head. **The Vulkan lane did not run an alternate SDF backend.** Previous claim that the uncorrected R3 script had run was false and is explicitly reconciled in #86 R11.

1. **Wrong CSG metric:** a correct signed zero-set under primitive min/max fails to retain true Euclidean boundary distances for overlap interiors, intersection exteriors and clips. In adversarial disk union erosion/intersection dilation, normalized finite Gaussian can produce **exact coverage 0 instead of 1** or **1 instead of 0** if the min/max field is treated as a true metric.
2. **Correct algebraic exceptions:** disk dilation distributes over union; disk erosion distributes over intersection. The opposite directions are not generally valid; eligibility must be per operator and direction.
3. **Wrong frame:** an exact transformed polygon convolved with the *unchanged* root Gaussian differed from correct parent-frame coverage by up to **0.08521 alpha** on six probes. A root AABB-as-membership test also changes geometric support.
4. **Clips and holes:** five orthogonal nested CSG cases; 44,850 deterministic signed-spread classifications, 86 wrong naïve min/max classifications, **38,969 definite certificates with zero observed errors**, and **5,881 uncertain requiring exact fallback**. Two independent finite-Gaussian integration methods agree to 1e-12 on those **zero-spread** cases only.
5. **Alternative exact indexing:** on one 64-leaf crosshatch with 4,352 exact boundary segments, an exact BVH reduced potential 3,660,032 exhaustive segment tests to 8,797 evaluated, matching all checked Euclidean distances to 1e-12. One Python-only run: indexed exact 841-query time **22.32 ms**; mandatory signed-field certificate plus BVH fallback **26.02–32.03 ms**. Smaller cases sometimes favored certification; **neither dominates**.
6. **Cost limits:** dense coordinate arrangement may create O(n²) cells and O(n³) cold work with recursive membership; recursive Boolean inclusion-exclusion can grow exponentially. A conservative preflight gate can reject useful content. Python wall times/tracemalloc do not measure native RunenRender CPU, GPU, total RSS or peak-live in-flight work.

## Semantic sufficiency matrix

Legend: **P** = evidence of restricted-case exactness in executed experiments or currently accepted subset, **C** = could be equivalent only with a separate certified evaluator (not yet proved end-to-end), **N** = representation alone lacks required semantics, **U** = full end-to-end capability unproved. All P claims are deliberately bounded.

| Property | A: current triangle/continuous | B: region/implicit DAG | C: SDF-first functional DAG | D: raster SDF/MSDF atlas authority | E: everything-SDF pipeline |
| --- | --- | --- | --- | --- | --- |
| Neutral membership, clips/holes | P admitted subset, U nested | P orthogonal oracle; C general | P sign in examples; C when certified | N for arbitrary subtexel/out-of-range shapes | N without separate region laws |
| Positive/negative **true** Euclidean spread | P root; U nested | C exact boundary witnesses; P restricted reference | N raw min/max; C exact refinement | N unconditional | N scalar-only |
| Finite Gaussian support **and** integrated coverage | P root; U nested | C independent integral in owner frame | N from field alone; C separate coverage | N as alpha-by-distance | N without normal paint/convolution |
| Ancestor affine + source-neutral support | P nested zero kernel; U nonzero | C with owning frames | C only with proven metric/frames | U resampling/range correctness | U |
| Text/image/gradient/group painter order | Existing F3E painter, not metric | Preserve existing painter | Preserve existing painter | N as painter replacement | N if genuinely field-only |
| Preflight, fail closed, cache & peak-live work | P subset, U complete F3F | U production | U production | U production | U |

A hypothetical 'all implicit geometry' is *not* rejected: it can be correct if it implements certified region membership, Euclidean morphology, parent-frame finite coverage and truthful bounded failure. It would no longer be a scalar SDF-only renderer, however.

## Resource/performance/migration comparison

| Candidate | Cold and query work | GPU uploads, cache, in-flight memory and incremental reuse | Architectural disposition |
| --- | --- | --- | --- |
| A retained implementation | Root F3F tested; nested nonzero kernel rejected | Existing GPU pipeline; nested full peak-live unresolved | **Retain ownership and current realization**; #81 must finish support |
| B implicit region graph + indexed boundary | Orthogonal BVH is fast on bounded probes; dense arrangement cold cost is poor | Unmeasured | Useful *private semantic/evaluation model*, not an adopted physical backend |
| C mandatory SDF-first with fallback | 13.1% uncertain on one corpus, 65.3% in one adverse crosshatch; exact BVH sometimes wins alone | Field maintenance/revalidation unmeasured | **Reject mandatory prepass**, allow eligible optional certificate |
| D raster-distance atlas | Resolution/range/feature-preservation proofs absent | Atlas eviction/repack/update/upload unmeasured | **Reject semantic authority**, permit future disposable cache tier only with evidence |
| E literal all-SDF | Signed distance has no image-color/gradient/ordered-blend semantics | Extra components needed to recreate current image pipeline | **Reject as simplification claim** |

No valid apples-to-apples RunenGPU performance conclusion can be derived: actual F1 equality, GPU time, upload traffic, field atlas bandwidth/range, CPU and GPU memory *peak including in-flight*, device/cache reconstruction, error tolerances, glyph repeats, rapidly changing paths, deep clips and shadows are all **unmeasured** for any proposed new backend. They remain future prerequisites for **positive backend selection**, and #81 separately must prove full current F3F behavior. Do not claim successful second-implementation Vulkan pixels.

## Primary-source prior art: license and backend boundaries

These references inform the architecture comparison; **none is incorporated, selected as a dependency, or accepted as an alternative F1 realization**. The recorded license facts are source-file observations, not a transitive-dependency compatibility or legal determination.

| Reference | Examined license evidence | Relevant implementation and authority boundary |
| --- | --- | --- |
| [RunenSDF](https://github.com/dornglut/runen-sdf/blob/main/Cargo.toml) | `GPL-3.0-only` in its workspace manifest; [LICENSE](https://github.com/dornglut/runen-sdf/blob/main/LICENSE), examined blob `f288702d2fa16d3cdf0035b15a9fcbc552cd88e7` | Host-neutral CPU signed-field queries; its [numerical contract](https://github.com/dornglut/runen-sdf/blob/main/docs/numerics.md) distinguishes signed values, conservative steps and proven exact metric. It owns neither RunenRender's source-neutral 2D effects nor RunenGPU work. |
| [Vello](https://github.com/linebender/vello/blob/main/ARCHITECTURE.md) | Workspace manifest `Apache-2.0 OR MIT`; [Apache](https://github.com/linebender/vello/blob/main/LICENSE-APACHE) blob `d9a10c0d8e868ebf8da0b3dc95bb0be634c34bfe`; [MIT](https://github.com/linebender/vello/blob/main/LICENSE-MIT) blob `48ff8e414b77650f3fd61c80e7df90cc354d5017` | Sparse Strips CPU/GPU pipelines preprocess paths on CPU; the GPU realization has wgpu/WebGL2 backends, with a separate experimental compute renderer. Neither backend replaces RunenGPU authority or proves F3F Euclidean nested morphology. |
| [msdfgen](https://github.com/Chlumsky/msdfgen) | [MIT LICENSE.txt](https://github.com/Chlumsky/msdfgen/blob/master/LICENSE.txt), examined blob `fbea359b250a2836df64322a3b38be572e8fee34` | C++ field generation for vector shapes/glyphs, finite texture resolution and distance range, with optional library extensions. Field generation and shader sampling are not a source-neutral Boolean-boundary/morphology/finite-Gaussian oracle. |
| [Skia](https://github.com/google/skia) | [LICENSE](https://github.com/google/skia/blob/main/LICENSE), examined blob `e32636d5668d51ca7a938ae6cef93c795c656d7c`: BSD-style redistribution and attribution/no-endorsement terms | Paths, images, paints, clipping and ordered composition are distinct concepts; a separately owned graphics implementation is prior art, not a RunenGPU-compatible replacement or an F1 conformance result. |

Version-specific third-party metadata, transitive license obligations, backend availability, maintenance, and integration cost **remain unassessed for any future dependency**. In particular, a license permission is neither semantic equivalence nor proof that a backend can be admitted through RunenGPU. This negative migration decision adds **zero** dependencies and retains accepted ownership.

## Concrete continuation for existing production #81 — no extra issue

At PR #85 head fe6d6ccb, source-neutral group traversal in src/runtime/execution_2d/shadow.rs uses SourceFrame.child_source: NeutralMesh and can propagate translated zero-kernel nested supports. Nonzero nested spread/sigma is explicitly rejected both in group_child_sources and src/runtime/execution_2d/mod.rs::admit_runs. Root continuous morphology and finite Gaussian in support.rs cannot magically turn a transformed, blurred completed group back into triangles for an ancestor. This is the exact, unresolved F3F-NS-02 gap.

#81 MUST: preserve pristine C and *completed child neutral support* as distinct values; represent a nested result compositionally (or with another equivalently certified private method) with its own effect frame, true morphological query, clip and support/coverage distinction; never feed an ancestor prior colored shadow pixels. Decide per operator whether true-distance is known, sign-only, algebraically exact or unsupported, and use certified fallback or typed precision/resource failure without target mutation. Reconcile both existing nested nonzero typed gates **only when** full implementation and Vulkan correctness exist. Bound graph depth, worst-case evaluation, Gaussian integration, transient masks, CPU/GPU work and in-flight peak. Independently test positive and negative signed spread, Gaussian, arbitrary ancestry, zero-alpha source/effects, offscreen casters, clip/opacity order, image destination, shaped-glyph geometry, exact phase/painter, replay and error paths. #81 cannot merge until its own complete proof, regardless of #86's outcome.

**No new redesign issue is needed** to retain #23's already-accepted multi-representation authority. A new production issue or ADR is required only if independent review proves a *new semantic/public ownership change* beyond #81, not simply because the internal implementation is no longer triangle-only.

## Decision review / explicit boundary

This document is a **proposed negative migration decision** to be independently reviewed and accepted in #86. It authorizes **no production changes**, is not a RunenSDF dependency proposal, and cannot close F3F #81 or parent #38. It rejects performance-motivated SDF-wide replatforming *without* inventing a matched GPU benchmark. #86 can close as a decision-complete **reject/defer SDF-wide adoption** investigation once its owner accepts the reviewed negative decision and the research evidence, even though a positive second backend remains unproved. Native GPU equivalence/peak-live measurements are not 'passed'; they are a gate to any later **positive** second-realization proposal and are a **separate, still-blocking #81** production obligation.
