# 2D composition contract

This document owns detailed immutable 2D semantics and physical realization requirements. See [ARCHITECTURE.md](../../ARCHITECTURE.md) for repository boundaries. The shadow law is accepted **semantic meaning**, not evidence of complete F3F execution.

## Ordinary group-shadow semantic law (F3F)

The immutable F1 group-shadow contract is source-neutral and independent of
the physical shadow renderer. This paragraph defines admitted *meaning*; until
F3F implementation acceptance, `execution_2d` continues to reject nonempty
shadow lists with a typed unsupported-content outcome.

For an attached group's **immediate-parent logical coordinate system**, let
`C` be the union of its direct children's neutral effect-support sets after
mapping them into that parent frame. An item's set is its transformed fill
geometry (respecting fill rule), centered stroke geometry (including caps and
joins), resolved image **destination** patch rectangles, or positioned shaped
glyph **outline** geometry, intersected with that item's conjunctive
immediate-parent-space clips. A child group's set includes its own child and
shadow supports after its own clips. These neutral sets are independent of
brush and foreground alpha, image texel alpha, item and group opacity, shadow
color alpha, atlas/MSDF sampling, and device/cache state. Intrinsic or physical
classes that cannot be represented remain typed capability failures, not
alpha-mask approximations. Empty or rank-degenerate filled geometry has empty
support, but transparent *color* never erases geometric support.

Each shadow in the ordered list independently consumes **the same**
pre-shadow set `C`; shadows do not derive geometry from previous shadows.
For signed finite spread `s`, its exact spread set is the Euclidean
Minkowski dilation `C ⊕ disk(s)` if `s > 0`, unchanged `C` if `s = 0`,
or the Euclidean disk erosion `C ⊖ disk(-s)` if `s < 0`.
Complete erosion has empty support. Apply the shadow's finite
`(offset_x, offset_y)` translation **in the group's immediate-parent
logical frame** after spread. For `sigma > 0`, the finite separable
Gaussian-style coverage blur has no support beyond three sigma per logical
axis; the **neutral support envelope** is the translated spread set
Minkowski-summed with the closed axis-aligned square
`[-3*sigma, 3*sigma] × [-3*sigma, 3*sigma]`. For `sigma = 0`,
blur is the identity. For `sigma > 0`, the reference alpha coverage is
the convolution of the translated spread set's indicator function with
separable, unit-integral one-dimensional kernels proportional to
`exp(-t*t/(2*sigma*sigma))` inside `[-3*sigma, 3*sigma]` and zero
outside. Normalization keeps fully covered interiors at unit coverage.
The square defines the cutoff/envelope, **not** the signed Euclidean
spread kernel. Bounded, deterministic physical blur weights inside that
cutoff may approximate this reference within independently established
quality/tolerance bounds; cached alpha is never propagated as the
neutral support envelope.

Effects are in the **immediate-parent** logical frame, *not* the
attached group's own local geometry frame. First map the children through
`group.local_to_parent` and then apply the parent-frame effect offset,
spread and blur. The group's clips also use that parent frame. An
**ancestor** affine subsequently maps the finished group color and neutral
support into its own frame, so transformed ancestor effects can become
non-axis-aligned in root space. For example, group-local `x=[0,1]`,
local-to-parent horizontal scale `2`, spread `+1`, and zero blur/
offset yield parent-space `x=[-1,3]`, **not** `[-2,4]`.

Paint shadow colors in their stable authored list order, with each later
shadow source-over the earlier translucent shadows in linear-premultiplied
space. Then compose the already-ordered child color **over** the completed
shadow colors; apply the group's conjunctive parent-space clips, then
its opacity **exactly once**, then source-over the entire atomic group
into its parent. Neutral output support propagated to an ancestor is the
union of `C` and every independently derived shadow envelope, intersected
with group clips and independent of group opacity and shadow color alpha.

Neutral support and complete finite effect extent MUST be derived **before**
early final-target culling. An invisible or fully offscreen primitive can
still affect the visible target through offset/spread/blur or through a
nested ancestor shadow. Private F3F implementations must preflight
geometry/precision, morphology, halo tiles, memory, GPU work and nested
intermediates before modifying caller-owned target content, retaining F3E's
single correlated 4×4 premultiplied composition and RunenGPU work evidence.
The implementation may choose disposable tessellation, masks, caches and
bounded approximations only where they preserve this semantic law.

The reusable semantic root is one lifetime-neutral immutable 2D composition
value. It represents resolved renderer meaning such as:

- exact painter and atomic-group order, with group opacity applied once;
- structural rectangle, rounded-rectangle, ellipse, and arbitrary path geometry,
  including explicit fill rule, centered stroke, cap, join, and miter semantics;
- item/group transforms, conjunctive clips, fills, strokes, and opacity;
- straight-alpha sRGB literal color meaning, gradient interpolation in
  premultiplied linear-sRGB, and ordered linear-light source-over composition;
- ordinary-shadow/effect meaning derived from neutral semantic support geometry
  rather than cached sampled source alpha, including effects from transparent
  child color;
- resolved image source/destination mapping plus immutable semantic resource
  identity;
- already-shaped glyph occurrences plus semantic shaped-resource identities
  whose immutable bindings carry exact font/face, variation, synthesis, size,
  and glyph facts required for realization.

This is a sibling renderer-semantic domain to the existing
surface/field/query representation protocols. Ordered 2D composition must not
be encoded as a `SurfaceQuery`, `FieldDistance`, display-list/GPU command
stream, source-framework paint type, or universal vector/field ontology merely
to reuse an existing physical path.

Source adapters project already-resolved renderer facts into RunenRender-owned
values. Source-framework publication/object identities, authoring policy, and
native presentation identity remain outside RunenRender semantic identity.

An ergonomic 2D authoring frontend MAY provide mutable builders, scoped group or
clip construction, and typed resource-binding conveniences only when the
resulting value is exactly the canonical immutable 2D composition/resource
model. Such authoring helpers are ephemeral construction state, not a second
composition authority, immediate GPU canvas, RunenUI replacement, or
source-framework display list.

One execution combines one immutable composition value with one compatible
immutable semantic-resource binding set and target/output fact set. Private
parallel preparation is permitted only when it cannot observe mutable source
state, alter painter/group order, or change semantic results.

## Admission and private realization

The 2D path preserves the same architectural law as the existing renderer
without reusing incompatible 3D semantic types:

```text
immutable 2D composition semantics
    -> semantic validation
    -> invocation-compatible resource and target binding
    -> semantically admissible private realizations
    -> private cost/quality/availability selection
    -> private compilation and retained derived state
    -> RunenGPU work
    -> source-neutral contribution/execution correlation
```

Semantic admissibility and physical policy are separate. Cache residency,
device capability, preprocessing cost, or a preferred implementation cannot
make a semantically incompatible realization admissible. A cache miss may
change cost or select another already-admitted equivalent realization; it must
not rewrite content, ordering, resource identity, color/composition meaning, or
text metrics.

A realization family is replaceable only within its admitted semantic
obligation. When one semantic class has a single initial realization, no second
implementation is required merely to demonstrate replaceability. A later
alternative must consume the same semantic input and prove the same admitted
semantics before policy may select it.

## Resources, text, and derived state

Semantic resource identity is distinct from every prepared or resident form:

```text
semantic resource identity
    -> immutable semantic binding / intrinsic representation
        -> private realization descriptor
            -> prepared/cache realization
                -> atlas / buffer / texture / RunenGPU residency
```

Cache keys may contain private reproducibility facts such as field quality,
range, generator revision, or tessellation parameters. Atlas coordinates,
cache buckets, device generations, and RunenGPU handles are never semantic
identity. Derived state must be reconstructible from immutable semantic facts or
fail with an owner-oriented structural/capability outcome.

Text is shaped before RunenRender. The renderer must not perform fallback,
line breaking, reshaping, font rediscovery, or metric authority. The initial
admitted text class is already-shaped scalable monochrome outline content with
supported exact synthesis. Its physical outline/field/coverage preparation is
renderer-private. Missing compatible derived state is reconstructed from the
immutable shaped resource before execution; another realization family is not
required solely for cache misses.

Intrinsic COLR, SVG, bitmap glyph content and unsupported synthesis remain
distinct capability classes and fail closed until an exact separately accepted
realization exists. They must not be silently flattened to monochrome outline
semantics.

The historical initial F2 production execution boundary is `execution_2d`.
It accepted a caller-owned RunenGPU color view plus logical canvas extent
and raster scale, admitted a bounded direct-root shaped-text subset, and returned
composable RunenGPU work wrapped with source-neutral execution correlation.
Retained semantic identity observations are separate from the private derived
field cache: cache discard may remove derived MSDF fields but cannot make an
already-observed resource identity legal to rebind. F2 retains reconstructible
CPU-side field data by semantic resource and quality tier while GPU
field textures, vertex buffers, sampler state, and draw bindings are
contribution-local disposable realization state. No global atlas or device
residency authority is introduced by this slice.

Solid vector items admit rectangles, canonical rounded rectangles, ellipses and
structural paths, both fill rules, centered strokes with caps/joins/miter,
affine transforms and item opacity. Private Lyon tessellation expands strokes
in local coordinates before transforming geometry. A contribution-local bounded
coverage surface resolves tessellation overlap before applying the semantic
brush/opacity once. Sampling, tolerances, mask layout and tessellation remain
private; unrepresentable precision or geometry/storage limits reject structurally.
Vector-only execution does not require shaped-field upload/filter capabilities.
Gradient brushes reuse the same coverage mask and per-item ordered contribution.
A private read-only buffer preserves authored stop order and premultiplied linear
RGB/alpha; the gradient compose pass integrates painted color at each covered
4×4 sample in local brush coordinates before applying item opacity once. Hard
stop and precision/storage limitations reject structurally rather than altering
immutable authored semantics. Shaped text retains F2's immutable outline font/shaping and
identity/translation geometry class. The F3E sample compositor applies
item and ancestor group opacity/clipping exactly once after rendering the
retained glyph coverage; non-translation text transforms and unsupported
intrinsic glyph classes remain fail-closed.

One ordered private lowering produces coverage/composition operations and
painter-ordered shaped-text draws. Consuming a prepared contribution appends the
complete sequence to caller-owned RunenGPU work, or authors a separate fragment.
Caller-owned `Render2dWorkBinding` keys connect prior target contents and the
resulting target output through RunenGPU typed import/export relationships.
These keys are graph wiring, not semantic identity or execution evidence.
Successful contribution evidence MUST require exact membership of **every** authored
work node in the same downstream submission plus its terminal `Completed` status.
Caller methods and opaque correlation types remain the same for single-node text
and multi-node mixed work. Non-painting content authors no synthetic node, output
or token; callers retain the prior target-content relationship in that case.
Legitimate no-work MUST NOT be represented by fabricated GPU work or identity.

The initial admitted image payload class is immutable tightly packed
unpremultiplied RGBA8 sRGB plus exact non-zero intrinsic extent and resolved
source/destination mapping. The F3C physical image path uses transient RGBA8-sRGB sampled textures uploaded
through RunenGPU. Its initial private nearest-texel policy reads only texels
intersecting each already-resolved source crop; fractional/affine destination
geometry is evaluated on the established 4×4 physical sample lattice. Source
sRGB is decoded to linear before premultiplication, and item opacity applies
once. Original source bytes and semantic identity are independent of uploads,
views and device residency. Sampling is private physical policy, not a new
source-framework fit/crop or public semantic filter flag; future alternatives
must preserve admitted F1 content meaning and accepted consumer evidence.
Resource budgets, GPU-format capabilities, and precision fail structurally
before contribution work is authored. Precision admission also bounds f32
subpixel loss from cancellation between large local coordinates and opposing
transforms, even when each scalar is individually representable. No shared image cache/residency backend
is introduced by this slice.

## Private realization boundary

Concrete implementation-library selection is not part of this semantic
contract. Implementations may privately use backend-neutral tessellation,
analytic coverage, signed-distance fields, direct curves, intrinsic sampled resources,
intermediate masks/targets, or other realization families when they preserve
the admitted semantic obligation and lower through RunenGPU. Concrete library,
algorithm, threshold, atlas, cache, batching, and pass-topology choices remain
replaceable implementation policy.

Adding, replacing, or deleting one private realization family does not require a
public semantic-contract change when the replacement preserves the same
admitted semantics. A new performance-driven selection requires evidence
appropriate to the claim; the architecture itself does not claim native-GPU
superiority for one family.

## Failure, evidence, and evolution

Caller-material failures remain owner-oriented and distinguish at least:

- malformed semantic composition;
- unsupported semantic or intrinsic capability;
- missing, incompatible, or unavailable semantic resource binding;
- valid content with no currently admitted realization;
- private realization/lowering failure;
- RunenShader compilation failure;
- RunenGPU preparation/submission/execution failure.

Native Present and source-publication failures remain downstream-owned.
Diagnostics identify the affected semantic/resource subject without exposing
private cache/backend identity as source truth.

RunenRender execution evidence identifies the admitted/executed renderer
contribution without depending on source-framework publication IDs or native
Present policy. For the initial F2 slice, that evidence is exact authored
RunenGPU work-node membership plus terminal completion, not labels, cache
identity, readback contents, or sentinel inference. The downstream combining
boundary owns the correspondence between that contribution, the source
publication, and terminal presentation.

While RunenRender is pre-1.0 and consumers are revision-pinned, a public 2D
semantic change requires explicit RunenRender authority, a validated accepted
revision, downstream compatibility review, and downstream repin/adaptation.
Do not introduce duplicate versioned types, compatibility aliases, runtime
schema-version fields, or persistence/wire machinery without a real
compatibility obligation.

The 2D semantic/compiler responsibility remains inside the RunenRender crate.
A separate public text/path/image renderer, `runen-render-2d` package, or
backend-specific public namespace would duplicate composition authority without
a demonstrated independent owner.

## Retained clipping and composition invariants

F3D established the retained clip invariant: independent binary 4×4 sample
masks are unioned within shapes and ANDed across the owner's conjunctive clips.
F3E reuses that exact private clipped-geometry/packed-mask implementation for
both items and groups. Clips remain in the owner's immediate-parent coordinate
system; a child's local transform never reapplies to its own clip.

F3E's one source-neutral compositor keeps premultiplied linear color at the
correlated samples through sibling painter order, image-patch ordering and
atomic nested-group composition. Item/image opacity is applied once to the
completed item; group opacity and clips are applied once to the completed group.
Only the final 16-sample root average source-overs the caller's original target.
A default identity group therefore has no separate image-formation law.
Nontranslation shaped-text transforms and ordinary shadows are not admitted.

The private implementation has bounded cropped/phase-aligned tiles, reusable
depth scratch, cumulative clip, tessellation, gradient/image-upload, shaped-field
and authored-work budgets, and typed capability, precision and resource
rejection before external mutation. Private MSDF cache residency is a
bounded, reconstructible successful-composition working set; immutable resource
identity observations remain separate from that derived cache. The separately
owned issue #72 governs long-lived identity retirement and continuous UI churn,
not a new semantic owner inside this compositor. No 2D display list,
backend-specific presentation surface or private GPU authority is introduced.
