# Renderer semantic integration contract

These detailed integration laws are summarized by [ARCHITECTURE.md](../../ARCHITECTURE.md). Public Rust contracts and tests establish current behavior.

## Public integration laws

The normal game-engine consumer is a source adapter or engine integration layer
that projects source-owned world, asset, UI, camera, and product facts into
RunenRender-owned semantic contracts. Standalone/framework consumers construct
the same canonical renderer semantics directly; they do not use a second
standalone semantic model. RunenRender MUST NOT read live mutable RunenECS,
RunenUI, Runenwerk, native-window, or product-frame state.

One bounded ordinary renderer invocation may aggregate:

```text
immutable RenderSceneSnapshot
+ RenderRequest
+ typed request-scoped semantic input bindings
+ current representation availability facts
+ physical output bindings
```

The aggregate is invocation plumbing, not a new semantic authority or global
revision. Scene meaning, request meaning, foreign semantic inputs, availability,
and physical destinations remain distinguishable. Current RunenGPU context and
device facts remain outside the aggregate until capability-dependent execution
admission.

Semantic planning remains a public responsibility and advanced inspection
surface; its public contract describes conditional renderer-semantic choices,
not one permanent materialized storage layout or scan algorithm. Planning MUST
remain free to use indexing, culling, sharing, lazy representation, or other
semantics-preserving scale strategies. Public API design MUST NOT require an
all-object × all-method materialization merely because the founding planner
currently uses one.

Observation/output correlation on the ordinary path MUST use request-owned
opaque correlation or an equivalently strong typed construction contract. Raw
collection positions may remain an internal ordered representation, but a
physical output binding, finite evaluation selection, prepared-output lookup,
capture request, or result correlation MUST NOT silently select a different
semantic output merely because two requests reuse the same numeric index.

Large immutable request-scoped semantic payloads MUST be cheap to retain across
binding normalization, admission, prepared execution, and result/evidence
formation. Public semantics do not prescribe `Arc` or another concrete storage
type, but ordinary ownership transfer MUST NOT require repeated deep copies of
large immutable sample arrays.

Caller-material ordinary failures MUST expose stable machine-actionable
owner-level outcomes. Callers MUST NOT need to parse diagnostic strings or
depend on private implementation error types to distinguish invalid invocation
facts, unavailable representations, unsupported semantics, target/capability
mismatch, private realization failure, RunenShader failure, or RunenGPU
execution failure.

Pre-1.0 public corrections use clean revision-pinned cutovers when no concrete
compatibility obligation exists. Historical overloads, raw-scope helpers, and
compatibility aliases MUST NOT remain solely to avoid updating repository-owned
consumers.

