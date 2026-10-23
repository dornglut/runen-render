# Retained headless scene-inspector example

The public-only [headless scene inspector](../../examples/headless_scene.rs) exercises
the retained ordinary API without Runenwerk, an ECS, native windows, or Present.
It constructs a renderer-owned sphere and plane, renders frame 001, commits a
sphere translation, retains the first immutable scene snapshot, then renders
frame 002 through the **same logical `RenderExecutionSession`**.

Each frame constructs a fresh request with request-owned opaque observation/output
handles for the same two semantic outputs: **spectral radiance at 550 nm** and
**object identity**. It forms one validated `RenderInvocation` after the physical
destinations are available, then explicitly prepares renderer work, submits it
through public RunenGPU, associates the exact prepared occurrence with that
submission, waits for terminal completion, reconciles the retained session,
and uses the resulting `AssociatedRenderOccurrence` for retained output
interpretation. Repeated numeric output positions across requests do not establish
output identity: both the request-owned output handle and the exact associated
occurrence/submission witness must match.

Run without writing files, or opt into locally inspectable diagnostic artifacts:

```sh
cargo run --example headless_scene
cargo run --example headless_scene -- --output ./runen-render-inspection
```

The second command writes:

```text
runen-render-inspection/
├── frame_001/
│   ├── radiance.png
│   ├── object_ids.png
│   └── evidence.json
├── frame_002/
│   ├── radiance.png
│   ├── object_ids.png
│   └── evidence.json
└── comparison.png
```

Both radiance PNGs use the same fixed 0.25 exposure over a *single* 550 nm
spectral-radiance lattice and display the result in grayscale; **this is not RGB
rendering**. `comparison.png` is a labeled side-by-side diagnostic of those
same grayscale mappings. The object-ID PNGs use one stable diagnostic mapping:
sphere=coral, plane=blue, undecoded=black. Those colors are not renderer
identity, and black is not a semantic background entity or definedness mask.

The retained radiance destination is the ordinary composable `R32Float`
carrier. Object identity remains on its `R32Uint` physical carrier. The first
frame's interpreted CPU diagnostics are retained before frame 002 may overwrite
those retained GPU destinations. Object identity is decoded through the exact
execution-local decoder from each associated occurrence; the example does not
reproduce private carrier rules.

Each `evidence.json` records the source scene revision, deterministic
request-local output positions (diagnostic ordering, not correlation identity),
exact associated-submission completion, carrier formats, adapter
backend, source-generation validation, visualization conventions, sphere
position, and renderer-owned temporal evidence. Persisting one
`RenderExecutionSession` does **not** imply temporal reuse: this example changes
the scene revision between frames, so the accepted temporal signature
invalidates prior history and frame 002 truthfully reports a history reset.

RunenRender also retains a separate **one-shot verified-result** workflow:
`submit_render_for_result` selects semantic-result verification before
submission and may later form a `RenderResult`. The retained example does not
form a `RenderResult`; its associated-occurrence radiance capture is maintained
physical output interpretation. These are distinct contracts and should not be
substituted for one another.

Failure semantics remain explicit. Dropping an unassociated prepared occurrence
abandons its provisional retained transition; association rejects submissions
missing the exact renderer-authored work; pending or failed associated
submissions cannot mint usable capture/decoder authority; and a newer completed
write invalidates stale retained-output interpretation. The caller continues to
own RunenGPU scheduling and optional readback submission.

PNG, JSON, comparison layout, labels, and filesystem persistence remain
example-local executable policy. They do not add image, artifact, or persistence
authority to RunenRender's production API.

The GPU-required Vulkan CI lane executes this retained two-frame public-API
conformance path. Without a suitable GPU, the executable reports the missing
adapter rather than pretending a render succeeded.

