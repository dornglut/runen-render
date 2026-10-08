// Private ABI: position, mask-local pixel coordinate, linear straight color.
// Coverage is an idempotent union on a 4x4 regular sample lattice, never alpha.
const COVERAGE_AXIS_SAMPLES: i32 = 4;
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) mask_pixel: vec2<f32>,
    @location(2) color: vec4<f32>,
}
struct VertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) mask_pixel: vec2<f32>,
    @location(1) color: vec4<f32>,
}
@vertex fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.mask_pixel = input.mask_pixel;
    output.color = input.color;
    return output;
}
@fragment fn fs_coverage() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0);
}
@group(0) @binding(0) var coverage_mask: texture_2d<f32>;
@fragment fn fs_compose(input: VertexOutput) -> @location(0) vec4<f32> {
    let base = vec2<i32>(floor(input.mask_pixel)) * COVERAGE_AXIS_SAMPLES;
    var sum = 0.0;
    for (var y = 0; y < COVERAGE_AXIS_SAMPLES; y += 1) {
        for (var x = 0; x < COVERAGE_AXIS_SAMPLES; x += 1) {
            sum += textureLoad(coverage_mask, base + vec2<i32>(x, y), 0).r;
        }
    }
    let alpha = input.color.a * (sum / f32(COVERAGE_AXIS_SAMPLES * COVERAGE_AXIS_SAMPLES));
    return vec4<f32>(input.color.rgb * alpha, alpha);
}
