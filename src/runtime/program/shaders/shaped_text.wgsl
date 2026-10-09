@group(0) @binding(0)
var field_texture: texture_2d<f32>;

@group(0) @binding(1)
var field_sampler: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) foreground: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) @invariant position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) foreground: vec4<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4<f32>(input.position, 0.0, 1.0);
    output.uv = input.uv;
    output.foreground = input.foreground;
    return output;
}

fn median3(a: f32, b: f32, c: f32) -> f32 {
    return max(min(a, b), min(max(a, b), c));
}

// One accepted F2 logical-pixel MSDF coverage rule. The sample-plane
// realization evaluates it at each ORIGINAL physical pixel center rather than
// changing smoothing derivatives by the 4x private color storage resolution.
fn msdf_coverage(sample: vec3<f32>, pixel_uv_width: vec2<f32>) -> f32 {
    let signed_distance = median3(sample.r, sample.g, sample.b) - 0.5;
    let texture_size = vec2<f32>(textureDimensions(field_texture, 0));
    let unit_range = vec2<f32>(4.0, 4.0) / texture_size;
    let screen_texture_size =
        1.0 / max(pixel_uv_width, vec2<f32>(0.000001, 0.000001));
    let screen_pixel_range = max(0.5 * dot(unit_range, screen_texture_size), 1.0);
    return clamp(screen_pixel_range * signed_distance + 0.5, 0.0, 1.0);
}

// F3E retains the original F2 logical-pixel MSDF estimate for all sixteen
// correlated subpixels. Structural clips and whole-item opacity are applied
// later, once, by the shared item/group compositor. The tile is aligned to
// full destination pixels, so local sample-phase modulo 4 is globally stable.
@fragment
fn fs_sample_projection(input: VertexOutput) -> @location(0) vec4<f32> {
    let sample_phase = vec2<i32>(floor(input.position.xy)) % vec2<i32>(4);
    let uv_dx = dpdx(input.uv);
    let uv_dy = dpdy(input.uv);
    let center_uv = input.uv
        + (1.5 - f32(sample_phase.x)) * uv_dx
        + (1.5 - f32(sample_phase.y)) * uv_dy;
    let msdf = textureSampleGrad(
        field_texture, field_sampler, center_uv, 4.0 * uv_dx, 4.0 * uv_dy
    ).rgb;
    let coverage = msdf_coverage(msdf, 4.0 * (abs(uv_dx) + abs(uv_dy)));
    let alpha = input.foreground.a * coverage;
    return vec4<f32>(input.foreground.rgb * alpha, alpha);
}
