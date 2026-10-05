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

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureSample(field_texture, field_sampler, input.uv).rgb;
    let signed_distance = median3(sample.r, sample.g, sample.b) - 0.5;
    let texture_size = vec2<f32>(textureDimensions(field_texture, 0));
    let unit_range = vec2<f32>(4.0, 4.0) / texture_size;
    let screen_texture_size =
        1.0 / max(fwidth(input.uv), vec2<f32>(0.000001, 0.000001));
    let screen_pixel_range = max(0.5 * dot(unit_range, screen_texture_size), 1.0);
    let coverage = clamp(screen_pixel_range * signed_distance + 0.5, 0.0, 1.0);
    let alpha = input.foreground.a * coverage;
    return vec4<f32>(input.foreground.rgb * alpha, alpha);
}
