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

// Renderer-private brush payload: four vec4 headers then two vec4s per stop.
// Header 0: kind, authored stop count, coverage origin x/y.
// Headers 1/2: inverse physical-to-local affine rows and item opacity.
// Header 3: linear start+direction or radial center+radius.
struct GradientPayload {
    header: array<vec4<f32>, 4>,
    stops: array<vec4<f32>>,
}
@group(0) @binding(1) var<storage, read> gradient: GradientPayload;

fn gradient_sample(physical: vec2<f32>) -> vec4<f32> {
    let lx = gradient.header[1];
    let ly = gradient.header[2];
    let local = vec2<f32>(
        dot(lx.xyz, vec3<f32>(physical, 1.0)),
        dot(ly.xyz, vec3<f32>(physical, 1.0))
    );
    let geometry = gradient.header[3];
    var t = 0.0;
    if (gradient.header[0].x == 1.0) {
        let delta = geometry.zw;
        t = dot(local - geometry.xy, delta) / dot(delta, delta);
    } else {
        t = distance(local, geometry.xy) / geometry.z;
    }
    t = clamp(t, 0.0, 1.0);
    let count = u32(gradient.header[0].y);
    var prior_offset = gradient.stops[0].x;
    var prior_color = gradient.stops[1];
    if (t <= prior_offset) {
        return prior_color;
    }
    for (var i = 1u; i < count; i += 1u) {
        let offset = gradient.stops[2u * i].x;
        let color = gradient.stops[2u * i + 1u];
        if (t <= offset) {
            if (t == offset) {
                return color;
            }
            let fraction = (t - prior_offset) / (offset - prior_offset);
            return mix(prior_color, color, fraction);
        }
        prior_offset = offset;
        prior_color = color;
    }
    return prior_color;
}
@fragment fn fs_gradient(input: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = floor(input.mask_pixel);
    let base = vec2<i32>(pixel) * COVERAGE_AXIS_SAMPLES;
    var result = vec4<f32>(0.0);
    for (var y = 0; y < COVERAGE_AXIS_SAMPLES; y += 1) {
        for (var x = 0; x < COVERAGE_AXIS_SAMPLES; x += 1) {
            let coverage = textureLoad(
                coverage_mask, base + vec2<i32>(x, y), 0
            ).r;
            if (coverage != 0.0) {
                let physical = gradient.header[0].zw + pixel
                    + (vec2<f32>(f32(x), f32(y)) + vec2<f32>(0.5)) /
                        f32(COVERAGE_AXIS_SAMPLES);
                result += gradient_sample(physical) * coverage;
            }
        }
    }
    return result * (gradient.header[1].w /
        f32(COVERAGE_AXIS_SAMPLES * COVERAGE_AXIS_SAMPLES));
}


// F3C physical policy: one nearest RGBA8-sRGB texel for each independently
// covered 4x4 physical sample. Mapping and patch order come from immutable F1.
// Source alpha is straight; textureLoad decodes sRGB to linear before premultiplication.
@group(0) @binding(2) var image_texture: texture_2d<f32>;
struct ImageParameters {
    header: array<vec4<f32>, 5>,
}
@group(0) @binding(3) var<storage, read> image_params: ImageParameters;

@fragment fn fs_image(input: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = floor(input.position.xy);
    let inverse_x = image_params.header[0];
    let inverse_y = image_params.header[1];
    let src = image_params.header[2];
    let dst = image_params.header[3];
    let image_size = vec2<i32>(textureDimensions(image_texture));
    let sample_min = clamp(vec2<i32>(floor(src.xy)), vec2<i32>(0), image_size - vec2<i32>(1));
    let sample_max = clamp(vec2<i32>(ceil(src.xy + src.zw)) - vec2<i32>(1),
        sample_min, image_size - vec2<i32>(1));
    var sum = vec4<f32>(0.0);
    for (var y = 0; y < COVERAGE_AXIS_SAMPLES; y += 1) {
        for (var x = 0; x < COVERAGE_AXIS_SAMPLES; x += 1) {
            let physical = pixel +
                (vec2<f32>(f32(x), f32(y)) + vec2<f32>(0.5)) / f32(COVERAGE_AXIS_SAMPLES);
            if (any(physical >= image_params.header[4].xy)) {
                continue;
            }
            let local = vec2<f32>(
                dot(inverse_x.xyz, vec3<f32>(physical, 1.0)),
                dot(inverse_y.xyz, vec3<f32>(physical, 1.0))
            );
            let relative = (local - dst.xy) / dst.zw;
            if (all(relative >= vec2<f32>(0.0)) &&
                all(relative < vec2<f32>(1.0))) {
                let source = src.xy + relative * src.zw;
                let texel = clamp(vec2<i32>(floor(source)), sample_min, sample_max);
                let straight_linear = textureLoad(image_texture, texel, 0);
                sum += vec4<f32>(straight_linear.rgb * straight_linear.a,
                    straight_linear.a);
            }
        }
    }
    return sum * (inverse_x.w / f32(COVERAGE_AXIS_SAMPLES * COVERAGE_AXIS_SAMPLES));
}
