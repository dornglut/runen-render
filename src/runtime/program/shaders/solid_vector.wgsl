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


// F3D private sample bitset: RG carries the 16 clip sample bits for one
// physical pixel, in the same 4x4 order as vector/image coverage. The sampled
// mask is a derived disposable resource, not a new semantic clip authority.
@group(0) @binding(4) var clip_bits: texture_2d<f32>;
struct ClipParameters {
    origin_and_extent: vec4<f32>,
}
@group(0) @binding(5) var<storage, read> clip_params: ClipParameters;

fn clip_hit(pixel: vec2<i32>, sx: i32, sy: i32) -> f32 {
    let coordinates = pixel - vec2<i32>(clip_params.origin_and_extent.xy);
    let dimension = vec2<i32>(textureDimensions(clip_bits));
    if (any(coordinates < vec2<i32>(0)) || any(coordinates >= dimension)) {
        return 0.0;
    }
    let packed = textureLoad(clip_bits, coordinates, 0);
    let lo = u32(round(packed.r * 255.0));
    let hi = u32(round(packed.g * 255.0));
    let bits = lo | (hi << 8u);
    let mask = 1u << u32(sy * COVERAGE_AXIS_SAMPLES + sx);
    return select(0.0, 1.0, (bits & mask) != 0u);
}

@fragment fn fs_compose_clipped(input: VertexOutput) -> @location(0) vec4<f32> {
    let base = vec2<i32>(floor(input.mask_pixel)) * COVERAGE_AXIS_SAMPLES;
    var sum = 0.0;
    for (var y = 0; y < COVERAGE_AXIS_SAMPLES; y += 1) {
        for (var x = 0; x < COVERAGE_AXIS_SAMPLES; x += 1) {
            sum += textureLoad(coverage_mask, base + vec2<i32>(x, y), 0).r *
                clip_hit(vec2<i32>(floor(input.position.xy)), x, y);
        }
    }
    let alpha = input.color.a * (sum / f32(COVERAGE_AXIS_SAMPLES * COVERAGE_AXIS_SAMPLES));
    return vec4<f32>(input.color.rgb * alpha, alpha);
}

@fragment fn fs_gradient_clipped(input: VertexOutput) -> @location(0) vec4<f32> {
    let pixel = floor(input.mask_pixel);
    let base = vec2<i32>(pixel) * COVERAGE_AXIS_SAMPLES;
    var result = vec4<f32>(0.0);
    for (var y = 0; y < COVERAGE_AXIS_SAMPLES; y += 1) {
        for (var x = 0; x < COVERAGE_AXIS_SAMPLES; x += 1) {
            let coverage = textureLoad(
                coverage_mask, base + vec2<i32>(x, y), 0
            ).r;
            if (coverage != 0.0 && clip_hit(vec2<i32>(floor(input.position.xy)), x, y) != 0.0) {
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

@fragment fn fs_image_clipped(input: VertexOutput) -> @location(0) vec4<f32> {
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
            if (clip_hit(vec2<i32>(pixel), x, y) == 0.0) {
                continue;
            }
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


// F3E GPU proof and future physical cutover: composited colors remain in
// one linear-premultiplied 4x4 sample space until the *single* root resolve.
// The sample texture is private derived work, not a source representation.
@fragment fn fs_sample_fill(input: VertexOutput) -> @location(0) vec4<f32> {
    let alpha = input.color.a;
    return vec4<f32>(input.color.rgb * alpha, alpha);
}

@group(0) @binding(6) var sample_layer: texture_2d<f32>;
@fragment fn fs_sample_resolve(input: VertexOutput) -> @location(0) vec4<f32> {
    // The interpolated coordinate is local to the cropped sample tile.
    let base = vec2<i32>(floor(input.mask_pixel)) * COVERAGE_AXIS_SAMPLES;
    var sum = vec4<f32>(0.0);
    for (var y = 0; y < COVERAGE_AXIS_SAMPLES; y += 1) {
        for (var x = 0; x < COVERAGE_AXIS_SAMPLES; x += 1) {
            sum += textureLoad(sample_layer, base + vec2<i32>(x, y), 0);
        }
    }
    return sum / f32(COVERAGE_AXIS_SAMPLES * COVERAGE_AXIS_SAMPLES);
}


// F3E isolated-group source-over at the same correlated physical sample.
// Child content is already premultiplied and composited onto transparent
// scratch. Vertex alpha is the group opacity, applied ONCE to that result.
@fragment fn fs_sample_merge(input: VertexOutput) -> @location(0) vec4<f32> {
    let sample = textureLoad(sample_layer, vec2<i32>(floor(input.position.xy)), 0);
    return sample * input.color.a;
}


// F3E group clip is an exact 4x4 binary sample mask in the owner's
// immediate parent coordinates. Unlike a resolved fractional-pixel alpha,
// the packed bit masks multiply the *completed* isolated group at each
// correlated physical sample before once-only group opacity/source-over.
@fragment fn fs_sample_merge_clipped(input: VertexOutput) -> @location(0) vec4<f32> {
    let local_sample = vec2<i32>(floor(input.position.xy));
    // The clip is authored in the owner's parent space; the interpolated
    // coordinate carries the corresponding global 4x4 sample location.
    // Texture addressing remains tile-local, including nonzero tile origins.
    let global_sample = vec2<i32>(floor(input.mask_pixel));
    let logical = global_sample / COVERAGE_AXIS_SAMPLES;
    let offset = global_sample % COVERAGE_AXIS_SAMPLES;
    let visible = clip_hit(logical, offset.x, offset.y);
    let sample = textureLoad(sample_layer, local_sample, 0);
    return sample * (input.color.a * visible);
}


// F3E physical sample-plane union coverage, not independently averaged alpha.
// Each vector's tessellation triangles first write an idempotent 4x4 mask;
// one source-over draw per covered *sample* then applies the brush/item alpha.
@fragment fn fs_sample_mask_fill(input: VertexOutput) -> @location(0) vec4<f32> {
    let sample = vec2<i32>(floor(input.position.xy));
    let coverage = textureLoad(coverage_mask, sample, 0).r;
    let alpha = input.color.a * coverage;
    return vec4<f32>(input.color.rgb * alpha, alpha);
}


// F3E item-level clip intersects exact union coverage at each correlated
// sample before applying the item's opacity once. The coverage texture uses
// tile-local coordinates; clip bits use the immediate-parent/global 4x4 phase.
@fragment fn fs_sample_mask_fill_clipped(input: VertexOutput) -> @location(0) vec4<f32> {
    let local_sample = vec2<i32>(floor(input.position.xy));
    let global_sample = vec2<i32>(floor(input.mask_pixel));
    let logical = global_sample / COVERAGE_AXIS_SAMPLES;
    let offset = global_sample % COVERAGE_AXIS_SAMPLES;
    let membership = textureLoad(coverage_mask, local_sample, 0).r;
    let visible = clip_hit(logical, offset.x, offset.y);
    let alpha = input.color.a * membership * visible;
    return vec4<f32>(input.color.rgb * alpha, alpha);
}


// F3E gradients sample the same F1 brush-space interpolation once at each
// actual 4x4 physical sample. GPU payload stops are already premultiplied
// linear RGBA, so group isolation and source-over need no color reinterpretation.
@fragment fn fs_sample_gradient(input: VertexOutput) -> @location(0) vec4<f32> {
    let local_sample = vec2<i32>(floor(input.position.xy));
    let physical_sample = floor(input.mask_pixel);
    let global_pixel = (physical_sample + vec2<f32>(0.5)) / f32(COVERAGE_AXIS_SAMPLES);
    let membership = textureLoad(coverage_mask, local_sample, 0).r;
    let painted = gradient_sample(global_pixel);
    return painted * (membership * gradient.header[1].w);
}

@fragment fn fs_sample_gradient_clipped(input: VertexOutput) -> @location(0) vec4<f32> {
    let local_sample = vec2<i32>(floor(input.position.xy));
    let physical_sample = floor(input.mask_pixel);
    let global_pixel = (physical_sample + vec2<f32>(0.5)) / f32(COVERAGE_AXIS_SAMPLES);
    let global_sample = vec2<i32>(physical_sample);
    let logical = global_sample / COVERAGE_AXIS_SAMPLES;
    let offset = global_sample % COVERAGE_AXIS_SAMPLES;
    let membership = textureLoad(coverage_mask, local_sample, 0).r;
    let visible = clip_hit(logical, offset.x, offset.y);
    let painted = gradient_sample(global_pixel);
    return painted * (membership * visible * gradient.header[1].w);
}


// F3E immutable RGBA8 image patch at ONE correlated 4x4 sample. Every
// patch composites at its full source alpha into a private item layer; the
// image ITEM opacity and item/group clips are applied ONCE when that
// completed layer is merged into the parent sample plane.
@fragment fn fs_sample_image(input: VertexOutput) -> @location(0) vec4<f32> {
    let physical = (floor(input.mask_pixel) + vec2<f32>(0.5)) /
        f32(COVERAGE_AXIS_SAMPLES);
    if (any(physical >= image_params.header[4].xy)) {
        return vec4<f32>(0.0);
    }
    let inverse_x = image_params.header[0];
    let inverse_y = image_params.header[1];
    let src = image_params.header[2];
    let dst = image_params.header[3];
    let local = vec2<f32>(
        dot(inverse_x.xyz, vec3<f32>(physical, 1.0)),
        dot(inverse_y.xyz, vec3<f32>(physical, 1.0))
    );
    let relative = (local - dst.xy) / dst.zw;
    if (any(relative < vec2<f32>(0.0)) || any(relative >= vec2<f32>(1.0))) {
        return vec4<f32>(0.0);
    }
    let image_size = vec2<i32>(textureDimensions(image_texture));
    let sample_min = clamp(vec2<i32>(floor(src.xy)),
        vec2<i32>(0), image_size - vec2<i32>(1));
    let sample_max = clamp(vec2<i32>(ceil(src.xy + src.zw)) - vec2<i32>(1),
        sample_min, image_size - vec2<i32>(1));
    let source = src.xy + relative * src.zw;
    let texel = clamp(vec2<i32>(floor(source)), sample_min, sample_max);
    let straight_linear = textureLoad(image_texture, texel, 0);
    return vec4<f32>(
        straight_linear.rgb * straight_linear.a,
        straight_linear.a
    );
}
