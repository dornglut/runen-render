@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output_words: array<u32>;

@group(0) @binding(2)
var<storage, read_write> defined_words: array<u32>;

@group(0) @binding(3)
var<storage, read_write> status_words: array<u32>;

@group(0) @binding(4)
var<storage, read_write> current_depth_words: array<u32>;

@group(0) @binding(5)
var<storage, read_write> current_hit_words: array<u32>;

fn observation_origin() -> vec3<f32> {
    return vec3<f32>(load_f32(8u), load_f32(9u), load_f32(10u));
}

fn sample_direction(index: u32) -> NormalizedDirection {
    if input_words[7u] == 2u {
        return normalize_checked(mul3(11u, vec3<f32>(0.0, 0.0, -1.0)));
    }

    let width = f32(input_words[1u]);
    let height = f32(input_words[2u]);
    let x = index % input_words[1u];
    let y = index / input_words[1u];
    var u = (f32(x) + 0.5) / width;
    var v = (f32(y) + 0.5) / height;
    if input_words[7u] == 3u {
        let phase = input_words[24u] % 4u;
        let phase_x = select(0.25, 0.75, phase == 1u || phase == 3u);
        let phase_y = select(0.25, 0.75, phase >= 2u);
        let requested_width = input_words[22u];
        let requested_height = input_words[23u];
        let requested_x = min(
            u32(floor((f32(x) + phase_x) * f32(requested_width) / width)),
            requested_width - 1u,
        );
        let requested_y = min(
            u32(floor((f32(y) + phase_y) * f32(requested_height) / height)),
            requested_height - 1u,
        );
        u = (f32(requested_x) + phase_x) / f32(requested_width);
        v = (f32(requested_y) + phase_y) / f32(requested_height);
    }
    let local = vec3<f32>(
        (2.0 * u - 1.0) * load_f32(20u) * load_f32(21u),
        (1.0 - 2.0 * v) * load_f32(20u),
        -1.0,
    );
    return normalize_checked(mul3(11u, local));
}

fn observation_forward() -> NormalizedDirection {
    return normalize_checked(mul3(11u, vec3<f32>(0.0, 0.0, -1.0)));
}

fn physical_output_index(index: u32) -> u32 {
    if input_words[7u] == 2u {
        return 0u;
    }
    let x = index % input_words[1u];
    let y = index / input_words[1u];
    return y * input_words[3u] + x;
}

fn invalidate(output_index: u32, sample_index: u32) {
    output_words[output_index] = 0u;
    defined_words[sample_index] = 0u;
    status_words[sample_index] = 1u;
    current_depth_words[output_index] = 0u;
    current_hit_words[output_index * 4u] = 0u;
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let group_index = workgroup.y * workgroups.x + workgroup.x;
    let sample_index = group_index * 64u + local_index;
    if sample_index >= input_words[0u] {
        return;
    }

    let execution_mode = input_words[6u];
    let output_index = physical_output_index(sample_index);
    let origin = observation_origin();
    let direction = sample_direction(sample_index);
    if execution_mode == 4u {
        // Renderer-private requested coverage. Always write one explicit terminal state for this
        // cell; padding is not a cell. Never access ordinary status/depth/hit scratch bindings.
        output_words[output_index] = 0u;
        defined_words[sample_index] = 0u; // Invalid
        if !finite_vec3(origin) || !direction.valid {
            return;
        }
        let primary = nearest_hit(origin, direction.value, 0u);
        if !primary.valid {
            return;
        }
        if !primary.found {
            defined_words[sample_index] = 1u; // KnownBackground
            return;
        }
        let forward = observation_forward();
        let position = origin + direction.value * primary.t;
        let depth = dot(position - origin, forward.value);
        if !forward.valid || !finite_vec3(position) || !finite_f32(depth) {
            return;
        }
        output_words[output_index] = bitcast<u32>(depth);
        defined_words[sample_index] = 2u; // Hit(finite current observation-forward depth)
        return;
    }
    if !finite_vec3(origin) || !direction.valid {
        invalidate(output_index, sample_index);
        return;
    }

    let hit = nearest_hit(origin, direction.value, 0u);
    if !hit.valid {
        invalidate(output_index, sample_index);
        return;
    }

    if execution_mode == 1u {
        if !hit.found {
            output_words[output_index] = bitcast<u32>(0.0);
            defined_words[sample_index] = 1u;
            current_depth_words[output_index] = 0u;
            current_hit_words[output_index * 4u] = 0u;
            return;
        }
        let position = origin + direction.value * hit.t;
        let forward = observation_forward();
        let depth = dot(position - origin, forward.value);
        if !finite_vec3(position) || !forward.valid || !finite_f32(depth) {
            invalidate(output_index, sample_index);
            return;
        }
        let radiance = direct_radiance(position, hit);
        if !radiance.valid {
            invalidate(output_index, sample_index);
            return;
        }
        output_words[output_index] = bitcast<u32>(radiance.value);
        defined_words[sample_index] = 1u;
        current_depth_words[output_index] = bitcast<u32>(depth);
        let hit_base = output_index * 4u;
        current_hit_words[hit_base] = 1u;
        current_hit_words[hit_base + 1u] = bitcast<u32>(position.x);
        current_hit_words[hit_base + 2u] = bitcast<u32>(position.y);
        current_hit_words[hit_base + 3u] = bitcast<u32>(position.z);
        return;
    }

    if execution_mode == 2u {
        if !hit.found {
            return;
        }
        let forward = observation_forward();
        if !forward.valid {
            invalidate(output_index, sample_index);
            return;
        }
        let position = origin + direction.value * hit.t;
        let depth = dot(position - origin, forward.value);
        if !finite_vec3(position) || !finite_f32(depth) {
            invalidate(output_index, sample_index);
            return;
        }
        output_words[output_index] = bitcast<u32>(depth);
        defined_words[sample_index] = 1u;
        return;
    }

    if execution_mode == 3u {
        if hit.found {
            output_words[output_index] = hit.code;
            defined_words[sample_index] = 1u;
        }
        return;
    }

    invalidate(output_index, sample_index);
}
