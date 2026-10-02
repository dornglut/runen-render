// Private P100 history revision 3: [stationary/display estimate, latest forward depth,
// display sample count, latest hit, raw latest phase radiance, actual hit point xyz].
// The four-phase stationary estimate has several anchors and is never motion-reprojected.
// One retained raw sample is eligible only when its anchor belongs to the current requested
// footprint and a fresh current-view primary query returns that exact represented point.
// Fresh maintained direct radiance must also match the retained value exactly. This establishes
// that the reused value is a valid current-view finite sample, including shadow visibility;
// depth/bounds only select a candidate. An accepted pair therefore has weight 1 + 1, while
// the newly retained motion lane always contains the raw current sample, never that pair.
// Exact equality deliberately rejects roundoff differences. It does not prove maximal reuse,
// a spatial tolerance, moving-object support or a general TAA/TAAU correspondence method.

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

@group(0) @binding(1)
var<storage, read_write> current_radiance_words: array<u32>;

@group(0) @binding(2)
var<storage, read> defined_words: array<u32>;

@group(0) @binding(3)
var<storage, read> current_depth_words: array<u32>;

@group(0) @binding(4)
var<storage, read> current_hit_words: array<u32>;

@group(0) @binding(5)
var<storage, read> previous_history_words: array<u32>;

@group(0) @binding(6)
var<storage, read_write> current_history_words: array<u32>;

@group(0) @binding(7)
var<storage, read> camera_words: array<u32>;

const CAMERA_DEPTH_ABSOLUTE_EPSILON: f32 = 0.001;
const CAMERA_DEPTH_RELATIVE_EPSILON: f32 = 0.001;
const INVALID_HISTORY_SAMPLE_COUNT: u32 = 4294967295u;

fn load_input_f32(index: u32) -> f32 {
    return bitcast<f32>(input_words[index]);
}

fn load_camera_f32(index: u32) -> f32 {
    return bitcast<f32>(camera_words[index]);
}

fn mul_camera3(base: u32, value: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(vec3<f32>(load_camera_f32(base), load_camera_f32(base + 1u), load_camera_f32(base + 2u)), value),
        dot(vec3<f32>(load_camera_f32(base + 3u), load_camera_f32(base + 4u), load_camera_f32(base + 5u)), value),
        dot(vec3<f32>(load_camera_f32(base + 6u), load_camera_f32(base + 7u), load_camera_f32(base + 8u)), value),
    );
}

fn current_origin() -> vec3<f32> {
    return vec3<f32>(load_input_f32(8u), load_input_f32(9u), load_input_f32(10u));
}

fn history_base(index: u32) -> u32 {
    return index * 8u;
}

fn current_output_index(index: u32) -> u32 {
    let x = index % input_words[1u];
    let y = index / input_words[1u];
    return y * input_words[3u] + x;
}

fn write_current(index: u32, radiance: f32, depth: f32, count: u32, hit: u32) {
    let base = history_base(index);
    current_history_words[base] = bitcast<u32>(radiance);
    current_history_words[base + 1u] = bitcast<u32>(depth);
    current_history_words[base + 2u] = count;
    current_history_words[base + 3u] = hit;
    // Slots 4..7 retain one coherent motion sample independently of the stationary estimate.
    current_history_words[base + 4u] = 2143289344u; // Invalid until a finite coherent anchor is written.
    current_history_words[base + 5u] = 0u;
    current_history_words[base + 6u] = 0u;
    current_history_words[base + 7u] = 0u;
}

fn write_current_motion_sample(index: u32, radiance: f32, point: vec3<f32>, valid: bool) {
    let base = history_base(index);
    current_history_words[base + 4u] = bitcast<u32>(radiance);
    if !valid || !finite_vec3(point) {
        return;
    }
    current_history_words[base + 5u] = bitcast<u32>(point.x);
    current_history_words[base + 6u] = bitcast<u32>(point.y);
    current_history_words[base + 7u] = bitcast<u32>(point.z);
}

fn matching_world_point(current_point: vec3<f32>, previous_point: vec3<f32>) -> bool {
    if !finite_vec3(current_point) || !finite_vec3(previous_point) {
        return false;
    }
    // Represented coordinates must agree exactly; no spatial tolerance certifies visibility.
    return all(current_point == previous_point);
}

// A candidate's value and geometry are independently evaluated in the current view.
// Outcome zero means one coherent retained sample was validated; every other outcome
// remains current-only. Diagnostic nearest lookup uses the same acceptance contract.
struct MotionCandidate {
    outcome: u32,
    previous_index: u32,
    projected_depth: f32,
    tolerance: f32,
    visible_point: vec3<f32>,
    validated_radiance: f32,
};

fn point_in_current_footprint(index: u32, point: vec3<f32>) -> bool {
    let local = mul_camera3(24u, point - current_origin());
    if !finite_vec3(local) || local.z >= 0.0 {
        return false;
    }
    let u = 0.5 + 0.5 * local.x / (-local.z * load_input_f32(20u) * load_input_f32(21u));
    let v = 0.5 - 0.5 * local.y / (-local.z * load_input_f32(20u));
    let cell = vec2<f32>(u * f32(input_words[22u]), v * f32(input_words[23u]));
    let x = f32(index % input_words[1u]);
    let y = f32(index / input_words[1u]);
    return finite_f32(cell.x) && finite_f32(cell.y)
        && cell.x >= x && cell.x < x + 1.0 && cell.y >= y && cell.y < y + 1.0;
}

fn motion_candidate(index: u32, point: vec3<f32>, hit: bool, nearest: bool) -> MotionCandidate {
    var result = MotionCandidate(1u, 4294967295u, 0.0, 0.0, vec3<f32>(0.0), 0.0);
    if !hit {
        return result;
    }
    let previous_origin = vec3<f32>(load_camera_f32(6u), load_camera_f32(7u), load_camera_f32(8u));
    let local = mul_camera3(9u, point - previous_origin);
    if !finite_vec3(local) || local.z >= 0.0 {
        result.outcome = 2u;
        return result;
    }
    let projected_x = local.x / (-local.z * load_camera_f32(21u) * load_camera_f32(22u));
    let projected_y = local.y / (-local.z * load_camera_f32(21u));
    let u = projected_x * 0.5 + 0.5;
    let v = 0.5 - projected_y * 0.5;
    if !finite_f32(u) || !finite_f32(v) || u < 0.0 || u >= 1.0 || v < 0.0 || v >= 1.0 {
        result.outcome = 3u;
        return result;
    }
    let width = input_words[22u];
    let height = input_words[23u];
    var cell = floor(vec2<f32>(u * f32(width), v * f32(height)));
    if nearest {
        let previous_phase = (input_words[24u] + 3u) % 4u;
        let phase_x = select(0.25, 0.75, previous_phase == 1u || previous_phase == 3u);
        let phase_y = select(0.25, 0.75, previous_phase >= 2u);
        cell = floor(vec2<f32>(u * f32(width) - phase_x + 0.5, v * f32(height) - phase_y + 0.5));
        if cell.x < 0.0 || cell.x >= f32(width) || cell.y < 0.0 || cell.y >= f32(height) {
            result.outcome = 3u;
            return result;
        }
    }
    result.previous_index = min(u32(cell.y), height - 1u) * width + min(u32(cell.x), width - 1u);
    let base = history_base(result.previous_index);
    if previous_history_words[base + 3u] == 0u {
        result.outcome = 4u;
        return result;
    }
    let previous_depth = bitcast<f32>(previous_history_words[base + 1u]);
    let previous_forward = vec3<f32>(load_camera_f32(18u), load_camera_f32(19u), load_camera_f32(20u));
    result.projected_depth = dot(point - previous_origin, previous_forward);
    result.tolerance = CAMERA_DEPTH_ABSOLUTE_EPSILON
        + CAMERA_DEPTH_RELATIVE_EPSILON * max(abs(result.projected_depth), abs(previous_depth));
    if !finite_f32(previous_depth) || !finite_f32(result.projected_depth)
        || abs(result.projected_depth - previous_depth) > result.tolerance {
        result.outcome = 5u;
        return result;
    }
    let count = previous_history_words[base + 2u];
    if count == 0u || count == INVALID_HISTORY_SAMPLE_COUNT {
        result.outcome = 6u;
        return result;
    }
    let radiance = bitcast<f32>(previous_history_words[base + 4u]);
    let anchor = vec3<f32>(
        bitcast<f32>(previous_history_words[base + 5u]),
        bitcast<f32>(previous_history_words[base + 6u]),
        bitcast<f32>(previous_history_words[base + 7u]),
    );
    if !finite_f32(radiance) || !finite_vec3(anchor) {
        result.outcome = 7u;
        return result;
    }
    if !point_in_current_footprint(index, anchor) {
        result.outcome = 8u;
        return result;
    }
    let direction = normalize_checked(anchor - current_origin());
    if !direction.valid {
        result.outcome = 9u;
        return result;
    }
    let visible_hit = nearest_hit(current_origin(), direction.value, 0u);
    if !visible_hit.valid || !visible_hit.found {
        result.outcome = 9u;
        return result;
    }
    result.visible_point = current_origin() + direction.value * visible_hit.t;
    if !point_in_current_footprint(index, result.visible_point)
        || !matching_world_point(anchor, result.visible_point) {
        result.outcome = 10u;
        return result;
    }
    let validated = direct_radiance(result.visible_point, visible_hit);
    result.validated_radiance = validated.value;
    // Exact value equality introduces no guessed lighting/visibility tolerance.
    if !validated.valid || !finite_f32(validated.value)
        || bitcast<u32>(validated.value) != bitcast<u32>(radiance) {
        result.outcome = 11u;
        return result;
    }
    result.outcome = 0u;
    return result;
}

// Thirty-two fixed cells are bounded delivery diagnostics, separate from history semantics.
// The first five include the investigated 640x480 cells; the others sample the floor.
fn diagnostic_cell(probe: u32) -> u32 {
    var point = vec2<u32>(0u);
    switch probe {
        case 0u: { point = vec2<u32>(223u, 257u); }
        case 1u: { point = vec2<u32>(213u, 247u); }
        case 2u: { point = vec2<u32>(177u, 337u); }
        case 3u: { point = vec2<u32>(224u, 257u); }
        case 4u: { point = vec2<u32>(212u, 247u); }
        default: { point = vec2<u32>(32u + ((probe - 5u) % 9u) * 64u, 360u + ((probe - 5u) / 9u) * 32u); }
    }
    let x = min(point.x * input_words[22u] / 640u, input_words[22u] - 1u);
    let y = min(point.y * input_words[23u] / 480u, input_words[23u] - 1u);
    return y * input_words[22u] + x;
}

fn write_motion_diagnostics(index: u32, point: vec3<f32>, depth: f32, radiance: f32,
    hit: bool, candidate: MotionCandidate, output: f32) {
    if camera_words[33u] == 0u {
        return;
    }
    for (var probe = 0u; probe < 32u; probe = probe + 1u) {
        if diagnostic_cell(probe) != index {
            continue;
        }
        let base = input_words[0u] * 8u + probe * 32u;
        current_history_words[base] = index;
        current_history_words[base + 1u] = candidate.previous_index;
        current_history_words[base + 2u] = input_words[24u];
        current_history_words[base + 3u] = (input_words[24u] + 3u) % 4u;
        current_history_words[base + 4u] = select(0u, 1u, hit);
        current_history_words[base + 5u] = bitcast<u32>(depth);
        current_history_words[base + 6u] = bitcast<u32>(radiance);
        current_history_words[base + 7u] = bitcast<u32>(output);
        current_history_words[base + 8u] = candidate.outcome;
        current_history_words[base + 9u] = bitcast<u32>(candidate.projected_depth);
        current_history_words[base + 10u] = bitcast<u32>(candidate.tolerance);
        current_history_words[base + 11u] = bitcast<u32>(point.x);
        current_history_words[base + 12u] = bitcast<u32>(point.y);
        current_history_words[base + 13u] = bitcast<u32>(point.z);
        if candidate.previous_index != 4294967295u {
            let previous = history_base(candidate.previous_index);
            current_history_words[base + 14u] = previous_history_words[previous + 3u];
            current_history_words[base + 15u] = previous_history_words[previous + 1u];
            current_history_words[base + 16u] = previous_history_words[previous];
            current_history_words[base + 17u] = previous_history_words[previous + 2u];
            current_history_words[base + 18u] = previous_history_words[previous + 4u];
            current_history_words[base + 19u] = previous_history_words[previous + 5u];
            current_history_words[base + 20u] = previous_history_words[previous + 6u];
            current_history_words[base + 21u] = previous_history_words[previous + 7u];
        }
        current_history_words[base + 22u] = bitcast<u32>(candidate.visible_point.x);
        current_history_words[base + 23u] = bitcast<u32>(candidate.visible_point.y);
        current_history_words[base + 24u] = bitcast<u32>(candidate.visible_point.z);
        current_history_words[base + 25u] = bitcast<u32>(candidate.validated_radiance);
        let nearest = motion_candidate(index, point, hit, true);
        current_history_words[base + 26u] = nearest.previous_index;
        current_history_words[base + 27u] = nearest.outcome;
        current_history_words[base + 28u] = bitcast<u32>(nearest.projected_depth);
        current_history_words[base + 29u] = bitcast<u32>(nearest.tolerance);
        var nearest_radiance = radiance;
        if nearest.previous_index != 4294967295u {
            let previous = history_base(nearest.previous_index);
            current_history_words[base + 30u] = previous_history_words[previous + 4u];
            if nearest.outcome == 0u {
                nearest_radiance = (radiance + bitcast<f32>(previous_history_words[previous + 4u])) * 0.5;
            }
        }
        current_history_words[base + 31u] = bitcast<u32>(nearest_radiance);
    }
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

    let output_index = current_output_index(sample_index);
    if defined_words[sample_index] == 0u {
        current_radiance_words[output_index] = 0u;
        write_current(sample_index, 0.0, 0.0, INVALID_HISTORY_SAMPLE_COUNT, 0u);
        write_motion_diagnostics(sample_index, vec3<f32>(0.0), 0.0, 0.0, false,
            MotionCandidate(12u, 4294967295u, 0.0, 0.0, vec3<f32>(0.0), 0.0), 0.0);
        return;
    }

    let current_radiance = bitcast<f32>(current_radiance_words[output_index]);
    let current_hit_base = output_index * 4u;
    let current_hit_word = min(current_hit_words[current_hit_base], 1u);
    let current_hit = current_hit_word != 0u;
    var current_depth = 0.0;
    if current_hit {
        current_depth = bitcast<f32>(current_depth_words[output_index]);
        if !finite_f32(current_depth) {
            current_radiance_words[output_index] = 0u;
            write_current(sample_index, 0.0, 0.0, INVALID_HISTORY_SAMPLE_COUNT, 0u);
            write_motion_diagnostics(sample_index, vec3<f32>(0.0), 0.0, 0.0, false,
                MotionCandidate(13u, 4294967295u, 0.0, 0.0, vec3<f32>(0.0), 0.0), 0.0);
            return;
        }
    }
    let current_sample = select(0.0, current_radiance, current_hit);
    var current_point = vec4<f32>(0.0);
    if current_hit {
        let point = vec3<f32>(
            bitcast<f32>(current_hit_words[current_hit_base + 1u]),
            bitcast<f32>(current_hit_words[current_hit_base + 2u]),
            bitcast<f32>(current_hit_words[current_hit_base + 3u]),
        );
        if !finite_vec3(point) || !finite_f32(current_sample) {
            current_radiance_words[output_index] = 0u;
            write_current(sample_index, 0.0, 0.0, INVALID_HISTORY_SAMPLE_COUNT, 0u);
            write_motion_diagnostics(sample_index, vec3<f32>(0.0), 0.0, 0.0, false,
                MotionCandidate(13u, 4294967295u, 0.0, 0.0, vec3<f32>(0.0), 0.0), 0.0);
            return;
        }
        current_point = vec4<f32>(point, 1.0);
    }

    let previous_available = camera_words[0u] != 0u;
    if !previous_available {
        current_radiance_words[output_index] = bitcast<u32>(current_sample);
        write_current(
            sample_index,
            current_sample,
            current_depth,
            1u,
            current_hit_word,
        );
        write_current_motion_sample(sample_index, current_sample, current_point.xyz, current_point.w != 0.0);
        return;
    }

    let pose_changed = camera_words[1u] != 0u;
    if pose_changed {
        let candidate = motion_candidate(sample_index, current_point.xyz, current_hit, false);
        var reconstructed = current_sample;
        var count = 1u;
        if candidate.outcome == 0u && camera_words[34u] == 0u {
            let previous = history_base(candidate.previous_index);
            let coherent_radiance = bitcast<f32>(previous_history_words[previous + 4u]);
            reconstructed = (current_sample + coherent_radiance) * 0.5;
            count = 2u;
        }
        current_radiance_words[output_index] = bitcast<u32>(reconstructed);
        write_current(sample_index, reconstructed, current_depth, count, current_hit_word);
        write_current_motion_sample(sample_index, current_sample, current_point.xyz, current_point.w != 0.0);
        write_motion_diagnostics(sample_index, current_point.xyz, current_depth, current_sample,
            current_hit, candidate, reconstructed);
        return;
    }

    // Same-pose reconstruction is a bounded four-phase estimator, separate from moving-camera
    // reprojection. A defined miss contributes the real radiance sample zero instead of erasing
    // foreground history from another phase.
    let same_pose_completed_frames = min(camera_words[23u], 4u);
    if same_pose_completed_frames == 0u {
        current_radiance_words[output_index] = bitcast<u32>(current_sample);
        write_current(
            sample_index,
            current_sample,
            current_depth,
            1u,
            current_hit_word,
        );
        write_current_motion_sample(sample_index, current_sample, current_point.xyz, current_point.w != 0.0);
        return;
    }

    let previous_base = history_base(sample_index);
    let previous_raw_count = previous_history_words[previous_base + 2u];
    if previous_raw_count == INVALID_HISTORY_SAMPLE_COUNT {
        current_radiance_words[output_index] = 0u;
        write_current(
            sample_index,
            0.0,
            current_depth,
            INVALID_HISTORY_SAMPLE_COUNT,
            current_hit_word,
        );
        write_current_motion_sample(sample_index, current_sample, current_point.xyz, current_point.w != 0.0);
        return;
    }
    let previous_radiance = bitcast<f32>(previous_history_words[previous_base]);
    let previous_count = min(previous_raw_count, 4u);
    if previous_count == 0u || !finite_f32(previous_radiance) {
        current_radiance_words[output_index] = bitcast<u32>(current_sample);
        write_current(
            sample_index,
            current_sample,
            current_depth,
            1u,
            current_hit_word,
        );
        write_current_motion_sample(sample_index, current_sample, current_point.xyz, current_point.w != 0.0);
        return;
    }

    if same_pose_completed_frames >= 4u && previous_count >= 4u {
        // Radiance is formed and must remain invariant. Refresh the separate raw coherent
        // sample and its geometric anchor for later moving-camera validation.
        current_radiance_words[output_index] = bitcast<u32>(previous_radiance);
        write_current(
            sample_index,
            previous_radiance,
            current_depth,
            previous_count,
            current_hit_word,
        );
        write_current_motion_sample(sample_index, current_sample, current_point.xyz, current_point.w != 0.0);
        return;
    }

    let weight = f32(previous_count);
    let reconstructed = (previous_radiance * weight + current_sample) / (weight + 1.0);
    current_radiance_words[output_index] = bitcast<u32>(reconstructed);
    write_current(
        sample_index,
        reconstructed,
        current_depth,
        min(previous_count + 1u, 4u),
        current_hit_word,
    );
    write_current_motion_sample(sample_index, current_sample, current_point.xyz, current_point.w != 0.0);
}
