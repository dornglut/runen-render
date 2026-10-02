@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

@group(0) @binding(1)
var<storage, read> current_words: array<u32>;

@group(0) @binding(2)
var<storage, read> defined_words: array<u32>;

@group(0) @binding(3)
var<storage, read_write> history_words: array<u32>;

@group(0) @binding(4)
var<storage, read_write> history_sample_counts: array<u32>;

const INVALID_HISTORY_SAMPLE_COUNT: u32 = 4294967295u;

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let group_index = workgroup.y * workgroups.x + workgroup.x;
    let evaluation_index = group_index * 64u + local_index;
    let evaluation_count = input_words[0u];
    if evaluation_index >= evaluation_count {
        return;
    }

    let evaluation_width = input_words[1u];
    let evaluation_height = input_words[2u];
    let evaluation_stride = input_words[3u];
    let evaluation_x = evaluation_index % evaluation_width;
    let evaluation_y = evaluation_index / evaluation_width;

    let requested_width = input_words[22u];
    let requested_height = input_words[23u];
    let phase = input_words[24u] % 4u;
    let phase_x = select(0.25, 0.75, phase == 1u || phase == 3u);
    let phase_y = select(0.25, 0.75, phase >= 2u);
    let requested_x = min(
        u32(floor((f32(evaluation_x) + phase_x) * f32(requested_width) / f32(evaluation_width))),
        requested_width - 1u,
    );
    let requested_y = min(
        u32(floor((f32(evaluation_y) + phase_y) * f32(requested_height) / f32(evaluation_height))),
        requested_height - 1u,
    );

    let history_stride = input_words[27u];
    let history_index = requested_y * history_stride + requested_x;
    if defined_words[evaluation_index] == 0u {
        // An undefined member of the bounded sequence invalidates this requested cell for the
        // entire compatible history generation. Later phases must not silently form a partial
        // estimate from the remaining samples.
        history_words[history_index] = 0u;
        history_sample_counts[history_index] = INVALID_HISTORY_SAMPLE_COUNT;
        return;
    }

    let retained_sample_count = history_sample_counts[history_index];
    if retained_sample_count == INVALID_HISTORY_SAMPLE_COUNT {
        // Fail closed until the owning history generation is recreated/reset.
        history_words[history_index] = 0u;
        return;
    }

    // One compatible global four-phase sequence is the finite estimator. History age is advanced
    // only after correlated submission completion, so age >= 4 means this history generation has
    // already formed its bounded estimate.
    if input_words[26u] >= 4u {
        return;
    }

    let current_index = evaluation_y * evaluation_stride + evaluation_x;
    let current = bitcast<f32>(current_words[current_index]);
    let sample_count = min(retained_sample_count, 4u);
    if sample_count == 0u {
        history_words[history_index] = bitcast<u32>(current);
    } else {
        let previous = bitcast<f32>(history_words[history_index]);
        let weight = f32(sample_count);
        let reconstructed = (previous * weight + current) / (weight + 1.0);
        history_words[history_index] = bitcast<u32>(reconstructed);
    }
    history_sample_counts[history_index] = min(sample_count + 1u, 4u);
}
