// Phase-aligned current-only availability resolve for sub-native static temporal outputs.
//
// Only a positively retained count may produce a compatible static mean.
// A primary sample at this phase is carried separately from the retained mean,
// and every other unresolved cell receives at most one *current* nearest-hit /
// direct-radiance query. Unresolved output words are a signaling NaN, never black.
// A physical sample is not a certified footprint integral or a history contributor.

@group(0) @binding(0)
var<storage, read> input_words: array<u32>;

@group(0) @binding(1)
var<storage, read> history_words: array<u32>;

@group(0) @binding(2)
var<storage, read> history_sample_counts: array<u32>;

@group(0) @binding(3)
var<storage, read_write> resolved_words: array<u32>;

@group(0) @binding(4)
var<storage, read> phase_visited_words: array<u32>;

@group(0) @binding(5)
var<storage, read_write> availability_words: array<u32>;

const INVALID_HISTORY_SAMPLE_COUNT: u32 = 4294967295u;
const UNRESOLVED_RADIANCE_BITS: u32 = 2143289344u;

fn unresolved(physical: u32, cell: u32) {
    resolved_words[physical] = UNRESOLVED_RADIANCE_BITS;
    availability_words[cell] = 0u;
}

fn current_phase_radiance(x: u32, y: u32) -> ScalarEvaluation {
    let phase = input_words[24u] % 4u;
    let phase_x = select(0.25, 0.75, phase == 1u || phase == 3u);
    let phase_y = select(0.25, 0.75, phase >= 2u);
    let requested_width = f32(input_words[22u]);
    let requested_height = f32(input_words[23u]);
    let u = (f32(x) + phase_x) / requested_width;
    let v = (f32(y) + phase_y) / requested_height;
    let local = vec3<f32>(
        (2.0 * u - 1.0) * load_f32(20u) * load_f32(21u),
        (1.0 - 2.0 * v) * load_f32(20u),
        -1.0,
    );
    let origin = vec3<f32>(load_f32(8u), load_f32(9u), load_f32(10u));
    let direction = normalize_checked(mul3(11u, local));
    if !finite_vec3(origin) || !direction.valid {
        return ScalarEvaluation(false, 0.0);
    }
    let primary = nearest_hit(origin, direction.value, 0u);
    if !primary.valid {
        return ScalarEvaluation(false, 0.0);
    }
    // Same-ray KnownBackground is defined zero only under this exact maintained
    // evaluator's no-environment / no-hit-zero policy.
    if !primary.found {
        return ScalarEvaluation(true, 0.0);
    }
    let position = origin + direction.value * primary.t;
    let forward = normalize_checked(mul3(11u, vec3<f32>(0.0, 0.0, -1.0)));
    let depth = dot(position - origin, forward.value);
    if !finite_vec3(position) || !forward.valid || !finite_f32(depth) {
        return ScalarEvaluation(false, 0.0);
    }
    // Hit(depth) is visibility alone: illuminate via the *same* shared
    // renderer-owned direct_radiance implementation used by the primary.
    let evaluated = direct_radiance(position, primary);
    return ScalarEvaluation(evaluated.valid && finite_f32(evaluated.value), evaluated.value);
}

@compute @workgroup_size(64)
fn main(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(num_workgroups) workgroups: vec3<u32>,
    @builtin(local_invocation_index) local_index: u32,
) {
    let group_index = workgroup.y * workgroups.x + workgroup.x;
    let cell = group_index * 64u + local_index;
    let width = input_words[22u];
    let height = input_words[23u];
    if cell >= width * height {
        return;
    }
    let x = cell % width;
    let y = cell / width;
    let physical = y * input_words[27u] + x;
    let visited = phase_visited_words[cell];
    let count = history_sample_counts[physical];

    // A failed current-phase primary has already disproved availability;
    // do not evaluate it twice or rescue invalidated historical radiance.
    if visited == 1u || visited > 2u {
        unresolved(physical, cell);
        return;
    }

    // The current reconstruction can seed count=1 in this very submission.
    // That value is a current physical sample, not previously completed
    // compatible history. Multi-phase means (count>=2) remain retained
    // static estimator output when the current primary also contributes.
    if visited == 2u && (count == 1u || count == INVALID_HISTORY_SAMPLE_COUNT) {
        let current = bitcast<f32>(resolved_words[physical]);
        if finite_f32(current) {
            availability_words[cell] = 2u; // Provisional current-phase sample.
        } else {
            unresolved(physical, cell);
        }
        return;
    }

    if count != 0u && count != INVALID_HISTORY_SAMPLE_COUNT {
        let retained = bitcast<f32>(history_words[physical]);
        if finite_f32(retained) {
            resolved_words[physical] = bitcast<u32>(retained);
            availability_words[cell] = 1u; // Compatible finite static estimate.
            return;
        }
        // Invalid numerical history is not certified radiance. Fall through
        // to an actual current sample or same-phase current-only query.
    }

    if visited == 2u {
        // The current evaluated sample was scattered by the temporal pass.
        // It is truthful even when an earlier phase permanently invalidated
        // that generation's *aggregate* estimator.
        let current = bitcast<f32>(resolved_words[physical]);
        if finite_f32(current) {
            availability_words[cell] = 2u; // Provisional current-phase sample.
        } else {
            unresolved(physical, cell);
        }
        return;
    }

    // Only a cell *not sampled by the primary phase* reaches this query.
    // The evaluated result never modifies history or its phase/sample count.
    let evaluated = current_phase_radiance(x, y);
    if evaluated.valid {
        resolved_words[physical] = bitcast<u32>(evaluated.value);
        availability_words[cell] = 2u; // Provisional current-phase sample.
    } else {
        unresolved(physical, cell);
    }
}
