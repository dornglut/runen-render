// Renderer-private scene query and direct-light evaluation shared by evaluation and history validation.
struct Hit {
    valid: bool,
    found: bool,
    t: f32,
    code: u32,
    normal_scene: vec3<f32>,
    reflectance: f32,
};

struct NormalizedDirection {
    valid: bool,
    value: vec3<f32>,
};

struct ScalarEvaluation {
    valid: bool,
    value: f32,
};

struct FieldEvaluation {
    valid: bool,
    inside: bool,
    value: f32,
};

struct RayInterval {
    valid: bool,
    found: bool,
    t_min: f32,
    t_max: f32,
};

const FIELD_MAX_STEPS: u32 = 128u;
const FIELD_DIRECTION_EPSILON: f32 = 0.0000001;
const FIELD_COORDINATE_EPSILON: f32 = 0.0001;
const FIELD_NUMERIC_HIT_EPSILON: f32 = 0.00001;

const F32_EXPONENT_MASK: u32 = 2139095040u;

fn finite_f32(value: f32) -> bool {
    return (bitcast<u32>(value) & F32_EXPONENT_MASK) != F32_EXPONENT_MASK;
}

fn finite_vec3(value: vec3<f32>) -> bool {
    return finite_f32(value.x) && finite_f32(value.y) && finite_f32(value.z);
}

fn load_f32(index: u32) -> f32 {
    return bitcast<f32>(input_words[index]);
}

fn miss() -> Hit {
    return Hit(true, false, 0.0, 0u, vec3<f32>(0.0), 0.0);
}

fn invalid_hit() -> Hit {
    return Hit(false, false, 0.0, 0u, vec3<f32>(0.0), 0.0);
}

fn normalize_checked(value: vec3<f32>) -> NormalizedDirection {
    let magnitude_squared = dot(value, value);
    if !finite_f32(magnitude_squared) || magnitude_squared <= 0.0 {
        return NormalizedDirection(false, vec3<f32>(0.0));
    }
    let normalized = value / sqrt(magnitude_squared);
    return NormalizedDirection(finite_vec3(normalized), normalized);
}

fn mul3(base: u32, value: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(vec3<f32>(load_f32(base), load_f32(base + 1u), load_f32(base + 2u)), value),
        dot(vec3<f32>(load_f32(base + 3u), load_f32(base + 4u), load_f32(base + 5u)), value),
        dot(vec3<f32>(load_f32(base + 6u), load_f32(base + 7u), load_f32(base + 8u)), value),
    );
}

fn geometry_base(index: u32) -> u32 {
    return 30u + index * 40u;
}

fn emitter_base(index: u32) -> u32 {
    return input_words[29u] + index * 4u;
}

fn to_local_point(base: u32, point_scene: vec3<f32>) -> vec3<f32> {
    let translation = vec3<f32>(
        load_f32(base + 13u),
        load_f32(base + 14u),
        load_f32(base + 15u),
    );
    return mul3(base + 4u, point_scene - translation);
}

fn to_local_direction(base: u32, direction_scene: vec3<f32>) -> vec3<f32> {
    return mul3(base + 4u, direction_scene);
}

fn normal_to_scene(base: u32, normal_local: vec3<f32>) -> NormalizedDirection {
    return normalize_checked(mul3(base + 16u, normal_local));
}

fn intersect_sphere(base: u32, origin_scene: vec3<f32>, direction_scene: vec3<f32>) -> Hit {
    let origin = to_local_point(base, origin_scene);
    let direction = to_local_direction(base, direction_scene);
    if !finite_vec3(origin) || !finite_vec3(direction) {
        return invalid_hit();
    }

    let center = vec3<f32>(
        load_f32(base + 25u),
        load_f32(base + 26u),
        load_f32(base + 27u),
    );
    let radius = load_f32(base + 28u);
    let relative = origin - center;
    let a = dot(direction, direction);
    let b = 2.0 * dot(relative, direction);
    let c = dot(relative, relative) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;
    if !finite_f32(a) || a <= 0.0 || !finite_f32(discriminant) {
        return invalid_hit();
    }
    if discriminant < 0.0 {
        return miss();
    }

    let root = sqrt(discriminant);
    let denominator = 2.0 * a;
    let first = (-b - root) / denominator;
    let second = (-b + root) / denominator;
    if !finite_f32(first) || !finite_f32(second) {
        return invalid_hit();
    }

    var found = false;
    var t = 0.0;
    if first >= 0.0 {
        found = true;
        t = first;
    }
    if second >= 0.0 && (!found || second < t) {
        found = true;
        t = second;
    }
    if !found {
        return miss();
    }

    let hit_local = origin + direction * t;
    if !finite_vec3(hit_local) {
        return invalid_hit();
    }
    let normal = normalize_checked(hit_local - center);
    if !normal.valid {
        return invalid_hit();
    }
    let normal_scene = normal_to_scene(base, normal.value);
    if !normal_scene.valid {
        return invalid_hit();
    }
    return Hit(
        true,
        true,
        t,
        input_words[base + 1u],
        normal_scene.value,
        load_f32(base + 2u),
    );
}

fn intersect_plane(base: u32, origin_scene: vec3<f32>, direction_scene: vec3<f32>) -> Hit {
    let origin = to_local_point(base, origin_scene);
    let direction = to_local_direction(base, direction_scene);
    if !finite_vec3(origin) || !finite_vec3(direction) {
        return invalid_hit();
    }

    let point = vec3<f32>(
        load_f32(base + 25u),
        load_f32(base + 26u),
        load_f32(base + 27u),
    );
    let normal = normalize_checked(vec3<f32>(
        load_f32(base + 28u),
        load_f32(base + 29u),
        load_f32(base + 30u),
    ));
    if !normal.valid {
        return invalid_hit();
    }

    let denominator = dot(normal.value, direction);
    if !finite_f32(denominator) {
        return invalid_hit();
    }
    if denominator == 0.0 {
        return miss();
    }
    let t = dot(point - origin, normal.value) / denominator;
    if !finite_f32(t) {
        return invalid_hit();
    }
    if t < 0.0 {
        return miss();
    }

    let normal_scene = normal_to_scene(base, normal.value);
    if !normal_scene.valid {
        return invalid_hit();
    }
    return Hit(
        true,
        true,
        t,
        input_words[base + 1u],
        normal_scene.value,
        load_f32(base + 2u),
    );
}

fn field_origin(base: u32) -> vec3<f32> {
    return vec3<f32>(
        load_f32(base + 25u),
        load_f32(base + 26u),
        load_f32(base + 27u),
    );
}

fn field_spacing(base: u32) -> vec3<f32> {
    return vec3<f32>(
        load_f32(base + 28u),
        load_f32(base + 29u),
        load_f32(base + 30u),
    );
}

fn field_dimensions(base: u32) -> vec3<u32> {
    return vec3<u32>(
        input_words[base + 31u],
        input_words[base + 32u],
        input_words[base + 33u],
    );
}

fn field_max_local(base: u32) -> vec3<f32> {
    let origin = field_origin(base);
    let spacing = field_spacing(base);
    let dimensions = field_dimensions(base);
    return origin + spacing * vec3<f32>(
        f32(dimensions.x - 1u),
        f32(dimensions.y - 1u),
        f32(dimensions.z - 1u),
    );
}

fn field_sample_index(base: u32, x: u32, y: u32, z: u32) -> u32 {
    let dimensions = field_dimensions(base);
    return input_words[base + 34u]
        + z * dimensions.x * dimensions.y
        + y * dimensions.x
        + x;
}

fn field_sample(base: u32, point_local: vec3<f32>) -> FieldEvaluation {
    let origin = field_origin(base);
    let spacing = field_spacing(base);
    let dimensions = field_dimensions(base);
    if !finite_vec3(origin) || !finite_vec3(spacing)
        || spacing.x <= 0.0 || spacing.y <= 0.0 || spacing.z <= 0.0
        || dimensions.x < 2u || dimensions.y < 2u || dimensions.z < 2u
        || !finite_vec3(point_local)
    {
        return FieldEvaluation(false, false, 0.0);
    }

    let grid = (point_local - origin) / spacing;
    let grid_max = vec3<f32>(
        f32(dimensions.x - 1u),
        f32(dimensions.y - 1u),
        f32(dimensions.z - 1u),
    );
    if any(grid < vec3<f32>(-FIELD_COORDINATE_EPSILON))
        || any(grid > grid_max + vec3<f32>(FIELD_COORDINATE_EPSILON))
    {
        return FieldEvaluation(true, false, 0.0);
    }
    let clamped = clamp(grid, vec3<f32>(0.0), grid_max);
    let lower = vec3<u32>(
        u32(floor(clamped.x)),
        u32(floor(clamped.y)),
        u32(floor(clamped.z)),
    );
    let upper = min(lower + vec3<u32>(1u), dimensions - vec3<u32>(1u));
    let weight = clamped - vec3<f32>(
        f32(lower.x),
        f32(lower.y),
        f32(lower.z),
    );

    let c000 = load_f32(field_sample_index(base, lower.x, lower.y, lower.z));
    let c100 = load_f32(field_sample_index(base, upper.x, lower.y, lower.z));
    let c010 = load_f32(field_sample_index(base, lower.x, upper.y, lower.z));
    let c110 = load_f32(field_sample_index(base, upper.x, upper.y, lower.z));
    let c001 = load_f32(field_sample_index(base, lower.x, lower.y, upper.z));
    let c101 = load_f32(field_sample_index(base, upper.x, lower.y, upper.z));
    let c011 = load_f32(field_sample_index(base, lower.x, upper.y, upper.z));
    let c111 = load_f32(field_sample_index(base, upper.x, upper.y, upper.z));
    if !finite_f32(c000) || !finite_f32(c100) || !finite_f32(c010) || !finite_f32(c110)
        || !finite_f32(c001) || !finite_f32(c101) || !finite_f32(c011) || !finite_f32(c111)
    {
        return FieldEvaluation(false, false, 0.0);
    }

    let c00 = mix(c000, c100, weight.x);
    let c10 = mix(c010, c110, weight.x);
    let c01 = mix(c001, c101, weight.x);
    let c11 = mix(c011, c111, weight.x);
    let c0 = mix(c00, c10, weight.y);
    let c1 = mix(c01, c11, weight.y);
    let value = mix(c0, c1, weight.z);
    return FieldEvaluation(finite_f32(value), true, value);
}

fn field_interval(base: u32, origin_scene: vec3<f32>, direction_scene: vec3<f32>) -> RayInterval {
    let origin = to_local_point(base, origin_scene);
    let direction = to_local_direction(base, direction_scene);
    let minimum = field_origin(base);
    let maximum = field_max_local(base);
    if !finite_vec3(origin) || !finite_vec3(direction)
        || !finite_vec3(minimum) || !finite_vec3(maximum)
    {
        return RayInterval(false, false, 0.0, 0.0);
    }

    var t_min = 0.0;
    var t_max = 1.0e30;

    if abs(direction.x) <= FIELD_DIRECTION_EPSILON {
        if origin.x < minimum.x || origin.x > maximum.x {
            return RayInterval(true, false, 0.0, 0.0);
        }
    } else {
        let first = (minimum.x - origin.x) / direction.x;
        let second = (maximum.x - origin.x) / direction.x;
        if !finite_f32(first) || !finite_f32(second) {
            return RayInterval(false, false, 0.0, 0.0);
        }
        t_min = max(t_min, min(first, second));
        t_max = min(t_max, max(first, second));
        if t_max < t_min {
            return RayInterval(true, false, 0.0, 0.0);
        }
    }

    if abs(direction.y) <= FIELD_DIRECTION_EPSILON {
        if origin.y < minimum.y || origin.y > maximum.y {
            return RayInterval(true, false, 0.0, 0.0);
        }
    } else {
        let first = (minimum.y - origin.y) / direction.y;
        let second = (maximum.y - origin.y) / direction.y;
        if !finite_f32(first) || !finite_f32(second) {
            return RayInterval(false, false, 0.0, 0.0);
        }
        t_min = max(t_min, min(first, second));
        t_max = min(t_max, max(first, second));
        if t_max < t_min {
            return RayInterval(true, false, 0.0, 0.0);
        }
    }

    if abs(direction.z) <= FIELD_DIRECTION_EPSILON {
        if origin.z < minimum.z || origin.z > maximum.z {
            return RayInterval(true, false, 0.0, 0.0);
        }
    } else {
        let first = (minimum.z - origin.z) / direction.z;
        let second = (maximum.z - origin.z) / direction.z;
        if !finite_f32(first) || !finite_f32(second) {
            return RayInterval(false, false, 0.0, 0.0);
        }
        t_min = max(t_min, min(first, second));
        t_max = min(t_max, max(first, second));
        if t_max < t_min {
            return RayInterval(true, false, 0.0, 0.0);
        }
    }

    if !finite_f32(t_min) || !finite_f32(t_max) {
        return RayInterval(false, false, 0.0, 0.0);
    }
    return RayInterval(true, true, t_min, t_max);
}

fn field_normal_local(base: u32, point_local: vec3<f32>) -> NormalizedDirection {
    let minimum = field_origin(base);
    let maximum = field_max_local(base);
    let spacing = field_spacing(base);
    var gradient = vec3<f32>(0.0);

    let x0 = max(point_local.x - 0.5 * spacing.x, minimum.x);
    let x1 = min(point_local.x + 0.5 * spacing.x, maximum.x);
    if x1 <= x0 {
        return NormalizedDirection(false, vec3<f32>(0.0));
    }
    let x_low = field_sample(base, vec3<f32>(x0, point_local.y, point_local.z));
    let x_high = field_sample(base, vec3<f32>(x1, point_local.y, point_local.z));
    if !x_low.valid || !x_low.inside || !x_high.valid || !x_high.inside {
        return NormalizedDirection(false, vec3<f32>(0.0));
    }
    gradient.x = (x_high.value - x_low.value) / (x1 - x0);

    let y0 = max(point_local.y - 0.5 * spacing.y, minimum.y);
    let y1 = min(point_local.y + 0.5 * spacing.y, maximum.y);
    if y1 <= y0 {
        return NormalizedDirection(false, vec3<f32>(0.0));
    }
    let y_low = field_sample(base, vec3<f32>(point_local.x, y0, point_local.z));
    let y_high = field_sample(base, vec3<f32>(point_local.x, y1, point_local.z));
    if !y_low.valid || !y_low.inside || !y_high.valid || !y_high.inside {
        return NormalizedDirection(false, vec3<f32>(0.0));
    }
    gradient.y = (y_high.value - y_low.value) / (y1 - y0);

    let z0 = max(point_local.z - 0.5 * spacing.z, minimum.z);
    let z1 = min(point_local.z + 0.5 * spacing.z, maximum.z);
    if z1 <= z0 {
        return NormalizedDirection(false, vec3<f32>(0.0));
    }
    let z_low = field_sample(base, vec3<f32>(point_local.x, point_local.y, z0));
    let z_high = field_sample(base, vec3<f32>(point_local.x, point_local.y, z1));
    if !z_low.valid || !z_low.inside || !z_high.valid || !z_high.inside {
        return NormalizedDirection(false, vec3<f32>(0.0));
    }
    gradient.z = (z_high.value - z_low.value) / (z1 - z0);

    return normalize_checked(gradient);
}

fn intersect_field(base: u32, origin_scene: vec3<f32>, direction_scene: vec3<f32>) -> Hit {
    let interval = field_interval(base, origin_scene, direction_scene);
    if !interval.valid {
        return invalid_hit();
    }
    if !interval.found {
        return miss();
    }

    let scene_scale = load_f32(base + 3u);
    let query_error_local = load_f32(base + 35u);
    let spacing = field_spacing(base);
    if !finite_f32(scene_scale) || scene_scale <= 0.0
        || !finite_f32(query_error_local) || query_error_local < 0.0
        || !finite_vec3(spacing)
    {
        return invalid_hit();
    }
    let query_error_scene = query_error_local * scene_scale;
    let minimum_spacing_scene = min(spacing.x, min(spacing.y, spacing.z)) * scene_scale;
    if !finite_f32(query_error_scene) || !finite_f32(minimum_spacing_scene)
        || minimum_spacing_scene <= 0.0
    {
        return invalid_hit();
    }
    let numeric_hit_epsilon =
        max(FIELD_NUMERIC_HIT_EPSILON, minimum_spacing_scene * FIELD_NUMERIC_HIT_EPSILON);

    var t = interval.t_min;
    for (var step = 0u; step < FIELD_MAX_STEPS; step = step + 1u) {
        if t > interval.t_max + numeric_hit_epsilon {
            return miss();
        }
        let point_scene = origin_scene + direction_scene * t;
        let point_local = to_local_point(base, point_scene);
        let evaluation = field_sample(base, point_local);
        if !evaluation.valid || !evaluation.inside {
            return invalid_hit();
        }

        let estimate_scene = evaluation.value * scene_scale;
        if !finite_f32(estimate_scene) {
            return invalid_hit();
        }
        let absolute_estimate = abs(estimate_scene);
        if absolute_estimate <= query_error_scene + numeric_hit_epsilon {
            let normal_local = field_normal_local(base, point_local);
            if !normal_local.valid {
                return invalid_hit();
            }
            let normal_scene = normal_to_scene(base, normal_local.value);
            if !normal_scene.valid {
                return invalid_hit();
            }
            return Hit(
                true,
                true,
                t,
                input_words[base + 1u],
                normal_scene.value,
                load_f32(base + 2u),
            );
        }

        let safe_step = max(0.0, absolute_estimate - query_error_scene);
        if !finite_f32(safe_step) {
            return invalid_hit();
        }
        if safe_step <= numeric_hit_epsilon {
            let normal_local = field_normal_local(base, point_local);
            if !normal_local.valid {
                return invalid_hit();
            }
            let normal_scene = normal_to_scene(base, normal_local.value);
            if !normal_scene.valid {
                return invalid_hit();
            }
            return Hit(
                true,
                true,
                t,
                input_words[base + 1u],
                normal_scene.value,
                load_f32(base + 2u),
            );
        }

        let next_t = t + safe_step;
        if !finite_f32(next_t) || next_t <= t {
            return invalid_hit();
        }
        t = next_t;
    }
    return invalid_hit();
}

fn intersect_geometry(base: u32, origin_scene: vec3<f32>, direction_scene: vec3<f32>) -> Hit {
    if input_words[base] == 2u {
        return intersect_plane(base, origin_scene, direction_scene);
    }
    if input_words[base] == 3u {
        return intersect_field(base, origin_scene, direction_scene);
    }
    if input_words[base] == 1u {
        return intersect_sphere(base, origin_scene, direction_scene);
    }
    return invalid_hit();
}

fn nearest_hit(origin_scene: vec3<f32>, direction_scene: vec3<f32>, ignored_code: u32) -> Hit {
    let geometry_count = input_words[4u];
    var nearest = miss();
    var index = 0u;
    loop {
        if index >= geometry_count {
            break;
        }
        let base = geometry_base(index);
        let code = input_words[base + 1u];
        if code != ignored_code {
            let candidate = intersect_geometry(base, origin_scene, direction_scene);
            if !candidate.valid {
                return invalid_hit();
            }
            if candidate.found && (!nearest.found || candidate.t < nearest.t) {
                nearest = candidate;
            }
        }
        index = index + 1u;
    }
    return nearest;
}

fn direct_radiance(hit_position: vec3<f32>, hit: Hit) -> ScalarEvaluation {
    let emitter_count = input_words[5u];
    var total = 0.0;
    var index = 0u;
    loop {
        if index >= emitter_count {
            break;
        }
        let base = emitter_base(index);
        let direction = normalize_checked(vec3<f32>(
            load_f32(base),
            load_f32(base + 1u),
            load_f32(base + 2u),
        ));
        if !direction.valid {
            return ScalarEvaluation(false, 0.0);
        }

        let cosine = max(dot(hit.normal_scene, direction.value), 0.0);
        if !finite_f32(cosine) {
            return ScalarEvaluation(false, 0.0);
        }
        if cosine > 0.0 {
            let blocker = nearest_hit(hit_position, direction.value, hit.code);
            if !blocker.valid {
                return ScalarEvaluation(false, 0.0);
            }
            if !blocker.found {
                let contribution =
                    hit.reflectance * load_f32(base + 3u) * cosine / 3.14159265358979323846;
                total = total + contribution;
                if !finite_f32(total) {
                    return ScalarEvaluation(false, 0.0);
                }
            }
        }
        index = index + 1u;
    }
    return ScalarEvaluation(true, total);
}
