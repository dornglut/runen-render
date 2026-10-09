//! Private maintained host <-> WGSL ABI authority.
//!
//! These constants name the physical word layout consumed by the maintained WGSL programs.
//! WGSL remains exact source authority for shader behavior; parity tests below prove the intentional
//! duplicated indices and numeric policy constants without introducing a generated shader format.

pub(crate) mod header {
    pub(crate) const WORDS: usize = 30;
    pub(crate) const SAMPLE_COUNT: usize = 0;
    pub(crate) const EVALUATION_WIDTH: usize = 1;
    pub(crate) const EVALUATION_HEIGHT: usize = 2;
    pub(crate) const ROW_STRIDE_WORDS: usize = 3;
    pub(crate) const GEOMETRY_COUNT: usize = 4;
    pub(crate) const EMITTER_COUNT: usize = 5;
    pub(crate) const EXECUTION_MODE: usize = 6;
    pub(crate) const OBSERVATION_KIND: usize = 7;
    pub(crate) const OBSERVATION_TRANSLATION: usize = 8;
    pub(crate) const OBSERVATION_SCENE_TO_LOCAL: usize = 11;
    pub(crate) const PERSPECTIVE_TAN_HALF_FOV: usize = 20;
    pub(crate) const PERSPECTIVE_ASPECT_RATIO: usize = 21;
    pub(crate) const REQUESTED_WIDTH: usize = 22;
    pub(crate) const REQUESTED_HEIGHT: usize = 23;
    pub(crate) const TEMPORAL_PHASE: usize = 24;
    pub(crate) const TEMPORAL_SEQUENCE_REVISION: usize = 25;
    pub(crate) const TEMPORAL_HISTORY_AGE: usize = 26;
    pub(crate) const TEMPORAL_HISTORY_ROW_STRIDE: usize = 27;
    pub(crate) const TEMPORAL_RECONSTRUCTION_REVISION: usize = 28;
    pub(crate) const EMITTER_OFFSET: usize = 29;
}

pub(crate) mod geometry {
    pub(crate) const WORDS: usize = 40;
    pub(crate) const SHAPE: usize = 0;
    pub(crate) const OBJECT_CODE: usize = 1;
    pub(crate) const RADIANCE_REFLECTANCE: usize = 2;
    pub(crate) const FIELD_SCENE_SCALE: usize = 3;
    pub(crate) const SCENE_TO_LOCAL: usize = 4;
    pub(crate) const TRANSLATION: usize = 13;
    pub(crate) const NORMAL_LOCAL_TO_SCENE: usize = 16;
    pub(crate) const SHAPE_DATA: usize = 25;
    pub(crate) const SPHERE_RADIUS: usize = 28;
    pub(crate) const PLANE_NORMAL: usize = 28;
    pub(crate) const FIELD_ORIGIN: usize = 25;
    pub(crate) const FIELD_SAMPLE_SPACING: usize = 28;
    pub(crate) const FIELD_DIMENSION_X: usize = 31;
    pub(crate) const FIELD_DIMENSION_Y: usize = 32;
    pub(crate) const FIELD_DIMENSION_Z: usize = 33;
    pub(crate) const FIELD_SAMPLE_OFFSET: usize = 34;
    pub(crate) const FIELD_MAX_QUERY_ERROR: usize = 35;
}

pub(crate) mod emitter {
    pub(crate) const WORDS: usize = 4;
    pub(crate) const DIRECTION: usize = 0;
    pub(crate) const SPECTRAL_IRRADIANCE: usize = 3;
}

pub(crate) mod execution_mode {
    pub(crate) const RADIANCE: u32 = 1;
    pub(crate) const FORWARD_DEPTH: u32 = 2;
    pub(crate) const OBJECT_IDENTITY: u32 = 3;
    pub(crate) const REQUESTED_COVERAGE: u32 = 4;
}

pub(crate) mod observation_kind {
    pub(crate) const PERSPECTIVE: u32 = 1;
    pub(crate) const PROBE: u32 = 2;
    pub(crate) const PERSPECTIVE_FOOTPRINT: u32 = 3;
}

pub(crate) mod shape {
    pub(crate) const SPHERE: u32 = 1;
    pub(crate) const PLANE: u32 = 2;
    pub(crate) const FIELD: u32 = 3;
}

pub(crate) mod temporal {
    pub(crate) const SEQUENCE_REVISION: u32 = 1;
    pub(crate) const PHASE_COUNT: u32 = 4;
}

pub(crate) mod requested_coverage {
    pub(crate) const POLICY_REVISION: u32 = 1;
}

pub(crate) mod camera {
    pub(crate) const DEPTH_POLICY_REVISION: u32 = 1;
    pub(crate) const DEPTH_ABSOLUTE_EPSILON: f32 = 0.001;
    pub(crate) const DEPTH_RELATIVE_EPSILON: f32 = 0.001;

    pub(crate) mod parameters {
        pub(crate) const WORDS: usize = 35;
        pub(crate) const PREVIOUS_AVAILABLE: usize = 0;
        pub(crate) const POSE_CHANGED: usize = 1;
        pub(crate) const DEPTH_POLICY_REVISION: usize = 2;
        pub(crate) const REPROJECTION_REVISION: usize = 3;
        pub(crate) const DEPTH_ABSOLUTE_EPSILON: usize = 4;
        pub(crate) const DEPTH_RELATIVE_EPSILON: usize = 5;
        pub(crate) const PREVIOUS_ORIGIN: usize = 6;
        pub(crate) const PREVIOUS_SCENE_TO_OBSERVATION: usize = 9;
        pub(crate) const PREVIOUS_FORWARD: usize = 18;
        pub(crate) const PREVIOUS_TAN_HALF_FOV: usize = 21;
        pub(crate) const PREVIOUS_ASPECT_RATIO: usize = 22;
        pub(crate) const SAME_POSE_COMPLETED_FRAMES: usize = 23;
        pub(crate) const CURRENT_SCENE_TO_OBSERVATION: usize = 24;
        pub(crate) const MOTION_DIAGNOSTICS_ENABLED: usize = 33;
        pub(crate) const CURRENT_ONLY_FIRST_MOTION: usize = 34;
    }

    pub(crate) mod history {
        pub(crate) const WORDS_PER_SAMPLE: u64 = 8;
        pub(crate) const CURRENT_HIT_WORDS_PER_SAMPLE: u64 = 4;
        pub(crate) const DIAGNOSTIC_PROBE_COUNT: u64 = 32;
        pub(crate) const DIAGNOSTIC_WORDS_PER_PROBE: u64 = 32;
        pub(crate) const DIAGNOSTIC_WORDS: u64 =
            DIAGNOSTIC_PROBE_COUNT * DIAGNOSTIC_WORDS_PER_PROBE;
    }
}

pub(crate) const WORKGROUP_SIZE: u32 = 64;

pub(crate) const COMPOSITION_VERTEX_STRIDE: u64 = 32;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::carrier;
    use crate::runtime::program::{
        CAMERA_REPROJECTION_REVISION, CAMERA_REPROJECTION_WGSL, EVALUATOR_WGSL,
        MAINTAINED_EVALUATOR_REVISION, SCENE_QUERY_WGSL, TEMPORAL_RECONSTRUCTION_REVISION,
        TEMPORAL_RECONSTRUCTION_WGSL,
    };

    fn indexed(name: &str, index: usize) -> String {
        format!("{name}[{index}u]")
    }

    #[test]
    fn vector_coverage_sampling_matches_the_exact_shader_contract() {
        assert!(super::super::VECTOR_WGSL.contains(&format!(
            "const COVERAGE_AXIS_SAMPLES: i32 = {VECTOR_COVERAGE_AXIS_SAMPLES};"
        )));
        assert_eq!(COMPOSITION_VERTEX_STRIDE, 8 * size_of::<f32>() as u64);
    }

    fn based(name: &str, index: usize) -> String {
        format!("{name}[base + {index}u]")
    }

    #[test]
    fn host_header_offsets_match_maintained_wgsl_reads() {
        for (source, indexes) in [
            (
                EVALUATOR_WGSL.as_str(),
                &[
                    header::SAMPLE_COUNT,
                    header::EVALUATION_WIDTH,
                    header::EVALUATION_HEIGHT,
                    header::ROW_STRIDE_WORDS,
                    header::EXECUTION_MODE,
                    header::OBSERVATION_KIND,
                    header::REQUESTED_WIDTH,
                    header::REQUESTED_HEIGHT,
                    header::TEMPORAL_PHASE,
                ][..],
            ),
            (
                TEMPORAL_RECONSTRUCTION_WGSL,
                &[
                    header::SAMPLE_COUNT,
                    header::EVALUATION_WIDTH,
                    header::EVALUATION_HEIGHT,
                    header::ROW_STRIDE_WORDS,
                    header::REQUESTED_WIDTH,
                    header::REQUESTED_HEIGHT,
                    header::TEMPORAL_PHASE,
                    header::TEMPORAL_HISTORY_AGE,
                    header::TEMPORAL_HISTORY_ROW_STRIDE,
                ][..],
            ),
            (
                CAMERA_REPROJECTION_WGSL.as_str(),
                &[
                    header::SAMPLE_COUNT,
                    header::EVALUATION_WIDTH,
                    header::ROW_STRIDE_WORDS,
                    header::REQUESTED_WIDTH,
                    header::REQUESTED_HEIGHT,
                    header::TEMPORAL_PHASE,
                ][..],
            ),
        ] {
            for index in indexes {
                assert!(
                    source.contains(&indexed("input_words", *index)),
                    "maintained WGSL must consume input_words[{index}]"
                );
            }
        }
    }

    #[test]
    fn geometry_and_emitter_offsets_match_scene_query_wgsl_reads() {
        for index in [
            geometry::OBJECT_CODE,
            geometry::RADIANCE_REFLECTANCE,
            geometry::FIELD_SCENE_SCALE,
            geometry::SCENE_TO_LOCAL,
            geometry::TRANSLATION,
            geometry::NORMAL_LOCAL_TO_SCENE,
            geometry::SHAPE_DATA,
            geometry::SPHERE_RADIUS,
            geometry::FIELD_DIMENSION_X,
            geometry::FIELD_DIMENSION_Y,
            geometry::FIELD_DIMENSION_Z,
            geometry::FIELD_SAMPLE_OFFSET,
            geometry::FIELD_MAX_QUERY_ERROR,
        ] {
            assert!(
                SCENE_QUERY_WGSL.contains(&based("input_words", index))
                    || SCENE_QUERY_WGSL.contains(&format!("base + {index}u")),
                "scene-query WGSL must retain geometry offset {index}"
            );
        }
        assert!(SCENE_QUERY_WGSL.contains(&indexed("input_words", header::EMITTER_OFFSET)));
        assert!(SCENE_QUERY_WGSL.contains(&format!("index * {}u", emitter::WORDS)));
        assert!(SCENE_QUERY_WGSL.contains(&format!("base + {}u", emitter::SPECTRAL_IRRADIANCE)));
    }

    #[test]
    fn camera_parameter_history_and_numeric_policy_match_wgsl() {
        for index in [
            camera::parameters::PREVIOUS_AVAILABLE,
            camera::parameters::POSE_CHANGED,
            camera::parameters::PREVIOUS_ORIGIN,
            camera::parameters::PREVIOUS_FORWARD,
            camera::parameters::PREVIOUS_TAN_HALF_FOV,
            camera::parameters::PREVIOUS_ASPECT_RATIO,
            camera::parameters::SAME_POSE_COMPLETED_FRAMES,
            camera::parameters::MOTION_DIAGNOSTICS_ENABLED,
            camera::parameters::CURRENT_ONLY_FIRST_MOTION,
        ] {
            assert!(
                CAMERA_REPROJECTION_WGSL.contains(&indexed("camera_words", index))
                    || CAMERA_REPROJECTION_WGSL.contains(&format!("load_camera_f32({index}u)")),
                "camera WGSL must retain camera parameter offset {index}"
            );
        }

        assert!(
            CAMERA_REPROJECTION_WGSL
                .contains(&format!("index * {}u", camera::history::WORDS_PER_SAMPLE))
        );
        assert!(CAMERA_REPROJECTION_WGSL.contains(&format!(
            "output_index * {}u",
            camera::history::CURRENT_HIT_WORDS_PER_SAMPLE
        )));
        assert!(CAMERA_REPROJECTION_WGSL.contains(&format!(
            "const CAMERA_DEPTH_ABSOLUTE_EPSILON: f32 = {};",
            camera::DEPTH_ABSOLUTE_EPSILON
        )));
        assert!(CAMERA_REPROJECTION_WGSL.contains(&format!(
            "const CAMERA_DEPTH_RELATIVE_EPSILON: f32 = {};",
            camera::DEPTH_RELATIVE_EPSILON
        )));
        assert!(CAMERA_REPROJECTION_WGSL.contains(&format!(
            "mul_camera3({}u",
            camera::parameters::PREVIOUS_SCENE_TO_OBSERVATION
        )));
        assert!(CAMERA_REPROJECTION_WGSL.contains(&format!(
            "mul_camera3({}u",
            camera::parameters::CURRENT_SCENE_TO_OBSERVATION
        )));
        assert_eq!(carrier::WORD_BYTES, core::mem::size_of::<u32>());
    }

    #[test]
    fn program_and_policy_revisions_are_explicit() {
        assert_eq!(MAINTAINED_EVALUATOR_REVISION, 3);
        assert_eq!(TEMPORAL_RECONSTRUCTION_REVISION, 2);
        assert_eq!(CAMERA_REPROJECTION_REVISION, 3);
        assert_eq!(requested_coverage::POLICY_REVISION, 1);
        assert_eq!(temporal::SEQUENCE_REVISION, 1);
        assert_eq!(camera::DEPTH_POLICY_REVISION, 1);
    }

    #[test]
    fn record_widths_and_mode_values_match_maintained_contract() {
        assert_eq!(header::WORDS, header::EMITTER_OFFSET + 1);
        const { assert!(geometry::FIELD_MAX_QUERY_ERROR < geometry::WORDS) };
        assert_eq!(emitter::WORDS, emitter::SPECTRAL_IRRADIANCE + 1);
        assert_eq!(
            camera::parameters::WORDS,
            camera::parameters::CURRENT_ONLY_FIRST_MOTION + 1
        );
        assert_eq!(execution_mode::RADIANCE, 1);
        assert_eq!(execution_mode::FORWARD_DEPTH, 2);
        assert_eq!(execution_mode::OBJECT_IDENTITY, 3);
        assert_eq!(execution_mode::REQUESTED_COVERAGE, 4);
        for mode in [
            execution_mode::RADIANCE,
            execution_mode::FORWARD_DEPTH,
            execution_mode::OBJECT_IDENTITY,
            execution_mode::REQUESTED_COVERAGE,
        ] {
            assert!(
                EVALUATOR_WGSL.contains(&format!("execution_mode == {mode}u")),
                "evaluator WGSL must retain execution mode {mode}"
            );
        }

        assert_eq!(observation_kind::PERSPECTIVE, 1);
        assert_eq!(observation_kind::PROBE, 2);
        assert_eq!(observation_kind::PERSPECTIVE_FOOTPRINT, 3);
        for kind in [
            observation_kind::PROBE,
            observation_kind::PERSPECTIVE_FOOTPRINT,
        ] {
            assert!(
                EVALUATOR_WGSL.contains(&format!("input_words[7u] == {kind}u")),
                "evaluator WGSL must retain observation kind {kind}"
            );
        }

        assert_eq!(shape::SPHERE, 1);
        assert_eq!(shape::PLANE, 2);
        assert_eq!(shape::FIELD, 3);
        for shape in [shape::SPHERE, shape::PLANE, shape::FIELD] {
            assert!(
                SCENE_QUERY_WGSL.contains(&format!("input_words[base] == {shape}u")),
                "scene-query WGSL must retain shape value {shape}"
            );
        }

        assert_eq!(temporal::PHASE_COUNT, 4);
        assert_eq!(WORKGROUP_SIZE, 64);
    }
}
