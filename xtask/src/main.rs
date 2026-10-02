use std::{
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const REQUIRED_FILES: &[&str] = &[
    ".cargo/config.toml",
    ".github/workflows/validation.yml",
    ".gitignore",
    "AGENTS.md",
    "ARCHITECTURE.md",
    "BOOTSTRAP.md",
    "Cargo.lock",
    "Cargo.toml",
    "LICENSE",
    "LICENSING.md",
    "README.md",
    "TESTING.md",
    "rust-toolchain.toml",
    "src/lib.rs",
    "xtask/Cargo.toml",
    "xtask/src/main.rs",
];

const ACTIVE_IDENTITY_FILES: &[&str] = &[
    "Cargo.toml",
    "README.md",
    "AGENTS.md",
    "ARCHITECTURE.md",
    "TESTING.md",
    "src/lib.rs",
];

const STALE_ACTIVE_IDENTITY: &[&str] = &[
    "rust-framework-template",
    "MIT OR Apache-2.0",
    "Apache License 2.0",
];

const TRANSFER_PRODUCTION_FILES: &[&str] = &[
    "src/admission.rs",
    "src/appearance.rs",
    "src/derived_state.rs",
    "src/derived_transform.rs",
    "src/deterministic_admission.rs",
    "src/deterministic_capture.rs",
    "src/deterministic_carrier.rs",
    "src/deterministic_execution.rs",
    "src/deterministic_verification.rs",
    "src/deterministic_verification/numeric.rs",
    "src/deterministic_verification/observation.rs",
    "src/deterministic_verification/semantic.rs",
    "src/field_input.rs",
    "src/lib.rs",
    "src/lowering.rs",
    "src/maintained_method.rs",
    "src/method.rs",
    "src/ordinary.rs",
    "src/output_result.rs",
    "src/participation.rs",
    "src/render_result.rs",
    "src/representation.rs",
    "src/request.rs",
    "src/scene/mod.rs",
    "src/semantic_binding.rs",
    "src/semantic_plan.rs",
    "src/shader_bridge.rs",
    "src/space_time.rs",
    "src/surface_input.rs",
    "src/surface_result.rs",
];

const FORBIDDEN_PRODUCTION_MARKERS: &[&str] = &[
    "RUNENWERK_",
    "crate::plugins::render",
    "crate::plugins::world",
    "crate::plugins::ui",
    "bevy_",
    "winit::",
    "std::env",
    "std::fs",
    "serde_json",
    "wgpu::",
];

fn main() {
    let result = match env::args().nth(1).as_deref() {
        Some("validate") => validate(),
        _ => Err("usage: cargo xtask validate".to_owned()),
    };

    if let Err(error) = result {
        eprintln!("validation failed: {error}");
        std::process::exit(1);
    }
}

fn validate() -> Result<(), String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask manifest must have a workspace root")
        .to_path_buf();

    validate_required_files(&root)?;
    validate_product_identity(&root)?;
    validate_transfer_contract(&root)?;

    let initial_state = git_status(&root)?;
    if !initial_state.is_empty() {
        return Err(format!(
            "repository must be clean before validation:\n{initial_state}"
        ));
    }

    run(&root, "cargo", &["fmt", "--all", "--", "--check"])?;
    run(&root, "cargo", &["test", "--workspace", "--locked"])?;
    run(
        &root,
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--locked",
            "--",
            "-D",
            "warnings",
        ],
    )?;
    run_with_env(
        &root,
        "cargo",
        &["doc", "--workspace", "--no-deps", "--locked"],
        &[("RUSTDOCFLAGS", "-D warnings")],
    )?;
    run(
        &root,
        "cargo",
        &[
            "+1.93.0",
            "check",
            "--workspace",
            "--all-targets",
            "--locked",
        ],
    )?;
    run(&root, "git", &["diff", "--check"])?;
    run(&root, "git", &["diff", "--cached", "--check"])?;

    let final_state = git_status(&root)?;
    if final_state != initial_state {
        return Err(format!(
            "validation changed repository state:\nbefore:\n{initial_state}after:\n{final_state}"
        ));
    }

    Ok(())
}

fn validate_required_files(root: &Path) -> Result<(), String> {
    for relative_path in REQUIRED_FILES {
        let path = root.join(relative_path);
        if !path.is_file() {
            return Err(format!("required file is missing: {relative_path}"));
        }
    }

    Ok(())
}

fn validate_product_identity(root: &Path) -> Result<(), String> {
    let manifest = read_file(root, "Cargo.toml")?;
    for required in [
        "name = \"runen-render\"",
        "version = \"0.1.0\"",
        "edition = \"2024\"",
        "rust-version = \"1.93.0\"",
        "license.workspace = true",
        "repository = \"https://github.com/dornglut/runen-render\"",
        "description = \"Reusable semantic rendering and maintained image-formation framework\"",
        "publish = false",
        "[workspace.package]",
        "license = \"GPL-3.0-only\"",
        "[features]\ndefault = []",
    ] {
        require_contains("Cargo.toml", &manifest, required)?;
    }

    if manifest.contains("unsafe_code = \"forbid\"") {
        return Err("Cargo.toml contains the unaccepted template unsafe-code policy".to_owned());
    }

    let lockfile = read_file(root, "Cargo.lock")?;
    require_contains("Cargo.lock", &lockfile, "name = \"runen-render\"")?;
    if lockfile.contains("name = \"rust-framework-template\"") {
        return Err("Cargo.lock contains stale framework-template identity".to_owned());
    }

    let license = read_file(root, "LICENSE")?;
    for required in [
        "GNU GENERAL PUBLIC LICENSE",
        "Version 3, 29 June 2007",
        "END OF TERMS AND CONDITIONS",
        "How to Apply These Terms to Your New Programs",
    ] {
        require_contains("LICENSE", &license, required)?;
    }
    if license.len() < 10_000 {
        return Err("LICENSE is shorter than a complete GPLv3 text".to_owned());
    }

    for relative_path in ACTIVE_IDENTITY_FILES {
        let contents = read_file(root, relative_path)?;
        for stale_identity in STALE_ACTIVE_IDENTITY {
            if contents.contains(stale_identity) {
                return Err(format!(
                    "active identity file {relative_path} contains stale product identity or license: {stale_identity}"
                ));
            }
        }
    }

    Ok(())
}

fn validate_transfer_contract(root: &Path) -> Result<(), String> {
    let manifest = read_file(root, "Cargo.toml")?;
    for required in [
        "runen-gpu = { git = \"https://github.com/dornglut/runen-gpu\", rev = \"789b430fdefeda89bfe59de86d548618b8f8ab9a\" }",
        "runen-shader = { git = \"https://github.com/dornglut/runen-shader\", rev = \"406f8165da92caa2d296b3a2f774ba534279870d\" }",
    ] {
        require_contains("Cargo.toml", &manifest, required)?;
    }
    for moving in ["branch =", "tag ="] {
        if manifest.contains(moving) {
            return Err(format!(
                "Cargo.toml contains moving sibling dependency authority: {moving}"
            ));
        }
    }

    for relative_path in TRANSFER_PRODUCTION_FILES {
        let path = root.join(relative_path);
        if !path.is_file() {
            return Err(format!(
                "required transferred production file is missing: {relative_path}"
            ));
        }
        let contents = read_file(root, relative_path)?;
        for marker in FORBIDDEN_PRODUCTION_MARKERS {
            if contents.contains(marker) {
                return Err(format!(
                    "production source {relative_path} contains forbidden predecessor/product coupling: {marker}"
                ));
            }
        }
    }

    for relative_path in [
        "src/deterministic_scene_query.wgsl",
        "src/deterministic_execution.wgsl",
        "src/deterministic_temporal_reconstruction.wgsl",
        "src/deterministic_camera_reprojection.wgsl",
        "tests/ordinary_public_api.rs",
    ] {
        if !root.join(relative_path).is_file() {
            return Err(format!("required RX authority file is missing: {relative_path}"));
        }
    }

    let lib = read_file(root, "src/lib.rs")?;
    for required in [
        "pub mod admission;",
        "pub mod appearance;",
        "pub mod derived_state;",
        "pub mod field_input;",
        "pub mod lowering;",
        "pub mod method;",
        "pub mod output_result;",
        "pub mod participation;",
        "pub mod representation;",
        "pub mod request;",
        "pub mod scene;",
        "pub mod semantic_plan;",
        "pub mod space_time;",
        "pub mod surface_input;",
        "pub mod surface_result;",
        "mod deterministic_execution;",
        "mod ordinary;",
        "mod shader_bridge;",
        "pub use ordinary::*;",
    ] {
        require_contains("src/lib.rs", &lib, required)?;
    }
    for forbidden in [
        "pub mod deterministic_",
        "pub use deterministic_",
        "pub mod maintained_method",
        "pub mod ordinary",
        "pub mod shader_bridge",
    ] {
        if lib.contains(forbidden) {
            return Err(format!(
                "src/lib.rs leaks maintained implementation vocabulary: {forbidden}"
            ));
        }
    }

    let public_consumer = read_file(root, "tests/ordinary_public_api.rs")?;
    if public_consumer.contains("Deterministic") {
        return Err(
            "ordinary public package consumer references deterministic implementation vocabulary"
                .to_owned(),
        );
    }

    let workflow = read_file(root, ".github/workflows/validation.yml")?;
    for required in [
        "name: RunenRender Vulkan conformance",
        "RUNEN_RENDER_REQUIRE_GPU: '1'",
        "mesa-vulkan-drivers",
        "WGPU_BACKEND=vulkan",
        "cargo +stable test -p runen-render --locked -- --nocapture --test-threads=1",
    ] {
        require_contains(".github/workflows/validation.yml", &workflow, required)?;
    }

    Ok(())
}

fn read_file(root: &Path, relative_path: &str) -> Result<String, String> {
    fs::read_to_string(root.join(relative_path))
        .map_err(|error| format!("failed to read {relative_path}: {error}"))
}

fn require_contains(file: &str, contents: &str, required: &str) -> Result<(), String> {
    if contents.contains(required) {
        Ok(())
    } else {
        Err(format!(
            "{file} is missing required product contract: {required}"
        ))
    }
}

fn git_status(root: &Path) -> Result<String, String> {
    output(
        root,
        "git",
        &["status", "--porcelain", "--untracked-files=all"],
    )
}

fn run(root: &Path, program: &str, args: &[&str]) -> Result<(), String> {
    run_with_env(root, program, args, &[])
}

fn run_with_env(
    root: &Path,
    program: &str,
    args: &[&str],
    environment: &[(&str, &str)],
) -> Result<(), String> {
    let mut command = Command::new(program);
    command
        .args(args)
        .current_dir(root)
        .envs(environment.iter().copied())
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    let status = command
        .status()
        .map_err(|error| format!("failed to execute {program}: {error}"))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!("{program} {} exited with {status}", args.join(" ")))
    }
}

fn output(root: &Path, program: &str, args: &[&str]) -> Result<String, String> {
    let result = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| format!("failed to execute {program}: {error}"))?;

    if !result.status.success() {
        return Err(format!(
            "{program} {} exited with {}:\n{}",
            args.join(" "),
            result.status,
            String::from_utf8_lossy(&result.stderr)
        ));
    }

    String::from_utf8(result.stdout)
        .map_err(|error| format!("{program} produced invalid UTF-8: {error}"))
}
