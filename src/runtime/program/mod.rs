pub(crate) mod abi;

use runen_gpu::{
    GpuAdmittedProgramSource, GpuProgramSourceError, GpuProgramSourceIdentity, GpuProgramSourceKey,
    GpuProgramSourceOwnerId, GpuProgramSourceProvenance, GpuProgramSourceRegistry,
    GpuProgramSourceRevision,
};
use runen_shader::{
    ShaderArtifact, ShaderCompilationInput, ShaderCompilationInvocation, ShaderCompilationOutcome,
    ShaderCompiler, ShaderCompilerRealization, ShaderDiagnostic, ShaderInvariantError,
    ShaderModuleIdentity, ShaderPackageIdentity, ShaderSourceRevision, ShaderSourceSnapshot,
    ShaderSourceUnitIdentity,
};
use std::error::Error;
use std::fmt;
use std::sync::{LazyLock, OnceLock};

const RUNEN_RENDER_SHADER_PACKAGE_ID: u64 = 1;
const EVALUATOR_MODULE_ID: u64 = 1;
const EVALUATOR_SOURCE_UNIT_ID: u64 = 1;
const TEMPORAL_MODULE_ID: u64 = 2;
const TEMPORAL_SOURCE_UNIT_ID: u64 = 2;
const CAMERA_MODULE_ID: u64 = 3;
const CAMERA_SOURCE_UNIT_ID: u64 = 3;
const SHAPED_TEXT_MODULE_ID: u64 = 4;
const SHAPED_TEXT_SOURCE_UNIT_ID: u64 = 4;
const VECTOR_MODULE_ID: u64 = 5;
const VECTOR_SOURCE_UNIT_ID: u64 = 5;
const VECTOR_REVISION: u64 = 1;
const VECTOR_WGSL: &str = include_str!("shaders/solid_vector.wgsl");

pub(crate) const MAINTAINED_EVALUATOR_REVISION: u64 = 3;
pub(crate) const TEMPORAL_RECONSTRUCTION_REVISION: u32 = 2;
pub(crate) const CAMERA_REPROJECTION_REVISION: u32 = 3;
pub(crate) const SHAPED_TEXT_REVISION: u64 = 1;

pub(crate) const SCENE_QUERY_WGSL: &str = include_str!("shaders/scene_query.wgsl");
pub(crate) static EVALUATOR_WGSL: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{SCENE_QUERY_WGSL}\n{}",
        include_str!("shaders/evaluator.wgsl")
    )
});
pub(crate) const TEMPORAL_RECONSTRUCTION_WGSL: &str =
    include_str!("shaders/temporal_reconstruction.wgsl");
pub(crate) static CAMERA_REPROJECTION_WGSL: LazyLock<String> = LazyLock::new(|| {
    format!(
        "{SCENE_QUERY_WGSL}\n{}",
        include_str!("shaders/camera_reprojection.wgsl")
    )
});
pub(crate) const SHAPED_TEXT_WGSL: &str = include_str!("shaders/shaped_text.wgsl");

static EVALUATOR_PROGRAM: OnceLock<RenderMaintainedProgram> = OnceLock::new();
static TEMPORAL_RECONSTRUCTION_PROGRAM: OnceLock<RenderMaintainedProgram> = OnceLock::new();
static CAMERA_REPROJECTION_PROGRAM: OnceLock<RenderMaintainedProgram> = OnceLock::new();
static SHAPED_TEXT_PROGRAM: OnceLock<RenderMaintainedProgram> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaintainedShaderProgram {
    Evaluator,
    TemporalReconstruction,
    CameraReprojection,
    ShapedText,
    Vector,
}

impl MaintainedShaderProgram {
    const fn label(self) -> &'static str {
        match self {
            Self::Evaluator => "maintained evaluator",
            Self::TemporalReconstruction => "temporal reconstruction",
            Self::CameraReprojection => "camera reprojection",
            Self::ShapedText => "2D shaped text",
            Self::Vector => "2D solid vector coverage",
        }
    }

    const fn gpu_key(self) -> &'static str {
        match self {
            Self::Evaluator => "runenrender.maintained.deterministic",
            Self::TemporalReconstruction => "runenrender.maintained.temporal_reconstruction",
            Self::CameraReprojection => "runenrender.maintained.camera_reprojection",
            Self::ShapedText => "runenrender.maintained.shaped_text",
            Self::Vector => "runenrender.maintained.solid_vector",
        }
    }

    const fn module_raw(self) -> u64 {
        match self {
            Self::Evaluator => EVALUATOR_MODULE_ID,
            Self::TemporalReconstruction => TEMPORAL_MODULE_ID,
            Self::CameraReprojection => CAMERA_MODULE_ID,
            Self::ShapedText => SHAPED_TEXT_MODULE_ID,
            Self::Vector => VECTOR_MODULE_ID,
        }
    }

    const fn source_unit_raw(self) -> u64 {
        match self {
            Self::Evaluator => EVALUATOR_SOURCE_UNIT_ID,
            Self::TemporalReconstruction => TEMPORAL_SOURCE_UNIT_ID,
            Self::CameraReprojection => CAMERA_SOURCE_UNIT_ID,
            Self::ShapedText => SHAPED_TEXT_SOURCE_UNIT_ID,
            Self::Vector => VECTOR_SOURCE_UNIT_ID,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RenderRunenShaderCompilationError {
    Invariant {
        program: &'static str,
        source: ShaderInvariantError,
    },
    Rejected {
        program: &'static str,
        diagnostics: Vec<ShaderDiagnostic>,
    },
    Unsupported {
        program: &'static str,
        diagnostics: Vec<ShaderDiagnostic>,
    },
    Failed {
        program: &'static str,
        diagnostics: Vec<ShaderDiagnostic>,
    },
    CanonicalBytesChanged {
        program: &'static str,
    },
}

impl RenderRunenShaderCompilationError {
    fn diagnostic_summary(diagnostics: &[ShaderDiagnostic]) -> &str {
        diagnostics
            .first()
            .map(ShaderDiagnostic::summary)
            .unwrap_or("<no diagnostic>")
    }
}

impl fmt::Display for RenderRunenShaderCompilationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invariant { program, source } => {
                write!(
                    formatter,
                    "RunenShader invariant failure for {program}: {source}"
                )
            }
            Self::Rejected {
                program,
                diagnostics,
            } => write!(
                formatter,
                "RunenShader rejected {program}: {}",
                Self::diagnostic_summary(diagnostics)
            ),
            Self::Unsupported {
                program,
                diagnostics,
            } => write!(
                formatter,
                "RunenShader does not support {program}: {}",
                Self::diagnostic_summary(diagnostics)
            ),
            Self::Failed {
                program,
                diagnostics,
            } => write!(
                formatter,
                "RunenShader failed to compile {program}: {}",
                Self::diagnostic_summary(diagnostics)
            ),
            Self::CanonicalBytesChanged { program } => write!(
                formatter,
                "RunenShader exact-WGSL artifact changed canonical bytes for {program}"
            ),
        }
    }
}

impl Error for RenderRunenShaderCompilationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Invariant { source, .. } => Some(source),
            Self::Rejected { .. }
            | Self::Unsupported { .. }
            | Self::Failed { .. }
            | Self::CanonicalBytesChanged { .. } => None,
        }
    }
}

#[derive(Debug)]
pub(crate) enum RenderMaintainedProgramBuildError {
    RunenShader(RenderRunenShaderCompilationError),
    RunenGpu {
        stage: &'static str,
        source: GpuProgramSourceError,
    },
}

#[derive(Debug)]
pub(crate) struct RenderMaintainedProgram {
    // Retain the accepted RunenShader artifact for the same renderer lifetime as its RunenGPU
    // admission even when production execution only needs the admitted source handle.
    _artifact: ShaderArtifact,
    admitted: GpuAdmittedProgramSource,
}

impl RenderMaintainedProgram {
    pub(crate) fn admitted(&self) -> &GpuAdmittedProgramSource {
        &self.admitted
    }

    #[cfg(test)]
    pub(crate) fn artifact(&self) -> &ShaderArtifact {
        &self._artifact
    }
}

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct RenderMaintainedProgramSources {
    evaluator: RenderMaintainedProgram,
    temporal_reconstruction: RenderMaintainedProgram,
    camera_reprojection: RenderMaintainedProgram,
    shaped_text: RenderMaintainedProgram,
}

#[cfg(test)]
impl RenderMaintainedProgramSources {
    pub(crate) fn evaluator(&self) -> &GpuAdmittedProgramSource {
        self.evaluator.admitted()
    }

    pub(crate) fn temporal_reconstruction(&self) -> &GpuAdmittedProgramSource {
        self.temporal_reconstruction.admitted()
    }

    pub(crate) fn camera_reprojection(&self) -> &GpuAdmittedProgramSource {
        self.camera_reprojection.admitted()
    }

    pub(crate) fn shaped_text(&self) -> &GpuAdmittedProgramSource {
        self.shaped_text.admitted()
    }

    pub(crate) fn evaluator_artifact(&self) -> &ShaderArtifact {
        self.evaluator.artifact()
    }

    pub(crate) fn temporal_reconstruction_artifact(&self) -> &ShaderArtifact {
        self.temporal_reconstruction.artifact()
    }

    pub(crate) fn camera_reprojection_artifact(&self) -> &ShaderArtifact {
        self.camera_reprojection.artifact()
    }

    pub(crate) fn shaped_text_artifact(&self) -> &ShaderArtifact {
        self.shaped_text.artifact()
    }
}

#[derive(Clone, Copy)]
struct MaintainedShaderSpec<'a> {
    program: MaintainedShaderProgram,
    revision: u64,
    wgsl: &'a str,
}

#[cfg(test)]
pub(crate) fn build_maintained_evaluator_program(
    revision: u64,
    wgsl: &str,
) -> Result<RenderMaintainedProgram, RenderMaintainedProgramBuildError> {
    build_maintained_program(MaintainedShaderSpec {
        program: MaintainedShaderProgram::Evaluator,
        revision,
        wgsl,
    })
}

#[cfg(test)]
pub(crate) fn build_temporal_reconstruction_program(
    revision: u64,
    wgsl: &str,
) -> Result<RenderMaintainedProgram, RenderMaintainedProgramBuildError> {
    build_maintained_program(MaintainedShaderSpec {
        program: MaintainedShaderProgram::TemporalReconstruction,
        revision,
        wgsl,
    })
}

#[cfg(test)]
pub(crate) fn build_camera_reprojection_program(
    revision: u64,
    wgsl: &str,
) -> Result<RenderMaintainedProgram, RenderMaintainedProgramBuildError> {
    build_maintained_program(MaintainedShaderSpec {
        program: MaintainedShaderProgram::CameraReprojection,
        revision,
        wgsl,
    })
}

#[cfg(test)]
pub(crate) fn build_shaped_text_program(
    revision: u64,
    wgsl: &str,
) -> Result<RenderMaintainedProgram, RenderMaintainedProgramBuildError> {
    build_maintained_program(MaintainedShaderSpec {
        program: MaintainedShaderProgram::ShapedText,
        revision,
        wgsl,
    })
}

pub(crate) fn retained_maintained_evaluator_source()
-> Result<GpuAdmittedProgramSource, RenderMaintainedProgramBuildError> {
    retained_program_source(
        &EVALUATOR_PROGRAM,
        MaintainedShaderSpec {
            program: MaintainedShaderProgram::Evaluator,
            revision: MAINTAINED_EVALUATOR_REVISION,
            wgsl: EVALUATOR_WGSL.as_str(),
        },
    )
}

pub(crate) fn retained_temporal_reconstruction_source()
-> Result<GpuAdmittedProgramSource, RenderMaintainedProgramBuildError> {
    retained_program_source(
        &TEMPORAL_RECONSTRUCTION_PROGRAM,
        MaintainedShaderSpec {
            program: MaintainedShaderProgram::TemporalReconstruction,
            revision: u64::from(TEMPORAL_RECONSTRUCTION_REVISION),
            wgsl: TEMPORAL_RECONSTRUCTION_WGSL,
        },
    )
}

pub(crate) fn retained_camera_reprojection_source()
-> Result<GpuAdmittedProgramSource, RenderMaintainedProgramBuildError> {
    retained_program_source(
        &CAMERA_REPROJECTION_PROGRAM,
        MaintainedShaderSpec {
            program: MaintainedShaderProgram::CameraReprojection,
            revision: u64::from(CAMERA_REPROJECTION_REVISION),
            wgsl: CAMERA_REPROJECTION_WGSL.as_str(),
        },
    )
}

pub(crate) fn retained_shaped_text_source()
-> Result<GpuAdmittedProgramSource, RenderMaintainedProgramBuildError> {
    retained_program_source(
        &SHAPED_TEXT_PROGRAM,
        MaintainedShaderSpec {
            program: MaintainedShaderProgram::ShapedText,
            revision: SHAPED_TEXT_REVISION,
            wgsl: SHAPED_TEXT_WGSL,
        },
    )
}

pub(crate) fn retained_vector_source()
-> Result<GpuAdmittedProgramSource, RenderMaintainedProgramBuildError> {
    static PROGRAM: OnceLock<RenderMaintainedProgram> = OnceLock::new();
    retained_program_source(
        &PROGRAM,
        MaintainedShaderSpec {
            program: MaintainedShaderProgram::Vector,
            revision: VECTOR_REVISION,
            wgsl: VECTOR_WGSL,
        },
    )
}

fn retained_program_source(
    cell: &'static OnceLock<RenderMaintainedProgram>,
    spec: MaintainedShaderSpec<'_>,
) -> Result<GpuAdmittedProgramSource, RenderMaintainedProgramBuildError> {
    if let Some(program) = cell.get() {
        return Ok(program.admitted().clone());
    }

    let program = build_maintained_program(spec)?;
    let _ = cell.set(program);

    Ok(cell
        .get()
        .expect("successful maintained program admission must initialize its retained cell")
        .admitted()
        .clone())
}

fn build_maintained_program(
    spec: MaintainedShaderSpec<'_>,
) -> Result<RenderMaintainedProgram, RenderMaintainedProgramBuildError> {
    let mut compiler = ShaderCompiler::new();
    let artifact = compile_exact_program(&mut compiler, spec)
        .map_err(RenderMaintainedProgramBuildError::RunenShader)?;

    let mut registry =
        GpuProgramSourceRegistry::new(1, spec.wgsl.len().max(1)).map_err(|source| {
            RenderMaintainedProgramBuildError::RunenGpu {
                stage: "maintained program source registry",
                source,
            }
        })?;
    let owner = GpuProgramSourceOwnerId::allocate().map_err(|source| {
        RenderMaintainedProgramBuildError::RunenGpu {
            stage: "maintained program source owner",
            source,
        }
    })?;

    admit_artifact(&mut registry, owner, spec, artifact)
}

#[cfg(test)]
pub(crate) fn build_maintained_program_sources()
-> Result<RenderMaintainedProgramSources, RenderMaintainedProgramBuildError> {
    Ok(RenderMaintainedProgramSources {
        evaluator: build_maintained_evaluator_program(
            MAINTAINED_EVALUATOR_REVISION,
            EVALUATOR_WGSL.as_str(),
        )?,
        temporal_reconstruction: build_temporal_reconstruction_program(
            u64::from(TEMPORAL_RECONSTRUCTION_REVISION),
            TEMPORAL_RECONSTRUCTION_WGSL,
        )?,
        camera_reprojection: build_camera_reprojection_program(
            u64::from(CAMERA_REPROJECTION_REVISION),
            CAMERA_REPROJECTION_WGSL.as_str(),
        )?,
        shaped_text: build_shaped_text_program(SHAPED_TEXT_REVISION, SHAPED_TEXT_WGSL)?,
    })
}
fn compile_exact_program(
    compiler: &mut ShaderCompiler,
    spec: MaintainedShaderSpec<'_>,
) -> Result<ShaderArtifact, RenderRunenShaderCompilationError> {
    let package = ShaderPackageIdentity::try_from_raw(RUNEN_RENDER_SHADER_PACKAGE_ID)
        .expect("RunenRender shader package identity is nonzero");
    let module = ShaderModuleIdentity::try_from_raw(spec.program.module_raw())
        .expect("RunenRender shader module identity is nonzero");
    let source_unit = ShaderSourceUnitIdentity::try_from_raw(spec.program.source_unit_raw())
        .expect("RunenRender shader source-unit identity is nonzero");
    let revision = ShaderSourceRevision::try_from_raw(spec.revision)
        .expect("maintained shader revisions are nonzero");
    let source = ShaderSourceSnapshot::new(source_unit, revision, spec.wgsl);
    let input = ShaderCompilationInput::exact_wgsl(package, module, source);
    let invocation =
        ShaderCompilationInvocation::new(input, ShaderCompilerRealization::Naga3001ExactWgslGateV1);
    let artifact = match compiler.compile(&invocation) {
        Ok(ShaderCompilationOutcome::Accepted(artifact)) => artifact,
        Ok(ShaderCompilationOutcome::Rejected(diagnostics)) => {
            return Err(RenderRunenShaderCompilationError::Rejected {
                program: spec.program.label(),
                diagnostics,
            });
        }
        Ok(ShaderCompilationOutcome::Unsupported(diagnostics)) => {
            return Err(RenderRunenShaderCompilationError::Unsupported {
                program: spec.program.label(),
                diagnostics,
            });
        }
        Ok(ShaderCompilationOutcome::Failed(diagnostics)) => {
            return Err(RenderRunenShaderCompilationError::Failed {
                program: spec.program.label(),
                diagnostics,
            });
        }
        Err(source) => {
            return Err(RenderRunenShaderCompilationError::Invariant {
                program: spec.program.label(),
                source,
            });
        }
    };

    if artifact.canonical_wgsl().as_bytes() != spec.wgsl.as_bytes() {
        return Err(RenderRunenShaderCompilationError::CanonicalBytesChanged {
            program: spec.program.label(),
        });
    }
    Ok(artifact)
}

fn admit_artifact(
    registry: &mut GpuProgramSourceRegistry,
    owner: GpuProgramSourceOwnerId,
    spec: MaintainedShaderSpec<'_>,
    artifact: ShaderArtifact,
) -> Result<RenderMaintainedProgram, RenderMaintainedProgramBuildError> {
    let key = GpuProgramSourceKey::new(spec.program.gpu_key()).map_err(|source| {
        RenderMaintainedProgramBuildError::RunenGpu {
            stage: "maintained program source key",
            source,
        }
    })?;
    let revision = GpuProgramSourceRevision::try_from_raw(spec.revision).map_err(|source| {
        RenderMaintainedProgramBuildError::RunenGpu {
            stage: "maintained program source revision",
            source,
        }
    })?;
    let identity = GpuProgramSourceIdentity::new(owner, key, revision);
    let provenance = GpuProgramSourceProvenance::new(
        "runen-render RunenShader bridge",
        Some(format!(
            "{} exact-WGSL artifact via {}",
            spec.program.label(),
            ShaderCompilerRealization::Naga3001ExactWgslGateV1.diagnostic_label()
        )),
    )
    .map_err(|source| RenderMaintainedProgramBuildError::RunenGpu {
        stage: "maintained program source provenance",
        source,
    })?;
    let admitted = registry
        .admit_wgsl(identity, artifact.canonical_wgsl(), provenance)
        .map_err(|source| RenderMaintainedProgramBuildError::RunenGpu {
            stage: "maintained RunenShader artifact admission",
            source,
        })?;

    Ok(RenderMaintainedProgram {
        _artifact: artifact,
        admitted,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shaped_text_program_is_part_of_the_maintained_exact_wgsl_set() {
        let programs = build_maintained_program_sources()
            .expect("all maintained RunenRender programs must compile and admit");
        let _ = programs.shaped_text();
        assert_eq!(
            programs.shaped_text_artifact().canonical_wgsl().as_bytes(),
            SHAPED_TEXT_WGSL.as_bytes()
        );
    }
}
