use runen_gpu::{
    GpuAdmittedProgramSource, GpuProgramSourceError, GpuProgramSourceIdentity,
    GpuProgramSourceKey, GpuProgramSourceOwnerId, GpuProgramSourceProvenance,
    GpuProgramSourceRegistry, GpuProgramSourceRevision,
};
use runen_shader::{
    ShaderArtifact, ShaderCompilationInput, ShaderCompilationInvocation, ShaderCompilationOutcome,
    ShaderCompiler, ShaderCompilerRealization, ShaderDiagnostic, ShaderInvariantError,
    ShaderModuleIdentity, ShaderPackageIdentity, ShaderSourceRevision, ShaderSourceSnapshot,
    ShaderSourceUnitIdentity,
};
use std::error::Error;
use std::fmt;

const RUNEN_RENDER_SHADER_PACKAGE_ID: u64 = 1;
const EVALUATOR_MODULE_ID: u64 = 1;
const EVALUATOR_SOURCE_UNIT_ID: u64 = 1;
const TEMPORAL_MODULE_ID: u64 = 2;
const TEMPORAL_SOURCE_UNIT_ID: u64 = 2;
const CAMERA_MODULE_ID: u64 = 3;
const CAMERA_SOURCE_UNIT_ID: u64 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MaintainedShaderProgram {
    Evaluator,
    TemporalReconstruction,
    CameraReprojection,
}

impl MaintainedShaderProgram {
    const fn label(self) -> &'static str {
        match self {
            Self::Evaluator => "maintained evaluator",
            Self::TemporalReconstruction => "temporal reconstruction",
            Self::CameraReprojection => "camera reprojection",
        }
    }

    const fn gpu_key(self) -> &'static str {
        match self {
            Self::Evaluator => "runenrender.maintained.deterministic",
            Self::TemporalReconstruction => "runenrender.maintained.temporal_reconstruction",
            Self::CameraReprojection => "runenrender.maintained.camera_reprojection",
        }
    }

    const fn module_raw(self) -> u64 {
        match self {
            Self::Evaluator => EVALUATOR_MODULE_ID,
            Self::TemporalReconstruction => TEMPORAL_MODULE_ID,
            Self::CameraReprojection => CAMERA_MODULE_ID,
        }
    }

    const fn source_unit_raw(self) -> u64 {
        match self {
            Self::Evaluator => EVALUATOR_SOURCE_UNIT_ID,
            Self::TemporalReconstruction => TEMPORAL_SOURCE_UNIT_ID,
            Self::CameraReprojection => CAMERA_SOURCE_UNIT_ID,
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
                write!(formatter, "RunenShader invariant failure for {program}: {source}")
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
struct RetainedMaintainedProgram {
    artifact: ShaderArtifact,
    admitted: GpuAdmittedProgramSource,
}

#[derive(Debug)]
pub(crate) struct RenderMaintainedProgramSources {
    evaluator: RetainedMaintainedProgram,
    temporal_reconstruction: RetainedMaintainedProgram,
    camera_reprojection: RetainedMaintainedProgram,
}

impl RenderMaintainedProgramSources {
    pub(crate) fn evaluator(&self) -> &GpuAdmittedProgramSource {
        &self.evaluator.admitted
    }

    pub(crate) fn temporal_reconstruction(&self) -> &GpuAdmittedProgramSource {
        &self.temporal_reconstruction.admitted
    }

    pub(crate) fn camera_reprojection(&self) -> &GpuAdmittedProgramSource {
        &self.camera_reprojection.admitted
    }

    #[cfg(test)]
    pub(crate) fn evaluator_artifact(&self) -> &ShaderArtifact {
        &self.evaluator.artifact
    }

    #[cfg(test)]
    pub(crate) fn temporal_reconstruction_artifact(&self) -> &ShaderArtifact {
        &self.temporal_reconstruction.artifact
    }

    #[cfg(test)]
    pub(crate) fn camera_reprojection_artifact(&self) -> &ShaderArtifact {
        &self.camera_reprojection.artifact
    }
}

#[derive(Clone, Copy)]
struct MaintainedShaderSpec<'a> {
    program: MaintainedShaderProgram,
    revision: u64,
    wgsl: &'a str,
}

pub(crate) fn build_maintained_program_sources(
    evaluator_revision: u64,
    evaluator_wgsl: &str,
    temporal_revision: u64,
    temporal_wgsl: &str,
    camera_revision: u64,
    camera_wgsl: &str,
) -> Result<RenderMaintainedProgramSources, RenderMaintainedProgramBuildError> {
    let specs = [
        MaintainedShaderSpec {
            program: MaintainedShaderProgram::Evaluator,
            revision: evaluator_revision,
            wgsl: evaluator_wgsl,
        },
        MaintainedShaderSpec {
            program: MaintainedShaderProgram::TemporalReconstruction,
            revision: temporal_revision,
            wgsl: temporal_wgsl,
        },
        MaintainedShaderSpec {
            program: MaintainedShaderProgram::CameraReprojection,
            revision: camera_revision,
            wgsl: camera_wgsl,
        },
    ];

    let mut compiler = ShaderCompiler::new();
    let evaluator = compile_exact_program(&mut compiler, specs[0])
        .map_err(RenderMaintainedProgramBuildError::RunenShader)?;
    let temporal_reconstruction = compile_exact_program(&mut compiler, specs[1])
        .map_err(RenderMaintainedProgramBuildError::RunenShader)?;
    let camera_reprojection = compile_exact_program(&mut compiler, specs[2])
        .map_err(RenderMaintainedProgramBuildError::RunenShader)?;

    let total_source_bytes = specs
        .iter()
        .try_fold(0usize, |total, spec| total.checked_add(spec.wgsl.len()))
        .unwrap_or(usize::MAX);
    let mut registry = GpuProgramSourceRegistry::new(3, total_source_bytes.max(1)).map_err(
        |source| RenderMaintainedProgramBuildError::RunenGpu {
            stage: "maintained program source registry",
            source,
        },
    )?;
    let owner = GpuProgramSourceOwnerId::allocate().map_err(|source| {
        RenderMaintainedProgramBuildError::RunenGpu {
            stage: "maintained program source owner",
            source,
        }
    })?;

    Ok(RenderMaintainedProgramSources {
        evaluator: admit_artifact(&mut registry, owner, specs[0], evaluator)?,
        temporal_reconstruction: admit_artifact(
            &mut registry,
            owner,
            specs[1],
            temporal_reconstruction,
        )?,
        camera_reprojection: admit_artifact(
            &mut registry,
            owner,
            specs[2],
            camera_reprojection,
        )?,
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
    let invocation = ShaderCompilationInvocation::new(
        input,
        ShaderCompilerRealization::Naga3001ExactWgslGateV1,
    );
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
) -> Result<RetainedMaintainedProgram, RenderMaintainedProgramBuildError> {
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

    Ok(RetainedMaintainedProgram { artifact, admitted })
}
