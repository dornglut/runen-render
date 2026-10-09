//! Source-neutral composable execution boundary for the admitted two-dimensional execution classes.

use crate::composition_2d::{
    Render2dComposition, Render2dResourceBindingError, Render2dResourceBindings, Render2dResourceId,
};
use crate::runtime::execution_2d::Render2dExecutionState;
use core::{error::Error, fmt};
use runen_gpu::{
    GpuContext, GpuExportKey, GpuRenderOperation, GpuSubmission, GpuSubmissionFailureKind,
    GpuSubmissionId, GpuSubmissionStatus, GpuTextureFormat, GpuTextureViewHandle,
    GpuWorkAuthoringError, GpuWorkFragment, GpuWorkFragmentBuilder, GpuWorkNodeId,
};

/// Caller-owned RunenGPU target-content relationships for one 2D contribution.
///
/// Keys describe graph causality, never semantic resource identity or execution
/// evidence. Use a distinct output key for each painting contribution. A prior
/// producer is required when another fragment writes this target in the same graph.
/// Otherwise the target must already have initialized contents. RunenGPU validates
/// those relationships and initialization before submission.
#[derive(Clone, Debug)]
pub struct Render2dWorkBinding {
    prior: Option<GpuExportKey>,
    output: GpuExportKey,
}

impl Render2dWorkBinding {
    /// Names the output of a contribution loading existing target contents.
    #[must_use]
    pub const fn new(output: GpuExportKey) -> Self {
        Self {
            prior: None,
            output,
        }
    }

    /// Binds the target contents to a prior producer in the same graph.
    #[must_use]
    pub fn after(mut self, prior: GpuExportKey) -> Self {
        self.prior = Some(prior);
        self
    }

    /// Returns the optional prior target-content producer key.
    #[must_use]
    pub const fn prior(&self) -> Option<&GpuExportKey> {
        self.prior.as_ref()
    }

    /// Returns the caller-selected target-content output key.
    #[must_use]
    pub const fn output(&self) -> &GpuExportKey {
        &self.output
    }
}

/// Immutable invocation facts for one 2D color target.
///
/// Logical canvas extent and raster scale are invocation facts only. They do not enter
/// immutable composition identity.
#[derive(Clone, Debug)]
pub struct Render2dTarget {
    view: GpuTextureViewHandle,
    logical_width: f64,
    logical_height: f64,
    raster_scale: f64,
}

impl Render2dTarget {
    /// Creates one finite positive logical target mapping.
    ///
    /// Descriptor/capability compatibility with the supplied RunenGPU context is checked during
    /// preparation because those facts are context-dependent.
    pub fn new(
        view: GpuTextureViewHandle,
        logical_width: f64,
        logical_height: f64,
        raster_scale: f64,
    ) -> Result<Self, Render2dTargetError> {
        if !logical_width.is_finite()
            || !logical_height.is_finite()
            || logical_width <= 0.0
            || logical_height <= 0.0
        {
            return Err(Render2dTargetError::InvalidLogicalExtent);
        }
        if !raster_scale.is_finite() || raster_scale <= 0.0 {
            return Err(Render2dTargetError::InvalidRasterScale);
        }
        Ok(Self {
            view,
            logical_width,
            logical_height,
            raster_scale,
        })
    }

    /// Returns the caller-owned backend-neutral RunenGPU color view.
    #[must_use]
    pub const fn view(&self) -> &GpuTextureViewHandle {
        &self.view
    }

    /// Returns logical canvas width.
    #[must_use]
    pub const fn logical_width(&self) -> f64 {
        self.logical_width
    }

    /// Returns logical canvas height.
    #[must_use]
    pub const fn logical_height(&self) -> f64 {
        self.logical_height
    }

    /// Returns exact logical-to-physical raster scale.
    #[must_use]
    pub const fn raster_scale(&self) -> f64 {
        self.raster_scale
    }
}

/// Invalid source-neutral F2 target mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dTargetError {
    /// Logical extent must be finite and positive.
    InvalidLogicalExtent,
    /// Raster scale must be finite and positive.
    InvalidRasterScale,
}

impl fmt::Display for Render2dTargetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLogicalExtent => {
                write!(
                    formatter,
                    "2D logical target extent must be finite and positive"
                )
            }
            Self::InvalidRasterScale => {
                write!(
                    formatter,
                    "2D target raster scale must be finite and positive"
                )
            }
        }
    }
}

impl Error for Render2dTargetError {}

/// Exact reason target facts cannot be admitted by the bounded F2 execution path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dTargetAdmissionError {
    /// Continuous logical-to-physical extent cannot be represented by the initial F2 path.
    PhysicalExtentOutOfRange,
    /// Bound texture extent does not equal the required ceiling of the continuous canvas.
    PhysicalExtentMismatch {
        /// Required physical pixel width.
        expected_width: u32,
        /// Required physical pixel height.
        expected_height: u32,
        /// Bound parent texture width.
        actual_width: u32,
        /// Bound parent texture height.
        actual_height: u32,
    },
    /// The view is not the exact full base-mip, one-layer D2 target required by F2.
    UnsupportedViewShape,
    /// The target is not single-sampled.
    UnsupportedSampleCount,
    /// Target format is outside the initial normalized sRGB set.
    UnsupportedFormat {
        /// Effective target view format.
        format: GpuTextureFormat,
    },
    /// The parent texture does not declare color-attachment usage.
    MissingColorAttachmentUsage,
    /// The admitted context does not expose blendable color-target support for the target format.
    TargetFormatNotBlendable,
    /// The admitted context cannot realize the private filtered field texture contract.
    FieldFormatUnsupported,
    /// The admitted context lacks the unified correlated Rgba16Float sample plane.
    SamplePlaneFormatUnsupported,
    /// Target or private field dimensions exceed the admitted device/workload limit.
    TextureDimensionLimitExceeded,
    /// The admitted context lacks render-pipeline execution.
    RenderPipelineNotAdmitted,
}

impl fmt::Display for Render2dTargetAdmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PhysicalExtentOutOfRange => {
                write!(
                    formatter,
                    "2D continuous physical target extent is not representable"
                )
            }
            Self::PhysicalExtentMismatch {
                expected_width,
                expected_height,
                actual_width,
                actual_height,
            } => write!(
                formatter,
                "2D target extent must be {expected_width}x{expected_height}, observed {actual_width}x{actual_height}"
            ),
            Self::UnsupportedViewShape => write!(
                formatter,
                "2D target must be a full base-mip single-layer D2 view of a single-layer D2 texture"
            ),
            Self::UnsupportedSampleCount => {
                write!(formatter, "2D F2 target must be single-sampled")
            }
            Self::UnsupportedFormat { format } => write!(
                formatter,
                "2D F2 target format {format:?} is outside the admitted normalized sRGB set"
            ),
            Self::MissingColorAttachmentUsage => {
                write!(
                    formatter,
                    "2D F2 target does not declare color-attachment usage"
                )
            }
            Self::TargetFormatNotBlendable => write!(
                formatter,
                "2D F2 target format is not admitted as blendable color attachment"
            ),
            Self::FieldFormatUnsupported => write!(
                formatter,
                "2D F2 private field format is not sampled/filterable/copy-destination capable"
            ),
            Self::SamplePlaneFormatUnsupported => write!(
                formatter,
                "2D correlated F1 rendering requires admitted Rgba16Float sample-plane and Rgba8 coverage roles"
            ),
            Self::TextureDimensionLimitExceeded => write!(
                formatter,
                "2D F2 texture extent exceeds admitted device or workload limits"
            ),
            Self::RenderPipelineNotAdmitted => {
                write!(
                    formatter,
                    "2D F2 requires admitted render-pipeline execution"
                )
            }
        }
    }
}

impl Error for Render2dTargetAdmissionError {}

/// Bounded semantic class rejected by the Counter-critical F2 admission gate.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dUnsupportedContent {
    /// Nested groups are outside this slice.
    Group {
        /// Root painter-order index.
        root_index: usize,
    },
    /// Primitive or brush class is outside the admitted execution subset.
    Primitive {
        /// Root painter-order index.
        root_index: usize,
    },
    /// Item clips are outside this slice.
    Clips {
        /// Root painter-order index.
        root_index: usize,
    },
    /// Non-opaque shaped-text item opacity is outside this slice.
    Opacity {
        /// Root painter-order index.
        root_index: usize,
    },
    /// Non-translation shaped-text transform is outside this slice.
    Transform {
        /// Root painter-order index.
        root_index: usize,
    },
    /// Initial field realization requires a positive finite logical font size representable as f32.
    FontSize {
        /// Affected semantic resource identity.
        resource_id: Render2dResourceId,
    },
}

impl fmt::Display for Render2dUnsupportedContent {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Group { root_index } => {
                write!(formatter, "2D F2 root entry {root_index} is a group")
            }
            Self::Primitive { root_index } => write!(
                formatter,
                "2D root entry {root_index} carries unsupported primitive or brush content"
            ),
            Self::Clips { root_index } => {
                write!(formatter, "2D F2 root item {root_index} carries clips")
            }
            Self::Opacity { root_index } => write!(
                formatter,
                "2D F2 root item {root_index} carries non-opaque item opacity"
            ),
            Self::Transform { root_index } => write!(
                formatter,
                "2D F2 root item {root_index} carries a non-translation transform"
            ),
            Self::FontSize { resource_id } => write!(
                formatter,
                "2D F2 shaped resource {} has an unsupported logical font size",
                resource_id.get()
            ),
        }
    }
}

impl Error for Render2dUnsupportedContent {}

/// Intrinsic glyph representation rejected by the initial scalable-outline path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dUnsupportedGlyphKind {
    /// COLR version 0 content.
    ColrV0,
    /// COLR version 1 content.
    ColrV1,
    /// Embedded bitmap content.
    Bitmap,
    /// SVG glyph content.
    Svg,
    /// Faux emboldening synthesis.
    FauxBold,
}

/// Source-neutral shaped-text realization failure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dShapedTextError {
    /// Font bytes/face cannot be interpreted by the maintained outline reader.
    InvalidFont {
        /// Affected semantic resource identity.
        resource_id: Render2dResourceId,
    },
    /// One intrinsic glyph class is not admitted by this slice.
    UnsupportedGlyph {
        /// Affected semantic resource identity.
        resource_id: Render2dResourceId,
        /// Upstream-shaped glyph id.
        glyph_id: Option<u32>,
        /// Unsupported intrinsic/synthesis class.
        kind: Render2dUnsupportedGlyphKind,
    },
    /// Scalable outline extraction or conversion failed.
    InvalidOutline {
        /// Affected semantic resource identity.
        resource_id: Render2dResourceId,
        /// Upstream-shaped glyph id.
        glyph_id: u32,
    },
    /// Private field extent cannot fit the admitted texture dimension bound.
    GlyphExtentExceedsLimit {
        /// Affected semantic resource identity.
        resource_id: Render2dResourceId,
        /// Upstream-shaped glyph id.
        glyph_id: u32,
        /// Generated field width.
        width: u32,
        /// Generated field height.
        height: u32,
        /// Admitted maximum 2D texture dimension.
        maximum: u32,
    },
    /// Sum of retained shaped field allocations exceeds the F2/F3E invocation budget.
    FieldBudgetExceeded {
        /// Affected immutable shaped resource.
        resource_id: Render2dResourceId,
        /// Already-shaped glyph that would exceed the aggregate budget.
        glyph_id: u32,
        /// Maximum cumulative RGBA8 bytes admitted for shaped field data.
        maximum_bytes: u64,
    },
}

impl fmt::Display for Render2dShapedTextError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidFont { resource_id } => write!(
                formatter,
                "2D shaped resource {} contains an invalid font face",
                resource_id.get()
            ),
            Self::UnsupportedGlyph {
                resource_id,
                glyph_id,
                kind,
            } => write!(
                formatter,
                "2D shaped resource {} glyph {:?} uses unsupported {kind:?} content",
                resource_id.get(),
                glyph_id
            ),
            Self::InvalidOutline {
                resource_id,
                glyph_id,
            } => write!(
                formatter,
                "2D shaped resource {} glyph {glyph_id} has an invalid scalable outline",
                resource_id.get()
            ),
            Self::GlyphExtentExceedsLimit {
                resource_id,
                glyph_id,
                width,
                height,
                maximum,
            } => write!(
                formatter,
                "2D shaped resource {} glyph {glyph_id} field {width}x{height} exceeds admitted dimension {maximum}",
                resource_id.get()
            ),
            Self::FieldBudgetExceeded {
                resource_id,
                glyph_id,
                maximum_bytes,
            } => write!(
                formatter,
                "2D shaped resource {} glyph {glyph_id} exceeds the {maximum_bytes}-byte cumulative retained field budget",
                resource_id.get()
            ),
        }
    }
}

impl Error for Render2dShapedTextError {}

/// Machine-actionable admission failure for the one correlated 4x4 sample compiler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dSampleSpaceError {
    /// Aggregate geometry, scratch, tile replay, storage or work-node budget exceeded.
    ResourceLimit,
    /// An authored affine cannot preserve its required physical precision.
    PrecisionLimit,
}

/// Failure to prepare one bounded F1/F2/F3 2D contribution.
#[derive(Debug)]
pub enum Render2dExecutionError {
    /// A vector item cannot be realized within current precision or resource limits.
    Vector {
        /// Root painter-order index of the affected item.
        root_index: usize,
        /// Stable renderer-owned failure class.
        kind: Render2dVectorError,
    },
    /// A semantic clip cannot be realized at the admitted physical precision/resources.
    Clip {
        /// Root painter-order index of the affected clipped item.
        root_index: usize,
        /// Stable clip realization failure.
        kind: Render2dClipError,
    },
    /// An immutable image patch cannot be realized under physical precision or resource limits.
    Image {
        /// Root painter-order index of the affected image.
        root_index: usize,
        /// Stable source-neutral realization failure.
        kind: Render2dImageError,
    },
    /// Composition/resource compatibility failed before maintained realization.
    ResourceBindings(Render2dResourceBindingError),
    /// Invocation target facts are not admitted.
    Target(Render2dTargetAdmissionError),
    /// Composition content is outside the admitted supported subset.
    UnsupportedContent(Render2dUnsupportedContent),
    /// A nested composition entry requires an unadmitted semantic capability.
    UnsupportedEntry {
        /// Exact zero-based authored F1 root/child occurrence path.
        path: Vec<usize>,
        /// Stable renderer-owned unsupported content class.
        kind: Render2dUnsupportedContent,
    },
    /// An already-observed semantic identity was rebound to different immutable content.
    ResourceIdentityRebound {
        /// Rebound semantic identity.
        resource_id: Render2dResourceId,
    },
    /// Shaped-text intrinsic realization failed.
    ShapedText(Render2dShapedTextError),
    /// Maintained program realization failed at an owner-oriented stage.
    Program {
        /// Maintained stage.
        stage: &'static str,
        /// Underlying diagnostic text.
        detail: String,
    },
    /// Correlated-sample preparation rejects unavailable resources or precision before mutation.
    SampleSpace {
        /// Stable admission failure class.
        kind: Render2dSampleSpaceError,
        /// Exact nested entry path when attributable to one authored entry.
        path: Option<Vec<usize>>,
        /// Renderer-owned diagnostic for the failed preflight budget or projection.
        detail: String,
    },
    /// RunenGPU lowering failed at an owner-oriented stage.
    Gpu {
        /// Lowering stage.
        stage: &'static str,
        /// Underlying diagnostic text.
        detail: String,
    },
}

impl fmt::Display for Render2dExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vector { root_index, kind } => {
                write!(formatter, "2D vector item {root_index}: {kind:?}")
            }
            Self::Image { root_index, kind } => {
                write!(formatter, "2D image item {root_index}: {kind:?}")
            }
            Self::Clip { root_index, kind } => {
                write!(formatter, "2D clipped item {root_index}: {kind:?}")
            }
            Self::ResourceBindings(error) => error.fmt(formatter),
            Self::Target(error) => error.fmt(formatter),
            Self::UnsupportedContent(error) => error.fmt(formatter),
            Self::UnsupportedEntry { path, kind } => {
                write!(formatter, "2D entry path {path:?}: {kind}")
            },
            Self::ResourceIdentityRebound { resource_id } => write!(
                formatter,
                "2D semantic resource {} was rebound to different immutable content",
                resource_id.get()
            ),
            Self::ShapedText(error) => error.fmt(formatter),
            Self::Program { stage, detail } => {
                write!(
                    formatter,
                    "2D maintained program failure during {stage}: {detail}"
                )
            }
            Self::SampleSpace { kind, path, detail } => {
                write!(formatter, "2D correlated sample {kind:?}")?;
                if let Some(path) = path {
                    write!(formatter, " at entry path {path:?}")?;
                }
                write!(formatter, ": {detail}")
            }
            Self::Gpu { stage, detail } => {
                write!(
                    formatter,
                    "2D RunenGPU lowering failure during {stage}: {detail}"
                )
            }
        }
    }
}

impl Error for Render2dExecutionError {}

/// Stable conjunctive clip realization failure without a public mask or stencil backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dClipError {
    /// Geometry or masking cannot preserve finite physical sample precision.
    PrecisionLimit,
    /// Tessellation input, temporary raster work, or mask memory exceeds policy.
    ResourceLimit,
    /// Device workload lacks private RGBA8 mask sampling/transfer capabilities.
    FormatUnsupported,
    /// Accepted source-neutral clip shape cannot be tessellated faithfully.
    TessellationFailed,
}

/// Stable image physical realization failure, without exposing texture/backend identity.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dImageError {
    /// Image dimensions, total upload bytes or prepared GPU storage exceed bounded limits.
    ResourceLimit,
    /// Geometry, mapping or affine inverse cannot preserve required physical precision.
    PrecisionLimit,
    /// The admitted device lacks RGBA8-sRGB texture sampling/upload capabilities.
    FormatUnsupported,
}

/// Renderer-owned vector realization failure, independent of the private tessellator.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dVectorError {
    /// Coordinates or tolerance cannot preserve the admitted physical precision.
    PrecisionLimit,
    /// Geometry or intermediate storage exceeds the admitted bounded realization.
    ResourceLimit,
    /// The current device contract lacks the required coverage-target format roles.
    CoverageFormatUnsupported,
    /// Valid structural geometry could not be tessellated.
    TessellationFailed,
}

impl From<Render2dResourceBindingError> for Render2dExecutionError {
    fn from(error: Render2dResourceBindingError) -> Self {
        Self::ResourceBindings(error)
    }
}

impl From<Render2dTargetAdmissionError> for Render2dExecutionError {
    fn from(error: Render2dTargetAdmissionError) -> Self {
        Self::Target(error)
    }
}

impl From<Render2dUnsupportedContent> for Render2dExecutionError {
    fn from(error: Render2dUnsupportedContent) -> Self {
        Self::UnsupportedContent(error)
    }
}

impl From<Render2dShapedTextError> for Render2dExecutionError {
    fn from(error: Render2dShapedTextError) -> Self {
        Self::ShapedText(error)
    }
}

/// Retained 2D execution owner.
///
/// Semantic identity observations survive private field-cache discard. Derived field data does
/// not. A fresh executor reconstructs compatible private state from immutable bindings.
#[derive(Debug, Default)]
pub struct Render2dExecutor {
    state: Render2dExecutionState,
}

impl Render2dExecutor {
    /// Creates an empty retained execution owner.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Discards only derived shaped-field data.
    ///
    /// Previously observed semantic identity/value associations remain retained so cache discard
    /// cannot legalize semantic rebinding.
    pub fn discard_cache(&mut self) {
        self.state.discard_cache();
    }

    /// Validates, realizes, lowers and transactionally commits one composable 2D contribution.
    pub fn prepare(
        &mut self,
        context: &GpuContext,
        composition: &Render2dComposition,
        bindings: &Render2dResourceBindings,
        target: &Render2dTarget,
    ) -> Result<Render2dPreparedContribution, Render2dExecutionError> {
        self.state.prepare(context, composition, bindings, target)
    }
}

/// Prepared private realization ready to author into composable RunenGPU work.
///
/// Preparation is not execution evidence. Consume this value through `append_to`
/// for caller-owned lexical ordering, or `into_fragment` for typed cross-fragment
/// composition. Both paths author the same ordered lowered operations. Valid non-painting
/// content produces no node, output, or execution token.
#[derive(Debug)]
pub struct Render2dPreparedContribution {
    render: Vec<GpuRenderOperation>,
    target: GpuTextureViewHandle,
}

impl Render2dPreparedContribution {
    pub(crate) fn new(render: Vec<GpuRenderOperation>, target: GpuTextureViewHandle) -> Self {
        Self { render, target }
    }

    /// Returns whether this contribution contains actual RunenRender-authored render work.
    #[must_use]
    pub const fn has_render_work(&self) -> bool {
        !self.render.is_empty()
    }

    /// Appends the exact renderer-authored operations to a caller-owned fragment.
    ///
    /// The caller owns surrounding work and lexical ordering, including clear and
    /// terminal work. RunenGPU derives resource hazards from the appended operation.
    /// The returned single-use token identifies every actual newly authored node.
    pub fn append_to(
        self,
        builder: &mut GpuWorkFragmentBuilder,
    ) -> Result<Option<Render2dContributionToken>, GpuWorkAuthoringError> {
        let mut nodes = Vec::with_capacity(self.render.len());
        for render in self.render {
            nodes.push(builder.operation("render admitted 2D composition", render)?);
        }
        Ok((!nodes.is_empty()).then_some(Render2dContributionToken { nodes }))
    }

    /// Authors a separate fragment with native target-content import/output wiring.
    ///
    /// The caller supplies graph relationship keys and composes/submits the graph.
    /// Empty contributions have no output; keep the prior relationship in that case.
    pub fn into_fragment(
        self,
        work_binding: &Render2dWorkBinding,
    ) -> Result<(GpuWorkFragment, Option<Render2dContributionToken>), GpuWorkAuthoringError> {
        let mut token = None;
        let fragment =
            GpuWorkFragment::build("runen-render 2D composition contribution", |builder| {
                let target = self.target.clone();
                token = self.append_to(builder)?;
                if token.is_some() {
                    crate::runtime::execution_2d::add_target_boundary(
                        builder,
                        &target,
                        work_binding,
                    )?;
                }
                Ok(())
            })?;
        Ok((fragment, token))
    }
}

/// Single-use exact authored-work correlation for one prepared 2D contribution.
#[derive(Debug)]
pub struct Render2dContributionToken {
    nodes: Vec<GpuWorkNodeId>,
}

impl Render2dContributionToken {
    /// Returns the exact work-node identities authored into the caller's RunenGPU
    /// fragment, in the contribution's painter order. The first and last nodes
    /// are its non-data control-order frontiers.
    ///
    /// These IDs allow the caller to add fragment-local `GpuExplicitOrder`
    /// constraints before closing the builder. They do not confer execution or
    /// presentation evidence and do not change RunenGPU's hazard authority.
    /// A nonpainting contribution produces no token.
    #[must_use]
    pub fn authored_nodes(&self) -> &[GpuWorkNodeId] {
        &self.nodes
    }

    /// Consumes the token and proves every exact authored node participated and completed.
    pub fn completed_by(
        self,
        submission: &GpuSubmission,
    ) -> Result<Render2dContributionEvidence, Render2dContributionEvidenceError> {
        if !self
            .nodes
            .iter()
            .all(|node| submission.contains_work_node(node))
        {
            return Err(Render2dContributionEvidenceError::MissingWorkNode);
        }
        match submission.status() {
            GpuSubmissionStatus::Completed => Ok(Render2dContributionEvidence {
                submission_id: submission.id(),
            }),
            GpuSubmissionStatus::Accepted => {
                Err(Render2dContributionEvidenceError::SubmissionNotCompleted)
            }
            GpuSubmissionStatus::Failed(failure) => {
                Err(Render2dContributionEvidenceError::SubmissionFailed {
                    kind: failure.kind(),
                })
            }
        }
    }
}

/// Successful source-neutral proof that one exact prepared F2 contribution completed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Render2dContributionEvidence {
    submission_id: GpuSubmissionId,
}

impl Render2dContributionEvidence {
    /// Returns terminal submission identity carrying the exact authored 2D work nodes.
    #[must_use]
    pub const fn submission_id(self) -> GpuSubmissionId {
        self.submission_id
    }
}

/// Failure to derive successful contribution evidence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Render2dContributionEvidenceError {
    /// The supplied submission did not contain the exact authored 2D work-node identities.
    MissingWorkNode,
    /// Exact work-node membership exists, but the submission remains in-flight.
    SubmissionNotCompleted,
    /// Exact work-node membership exists, but the submission terminalized unsuccessfully.
    SubmissionFailed {
        /// Generic RunenGPU terminal failure class.
        kind: GpuSubmissionFailureKind,
    },
}

impl fmt::Display for Render2dContributionEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingWorkNode => write!(
                formatter,
                "submission does not contain every exact prepared 2D contribution work node"
            ),
            Self::SubmissionNotCompleted => write!(
                formatter,
                "submission contains the 2D contribution but has not terminalized successfully"
            ),
            Self::SubmissionFailed { kind } => write!(
                formatter,
                "submission contains the 2D contribution but failed with {kind:?}"
            ),
        }
    }
}

impl Error for Render2dContributionEvidenceError {}

#[cfg(test)]
mod evidence_tests;
