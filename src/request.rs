use super::space_time::{
    CanonicalF64, RenderAffineTransform3, RenderSemanticValueError, RenderTimeInterval,
};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderRequestValidationError {
    SemanticValue(RenderSemanticValueError),
    NonPositive { field: &'static str },
    Negative { field: &'static str },
    DegenerateObservationFrame,
    PerspectiveFieldOfViewOutOfRange,
    SamplingConeOutOfRange,
    PerspectiveLatticeCellRequiresPerspectiveObservation,
    InvalidLatticeDimensions,
    IdentityToleranceMustBeExact,
    EmptyObservations,
    EmptyOutputs,
    OutputObservationOutOfRange { observation_index: usize },
    ObservationOutsideRenderInterval { observation_index: usize },
    ProbeRequiresScalarTopology { observation_index: usize },
    PerspectiveRequiresSampleLatticeTopology { observation_index: usize },
    ObservationHasNoOutputs { observation_index: usize },
}

impl From<RenderSemanticValueError> for RenderRequestValidationError {
    fn from(value: RenderSemanticValueError) -> Self {
        Self::SemanticValue(value)
    }
}

impl fmt::Display for RenderRequestValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SemanticValue(error) => fmt::Display::fmt(error, f),
            Self::NonPositive { field } => write!(f, "{field} must be greater than zero"),
            Self::Negative { field } => write!(f, "{field} must be non-negative"),
            Self::DegenerateObservationFrame => {
                write!(f, "observation frame linear basis must be invertible")
            }
            Self::PerspectiveFieldOfViewOutOfRange => {
                write!(
                    f,
                    "perspective field of view must be greater than zero and less than pi radians"
                )
            }
            Self::SamplingConeOutOfRange => {
                write!(
                    f,
                    "sampling cone half-angle must be greater than zero and less than pi/2 radians"
                )
            }
            Self::PerspectiveLatticeCellRequiresPerspectiveObservation => write!(
                f,
                "perspective lattice-cell sampling support requires a perspective observation"
            ),
            Self::InvalidLatticeDimensions => {
                write!(f, "sample lattice dimensions must both be non-zero")
            }
            Self::IdentityToleranceMustBeExact => {
                write!(
                    f,
                    "object-identity output requires exact semantic tolerance"
                )
            }
            Self::EmptyObservations => write!(f, "render request must contain an observation"),
            Self::EmptyOutputs => write!(f, "render request must contain an output"),
            Self::OutputObservationOutOfRange { observation_index } => write!(
                f,
                "requested output references missing observation index {observation_index}"
            ),
            Self::ObservationOutsideRenderInterval { observation_index } => write!(
                f,
                "observation index {observation_index} has shutter support outside the render interval"
            ),
            Self::ProbeRequiresScalarTopology { observation_index } => write!(
                f,
                "probe observation index {observation_index} requires scalar result topology"
            ),
            Self::PerspectiveRequiresSampleLatticeTopology { observation_index } => write!(
                f,
                "perspective observation index {observation_index} requires 2D sample-lattice result topology"
            ),
            Self::ObservationHasNoOutputs { observation_index } => write!(
                f,
                "observation index {observation_index} has no requested outputs"
            ),
        }
    }
}

impl Error for RenderRequestValidationError {}

fn validate_observation_frame(
    observation_to_scene: RenderAffineTransform3,
) -> Result<(), RenderRequestValidationError> {
    let matrix = observation_to_scene.row_major_3x4();
    let linear = [
        matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], matrix[6], matrix[8], matrix[9],
        matrix[10],
    ];
    let scale = linear.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(RenderRequestValidationError::DegenerateObservationFrame);
    }

    let m00 = linear[0] / scale;
    let m01 = linear[1] / scale;
    let m02 = linear[2] / scale;
    let m10 = linear[3] / scale;
    let m11 = linear[4] / scale;
    let m12 = linear[5] / scale;
    let m20 = linear[6] / scale;
    let m21 = linear[7] / scale;
    let m22 = linear[8] / scale;
    let determinant = m00 * (m11 * m22 - m12 * m21) - m01 * (m10 * m22 - m12 * m20)
        + m02 * (m10 * m21 - m11 * m20);
    if determinant == 0.0 {
        return Err(RenderRequestValidationError::DegenerateObservationFrame);
    }
    Ok(())
}

/// Semantic support associated with one logical observation sample.
///
/// This describes which scene-space angular/image support may contribute to requested meaning. It
/// deliberately does not describe sample counts, sequences, adaptive policy, work distribution, or
/// any other algorithmic sampling strategy. Perspective lattice-cell support is resolved against
/// the exact requested output lattice; it does not carry physical or internal render dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderSamplingSupport {
    kind: RenderSamplingSupportKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RenderSamplingSupportKind {
    IdealRay,
    PerspectiveLatticeCell,
    Cone { half_angle_radians: CanonicalF64 },
}

impl RenderSamplingSupport {
    pub const fn ideal_ray() -> Self {
        Self {
            kind: RenderSamplingSupportKind::IdealRay,
        }
    }

    /// Requests the perspective lattice-cell / pixel-footprint support associated with each
    /// logical sample of a perspective output lattice.
    ///
    /// The concrete footprint is derived from the unjittered observation plus the selected
    /// output's requested lattice dimensions. Physical/internal extents and sample phase are not
    /// semantic state.
    pub const fn perspective_lattice_cell() -> Self {
        Self {
            kind: RenderSamplingSupportKind::PerspectiveLatticeCell,
        }
    }

    pub fn cone(half_angle_radians: f64) -> Result<Self, RenderRequestValidationError> {
        let half_angle_radians = CanonicalF64::new(half_angle_radians, "half_angle_radians")?;
        if half_angle_radians.get() <= 0.0
            || half_angle_radians.get() >= std::f64::consts::FRAC_PI_2
        {
            return Err(RenderRequestValidationError::SamplingConeOutOfRange);
        }
        Ok(Self {
            kind: RenderSamplingSupportKind::Cone { half_angle_radians },
        })
    }

    pub const fn is_ideal_ray(self) -> bool {
        matches!(self.kind, RenderSamplingSupportKind::IdealRay)
    }

    pub const fn is_perspective_lattice_cell(self) -> bool {
        matches!(self.kind, RenderSamplingSupportKind::PerspectiveLatticeCell)
    }

    pub fn cone_half_angle_radians(self) -> Option<f64> {
        match self.kind {
            RenderSamplingSupportKind::IdealRay
            | RenderSamplingSupportKind::PerspectiveLatticeCell => None,
            RenderSamplingSupportKind::Cone { half_angle_radians } => {
                Some(half_angle_radians.get())
            }
        }
    }
}

/// A perspective observation in the canonical renderer observation frame.
///
/// Observation-local coordinates are right-handed with +X to the logical right, +Y to the logical
/// top, and -Z as the canonical forward axis. `observation_to_scene` maps that frame into renderer
/// scene coordinates and must have an invertible linear basis. The vertical field of view is the
/// full angle about -Z and `aspect_ratio` is logical width divided by logical height. No physical
/// surface, pixel format, device, or sampling algorithm is implied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderPerspectiveObservation {
    observation_to_scene: RenderAffineTransform3,
    vertical_field_of_view_radians: CanonicalF64,
    aspect_ratio: CanonicalF64,
    shutter: RenderTimeInterval,
    sampling_support: RenderSamplingSupport,
}

impl RenderPerspectiveObservation {
    pub fn new(
        observation_to_scene: RenderAffineTransform3,
        vertical_field_of_view_radians: f64,
        aspect_ratio: f64,
        shutter: RenderTimeInterval,
        sampling_support: RenderSamplingSupport,
    ) -> Result<Self, RenderRequestValidationError> {
        validate_observation_frame(observation_to_scene)?;
        let vertical_field_of_view_radians = CanonicalF64::new(
            vertical_field_of_view_radians,
            "vertical_field_of_view_radians",
        )?;
        if vertical_field_of_view_radians.get() <= 0.0
            || vertical_field_of_view_radians.get() >= std::f64::consts::PI
        {
            return Err(RenderRequestValidationError::PerspectiveFieldOfViewOutOfRange);
        }
        let aspect_ratio = CanonicalF64::new(aspect_ratio, "aspect_ratio")?;
        if aspect_ratio.get() <= 0.0 {
            return Err(RenderRequestValidationError::NonPositive {
                field: "aspect_ratio",
            });
        }
        Ok(Self {
            observation_to_scene,
            vertical_field_of_view_radians,
            aspect_ratio,
            shutter,
            sampling_support,
        })
    }

    pub const fn observation_to_scene(self) -> RenderAffineTransform3 {
        self.observation_to_scene
    }

    pub fn vertical_field_of_view_radians(self) -> f64 {
        self.vertical_field_of_view_radians.get()
    }

    pub fn aspect_ratio(self) -> f64 {
        self.aspect_ratio.get()
    }

    pub const fn shutter(self) -> RenderTimeInterval {
        self.shutter
    }

    pub const fn sampling_support(self) -> RenderSamplingSupport {
        self.sampling_support
    }
}

/// A scalar renderer probe oriented by the canonical observation frame.
///
/// The probe evaluates around the frame's -Z forward axis. `observation_to_scene` must have an
/// invertible linear basis. This is renderer-semantic observation meaning; it does not imply an
/// image lattice, physical target, or GPU resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderProbeObservation {
    observation_to_scene: RenderAffineTransform3,
    shutter: RenderTimeInterval,
    sampling_support: RenderSamplingSupport,
}

impl RenderProbeObservation {
    pub fn new(
        observation_to_scene: RenderAffineTransform3,
        shutter: RenderTimeInterval,
        sampling_support: RenderSamplingSupport,
    ) -> Result<Self, RenderRequestValidationError> {
        validate_observation_frame(observation_to_scene)?;
        if sampling_support.is_perspective_lattice_cell() {
            return Err(
                RenderRequestValidationError::PerspectiveLatticeCellRequiresPerspectiveObservation,
            );
        }
        Ok(Self {
            observation_to_scene,
            shutter,
            sampling_support,
        })
    }

    pub const fn observation_to_scene(self) -> RenderAffineTransform3 {
        self.observation_to_scene
    }

    pub const fn shutter(self) -> RenderTimeInterval {
        self.shutter
    }

    pub const fn sampling_support(self) -> RenderSamplingSupport {
        self.sampling_support
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderObservationSpec {
    Perspective(RenderPerspectiveObservation),
    Probe(RenderProbeObservation),
}

impl RenderObservationSpec {
    pub const fn shutter(self) -> RenderTimeInterval {
        match self {
            Self::Perspective(observation) => observation.shutter(),
            Self::Probe(observation) => observation.shutter(),
        }
    }

    pub const fn is_probe(self) -> bool {
        matches!(self, Self::Probe(_))
    }
}

/// Renderer-semantic radiometric representation independent of result topology and numeric storage.
///
/// R2 deliberately starts with one precise founding domain rather than an underspecified RGB/color
/// alias: spectral radiance at one explicit wavelength. The represented scalar is spectral radiance
/// per unit wavelength with units W·sr⁻¹·m⁻³. Colorimetric reconstruction, channel sets, and broader
/// representation families remain later concerns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderRadiometricRepresentation {
    wavelength_meters: CanonicalF64,
}

impl RenderRadiometricRepresentation {
    pub fn spectral_at_wavelength_meters(
        wavelength_meters: f64,
    ) -> Result<Self, RenderRequestValidationError> {
        let wavelength_meters =
            CanonicalF64::new(wavelength_meters, "radiometric_wavelength_meters")?;
        if wavelength_meters.get() <= 0.0 {
            return Err(RenderRequestValidationError::NonPositive {
                field: "radiometric_wavelength_meters",
            });
        }
        Ok(Self { wavelength_meters })
    }

    pub fn wavelength_meters(self) -> f64 {
        self.wavelength_meters.get()
    }
}

/// Semantic distance meaning relative to the observation frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderDistanceConvention {
    /// Euclidean distance in metres from the observation origin to the represented point along the
    /// logical sample ray.
    RayDistance,
    /// Scalar projection in metres of the represented point displacement onto the normalized
    /// scene-space direction obtained by transforming the observation frame's -Z forward vector.
    ObservationForwardDepth,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RenderOutputValue {
    Radiance {
        representation: RenderRadiometricRepresentation,
    },
    Distance {
        convention: RenderDistanceConvention,
    },
    /// Renderer-local object identity. Result values refer to `RenderObjectId` semantics, never
    /// source/ECS/product or RunenGPU physical identity.
    ObjectIdentity,
}

/// Logical result shape, independent of physical texture/buffer representation.
///
/// For a perspective observation, `SampleLattice2D` spans the full perspective domain. Logical X
/// increases left-to-right and logical Y increases top-to-bottom. Cell `(x, y)` has its semantic
/// center at `((x + 0.5) / width, (y + 0.5) / height)` across that domain; the observation's
/// `RenderSamplingSupport` describes support around the resulting center ray. This convention says
/// nothing about physical memory order, numeric format, or how an algorithm realizes that support.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderResultTopology {
    kind: RenderResultTopologyKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RenderResultTopologyKind {
    Scalar,
    SampleLattice2D { width: u32, height: u32 },
}

impl RenderResultTopology {
    pub const fn scalar() -> Self {
        Self {
            kind: RenderResultTopologyKind::Scalar,
        }
    }

    pub fn sample_lattice_2d(
        width: u32,
        height: u32,
    ) -> Result<Self, RenderRequestValidationError> {
        if width == 0 || height == 0 {
            return Err(RenderRequestValidationError::InvalidLatticeDimensions);
        }
        Ok(Self {
            kind: RenderResultTopologyKind::SampleLattice2D { width, height },
        })
    }

    pub const fn is_scalar(self) -> bool {
        matches!(self.kind, RenderResultTopologyKind::Scalar)
    }

    pub const fn sample_lattice_dimensions(self) -> Option<(u32, u32)> {
        match self.kind {
            RenderResultTopologyKind::Scalar => None,
            RenderResultTopologyKind::SampleLattice2D { width, height } => Some((width, height)),
        }
    }
}

/// Semantic acceptance tolerance, independent of physical numeric storage and packing.
///
/// For scalar R2 outputs, `Exact` permits no deviation from the exact semantic quantity;
/// `Absolute(max_error)` permits absolute error no greater than `max_error` in that output's
/// semantic units; and `Relative(max_fraction)` permits absolute error no greater than
/// `max_fraction * abs(exact_value)`. A zero exact value therefore requires zero absolute error
/// under relative tolerance. Object identity is categorical and consequently accepts only `Exact`.
/// None of these choices selects an `f16`/`f32`/`f64`, texture format, packing, quantization, or
/// execution method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderSemanticTolerance {
    kind: RenderSemanticToleranceKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum RenderSemanticToleranceKind {
    Exact,
    Absolute { max_error: CanonicalF64 },
    Relative { max_fraction: CanonicalF64 },
}

impl RenderSemanticTolerance {
    pub const fn exact() -> Self {
        Self {
            kind: RenderSemanticToleranceKind::Exact,
        }
    }

    pub fn absolute(max_error: f64) -> Result<Self, RenderRequestValidationError> {
        let max_error = CanonicalF64::new(max_error, "absolute_semantic_tolerance")?;
        if max_error.get() < 0.0 {
            return Err(RenderRequestValidationError::Negative {
                field: "absolute_semantic_tolerance",
            });
        }
        Ok(Self {
            kind: RenderSemanticToleranceKind::Absolute { max_error },
        })
    }

    pub fn relative(max_fraction: f64) -> Result<Self, RenderRequestValidationError> {
        let max_fraction = CanonicalF64::new(max_fraction, "relative_semantic_tolerance")?;
        if max_fraction.get() < 0.0 {
            return Err(RenderRequestValidationError::Negative {
                field: "relative_semantic_tolerance",
            });
        }
        Ok(Self {
            kind: RenderSemanticToleranceKind::Relative { max_fraction },
        })
    }

    pub const fn is_exact(self) -> bool {
        matches!(self.kind, RenderSemanticToleranceKind::Exact)
    }

    pub fn absolute_max_error(self) -> Option<f64> {
        match self.kind {
            RenderSemanticToleranceKind::Absolute { max_error } => Some(max_error.get()),
            _ => None,
        }
    }

    pub fn relative_max_fraction(self) -> Option<f64> {
        match self.kind {
            RenderSemanticToleranceKind::Relative { max_fraction } => Some(max_fraction.get()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderOutputSpec {
    value: RenderOutputValue,
    topology: RenderResultTopology,
    tolerance: RenderSemanticTolerance,
}

impl RenderOutputSpec {
    pub fn new(
        value: RenderOutputValue,
        topology: RenderResultTopology,
        tolerance: RenderSemanticTolerance,
    ) -> Result<Self, RenderRequestValidationError> {
        if matches!(value, RenderOutputValue::ObjectIdentity) && !tolerance.is_exact() {
            return Err(RenderRequestValidationError::IdentityToleranceMustBeExact);
        }
        Ok(Self {
            value,
            topology,
            tolerance,
        })
    }

    pub const fn value(self) -> RenderOutputValue {
        self.value
    }

    pub const fn topology(self) -> RenderResultTopology {
        self.topology
    }

    pub const fn tolerance(self) -> RenderSemanticTolerance {
        self.tolerance
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderRequestedOutput {
    observation_index: usize,
    spec: RenderOutputSpec,
}

impl RenderRequestedOutput {
    pub const fn new(observation_index: usize, spec: RenderOutputSpec) -> Self {
        Self {
            observation_index,
            spec,
        }
    }

    pub const fn observation_index(self) -> usize {
        self.observation_index
    }

    pub const fn spec(self) -> RenderOutputSpec {
        self.spec
    }
}

/// R2 renderer-semantic request envelope.
///
/// All render/shutter times in one request are values on the same renderer-semantic timeline used
/// by the paired scene snapshot's temporal state. Source clocks, simulation ticks, wall clocks, and
/// device generations are projected by integration code and are not carried as request identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderRequest {
    render_interval: RenderTimeInterval,
    observations: Vec<RenderObservationSpec>,
    outputs: Vec<RenderRequestedOutput>,
}

impl RenderRequest {
    pub fn new(
        render_interval: RenderTimeInterval,
        observations: Vec<RenderObservationSpec>,
        outputs: Vec<RenderRequestedOutput>,
    ) -> Result<Self, RenderRequestValidationError> {
        if observations.is_empty() {
            return Err(RenderRequestValidationError::EmptyObservations);
        }
        if outputs.is_empty() {
            return Err(RenderRequestValidationError::EmptyOutputs);
        }

        for (observation_index, observation) in observations.iter().copied().enumerate() {
            if !render_interval.contains(observation.shutter()) {
                return Err(
                    RenderRequestValidationError::ObservationOutsideRenderInterval {
                        observation_index,
                    },
                );
            }
        }

        let mut output_counts = vec![0usize; observations.len()];
        for output in &outputs {
            let observation_index = output.observation_index();
            let Some(observation) = observations.get(observation_index).copied() else {
                return Err(RenderRequestValidationError::OutputObservationOutOfRange {
                    observation_index,
                });
            };
            if observation.is_probe() && !output.spec().topology().is_scalar() {
                return Err(RenderRequestValidationError::ProbeRequiresScalarTopology {
                    observation_index,
                });
            }
            if matches!(observation, RenderObservationSpec::Perspective(_))
                && output.spec().topology().is_scalar()
            {
                return Err(
                    RenderRequestValidationError::PerspectiveRequiresSampleLatticeTopology {
                        observation_index,
                    },
                );
            }
            output_counts[observation_index] += 1;
        }

        if let Some(observation_index) = output_counts.iter().position(|count| *count == 0) {
            return Err(RenderRequestValidationError::ObservationHasNoOutputs {
                observation_index,
            });
        }

        Ok(Self {
            render_interval,
            observations,
            outputs,
        })
    }

    pub const fn render_interval(&self) -> RenderTimeInterval {
        self.render_interval
    }

    pub fn observations(&self) -> &[RenderObservationSpec] {
        &self.observations
    }

    pub fn outputs(&self) -> &[RenderRequestedOutput] {
        &self.outputs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space_time::RenderTimePoint;

    fn interval(start: f64, end: f64) -> RenderTimeInterval {
        RenderTimeInterval::new(
            RenderTimePoint::from_seconds(start).expect("finite time"),
            RenderTimePoint::from_seconds(end).expect("finite time"),
        )
        .expect("ordered interval")
    }

    fn probe_observation(shutter: RenderTimeInterval) -> RenderObservationSpec {
        RenderObservationSpec::Probe(
            RenderProbeObservation::new(
                RenderAffineTransform3::identity(),
                shutter,
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("valid probe"),
        )
    }

    fn perspective_observation(shutter: RenderTimeInterval) -> RenderObservationSpec {
        RenderObservationSpec::Perspective(
            RenderPerspectiveObservation::new(
                RenderAffineTransform3::identity(),
                std::f64::consts::FRAC_PI_2,
                1.0,
                shutter,
                RenderSamplingSupport::ideal_ray(),
            )
            .expect("valid perspective"),
        )
    }

    fn radiance(topology: RenderResultTopology) -> RenderOutputSpec {
        RenderOutputSpec::new(
            RenderOutputValue::Radiance {
                representation: RenderRadiometricRepresentation::spectral_at_wavelength_meters(
                    550e-9,
                )
                .expect("valid radiometric wavelength"),
            },
            topology,
            RenderSemanticTolerance::relative(0.01).expect("valid tolerance"),
        )
        .expect("valid radiance output")
    }

    #[test]
    fn perspective_lattice_cell_support_is_semantic_and_non_algorithmic() {
        let support = RenderSamplingSupport::perspective_lattice_cell();
        assert!(support.is_perspective_lattice_cell());
        assert!(!support.is_ideal_ray());
        assert_eq!(support.cone_half_angle_radians(), None);
    }

    #[test]
    fn probe_rejects_perspective_lattice_cell_support() {
        assert_eq!(
            RenderProbeObservation::new(
                RenderAffineTransform3::identity(),
                interval(0.0, 0.0),
                RenderSamplingSupport::perspective_lattice_cell(),
            ),
            Err(RenderRequestValidationError::PerspectiveLatticeCellRequiresPerspectiveObservation)
        );
    }

    #[test]
    fn perspective_observation_is_independent_of_physical_surface_state() {
        let observation = RenderPerspectiveObservation::new(
            RenderAffineTransform3::identity(),
            std::f64::consts::FRAC_PI_2,
            16.0 / 9.0,
            interval(1.0, 1.01),
            RenderSamplingSupport::ideal_ray(),
        )
        .expect("valid perspective observation");
        assert_eq!(observation.aspect_ratio(), 16.0 / 9.0);
        assert!(observation.sampling_support().is_ideal_ray());
    }

    #[test]
    fn observation_frames_reject_degenerate_linear_basis() {
        let degenerate = RenderAffineTransform3::from_row_major_3x4([
            1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ])
        .expect("finite transform");
        assert_eq!(
            RenderPerspectiveObservation::new(
                degenerate,
                std::f64::consts::FRAC_PI_2,
                1.0,
                interval(0.0, 0.0),
                RenderSamplingSupport::ideal_ray(),
            ),
            Err(RenderRequestValidationError::DegenerateObservationFrame)
        );
        assert_eq!(
            RenderProbeObservation::new(
                degenerate,
                interval(0.0, 0.0),
                RenderSamplingSupport::ideal_ray(),
            ),
            Err(RenderRequestValidationError::DegenerateObservationFrame)
        );
    }

    #[test]
    fn perspective_and_cone_reject_zero_and_upper_boundary() {
        assert_eq!(
            RenderPerspectiveObservation::new(
                RenderAffineTransform3::identity(),
                0.0,
                1.0,
                interval(0.0, 0.0),
                RenderSamplingSupport::ideal_ray(),
            ),
            Err(RenderRequestValidationError::PerspectiveFieldOfViewOutOfRange)
        );
        assert_eq!(
            RenderPerspectiveObservation::new(
                RenderAffineTransform3::identity(),
                std::f64::consts::PI,
                1.0,
                interval(0.0, 0.0),
                RenderSamplingSupport::ideal_ray(),
            ),
            Err(RenderRequestValidationError::PerspectiveFieldOfViewOutOfRange)
        );
        assert_eq!(
            RenderSamplingSupport::cone(0.0),
            Err(RenderRequestValidationError::SamplingConeOutOfRange)
        );
        assert_eq!(
            RenderSamplingSupport::cone(std::f64::consts::FRAC_PI_2),
            Err(RenderRequestValidationError::SamplingConeOutOfRange)
        );
    }

    #[test]
    fn probe_request_uses_scalar_topology_without_image_lattice() {
        let render_interval = interval(0.0, 1.0);
        let probe = probe_observation(interval(0.25, 0.25));
        let output = RenderRequestedOutput::new(0, radiance(RenderResultTopology::scalar()));
        let request = RenderRequest::new(render_interval, vec![probe], vec![output])
            .expect("scalar probe request should validate");
        assert!(request.outputs()[0].spec().topology().is_scalar());
    }

    #[test]
    fn coordinated_observations_support_distinct_result_topologies() {
        let render_interval = interval(0.0, 1.0);
        let perspective = perspective_observation(interval(0.0, 0.5));
        let probe = probe_observation(interval(0.5, 0.5));
        let lattice = RenderResultTopology::sample_lattice_2d(640, 480).expect("valid lattice");
        let request = RenderRequest::new(
            render_interval,
            vec![perspective, probe],
            vec![
                RenderRequestedOutput::new(0, radiance(lattice)),
                RenderRequestedOutput::new(1, radiance(RenderResultTopology::scalar())),
            ],
        )
        .expect("coordinated request should validate");
        assert_eq!(request.observations().len(), 2);
        assert_eq!(
            request.outputs()[0]
                .spec()
                .topology()
                .sample_lattice_dimensions(),
            Some((640, 480))
        );
    }

    #[test]
    fn observation_topologies_and_shutter_support_reject_incoherent_requests() {
        let outside_probe = probe_observation(interval(2.0, 2.0));
        let lattice = RenderResultTopology::sample_lattice_2d(4, 4).expect("valid lattice");
        assert_eq!(
            RenderRequest::new(
                interval(0.0, 1.0),
                vec![outside_probe],
                vec![RenderRequestedOutput::new(0, radiance(lattice))]
            ),
            Err(
                RenderRequestValidationError::ObservationOutsideRenderInterval {
                    observation_index: 0
                }
            )
        );

        let inside_probe = probe_observation(interval(0.5, 0.5));
        assert_eq!(
            RenderRequest::new(
                interval(0.0, 1.0),
                vec![inside_probe],
                vec![RenderRequestedOutput::new(0, radiance(lattice))]
            ),
            Err(RenderRequestValidationError::ProbeRequiresScalarTopology {
                observation_index: 0
            })
        );

        let perspective = perspective_observation(interval(0.5, 0.5));
        assert_eq!(
            RenderRequest::new(
                interval(0.0, 1.0),
                vec![perspective],
                vec![RenderRequestedOutput::new(
                    0,
                    radiance(RenderResultTopology::scalar()),
                )]
            ),
            Err(
                RenderRequestValidationError::PerspectiveRequiresSampleLatticeTopology {
                    observation_index: 0
                }
            )
        );
    }

    #[test]
    fn request_rejects_invalid_output_references_and_unserved_observations() {
        let first = probe_observation(interval(0.0, 0.0));
        assert_eq!(
            RenderRequest::new(
                interval(0.0, 1.0),
                vec![first],
                vec![RenderRequestedOutput::new(
                    1,
                    radiance(RenderResultTopology::scalar()),
                )],
            ),
            Err(RenderRequestValidationError::OutputObservationOutOfRange {
                observation_index: 1
            })
        );

        let first = probe_observation(interval(0.0, 0.0));
        let second = probe_observation(interval(1.0, 1.0));
        assert_eq!(
            RenderRequest::new(
                interval(0.0, 1.0),
                vec![first, second],
                vec![RenderRequestedOutput::new(
                    0,
                    radiance(RenderResultTopology::scalar()),
                )],
            ),
            Err(RenderRequestValidationError::ObservationHasNoOutputs {
                observation_index: 1
            })
        );
    }

    #[test]
    fn spectral_radiance_requires_explicit_positive_wavelength() {
        let representation = RenderRadiometricRepresentation::spectral_at_wavelength_meters(550e-9)
            .expect("positive wavelength should validate");
        assert_eq!(representation.wavelength_meters(), 550e-9);
        assert_eq!(
            RenderRadiometricRepresentation::spectral_at_wavelength_meters(0.0),
            Err(RenderRequestValidationError::NonPositive {
                field: "radiometric_wavelength_meters"
            })
        );
        assert!(matches!(
            RenderRadiometricRepresentation::spectral_at_wavelength_meters(f64::NAN),
            Err(RenderRequestValidationError::SemanticValue(
                RenderSemanticValueError::NonFinite { .. }
            ))
        ));
    }

    #[test]
    fn output_meaning_is_orthogonal_to_topology_and_numeric_storage() {
        let lattice = RenderResultTopology::sample_lattice_2d(2, 2).expect("valid lattice");
        let distance = RenderOutputSpec::new(
            RenderOutputValue::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth,
            },
            lattice,
            RenderSemanticTolerance::absolute(0.001).expect("valid tolerance"),
        )
        .expect("valid distance output");
        assert!(matches!(
            distance.value(),
            RenderOutputValue::Distance {
                convention: RenderDistanceConvention::ObservationForwardDepth
            }
        ));
        assert_eq!(distance.tolerance().absolute_max_error(), Some(0.001));
    }

    #[test]
    fn semantic_tolerance_allows_zero_but_rejects_negative_values() {
        assert_eq!(
            RenderSemanticTolerance::absolute(0.0)
                .expect("zero absolute tolerance is valid")
                .absolute_max_error(),
            Some(0.0)
        );
        assert_eq!(
            RenderSemanticTolerance::relative(0.0)
                .expect("zero relative tolerance is valid")
                .relative_max_fraction(),
            Some(0.0)
        );
        assert_eq!(
            RenderSemanticTolerance::absolute(-0.1),
            Err(RenderRequestValidationError::Negative {
                field: "absolute_semantic_tolerance"
            })
        );
        assert_eq!(
            RenderSemanticTolerance::relative(-0.1),
            Err(RenderRequestValidationError::Negative {
                field: "relative_semantic_tolerance"
            })
        );
    }

    #[test]
    fn identity_output_requires_exact_semantic_tolerance() {
        let tolerance = RenderSemanticTolerance::relative(0.0).expect("valid tolerance");
        assert_eq!(
            RenderOutputSpec::new(
                RenderOutputValue::ObjectIdentity,
                RenderResultTopology::scalar(),
                tolerance
            ),
            Err(RenderRequestValidationError::IdentityToleranceMustBeExact)
        );
    }
}
