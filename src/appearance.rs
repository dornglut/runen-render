//! R3 minimum renderer-semantic appearance contracts.
//!
//! These values describe renderer meaning only. Authored material graphs, compiler IR, WGSL,
//! shader resources, texture bindings, and GPU realization remain outside this module.

use super::space_time::{CanonicalF64, RenderSemanticValueError};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderAppearanceValidationError {
    SemanticValue(RenderSemanticValueError),
    DiffuseReflectanceOutOfRange,
    ZeroDirection,
    NegativeSpectralIrradiance,
}

impl From<RenderSemanticValueError> for RenderAppearanceValidationError {
    fn from(value: RenderSemanticValueError) -> Self {
        Self::SemanticValue(value)
    }
}

impl fmt::Display for RenderAppearanceValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SemanticValue(error) => fmt::Display::fmt(error, f),
            Self::DiffuseReflectanceOutOfRange => {
                write!(
                    f,
                    "diffuse reflectance must lie in the closed interval [0, 1]"
                )
            }
            Self::ZeroDirection => write!(f, "directional emitter direction must be non-zero"),
            Self::NegativeSpectralIrradiance => {
                write!(
                    f,
                    "directional emitter spectral irradiance must be non-negative"
                )
            }
        }
    }
}

impl Error for RenderAppearanceValidationError {}

/// A spectrally neutral Lambertian diffuse material.
///
/// `reflectance` is the dimensionless fraction of incident spectral irradiance reflected by the
/// diffuse model and therefore lies in `[0, 1]`. R3 deliberately does not import authored material
/// graphs or define a general BSDF/material ontology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderDiffuseMaterial {
    reflectance: CanonicalF64,
}

impl RenderDiffuseMaterial {
    pub fn new(reflectance: f64) -> Result<Self, RenderAppearanceValidationError> {
        let reflectance = CanonicalF64::new(reflectance, "diffuse_reflectance")?;
        if !(0.0..=1.0).contains(&reflectance.get()) {
            return Err(RenderAppearanceValidationError::DiffuseReflectanceOutOfRange);
        }
        Ok(Self { reflectance })
    }

    pub fn reflectance(self) -> f64 {
        self.reflectance.get()
    }
}

/// Minimum directional-emitter semantics for the later monochromatic direct-light proof.
///
/// `direction_to_source_scene` is a canonical unit vector in renderer scene coordinates pointing
/// from the shaded point toward the source. The constructor canonicalizes any finite non-zero input
/// direction to unit length, so positive scalar multiples have identical semantic meaning.
///
/// `spectral_irradiance_w_m3` is spectral irradiance per unit wavelength at
/// `wavelength_meters`, with SI units W·m⁻³. It is independent of shader storage, texture formats,
/// device resources, and execution method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RenderDirectionalEmitter {
    direction_to_source_scene: [CanonicalF64; 3],
    wavelength_meters: CanonicalF64,
    spectral_irradiance_w_m3: CanonicalF64,
}

impl RenderDirectionalEmitter {
    pub fn new(
        direction_to_source_scene: [f64; 3],
        wavelength_meters: f64,
        spectral_irradiance_w_m3: f64,
    ) -> Result<Self, RenderAppearanceValidationError> {
        let direction_to_source_scene = canonical_unit_direction(direction_to_source_scene)?;
        let wavelength_meters = CanonicalF64::new(wavelength_meters, "emitter_wavelength_meters")?;
        if wavelength_meters.get() <= 0.0 {
            return Err(RenderAppearanceValidationError::SemanticValue(
                RenderSemanticValueError::NonPositive {
                    field: "emitter_wavelength_meters",
                },
            ));
        }
        let spectral_irradiance_w_m3 =
            CanonicalF64::new(spectral_irradiance_w_m3, "emitter_spectral_irradiance_w_m3")?;
        if spectral_irradiance_w_m3.get() < 0.0 {
            return Err(RenderAppearanceValidationError::NegativeSpectralIrradiance);
        }
        Ok(Self {
            direction_to_source_scene,
            wavelength_meters,
            spectral_irradiance_w_m3,
        })
    }

    pub fn direction_to_source_scene(self) -> [f64; 3] {
        self.direction_to_source_scene.map(CanonicalF64::get)
    }

    pub fn wavelength_meters(self) -> f64 {
        self.wavelength_meters.get()
    }

    pub fn spectral_irradiance_w_m3(self) -> f64 {
        self.spectral_irradiance_w_m3.get()
    }
}

fn canonical_unit_direction(
    direction: [f64; 3],
) -> Result<[CanonicalF64; 3], RenderAppearanceValidationError> {
    let values = [
        CanonicalF64::new(direction[0], "directional_emitter_direction")?.get(),
        CanonicalF64::new(direction[1], "directional_emitter_direction")?.get(),
        CanonicalF64::new(direction[2], "directional_emitter_direction")?.get(),
    ];
    let scale = values.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(RenderAppearanceValidationError::ZeroDirection);
    }

    let scaled = values.map(|value| value / scale);
    let length = scaled
        .iter()
        .copied()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    let normalized = scaled.map(|value| value / length);
    Ok([
        CanonicalF64::new(normalized[0], "directional_emitter_direction")?,
        CanonicalF64::new(normalized[1], "directional_emitter_direction")?,
        CanonicalF64::new(normalized[2], "directional_emitter_direction")?,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diffuse_material_enforces_physical_reflectance_domain() {
        assert_eq!(
            RenderDiffuseMaterial::new(0.25)
                .expect("valid diffuse reflectance")
                .reflectance(),
            0.25
        );
        assert_eq!(
            RenderDiffuseMaterial::new(-0.1),
            Err(RenderAppearanceValidationError::DiffuseReflectanceOutOfRange)
        );
        assert_eq!(
            RenderDiffuseMaterial::new(1.1),
            Err(RenderAppearanceValidationError::DiffuseReflectanceOutOfRange)
        );
    }

    #[test]
    fn directional_emitter_canonicalizes_direction_and_validates_radiometry() {
        let emitter = RenderDirectionalEmitter::new([0.0, 2.0, 0.0], 550e-9, 12.0)
            .expect("valid directional emitter");
        assert_eq!(emitter.direction_to_source_scene(), [0.0, 1.0, 0.0]);
        assert_eq!(emitter.wavelength_meters(), 550e-9);
        assert_eq!(emitter.spectral_irradiance_w_m3(), 12.0);

        assert_eq!(
            RenderDirectionalEmitter::new([0.0, 0.0, 0.0], 550e-9, 1.0),
            Err(RenderAppearanceValidationError::ZeroDirection)
        );
        assert!(matches!(
            RenderDirectionalEmitter::new([1.0, 0.0, 0.0], f64::NAN, 1.0),
            Err(RenderAppearanceValidationError::SemanticValue(
                RenderSemanticValueError::NonFinite { .. }
            ))
        ));
        assert_eq!(
            RenderDirectionalEmitter::new([1.0, 0.0, 0.0], 550e-9, -1.0),
            Err(RenderAppearanceValidationError::NegativeSpectralIrradiance)
        );
    }
}
