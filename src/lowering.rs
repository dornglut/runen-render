//! R6 renderer-owned lowering products.
//!
//! R1-R5 establish semantic scene/request meaning, legal method/representation choices, current
//! admission, and physical output bindings. R6 is the first boundary allowed to turn that admitted
//! renderer meaning into concrete RunenGPU work. It does not re-plan semantics and it does not own
//! RunenGPU preparation, hazard analysis, physical realization, submission, or backend execution.

use super::admission::AdmittedRenderPlan;
use runen_gpu::GpuWorkFragment;

/// Concrete renderer work derived from exactly one admitted R1-R5 plan.
///
/// The admitted plan is retained rather than copied into a second set of renderer-semantic fields.
/// This keeps method, representation, output, approximation, scene, request, and physical binding
/// correlation anchored to the authority that admitted them. The contained RunenGPU fragments are
/// backend-neutral logical work; RunenGPU remains responsible for preparing and executing them.
///
/// A work set may contain more than one device-local fragment, private observation work, merge work,
/// or presentation work. R6 therefore does not encode a permanent one-fragment/one-device
/// assumption.
#[derive(Debug, Clone)]
pub struct RenderWorkSet {
    admitted_plan: AdmittedRenderPlan,
    fragments: Vec<GpuWorkFragment>,
}

impl RenderWorkSet {
    /// Construct a work set after renderer-owned lowering has produced complete checked RunenGPU
    /// work.
    ///
    /// This remains crate-private so callers cannot attach arbitrary GPU work to semantic renderer
    /// authority. Maintained renderer execution is responsible for selecting the exact admitted
    /// method and deriving every fragment before crossing into RunenGPU.
    pub(crate) fn from_lowering(
        admitted_plan: &AdmittedRenderPlan,
        fragments: Vec<GpuWorkFragment>,
    ) -> Self {
        Self {
            admitted_plan: admitted_plan.clone(),
            fragments,
        }
    }

    /// The exact R5 admission from which this work set was lowered.
    pub const fn admitted_plan(&self) -> &AdmittedRenderPlan {
        &self.admitted_plan
    }

    /// Backend-neutral logical RunenGPU work authored by renderer lowering.
    pub fn fragments(&self) -> &[GpuWorkFragment] {
        &self.fragments
    }
}
