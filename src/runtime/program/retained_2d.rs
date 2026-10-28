//! Bounded reuse of RunenGPU-validated, maintained 2D program contracts.
//!
//! Only the nine maintained vector fragment variants and shaped text have slots.
//! RunenGPU remains the sole shader analyzer and program-contract authority.
//! These records are CPU-side immutable contracts, not GPU pipelines or device
//! capability permits. Errors are never retained.

use runen_gpu::{
    GpuAdmittedProgramSource, GpuBindingLayoutRefinement, GpuEntryPointName,
    GpuProgramContractError, GpuProgramDescriptor,
};
use std::sync::{LazyLock, Mutex};

const MAINTAINED_2D_VARIANTS: usize = 10;
const SHAPED_TEXT_SLOT: usize = 9;

struct ValidatedProgram {
    descriptor: GpuProgramDescriptor,
    entry_points: [GpuEntryPointName; 2],
    refinements: Vec<GpuBindingLayoutRefinement>,
}

struct Maintained2dDescriptors {
    slots: [Option<ValidatedProgram>; MAINTAINED_2D_VARIANTS],
}

impl Default for Maintained2dDescriptors {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
        }
    }
}

impl Maintained2dDescriptors {
    fn get_or_validate(
        &mut self,
        slot: usize,
        source: GpuAdmittedProgramSource,
        entry_points: [GpuEntryPointName; 2],
        refinements: Vec<GpuBindingLayoutRefinement>,
        validate: impl FnOnce(
            GpuAdmittedProgramSource,
            [GpuEntryPointName; 2],
            Vec<GpuBindingLayoutRefinement>,
        ) -> Result<GpuProgramDescriptor, GpuProgramContractError>,
    ) -> Result<GpuProgramDescriptor, GpuProgramContractError> {
        // Record identity implies exact canonical bytes, revision and admission
        // provenance. Different selected interfaces/refinements never alias.
        if let Some(existing) = self.slots[slot].as_ref() {
            if existing.descriptor.source().is_same_record(&source)
                && existing.entry_points == entry_points
                && existing.refinements == refinements
            {
                return Ok(existing.descriptor.clone());
            }
        }

        // Do not replace a successful contract unless the new one validates.
        // A failed construction preserves the previous record and retry behavior.
        let descriptor = validate(source, entry_points.clone(), refinements.clone())?;
        self.slots[slot] = Some(ValidatedProgram {
            descriptor: descriptor.clone(),
            entry_points,
            refinements,
        });
        Ok(descriptor)
    }
}

// The lock covers cold validation, not just insertion, so concurrent callers
// cannot repeatedly analyze the same first-use contract.
static PROGRAMS: LazyLock<Mutex<Maintained2dDescriptors>> =
    LazyLock::new(|| Mutex::new(Maintained2dDescriptors::default()));

fn retain(
    slot: usize,
    source: GpuAdmittedProgramSource,
    entry_points: [GpuEntryPointName; 2],
    refinements: Vec<GpuBindingLayoutRefinement>,
) -> Result<GpuProgramDescriptor, GpuProgramContractError> {
    PROGRAMS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get_or_validate(
            slot,
            source,
            entry_points,
            refinements,
            |source, entries, bindings| GpuProgramDescriptor::new(source, entries, bindings),
        )
}

pub(crate) fn retained_vector_2d_descriptor(
    fragment: &'static str,
    source: GpuAdmittedProgramSource,
    entry_points: [GpuEntryPointName; 2],
    refinements: Vec<GpuBindingLayoutRefinement>,
) -> Result<GpuProgramDescriptor, GpuProgramContractError> {
    let slot = match fragment {
        "fs_coverage" => 0,
        "fs_sample_mask_fill" => 1,
        "fs_sample_merge" => 2,
        "fs_sample_merge_clipped" => 3,
        "fs_sample_mask_fill_clipped" => 4,
        "fs_sample_gradient" => 5,
        "fs_sample_gradient_clipped" => 6,
        "fs_sample_image" => 7,
        "fs_sample_resolve" => 8,
        // Unknown/new contracts still receive full RunenGPU validation;
        // only the maintained finite set is retained.
        _ => return GpuProgramDescriptor::new(source, entry_points, refinements),
    };
    retain(slot, source, entry_points, refinements)
}

pub(crate) fn retained_shaped_text_2d_descriptor(
    source: GpuAdmittedProgramSource,
    entry_points: [GpuEntryPointName; 2],
    refinements: Vec<GpuBindingLayoutRefinement>,
) -> Result<GpuProgramDescriptor, GpuProgramContractError> {
    retain(SHAPED_TEXT_SLOT, source, entry_points, refinements)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::program::{
        MaintainedShaderProgram, MaintainedShaderSpec, VECTOR_REVISION, VECTOR_WGSL,
        build_maintained_program, retained_vector_source,
    };
    use runen_gpu::{GpuBindingKey, GpuTextureSampleClass};
    use std::cell::Cell;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn entries(fragment: &str) -> [GpuEntryPointName; 2] {
        [
            GpuEntryPointName::new("vs_main").expect("maintained vertex name"),
            GpuEntryPointName::new(fragment).expect("valid fragment name"),
        ]
    }

    fn checked(
        calls: &Cell<usize>,
        source: GpuAdmittedProgramSource,
        entries: [GpuEntryPointName; 2],
        refinements: Vec<GpuBindingLayoutRefinement>,
    ) -> Result<GpuProgramDescriptor, GpuProgramContractError> {
        calls.set(calls.get() + 1);
        GpuProgramDescriptor::new(source, entries, refinements)
    }

    #[test]
    fn repeated_maintained_preparation_reuses_the_same_validated_record() {
        let source = retained_vector_source().expect("maintained vector source");
        let first = retained_vector_2d_descriptor(
            "fs_coverage",
            source.clone(),
            entries("fs_coverage"),
            Vec::new(),
        )
        .expect("first program validation");
        let again = retained_vector_2d_descriptor(
            "fs_coverage",
            source,
            entries("fs_coverage"),
            Vec::new(),
        )
        .expect("reused program");
        assert!(first.is_same_record(&again));
    }

    #[test]
    fn warm_contract_skips_real_construction_but_entrypoints_and_refinements_do_not_alias() {
        let source = retained_vector_source().expect("maintained vector source");
        let mut cache = Maintained2dDescriptors::default();
        let calls = Cell::new(0);
        let same = entries("fs_sample_merge");
        let unfilterable = vec![
            GpuBindingLayoutRefinement::new(
                GpuBindingKey::try_new(0, 6).expect("sample binding key"),
            )
            .with_texture_sample_class(GpuTextureSampleClass::FloatUnfilterable),
        ];
        let initial = cache
            .get_or_validate(
                2,
                source.clone(),
                same.clone(),
                unfilterable.clone(),
                |s, e, r| checked(&calls, s, e, r),
            )
            .expect("valid descriptor");
        let warm = cache
            .get_or_validate(2, source.clone(), same.clone(), unfilterable, |s, e, r| {
                checked(&calls, s, e, r)
            })
            .expect("warm descriptor");
        assert!(initial.is_same_record(&warm));
        assert_eq!(calls.get(), 1, "warm preparation must skip Naga admission");

        let refined = vec![
            GpuBindingLayoutRefinement::new(
                GpuBindingKey::try_new(0, 6).expect("sample binding key"),
            )
            .with_texture_sample_class(GpuTextureSampleClass::FloatFilterable),
        ];
        let changed = cache
            .get_or_validate(2, source.clone(), same, refined, |s, e, r| {
                checked(&calls, s, e, r)
            })
            .expect("valid distinct refinement");
        assert!(!initial.is_same_record(&changed));
        assert_eq!(calls.get(), 2);

        let other = cache
            .get_or_validate(2, source, entries("fs_coverage"), Vec::new(), |s, e, r| {
                checked(&calls, s, e, r)
            })
            .expect("different selected entrypoint");
        assert!(!changed.is_same_record(&other));
        assert_eq!(calls.get(), 3);
        assert_eq!(cache.slots.iter().flatten().count(), 1);
    }

    #[test]
    fn revision_and_canonical_content_changes_require_fresh_gpu_validation() {
        let mut cache = Maintained2dDescriptors::default();
        let calls = Cell::new(0);
        let source = retained_vector_source().expect("maintained source");
        let first = cache
            .get_or_validate(0, source, entries("fs_coverage"), Vec::new(), |s, e, r| {
                checked(&calls, s, e, r)
            })
            .expect("initial revision");

        let next = build_maintained_program(MaintainedShaderSpec {
            program: MaintainedShaderProgram::Vector,
            revision: VECTOR_REVISION + 1,
            wgsl: VECTOR_WGSL,
        })
        .expect("different admitted revision");
        let revised = cache
            .get_or_validate(
                0,
                next.admitted().clone(),
                entries("fs_coverage"),
                Vec::new(),
                |s, e, r| checked(&calls, s, e, r),
            )
            .expect("fresh revised descriptor");
        assert!(!first.is_same_record(&revised));

        let content = format!("{VECTOR_WGSL}\n// changed content for revision test\n");
        let different = build_maintained_program(MaintainedShaderSpec {
            program: MaintainedShaderProgram::Vector,
            revision: VECTOR_REVISION + 2,
            wgsl: &content,
        })
        .expect("different canonical bytes");
        let updated = cache
            .get_or_validate(
                0,
                different.admitted().clone(),
                entries("fs_coverage"),
                Vec::new(),
                |s, e, r| checked(&calls, s, e, r),
            )
            .expect("fresh content descriptor");
        assert!(!revised.is_same_record(&updated));
        assert_eq!(calls.get(), 3);
        assert_eq!(cache.slots.iter().flatten().count(), 1);
    }

    #[test]
    fn failed_selected_entrypoint_keeps_typed_error_and_cannot_poison_a_valid_record() {
        let mut cache = Maintained2dDescriptors::default();
        let calls = Cell::new(0);
        let source = retained_vector_source().expect("maintained source");
        let first = cache
            .get_or_validate(
                0,
                source.clone(),
                entries("fs_coverage"),
                Vec::new(),
                |s, e, r| checked(&calls, s, e, r),
            )
            .expect("valid descriptor");
        let error: GpuProgramContractError = cache
            .get_or_validate(
                0,
                source.clone(),
                entries("fs_missing_from_maintained_source"),
                Vec::new(),
                |s, e, r| checked(&calls, s, e, r),
            )
            .expect_err("unknown entrypoint must fail with original GPU contract error");
        let _ = error;
        let again = cache
            .get_or_validate(0, source, entries("fs_coverage"), Vec::new(), |s, e, r| {
                checked(&calls, s, e, r)
            })
            .expect("prior valid contract survived failed admission");
        assert!(first.is_same_record(&again));
        assert_eq!(calls.get(), 2, "failed contract is not cached");
    }

    #[test]
    fn concurrent_cold_requests_perform_one_authoritative_analysis() {
        let cache = Arc::new(Mutex::new(Maintained2dDescriptors::default()));
        let calls = Arc::new(AtomicUsize::new(0));
        let source = retained_vector_source().expect("maintained source");
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let calls = Arc::clone(&calls);
                let source = source.clone();
                std::thread::spawn(move || {
                    cache
                        .lock()
                        .expect("test mutex")
                        .get_or_validate(
                            0,
                            source,
                            entries("fs_coverage"),
                            Vec::new(),
                            |s, e, r| {
                                calls.fetch_add(1, Ordering::SeqCst);
                                GpuProgramDescriptor::new(s, e, r)
                            },
                        )
                        .expect("concurrent descriptor")
                })
            })
            .collect();
        let records: Vec<_> = handles
            .into_iter()
            .map(|handle| handle.join().expect("thread result"))
            .collect();
        assert!(records.iter().all(|r| r.is_same_record(&records[0])));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}
