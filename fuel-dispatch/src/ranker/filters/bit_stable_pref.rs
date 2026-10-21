// SPDX-License-Identifier: MIT OR Apache-2.0
//! Soft preference: when at least `min_remaining` bit-stable
//! candidates exist, drop the non-bit-stable ones **that share a device
//! with a bit-stable candidate**. Otherwise the pipeline skips this filter
//! (preferences yield to availability).
//!
//! The preference selects a KERNEL on a device; it does not select a DEVICE.
//! Before 0.20.42 it dropped every non-bit-stable candidate whenever any
//! bit-stable one existed, on any device, so a GPU kernel that is honestly
//! non-bit-stable (the Vulkan f32 matmul: subgroup / FADD order is not pinned)
//! could never be compared on cost with the always-bit-stable CPU kernel, and a
//! whole Qwen3 decode step ran on the CPU (4.3 s instead of 0.33 s;
//! `docs/research/qwen3-vulkan-decode-profile.md`). A hard determinism
//! requirement is still available through `PrecisionRequirement::require_bit_stable`
//! (`PrecisionFloorFilter`, strict reproducibility mode), which this filter
//! does not touch.
//!
//! Phase 1.3 of the picker-work arc. This filter is what makes
//! the default picker behavior "bit-stable when there's a choice"
//! while never *requiring* it — that's [`super::PrecisionFloorFilter`]
//! with `require_bit_stable: true`.

use crate::ranker::candidate::Candidate;
use crate::ranker::filter::{AlternativeFilter, FilterClass, FilterContext};

/// Soft filter that keeps only bit-stable-on-same-hardware
/// candidates, as long as doing so leaves at least `min_remaining`
/// alternatives in play.
#[derive(Copy, Clone, Debug)]
pub struct BitStablePreferenceFilter {
    pub min_remaining: usize,
}

impl Default for BitStablePreferenceFilter {
    fn default() -> Self {
        Self { min_remaining: 1 }
    }
}

impl AlternativeFilter for BitStablePreferenceFilter {
    fn filter(&self, alts: &[Candidate], _ctx: &FilterContext) -> Vec<usize> {
        // Devices that have at least one bit-stable candidate of their own.
        let stable_devices: Vec<_> = alts
            .iter()
            .filter(|c| c.precision.bit_stable_on_same_hardware)
            .map(|c| c.device)
            .collect();
        if stable_devices.is_empty() {
            // Nothing bit-stable anywhere: nothing to prefer (the chain
            // soft-skips an empty result).
            return Vec::new();
        }
        // The preference chooses a KERNEL on a device; it must not choose a
        // DEVICE. A candidate is dropped only when a bit-stable candidate
        // exists on its own device, so a device whose kernels are honestly
        // non-bit-stable (e.g. a GPU matmul with unpinned subgroup order) is
        // still ranked against the others by cost.
        alts.iter()
            .enumerate()
            .filter_map(|(i, c)| {
                (c.precision.bit_stable_on_same_hardware || !stable_devices.contains(&c.device))
                    .then_some(i)
            })
            .collect()
    }

    fn classification(&self) -> FilterClass {
        FilterClass::Soft {
            min_remaining: self.min_remaining,
        }
    }

    fn name(&self) -> &'static str {
        "bit-stable-preference"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{KernelCaps, OpParams};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::probe::BackendId;
    use fuel_ir::{DType, DeviceLocation, Layout, Result, Shape};
    use fuel_memory::Storage;
    use std::sync::{Arc, RwLock};

    fn noop(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn candidate(bit_stable: bool) -> Candidate {
        Candidate {
            kernel: noop,
            caps: KernelCaps::empty(),
            backend: BackendId::Cpu,
            device: DeviceLocation::Cpu,
            precision: PrecisionGuarantee {
                bit_stable_on_same_hardware: bit_stable,
                ..PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU
            },
            static_cost: Default::default(),
            inbound_transfer_ns: 0,
            op_params: OpParams::None,
            coupling: Vec::new(),
            kernel_source: "",
            kernel_revision_hash: 0,
        }
    }

    fn candidate_on(bit_stable: bool, backend: BackendId, device: DeviceLocation) -> Candidate {
        Candidate {
            backend,
            device,
            ..candidate(bit_stable)
        }
    }

    fn ctx<'a>() -> FilterContext<'a> {
        static LAYOUTS: std::sync::OnceLock<Vec<Layout>> = std::sync::OnceLock::new();
        let layouts = LAYOUTS.get_or_init(|| vec![Layout::contiguous(Shape::from(vec![4]))]);
        FilterContext::new(OpKind::AddElementwise, &[DType::F32], layouts)
    }

    #[test]
    fn keeps_only_bit_stable_when_available() {
        let f = BitStablePreferenceFilter::default();
        let alts = vec![
            candidate(false),
            candidate(true),
            candidate(false),
            candidate(true),
        ];
        assert_eq!(f.filter(&alts, &ctx()), vec![1, 3]);
    }

    #[test]
    fn non_bit_stable_candidate_on_another_device_is_kept() {
        // The Qwen3-on-Vulkan shape: a non-bit-stable GPU kernel and a
        // bit-stable CPU kernel. The preference must not pick the device.
        let f = BitStablePreferenceFilter::default();
        let gpu = DeviceLocation::Vulkan { gpu_id: 0 };
        let alts = vec![
            candidate_on(false, BackendId::Vulkan, gpu),
            candidate_on(true, BackendId::Cpu, DeviceLocation::Cpu),
        ];
        assert_eq!(f.filter(&alts, &ctx()), vec![0, 1]);
    }

    #[test]
    fn non_bit_stable_sibling_on_the_same_device_is_still_dropped() {
        // Sabotage sibling of the test above: the same preference still
        // removes a non-bit-stable kernel when a bit-stable one shares ITS
        // device, so the cross-device test passes because of the device
        // grouping, not because the filter stopped filtering.
        let f = BitStablePreferenceFilter::default();
        let gpu = DeviceLocation::Vulkan { gpu_id: 0 };
        let alts = vec![
            candidate_on(false, BackendId::Vulkan, gpu),
            candidate_on(true, BackendId::Vulkan, gpu),
            candidate_on(true, BackendId::Cpu, DeviceLocation::Cpu),
        ];
        assert_eq!(f.filter(&alts, &ctx()), vec![1, 2]);
    }

    #[test]
    fn returns_empty_when_no_bit_stable_candidates() {
        // Result is empty — the chain pipeline will soft-skip this
        // filter (its classification is Soft { min_remaining: 1 }).
        let f = BitStablePreferenceFilter::default();
        let alts = vec![candidate(false), candidate(false)];
        assert!(f.filter(&alts, &ctx()).is_empty());
    }

    #[test]
    fn classification_is_soft_with_min_remaining() {
        let f = BitStablePreferenceFilter { min_remaining: 2 };
        assert_eq!(f.classification(), FilterClass::Soft { min_remaining: 2 },);
    }

    #[test]
    fn min_remaining_drives_chain_skip_decision() {
        // Verify integration with apply_filter_chain: with two
        // candidates (only one bit-stable) and min_remaining=2, the
        // filter returns [1] which is below the threshold → pipeline
        // skips.
        use crate::ranker::alternative_set::AlternativeSet;
        use crate::ranker::chain::apply_filter_chain;
        let mut set = AlternativeSet::from_candidates(vec![candidate(false), candidate(true)]);
        let filters: Vec<Box<dyn AlternativeFilter>> =
            vec![Box::new(BitStablePreferenceFilter { min_remaining: 2 })];
        apply_filter_chain(&mut set, &filters, &ctx()).expect("soft skip succeeds");
        assert_eq!(set.len(), 2, "below min_remaining → filter skipped");
    }

    #[test]
    fn applies_when_at_or_above_min_remaining() {
        use crate::ranker::alternative_set::AlternativeSet;
        use crate::ranker::chain::apply_filter_chain;
        let mut set = AlternativeSet::from_candidates(vec![
            candidate(true),
            candidate(false),
            candidate(true),
        ]);
        let filters: Vec<Box<dyn AlternativeFilter>> =
            vec![Box::new(BitStablePreferenceFilter { min_remaining: 2 })];
        apply_filter_chain(&mut set, &filters, &ctx()).expect("applies");
        assert_eq!(set.len(), 2, "two bit-stable survive; filter applied");
        assert!(
            set.alternatives()
                .iter()
                .all(|c| c.precision.bit_stable_on_same_hardware)
        );
    }
}
