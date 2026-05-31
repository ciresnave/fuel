//! Soft preference: when the decision point's inputs are
//! non-contiguous AND at least one candidate advertises
//! `caps.strided_input`, drop the candidates that don't. Avoids the
//! executor's auto-Contiguize materialize step.
//!
//! Phase 1.3 of the picker-work arc.
//!
//! # When this fires vs no-ops
//!
//! - Every input is contiguous → no-op (returns all indices). The
//!   strided-input capability provides no benefit if the kernel
//!   would see contiguous bytes anyway.
//! - At least one input is non-contiguous → prefer kernels with
//!   `caps.strided_input = true`. If none qualify, the chain
//!   pipeline skips per `min_remaining`.

use crate::ranker::candidate::Candidate;
use crate::ranker::filter::{AlternativeFilter, FilterClass, FilterContext};

/// Soft filter that prefers kernels with `caps.strided_input` when
/// the input layouts demand it.
#[derive(Copy, Clone, Debug)]
pub struct StridedInputPreferenceFilter {
    pub min_remaining: usize,
}

impl Default for StridedInputPreferenceFilter {
    fn default() -> Self {
        Self { min_remaining: 1 }
    }
}

impl AlternativeFilter for StridedInputPreferenceFilter {
    fn filter(&self, alts: &[Candidate], ctx: &FilterContext) -> Vec<usize> {
        // No-op when every input is contiguous — the strided
        // capability is only valuable when the kernel can skip the
        // auto-Contiguize materialize step the executor would
        // otherwise run.
        let any_non_contig = ctx.input_layouts.iter().any(|l| !l.is_contiguous());
        if !any_non_contig {
            return (0..alts.len()).collect();
        }
        alts.iter()
            .enumerate()
            .filter_map(|(i, c)| c.caps.strided_input.then_some(i))
            .collect()
    }

    fn classification(&self) -> FilterClass {
        FilterClass::Soft {
            min_remaining: self.min_remaining,
        }
    }

    fn name(&self) -> &'static str {
        "strided-input-preference"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::PrecisionGuarantee;
    use crate::kernel::{KernelCaps, OpParams};
    use fuel_core_types::dispatch::OpKind;
    use fuel_core_types::probe::BackendId;
    use fuel_core_types::{DType, DeviceLocation, Layout, Result, Shape, StrideVec};
    use fuel_storage::Storage;
    use smallvec::smallvec;
    use std::sync::{Arc, RwLock};

    fn noop(
        _i: &[Arc<RwLock<Storage>>],
        _o: &mut [Arc<RwLock<Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> Result<()> {
        Ok(())
    }

    fn candidate(strided: bool) -> Candidate {
        Candidate {
            kernel: noop,
            caps: if strided { KernelCaps::strided_input() } else { KernelCaps::empty() },
            backend: BackendId::Cpu,
            device: DeviceLocation::Cpu,
            precision: PrecisionGuarantee::PRIMITIVE_DETERMINISTIC_CPU,
            static_cost: Default::default(),
            op_params: OpParams::None,
            coupling: Vec::new(),
        }
    }

    fn contig_ctx<'a>() -> FilterContext<'a> {
        static LAYOUTS: std::sync::OnceLock<Vec<Layout>> = std::sync::OnceLock::new();
        let layouts = LAYOUTS.get_or_init(|| vec![Layout::contiguous(Shape::from(vec![4]))]);
        FilterContext::new(OpKind::AddElementwise, &[DType::F32], layouts)
    }

    fn non_contig_ctx<'a>() -> FilterContext<'a> {
        static LAYOUTS: std::sync::OnceLock<Vec<Layout>> = std::sync::OnceLock::new();
        let layouts = LAYOUTS.get_or_init(|| {
            // Non-contiguous: stride 2 on a length-4 axis.
            let shape = Shape::from(vec![4]);
            let strides: StrideVec = smallvec![2isize];
            vec![Layout::new(shape, strides, 0)]
        });
        FilterContext::new(OpKind::AddElementwise, &[DType::F32], layouts)
    }

    #[test]
    fn contiguous_inputs_is_no_op() {
        let f = StridedInputPreferenceFilter::default();
        let alts = vec![candidate(false), candidate(true), candidate(false)];
        assert_eq!(f.filter(&alts, &contig_ctx()), vec![0, 1, 2]);
    }

    #[test]
    fn non_contiguous_input_prefers_strided_capable() {
        let f = StridedInputPreferenceFilter::default();
        let alts = vec![candidate(false), candidate(true), candidate(false), candidate(true)];
        assert_eq!(f.filter(&alts, &non_contig_ctx()), vec![1, 3]);
    }

    #[test]
    fn no_strided_capable_returns_empty_then_chain_skips() {
        // No candidate is strided-capable, so the filter returns
        // empty — the chain pipeline soft-skips because
        // min_remaining=1 means "leave at least one" and zero <
        // one.
        use crate::ranker::alternative_set::{AlternativeSet, DEFAULT_MAX_N};
        use crate::ranker::chain::apply_filter_chain;
        let f = StridedInputPreferenceFilter::default();
        let mut set = AlternativeSet::from_candidates(
            vec![candidate(false), candidate(false)],
            DEFAULT_MAX_N,
        );
        let filters: Vec<Box<dyn AlternativeFilter>> = vec![Box::new(f)];
        apply_filter_chain(&mut set, &filters, &non_contig_ctx()).expect("soft skip");
        assert_eq!(set.len(), 2, "no strided-capable → filter skipped");
    }

    #[test]
    fn classification_is_soft_with_default_min_remaining_one() {
        let f = StridedInputPreferenceFilter::default();
        assert_eq!(f.classification(), FilterClass::Soft { min_remaining: 1 });
        assert_eq!(f.name(), "strided-input-preference");
    }
}
