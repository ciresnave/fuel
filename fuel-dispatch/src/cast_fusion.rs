//! Cast-fusion capability predicate — builds the
//! [`fuel_graph::opt::CapabilityPredicate`] consumed by
//! `CastFusionRule` from the live `KernelBindingTable`.
//!
//! The rule itself lives in `fuel-graph` and is decoupled from the
//! binding table by design (`fuel-graph` doesn't know what
//! `BackendId` means). The predicate factory here closes the gap:
//! it walks the binding table to answer "does any registered
//! backend have a kernel for `(op_kind, dtypes)`?"
//!
//! ## Usage
//!
//! ```rust,ignore
//! use fuel_graph::opt::{CastFusionRule, RuleRegistry};
//! use fuel_storage::cast_fusion::cast_fusion_predicate;
//!
//! let registry = RuleRegistry::default_rules()
//!     .with_rule(Box::new(CastFusionRule::new(cast_fusion_predicate())));
//! ```
//!
//! The predicate consults [`crate::dispatch::global_bindings`] at
//! call time, so it picks up registrations done after construction.

use std::sync::Arc;

use fuel_core_types::{probe::BackendId, DType};
use fuel_core_types::dispatch::OpKind;
use fuel_graph::opt::CapabilityPredicate;
use fuel_graph::Op;

use crate::dispatch::global_bindings;

/// Build a [`CapabilityPredicate`] backed by the process-wide
/// [`crate::dispatch::global_bindings`].
///
/// The predicate answers `true` if at least one registered backend
/// has a kernel for `(op, dtypes)`. The route picker chooses
/// among the backends later; the cast-fusion rule only cares that
/// *some* backend can execute the rewritten consumer.
///
/// Op → OpKind mapping is partial: the `op_kind` helper covers
/// the ops we expect to benefit most from cast-fusion (elementwise
/// unary/binary, reductions, MatMul, fused linear, attention). Ops
/// outside this set always return `false` from the predicate; the
/// cast survives and dispatch falls back to the existing
/// `Cast → Op` chain. Extending coverage is mechanical — add a
/// match arm to [`op_kind`].
pub fn cast_fusion_predicate() -> CapabilityPredicate {
    Arc::new(|op: &Op, dtypes: &[DType]| -> bool {
        let Some(kind) = op_kind(op) else { return false; };
        let bindings = global_bindings();
        // Try every registered backend. The route picker decides
        // which to use later; the rule only needs to know *some*
        // backend can execute the rewritten consumer.
        for backend in ALL_BACKENDS {
            if bindings.lookup(kind, dtypes, *backend).is_ok() {
                return true;
            }
        }
        false
    })
}

/// Every BackendId variant the cast-fusion predicate consults.
/// Iterating an enum requires either an explicit list or
/// `strum::EnumIter` — the explicit list is cheap and predicate
/// performance isn't sensitive (one cast-fusion check per Cast →
/// Op pair per optimization pass).
const ALL_BACKENDS: &[BackendId] = &[
    BackendId::Cpu,
    BackendId::Cuda,
    BackendId::Vulkan,
    BackendId::Metal,
    BackendId::Mkl,
    BackendId::Aocl,
];

/// Map a `fuel_graph::Op` variant to its corresponding `OpKind`.
/// Returns `None` for ops that don't carry an `OpKind` mapping
/// (the cast-fusion predicate then conservatively declines the
/// rewrite, leaving the Cast in place).
///
/// The mapping is partial by design. Adding a variant to either
/// `Op` or `OpKind` doesn't break this function — it just means
/// the new variant's cast-fusion opportunities are missed until
/// an arm is added.
fn op_kind(op: &Op) -> Option<OpKind> {
    match op {
        // --- elementwise binary ---
        Op::Add => Some(OpKind::AddElementwise),
        Op::Sub => Some(OpKind::SubElementwise),
        Op::Mul => Some(OpKind::MulElementwise),
        Op::Div => Some(OpKind::DivElementwise),
        Op::Maximum => Some(OpKind::MaximumElementwise),
        Op::Minimum => Some(OpKind::MinimumElementwise),
        Op::Pow => Some(OpKind::PowElementwise),
        Op::Rem => Some(OpKind::RemElementwise),

        // --- elementwise unary ---
        Op::Relu => Some(OpKind::ReluElementwise),
        Op::Neg => Some(OpKind::NegElementwise),
        Op::Sqr => Some(OpKind::SqrElementwise),
        Op::Sqrt => Some(OpKind::SqrtElementwise),
        Op::Recip => Some(OpKind::RecipElementwise),
        Op::Abs => Some(OpKind::AbsElementwise),
        Op::Tanh => Some(OpKind::TanhElementwise),
        Op::Exp => Some(OpKind::ExpElementwise),
        Op::Log => Some(OpKind::LogElementwise),
        Op::Sin => Some(OpKind::SinElementwise),
        Op::Cos => Some(OpKind::CosElementwise),
        Op::Sigmoid => Some(OpKind::SigmoidElementwise),
        Op::Silu => Some(OpKind::SiluElementwise),
        Op::Gelu => Some(OpKind::GeluElementwise),
        Op::Step => Some(OpKind::StepElementwise),
        Op::Floor => Some(OpKind::FloorElementwise),
        Op::Ceil => Some(OpKind::CeilElementwise),
        Op::Round => Some(OpKind::RoundElementwise),
        Op::Sign => Some(OpKind::SignElementwise),
        Op::Erf => Some(OpKind::ErfElementwise),
        Op::GeluErf => Some(OpKind::GeluErfElementwise),
        Op::Rsqrt => Some(OpKind::RsqrtElementwise),

        // --- compares (T → U8) ---
        Op::Equal => Some(OpKind::EqualElementwise),
        Op::Ne => Some(OpKind::NotEqualElementwise),
        Op::Lt => Some(OpKind::LessElementwise),
        Op::Le => Some(OpKind::LessEqualElementwise),
        Op::Gt => Some(OpKind::GreaterElementwise),
        Op::Ge => Some(OpKind::GreaterEqualElementwise),

        // --- reductions (scalar + along-one-dim) ---
        // Both Op::*All and Op::*Dim map to the same OpKind; the
        // binding-table dispatch threads dim info through
        // OpParams::Reduce, not the OpKind key.
        Op::SumAll | Op::SumDim(_) => Some(OpKind::SumReduce),
        Op::MaxAll | Op::MaxDim(_) => Some(OpKind::MaxReduce),
        Op::MinAll | Op::MinDim(_) => Some(OpKind::MinReduce),
        Op::MeanAll | Op::MeanDim(_) => Some(OpKind::MeanReduce),
        Op::ReduceSumTo(_) => Some(OpKind::ReduceSumTo),
        Op::ReduceMaxTo(_) => Some(OpKind::ReduceMaxTo),

        // --- dense linear algebra ---
        Op::MatMul => Some(OpKind::MatMul),
        // Phase 7.6 step 5 (2026-05-11): FusedLinear lives behind
        // `Op::Fused(FUSED_LINEAR, _)` now. The cast-fusion rule
        // doesn't need to match it specially — the route through
        // Op::Fused already carries the correct dtype info on the
        // operand nodes; if cast-fusion ever needs to fire on a
        // fused-linear consumer, add an Op::Fused(FUSED_LINEAR, _)
        // arm.

        // Everything else is a structural or specialized op the
        // cast-fusion rule doesn't target today. Notably: Cast
        // itself (the rule doesn't fire on the Cast node), Copy /
        // Move / Release (residency), view ops, FlashAttn /
        // PagedAttn (specialized inputs), Affine (carries a
        // float scalar in the variant — predicate can't easily
        // synthesize a matching binding-table key), Where /
        // MaskedFill (multi-dtype inputs without a clean
        // "single-cast-input" pattern), Triu / Tril / Roll /
        // Flip / Pad / CumSum / IndexSelect / Gather / etc.
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_core_types::Shape;
    use fuel_graph::opt::{CastFusionRule, RuleRegistry};
    use fuel_graph::Tensor;
    use fuel_graph::topo_order_multi;
    use std::sync::Arc as StdArc;

    fn cpu_dev() -> &'static StdArc<dyn fuel_core_types::DynBackendDevice> {
        use std::sync::OnceLock;
        static D: OnceLock<StdArc<dyn fuel_core_types::DynBackendDevice>> = OnceLock::new();
        D.get_or_init(|| StdArc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
    }

    /// Sanity: predicate returns true for an (op, dtypes) combo
    /// that is registered in the default CPU bindings. We don't
    /// rely on a *specific* registration — the binding table is
    /// shared across tests — but `(NegElementwise, [F32, F32])` is
    /// part of the standard CPU registration set so it should
    /// always be there.
    #[test]
    fn predicate_accepts_registered_op() {
        let predicate = cast_fusion_predicate();
        // Build the dtype vector the predicate expects:
        // [input dtypes ..., output dtype]. For unary Neg: input
        // F32, output F32.
        let dtypes = [DType::F32, DType::F32];
        assert!(predicate(&Op::Neg, &dtypes));
    }

    /// Predicate returns false for ops outside its OpKind mapping.
    /// Op::Const has no kernel-dispatch shape; the predicate
    /// declines and the cast (if any) is preserved.
    #[test]
    fn predicate_rejects_unmapped_op() {
        let predicate = cast_fusion_predicate();
        let dtypes = [DType::F32];
        assert!(!predicate(&Op::Const, &dtypes));
    }

    /// End-to-end: a graph `x:f32 → Cast(BF16) → Neg(BF16)`
    /// optimized with the predicate-backed CastFusionRule
    /// eliminates the Cast (since the CPU backend registers
    /// `(NegElementwise, [F32, F32])` and `[BF16, BF16]`).
    #[test]
    fn end_to_end_cast_neg_collapses_to_neg() {
        // Force the global binding table to be initialized (this
        // also wires the standard CPU registrations).
        let _bindings = global_bindings();

        let x = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        let xc = x.cast(DType::BF16);
        let y = xc.neg();
        let graph = y.graph().clone();

        let registry = RuleRegistry::new()
            .with_rule(Box::new(CastFusionRule::new(cast_fusion_predicate())));
        let new_roots = registry.optimize_to_fixpoint(&graph, &[y.id()]);
        assert_eq!(new_roots.len(), 1);
        let new_root = new_roots[0];

        let g = graph.read().unwrap();
        // No Cast remains reachable from the new root.
        let reachable = topo_order_multi(&g, &[new_root]);
        let cast_count = reachable.iter()
            .filter(|&&n| matches!(g.node(n).op, Op::Cast(_)))
            .count();
        assert_eq!(cast_count, 0,
            "Cast should be eliminated when NegElementwise is registered for both [F32,F32] and [BF16,BF16]");
        // The rewritten Neg consumes the original F32 Const directly.
        assert!(matches!(g.node(new_root).op, Op::Neg));
        assert_eq!(g.node(new_root).inputs, vec![x.id()]);
    }

    /// The end-to-end test above also verifies the predicate
    /// answers for the *source* dtype path: the rewritten
    /// `Neg(F32)` must have a registered kernel. If only the
    /// target-dtype path were registered (`Neg(BF16)` but not
    /// `Neg(F32)`) the rule should *not* fire — but we don't have
    /// a clean way to deregister bindings mid-test, so we trust
    /// the unit tests at the graph layer to verify the no-fire
    /// path and check the live binding table here for the
    /// "yes-fire" side only.
    #[test]
    fn live_binding_table_contains_neg_for_both_dtypes() {
        let _bindings = global_bindings();
        let predicate = cast_fusion_predicate();
        assert!(predicate(&Op::Neg, &[DType::F32, DType::F32]),
            "fuel-cpu-backend's standard registration includes Neg[F32]");
        assert!(predicate(&Op::Neg, &[DType::BF16, DType::BF16]),
            "fuel-cpu-backend's standard registration includes Neg[BF16]");
    }

}
