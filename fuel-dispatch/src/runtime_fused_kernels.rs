//! Tier-2 runtime fused-op **kernel** sidecar
//! (`docs/specs/runtime-fused-op-registration.md` §6).
//!
//! The static [`crate::fused::FusedKernelRegistry`] is a frozen `OnceLock`, so a
//! synthesized/adopted runtime kernel cannot register into it. This parallel
//! `RwLock` registry holds runtime fused-op kernels (keyed by [`FusedOpId`],
//! exactly like the static one), and [`fused_kernel_available`] is the
//! capability predicate the optimizer's gate
//! ([`fuel_graph::opt::RuleRegistry::capability_gated_rules`]) consults —
//! static **or** runtime.
//!
//! [`adopt_runtime_fused`] is the entry point: it registers the runtime op's
//! recipe (the `fuel-graph` metadata sidecar) **and** binds its kernel here, so
//! the gate sees `has_kernel = true` and the op fuses + dispatches like a static
//! one. The kernel-present half of the design that "already works" because
//! `FusedKernelRegistry` is `FusedOpId`-keyed — this is the missing extensible
//! half that lets a *runtime* id participate.

use std::sync::{OnceLock, RwLock};

use fuel_ir::{DType, probe::BackendId};
use fuel_graph::jit::PatternNode;
use fuel_graph::registry::FusedOpId;

use crate::fused::{
    BackendImpl, FusedKernelRegistry, KernelRevisionHash, PrecisionGuarantee,
    default_kernel_registry,
};
use crate::kernel::{KernelCaps, KernelRef};

/// The process-global runtime-kernel registry — append-only, behind a lock
/// because (unlike the static `OnceLock`) it grows across the run as ops are
/// adopted. Reads (dispatch/gate, the hot direction) take the read lock.
fn runtime_kernels() -> &'static RwLock<FusedKernelRegistry> {
    static R: OnceLock<RwLock<FusedKernelRegistry>> = OnceLock::new();
    R.get_or_init(|| RwLock::new(FusedKernelRegistry::new()))
}

/// Bind a kernel for a runtime fused op (id `>= RUNTIME_FUSED_BASE`).
pub fn register_runtime_kernel(id: FusedOpId, backend: BackendId, impl_: BackendImpl) {
    runtime_kernels().write().unwrap().register(id, backend, impl_);
}

/// Look up a runtime fused op's kernel for `backend`.
pub fn lookup_runtime_kernel(id: FusedOpId, backend: BackendId) -> Option<BackendImpl> {
    runtime_kernels().read().unwrap().lookup(id, backend)
}

/// The capability predicate the optimizer's gate consults: is there an
/// admissible kernel for `(id, backend)` — static **or** runtime-adopted? The
/// dispatch layer passes `|id| fused_kernel_available(id, backend)` to
/// `capability_gated_rules`, so a runtime op fuses only once its kernel is bound.
pub fn fused_kernel_available(id: FusedOpId, backend: BackendId) -> bool {
    default_kernel_registry().lookup(id, backend).is_some()
        || lookup_runtime_kernel(id, backend).is_some()
}

/// **TEST-ONLY.** Reset BOTH runtime-fused sidecars — the kernel registry here
/// and the `fuel-graph` metadata sidecar — together, because clearing metadata
/// restarts the id allocator and a reused id must never resolve a stale
/// kernel. Kernels are cleared FIRST so no window exists where a fresh id sees
/// an old binding. Callers in one test binary share the process: serialize
/// with every other adopting test (a bare reset races — dd-shapes
/// coordination, 2026-07-08). `#[doc(hidden)] pub`, not `#[cfg(test)]`:
/// integration tests compile this crate without `cfg(test)`.
#[doc(hidden)]
pub fn clear_runtime_fused_for_tests() {
    *runtime_kernels().write().unwrap() = FusedKernelRegistry::new();
    fuel_graph::runtime_fused::clear_runtime_fused_for_tests();
}

/// Adopt a synthesized/imported runtime fused op: register its recipe (the
/// `region`) in the `fuel-graph` sidecar **and** bind its `kernel` here, then
/// return the freshly-allocated runtime [`FusedOpId`]. After this the capability
/// gate sees the op as fusable on `backend`.
///
/// Takes the *resolved* parts (the region + the bound `KernelRef`), not a
/// `JitResponse` — the `JitResponse`/`SynthesizedKernel` destructuring (+ the
/// `entry_point → KernelRef` link-registry resolution) happens at the seam-call
/// site, so `fuel-dispatch` stays free of the `fuel-kernel-seam` envelope crate.
/// Returns `None` if the region is not registrable (non-decomposable / bad
/// binds — surfaced by `register_runtime_fused`).
pub fn adopt_runtime_fused(
    name: impl Into<String>,
    region: PatternNode,
    kernel: KernelRef,
    dtypes: Vec<DType>,
    backend: BackendId,
) -> Option<FusedOpId> {
    let id = fuel_graph::runtime_fused::register_runtime_fused(name, region).ok()?;
    // `BackendImpl` is `Copy` / `&'static [DType]`; a runtime op lives for the
    // process, so leaking its dtype tuple to `'static` is sound (not a per-call
    // leak — one per adopted op).
    let dtypes: &'static [DType] = Box::leak(dtypes.into_boxed_slice());
    // The fused-op cost sentinel (not a private zero-cost twin): so the
    // adopted op's Layer-1 cost is composed from its recipe
    // (`crate::fused_cost::fused_layer1_cost` → `cost_from_decompose`)
    // rather than pricing at a spurious zero. A `cost_expr`/Judge override
    // still supersedes it (measured › declared › composed › zero). This is
    // the runtime-fused case spec §6 flags as most sentinel-zero-prone.
    let impl_ = BackendImpl {
        kernel,
        dtypes,
        cost: crate::fkc::fused_unknown_cost,
        precision: PrecisionGuarantee::UNAUDITED,
        caps: KernelCaps::empty(),
        revision: KernelRevisionHash::UNTRACKED,
    };
    register_runtime_kernel(id, backend, impl_);
    Some(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_ir::Layout;
    use fuel_graph::jit::{OpAttrs, OpTag};
    use std::sync::{Arc, RwLock as StdRwLock};

    fn noop_kernel(
        _inputs: &[Arc<StdRwLock<fuel_memory::Storage>>],
        _outputs: &mut [Arc<StdRwLock<fuel_memory::Storage>>],
        _layouts: &[Layout],
        _params: &crate::kernel::OpParams,
    ) -> fuel_ir::Result<()> {
        Ok(())
    }

    fn relu_add() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Relu,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Add,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        }
    }

    #[test]
    fn adopt_registers_op_plus_kernel_and_the_gate_predicate_sees_it() {
        let id = adopt_runtime_fused(
            "test::adopt::relu_add",
            relu_add(),
            noop_kernel as KernelRef,
            vec![DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
        )
        .expect("relu(add) is a registrable region");

        assert!(id.is_runtime(), "allocated a runtime id");
        assert!(lookup_runtime_kernel(id, BackendId::Cpu).is_some(), "kernel bound on Cpu");
        assert!(
            fused_kernel_available(id, BackendId::Cpu),
            "the capability predicate sees the adopted op on Cpu",
        );
        assert!(
            !fused_kernel_available(id, BackendId::Cuda),
            "but not on a backend the kernel wasn't adopted for",
        );
        assert!(
            fuel_graph::runtime_fused::runtime_region(id).is_some(),
            "the recipe (region) is registered in the fuel-graph sidecar",
        );
    }

    #[test]
    fn adopted_runtime_op_carries_the_cost_from_decompose_sentinel() {
        // The adopted op's cost is the fused sentinel — so its Layer-1 cost
        // is composed from its recipe (`fused_cost::fused_layer1_cost`),
        // never a spurious zero (spec §6: runtime-fused ops are the most
        // sentinel-zero-prone).
        let id = adopt_runtime_fused(
            "test::adopt::relu_add::cost_sentinel",
            relu_add(),
            noop_kernel as KernelRef,
            vec![DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
        )
        .expect("relu(add) is a registrable region");
        let impl_ = lookup_runtime_kernel(id, BackendId::Cpu).expect("kernel bound");
        assert!(
            crate::fused_cost::is_fused_cost_sentinel(impl_.cost),
            "adopted runtime op carries the fused cost sentinel (→ cost-from-decompose)",
        );
    }
}
