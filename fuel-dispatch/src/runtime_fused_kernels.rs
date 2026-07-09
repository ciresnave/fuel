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

use fuel_ir::{DType, dispatch::OpKind, probe::BackendId};
use fuel_graph::jit::PatternNode;
use fuel_graph::registry::{FusedOpId, FusedOps};

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
///
/// **SHARED-CONSUMER COORDINATION (dd-shapes / flash-arm registry seam,
/// 2026-07-08).** This predicate has two independent callers with different
/// invariants, and any change here must hold both:
/// - the **Tier-2 runtime-fusion / JIT program** calls this for *runtime* ids
///   (`id.is_runtime()`, allocated `>= FusedOpId::RUNTIME_FUSED_BASE` by
///   [`adopt_runtime_fused`]) — those resolve *only* through the
///   [`runtime_kernels`] sidecar (`lookup_runtime_kernel`), never the static
///   registry or the binding table below;
/// - the **decode flash-arm gate**
///   (`crate::decode_flash::FlashArmCapability::production`) calls this for
///   the *static* id `FusedOps::FLASH_ATTN` on `BackendId::Cuda` — and that
///   call was always `false`, making the CUDA flash-decode arm unreachable on
///   every host. Root cause: the CUDA `flash_decoding` kernel
///   (`baracuda_dispatch::register_cuda_flash_decoding_from_contract`)
///   registers ONLY into the primitive `KernelBindingTable` under
///   `(OpKind::FlashAttn, [f16|bf16;4], Cuda)` — its FKC import even
///   `debug_assert!`s `provider.fused.is_empty()` — so it never reaches
///   [`default_kernel_registry`], which today holds CPU-only `FLASH_ATTN`
///   `BackendImpl`s. `default_kernel_registry` is a frozen `OnceLock`
///   (architecture v1.0: no runtime extensibility) built from
///   `register_default_kernels`, which has no `cfg(feature = "cuda")` arm —
///   so *mirroring* the CUDA registration into it isn't a clean option
///   without threading the FKC-resolved kernel fn pointer back out to a
///   static-registration call site. **Fix: bridge, not mirror.** For a
///   *static* id only (`!id.is_runtime()` — the runtime sidecar path above is
///   untouched), additionally consult [`crate::dispatch::global_bindings`]
///   for ANY entry keyed by the id's corresponding [`OpKind`] on `backend`
///   (dtype-blind — this predicate answers "is *a* kernel bound", not "is
///   *this exact dtype tuple* bound"; dtype admissibility is a separate gate,
///   e.g. [`crate::decode_flash::flash_decode_admissible`]). Scoped to
///   `FLASH_ATTN` today (the one defect this fixes); other static fused ids
///   with the same CPU-registry/CUDA-binding-table split (several CUDA FKC
///   families follow the identical `provider.fused.is_empty()` pattern) are
///   an out-of-scope, separately-verifiable follow-up — widen
///   [`static_fused_id_to_binding_table_op_kind`] deliberately, one id at a
///   time, not by guessing a blanket mapping.
pub fn fused_kernel_available(id: FusedOpId, backend: BackendId) -> bool {
    default_kernel_registry().lookup(id, backend).is_some()
        || lookup_runtime_kernel(id, backend).is_some()
        || static_binding_table_bridge(id, backend)
}

/// The static-id half of the [`fused_kernel_available`] bridge (see its doc
/// comment for the full defect writeup). Returns `false` immediately for a
/// runtime id — this is purely additive for the static-id path and must
/// never perturb the runtime-fusion program's sidecar-only resolution.
fn static_binding_table_bridge(id: FusedOpId, backend: BackendId) -> bool {
    if id.is_runtime() {
        return false;
    }
    let Some(op_kind) = static_fused_id_to_binding_table_op_kind(id) else {
        return false;
    };
    crate::dispatch::global_bindings()
        .iter_keys()
        .any(|(op, _dtypes, b)| op == op_kind && b == backend)
}

/// The (deliberately small) set of *static* [`FusedOpId`]s whose kernel is
/// known to register into the primitive `KernelBindingTable` under a
/// corresponding [`OpKind`] rather than (only) [`default_kernel_registry`].
/// See [`fused_kernel_available`]'s doc comment — widen this one id at a
/// time as each is verified, not speculatively.
fn static_fused_id_to_binding_table_op_kind(id: FusedOpId) -> Option<OpKind> {
    if id == FusedOps::FLASH_ATTN { Some(OpKind::FlashAttn) } else { None }
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

    // ---- flash-arm registry-seam bridge (dd-shapes, 2026-07-08) -----------

    /// GREEN target (cuda build only): the CUDA `flash_decoding` kernel
    /// registers only into `KernelBindingTable` (`register_cuda_flash_
    /// decoding_from_contract`, never `default_kernel_registry`), so before
    /// the bridge this was unconditionally `false` on every host — the
    /// flash-arm registry-seam defect this module's doc comment documents.
    /// Forces `global_bindings()` to initialize (which runs the CUDA
    /// registration under `#[cfg(feature = "cuda")]`) then asserts the
    /// static-id bridge sees it.
    #[test]
    #[cfg(feature = "cuda")]
    fn fused_kernel_available_bridges_static_flash_attn_from_binding_table_on_cuda() {
        let _guard = crate::dispatch::global_bindings();
        drop(_guard);
        assert!(
            fused_kernel_available(FusedOps::FLASH_ATTN, BackendId::Cuda),
            "the CUDA flash_decoding binding registers only into KernelBindingTable \
             ((OpKind::FlashAttn, [f16|bf16;4], Cuda)); fused_kernel_available must \
             bridge to it for the static FLASH_ATTN id",
        );
    }

    /// GUARD (non-cuda build only): with no CUDA registration ever run, the
    /// static-id bridge has nothing to find — `fused_kernel_available` stays
    /// `false` on `Cuda`, matching every CPU/Vulkan-only build today.
    #[test]
    #[cfg(not(feature = "cuda"))]
    fn fused_kernel_available_flash_attn_false_on_cuda_without_cuda_registration() {
        assert!(
            !fused_kernel_available(FusedOps::FLASH_ATTN, BackendId::Cuda),
            "no cuda registration ran in this build ⇒ the static bridge finds nothing",
        );
    }

    /// COORDINATION GUARD: the static-id `KernelBindingTable` bridge added
    /// for `FLASH_ATTN` must be a strict no-op for *runtime* ids — the
    /// Tier-2 runtime-fusion program (`jit_adopt.rs`, `runtime_fused_arm.rs`)
    /// relies on a runtime id resolving ONLY through the `runtime_kernels`
    /// sidecar, never the binding table. Mirrors
    /// `adopt_registers_op_plus_kernel_and_the_gate_predicate_sees_it`'s
    /// setup, with an extra assertion pinning that a runtime id adopted on
    /// one backend is not "rescued" by the static bridge on another.
    #[test]
    fn fused_kernel_available_runtime_id_still_routes_through_the_sidecar_untouched() {
        let id = adopt_runtime_fused(
            "test::adopt::flash_bridge_guard::relu_add",
            relu_add(),
            noop_kernel as KernelRef,
            vec![DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
        )
        .expect("relu(add) is a registrable region");

        assert!(id.is_runtime(), "allocated a runtime id");
        assert!(
            fused_kernel_available(id, BackendId::Cpu),
            "a runtime id still resolves via the runtime sidecar (unaffected by the bridge)",
        );
        assert!(
            !fused_kernel_available(id, BackendId::Cuda),
            "not adopted on Cuda, and the static-id bridge must never rescue a runtime id \
             (it returns false immediately for any id.is_runtime())",
        );
    }
}
