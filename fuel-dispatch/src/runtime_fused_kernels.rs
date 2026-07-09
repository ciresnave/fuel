//! Runtime fused-op kernel registration — the **adopt/gate facade** over the
//! ONE binding registry.
//!
//! The transitional `FusedOpId`-keyed kernel *sidecar* that used to live here
//! is **folded** (the 2026-07-08 decisions-log end-state): a runtime kernel now
//! registers into the global binding table
//! ([`crate::dispatch::extend_global_bindings`]) under
//! [`BindingKey::RuntimeFused`], exactly like a startup kernel registers under
//! `BindingKey::Static` — same table, same entry shape, same dtype-keyed
//! dispatch lookup. This module keeps the runtime-op-shaped API
//! ([`adopt_runtime_fused`], [`fused_kernel_available`],
//! [`lookup_runtime_kernel`]) so callers stay id-oriented, but there is no
//! second registry behind it.
//!
//! Cost note: a runtime row is stored with the binding-native `unknown_cost`
//! sentinel (never a lying zero). Deriving its Layer-1 estimate from the op's
//! recipe (`decompose_region`, keyed by the `RuntimeFused` id in the binding
//! key — no stored closure needed) is the plan-path pricing follow-up; today a
//! runtime arm is sparse-skip unpriced there, which is safe (arm 0 stays the
//! runnability fallback).

use fuel_graph::jit::PatternNode;
use fuel_graph::registry::FusedOpId;
use fuel_ir::DType;
use fuel_ir::probe::BackendId;

use crate::dispatch::{extend_global_bindings, global_bindings};
use crate::fused::{PrecisionGuarantee, default_kernel_registry};
use crate::kernel::{BindingKey, KernelCaps, KernelRef};

/// A runtime-fused binding row, as [`lookup_runtime_kernel`] returns it — a
/// diagnostic/test view (dispatch resolves dtype-precisely through the binding
/// table itself; see `compile_one`'s `is_runtime` arm).
#[derive(Clone, Debug)]
pub struct RuntimeKernelBinding {
    pub kernel: KernelRef,
    pub caps: KernelCaps,
    pub precision: PrecisionGuarantee,
    /// The per-operand dtype tuple the kernel registered for (inputs in
    /// order, then output) — the binding key's dtype half.
    pub dtypes: Vec<DType>,
}

/// Bind a kernel for a runtime fused op (id `>= RUNTIME_FUSED_BASE`) into the
/// global binding table under `BindingKey::RuntimeFused(id)`. `dtypes` is the
/// per-operand tuple (inputs in order, then output) — the same key shape every
/// static registration uses.
pub fn register_runtime_kernel(
    id: FusedOpId,
    dtypes: &[DType],
    backend: BackendId,
    kernel: KernelRef,
) {
    extend_global_bindings(|t| t.register(BindingKey::RuntimeFused(id), dtypes, backend, kernel));
}

/// The first runtime-fused binding for `(id, backend)`, any dtype tuple.
pub fn lookup_runtime_kernel(id: FusedOpId, backend: BackendId) -> Option<RuntimeKernelBinding> {
    let table = global_bindings();
    table.first_runtime_fused(id, backend).map(|(dtypes, e)| RuntimeKernelBinding {
        kernel: e.kernel,
        caps: e.caps,
        precision: e.precision,
        dtypes: dtypes.to_vec(),
    })
}

/// The capability predicate the optimizer's gate consults: is there an
/// admissible kernel for `(id, backend)` — static **or** runtime-registered?
/// The dispatch layer passes `|id| fused_kernel_available(id, backend)` to
/// `capability_gated_rules`, so a runtime op fuses only once its kernel is
/// bound. Id-level (coarse) by design; the dtype-precise check is the
/// dispatch-time binding lookup itself.
pub fn fused_kernel_available(id: FusedOpId, backend: BackendId) -> bool {
    default_kernel_registry().lookup(id, backend).is_some()
        || global_bindings().has_runtime_fused(id, backend)
}

/// **TEST-ONLY.** Reset the runtime-fused world: drop every
/// `BindingKey::RuntimeFused` row from the global binding table AND clear the
/// `fuel-graph` metadata sidecar — together, because clearing metadata restarts
/// the id allocator and a reused id must never resolve a stale kernel. Bindings
/// are cleared FIRST so no window exists where a fresh id sees an old row.
/// Callers in one test binary share the process: serialize with every other
/// adopting test (a bare reset races — dd-shapes coordination, 2026-07-08).
/// `#[doc(hidden)] pub`, not `#[cfg(test)]`: integration tests compile this
/// crate without `cfg(test)`.
#[doc(hidden)]
pub fn clear_runtime_fused_for_tests() {
    extend_global_bindings(|t| t.remove_runtime_fused_for_tests());
    fuel_graph::runtime_fused::clear_runtime_fused_for_tests();
}

/// Adopt a synthesized/imported runtime fused op: register its recipe (the
/// `region`) in the `fuel-graph` metadata sidecar **and** bind its kernel in
/// the global binding table, then return the freshly-allocated runtime
/// [`FusedOpId`]. After this the capability gate sees the op as fusable on
/// `backend`, and the executor's `is_runtime` arm resolves it dtype-precisely
/// from the same table every static kernel lives in.
///
/// Takes the *resolved* parts (the region + the bound `KernelRef`), not a
/// `JitResponse` — the `JitResponse`/`SynthArtifact` destructuring (+ the
/// artifact-load step) happens at the seam-call site (`jit_adopt`), so the
/// core dispatch layer stays free of the `fuel-kernel-seam` envelope crate.
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
    register_runtime_kernel(id, &dtypes, backend, kernel);
    Some(id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fuel_graph::jit::{OpAttrs, OpTag};
    use fuel_ir::Layout;
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
        let row = lookup_runtime_kernel(id, BackendId::Cpu).expect("row bound on Cpu");
        assert_eq!(
            row.dtypes,
            vec![DType::F32, DType::F32, DType::F32],
            "the binding key carries the per-operand dtype tuple",
        );
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
            "the recipe (region) is registered in the fuel-graph metadata sidecar",
        );
    }

    /// The fold invariant: a runtime kernel lives in the SAME global binding
    /// table as static kernels — resolvable through the ordinary dtype-keyed
    /// `lookup_with_caps` under its `RuntimeFused` key, with the exact adopted
    /// kernel pointer. (No second registry behind the facade.)
    #[test]
    fn adopted_kernel_lives_in_the_one_binding_table() {
        let id = adopt_runtime_fused(
            "test::adopt::relu_add::one_registry",
            relu_add(),
            noop_kernel as KernelRef,
            vec![DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
        )
        .expect("registrable region");
        let (kernel, _caps) = global_bindings()
            .lookup_with_caps(
                BindingKey::RuntimeFused(id),
                &[DType::F32, DType::F32, DType::F32],
                BackendId::Cpu,
            )
            .expect("resolves through the ordinary binding lookup");
        assert!(
            std::ptr::fn_addr_eq(kernel, noop_kernel as KernelRef),
            "the exact adopted kernel, from the one registry",
        );
        // …and the dtype-precise lookup is an honest miss on a wrong tuple.
        assert!(
            global_bindings()
                .lookup_with_caps(
                    BindingKey::RuntimeFused(id),
                    &[DType::F64, DType::F64, DType::F64],
                    BackendId::Cpu,
                )
                .is_err(),
            "wrong dtype tuple ⇒ NoBackendForOp, not a wrong-kernel bind",
        );
    }
}
