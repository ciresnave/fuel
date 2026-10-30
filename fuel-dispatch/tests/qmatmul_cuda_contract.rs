// SPDX-License-Identifier: MIT OR Apache-2.0
//! Host-side check of the CUDA QMatMul contract (ROADMAP row 105).
//!
//! `register_cuda_qmatmul_from_contract` is `cuda`-gated, so no CI leg builds it. This test imports
//! the SAME contract text through the real importer with a stub link registry and reads the binding
//! back, so a contract that does not import, or keys the wrong dtypes, fails on every host.
//!
//! Controls: the stub resolves exactly the contract's entry point (a typo is an unresolved-symbol
//! error, checked below), the key is read back from the table, and the CPU contract's key is not
//! this one (the binding is CUDA-only).

use std::sync::{Arc, RwLock};

use fuel_dispatch::fkc::{LinkRegistry, import_bundle_str};
use fuel_dispatch::fused::FusedKernelRegistry;
use fuel_dispatch::kernel::{KernelBindingTable, KernelRef, OpParams};
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, Layout, Result};
use fuel_memory::Storage;

const CONTRACT: &str = include_str!("../kernel-contracts/cuda/qmatmul.fkc.md");
const SYMBOL: &str = "fuel_cuda_backend::fkc::qmatmul_f32";
const KEY: [DType; 3] = [DType::F32, DType::U32, DType::F32];

fn stub_kernel(
    _: &[Arc<RwLock<Storage>>],
    _: &mut [Arc<RwLock<Storage>>],
    _: &[Layout],
    _: &OpParams,
) -> Result<()> {
    Ok(())
}

struct Link(&'static str);
impl LinkRegistry for Link {
    fn resolve_primitive(&self, symbol: &str) -> Option<KernelRef> {
        (symbol == self.0).then_some(stub_kernel as KernelRef)
    }
    fn resolve_fused(&self, _symbol: &str) -> Option<KernelRef> {
        None
    }
}

#[test]
fn qmatmul_contract_registers_one_cuda_binding_with_the_f32_u32_f32_key() {
    let provider = import_bundle_str(CONTRACT, &Link(SYMBOL)).expect("contract must import");
    assert!(provider.fused.is_empty(), "QMatMul is a primitive OpKind");
    let mut table = KernelBindingTable::new();
    let mut fused = FusedKernelRegistry::new();
    provider
        .register_into(&mut table, &mut fused)
        .expect("contract must register");

    let keys: Vec<_> = table
        .iter_keys()
        .filter(|(op, _, _)| *op == OpKind::QMatMul)
        .map(|(_, d, b)| (d.to_vec(), b))
        .collect();
    assert_eq!(
        keys,
        vec![(KEY.to_vec(), BackendId::Cuda)],
        "exactly one QMatMul binding, keyed [activations, weight, out], on CUDA"
    );
    // The same lookup the executor does, with the same key.
    table
        .lookup(OpKind::QMatMul, &KEY, BackendId::Cuda)
        .expect("executor-shaped lookup resolves");
    // Sabotage sibling: the contract binds CUDA only, not the CPU key.
    assert!(
        table.lookup(OpKind::QMatMul, &KEY, BackendId::Cpu).is_err(),
        "the CUDA contract must not bind QMatMul on the CPU backend"
    );
}

#[test]
fn a_misspelled_entry_point_is_refused_not_silently_dropped() {
    let err = import_bundle_str(CONTRACT, &Link("fuel_cuda_backend::fkc::qmatmul_f16"))
        .expect_err("an unresolved entry point must be an error");
    assert!(
        format!("{err}").contains("qmatmul_f32"),
        "error should name the unresolved symbol: {err}"
    );
}

/// The accuracy claim must survive the import: the planner's `AccuracyPreferenceFilter` drops a
/// kernel with no `max_relative`/`max_absolute` claim whenever a bit-stable CPU sibling exists,
/// which is how the production realize missed this binding (ROADMAP row 114).
#[test]
fn the_declared_accuracy_claim_reaches_the_binding_entry() {
    let provider = import_bundle_str(CONTRACT, &Link(SYMBOL)).expect("contract must import");
    let mut table = KernelBindingTable::new();
    let mut fused = FusedKernelRegistry::new();
    provider
        .register_into(&mut table, &mut fused)
        .expect("contract must register");
    let entries = table.lookup_alternatives(OpKind::QMatMul, &KEY, BackendId::Cuda);
    assert_eq!(entries.len(), 1, "one CUDA QMatMul entry");
    let p = entries[0].precision;
    assert_eq!(
        p.max_relative,
        Some(4.0e-4),
        "the ledger-backed bound must reach the entry: {p:?}"
    );
    assert_eq!(p.max_absolute, Some(4.0e-4), "{p:?}");
    assert!(
        !p.bit_stable_on_same_hardware,
        "the kernels are not bit-stable"
    );
    assert!(
        fuel_dispatch::ranker::filters::meets_accuracy_bound(
            &p,
            &fuel_dispatch::ranker::filters::DEFAULT_ACCURACY_BOUND
        ),
        "the entry must meet the planner's default accuracy bound, or it loses to the CPU kernel"
    );
}
