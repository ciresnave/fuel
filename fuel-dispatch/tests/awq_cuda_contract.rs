// SPDX-License-Identifier: MIT OR Apache-2.0
//! Host-side check of the CUDA AWQ matmul contract (ROADMAP row 31a).
//!
//! `register_cuda_awq_matmul_from_contract` is `cuda`-gated, so no CI leg builds
//! it. This test imports the SAME contract text through the real importer with a
//! stub link registry and reads the binding back, so a contract that does not
//! import, or that keys the wrong dtypes, fails on every host.
//!
//! Controls: the stub resolves exactly the contract's entry point (a symbol typo
//! in the contract is an `UnknownEntryPoint`, checked below), and the key is
//! read back from the table rather than inferred from the contract text.

use std::sync::{Arc, RwLock};

use fuel_dispatch::fkc::{LinkRegistry, import_bundle_str};
use fuel_dispatch::fused::FusedKernelRegistry;
use fuel_dispatch::kernel::{KernelBindingTable, KernelRef, OpParams};
use fuel_ir::dispatch::OpKind;
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, Layout, Result};
use fuel_memory::Storage;

const CONTRACT: &str = include_str!("../kernel-contracts/cuda/awq_matmul.fkc.md");
const SYMBOL: &str = "fuel_cuda_backend::fkc::awq_matmul_f16";

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
fn awq_contract_registers_one_cuda_binding_with_the_four_input_f16_key() {
    let provider = import_bundle_str(CONTRACT, &Link(SYMBOL)).expect("contract must import");
    assert!(provider.fused.is_empty(), "AwqMatmul is a primitive OpKind");
    let mut table = KernelBindingTable::new();
    let mut fused = FusedKernelRegistry::new();
    provider
        .register_into(&mut table, &mut fused)
        .expect("contract must register");

    let keys: Vec<_> = table
        .iter_keys()
        .filter(|(op, _, _)| *op == OpKind::AwqMatmul)
        .map(|(_, d, b)| (d.to_vec(), b))
        .collect();
    assert_eq!(
        keys,
        vec![(
            vec![DType::F16, DType::U32, DType::U32, DType::F16, DType::F16],
            BackendId::Cuda
        )],
        "exactly one AwqMatmul binding, keyed [activations, weight, zeros, scales, out]"
    );
    // The same lookup the executor does, with the same key.
    table
        .lookup(
            OpKind::AwqMatmul,
            &[DType::F16, DType::U32, DType::U32, DType::F16, DType::F16],
            BackendId::Cuda,
        )
        .expect("executor-shaped lookup resolves");
    // Sabotage sibling: an F32-activation key is NOT bound (it is served only
    // by an explicit lowering to the decompose; GAP-367).
    assert!(
        table
            .lookup(
                OpKind::AwqMatmul,
                &[DType::F32, DType::U32, DType::U32, DType::F16, DType::F32],
                BackendId::Cuda,
            )
            .is_err(),
        "F32 activations have no CUDA binding; they need an explicit lowering"
    );
}

#[test]
fn a_misspelled_entry_point_is_refused_not_silently_dropped() {
    let err = import_bundle_str(CONTRACT, &Link("fuel_cuda_backend::fkc::awq_matmul_f32"))
        .expect_err("an unresolved entry point must be an error");
    assert!(
        format!("{err}").contains("awq_matmul_f16"),
        "error should name the unresolved symbol: {err}"
    );
}
