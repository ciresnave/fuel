//! Live-CUDA tests for the in-place unary activations registered at
//! `(OpKind::{Relu,Silu,Gelu,Tanh,Sigmoid}Inplace, [f32, f32], Cuda)`.
//! Phase 3e of the in-place ops infrastructure
//! (`docs/session-prompts/in-place-ops-infrastructure.md`).
//!
//! Each kernel reuses the same baracuda symbol as its non-inplace
//! cousin but the wrapper passes the target's pointer for both `x`
//! and `y`. The executor's `WorkItemKind::InplaceKernel` arm passes
//! the target as `outputs[0]` with `inputs=[]`; the wrapper acquires
//! the write lock and dispatches.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_storage::{
    baracuda_dispatch::register_baracuda_cuda_kernels,
    dispatch::register_cuda_kernels,
    kernel::{KernelBindingTable, OpParams},
    BackendStorage, Storage,
};

fn dev_or_skip() -> Option<CudaDevice> {
    CudaDevice::new(0).ok()
}

fn dual_table() -> KernelBindingTable {
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);
    table
}

fn upload<T: bytemuck::Pod>(dev: &CudaDevice, dt: DType, host: &[T]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda), dt)
}

fn download_bytes(s: &Storage) -> Vec<u8> {
    match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    }
}

/// Dispatch an in-place unary op on CUDA: looks up the kernel under
/// `(op, [F32, F32], Cuda)` from the dual binding table, runs it
/// against `target_initial` seeded as the target storage, returns the
/// downloaded post-mutation bytes.
fn run_unary_inplace(
    table: &KernelBindingTable,
    op: OpKind,
    expected: fuel_storage::KernelRef,
    target_initial: &[f32],
) -> Vec<u8> {
    let dev = CudaDevice::new(0).expect("cuda");
    let target = upload(&dev, DType::F32, target_initial);
    let target_arc = Arc::new(RwLock::new(target));
    let alternatives = table.lookup_alternatives(op, &[DType::F32, DType::F32], BackendId::Cuda);
    assert!(
        !alternatives.is_empty(),
        "no alternatives at ({op:?}, [F32, F32], Cuda)",
    );
    let expected_ptr = expected as usize;
    let kernel = alternatives
        .iter()
        .map(|a| a.kernel)
        .find(|k| (*k as usize) == expected_ptr)
        .expect("expected baracuda KernelRef not registered");
    kernel(&[], &mut [target_arc.clone()], &[], &OpParams::None)
        .expect("inplace unary kernel call");
    download_bytes(&target_arc.read().unwrap())
}

#[test]
#[ignore]
fn baracuda_relu_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input = [-1.0_f32, 0.0, 1.0, 2.0];
    let out = run_unary_inplace(
        &table,
        OpKind::ReluInplace,
        fuel_storage::baracuda_dispatch::unary::relu_inplace_f32,
        &input,
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[0.0_f32, 0.0, 1.0, 2.0]);
}

#[test]
#[ignore]
fn baracuda_silu_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input = [0.0_f32, 1.0, -1.0];
    let out = run_unary_inplace(
        &table,
        OpKind::SiluInplace,
        fuel_storage::baracuda_dispatch::unary::silu_inplace_f32,
        &input,
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    // Silu(x) = x · sigmoid(x): Silu(0)=0, Silu(1)≈0.731, Silu(-1)≈-0.269
    assert!((got[0] - 0.0).abs() < 1e-6);
    assert!((got[1] - 0.7310585).abs() < 1e-4);
    assert!((got[2] - (-0.26894143)).abs() < 1e-4);
}

#[test]
#[ignore]
fn baracuda_gelu_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input = [0.0_f32, 1.0, -1.0];
    let out = run_unary_inplace(
        &table,
        OpKind::GeluInplace,
        fuel_storage::baracuda_dispatch::unary::gelu_inplace_f32,
        &input,
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    // Gelu tanh-approx: Gelu(0)=0, Gelu(1)≈0.8412, Gelu(-1)≈-0.1588
    assert!((got[0] - 0.0).abs() < 1e-6);
    assert!((got[1] - 0.8411920).abs() < 1e-3);
    assert!((got[2] - (-0.1588080)).abs() < 1e-3);
}

#[test]
#[ignore]
fn baracuda_tanh_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input = [0.0_f32, 1.0, -1.0, 100.0];
    let out = run_unary_inplace(
        &table,
        OpKind::TanhInplace,
        fuel_storage::baracuda_dispatch::unary::tanh_inplace_f32,
        &input,
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert!((got[0] - 0.0).abs() < 1e-6);
    assert!((got[1] - 0.7615942).abs() < 1e-4);
    assert!((got[2] - (-0.7615942)).abs() < 1e-4);
    assert!((got[3] - 1.0).abs() < 1e-6);
}

#[test]
#[ignore]
fn baracuda_sigmoid_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input = [0.0_f32, 1.0, -1.0];
    let out = run_unary_inplace(
        &table,
        OpKind::SigmoidInplace,
        fuel_storage::baracuda_dispatch::unary::sigmoid_inplace_f32,
        &input,
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    // Sigmoid(0)=0.5, Sigmoid(1)≈0.731, Sigmoid(-1)≈0.269
    assert!((got[0] - 0.5).abs() < 1e-6);
    assert!((got[1] - 0.7310586).abs() < 1e-4);
    assert!((got[2] - 0.26894143).abs() < 1e-4);
}
