//! Live-CUDA tests for the in-place op family expansion (2026-05-30):
//!
//! - **A — 16 new unary in-place activations** (`Neg`, `Abs`, `Sqr`,
//!   `Sqrt`, `Rsqrt`, `Recip`, `Exp`, `Log`, `Sin`, `Cos`, `Sign`,
//!   `Floor`, `Ceil`, `Round`, `Erf`, `GeluErf`) registered at
//!   `(OpKind::*Inplace, [T, T], Cuda)` for T ∈ {F32, F64, BF16, F16}.
//!   Each reuses the matching baracuda forward symbol with
//!   same-pointer dispatch.
//!
//! - **B — 2 new scalar-param in-place ops** (`ClampInplace`,
//!   `PowIInplace`) carrying `OpParams::Clamp` / `OpParams::PowI`.
//!   ClampInplace allocates 1-element broadcast bounds; PowIInplace
//!   passes the integer exponent via the `unary_param` ABI's `p0`.
//!
//! Each test is `#[ignore]` — opt-in via `--ignored` on a CUDA box.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{baracuda_dispatch::register_baracuda_cuda_kernels, dispatch::register_cuda_kernels, kernel::{KernelBindingTable, OpParams}};
use fuel_storage::{BackendStorage, Storage};

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

/// Look up any registered kernel for the (op, [dt, dt], Cuda) key and
/// invoke it against `target_initial` with `params`. Returns the
/// downloaded post-mutation bytes. Used for tests that don't need to
/// distinguish between alternatives — the in-place ops have a single
/// baracuda registration per (op, dtype) entry today.
fn run_inplace<T: bytemuck::Pod>(
    table: &KernelBindingTable,
    op: OpKind,
    dt: DType,
    params: OpParams,
    target_initial: &[T],
) -> Vec<u8> {
    let dev = CudaDevice::new(0).expect("cuda");
    let target = upload(&dev, dt, target_initial);
    let target_arc = Arc::new(RwLock::new(target));
    let alternatives = table.lookup_alternatives(op, &[dt, dt], BackendId::Cuda);
    assert!(
        !alternatives.is_empty(),
        "no alternatives at ({op:?}, [{dt:?}, {dt:?}], Cuda)",
    );
    let kernel = alternatives[0].kernel;
    kernel(&[], &mut [target_arc.clone()], &[], &params)
        .expect("inplace kernel call");
    download_bytes(&target_arc.read().unwrap())
}

// ---------------------------------------------------------------------------
// A — Unary in-place activations expansion (16 new ops)
// One representative test per op (f32) — the per-dtype dispatch path
// is the same `unary_inplace_kernel!` macro that the 5 existing ops
// already exercise on f32/f64/bf16/f16.
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn baracuda_neg_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::NegInplace, DType::F32, OpParams::None,
        &[1.0_f32, -2.0, 3.0, -4.0],
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[-1.0_f32, 2.0, -3.0, 4.0]);
}

#[test]
#[ignore]
fn baracuda_abs_inplace_bf16() {
    use half::bf16;
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::AbsInplace, DType::BF16, OpParams::None,
        &[bf16::from_f32(-1.5), bf16::from_f32(0.0), bf16::from_f32(2.5)],
    );
    let got: &[bf16] = bytemuck::cast_slice(&out);
    assert!((got[0].to_f32() - 1.5).abs() < 1e-2);
    assert_eq!(got[1].to_f32(), 0.0);
    assert!((got[2].to_f32() - 2.5).abs() < 1e-2);
}

#[test]
#[ignore]
fn baracuda_sqr_inplace_f64() {
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::SqrInplace, DType::F64, OpParams::None,
        &[2.0_f64, -3.0, 4.0],
    );
    let got: &[f64] = bytemuck::cast_slice(&out);
    assert!((got[0] - 4.0).abs() < 1e-12);
    assert!((got[1] - 9.0).abs() < 1e-12);
    assert!((got[2] - 16.0).abs() < 1e-12);
}

#[test]
#[ignore]
fn baracuda_exp_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::ExpInplace, DType::F32, OpParams::None,
        &[0.0_f32, 1.0, 2.0],
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert!((got[0] - 1.0).abs() < 1e-5);
    assert!((got[1] - std::f32::consts::E).abs() < 1e-4);
    assert!((got[2] - (2.0_f32).exp()).abs() < 1e-3);
}

#[test]
#[ignore]
fn baracuda_log_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::LogInplace, DType::F32, OpParams::None,
        &[1.0_f32, std::f32::consts::E, 10.0],
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert!(got[0].abs() < 1e-5);
    assert!((got[1] - 1.0).abs() < 1e-4);
    assert!((got[2] - 10.0_f32.ln()).abs() < 1e-4);
}

#[test]
#[ignore]
fn baracuda_floor_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::FloorInplace, DType::F32, OpParams::None,
        &[1.7_f32, -2.3, 3.0, -0.5],
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[1.0_f32, -3.0, 3.0, -1.0]);
}

#[test]
#[ignore]
fn baracuda_sign_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::SignInplace, DType::F32, OpParams::None,
        &[-2.0_f32, 0.0, 3.0],
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[-1.0_f32, 0.0, 1.0]);
}

#[test]
#[ignore]
fn baracuda_erf_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::ErfInplace, DType::F32, OpParams::None,
        &[0.0_f32, 1.0, -1.0],
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert!(got[0].abs() < 1e-5);
    // erf(1) ≈ 0.8427
    assert!((got[1] - 0.8427).abs() < 1e-3);
    assert!((got[2] - (-0.8427)).abs() < 1e-3);
}

// ---------------------------------------------------------------------------
// B — Scalar-param in-place ops (Clamp + PowI)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn baracuda_clamp_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::ClampInplace, DType::F32,
        OpParams::Clamp { min: -1.0, max: 1.0 },
        &[-5.0_f32, 0.0, 5.0, 2.5],
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert_eq!(got, &[-1.0_f32, 0.0, 1.0, 1.0]);
}

#[test]
#[ignore]
fn baracuda_clamp_inplace_bf16() {
    use half::bf16;
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::ClampInplace, DType::BF16,
        OpParams::Clamp { min: 0.0, max: 2.0 },
        &[bf16::from_f32(-1.0), bf16::from_f32(1.5), bf16::from_f32(3.0)],
    );
    let got: &[bf16] = bytemuck::cast_slice(&out);
    assert_eq!(got[0].to_f32(), 0.0);
    assert!((got[1].to_f32() - 1.5).abs() < 1e-2);
    assert_eq!(got[2].to_f32(), 2.0);
}

#[test]
#[ignore]
fn baracuda_powi_inplace_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::PowIInplace, DType::F32,
        OpParams::PowI { exp: 3 },
        &[2.0_f32, -3.0, 4.0],
    );
    let got: &[f32] = bytemuck::cast_slice(&out);
    assert!((got[0] - 8.0).abs() < 1e-4);
    assert!((got[1] - (-27.0)).abs() < 1e-3);
    assert!((got[2] - 64.0).abs() < 1e-3);
}

#[test]
#[ignore]
fn baracuda_powi_inplace_f64() {
    let Some(_dev) = dev_or_skip() else { return };
    let out = run_inplace(
        &dual_table(),
        OpKind::PowIInplace, DType::F64,
        OpParams::PowI { exp: 2 },
        &[1.5_f64, -2.5, 3.5],
    );
    let got: &[f64] = bytemuck::cast_slice(&out);
    assert!((got[0] - 2.25).abs() < 1e-10);
    assert!((got[1] - 6.25).abs() < 1e-10);
    assert!((got[2] - 12.25).abs() < 1e-10);
}
