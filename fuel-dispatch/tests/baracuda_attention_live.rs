//! Live-CUDA tests for baracuda-kernels-sys-backed attention
//! primitives.

#![cfg(feature = "cuda")]

use std::sync::{Arc, RwLock};

use fuel_core_types::{dispatch::OpKind, probe::BackendId, DType};
use fuel_cuda_backend::{CudaDevice, CudaStorageBytes};
use fuel_dispatch::{baracuda_dispatch::register_baracuda_cuda_kernels, dispatch::register_cuda_kernels, kernel::{KernelBindingTable, OpParams}};
use fuel_memory::{BackendStorage, Storage};

fn dev_or_skip() -> Option<CudaDevice> {
    CudaDevice::new(0).ok()
}

fn upload_f32(dev: &CudaDevice, host: &[f32]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda_bytes = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda_bytes), DType::F32)
}

fn download_f32(s: &Storage) -> Vec<f32> {
    let bytes = match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    };
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn pick_alt(
    table: &KernelBindingTable,
    op: OpKind,
    expected: fuel_dispatch::KernelRef,
) -> fuel_dispatch::KernelRef {
    let alternatives =
        table.lookup_alternatives(op, &[DType::F32, DType::F32], BackendId::Cuda);
    let expected_ptr = expected as usize;
    for alt in alternatives {
        if (alt.kernel as usize) == expected_ptr {
            return alt.kernel;
        }
    }
    panic!("expected baracuda KernelRef not found")
}

/// RoPE applied at sequence position 0 with default base 10000 is
/// the identity transform (angle θ = 0 → cos=1, sin=0, output = x).
/// Use this for a no-arithmetic-error smoke test on the live GPU.
#[test]
#[ignore]
fn baracuda_rope_f32_at_seq_position_zero_is_identity() {
    let Some(_dev) = dev_or_skip() else { return };
    let dev = CudaDevice::new(0).expect("cuda");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // [outer_count=1, seq=1, head_dim=4] — RoPE rotates pairs
    // (x_0, x_1) and (x_2, x_3). At seq position 0, angle is 0 →
    // identity.
    let input = [1.0_f32, 2.0, 3.0, 4.0];
    let src = upload_f32(&dev, &input);
    let out_bytes = CudaStorageBytes::alloc(&dev, input.len() * 4).expect("alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(
        &table,
        OpKind::Rope,
        fuel_dispatch::baracuda_dispatch::attention::rope_f32,
    );
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::Rope {
            outer_count: 1,
            seq: 1,
            head_dim: 4,
        },
    )
    .expect("kernel call");

    let got = download_f32(&out_arc.read().unwrap());
    // Identity at pos 0: output == input within fp32 tolerance.
    for (g, e) in got.iter().zip(input.iter()) {
        assert!(
            (g - e).abs() < 1e-5,
            "got {got:?} expected {input:?}",
        );
    }
}

/// Two sequence positions, head_dim=4. Verify RoPE produces a
/// non-trivial rotation at pos 1 — angle θ_0 = 1 (i=0 →
/// base^0 = 1). The pair (x_0, x_1) at pos 1 rotates by 1 rad.
///
/// Reference: `(cos(1)·x_0 - sin(1)·x_1, sin(1)·x_0 + cos(1)·x_1)`.
#[test]
#[ignore]
fn baracuda_rope_f32_at_seq_position_one_rotates_pair() {
    let Some(_dev) = dev_or_skip() else { return };
    let dev = CudaDevice::new(0).expect("cuda");
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    register_baracuda_cuda_kernels(&mut table);

    // [outer_count=1, seq=2, head_dim=4] flat = 8 elements:
    // [pos0_x0, pos0_x1, pos0_x2, pos0_x3, pos1_x0, pos1_x1, pos1_x2, pos1_x3]
    let input: [f32; 8] = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    let src = upload_f32(&dev, &input);
    let out_bytes = CudaStorageBytes::alloc(&dev, input.len() * 4).expect("alloc");
    let out = Storage::new(BackendStorage::Cuda(out_bytes), DType::F32);

    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(
        &table,
        OpKind::Rope,
        fuel_dispatch::baracuda_dispatch::attention::rope_f32,
    );
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::Rope {
            outer_count: 1,
            seq: 2,
            head_dim: 4,
        },
    )
    .expect("kernel call");

    let got = download_f32(&out_arc.read().unwrap());

    // Pos 0 (angle = 0): identity.
    for i in 0..4 {
        assert!(
            (got[i] - input[i]).abs() < 1e-5,
            "pos0 idx {i}: got {} expected {}",
            got[i],
            input[i],
        );
    }
    // Pos 1, pair (x_0=1, x_1=0): θ_0 = 1 · base^0 = 1.
    //   out_x0 = cos(1) · 1 - sin(1) · 0 = cos(1) ≈ 0.5403
    //   out_x1 = sin(1) · 1 + cos(1) · 0 = sin(1) ≈ 0.8415
    let cos1 = 1.0_f32.cos();
    let sin1 = 1.0_f32.sin();
    assert!(
        (got[4] - cos1).abs() < 1e-4,
        "pos1 x0: got {} expected cos(1) = {cos1}",
        got[4],
    );
    assert!(
        (got[5] - sin1).abs() < 1e-4,
        "pos1 x1: got {} expected sin(1) = {sin1}",
        got[5],
    );
}
