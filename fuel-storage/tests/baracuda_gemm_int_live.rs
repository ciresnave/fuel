//! Live-CUDA tests for baracuda-kernels-sys int8 GEMM. The s8 / u8
//! RRR Identity SKUs land at the `(OpKind::MatMul, [I8, I8, I8],
//! Cuda)` and `[U8, U8, U8]` keys — first non-FP MatMul coverage on
//! Fuel's binding table.
//!
//! Test shape: small rank-2 matmuls with hand-checkable results.
//! `mma.sync.m16n8k32` ties the kernel to M%16==0 / N%8==0 / K%32==0
//! alignment via baracuda's `_can_implement` host-side check —
//! `should_fail`-style alignment tests live in baracuda's own crate.
//! Here we use shapes that match the kernel's tile multiples.

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

fn pick_alt(
    table: &KernelBindingTable,
    op: OpKind,
    dtypes: &[DType],
    expected: fuel_storage::KernelRef,
) -> fuel_storage::KernelRef {
    let alternatives = table.lookup_alternatives(op, dtypes, BackendId::Cuda);
    assert!(
        !alternatives.is_empty(),
        "no alternatives at ({op:?}, {dtypes:?}, Cuda)",
    );
    let expected_ptr = expected as usize;
    for alt in alternatives {
        if (alt.kernel as usize) == expected_ptr {
            return alt.kernel;
        }
    }
    panic!("expected baracuda KernelRef not found")
}

fn upload_i8(dev: &CudaDevice, host: &[i8]) -> Storage {
    let bytes: &[u8] = bytemuck::cast_slice(host);
    let cuda = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda), DType::I8)
}

fn upload_u8(dev: &CudaDevice, host: &[u8]) -> Storage {
    let cuda = CudaStorageBytes::from_cpu_bytes(dev, host).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda), DType::U8)
}

fn alloc_out(dev: &CudaDevice, dt: DType, n_elems: usize) -> Storage {
    let buf = CudaStorageBytes::alloc(dev, n_elems).expect("alloc");
    Storage::new(BackendStorage::Cuda(buf), dt)
}

fn download(s: &Storage) -> Vec<u8> {
    match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    }
}

/// Reference s8 matmul: row-major A[M, K] @ B[K, N] → C[M, N], i32
/// accumulator, saturating cast to i8 on store.
fn ref_gemm_s8(a: &[i8], b: &[i8], m: usize, n: usize, k: usize) -> Vec<i8> {
    let mut out = vec![0i8; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc: i32 = 0;
            for kk in 0..k {
                acc += (a[row * k + kk] as i32) * (b[kk * n + col] as i32);
            }
            out[row * n + col] = acc.clamp(i8::MIN as i32, i8::MAX as i32) as i8;
        }
    }
    out
}

/// Reference u8 matmul: same shape, u32 accumulator, saturating cast
/// to u8 on store.
fn ref_gemm_u8(a: &[u8], b: &[u8], m: usize, n: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0u8; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc: u32 = 0;
            for kk in 0..k {
                acc += (a[row * k + kk] as u32) * (b[kk * n + col] as u32);
            }
            out[row * n + col] = acc.min(u8::MAX as u32) as u8;
        }
    }
    out
}

#[test]
#[ignore]
fn baracuda_gemm_s8_rrr_identity_matches_reference() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    // Tile-aligned shape: M=16, N=8, K=32 (the kernel's native
    // `mma.sync.m16n8k32` tile).
    let m = 16;
    let n = 8;
    let k = 32;

    // Small magnitudes so accumulator stays well within i32 range and
    // saturating-cast on store doesn't bias the comparison.
    let a: Vec<i8> = (0..m * k).map(|i| ((i as i32 % 7) - 3) as i8).collect();
    let b: Vec<i8> = (0..k * n).map(|i| ((i as i32 % 5) - 2) as i8).collect();

    let dev = CudaDevice::new(0).expect("cuda");
    let lhs = upload_i8(&dev, &a);
    let rhs = upload_i8(&dev, &b);
    let out = alloc_out(&dev, DType::I8, m * n);
    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = pick_alt(
        &table,
        OpKind::MatMul,
        &[DType::I8, DType::I8, DType::I8],
        fuel_storage::baracuda_dispatch::gemm_int::gemm_s8_rrr,
    );
    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::Matmul {
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            m,
            n,
            k,
        },
    )
    .expect("kernel call");

    let got_bytes = download(&out_arc.read().unwrap());
    let got: &[i8] = bytemuck::cast_slice(&got_bytes);
    let want = ref_gemm_s8(&a, &b, m, n, k);
    assert_eq!(got, &want[..]);
}

#[test]
#[ignore]
fn baracuda_gemm_u8_rrr_identity_matches_reference() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let m = 16;
    let n = 8;
    let k = 32;

    // Small magnitudes; product chain stays under u8::MAX so the
    // saturating cast is a no-op for the test.
    let a: Vec<u8> = (0..m * k).map(|i| (i % 3) as u8).collect();
    let b: Vec<u8> = (0..k * n).map(|i| (i % 2) as u8).collect();

    let dev = CudaDevice::new(0).expect("cuda");
    let lhs = upload_u8(&dev, &a);
    let rhs = upload_u8(&dev, &b);
    let out = alloc_out(&dev, DType::U8, m * n);
    let lhs_arc = Arc::new(RwLock::new(lhs));
    let rhs_arc = Arc::new(RwLock::new(rhs));
    let out_arc = Arc::new(RwLock::new(out));

    let kernel = pick_alt(
        &table,
        OpKind::MatMul,
        &[DType::U8, DType::U8, DType::U8],
        fuel_storage::baracuda_dispatch::gemm_int::gemm_u8_rrr,
    );
    kernel(
        &[lhs_arc.clone(), rhs_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::Matmul {
            lhs_batch_dims: vec![],
            rhs_batch_dims: vec![],
            m,
            n,
            k,
        },
    )
    .expect("kernel call");

    let got = download(&out_arc.read().unwrap());
    let want = ref_gemm_u8(&a, &b, m, n, k);
    assert_eq!(got, want);
}

/// At `[I8, I8, I8]` only the baracuda alternative exists (no PTX
/// int8 matmul). Verify lookup returns ≥1 alternative.
#[test]
fn int_gemm_registers_baracuda_only_alternative() {
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    let before = table
        .lookup_alternatives(
            OpKind::MatMul,
            &[DType::I8, DType::I8, DType::I8],
            BackendId::Cuda,
        )
        .len();
    register_baracuda_cuda_kernels(&mut table);
    let after = table
        .lookup_alternatives(
            OpKind::MatMul,
            &[DType::I8, DType::I8, DType::I8],
            BackendId::Cuda,
        )
        .len();
    assert_eq!(before, 0, "no I8 matmul registered by the PTX path");
    assert_eq!(after, 1, "baracuda registers exactly one I8 matmul alternative");
}
