//! Live-CUDA tests for baracuda-kernels-sys-backed Cast registered as
//! sibling alternatives at `(OpKind::Cast, [src_dt, dst_dt], Cuda)`
//! decision points.
//!
//! See `baracuda_unary_live.rs` for the dual-table + pick-by-fn-pointer
//! pattern.
//!
//! Cast is the first family where the binding-table key uses two
//! *different* dtypes (everything before this has been `[dt, dt]` for
//! same-dtype ops). The fan-out is ~49 distinct (src, dst) pairs;
//! these tests sample a few representative ones — full coverage would
//! 49× this file with no extra information per test.

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

/// Look up the baracuda kernel at `(op, dtypes, Cuda)` by function-
/// pointer identity. Tolerates pairs where baracuda is the sole
/// registered impl (e.g. I32-touching pairs that the PTX path doesn't
/// register).
fn pick_alt(
    table: &KernelBindingTable,
    op: OpKind,
    dtypes: &[DType],
    expected: fuel_storage::KernelRef,
) -> fuel_storage::KernelRef {
    let alternatives = table.lookup_alternatives(op, dtypes, BackendId::Cuda);
    assert!(
        !alternatives.is_empty(),
        "expected ≥ 1 alternative at ({op:?}, {dtypes:?}, Cuda); got 0",
    );
    let expected_ptr = expected as usize;
    for alt in alternatives {
        if (alt.kernel as usize) == expected_ptr {
            return alt.kernel;
        }
    }
    panic!(
        "expected baracuda KernelRef not found among {} alternatives at ({op:?}, {dtypes:?})",
        table
            .lookup_alternatives(op, dtypes, BackendId::Cuda)
            .len(),
    )
}

/// Upload typed host bytes to CUDA + wrap as a `Storage` of dtype `dt`.
fn upload_bytes(dev: &CudaDevice, dt: DType, bytes: &[u8]) -> Storage {
    let cuda = CudaStorageBytes::from_cpu_bytes(dev, bytes).expect("h2d");
    Storage::new(BackendStorage::Cuda(cuda), dt)
}

fn alloc_out(dev: &CudaDevice, dt: DType, n_elems: usize) -> Storage {
    let elem_size = match dt {
        DType::F32 | DType::I32 | DType::U32 => 4,
        DType::F64 | DType::I64 => 8,
        DType::F16 | DType::BF16 => 2,
        DType::U8 | DType::F8E4M3 => 1,
        other => panic!("unsupported test dtype {other:?}"),
    };
    let buf = CudaStorageBytes::alloc(dev, n_elems * elem_size).expect("alloc");
    Storage::new(BackendStorage::Cuda(buf), dt)
}

fn download_bytes(s: &Storage) -> Vec<u8> {
    match &s.inner {
        BackendStorage::Cuda(c) => c.to_cpu_bytes().expect("d2h"),
        _ => panic!("not on CUDA"),
    }
}

/// Run the baracuda Cast wrapper end-to-end via the binding table.
/// Returns the raw output bytes for the caller to reinterpret.
fn run_cast(
    table: &KernelBindingTable,
    src_dt: DType,
    dst_dt: DType,
    src_bytes: &[u8],
    n_elems: usize,
) -> Vec<u8> {
    let dev = CudaDevice::new(0).expect("cuda");
    let src = upload_bytes(&dev, src_dt, src_bytes);
    let out = alloc_out(&dev, dst_dt, n_elems);
    let src_arc = Arc::new(RwLock::new(src));
    let out_arc = Arc::new(RwLock::new(out));
    let kernel = pick_alt(
        table,
        OpKind::Cast,
        &[src_dt, dst_dt],
        fuel_storage::baracuda_dispatch::cast::cast_baracuda_wrapper,
    );
    kernel(
        &[src_arc.clone()],
        &mut [out_arc.clone()],
        &[],
        &OpParams::None,
    )
    .expect("kernel call");
    download_bytes(&out_arc.read().unwrap())
}

#[test]
#[ignore]
fn baracuda_cast_f32_to_f64() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f32> = vec![1.5, -2.25, 0.0, 3.75];
    let src_bytes: &[u8] = bytemuck::cast_slice(&input);
    let out_bytes = run_cast(&table, DType::F32, DType::F64, src_bytes, input.len());
    let got: &[f64] = bytemuck::cast_slice(&out_bytes);
    let want: Vec<f64> = input.iter().map(|&x| x as f64).collect();
    assert_eq!(got, &want[..]);
}

#[test]
#[ignore]
fn baracuda_cast_f64_to_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f64> = vec![1.5, -2.25, 0.0, 3.75];
    let src_bytes: &[u8] = bytemuck::cast_slice(&input);
    let out_bytes = run_cast(&table, DType::F64, DType::F32, src_bytes, input.len());
    let got: &[f32] = bytemuck::cast_slice(&out_bytes);
    let want: Vec<f32> = input.iter().map(|&x| x as f32).collect();
    assert_eq!(got, &want[..]);
}

#[test]
#[ignore]
fn baracuda_cast_f32_to_i32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f32> = vec![1.7, -2.3, 0.0, 3.99];
    let src_bytes: &[u8] = bytemuck::cast_slice(&input);
    let out_bytes = run_cast(&table, DType::F32, DType::I32, src_bytes, input.len());
    let got: &[i32] = bytemuck::cast_slice(&out_bytes);
    // C-style trunc-toward-zero conversion (baracuda matches CUDA's __float2int_rz).
    assert_eq!(got, &[1, -2, 0, 3]);
}

#[test]
#[ignore]
fn baracuda_cast_i32_to_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<i32> = vec![-7, 0, 1, 1234];
    let src_bytes: &[u8] = bytemuck::cast_slice(&input);
    let out_bytes = run_cast(&table, DType::I32, DType::F32, src_bytes, input.len());
    let got: &[f32] = bytemuck::cast_slice(&out_bytes);
    let want: Vec<f32> = input.iter().map(|&x| x as f32).collect();
    assert_eq!(got, &want[..]);
}

#[test]
#[ignore]
fn baracuda_cast_u32_collapses_to_i32_at_ffi() {
    // Verify the U32 → i32 reinterpret trick: bit-identical bytes for
    // non-negative values, so cast_u32_to_f32 must agree with
    // cast_i32_to_f32 byte-for-byte.
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input_u32: Vec<u32> = vec![0, 1, 7, 1234];
    let src_bytes: &[u8] = bytemuck::cast_slice(&input_u32);
    let out_bytes = run_cast(&table, DType::U32, DType::F32, src_bytes, input_u32.len());
    let got: &[f32] = bytemuck::cast_slice(&out_bytes);
    let want: Vec<f32> = input_u32.iter().map(|&x| x as f32).collect();
    assert_eq!(got, &want[..]);
}

#[test]
#[ignore]
fn baracuda_cast_u8_to_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<u8> = vec![0, 1, 255, 128];
    let out_bytes = run_cast(&table, DType::U8, DType::F32, &input, input.len());
    let got: &[f32] = bytemuck::cast_slice(&out_bytes);
    assert_eq!(got, &[0.0, 1.0, 255.0, 128.0]);
}

#[test]
#[ignore]
fn baracuda_cast_i64_to_f32() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<i64> = vec![-7, 0, 1, 1234];
    let src_bytes: &[u8] = bytemuck::cast_slice(&input);
    let out_bytes = run_cast(&table, DType::I64, DType::F32, src_bytes, input.len());
    let got: &[f32] = bytemuck::cast_slice(&out_bytes);
    let want: Vec<f32> = input.iter().map(|&x| x as f32).collect();
    assert_eq!(got, &want[..]);
}

/// Sole-source check (CPU-only, no GPU). Post-fuel-cuda-kernels-cleanup
/// (2026-05-25): baracuda is the single source of truth for CUDA Cast;
/// the legacy PTX path no longer registers duplicate alternatives.
#[test]
fn baracuda_is_sole_cast_source() {
    let mut table = KernelBindingTable::new();
    register_cuda_kernels(&mut table);
    let before = table
        .lookup_alternatives(OpKind::Cast, &[DType::F32, DType::F64], BackendId::Cuda)
        .len();
    register_baracuda_cuda_kernels(&mut table);
    let after = table
        .lookup_alternatives(OpKind::Cast, &[DType::F32, DType::F64], BackendId::Cuda)
        .len();
    assert_eq!(before, 0, "PTX path no longer registers F32→F64 cast");
    assert_eq!(after, 1, "baracuda is the sole F32→F64 cast source");
}

// ---- F8E4M3 casts (alpha.29 CastSubBytePlan, OCP/NV FP8 family) ----------

/// F32 → F8E4M3 → F32 round-trip. F8E4M3 has ~3 bits of mantissa, so
/// values lose precision but should round to nearby representable
/// values. We pick inputs that hit exact representable points to keep
/// the assertion tight.
#[test]
#[ignore]
fn baracuda_cast_f32_to_f8e4m3_round_trip() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    // 0, ±0.5, ±1.0, ±2.0, ±4.0 — all exactly representable in F8E4M3.
    let input: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0, 4.0];
    let src_bytes: &[u8] = bytemuck::cast_slice(&input);
    let mid = run_cast(&table, DType::F32, DType::F8E4M3, src_bytes, input.len());
    assert_eq!(mid.len(), input.len(), "F32 → F8E4M3 should produce 1 byte per element");

    let back = run_cast(&table, DType::F8E4M3, DType::F32, &mid, input.len());
    let got: &[f32] = bytemuck::cast_slice(&back);
    for (i, (&want, &g)) in input.iter().zip(got.iter()).enumerate() {
        assert_eq!(
            want, g,
            "F32 → F8E4M3 → F32 mismatch at index {i}: want {want}, got {g}",
        );
    }
}

/// F8E4M3 → BF16 → F8E4M3 round-trip for the same exactly-representable
/// set. Verifies the F16/BF16 sibling paths also land cleanly.
#[test]
#[ignore]
fn baracuda_cast_f8e4m3_through_bf16_round_trip() {
    let Some(_dev) = dev_or_skip() else { return };
    let table = dual_table();
    let input: Vec<f32> = vec![0.0, 0.5, 1.0, 2.0];
    let src_bytes: &[u8] = bytemuck::cast_slice(&input);
    // F32 → F8E4M3.
    let fp8 = run_cast(&table, DType::F32, DType::F8E4M3, src_bytes, input.len());
    // F8E4M3 → BF16.
    let bf = run_cast(&table, DType::F8E4M3, DType::BF16, &fp8, input.len());
    assert_eq!(bf.len(), input.len() * 2, "BF16 is 2 bytes/elem");
    // BF16 → F8E4M3.
    let fp8_back = run_cast(&table, DType::BF16, DType::F8E4M3, &bf, input.len());
    assert_eq!(fp8, fp8_back, "F8E4M3 → BF16 → F8E4M3 should be bit-stable for representable inputs");
}

#[test]
fn f8e4m3_cast_registered_for_3_target_dtypes() {
    let table = dual_table();
    for other in [DType::F32, DType::F16, DType::BF16] {
        for (src, dst) in [(DType::F8E4M3, other), (other, DType::F8E4M3)] {
            let alts = table.lookup_alternatives(
                OpKind::Cast,
                &[src, dst],
                BackendId::Cuda,
            );
            assert!(
                !alts.is_empty(),
                "no Cast CUDA registration for ({src:?} → {dst:?})",
            );
        }
    }
}
