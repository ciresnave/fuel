//! Dense FP matmul over baracuda's Phase 74 `gemm_dense_*` facade
//! (alpha.67) — the cuBLAS-backed flat-C GEMM family that answers
//! Fuel's 2026-06-10 ask and retires the last hand-written matmul
//! path in this crate (`byte_kernels::matmul_{f32,bf16,f16}`).
//!
//! One generic launcher serves all four dtypes; per-dtype entry
//! points are macro manifest lines like the rest of `baracuda::*`.
//! Compared to the retired paths this adds:
//!
//! - **f64 matmul** — net-new CUDA coverage (the cuBLAS residue was
//!   f32/bf16/f16 only).
//! - **GQA / broadcast batch for bf16 + f16** — the CUTLASS byte path
//!   rejected per-axis broadcast batches; the facade's per-slot loop
//!   (and `stride_b = 0` single-call broadcast) serves every dtype
//!   uniformly.
//!
//! Layout contract: operands are packed row-major per batch slot
//! (`A: [M, K]`, `B: [K, N]`, `D: [M, N]`, layout tag 0 = RRR), which
//! is what the byte-length validation below enforces — identical to
//! the retired paths' contract, so the auto-Contiguize gate upstream
//! is unchanged. The facade itself accepts padded leading dims; Fuel
//! can exploit that later by relaxing the validation, not the launch.
//!
//! Precision: f32 is true IEEE binary32 (cuBLAS default math mode,
//! NOT TF32 — but the process-wide `NVIDIA_TF32_OVERRIDE=1` env var
//! would force TF32 inside cuBLAS; don't set it). f16/bf16 accumulate
//! in f32; f64 in f64. Run-to-run bitwise reproducibility holds under
//! cuBLAS's single-active-stream condition.

use std::sync::Arc;

use fuel_core_types::Result;

use crate::byte_storage::CudaStorageBytes;

use baracuda_kernels_sys as sys;

/// Validated batch geometry shared by every dtype's entry point.
struct MatmulDims {
    /// Per-axis lhs/rhs batch repeat factor (1 = equal, >1 = GQA).
    n_rep: Vec<usize>,
    lhs_batch_count: usize,
    rhs_batch_count: usize,
    lhs_per_batch: usize,
    rhs_per_batch: usize,
    out_per_batch: usize,
    need_out: usize,
}

/// Port of the retired `byte_kernels::matmul_f32` validation: batch
/// ranks must match, per-axis dims must be equal or GQA-divisible,
/// byte lengths must match the packed row-major contract, and both
/// operands must live on one device. Also bounds every dimension the
/// FFI receives as `i32`.
fn validate_dims(
    label: &'static str,
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
    elem: usize,
) -> Result<MatmulDims> {
    let err = |msg: String| fuel_core_types::Error::Msg(msg).bt();
    if lhs_batch_dims.len() != rhs_batch_dims.len() {
        return Err(err(format!(
            "{label}: batch ranks must match (lhs={}, rhs={}); fuel-graph's \
             auto-broadcast equalizes them at graph construction time",
            lhs_batch_dims.len(),
            rhs_batch_dims.len(),
        )));
    }
    let batch_rank = lhs_batch_dims.len();
    let mut n_rep: Vec<usize> = Vec::with_capacity(batch_rank);
    for i in 0..batch_rank {
        let la = lhs_batch_dims[i];
        let ra = rhs_batch_dims[i];
        if la == ra {
            n_rep.push(1);
        } else if ra > 0 && la > ra && la % ra == 0 {
            n_rep.push(la / ra);
        } else {
            return Err(err(format!(
                "{label}: batch dim {i} disallowed combination (lhs={la}, rhs={ra}); \
                 must be equal or GQA-divisible (lhs > rhs && lhs % rhs == 0)",
            )));
        }
    }
    let lhs_per_batch = m.saturating_mul(k);
    let rhs_per_batch = k.saturating_mul(n);
    let out_per_batch = m.saturating_mul(n);
    let lhs_batch_count: usize = lhs_batch_dims.iter().product::<usize>().max(1);
    let rhs_batch_count: usize = rhs_batch_dims.iter().product::<usize>().max(1);
    let need_lhs = lhs_batch_count.saturating_mul(lhs_per_batch).saturating_mul(elem);
    let need_rhs = rhs_batch_count.saturating_mul(rhs_per_batch).saturating_mul(elem);
    let need_out = lhs_batch_count.saturating_mul(out_per_batch).saturating_mul(elem);
    if lhs.len_bytes() != need_lhs {
        return Err(err(format!(
            "{label}: lhs bytes={} doesn't match shape {:?} + [{m}, {k}]",
            lhs.len_bytes(),
            lhs_batch_dims,
        )));
    }
    if rhs.len_bytes() != need_rhs {
        return Err(err(format!(
            "{label}: rhs bytes={} doesn't match shape {:?} + [{k}, {n}]",
            rhs.len_bytes(),
            rhs_batch_dims,
        )));
    }
    if rhs.device().id() != lhs.device().id() {
        return Err(err(format!(
            "{label}: lhs and rhs are on different CUDA devices; cross-device \
             matmul is the caller's responsibility (insert Op::Move first)",
        )));
    }
    let i32_max = i32::MAX as usize;
    if m > i32_max || n > i32_max || k > i32_max || lhs_batch_count > i32_max {
        return Err(err(format!(
            "{label}: dimension exceeds i32 range (m={m}, n={n}, k={k}, \
             batch={lhs_batch_count}); the gemm_dense FFI takes i32 dims",
        )));
    }
    Ok(MatmulDims {
        n_rep,
        lhs_batch_count,
        rhs_batch_count,
        lhs_per_batch,
        rhs_per_batch,
        out_per_batch,
        need_out,
    })
}

/// One dtype's matmul entry point over `gemm_dense_<dt>_run`.
///
/// `$scalar` is the α/β scalar type the FFI symbol takes (`f32` for
/// the f32/f16/bf16 symbols, `f64` for f64), NOT the storage dtype.
macro_rules! gemm_dense_matmul {
    ($name:ident, $run:path, $scalar:ty, $elem:expr, $label:expr $(,)?) => {
        #[doc = concat!(
            "Dense `", $label, "` matmul via baracuda's Phase 74 ",
            "`gemm_dense` facade (layout 0 = RRR, packed operands). ",
            "Equal batches launch once (strided batch); a broadcast ",
            "rhs (`rhs_batch_count == 1`) launches once with ",
            "`stride_b = 0`; general GQA loops per lhs slot.",
        )]
        pub fn $name(
            lhs: &CudaStorageBytes,
            rhs: &CudaStorageBytes,
            lhs_batch_dims: &[usize],
            rhs_batch_dims: &[usize],
            m: usize,
            n: usize,
            k: usize,
        ) -> Result<CudaStorageBytes> {
            let elem: usize = $elem;
            let dims = validate_dims(
                $label, lhs, rhs, lhs_batch_dims, rhs_batch_dims, m, n, k, elem,
            )?;
            let device = lhs.device().clone();
            if dims.need_out == 0 {
                return CudaStorageBytes::alloc(&device, 0);
            }
            let out = device.alloc_zeros::<u8>(dims.need_out)?;
            let stream = device.stream().as_raw() as *mut std::ffi::c_void;
            let (lda, ldb, ldd) = (k.max(1) as i64, n.max(1) as i64, n.max(1) as i64);
            let alpha: $scalar = 1.0;
            let beta: $scalar = 0.0;
            let lhs_base = lhs.buffer().as_raw().0;
            let rhs_base = rhs.buffer().as_raw().0;
            let out_base = out.as_raw().0;

            let all_equal = dims.n_rep.iter().all(|&r| r == 1);
            let broadcast_rhs = dims.rhs_batch_count == 1;
            if all_equal || broadcast_rhs {
                // Single strided-batch launch. `stride_b = 0`
                // broadcasts the lone rhs across every lhs slot.
                let stride_b = if broadcast_rhs && !all_equal {
                    0
                } else {
                    dims.rhs_per_batch as i64
                };
                // SAFETY: pointers validated against the packed
                // byte-length contract above; `stream` belongs to the
                // operands' device; α/β passed by value per the
                // facade ABI. Sync follows (sync KernelRef contract).
                let status = unsafe {
                    $run(
                        m as i32, n as i32, k as i32,
                        dims.lhs_batch_count as i32,
                        0, // layout: RRR
                        alpha, beta,
                        lhs_base as *const std::ffi::c_void, lda,
                        dims.lhs_per_batch as i64,
                        rhs_base as *const std::ffi::c_void, ldb,
                        stride_b,
                        out_base as *mut std::ffi::c_void, ldd,
                        dims.out_per_batch as i64,
                        std::ptr::null_mut(), 0,
                        stream,
                    )
                };
                crate::baracuda::status::check(status, $label)?;
            } else {
                // General GQA: decode each lhs flat batch index to a
                // multi-index, map per-axis through n_rep to the rhs
                // slot, one `batch = 1` launch per slot (strides
                // ignored at batch == 1). Mirrors the CPU kernel's
                // per-batch loop.
                let batch_rank = lhs_batch_dims.len();
                let mut lhs_multi = vec![0usize; batch_rank];
                for b in 0..dims.lhs_batch_count {
                    let mut rem = b;
                    for d in (0..batch_rank).rev() {
                        let s = lhs_batch_dims[d];
                        lhs_multi[d] = rem % s;
                        rem /= s;
                    }
                    let mut rhs_b = 0usize;
                    for d in 0..batch_rank {
                        rhs_b = rhs_b * rhs_batch_dims[d] + (lhs_multi[d] / dims.n_rep[d]);
                    }
                    let lhs_off = (b * dims.lhs_per_batch * elem) as u64;
                    let rhs_off = (rhs_b * dims.rhs_per_batch * elem) as u64;
                    let out_off = (b * dims.out_per_batch * elem) as u64;
                    // SAFETY: offsets stay within the validated byte
                    // ranges (b < lhs_batch_count, rhs_b <
                    // rhs_batch_count by construction).
                    let status = unsafe {
                        $run(
                            m as i32, n as i32, k as i32,
                            1, // batch
                            0, // layout: RRR
                            alpha, beta,
                            (lhs_base + lhs_off) as *const std::ffi::c_void, lda, 0,
                            (rhs_base + rhs_off) as *const std::ffi::c_void, ldb, 0,
                            (out_base + out_off) as *mut std::ffi::c_void, ldd, 0,
                            std::ptr::null_mut(), 0,
                            stream,
                        )
                    };
                    crate::baracuda::status::check(status, $label)?;
                }
            }
            device.synchronize()?;
            Ok(CudaStorageBytes::from_parts(
                Arc::new(out),
                device,
                dims.need_out,
            ))
        }
    };
}

gemm_dense_matmul!(matmul_f32,  sys::baracuda_kernels_gemm_dense_f32_run,  f32, 4, "gemm_dense_f32");
gemm_dense_matmul!(matmul_f64,  sys::baracuda_kernels_gemm_dense_f64_run,  f64, 8, "gemm_dense_f64");
gemm_dense_matmul!(matmul_f16,  sys::baracuda_kernels_gemm_dense_f16_run,  f32, 2, "gemm_dense_f16");
gemm_dense_matmul!(matmul_bf16, sys::baracuda_kernels_gemm_dense_bf16_run, f32, 2, "gemm_dense_bf16");
