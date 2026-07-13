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

use fuel_ir::Result;

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
    let err = |msg: String| fuel_ir::Error::Msg(msg).bt();
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
        ::paste::paste! {
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
            Ok(CudaStorageBytes::from_parts(
                Arc::new(out),
                device,
                dims.need_out,
            ))
        }

        #[doc = concat!(
            "Write-into-output variant of dense `", $label, "` matmul — ",
            "writes into the caller-provided `out` (no output alloc; ",
            "CapturedRun capture mode). Byte-identical launch(es) to `",
            stringify!($name), "` for a same-sized `out`; the only ",
            "difference is the output base pointer comes from `out` and ",
            "no device allocation happens.",
        )]
        #[allow(clippy::too_many_arguments)]
        pub fn [<$name _into>](
            lhs: &CudaStorageBytes,
            rhs: &CudaStorageBytes,
            lhs_batch_dims: &[usize],
            rhs_batch_dims: &[usize],
            m: usize,
            n: usize,
            k: usize,
            out: &CudaStorageBytes,
        ) -> Result<()> {
            let elem: usize = $elem;
            let dims = validate_dims(
                $label, lhs, rhs, lhs_batch_dims, rhs_batch_dims, m, n, k, elem,
            )?;
            let device = lhs.device().clone();
            if dims.need_out == 0 {
                return Ok(());
            }
            if out.len_bytes() < dims.need_out {
                return Err(fuel_ir::Error::Msg(format!(
                    "{}: write-into output buffer too small ({} < {} bytes)",
                    $label,
                    out.len_bytes(),
                    dims.need_out,
                ))
                .bt());
            }
            let stream = device.stream().as_raw() as *mut std::ffi::c_void;
            let (lda, ldb, ldd) = (k.max(1) as i64, n.max(1) as i64, n.max(1) as i64);
            let alpha: $scalar = 1.0;
            let beta: $scalar = 0.0;
            let lhs_base = lhs.buffer().as_raw().0;
            let rhs_base = rhs.buffer().as_raw().0;
            let out_base = out.buffer().as_raw().0;

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
            Ok(())
        }
        }
    };
}

gemm_dense_matmul!(matmul_f32,  sys::baracuda_kernels_gemm_dense_f32_run,  f32, 4, "gemm_dense_f32");
gemm_dense_matmul!(matmul_f64,  sys::baracuda_kernels_gemm_dense_f64_run,  f64, 8, "gemm_dense_f64");
gemm_dense_matmul!(matmul_f16,  sys::baracuda_kernels_gemm_dense_f16_run,  f32, 2, "gemm_dense_f16");
gemm_dense_matmul!(matmul_bf16, sys::baracuda_kernels_gemm_dense_bf16_run, f32, 2, "gemm_dense_bf16");

// =============================================================================
// cuBLAS same-hardware determinism audit (task-cublas-audit, 2026-07-10/11)
// =============================================================================
//
// Empirical Part 3 of the audit documented in
// `.superpowers/sdd/task-cublas-audit-report.md`: does `matmul_f32`
// (`cublasGemmEx` / `cublasGemmStridedBatchedEx` under
// `CUBLAS_GEMM_DEFAULT`, IEEE binary32 compute, via baracuda's pooled
// cuBLAS handle) produce bit-identical output for bit-identical input
// across (a) ≥100 repeat calls, (b) concurrent GPU load on a SEPARATE
// CUDA context/stream, and (c) fresh process restarts (a golden file
// on disk carries the expected bytes across `cargo test` invocations)?
//
// Shapes mirror `fuel-core`'s
// `forward_with_kv_context_captured_matches_persistent` fixture
// (`LlamaConfig { dim: 128, n_heads: 4, n_kv_heads: 2, head_dim: 32,
// ffn_dim: 512, vocab_size: 512 }`) at a real decode step (`seq == 1`,
// so `M == 1` — a GEMV-shaped `cublasGemmEx` launch): Q/O projection
// (`dim -> dim`), K/V projection (`dim -> kv_dim`, GQA-narrow), FFN
// up-proj (`dim -> ffn_dim`) and down-proj (`ffn_dim -> dim`), plus a
// batched attention-score-shaped launch (`batch == n_heads`) to
// exercise `cublasGemmStridedBatchedEx` too.
#[cfg(test)]
mod determinism_audit {
    use super::*;
    use crate::CudaDevice;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    fn dev_or_skip() -> Option<CudaDevice> {
        match CudaDevice::new(0) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("no CUDA device; skipping: {e:?}");
                None
            }
        }
    }

    /// Deterministic pseudo-random `f32` fill (fixed seed, xorshift64* —
    /// no external RNG dependency, no reliance on host time/entropy).
    /// MUST produce the exact same bytes on every process invocation:
    /// the cross-process golden-file check below depends on identical
    /// inputs, not just identical code.
    fn fill_deterministic(len: usize, seed: u64) -> Vec<f32> {
        let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map top 53 bits to [-1.0, 1.0); avoids the degenerate
                // all-zero / all-equal inputs that could mask real
                // nondeterminism behind trivial arithmetic.
                (((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
            })
            .collect()
    }

    fn to_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    struct Shape {
        label: &'static str,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    }

    /// Real decode-graph matmul shapes (see module comment) plus one
    /// batched shape to exercise `cublasGemmStridedBatchedEx`.
    const SHAPES: &[Shape] = &[
        Shape { label: "qkv_o_proj_dim128_gemv", batch: 1, m: 1, k: 128, n: 128 },
        Shape { label: "kv_proj_narrow_gqa",     batch: 1, m: 1, k: 128, n: 64 },
        Shape { label: "ffn_up_512",             batch: 1, m: 1, k: 128, n: 512 },
        Shape { label: "ffn_down_512",           batch: 1, m: 1, k: 512, n: 128 },
        Shape { label: "attn_scores_batched4",   batch: 4, m: 1, k: 32, n: 16 },
    ];

    /// Run one `matmul_f32` call for `shape` against fixed `lhs`/`rhs`
    /// device buffers and read the result back to host bytes.
    fn run_once(shape: &Shape, lhs: &CudaStorageBytes, rhs: &CudaStorageBytes) -> Vec<u8> {
        let lhs_batch_dims = vec![shape.batch];
        let rhs_batch_dims = vec![shape.batch];
        let out = matmul_f32(lhs, rhs, &lhs_batch_dims, &rhs_batch_dims, shape.m, shape.n, shape.k)
            .expect("matmul_f32 launch");
        out.to_cpu_bytes().expect("to_cpu_bytes (device-synchronizing D2H)")
    }

    /// Part 3.1 + 3.3: repeat-call bit-exactness (≥100 calls per shape)
    /// AND cross-process agreement (a golden file on disk is written on
    /// the first process to reach this shape and compared against on
    /// every subsequent process — run this test as 2-3 SEPARATE `cargo
    /// test` invocations to exercise the cross-process leg; a single
    /// invocation only exercises the repeat-call leg).
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn repeat_call_bit_exact_and_cross_process() {
        const ITERS: usize = 150;
        let Some(dev) = dev_or_skip() else { return };

        for shape in SHAPES {
            let lhs_data = fill_deterministic(shape.batch * shape.m * shape.k, 0xC0FFEE ^ shape.n as u64);
            let rhs_data = fill_deterministic(shape.batch * shape.k * shape.n, 0xFACADE ^ shape.k as u64);
            let lhs = CudaStorageBytes::from_cpu_bytes(&dev, &to_bytes(&lhs_data)).expect("lhs upload");
            let rhs = CudaStorageBytes::from_cpu_bytes(&dev, &to_bytes(&rhs_data)).expect("rhs upload");

            let first = run_once(shape, &lhs, &rhs);
            for i in 1..ITERS {
                let got = run_once(shape, &lhs, &rhs);
                assert_eq!(
                    got, first,
                    "cuBLAS determinism audit FAILED: shape={} iteration={} diverged from \
                     iteration 0 (repeat-call, same process, single stream)",
                    shape.label, i,
                );
            }
            eprintln!(
                "[determinism_audit] shape={} : {ITERS} repeat calls bit-identical (in-process)",
                shape.label
            );

            // Cross-process leg: compare against (or seed) a golden file.
            let golden_path = std::env::temp_dir()
                .join(format!("fuel_cublas_audit_golden_{}.bin", shape.label));
            if golden_path.exists() {
                let golden = std::fs::read(&golden_path).expect("read golden file");
                assert_eq!(
                    golden, first,
                    "cuBLAS determinism audit FAILED: shape={} result differs from a PRIOR \
                     process's golden output (fresh cuBLAS handle / context) at {}",
                    shape.label,
                    golden_path.display(),
                );
                eprintln!(
                    "[determinism_audit] shape={} : matches prior-process golden at {}",
                    shape.label,
                    golden_path.display()
                );
            } else {
                std::fs::write(&golden_path, &first).expect("write golden file");
                eprintln!(
                    "[determinism_audit] shape={} : wrote golden file at {} (first process to run)",
                    shape.label,
                    golden_path.display()
                );
            }
        }
    }

    /// Part 3.2: repeat-call bit-exactness while a SEPARATE CUDA
    /// context/stream (a second `CudaDevice::new(0)` — baracuda's
    /// gemm_dense facade pools cuBLAS handles keyed by context, so this
    /// genuinely exercises a second, concurrently-active stream, not
    /// just a second thread serialized behind the same stream) hammers
    /// the GPU with its own GEMM launches in a tight loop. This is
    /// exactly the condition NVIDIA's cuBLAS docs flag as the one where
    /// the same-hardware bit-reproducibility guarantee "no longer
    /// holds" (multiple concurrent streams).
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn repeat_call_bit_exact_under_concurrent_stream_load() {
        const ITERS: usize = 150;
        let Some(dev) = dev_or_skip() else { return };

        // Background "noisy neighbor": its own context + stream, its
        // own cuBLAS handle (pooled per-context), hammering a
        // deliberately-different, larger GEMM shape continuously.
        let stop = Arc::new(AtomicBool::new(false));
        let stop_bg = stop.clone();
        let bg_handle = std::thread::spawn(move || {
            let Some(bg_dev) = dev_or_skip() else { return 0usize };
            let bg_shape = Shape { label: "bg_noise", batch: 1, m: 64, k: 512, n: 512 };
            let lhs_data = fill_deterministic(bg_shape.m * bg_shape.k, 0x1234_5678);
            let rhs_data = fill_deterministic(bg_shape.k * bg_shape.n, 0x8765_4321);
            let lhs = CudaStorageBytes::from_cpu_bytes(&bg_dev, &to_bytes(&lhs_data)).expect("bg lhs upload");
            let rhs = CudaStorageBytes::from_cpu_bytes(&bg_dev, &to_bytes(&rhs_data)).expect("bg rhs upload");
            let mut count = 0usize;
            while !stop_bg.load(Ordering::Relaxed) {
                let _ = run_once(&bg_shape, &lhs, &rhs);
                count += 1;
            }
            count
        });

        // Give the background thread a moment to actually start
        // launching before we begin the measured loop, so the
        // concurrent-stream window covers the whole measured run, not
        // just the tail of it.
        std::thread::sleep(std::time::Duration::from_millis(50));

        for shape in SHAPES {
            let lhs_data = fill_deterministic(shape.batch * shape.m * shape.k, 0x5EED_0001 ^ shape.n as u64);
            let rhs_data = fill_deterministic(shape.batch * shape.k * shape.n, 0x5EED_0002 ^ shape.k as u64);
            let lhs = CudaStorageBytes::from_cpu_bytes(&dev, &to_bytes(&lhs_data)).expect("lhs upload");
            let rhs = CudaStorageBytes::from_cpu_bytes(&dev, &to_bytes(&rhs_data)).expect("rhs upload");

            let first = run_once(shape, &lhs, &rhs);
            for i in 1..ITERS {
                let got = run_once(shape, &lhs, &rhs);
                assert_eq!(
                    got, first,
                    "cuBLAS determinism audit FAILED under concurrent GPU load: shape={} \
                     iteration={} diverged from iteration 0 while a second context/stream was \
                     concurrently launching GEMMs",
                    shape.label, i,
                );
            }
            eprintln!(
                "[determinism_audit] shape={} : {ITERS} repeat calls bit-identical UNDER \
                 concurrent cross-stream GPU load",
                shape.label
            );
        }

        stop.store(true, Ordering::Relaxed);
        let bg_count = bg_handle.join().expect("bg thread join");
        eprintln!(
            "[determinism_audit] background noisy-neighbor thread completed {bg_count} GEMM \
             launches on a separate context/stream during the measured window"
        );
        assert!(
            bg_count > 0,
            "background concurrent-load thread never got to launch a single GEMM — the \
             concurrent-load leg of this test did not actually exercise concurrency; treat \
             a pass here as inconclusive, not confirmatory"
        );
    }
}
