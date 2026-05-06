//! Byte-level CUDA kernels — Phase 7.5 unified-storage migration.
//!
//! These kernels operate on `CudaStorageBytes` (raw `DeviceBuffer<u8>`)
//! rather than the dtype-tagged legacy `CudaStorage` enum. Dispatch
//! to the right CUDA function happens via wrappers in
//! `fuel-storage::dispatch::register_cuda_kernels`; the typed kernel
//! functions in `fuel-cuda-kernels` are launched by passing
//! `&DeviceBuffer<u8>` as the kernel arg — at the CUDA driver level
//! the typed pointer (`f32*`, `f64*`, etc.) and the byte pointer have
//! the same value, and the kernel's compiled code interprets the
//! bytes per its declared type.
//!
//! The kernels in `fuel-cuda-kernels` (e.g. `badd_f32`) accept the
//! signature `(elem_count, ndims, dims_strides_or_null, lhs, rhs,
//! out)`. A null `dims_strides_or_null` selects the kernel's
//! contiguous fast path; the unified executor's auto-Contiguize pass
//! guarantees inputs are contiguous before kernel call, so the
//! wrappers always pass null.

use std::sync::Arc;

use fuel_core_types::{DType, Layout, Result};
use fuel_cuda_kernels as kernels;

use crate::builder_arg as barg;
use crate::byte_storage::CudaStorageBytes;
use crate::device::LaunchConfig;
use crate::error::WrapErr;
use crate::storage::SlicePtrOrNull;

/// Phase 7.5 first CUDA kernel through the unified path.
/// Element-wise add of two F32 `CudaStorageBytes`. Both inputs must
/// have the same byte length (== same element count for F32). Output
/// is freshly allocated on the same device as `lhs`; caller is
/// responsible for storing it where the unified executor expects it.
///
/// Auto-Contiguize is assumed: this wrapper passes null for the
/// dims/strides side-band, selecting the kernel's contiguous fast
/// path. Strided inputs through the unified path are an A5 follow-on
/// (Layout-on-KernelRef extension).
pub fn add_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "badd_f32")
}

/// Element-wise subtraction (lhs - rhs) of two F32 `CudaStorageBytes`.
/// Same shape as [`add_elementwise_f32`]; only the launched kernel
/// name differs.
pub fn sub_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "bsub_f32")
}

/// Element-wise multiplication (lhs * rhs) of two F32 `CudaStorageBytes`.
pub fn mul_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "bmul_f32")
}

/// Element-wise division (lhs / rhs) of two F32 `CudaStorageBytes`.
pub fn div_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "bdiv_f32")
}

/// Element-wise maximum (max(lhs, rhs)) of two F32 `CudaStorageBytes`.
pub fn maximum_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "bmaximum_f32")
}

/// Element-wise minimum (min(lhs, rhs)) of two F32 `CudaStorageBytes`.
pub fn minimum_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
) -> Result<CudaStorageBytes> {
    binary_elementwise_f32(lhs, rhs, "bminimum_f32")
}

/// Element-wise ReLU (max(x, 0)) of one F32 `CudaStorageBytes`.
/// First unary op through the unified binding table; extracts the
/// shared [`unary_elementwise_f32`] helper for the rest of the F32
/// unary fanout to delegate to.
pub fn relu_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "urelu_f32")
}

/// Element-wise negation (-x) of one F32 `CudaStorageBytes`.
pub fn neg_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "uneg_f32")
}

/// Element-wise square (x * x) of one F32 `CudaStorageBytes`.
pub fn sqr_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "usqr_f32")
}

/// Element-wise square root (sqrt(x)) of one F32 `CudaStorageBytes`.
pub fn sqrt_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "usqrt_f32")
}

/// Element-wise reciprocal (1/x) of one F32 `CudaStorageBytes`.
pub fn recip_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "urecip_f32")
}

/// Element-wise absolute value (|x|) of one F32 `CudaStorageBytes`.
pub fn abs_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "uabs_f32")
}

/// Element-wise hyperbolic tangent (tanh(x)) of one F32 `CudaStorageBytes`.
pub fn tanh_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "utanh_f32")
}

/// Element-wise exp(x) of one F32 `CudaStorageBytes`.
pub fn exp_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "uexp_f32")
}

/// Element-wise natural log (ln(x)) of one F32 `CudaStorageBytes`.
pub fn log_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "ulog_f32")
}

/// Element-wise sin(x) of one F32 `CudaStorageBytes`.
pub fn sin_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "usin_f32")
}

/// Element-wise cos(x) of one F32 `CudaStorageBytes`.
pub fn cos_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "ucos_f32")
}

/// Element-wise sigmoid (1 / (1 + exp(-x))) of one F32 `CudaStorageBytes`.
pub fn sigmoid_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "usigmoid_f32")
}

/// Element-wise SiLU (x * sigmoid(x)) of one F32 `CudaStorageBytes`.
pub fn silu_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "usilu_f32")
}

/// Element-wise GELU (tanh approximation) of one F32 `CudaStorageBytes`.
/// Maps to `ugelu_f32` (the kernel's `gelu_fwd`); the erf variant is
/// `ugelu_erf_f32` and is exposed by `OpKind::GeluErfElementwise` if/when
/// it's added to the binding table.
pub fn gelu_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "ugelu_f32")
}

/// Element-wise Heaviside step (1.0 if x > 0 else 0.0) of one F32
/// `CudaStorageBytes`. Maps to `ustep_f32`, which was added to
/// `fuel-cuda-kernels::UNARY` (via `unary.cu`) in the same commit
/// that introduced this wrapper — the rest of the legacy unary
/// kernels predated it.
pub fn step_elementwise_f32(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
    unary_elementwise_f32(src, "ustep_f32")
}

/// Sum-reduce one F32 `CudaStorageBytes` along the dims listed in
/// `reduce_dims`. First reduction op through the unified binding
/// table; extracts the shared [`reduce_f32`] helper for Max/Min/Mean
/// to delegate to. Output is freshly allocated, sized
/// `prod(non-reduced dims) * sizeof(f32)`.
pub fn sum_reduce_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    reduce_dims: &[usize],
) -> Result<CudaStorageBytes> {
    reduce_f32(src, input_layout, reduce_dims, "fast_sum_f32")
}

/// Max-reduce one F32 `CudaStorageBytes` along the dims listed in
/// `reduce_dims`. Same shape contract as [`sum_reduce_f32`]; only
/// the launched kernel name differs.
pub fn max_reduce_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    reduce_dims: &[usize],
) -> Result<CudaStorageBytes> {
    reduce_f32(src, input_layout, reduce_dims, "fast_max_f32")
}

/// Min-reduce one F32 `CudaStorageBytes` along the dims listed in
/// `reduce_dims`. Same shape contract as [`sum_reduce_f32`].
pub fn min_reduce_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    reduce_dims: &[usize],
) -> Result<CudaStorageBytes> {
    reduce_f32(src, input_layout, reduce_dims, "fast_min_f32")
}

/// Mean-reduce one F32 `CudaStorageBytes` along the dims listed in
/// `reduce_dims`. Composed: launch `fast_sum_f32` (via the shared
/// reduce helper), then launch `affine_f32` with `mul = 1/divisor`
/// and `add = 0` to scale the sum into the mean. Mirrors the CPU
/// `mean_reduce_f32` (sum then in-place scale). The two-launch
/// shape avoids needing a dedicated `fast_mean_f32` PTX kernel; if
/// profiling later shows the second launch matters, a fused kernel
/// is the natural follow-on.
pub fn mean_reduce_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    reduce_dims: &[usize],
) -> Result<CudaStorageBytes> {
    let sum = reduce_f32(src, input_layout, reduce_dims, "fast_sum_f32")?;
    let src_dims = input_layout.shape().dims();
    let divisor: usize = reduce_dims.iter().map(|&d| src_dims[d]).product();
    if divisor == 0 {
        return Err(fuel_core_types::Error::Msg(
            "mean_reduce_f32: divisor zero (reduced dim has size 0)".to_string(),
        )
        .bt());
    }
    let inv = 1.0_f32 / divisor as f32;
    affine_f32(&sum, inv, 0.0)
}

/// Batched row-major F32 matmul through cuBLAS, on byte-shaped inputs.
/// Shape contract per `OpParams::Matmul`: `lhs [..lhs_batch.., m, k] @
/// rhs [..rhs_batch.., k, n] → out [..lhs_batch.., m, n]`. Inputs are
/// guaranteed contiguous by the executor's auto-Contiguize pass, so
/// per-batch element strides are `m*k`, `k*n`, `m*n` respectively.
///
/// The cuBLAS row-major-via-col-major trick: pass our `rhs` as cuBLAS
/// `A` and our `lhs` as cuBLAS `B`, swap `m` and `n` in the call, and
/// use no transposes. cuBLAS computes `C^T = B^T × A^T` in col-major
/// terms, which equals `A_row × B_row` viewed back as row-major. See
/// the legacy `matmul_via_cublas` (`storage.rs::CudaStorage::matmul`)
/// — same mechanic.
///
/// Two paths:
/// - **Equal-batch fast path** (all per-axis dims match): single
///   `gemm_strided_batched_ex` call with `batch_count = lhs_batch_count`.
/// - **GQA per-batch loop** (per-axis `lhs_dim = n_rep_axis * rhs_dim`):
///   one `gemm_ex` call per lhs batch slot, with the rhs slot index
///   computed via the per-axis `n_rep` mapping (mirrors CPU's
///   `matmul_f32`). Slow but correct for any GQA pattern; if profiling
///   shows it matters, the natural follow-on is per-rhs-slot grouping
///   for innermost-axis-only n_rep (the GQA-attention common case).
pub fn matmul_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_batch_dims: &[usize],
    rhs_batch_dims: &[usize],
    m: usize,
    n: usize,
    k: usize,
) -> Result<CudaStorageBytes> {
    if lhs_batch_dims.len() != rhs_batch_dims.len() {
        return Err(fuel_core_types::Error::Msg(format!(
            "matmul_f32: batch ranks must match (lhs={}, rhs={}); fuel-graph's \
             auto-broadcast equalizes them at graph construction time",
            lhs_batch_dims.len(),
            rhs_batch_dims.len(),
        ))
        .bt());
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
            return Err(fuel_core_types::Error::Msg(format!(
                "matmul_f32: batch dim {i} disallowed combination (lhs={la}, rhs={ra}); \
                 must be equal or GQA-divisible (lhs > rhs && lhs % rhs == 0)",
            ))
            .bt());
        }
    }
    let elem = std::mem::size_of::<f32>();
    let lhs_per_batch = m.saturating_mul(k);
    let rhs_per_batch = k.saturating_mul(n);
    let out_per_batch = m.saturating_mul(n);
    let lhs_batch_count: usize = lhs_batch_dims.iter().product::<usize>().max(1);
    let rhs_batch_count: usize = rhs_batch_dims.iter().product::<usize>().max(1);
    let need_lhs = lhs_batch_count.saturating_mul(lhs_per_batch).saturating_mul(elem);
    let need_rhs = rhs_batch_count.saturating_mul(rhs_per_batch).saturating_mul(elem);
    let need_out = lhs_batch_count.saturating_mul(out_per_batch).saturating_mul(elem);
    if lhs.len_bytes() != need_lhs {
        return Err(fuel_core_types::Error::Msg(format!(
            "matmul_f32: lhs bytes={} doesn't match shape {:?} + [{m}, {k}] (f32)",
            lhs.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    if rhs.len_bytes() != need_rhs {
        return Err(fuel_core_types::Error::Msg(format!(
            "matmul_f32: rhs bytes={} doesn't match shape {:?} + [{k}, {n}] (f32)",
            rhs.len_bytes(),
            rhs_batch_dims,
        ))
        .bt());
    }
    let device = lhs.device().clone();
    if rhs.device().id() != device.id() {
        return Err(fuel_core_types::Error::Msg(
            "matmul_f32: lhs and rhs are on different CUDA devices; cross-device \
             matmul is the caller's responsibility (insert Op::Move first)"
                .to_string(),
        )
        .bt());
    }
    if need_out == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out = device.alloc_zeros::<u8>(need_out)?;

    use baracuda_cublas::{cublasComputeType_t, cudaDataType_t, Op};
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let alpha_ptr = (&alpha) as *const f32 as *const std::ffi::c_void;
    let beta_ptr = (&beta) as *const f32 as *const std::ffi::c_void;
    // cuBLAS A = our rhs (logical [k, n] row-major, viewed col-major
    // as [n, k]). lda = n. cuBLAS B = our lhs (logical [m, k] row-
    // major, viewed col-major as [k, m]). ldb = k. cuBLAS C = our out
    // (logical [m, n] row-major, viewed col-major as [n, m]). ldc = n.
    let lda = n.max(1) as i32;
    let ldb = k.max(1) as i32;
    let ldc = n.max(1) as i32;
    let cublas = device.cublas_handle();
    let compute_type = cublasComputeType_t::Compute32F;
    let lhs_base = lhs.buffer().as_raw().0;
    let rhs_base = rhs.buffer().as_raw().0;
    let out_base = out.as_raw().0;

    let all_equal = n_rep.iter().all(|&r| r == 1);
    if all_equal {
        let a_ptr = rhs_base as *const std::ffi::c_void;
        let b_ptr = lhs_base as *const std::ffi::c_void;
        let c_ptr = out_base as *mut std::ffi::c_void;
        // SAFETY: pointers are valid for the call (lhs, rhs, out
        // outlive the launch); shape parameters match byte-length
        // validation above. Sync follows so result is observable on
        // return (sync KernelRef per locked design decision).
        unsafe {
            baracuda_cublas::gemm_strided_batched_ex(
                &cublas.0,
                Op::N,
                Op::N,
                n as i32,                       // cuBLAS m
                m as i32,                       // cuBLAS n
                k as i32,                       // cuBLAS k
                alpha_ptr,
                a_ptr,                          // cuBLAS A = our rhs
                cudaDataType_t::R_32F,
                lda,
                rhs_per_batch as i64,           // stride_a
                b_ptr,                          // cuBLAS B = our lhs
                cudaDataType_t::R_32F,
                ldb,
                lhs_per_batch as i64,           // stride_b
                beta_ptr,
                c_ptr,                          // cuBLAS C = our out
                cudaDataType_t::R_32F,
                ldc,
                out_per_batch as i64,           // stride_c
                lhs_batch_count as i32,
                compute_type,
                99_i32,                         // CUBLAS_GEMM_DEFAULT
            )
        }
        .map_err(|e| fuel_core_types::Error::Msg(format!("cublas gemm: {e:?}")).bt())?;
    } else {
        // GQA path: walk lhs flat batch index in row-major, decode to
        // multi-index, encode rhs flat batch index via per-axis n_rep
        // mapping, single gemm per batch. Mirrors CPU's per-batch
        // loop in `fuel-cpu-backend::byte_kernels::matmul_f32`.
        let mut lhs_multi = vec![0usize; batch_rank];
        for b in 0..lhs_batch_count {
            let mut rem = b;
            for d in (0..batch_rank).rev() {
                let s = lhs_batch_dims[d];
                lhs_multi[d] = rem % s;
                rem /= s;
            }
            let mut rhs_b = 0usize;
            for d in 0..batch_rank {
                rhs_b = rhs_b * rhs_batch_dims[d] + (lhs_multi[d] / n_rep[d]);
            }
            let lhs_off_bytes = (b * lhs_per_batch * elem) as u64;
            let rhs_off_bytes = (rhs_b * rhs_per_batch * elem) as u64;
            let out_off_bytes = (b * out_per_batch * elem) as u64;
            let a_ptr = (rhs_base + rhs_off_bytes) as *const std::ffi::c_void;
            let b_ptr = (lhs_base + lhs_off_bytes) as *const std::ffi::c_void;
            let c_ptr = (out_base + out_off_bytes) as *mut std::ffi::c_void;
            // SAFETY: pointer offsets are within validated byte ranges
            // (b < lhs_batch_count and rhs_b < rhs_batch_count by
            // construction; per-batch byte counts verified above).
            unsafe {
                baracuda_cublas::gemm_ex(
                    &cublas.0,
                    Op::N,
                    Op::N,
                    n as i32,
                    m as i32,
                    k as i32,
                    alpha_ptr,
                    a_ptr,
                    cudaDataType_t::R_32F,
                    lda,
                    b_ptr,
                    cudaDataType_t::R_32F,
                    ldb,
                    beta_ptr,
                    c_ptr,
                    cudaDataType_t::R_32F,
                    ldc,
                    compute_type,
                    99_i32,
                )
            }
            .map_err(|e| fuel_core_types::Error::Msg(format!("cublas gemm: {e:?}")).bt())?;
        }
    }
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, need_out))
}

/// Element-wise affine `y = mul * x + add` for one F32 `CudaStorageBytes`.
/// Backs `OpKind::Affine` (and is the building block `mean_reduce_f32`
/// uses for its post-sum scaling step). The legacy `Affine` struct
/// in `storage.rs` provides the same math; this is the byte-level
/// path through the unified binding table.
///
/// Allocates a fresh output buffer (the affine kernel's signature
/// has separate `inp` and `out` pointers, and the wrapper takes
/// `&out` mutably so it can't alias `inp`). Output size matches
/// input size.
pub fn affine_f32(src: &CudaStorageBytes, mul: f32, add: f32) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if src.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "affine_f32: src.len_bytes={} not a multiple of f32 size",
            src.len_bytes(),
        ))
        .bt());
    }
    let elem_count = src.len_bytes() / elem;
    let device = src.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let mut out = device.alloc_zeros::<u8>(src.len_bytes())?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let func = device.get_or_load_func("affine_f32", &kernels::AFFINE)?;
    // Affine kernel signature: (numel, num_dims, info, inp, out, mul, add).
    let dims_strides: SlicePtrOrNull<usize> = SlicePtrOrNull::Null;
    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, 1_usize); // ndims (ignored on the contiguous path)
    dims_strides.builder_arg(&mut builder);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    barg!(builder, mul);
    barg!(builder, add);
    // SAFETY: kernel signature matches the args above — same shape
    // as the legacy `Map1::f` for `Affine`, just on byte buffers.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out),
        device,
        src.len_bytes(),
    ))
}

/// Element-wise dtype cast. Element count is preserved; the byte
/// length of the output differs from the input when source and
/// destination have different `size_in_bytes`. Picks the
/// `cast_<src>_<dst>` kernel from `fuel_cuda_kernels::CAST` based on
/// the dtype pair; missing-kernel cases (e.g. an FP8 cast on a GPU
/// where FP8 wasn't compiled in) surface at kernel-load time with the
/// kernel name in the error.
///
/// Sub-byte source/destination types (`F4`/`F6E2M3`/`F6E3M2`) are not
/// supported — they would need a packed-bytes representation that the
/// unified storage doesn't currently expose. Sub-byte arrives as a
/// follow-up if/when those dtypes become load-bearing.
pub fn cast(
    src: &CudaStorageBytes,
    src_dtype: DType,
    dst_dtype: DType,
) -> Result<CudaStorageBytes> {
    let src_elem_size = src_dtype.size_in_bytes();
    let dst_elem_size = dst_dtype.size_in_bytes();
    if src_elem_size == 0 || dst_elem_size == 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "cast({src_dtype:?} -> {dst_dtype:?}): sub-byte dtypes \
             are not supported through the unified path"
        ))
        .bt());
    }
    if src.len_bytes() % src_elem_size != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "cast({src_dtype:?} -> {dst_dtype:?}): src.len_bytes={} \
             not a multiple of src elem size {}",
            src.len_bytes(),
            src_elem_size,
        ))
        .bt());
    }
    let elem_count = src.len_bytes() / src_elem_size;
    let device = src.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_bytes = elem_count * dst_elem_size;
    let mut out = device.alloc_zeros::<u8>(out_bytes)?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let kernel_name = format!("cast_{}_{}", src_dtype.as_str(), dst_dtype.as_str());
    let func = device.get_or_load_func(&kernel_name, &kernels::CAST)?;
    // Cast kernel signature: (numel, num_dims, info, inp, out).
    // info=null selects the contiguous fast path.
    let dims_strides: SlicePtrOrNull<usize> = SlicePtrOrNull::Null;
    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, 1_usize); // ndims (ignored on the contiguous path)
    dims_strides.builder_arg(&mut builder);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — same shape as
    // the legacy `to_dtype` impl in `storage.rs::CudaStorage`.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out),
        device,
        out_bytes,
    ))
}

/// Shared launch path for F32 elementwise binary ops. Validates equal
/// byte lengths, allocates a fresh device buffer, launches the
/// fuel-cuda-kernels BINARY function identified by `kernel_name`,
/// and returns the result. Synchronizes the default stream so the
/// result is observable on return (sync KernelRef per locked design
/// decision).
fn binary_elementwise_f32(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    kernel_name: &'static str,
) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if lhs.len_bytes() != rhs.len_bytes() {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: lhs.len_bytes={} != rhs.len_bytes={}",
            lhs.len_bytes(),
            rhs.len_bytes(),
        ))
        .bt());
    }
    if lhs.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: lhs.len_bytes={} not a multiple of f32 size",
            lhs.len_bytes(),
        ))
        .bt());
    }
    let elem_count = lhs.len_bytes() / elem;
    let device = lhs.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let mut out = device.alloc_zeros::<u8>(lhs.len_bytes())?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let func = device.get_or_load_func(kernel_name, &kernels::BINARY)?;
    let dims_strides: SlicePtrOrNull<usize> = SlicePtrOrNull::Null;
    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, 1_usize); // ndims (ignored on the contiguous path)
    dims_strides.builder_arg(&mut builder);
    builder.arg(lhs.buffer());
    builder.arg(rhs.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — same shape as
    // the existing legacy `Map2::f` for `BinaryOpT`, just on byte
    // buffers. Kernel-side validation is the same.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out),
        device,
        lhs.len_bytes(),
    ))
}

/// Shared launch path for F32 elementwise unary ops. Mirrors
/// [`binary_elementwise_f32`] but with a single input. The
/// fuel-cuda-kernels UNARY function signature is
/// `(elem_count, ndims, dims_strides_or_null, src, out)` — same as
/// the legacy `Map1::f` for `UnaryOpT`. A null `dims_strides_or_null`
/// selects the contiguous fast path; auto-Contiguize guarantees
/// that on the unified path.
fn unary_elementwise_f32(
    src: &CudaStorageBytes,
    kernel_name: &'static str,
) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if src.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: src.len_bytes={} not a multiple of f32 size",
            src.len_bytes(),
        ))
        .bt());
    }
    let elem_count = src.len_bytes() / elem;
    let device = src.device().clone();
    if elem_count == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let mut out = device.alloc_zeros::<u8>(src.len_bytes())?;
    let cfg = LaunchConfig::for_num_elems(elem_count as u32);
    let func = device.get_or_load_func(kernel_name, &kernels::UNARY)?;
    let dims_strides: SlicePtrOrNull<usize> = SlicePtrOrNull::Null;
    let mut builder = func.builder();
    barg!(builder, elem_count);
    barg!(builder, 1_usize); // ndims (ignored on the contiguous path)
    dims_strides.builder_arg(&mut builder);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — same shape as
    // the legacy `Map1::f` for `UnaryOpT`, just on byte buffers.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out),
        device,
        src.len_bytes(),
    ))
}

/// Shared launch path for F32 reductions (Sum/Max/Min). Mirrors the
/// legacy `Map1Any` for `FastReduce` (storage.rs:317): reorders dims
/// so reduced axes come last, builds a `[dims | strides]` device
/// buffer, and launches with `grid_dim = dst_el` and `block_dim =
/// next_power_of_two(min(1024, el_to_sum_per_block))`. The kernel
/// signature is `(src_numel, el_to_sum_per_block, num_dims, info,
/// src, dst)`.
///
/// Auto-Contiguize guarantees the input is contiguous before this
/// runs, so `input_layout.stride()` is the row-major stride. The
/// strides side-band is still passed because the kernel uses
/// `get_strided_index` unconditionally.
fn reduce_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    reduce_dims: &[usize],
    kernel_name: &'static str,
) -> Result<CudaStorageBytes> {
    let elem = std::mem::size_of::<f32>();
    if src.len_bytes() % elem != 0 {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: src.len_bytes={} not a multiple of f32 size",
            src.len_bytes(),
        ))
        .bt());
    }
    let src_dims = input_layout.shape().dims();
    let src_stride = input_layout.stride();
    let src_el: usize = src_dims.iter().product();
    if src_el * elem != src.len_bytes() {
        return Err(fuel_core_types::Error::Msg(format!(
            "{kernel_name}: src element count {} (from layout shape {:?}) \
             disagrees with byte length {} / sizeof(f32)",
            src_el,
            src_dims,
            src.len_bytes(),
        ))
        .bt());
    }

    // Reorder dims/strides so the reduced axes are at the end —
    // matches the legacy `FastReduce::f` precondition that the
    // kernel iterates over the last `el_to_sum_per_block` elements
    // per block.
    let mut dims = Vec::with_capacity(src_dims.len());
    let mut stride = Vec::with_capacity(src_dims.len());
    let mut dst_el: usize = 1;
    for (dim_idx, &d) in src_dims.iter().enumerate() {
        if !reduce_dims.contains(&dim_idx) {
            dst_el *= d;
            dims.push(d);
            stride.push(src_stride[dim_idx]);
        }
    }
    for &dim_idx in reduce_dims.iter() {
        dims.push(src_dims[dim_idx]);
        stride.push(src_stride[dim_idx]);
    }

    let dst_bytes = dst_el * elem;
    let device = src.device().clone();
    if src_el == 0 || dst_el == 0 {
        return CudaStorageBytes::alloc(&device, dst_bytes);
    }
    let el_to_sum_per_block = src_el / dst_el;
    // Pow-of-two block size so the in-block parallel reduction's
    // halving loop is well-defined (matches legacy).
    let block_dim = usize::min(1024, el_to_sum_per_block).next_power_of_two();
    let cfg = LaunchConfig {
        grid_dim: (dst_el as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut out = device.alloc_zeros::<u8>(dst_bytes)?;
    let ds = device.clone_htod(&[dims.as_slice(), stride.as_slice()].concat())?;
    let func = device.get_or_load_func(kernel_name, &kernels::REDUCE)?;
    let mut builder = func.builder();
    barg!(builder, src_el);
    barg!(builder, el_to_sum_per_block);
    barg!(builder, src_dims.len());
    builder.arg(&ds);
    builder.arg(src.buffer());
    builder.arg(&mut out);
    // SAFETY: kernel signature matches the args above — same shape as
    // the legacy `FastReduce::f`, just on byte buffers.
    unsafe { builder.launch(cfg) }.w()?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out), device, dst_bytes))
}
