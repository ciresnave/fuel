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


/// Compute the list of input axes that need reducing to align with a
/// broadcast-compatible target shape. Mirrors the CPU
/// `align_reduce_to` validation: target left-pads with 1s to match
/// input rank; an axis is reduced when the padded target dim is 1
/// and the input dim is greater than 1.
fn reduce_dims_from_shapes(
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<Vec<usize>> {
    if output_shape.len() > input_shape.len() {
        return Err(fuel_core_types::Error::Msg(format!(
            "reduce_to: output rank {} exceeds input rank {}",
            output_shape.len(), input_shape.len(),
        )).bt());
    }
    let pad = input_shape.len() - output_shape.len();
    let mut padded = vec![1_usize; pad];
    padded.extend_from_slice(output_shape);
    let mut reduce_dims: Vec<usize> = Vec::new();
    for (axis, (&s, &t)) in input_shape.iter().zip(padded.iter()).enumerate() {
        if t == s {
            // Pass-through axis.
        } else if t == 1 {
            // Axis being reduced. Only push when input dim > 1; if
            // input dim is also 1 the reduction is a no-op and the
            // existing reduce_f32 kernel handles it correctly via the
            // empty-stride single-element path either way.
            if s > 1 {
                reduce_dims.push(axis);
            }
        } else {
            return Err(fuel_core_types::Error::Msg(format!(
                "reduce_to: axis {axis} target {t} must be 1 or input {s}",
            )).bt());
        }
    }
    Ok(reduce_dims)
}

/// Sum-reduce a CUDA F32 tensor to a smaller broadcast-compatible
/// shape. Maps the broadcast-aligned target shape to a list of
/// reduce dims and dispatches through the existing `fast_sum_f32`
/// kernel. The output's byte count matches what the executor
/// pre-allocates for `output_shape` (since the reduced byte count is
/// determined entirely by which dims are reduced, regardless of
/// whether they're dropped or kept as size-1).
///
/// Mirrors the CPU `reduce_sum_to_f32` byte kernel; on CUDA the
/// keepdim form is free because the result bytes are the same as
/// dropping the reduced dim — only the metadata shape differs and
/// is set by the wrapper's pre-allocated output.
pub fn reduce_sum_to_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<CudaStorageBytes> {
    let reduce_dims = reduce_dims_from_shapes(input_shape, output_shape)?;
    // Empty reduce_dims is an "identity reduce_to" — input_shape ==
    // padded(output_shape) on every axis. The reduce kernel handles
    // it correctly (each output element sums over one input element)
    // with the cost of one extra kernel launch; not a hot path so
    // not worth a special-case.
    reduce_f32(src, input_layout, &reduce_dims, "fast_sum_f32")
}

/// Max-reduce a CUDA F32 tensor to a smaller broadcast-compatible
/// shape — the max-symmetric counterpart of [`reduce_sum_to_f32`].
pub fn reduce_max_to_f32(
    src: &CudaStorageBytes,
    input_layout: &Layout,
    input_shape: &[usize],
    output_shape: &[usize],
) -> Result<CudaStorageBytes> {
    let reduce_dims = reduce_dims_from_shapes(input_shape, output_shape)?;
    reduce_f32(src, input_layout, &reduce_dims, "fast_max_f32")
}

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

/// CUTLASS bf16 matmul through the unified byte-storage substrate.
/// Mirrors [`matmul_f32`]'s argument shape but routes through the
/// alpha.13 `LayoutSku::Rrr` SKU in [`crate::cutlass::cutlass_matmul_bf16`]
/// instead of cuBLAS — no row-major-via-col-major transpose trick is
/// needed because CUTLASS Rrr already matches `Op::MatMul`'s
/// activation-row-major @ weight-row-major shape.
///
/// Equal-batch coverage only: per-axis `lhs_batch_dims == rhs_batch_dims`.
/// GQA (per-axis broadcast with `lhs % rhs == 0`) is rejected and the
/// caller should split it upstream — `BatchedGemmPlan` (Phase B6) is
/// the natural follow-on for native batched dispatch.
pub fn matmul_bf16(
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
            "matmul_bf16: batch ranks must match (lhs={}, rhs={})",
            lhs_batch_dims.len(),
            rhs_batch_dims.len(),
        ))
        .bt());
    }
    for (i, (&la, &ra)) in lhs_batch_dims.iter().zip(rhs_batch_dims.iter()).enumerate() {
        if la != ra {
            return Err(fuel_core_types::Error::Msg(format!(
                "matmul_bf16: GQA / broadcast batch (axis {i}: lhs={la}, rhs={ra}) \
                 not supported yet on the CUTLASS bf16 path; split upstream or \
                 wait for BatchedGemmPlan (Phase B6)",
            ))
            .bt());
        }
    }
    let elem = std::mem::size_of::<half::bf16>();
    let lhs_per_batch = m.saturating_mul(k);
    let rhs_per_batch = k.saturating_mul(n);
    let batch_count: usize = lhs_batch_dims.iter().product::<usize>().max(1);
    let need_lhs = batch_count
        .saturating_mul(lhs_per_batch)
        .saturating_mul(elem);
    let need_rhs = batch_count
        .saturating_mul(rhs_per_batch)
        .saturating_mul(elem);
    if lhs.len_bytes() != need_lhs {
        return Err(fuel_core_types::Error::Msg(format!(
            "matmul_bf16: lhs bytes={} doesn't match shape {:?} + [{m}, {k}] (bf16)",
            lhs.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    if rhs.len_bytes() != need_rhs {
        return Err(fuel_core_types::Error::Msg(format!(
            "matmul_bf16: rhs bytes={} doesn't match shape {:?} + [{k}, {n}] (bf16)",
            rhs.len_bytes(),
            rhs_batch_dims,
        ))
        .bt());
    }
    crate::cutlass::cutlass_matmul_bf16(lhs, rhs, batch_count, m, n, k)
}

/// CUTLASS f16 matmul through the unified byte-storage substrate.
/// Mirror of [`matmul_bf16`] at `f16` dtype; routes through the
/// alpha.13 `LayoutSku::Rrr` SKU as well (f16 + bf16 Rrr ship in the
/// same alpha.9 batch). Same equal-batch-only limitation: GQA is
/// rejected pending BatchedGemmPlan (Phase B6).
pub fn matmul_f16(
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
            "matmul_f16: batch ranks must match (lhs={}, rhs={})",
            lhs_batch_dims.len(),
            rhs_batch_dims.len(),
        ))
        .bt());
    }
    for (i, (&la, &ra)) in lhs_batch_dims.iter().zip(rhs_batch_dims.iter()).enumerate() {
        if la != ra {
            return Err(fuel_core_types::Error::Msg(format!(
                "matmul_f16: GQA / broadcast batch (axis {i}: lhs={la}, rhs={ra}) \
                 not supported yet on the CUTLASS f16 path; split upstream or \
                 wait for BatchedGemmPlan (Phase B6)",
            ))
            .bt());
        }
    }
    let elem = std::mem::size_of::<half::f16>();
    let lhs_per_batch = m.saturating_mul(k);
    let rhs_per_batch = k.saturating_mul(n);
    let batch_count: usize = lhs_batch_dims.iter().product::<usize>().max(1);
    let need_lhs = batch_count
        .saturating_mul(lhs_per_batch)
        .saturating_mul(elem);
    let need_rhs = batch_count
        .saturating_mul(rhs_per_batch)
        .saturating_mul(elem);
    if lhs.len_bytes() != need_lhs {
        return Err(fuel_core_types::Error::Msg(format!(
            "matmul_f16: lhs bytes={} doesn't match shape {:?} + [{m}, {k}] (f16)",
            lhs.len_bytes(),
            lhs_batch_dims,
        ))
        .bt());
    }
    if rhs.len_bytes() != need_rhs {
        return Err(fuel_core_types::Error::Msg(format!(
            "matmul_f16: rhs bytes={} doesn't match shape {:?} + [{k}, {n}] (f16)",
            rhs.len_bytes(),
            rhs_batch_dims,
        ))
        .bt());
    }
    crate::cutlass::cutlass_matmul_f16(lhs, rhs, batch_count, m, n, k)
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
    let src_stride = input_layout.stride_unsigned();
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
    let mut dims: Vec<usize> = Vec::with_capacity(src_dims.len());
    let mut stride: Vec<usize> = Vec::with_capacity(src_dims.len());
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
