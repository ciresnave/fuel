//! Concat kernels from `baracuda-kernels-sys` — `concat2` (binary
//! concatenate along one dim) over `{F32, F64, F16, BF16}`. Baracuda
//! only ships the binary form; N-ary concat (Fuel's `OpKind::Concat`)
//! chains N-1 binary calls.
//!
//! ## Shape spec
//!
//! Concat reshapes input/output into rank-3 `[outer_count, dim_size,
//! inner_count]` where:
//! - `outer_count` = product of dims before the concat dim
//! - `inner_count` = product of dims after the concat dim
//! - `dim_size` = a's or b's size along the concat dim
//!
//! Output shape: `[outer_count, sum_dim_sizes, inner_count]`. baracuda's
//! kernel reads `concat_dim` (set to 1 here for the rank-3 reshape) and
//! `split_offset` = `a.shape[concat_dim]`.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_core_types::{Error, Result};

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

type Concat2Run = unsafe extern "C" fn(
    output_numel: i64,
    rank: i32,
    output_shape: *const i32,
    concat_dim: i32,
    split_offset: i32,
    stride_a: *const i64,
    stride_b: *const i64,
    stride_y: *const i64,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Run a single binary concat: `y = cat(a, b)` along the (rank-3
/// reshaped) middle dim with sizes `a_dim` and `b_dim`. The rank-3
/// reshape lets one kernel cover Fuel's arbitrary-rank Concat by
/// flattening before/after the concat dim.
fn concat2_run(
    a: &CudaStorageBytes,
    b: &CudaStorageBytes,
    outer_count: usize,
    a_dim: usize,
    b_dim: usize,
    inner_count: usize,
    kernel: Concat2Run,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    let device = a.device().clone();
    let out_dim = a_dim + b_dim;
    let out_numel = outer_count * out_dim * inner_count;
    let out_bytes = out_numel * dtype_size_bytes;
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let i32_or = |dim_index: usize, dim_value: usize| -> Result<i32> {
        i32::try_from(dim_value).map_err(|_| {
            Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label,
                dim_index,
                dim_value,
            })
        })
    };

    let oc = i32_or(0, outer_count)?;
    let out_d = i32_or(1, out_dim)?;
    let ic = i32_or(2, inner_count)?;
    let split = i32_or(3, a_dim)?;

    // Output shape (rank-3): [outer_count, out_dim, inner_count].
    let output_shape: [i32; 3] = [oc, out_d, ic];
    // Per-input strides (row-major contig for both input shapes):
    let stride_a: [i64; 3] = [
        (a_dim * inner_count) as i64,
        inner_count as i64,
        1,
    ];
    let stride_b: [i64; 3] = [
        (b_dim * inner_count) as i64,
        inner_count as i64,
        1,
    ];
    // Output stride: row-major contig over [outer_count, out_dim, inner_count].
    let stride_y: [i64; 3] = [
        (out_dim * inner_count) as i64,
        inner_count as i64,
        1,
    ];

    let a_ptr = a.buffer().as_raw().0 as *const std::ffi::c_void;
    let b_ptr = b.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    // SAFETY: pointers + extents validated above; stream lives on the
    // device for the call's duration; workspace null/0 (no scratch).
    let status = unsafe {
        kernel(
            out_numel as i64,
            3,
            output_shape.as_ptr(),
            1, // concat along the middle dim (the only non-flat one)
            split,
            stride_a.as_ptr(),
            stride_b.as_ptr(),
            stride_y.as_ptr(),
            a_ptr,
            b_ptr,
            y_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    device.synchronize()?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        out_bytes,
    ))
}

/// N-ary concat via N-1 chained `concat2` calls. The first call
/// produces `cat(inputs[0], inputs[1])`; each subsequent call appends
/// one more input. Intermediate buffers are dropped when their
/// reference count hits zero (i.e. immediately after they're consumed
/// by the next concat2).
///
/// `input_dim_sizes` carries the per-input size along the concat dim
/// (length N matches `inputs.len()`).
fn concat_n_chain(
    inputs: &[&CudaStorageBytes],
    outer_count: usize,
    input_dim_sizes: &[usize],
    inner_count: usize,
    kernel: Concat2Run,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    if inputs.len() != input_dim_sizes.len() {
        return Err(Error::Msg(format!(
            "{op_label}: inputs.len()={} != input_dim_sizes.len()={}",
            inputs.len(),
            input_dim_sizes.len(),
        ))
        .bt());
    }
    match inputs.len() {
        0 => Err(Error::Msg(format!("{op_label}: zero inputs")).bt()),
        1 => {
            // Single-input concat is the identity. Clone via to_cpu +
            // upload? No — produce a fresh buffer that's a byte copy.
            let device = inputs[0].device().clone();
            let host = inputs[0].to_cpu_bytes()?;
            CudaStorageBytes::from_cpu_bytes(&device, &host)
        }
        _ => {
            // Pair off: acc = inputs[0]; for i in 1..N, acc = concat2(acc, inputs[i]).
            let mut acc = concat2_run(
                inputs[0],
                inputs[1],
                outer_count,
                input_dim_sizes[0],
                input_dim_sizes[1],
                inner_count,
                kernel,
                op_label,
                dtype_size_bytes,
            )?;
            let mut acc_dim = input_dim_sizes[0] + input_dim_sizes[1];
            for i in 2..inputs.len() {
                acc = concat2_run(
                    &acc,
                    inputs[i],
                    outer_count,
                    acc_dim,
                    input_dim_sizes[i],
                    inner_count,
                    kernel,
                    op_label,
                    dtype_size_bytes,
                )?;
                acc_dim += input_dim_sizes[i];
            }
            Ok(acc)
        }
    }
}

macro_rules! concat_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` — N-ary concat via N-1 chained concat2 calls.")]
            pub fn $name(
                inputs: &[&CudaStorageBytes],
                outer_count: usize,
                input_dim_sizes: &[usize],
                inner_count: usize,
            ) -> Result<CudaStorageBytes> {
                concat_n_chain(
                    inputs,
                    outer_count,
                    input_dim_sizes,
                    inner_count,
                    sys::[<baracuda_kernels_concat2_ $sys_stem _run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

concat_kernel!(concat_f32, f32, 4, "concat_f32");
concat_kernel!(concat_f64, f64, 8, "concat_f64");
concat_kernel!(concat_f16, f16, 2, "concat_f16");
concat_kernel!(concat_bf16, bf16, 2, "concat_bf16");
