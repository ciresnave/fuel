//! Concat kernels from `baracuda-kernels-sys` — `concat2` (binary
//! concatenate along one dim) over `{F32, F64, F16, BF16}`. Baracuda
//! only ships the binary form; N-ary concat (Fuel's `OpKind::Concat`)
//! chains N-1 binary calls.
//!
//! ## Stride-aware dispatch
//!
//! Baracuda's concat2 FFI is shape+stride driven — it always takes
//! `output_shape`, per-input `stride_a` / `stride_b`, and `stride_y`.
//! Earlier Fuel-side wiring synthesized contig rank-3 strides from
//! `(outer, dim, inner)` factoring; the alpha.31 update threads the
//! input's true rank-N layout (shape + strides) through the FFI, with
//! `concat_dim` carrying the actual axis from `OpParams::Concat`.
//!
//! When the dispatch wrapper omits the input layouts (no Fuel layout
//! passed), the wrapper falls back to the historic rank-3 contig
//! reshape using `outer_count, dim_size, inner_count` from
//! `OpParams::Concat`.
//!
//! ## Chained N-ary semantics
//!
//! The first `concat2` consumes inputs[0] and inputs[1] at their
//! actual layouts. Each subsequent iteration consumes `acc` (the
//! fresh contig output of the previous step) plus inputs[i] (its
//! actual layout). The accumulator's layout is therefore always
//! `Layout::contiguous(partial_output_shape)`.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::{Error, Layout, Result, Shape};

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

fn i32_or(
    op_label: &'static str,
    dim_index: usize,
    dim_value: usize,
) -> Result<i32> {
    i32::try_from(dim_value).map_err(|_| {
        Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
            op: op_label, dim_index, dim_value,
        })
    })
}

/// Contig stride array over `dims` (row-major).
fn contig_stride(dims: &[usize]) -> Vec<i64> {
    let rank = dims.len();
    let mut s = vec![1_i64; rank];
    for d in (0..rank.saturating_sub(1)).rev() {
        s[d] = s[d + 1] * dims[d + 1] as i64;
    }
    s
}

/// Convert a layout (or its synthetic equivalent) to the rank-N
/// `(shape_i32, stride_i64)` baracuda's FFI expects.
fn shape_strides_from_layout(
    layout: &Layout,
    op_label: &'static str,
) -> Result<(Vec<i32>, Vec<i64>)> {
    let dims = layout.shape().dims();
    let rank = dims.len();
    let mut shape_i32: Vec<i32> = Vec::with_capacity(rank);
    for (i, &d) in dims.iter().enumerate() {
        shape_i32.push(i32_or(op_label, i, d)?);
    }
    let strides_i64: Vec<i64> = layout.stride().iter().map(|&s| s as i64).collect();
    Ok((shape_i32, strides_i64))
}

/// Single binary concat: `y = cat(a, b)` along `axis`. `a` and `b`
/// must agree on every dim except `axis`. Output layout is contig
/// over the merged shape (axis dim = a_dim + b_dim).
#[allow(clippy::too_many_arguments)]
fn concat2_run(
    a: &CudaStorageBytes,
    a_layout: &Layout,
    b: &CudaStorageBytes,
    b_layout: &Layout,
    axis: usize,
    kernel: Concat2Run,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<(CudaStorageBytes, Layout)> {
    let device = a.device().clone();
    let a_dims = a_layout.shape().dims();
    let b_dims = b_layout.shape().dims();
    if a_dims.len() != b_dims.len() {
        return Err(Error::Msg(format!(
            "{op_label}: input ranks differ (a={}, b={})",
            a_dims.len(), b_dims.len(),
        )).bt());
    }
    let rank = a_dims.len();
    if axis >= rank {
        return Err(Error::Msg(format!(
            "{op_label}: axis {axis} out of range for rank {rank}",
        )).bt());
    }
    let mut out_dims = a_dims.to_vec();
    out_dims[axis] = a_dims[axis] + b_dims[axis];
    let out_numel: usize = out_dims.iter().product();
    let out_bytes = out_numel * dtype_size_bytes;
    let out_layout = Layout::contiguous(Shape::from_dims(&out_dims));
    if out_bytes == 0 {
        return Ok((CudaStorageBytes::alloc(&device, 0)?, out_layout));
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let mut output_shape_i32: Vec<i32> = Vec::with_capacity(rank);
    for (i, &d) in out_dims.iter().enumerate() {
        output_shape_i32.push(i32_or(op_label, i, d)?);
    }
    let (_, stride_a_i64) = shape_strides_from_layout(a_layout, op_label)?;
    let (_, stride_b_i64) = shape_strides_from_layout(b_layout, op_label)?;
    let stride_y_i64 = contig_stride(&out_dims);
    let split = i32_or(op_label, axis, a_dims[axis])?;
    let concat_dim = axis as i32;
    let rank_i32 = rank as i32;

    let a_ptr = a.buffer().as_raw().0 as *const std::ffi::c_void;
    let b_ptr = b.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    // SAFETY: pointers + extents validated above; shape/stride
    // buffers owned through the call; workspace null/0 (no scratch
    // needed for concat).
    let status = unsafe {
        kernel(
            out_numel as i64,
            rank_i32,
            output_shape_i32.as_ptr(),
            concat_dim,
            split,
            stride_a_i64.as_ptr(),
            stride_b_i64.as_ptr(),
            stride_y_i64.as_ptr(),
            a_ptr, b_ptr, y_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    Ok((
        CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes),
        out_layout,
    ))
}

/// Write-into sibling of [`concat2_run`] (CapturedRun executor build-out,
/// 4b-γ): `dest[..] = cat(a, b)` along `axis`, writing into `dest`'s
/// EXISTING device buffer instead of allocating a fresh one — device
/// allocation is illegal inside a CUDA-graph capture scope. Same body as
/// `concat2_run` minus the `device.alloc_zeros`; `dest`'s byte length must
/// already equal the computed `out_bytes` (validated, typed `Error::Msg`
/// on mismatch).
///
/// **N==2 only** — this is the primitive the N-ary write-into concat would
/// need to chain, but that chaining (a scratch-workspace-backed
/// intermediate accumulator) isn't built (no consumer: the real decode
/// graph's only Concat is the rope rotate-half's 2-input `concat(-x2,
/// x1)`). See the per-dtype `concat_*_into` entry points below, which
/// refuse N>2 explicitly rather than silently falling back to allocating.
#[allow(clippy::too_many_arguments)]
pub fn concat2_into(
    a: &CudaStorageBytes,
    a_layout: &Layout,
    b: &CudaStorageBytes,
    b_layout: &Layout,
    axis: usize,
    dest: &CudaStorageBytes,
    kernel: Concat2Run,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<()> {
    let device = a.device().clone();
    let a_dims = a_layout.shape().dims();
    let b_dims = b_layout.shape().dims();
    if a_dims.len() != b_dims.len() {
        return Err(Error::Msg(format!(
            "{op_label}_into: input ranks differ (a={}, b={})",
            a_dims.len(), b_dims.len(),
        )).bt());
    }
    let rank = a_dims.len();
    if axis >= rank {
        return Err(Error::Msg(format!(
            "{op_label}_into: axis {axis} out of range for rank {rank}",
        )).bt());
    }
    let mut out_dims = a_dims.to_vec();
    out_dims[axis] = a_dims[axis] + b_dims[axis];
    let out_numel: usize = out_dims.iter().product();
    let out_bytes = out_numel * dtype_size_bytes;
    if dest.len_bytes() != out_bytes {
        return Err(Error::Msg(format!(
            "{op_label}_into: dest buffer is {} bytes, expected {out_bytes} for output \
             shape {out_dims:?}",
            dest.len_bytes(),
        )).bt());
    }
    if out_bytes == 0 {
        // Empty tensor — no-op write, don't call the kernel.
        return Ok(());
    }
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;

    let mut output_shape_i32: Vec<i32> = Vec::with_capacity(rank);
    for (i, &d) in out_dims.iter().enumerate() {
        output_shape_i32.push(i32_or(op_label, i, d)?);
    }
    let (_, stride_a_i64) = shape_strides_from_layout(a_layout, op_label)?;
    let (_, stride_b_i64) = shape_strides_from_layout(b_layout, op_label)?;
    let stride_y_i64 = contig_stride(&out_dims);
    let split = i32_or(op_label, axis, a_dims[axis])?;
    let concat_dim = axis as i32;
    let rank_i32 = rank as i32;

    let a_ptr = a.buffer().as_raw().0 as *const std::ffi::c_void;
    let b_ptr = b.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = dest.buffer().as_raw().0 as *mut std::ffi::c_void;

    // SAFETY: pointers + extents validated above; shape/stride
    // buffers owned through the call; workspace null/0 (no scratch
    // needed for concat).
    let status = unsafe {
        kernel(
            out_numel as i64,
            rank_i32,
            output_shape_i32.as_ptr(),
            concat_dim,
            split,
            stride_a_i64.as_ptr(),
            stride_b_i64.as_ptr(),
            stride_y_i64.as_ptr(),
            a_ptr, b_ptr, y_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)
}

/// N-ary concat via N-1 chained `concat2` calls. The first call
/// consumes `inputs[0]` + `inputs[1]` at their actual layouts; each
/// subsequent iteration consumes the contig accumulator + `inputs[i]`.
#[allow(clippy::too_many_arguments)]
fn concat_n_chain(
    inputs: &[&CudaStorageBytes],
    input_layouts: &[Layout],
    axis: usize,
    kernel: Concat2Run,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    if inputs.len() != input_layouts.len() {
        return Err(Error::Msg(format!(
            "{op_label}: inputs.len()={} != input_layouts.len()={}",
            inputs.len(), input_layouts.len(),
        )).bt());
    }
    match inputs.len() {
        0 => Err(Error::Msg(format!("{op_label}: zero inputs")).bt()),
        1 => {
            // Single-input concat is the identity — clone via host
            // roundtrip to produce a fresh contig buffer.
            let device = inputs[0].device().clone();
            let host = inputs[0].to_cpu_bytes()?;
            CudaStorageBytes::from_cpu_bytes(&device, &host)
        }
        _ => {
            let (mut acc, mut acc_layout) = concat2_run(
                inputs[0], &input_layouts[0],
                inputs[1], &input_layouts[1],
                axis, kernel, op_label, dtype_size_bytes,
            )?;
            for i in 2..inputs.len() {
                let (next_acc, next_layout) = concat2_run(
                    &acc, &acc_layout,
                    inputs[i], &input_layouts[i],
                    axis, kernel, op_label, dtype_size_bytes,
                )?;
                acc = next_acc;
                acc_layout = next_layout;
            }
            Ok(acc)
        }
    }
}

/// Build per-input contig rank-3 layouts when the dispatch wrapper
/// doesn't supply input layouts (back-compat path for callers using
/// the legacy `(outer, dim, inner)` factoring).
fn synthetic_input_layouts(
    outer_count: usize,
    input_dim_sizes: &[usize],
    inner_count: usize,
) -> Vec<Layout> {
    input_dim_sizes
        .iter()
        .map(|&d| Layout::contiguous(Shape::from_dims(&[outer_count, d, inner_count])))
        .collect()
}

macro_rules! concat_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` — N-ary concat via N-1 chained concat2 calls.")]
            ///
            /// `input_layouts`, when present, lets the kernel walk the
            /// inputs' true rank-N layouts (stride-aware path).
            /// `input_layouts == None` falls back to a synthetic rank-3
            /// `[outer, dim, inner]` contig reshape — matches the
            /// pre-stride-aware behavior bit-for-bit.
            pub fn $name(
                inputs: &[&CudaStorageBytes],
                input_layouts: Option<&[Layout]>,
                axis: usize,
                outer_count: usize,
                input_dim_sizes: &[usize],
                inner_count: usize,
            ) -> Result<CudaStorageBytes> {
                match input_layouts {
                    Some(l) => {
                        if l.len() != inputs.len() {
                            return Err(Error::Msg(format!(
                                "{}: input_layouts.len()={} != inputs.len()={}",
                                $op_label, l.len(), inputs.len(),
                            )).bt());
                        }
                        concat_n_chain(
                            inputs, l, axis,
                            sys::[<baracuda_kernels_concat2_ $sys_stem _run>],
                            $op_label, $dtype_size,
                        )
                    }
                    None => {
                        let layouts_owned = synthetic_input_layouts(
                            outer_count, input_dim_sizes, inner_count,
                        );
                        // Synthetic rank-3 reshape — axis is the middle dim.
                        concat_n_chain(
                            inputs, &layouts_owned, 1,
                            sys::[<baracuda_kernels_concat2_ $sys_stem _run>],
                            $op_label, $dtype_size,
                        )
                    }
                }
            }
        }
    };
}

concat_kernel!(concat_f32, f32, 4, "concat_f32");
concat_kernel!(concat_f64, f64, 8, "concat_f64");
concat_kernel!(concat_f16, f16, 2, "concat_f16");
concat_kernel!(concat_bf16, bf16, 2, "concat_bf16");

/// Write-into sibling of [`concat_kernel!`] (CapturedRun executor
/// build-out, 4b-γ): **only N==2 has a write-into path** — refuses N>2
/// cleanly (typed `Error::Msg`, never silently falls back to allocating)
/// rather than needing the caller to have already checked. N-ary
/// write-into concat is a real future gap if a consumer needs it; not
/// built here (no consumer today — decode's only Concat is the rope
/// rotate-half's 2-input `concat(-x2, x1)`).
macro_rules! concat_kernel_into {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!(
                "Write-into sibling of [`", stringify!($name), "`] — N==2 only. ",
                "N>2 refuses cleanly (`Error::Msg`); see the doc comment on ",
                "`concat_kernel_into!`.",
            )]
            pub fn [<$name _into>](
                inputs: &[&CudaStorageBytes],
                input_layouts: Option<&[Layout]>,
                axis: usize,
                outer_count: usize,
                input_dim_sizes: &[usize],
                inner_count: usize,
                dest: &CudaStorageBytes,
            ) -> Result<()> {
                if inputs.len() != 2 {
                    return Err(Error::Msg(format!(
                        "{}_into: N-ary (N>2) concat has no write-into path yet — only \
                         2-input concat is capture-safe today (got {} inputs)",
                        $op_label, inputs.len(),
                    )).bt());
                }
                let kernel = sys::[<baracuda_kernels_concat2_ $sys_stem _run>];
                match input_layouts {
                    Some(l) => {
                        if l.len() != inputs.len() {
                            return Err(Error::Msg(format!(
                                "{}_into: input_layouts.len()={} != inputs.len()={}",
                                $op_label, l.len(), inputs.len(),
                            )).bt());
                        }
                        concat2_into(
                            inputs[0], &l[0], inputs[1], &l[1], axis, dest,
                            kernel, $op_label, $dtype_size,
                        )
                    }
                    None => {
                        // Synthetic rank-3 reshape — axis is the middle dim.
                        // Matches concat_kernel!'s existing None-path fallback.
                        let layouts_owned = synthetic_input_layouts(
                            outer_count, input_dim_sizes, inner_count,
                        );
                        concat2_into(
                            inputs[0], &layouts_owned[0], inputs[1], &layouts_owned[1], 1, dest,
                            kernel, $op_label, $dtype_size,
                        )
                    }
                }
            }
        }
    };
}

concat_kernel_into!(concat_f32, f32, 4, "concat_f32");
concat_kernel_into!(concat_f64, f64, 8, "concat_f64");
concat_kernel_into!(concat_f16, f16, 2, "concat_f16");
concat_kernel_into!(concat_bf16, bf16, 2, "concat_bf16");

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaDevice;

    fn dev_or_skip() -> Option<CudaDevice> {
        match CudaDevice::new(0) {
            Ok(d) => Some(d),
            Err(e) => {
                eprintln!("no CUDA device; skipping: {e:?}");
                None
            }
        }
    }

    /// `concat2_into` (write-into) must byte-match `concat2_run`
    /// (allocating) for the same 2-input case — CapturedRun executor
    /// build-out, 4b-γ.
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn concat2_into_matches_concat2_run_2input() {
        let Some(dev) = dev_or_skip() else { return };
        let a_data: Vec<f32> = (0..6).map(|i| i as f32).collect(); // [2, 3]
        let b_data: Vec<f32> = (100..106).map(|i| i as f32).collect(); // [2, 3]
        let a_bytes: Vec<u8> = a_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = b_data.iter().flat_map(|f| f.to_le_bytes()).collect();
        let a = CudaStorageBytes::from_cpu_bytes(&dev, &a_bytes).unwrap();
        let b = CudaStorageBytes::from_cpu_bytes(&dev, &b_bytes).unwrap();
        let a_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
        let b_layout = Layout::contiguous(Shape::from_dims(&[2, 3]));
        let kernel = sys::baracuda_kernels_concat2_f32_run;

        let (expect, expect_layout) = concat2_run(
            &a, &a_layout, &b, &b_layout, 1, kernel, "concat_f32_test", 4,
        )
        .unwrap();
        let expect_bytes = expect.to_cpu_bytes().unwrap();
        assert_eq!(expect_layout.shape().dims(), &[2, 6]);

        let dest = CudaStorageBytes::alloc(&dev, expect.len_bytes()).unwrap();
        concat2_into(
            &a, &a_layout, &b, &b_layout, 1, &dest, kernel, "concat_f32_test", 4,
        )
        .unwrap();
        let got_bytes = dest.to_cpu_bytes().unwrap();

        assert_eq!(
            got_bytes, expect_bytes,
            "concat2_into must byte-match concat2_run"
        );
    }

    /// `concat_f32_into` refuses N>2 cleanly (typed error, never silently
    /// falls back to allocating) — the invariant `capture_decode`'s
    /// validation loop depends on.
    #[test]
    #[ignore = "requires a live CUDA device"]
    fn concat_f32_into_refuses_n_ary() {
        let Some(dev) = dev_or_skip() else { return };
        let a = CudaStorageBytes::from_cpu_bytes(&dev, &[0u8; 12]).unwrap();
        let b = CudaStorageBytes::from_cpu_bytes(&dev, &[0u8; 12]).unwrap();
        let c = CudaStorageBytes::from_cpu_bytes(&dev, &[0u8; 12]).unwrap();
        let dest = CudaStorageBytes::alloc(&dev, 36).unwrap();
        let inputs: [&CudaStorageBytes; 3] = [&a, &b, &c];
        let err = concat_f32_into(&inputs, None, 1, 1, &[3, 3, 3], 1, &dest);
        assert!(err.is_err(), "N=3 concat_f32_into must refuse, not silently allocate");
    }
}
