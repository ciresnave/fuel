//! PowI (integer-exponent power) kernels from `baracuda-kernels-sys`.
//! Forward `y = x^n` and backward `dx = n * x^(n-1) * dy` (alpha.31).
//!
//! ## ABI
//!
//! Forward: `fn run(numel, x, y, p0=exp_f32, p1=unused, ws, ws_b, stream)`.
//! Backward: `fn run(numel, dy, x, dx, p0=exp_f32, p1=unused, ws, ws_b, stream)`.
//!
//! The exponent `n: i32` is reinterpret-cast to `f32` and shipped via
//! `p0` (round-trip exact for `|n| ≤ 2^24`).
//!
//! ## Contig vs strided
//!
//! Baracuda alpha.31 ships `<sym>_strided_run` siblings for both FW
//! and BW. The wrapper picks per-call via
//! `is_contiguous_zero_offset(layout)`.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::{Error, Layout, Result, Shape};

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

type PowiRun = unsafe extern "C" fn(
    numel: i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    p0: f32,
    p1: f32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

type PowiStridedRun = unsafe extern "C" fn(
    numel: i64,
    rank: i32,
    shape: *const i32,
    stride_x: *const i64,
    stride_y: *const i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    p0: f32,
    p1: f32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

type PowiBackwardRun = unsafe extern "C" fn(
    numel: i64,
    dy: *const std::ffi::c_void,
    x: *const std::ffi::c_void,
    dx: *mut std::ffi::c_void,
    p0: f32,
    p1: f32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

type PowiBackwardStridedRun = unsafe extern "C" fn(
    numel: i64,
    rank: i32,
    shape: *const i32,
    stride_x: *const i64,
    stride_dy: *const i64,
    stride_dx: *const i64,
    x: *const std::ffi::c_void,
    dy: *const std::ffi::c_void,
    dx: *mut std::ffi::c_void,
    p0: f32,
    p1: f32,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

fn is_contiguous_zero_offset(layout: &Layout) -> bool {
    layout.start_offset() == 0 && layout.is_contiguous()
}

fn build_strided_args(
    layout: &Layout,
    op_label: &'static str,
) -> Result<(Vec<i32>, Vec<i64>, Vec<i64>)> {
    let dims = layout.shape().dims();
    let rank = dims.len();
    if rank == 0 {
        return Err(Error::Msg(format!("{op_label}: rank-0 input not supported")).bt());
    }
    let mut shape_i32: Vec<i32> = Vec::with_capacity(rank);
    for (i, &d) in dims.iter().enumerate() {
        shape_i32.push(i32::try_from(d).map_err(|_| {
            Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                op: op_label, dim_index: i, dim_value: d,
            })
        })?);
    }
    let stride_x: Vec<i64> = layout.stride().iter().map(|&s| s as i64).collect();
    let stride_y: Vec<i64> = {
        let mut s = vec![1_i64; rank];
        for d in (0..rank.saturating_sub(1)).rev() {
            s[d] = s[d + 1] * dims[d + 1] as i64;
        }
        s
    };
    Ok((shape_i32, stride_x, stride_y))
}

fn powi_run(
    src: &CudaStorageBytes,
    src_layout: Option<&Layout>,
    exp: i32,
    dtype_size_bytes: usize,
    contig: PowiRun,
    strided: PowiStridedRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = src.device().clone();
    let owned_layout;
    let layout = match src_layout {
        Some(l) => l,
        None => {
            let elems = src.len_bytes() / dtype_size_bytes.max(1);
            owned_layout = Layout::contiguous(Shape::from_dims(&[elems]));
            &owned_layout
        }
    };
    let numel = layout.shape().elem_count() as i64;
    if numel == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_bytes = (numel as usize) * dtype_size_bytes;
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;
    let p0 = exp as f32;

    let status = if is_contiguous_zero_offset(layout) {
        // SAFETY: pointers + numel validated above; workspace null/0.
        unsafe {
            contig(numel, x_ptr, y_ptr, p0, 0.0_f32, scratch.as_raw(), scratch.bytes(), stream)
        }
    } else {
        let (shape_i32, stride_x, stride_y) = build_strided_args(layout, op_label)?;
        let rank = shape_i32.len() as i32;
        // SAFETY: shape/stride buffers owned through the call.
        unsafe {
            strided(
                numel, rank, shape_i32.as_ptr(),
                stride_x.as_ptr(), stride_y.as_ptr(),
                x_ptr, y_ptr, p0, 0.0_f32,
                scratch.as_raw(), scratch.bytes(), stream,
            )
        }
    };
    check(status, op_label)?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes))
}

/// Backward `dx = n * x^(n-1) * dy`. The forward kernel's `x` is the
/// original input; `dy` is the upstream gradient; `dx` is the result.
fn powi_backward_run(
    x: &CudaStorageBytes,
    dy: &CudaStorageBytes,
    src_layout: Option<&Layout>,
    exp: i32,
    dtype_size_bytes: usize,
    contig: PowiBackwardRun,
    strided: PowiBackwardStridedRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    let device = x.device().clone();
    let owned_layout;
    let layout = match src_layout {
        Some(l) => l,
        None => {
            let elems = x.len_bytes() / dtype_size_bytes.max(1);
            owned_layout = Layout::contiguous(Shape::from_dims(&[elems]));
            &owned_layout
        }
    };
    let numel = layout.shape().elem_count() as i64;
    if numel == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_bytes = (numel as usize) * dtype_size_bytes;
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = x.buffer().as_raw().0 as *const std::ffi::c_void;
    let dy_ptr = dy.buffer().as_raw().0 as *const std::ffi::c_void;
    let dx_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;
    let p0 = exp as f32;

    let status = if is_contiguous_zero_offset(layout) {
        // SAFETY: pointers + numel validated above; workspace null/0.
        unsafe {
            contig(numel, dy_ptr, x_ptr, dx_ptr, p0, 0.0_f32,
                   scratch.as_raw(), scratch.bytes(), stream)
        }
    } else {
        // Strided BW: x, dy, dx share the same layout (they're all
        // shape `[numel-by-rank]`); pass stride_x for all three since
        // baracuda's signature requires them separately but they
        // match here.
        let (shape_i32, stride_x, stride_y) = build_strided_args(layout, op_label)?;
        let rank = shape_i32.len() as i32;
        // dy reads at input's layout; dx writes contig.
        let stride_dy = stride_x.clone();
        let stride_dx = stride_y.clone();
        // SAFETY: shape/stride buffers owned through the call.
        unsafe {
            strided(
                numel, rank, shape_i32.as_ptr(),
                stride_x.as_ptr(), stride_dy.as_ptr(), stride_dx.as_ptr(),
                x_ptr, dy_ptr, dx_ptr, p0, 0.0_f32,
                scratch.as_raw(), scratch.bytes(), stream,
            )
        }
    };
    check(status, op_label)?;
    Ok(CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes))
}

macro_rules! powi_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` FW kernel (contig + strided dispatch).")]
            pub fn $name(
                src: &CudaStorageBytes,
                src_layout: Option<&Layout>,
                exp: i32,
            ) -> Result<CudaStorageBytes> {
                powi_run(
                    src,
                    src_layout,
                    exp,
                    $dtype_size,
                    sys::[<baracuda_kernels_unary_powi_ $sys_stem _run>],
                    sys::[<baracuda_kernels_unary_powi_ $sys_stem _strided_run>],
                    $op_label,
                )
            }
        }
    };
}

macro_rules! powi_backward_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` BW kernel (contig + strided dispatch).")]
            pub fn $name(
                x: &CudaStorageBytes,
                dy: &CudaStorageBytes,
                src_layout: Option<&Layout>,
                exp: i32,
            ) -> Result<CudaStorageBytes> {
                powi_backward_run(
                    x, dy, src_layout, exp, $dtype_size,
                    sys::[<baracuda_kernels_unary_powi_backward_ $sys_stem _run>],
                    sys::[<baracuda_kernels_unary_powi_backward_ $sys_stem _strided_run>],
                    $op_label,
                )
            }
        }
    };
}

powi_kernel!(powi_f32, f32, 4, "powi_f32");
powi_kernel!(powi_f64, f64, 8, "powi_f64");
powi_kernel!(powi_f16, f16, 2, "powi_f16");
powi_kernel!(powi_bf16, bf16, 2, "powi_bf16");

powi_backward_kernel!(powi_backward_f32, f32, 4, "powi_backward_f32");
powi_backward_kernel!(powi_backward_f64, f64, 8, "powi_backward_f64");
powi_backward_kernel!(powi_backward_f16, f16, 2, "powi_backward_f16");
powi_backward_kernel!(powi_backward_bf16, bf16, 2, "powi_backward_bf16");

/// In-place PowI — reuses the contig forward symbol with same-pointer
/// dispatch for x and y. Safe for elementwise param-unary (no
/// cross-thread aliasing).
fn powi_inplace_run(
    target: &mut CudaStorageBytes,
    exp: i32,
    dtype_size_bytes: usize,
    contig: PowiRun,
    op_label: &'static str,
) -> Result<()> {
    let numel = (target.len_bytes() / dtype_size_bytes.max(1)) as i64;
    if numel == 0 {
        return Ok(());
    }
    let device = target.device().clone();
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let ptr_mut = target.buffer().as_raw().0 as *mut std::ffi::c_void;
    let ptr_const = ptr_mut as *const std::ffi::c_void;
    let p0 = exp as f32;
    // SAFETY: same buffer for x + y is safe for elementwise param-unary
    // kernels (no cross-thread aliasing); pointers / stream / scratch
    // validated above.
    let status = unsafe {
        contig(
            numel, ptr_const, ptr_mut, p0, 0.0_f32,
            scratch.as_raw(), scratch.bytes(), stream,
        )
    };
    check(status, op_label)?;
    Ok(())
}

macro_rules! powi_inplace_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("In-place baracuda `", $op_label, "` kernel — mutates `target`.")]
            pub fn $name(target: &mut CudaStorageBytes, exp: i32) -> Result<()> {
                powi_inplace_run(
                    target, exp, $dtype_size,
                    sys::[<baracuda_kernels_unary_powi_ $sys_stem _run>],
                    $op_label,
                )
            }
        }
    };
}

powi_inplace_kernel!(powi_inplace_f32,  f32,  4, "powi_inplace_f32");
powi_inplace_kernel!(powi_inplace_f64,  f64,  8, "powi_inplace_f64");
powi_inplace_kernel!(powi_inplace_f16,  f16,  2, "powi_inplace_f16");
powi_inplace_kernel!(powi_inplace_bf16, bf16, 2, "powi_inplace_bf16");
