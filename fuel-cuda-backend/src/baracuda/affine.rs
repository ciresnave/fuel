//! Affine kernels from `baracuda-kernels-sys` — `y = a * x + b` with
//! scalar `(a, b)`.
//!
//! ## Contig vs strided dispatch
//!
//! Baracuda alpha.31 ships `<sym>_strided_run` siblings for every
//! affine dtype. The driver picks per-call via
//! `is_contiguous_zero_offset(layout)`; the contig fast path uses the
//! existing `<sym>_run` ABI and the strided path passes
//! `(rank, shape, stride_x, stride_y)` over the input's true rank-N
//! layout.
//!
//! ## Per-dtype scalar typing
//!
//! Baracuda's affine signature carries the scalar params in the storage
//! dtype's natural arithmetic type:
//!
//! | Storage dtype | `a, b` C type | Note                            |
//! |---------------|---------------|---------------------------------|
//! | `f32`         | `f32`         |                                 |
//! | `f64`         | `f64`         |                                 |
//! | `f16`         | `f32`         | f32 compute, f16 storage        |
//! | `bf16`        | `f32`         | f32 compute, bf16 storage       |
//! | `i32`         | `i32`         |                                 |
//! | `i64`         | `i64`         |                                 |
//! | `u8`          | `u8`          |                                 |
//!
//! Fuel's `OpParams::Affine { mul: f64, add: f64 }` always holds f64
//! params. The wrappers cast to the per-dtype scalar type at the FFI
//! boundary.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::{Layout, Result, Shape};

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

/// Picks the source layout, falling back to a rank-1 contig view of
/// the byte buffer when the caller doesn't pass one (legacy call
/// sites that haven't migrated to layout-passing).
fn resolve_layout<'a>(
    src: &CudaStorageBytes,
    src_layout: Option<&'a Layout>,
    dtype_size_bytes: usize,
    storage: &'a mut Option<Layout>,
) -> &'a Layout {
    if let Some(l) = src_layout {
        return l;
    }
    let elems = src.len_bytes() / dtype_size_bytes.max(1);
    *storage = Some(Layout::contiguous(Shape::from_dims(&[elems])));
    storage.as_ref().unwrap()
}

fn is_contiguous_zero_offset(layout: &Layout) -> bool {
    layout.start_offset() == 0 && layout.is_contiguous()
}

/// Build rank-N `shape` + `stride_x` + contig `stride_y` arrays for
/// the strided FFI. `stride_y` is the contig stride over `dims` since
/// baracuda's output is freshly allocated contig.
fn build_strided_args(
    layout: &Layout,
    op_label: &'static str,
) -> Result<(Vec<i32>, Vec<i64>, Vec<i64>)> {
    let dims = layout.shape().dims();
    let rank = dims.len();
    if rank == 0 {
        return Err(fuel_ir::Error::Msg(
            format!("{op_label}: rank-0 input not supported"),
        ).bt());
    }
    let mut shape_i32: Vec<i32> = Vec::with_capacity(rank);
    for (i, &d) in dims.iter().enumerate() {
        shape_i32.push(i32::try_from(d).map_err(|_| {
            fuel_ir::Error::cuda(
                crate::error::CudaError::BaracudaShapeOverflow {
                    op: op_label, dim_index: i, dim_value: d,
                },
            )
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

/// Generate one per-dtype affine wrapper that picks contig vs strided
/// at the FFI boundary. The macro expands per (dtype-name, scalar-type,
/// contig-fn-sym, strided-fn-sym, dtype-size).
macro_rules! affine_kernel {
    ($name:ident, $scalar:ty, $contig_sym:ident, $strided_sym:ident, $dtype_size:expr, $op_label:expr) => {
        pub fn $name(
            src: &CudaStorageBytes,
            src_layout: Option<&Layout>,
            mul: $scalar,
            add: $scalar,
        ) -> Result<CudaStorageBytes> {
            let op_label = $op_label;
            let device = src.device().clone();
            let mut owned: Option<Layout> = None;
            let layout = resolve_layout(src, src_layout, $dtype_size, &mut owned);
            let numel = layout.shape().elem_count() as i64;
            if numel == 0 {
                return CudaStorageBytes::alloc(&device, 0);
            }
            let out_bytes = (numel as usize) * $dtype_size;
            let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
            let stream = device.stream().as_raw() as *mut std::ffi::c_void;
            let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
            let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;
            let scratch = Workspace::alloc(&device, 0)?;

            let status = if is_contiguous_zero_offset(layout) {
                // SAFETY: pointers + numel validated; scratch null/0.
                unsafe {
                    sys::$contig_sym(
                        numel, x_ptr, y_ptr, mul, add,
                        scratch.as_raw(), scratch.bytes(), stream,
                    )
                }
            } else {
                let (shape_i32, stride_x, stride_y) = build_strided_args(layout, op_label)?;
                let rank = shape_i32.len() as i32;
                // SAFETY: shape/stride buffers owned through the call.
                unsafe {
                    sys::$strided_sym(
                        numel,
                        rank,
                        shape_i32.as_ptr(),
                        stride_x.as_ptr(),
                        stride_y.as_ptr(),
                        x_ptr, y_ptr, mul, add,
                        scratch.as_raw(), scratch.bytes(), stream,
                    )
                }
            };
            check(status, op_label)?;
            Ok(CudaStorageBytes::from_parts(
                Arc::new(out_buf), device, out_bytes,
            ))
        }
    };
}

affine_kernel!(
    affine_f32, f32,
    baracuda_kernels_affine_f32_run,
    baracuda_kernels_affine_f32_strided_run,
    4, "affine_f32"
);
affine_kernel!(
    affine_f64, f64,
    baracuda_kernels_affine_f64_run,
    baracuda_kernels_affine_f64_strided_run,
    8, "affine_f64"
);
affine_kernel!(
    affine_f16, f32,
    baracuda_kernels_affine_f16_run,
    baracuda_kernels_affine_f16_strided_run,
    2, "affine_f16"
);
affine_kernel!(
    affine_bf16, f32,
    baracuda_kernels_affine_bf16_run,
    baracuda_kernels_affine_bf16_strided_run,
    2, "affine_bf16"
);
affine_kernel!(
    affine_i32, i32,
    baracuda_kernels_affine_i32_run,
    baracuda_kernels_affine_i32_strided_run,
    4, "affine_i32"
);
affine_kernel!(
    affine_i64, i64,
    baracuda_kernels_affine_i64_run,
    baracuda_kernels_affine_i64_strided_run,
    8, "affine_i64"
);
affine_kernel!(
    affine_u8, u8,
    baracuda_kernels_affine_u8_run,
    baracuda_kernels_affine_u8_strided_run,
    1, "affine_u8"
);

/// In-place affine on CUDA via baracuda's
/// `baracuda_kernels_affine_inplace_*_run`. Single-pointer ABI:
/// `y = mul * y + offset`. As of alpha.62 all 4 FP dtypes ship
/// (f32, f64, bf16, f16); the half-precision variants pivot scalars
/// through f32 at the FFI boundary (matching the forward
/// `affine_{bf16,f16}_run` convention).
///
/// No strided in-place variant exists in baracuda. The executor's
/// `WorkItemKind::InplaceKernel` arm rejects strided targets up front
/// (contig + zero-offset only), so callers see a clear error rather
/// than a kernel mismatch.
macro_rules! affine_inplace_kernel {
    ($name:ident, $scalar:ty, $sym:ident, $dtype_size:expr, $op_label:expr) => {
        pub fn $name(
            target: &mut CudaStorageBytes,
            mul: $scalar,
            add: $scalar,
        ) -> Result<()> {
            let op_label = $op_label;
            let device = target.device().clone();
            let numel = (target.len_bytes() / $dtype_size) as i64;
            if numel == 0 {
                return Ok(());
            }
            let stream = device.stream().as_raw() as *mut std::ffi::c_void;
            let y_ptr = target.buffer().as_raw().0 as *mut std::ffi::c_void;
            let scratch = Workspace::alloc(&device, 0)?;
            // SAFETY: y_ptr valid for `numel * dtype_size` bytes;
            // scratch null/0; stream comes from baracuda driver.
            let status = unsafe {
                sys::$sym(
                    numel,
                    mul,
                    add,
                    y_ptr,
                    scratch.as_raw(),
                    scratch.bytes(),
                    stream,
                )
            };
            check(status, op_label)?;
            Ok(())
        }
    };
}

affine_inplace_kernel!(
    affine_inplace_f32, f32,
    baracuda_kernels_affine_inplace_f32_run,
    4, "affine_inplace_f32"
);
affine_inplace_kernel!(
    affine_inplace_f64, f64,
    baracuda_kernels_affine_inplace_f64_run,
    8, "affine_inplace_f64"
);
// alpha.61 added bf16 + f16 in response to Fuel's ask
// (docs/baracuda-ask-inplace-ops-2026-05-30.md Item 1). Scalars
// pivot through f32; storage stays at the half-precision dtype.
affine_inplace_kernel!(
    affine_inplace_bf16, f32,
    baracuda_kernels_affine_inplace_bf16_run,
    2, "affine_inplace_bf16"
);
affine_inplace_kernel!(
    affine_inplace_f16, f32,
    baracuda_kernels_affine_inplace_f16_run,
    2, "affine_inplace_f16"
);
