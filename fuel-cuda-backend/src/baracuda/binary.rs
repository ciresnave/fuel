//! Binary elementwise kernels from `baracuda-kernels-sys`.
//!
//! Each kernel has signature
//! `(numel, a, b, y, workspace, workspace_bytes, stream) -> i32`
//! for the contiguous path and
//! `(numel, rank, shape, stride_a, stride_b, stride_y, a, b, y, …)`
//! for the strided path.
//!
//! ## Coverage today
//!
//! - FP math: add / sub / mul / div / maximum / minimum / pow / rem,
//!   across the four-dtype family (F32 / F16 / BF16 / F64).
//!
//! Fuel's `RemElementwise` is contractually PyTorch-style
//! (`a - floor(a/b) * b`, sign follows the divisor) — that maps to
//! baracuda's `binary_mod_*` (Python-style modulo), NOT
//! `binary_remainder_*` (C99 `fmod`, sign follows the dividend).
//!
//! Ops baracuda ships that Fuel doesn't yet have OpKinds for
//! (`atan2`, `copysign`, `hypot`, `fmax`, `fmin`, `nextafter`,
//! `remainder` (fmod-style), `floor_divide`, `lerp`) and `BinaryCmp*`
//! (output dtype Bool — Fuel doesn't have Bool today) are wired up
//! incrementally as Fuel grows those primitive ops.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::{DType, Layout, Result, Shape};

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

type BinaryContigRun = unsafe extern "C" fn(
    numel: i64,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

type BinaryStridedRun = unsafe extern "C" fn(
    numel: i64,
    rank: i32,
    shape: *const i32,
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

/// Three layouts (a, b, output) folded into one stride buffer per
/// tensor. Used by the strided binary path when inputs are
/// broadcast-shaped or non-contiguous.
struct BinaryStrides {
    rank: i32,
    shape: Vec<i32>,
    stride_a: Vec<i64>,
    stride_b: Vec<i64>,
    stride_y: Vec<i64>,
}

impl BinaryStrides {
    fn from(
        a_layout: &Layout,
        b_layout: &Layout,
        y_layout: &Layout,
        op_label: &'static str,
    ) -> Result<Self> {
        let dims = y_layout.shape().dims();
        if a_layout.shape().dims() != dims || b_layout.shape().dims() != dims {
            return Err(fuel_ir::Error::Msg(format!(
                "{op_label}: a / b / y shapes must match after broadcast \
                 (a={:?}, b={:?}, y={:?})",
                a_layout.shape().dims(),
                b_layout.shape().dims(),
                dims,
            ))
            .bt());
        }
        let mut shape = Vec::with_capacity(dims.len());
        for (i, &d) in dims.iter().enumerate() {
            shape.push(i32::try_from(d).map_err(|_| {
                fuel_ir::Error::cuda(crate::error::CudaError::BaracudaShapeOverflow {
                    op: op_label,
                    dim_index: i,
                    dim_value: d,
                })
            })?);
        }
        Ok(Self {
            rank: dims.len() as i32,
            shape,
            stride_a: a_layout.stride().iter().map(|&s| s as i64).collect(),
            stride_b: b_layout.stride().iter().map(|&s| s as i64).collect(),
            stride_y: y_layout.stride().iter().map(|&s| s as i64).collect(),
        })
    }
}

/// Core binary-elementwise driver. Mirrors `unary_run`'s shape but
/// takes two input pointers + their strides. Allocates a fresh output
/// and delegates the launch to [`binary_run_into`].
fn binary_run(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_layout: Option<&Layout>,
    rhs_layout: Option<&Layout>,
    contig_run: BinaryContigRun,
    strided_run: BinaryStridedRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<CudaStorageBytes> {
    // Size the output from lhs (post-broadcast lhs shape == output shape
    // at this layer — the graph inserts explicit Op::BroadcastTo, so both
    // operands already carry the output dims). `binary_run_into` re-derives
    // and validates the same numel against the buffer it's handed.
    let numel = match lhs_layout {
        Some(l) => l.shape().elem_count(),
        None => lhs.len_bytes() / dtype_size_bytes.max(1),
    };
    let out_bytes = numel * dtype_size_bytes;
    let device = lhs.device().clone();
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let out = CudaStorageBytes::from_parts(Arc::new(out_buf), device, out_bytes);
    binary_run_into(
        lhs, rhs, lhs_layout, rhs_layout, &out, contig_run, strided_run, op_label,
        dtype_size_bytes,
    )?;
    Ok(out)
}

/// Write-into-output binary driver (CapturedRun executor build-out).
///
/// Identical elementwise math to [`binary_run`], but writes into the
/// caller-provided `out` buffer instead of allocating one. This is the
/// enabler for the pipelined executor's persistent-output (capture) mode:
/// a FIXED-ADDRESS output buffer is written in place so **no device
/// allocation happens** — mandatory inside a CUDA-graph capture scope,
/// where both alloc and host sync are illegal. Byte-identical result to
/// the alloc-and-return path for a same-sized `out`.
///
/// `out` must already hold at least `numel * dtype_size_bytes` bytes
/// (the executor pre-sizes it from the node's output shape); a smaller
/// buffer is a surfaced error, never an out-of-bounds device write.
#[allow(clippy::too_many_arguments)]
fn binary_run_into(
    lhs: &CudaStorageBytes,
    rhs: &CudaStorageBytes,
    lhs_layout: Option<&Layout>,
    rhs_layout: Option<&Layout>,
    out: &CudaStorageBytes,
    contig_run: BinaryContigRun,
    strided_run: BinaryStridedRun,
    op_label: &'static str,
    dtype_size_bytes: usize,
) -> Result<()> {
    let derived_lhs_layout;
    let derived_rhs_layout;
    let lhs_l = match lhs_layout {
        Some(l) => l,
        None => {
            let elems = lhs.len_bytes() / dtype_size_bytes.max(1);
            derived_lhs_layout = Layout::contiguous(Shape::from_dims(&[elems]));
            &derived_lhs_layout
        }
    };
    let rhs_l = match rhs_layout {
        Some(l) => l,
        None => {
            let elems = rhs.len_bytes() / dtype_size_bytes.max(1);
            derived_rhs_layout = Layout::contiguous(Shape::from_dims(&[elems]));
            &derived_rhs_layout
        }
    };
    let numel: i64 = lhs_l.shape().elem_count() as i64;
    let out_bytes = (numel as usize) * dtype_size_bytes;
    let device = lhs.device().clone();
    if rhs.device().id() != device.id() {
        return Err(fuel_ir::Error::Msg(format!(
            "{op_label}: lhs and rhs on different CUDA devices",
        ))
        .bt());
    }
    if out_bytes == 0 {
        return Ok(());
    }
    if out.len_bytes() < out_bytes {
        return Err(fuel_ir::Error::Msg(format!(
            "{op_label}: write-into output buffer too small ({} < {} bytes)",
            out.len_bytes(),
            out_bytes,
        ))
        .bt());
    }
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let a_ptr = lhs.buffer().as_raw().0 as *const std::ffi::c_void;
    let b_ptr = rhs.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out.buffer().as_raw().0 as *mut std::ffi::c_void;

    let contig = lhs_l.is_contiguous()
        && lhs_l.start_offset() == 0
        && rhs_l.is_contiguous()
        && rhs_l.start_offset() == 0;
    let status = if contig {
        // SAFETY: pointers + lengths validated; stream lives on device;
        // workspace is null because elementwise binary needs none.
        unsafe {
            contig_run(
                numel,
                a_ptr,
                b_ptr,
                y_ptr,
                scratch.as_raw(),
                scratch.bytes(),
                stream,
            )
        }
    } else {
        let out_layout = Layout::contiguous(lhs_l.shape());
        let s = BinaryStrides::from(lhs_l, rhs_l, &out_layout, op_label)?;
        // SAFETY: shape/stride buffers owned by `s`; pointers above.
        unsafe {
            strided_run(
                numel,
                s.rank,
                s.shape.as_ptr(),
                s.stride_a.as_ptr(),
                s.stride_b.as_ptr(),
                s.stride_y.as_ptr(),
                a_ptr,
                b_ptr,
                y_ptr,
                scratch.as_raw(),
                scratch.bytes(),
                stream,
            )
        }
    };
    check(status, op_label)?;
    Ok(())
}

/// Manifest macro for one (kind, dtype) binary entry. Emits BOTH the
/// allocating entry (`$name` → `Result<CudaStorageBytes>`) and its
/// write-into sibling (`$name _into` → writes into a caller-provided
/// `out`, `Result<()>`) — the latter powers the executor's
/// persistent-output CUDA-graph capture mode (no alloc inside capture).
macro_rules! binary_kernel {
    ($name:ident, $sys_stem:ident, $dtype_size:expr, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda binary `", $op_label, "` kernel.")]
            pub fn $name(
                lhs: &CudaStorageBytes,
                rhs: &CudaStorageBytes,
                lhs_layout: Option<&Layout>,
                rhs_layout: Option<&Layout>,
            ) -> Result<CudaStorageBytes> {
                binary_run(
                    lhs,
                    rhs,
                    lhs_layout,
                    rhs_layout,
                    sys::[<baracuda_kernels_binary_ $sys_stem _run>],
                    sys::[<baracuda_kernels_binary_ $sys_stem _strided_run>],
                    $op_label,
                    $dtype_size,
                )
            }

            #[doc = concat!(
                "Write-into-output variant of baracuda binary `", $op_label,
                "` — writes into `out` (no alloc; CapturedRun capture mode)."
            )]
            pub fn [<$name _into>](
                lhs: &CudaStorageBytes,
                rhs: &CudaStorageBytes,
                lhs_layout: Option<&Layout>,
                rhs_layout: Option<&Layout>,
                out: &CudaStorageBytes,
            ) -> Result<()> {
                binary_run_into(
                    lhs,
                    rhs,
                    lhs_layout,
                    rhs_layout,
                    out,
                    sys::[<baracuda_kernels_binary_ $sys_stem _run>],
                    sys::[<baracuda_kernels_binary_ $sys_stem _strided_run>],
                    $op_label,
                    $dtype_size,
                )
            }
        }
    };
}

// ---------------------------------------------------------------------------
// F32 binary kernels
// ---------------------------------------------------------------------------

binary_kernel!(binary_add_f32, add_f32, 4, "binary_add_f32");
binary_kernel!(binary_sub_f32, sub_f32, 4, "binary_sub_f32");
binary_kernel!(binary_mul_f32, mul_f32, 4, "binary_mul_f32");
binary_kernel!(binary_div_f32, div_f32, 4, "binary_div_f32");
binary_kernel!(binary_maximum_f32, maximum_f32, 4, "binary_maximum_f32");
binary_kernel!(binary_minimum_f32, minimum_f32, 4, "binary_minimum_f32");

// Pow: elementwise `pow(a[i], b[i])`, IEEE-754 NaN semantics — matches
// Fuel's `PowElementwise` contract (`powf`/`powf` on CPU).
binary_kernel!(binary_pow_f32, pow_f32, 4, "binary_pow_f32");
// Rem: Fuel-side name is `rem` but the FFI stem is baracuda's `mod`
// (Python-style modulo, sign of divisor) — the semantic match for
// Fuel's PyTorch-convention `RemElementwise`. See the module doc.
binary_kernel!(binary_rem_f32, mod_f32, 4, "binary_mod_f32");

// ---------------------------------------------------------------------------
// F16 / BF16 / F64 binary kernels — mirrors of F32 above
// ---------------------------------------------------------------------------

binary_kernel!(binary_add_f16, add_f16, 2, "binary_add_f16");
binary_kernel!(binary_sub_f16, sub_f16, 2, "binary_sub_f16");
binary_kernel!(binary_mul_f16, mul_f16, 2, "binary_mul_f16");
binary_kernel!(binary_div_f16, div_f16, 2, "binary_div_f16");
binary_kernel!(binary_maximum_f16, maximum_f16, 2, "binary_maximum_f16");
binary_kernel!(binary_minimum_f16, minimum_f16, 2, "binary_minimum_f16");
binary_kernel!(binary_pow_f16, pow_f16, 2, "binary_pow_f16");
binary_kernel!(binary_rem_f16, mod_f16, 2, "binary_mod_f16");

binary_kernel!(binary_add_bf16, add_bf16, 2, "binary_add_bf16");
binary_kernel!(binary_sub_bf16, sub_bf16, 2, "binary_sub_bf16");
binary_kernel!(binary_mul_bf16, mul_bf16, 2, "binary_mul_bf16");
binary_kernel!(binary_div_bf16, div_bf16, 2, "binary_div_bf16");
binary_kernel!(binary_maximum_bf16, maximum_bf16, 2, "binary_maximum_bf16");
binary_kernel!(binary_minimum_bf16, minimum_bf16, 2, "binary_minimum_bf16");
binary_kernel!(binary_pow_bf16, pow_bf16, 2, "binary_pow_bf16");
binary_kernel!(binary_rem_bf16, mod_bf16, 2, "binary_mod_bf16");

binary_kernel!(binary_add_f64, add_f64, 8, "binary_add_f64");
binary_kernel!(binary_sub_f64, sub_f64, 8, "binary_sub_f64");
binary_kernel!(binary_mul_f64, mul_f64, 8, "binary_mul_f64");
binary_kernel!(binary_div_f64, div_f64, 8, "binary_div_f64");
binary_kernel!(binary_maximum_f64, maximum_f64, 8, "binary_maximum_f64");
binary_kernel!(binary_minimum_f64, minimum_f64, 8, "binary_minimum_f64");
binary_kernel!(binary_pow_f64, pow_f64, 8, "binary_pow_f64");
binary_kernel!(binary_rem_f64, mod_f64, 8, "binary_mod_f64");

/// Byte-size lookup for binary-elementwise dtypes.
pub fn dtype_byte_size(dt: DType) -> usize {
    match dt {
        DType::F32 => 4,
        DType::F64 => 8,
        DType::F16 | DType::BF16 => 2,
        _ => 0,
    }
}
