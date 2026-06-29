//! Cast kernels from `baracuda-kernels-sys` — the 8×8 = 64 dtype-pair
//! cross product over `{f32, f64, f16, bf16, i32, i64, u8, i8}`.
//!
//! ## Shape
//!
//! Each `baracuda_kernels_cast_<src>_<dst>_run` has the same shape:
//!
//! ```text
//! fn(numel, x, y, workspace, workspace_bytes, stream) -> i32
//! ```
//!
//! No strided variant — Cast is contig-only on the baracuda surface.
//! Fuel's executor inserts a `Contiguize` op before non-contig consumers
//! and the multi-dtype Cast key includes the input dtype, so by the
//! time the wrapper fires the input is contig.
//!
//! ## U32 → i32 reinterpretation
//!
//! Fuel uses `DType::U32` where baracuda exposes `i32`. For
//! non-negative values (the only case Fuel currently produces — U32
//! holds indices and probe counters), the bit pattern is identical and
//! a reinterpret cast is correct. Same trick as the indexing family
//! (see [`super::indexing`]). For destination U32 the kernel writes
//! i32 bytes into the output buffer; downstream readers re-interpret
//! the same bytes as U32.
//!
//! ## What's not registered
//!
//! - **I8 pairs.** Fuel has no `DType::I8`. Baracuda's 16 I8-touching
//!   pairs (i8↔*) stay live in the FFI but aren't surfaced.
//! - **Sub-byte dtypes outside OCP/NV FP8.**
//!   * `F8E4M3 ↔ {F32, F16, BF16}` IS wired (alpha.29's CastSubBytePlan
//!     family — see entries below the I8 block).
//!   * F8E5M2 / S4 / U4 / Bool — baracuda alpha.29 ships these but
//!     Fuel's DType enum doesn't carry them; would require the
//!     multi-crate I8-style cascade before any registration here.
//!   * F6E2M3 / F6E3M2 / F4 / F8E8M0 — Fuel HAS these in DType, but
//!     they're MX (Microscaling) formats with separate scale tensors,
//!     distinct from baracuda's OCP/NV CastSubByte family. baracuda
//!     alpha.29 doesn't cover them (separate kernel family). Real
//!     baracuda gap when Fuel grows a consumer.

use std::sync::Arc;

use baracuda_kernels_sys as sys;
use fuel_ir::{DType, Error, Result};

use crate::byte_storage::CudaStorageBytes;

use super::scratch::Workspace;
use super::status::check;

type CastRun = unsafe extern "C" fn(
    numel: i64,
    x: *const std::ffi::c_void,
    y: *mut std::ffi::c_void,
    workspace: *mut std::ffi::c_void,
    workspace_bytes: usize,
    stream: *mut std::ffi::c_void,
) -> i32;

/// Core cast driver. Owns the alloc/launch/sync flow common to every
/// (src, dst) pair. Allocates a fresh output buffer of `numel *
/// dst_size_bytes` bytes; the input must hold an integer multiple of
/// `src_size_bytes` bytes.
fn cast_run(
    src: &CudaStorageBytes,
    src_size_bytes: usize,
    dst_size_bytes: usize,
    kernel: CastRun,
    op_label: &'static str,
) -> Result<CudaStorageBytes> {
    if src_size_bytes == 0 || dst_size_bytes == 0 {
        return Err(Error::Msg(format!(
            "{op_label}: zero-byte dtype (src={src_size_bytes}, dst={dst_size_bytes})",
        ))
        .bt());
    }
    if src.len_bytes() % src_size_bytes != 0 {
        return Err(Error::Msg(format!(
            "{op_label}: src.len_bytes={} not a multiple of src elem size {}",
            src.len_bytes(),
            src_size_bytes,
        ))
        .bt());
    }
    let numel = src.len_bytes() / src_size_bytes;
    let out_bytes = numel * dst_size_bytes;
    let device = src.device().clone();
    if out_bytes == 0 {
        return CudaStorageBytes::alloc(&device, 0);
    }
    let out_buf = device.alloc_zeros::<u8>(out_bytes)?;
    let scratch = Workspace::alloc(&device, 0)?;
    let stream = device.stream().as_raw() as *mut std::ffi::c_void;
    let x_ptr = src.buffer().as_raw().0 as *const std::ffi::c_void;
    let y_ptr = out_buf.as_raw().0 as *mut std::ffi::c_void;

    // SAFETY: device pointers + sizes validated above; stream lives on
    // CudaDevice for the call's duration; workspace null/0 (no scratch
    // needed for cast).
    let status = unsafe {
        kernel(
            numel as i64,
            x_ptr,
            y_ptr,
            scratch.as_raw(),
            scratch.bytes(),
            stream,
        )
    };
    check(status, op_label)?;
    Ok(CudaStorageBytes::from_parts(
        Arc::new(out_buf),
        device,
        out_bytes,
    ))
}

/// Per-pair public entry. Generated per (src_size, dst_size, FFI stem)
/// from the manifest below; lets tests and the dispatcher both reach
/// individual pairs by symbol name.
macro_rules! cast_kernel {
    ($name:ident, $src_size:expr, $dst_size:expr, $sys_stem:ident, $op_label:expr $(,)?) => {
        ::paste::paste! {
            #[doc = concat!("Baracuda `", $op_label, "` kernel.")]
            pub fn $name(src: &CudaStorageBytes) -> Result<CudaStorageBytes> {
                cast_run(
                    src,
                    $src_size,
                    $dst_size,
                    sys::[<baracuda_kernels_cast_ $sys_stem _run>],
                    $op_label,
                )
            }
        }
    };
}

// ---------------------------------------------------------------------------
// 8×8 cast manifest. Sizes are in bytes:
//   f32 = 4, f64 = 8, f16 = 2, bf16 = 2, i32 = 4, i64 = 8, u8 = 1, i8 = 1.
// I8 pairs aren't registered into Fuel's binding table (no DType::I8)
// but the wrappers are still defined for completeness / future use.
// ---------------------------------------------------------------------------

// f32 -> *
cast_kernel!(cast_f32_to_f32, 4, 4, f32_f32, "cast_f32_to_f32");
cast_kernel!(cast_f32_to_f64, 4, 8, f32_f64, "cast_f32_to_f64");
cast_kernel!(cast_f32_to_f16, 4, 2, f32_f16, "cast_f32_to_f16");
cast_kernel!(cast_f32_to_bf16, 4, 2, f32_bf16, "cast_f32_to_bf16");
cast_kernel!(cast_f32_to_i32, 4, 4, f32_i32, "cast_f32_to_i32");
cast_kernel!(cast_f32_to_i64, 4, 8, f32_i64, "cast_f32_to_i64");
cast_kernel!(cast_f32_to_u8, 4, 1, f32_u8, "cast_f32_to_u8");
cast_kernel!(cast_f32_to_i8, 4, 1, f32_i8, "cast_f32_to_i8");

// f64 -> *
cast_kernel!(cast_f64_to_f32, 8, 4, f64_f32, "cast_f64_to_f32");
cast_kernel!(cast_f64_to_f64, 8, 8, f64_f64, "cast_f64_to_f64");
cast_kernel!(cast_f64_to_f16, 8, 2, f64_f16, "cast_f64_to_f16");
cast_kernel!(cast_f64_to_bf16, 8, 2, f64_bf16, "cast_f64_to_bf16");
cast_kernel!(cast_f64_to_i32, 8, 4, f64_i32, "cast_f64_to_i32");
cast_kernel!(cast_f64_to_i64, 8, 8, f64_i64, "cast_f64_to_i64");
cast_kernel!(cast_f64_to_u8, 8, 1, f64_u8, "cast_f64_to_u8");
cast_kernel!(cast_f64_to_i8, 8, 1, f64_i8, "cast_f64_to_i8");

// f16 -> *
cast_kernel!(cast_f16_to_f32, 2, 4, f16_f32, "cast_f16_to_f32");
cast_kernel!(cast_f16_to_f64, 2, 8, f16_f64, "cast_f16_to_f64");
cast_kernel!(cast_f16_to_f16, 2, 2, f16_f16, "cast_f16_to_f16");
cast_kernel!(cast_f16_to_bf16, 2, 2, f16_bf16, "cast_f16_to_bf16");
cast_kernel!(cast_f16_to_i32, 2, 4, f16_i32, "cast_f16_to_i32");
cast_kernel!(cast_f16_to_i64, 2, 8, f16_i64, "cast_f16_to_i64");
cast_kernel!(cast_f16_to_u8, 2, 1, f16_u8, "cast_f16_to_u8");
cast_kernel!(cast_f16_to_i8, 2, 1, f16_i8, "cast_f16_to_i8");

// bf16 -> *
cast_kernel!(cast_bf16_to_f32, 2, 4, bf16_f32, "cast_bf16_to_f32");
cast_kernel!(cast_bf16_to_f64, 2, 8, bf16_f64, "cast_bf16_to_f64");
cast_kernel!(cast_bf16_to_f16, 2, 2, bf16_f16, "cast_bf16_to_f16");
cast_kernel!(cast_bf16_to_bf16, 2, 2, bf16_bf16, "cast_bf16_to_bf16");
cast_kernel!(cast_bf16_to_i32, 2, 4, bf16_i32, "cast_bf16_to_i32");
cast_kernel!(cast_bf16_to_i64, 2, 8, bf16_i64, "cast_bf16_to_i64");
cast_kernel!(cast_bf16_to_u8, 2, 1, bf16_u8, "cast_bf16_to_u8");
cast_kernel!(cast_bf16_to_i8, 2, 1, bf16_i8, "cast_bf16_to_i8");

// i32 -> *
cast_kernel!(cast_i32_to_f32, 4, 4, i32_f32, "cast_i32_to_f32");
cast_kernel!(cast_i32_to_f64, 4, 8, i32_f64, "cast_i32_to_f64");
cast_kernel!(cast_i32_to_f16, 4, 2, i32_f16, "cast_i32_to_f16");
cast_kernel!(cast_i32_to_bf16, 4, 2, i32_bf16, "cast_i32_to_bf16");
cast_kernel!(cast_i32_to_i32, 4, 4, i32_i32, "cast_i32_to_i32");
cast_kernel!(cast_i32_to_i64, 4, 8, i32_i64, "cast_i32_to_i64");
cast_kernel!(cast_i32_to_u8, 4, 1, i32_u8, "cast_i32_to_u8");
cast_kernel!(cast_i32_to_i8, 4, 1, i32_i8, "cast_i32_to_i8");

// i64 -> *
cast_kernel!(cast_i64_to_f32, 8, 4, i64_f32, "cast_i64_to_f32");
cast_kernel!(cast_i64_to_f64, 8, 8, i64_f64, "cast_i64_to_f64");
cast_kernel!(cast_i64_to_f16, 8, 2, i64_f16, "cast_i64_to_f16");
cast_kernel!(cast_i64_to_bf16, 8, 2, i64_bf16, "cast_i64_to_bf16");
cast_kernel!(cast_i64_to_i32, 8, 4, i64_i32, "cast_i64_to_i32");
cast_kernel!(cast_i64_to_i64, 8, 8, i64_i64, "cast_i64_to_i64");
cast_kernel!(cast_i64_to_u8, 8, 1, i64_u8, "cast_i64_to_u8");
cast_kernel!(cast_i64_to_i8, 8, 1, i64_i8, "cast_i64_to_i8");

// u8 -> *
cast_kernel!(cast_u8_to_f32, 1, 4, u8_f32, "cast_u8_to_f32");
cast_kernel!(cast_u8_to_f64, 1, 8, u8_f64, "cast_u8_to_f64");
cast_kernel!(cast_u8_to_f16, 1, 2, u8_f16, "cast_u8_to_f16");
cast_kernel!(cast_u8_to_bf16, 1, 2, u8_bf16, "cast_u8_to_bf16");
cast_kernel!(cast_u8_to_i32, 1, 4, u8_i32, "cast_u8_to_i32");
cast_kernel!(cast_u8_to_i64, 1, 8, u8_i64, "cast_u8_to_i64");
cast_kernel!(cast_u8_to_u8, 1, 1, u8_u8, "cast_u8_to_u8");
cast_kernel!(cast_u8_to_i8, 1, 1, u8_i8, "cast_u8_to_i8");

// i8 -> * (defined but not registered — no DType::I8 in Fuel)
cast_kernel!(cast_i8_to_f32, 1, 4, i8_f32, "cast_i8_to_f32");
cast_kernel!(cast_i8_to_f64, 1, 8, i8_f64, "cast_i8_to_f64");
cast_kernel!(cast_i8_to_f16, 1, 2, i8_f16, "cast_i8_to_f16");
cast_kernel!(cast_i8_to_bf16, 1, 2, i8_bf16, "cast_i8_to_bf16");
cast_kernel!(cast_i8_to_i32, 1, 4, i8_i32, "cast_i8_to_i32");
cast_kernel!(cast_i8_to_i64, 1, 8, i8_i64, "cast_i8_to_i64");
cast_kernel!(cast_i8_to_u8, 1, 1, i8_u8, "cast_i8_to_u8");
cast_kernel!(cast_i8_to_i8, 1, 1, i8_i8, "cast_i8_to_i8");

// F8E4M3 ↔ {f32, f16, bf16} — alpha.29's CastSubBytePlan family.
// baracuda exposes these as `fp8e4m3_*` / `*_fp8e4m3` symbol stems.
// Fuel's DType::F8E4M3 maps to a 1-byte element; the FP8 OCP/NV
// format (Fp8E4M3 / Fp8E5M2) is what baracuda ships casts for —
// distinct from the MX-format F4/F6/F8E8M0 dtypes Fuel also carries
// (those have separate scale tensors and would need different
// baracuda kernels, not in alpha.29).
//
// F8E5M2 isn't in Fuel's DType enum yet, so its 6 pairs are
// available in baracuda but not registered here.
cast_kernel!(cast_f8e4m3_to_f32,  1, 4, fp8e4m3_f32,  "cast_f8e4m3_to_f32");
cast_kernel!(cast_f8e4m3_to_f16,  1, 2, fp8e4m3_f16,  "cast_f8e4m3_to_f16");
cast_kernel!(cast_f8e4m3_to_bf16, 1, 2, fp8e4m3_bf16, "cast_f8e4m3_to_bf16");
cast_kernel!(cast_f32_to_f8e4m3,  4, 1, f32_fp8e4m3,  "cast_f32_to_f8e4m3");
cast_kernel!(cast_f16_to_f8e4m3,  2, 1, f16_fp8e4m3,  "cast_f16_to_f8e4m3");
cast_kernel!(cast_bf16_to_f8e4m3, 2, 1, bf16_fp8e4m3, "cast_bf16_to_f8e4m3");

/// Fuel `DType` → baracuda dtype tag. Returns `None` for dtypes
/// baracuda's cast surface doesn't expose. `U32` collapses to `i32` —
/// see module docs for why this is safe for Fuel's usage.
fn baracuda_dtype_tag(dt: DType) -> Option<BaracudaCastDt> {
    Some(match dt {
        DType::F32 => BaracudaCastDt::F32,
        DType::F64 => BaracudaCastDt::F64,
        DType::F16 => BaracudaCastDt::F16,
        DType::BF16 => BaracudaCastDt::Bf16,
        DType::I32 | DType::U32 => BaracudaCastDt::I32,
        DType::I64 => BaracudaCastDt::I64,
        DType::U8 => BaracudaCastDt::U8,
        DType::F8E4M3 => BaracudaCastDt::F8E4M3,
        _ => return None,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BaracudaCastDt {
    F32,
    F64,
    F16,
    Bf16,
    I32,
    I64,
    U8,
    F8E4M3,
}

/// Dispatch a single (src_dtype, dst_dtype) pair to the matching
/// baracuda `cast_<src>_<dst>_run` symbol. Returns an error for pairs
/// outside the baracuda surface (sub-byte dtypes, I16, F8 family, …).
///
/// This is what `fuel-storage`'s `cast_cuda_baracuda_wrapper` calls
/// after reading the input/output Storage dtypes.
pub fn dispatch(
    src: &CudaStorageBytes,
    src_dt: DType,
    dst_dt: DType,
) -> Result<CudaStorageBytes> {
    let src_tag = baracuda_dtype_tag(src_dt).ok_or_else(|| {
        Error::Msg(format!("baracuda cast: src dtype {src_dt:?} not supported")).bt()
    })?;
    let dst_tag = baracuda_dtype_tag(dst_dt).ok_or_else(|| {
        Error::Msg(format!("baracuda cast: dst dtype {dst_dt:?} not supported")).bt()
    })?;
    use BaracudaCastDt::*;
    match (src_tag, dst_tag) {
        (F32, F32) => cast_f32_to_f32(src),
        (F32, F64) => cast_f32_to_f64(src),
        (F32, F16) => cast_f32_to_f16(src),
        (F32, Bf16) => cast_f32_to_bf16(src),
        (F32, I32) => cast_f32_to_i32(src),
        (F32, I64) => cast_f32_to_i64(src),
        (F32, U8) => cast_f32_to_u8(src),

        (F64, F32) => cast_f64_to_f32(src),
        (F64, F64) => cast_f64_to_f64(src),
        (F64, F16) => cast_f64_to_f16(src),
        (F64, Bf16) => cast_f64_to_bf16(src),
        (F64, I32) => cast_f64_to_i32(src),
        (F64, I64) => cast_f64_to_i64(src),
        (F64, U8) => cast_f64_to_u8(src),

        (F16, F32) => cast_f16_to_f32(src),
        (F16, F64) => cast_f16_to_f64(src),
        (F16, F16) => cast_f16_to_f16(src),
        (F16, Bf16) => cast_f16_to_bf16(src),
        (F16, I32) => cast_f16_to_i32(src),
        (F16, I64) => cast_f16_to_i64(src),
        (F16, U8) => cast_f16_to_u8(src),

        (Bf16, F32) => cast_bf16_to_f32(src),
        (Bf16, F64) => cast_bf16_to_f64(src),
        (Bf16, F16) => cast_bf16_to_f16(src),
        (Bf16, Bf16) => cast_bf16_to_bf16(src),
        (Bf16, I32) => cast_bf16_to_i32(src),
        (Bf16, I64) => cast_bf16_to_i64(src),
        (Bf16, U8) => cast_bf16_to_u8(src),

        (I32, F32) => cast_i32_to_f32(src),
        (I32, F64) => cast_i32_to_f64(src),
        (I32, F16) => cast_i32_to_f16(src),
        (I32, Bf16) => cast_i32_to_bf16(src),
        (I32, I32) => cast_i32_to_i32(src),
        (I32, I64) => cast_i32_to_i64(src),
        (I32, U8) => cast_i32_to_u8(src),

        (I64, F32) => cast_i64_to_f32(src),
        (I64, F64) => cast_i64_to_f64(src),
        (I64, F16) => cast_i64_to_f16(src),
        (I64, Bf16) => cast_i64_to_bf16(src),
        (I64, I32) => cast_i64_to_i32(src),
        (I64, I64) => cast_i64_to_i64(src),
        (I64, U8) => cast_i64_to_u8(src),

        (U8, F32) => cast_u8_to_f32(src),
        (U8, F64) => cast_u8_to_f64(src),
        (U8, F16) => cast_u8_to_f16(src),
        (U8, Bf16) => cast_u8_to_bf16(src),
        (U8, I32) => cast_u8_to_i32(src),
        (U8, I64) => cast_u8_to_i64(src),
        (U8, U8) => cast_u8_to_u8(src),

        // F8E4M3 ↔ {F32, F16, BF16}. Baracuda alpha.29's CastSubBytePlan
        // family. Pairs outside this set (F8E4M3 ↔ {I32, I64, U8, F64,
        // F8E4M3-to-F8E4M3}) aren't shipped — they'd require f32-detour
        // chaining if Fuel ever needs them.
        (F8E4M3, F32)    => cast_f8e4m3_to_f32(src),
        (F8E4M3, F16)    => cast_f8e4m3_to_f16(src),
        (F8E4M3, Bf16)   => cast_f8e4m3_to_bf16(src),
        (F32,    F8E4M3) => cast_f32_to_f8e4m3(src),
        (F16,    F8E4M3) => cast_f16_to_f8e4m3(src),
        (Bf16,   F8E4M3) => cast_bf16_to_f8e4m3(src),

        // Unsupported pairs through baracuda's CastSubByte family.
        (F8E4M3, _) | (_, F8E4M3) => Err(Error::Msg(format!(
            "baracuda cast: F8E4M3 ↔ {:?} / {:?} not in baracuda alpha.29's \
             CastSubBytePlan surface (supported: F8E4M3 ↔ {{F32, F16, BF16}}). \
             Compose via an intermediate f32 cast if needed.",
            src_dt, dst_dt,
        )).bt()),
    }
}
