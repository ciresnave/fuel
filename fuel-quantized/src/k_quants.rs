// SPDX-License-Identifier: MIT OR Apache-2.0
// Derived from candle (https://github.com/huggingface/candle, MIT OR Apache-2.0): `candle-core/src/quantized/k_quants.rs`
// at commit 549eacb011f25b2c6fc765d6c3dbd26e42f72ee6, modified here. candle's code is itself a port of ggml / llama.cpp (MIT), which its own comments cite by commit.
// Contributors, how this was verified and the licence texts: THIRD-PARTY-NOTICES.md (section "candle").
use crate::utils::{
    get_scale_min_k4, group_for_dequantization, group_for_quantization, make_q3_quants,
    make_qkx1_quants, make_qkx3_quants, make_qp_quants, make_qx_quants, nearest_int,
};
use byteorder::{ByteOrder, LittleEndian};
use fuel_ir::Result;
use fuel_ir::quantized::GgmlDType;
use half::{bf16, f16, slice::HalfFloatSliceExt};
use rayon::prelude::*;

// Default to QK_K 256 rather than 64.
pub const QK_K: usize = 256;
pub const K_SCALE_SIZE: usize = 12;

pub const QK4_0: usize = 32;
pub const QK4_1: usize = 32;
pub const QK5_0: usize = 32;
pub const QK5_1: usize = 32;
pub const QK8_0: usize = 32;
pub const QK8_1: usize = 32;

/// A GGML block type (or a dense scalar standing in for one).
///
/// `bytemuck::Pod` is a supertrait on purpose: every implementor is
/// `#[repr(C)]` plain data with no padding and no invalid bit pattern, so a
/// byte slice of the right length and alignment IS a valid `[Self]`, and
/// all-zero bytes ARE a valid `Self`. Making that a bound the compiler checks
/// (the derive refuses a type with padding) replaces the hand-written
/// `unsafe` casts that each asserted it in a comment.
pub trait GgmlType: Sized + Clone + Send + Sync + bytemuck::Pod {
    const DTYPE: GgmlDType;
    const BLCK_SIZE: usize;

    /// The all-zero value. Safe by construction: `Self: Pod` implies `Zeroable`.
    fn zeros() -> Self {
        <Self as bytemuck::Zeroable>::zeroed()
    }
    fn to_float(xs: &[Self], ys: &mut [f32]);
    fn from_float(xs: &[f32], ys: &mut [Self]);
    fn from_float_imatrix(
        _xs: &[f32],
        _ys: &mut [Self],
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) {
        panic!(
            "`from_float_imatrix` is unimplemented for {:?}",
            Self::DTYPE
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ4_0 {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_0>() == 18);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ4_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qs: [u8; QK4_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ4_1>() == 20);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ5_0 {
    pub(crate) d: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_0 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_0>() == 22);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ5_1 {
    pub(crate) d: f16,
    pub(crate) m: f16,
    pub(crate) qh: [u8; 4],
    pub(crate) qs: [u8; QK5_1 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockQ5_1>() == 24);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ8_0 {
    pub(crate) d: f16,
    pub(crate) qs: [i8; QK8_0],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_0>() == 34);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ8_1 {
    pub(crate) d: f16,
    pub(crate) s: f16,
    pub(crate) qs: [i8; QK8_1],
}
const _: () = assert!(std::mem::size_of::<BlockQ8_1>() == 36);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ2K {
    pub(crate) scales: [u8; QK_K / 16],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) d: f16,
    pub(crate) dmin: f16,
}
const _: () = assert!(QK_K / 16 + QK_K / 4 + 2 * 2 == std::mem::size_of::<BlockQ2K>());

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ3K {
    pub(crate) hmask: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) scales: [u8; 12],
    pub(crate) d: f16,
}
const _: () = assert!(QK_K / 8 + QK_K / 4 + 12 + 2 == std::mem::size_of::<BlockQ3K>());

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
// https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/k_quants.h#L82
#[repr(C)]
pub struct BlockQ4K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () = assert!(QK_K / 2 + K_SCALE_SIZE + 2 * 2 == std::mem::size_of::<BlockQ4K>());

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ5K {
    pub(crate) d: f16,
    pub(crate) dmin: f16,
    pub(crate) scales: [u8; K_SCALE_SIZE],
    pub(crate) qh: [u8; QK_K / 8],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () =
    assert!(QK_K / 8 + QK_K / 2 + 2 * 2 + K_SCALE_SIZE == std::mem::size_of::<BlockQ5K>());

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ6K {
    pub(crate) ql: [u8; QK_K / 2],
    pub(crate) qh: [u8; QK_K / 4],
    pub(crate) scales: [i8; QK_K / 16],
    pub(crate) d: f16,
}
const _: () = assert!(3 * QK_K / 4 + QK_K / 16 + 2 == std::mem::size_of::<BlockQ6K>());

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockQ8K {
    pub(crate) d: f32,
    pub(crate) qs: [i8; QK_K],
    pub(crate) bsums: [i16; QK_K / 16],
}
const _: () = assert!(4 + QK_K + QK_K / 16 * 2 == std::mem::size_of::<BlockQ8K>());

/// View a block slice as its raw bytes: GGUF's on-disk layout, byte for byte.
///
/// Infallible: `u8` has alignment 1 and every `T: GgmlType` is `Pod`, so this
/// is a plain reinterpretation with no length or alignment condition.
pub fn blocks_as_bytes<T: GgmlType>(blocks: &[T]) -> &[u8] {
    bytemuck::cast_slice(blocks)
}

/// Reinterpret GGUF bytes as a dense `[T]` array.
///
/// Borrowed when `bytes` happens to be aligned for `T`, **copied** into an
/// aligned `Vec<T>` otherwise: a `&[u8]` guarantees only 1-byte alignment, and
/// a sub-slice taken at a tensor offset inside an mmap need not be aligned for
/// the `f16`/`f32` fields of a block, so assuming it is would be unsound.
///
/// A trailing partial block is dropped, matching the pre-existing
/// per-model `cpu_dequant_*_bytes` callers; callers that must reject a ragged
/// length check `bytes.len() % size_of::<T>()` themselves.
pub fn blocks_from_bytes_cow<T: GgmlType>(bytes: &[u8]) -> std::borrow::Cow<'_, [T]> {
    let size = std::mem::size_of::<T>();
    let whole = &bytes[..(bytes.len() / size) * size];
    match bytemuck::try_cast_slice::<u8, T>(whole) {
        Ok(blocks) => std::borrow::Cow::Borrowed(blocks),
        Err(_) => std::borrow::Cow::Owned(
            whole
                .chunks_exact(size)
                .map(bytemuck::pod_read_unaligned::<T>)
                .collect(),
        ),
    }
}

impl GgmlType for BlockQ4_0 {
    const DTYPE: GgmlDType = GgmlDType::Q4_0;
    const BLCK_SIZE: usize = QK4_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1525
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        let qk = Self::BLCK_SIZE;
        debug_assert!(
            k.is_multiple_of(qk),
            "dequantize_row_q4_0: {k} is not divisible by {qk}"
        );

        let nb = k / qk;
        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..(qk / 2) {
                let x0 = (xs[i].qs[j] & 0x0F) as i16 - 8;
                let x1 = (xs[i].qs[j] >> 4) as i16 - 8;

                ys[i * qk + j] = (x0 as f32) * d;
                ys[i * qk + j + qk / 2] = (x1 as f32) * d;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q4_0
        let qk = Self::BLCK_SIZE;
        let k = xs.len();
        debug_assert!(k.is_multiple_of(qk), "{k} is not divisible by {qk}");
        debug_assert_eq!(
            ys.len(),
            k / qk,
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            qk,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let mut max = 0f32;

            let xs = &xs[i * qk..(i + 1) * qk];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -8.0;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);

            for (j, q) in ys.qs.iter_mut().enumerate() {
                let x0 = xs[j] * id;
                let x1 = xs[qk / 2 + j] * id;
                let xi0 = u8::min(15, (x0 + 8.5) as u8);
                let xi1 = u8::min(15, (x1 + 8.5) as u8);
                *q = xi0 | (xi1 << 4)
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/b5ffb2849d23afe73647f68eec7b68187af09be6/ggml.c#L2361C10-L2361C122
}

impl GgmlType for BlockQ4_1 {
    const DTYPE: GgmlDType = GgmlDType::Q4_1;
    const BLCK_SIZE: usize = QK4_1;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q4_1
        let qk = Self::BLCK_SIZE;

        debug_assert_eq!(
            ys.len() * qk,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            qk,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * qk..(i + 1) * qk];

            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &x in xs.iter() {
                min = f32::min(x, min);
                max = f32::max(x, max);
            }
            let d = (max - min) / ((1 << 4) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            ys.m = f16::from_f32(min);

            for (j, q) in ys.qs.iter_mut().take(qk / 2).enumerate() {
                let x0 = (xs[j] - min) * id;
                let x1 = (xs[qk / 2 + j] - min) * id;

                let xi0 = u8::min(15, (x0 + 0.5) as u8);
                let xi1 = u8::min(15, (x1 + 0.5) as u8);

                *q = xi0 | (xi1 << 4);
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1545
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK4_1),
            "dequantize_row_q4_1: {k} is not divisible by {QK4_1}"
        );

        let nb = k / QK4_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let m = xs[i].m.to_f32();

            for j in 0..(QK4_1 / 2) {
                let x0 = xs[i].qs[j] & 0x0F;
                let x1 = xs[i].qs[j] >> 4;

                ys[i * QK4_1 + j] = (x0 as f32) * d + m;
                ys[i * QK4_1 + j + QK4_1 / 2] = (x1 as f32) * d + m;
            }
        }
    }
}

impl GgmlType for BlockQ5_0 {
    const DTYPE: GgmlDType = GgmlDType::Q5_0;
    const BLCK_SIZE: usize = QK5_0;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q5_0
        debug_assert_eq!(
            ys.len() * Self::BLCK_SIZE,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];

            let mut amax = 0f32;
            let mut max = 0f32;
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            let d = max / -16.;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            let mut qh = 0u32;
            for j in 0..Self::BLCK_SIZE / 2 {
                let x0 = xs[j] * id;
                let x1 = xs[j + Self::BLCK_SIZE / 2] * id;
                let xi0 = ((x0 + 16.5) as i8).min(31) as u8;
                let xi1 = ((x1 + 16.5) as i8).min(31) as u8;
                ys.qs[j] = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
                qh |= ((xi0 as u32 & 0x10) >> 4) << j;
                qh |= ((xi1 as u32 & 0x10) >> 4) << (j + Self::BLCK_SIZE / 2);
            }
            LittleEndian::write_u32(&mut ys.qh, qh)
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1566
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK5_0),
            "dequantize_row_q5_0: {k} is not divisible by {QK5_0}"
        );
        let nb = k / QK5_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let qh: u32 = LittleEndian::read_u32(&xs[i].qh);

            for j in 0..(QK5_0 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

                let x0 = ((xs[i].qs[j] & 0x0F) | xh_0) as i32 - 16;
                let x1 = ((xs[i].qs[j] >> 4) | xh_1) as i32 - 16;

                ys[i * QK5_0 + j] = (x0 as f32) * d;
                ys[i * QK5_0 + j + QK5_0 / 2] = (x1 as f32) * d;
            }
        }
    }
}

impl GgmlType for BlockQ5_1 {
    const DTYPE: GgmlDType = GgmlDType::Q5_1;
    const BLCK_SIZE: usize = QK5_1;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q5_1
        let qk = Self::BLCK_SIZE;
        debug_assert_eq!(
            ys.len() * qk,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            qk,
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let xs = &xs[i * qk..(i + 1) * qk];

            let mut min = f32::INFINITY;
            let mut max = f32::NEG_INFINITY;
            for &x in xs.iter() {
                min = f32::min(x, min);
                max = f32::max(x, max);
            }
            let d = (max - min) / ((1 << 5) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            ys.m = f16::from_f32(min);

            let mut qh = 0u32;
            for (j, q) in ys.qs.iter_mut().take(qk / 2).enumerate() {
                let x0 = (xs[j] - min) * id;
                let x1 = (xs[qk / 2 + j] - min) * id;

                let xi0 = (x0 + 0.5) as u8;
                let xi1 = (x1 + 0.5) as u8;

                *q = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
                // get the 5-th bit and store it in qh at the right position
                qh |= ((xi0 as u32 & 0x10) >> 4) << j;
                qh |= ((xi1 as u32 & 0x10) >> 4) << (j + qk / 2);
            }
            LittleEndian::write_u32(&mut ys.qh, qh);
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1592
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK5_1),
            "dequantize_row_q5_1: {k} is not divisible by {QK5_1}"
        );

        let nb = k / QK5_1;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let m = xs[i].m.to_f32();
            let qh: u32 = LittleEndian::read_u32(&xs[i].qh);

            for j in 0..(QK5_1 / 2) {
                let xh_0 = (((qh >> j) << 4) & 0x10) as u8;
                let xh_1 = ((qh >> (j + 12)) & 0x10) as u8;

                let x0 = (xs[i].qs[j] & 0x0F) | xh_0;
                let x1 = (xs[i].qs[j] >> 4) | xh_1;

                ys[i * QK5_1 + j] = (x0 as f32) * d + m;
                ys[i * QK5_1 + j + QK5_1 / 2] = (x1 as f32) * d + m;
            }
        }
    }
}

impl GgmlType for BlockQ8_0 {
    const DTYPE: GgmlDType = GgmlDType::Q8_0;
    const BLCK_SIZE: usize = QK8_0;

    // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L1619
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK8_0),
            "dequantize_row_q8_0: {k} is not divisible by {QK8_0}"
        );

        let nb = k / QK8_0;

        for i in 0..nb {
            let d = xs[i].d.to_f32();

            for j in 0..QK8_0 {
                ys[i * QK8_0 + j] = xs[i].qs[j] as f32 * d;
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q8_0
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(Self::BLCK_SIZE),
            "{k} is not divisible by {}",
            Self::BLCK_SIZE
        );
        debug_assert_eq!(
            ys.len(),
            k / Self::BLCK_SIZE,
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            for (y, &x) in ys.qs.iter_mut().zip(xs.iter()) {
                *y = f32::round(x * id) as i8
            }
        }
    }
}

impl GgmlType for BlockQ8_1 {
    const DTYPE: GgmlDType = GgmlDType::Q8_1;
    const BLCK_SIZE: usize = QK8_1;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        // quantize_row_q8_1
        debug_assert_eq!(
            ys.len() * Self::BLCK_SIZE,
            xs.len(),
            "size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        for (i, ys) in ys.iter_mut().enumerate() {
            let mut amax = 0f32;
            let xs = &xs[i * Self::BLCK_SIZE..(i + 1) * Self::BLCK_SIZE];
            for &x in xs.iter() {
                amax = amax.max(x.abs())
            }
            let d = amax / ((1 << 7) - 1) as f32;
            let id = if d != 0f32 { 1. / d } else { 0. };
            ys.d = f16::from_f32(d);
            let mut sum = 0i32;
            for j in 0..Self::BLCK_SIZE / 2 {
                let v0 = xs[j] * id;
                let v1 = xs[j + Self::BLCK_SIZE / 2] * id;
                ys.qs[j] = f32::round(v0) as i8;
                ys.qs[j + Self::BLCK_SIZE / 2] = f32::round(v1) as i8;
                sum += ys.qs[j] as i32 + ys.qs[j + Self::BLCK_SIZE / 2] as i32;
            }
            ys.s = f16::from_f32(sum as f32) * ys.d;
        }
    }

    fn to_float(_xs: &[Self], _ys: &mut [f32]) {
        unimplemented!("no support for vec-dot on Q8_1") // GAP(GAP-125)
    }
}

impl GgmlType for BlockQ2K {
    const DTYPE: GgmlDType = GgmlDType::Q2K;
    const BLCK_SIZE: usize = QK_K;

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L279
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        const Q4SCALE: f32 = 15.0;

        for (block, x) in group_for_quantization(xs, ys) {
            //calculate scales and mins
            let mut mins: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];

            for (j, x_scale_slice) in x.chunks(16).enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(3, 5, x_scale_slice);
            }
            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            if max_scale > 0.0 {
                let iscale = Q4SCALE / max_scale;
                for (j, scale) in scales.iter().enumerate().take(QK_K / 16) {
                    block.scales[j] = nearest_int(iscale * scale) as u8;
                }
                block.d = f16::from_f32(max_scale / Q4SCALE);
            } else {
                for j in 0..QK_K / 16 {
                    block.scales[j] = 0;
                }
                block.d = f16::from_f32(0.0);
            }

            if max_min > 0.0 {
                let iscale = Q4SCALE / max_min;
                for (j, scale) in block.scales.iter_mut().enumerate() {
                    let l = nearest_int(iscale * mins[j]) as u8;
                    *scale |= l << 4;
                }
                block.dmin = f16::from_f32(max_min / Q4SCALE);
            } else {
                block.dmin = f16::from_f32(0.0);
            }

            let mut big_l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let d = block.d.to_f32() * (block.scales[j] & 0xF) as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * (block.scales[j] >> 4) as f32;
                for ii in 0..16 {
                    let ll = nearest_int((x[16 * j + ii] + dm) / d).clamp(0, 3);
                    big_l[16 * j + ii] = ll as u8;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for ll in 0..32 {
                    block.qs[j / 4 + ll] = big_l[j + ll]
                        | (big_l[j + ll + 32] << 2)
                        | (big_l[j + ll + 64] << 4)
                        | (big_l[j + ll + 96] << 6);
                }
            }
        }
    }

    fn from_float_imatrix(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        for (sblk_idx, (block, x)) in group_for_quantization(xs, ys).into_iter().enumerate() {
            let mut mins: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut weights: [f32; 16] = [0.0; 16];
            let mut sw: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut ls: [u8; QK_K / 16] = [0; QK_K / 16];
            let mut lm: [u8; QK_K / 16] = [0; QK_K / 16];

            let sum_x2 = x.iter().map(|x| x * x).sum::<f32>();
            let sigma2 = sum_x2 / QK_K as f32;
            for (j, x_scale_slice) in x.as_chunks::<16>().0.iter().enumerate() {
                for (l, (w_elem, x_elem)) in weights.iter_mut().zip(x_scale_slice).enumerate() {
                    let imatrix_row = sblk_idx % (n_per_row / QK_K);
                    let imatrix_w = imatrix_weights[imatrix_row * QK_K + 16 * j + l];
                    *w_elem = imatrix_w * (sigma2 + x_elem * x_elem).sqrt();
                }
                let sumw = weights.iter().sum::<f32>();
                sw[j] = sumw;
                (scales[j], mins[j]) =
                    make_qkx3_quants(3, x_scale_slice, Some(&weights), -0.9, 0.05, 36, false);
            }

            let d_block = make_qp_quants(QK_K / 16, 15, &scales, &mut ls, &sw);
            let m_block = make_qp_quants(QK_K / 16, 15, &mins, &mut lm, &sw);

            block.d = f16::from_f32(d_block);
            block.dmin = f16::from_f32(m_block);

            for j in 0..QK_K / 16 {
                block.scales[j] = ls[j] | (lm[j] << 4);
            }

            let mut big_l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let d = block.d.to_f32() * (block.scales[j] & 0xF) as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * (block.scales[j] >> 4) as f32;
                for ii in 0..16 {
                    let ll = nearest_int((x[16 * j + ii] + dm) / d).clamp(0, 3);
                    big_l[16 * j + ii] = ll as u8;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for ll in 0..32 {
                    block.qs[j / 4 + ll] = big_l[j + ll]
                        | (big_l[j + ll + 32] << 2)
                        | (big_l[j + ll + 64] << 4)
                        | (big_l[j + ll + 96] << 6);
                }
            }
        }
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L354
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in group_for_dequantization(xs, ys) {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();

            let mut is = 0;

            for (y_block, qs) in y
                .as_chunks_mut::<128>()
                .0
                .iter_mut()
                .zip(block.qs.as_chunks::<32>().0.iter())
            {
                // Step by 32 over q.
                let mut shift = 0;
                let mut y_block_index = 0;
                for _j in 0..4 {
                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for q in &qs[..16] {
                        let y = dl * ((q >> shift) & 3) as f32 - ml;
                        y_block[y_block_index] = y;
                        y_block_index += 1;
                    }

                    let sc = block.scales[is];
                    is += 1;
                    let dl = d * (sc & 0xF) as f32;
                    let ml = min * (sc >> 4) as f32;
                    for q in &qs[16..] {
                        let y = dl * ((q >> shift) & 3) as f32 - ml;
                        y_block[y_block_index] = y;
                        y_block_index += 1;
                    }

                    shift += 2;
                }
            }
        }
    }
}

impl GgmlType for BlockQ3K {
    const DTYPE: GgmlDType = GgmlDType::Q3K;
    const BLCK_SIZE: usize = QK_K;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in group_for_quantization(xs, ys) {
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];
            for (j, x_scale_slice) in x.as_chunks::<16>().0.iter().enumerate() {
                scales[j] = make_q3_quants(x_scale_slice, 4, true);
            }

            // Get max scale by absolute value.
            let mut max_scale: f32 = 0.0;
            for &scale in scales.iter() {
                if scale.abs() > max_scale.abs() {
                    max_scale = scale;
                }
            }

            block.scales.fill(0);

            if max_scale != 0.0 {
                let iscale = -32.0 / max_scale;
                for (j, scale) in scales.iter().enumerate() {
                    let l_val = nearest_int(iscale * scale);
                    let l_val = l_val.clamp(-32, 31) + 32;
                    if j < 8 {
                        block.scales[j] = (l_val & 0xF) as u8;
                    } else {
                        block.scales[j - 8] |= ((l_val & 0xF) << 4) as u8;
                    }
                    let l_val = l_val >> 4;
                    block.scales[j % 4 + 8] |= (l_val << (2 * (j / 4))) as u8;
                }
                block.d = f16::from_f32(1.0 / iscale);
            } else {
                block.d = f16::from_f32(0.0);
            }

            let mut l: [i8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 16 {
                let sc = if j < 8 {
                    block.scales[j] & 0xF
                } else {
                    block.scales[j - 8] >> 4
                };
                let sc = (sc | (((block.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i8 - 32;
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    for ii in 0..16 {
                        let l_val = nearest_int(x[16 * j + ii] / d);
                        l[16 * j + ii] = (l_val.clamp(-4, 3) + 4) as i8;
                    }
                }
            }

            block.hmask.fill(0);
            let mut m = 0;
            let mut hm = 1;

            for ll in l.iter_mut() {
                if *ll > 3 {
                    block.hmask[m] |= hm;
                    *ll -= 4;
                }
                m += 1;
                if m == QK_K / 8 {
                    m = 0;
                    hm <<= 1;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for l_val in 0..32 {
                    block.qs[j / 4 + l_val] = (l[j + l_val]
                        | (l[j + l_val + 32] << 2)
                        | (l[j + l_val + 64] << 4)
                        | (l[j + l_val + 96] << 6))
                        as u8;
                }
            }
        }
    }

    fn from_float_imatrix(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        for (sblk_idx, (block, x)) in group_for_quantization(xs, ys).into_iter().enumerate() {
            let mut scales: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut weights: [f32; 16] = [0.0; 16];
            let mut sw: [f32; QK_K / 16] = [0.0; QK_K / 16];
            let mut ls: [i8; QK_K / 16] = [0; QK_K / 16];
            let mut l: [i8; QK_K] = [0; QK_K];

            let sum_x2 = x.iter().map(|x| x * x).sum::<f32>();
            let sigma2 = 2. * sum_x2 / QK_K as f32;

            for (j, x_scale_slice) in x.as_chunks::<16>().0.iter().enumerate() {
                for (l_idx, (w_elem, x_elem)) in weights.iter_mut().zip(x_scale_slice).enumerate() {
                    let imatrix_row = sblk_idx % (n_per_row / QK_K);
                    let imatrix_w = imatrix_weights[imatrix_row * QK_K + 16 * j + l_idx];
                    *w_elem = imatrix_w * (sigma2 + x_elem * x_elem).sqrt();
                }
                let sumw = weights.iter().sum::<f32>();
                sw[j] = sumw;
                scales[j] = unsafe {
                    make_qx_quants(
                        16,
                        4,
                        x_scale_slice.as_ptr(),
                        l.as_mut_ptr().add(16 * j),
                        1,
                        weights.as_ptr(),
                    )
                };
            }

            block.scales.fill(0);
            let d_block = unsafe {
                make_qx_quants(
                    QK_K / 16,
                    32,
                    scales.as_ptr(),
                    ls.as_mut_ptr(),
                    1,
                    sw.as_ptr(),
                )
            };
            block.d = f16::from_f32(d_block);
            for (j, l_val) in ls.iter().enumerate().take(QK_K / 16) {
                if j < 8 {
                    block.scales[j] = (l_val & 0xF) as u8;
                } else {
                    block.scales[j - 8] |= ((l_val & 0xF) << 4) as u8;
                }
                let l_val = l_val >> 4;
                block.scales[j % 4 + 8] |= (l_val << (2 * (j / 4))) as u8;
            }

            for j in 0..QK_K / 16 {
                let sc = if j < 8 {
                    block.scales[j] & 0xF
                } else {
                    block.scales[j - 8] >> 4
                };
                let sc = (sc | (((block.scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i8 - 32;
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    for ii in 0..16 {
                        let l_val = nearest_int(x[16 * j + ii] / d);
                        l[16 * j + ii] = (l_val.clamp(-4, 3) + 4) as i8;
                    }
                }
            }

            block.hmask.fill(0);
            let mut m = 0;
            let mut hm = 1;

            for ll in l.iter_mut() {
                if *ll > 3 {
                    block.hmask[m] |= hm;
                    *ll -= 4;
                }
                m += 1;
                if m == QK_K / 8 {
                    m = 0;
                    hm <<= 1;
                }
            }

            for j in (0..QK_K).step_by(128) {
                for l_val in 0..32 {
                    block.qs[j / 4 + l_val] = (l[j + l_val]
                        | (l[j + l_val + 32] << 2)
                        | (l[j + l_val + 64] << 4)
                        | (l[j + l_val + 96] << 6))
                        as u8;
                }
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L533
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        const KMASK1: u32 = 0x03030303;
        const KMASK2: u32 = 0x0f0f0f0f;

        for (block, y) in group_for_dequantization(xs, ys) {
            //Reconstruct the scales
            let mut aux = [0; 4];
            LittleEndian::read_u32_into(&block.scales, &mut aux[0..3]);

            let tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
            aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
            aux[0] = (aux[0] & KMASK2) | (((tmp) & KMASK1) << 4);
            aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);

            //Transfer the scales into an i8 array
            let scales: &mut [i8] =
                unsafe { std::slice::from_raw_parts_mut(aux.as_mut_ptr() as *mut i8, 16) };

            let d_all = block.d.to_f32();
            let mut m = 1;
            let mut is = 0;

            // Dequantize both 128 long blocks
            // 32 qs values per 128 long block
            // Each 16 elements get a scale
            for (y, qs) in y
                .as_chunks_mut::<128>()
                .0
                .iter_mut()
                .zip(block.qs.as_chunks::<32>().0.iter())
            {
                let mut shift = 0;
                for shift_scoped_y in y.as_chunks_mut::<32>().0.iter_mut() {
                    for (scale_index, scale_scoped_y) in shift_scoped_y
                        .as_chunks_mut::<16>()
                        .0
                        .iter_mut()
                        .enumerate()
                    {
                        let dl = d_all * (scales[is] as f32 - 32.0);
                        for (i, inner_y) in scale_scoped_y.iter_mut().enumerate() {
                            let new_y = dl
                                * (((qs[i + 16 * scale_index] >> shift) & 3) as i8
                                    - if (block.hmask[i + 16 * scale_index] & m) == 0 {
                                        4
                                    } else {
                                        0
                                    }) as f32;
                            *inner_y = new_y;
                        }
                        // 16 block finished => advance scale index
                        is += 1;
                    }
                    // 32 block finished => increase shift and m
                    shift += 2;
                    m <<= 1;
                }
            }
        }
    }
}

impl GgmlType for BlockQ4K {
    const DTYPE: GgmlDType = GgmlDType::Q4K;
    const BLCK_SIZE: usize = QK_K;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in group_for_quantization(xs, ys) {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.as_chunks::<32>().0.iter().enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(15, 5, x_scale_slice);
            }

            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

            for j in 0..QK_K / 32 {
                let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    block.scales[j] = ls;
                    block.scales[j + 4] = lm;
                } else {
                    block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    block.scales[j - 4] |= (ls >> 4) << 6;
                    block.scales[j] |= (lm >> 4) << 6;
                }
            }

            block.d = f16::from_f32(max_scale / 63.0);
            block.dmin = f16::from_f32(max_min / 63.0);

            let mut l: [u8; QK_K] = [0; QK_K];

            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    let dm = block.dmin.to_f32() * m as f32;
                    for ii in 0..32 {
                        let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                        l[32 * j + ii] = l_val.clamp(0, 15) as u8;
                    }
                }
            }

            let q = &mut block.qs;
            for j in (0..QK_K).step_by(64) {
                for l_val in 0..32 {
                    let offset_index = (j / 64) * 32 + l_val;
                    q[offset_index] = l[j + l_val] | (l[j + l_val + 32] << 4);
                }
            }
        }
    }

    fn from_float_imatrix(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        for (sblk_idx, (block, x)) in group_for_quantization(xs, ys).into_iter().enumerate() {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut weights: [f32; 32] = [0.0; 32];
            let mut sw: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut ls: [u8; QK_K / 32] = [0; QK_K / 32];
            let mut lm: [u8; QK_K / 32] = [0; QK_K / 32];

            let sum_x2 = x.iter().map(|x| x * x).sum::<f32>();
            let sigma2 = 2. * sum_x2 / QK_K as f32;

            for (j, x_scale_slice) in x.as_chunks::<32>().0.iter().enumerate() {
                for (l, (w_elem, x_elem)) in weights.iter_mut().zip(x_scale_slice).enumerate() {
                    let imatrix_row = sblk_idx % (n_per_row / QK_K);
                    let imatrix_w = imatrix_weights[imatrix_row * QK_K + 32 * j + l];
                    *w_elem = imatrix_w * (sigma2 + x_elem * x_elem).sqrt();
                }
                let sumw = weights.iter().sum::<f32>();
                sw[j] = sumw;
                (scales[j], mins[j]) =
                    make_qkx3_quants(15, x_scale_slice, Some(&weights), -0.9, 0.05, 36, false);
            }

            let d_block = make_qp_quants(QK_K / 32, 63, &scales, &mut ls, &sw);
            let m_block = make_qp_quants(QK_K / 32, 63, &mins, &mut lm, &sw);
            for j in 0..QK_K / 32 {
                let ls_val = ls[j];
                let lm_val = lm[j];
                if j < 4 {
                    block.scales[j] = ls_val;
                    block.scales[j + 4] = lm_val;
                } else {
                    block.scales[j + 4] = (ls_val & 0xF) | ((lm_val & 0xF) << 4);
                    block.scales[j - 4] |= (ls_val >> 4) << 6;
                    block.scales[j] |= (lm_val >> 4) << 6;
                }
            }

            block.d = f16::from_f32(d_block);
            block.dmin = f16::from_f32(m_block);

            let mut l: [u8; QK_K] = [0; QK_K];
            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    let dm = block.dmin.to_f32() * m as f32;
                    for ii in 0..32 {
                        let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                        l[32 * j + ii] = l_val.clamp(0, 15) as u8;
                    }
                }
            }

            let q = &mut block.qs;
            for j in (0..QK_K).step_by(64) {
                for l_val in 0..32 {
                    let offset_index = (j / 64) * 32 + l_val;
                    q[offset_index] = l[j + l_val] | (l[j + l_val + 32] << 4);
                }
            }
        }
    }
    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L735
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in group_for_dequantization(xs, ys) {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let q = &block.qs;
            let mut is = 0;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let q = &q[j / 2..j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for q in q {
                    y[ys_index] = d1 * (q & 0xF) as f32 - m1;
                    ys_index += 1;
                }
                for q in q {
                    y[ys_index] = d2 * (q >> 4) as f32 - m2;
                    ys_index += 1;
                }
                is += 2;
            }
        }
    }
}

// https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
impl GgmlType for BlockQ5K {
    const DTYPE: GgmlDType = GgmlDType::Q5K;
    const BLCK_SIZE: usize = QK_K;

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L793
    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (block, x) in group_for_quantization(xs, ys) {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];

            for (j, x_scale_slice) in x.as_chunks::<32>().0.iter().enumerate() {
                (scales[j], mins[j]) = make_qkx1_quants(31, 5, x_scale_slice);
            }

            // get max scale and max min and ensure they are >= 0.0
            let max_scale = scales.iter().fold(0.0, |max, &val| val.max(max));
            let max_min = mins.iter().fold(0.0, |max, &val| val.max(max));

            let inv_scale = if max_scale > 0.0 {
                63.0 / max_scale
            } else {
                0.0
            };
            let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
            for j in 0..QK_K / 32 {
                let ls = nearest_int(inv_scale * scales[j]).min(63) as u8;
                let lm = nearest_int(inv_min * mins[j]).min(63) as u8;
                if j < 4 {
                    block.scales[j] = ls;
                    block.scales[j + 4] = lm;
                } else {
                    block.scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
                    block.scales[j - 4] |= (ls >> 4) << 6;
                    block.scales[j] |= (lm >> 4) << 6;
                }
            }
            block.d = f16::from_f32(max_scale / 63.0);
            block.dmin = f16::from_f32(max_min / 63.0);

            let mut l: [u8; QK_K] = [0; QK_K];
            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d == 0.0 {
                    continue;
                }
                let dm = block.dmin.to_f32() * m as f32;
                for ii in 0..32 {
                    let ll = nearest_int((x[32 * j + ii] + dm) / d);
                    l[32 * j + ii] = ll.clamp(0, 31) as u8;
                }
            }

            let qh = &mut block.qh;
            let ql = &mut block.qs;
            qh.fill(0);

            let mut m1 = 1;
            let mut m2 = 2;
            for n in (0..QK_K).step_by(64) {
                let offset = (n / 64) * 32;
                for j in 0..32 {
                    let mut l1 = l[n + j];
                    if l1 > 15 {
                        l1 -= 16;
                        qh[j] |= m1;
                    }
                    let mut l2 = l[n + j + 32];
                    if l2 > 15 {
                        l2 -= 16;
                        qh[j] |= m2;
                    }
                    ql[offset + j] = l1 | (l2 << 4);
                }
                m1 <<= 2;
                m2 <<= 2;
            }
        }
    }

    fn from_float_imatrix(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        for (sblk_idx, (block, x)) in group_for_quantization(xs, ys).into_iter().enumerate() {
            let mut mins: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut scales: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut weights: [f32; 32] = [0.0; 32];
            let mut sw: [f32; QK_K / 32] = [0.0; QK_K / 32];
            let mut ls: [u8; QK_K / 32] = [0; QK_K / 32];
            let mut lm: [u8; QK_K / 32] = [0; QK_K / 32];

            let sum_x2 = x.iter().map(|x| x * x).sum::<f32>();
            let sigma2 = 2. * sum_x2 / QK_K as f32;

            for (j, x_scale_slice) in x.as_chunks::<32>().0.iter().enumerate() {
                for (l, (w_elem, x_elem)) in weights.iter_mut().zip(x_scale_slice).enumerate() {
                    let imatrix_row = sblk_idx % (n_per_row / QK_K);
                    let imatrix_w = imatrix_weights[imatrix_row * QK_K + 32 * j + l];
                    *w_elem = imatrix_w * (sigma2 + x_elem * x_elem).sqrt();
                }
                let sumw = weights.iter().sum::<f32>();
                sw[j] = sumw;
                (scales[j], mins[j]) =
                    make_qkx3_quants(31, x_scale_slice, Some(&weights), -0.9, 0.05, 36, false);
            }

            let d_block = make_qp_quants(QK_K / 32, 63, &scales, &mut ls, &sw);
            let m_block = make_qp_quants(QK_K / 32, 63, &mins, &mut lm, &sw);
            for j in 0..QK_K / 32 {
                let ls_val = ls[j].min(63);
                let lm_val = lm[j].min(63);
                if j < 4 {
                    block.scales[j] = ls_val;
                    block.scales[j + 4] = lm_val;
                } else {
                    block.scales[j + 4] = (ls_val & 0xF) | ((lm_val & 0xF) << 4);
                    block.scales[j - 4] |= (ls_val >> 4) << 6;
                    block.scales[j] |= (lm_val >> 4) << 6;
                }
            }

            block.d = f16::from_f32(d_block);
            block.dmin = f16::from_f32(m_block);

            let mut l: [u8; QK_K] = [0; QK_K];
            for j in 0..QK_K / 32 {
                let (sc, m) = get_scale_min_k4(j, &block.scales);
                let d = block.d.to_f32() * sc as f32;
                if d != 0.0 {
                    let dm = block.dmin.to_f32() * m as f32;
                    for ii in 0..32 {
                        let l_val = nearest_int((x[32 * j + ii] + dm) / d);
                        l[32 * j + ii] = l_val.clamp(0, 31) as u8;
                    }
                }
            }

            let qh = &mut block.qh;
            let ql = &mut block.qs;
            qh.fill(0);

            let mut m1 = 1;
            let mut m2 = 2;
            for n in (0..QK_K).step_by(64) {
                let offset = (n / 64) * 32;
                for j in 0..32 {
                    let mut l1 = l[n + j];
                    if l1 > 15 {
                        l1 -= 16;
                        qh[j] |= m1;
                    }
                    let mut l2 = l[n + j + 32];
                    if l2 > 15 {
                        l2 -= 16;
                        qh[j] |= m2;
                    }
                    ql[offset + j] = l1 | (l2 << 4);
                }
                m1 <<= 2;
                m2 <<= 2;
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L928
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (block, y) in group_for_dequantization(xs, ys) {
            let d = block.d.to_f32();
            let min = block.dmin.to_f32();
            let ql = &block.qs;
            let qh = &block.qh;
            let mut is = 0;
            let mut u1 = 1;
            let mut u2 = 2;
            let mut ys_index = 0;

            for j in (0..QK_K).step_by(64) {
                let ql = &ql[j / 2..j / 2 + 32];
                let (sc, m) = get_scale_min_k4(is, &block.scales);
                let d1 = d * sc as f32;
                let m1 = min * m as f32;
                let (sc, m) = get_scale_min_k4(is + 1, &block.scales);
                let d2 = d * sc as f32;
                let m2 = min * m as f32;
                for (ql, qh) in ql.iter().zip(qh) {
                    let to_add = if qh & u1 != 0 { 16f32 } else { 0f32 };
                    y[ys_index] = d1 * ((ql & 0xF) as f32 + to_add) - m1;
                    ys_index += 1;
                }
                for (ql, qh) in ql.iter().zip(qh) {
                    let to_add = if qh & u2 != 0 { 16f32 } else { 0f32 };
                    y[ys_index] = d2 * ((ql >> 4) as f32 + to_add) - m2;
                    ys_index += 1;
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
    }
}

impl GgmlType for BlockQ6K {
    const DTYPE: GgmlDType = GgmlDType::Q6K;
    const BLCK_SIZE: usize = QK_K;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len() * Self::BLCK_SIZE,
            "quantize_row_q6k: size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        let mut l = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];
        let mut x = xs.as_ptr();
        let l = l.as_mut_ptr();
        unsafe {
            for y in ys.iter_mut() {
                let mut max_scale = 0f32;
                let mut max_abs_scale = 0f32;
                for (ib, scale_) in scales.iter_mut().enumerate() {
                    let scale =
                        make_qx_quants(16, 32, x.add(16 * ib), l.add(16 * ib), 1, std::ptr::null());
                    *scale_ = scale;
                    let abs_scale = scale.abs();
                    if abs_scale > max_abs_scale {
                        max_abs_scale = abs_scale;
                        max_scale = scale
                    }
                }

                let iscale = -128f32 / max_scale;
                y.d = f16::from_f32(1.0 / iscale);

                for (y_scale, scale) in y.scales.iter_mut().zip(scales.iter()) {
                    *y_scale = nearest_int(iscale * scale).min(127) as i8
                }

                for (j, &y_scale) in y.scales.iter().enumerate() {
                    let d = y.d.to_f32() * y_scale as f32;
                    if d == 0. {
                        continue;
                    }
                    for ii in 0..16 {
                        let ll = nearest_int(*x.add(16 * j + ii) / d).clamp(-32, 31);
                        *l.add(16 * j + ii) = (ll + 32) as i8
                    }
                }

                let mut ql = y.ql.as_mut_ptr();
                let mut qh = y.qh.as_mut_ptr();

                for j in (0..QK_K).step_by(128) {
                    for l_idx in 0..32 {
                        let q1 = *l.add(j + l_idx) & 0xF;
                        let q2 = *l.add(j + l_idx + 32) & 0xF;
                        let q3 = *l.add(j + l_idx + 64) & 0xF;
                        let q4 = *l.add(j + l_idx + 96) & 0xF;
                        *ql.add(l_idx) = (q1 | (q3 << 4)) as u8;
                        *ql.add(l_idx + 32) = (q2 | (q4 << 4)) as u8;
                        *qh.add(l_idx) = ((*l.add(j + l_idx) >> 4)
                            | ((*l.add(j + l_idx + 32) >> 4) << 2)
                            | ((*l.add(j + l_idx + 64) >> 4) << 4)
                            | ((*l.add(j + l_idx + 96) >> 4) << 6))
                            as u8;
                    }
                    ql = ql.add(64);
                    qh = qh.add(32);
                }

                x = x.add(QK_K)
            }
        }
    }

    fn from_float_imatrix(xs: &[f32], ys: &mut [Self], imatrix_weights: &[f32], n_per_row: usize) {
        debug_assert_eq!(
            xs.len(),
            ys.len() * Self::BLCK_SIZE,
            "quantize_row_q6k imatrix: size mismatch {} {} {}",
            xs.len(),
            ys.len(),
            Self::BLCK_SIZE
        );
        let mut l = [0i8; QK_K];
        let mut scales = [0f32; QK_K / 16];
        let mut x = xs.as_ptr();
        let imatrix_weights = imatrix_weights.as_ptr();
        let l = l.as_mut_ptr();
        unsafe {
            for (sblk_idx, y) in ys.iter_mut().enumerate() {
                let mut max_scale = 0f32;
                let mut max_abs_scale = 0f32;
                for (ib, scale_) in scales.iter_mut().enumerate() {
                    let imatrix_row = sblk_idx % (n_per_row / QK_K);
                    let scale = make_qx_quants(
                        16,
                        32,
                        x.add(16 * ib),
                        l.add(16 * ib),
                        1,
                        imatrix_weights.add(QK_K * imatrix_row + 16 * ib),
                    );
                    *scale_ = scale;
                    let abs_scale = scale.abs();
                    if abs_scale > max_abs_scale {
                        max_abs_scale = abs_scale;
                        max_scale = scale
                    }
                }

                let iscale = -128f32 / max_scale;
                y.d = f16::from_f32(1.0 / iscale);

                for (y_scale, scale) in y.scales.iter_mut().zip(scales.iter()) {
                    *y_scale = nearest_int(iscale * scale).min(127) as i8
                }

                for (j, &y_scale) in y.scales.iter().enumerate() {
                    let d = y.d.to_f32() * y_scale as f32;
                    if d == 0. {
                        continue;
                    }
                    for ii in 0..16 {
                        let ll = nearest_int(*x.add(16 * j + ii) / d).clamp(-32, 31);
                        *l.add(16 * j + ii) = (ll + 32) as i8
                    }
                }

                let mut ql = y.ql.as_mut_ptr();
                let mut qh = y.qh.as_mut_ptr();

                for j in (0..QK_K).step_by(128) {
                    for l_idx in 0..32 {
                        let q1 = *l.add(j + l_idx) & 0xF;
                        let q2 = *l.add(j + l_idx + 32) & 0xF;
                        let q3 = *l.add(j + l_idx + 64) & 0xF;
                        let q4 = *l.add(j + l_idx + 96) & 0xF;
                        *ql.add(l_idx) = (q1 | (q3 << 4)) as u8;
                        *ql.add(l_idx + 32) = (q2 | (q4 << 4)) as u8;
                        *qh.add(l_idx) = ((*l.add(j + l_idx) >> 4)
                            | ((*l.add(j + l_idx + 32) >> 4) << 2)
                            | ((*l.add(j + l_idx + 64) >> 4) << 4)
                            | ((*l.add(j + l_idx + 96) >> 4) << 6))
                            as u8;
                    }
                    ql = ql.add(64);
                    qh = qh.add(32);
                }

                x = x.add(QK_K)
            }
        }
    }

    // https://github.com/ggerganov/llama.cpp/blob/8183159cf3def112f6d1fe94815fce70e1bffa12/k_quants.c#L1067
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK_K),
            "dequantize_row_q6k: {k} is not divisible by {QK_K}"
        );

        for (idx_x, x) in xs.iter().enumerate() {
            let d = x.d.to_f32();
            let ql = &x.ql;
            let qh = &x.qh;
            let sc = &x.scales;
            for n in (0..QK_K).step_by(128) {
                let idx = n / 128;
                let ys = &mut ys[idx_x * QK_K + n..];
                let sc = &sc[8 * idx..];
                let ql = &ql[64 * idx..];
                let qh = &qh[32 * idx..];
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = ((ql[l] & 0xF) | ((qh[l] & 3) << 4)) as i8 - 32;
                    let q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) as i8 - 32;
                    let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) as i8 - 32;
                    let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) as i8 - 32;
                    ys[l] = d * sc[is] as f32 * q1 as f32;
                    ys[l + 32] = d * sc[is + 2] as f32 * q2 as f32;
                    ys[l + 64] = d * sc[is + 4] as f32 * q3 as f32;
                    ys[l + 96] = d * sc[is + 6] as f32 * q4 as f32;
                }
            }
        }
    }
}

impl GgmlType for BlockQ8K {
    const DTYPE: GgmlDType = GgmlDType::Q8K;
    const BLCK_SIZE: usize = QK_K;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        let k = xs.len();
        debug_assert!(
            k.is_multiple_of(QK_K),
            "quantize_row_q8k: {k} is not divisible by {QK_K}"
        );
        for (i, y) in ys.iter_mut().enumerate() {
            let mut max = 0f32;
            let mut amax = 0f32;
            let xs = &xs[i * QK_K..(i + 1) * QK_K];
            for &x in xs.iter() {
                if amax < x.abs() {
                    amax = x.abs();
                    max = x;
                }
            }
            if amax == 0f32 {
                y.d = 0f32;
                y.qs.fill(0)
            } else {
                let iscale = -128f32 / max;
                for (j, q) in y.qs.iter_mut().enumerate() {
                    // ggml uses nearest_int with bit magic here, maybe we want the same
                    // but we would have to test and benchmark it.
                    let v = (iscale * xs[j]).round();
                    *q = v.min(127.) as i8
                }
                for j in 0..QK_K / 16 {
                    let mut sum = 0i32;
                    for ii in 0..16 {
                        sum += y.qs[j * 16 + ii] as i32
                    }
                    y.bsums[j] = sum as i16
                }
                y.d = 1.0 / iscale
            }
        }
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        let k = ys.len();
        debug_assert!(
            k.is_multiple_of(QK_K),
            "dequantize_row_q8k: {k} is not divisible by {QK_K}"
        );
        for (i, x) in xs.iter().enumerate() {
            for (j, &q) in x.qs.iter().enumerate() {
                ys[i * QK_K + j] = x.d * q as f32
            }
        }
    }
}

/// Plain f32 dot product, eight independent lanes so the compiler can vectorise it.
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    let tail: f32 = ra.iter().zip(rb).map(|(x, y)| x * y).sum();
    for (xa, xb) in ca.iter().zip(cb) {
        for l in 0..8 {
            acc[l] += xa[l] * xb[l];
        }
    }
    acc.iter().sum::<f32>() + tail
}

/// `dst[m, n] = lhs[m, k] x rhs_t[n, k]^T` where `rhs_t` holds quantized weight rows.
///
/// The one matmul route for quantized weights: every weight row is converted to f32 (`to_float`; the
/// block scale is applied to each value here, BEFORE any multiply) and dotted with the f32
/// activation row. There is no activation quantisation, no integer multiply, and no scaling of an
/// integer block sum: the integer dot kernels (Q8_0 x Q8_0, Q4_0 x Q8_0, the K-quant two-level
/// paths and their AVX2 and NEON versions) were removed from every published crate on 2026-11-01
/// (ROADMAP board 307).
pub fn matmul<T: GgmlType>(
    (m, k, n): (usize, usize, usize),
    lhs: &[f32],
    rhs_t: &[T],
    dst: &mut [f32],
) -> Result<()> {
    if m * k != lhs.len() || m * n != dst.len() {
        fuel_ir::bail!(
            "quantized matmul: bad lengths lhs {} dst {} for ({m},{k},{n})",
            lhs.len(),
            dst.len()
        );
    }
    let k_in_blocks = k.div_ceil(T::BLCK_SIZE);
    // Result laid out column-major (one chunk of `m` values per weight row) so each parallel task
    // owns a disjoint chunk; transposed into `dst` afterwards (a no-op copy for m == 1).
    let mut tmp = vec![0f32; m * n];
    tmp.par_chunks_mut(m)
        .enumerate()
        .with_min_len(64)
        .for_each_init(
            || vec![0f32; k_in_blocks * T::BLCK_SIZE],
            |buf, (col, out)| {
                T::to_float(&rhs_t[col * k_in_blocks..(col + 1) * k_in_blocks], buf);
                for (row, o) in out.iter_mut().enumerate() {
                    *o = dot_f32(&buf[..k], &lhs[row * k..(row + 1) * k]);
                }
            },
        );
    for col in 0..n {
        for row in 0..m {
            dst[row * n + col] = tmp[col * m + row];
        }
    }
    Ok(())
}

/// [`matmul`] with an f16 activation and an f16 result (computed in f32).
pub fn matmul_f16<T: GgmlType>(
    mkn: (usize, usize, usize),
    lhs: &[f16],
    rhs_t: &[T],
    dst: &mut [f16],
) -> Result<()> {
    let (m, k, n) = mkn;
    if m * k != lhs.len() {
        fuel_ir::bail!("unexpected lhs length {} {mkn:?}", lhs.len());
    }
    let lhs32: Vec<f32> = lhs.iter().map(|x| x.to_f32()).collect();
    let mut dst32 = vec![0f32; m * n];
    matmul(mkn, &lhs32, rhs_t, &mut dst32)?;
    for (d, s) in dst.iter_mut().zip(dst32) {
        *d = f16::from_f32(s);
    }
    Ok(())
}

impl GgmlType for f32 {
    const DTYPE: GgmlDType = GgmlDType::F32;
    const BLCK_SIZE: usize = 1;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.copy_from_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.copy_from_slice(xs);
    }
}

impl GgmlType for f16 {
    const DTYPE: GgmlDType = GgmlDType::F16;
    const BLCK_SIZE: usize = 1;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.convert_from_f32_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        xs.convert_to_f32_slice(ys);
    }
}

impl GgmlType for bf16 {
    const DTYPE: GgmlDType = GgmlDType::BF16;
    const BLCK_SIZE: usize = 1;

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        ys.convert_from_f32_slice(xs);
    }

    fn to_float(xs: &[Self], ys: &mut [f32]) {
        debug_assert_eq!(
            xs.len(),
            ys.len(),
            "size mismatch xs {} != ys {}",
            xs.len(),
            ys.len()
        );
        xs.convert_to_f32_slice(ys);
    }
}

#[cfg(test)]
mod matmul_tests {
    use super::*;

    /// A small deterministic matrix: values in about [-1, 1], no two rows alike.
    fn fill(len: usize, seed: u32) -> Vec<f32> {
        (0..len as u32)
            .map(|i| {
                let x = (i
                    .wrapping_mul(2654435761)
                    .wrapping_add(seed.wrapping_mul(40503)))
                    >> 8;
                (x % 2001) as f32 / 1000.0 - 1.0
            })
            .collect()
    }

    /// The matmul is the f32 matmul of the DEQUANTISED weights with the f32 activations (so it
    /// agrees with a plain f32 reference to rounding), for several rows of activations at once.
    #[test]
    fn matmul_equals_the_f32_matmul_of_the_dequantised_weights() {
        let (m, k, n) = (3usize, 64usize, 8usize);
        let w = fill(n * k, 7);
        let x = fill(m * k, 11);
        let mut blocks = vec![BlockQ4_0::zeros(); n * k / QK4_0];
        BlockQ4_0::from_float(&w, &mut blocks);
        let mut deq = vec![0f32; n * k];
        BlockQ4_0::to_float(&blocks, &mut deq);
        let mut want = vec![0f32; m * n];
        for r in 0..m {
            for c in 0..n {
                want[r * n + c] = (0..k).map(|i| x[r * k + i] * deq[c * k + i]).sum();
            }
        }
        let mut got = vec![0f32; m * n];
        matmul((m, k, n), &x, &blocks, &mut got).expect("quantized matmul");
        assert!(
            want.iter().any(|v| v.abs() > 0.1),
            "control: the reference is not all zeros"
        );
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g - w).abs() <= 1e-3 * w.abs().max(1.0),
                "matmul {g} vs reference {w}"
            );
        }
    }

    /// The f16 entry point is the f32 one with the activation rounded to f16 and the result rounded
    /// back, so it agrees with the f32 matmul to f16 precision.
    #[test]
    fn matmul_f16_agrees_with_matmul_to_f16_precision() {
        let (m, k, n) = (2usize, 64usize, 8usize);
        let w = fill(n * k, 3);
        let x = fill(m * k, 5);
        let mut blocks = vec![BlockQ4_0::zeros(); n * k / QK4_0];
        BlockQ4_0::from_float(&w, &mut blocks);
        let x16: Vec<f16> = x.iter().map(|v| f16::from_f32(*v)).collect();
        let x_back: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
        let mut want = vec![0f32; m * n];
        matmul((m, k, n), &x_back, &blocks, &mut want).expect("matmul");
        let mut got = vec![f16::ZERO; m * n];
        matmul_f16((m, k, n), &x16, &blocks, &mut got).expect("matmul_f16");
        assert!(want.iter().any(|v| v.abs() > 0.1), "control: not all zeros");
        for (g, w) in got.iter().zip(&want) {
            assert!(
                (g.to_f32() - w).abs() <= 2e-3 * w.abs().max(1.0),
                "matmul_f16 {g} vs {w}"
            );
        }
    }
}
