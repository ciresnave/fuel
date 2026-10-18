// Derived in part from llama.cpp / ggml (MIT License, Copyright (c) 2023-2026 The ggml authors).
// The full licence text is in THIRD-PARTY-NOTICES.md at the repository root. No licence identifier line
// is stamped here: the file derives from somebody else's copyright.
//! `TQ1_0`, `TQ2_0` (ternary) and `MXFP4`, `NVFP4` (4-bit float) block formats
//! (GAP-211 C). DECODE-ONLY.
//!
//! # Provenance
//!
//! Block layouts and dequantization are llama.cpp's / ggml's (MIT License,
//! Copyright (c) 2023-2026 The ggml authors), checked against the source at
//! commit `9d57ce456c` (the pin recorded on [`GgmlDType`]):
//!
//! - `ggml/src/ggml-common.h`: `block_tq1_0`, `block_tq2_0`, `block_mxfp4`,
//!   `block_nvfp4`, `kvalues_fp4`.
//! - `ggml/src/ggml-quants.c`: `dequantize_row_tq1_0`, `dequantize_row_tq2_0`,
//!   `dequantize_row_mxfp4`, `dequantize_row_nvfp4`.
//! - `ggml/src/ggml-impl.h`: `ggml_e8m0_to_fp32_half`, `ggml_ue4m3_to_fp32`.
//! - `ggml/include/ggml.h`: `GGML_TYPE_TQ1_0 = 34`, `TQ2_0 = 35`, `MXFP4 = 39`,
//!   `NVFP4 = 40`.
//!
//! # Decode-only
//!
//! Like the grid-table IQ formats, these load, dequantize and multiply
//! (dequantize-then-dot) but cannot be produced here:
//! `GgmlDType::supports_quantize` is `false` and the fallible quantize entry
//! points refuse them first.

use crate::k_quants::{BlockQ8_0, BlockQ8K, GgmlType, QK_K};
use fuel_ir::quantized::GgmlDType;
use half::f16;

pub const QK_MXFP4: usize = 32;
pub const QK_NVFP4: usize = 64;
const QK_NVFP4_SUB: usize = 16;

/// `kvalues_fp4` from `ggml-common.h`: E2M1 values, doubled (`kvalues_mxfp4` is
/// an alias).
pub const KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockTq1_0 {
    pub(crate) qs: [u8; (QK_K - 4 * QK_K / 64) / 5],
    pub(crate) qh: [u8; QK_K / 64],
    pub(crate) d: f16,
}
const _: () = assert!(std::mem::size_of::<BlockTq1_0>() == 54);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockTq2_0 {
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) d: f16,
}
const _: () = assert!(std::mem::size_of::<BlockTq2_0>() == 66);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockMxfp4 {
    pub(crate) e: u8,
    pub(crate) qs: [u8; QK_MXFP4 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockMxfp4>() == 17);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockNvfp4 {
    pub(crate) d: [u8; QK_NVFP4 / QK_NVFP4_SUB],
    pub(crate) qs: [u8; QK_NVFP4 / 2],
}
const _: () = assert!(std::mem::size_of::<BlockNvfp4>() == 36);

/// `ggml_e8m0_to_fp32_half`: half of `2^(x-127)`, with the two smallest codes
/// as denormals. Code 255 (NaN upstream) is not special-cased upstream either.
fn e8m0_to_f32_half(x: u8) -> f32 {
    let bits: u32 = if x < 2 {
        0x0020_0000u32 << x
    } else {
        u32::from(x - 1) << 23
    };
    f32::from_bits(bits)
}

/// `x * 2^e` for an exponent in the normal range; exact (no `powi` rounding).
fn ldexp(x: f32, e: i32) -> f32 {
    x * f32::from_bits(((e + 127) as u32) << 23)
}

/// `ggml_ue4m3_to_fp32`: unsigned 4-exponent-bit (bias 7), 3-mantissa-bit float,
/// times 0.5 to match the doubled `kvalues_fp4`. Codes 0 and 0x7F decode to 0.
fn ue4m3_to_f32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = i32::from((x >> 3) & 0xF);
    let man = f32::from(x & 0x7);
    let raw = if exp == 0 {
        ldexp(man, -9)
    } else {
        ldexp(1.0 + man / 8.0, exp - 7)
    };
    raw * 0.5
}

/// One base-3 digit of a TQ1_0 byte, as upstream extracts it: -1, 0 or +1.
fn tq1_digit(byte: u8, pow3: u8) -> f32 {
    let q = byte.wrapping_mul(pow3);
    let xi = ((u16::from(q) * 3) >> 8) as i16;
    f32::from(xi - 1)
}

impl BlockTq1_0 {
    fn decode(&self, y: &mut [f32]) {
        const POW3: [u8; 5] = [1, 3, 9, 27, 81];
        let d = self.d.to_f32();
        let mut out = 0;
        // 48 bytes of `qs`: one run of 32, then one of 16 (upstream's two loops).
        let (head, tail) = self.qs.split_at(self.qs.len() - self.qs.len() % 32);
        let runs32 = head.as_chunks::<32>().0.iter().map(|c| &c[..]);
        let runs16 = tail.as_chunks::<16>().0.iter().map(|c| &c[..]);
        for chunk in runs32.chain(runs16) {
            for p in POW3 {
                for q in chunk {
                    y[out] = tq1_digit(*q, p) * d;
                    out += 1;
                }
            }
        }
        for p in &POW3[..4] {
            for q in &self.qh {
                y[out] = tq1_digit(*q, *p) * d;
                out += 1;
            }
        }
        debug_assert_eq!(out, QK_K);
    }
}

impl BlockTq2_0 {
    fn decode(&self, y: &mut [f32]) {
        let d = self.d.to_f32();
        let mut out = 0;
        for chunk in self.qs.as_chunks::<32>().0 {
            for l in 0..4 {
                for q in chunk {
                    let v = ((q >> (l * 2)) & 3) as i8;
                    y[out] = f32::from(v - 1) * d;
                    out += 1;
                }
            }
        }
    }
}

impl BlockMxfp4 {
    fn decode(&self, y: &mut [f32]) {
        let d = e8m0_to_f32_half(self.e);
        for (j, q) in self.qs.iter().enumerate() {
            y[j] = f32::from(KVALUES_FP4[usize::from(q & 0x0F)]) * d;
            y[j + QK_MXFP4 / 2] = f32::from(KVALUES_FP4[usize::from(q >> 4)]) * d;
        }
    }
}

impl BlockNvfp4 {
    fn decode(&self, y: &mut [f32]) {
        for (s, scale) in self.d.iter().enumerate() {
            let d = ue4m3_to_f32(*scale);
            let yb = &mut y[s * QK_NVFP4_SUB..(s + 1) * QK_NVFP4_SUB];
            let qs = &self.qs[s * QK_NVFP4_SUB / 2..(s + 1) * QK_NVFP4_SUB / 2];
            for (j, q) in qs.iter().enumerate() {
                yb[j] = f32::from(KVALUES_FP4[usize::from(q & 0x0F)]) * d;
                yb[j + QK_NVFP4_SUB / 2] = f32::from(KVALUES_FP4[usize::from(q >> 4)]) * d;
            }
        }
    }
}

/// Reference dot product: dequantize both sides, multiply. Correct by
/// construction from `to_float`.
fn dot_via_to_float<X: GgmlType, Y: GgmlType>(n: usize, xs: &[X], ys: &[Y]) -> f32 {
    let mut a = vec![0f32; n];
    let mut b = vec![0f32; n];
    X::to_float(&xs[..n / X::BLCK_SIZE], &mut a);
    Y::to_float(&ys[..n / Y::BLCK_SIZE], &mut b);
    a.iter().zip(&b).map(|(x, y)| x * y).sum()
}

macro_rules! decode_only_type {
    ($blk:ty, $dtype:expr, $qk:expr, $vecdot:ty) => {
        impl GgmlType for $blk {
            const DTYPE: GgmlDType = $dtype;
            const BLCK_SIZE: usize = $qk;
            type VecDotType = $vecdot;

            fn to_float(xs: &[Self], ys: &mut [f32]) {
                for (b, y) in xs.iter().zip(ys.chunks_exact_mut($qk)) {
                    b.decode(y);
                }
            }

            fn from_float(_xs: &[f32], _ys: &mut [Self]) {
                // Unreachable through the fallible entry points, which check
                // `GgmlDType::supports_quantize` first.
                panic!(
                    "{:?} is decode-only: Fuel cannot quantize to it (GAP-211 C)",
                    $dtype
                );
            }

            fn vec_dot(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
                dot_via_to_float(n, xs, ys)
            }

            fn vec_dot_unopt(n: usize, xs: &[Self], ys: &[Self::VecDotType]) -> f32 {
                dot_via_to_float(n, xs, ys)
            }
        }
    };
}

decode_only_type!(BlockTq1_0, GgmlDType::Tq1_0, QK_K, BlockQ8K);
decode_only_type!(BlockTq2_0, GgmlDType::Tq2_0, QK_K, BlockQ8K);
decode_only_type!(BlockMxfp4, GgmlDType::Mxfp4, QK_MXFP4, BlockQ8_0);
decode_only_type!(BlockNvfp4, GgmlDType::Nvfp4, QK_NVFP4, BlockQ8_0);

#[cfg(test)]
mod tests {
    use super::*;

    fn one<T: GgmlType>(b: T) -> Vec<f32> {
        let mut y = vec![0f32; T::BLCK_SIZE];
        T::to_float(&[b], &mut y);
        y
    }

    #[test]
    fn geometry_agrees_with_ggml_dtype() {
        fn check<T: GgmlType>() {
            assert_eq!(
                std::mem::size_of::<T>(),
                T::DTYPE.type_size(),
                "{:?}",
                T::DTYPE
            );
            assert_eq!(T::BLCK_SIZE, T::DTYPE.block_size(), "{:?}", T::DTYPE);
            assert!(!T::DTYPE.supports_quantize());
            assert!(T::DTYPE.check_quantize().is_err());
        }
        check::<BlockTq1_0>();
        check::<BlockTq2_0>();
        check::<BlockMxfp4>();
        check::<BlockNvfp4>();
        assert!(GgmlDType::Q4K.supports_quantize());
    }

    /// `ggml_e8m0_to_fp32_half`: code 127 is 2^0 halved; 0 and 1 are the denormals.
    #[test]
    fn e8m0_half_scale_cases() {
        assert_eq!(e8m0_to_f32_half(127), 0.5);
        assert_eq!(e8m0_to_f32_half(128), 1.0);
        assert_eq!(e8m0_to_f32_half(1), f32::from_bits(0x0040_0000));
        assert_eq!(e8m0_to_f32_half(0), f32::from_bits(0x0020_0000));
    }

    /// `ggml_ue4m3_to_fp32`, hand-evaluated: 0x38 = 1.0 * 0.5; 0x01 is the
    /// smallest denormal 2^-9 * 0.5; 0x7E = 1.75 * 2^8 * 0.5; 0 and 0x7F are 0.
    #[test]
    fn ue4m3_scale_cases() {
        assert_eq!(ue4m3_to_f32(0x38), 0.5);
        assert_eq!(ue4m3_to_f32(0x01), 2f32.powi(-10));
        assert_eq!(ue4m3_to_f32(0x7E), 224.0);
        assert_eq!(ue4m3_to_f32(0x00), 0.0);
        assert_eq!(ue4m3_to_f32(0x7F), 0.0);
    }

    #[test]
    fn mxfp4_dequantizes_as_the_ggml_formula() {
        let mut b = BlockMxfp4::zeros();
        b.e = 128; // d = 1.0
        b.qs[0] = 0x21; // low code 1 -> 1, high code 2 -> 2
        b.qs[1] = 0xF8; // low code 8 -> 0, high code 15 -> -12
        let y = one(b);
        assert_eq!(y[0], 1.0);
        assert_eq!(y[16], 2.0);
        assert_eq!(y[1], 0.0);
        assert_eq!(y[17], -12.0);
        assert_eq!(y[2], 0.0); // untouched
    }

    #[test]
    fn nvfp4_each_subblock_uses_its_own_scale() {
        let mut b = BlockNvfp4::zeros();
        b.d = [0x38, 0x38 + 8, 0x7F, 0x38]; // 0.5, 1.0, 0 (reserved), 0.5
        b.qs[0] = 0x42; // sub-block 0: low code 2 -> 2, high code 4 -> 4
        b.qs[8] = 0x42; // sub-block 1, same codes
        b.qs[16] = 0x42; // sub-block 2, scale 0
        let y = one(b);
        assert_eq!(y[0], 1.0); // 2 * 0.5
        assert_eq!(y[8], 2.0); // 4 * 0.5
        assert_eq!(y[16], 2.0); // 2 * 1.0
        assert_eq!(y[24], 4.0);
        assert_eq!(y[32], 0.0);
        assert_eq!(y[40], 0.0);
    }

    #[test]
    fn tq2_0_dequantizes_as_the_ggml_formula() {
        let mut b = BlockTq2_0::zeros();
        b.d = f16::from_f32(2.0);
        b.qs[0] = 0b11_10_01_00;
        let y = one(b);
        assert_eq!(y[0], -2.0); // q 0
        assert_eq!(y[32], 0.0); // q 1
        assert_eq!(y[64], 2.0); // q 2
        assert_eq!(y[96], 4.0); // q 3
        assert_eq!(y[1], -2.0); // untouched byte: q 0
    }

    /// Byte 171 is the trit run (2,0,0,0,0): ceil(162 * 256 / 243). Digit 0 is
    /// +1, digit 1 is -1. Checked in the 32-run, the 16-run and the `qh` tail.
    #[test]
    fn tq1_0_dequantizes_as_the_ggml_formula() {
        let mut b = BlockTq1_0::zeros();
        b.d = f16::from_f32(2.0);
        b.qs[0] = 171; // 32-run
        b.qs[32] = 171; // 16-run
        b.qh[0] = 171;
        let y = one(b);
        assert_eq!(y[0], 2.0);
        assert_eq!(y[32], -2.0);
        assert_eq!(y[160], 2.0); // 16-run starts after 5 * 32
        assert_eq!(y[160 + 16], -2.0);
        assert_eq!(y[240], 2.0); // qh tail starts after 160 + 80
        assert_eq!(y[244], -2.0);
        assert_eq!(y[1], -2.0); // zero byte: all digits -1
    }
}
