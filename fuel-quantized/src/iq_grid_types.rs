// Derived in part from llama.cpp / ggml (MIT License, Copyright (c) 2023-2026 The ggml authors).
// The full licence text is in THIRD-PARTY-NOTICES.md at the repository root. No licence identifier line
// is stamped here: the file derives from somebody else's copyright.
//! `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ1_S` and `IQ1_M` block
//! formats (GAP-211 B2): the grid-table IQ quantizations. DECODE-ONLY.
//!
//! # Provenance
//!
//! Block layouts and dequantization are llama.cpp's / ggml's (MIT License,
//! Copyright (c) 2023-2026 The ggml authors), checked against the source at
//! commit `9d57ce456c` (the pin recorded on [`GgmlDType`]):
//!
//! - `ggml/src/ggml-common.h`: `block_iq2_xxs`, `block_iq2_xs`, `block_iq2_s`,
//!   `block_iq3_xxs`, `block_iq3_s`, `block_iq1_s`, `block_iq1_m`, `IQ1S_DELTA`.
//! - `ggml/src/ggml-quants.c`: `dequantize_row_iq2_xxs` ... `dequantize_row_iq1_m`.
//! - the lookup tables themselves are in [`crate::iq_grids`], generated from
//!   `ggml-common.h` by `scripts/gen_iq_grids.py`.
//!
//! # Decode-only
//!
//! llama.cpp's quantizers for these formats search for grid points under an
//! importance matrix and are out of scope here. [`GgmlType::from_float`] is
//! therefore not provided: it panics naming the type, and the fallible
//! quantize entry points refuse these dtypes first
//! (`GgmlDType::supports_quantize`). A model stored in these formats loads,
//! dequantizes and multiplies (dequantize-then-dot); it cannot be produced here.

use crate::iq_grids::*;
use crate::k_quants::{BlockQ8K, GgmlType, QK_K};
use fuel_ir::quantized::GgmlDType;
use half::f16;

const IQ1S_DELTA: f32 = 0.125;

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockIq2Xxs {
    pub(crate) d: f16,
    pub(crate) qs: [u16; QK_K / 8],
}
const _: () = assert!(std::mem::size_of::<BlockIq2Xxs>() == 66);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockIq2Xs {
    pub(crate) d: f16,
    pub(crate) qs: [u16; QK_K / 8],
    pub(crate) scales: [u8; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIq2Xs>() == 74);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockIq2S {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) qh: [u8; QK_K / 32],
    pub(crate) scales: [u8; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIq2S>() == 82);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockIq3Xxs {
    pub(crate) d: f16,
    pub(crate) qs: [u8; 3 * QK_K / 8],
}
const _: () = assert!(std::mem::size_of::<BlockIq3Xxs>() == 98);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockIq3S {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 4],
    pub(crate) qh: [u8; QK_K / 32],
    pub(crate) signs: [u8; QK_K / 8],
    pub(crate) scales: [u8; QK_K / 64],
}
const _: () = assert!(std::mem::size_of::<BlockIq3S>() == 110);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockIq1S {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK_K / 8],
    pub(crate) qh: [u16; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIq1S>() == 50);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockIq1M {
    pub(crate) qs: [u8; QK_K / 8],
    pub(crate) qh: [u8; QK_K / 16],
    pub(crate) scales: [u8; QK_K / 32],
}
const _: () = assert!(std::mem::size_of::<BlockIq1M>() == 56);

/// `(signs & kmask_iq2xs[j]) ? -1.f : 1.f`.
fn sign(signs: u8, j: usize) -> f32 {
    if signs & KMASK_IQ2XS[j] != 0 {
        -1.0
    } else {
        1.0
    }
}

impl BlockIq2Xxs {
    // ggml-quants.c `dequantize_row_iq2_xxs`.
    pub(crate) fn decode(&self, y: &mut [f32]) {
        let d = self.d.to_f32();
        for ib32 in 0..QK_K / 32 {
            let q = &self.qs[4 * ib32..4 * ib32 + 4];
            let aux0 = u32::from(q[0]) | (u32::from(q[1]) << 16);
            let aux1 = u32::from(q[2]) | (u32::from(q[3]) << 16);
            let aux8 = aux0.to_le_bytes();
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = IQ2XXS_GRID[aux8[l] as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    y[32 * ib32 + 8 * l + j] = db * f32::from(grid[j]) * sign(signs, j);
                }
            }
        }
    }
}

impl BlockIq2Xs {
    // ggml-quants.c `dequantize_row_iq2_xs`.
    pub(crate) fn decode(&self, y: &mut [f32]) {
        let d = self.d.to_f32();
        for ib32 in 0..QK_K / 32 {
            let db = [
                d * (0.5 + (self.scales[ib32] & 0xf) as f32) * 0.25,
                d * (0.5 + (self.scales[ib32] >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let q = self.qs[4 * ib32 + l];
                let grid = IQ2XS_GRID[(q & 511) as usize].to_le_bytes();
                let signs = KSIGNS_IQ2XS[(q >> 9) as usize];
                for j in 0..8 {
                    y[32 * ib32 + 8 * l + j] = db[l / 2] * f32::from(grid[j]) * sign(signs, j);
                }
            }
        }
    }
}

impl BlockIq2S {
    // ggml-quants.c `dequantize_row_iq2_s`.
    pub(crate) fn decode(&self, y: &mut [f32]) {
        let d = self.d.to_f32();
        let signs_base = QK_K / 8;
        for ib32 in 0..QK_K / 32 {
            let db = [
                d * (0.5 + (self.scales[ib32] & 0xf) as f32) * 0.25,
                d * (0.5 + (self.scales[ib32] >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let dl = db[l / 2];
                let idx = usize::from(self.qs[4 * ib32 + l])
                    | ((usize::from(self.qh[ib32]) << (8 - 2 * l)) & 0x300);
                let grid = IQ2S_GRID[idx].to_le_bytes();
                let signs = self.qs[signs_base + 4 * ib32 + l];
                for j in 0..8 {
                    y[32 * ib32 + 8 * l + j] = dl * f32::from(grid[j]) * sign(signs, j);
                }
            }
        }
    }
}

impl BlockIq3Xxs {
    // ggml-quants.c `dequantize_row_iq3_xxs`.
    pub(crate) fn decode(&self, y: &mut [f32]) {
        let d = self.d.to_f32();
        let scales_and_signs = QK_K / 4;
        for ib32 in 0..QK_K / 32 {
            let o = scales_and_signs + 4 * ib32;
            let aux32 =
                u32::from_le_bytes([self.qs[o], self.qs[o + 1], self.qs[o + 2], self.qs[o + 3]]);
            let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let g1 = IQ3XXS_GRID[self.qs[8 * ib32 + 2 * l] as usize].to_le_bytes();
                let g2 = IQ3XXS_GRID[self.qs[8 * ib32 + 2 * l + 1] as usize].to_le_bytes();
                for j in 0..4 {
                    y[32 * ib32 + 8 * l + j] = db * f32::from(g1[j]) * sign(signs, j);
                    y[32 * ib32 + 8 * l + j + 4] = db * f32::from(g2[j]) * sign(signs, j + 4);
                }
            }
        }
    }
}

impl BlockIq3S {
    // ggml-quants.c `dequantize_row_iq3_s`.
    pub(crate) fn decode(&self, y: &mut [f32]) {
        let d = self.d.to_f32();
        for pair in 0..QK_K / 64 {
            let db = [
                d * (1 + 2 * i32::from(self.scales[pair] & 0xf)) as f32,
                d * (1 + 2 * i32::from(self.scales[pair] >> 4)) as f32,
            ];
            for (half, dbh) in db.iter().enumerate() {
                let ib32 = 2 * pair + half;
                let qh = usize::from(self.qh[ib32]);
                let qs = &self.qs[8 * ib32..8 * ib32 + 8];
                for l in 0..4 {
                    let i1 = usize::from(qs[2 * l]) | ((qh << (8 - 2 * l)) & 256);
                    let i2 = usize::from(qs[2 * l + 1]) | ((qh << (7 - 2 * l)) & 256);
                    let g1 = IQ3S_GRID[i1].to_le_bytes();
                    let g2 = IQ3S_GRID[i2].to_le_bytes();
                    let signs = self.signs[4 * ib32 + l];
                    for j in 0..4 {
                        y[32 * ib32 + 8 * l + j] = dbh * f32::from(g1[j]) * sign(signs, j);
                        y[32 * ib32 + 8 * l + j + 4] = dbh * f32::from(g2[j]) * sign(signs, j + 4);
                    }
                }
            }
        }
    }
}

impl BlockIq1S {
    // ggml-quants.c `dequantize_row_iq1_s`.
    pub(crate) fn decode(&self, y: &mut [f32]) {
        let d = self.d.to_f32();
        for ib in 0..QK_K / 32 {
            let qh = self.qh[ib];
            let dl = d * (2 * i32::from((qh >> 12) & 7) + 1) as f32;
            let delta = if qh & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4 {
                let idx =
                    usize::from(self.qs[4 * ib + l]) | ((usize::from(qh >> (3 * l)) & 7) << 8);
                let grid = IQ1S_GRID[idx].to_le_bytes();
                for j in 0..8 {
                    y[32 * ib + 8 * l + j] = dl * (f32::from(grid[j] as i8) + delta);
                }
            }
        }
    }
}

impl BlockIq1M {
    fn sc(&self, i: usize) -> u16 {
        u16::from_le_bytes([self.scales[2 * i], self.scales[2 * i + 1]])
    }

    // ggml-quants.c `dequantize_row_iq1_m`.
    pub(crate) fn decode(&self, y: &mut [f32]) {
        let sc = [self.sc(0), self.sc(1), self.sc(2), self.sc(3)];
        let scale =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f16::from_bits(scale).to_f32();
        for ib in 0..QK_K / 32 {
            let dl1 = d * (2 * i32::from((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) + 1) as f32;
            let dl2 = d * (2 * i32::from((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1) as f32;
            let qs = &self.qs[4 * ib..4 * ib + 4];
            let qh = &self.qh[2 * ib..2 * ib + 2];
            let idx = [
                usize::from(qs[0]) | ((usize::from(qh[0]) << 8) & 0x700),
                usize::from(qs[1]) | ((usize::from(qh[0]) << 4) & 0x700),
                usize::from(qs[2]) | ((usize::from(qh[1]) << 8) & 0x700),
                usize::from(qs[3]) | ((usize::from(qh[1]) << 4) & 0x700),
            ];
            let bit = |b: u8, m: u8| if b & m != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
            let delta = [
                bit(qh[0], 0x08),
                bit(qh[0], 0x80),
                bit(qh[1], 0x08),
                bit(qh[1], 0x80),
            ];
            for l in 0..4 {
                let dl = if l < 2 { dl1 } else { dl2 };
                let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                for j in 0..8 {
                    y[32 * ib + 8 * l + j] = dl * (f32::from(grid[j] as i8) + delta[l]);
                }
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
    ($blk:ty, $dtype:expr) => {
        impl GgmlType for $blk {
            const DTYPE: GgmlDType = $dtype;
            const BLCK_SIZE: usize = QK_K;
            type VecDotType = BlockQ8K;

            fn to_float(xs: &[Self], ys: &mut [f32]) {
                for (b, y) in xs.iter().zip(ys.as_chunks_mut::<QK_K>().0.iter_mut()) {
                    b.decode(y);
                }
            }

            fn from_float(_xs: &[f32], _ys: &mut [Self]) {
                // Unreachable through the fallible entry points, which check
                // `GgmlDType::supports_quantize` first.
                panic!(
                    "{:?} is decode-only: Fuel cannot quantize to it (GAP-211 B2)",
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

decode_only_type!(BlockIq2Xxs, GgmlDType::Iq2Xxs);
decode_only_type!(BlockIq2Xs, GgmlDType::Iq2Xs);
decode_only_type!(BlockIq2S, GgmlDType::Iq2S);
decode_only_type!(BlockIq3Xxs, GgmlDType::Iq3Xxs);
decode_only_type!(BlockIq3S, GgmlDType::Iq3S);
decode_only_type!(BlockIq1S, GgmlDType::Iq1S);
decode_only_type!(BlockIq1M, GgmlDType::Iq1M);

#[cfg(test)]
mod tests {
    use super::*;

    fn one<T: GgmlType>(b: T) -> Vec<f32> {
        let mut y = vec![0f32; QK_K];
        T::to_float(&[b], &mut y);
        y
    }

    /// Properties of the tables that do not depend on reading them back from the
    /// generated file: `ksigns` is "7-bit pattern plus an even-parity bit",
    /// `kmask` is the eight single-bit masks, and each grid only ever holds the
    /// byte values its format is built from.
    #[test]
    fn tables_have_the_structure_ggml_builds_them_with() {
        for (i, s) in KSIGNS_IQ2XS.iter().enumerate() {
            let parity = (i as u32).count_ones() & 1;
            assert_eq!(u32::from(*s), i as u32 | (parity << 7), "ksigns[{i}]");
        }
        for (j, m) in KMASK_IQ2XS.iter().enumerate() {
            assert_eq!(*m, 1u8 << j);
        }
        let all_in = |grid: &[u64], allowed: &[u8]| {
            grid.iter()
                .all(|e| e.to_le_bytes().iter().all(|b| allowed.contains(b)))
        };
        assert!(all_in(&IQ2XXS_GRID, &[0x08, 0x19, 0x2b]));
        assert!(all_in(&IQ2XS_GRID, &[0x08, 0x19, 0x2b]));
        assert!(all_in(&IQ2S_GRID, &[0x08, 0x19, 0x2b]));
        assert!(all_in(&IQ1S_GRID, &[0xff, 0x00, 0x01]));
        assert!(
            IQ3S_GRID
                .iter()
                .all(|e| e.to_le_bytes().iter().all(|b| *b % 2 == 1 && *b <= 15))
        );
    }

    /// Block geometry agrees between `GgmlDType`, the `GgmlType` impls and the
    /// structs (the structs carry `size_of` asserts of their own).
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
        check::<BlockIq2Xxs>();
        check::<BlockIq2Xs>();
        check::<BlockIq2S>();
        check::<BlockIq3Xxs>();
        check::<BlockIq3S>();
        check::<BlockIq1S>();
        check::<BlockIq1M>();
        assert!(GgmlDType::Q4K.supports_quantize());
    }

    /// Hand-evaluated `dequantize_row_iq2_xxs`. d = 1, group 0 picks grid[0]
    /// (all 0x08) then grid[1] (0x2b, then 0x08s), scale nibble 15 gives
    /// db = 15.5 * 0.25 = 3.875, sign index 1 is 0x81 (bits 0 and 7 negative).
    #[test]
    fn iq2_xxs_dequantizes_as_the_ggml_formula() {
        let mut b = BlockIq2Xxs::zeros();
        b.d = f16::from_f32(1.0);
        b.qs[0] = 0x0100; // aux8 = [0, 1, 0, 0]
        b.qs[2] = 1; // sign index of l = 0
        b.qs[3] = 0xF000; // scale 15
        let y = one(b);
        assert_eq!(y[0], -31.0);
        assert_eq!(y[1], 31.0);
        assert_eq!(y[7], -31.0);
        assert_eq!(y[8], 3.875 * 43.0); // grid[1] byte 0 = 0x2b, no sign
        assert_eq!(y[9], 31.0);
        assert_eq!(y[32], 1.0); // untouched group: 0.125 * 8
    }

    /// Hand-evaluated `dequantize_row_iq2_xs`: scales byte 0x21 gives db0 =
    /// 1.5 * 0.25 = 0.375 (l = 0, 1) and db1 = 2.5 * 0.25 = 0.625 (l = 2, 3).
    #[test]
    fn iq2_xs_dequantizes_as_the_ggml_formula() {
        let mut b = BlockIq2Xs::zeros();
        b.d = f16::from_f32(1.0);
        b.scales[0] = 0x21;
        b.qs[0] = (1 << 9) | 1; // grid[1], sign index 1 (0x81)
        let y = one(b);
        assert_eq!(y[0], -(0.375 * 43.0));
        assert_eq!(y[1], 3.0);
        assert_eq!(y[7], -3.0);
        assert_eq!(y[8], 3.0); // l = 1: grid[0]
        assert_eq!(y[16], 5.0); // l = 2 uses db1
    }

    /// Hand-evaluated `dequantize_row_iq2_s`: signs come from qs[32..].
    #[test]
    fn iq2_s_dequantizes_as_the_ggml_formula() {
        let mut b = BlockIq2S::zeros();
        b.d = f16::from_f32(1.0);
        b.scales[0] = 0x21;
        b.qs[0] = 1; // grid[1]
        b.qs[QK_K / 8] = 0x01; // sign byte of l = 0: bit 0 negative
        let y = one(b);
        assert_eq!(y[0], -(0.375 * 43.0));
        assert_eq!(y[1], 3.0);
        assert_eq!(y[8], 3.0);
        assert_eq!(y[16], 5.0);
    }

    /// Hand-evaluated `dequantize_row_iq3_xxs`: scale nibble 3 gives
    /// db = 3.5 * 0.5 = 1.75; grid[1] = 0x04040414 (byte 0 = 0x14).
    #[test]
    fn iq3_xxs_dequantizes_as_the_ggml_formula() {
        let mut b = BlockIq3Xxs::zeros();
        b.d = f16::from_f32(1.0);
        b.qs[0] = 1;
        let aux32: u32 = (3 << 28) | 1; // scale 3, sign index 1 (0x81) for l = 0
        b.qs[QK_K / 4..QK_K / 4 + 4].copy_from_slice(&aux32.to_le_bytes());
        let y = one(b);
        assert_eq!(y[0], -(1.75 * 20.0));
        assert_eq!(y[1], 7.0);
        assert_eq!(y[4], 7.0);
        assert_eq!(y[7], -7.0);
    }

    /// Hand-evaluated `dequantize_row_iq3_s`: scales 0x21 gives db1 = 3,
    /// db2 = 5; grid[1] = 0x01010103 (byte 0 = 3), grid[0] all 1.
    #[test]
    fn iq3_s_dequantizes_as_the_ggml_formula() {
        let mut b = BlockIq3S::zeros();
        b.d = f16::from_f32(1.0);
        b.scales[0] = 0x21;
        b.qs[0] = 1;
        b.signs[0] = 0x01;
        let y = one(b);
        assert_eq!(y[0], -9.0);
        assert_eq!(y[1], 3.0);
        assert_eq!(y[4], 3.0);
        assert_eq!(y[32], 5.0); // second half of the pair uses db2
    }

    /// Hand-evaluated `dequantize_row_iq1_s`: grid[1] = 0xffffffffffffff01
    /// (byte 0 = +1, rest -1), delta +-0.125, dl = d * (2 * s + 1).
    #[test]
    fn iq1_s_dequantizes_as_the_ggml_formula() {
        let mut b = BlockIq1S::zeros();
        b.d = f16::from_f32(1.0);
        b.qs[0] = 1;
        let y = one(b);
        assert_eq!(y[0], 1.125);
        assert_eq!(y[1], -0.875);
        assert_eq!(y[8], -0.875);
        b.qh[0] = 0x9000; // scale field 1 -> dl = 3, sign bit -> delta -0.125
        let y = one(b);
        assert_eq!(y[0], 3.0 * (1.0 - 0.125));
        assert_eq!(y[1], 3.0 * (-1.0 - 0.125));
    }

    /// Hand-evaluated `dequantize_row_iq1_m`: d is assembled from the top
    /// nibbles of the four scale words (0x3c00 = 1.0), the low 9 bits of
    /// scales[0] hold the 3-bit sub-block scales.
    #[test]
    fn iq1_m_dequantizes_as_the_ggml_formula() {
        let mut b = BlockIq1M::zeros();
        b.scales = [0x01, 0x00, 0x00, 0x00, 0x00, 0xc0, 0x00, 0x30];
        b.qs[0] = 1;
        let y = one(b);
        assert_eq!(y[0], 3.0 * 1.125); // dl1 = 2 * 1 + 1
        assert_eq!(y[1], 3.0 * -0.875);
        assert_eq!(y[8], 3.0 * -0.875);
        assert_eq!(y[16], -0.875); // l = 2 uses dl2 = 1
        assert_eq!(y[32], -0.875);
        b.qh[0] = 0x08; // delta[0] negative
        let y = one(b);
        assert_eq!(y[0], 3.0 * 0.875);
    }

    /// The decode-only types refuse quantizing with a typed error naming them.
    #[test]
    fn quantizing_to_a_decode_only_type_is_a_typed_error_naming_it() {
        let e = GgmlDType::Iq2Xxs.check_quantize().unwrap_err().to_string();
        assert!(e.contains("Iq2Xxs") && e.contains("no quantizer"), "{e}");
    }
}
