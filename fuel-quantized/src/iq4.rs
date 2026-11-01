// Derived in part from llama.cpp / ggml (MIT License, Copyright (c) 2023-2026 The ggml authors).
// The full licence text is in THIRD-PARTY-NOTICES.md at the repository root. No licence identifier line
// is stamped here: the file derives from somebody else's copyright.
//! `IQ4_NL` and `IQ4_XS` block formats (GAP-211 B1).
//!
//! # Provenance
//!
//! These two formats and their dequantization are llama.cpp's / ggml's
//! (MIT License, Copyright (c) 2023-2026 The ggml authors). What is taken from
//! there is the on-disk layout, the 16-entry non-linear codebook and the
//! dequantization arithmetic, each checked against the source at commit
//! `9d57ce456c` (the pin recorded on [`GgmlDType`]):
//!
//! - `ggml/src/ggml-common.h`: `QK4_NL = 32`, `block_iq4_nl` (`ggml_half d;
//!   uint8_t qs[QK4_NL/2]`), `block_iq4_xs` (`ggml_half d; uint16_t scales_h;
//!   uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]`), and `kvalues_iq4nl`.
//! - `ggml/src/ggml-quants.c`: `dequantize_row_iq4_nl`, `dequantize_row_iq4_xs`.
//! - `ggml/include/ggml.h`: `GGML_TYPE_IQ4_NL = 20`, `GGML_TYPE_IQ4_XS = 23`.
//!
//! The quantizer ([`GgmlType::from_float`]) is NOT llama.cpp's: llama.cpp
//! searches for the scale that minimises weighted error. This one picks the
//! scale that maps the block's largest-magnitude value onto the end of the
//! codebook and rounds to the nearest codebook entry. It writes valid blocks
//! llama.cpp reads, but not bit-identical ones, and is less accurate; it
//! exists so the formats satisfy the whole `GgmlType` contract without a
//! panic path.

use crate::k_quants::{GgmlType, QK_K};
use fuel_ir::quantized::GgmlDType;
use half::f16;

pub const QK4_NL: usize = 32;

/// `kvalues_iq4nl` from `ggml-common.h`: the codebook both formats index with a
/// 4-bit code.
pub const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockIq4Nl {
    pub(crate) d: f16,
    pub(crate) qs: [u8; QK4_NL / 2],
}
const _: () = assert!(std::mem::size_of::<BlockIq4Nl>() == 2 + QK4_NL / 2);

#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct BlockIq4Xs {
    pub(crate) d: f16,
    pub(crate) scales_h: u16,
    pub(crate) scales_l: [u8; QK_K / 64],
    pub(crate) qs: [u8; QK_K / 2],
}
const _: () = assert!(std::mem::size_of::<BlockIq4Xs>() == 2 + 2 + QK_K / 64 + QK_K / 2);

/// Index of the codebook entry nearest `v`.
fn nearest_code(v: f32) -> u8 {
    let mut best = 0u8;
    let mut best_err = f32::INFINITY;
    for (i, &k) in KVALUES_IQ4NL.iter().enumerate() {
        let err = (v - f32::from(k)).abs();
        if err < best_err {
            best_err = err;
            best = i as u8;
        }
    }
    best
}

/// The scale that maps the value of largest magnitude onto codebook entry 0
/// (-127): `max / -127`, sign included so the extreme lands on the long end.
fn block_scale(xs: &[f32]) -> f32 {
    let mut amax = 0f32;
    let mut max = 0f32;
    for &x in xs {
        if amax < x.abs() {
            amax = x.abs();
            max = x;
        }
    }
    max / f32::from(KVALUES_IQ4NL[0])
}

/// Quantize 32 values with scale `d` into 16 packed bytes (low nibble =
/// element `j`, high nibble = element `j + 16`, as `dequantize_row_iq4_nl`).
fn pack32(xs: &[f32], d: f32, qs: &mut [u8]) {
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };
    for j in 0..16 {
        qs[j] = nearest_code(xs[j] * id) | (nearest_code(xs[j + 16] * id) << 4);
    }
}

impl GgmlType for BlockIq4Nl {
    const DTYPE: GgmlDType = GgmlDType::Iq4Nl;
    const BLCK_SIZE: usize = QK4_NL;

    // ggml-quants.c `dequantize_row_iq4_nl`.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (b, y) in xs.iter().zip(ys.as_chunks_mut::<QK4_NL>().0.iter_mut()) {
            let d = b.d.to_f32();
            for j in 0..QK4_NL / 2 {
                y[j] = d * f32::from(KVALUES_IQ4NL[(b.qs[j] & 0xf) as usize]);
                y[j + QK4_NL / 2] = d * f32::from(KVALUES_IQ4NL[(b.qs[j] >> 4) as usize]);
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (x, y) in xs.as_chunks::<QK4_NL>().0.iter().zip(ys.iter_mut()) {
            let d = block_scale(x);
            y.d = f16::from_f32(d);
            // Quantize against the scale as stored (f16-rounded), so the
            // codes are chosen for what `to_float` will multiply by.
            pack32(x, y.d.to_f32(), &mut y.qs);
        }
    }
}

impl GgmlType for BlockIq4Xs {
    const DTYPE: GgmlDType = GgmlDType::Iq4Xs;
    const BLCK_SIZE: usize = QK_K;

    // ggml-quants.c `dequantize_row_iq4_xs`.
    fn to_float(xs: &[Self], ys: &mut [f32]) {
        for (b, y) in xs.iter().zip(ys.as_chunks_mut::<QK_K>().0.iter_mut()) {
            let d = b.d.to_f32();
            for ib in 0..QK_K / 32 {
                let ls = ((b.scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32
                    | ((((b.scales_h >> (2 * ib)) & 3) as i32) << 4);
                let dl = d * (ls - 32) as f32;
                let qs = &b.qs[ib * 16..ib * 16 + 16];
                let out = &mut y[ib * 32..ib * 32 + 32];
                for j in 0..16 {
                    out[j] = dl * f32::from(KVALUES_IQ4NL[(qs[j] & 0xf) as usize]);
                    out[j + 16] = dl * f32::from(KVALUES_IQ4NL[(qs[j] >> 4) as usize]);
                }
            }
        }
    }

    fn from_float(xs: &[f32], ys: &mut [Self]) {
        for (x, y) in xs.as_chunks::<QK_K>().0.iter().zip(ys.iter_mut()) {
            // Per-32 sub-block scales, then one f16 super-scale `d` such that
            // each sub-scale is `d * (ls - 32)` with `ls` in 0..=63.
            let mut sub = [0f32; QK_K / 32];
            let mut dmax = 0f32;
            for (ib, s) in sub.iter_mut().enumerate() {
                *s = block_scale(&x[ib * 32..ib * 32 + 32]);
                dmax = dmax.max(s.abs());
            }
            let d = f16::from_f32(dmax / 31.0);
            let df = d.to_f32();
            y.d = d;
            y.scales_h = 0;
            y.scales_l = [0; QK_K / 64];
            for (ib, &s) in sub.iter().enumerate() {
                let ls = if df != 0.0 {
                    ((s / df).round() as i32).clamp(-32, 31) + 32
                } else {
                    32
                } as u16;
                y.scales_l[ib / 2] |= ((ls & 0xf) as u8) << (4 * (ib % 2));
                y.scales_h |= (ls >> 4) << (2 * ib);
                let dl = df * (i32::from(ls) - 32) as f32;
                pack32(
                    &x[ib * 32..ib * 32 + 32],
                    dl,
                    &mut y.qs[ib * 16..ib * 16 + 16],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The codebook as `ggml-common.h` prints it (`kvalues_iq4nl`), spelled out
    /// a second time so an edit to [`KVALUES_IQ4NL`] cannot silently pass.
    #[test]
    fn codebook_matches_ggml_common_h() {
        let printed: [i8; 16] = [
            -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
        ];
        assert_eq!(KVALUES_IQ4NL, printed);
    }

    /// Hand-evaluated `dequantize_row_iq4_nl`: d = 0.5, byte j = (j) | (15-j)<<4,
    /// so y[j] = 0.5*kv[j] and y[j+16] = 0.5*kv[15-j]. Spot values are
    /// literal arithmetic (kv[0] = -127 -> -63.5; kv[15] = 113 -> 56.5), not
    /// read back through the table under test.
    #[test]
    fn iq4_nl_dequantizes_as_the_ggml_formula() {
        let mut qs = [0u8; 16];
        for (j, q) in qs.iter_mut().enumerate() {
            *q = j as u8 | ((15 - j as u8) << 4);
        }
        let blk = BlockIq4Nl {
            d: f16::from_f32(0.5),
            qs,
        };
        let mut y = [0f32; 32];
        BlockIq4Nl::to_float(&[blk], &mut y);
        assert_eq!(y[0], -63.5);
        assert_eq!(y[15], 56.5);
        assert_eq!(y[16], 56.5); // high nibble of byte 0 is 15
        assert_eq!(y[31], -63.5); // high nibble of byte 15 is 0
        assert_eq!(y[8], 0.5); // kv[8] = 1
    }

    /// Hand-evaluated `dequantize_row_iq4_xs`: d = 2.0, sub-block 0 has
    /// ls = 33 (scales_l[0] low nibble 1, scales_h bits 0..2 = 0b10) so
    /// dl = 2*(33-32) = 2; sub-block 1 has ls = 30 (scales_l[0] high nibble
    /// 14, scales_h bits 2..4 = 0b01 -> 14 | 16 = 30) so dl = 2*(30-32) = -4.
    #[test]
    fn iq4_xs_dequantizes_as_the_ggml_formula() {
        let mut blk = BlockIq4Xs::zeros();
        blk.d = f16::from_f32(2.0);
        blk.scales_l[0] = 0xE1;
        blk.scales_h = 0b0000_0110; // sub-block0: 0b10, sub-block1: 0b01
        blk.qs[0] = 0x80; // sub-block 0, j=0: lo code 0 (-127), hi code 8 (1)
        blk.qs[16] = 0x0F; // sub-block 1, j=0: lo code 15 (113), hi code 0 (-127)
        let mut y = vec![0f32; QK_K];
        BlockIq4Xs::to_float(&[blk], &mut y);
        assert_eq!(y[0], 2.0 * -127.0);
        assert_eq!(y[16], 2.0 * 1.0);
        assert_eq!(y[32], -4.0 * 113.0);
        assert_eq!(y[48], -4.0 * -127.0);
    }

    fn wave(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) * 0.37).sin() * 3.0 + ((i as f32) * 0.11).cos())
            .collect()
    }

    /// The quantizer's blocks must round-trip through `to_float` within the
    /// codebook's resolution (largest gap 24 of 127 -> half-gap ~ 0.10 of the
    /// block max), for both formats and for a block whose max is negative.
    #[test]
    fn quantize_then_dequantize_stays_within_codebook_resolution() {
        for sign in [1.0f32, -1.0] {
            let x: Vec<f32> = wave(QK_K).into_iter().map(|v| v * sign).collect();
            let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));

            let mut nl = vec![BlockIq4Nl::zeros(); QK_K / QK4_NL];
            BlockIq4Nl::from_float(&x, &mut nl);
            let mut y = vec![0f32; QK_K];
            BlockIq4Nl::to_float(&nl, &mut y);
            let nl_err = x
                .iter()
                .zip(&y)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));

            let mut xs = vec![BlockIq4Xs::zeros()];
            BlockIq4Xs::from_float(&x, &mut xs);
            BlockIq4Xs::to_float(&xs, &mut y);
            let xs_err = x
                .iter()
                .zip(&y)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()));

            assert!(nl_err <= 0.14 * amax, "iq4_nl err {nl_err} vs amax {amax}");
            assert!(xs_err <= 0.20 * amax, "iq4_xs err {xs_err} vs amax {amax}");
            // Control: the bound is not vacuous -- an all-zero block is far off.
            assert!(amax > 1.0);
        }
    }

    /// An all-zero input must not divide by zero or produce NaN.
    #[test]
    fn all_zero_input_quantizes_to_zero() {
        let x = vec![0f32; QK_K];
        let mut xs = vec![BlockIq4Xs::zeros()];
        BlockIq4Xs::from_float(&x, &mut xs);
        let mut y = vec![1f32; QK_K];
        BlockIq4Xs::to_float(&xs, &mut y);
        assert!(y.iter().all(|v| *v == 0.0));
    }

    /// The quantized matmul against IQ4_NL weights equals the dot of the dequantized row with the f32
    /// activation row.
    #[test]
    fn matmul_matches_the_dequantized_dot() {
        let x = wave(QK_K);
        let a: Vec<f32> = wave(QK_K).iter().map(|v| v * 0.5 + 0.1).collect();
        let mut w = vec![BlockIq4Nl::zeros(); QK_K / QK4_NL];
        BlockIq4Nl::from_float(&x, &mut w);
        let mut wf = vec![0f32; QK_K];
        BlockIq4Nl::to_float(&w, &mut wf);
        let want: f32 = wf.iter().zip(&a).map(|(p, r)| p * r).sum();
        let mut got = [0f32; 1];
        crate::k_quants::matmul((1, QK_K, 1), &a, &w, &mut got).expect("matmul");
        assert!(
            (got[0] - want).abs() <= 1e-3 * want.abs().max(1.0),
            "{} vs {want}",
            got[0]
        );
    }
}
