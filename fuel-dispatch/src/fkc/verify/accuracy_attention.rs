// SPDX-License-Identifier: MIT OR Apache-2.0
//! Measures the registered CUDA `FlashAttn` decode kernel (`flash_decoding`, f16) against an
//! `f64` host attention of the same f16 inputs (ROADMAP row 39).
//!
//! This is evidence, not a claim: nothing here writes the ledger. The live run is
//! `fuel-dev-tools check-cuda-accuracy` (a `#[ignore]`d test would breach the ignore ceiling, and
//! a CUDA `skip` outside one breaches `skip_sites_are_opt_in`). It replaces the test the
//! `flash-attn` contract cites for its `max_absolute: 0.001` (`flash_attn_tests.rs`, which lived
//! in the dissolved `fuel-flash-attn-cuda` crate and exists nowhere now). The dense `flash-attn`
//! and `flash-attn-varlen` contracts bind no kernel (their `fuel_flash_attn_cuda` link registry
//! does not exist), so the decode kernel is the only registered CUDA `FlashAttn` to measure.
//! Half-precision claims, bf16 included, are the next row-39 item.

use fuel_ir::DType;
use fuel_ir::dispatch::OpKind;
use half::f16;

use super::accuracy_common::{Claim, OpAccuracy, Runner, case, scaled};
use super::bit_stability::{HostTensor, VerifyError};
use crate::kernel::{BindingEntry, OpParams};

/// Geometry of one decode case. `k_len < sk` is the live prefix of a capacity-`sk` cache;
/// `hq != hkv` is grouped-query attention.
#[derive(Clone, Copy)]
pub(crate) struct Geometry {
    pub b: usize,
    pub hq: usize,
    pub hkv: usize,
    pub sk: usize,
    pub d: usize,
    pub k_len: usize,
}

const fn geo(b: usize, hq: usize, hkv: usize, sk: usize, d: usize, k_len: usize) -> Geometry {
    Geometry {
        b,
        hq,
        hkv,
        sk,
        d,
        k_len,
    }
}

/// The measured decode cases.
pub(crate) const DECODE_CASES: &[Geometry] = &[
    geo(1, 16, 8, 512, 128, 512),
    geo(1, 16, 8, 512, 128, 300),
    geo(2, 8, 2, 64, 64, 64),
    geo(1, 4, 4, 33, 32, 17),
    geo(1, 8, 8, 2048, 128, 2048),
];

/// Round to f16, as the kernel's input does, and return the bytes and the rounded values.
fn to_f16(shape: Vec<usize>, vals: &[f32]) -> (HostTensor, Vec<f32>) {
    let halves: Vec<f16> = vals.iter().map(|&v| f16::from_f32(v)).collect();
    let rounded = halves.iter().map(|h| h.to_f32()).collect();
    let tensor = HostTensor {
        dtype: DType::F16,
        shape,
        bytes: bytemuck::cast_slice(&halves).to_vec(),
    };
    (tensor, rounded)
}

/// Decode an f16 kernel output to `f32`.
pub(crate) fn f16s(t: &HostTensor) -> Vec<f32> {
    t.bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| f16::from_bits(u16::from_le_bytes(*c)).to_f32())
        .collect()
}

/// One query head of [`reference_decode`]: `softmax(qrow . k * scale) . v` over the first
/// `k_len` rows of the kv head starting at element `base`, accumulated into `cell` (`[d]`).
fn attend_head(
    cell: &mut [f64],
    qrow: &[f32],
    (k, v): (&[f32], &[f32]),
    base: usize,
    k_len: usize,
    scale: f64,
) {
    let d = cell.len();
    let dot = |krow: &[f32]| -> f64 {
        scale
            * qrow
                .iter()
                .zip(krow)
                .map(|(a, b)| *a as f64 * *b as f64)
                .sum::<f64>()
    };
    let scores: Vec<f64> = (0..k_len).map(|j| dot(&k[base + j * d..][..d])).collect();
    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let total: f64 = exps.iter().sum();
    for (j, w) in exps.iter().enumerate() {
        for (c, x) in cell.iter_mut().zip(&v[base + j * d..][..d]) {
            *c += w / total * *x as f64;
        }
    }
}

/// `f64` single-query attention over the live prefix `[0, k_len)` of a `[b, hkv, sk, d]` cache,
/// query head `h` reading kv head `h / (hq / hkv)`. Returns `[b, hq, d]`.
pub(crate) fn reference_decode(
    q: &[f32],
    (k, v): (&[f32], &[f32]),
    g: Geometry,
    scale: f64,
) -> Vec<f64> {
    let Geometry {
        b,
        hq,
        hkv,
        sk,
        d,
        k_len,
    } = g;
    let group = hq / hkv;
    let mut out = vec![0.0; b * hq * d];
    for (row, cell) in out.chunks_exact_mut(d).enumerate() {
        let (bi, h) = (row / hq, row % hq);
        let base = (bi * hkv + h / group) * sk * d;
        attend_head(cell, &q[row * d..][..d], (k, v), base, k_len, scale);
    }
    out
}

/// Measure the f16 `flash_decoding` kernel in `entry` over [`DECODE_CASES`], inputs uniform in
/// `[-1, 1)` rounded to f16. `run` decodes the f16 output to `f32`.
pub(crate) fn measure_flash_decoding(
    run: &Runner,
    entry: &BindingEntry,
    claim: Claim,
) -> Result<OpAccuracy, VerifyError> {
    let mut cases = Vec::new();
    for (idx, &g) in DECODE_CASES.iter().enumerate() {
        let Geometry {
            b,
            hq,
            hkv,
            sk,
            d,
            k_len,
        } = g;
        let seed = 0xA000 + idx as u64 * 3;
        let (qt, q) = to_f16(vec![b, hq, 1, d], &scaled(b * hq * d, seed, 1.0));
        let kv_len = b * hkv * sk * d;
        let (kt, k) = to_f16(vec![b, hkv, sk, d], &scaled(kv_len, seed + 1, 1.0));
        let (vt, v) = to_f16(vec![b, hkv, sk, d], &scaled(kv_len, seed + 2, 1.0));
        let scale = 1.0 / (d as f64).sqrt();
        let params = OpParams::FlashAttn {
            b,
            hq,
            hkv,
            sq: 1,
            sk,
            d,
            k_len,
            softmax_scale: scale as f32,
            causal: false,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
        };
        let got = run(entry, params, &[qt, kt, vt], vec![b, hq, 1, d])?;
        let want = reference_decode(&q, (&k, &v), g, scale);
        cases.push(case(
            format!("b={b} hq={hq} hkv={hkv} sk={sk} d={d} k_len={k_len}"),
            &got,
            &want,
        ));
    }
    Ok(OpAccuracy::new(
        OpKind::FlashAttn,
        &[DType::F16; 4],
        claim,
        entry,
        cases,
        "f64 softmax(q.k * scale).v of the same f16 inputs",
        "uniform [-1, 1) rounded to f16",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fkc::verify::accuracy_common::find_entry;
    use crate::kernel::KernelBindingTable;
    use fuel_ir::probe::BackendId;

    /// The default placement bound; the contract's `flash-attn` section declares the same number.
    const BOUND: Claim = Claim {
        max_relative: 1.0e-3,
        max_absolute: 1.0e-3,
    };

    fn entry_table() -> KernelBindingTable {
        let mut table = KernelBindingTable::new();
        crate::baracuda_dispatch::register_baracuda_cuda_kernels(&mut table);
        table
    }

    #[test]
    fn reference_attends_the_live_prefix_with_grouped_heads() {
        // One batch, two query heads sharing one kv head, d = 1, sk = 3 but k_len = 2: the
        // third (capacity-only) value must not contribute.
        let q = [0.0, 0.0];
        let k = [0.0, 0.0, 9.0];
        let v = [2.0, 4.0, 1000.0];
        let out = reference_decode(&q, (&k, &v), geo(1, 2, 1, 3, 1, 2), 1.0);
        assert_eq!(out, [3.0, 3.0]);
    }

    /// The retained sabotage sibling of the live test: a runner returning the f16 rounding of the
    /// reference measures about nothing, and the same runner with one element perturbed measures
    /// that perturbation. Without the second half a harness comparing the reference with itself
    /// would pass every bound.
    #[test]
    fn the_measurement_sees_a_perturbed_output() {
        let table = entry_table();
        let entry = find_entry(&table, OpKind::FlashAttn, &[DType::F16; 4], BackendId::Cuda)
            .expect("f16 flash_decoding entry");
        let host = |perturb: f32| {
            move |_e: &BindingEntry,
                  p: OpParams,
                  ins: &[HostTensor],
                  _s: Vec<usize>|
                  -> Result<Vec<f32>, VerifyError> {
                let OpParams::FlashAttn {
                    b,
                    hq,
                    hkv,
                    sk,
                    d,
                    k_len,
                    softmax_scale,
                    ..
                } = p
                else {
                    panic!("flash params expected");
                };
                let [q, k, v] = [0, 1, 2].map(|i| f16s(&ins[i]));
                let mut out: Vec<f32> = reference_decode(
                    &q,
                    (&k, &v),
                    geo(b, hq, hkv, sk, d, k_len),
                    softmax_scale as f64,
                )
                .into_iter()
                .map(|x| f16::from_f64(x).to_f32())
                .collect();
                out[0] += perturb;
                Ok(out)
            }
        };
        let exact = measure_flash_decoding(&host(0.0), entry, BOUND).expect("measures");
        assert!(
            exact.worst_abs < 1.0e-3,
            "rounding only: {}",
            exact.worst_abs
        );
        let off = measure_flash_decoding(&host(0.5), entry, BOUND).expect("measures");
        assert!(off.worst_abs >= 0.4, "perturbation seen: {}", off.worst_abs);
    }
}
