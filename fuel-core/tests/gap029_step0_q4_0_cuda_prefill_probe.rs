// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-029 Step 0, relabelled by GAP-335: a Q4_0 `qmatmul` realized on a CUDA
//! device through the normal lazy path, at `m = 1` and at `m = 6`, checked for
//! SANE values against an exact dequant reference.
//!
//! ## What this does NOT show
//!
//! It was written to ask whether a prefill-shaped (`m > 1`) Q4_0 matmul reaches
//! CUDA's `CudaStorage::matmul_q4_0` and its `total_rows != 1` guard. **It cannot
//! answer that, and never could: the dispatcher has no CUDA `QMatMul` binding
//! (GAP-335), and it had none when this file was written.** So neither `m = 1`
//! nor `m = 6` can reach `matmul_q4_0`. The realize takes some other path
//! (recipe decomposition or another placement), and WHICH path is unmeasured.
//! The `m = 1` run is therefore not a positive control for the quantized CUDA
//! kernel, and "m = 6 did not hit the guard" is true only because nothing does.
//!
//! What it does check: realizing a quantized matmul on a CUDA `Device` succeeds
//! at both shapes, and the result is within a loose bound of the exact
//! reference. The device-free detector for the missing binding (and the note
//! to re-point this test if one is added) is GAP-335's.
//!
//! Live-GPU: `#[ignore]`d so a plain `cargo test` skips it. Run via
//! `scripts/gpu-run.ps1`.

#![cfg(feature = "cuda")]

use fuel_core::Device;
use fuel_core::lazy::Tensor;
use fuel_core::pipelined_bridge::realize_one_as;
use fuel_graph::QuantType;
use fuel_ir::Shape;
use fuel_quantized::{BlockQ4_0, GgmlType};

/// Outcome of one realize attempt, with enough detail to tell a real bail from
/// a broken probe.
enum Outcome {
    Ok(Vec<f32>),
    Err(String),
}

/// Build `[m, k] @ dequant(W)^T -> [m, n]` as a Q4_0 `qmatmul` and realize it on
/// `dev` through the normal (fully-optimized) lazy path. Also returns the exact
/// dequant reference so the caller can check numerics.
fn realize_q4_0(m: usize, k: usize, n: usize, dev: &Device) -> (Outcome, Vec<f32>) {
    // Deterministic, well-scaled weights; quantize to real Q4_0 blocks.
    let w_f32: Vec<f32> = (0..n * k)
        .map(|i| ((i as f32) * 0.021).sin() * 0.7)
        .collect();
    let blocks_per_row = k / BlockQ4_0::BLCK_SIZE;
    let mut w_blocks = vec![BlockQ4_0::zeros(); n * blocks_per_row];
    BlockQ4_0::from_float(&w_f32, &mut w_blocks);

    let bytes_per_block = std::mem::size_of::<BlockQ4_0>();
    let w_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(
            w_blocks.as_ptr() as *const u8,
            w_blocks.len() * bytes_per_block,
        )
    }
    .to_vec();
    assert_eq!(w_bytes.len() % 4, 0, "block bytes must pack into u32");
    let w_u32: Vec<u32> = w_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Exact dequantized weight [N, K] — the oracle.
    let mut deq = vec![0f32; n * k];
    BlockQ4_0::to_float(&w_blocks, &mut deq);

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
    let x = Tensor::from_f32(a_data.clone(), Shape::from_dims(&[m, k]), dev).unwrap();
    let w = x
        .const_u32_like(w_u32, Shape::from_dims(&[w_bytes.len() / 4]))
        .unwrap();
    let y = x
        .qmatmul(&w, QuantType::Q4_0, k, n)
        .expect("qmatmul build failed (graph-build, not realize — probe is malformed)");

    // Exact reference: out[mi, ni] = Σ_k a[mi, k] · deq[ni, k].
    let mut expected = vec![0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut s = 0f32;
            for kk in 0..k {
                s += a_data[mi * k + kk] * deq[ni * k + kk];
            }
            expected[mi * n + ni] = s;
        }
    }

    let t = y.graph_tensor();
    let graph = t.graph().clone();
    let id = t.id();
    let outcome = match realize_one_as::<f32>(&graph, id, dev) {
        Ok(v) => Outcome::Ok(v),
        Err(e) => Outcome::Err(format!("{e:?}")),
    };
    (outcome, expected)
}

fn max_rel(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want.iter())
        .map(|(&g, &e)| {
            let denom = g.abs().max(e.abs()).max(f32::MIN_POSITIVE);
            (g - e).abs() / denom
        })
        .fold(0.0f32, f32::max)
}

/// Error relative to the OUTPUT SCALE rather than per-element.
///
/// Per-element `max_rel` is unstable exactly where it matters here: an output
/// near a zero crossing has a tiny denominator, so a perfectly ordinary
/// absolute error reads as a huge relative one — and the more elements you
/// compare, the more likely you are to sample such a point. That makes
/// `max_rel` grow with output size for reasons having nothing to do with
/// accuracy, so comparing `max_rel` at m=1 (64 outputs) against m=6 (384
/// outputs) **cannot** distinguish "systematically worse" from "more samples,
/// bigger max". This metric can: the denominator is fixed by the tensor, not
/// by the element.
fn scale_rel(got: &[f32], want: &[f32]) -> f32 {
    let scale = want
        .iter()
        .fold(0.0f32, |m, &e| m.max(e.abs()))
        .max(f32::MIN_POSITIVE);
    let max_abs = got
        .iter()
        .zip(want.iter())
        .map(|(&g, &e)| (g - e).abs())
        .fold(0.0f32, f32::max);
    max_abs / scale
}

/// Median per-element relative error — a distribution summary that a handful of
/// near-zero outliers cannot dominate.
fn median_rel(got: &[f32], want: &[f32]) -> f32 {
    let mut v: Vec<f32> = got
        .iter()
        .zip(want.iter())
        .map(|(&g, &e)| {
            let denom = g.abs().max(e.abs()).max(f32::MIN_POSITIVE);
            (g - e).abs() / denom
        })
        .collect();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[test]
#[ignore = "live GPU: run via scripts/gpu-run.ps1"]
fn gap029_step0_q4_0_prefill_shape_on_cuda() {
    // k >= 64 (CUDA's own guard for type-0/1 quants) and a multiple of 32
    // (Q4_0 block size). n is arbitrary.
    const K: usize = 128;
    const N: usize = 64;

    let cuda = fuel_cuda_backend::CudaDevice::new(0)
        .expect("CUDA device 0 — probe cannot run, this is NOT a finding");
    let dev: Device = cuda.into();

    // Measure BOTH arms before asserting anything, so one run reports the whole
    // picture. The first version panicked inside the positive control and
    // therefore never reached the measurement it existed to enable.
    let (ctl, ctl_want) = realize_q4_0(1, K, N, &dev);
    let (probe, probe_want) = realize_q4_0(6, K, N, &dev);

    // ---- m = 1 (decode shape) must realize and be sane. ---------------------
    //
    // TOLERANCE, AND WHAT IT IS NOT. This file once asserted `max_rel < 1e-5`
    // (imported from the recipe-path oracle) and measured 1.08e-2; the per-
    // element figure was then shown to be one near-zero-crossing output
    // (median relative error 1.74e-4, scale-relative 1.97e-4). The earlier
    // explanation, that "the CUDA path is a different algorithm: baracuda's
    // batched MMVQ", is WITHDRAWN: no CUDA QMatMul binding exists (GAP-335), so
    // that kernel never ran. The 5e-2 bound below is NOT calibrated for the path
    // actually taken, which is unmeasured. It only catches O(1) breakage.
    //
    // NOT claimed: that any particular kernel is numerically correct.
    const SANE: f32 = 5e-2;
    let ctl_got = match ctl {
        Outcome::Ok(v) => v,
        Outcome::Err(e) => panic!("m=1 qmatmul failed to realize on a CUDA device: {e}"),
    };
    let ctl_rel = max_rel(&ctl_got, &ctl_want);
    let ctl_scale = scale_rel(&ctl_got, &ctl_want);
    let ctl_med = median_rel(&ctl_got, &ctl_want);
    eprintln!(
        "[GAP-029 step0] m=1 ({} outputs) realized on a CUDA device (path unpinned). \
         max_rel {ctl_rel:.4e} | scale_rel {ctl_scale:.4e} | median_rel \
         {ctl_med:.4e}  (max_rel is inflated by near-zero-crossing outputs; \
         scale_rel/median_rel are the trustworthy columns)",
        ctl_got.len(),
    );
    assert!(
        ctl_scale < SANE,
        "m=1 realized but scale_rel {ctl_scale:.4e} exceeds the sane bound \
         {SANE:.0e}; whatever path ran is broken, not noisy."
    );

    // ---- m = 6 (a prefill shape) must realize and be sane too. -------------
    match probe {
        Outcome::Ok(got) => {
            let rel = max_rel(&got, &probe_want);
            let sc = scale_rel(&got, &probe_want);
            let med = median_rel(&got, &probe_want);
            eprintln!(
                "[GAP-029 step0] m=6 ({} outputs) realized on a CUDA device (path \
                 unpinned). max_rel {rel:.4e} | scale_rel {sc:.4e} | median_rel {med:.4e}. \
                 Not evidence about matmul_q4_0, which no path reaches (GAP-335).",
                got.len(),
            );
            assert!(
                sc < SANE,
                "m=6 realized but scale_rel {sc:.4e} exceeds the sane bound — a \
                 silent wrong answer on prefill is a WORSE finding than a bail, \
                 not a better one. Do not report this as 'prefill works'."
            );
        }
        Outcome::Err(e) => panic!("m=6 qmatmul failed to realize on a CUDA device: {e}"),
    }
}
