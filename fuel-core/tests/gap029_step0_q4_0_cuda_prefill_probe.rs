//! GAP-029 Step 0 — **measurement, not a unit test**: does a Q4_0 `qmatmul` at
//! `m > 1` (a *prefill* shape) actually reach CUDA's `matmul_q4_0`, which
//! hard-`bail!`s on `total_rows != 1`?
//!
//! ## Why this probe and not the obvious one
//!
//! The obvious probe — call `CudaStorage::matmul_q4_0` directly at `m = 6` —
//! is a **tautology**. The guard is literally
//! `let total_rows = batch * m; if total_rows != 1 { bail!(...) }`
//! (`fuel-cuda-backend/src/storage.rs:3566-3571`, and identically at `:3617`
//! for Q4_K_M), so a direct call can only ever confirm that an `if` statement
//! works. It says nothing about whether a real prefill ever gets there.
//!
//! The load-bearing unknown is **one level up: does placement/dispatch route a
//! Q4_0 `m > 1` matmul to CUDA at all**, or does the fused node decompose to
//! `Slice` + `MatMul` primitives (the Increment-C recipe) and realize without
//! ever touching the quantized CUDA kernel? GAP-007's resolution *predicts* it
//! routes there and breaks, because `BackendCapabilities` has no extent
//! dimension and so cannot express "M must be 1" (GAP-031) — but that
//! prediction is itself a code read, which is the exact thing this probe exists
//! to stop relying on.
//!
//! So this realizes through the **normal lazy path** (`realize_one_as`, full
//! optimizer — *not* `lowering_only`, which would force the decompose and
//! answer a different question). That is the only version that can distinguish
//! **broken** from **falls back cleanly**, and the only one whose result
//! transfers to Lightbulb's `model.prefill(&ids, &mut st)` — which passes the
//! whole encoded prompt in one call, so `m = prompt_tokens` on every request.
//!
//! ## The positive control is load-bearing
//!
//! A malformed probe and a real `bail!` are **indistinguishable from an exit
//! code**, and a red result would read exactly like "my probe was broken". So
//! `m = 1` runs first and must both **succeed** and be **numerically correct**
//! against an exact `BlockQ4_0::to_float` dequant reference. If the control
//! fails, the `m = 6` result is uninterpretable and must not be reported as a
//! finding.
//!
//! Note the control also discriminates *which path ran*: if `m = 6` fails with
//! the literal `only M=1 supported on CUDA today` text, that string is a
//! positive artifact that `matmul_q4_0` was genuinely reached — as opposed to
//! failing for some unrelated reason.
//!
//! Live-GPU: `#[ignore]`d so a plain `cargo test` skips it. Run via
//! `scripts/gpu-run.ps1`.

#![cfg(feature = "cuda")]

use fuel_core::Device;
use fuel_core::lazy::LazyTensor;
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
    let w_f32: Vec<f32> = (0..n * k).map(|i| ((i as f32) * 0.021).sin() * 0.7).collect();
    let blocks_per_row = k / BlockQ4_0::BLCK_SIZE;
    let mut w_blocks = vec![BlockQ4_0::zeros(); n * blocks_per_row];
    BlockQ4_0::from_float(&w_f32, &mut w_blocks);

    let bytes_per_block = std::mem::size_of::<BlockQ4_0>();
    let w_bytes: Vec<u8> = unsafe {
        std::slice::from_raw_parts(w_blocks.as_ptr() as *const u8, w_blocks.len() * bytes_per_block)
    }
    .to_vec();
    assert_eq!(w_bytes.len() % 4, 0, "block bytes must pack into u32");
    let w_u32: Vec<u32> = w_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // Exact dequantized weight [N, K] — the oracle.
    let mut deq = vec![0f32; n * k];
    BlockQ4_0::to_float(&w_blocks, &mut deq);

    let a_data: Vec<f32> = (0..m * k).map(|i| ((i as f32) * 0.013).cos()).collect();
    let x = LazyTensor::from_f32(a_data.clone(), Shape::from_dims(&[m, k]), dev);
    let w = x.const_u32_like(w_u32, Shape::from_dims(&[w_bytes.len() / 4]));
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

    // ---- POSITIVE CONTROL: m = 1 (decode shape) must work AND be correct. --
    // If this fails, the m>1 result below is uninterpretable.
    let (ctl, ctl_want) = realize_q4_0(1, K, N, &dev);
    let ctl_got = match ctl {
        Outcome::Ok(v) => v,
        Outcome::Err(e) => panic!(
            "POSITIVE CONTROL FAILED at m=1 on CUDA: {e}\n\
             The probe could not exercise the quantized CUDA path at all, so \
             the m>1 result proves nothing. This is a broken probe, NOT a \
             GAP-007 finding."
        ),
    };
    let ctl_rel = max_rel(&ctl_got, &ctl_want);
    assert!(
        ctl_rel < 1e-5,
        "POSITIVE CONTROL numerically wrong at m=1 (max_rel {ctl_rel}) — the \
         CUDA quantized path ran but produced garbage, so the m>1 result is \
         uninterpretable."
    );
    eprintln!("[GAP-029 step0] positive control m=1: OK, max_rel {ctl_rel:.3e}");

    // ---- THE MEASUREMENT: m = 6 (a prefill shape). -------------------------
    let (probe, probe_want) = realize_q4_0(6, K, N, &dev);
    match probe {
        Outcome::Ok(got) => {
            let rel = max_rel(&got, &probe_want);
            eprintln!(
                "[GAP-029 step0] RESULT: m=6 SUCCEEDED on CUDA (max_rel {rel:.3e}).\n\
                 => Dispatch did NOT route a Q4_0 m>1 matmul into matmul_q4_0's \
                 M=1 guard. GAP-007 does not gate quantized prefill on this path."
            );
            assert!(
                rel < 1e-5,
                "m=6 realized but is numerically WRONG (max_rel {rel}) — that is \
                 a worse finding than a bail: a silent wrong answer on prefill."
            );
        }
        Outcome::Err(e) => {
            let reached_guard = e.contains("only M=1 supported on CUDA today");
            eprintln!(
                "[GAP-029 step0] RESULT: m=6 FAILED on CUDA.\n\
                 reached matmul_q4_0's M=1 guard: {reached_guard}\n\
                 error: {e}"
            );
            assert!(
                reached_guard,
                "m=6 failed, but NOT with the M=1 guard message — so this is some \
                 other failure and must not be reported as a GAP-007 \
                 confirmation. error: {e}"
            );
        }
    }
}
