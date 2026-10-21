// SPDX-License-Identifier: MIT OR Apache-2.0
//! Ordering of dependent Vulkan batches that are submitted without a host wait.
//!
//! Each round records `z = x @ W1` (an M = 1 matmul against a 1,024 x 151,936
//! matrix, a few milliseconds on the GPU) and submits it as its OWN batch with
//! `submit_pending`, then records `y = z @ W2` (reads ALL of `z`) and submits that
//! as the next batch, without waiting in between. If the second batch could start
//! before the first finished it would read a partly written `z`. After all rounds
//! every `y` is compared with an f64 CPU chain.
//!
//! The recorder starts each batch with a memory barrier covering earlier
//! submissions (`Recorder::record_inter_batch_barrier`); the Vulkan spec says that
//! barrier's first scope includes earlier submissions on the queue, which is what
//! makes this correct without host waits.
//!
//! What this probe shows. It passes with the barrier, and it also passed with the
//! barrier call removed on the NVIDIA GPU it was written on (consecutive
//! submissions did not overlap there), so a wrong number is not what shows the
//! barrier is needed. Run it under the Khronos validation layer with
//! synchronization validation (core checks left ON, `validate_sync = true`):
//! with the barrier it reports no `SYNC-HAZARD`; with the barrier call removed it
//! reports `SYNC-HAZARD-READ-AFTER-WRITE` across the submissions. (With core
//! checks disabled the layer printed nothing even for a known hazard.)
//!
//! ```sh
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-vulkan-backend --example batch_order_probe
//! ```
//!
//! An example rather than an `#[ignore]` test (the ignore total sits at its
//! ceiling). Exits non-zero if any round is off its reference.

use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
use std::sync::Arc;

const K: usize = 1024;
const WIDE: usize = 151_936;
const ROUNDS: usize = 6;

fn bytes_of(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn floats_of(b: &[u8]) -> Vec<f32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Entries in roughly +-1/sqrt(rows), so a product stays near unit scale.
fn weights(rows: usize, cols: usize, seed: usize) -> Vec<f32> {
    let scale = 2.0 / (rows as f32).sqrt();
    (0..rows * cols)
        .map(|i| ((((i + seed) * 7919) % 1000) as f32 / 1000.0 - 0.5) * scale)
        .collect()
}

fn vecmat(x: &[f64], w: &[f32], cols: usize) -> Vec<f64> {
    let mut out = vec![0f64; cols];
    for (k, &xk) in x.iter().enumerate() {
        for (o, &wv) in out.iter_mut().zip(&w[k * cols..(k + 1) * cols]) {
            *o += xk * wv as f64;
        }
    }
    out
}

fn main() {
    let be = Arc::new(
        VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
            .expect("a Vulkan device (PreferDiscrete)"),
    );
    let w1_host = weights(K, WIDE, 1);
    let w2_host = weights(WIDE, K, 2);
    let x_host: Vec<f32> = (0..K).map(|i| ((i % 17) as f32 - 8.0) / 8.0).collect();
    let w1 = be.upload_bytes_handle(&bytes_of(&w1_host)).unwrap();
    let w2 = be.upload_bytes_handle(&bytes_of(&w2_host)).unwrap();
    let mut xs = vec![be.upload_bytes_handle(&bytes_of(&x_host)).unwrap()];
    let mut zs = Vec::new();
    for _ in 0..ROUNDS {
        zs.push(be.alloc_bytes_handle(WIDE * 4).unwrap());
        xs.push(be.alloc_bytes_handle(K * 4).unwrap());
    }
    let mut batches = Vec::new();
    for r in 0..ROUNDS {
        let (done, rest) = xs.split_at_mut(r + 1);
        be.matmul_f32_bytes(&done[r], &w1, &mut zs[r], &[], &[], 1, WIDE, K)
            .unwrap();
        batches.push(be.submit_pending().unwrap().expect("a batch"));
        be.matmul_f32_bytes(&zs[r], &w2, &mut rest[0], &[], &[], 1, K, WIDE)
            .unwrap();
        batches.push(be.submit_pending().unwrap().expect("a batch"));
    }
    for b in batches {
        be.wait_submitted(b).unwrap();
    }
    let mut want: Vec<f64> = x_host.iter().map(|&v| v as f64).collect();
    let mut bad = 0;
    for r in 0..ROUNDS {
        want = vecmat(&vecmat(&want, &w1_host, WIDE), &w2_host, K);
        let got = floats_of(&be.download_bytes(&xs[r + 1]).unwrap());
        let scale = want.iter().fold(0f64, |m, v| m.max(v.abs())).max(1e-30);
        let err = got
            .iter()
            .zip(&want)
            .fold(0f64, |m, (g, w)| m.max((*g as f64 - w).abs()))
            / scale;
        println!("round {}: max error relative to max |y| = {err:.2e}", r + 1);
        if err > 1e-3 {
            bad += 1;
        }
    }
    if bad != 0 {
        eprintln!("{bad} round(s) off the f64 chain");
        std::process::exit(1);
    }
}
