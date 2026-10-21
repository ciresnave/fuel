// SPDX-License-Identifier: MIT OR Apache-2.0
//! Cross-check of the Vulkan timestamp-query GPU timing (ROADMAP item 22)
//! against the host clock.
//!
//! Records `n` decode-attention dispatches (`flash_attn_f32`, 8 heads, 4096 keys)
//! into one batch, drains it with a readback, and compares the GPU time the
//! timestamp queries report for the batch (`BATCH_TOTAL`, first to last
//! timestamp) with the host wall time of record + submit + wait + readback. As
//! `n` grows the fixed submit/wait/readback cost amortises, so the two should
//! converge; GPU time may never exceed the wall time.
//!
//! ```sh
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-vulkan-backend --example gpu_time_probe
//! ```
//!
//! An example rather than an `#[ignore]` test (the ignore total sits at its
//! ceiling).

use fuel_vulkan_backend::{BATCH_TOTAL, DeviceSelection, VulkanBackend};
use std::time::Instant;

fn main() {
    let be = VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
        .expect("a Vulkan device (PreferDiscrete)");
    assert!(be.set_gpu_timing(true), "timestamp queries unsupported");
    let (hq, hkv, sk, d) = (8usize, 8usize, 4096usize, 128usize);
    let fill = |n: usize| -> Vec<u8> {
        (0..n)
            .flat_map(|i| (((i % 97) as f32) * 0.01 - 0.4).to_le_bytes())
            .collect()
    };
    let q = be.upload_bytes(&fill(hq * d)).unwrap();
    let k = be.upload_bytes(&fill(hkv * sk * d)).unwrap();
    let v = be.upload_bytes(&fill(hkv * sk * d)).unwrap();
    let mut out = be.alloc_bytes(hq * d * 4).unwrap();
    let scale = 1.0 / (d as f32).sqrt();
    println!(
        "{:>6} {:>12} {:>12} {:>14} {:>12}",
        "n", "wall ms", "gpu ms", "gpu/wall", "gpu us/disp"
    );
    for &n in &[1usize, 10, 50, 200, 400] {
        let mut rows = Vec::new();
        for rep in 0..7 {
            be.reset_gpu_time();
            let t = Instant::now();
            for _ in 0..n {
                be.flash_attn_f32_bytes(
                    &q, &k, &v, None, &mut out, 1, hq, hkv, 1, sk, d, scale, false,
                )
                .unwrap();
            }
            be.download_bytes(&out).unwrap();
            let wall = t.elapsed().as_secs_f64() * 1e3;
            let gpu: f64 = be
                .gpu_time_snapshot()
                .iter()
                .filter(|r| r.0 == BATCH_TOTAL)
                .map(|r| r.2 as f64 / 1e6)
                .sum();
            if rep >= 2 {
                rows.push((wall, gpu));
            }
        }
        rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (wall, gpu) = rows[rows.len() / 2];
        println!(
            "{n:>6} {wall:>12.3} {gpu:>12.3} {:>14.2} {:>12.1}",
            gpu / wall,
            gpu * 1e3 / n as f64
        );
        assert!(gpu <= wall * 1.001, "GPU time exceeds wall time");
    }
}
