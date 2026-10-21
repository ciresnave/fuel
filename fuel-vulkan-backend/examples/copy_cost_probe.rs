// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fixed and per-byte cost of Vulkan host<->device copies (ROADMAP item 21).
//!
//! For each size, the median wall time of: `upload_bytes` (host to device, a
//! fresh allocation each call), `download_bytes` (device to host, GPU idle),
//! and `slot_copy_to_new_handle` followed by a `download_bytes` of the copy
//! (device to device, then read back, so the copy is observable). A 4-byte copy
//! against a 4 MB copy says whether the cost is a fixed submit/fence/staging
//! overhead or bandwidth.
//!
//! ```sh
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-vulkan-backend --example copy_cost_probe
//! ```
//!
//! An example rather than an `#[ignore]` test (the ignore total sits at its
//! ceiling).

use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};
use std::time::Instant;

fn median_us(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let be = std::sync::Arc::new(
        VulkanBackend::with_selection(DeviceSelection::PreferDiscrete)
            .expect("a Vulkan device (PreferDiscrete)"),
    );
    println!(
        "{:>10} {:>12} {:>14} {:>22}   (median us over 100 reps after 10 warm-ups)",
        "bytes", "H2D upload", "D2H download", "D2D copy + readback"
    );
    for &n in &[
        4usize,
        1024,
        4096,
        65_536,
        1 << 20,
        4 << 20,
        16 << 20,
        64 << 20,
    ] {
        let host: Vec<u8> = (0..n).map(|i| (i % 251) as u8).collect();
        let resident = be.upload_bytes(&host).expect("upload");
        let (mut up, mut down, mut d2d) = (Vec::new(), Vec::new(), Vec::new());
        for rep in 0..110 {
            let t = Instant::now();
            let h = be.upload_bytes(&host).expect("upload");
            let a = t.elapsed().as_secs_f64() * 1e6;
            drop(h);
            let t = Instant::now();
            let back = be.download_bytes(&resident).expect("download");
            let b = t.elapsed().as_secs_f64() * 1e6;
            assert_eq!(back.len(), n);
            let t = Instant::now();
            let c = be
                .slot_copy_to_new_handle(&resident, 0, n)
                .expect("device copy");
            let _ = be.download_bytes(&c).expect("download copy");
            let d = t.elapsed().as_secs_f64() * 1e6;
            if rep >= 10 {
                up.push(a);
                down.push(b);
                d2d.push(d);
            }
        }
        println!(
            "{n:>10} {:>12.1} {:>14.1} {:>22.1}",
            median_us(up),
            median_us(down),
            median_us(d2d)
        );
    }
}
