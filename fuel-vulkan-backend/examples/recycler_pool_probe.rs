// SPDX-License-Identifier: MIT OR Apache-2.0
//! Worst-case VRAM held by the Vulkan buffer recycler (ROADMAP item 23).
//!
//! Allocates, for 64 distinct sizes between 2 KiB and 24 MiB, as many buffers as the
//! recycler's per-bucket depth allows (about 16 MiB per size), frees them all, and
//! reads the pooled bytes before and after one trimming allocation. The pool's own
//! caps (64 buckets, 512 MB total) apply at that trim.
//!
//! ```text
//! pwsh scripts/gpu-run.ps1 -Project fuel -- cargo run -j 4 --release \
//!     -p fuel-vulkan-backend --example recycler_pool_probe
//! ```

use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

fn depth(size: u64) -> usize {
    ((16u64 * 1024 * 1024 / size.max(1)) as usize).clamp(4, 1024)
}

fn main() {
    let be =
        VulkanBackend::with_selection(DeviceSelection::PreferDiscrete).expect("vulkan backend");
    println!("device: {}", be.device_name);
    let (lo, hi) = (2048f64, 24.0 * 1024.0 * 1024.0);
    let sizes: Vec<u64> = (0..64)
        .map(|i| (lo * (hi / lo).powf(i as f64 / 63.0)) as u64)
        .collect();
    let mut live = Vec::new();
    let mut requested = 0u64;
    for &s in &sizes {
        for _ in 0..depth(s) {
            live.push(be.alloc_bytes(s as usize).expect("alloc"));
            requested += s;
        }
    }
    println!(
        "allocated {} buffers over {} distinct sizes: {:.1} MiB live",
        live.len(),
        sizes.len(),
        requested as f64 / 1048576.0
    );
    drop(live);
    println!(
        "pooled after free (before any trim): {:.1} MiB",
        be.recycler_pooled_bytes() as f64 / 1048576.0
    );
    let _probe = be.alloc_bytes(3).expect("trigger alloc");
    println!(
        "pooled after one trimming alloc:      {:.1} MiB (cap 512 MiB)",
        be.recycler_pooled_bytes() as f64 / 1048576.0
    );
}
