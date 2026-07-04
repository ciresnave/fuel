//! Live-device regression guard for the Vulkan realize memory-reclaim
//! bug (the D1 replan decode path OOM'd on 12 GB after ~4 full realizes).
//!
//! Gated `#[ignore]` — run with:
//!
//! ```sh
//! cargo test -p fuel-vulkan-backend --test recycler_reclaim_live -- --ignored --nocapture
//! ```
//!
//! ## The bug this pins
//!
//! `VulkanBuffer::drop` pushes every freed device buffer into the
//! backend's `buffer_pool` recycler. Before the fix, only the legacy
//! typed `alloc_device` path ever consulted OR trimmed that pool — the
//! production byte-storage substrate (`alloc_bytes` / `upload_bytes`,
//! which the pipelined-executor realize path is built on) allocated
//! fresh from VMA every time and never reused or trimmed. So a realize
//! loop that re-uploads the same weight working-set every iteration
//! (exactly the D1 replan decode path) grew the pool by one working-set
//! per realize until `ERROR_OUT_OF_DEVICE_MEMORY`.
//!
//! This test reproduces the leak signature at a few-MB scale — no 1.1 B
//! model needed — by measuring the fuel-side recycler occupancy
//! (`recycler_pooled_bytes`, deterministic) across N repeated full
//! realizes. A push-only recycler grows ~linearly in N; a correct one
//! (reuse + bounded trim) holds ~one working-set regardless of N. The
//! driver-reported `vram_used` is printed alongside as a coarse
//! cross-check (VMA block granularity makes it noisy, so it is not
//! hard-asserted).

use fuel_vulkan_backend::{DeviceSelection, VulkanBackend};

fn backend_or_skip() -> Option<VulkanBackend> {
    match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("no Vulkan device; skipping: {e:?}");
            None
        }
    }
}

/// One "full realize" uploads a fixed weight working-set (several MB of
/// distinct sizes) held simultaneously, then drops it all at end-of-
/// realize — mirroring a D1 `forward_with_kv_context` + full realize
/// per token, whose weight Consts are uploaded fresh each token and
/// dropped when the local executor cache dies. The SAME sizes recur
/// every realize (weights don't change size), so a correct recycler
/// REUSES the freed buffers and the pool stays at ~one working-set.
fn one_realize(b: &VulkanBackend, sizes: &[usize]) {
    let mut held = Vec::with_capacity(sizes.len());
    for &s in sizes {
        let src = vec![0u8; s];
        held.push(b.upload_bytes(&src).expect("h2d weight upload"));
    }
    // End of realize: `held` drops here → every VulkanBuffer Arc hits 0
    // → VulkanBuffer::drop returns each buffer to the recycler pool.
}

#[test]
#[ignore = "requires a live Vulkan device"]
fn recycler_reclaims_across_full_realizes_no_unbounded_growth() {
    let Some(b) = backend_or_skip() else { return };

    // A ~10 MB working set of distinct recurring sizes (stand-in for a
    // model's weight tensors). Distinct sizes exercise several buckets;
    // recurrence is what a correct recycler must exploit for reuse.
    let sizes: [usize; 4] = [
        4 * 1024 * 1024,
        3 * 1024 * 1024,
        2 * 1024 * 1024,
        1 * 1024 * 1024,
    ];
    let working_set: u64 = sizes.iter().map(|&s| s as u64).sum();

    // Warm-up realize establishes the steady-state pool + lets VMA
    // reserve its device-memory blocks, so the later delta is clean.
    one_realize(&b, &sizes);
    b.synchronize_pending().ok();
    let pool_after_1 = b.recycler_pooled_bytes();
    let vram_after_1 = b.vram_used();

    const N: usize = 20;
    for _ in 0..N {
        one_realize(&b, &sizes);
        // synchronize_pending between realizes is exactly what did NOT
        // help in the field — the leak is the untrimmed pool, not
        // deferred-batch retirement. Kept here to mirror the real loop.
        b.synchronize_pending().ok();
    }
    let pool_after_n = b.recycler_pooled_bytes();
    let vram_after_n = b.vram_used();

    eprintln!("working_set              = {working_set} bytes");
    eprintln!("pool_bytes after realize 1     = {pool_after_1}");
    eprintln!("pool_bytes after realize {}    = {pool_after_n}", N + 1);
    eprintln!("vram_used  after realize 1     = {vram_after_1}");
    eprintln!("vram_used  after realize {}    = {vram_after_n}", N + 1);
    let vram_delta = vram_after_n.saturating_sub(vram_after_1);
    eprintln!("vram_used delta (1 -> {})       = {vram_delta}", N + 1);

    // A correctly-bounded recycler holds ~one working-set no matter how
    // many realizes ran (reuse keeps the pool flat; the trim caps it).
    // Allow 2× slack for best-fit rounding / an in-flight extra bucket.
    // A push-only recycler holds ~(N+1)× working-set — orders over this.
    let bound = working_set * 2;
    assert!(
        pool_after_n <= bound,
        "Vulkan recycler-reclaim leak: pool held {pool_after_n} bytes after \
         {} full realizes (working set {working_set}, bound {bound}). The \
         recycler grew ~linearly instead of reusing/trimming freed device \
         buffers — the D1-decode OOM signature.",
        N + 1,
    );
}
