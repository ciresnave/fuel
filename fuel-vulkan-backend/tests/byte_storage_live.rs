//! Live-device tests for the Phase 7.5 A4 substrate methods on
//! [`VulkanBackend`] / [`VulkanStorageBytes`]. Gated `#[ignore]` —
//! run with:
//!
//! ```sh
//! cargo test -p fuel-vulkan-backend --test byte_storage_live -- --ignored --nocapture
//! ```

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

/// Smoke: alloc_bytes(byte_count) reports the right len_bytes and is
/// readable back via download_bytes.
#[test]
#[ignore]
fn alloc_then_download() {
    let Some(b) = backend_or_skip() else { return };
    let storage = b.alloc_bytes(32).expect("alloc");
    assert_eq!(storage.len_bytes(), 32);
    let got = b.download_bytes(&storage).expect("d2h");
    // alloc_bytes does NOT zero — the GPU buffer is uninitialized.
    // We only assert the length round-trips, not the content.
    assert_eq!(got.len(), 32);
}

/// H2D + D2H roundtrip: upload bytes, read them back, expect exact
/// byte equality.
#[test]
#[ignore]
fn upload_download_roundtrip_preserves_bytes() {
    let Some(b) = backend_or_skip() else { return };
    let src: Vec<u8> = (0..=255).collect();
    let storage = b.upload_bytes(&src).expect("h2d");
    assert_eq!(storage.len_bytes(), src.len());
    let got = b.download_bytes(&storage).expect("d2h");
    assert_eq!(got, src);
}

/// Zero-length transfer is sound: upload(empty) and alloc(0) both
/// succeed; download produces an empty Vec.
#[test]
#[ignore]
fn zero_length_transfers_round_trip() {
    let Some(b) = backend_or_skip() else { return };
    let from_empty = b.upload_bytes(&[]).expect("h2d 0");
    assert_eq!(from_empty.len_bytes(), 0);
    let got = b.download_bytes(&from_empty).expect("d2h 0");
    assert!(got.is_empty());

    let storage = b.alloc_bytes(0).expect("alloc 0");
    assert_eq!(storage.len_bytes(), 0);
    let got = b.download_bytes(&storage).expect("d2h 0 (alloc)");
    assert!(got.is_empty());
}

/// Larger transfer: 1 MiB pattern, exercises the non-trivial copy path.
#[test]
#[ignore]
fn one_mib_roundtrip_preserves_bytes() {
    let Some(b) = backend_or_skip() else { return };
    let src: Vec<u8> = (0..1024 * 1024).map(|i| (i & 0xFF) as u8).collect();
    let storage = b.upload_bytes(&src).expect("h2d");
    let got = b.download_bytes(&storage).expect("d2h");
    assert_eq!(got, src);
}

/// BDA: a device-resident storage buffer yields a valid non-zero device
/// address — proving the `bufferDeviceAddress` device feature + the BDA
/// allocator option + the `SHADER_DEVICE_ADDRESS` buffer usage are all wired.
/// This is exactly the value the FDX Vulkan path (spec §3.3.1) sources as a
/// `kDLVulkan` tensor's `data` (the base; `byte_offset` is folded at dispatch).
#[test]
#[ignore]
fn device_storage_has_valid_buffer_device_address() {
    let Some(b) = backend_or_skip() else { return };
    let storage = b.alloc_bytes(256).expect("alloc");
    let buf = storage.buffer_opt().expect("device-resident storage must carry a buffer");
    let addr = buf
        .device_address()
        .expect("device_address must succeed (BDA feature + usage must be enabled)");
    assert_ne!(addr, 0, "a SHADER_DEVICE_ADDRESS buffer must have a non-zero device address");
    eprintln!("BDA device_address = {addr:#018x}");
}

/// Regression: enabling BDA (the device feature + the address usage bit on
/// every storage buffer) must NOT disturb the existing descriptor/transfer
/// path. The same storage that exposes a device address still round-trips
/// H2D/D2H byte-for-byte.
#[test]
#[ignore]
fn bda_does_not_disturb_transfer_path() {
    let Some(b) = backend_or_skip() else { return };
    let src: Vec<u8> = (0..=255).collect();
    let storage = b.upload_bytes(&src).expect("h2d");
    let addr = storage
        .buffer_opt()
        .expect("buffer")
        .device_address()
        .expect("bda");
    assert_ne!(addr, 0);
    let got = b.download_bytes(&storage).expect("d2h");
    assert_eq!(got, src, "BDA enablement must not disturb the transfer path");
}

/// Step E A4b-2: the async submit/wait split — `submit_pending` (submit a
/// recorded batch WITHOUT waiting) + `wait_submitted` (wait the fence, then
/// release the in-flight batch). Records ONE affine dispatch into the deferred
/// batch, submits it async, waits via the returned `SubmittedBatch`, and reads
/// the result back — proving (a) the submitted batch computes correctly, (b) the
/// `SubmittedBatch` keeps the CB/descriptors/transients alive across the wait
/// (no use-after-free), and (c) `submit_pending` is idempotent (a second call on
/// an empty batch returns `None`).
///
/// out = input * 2 + 1, for input = [1,2,3,4] ⇒ [3,5,7,9].
#[test]
#[ignore = "requires a live Vulkan device"]
fn submit_pending_then_wait_submitted_computes_and_releases() {
    use fuel_ir::{Layout, Shape};

    let Some(b) = backend_or_skip() else { return };

    let input_f32: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut in_bytes = Vec::with_capacity(16);
    for v in input_f32 {
        in_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let input = b.upload_bytes(&in_bytes).expect("h2d input");
    let mut out = b.alloc_bytes(16).expect("alloc out");

    let layout = Layout::contiguous(Shape::from_dims(&[4]));
    // Records into the deferred batch (flush_pending only flushes at the TDR cap,
    // so nothing is submitted yet).
    b.affine_f32_bytes(&input, &mut out, 2.0, 1.0, &layout)
        .expect("affine records");

    // ASYNC submit: returns the in-flight batch WITHOUT waiting.
    let batch = b
        .submit_pending()
        .expect("submit_pending ok")
        .expect("non-empty batch ⇒ Some(SubmittedBatch)");

    // Deferred wait: blocks on the fence, retires pools, then releases the batch
    // (frees CB/descriptors/transients/pool — must be post-fence, UAF-safe).
    b.wait_submitted(batch).expect("wait_submitted ok");

    // A second submit sees an empty batch ⇒ None (idempotent).
    assert!(
        b.submit_pending().expect("submit_pending ok 2").is_none(),
        "submit_pending on an empty batch must return None",
    );

    let got = b.download_bytes(&out).expect("d2h out");
    let got_f32: Vec<f32> = got
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got_f32, vec![3.0_f32, 5.0, 7.0, 9.0]);
}

/// Follow-on #2 (the error-path UAF safety net): a `SubmittedBatch` dropped
/// WITHOUT `wait_submitted` — the shape of a `?` unwinding the realize loop with
/// in-flight Vulkan batches still queued — must still complete safely. The
/// `SubmittedBatch::Drop` safety net fence-waits (because `consumed == false`)
/// BEFORE the CB / descriptor sets / transient buffers / pool free, so the GPU
/// has finished the command buffer and freeing it can't race the GPU (the UAF
/// this fix closes). Here we submit an affine batch, then DROP the returned
/// `SubmittedBatch` directly (never calling `wait_submitted`); the test
/// completing without a hang / GPU fault, the batch being idempotently consumed
/// (a second `submit_pending` ⇒ `None`), and the output reading back correct all
/// confirm the Drop path waited + released cleanly.
#[test]
#[ignore = "requires a live Vulkan device"]
fn submitted_batch_dropped_without_wait_is_uaf_safe() {
    use fuel_ir::{Layout, Shape};

    let Some(b) = backend_or_skip() else { return };

    let input_f32: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut in_bytes = Vec::with_capacity(16);
    for v in input_f32 {
        in_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let input = b.upload_bytes(&in_bytes).expect("h2d input");
    let mut out = b.alloc_bytes(16).expect("alloc out");

    let layout = Layout::contiguous(Shape::from_dims(&[4]));
    b.affine_f32_bytes(&input, &mut out, 2.0, 1.0, &layout)
        .expect("affine records");

    // ASYNC submit, then DROP the in-flight batch WITHOUT wait_submitted — the
    // error-unwind shape. SubmittedBatch::Drop (consumed == false) fence-waits
    // here before freeing the CB/descriptors/transients/pool.
    {
        let batch = b
            .submit_pending()
            .expect("submit_pending ok")
            .expect("non-empty batch ⇒ Some(SubmittedBatch)");
        drop(batch); // <- the safety net fires: fence-wait, THEN free. No UAF.
    }

    // The batch was submitted (and its Drop waited), so a second submit is empty.
    assert!(
        b.submit_pending().expect("submit_pending ok 2").is_none(),
        "the dropped batch was already submitted ⇒ empty open batch ⇒ None",
    );

    // The GPU finished (the Drop waited the fence), so the output is correct.
    let got = b.download_bytes(&out).expect("d2h out");
    let got_f32: Vec<f32> = got
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    assert_eq!(got_f32, vec![3.0_f32, 5.0, 7.0, 9.0]);
}

/// Step E A2.1 (deferred-deletion / retain-until-fence) — the MECHANISM, isolated.
///
/// Proves the executor's deferred-eviction primitive: a DATA buffer evicted from
/// the cache while an in-flight `SubmittedBatch` may read it is RETAINED on that
/// batch (`SubmittedBatch::retain_buffer`) instead of host-blocking on a drain,
/// and freed only POST-fence on the batch's `Drop` (follow-on #2). This is
/// deterministic (no wall-clock — the box's timer is noisy per follow-on #1): we
/// observe the buffer's `Arc` lifetime via a `Weak`.
///
/// Sequence (mirrors `defer_evicted_vulkan_buffer` in the executor):
///   1. Record an affine op into the open batch; `submit_pending` → an in-flight
///      `SubmittedBatch` whose CB reads `out`'s buffer.
///   2. Take `out`'s device buffer `Arc`, retain a clone on the batch, and a
///      `Weak` to observe its life. Then DROP every other strong ref (the `out`
///      storage + our `Arc`), so the ONLY strong ref left is the one inside the
///      batch — exactly the post-`cache.remove` state.
///   3. ASSERT the buffer is STILL ALIVE (`Weak::upgrade().is_some()`): the
///      retain kept it out of the recycler even though the cache "evicted" it —
///      i.e. NO eviction-time free, the GPU can still read it. (This is the
///      no-UAF property: the buffer outlives the eviction.)
///   4. `wait_submitted(batch)` → the batch drops → its retained `Arc` drops.
///   5. ASSERT the buffer is now GONE (`Weak::upgrade().is_none()`): freed
///      POST-fence, exactly when `wait_submitted` released the batch — proving
///      retain-until-fence (the deferred free actually happens; no leak).
#[test]
#[ignore = "requires a live Vulkan device"]
fn evicted_buffer_retained_on_batch_frees_post_fence() {
    use fuel_ir::{Layout, Shape};
    use std::sync::Arc;

    let Some(b) = backend_or_skip() else { return };

    let input_f32: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let mut in_bytes = Vec::with_capacity(16);
    for v in input_f32 {
        in_bytes.extend_from_slice(&v.to_le_bytes());
    }
    let input = b.upload_bytes(&in_bytes).expect("h2d input");
    let mut out = b.alloc_bytes(16).expect("alloc out");

    let layout = Layout::contiguous(Shape::from_dims(&[4]));
    b.affine_f32_bytes(&input, &mut out, 2.0, 1.0, &layout)
        .expect("affine records");

    // Submit WITHOUT waiting — the CB now reads `out`'s buffer on the GPU.
    let mut batch = b
        .submit_pending()
        .expect("submit_pending ok")
        .expect("non-empty batch ⇒ Some(SubmittedBatch)");

    // The evicted buffer: take its device Arc, retain a clone on the in-flight
    // batch (what the executor does), and a Weak to watch its lifetime.
    let out_arc: Arc<_> = out
        .device_buffer_arc()
        .expect("out is device-resident ⇒ Some(Arc<VulkanBuffer>)");
    let weak = Arc::downgrade(&out_arc);
    batch.retain_buffer(out_arc.clone());

    // "Evict": drop every strong ref EXCEPT the one inside the batch — the
    // post-`cache.remove(&destroyed)` state where only the deferred retain holds
    // the buffer alive.
    drop(out_arc);
    drop(out);

    // The buffer must still be ALIVE — retained by the in-flight batch, NOT freed
    // at eviction time (no UAF: the GPU can still read it). If the retain were a
    // no-op (the pre-A2.1 drain-then-free), this strong ref would be gone.
    assert!(
        weak.upgrade().is_some(),
        "A2.1: an evicted buffer retained on an in-flight batch must STILL be \
         alive after the cache drops its ref — freeing it now would be a UAF",
    );

    // Wait the fence + release the batch → the retained Arc drops POST-fence.
    b.wait_submitted(batch).expect("wait_submitted ok");

    // Now the buffer is freed — exactly when the batch (post-fence) released it.
    // Proves retain-until-fence: the deferred free DID happen (no leak), and only
    // AFTER the GPU finished (the fence the wait blocked on).
    assert!(
        weak.upgrade().is_none(),
        "A2.1: the retained buffer must be freed once the batch is released \
         post-fence (deferred-deletion completes — no permanent leak)",
    );
}
