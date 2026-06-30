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
