// SPDX-License-Identifier: MIT OR Apache-2.0
//! GAP-331: proves `mmvq_run` calls its argument checks before launching.
//! The checks themselves are unit-tested without a device in
//! `src/baracuda/gguf.rs`. This test only shows they are wired in.
//!
//! Only an in-bounds case is used here. With the one-byte offset below,
//! the W read covers bytes 1..37 of a 37-byte buffer. So if the check were
//! missing, the kernel would still read inside the buffer. No arm of this
//! test can cause an out-of-bounds device read.
//!
//! Gated `#[ignore]`: run with `cargo test -p fuel-cuda-backend -- --ignored`
//! under `scripts/gpu-run.ps1`, like the sibling `*_live.rs` tests.

use fuel_cuda_backend::{CudaDevice, CudaStorageBytes, baracuda::gguf};

/// Acquire CUDA device 0. GAP-224: asserts the machine-wide GPU mutex is
/// held. GAP-157: a missing device is a failure, not a silent skip.
fn dev() -> CudaDevice {
    fuel_test_support::require_gpu_run_lock();
    fuel_test_support::required_ok("CUDA device 0", CudaDevice::new(0))
}

/// One Q4_0 row of 64 cols is 2 blocks of 18 bytes. All-zero blocks have
/// scale 0, so the product is exactly 0.
#[test]
#[ignore]
fn mmvq_declines_a_misaligned_offset_before_launch() {
    let dev = dev();
    let act = CudaStorageBytes::from_cpu_bytes(&dev, &[0_u8; 64 * 4]).expect("h2d activations");

    // Q4_0's offset alignment is 2, so offset 1 must be declined.
    let w = CudaStorageBytes::from_cpu_bytes(&dev, &[0_u8; 1 + 36]).expect("h2d weights");
    let err = gguf::mmvq_q4_0(&w, &act, None, 1, 64, 1).expect_err("offset 1 must be declined");
    let msg = err.to_string();
    assert!(
        msg.contains("mmvq_q4_0: w_start_byte_offset (1) is not a multiple of 2 bytes"),
        "wrong decline: {msg}"
    );

    // Control: an aligned offset reaches the kernel and runs. This shows the
    // decline above is not every call failing for some other reason.
    let w = CudaStorageBytes::from_cpu_bytes(&dev, &[0_u8; 2 + 36]).expect("h2d weights");
    let out = gguf::mmvq_q4_0(&w, &act, None, 2, 64, 1).expect("offset 2 is valid");
    let bytes = out.to_cpu_bytes().expect("d2h");
    assert_eq!(bytes.len(), 4, "one output row of f32");
    assert_eq!(f32::from_le_bytes(bytes[..4].try_into().unwrap()), 0.0);
}
