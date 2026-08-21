// SPDX-License-Identifier: MIT OR Apache-2.0
//! Live-device tests for `impl BackendRuntime for CudaDevice` — the
//! contract-v0.3 memory-pressure surface backed by baracuda alpha.66's
//! `cuMemGetInfo` wrappers. Gated `#[ignore]` — run with
//! `cargo test -p fuel-cuda-backend --test backend_runtime_live -- --ignored`
//! on a host with an NVIDIA GPU + CUDA Runtime SDK installed.

use fuel_backend_contract::backend::BackendRuntime;
use fuel_cuda_backend::CudaDevice;
use fuel_ir::backend::FitStatus;

/// Acquire CUDA device 0 for a live test. GAP-224: asserts the machine-wide GPU
/// mutex is held (`require_gpu_run_lock`); GAP-157: a missing device is a FAILURE
/// (`required_ok`), not a silent skip. On a device-less box, an `--ignored` run of
/// this file now FAILS loudly (intended) rather than passing green having asserted
/// nothing.
fn dev() -> CudaDevice {
    fuel_test_support::require_gpu_run_lock();
    fuel_test_support::required_ok("CUDA device 0", CudaDevice::new(0))
}

/// `available_bytes` / `total_bytes` report real, self-consistent
/// numbers on a live GPU: total > 0 and available ≤ total. (The driver
/// query never fabricates; on a working device both are `Some`.)
#[test]
#[ignore]
fn cuda_backend_runtime_reports_sensible_memory() {
    let dev = dev();

    let avail = dev.available_bytes();
    let total = dev.total_bytes();
    eprintln!("CUDA memory: available={avail:?} total={total:?}");

    let (avail, total) = match (avail, total) {
        (Some(a), Some(t)) => (a, t),
        other => panic!("live GPU should report Some/Some, got {other:?}"),
    };
    assert!(total > 0, "total VRAM must be positive");
    assert!(
        avail <= total,
        "available ({avail}) must not exceed total ({total})"
    );
}

/// `would_fit` (default trait derivation) classifies allocations
/// against live state: a 1-byte alloc fits, an alloc larger than total
/// VRAM never fits.
#[test]
#[ignore]
fn cuda_backend_runtime_would_fit_classifies() {
    let dev = dev();
    let Some(total) = dev.total_bytes() else {
        panic!("live GPU should report a total")
    };

    // A 1-byte allocation must fit (Comfortable or Tight depending on
    // current load) — never WontFit / Unknown on a healthy device.
    match dev.would_fit(1) {
        FitStatus::Comfortable | FitStatus::Tight => {}
        other => panic!("1-byte alloc should fit, got {other:?}"),
    }

    // An allocation larger than the entire device cannot fit.
    assert_eq!(dev.would_fit(total + 1), FitStatus::WontFit);
}

/// The query is robust to being polled from a thread that never made
/// the device's context current — the impl pushes/pops the context
/// internally. Spawn a fresh thread and confirm it still gets a signal.
#[test]
#[ignore]
fn cuda_backend_runtime_works_off_dispatch_thread() {
    let dev = dev();
    let handle = std::thread::spawn(move || dev.available_bytes());
    let avail = handle.join().expect("polling thread panicked");
    assert!(
        avail.is_some(),
        "available_bytes should be Some when polled from a fresh thread \
         (context push/pop makes the query thread-independent)"
    );
}
