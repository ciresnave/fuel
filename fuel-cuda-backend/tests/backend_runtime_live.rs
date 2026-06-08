//! Live-device tests for `impl BackendRuntime for CudaDevice` — the
//! contract-v0.3 memory-pressure surface backed by baracuda alpha.66's
//! `cuMemGetInfo` wrappers. Gated `#[ignore]` — run with
//! `cargo test -p fuel-cuda-backend --test backend_runtime_live -- --ignored`
//! on a host with an NVIDIA GPU + CUDA Runtime SDK installed.

use fuel_core_types::backend::{BackendRuntime, FitStatus};
use fuel_cuda_backend::CudaDevice;

fn dev_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("no CUDA device; skipping: {e:?}");
            None
        }
    }
}

/// `available_bytes` / `total_bytes` report real, self-consistent
/// numbers on a live GPU: total > 0 and available ≤ total. (The driver
/// query never fabricates; on a working device both are `Some`.)
#[test]
#[ignore]
fn cuda_backend_runtime_reports_sensible_memory() {
    let Some(dev) = dev_or_skip() else { return };

    let avail = dev.available_bytes();
    let total = dev.total_bytes();
    eprintln!("CUDA memory: available={avail:?} total={total:?}");

    let (avail, total) = match (avail, total) {
        (Some(a), Some(t)) => (a, t),
        other => panic!("live GPU should report Some/Some, got {other:?}"),
    };
    assert!(total > 0, "total VRAM must be positive");
    assert!(avail <= total, "available ({avail}) must not exceed total ({total})");
}

/// `would_fit` (default trait derivation) classifies allocations
/// against live state: a 1-byte alloc fits, an alloc larger than total
/// VRAM never fits.
#[test]
#[ignore]
fn cuda_backend_runtime_would_fit_classifies() {
    let Some(dev) = dev_or_skip() else { return };
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
    let Some(dev) = dev_or_skip() else { return };
    let handle = std::thread::spawn(move || dev.available_bytes());
    let avail = handle.join().expect("polling thread panicked");
    assert!(
        avail.is_some(),
        "available_bytes should be Some when polled from a fresh thread \
         (context push/pop makes the query thread-independent)"
    );
}
