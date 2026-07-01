//! Live-CUDA realize under Step E A3 (CUDA async dispatch).
//!
//! A3 removed the per-op `device.synchronize()` from every baracuda compute op
//! and routed `alloc_zeros` → `zeros_async`, so kernels pipeline on the single
//! per-device stream and op buffers (outputs + scratch) free STREAM-ORDERED on
//! `Drop` (`cuMemFreeAsync` on the retained origin stream). Correctness now
//! rests on two things this suite stresses:
//!   1. same-stream submission order = execution order carries producer→consumer
//!      deps with no inline sync, and
//!   2. an intermediate dropped while its kernel is still in flight is freed
//!      *after* that kernel (stream-ordered), so the pool can't hand its block
//!      to a later alloc that overwrites a value still being read.
//! A UAF or ordering regression in the async change corrupts the output; a sync
//! version and a correct async version both produce the references below (so
//! this is a correctness-regression guard, not a born-red test — removing the
//! syncs doesn't change the *math*, only when it runs). The references are the
//! exact bytes the CPU `--lib` suite and the live-Vulkan `vulkan_bridge_realize_live`
//! twin assert, so the `optimize_graph` realize path is proven identical on CUDA.
//!
//! Gated `#[ignore]`; requires a live NVIDIA GPU + CUDA Runtime SDK. Run:
//!   cargo test -p fuel-core --features cuda --test cuda_async_realize_live -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use fuel_core::lazy::LazyTensor;
use fuel_cuda_backend::CudaDevice;
use fuel_ir::{DType, DeviceLocation, Shape};

fn dev_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("no CUDA device; skipping: {e:?}");
            None
        }
    }
}

/// `(a + b) * a` realized on CUDA matches the host oracle `[11, 44, 99, 176]` —
/// the same value the CPU suite and the Vulkan twin assert. A two-op chain
/// pipelined on the stream with no per-op sync (A3): `add`'s output is consumed
/// by `mul` in submission order.
#[test]
#[ignore = "requires a live CUDA device"]
fn mul_add_realize_on_cuda_matches_reference() {
    let Some(dev) = dev_or_skip() else { return };

    let a = LazyTensor::from_f32(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        &fuel_core::Device::cpu(),
    );
    // `const_f32_like` keeps `b` in `a`'s graph (a bare second `from_f32` would
    // mint a separate graph and `add` across graphs would fail).
    let b = a.const_f32_like(vec![10.0_f32, 20.0, 30.0, 40.0], Shape::from_dims(&[4]));
    let c = a.add(&b).expect("add").mul(&a).expect("mul");
    assert_eq!(c.dtype(), DType::F32);

    let out = c.realize_f32_cuda(&dev);
    assert_eq!(out, vec![11.0_f32, 44.0, 99.0, 176.0]);
}

/// Step E A3 (CUDA async): a DEEPER chain with fan-out, so several CUDA ops
/// accumulate in flight on the stream and an intermediate is consumed twice
/// (`t1`) then dropped while later ops are still pending — stressing
/// same-stream producer→consumer ordering and the stream-ordered free of
/// intermediates. The host oracle:
///   t1 = a+b = [11,22,33,44];  t2 = t1*a = [11,44,99,176]
///   t3 = t2+t1 = [22,66,132,220];  out = t3*a = [22,132,396,880]
/// Correct bytes under deferred (pipelined) dispatch ⇒ A3's ordering + free
/// are sound.
#[test]
#[ignore = "requires a live CUDA device"]
fn deep_chain_realize_on_cuda_matches_reference() {
    let Some(dev) = dev_or_skip() else { return };

    let a = LazyTensor::from_f32(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        &fuel_core::Device::cpu(),
    );
    let b = a.const_f32_like(vec![10.0_f32, 20.0, 30.0, 40.0], Shape::from_dims(&[4]));

    let t1 = a.add(&b).expect("add");
    let t2 = t1.mul(&a).expect("mul");
    let t3 = t2.add(&t1).expect("add2"); // reuses t1 (fan-out)
    let out_t = t3.mul(&a).expect("mul2");

    let out = out_t.realize_f32_cuda(&dev);
    assert_eq!(out, vec![22.0_f32, 132.0, 396.0, 880.0]);

    // Step E Phase C / B1 — single-device in-flight counter BALANCE. This deep
    // CUDA chain submits several events (one per producing node) and drains them
    // all at realize-end (drain_handles + the per-copy/eviction waits). After the
    // drain the per-device count MUST be 0: every CudaCompletion::new (+1) was
    // matched by exactly one Drop (-1), covering wait, drain, and eviction-drop.
    // No leak, no underflow. B1 is behavior-preserving — the byte-exact assert
    // above is unchanged by the counter.
    assert_eq!(
        fuel_dispatch::dispatch::inflight_count(DeviceLocation::Cuda { gpu_id: 0 }),
        0,
        "B1: CUDA in-flight count must return to 0 after the chain fully drains",
    );
}

/// Pool-reuse / stream-ordered-free pressure: a longer add-chain on a larger
/// buffer (1024 elems) so the stream-ordered mem-pool repeatedly frees and
/// re-hands the same footprint across many in-flight ops. If a freed
/// intermediate's block were reissued before its kernel finished (a
/// stream-ordering bug in the async free), the accumulated result would be
/// corrupted. `t = a; t += 1.0  (x32)` ⇒ `a + 32`, exact in f32 for these
/// magnitudes; computed against a host mirror.
#[test]
#[ignore = "requires a live CUDA device"]
fn long_chain_pool_reuse_on_cuda_matches_reference() {
    let Some(dev) = dev_or_skip() else { return };

    const N: usize = 1024;
    const STEPS: usize = 32;

    let a_host: Vec<f32> = (0..N).map(|i| (i % 17) as f32).collect();
    let a = LazyTensor::from_f32(a_host.clone(), Shape::from_dims(&[N]), &fuel_core::Device::cpu());
    let one = a.const_f32_like(vec![1.0_f32; N], Shape::from_dims(&[N]));

    let mut t = a.add(&one).expect("add 1");
    for _ in 1..STEPS {
        t = t.add(&one).expect("add step");
    }

    let out = t.realize_f32_cuda(&dev);
    let expected: Vec<f32> = a_host.iter().map(|v| v + STEPS as f32).collect();
    assert_eq!(out, expected);
}
