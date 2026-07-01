//! Live-device tests for element-wise `Op::Recip` and `Op::Abs`
//! realized on CUDA.
//!
//! Ported in executor-unification Session 1 (re-audit `846f7908`) from
//! `fuel-cuda-backend/tests/recip_abs_realize_live.rs`, where they
//! pinned the legacy `CudaGraphExecutor::eval_node` realize arms. That
//! struct is retiring; the executor under test is now the production
//! path — `LazyTensor::realize_f32_cuda` →
//! `pipelined_bridge::realize_one_as` → `PipelinedExecutor` →
//! binding-table dispatch onto baracuda's `unary_recip_f32` /
//! `unary_abs_f32`. The file moved here because the pipelined entries
//! live in fuel-core (fuel-cuda-backend cannot depend on fuel-core).
//! Reference values and tolerances are unchanged from the original.
//!
//! Gated `#[ignore]` — run with
//! `cargo test -p fuel-core --features cuda --test recip_abs_realize_live -- --ignored`
//! on a host with an NVIDIA GPU + CUDA Runtime SDK installed.

#![cfg(feature = "cuda")]

use fuel_core::lazy::LazyTensor;
use fuel_ir::{DType, Shape};
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

/// `Op::Recip` realized on CUDA matches `1.0 / x` element-wise.
/// Asserts near-bit-exact (the baracuda recip kernel is expected to use
/// IEEE-correct division, matching host `1.0 / x` to within a few ULP
/// — the integers we pick survive any 1-ULP wobble).
#[test]
#[ignore]
fn recip_realize_on_cuda_matches_reference() {
    let Some(dev) = dev_or_skip() else { return };

    let a = LazyTensor::from_f32(
        vec![2.0_f32, 4.0, 8.0, 16.0],
        Shape::from_dims(&[4]),
        &fuel_core::Device::cpu(),
    );
    let r = a.recip();
    assert_eq!(r.dtype(), DType::F32);

    let out = r.realize_f32_cuda(&dev);
    assert_eq!(out.len(), 4);
    let expected = [0.5_f32, 0.25, 0.125, 0.0625];
    for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(diff < 1e-7, "index {i}: got={got}, want={want} (diff={diff})");
    }
}

/// `Op::Abs` realized on CUDA matches `x.abs()` element-wise. The
/// abs kernel does a sign-bit clear, so this is bit-exact.
#[test]
#[ignore]
fn abs_realize_on_cuda_matches_reference() {
    let Some(dev) = dev_or_skip() else { return };

    let a = LazyTensor::from_f32(
        vec![-3.0_f32, 0.0, 3.0, -1.5, 2.5],
        Shape::from_dims(&[5]),
        &fuel_core::Device::cpu(),
    );
    let b = a.abs();
    assert_eq!(b.dtype(), DType::F32);

    let out = b.realize_f32_cuda(&dev);
    assert_eq!(out, &[3.0_f32, 0.0, 3.0, 1.5, 2.5]);
}
