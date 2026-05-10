//! Live-device tests for the `Op::Recip` and `Op::Abs` realize arms
//! added to [`CudaGraphExecutor::eval_node`]. Gated `#[ignore]` —
//! run with `cargo test -p fuel-cuda-backend -- --ignored` on a host
//! with an NVIDIA GPU + CUDA Runtime SDK installed.

use fuel_core_types::{DType, Shape};
use fuel_cuda_backend::{CudaDevice, CudaGraphExecutor};
use fuel_graph::Tensor;
use std::sync::Arc;

fn dev_or_skip() -> Option<CudaDevice> {
    match CudaDevice::new(0) {
        Ok(d) => Some(d),
        Err(e) => {
            eprintln!("no CUDA device; skipping: {e:?}");
            None
        }
    }
}

fn cpu_dev() -> &'static Arc<dyn fuel_core_types::DynBackendDevice> {
    static D: std::sync::OnceLock<Arc<dyn fuel_core_types::DynBackendDevice>>
        = std::sync::OnceLock::new();
    D.get_or_init(|| Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
}

/// `Op::Recip` realized on CUDA matches `1.0 / x` element-wise.
/// Asserts bit-exact (the `urecip` PTX kernel is expected to use
/// IEEE-correct division, matching host `1.0 / x` to within a few ULP
/// — the integers we pick survive any 1-ULP wobble).
#[test]
#[ignore]
fn recip_realize_on_cuda_matches_reference() {
    let Some(dev) = dev_or_skip() else { return };
    let mut exec = CudaGraphExecutor::new(dev);

    let a = Tensor::from_f32(
        vec![2.0_f32, 4.0, 8.0, 16.0],
        Shape::from_dims(&[4]),
        cpu_dev(),
    );
    let r = a.recip();
    assert_eq!(r.dtype(), DType::F32);

    let out = exec.realize_f32(&r);
    let s = out.as_slice();
    assert_eq!(s.len(), 4);
    let expected = [0.5_f32, 0.25, 0.125, 0.0625];
    for (i, (&got, &want)) in s.iter().zip(expected.iter()).enumerate() {
        let diff = (got - want).abs();
        assert!(diff < 1e-7, "index {i}: got={got}, want={want} (diff={diff})");
    }
}

/// `Op::Abs` realized on CUDA matches `x.abs()` element-wise. The
/// `uabs` PTX kernel does a sign-bit clear, so this is bit-exact.
#[test]
#[ignore]
fn abs_realize_on_cuda_matches_reference() {
    let Some(dev) = dev_or_skip() else { return };
    let mut exec = CudaGraphExecutor::new(dev);

    let a = Tensor::from_f32(
        vec![-3.0_f32, 0.0, 3.0, -1.5, 2.5],
        Shape::from_dims(&[5]),
        cpu_dev(),
    );
    let b = a.abs();
    assert_eq!(b.dtype(), DType::F32);

    let out = exec.realize_f32(&b);
    assert_eq!(out.as_slice(), &[3.0_f32, 0.0, 3.0, 1.5, 2.5]);
}
