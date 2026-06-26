//! Live-Vulkan bridge realize — proves the `optimize_graph` realize
//! path (the single path after Phase A PR-A3b-2) realizes correctly on
//! the Vulkan backend through `pipelined_bridge`
//! (`LazyTensor::realize_f32_vulkan` → `pipelined_bridge::realize_one_as`
//! → `PipelinedExecutor`), matching the host oracle. The path is
//! backend-agnostic (it lives in the generic `realize_one_as`), so this
//! is the Vulkan counterpart of the CPU `--lib` suite and the live-CUDA
//! `recip_abs_realize_live` / `phase_c_rotating_kv_cuda` smokes.
//!
//! Run:
//!
//!   cargo test -p fuel-core --features vulkan --test vulkan_bridge_realize_live -- --ignored --test-threads=1
//!
//! Gated `#[ignore]`; requires a live Vulkan device (RTX 4070 on the dev box).

#![cfg(feature = "vulkan")]

use std::sync::Arc;

use fuel_core::lazy::LazyTensor;
use fuel_ir::{DType, Shape};
use fuel_vulkan_backend::VulkanBackend;

fn backend_or_skip() -> Option<Arc<VulkanBackend>> {
    match VulkanBackend::new() {
        Ok(b) => Some(Arc::new(b)),
        Err(e) => {
            eprintln!("no Vulkan device; skipping: {e:?}");
            None
        }
    }
}

/// `(a + b) * a` realized on Vulkan through the bridge matches the host
/// oracle — the exact `[11, 44, 99, 176]` the CPU suite and the CUDA
/// smoke assert, so the `optimize_graph` realize path is correct on the
/// Vulkan backend too.
#[test]
#[ignore = "requires a live Vulkan device"]
fn mul_add_realize_on_vulkan_matches_reference() {
    let Some(backend) = backend_or_skip() else { return };

    let a = LazyTensor::from_f32(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        Shape::from_dims(&[4]),
        &fuel_core::Device::cpu(),
    );
    // `const_f32_like` keeps `b` in `a`'s graph (a bare second
    // `from_f32` would mint a separate graph and `add` across graphs
    // would fail).
    let b = a.const_f32_like(vec![10.0_f32, 20.0, 30.0, 40.0], Shape::from_dims(&[4]));
    // LazyTensor binary ops are fallible (shape/broadcast validation),
    // unlike the graph-`Tensor` ops; surface any error rather than panic
    // implicitly.
    let c = a.add(&b).expect("add").mul(&a).expect("mul");
    assert_eq!(c.dtype(), DType::F32);

    let out = c.realize_f32_vulkan(&backend);
    assert_eq!(out, vec![11.0_f32, 44.0, 99.0, 176.0]);
}
