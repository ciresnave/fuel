//! Phase 6b oracle-gate proof-of-mechanism: a single matmul runs on
//! CUDA with output matching the reference backend.
//!
//! PR #6's CUDA kernel parity suite validated rope / rms_norm /
//! matmul / matmul_q4_0 / matmul_q4_km as individual ops, each bit-
//! equivalent to the reference. This test is a Phase 6b-specific
//! gate confirming the `realize_f32_reference()` vs
//! `realize_f32_cuda(&mut exe)` oracle comparison loop — the thing
//! the Judge relies on to produce its `max_rel_error` numbers — is
//! wired correctly end-to-end.
//!
//! Two follow-ups are in flight separately:
//!
//! - `composed_subgraph_cuda_diverges_from_reference` — a
//!   matmul + rms_norm + matmul composition produces large
//!   divergence (~77% rel error) even though each op passes its
//!   PR #6 parity test in isolation. Memory entry
//!   `project_cuda_composed_divergence.md` tracks this.
//!
//! - Full LLaMA forward on CUDA produces NaN. Memory entry
//!   `project_cuda_llama_nan.md` tracks this.
//!
//! Neither blocks Phase 6b's probe → judge → dispatch machinery, all
//! of which operates regardless of how many anchors happen to be
//! in numerical parity today.
//!
//! Feature-gated on `cuda` and requires a CUDA device. Skips
//! cleanly when no CUDA visible.

#![cfg(feature = "cuda")]

use fuel_core::lazy::LazyTensor;
use fuel_core_types::{probe::BackendId, Shape};
use fuel_graph_executor::GraphExecutor;

#[test]
fn single_matmul_cuda_matches_reference_within_tolerance() {
    // Skip cleanly when no CUDA device is visible.
    let probe = fuel_core::probe::ProbeReport::probe_all();
    let has_cuda = probe.devices.iter().any(|d| d.backend == BackendId::Cuda);
    if !has_cuda {
        eprintln!("skipping: no CUDA device visible to Fuel");
        return;
    }

    // 32×48 @ 48×24 — deterministic deterministic inputs.
    let (m, k, n) = (32usize, 48, 24);
    let a_data: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 1.3e-3).sin()).collect();
    let b_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 1.7e-3).cos()).collect();
    let a = LazyTensor::from_f32(a_data, Shape::from_dims(&[m, k]));
    let b = a.const_f32_like(b_data, Shape::from_dims(&[k, n]));
    let c = a.matmul(&b);

    let reference = c.realize_f32_reference();

    let cuda_device = fuel_graph_cuda::CudaDevice::new(0)
        .expect("cuda device 0 should be available");
    let mut cuda_exe = GraphExecutor::new(
        fuel_graph_cuda::CudaBackend::new(cuda_device),
    );
    let cuda_out = c.realize_f32_cuda(&mut cuda_exe);

    assert_eq!(reference.len(), cuda_out.len());
    assert_eq!(reference.len(), m * n);

    // Tight tolerance — matmul bit-parity per PR #6's individual
    // kernel suite. Any drift is gemm sum-order accumulation and is
    // well under 1e-4 at these shapes.
    fuel_core::test_utils::assert_allclose_f32(&cuda_out, &reference, 1e-4, 1e-4);
}
