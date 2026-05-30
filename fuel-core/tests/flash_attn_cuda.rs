//! Phase 8 Tier 3 CUDA parity gate: baracuda's FA2 kernels
//! (alpha.60, accessed via `baracuda_kernels_fa2_sdpa_*_run_v2`)
//! match the reference attention_naive within F16 precision.
//!
//! Vendor tree is Dao-AILab FA2 v2.8.3 + the {160, 224, 512} head_dim
//! expansion absorbed from Candle/Fuel patches (PRs #245, #2688, #3417).
//! Gated on `flash-attn` (transitional) until the eager wrapper migrates
//! off `fuel-flash-attn-cuda-sys`; baracuda's `fa2` symbols are always-on
//! at the workspace level so the feature no longer pulls nvcc work.
//!
//! Architecture note: the lazy `LazyTensor::realize_f32_cuda` path was
//! moved to PipelinedExecutor in Phase 9c, and `Op::Fused(FLASH_ATTN, _)`
//! has NOT yet been registered in the binding-table dispatch (one of the
//! ~12 PipelinedExecutor parity gaps tracked in
//! `project_phase_7_6_step_9c_parity_audit`). Until that lands, this test
//! exercises the FA2 launcher through the trait-based `GraphExecutor` on
//! eager `Tensor`s — same `CudaBackend::flash_attn` trait method, same
//! `crate::flash_attn::launch` implementation, just a different executor
//! frame.

#![cfg(all(feature = "cuda", feature = "flash-attn"))]

use fuel_core_types::{DType, Shape, probe::BackendId};
use fuel_graph::Tensor;
use fuel_graph_executor::GraphExecutor;
use half::f16;
use std::sync::Arc;

fn rand_f16(shape: &[usize], seed: u32) -> Vec<f16> {
    let n: usize = shape.iter().product();
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let r = ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.5;
        v.push(f16::from_f32(r));
    }
    v
}

fn cuda_present() -> bool {
    fuel_core::probe::ProbeReport::probe_all()
        .devices.iter()
        .any(|d| d.backend == BackendId::Cuda)
}

fn cuda_executor() -> GraphExecutor<fuel_cuda_backend::CudaBackend> {
    let dev = fuel_cuda_backend::CudaDevice::new(0).expect("cuda device 0");
    GraphExecutor::new(fuel_cuda_backend::CudaBackend::new(dev))
}

fn cpu_dev() -> &'static Arc<dyn fuel_core_types::DynBackendDevice> {
    static D: std::sync::OnceLock<Arc<dyn fuel_core_types::DynBackendDevice>>
        = std::sync::OnceLock::new();
    D.get_or_init(|| Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
}

fn assert_close_f16(label: &str, vk: &[f32], reference: &[f32], atol: f32, rtol: f32) {
    assert_eq!(vk.len(), reference.len(), "{label}: length mismatch");
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_idx = 0;
    for (i, (&a, &b)) in vk.iter().zip(reference.iter()).enumerate() {
        let diff = (a - b).abs();
        let denom = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        if diff > max_abs { max_abs = diff; max_rel = rel; max_idx = i; }
    }
    eprintln!("{label}: max abs={max_abs} rel={max_rel} at idx {max_idx}");
    for (i, (&a, &b)) in vk.iter().zip(reference.iter()).enumerate() {
        let diff = (a - b).abs();
        let denom = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(
            diff < atol || rel < rtol,
            "{label}[{i}]: cuda={a} ref={b} (abs={diff} rel={rel})",
        );
    }
}

#[test]
fn cuda_flash_attn_basic_f16_no_mask() {
    if !cuda_present() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    // d=32 is the smallest head_dim baracuda FA2 supports (set is
    // {32, 64, 96, 128, 160, 192, 224, 256, 512}).
    let (b, h, sq, sk, d) = (1usize, 2, 16, 16, 32);
    let scale = 1.0_f32 / (d as f32).sqrt();
    let q_data = rand_f16(&[b, h, sq, d], 1);
    let k_data = rand_f16(&[b, h, sk, d], 2);
    let v_data = rand_f16(&[b, h, sk, d], 3);

    // Build the F16 graph + run via baracuda FA2 through the
    // CudaBackend::flash_attn trait method.
    let q = Tensor::from_f16(q_data.clone(), Shape::from_dims(&[b, h, sq, d]), cpu_dev());
    let k = q.const_f16_like(k_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let v = q.const_f16_like(v_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let out = q.flash_attn(&k, &v, None, scale, false, None, None, None)
        .cast(DType::F32);
    let mut exe = cuda_executor();
    let cuda = exe.realize_f32(&out);

    // Reference: same graph in F32 on the reference backend. F16 → F32
    // cast at the seed values keeps per-element rounding identical.
    let q_f32: Vec<f32> = q_data.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_data.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_data.iter().map(|x| x.to_f32()).collect();
    let q2 = Tensor::from_f32(q_f32, Shape::from_dims(&[b, h, sq, d]), cpu_dev());
    let k2 = q2.const_f32_like(k_f32, Shape::from_dims(&[b, h, sk, d]));
    let v2 = q2.const_f32_like(v_f32, Shape::from_dims(&[b, h, sk, d]));
    let out2 = q2.flash_attn(&k2, &v2, None, scale, false, None, None, None);
    let reference = fuel_reference_backend::exec::realize_f32(&out2);

    assert_close_f16("cuda flash-attn F16 basic", cuda.as_slice(), reference.as_slice(), 5e-3, 5e-3);
}

#[test]
fn cuda_flash_attn_causal_f16() {
    if !cuda_present() { return; }
    // d=32 is the smallest head_dim baracuda FA2 supports (set is
    // {32, 64, 96, 128, 160, 192, 224, 256, 512}).
    let (b, h, sq, sk, d) = (1usize, 2, 16, 16, 32);
    let scale = 1.0_f32 / (d as f32).sqrt();
    let q_data = rand_f16(&[b, h, sq, d], 4);
    let k_data = rand_f16(&[b, h, sk, d], 5);
    let v_data = rand_f16(&[b, h, sk, d], 6);

    let q = Tensor::from_f16(q_data.clone(), Shape::from_dims(&[b, h, sq, d]), cpu_dev());
    let k = q.const_f16_like(k_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let v = q.const_f16_like(v_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let out = q.flash_attn(&k, &v, None, scale, true, None, None, None)
        .cast(DType::F32);
    let mut exe = cuda_executor();
    let cuda = exe.realize_f32(&out);

    let q_f32: Vec<f32> = q_data.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_data.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_data.iter().map(|x| x.to_f32()).collect();
    let q2 = Tensor::from_f32(q_f32, Shape::from_dims(&[b, h, sq, d]), cpu_dev());
    let k2 = q2.const_f32_like(k_f32, Shape::from_dims(&[b, h, sk, d]));
    let v2 = q2.const_f32_like(v_f32, Shape::from_dims(&[b, h, sk, d]));
    let out2 = q2.flash_attn(&k2, &v2, None, scale, true, None, None, None);
    let reference = fuel_reference_backend::exec::realize_f32(&out2);

    assert_close_f16("cuda flash-attn F16 causal", cuda.as_slice(), reference.as_slice(), 5e-3, 5e-3);
}
