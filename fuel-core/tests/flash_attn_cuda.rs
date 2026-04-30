//! Phase 8 Tier 3 CUDA parity gate: the Dao-AILab FlashAttention v2
//! kernels (sm80, accessed via fuel-flash-attn-cuda-sys) match the
//! reference attention_naive within F16 precision.
//!
//! Only runs when the `flash-attn` feature is enabled on fuel-core +
//! fuel-graph-cuda — gates the heavy nvcc build behind an explicit opt-in.

#![cfg(all(feature = "cuda", feature = "flash-attn"))]

use fuel_core::lazy::LazyTensor;
use fuel_core_types::{DType, Shape, probe::BackendId};
use fuel_graph_executor::GraphExecutor;
use half::f16;

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

fn cuda_executor() -> GraphExecutor<fuel_graph_cuda::CudaBackend> {
    let dev = fuel_graph_cuda::CudaDevice::new(0).expect("cuda device 0");
    GraphExecutor::new(fuel_graph_cuda::CudaBackend::new(dev))
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
    let (b, h, sq, sk, d) = (1usize, 2, 16, 16, 16);
    let scale = 1.0_f32 / (d as f32).sqrt();
    let q_data = rand_f16(&[b, h, sq, d], 1);
    let k_data = rand_f16(&[b, h, sk, d], 2);
    let v_data = rand_f16(&[b, h, sk, d], 3);

    // Build the F16 lazy graph + run via CUDA flash-attn.
    let q = LazyTensor::from_f16(q_data.clone(), Shape::from_dims(&[b, h, sq, d]));
    let k = q.const_f16_like(k_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let v = q.const_f16_like(v_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let out = q.flash_attn(&k, &v, None, scale, false, None, None, None)
        .cast(DType::F32);

    let mut exe = cuda_executor();
    let cuda = out.realize_f32_cuda(&mut exe);

    // Reference: build the same graph in F32 (for tighter ref precision)
    // and realize on the reference backend. F16 → F32 cast at the seed
    // values keeps the per-element rounding identical.
    let q_f32: Vec<f32> = q_data.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_data.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_data.iter().map(|x| x.to_f32()).collect();
    let q2 = LazyTensor::from_f32(q_f32, Shape::from_dims(&[b, h, sq, d]));
    let k2 = q2.const_f32_like(k_f32, Shape::from_dims(&[b, h, sk, d]));
    let v2 = q2.const_f32_like(v_f32, Shape::from_dims(&[b, h, sk, d]));
    let out2 = q2.flash_attn(&k2, &v2, None, scale, false, None, None, None);
    let reference = out2.realize_f32_reference();

    // F16 precision tolerance — looser than the Vulkan F32 test.
    assert_close_f16("cuda flash-attn F16 basic", &cuda, &reference, 5e-3, 5e-3);
}

#[test]
fn cuda_flash_attn_causal_f16() {
    if !cuda_present() { return; }
    let (b, h, sq, sk, d) = (1usize, 2, 16, 16, 16);
    let scale = 1.0_f32 / (d as f32).sqrt();
    let q_data = rand_f16(&[b, h, sq, d], 4);
    let k_data = rand_f16(&[b, h, sk, d], 5);
    let v_data = rand_f16(&[b, h, sk, d], 6);

    let q = LazyTensor::from_f16(q_data.clone(), Shape::from_dims(&[b, h, sq, d]));
    let k = q.const_f16_like(k_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let v = q.const_f16_like(v_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let out = q.flash_attn(&k, &v, None, scale, true, None, None, None)
        .cast(DType::F32);

    let mut exe = cuda_executor();
    let cuda = out.realize_f32_cuda(&mut exe);

    let q_f32: Vec<f32> = q_data.iter().map(|x| x.to_f32()).collect();
    let k_f32: Vec<f32> = k_data.iter().map(|x| x.to_f32()).collect();
    let v_f32: Vec<f32> = v_data.iter().map(|x| x.to_f32()).collect();
    let q2 = LazyTensor::from_f32(q_f32, Shape::from_dims(&[b, h, sq, d]));
    let k2 = q2.const_f32_like(k_f32, Shape::from_dims(&[b, h, sk, d]));
    let v2 = q2.const_f32_like(v_f32, Shape::from_dims(&[b, h, sk, d]));
    let out2 = q2.flash_attn(&k2, &v2, None, scale, true, None, None, None);
    let reference = out2.realize_f32_reference();

    assert_close_f16("cuda flash-attn F16 causal", &cuda, &reference, 5e-3, 5e-3);
}
