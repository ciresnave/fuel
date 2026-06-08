//! Bisect the CUDA composed-graph divergence:
//!
//!     matmul → rms_norm → matmul   (~77% rel error)
//!
//! Each test below isolates one combination so we can pin which
//! op pair (or interaction) introduces the drift.
//!
//! Skips cleanly when no CUDA device is visible.

#![cfg(feature = "cuda")]

use fuel_core::lazy::LazyTensor;
use fuel_core_types::{probe::BackendId, Shape};
use fuel_graph_executor::GraphExecutor;
use std::sync::Arc;

fn gen_lcg(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
    }).collect()
}

fn cuda_present() -> bool {
    let probe = fuel_core::probe::ProbeReport::probe_all();
    probe.devices.iter().any(|d| d.backend == BackendId::Cuda)
}

fn realize_both(t: &LazyTensor) -> (Vec<f32>, Vec<f32>) {
    let reference = t.realize_f32();
    let cuda_device = fuel_cuda_backend::CudaDevice::new(0)
        .expect("cuda device 0 should be available");
    let cuda = t.realize_f32_cuda(&cuda_device);
    (reference, cuda)
}

fn report(label: &str, ref_out: &[f32], cuda_out: &[f32]) {
    assert_eq!(ref_out.len(), cuda_out.len(),
        "{label}: length mismatch ref {} vs cuda {}",
        ref_out.len(), cuda_out.len());
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (&r, &c) in ref_out.iter().zip(cuda_out.iter()) {
        let abs = (r - c).abs();
        let denom = r.abs().max(c.abs()).max(f32::MIN_POSITIVE);
        let rel = abs / denom;
        if abs > max_abs { max_abs = abs; }
        if rel > max_rel { max_rel = rel; }
    }
    println!("{label}:  max_abs = {max_abs:.4e}  max_rel = {max_rel:.4e}  (n={})",
        ref_out.len());
    println!("  ref[0..4]:  {:?}", &ref_out[..4.min(ref_out.len())]);
    println!("  cuda[0..4]: {:?}", &cuda_out[..4.min(cuda_out.len())]);
}

/// Same shapes as the failing composed test, so the comparison
/// is apples-to-apples across the bisect cases below.
struct Inputs {
    x:    LazyTensor,
    w1:   LazyTensor,
    w2:   LazyTensor,
}

fn build_inputs() -> Inputs {
    let seq = 8_usize;
    let dim_in = 16_usize;
    let dim_mid = 32_usize;
    let dim_out = 8_usize;
    let x_data: Vec<f32> = gen_lcg(12345, seq * dim_in);
    let x = LazyTensor::from_f32(x_data, Shape::from_dims(&[1, seq, dim_in]), &fuel_core::Device::cpu());
    let w1 = x.const_f32_like(
        Arc::<[f32]>::from(gen_lcg(24691, dim_in * dim_mid)),
        Shape::from_dims(&[dim_in, dim_mid]),
    );
    let w2 = x.const_f32_like(
        Arc::<[f32]>::from(gen_lcg(37037, dim_mid * dim_out)),
        Shape::from_dims(&[dim_mid, dim_out]),
    );
    Inputs { x, w1, w2 }
}

#[test]
fn bisect_a_just_first_matmul() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_present() { return Ok(()); }
    let i = build_inputs();
    let y = i.x.matmul(&i.w1)?;
    let (r, c) = realize_both(&y);
    report("A: x @ w1 (rank-3)", &r, &c);
    fuel_core::test_utils::assert_allclose_f32(&c, &r, 1e-3, 1e-3);
    Ok(())
}

#[test]
fn bisect_b_matmul_then_rmsnorm() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_present() { return Ok(()); }
    let i = build_inputs();
    let y = i.x.matmul(&i.w1)?.rms_norm_last_dim(1e-5)?;
    let (r, c) = realize_both(&y);
    report("B: matmul → rms_norm", &r, &c);
    fuel_core::test_utils::assert_allclose_f32(&c, &r, 1e-3, 1e-3);
    Ok(())
}

#[test]
fn bisect_c_just_rmsnorm() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_present() { return Ok(()); }
    // [1, 8, 32] direct rms_norm — skip the upstream matmul, supply
    // the "post-matmul" tensor directly so we isolate rms_norm's
    // behaviour from any contiguity / stride state matmul might leave
    // behind.
    let seq = 8_usize;
    let dim_mid = 32_usize;
    let data: Vec<f32> = gen_lcg(12345, seq * dim_mid);
    let x = LazyTensor::from_f32(data, Shape::from_dims(&[1, seq, dim_mid]), &fuel_core::Device::cpu());
    let y = x.rms_norm_last_dim(1e-5)?;
    let (r, c) = realize_both(&y);
    report("C: rms_norm only", &r, &c);
    fuel_core::test_utils::assert_allclose_f32(&c, &r, 1e-3, 1e-3);
    Ok(())
}

#[test]
fn bisect_d_rmsnorm_then_matmul() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_present() { return Ok(()); }
    let seq = 8_usize;
    let dim_mid = 32_usize;
    let dim_out = 8_usize;
    let data: Vec<f32> = gen_lcg(12345, seq * dim_mid);
    let x = LazyTensor::from_f32(data, Shape::from_dims(&[1, seq, dim_mid]), &fuel_core::Device::cpu());
    let w2 = x.const_f32_like(
        Arc::<[f32]>::from(gen_lcg(37037, dim_mid * dim_out)),
        Shape::from_dims(&[dim_mid, dim_out]),
    );
    let y = x.rms_norm_last_dim(1e-5)?.matmul(&w2)?;
    let (r, c) = realize_both(&y);
    report("D: rms_norm → matmul", &r, &c);
    fuel_core::test_utils::assert_allclose_f32(&c, &r, 1e-3, 1e-3);
    Ok(())
}

#[test]
fn bisect_e_full_chain() -> Result<(), Box<dyn std::error::Error>> {
    if !cuda_present() { return Ok(()); }
    let i = build_inputs();
    let y = i.x.matmul(&i.w1)?.rms_norm_last_dim(1e-5)?.matmul(&i.w2)?;
    let (r, c) = realize_both(&y);
    report("E: matmul → rms_norm → matmul (full)", &r, &c);
    fuel_core::test_utils::assert_allclose_f32(&c, &r, 1e-3, 1e-3);
    Ok(())
}
