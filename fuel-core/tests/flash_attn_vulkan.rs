//! Phase 8 Tier 2 Vulkan parity gate: the Slang flash_attention
//! shader matches the reference attention_naive within tolerance.
//!
//! Skipped (returns early) when no Vulkan device is available, so
//! CI machines without a GPU stay green. To force-run on a dev rig:
//! `cargo test --test flash_attn_vulkan -p fuel-core --features vulkan`.

#![cfg(feature = "vulkan")]

use fuel_core::lazy::LazyTensor;
use fuel_core_types::Shape;
use fuel_graph_executor::GraphExecutor;
use fuel_graph_vulkan::{DeviceSelection, VulkanBackend};

fn rand_f32(shape: &[usize], seed: u32) -> Vec<f32> {
    let n: usize = shape.iter().product();
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        v.push(((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.5);
    }
    v
}

fn vulkan_executor() -> Option<GraphExecutor<VulkanBackend>> {
    match VulkanBackend::with_selection(DeviceSelection::PreferDiscrete) {
        Ok(b) => Some(GraphExecutor::new(b)),
        Err(e) => {
            eprintln!("skipping: no Vulkan device ({e:?})");
            None
        }
    }
}

fn assert_close(label: &str, vk: &[f32], reference: &[f32], atol: f32, rtol: f32) {
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
            "{label}[{i}]: vk={a} ref={b} (abs={diff}, rel={rel})",
        );
    }
}

fn run_case(label: &str, b: usize, hq: usize, hkv: usize, sq: usize, sk: usize, d: usize, causal: bool) {
    let mut exe = match vulkan_executor() { Some(e) => e, None => return };
    let scale = 1.0_f32 / (d as f32).sqrt();
    let q_data = rand_f32(&[b, hq, sq, d], 1);
    let k_data = rand_f32(&[b, hkv, sk, d], 2);
    let v_data = rand_f32(&[b, hkv, sk, d], 3);

    let q = LazyTensor::from_f32(q_data, Shape::from_dims(&[b, hq, sq, d]), &fuel_core::Device::cpu());
    let k = q.const_f32_like(k_data, Shape::from_dims(&[b, hkv, sk, d]));
    let v = q.const_f32_like(v_data, Shape::from_dims(&[b, hkv, sk, d]));
    let out = q.flash_attn(&k, &v, None, scale, causal, None, None, None);
    let vk = out.realize_f32_vulkan(&mut exe);
    let reference = out.realize_f32_reference();
    assert_close(label, &vk, &reference, 5e-4, 5e-4);
}

#[test]
fn vulkan_flash_attn_basic_no_mask() {
    run_case("vulkan basic", 1, 2, 2, 16, 16, 16, false);
}

#[test]
fn vulkan_flash_attn_causal() {
    run_case("vulkan causal", 1, 2, 2, 16, 16, 16, true);
}

#[test]
fn vulkan_flash_attn_gqa_causal() {
    // Hq=8, Hkv=2 — LLaMA-style GQA.
    run_case("vulkan GQA", 1, 8, 2, 16, 16, 16, true);
}

#[test]
fn vulkan_flash_attn_uneven_seqlens() {
    // Seqlens not multiples of BR=BC=16.
    run_case("vulkan uneven", 1, 2, 2, 23, 19, 8, true);
}
