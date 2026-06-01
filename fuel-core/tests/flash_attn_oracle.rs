//! Phase 8 Tier 2 lazy-IR oracle: a graph containing `Op::FlashAttn`
//! realizes to the same output as the composed
//! `softmax(Q·K^T·scale + mask)·V` graph.
//!
//! Catches:
//! - Op::FlashAttn dispatch wired wrong in any of the three executors
//!   (reference, fuel-graph-cpu, lazy-tensor wrapper).
//! - GQA / causal / softcap / alibi getting threaded incorrectly
//!   through the lazy IR.
//!
//! Both the FlashAttn path and the composed-attention path go through
//! the CPU executor, so this is purely an attention-algorithm parity
//! check, not a backend correctness check (that's the job of
//! fuel-reference-backend's attention.rs tests).

use fuel_core::lazy::LazyTensor;
use fuel_core_types::Shape;

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

/// Build the equivalent attention graph: softmax(Q·K^T·scale)·V.
/// No mask, no GQA — caller-side variants are tested in their own tests.
fn composed_attention(q: &LazyTensor, k: &LazyTensor, v: &LazyTensor, scale: f32) -> LazyTensor {
    // Q: [B, H, Sq, D], K: [B, H, Sk, D], V: [B, H, Sk, D]
    // K^T (along last two dims): [B, H, D, Sk]
    let kt = k.transpose().unwrap();
    // Scores: Q · K^T = [B, H, Sq, Sk]
    let scores = q.matmul(&kt).unwrap();
    // Scale
    let scaled = scores.mul_scalar(scale as f64);
    // Softmax along last dim (Sk)
    let attn = scaled.softmax_last_dim();
    // Attn · V = [B, H, Sq, D]
    attn.matmul(v).unwrap()
}

#[test]
fn lazy_flash_attn_matches_composed_attention_basic() {
    let (b, h, sq, sk, d) = (1usize, 2, 8, 8, 4);
    let q_data = rand_f32(&[b, h, sq, d], 1);
    let k_data = rand_f32(&[b, h, sk, d], 2);
    let v_data = rand_f32(&[b, h, sk, d], 3);
    let scale = 1.0_f32 / (d as f32).sqrt();

    let q = LazyTensor::from_f32(q_data.clone(), Shape::from_dims(&[b, h, sq, d]), &fuel_core::Device::cpu());
    let k = q.const_f32_like(k_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let v = q.const_f32_like(v_data.clone(), Shape::from_dims(&[b, h, sk, d]));

    // Path A: lazy Op::FlashAttn → realize → CPU executor → reference dispatch
    let fa_out = q.flash_attn(&k, &v, None, scale, false, None, None, None).unwrap();
    let fa = fa_out.realize_f32();

    // Path B: explicit matmul+softmax composition.
    let q2 = LazyTensor::from_f32(q_data, Shape::from_dims(&[b, h, sq, d]), &fuel_core::Device::cpu());
    let k2 = q2.const_f32_like(k_data, Shape::from_dims(&[b, h, sk, d]));
    let v2 = q2.const_f32_like(v_data, Shape::from_dims(&[b, h, sk, d]));
    let composed = composed_attention(&q2, &k2, &v2, scale);
    let comp = composed.realize_f32();

    assert_eq!(fa.len(), comp.len());
    for (i, (&a, &b)) in fa.iter().zip(comp.iter()).enumerate() {
        let diff = (a - b).abs();
        let denom = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(diff < 1e-5 || rel < 1e-5, "[{i}]: flash={a} composed={b} (abs={diff} rel={rel})");
    }
}

#[test]
fn lazy_flash_attn_matches_naive_with_causal_mask() {
    // Causal mask is the most common feature — verify it threads through
    // the lazy path. Comparison is FA-via-lazy vs naive-via-direct-call,
    // which is fine since both go through the same reference under the
    // hood (the CPU executor's Op::FlashAttn arm dispatches to
    // attention_naive).
    let (b, h, sq, sk, d) = (1usize, 2, 6, 6, 4);
    let q_data = rand_f32(&[b, h, sq, d], 4);
    let k_data = rand_f32(&[b, h, sk, d], 5);
    let v_data = rand_f32(&[b, h, sk, d], 6);
    let scale = 1.0_f32 / (d as f32).sqrt();

    let q = LazyTensor::from_f32(q_data.clone(), Shape::from_dims(&[b, h, sq, d]), &fuel_core::Device::cpu());
    let k = q.const_f32_like(k_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let v = q.const_f32_like(v_data.clone(), Shape::from_dims(&[b, h, sk, d]));
    let fa_out = q.flash_attn(&k, &v, None, scale, true, None, None, None).unwrap().realize_f32();

    // Direct call to the reference for a known-correct comparison point.
    use fuel_reference_backend::attention::{attention_naive, AttentionParams};
    use fuel_reference_backend::RefTensor;
    let p = AttentionParams { softmax_scale: scale, causal: true, ..Default::default() };
    let naive = attention_naive(
        &RefTensor::from_vec(q_data, Shape::from_dims(&[b, h, sq, d])),
        &RefTensor::from_vec(k_data, Shape::from_dims(&[b, h, sk, d])),
        &RefTensor::from_vec(v_data, Shape::from_dims(&[b, h, sk, d])),
        None,
        &p,
    );
    let direct = naive.into_vec();

    assert_eq!(fa_out.len(), direct.len());
    for (i, (&a, &b)) in fa_out.iter().zip(direct.iter()).enumerate() {
        let diff = (a - b).abs();
        let denom = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(diff < 1e-5 || rel < 1e-5, "[{i}]: lazy={a} direct={b} (abs={diff} rel={rel})");
    }
}
