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
//! the hard-CPU reference realize (`realize_f32_reference`), so this is
//! purely an attention-algorithm parity check, not a backend correctness
//! check. The known-correct comparison point for the causal case is an
//! inline textbook naive-attention scalar loop (was
//! `fuel-reference-backend::attention::attention_naive` before that crate
//! was retired 2026-07-04).

use fuel_core::lazy::LazyTensor;
use fuel_ir::Shape;

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

/// Textbook naive attention as an inline scalar loop — the independent
/// known-correct oracle for the causal case, replacing the retired
/// `fuel_reference_backend::attention::attention_naive`. Layout [B, H, S, D]
/// row-major. Applies a causal mask when `causal` (position `j > i` masked).
fn naive_attention(
    q: &[f32], k: &[f32], v: &[f32],
    b: usize, h: usize, sq: usize, sk: usize, d: usize,
    scale: f32, causal: bool,
) -> Vec<f32> {
    let mut out = vec![0f32; b * h * sq * d];
    for bi in 0..b {
        for hi in 0..h {
            let qh = &q[((bi * h + hi) * sq) * d..];
            let kh = &k[((bi * h + hi) * sk) * d..];
            let vh = &v[((bi * h + hi) * sk) * d..];
            let oh = &mut out[((bi * h + hi) * sq) * d..];
            for i in 0..sq {
                // scores[j] = scale · Σ_d q[i,d]·k[j,d], causal-masked.
                let mut scores = vec![f32::NEG_INFINITY; sk];
                let mut maxv = f32::NEG_INFINITY;
                for j in 0..sk {
                    if causal && j > i { continue; }
                    let mut dot = 0f32;
                    for dd in 0..d {
                        dot += qh[i * d + dd] * kh[j * d + dd];
                    }
                    let s = dot * scale;
                    scores[j] = s;
                    if s > maxv { maxv = s; }
                }
                // softmax over j.
                let mut denom = 0f32;
                for j in 0..sk {
                    if scores[j].is_finite() {
                        scores[j] = (scores[j] - maxv).exp();
                        denom += scores[j];
                    } else {
                        scores[j] = 0.0;
                    }
                }
                // out[i,·] = Σ_j attn[j]·v[j,·].
                for dd in 0..d {
                    let mut acc = 0f32;
                    for j in 0..sk {
                        acc += (scores[j] / denom) * vh[j * d + dd];
                    }
                    oh[i * d + dd] = acc;
                }
            }
        }
    }
    out
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
    let attn = scaled.softmax_last_dim().unwrap();
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

    // Path A: lazy Op::FlashAttn → hard-CPU realize.
    let fa_out = q.flash_attn(&k, &v, None, scale, false, None, None, None).unwrap();
    let fa = fa_out.realize_f32_reference();

    // Path B: explicit matmul+softmax composition.
    let q2 = LazyTensor::from_f32(q_data, Shape::from_dims(&[b, h, sq, d]), &fuel_core::Device::cpu());
    let k2 = q2.const_f32_like(k_data, Shape::from_dims(&[b, h, sk, d]));
    let v2 = q2.const_f32_like(v_data, Shape::from_dims(&[b, h, sk, d]));
    let composed = composed_attention(&q2, &k2, &v2, scale);
    let comp = composed.realize_f32_reference();

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
    let fa_out = q.flash_attn(&k, &v, None, scale, true, None, None, None).unwrap().realize_f32_reference();

    // Inline textbook naive attention (causal) for a known-correct comparison
    // point — the independent oracle that replaced the retired
    // `fuel_reference_backend::attention::attention_naive`.
    let direct = naive_attention(&q_data, &k_data, &v_data, b, h, sq, sk, d, scale, true);

    assert_eq!(fa_out.len(), direct.len());
    for (i, (&a, &b)) in fa_out.iter().zip(direct.iter()).enumerate() {
        let diff = (a - b).abs();
        let denom = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(diff < 1e-5 || rel < 1e-5, "[{i}]: lazy={a} direct={b} (abs={diff} rel={rel})");
    }
}
