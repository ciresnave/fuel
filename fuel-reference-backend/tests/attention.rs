//! Phase 8 Tier 1 parity gate: tiled FlashAttention forward output
//! matches the naive math-definition reference within f32 precision,
//! across causal/window/alibi/softcap features and GQA variants.
//!
//! Backward parity is checked separately by finite-difference
//! gradcheck.

use fuel_core_types::Shape;
use fuel_reference_backend::attention::{
    attention_flash, attention_flash_backward, attention_naive, AttentionParams,
};
use fuel_reference_backend::RefTensor;

fn rand_f32(shape: &[usize], seed: u32) -> RefTensor<f32> {
    let n: usize = shape.iter().product();
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        let r = ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.5;
        v.push(r);
    }
    RefTensor::from_vec(v, Shape::from_dims(shape))
}

fn assert_close(a: &RefTensor<f32>, b: &RefTensor<f32>, atol: f32, rtol: f32, label: &str) {
    let av = a.as_slice();
    let bv = b.as_slice();
    assert_eq!(av.len(), bv.len(), "{label}: length mismatch {} vs {}", av.len(), bv.len());
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_idx = 0;
    for (i, (&x, &y)) in av.iter().zip(bv.iter()).enumerate() {
        let diff = (x - y).abs();
        let denom = x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        if diff > max_abs { max_abs = diff; max_rel = rel; max_idx = i; }
    }
    eprintln!("{label}: max abs={max_abs} rel={max_rel} at idx {max_idx}");
    for (i, (&x, &y)) in av.iter().zip(bv.iter()).enumerate() {
        let diff = (x - y).abs();
        let denom = x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(
            diff < atol || rel < rtol,
            "{label}[{i}]: naive={x} flash={y} (abs={diff}, rel={rel})",
        );
    }
}

#[test]
fn flash_matches_naive_basic_no_mask() {
    let (b, h, sq, sk, d) = (2, 4, 32, 32, 16);
    let q = rand_f32(&[b, h, sq, d], 1);
    let k = rand_f32(&[b, h, sk, d], 2);
    let v = rand_f32(&[b, h, sk, d], 3);
    let p = AttentionParams { softmax_scale: (d as f32).sqrt().recip(), ..Default::default() };
    let naive = attention_naive(&q, &k, &v, None, &p);
    let flash = attention_flash(&q, &k, &v, None, &p);
    assert_close(&naive, &flash, 1e-5, 1e-5, "naive vs flash basic");
}

#[test]
fn flash_matches_naive_causal() {
    let (b, h, sq, sk, d) = (1, 2, 24, 24, 8);
    let q = rand_f32(&[b, h, sq, d], 4);
    let k = rand_f32(&[b, h, sk, d], 5);
    let v = rand_f32(&[b, h, sk, d], 6);
    let p = AttentionParams {
        softmax_scale: (d as f32).sqrt().recip(),
        causal: true,
        ..Default::default()
    };
    let naive = attention_naive(&q, &k, &v, None, &p);
    let flash = attention_flash(&q, &k, &v, None, &p);
    assert_close(&naive, &flash, 1e-5, 1e-5, "naive vs flash causal");
}

#[test]
fn flash_matches_naive_sliding_window() {
    let (b, h, sq, sk, d) = (1, 2, 24, 24, 8);
    let q = rand_f32(&[b, h, sq, d], 7);
    let k = rand_f32(&[b, h, sk, d], 8);
    let v = rand_f32(&[b, h, sk, d], 9);
    let p = AttentionParams {
        softmax_scale: (d as f32).sqrt().recip(),
        causal: true,
        window_size_left: Some(4),
        ..Default::default()
    };
    let naive = attention_naive(&q, &k, &v, None, &p);
    let flash = attention_flash(&q, &k, &v, None, &p);
    assert_close(&naive, &flash, 1e-5, 1e-5, "naive vs flash sliding window");
}

#[test]
fn flash_matches_naive_alibi() {
    let (b, h, sq, sk, d) = (1, 4, 16, 16, 8);
    let q = rand_f32(&[b, h, sq, d], 10);
    let k = rand_f32(&[b, h, sk, d], 11);
    let v = rand_f32(&[b, h, sk, d], 12);
    // Per-head ALiBi slopes — typical pattern uses 1/2^(8·h/H) but
    // any deterministic per-head value works for parity.
    let slopes = RefTensor::from_vec(
        vec![0.5_f32, 0.25, 0.125, 0.0625],
        Shape::from_dims(&[h]),
    );
    let p = AttentionParams {
        softmax_scale: (d as f32).sqrt().recip(),
        causal: true,
        ..Default::default()
    };
    let naive = attention_naive(&q, &k, &v, Some(&slopes), &p);
    let flash = attention_flash(&q, &k, &v, Some(&slopes), &p);
    assert_close(&naive, &flash, 1e-5, 1e-5, "naive vs flash alibi");
}

#[test]
fn flash_matches_naive_softcap() {
    let (b, h, sq, sk, d) = (1, 2, 16, 16, 8);
    let q = rand_f32(&[b, h, sq, d], 13);
    let k = rand_f32(&[b, h, sk, d], 14);
    let v = rand_f32(&[b, h, sk, d], 15);
    let p = AttentionParams {
        softmax_scale: (d as f32).sqrt().recip(),
        softcap: Some(30.0), // Gemma-style logit cap
        ..Default::default()
    };
    let naive = attention_naive(&q, &k, &v, None, &p);
    let flash = attention_flash(&q, &k, &v, None, &p);
    assert_close(&naive, &flash, 1e-5, 1e-5, "naive vs flash softcap");
}

#[test]
fn flash_matches_naive_gqa() {
    // Hq=8, Hk=2 -> groups=4 (LLaMA-style GQA)
    let (b, hq, hk, sq, sk, d) = (1usize, 8, 2, 16, 16, 8);
    let q = rand_f32(&[b, hq, sq, d], 16);
    let k = rand_f32(&[b, hk, sk, d], 17);
    let v = rand_f32(&[b, hk, sk, d], 18);
    let p = AttentionParams {
        softmax_scale: (d as f32).sqrt().recip(),
        causal: true,
        ..Default::default()
    };
    let naive = attention_naive(&q, &k, &v, None, &p);
    let flash = attention_flash(&q, &k, &v, None, &p);
    assert_close(&naive, &flash, 1e-5, 1e-5, "naive vs flash GQA");
}

#[test]
fn flash_handles_seqlen_not_multiple_of_block() {
    // BR=BC=16 internally; pick seqlens that don't align.
    let (b, h, sq, sk, d) = (1, 2, 23, 19, 8);
    let q = rand_f32(&[b, h, sq, d], 19);
    let k = rand_f32(&[b, h, sk, d], 20);
    let v = rand_f32(&[b, h, sk, d], 21);
    let p = AttentionParams {
        softmax_scale: (d as f32).sqrt().recip(),
        causal: true,
        ..Default::default()
    };
    let naive = attention_naive(&q, &k, &v, None, &p);
    let flash = attention_flash(&q, &k, &v, None, &p);
    assert_close(&naive, &flash, 1e-5, 1e-5, "naive vs flash uneven seqlens");
}

/// Backward gradcheck: dQ, dK, dV from the analytical recompute
/// match finite-difference numerical gradients of `attention_naive`.
#[test]
fn flash_backward_matches_finite_difference() {
    let (b, h, sq, sk, d) = (1, 2, 6, 6, 4);
    let q = rand_f32(&[b, h, sq, d], 100);
    let k = rand_f32(&[b, h, sk, d], 101);
    let v = rand_f32(&[b, h, sk, d], 102);
    let do_grad = rand_f32(&[b, h, sq, d], 103);
    let p = AttentionParams {
        softmax_scale: (d as f32).sqrt().recip(),
        causal: true,
        ..Default::default()
    };
    // Loss = sum(out * dO). df/dx = dout/dx * dO -> equivalent gradient.
    let scalar_loss = |q_in: &RefTensor<f32>, k_in: &RefTensor<f32>, v_in: &RefTensor<f32>| -> f32 {
        let out = attention_naive(q_in, k_in, v_in, None, &p);
        out.as_slice()
            .iter()
            .zip(do_grad.as_slice().iter())
            .map(|(&o, &g)| o * g)
            .sum::<f32>()
    };
    let (dq, dk, dv) = attention_flash_backward(&q, &k, &v, &do_grad, None, &p);

    let h_step = 1e-3_f32;
    let numeric = |t: &RefTensor<f32>, idx: usize, kind: char| -> f32 {
        let mut bumped = t.as_slice().to_vec();
        bumped[idx] += h_step;
        let bumped_t = RefTensor::from_vec(bumped, t.shape().clone());
        let f_plus = match kind {
            'q' => scalar_loss(&bumped_t, &k, &v),
            'k' => scalar_loss(&q, &bumped_t, &v),
            'v' => scalar_loss(&q, &k, &bumped_t),
            _ => unreachable!(),
        };
        let mut bumped = t.as_slice().to_vec();
        bumped[idx] -= h_step;
        let bumped_t = RefTensor::from_vec(bumped, t.shape().clone());
        let f_minus = match kind {
            'q' => scalar_loss(&bumped_t, &k, &v),
            'k' => scalar_loss(&q, &bumped_t, &v),
            'v' => scalar_loss(&q, &k, &bumped_t),
            _ => unreachable!(),
        };
        (f_plus - f_minus) / (2.0 * h_step)
    };

    // Spot-check a sampling of indices (full check would be 3·B·H·S·D evals,
    // each requiring a forward pass — quadratic-ish). 8 random indices per
    // tensor catches sign/axis bugs reliably.
    let q_len = q.as_slice().len();
    let k_len = k.as_slice().len();
    let v_len = v.as_slice().len();
    for idx in [0, 1, q_len / 2, q_len - 1, 7, 13, 17, 23] {
        if idx >= q_len { continue; }
        let analytic = dq.as_slice()[idx];
        let num = numeric(&q, idx, 'q');
        let diff = (analytic - num).abs();
        let denom = analytic.abs().max(num.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(
            diff < 5e-3 || rel < 1e-2,
            "dQ[{idx}]: analytic={analytic} numeric={num} (abs={diff}, rel={rel})",
        );
    }
    for idx in [0, 1, k_len / 2, k_len - 1, 7, 13, 17, 23] {
        if idx >= k_len { continue; }
        let analytic = dk.as_slice()[idx];
        let num = numeric(&k, idx, 'k');
        let diff = (analytic - num).abs();
        let denom = analytic.abs().max(num.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(
            diff < 5e-3 || rel < 1e-2,
            "dK[{idx}]: analytic={analytic} numeric={num} (abs={diff}, rel={rel})",
        );
    }
    for idx in [0, 1, v_len / 2, v_len - 1, 7, 13, 17, 23] {
        if idx >= v_len { continue; }
        let analytic = dv.as_slice()[idx];
        let num = numeric(&v, idx, 'v');
        let diff = (analytic - num).abs();
        let denom = analytic.abs().max(num.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(
            diff < 5e-3 || rel < 1e-2,
            "dV[{idx}]: analytic={analytic} numeric={num} (abs={diff}, rel={rel})",
        );
    }
}
