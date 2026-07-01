//! Phase 6d Track 1 parity gate: paged-cache attention reference
//! produces the same output as plain `attention_naive` when the
//! block table is the trivial identity mapping.
//!
//! Specifically: pack a contiguous `[B, Hkv, Sk, D]` cache into the
//! paged layout `[num_blocks, block_size, Hkv, D]` with one block per
//! contiguous KV chunk and a per-batch block table that hands out
//! consecutive blocks. Result must match `attention_naive` with causal
//! masking enabled (paged attention's masking is implicit via
//! `context_lens`).

use fuel_ir::Shape;
use fuel_reference_backend::attention::{
    attention_naive, attention_paged_naive, AttentionParams,
};
use fuel_reference_backend::RefTensor;

fn rand_f32(shape: &[usize], seed: u32) -> RefTensor<f32> {
    let n: usize = shape.iter().product();
    let mut s = seed;
    let mut v = Vec::with_capacity(n);
    for _ in 0..n {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        v.push(((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.5);
    }
    RefTensor::from_vec(v, Shape::from_dims(shape))
}

/// Pack a `[B, Hkv, Sk, D]` contiguous cache into paged layout
/// `[num_blocks, block_size, Hkv, D]` plus a `[B, max_blocks]` table
/// mapping logical → physical blocks.
fn pack_paged(
    contig: &RefTensor<f32>,
    block_size: usize,
) -> (RefTensor<f32>, RefTensor<u32>, usize) {
    let dims = contig.shape().dims();
    let (b, hkv, sk, d) = (dims[0], dims[1], dims[2], dims[3]);
    let blocks_per_seq = (sk + block_size - 1) / block_size;
    let total_blocks = b * blocks_per_seq;
    let mut paged = vec![0.0_f32; total_blocks * block_size * hkv * d];
    let mut block_table = vec![0u32; b * blocks_per_seq];
    let cs = contig.as_slice();
    for bi in 0..b {
        for blk in 0..blocks_per_seq {
            let phys = bi * blocks_per_seq + blk;
            block_table[bi * blocks_per_seq + blk] = phys as u32;
            for slot in 0..block_size {
                let k_pos = blk * block_size + slot;
                if k_pos >= sk { continue; }
                for h in 0..hkv {
                    for di in 0..d {
                        let src = ((bi * hkv + h) * sk + k_pos) * d + di;
                        let dst = ((phys * block_size + slot) * hkv + h) * d + di;
                        paged[dst] = cs[src];
                    }
                }
            }
        }
    }
    let paged_t = RefTensor::from_vec(
        paged,
        Shape::from_dims(&[total_blocks, block_size, hkv, d]),
    );
    let bt_t = RefTensor::from_vec(
        block_table,
        Shape::from_dims(&[b, blocks_per_seq]),
    );
    (paged_t, bt_t, blocks_per_seq)
}

fn assert_close(a: &[f32], b: &[f32], atol: f32, rtol: f32, label: &str) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch");
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    let mut max_idx = 0;
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let denom = x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        if diff > max_abs { max_abs = diff; max_rel = rel; max_idx = i; }
    }
    eprintln!("{label}: max abs={max_abs} rel={max_rel} at idx {max_idx}");
    for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        let denom = x.abs().max(y.abs()).max(f32::MIN_POSITIVE);
        let rel = diff / denom;
        assert!(diff < atol || rel < rtol,
            "{label}[{i}]: paged={x} naive={y} (abs={diff}, rel={rel})");
    }
}

#[test]
fn paged_matches_naive_decode_full_context() {
    // Decode case: Sq=1, Sk=Lb=20 contiguous. Block size 8 -> 3 blocks per seq.
    // All sequences have the same length (= Sk), so the result must match
    // `attention_naive` with causal=true (last query attends to all positions).
    let (b, h, sq, sk, d) = (2usize, 4, 1, 20, 8);
    let block_size = 8;
    let q  = rand_f32(&[b, h, sq, d], 1);
    let k  = rand_f32(&[b, h, sk, d], 2);
    let v  = rand_f32(&[b, h, sk, d], 3);
    let scale = (d as f32).sqrt().recip();

    // Decode (Sq=1, q is the *new* token, sees all cached K). Paged
    // expresses this as `abs_pos = ctx_len - Sq + 0 = ctx_len - 1`,
    // admitting `k_pos <= ctx_len-1`, i.e. the whole cache. The
    // equivalent in `attention_naive` is causal=false (the diagonal
    // causal mask only matches paged when Sq spans the full
    // sequence; for Sq=1 they diverge unless mask is off).
    let p_naive = AttentionParams {
        softmax_scale: scale,
        causal: false,
        ..Default::default()
    };
    let naive = attention_naive(&q, &k, &v, None, &p_naive).into_vec();

    let (kc, bt, _) = pack_paged(&k, block_size);
    let (vc, _, _) = pack_paged(&v, block_size);
    let context_lens = RefTensor::from_vec(vec![sk as u32; b], Shape::from_dims(&[b]));
    let paged = attention_paged_naive(&q, &kc, &vc, &bt, &context_lens, None, scale, block_size, None).into_vec();

    assert_close(&paged, &naive, 1e-5, 1e-5, "paged vs naive decode");
}

#[test]
fn paged_handles_variable_context_lens() {
    // Different sequences in the batch have different lengths.
    // Pack the longest, then per-batch context_lens drives masking.
    let (b, h, sq, max_sk, d) = (3usize, 2, 1, 16, 8);
    let block_size = 4;
    let lens = [16usize, 9, 4]; // sequence lengths

    let q = rand_f32(&[b, h, sq, d], 4);
    // Build a [B, H, max_sk, D] cache with deterministic content.
    let k = rand_f32(&[b, h, max_sk, d], 5);
    let v = rand_f32(&[b, h, max_sk, d], 6);
    let scale = (d as f32).sqrt().recip();

    let (kc, bt, _) = pack_paged(&k, block_size);
    let (vc, _, _) = pack_paged(&v, block_size);
    let context_lens = RefTensor::from_vec(
        lens.iter().map(|&l| l as u32).collect::<Vec<_>>(),
        Shape::from_dims(&[b]),
    );
    let paged = attention_paged_naive(&q, &kc, &vc, &bt, &context_lens, None, scale, block_size, None).into_vec();

    // Per-sequence reference: slice K/V down to its true length, run naive
    // attention, splice the row into the expected output.
    let q_data = q.as_slice();
    let k_data = k.as_slice();
    let v_data = v.as_slice();
    let mut expected = vec![0.0_f32; b * h * sq * d];
    for bi in 0..b {
        let l = lens[bi];
        let q_b = RefTensor::from_vec(
            q_data[bi*h*sq*d .. (bi+1)*h*sq*d].to_vec(),
            Shape::from_dims(&[1, h, sq, d]),
        );
        let mut k_slice = vec![0.0_f32; h * l * d];
        let mut v_slice = vec![0.0_f32; h * l * d];
        for hi in 0..h {
            for kp in 0..l {
                for di in 0..d {
                    k_slice[(hi * l + kp) * d + di] = k_data[((bi * h + hi) * max_sk + kp) * d + di];
                    v_slice[(hi * l + kp) * d + di] = v_data[((bi * h + hi) * max_sk + kp) * d + di];
                }
            }
        }
        let k_b = RefTensor::from_vec(k_slice, Shape::from_dims(&[1, h, l, d]));
        let v_b = RefTensor::from_vec(v_slice, Shape::from_dims(&[1, h, l, d]));
        // Same Sq=1 decode logic: q sees all of K up to ctx_len; equivalent
        // to attention_naive without causal mask on K sliced to ctx_len.
        let p = AttentionParams { softmax_scale: scale, causal: false, ..Default::default() };
        let out_b = attention_naive(&q_b, &k_b, &v_b, None, &p).into_vec();
        expected[bi*h*sq*d .. (bi+1)*h*sq*d].copy_from_slice(&out_b);
    }

    assert_close(&paged, &expected, 1e-5, 1e-5, "paged vs per-sequence naive");
}

#[test]
fn paged_handles_gqa() {
    // Hq=8, Hkv=2 — LLaMA-3-style. Same length per sequence to match
    // attention_naive's contiguous-K interpretation.
    let (b, hq, hkv, sq, sk, d) = (1usize, 8, 2, 1, 16, 8);
    let block_size = 8;
    let q  = rand_f32(&[b, hq, sq, d], 7);
    let k  = rand_f32(&[b, hkv, sk, d], 8);
    let v  = rand_f32(&[b, hkv, sk, d], 9);
    let scale = (d as f32).sqrt().recip();

    // Decode (Sq=1, q is the *new* token, sees all cached K). Paged
    // expresses this as `abs_pos = ctx_len - Sq + 0 = ctx_len - 1`,
    // admitting `k_pos <= ctx_len-1`, i.e. the whole cache. The
    // equivalent in `attention_naive` is causal=false (the diagonal
    // causal mask only matches paged when Sq spans the full
    // sequence; for Sq=1 they diverge unless mask is off).
    let p_naive = AttentionParams {
        softmax_scale: scale,
        causal: false,
        ..Default::default()
    };
    let naive = attention_naive(&q, &k, &v, None, &p_naive).into_vec();

    let (kc, bt, _) = pack_paged(&k, block_size);
    let (vc, _, _) = pack_paged(&v, block_size);
    let context_lens = RefTensor::from_vec(vec![sk as u32; b], Shape::from_dims(&[b]));
    let paged = attention_paged_naive(&q, &kc, &vc, &bt, &context_lens, None, scale, block_size, None).into_vec();

    assert_close(&paged, &naive, 1e-5, 1e-5, "paged vs naive GQA");
}
