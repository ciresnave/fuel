//! Paged-decode parity (multi-session serving, paged-storage integration PS2):
//! `LlamaModel::forward_paged_step` — sessions' KV physically in `DeviceKvPool`
//! blocks, decoded via `Op::PagedAttn` — produces logits ε-close to the
//! contiguous `forward_with_kv_context` for the same token sequence.
//!
//! `Op::PagedAttn` is decode-only (Sq=1), so the paged path feeds EVERY token
//! (prompt + generated) one at a time; the contiguous path does a batched
//! prefill then decodes. They agree position-for-position because each token's
//! K/V depends only on tokens `0..=i` (causal), which are all resident by the
//! time token `i` is fed — the standard one-at-a-time ≡ batched-prefill identity.

use std::sync::Arc;

use fuel_core::Device;
use fuel_core::inference_context::{InferenceContext, KvCache};
use fuel_core::kv_block_pool::KvGeometry;
use fuel_core::kv_block_pool_device::{BlockKind, DeviceKvPool};
use fuel_core::lazy::{LayerWeights, LlamaConfig, LlamaModel, LlamaWeights};
use fuel_ir::DType;

/// Deterministic tiny weights sized to `cfg` (mirrors lazy.rs's test builder).
fn tiny_weights(cfg: &LlamaConfig, seed: u32) -> LlamaWeights {
    let mut s: u32 = seed;
    let mut next = || -> f32 {
        s = s.wrapping_mul(1103515245).wrapping_add(12345);
        ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
    };
    let mut vec_of = |n: usize| -> Arc<[f32]> {
        let v: Vec<f32> = (0..n).map(|_| next()).collect();
        Arc::from(v)
    };
    let kv_dim = cfg.n_kv_heads * cfg.head_dim;
    LlamaWeights {
        token_embedding: vec_of(cfg.vocab_size * cfg.dim),
        layers: (0..cfg.n_layers)
            .map(|_| LayerWeights {
                attn_q: vec_of(cfg.dim * cfg.dim).into(),
                attn_q_bias: None,
                attn_k: vec_of(cfg.dim * kv_dim).into(),
                attn_k_bias: None,
                attn_v: vec_of(cfg.dim * kv_dim).into(),
                attn_v_bias: None,
                attn_o: vec_of(cfg.dim * cfg.dim).into(),
                ffn_gate: vec_of(cfg.dim * cfg.ffn_dim).into(),
                ffn_up: vec_of(cfg.dim * cfg.ffn_dim).into(),
                ffn_down: vec_of(cfg.ffn_dim * cfg.dim).into(),
                attn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
                ffn_norm_gain: Arc::from(vec![1.0; cfg.dim]),
            })
            .collect(),
        final_norm_gain: Arc::from(vec![1.0; cfg.dim]),
        output: vec_of(cfg.dim * cfg.vocab_size).into(),
    }
}

fn assert_close(paged: &[f32], contig: &[f32], label: &str) {
    assert_eq!(paged.len(), contig.len(), "{label}: logits length");
    for (i, (&a, &b)) in paged.iter().zip(contig.iter()).enumerate() {
        let diff = (a - b).abs();
        let den = a.abs().max(b.abs()).max(f32::MIN_POSITIVE);
        assert!(
            diff < 1e-4 || diff / den < 1e-4,
            "{label}[{i}]: paged={a} contig={b} (abs={diff})",
        );
    }
}

fn parity_for(cfg: LlamaConfig) {
    let model = LlamaModel { config: cfg.clone(), weights: tiny_weights(&cfg, 9999) };
    let prompt = [1u32, 2, 3];
    let decode = [4u32, 5, 6];
    let all: Vec<u32> = prompt.iter().chain(decode.iter()).copied().collect();
    let max_seq_len = all.len();
    let dev = Device::cpu();

    // --- Contiguous reference: batched prefill, then decode one token/step. ---
    let mut cache = KvCache::with_capacity(
        cfg.n_layers, cfg.n_kv_heads, cfg.head_dim, max_seq_len, DType::F32, &dev,
    ).unwrap();
    let mut ctx = InferenceContext::new(dev.clone());
    let mut contig: Vec<Vec<f32>> = Vec::new();
    contig.push(model.forward_with_kv_context(&prompt, &mut cache, &mut ctx).unwrap());
    for &t in &decode {
        contig.push(model.forward_with_kv_context(&[t], &mut cache, &mut ctx).unwrap());
    }

    // --- Paged: feed EVERY token one at a time into the pool. ---
    let geom = KvGeometry {
        n_layers: cfg.n_layers,
        num_blocks: 32,
        block_size: 4,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        elem_size: 4,
    };
    let mut pool = DeviceKvPool::new(geom, DType::F32, &dev).unwrap();
    let session = pool.core_mut().open();
    let mut paged: Vec<Vec<f32>> = Vec::new();
    for &t in &all {
        paged.push(model.forward_paged_step(t, &mut pool, session).unwrap());
    }

    // Contiguous prefill returns position P-1 (predict token after the prompt),
    // = paged's P-th step; then each decode position lines up.
    let p = prompt.len();
    assert_close(&paged[p - 1], &contig[0], "prefill-last");
    for i in 0..decode.len() {
        assert_close(&paged[p + i], &contig[1 + i], &format!("decode-{i}"));
    }
}

#[test]
fn paged_forward_matches_contiguous_no_gqa() {
    parity_for(LlamaConfig {
        vocab_size: 16,
        dim: 16,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 4, // no GQA
        head_dim: 4,
        ffn_dim: 16,
        norm_eps: 1e-5,
        rope_base: 10000.0,
    });
}

#[test]
fn paged_forward_matches_contiguous_gqa() {
    parity_for(LlamaConfig {
        vocab_size: 16,
        dim: 16,
        n_layers: 2,
        n_heads: 4,
        n_kv_heads: 2, // GQA: n_rep = 2
        head_dim: 4,
        ffn_dim: 16,
        norm_eps: 1e-5,
        rope_base: 10000.0,
    });
}

/// PS4a: batched paged decode (`B = K`) matches the serial single-session path
/// per row. Each row of a batched step equals that session's standalone decode
/// — proving `Op::PagedAttn` at B=K routes each session to only its own blocks
/// (no cross-contamination) and the batched result is the B=1 result.
fn batched_matches_serial_for(cfg: LlamaConfig) {
    let model = LlamaModel { config: cfg.clone(), weights: tiny_weights(&cfg, 9999) };
    let dev = Device::cpu();
    let geom = || KvGeometry {
        n_layers: cfg.n_layers,
        num_blocks: 32,
        block_size: 4,
        n_kv_heads: cfg.n_kv_heads,
        head_dim: cfg.head_dim,
        elem_size: 4,
    };
    // Equal-length prompts → uniform position after prefill (the batching gate).
    let prompt_a = [1u32, 2, 3];
    let prompt_b = [4u32, 5, 6];
    let (decode_a, decode_b) = (7u32, 8u32);

    // Serial: each session alone (own pool), prefill token-by-token + one decode.
    let serial = |prompt: &[u32], tok: u32| -> Vec<f32> {
        let mut pool = DeviceKvPool::new(geom(), DType::F32, &dev).unwrap();
        let s = pool.core_mut().open();
        for &t in prompt {
            model.forward_paged_step(t, &mut pool, s).unwrap();
        }
        model.forward_paged_step(tok, &mut pool, s).unwrap()
    };
    let a_serial = serial(&prompt_a, decode_a);
    let b_serial = serial(&prompt_b, decode_b);

    // Batched: A + B on ONE shared pool, prefill each, then one batched step.
    let mut pool = DeviceKvPool::new(geom(), DType::F32, &dev).unwrap();
    let sa = pool.core_mut().open();
    let sb = pool.core_mut().open();
    for &t in &prompt_a {
        model.forward_paged_step(t, &mut pool, sa).unwrap();
    }
    for &t in &prompt_b {
        model.forward_paged_step(t, &mut pool, sb).unwrap();
    }
    let batched = model
        .forward_paged_step_batched(&[decode_a, decode_b], &mut pool, &[sa, sb])
        .unwrap();
    assert_eq!(batched.len(), 2);
    assert_close(&batched[0], &a_serial, "batched row 0 (A) == serial A");
    assert_close(&batched[1], &b_serial, "batched row 1 (B) == serial B");
}

#[test]
fn paged_batched_matches_serial_no_gqa() {
    batched_matches_serial_for(LlamaConfig {
        vocab_size: 16, dim: 16, n_layers: 2, n_heads: 4, n_kv_heads: 4,
        head_dim: 4, ffn_dim: 16, norm_eps: 1e-5, rope_base: 10000.0,
    });
}

#[test]
fn paged_batched_matches_serial_gqa() {
    batched_matches_serial_for(LlamaConfig {
        vocab_size: 16, dim: 16, n_layers: 2, n_heads: 4, n_kv_heads: 2,
        head_dim: 4, ffn_dim: 16, norm_eps: 1e-5, rope_base: 10000.0,
    });
}

fn tiny_cfg() -> LlamaConfig {
    LlamaConfig {
        vocab_size: 16, dim: 16, n_layers: 2, n_heads: 4, n_kv_heads: 4,
        head_dim: 4, ffn_dim: 16, norm_eps: 1e-5, rope_base: 10000.0,
    }
}

/// REGRESSION (adversarial-verification finding): decoding a session whose last
/// block is spliced/shared must copy-on-write, NOT write through the shared
/// physical block. A donates its prefix to B; A decodes into the (partial)
/// shared block, B decodes into it, then A decodes again — A's logits must equal
/// an A that never shared with B. Without the CoW guard, B's write clobbers A's
/// shared slot and A silently reads B's KV.
#[test]
fn paged_decode_into_spliced_prefix_does_not_corrupt_donor() {
    let cfg = tiny_cfg();
    let model = LlamaModel { config: cfg.clone(), weights: tiny_weights(&cfg, 9999) };
    let dev = Device::cpu();
    let geom = || KvGeometry {
        n_layers: cfg.n_layers, num_blocks: 32, block_size: 4,
        n_kv_heads: cfg.n_kv_heads, head_dim: cfg.head_dim, elem_size: 4,
    };
    let prompt = [1u32, 2, 3, 4, 5, 6]; // 6 tokens → block0 full, block1 partial (2/4)
    let (a_tok, b_tok) = (7u32, 8u32);

    // Reference: A alone — prefill, decode a_tok (pos 6), decode a_tok (pos 7).
    let a_ref = {
        let mut pool = DeviceKvPool::new(geom(), DType::F32, &dev).unwrap();
        let a = pool.core_mut().open();
        for &t in &prompt { model.forward_paged_step(t, &mut pool, a).unwrap(); }
        model.forward_paged_step(a_tok, &mut pool, a).unwrap(); // pos 6
        model.forward_paged_step(a_tok, &mut pool, a).unwrap() // pos 7 (compared)
    };

    // With B sharing A's prefix, interleaved so B writes A's shared slot AFTER A.
    let mut pool = DeviceKvPool::new(geom(), DType::F32, &dev).unwrap();
    let a = pool.core_mut().open();
    for &t in &prompt { model.forward_paged_step(t, &mut pool, a).unwrap(); }
    let b = pool.core_mut().open();
    pool.core_mut().splice(a, b, 0, 2).unwrap(); // B shares A's 2 blocks (block1 partial + shared)
    model.forward_paged_step(a_tok, &mut pool, a).unwrap(); // A writes pos 6 into shared block1 slot 2
    model.forward_paged_step(b_tok, &mut pool, b).unwrap(); // B decodes pos 6 → CoW (must NOT clobber A)
    let a_with_b = model.forward_paged_step(a_tok, &mut pool, a).unwrap(); // A pos 7 reads its own pos-6 KV

    assert_eq!(a_with_b, a_ref, "A's decode unaffected by B decoding into the shared prefix (CoW held)");
}

/// REGRESSION (adversarial-verification finding): a batched step that can't fit
/// the whole batch is ATOMIC — it errors before advancing ANY session, so the
/// batch stays uniform + retryable instead of wedging.
#[test]
fn batched_step_out_of_blocks_is_atomic() {
    let cfg = tiny_cfg();
    let model = LlamaModel { config: cfg.clone(), weights: tiny_weights(&cfg, 9999) };
    let dev = Device::cpu();
    // num_blocks 3, block_size 4: two sessions prefilled to 4 tokens use 2 blocks
    // (1 each); a boundary batched step needs 2 more but only 1 is free.
    let geom = KvGeometry {
        n_layers: cfg.n_layers, num_blocks: 3, block_size: 4,
        n_kv_heads: cfg.n_kv_heads, head_dim: cfg.head_dim, elem_size: 4,
    };
    let mut pool = DeviceKvPool::new(geom, DType::F32, &dev).unwrap();
    let sa = pool.core_mut().open();
    let sb = pool.core_mut().open();
    for &t in &[1u32, 2, 3, 4] { model.forward_paged_step(t, &mut pool, sa).unwrap(); }
    for &t in &[5u32, 6, 7, 8] { model.forward_paged_step(t, &mut pool, sb).unwrap(); }
    assert_eq!(pool.core().free_blocks(), 1, "2 of 3 blocks used, 1 free");
    let filled_before = (pool.core().filled_tokens(sa), pool.core().filled_tokens(sb));

    // Boundary batched step (pos 4, slot 0) needs 2 blocks, 1 free → Err, atomic.
    let r = model.forward_paged_step_batched(&[9, 10], &mut pool, &[sa, sb]);
    assert!(r.is_err(), "batch at a boundary needs 2 blocks, only 1 free");
    assert_eq!(
        (pool.core().filled_tokens(sa), pool.core().filled_tokens(sb)), filled_before,
        "no session partially advanced (atomic)",
    );
    assert_eq!(pool.core().free_blocks(), 1, "no block consumed on the rejected batch");
}

/// REGRESSION for the adversarial-verification "highest-risk gap": batched decode
/// with sessions spanning >1 block. The earlier batched parity uses 1-block
/// sessions, where the per-row block gather degenerates to a no-op; here each
/// session holds 2 blocks, exercising the flattened `[K, max_blk]` block_table
/// routing at B=K. Each batched row must still equal its standalone serial decode
/// (each row attends to ONLY its own two blocks).
#[test]
fn paged_batched_multiblock_matches_serial() {
    let cfg = tiny_cfg();
    let model = LlamaModel { config: cfg.clone(), weights: tiny_weights(&cfg, 9999) };
    let dev = Device::cpu();
    let geom = || KvGeometry {
        n_layers: cfg.n_layers, num_blocks: 32, block_size: 4,
        n_kv_heads: cfg.n_kv_heads, head_dim: cfg.head_dim, elem_size: 4,
    };
    // 5-token prompts → 2 blocks each (block_size 4); equal length → uniform.
    let prompt_a = [1u32, 2, 3, 4, 5];
    let prompt_b = [6u32, 7, 8, 9, 10];
    let (da, db) = (11u32, 12u32);

    let serial = |prompt: &[u32], tok: u32| -> Vec<f32> {
        let mut pool = DeviceKvPool::new(geom(), DType::F32, &dev).unwrap();
        let s = pool.core_mut().open();
        for &t in prompt {
            model.forward_paged_step(t, &mut pool, s).unwrap();
        }
        model.forward_paged_step(tok, &mut pool, s).unwrap()
    };
    let a_serial = serial(&prompt_a, da);
    let b_serial = serial(&prompt_b, db);

    let mut pool = DeviceKvPool::new(geom(), DType::F32, &dev).unwrap();
    let sa = pool.core_mut().open();
    let sb = pool.core_mut().open();
    for &t in &prompt_a {
        model.forward_paged_step(t, &mut pool, sa).unwrap();
    }
    for &t in &prompt_b {
        model.forward_paged_step(t, &mut pool, sb).unwrap();
    }
    assert_eq!(pool.core().session_blocks(sa), Some(2), "each session spans 2 blocks");
    let batched = model.forward_paged_step_batched(&[da, db], &mut pool, &[sa, sb]).unwrap();
    assert_close(&batched[0], &a_serial, "multi-block batched row A == serial A");
    assert_close(&batched[1], &b_serial, "multi-block batched row B == serial B");
}

/// SERVING-VALUE CORRECTNESS ANCHOR (prefix-sharing Task 1): a session that
/// REUSES a shared KV prefix through the registry (`register_prefix` +
/// `splice_prefix_from`) and prefills ONLY the suffix decodes token-for-token
/// IDENTICALLY to a from-scratch session that computed the WHOLE prompt.
///
/// This goes THROUGH the product API (not hand-rolled `splice`), so it is the
/// real end-to-end proof of the rung-1 serving claim. Two things are asserted:
///   1. **Decode parity:** the shared session's logits for `[suffix .. decode]`
///      equal the from-scratch session's logits at the SAME absolute positions.
///      This works because `forward_paged_step` derives the RoPE position from
///      `filled_tokens` (bumped to `shared_tokens` by the splice), so the sharer
///      computes the suffix at positions `N*block_size ..` — exactly where the
///      from-scratch prompt has those same tokens.
///   2. **Owner KV immutability:** the shared prefix's physical block bytes are
///      byte-unchanged after the sharer prefills + decodes. In rung-1 the prefix
///      is WHOLE filled blocks, so the sharer's first suffix write lands on a
///      FRESH block (`tok_pos/block_size == N`, a new append) — it never writes
///      the shared blocks, and CoW never enters the rung-1 path. If a bug wrote
///      into a shared slot instead, these bytes would change and this catches it.
#[test]
fn prefix_shared_session_decodes_like_from_scratch() {
    let cfg = tiny_cfg();
    let model = LlamaModel { config: cfg.clone(), weights: tiny_weights(&cfg, 9999) };
    let dev = Device::cpu();
    let block_size = 4usize;
    let geom = || KvGeometry {
        n_layers: cfg.n_layers, num_blocks: 32, block_size,
        n_kv_heads: cfg.n_kv_heads, head_dim: cfg.head_dim, elem_size: 4,
    };

    // A shared system prompt filling EXACTLY 2 whole blocks (8 tokens @ bs=4),
    // a per-request suffix (3 tokens), then K=3 decode steps.
    let prefix: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
    let suffix: [u32; 3] = [9, 10, 11];
    let decode: [u32; 3] = [12, 13, 14];
    let n_prefix_blocks = prefix.len() / block_size; // 2 whole blocks
    let full: Vec<u32> = prefix.iter().chain(suffix.iter()).copied().collect();

    // --- From-scratch reference: prefill the FULL prompt + decode, one at a time.
    // scratch[i] = logits after feeding the i-th token (predicts position i+1).
    let scratch: Vec<Vec<f32>> = {
        let mut pool = DeviceKvPool::new(geom(), DType::F32, &dev).unwrap();
        let s = pool.core_mut().open();
        let mut logits = Vec::new();
        for &t in full.iter().chain(decode.iter()) {
            logits.push(model.forward_paged_step(t, &mut pool, s).unwrap());
        }
        logits
    };

    // --- Shared path: donor fills the prefix; register it; sharer splices it. ---
    let mut pool = DeviceKvPool::new(geom(), DType::F32, &dev).unwrap();
    let donor = pool.core_mut().open();
    for &t in &prefix {
        model.forward_paged_step(t, &mut pool, donor).unwrap();
    }
    // Mint a registry prefix from the donor's first N (whole, filled) blocks,
    // then DISCARD the donor — the owner keeps the prefix alive (registry lifetime).
    let pid = pool.core_mut().register_prefix(donor, n_prefix_blocks).unwrap();
    pool.core_mut().discard(donor);

    // Sharer session: zero-copy splice the registered prefix, then prefill ONLY
    // the suffix. The splice returns the shared token count = the suffix offset.
    let sharer = pool.core_mut().open();
    let shared_tokens = pool.core_mut().splice_prefix_from(pid, sharer).unwrap();
    assert_eq!(shared_tokens, prefix.len(), "splice returns the shared prefix token count");
    assert_eq!(
        pool.core().filled_tokens(sharer), Some(prefix.len()),
        "sharer starts at the shared-prefix position (suffix prefill continues from here)",
    );

    // Snapshot the SHARED prefix's physical block bytes (owner-held, refcount 2).
    // The sharer's block table points at the same physical blocks the owner keeps
    // alive, so reading them through the sharer reads the owner's KV.
    let shared_phys: Vec<_> = (0..n_prefix_blocks)
        .map(|i| pool.core().resident_block(sharer, i).expect("shared block resident"))
        .collect();
    let read_shared = |pool: &DeviceKvPool, phys: &[_]| -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for &p in phys {
            for l in 0..cfg.n_layers {
                out.push(pool.read_block(l, BlockKind::K, p).unwrap());
                out.push(pool.read_block(l, BlockKind::V, p).unwrap());
            }
        }
        out
    };
    let shared_kv_before = read_shared(&pool, &shared_phys);

    // Prefill ONLY the suffix (positions N*block_size ..), then decode K tokens.
    let mut shared_logits: Vec<Vec<f32>> = Vec::new();
    for &t in suffix.iter().chain(decode.iter()) {
        shared_logits.push(model.forward_paged_step(t, &mut pool, sharer).unwrap());
    }

    // (2) Owner KV immutability: shared prefix blocks byte-unchanged.
    let shared_kv_after = read_shared(&pool, &shared_phys);
    assert_eq!(
        shared_kv_before, shared_kv_after,
        "the shared prefix owner's KV bytes must be untouched by the sharer's writes",
    );

    // (1) Decode parity: shared_logits[k] is the logit after feeding the k-th
    // suffix/decode token, at absolute position prefix.len()+k. Compare to the
    // from-scratch logit at the SAME position (scratch index prefix.len()+k).
    assert_eq!(shared_logits.len(), suffix.len() + decode.len());
    for (k, sl) in shared_logits.iter().enumerate() {
        assert_close(sl, &scratch[prefix.len() + k], &format!("prefix-shared-step-{k}"));
    }
}
