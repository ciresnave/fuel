//! CSM (Conversational Speech Model) — lazy port.
//!
//! CSM is Sesame's open speech-token model with a two-stage
//! transformer pipeline:
//!
//!   - **backbone** — a Llama-family LM that ingests the
//!     **summed multi-codebook + text** embedding of an interleaved
//!     `(audio_codebook_0..N-1, text_token)` frame and predicts
//!     the next frame's codebook-0 audio token via the
//!     `codebook0_head` linear.
//!
//!   - **decoder** — a smaller Llama-family LM that
//!     auto-regressively predicts the remaining codebooks
//!     `1..N-1` given the backbone hidden state and the previously
//!     predicted codebooks. The codebook-i token is sampled from
//!     `decoder_h · audio_head[i-1]` (a per-codebook output matrix).
//!
//! Two Llama models, each with its own KV cache, are coordinated
//! over the multi-codebook generation loop.
//!
//! Scope of this v1 port — the **embedding pipeline and heads**.
//! The two Llama forwards stay on the consumer side (the loop
//! mutates two separate KV caches across many `forward_embeds`
//! calls per generated audio frame, which is consumer-driven
//! glue). What this module ships:
//!
//!   - `embed_frame(audio_codes, text_tokens, tokens_mask, anchor)`
//!     — applies the per-codebook offset to the audio codes,
//!     looks up the audio and text embedding tables, gates by
//!     `tokens_mask`, and sums across the codebook axis. Output is
//!     a `(1, S, backbone_dim)` tensor ready for
//!     `backbone.forward_embeds(...)`.
//!
//!   - `codebook0_logits(backbone_h, anchor)` — applies the
//!     `codebook0_head` linear to the backbone hidden state.
//!
//!   - `project_to_decoder(curr_h, anchor)` — applies the
//!     `projection` linear (`backbone_dim → decoder_dim`) used to
//!     map backbone hidden to the decoder's hidden space.
//!
//!   - `audio_head_logits(decoder_h, codebook_idx, anchor)` —
//!     applies `audio_head[codebook_idx - 1]` (a per-codebook
//!     `(decoder_dim, audio_vocab_size)` matrix).
//!
//!   - `audio_embed_for_code(code, codebook_idx, anchor)` —
//!     looks up the codebook-i embedding (with codebook-i offset
//!     applied) for one sampled audio code. Used to assemble the
//!     `curr_h` input to each decoder step.
//!
//! v1 scope: F32, batch == 1, forward-only inference.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct CsmConfig {
    pub audio_num_codebooks: usize,
    pub audio_vocab_size: usize,
    pub text_vocab_size: usize,
    pub backbone_dim: usize,
    pub decoder_dim: usize,
}

impl CsmConfig {
    /// Canonical Sesame CSM: 32 audio codebooks of 2051 codes each,
    /// 128256-token Llama-3 text vocab, 2048-d backbone (Llama-1B),
    /// 1024-d decoder (Llama-100M).
    pub fn sesame() -> Self {
        Self {
            audio_num_codebooks: 32,
            audio_vocab_size: 2051,
            text_vocab_size: 128_256,
            backbone_dim: 2048,
            decoder_dim: 1024,
        }
    }
}

// ---- Weight structures ------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CsmWeights {
    /// `(audio_num_codebooks * audio_vocab_size, backbone_dim)`. The
    /// per-codebook tables are concatenated along the vocab axis;
    /// codebook-i lookups add `i * audio_vocab_size` to the code id.
    pub audio_embedding: Arc<[f32]>,
    /// `(text_vocab_size, backbone_dim)`.
    pub text_embedding: Arc<[f32]>,
    /// `(backbone_dim, audio_vocab_size)`.
    pub codebook0_head: WeightStorage,
    /// `(backbone_dim, decoder_dim)` — no bias.
    pub projection: WeightStorage,
    /// `(audio_num_codebooks - 1, decoder_dim, audio_vocab_size)`.
    /// `audio_head[i-1]` is the matmul matrix for codebook `i`
    /// (`i in 1..audio_num_codebooks`).
    pub audio_head: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct CsmModel {
    pub config: CsmConfig,
    pub weights: CsmWeights,
}

// ---- Helpers ---------------------------------------------------------------

impl CsmModel {
    /// Build the backbone embedding for one or more interleaved frames.
    ///
    /// Inputs (all u32):
    ///   - `audio_codes` — flat `seq_len * num_codebooks` audio code ids.
    ///     Code at frame `t` codebook `i` lives at index
    ///     `t * num_codebooks + i`. **No** per-codebook offset is
    ///     pre-applied; the wrapper adds `i * audio_vocab_size`.
    ///   - `text_tokens` — flat `seq_len` text token ids.
    ///   - `tokens_mask` — flat `seq_len * (num_codebooks + 1)` mask
    ///     (1 = active, 0 = inactive). Matches the eager
    ///     `(B, S, num_codebooks + 1)` layout.
    ///
    /// `anchor` is the graph anchor (any tensor on the target graph)
    /// — every constant table is materialized on `anchor`'s graph.
    ///
    /// Returns `(1, seq_len, backbone_dim)`.
    pub fn embed_frame(
        &self,
        audio_codes: &[u32],
        text_tokens: &[u32],
        tokens_mask: &[u8],
        anchor: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let cb = cfg.audio_num_codebooks;
        let seq = text_tokens.len();
        assert_eq!(audio_codes.len(), seq * cb,
            "audio_codes len {} != seq_len {seq} * num_codebooks {cb}",
            audio_codes.len());
        assert_eq!(tokens_mask.len(), seq * (cb + 1),
            "tokens_mask len {} != seq_len {seq} * (num_codebooks+1) {}",
            tokens_mask.len(), cb + 1);
        let bd = cfg.backbone_dim;

        // Apply per-codebook offset to audio codes.
        let mut offset_codes = Vec::with_capacity(audio_codes.len());
        for t in 0..seq {
            for i in 0..cb {
                let c = audio_codes[t * cb + i] as usize + i * cfg.audio_vocab_size;
                offset_codes.push(c as u32);
            }
        }
        let audio_ids = anchor.const_u32_like(
            offset_codes, Shape::from_dims(&[seq * cb]),
        );
        let audio_table = anchor.const_f32_like(
            Arc::clone(&self.weights.audio_embedding),
            Shape::from_dims(&[cb * cfg.audio_vocab_size, bd]),
        );
        let audio_emb = audio_table
            .index_select(0_usize, &audio_ids)?
            .reshape(Shape::from_dims(&[1, seq, cb, bd]))?;

        let text_ids = anchor.const_u32_like(
            text_tokens.to_vec(), Shape::from_dims(&[seq]),
        );
        let text_table = anchor.const_f32_like(
            Arc::clone(&self.weights.text_embedding),
            Shape::from_dims(&[cfg.text_vocab_size, bd]),
        );
        let text_emb = text_table
            .index_select(0_usize, &text_ids)?
            .reshape(Shape::from_dims(&[1, seq, 1, bd]))?;

        // Concatenate audio (cb columns) + text (1 column) along dim 2.
        let combined = audio_emb.concat(&text_emb, 2_usize)?;

        // Apply mask (broadcast over backbone_dim) and sum across codebook+1 axis.
        let mask_f32: Vec<f32> = tokens_mask.iter().map(|&b| b as f32).collect();
        let mask = anchor.const_f32_like(
            mask_f32, Shape::from_dims(&[1, seq, cb + 1, 1]),
        );
        let mask_b = mask.broadcast_to(Shape::from_dims(&[1, seq, cb + 1, bd]))?;
        let gated = combined.mul(&mask_b)?;
        Ok(gated.sum_dim(2_usize)?)
    }

    /// Apply the `codebook0_head` linear to the backbone hidden state.
    /// `backbone_h` shape `(1, S, backbone_dim)` → `(1, S, audio_vocab_size)`.
    pub fn codebook0_logits(&self, backbone_h: &LazyTensor) -> LazyTensor {
        let cfg = &self.config;
        self.weights.codebook0_head.apply_linear(
            backbone_h, cfg.backbone_dim, cfg.audio_vocab_size,
        )
    }

    /// Project a tensor from backbone hidden space to decoder hidden
    /// space (no bias). Used between `cat([h, c0_embed], 1)` and the
    /// decoder's `forward_embeds`.
    pub fn project_to_decoder(&self, curr_h: &LazyTensor) -> LazyTensor {
        let cfg = &self.config;
        self.weights.projection.apply_linear(
            curr_h, cfg.backbone_dim, cfg.decoder_dim,
        )
    }

    /// Apply `audio_head[codebook_idx - 1]` to a decoder hidden
    /// state to produce per-codebook logits.
    ///
    /// `codebook_idx` must be in `1..audio_num_codebooks` (codebook 0
    /// is predicted by `codebook0_logits` from the backbone hidden).
    /// `decoder_h` shape `(1, S, decoder_dim)` →
    /// `(1, S, audio_vocab_size)`.
    pub fn audio_head_logits(
        &self,
        decoder_h: &LazyTensor,
        codebook_idx: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert!(codebook_idx >= 1 && codebook_idx < cfg.audio_num_codebooks,
            "codebook_idx {codebook_idx} must be in 1..{}",
            cfg.audio_num_codebooks);
        let slab = cfg.decoder_dim * cfg.audio_vocab_size;
        let start = (codebook_idx - 1) * slab;
        let end = start + slab;
        // Slice the (num_codebooks - 1, decoder_dim, audio_vocab) tensor
        // by materializing the head-matrix slab as a fresh constant on
        // the same graph as `decoder_h`.
        let head_slice: Arc<[f32]> = Arc::from(
            self.weights.audio_head[start..end].to_vec(),
        );
        let head = decoder_h.const_f32_like(
            head_slice, Shape::from_dims(&[cfg.decoder_dim, cfg.audio_vocab_size]),
        );
        decoder_h.matmul(&head)
    }

    /// Look up one audio codebook embedding for a sampled code.
    /// Applies the codebook-i offset (`i * audio_vocab_size`) before
    /// the lookup. Result shape `(1, 1, backbone_dim)`.
    pub fn audio_embed_for_code(
        &self,
        code: u32,
        codebook_idx: usize,
        anchor: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let bd = cfg.backbone_dim;
        let offset = (codebook_idx * cfg.audio_vocab_size) as u32;
        let table = anchor.const_f32_like(
            Arc::clone(&self.weights.audio_embedding),
            Shape::from_dims(&[cfg.audio_num_codebooks * cfg.audio_vocab_size, bd]),
        );
        let id = anchor.const_u32_like(vec![code + offset], Shape::from_dims(&[1]));
        let emb = table
            .index_select(0_usize, &id)?
            .reshape(Shape::from_dims(&[1, 1, bd]))?;
        Ok(emb)
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }
    fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
    }
    fn ws(n: usize, nb: &mut dyn FnMut() -> f32) -> WeightStorage {
        WeightStorage::F32(vec_of(n, nb))
    }

    fn tiny_cfg() -> CsmConfig {
        CsmConfig {
            audio_num_codebooks: 3,
            audio_vocab_size: 5,
            text_vocab_size: 7,
            backbone_dim: 4,
            decoder_dim: 2,
        }
    }

    fn tiny_model() -> CsmModel {
        let cfg = tiny_cfg();
        let mut nb = rng_seed(2026);
        let weights = CsmWeights {
            audio_embedding: vec_of(
                cfg.audio_num_codebooks * cfg.audio_vocab_size * cfg.backbone_dim,
                &mut nb,
            ),
            text_embedding: vec_of(cfg.text_vocab_size * cfg.backbone_dim, &mut nb),
            codebook0_head: ws(cfg.backbone_dim * cfg.audio_vocab_size, &mut nb),
            projection: ws(cfg.backbone_dim * cfg.decoder_dim, &mut nb),
            audio_head: vec_of(
                (cfg.audio_num_codebooks - 1) * cfg.decoder_dim * cfg.audio_vocab_size,
                &mut nb,
            ),
        };
        CsmModel { config: cfg, weights }
    }

    fn anchor() -> LazyTensor {
        LazyTensor::from_f32(vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu())
    }

    #[test]
    fn embed_frame_shape_and_finite() {
        let model = tiny_model();
        let cb = model.config.audio_num_codebooks;
        let seq = 2;
        let audio_codes = vec![0_u32, 1, 2, 3, 4, 0]; // shape (2, 3)
        let text_tokens = vec![1_u32, 2];
        let mask = vec![1_u8; seq * (cb + 1)];
        let a = anchor();
        let out = model.embed_frame(&audio_codes, &text_tokens, &mask, &a).unwrap();
        assert_eq!(out.shape().dims(), &[1, seq, model.config.backbone_dim]);
        for &v in &out.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn embed_frame_mask_zeros_drop_contribution() {
        let model = tiny_model();
        let cb = model.config.audio_num_codebooks;
        let bd = model.config.backbone_dim;
        let a = anchor();
        // All-zero mask → result is the zero vector.
        let zero_mask = vec![0_u8; 1 * (cb + 1)];
        let zero_out = model.embed_frame(
            &vec![0_u32; cb], &vec![0_u32], &zero_mask, &a,
        ).unwrap().realize_f32();
        for &v in &zero_out {
            assert!(v.abs() < 1e-7, "all-zero mask must zero embed: {v}");
        }
        // All-active mask → result has bd elements all non-zero (with
        // overwhelming probability under random init).
        let one_mask = vec![1_u8; 1 * (cb + 1)];
        let one_out = model.embed_frame(
            &vec![0_u32; cb], &vec![0_u32], &one_mask, &a,
        ).unwrap().realize_f32();
        let any_nonzero = one_out.iter().any(|v| v.abs() > 1e-9);
        assert!(any_nonzero, "active mask must produce non-zero embed");
        let _ = bd;
    }

    #[test]
    fn codebook0_logits_shape() {
        let model = tiny_model();
        let cfg = &model.config;
        let h = LazyTensor::from_f32(
            (0..(1 * 2 * cfg.backbone_dim)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 2, cfg.backbone_dim]),
            &Device::cpu(),
        );
        let logits = model.codebook0_logits(&h);
        assert_eq!(logits.shape().dims(), &[1, 2, cfg.audio_vocab_size]);
    }

    #[test]
    fn project_and_audio_head_chain() {
        let model = tiny_model();
        let cfg = &model.config;
        let a = anchor();
        let curr_h = LazyTensor::from_f32(
            (0..(1 * 3 * cfg.backbone_dim)).map(|i| (i as f32) * 0.01).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, cfg.backbone_dim]),
            &Device::cpu(),
        );
        let proj = model.project_to_decoder(&curr_h);
        assert_eq!(proj.shape().dims(), &[1, 3, cfg.decoder_dim]);
        // Run audio_head_logits for codebook 1 (proj subs in for decoder hidden).
        let ci_logits = model.audio_head_logits(&proj, 1).unwrap();
        let _ = &a;
        assert_eq!(ci_logits.shape().dims(), &[1, 3, cfg.audio_vocab_size]);
        for &v in &ci_logits.realize_f32() { assert!(v.is_finite()); }
    }

    #[test]
    fn audio_embed_for_code_offsets_correctly() {
        let model = tiny_model();
        let cfg = &model.config;
        let a = anchor();
        let bd = cfg.backbone_dim;
        // Look up code 2 from codebook 0 — table row 2.
        let e0 = model.audio_embed_for_code(2, 0, &a).unwrap().realize_f32();
        // Look up code 2 from codebook 1 — table row 2 + 1*audio_vocab_size = 7.
        let e1 = model.audio_embed_for_code(2, 1, &a).unwrap().realize_f32();
        // The same eager byte offsets — read the underlying flat table.
        let table = &model.weights.audio_embedding;
        for d in 0..bd {
            assert!((e0[d] - table[2 * bd + d]).abs() < 1e-7);
            assert!((e1[d] - table[(cfg.audio_vocab_size + 2) * bd + d]).abs() < 1e-7);
        }
    }

    #[test]
    fn preset_sesame() {
        let p = CsmConfig::sesame();
        assert_eq!(p.audio_num_codebooks, 32);
        assert_eq!(p.audio_vocab_size, 2051);
        assert_eq!(p.backbone_dim, 2048);
        assert_eq!(p.decoder_dim, 1024);
    }
}
