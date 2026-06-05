//! NomicBert (nomic-embed-text-v1.x) ported to the lazy-graph API.
//!
//! Nussbaum et al. 2024. A BERT variant designed for text
//! embedding that swaps the standard BERT components in three
//! places:
//!
//!   1. **RoPE** instead of absolute position embeddings. The
//!      rotary base is small (`1000.0` rather than the LLM-typical
//!      `10000.0`) because the encoder is trained at relatively
//!      short sequence lengths. Both split-half and interleaved
//!      RoPE are supported via the `rotary_emb_interleaved`
//!      config flag. `rotary_emb_fraction` controls how many of
//!      the head-dim features get rotated; the remainder is
//!      passed through unchanged.
//!   2. **SwiGLU FFN**: `fc2(fc11(x) * silu(fc12(x)))`. `fc11`
//!      is the value path, `fc12` is the gate path. (The naming
//!      is inherited from the eager Fuel port and matches
//!      upstream weights.) The `n_inner` width is the gated MLP
//!      hidden dim, not `4 * n_embd`.
//!   3. **Fused QKV** projection (single `Wqkv` of shape
//!      `[n_embd, 3 * n_embd]`).
//!
//! Other distinctive bits:
//!
//!   - **Configurable biases** on QKV, output projection, fc11/fc12
//!     and fc2 — the upstream model uses `bias = false` everywhere
//!     but the eager port carries the booleans so we keep them.
//!   - **Optional token type embeddings** (`type_vocab_size > 0`).
//!   - **prenorm flag** picks between Pre-LN
//!     (`residual + sublayer(LN(x))`) and Post-LN
//!     (`LN(residual + sublayer(x))`). nomic-embed-text-v1.5
//!     uses post-LN; some variants enable prenorm.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. Optional additive attention
//! mask of shape `(1, 1, seq, seq)` for padding masking. The
//! returned tensor is per-token hidden states `(1, seq, n_embd)`.
//! For text embeddings, mean-pool over the sequence dim using
//! the attention mask, then L2-normalize the result. v1 keeps
//! the pooling/normalization out of the model itself — they're
//! a one-liner on the caller side and trivially composable.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_glm4::apply_interleaved_partial_rope;
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NomicBertActivation {
    /// SwiGLU (default for nomic-embed-text-v1.x).
    SwiGlu,
    Gelu,
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NomicBertConfig {
    pub vocab_size: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    /// MLP hidden dim (gated SwiGLU hidden, NOT `4 * n_embd`).
    pub n_inner: usize,
    pub n_positions: usize,
    /// 0 disables token type embeddings.
    pub type_vocab_size: usize,
    pub layer_norm_epsilon: f64,
    /// Fraction of `head_dim` that gets rotated. The rest is
    /// passed through unchanged. v1.5 uses `1.0`.
    pub rotary_emb_fraction: f64,
    pub rotary_emb_base: f64,
    /// `true` → interleaved (GPT-J) RoPE; `false` → split-half
    /// (GPT-NeoX / LLaMA) RoPE.
    pub rotary_emb_interleaved: bool,
    pub qkv_proj_bias: bool,
    pub mlp_fc1_bias: bool,
    pub mlp_fc2_bias: bool,
    pub activation: NomicBertActivation,
    /// `true` → Pre-LN; `false` → Post-LN (BERT-shape).
    pub prenorm: bool,
}

impl NomicBertConfig {
    pub fn head_dim(&self) -> usize {
        self.n_embd / self.n_head
    }
    pub fn rotary_emb_dim(&self) -> usize {
        (self.head_dim() as f64 * self.rotary_emb_fraction) as usize
    }
    /// `nomic-ai/nomic-embed-text-v1.5` preset.
    pub fn nomic_embed_text_v1_5() -> Self {
        Self {
            vocab_size: 30528,
            n_embd: 768,
            n_head: 12,
            n_layer: 12,
            n_inner: 3072,
            n_positions: 8192,
            type_vocab_size: 2,
            layer_norm_epsilon: 1e-12,
            rotary_emb_fraction: 1.0,
            rotary_emb_base: 1000.0,
            rotary_emb_interleaved: false,
            qkv_proj_bias: false,
            mlp_fc1_bias: false,
            mlp_fc2_bias: false,
            activation: NomicBertActivation::SwiGlu,
            prenorm: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NomicBertLayerWeights {
    /// `[n_embd, 3 * n_embd]` fused QKV.
    pub wqkv: WeightStorage,
    pub wqkv_bias: Option<Arc<[f32]>>,
    /// `[n_embd, n_embd]` attention output projection.
    pub out_proj: WeightStorage,
    pub out_proj_bias: Option<Arc<[f32]>>,
    /// LayerNorm applied after attention residual (post-LN) or
    /// before attention sublayer (pre-LN).
    pub norm1_gain: Arc<[f32]>,
    pub norm1_bias: Arc<[f32]>,
    /// SwiGLU value path. `[n_embd, n_inner]`.
    pub fc11: WeightStorage,
    pub fc11_bias: Option<Arc<[f32]>>,
    /// SwiGLU gate path. `[n_embd, n_inner]`.
    pub fc12: WeightStorage,
    pub fc12_bias: Option<Arc<[f32]>>,
    /// MLP down-projection. `[n_inner, n_embd]`.
    pub fc2: WeightStorage,
    pub fc2_bias: Option<Arc<[f32]>>,
    /// LayerNorm applied after MLP residual (post-LN) or
    /// before MLP sublayer (pre-LN).
    pub norm2_gain: Arc<[f32]>,
    pub norm2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct NomicBertWeights {
    pub word_embedding: Arc<[f32]>,
    /// `None` when `type_vocab_size == 0`.
    pub token_type_embedding: Option<Arc<[f32]>>,
    pub embed_ln_gain: Arc<[f32]>,
    pub embed_ln_bias: Arc<[f32]>,
    pub layers: Vec<NomicBertLayerWeights>,
}

#[derive(Debug, Clone)]
pub struct NomicBertModel {
    pub config: NomicBertConfig,
    pub weights: NomicBertWeights,
}

impl NomicBertModel {
    /// Run a forward pass.
    ///
    /// - `tokens`: input token ids, length `seq`.
    /// - `token_type_ids`: optional per-token type ids (length `seq`).
    ///   Defaults to all-zeros when the model has token type
    ///   embeddings; ignored otherwise.
    /// - `attention_mask`: optional additive mask of shape
    ///   `(1, 1, seq, seq)` with `0` for keep and `-inf` / large
    ///   negative for mask. Caller is responsible for building it.
    ///
    /// Returns per-token hidden states of shape `(1, seq, n_embd)`.
    pub fn forward(
        &self,
        tokens: &[u32],
        token_type_ids: Option<&[u32]>,
        attention_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert!(seq <= cfg.n_positions);
        assert!(
            cfg.n_head * cfg.head_dim() == cfg.n_embd,
            "n_head * head_dim must equal n_embd",
        );

        // ---- Embeddings -----------------------------------------------------
        let word_emb_t = LazyTensor::from_f32(
            weights.word_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.n_embd]),
            &Device::cpu(),
        );
        let token_ids = word_emb_t.const_u32_like(
            tokens.to_vec(),
            Shape::from_dims(&[seq]),
        );
        let mut embeds = word_emb_t
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_embd]))?;

        if let Some(tte) = &weights.token_type_embedding {
            let tte_t = word_emb_t.const_f32_like(
                Arc::clone(tte),
                Shape::from_dims(&[cfg.type_vocab_size, cfg.n_embd]),
            );
            let tt_ids: Vec<u32> = match token_type_ids {
                Some(ids) => {
                    assert_eq!(ids.len(), seq, "token_type_ids length must match tokens length");
                    ids.to_vec()
                }
                None => vec![0; seq],
            };
            let tt_id_t = word_emb_t.const_u32_like(
                tt_ids,
                Shape::from_dims(&[seq]),
            );
            let tt_emb = tte_t
                .index_select(0_usize, &tt_id_t)?
                .reshape(Shape::from_dims(&[batch, seq, cfg.n_embd]))?;
            embeds = embeds.add(&tt_emb)?;
        }

        let mut h = embeds.layer_norm_affine(std::sync::Arc::clone(&weights.embed_ln_gain), std::sync::Arc::clone(&weights.embed_ln_bias), cfg.layer_norm_epsilon)?;

        // ---- RoPE tables ----------------------------------------------------
        let rope_dim = cfg.rotary_emb_dim();
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rotary_emb_base, 0, seq, rope_dim,
        );

        // ---- Encoder blocks -------------------------------------------------
        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, attention_mask)?;
        }
        Ok(h)
    }

    /// Extract per-token features at the requested layer
    /// indices. Output shape per layer:
    /// `(1, seq, n_embd)`. NomicBert defaults to post-LN
    /// (the public release uses `prenorm: false`); with
    /// `prenorm: true` the captures are pre-LN features
    /// instead. Either way, each capture is the OUTPUT of
    /// the requested encoder layer.
    ///
    /// Layer-id contract: 0-based, strictly increasing, all in
    /// `[0, n_layer)`. Same shape contract as the BERT and
    /// DistilBERT hooks. The optional `token_type_ids` and
    /// `attention_mask` parameters thread through identically
    /// to `forward`.
    ///
    /// # Use cases
    ///
    ///   - **Multi-layer features** for `nomic-ai/nomic-embed-text-v1.5`
    ///     and similar Matryoshka-trained embedding models. The
    ///     v1.5 release exposes 6 output dimensions trained
    ///     jointly; some retrieval pipelines additionally
    ///     concat layer-N hidden states for better recall at
    ///     low embedding dims.
    ///   - **Layer-wise probing** of the RoPE-equipped
    ///     BERT-family backbone.
    pub fn forward_intermediate_layers(
        &self,
        tokens: &[u32],
        layer_ids: &[usize],
        token_type_ids: Option<&[u32]>,
        attention_mask: Option<&LazyTensor>,
    ) -> Result<Vec<LazyTensor>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        let batch = 1;
        assert!(seq > 0);
        assert!(seq <= cfg.n_positions);
        assert!(
            cfg.n_head * cfg.head_dim() == cfg.n_embd,
            "n_head * head_dim must equal n_embd",
        );
        assert!(!layer_ids.is_empty(), "layer_ids must not be empty");
        for w in layer_ids.windows(2) {
            assert!(w[0] < w[1], "layer_ids must be strictly increasing");
        }
        let depth = weights.layers.len();
        assert!(
            *layer_ids.last().unwrap() < depth,
            "layer_ids must all be in [0, n_layer = {depth})",
        );

        let word_emb_t = LazyTensor::from_f32(
            weights.word_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, cfg.n_embd]),
            &Device::cpu(),
        );
        let token_ids = word_emb_t.const_u32_like(
            tokens.to_vec(), Shape::from_dims(&[seq]),
        );
        let mut embeds = word_emb_t
            .index_select(0_usize, &token_ids)?
            .reshape(Shape::from_dims(&[batch, seq, cfg.n_embd]))?;
        if let Some(tte) = &weights.token_type_embedding {
            let tte_t = word_emb_t.const_f32_like(
                Arc::clone(tte),
                Shape::from_dims(&[cfg.type_vocab_size, cfg.n_embd]),
            );
            let tt_ids: Vec<u32> = match token_type_ids {
                Some(ids) => {
                    assert_eq!(ids.len(), seq, "token_type_ids length must match tokens length");
                    ids.to_vec()
                }
                None => vec![0; seq],
            };
            let tt_id_t = word_emb_t.const_u32_like(tt_ids, Shape::from_dims(&[seq]));
            let tt_emb = tte_t
                .index_select(0_usize, &tt_id_t)?
                .reshape(Shape::from_dims(&[batch, seq, cfg.n_embd]))?;
            embeds = embeds.add(&tt_emb)?;
        }
        let mut h = embeds.layer_norm_affine(std::sync::Arc::clone(&weights.embed_ln_gain), std::sync::Arc::clone(&weights.embed_ln_bias), cfg.layer_norm_epsilon)?;

        let rope_dim = cfg.rotary_emb_dim();
        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rotary_emb_base, 0, seq, rope_dim,
        );

        let mut out = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0;
        for (idx, layer) in weights.layers.iter().enumerate() {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin, attention_mask)?;
            if next_capture < layer_ids.len() && layer_ids[next_capture] == idx {
                out.push(h.clone());
                next_capture += 1;
            }
        }
        Ok(out)
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &NomicBertLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        attention_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        if cfg.prenorm {
            // Pre-LN: y = x + attn(LN(x)); z = y + ffn(LN(y)).
            let x_norm = x.layer_norm_affine(std::sync::Arc::clone(&layer.norm1_gain), std::sync::Arc::clone(&layer.norm1_bias), cfg.layer_norm_epsilon)?;
            let attn = self.attention(&x_norm, layer, rope_cos, rope_sin, attention_mask)?;
            let y = x.add(&attn)?;
            let y_norm = y.layer_norm_affine(std::sync::Arc::clone(&layer.norm2_gain), std::sync::Arc::clone(&layer.norm2_bias), cfg.layer_norm_epsilon)?;
            let mlp = self.mlp(&y_norm, layer)?;
            y.add(&mlp)
        } else {
            // Post-LN (BERT-shape): y = LN(x + attn(x)); z = LN(y + ffn(y)).
            let attn = self.attention(x, layer, rope_cos, rope_sin, attention_mask)?;
            let y = x.add(&attn)?.layer_norm_affine(std::sync::Arc::clone(&layer.norm1_gain), std::sync::Arc::clone(&layer.norm1_bias), cfg.layer_norm_epsilon)?;
            let mlp = self.mlp(&y, layer)?;
            Ok(y.add(&mlp)?.layer_norm_affine(std::sync::Arc::clone(&layer.norm2_gain), std::sync::Arc::clone(&layer.norm2_bias), cfg.layer_norm_epsilon)?)
        }
    }

    fn attention(
        &self,
        x: &LazyTensor,
        layer: &NomicBertLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        attention_mask: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let d = cfg.n_embd;
        let n_heads = cfg.n_head;
        let head_dim = cfg.head_dim();

        let qkv = layer.wqkv.apply_linear(x, d, 3 * d);
        let qkv = opt_bias(qkv, layer.wqkv_bias.as_ref(), 3 * d, x)?;
        // (batch, seq, 3 * d) → split Q / K / V on last dim.
        let q = qkv.slice(2_usize, 0, d)?;
        let k = qkv.slice(2_usize, d, d)?;
        let v = qkv.slice(2_usize, 2 * d, d)?;

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        // RoPE on the rotary prefix.
        let rope_dim = cfg.rotary_emb_dim();
        let q_r = apply_rope(&q, rope_cos, rope_sin, head_dim, rope_dim, cfg.rotary_emb_interleaved)?;
        let k_r = apply_rope(&k, rope_cos, rope_sin, head_dim, rope_dim, cfg.rotary_emb_interleaved)?;

        let scale = 1.0 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_r.transpose()?)?.mul_scalar(scale);
        let scores = match attention_mask {
            None => scores,
            Some(mask) => scores.broadcast_add(mask)?,
        };
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        let out = layer.out_proj.apply_linear(&merged, d, d);
        opt_bias(out, layer.out_proj_bias.as_ref(), d, x)
    }

    fn mlp(&self, x: &LazyTensor, layer: &NomicBertLayerWeights) -> Result<LazyTensor> {
        let cfg = &self.config;
        let d = cfg.n_embd;
        let h = cfg.n_inner;
        match cfg.activation {
            NomicBertActivation::SwiGlu => {
                let val = layer.fc11.apply_linear(x, d, h);
                let val = opt_bias(val, layer.fc11_bias.as_ref(), h, x)?;
                let gate = layer.fc12.apply_linear(x, d, h);
                let gate = opt_bias(gate, layer.fc12_bias.as_ref(), h, x)?;
                let inner = val.mul(&gate.silu())?;
                let down = layer.fc2.apply_linear(&inner, h, d);
                opt_bias(down, layer.fc2_bias.as_ref(), d, x)
            }
            NomicBertActivation::Gelu => {
                let up = layer.fc11.apply_linear(x, d, h);
                let up = opt_bias(up, layer.fc11_bias.as_ref(), h, x)?;
                let act = up.gelu_erf();
                let down = layer.fc2.apply_linear(&act, h, d);
                opt_bias(down, layer.fc2_bias.as_ref(), d, x)
            }
            NomicBertActivation::Relu => {
                let up = layer.fc11.apply_linear(x, d, h);
                let up = opt_bias(up, layer.fc11_bias.as_ref(), h, x)?;
                let act = up.relu();
                let down = layer.fc2.apply_linear(&act, h, d);
                opt_bias(down, layer.fc2_bias.as_ref(), d, x)
            }
        }
    }
}

fn apply_rope(
    qk: &LazyTensor,
    rope_cos: &LazyTensor,
    rope_sin: &LazyTensor,
    head_dim: usize,
    rope_dim: usize,
    interleaved: bool,
) -> Result<LazyTensor> {
    if rope_dim == 0 {
        return Ok(qk.clone());
    }
    if interleaved {
        return apply_interleaved_partial_rope(qk, rope_cos, rope_sin, head_dim, rope_dim);
    }
    if rope_dim == head_dim {
        return qk.rope_with_tables(rope_cos, rope_sin);
    }
    // Split-half partial rope: rotate the first rope_dim features,
    // pass the rest through unchanged.
    let shape = qk.shape();
    let dims = shape.dims();
    assert_eq!(dims.len(), 4);
    let pass_dim = head_dim - rope_dim;
    let rot = qk.slice(3_usize, 0, rope_dim)?;
    let pass = qk.slice(3_usize, rope_dim, pass_dim)?;
    let rotated = rot.rope_with_tables(rope_cos, rope_sin)?;
    rotated.concat(&pass, 3_usize)
}

fn opt_bias(
    x: LazyTensor,
    b: Option<&Arc<[f32]>>,
    n: usize,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let _ = (n, anchor);
    x.add_optional_trailing_bias(b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_cfg() -> NomicBertConfig {
        NomicBertConfig {
            vocab_size: 32,
            n_embd: 16,
            n_head: 4,
            n_layer: 2,
            n_inner: 24,
            n_positions: 16,
            type_vocab_size: 2,
            layer_norm_epsilon: 1e-12,
            rotary_emb_fraction: 1.0,
            rotary_emb_base: 1000.0,
            rotary_emb_interleaved: false,
            qkv_proj_bias: false,
            mlp_fc1_bias: false,
            mlp_fc2_bias: false,
            activation: NomicBertActivation::SwiGlu,
            prenorm: false,
        }
    }

    fn tiny_weights(cfg: &NomicBertConfig) -> NomicBertWeights {
        let mut s: u32 = 13579;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let d = cfg.n_embd;
        let n_inner = cfg.n_inner;
        let word_embedding = vec_of(cfg.vocab_size * d, &mut *nb);
        let token_type_embedding = if cfg.type_vocab_size > 0 {
            Some(vec_of(cfg.type_vocab_size * d, &mut *nb))
        } else {
            None
        };
        let embed_ln_gain = Arc::from(vec![1.0_f32; d]);
        let embed_ln_bias = Arc::from(vec![0.0_f32; d]);

        let layers: Vec<NomicBertLayerWeights> = (0..cfg.n_layer)
            .map(|_| NomicBertLayerWeights {
                wqkv: WeightStorage::F32(vec_of(d * 3 * d, &mut *nb)),
                wqkv_bias: if cfg.qkv_proj_bias { Some(vec_of(3 * d, &mut *nb)) } else { None },
                out_proj: WeightStorage::F32(vec_of(d * d, &mut *nb)),
                out_proj_bias: if cfg.qkv_proj_bias { Some(vec_of(d, &mut *nb)) } else { None },
                norm1_gain: Arc::from(vec![1.0_f32; d]),
                norm1_bias: Arc::from(vec![0.0_f32; d]),
                fc11: WeightStorage::F32(vec_of(d * n_inner, &mut *nb)),
                fc11_bias: if cfg.mlp_fc1_bias { Some(vec_of(n_inner, &mut *nb)) } else { None },
                fc12: WeightStorage::F32(vec_of(d * n_inner, &mut *nb)),
                fc12_bias: if cfg.mlp_fc1_bias { Some(vec_of(n_inner, &mut *nb)) } else { None },
                fc2: WeightStorage::F32(vec_of(n_inner * d, &mut *nb)),
                fc2_bias: if cfg.mlp_fc2_bias { Some(vec_of(d, &mut *nb)) } else { None },
                norm2_gain: Arc::from(vec![1.0_f32; d]),
                norm2_bias: Arc::from(vec![0.0_f32; d]),
            })
            .collect();

        NomicBertWeights {
            word_embedding,
            token_type_embedding,
            embed_ln_gain,
            embed_ln_bias,
            layers,
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = NomicBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4];
        let out = model.forward(&tokens, None, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.n_embd]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    /// Bidirectional attention — changing the last token must
    /// alter position 0's hidden state.
    #[test]
    fn bidirectional_attention() {
        let cfg = tiny_cfg();
        let model = NomicBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let toks_a = [1_u32, 2, 3, 4];
        let toks_b = [1_u32, 2, 3, 15];
        let a = model.forward(&toks_a, None, None).unwrap().realize_f32();
        let b = model.forward(&toks_b, None, None).unwrap().realize_f32();
        let d = cfg.n_embd;
        let mut max_diff = 0.0_f32;
        for i in 0..d {
            max_diff = max_diff.max((a[i] - b[i]).abs());
        }
        assert!(max_diff > 1e-6,
            "last-token change must affect position 0 (bidirectional), max_diff = {max_diff}");
    }

    /// Token type embedding is wired — changing the row 0 of the
    /// token-type table (the default type id at every position
    /// when `token_type_ids` is `None`) must alter the output.
    #[test]
    fn token_type_embedding_is_wired() {
        let cfg = tiny_cfg();
        let mut base = tiny_weights(&cfg);
        let mut tte = base.token_type_embedding.as_ref().unwrap().to_vec();
        for i in 0..cfg.n_embd {
            tte[i] += 1.0;
        }
        let mut modified = base.clone();
        modified.token_type_embedding = Some(Arc::from(tte));
        // Reset base's TTE to make sure we're not aliasing.
        base.token_type_embedding = Some(
            Arc::from(base.token_type_embedding.as_ref().unwrap().to_vec())
        );

        let m_a = NomicBertModel { config: cfg.clone(), weights: base };
        let m_b = NomicBertModel { config: cfg, weights: modified };
        let toks = [1_u32, 2, 3, 4];
        let a = m_a.forward(&toks, None, None).unwrap().realize_f32();
        let b = m_b.forward(&toks, None, None).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "token-type change must alter output, max_diff = {max_diff}");
    }

    /// SwiGLU MLP — verify the gate path is wired. Replace `fc12`
    /// with all-zeros: the gate becomes `silu(0) = 0`, so
    /// `val * silu(gate) = 0`, completely zeroing out the MLP
    /// contribution. If the model treated fc12 as a no-op (e.g.,
    /// fell through to a non-gated path), zeroing fc12 would not
    /// change the output. The output must differ.
    #[test]
    fn swiglu_gate_is_wired() {
        let cfg = tiny_cfg();
        let base = tiny_weights(&cfg);
        let mut modified = base.clone();
        let h = cfg.n_inner;
        let d = cfg.n_embd;
        modified.layers[0].fc12 = WeightStorage::F32(Arc::from(vec![0.0_f32; d * h]));

        let m_a = NomicBertModel { config: cfg.clone(), weights: base };
        let m_b = NomicBertModel { config: cfg, weights: modified };
        let toks = [1_u32, 2, 3, 4];
        let a = m_a.forward(&toks, None, None).unwrap().realize_f32();
        let b = m_b.forward(&toks, None, None).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "zeroing fc12 (gate) must alter output (SwiGLU MLP zeroed out), max_diff = {max_diff}");
    }

    /// Interleaved-RoPE variant runs end-to-end with finite output.
    #[test]
    fn interleaved_rope_runs() {
        let mut cfg = tiny_cfg();
        cfg.rotary_emb_interleaved = true;
        let model = NomicBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4];
        let out = model.forward(&tokens, None, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.n_embd]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    /// Pre-LN variant runs end-to-end with finite output.
    #[test]
    fn prenorm_runs() {
        let mut cfg = tiny_cfg();
        cfg.prenorm = true;
        let model = NomicBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4];
        let out = model.forward(&tokens, None, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.n_embd]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    /// `forward_intermediate_layers` returns per-layer features
    /// `(1, seq, n_embd)`. Mirrors the BERT-family hooks.
    #[test]
    fn forward_intermediate_layers_shape() {
        let cfg = tiny_cfg();
        let model = NomicBertModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens = [1_u32, 2, 3, 4];
        let outs = model.forward_intermediate_layers(&tokens, &[0_usize, 1], None, None).unwrap();
        assert_eq!(outs.len(), 2);
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.n_embd]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
        let a = outs[0].realize_f32();
        let b = outs[1].realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "layer 0 and layer 1 intermediates must differ, max_diff = {max_diff}");
    }
}
