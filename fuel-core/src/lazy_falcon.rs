//! Falcon (7B and similar) decoder ported to the lazy-graph API.
//!
//! Phase D LLM port. Falcon is architecturally distinct from
//! LLaMA-cousins:
//!   1. **Parallel attention + FFN** — `out = attn(ln(x)) + mlp(ln(x))
//!      + x` with a shared LayerNorm input. Two branches sum into one
//!      residual instead of LLaMA's serial two-residual flow.
//!   2. **LayerNorm** (with bias) — not RmsNorm. Both `gamma` and
//!      `beta` live per layer.
//!   3. **Multi-query attention** (n_head_kv == 1) by default — one
//!      shared K and one shared V for all attention heads. Implemented
//!      via the existing GQA replication code with `num_kv_heads = 1`.
//!   4. **Standard GELU MLP** — `down(gelu(up(x)))`, no gate path
//!      (h → 4h → h, two projections).
//!   5. **No final LayerNorm** post-decoder per the eager
//!      reference — wait, yes there is: `ln_f` after all decoder
//!      blocks. So: input embed → N × decoder block → final LN →
//!      lm_head.
//!   6. **Optional projection biases** — `cfg.bias` flag enables
//!      biases on Q/K/V/O/MLP linears.
//!
//! Custom [`FalconLayerWeights`] because LayerNorm has both gain and
//! bias (vs LLaMA RmsNorm's gain-only), and the MLP shape differs
//! (no gate path).
//!
//! # Scope (v1)
//!
//! - Forward-only, single sequence (`batch == 1`), no KV cache.
//! - Multi-query attention with `n_head_kv = 1` (the Falcon 7B
//!   default). The struct carries `n_head_kv` so non-MQA variants
//!   can be plugged in.
//! - No ALiBi, no `new_decoder_architecture` (Falcon-180B uses it;
//!   add when needed).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_ir::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct FalconConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    /// Per-layer K/V head count. `1` for default MQA Falcon-7B.
    pub n_head_kv: usize,
    pub layer_norm_epsilon: f64,
    pub max_position_embeddings: usize,
    /// True for Falcon-7B/40B/180B — parallel attention + FFN with a
    /// shared input LayerNorm.
    pub parallel_attn: bool,
    /// True when projection weights have additive biases.
    pub bias: bool,
    pub rope_theta: f64,
}

impl FalconConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `tiiuae/falcon-7b` defaults.
    pub fn falcon_7b() -> Self {
        Self {
            vocab_size: 65_024,
            hidden_size: 4544,
            num_hidden_layers: 32,
            num_attention_heads: 71,
            n_head_kv: 1,
            layer_norm_epsilon: 1e-5,
            max_position_embeddings: 2048,
            parallel_attn: true,
            bias: false,
            rope_theta: 10_000.0,
        }
    }
}

/// Falcon per-layer weights. Distinct from `crate::lazy::LayerWeights`
/// because:
///   - Input LN has both `gain` and `bias` (and an optional
///     post-attention LN when `parallel_attn == false`).
///   - MLP is just `up + down` (no gate).
#[derive(Debug, Clone)]
pub struct FalconLayerWeights {
    pub input_ln_gain: Arc<[f32]>,
    pub input_ln_bias: Arc<[f32]>,
    /// Present only when `parallel_attn == false` (Falcon-7B leaves
    /// this `None`).
    pub post_attn_ln: Option<(Arc<[f32]>, Arc<[f32]>)>,
    pub attn_q: WeightStorage,
    pub attn_q_bias: Option<Arc<[f32]>>,
    pub attn_k: WeightStorage,
    pub attn_k_bias: Option<Arc<[f32]>>,
    pub attn_v: WeightStorage,
    pub attn_v_bias: Option<Arc<[f32]>>,
    pub attn_dense: WeightStorage,
    pub attn_dense_bias: Option<Arc<[f32]>>,
    /// `[hidden_size, 4 * hidden_size]`.
    pub mlp_up: WeightStorage,
    pub mlp_up_bias: Option<Arc<[f32]>>,
    /// `[4 * hidden_size, hidden_size]`.
    pub mlp_down: WeightStorage,
    pub mlp_down_bias: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct FalconWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<FalconLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    pub output: WeightStorage,
}

#[derive(Debug, Clone)]
pub struct FalconModel {
    pub config: FalconConfig,
    pub weights: FalconWeights,
}

impl FalconModel {
    pub fn forward(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let h_norm = self.run_backbone(tokens, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Run the decoder forward up to the final LayerNorm and
    /// return per-token hidden states `(1, seq, hidden_size)`.
    /// Skips the `lm_head` projection. Mirrors the
    /// `forward_hidden` pattern across the LLM family —
    /// Falcon-specific bit is the final-LN uses gain+bias
    /// affine (LayerNorm, not RmsNorm).
    pub fn forward_hidden(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        self.run_backbone(tokens, start_pos)
    }

    /// Multimodal entry point. Skips token embedding; runs the decoder
    /// over pre-embedded inputs. Falcon does NOT scale embeddings.
    pub fn forward_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let h_norm = self.run_backbone_embeds(embeds, start_pos)?;
        self.apply_lm_head(&h_norm)
    }

    /// Hidden-state variant of [`Self::forward_embeds`].
    pub fn forward_hidden_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        self.run_backbone_embeds(embeds, start_pos)
    }

    /// Build per-token embeddings without running the decoder.
    pub fn embed_tokens_anchored(
        &self, anchor: &LazyTensor, tokens: &[u32],
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        anchor.embed_tokens_anchored(
            self.weights.token_embedding.clone(),
            cfg.vocab_size, cfg.hidden_size, tokens,
        )
    }

    fn apply_lm_head(&self, h_norm: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        Ok(self.weights.output.apply_linear(h_norm, cfg.hidden_size, cfg.vocab_size))
    }

    /// Shared backbone: embed → RoPE → per-layer parallel
    /// attn + MLP (Falcon's parallel structure) → final
    /// LayerNorm.
    fn run_backbone(&self, tokens: &[u32], start_pos: usize) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let seq = tokens.len();
        assert!(seq > 0, "FalconModel: tokens must be non-empty");

        let h = LazyTensor::embed_tokens(
            weights.token_embedding.clone(), cfg.vocab_size, cfg.hidden_size, tokens, &Device::cpu(),
        )?;
        self.run_backbone_embeds(&h, start_pos)
    }

    fn run_backbone_embeds(
        &self, embeds: &LazyTensor, start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = embeds.shape();
        let dims = dims.dims();
        if dims.len() != 3 || dims[0] != 1 || dims[2] != cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "FalconModel::forward_embeds: expected embeds shape (1, seq, hidden_size={}), got {:?}",
                cfg.hidden_size, dims,
            )).bt());
        }
        let seq = dims[1];
        if seq == 0 {
            return Err(crate::Error::Msg(
                "FalconModel::forward_embeds: seq must be > 0".into(),
            ).bt());
        }
        let head_dim = cfg.head_dim();
        if cfg.num_attention_heads * head_dim != cfg.hidden_size {
            return Err(crate::Error::Msg(
                "FalconConfig: num_attention_heads * head_dim must equal hidden_size".into(),
            ).bt());
        }
        if cfg.n_head_kv == 0 || cfg.num_attention_heads % cfg.n_head_kv != 0 {
            return Err(crate::Error::Msg(
                "FalconConfig: num_attention_heads must be a positive multiple of n_head_kv".into(),
            ).bt());
        }
        let mut h = embeds.clone();

        let (rope_cos, rope_sin) = h.rope_tables_const(
            cfg.rope_theta, start_pos, seq, head_dim,
        );

        for layer in &weights.layers {
            h = self.apply_layer(&h, layer, &rope_cos, &rope_sin)?;
        }
        h.layer_norm_affine(
            std::sync::Arc::clone(&weights.final_ln_gain),
            std::sync::Arc::clone(&weights.final_ln_bias),
            cfg.layer_norm_epsilon,
        )
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &FalconLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let head_dim = cfg.head_dim();
        let x_shape = x.shape();
        let dims = x_shape.dims();
        let batch = dims[0];
        let seq = dims[1];
        let kv_dim = cfg.n_head_kv * head_dim;

        // Shared input LayerNorm for attention (and FFN in parallel mode).
        let x_ln = x.layer_norm_affine(std::sync::Arc::clone(&layer.input_ln_gain), std::sync::Arc::clone(&layer.input_ln_bias), cfg.layer_norm_epsilon)?;

        let attn_output = self.attention(&x_ln, layer, rope_cos, rope_sin, batch, seq, head_dim, kv_dim)?;

        if cfg.parallel_attn {
            // `out = attn(ln(x)) + mlp(ln(x)) + x` — both branches use
            // the SAME ln(x) input, and a single residual sums them.
            let mlp_output = self.mlp(&x_ln, layer, batch, seq)?;
            let summed = attn_output.add(&mlp_output)?;
            x.add(&summed)
        } else {
            // Serial: `h1 = attn(ln(x)) + x; out = mlp(ln'(h1)) + h1`.
            let h1 = x.add(&attn_output)?;
            let h1_ln = match &layer.post_attn_ln {
                Some((g, b)) => h1.layer_norm_affine(std::sync::Arc::clone(&g), std::sync::Arc::clone(&b), cfg.layer_norm_epsilon)?,
                None => h1.clone(),
            };
            let mlp_output = self.mlp(&h1_ln, layer, batch, seq)?;
            h1.add(&mlp_output)
        }
    }

    fn attention(
        &self,
        x_ln: &LazyTensor,
        layer: &FalconLayerWeights,
        rope_cos: &LazyTensor,
        rope_sin: &LazyTensor,
        batch: usize,
        seq: usize,
        head_dim: usize,
        kv_dim: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let q = layer.attn_q.apply_linear(x_ln, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.attn_q_bias.as_ref())?;
        let k = layer.attn_k.apply_linear(x_ln, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_k_bias.as_ref())?;
        let v = layer.attn_v.apply_linear(x_ln, cfg.hidden_size, kv_dim).add_optional_trailing_bias(layer.attn_v_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(cfg.num_attention_heads, head_dim)?;
        let k = k.split_heads(cfg.n_head_kv, head_dim)?;
        let v = v.split_heads(cfg.n_head_kv, head_dim)?;

        let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
        let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

        // Multi-query attention: broadcast K/V from n_head_kv → num_heads.
        let n_rep = cfg.num_attention_heads / cfg.n_head_kv;
        let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
        let v_full = v.repeat_interleave(1_usize, n_rep)?;

        let k_t = k_full.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let mask = LazyTensor::additive_causal_mask_like(&x_ln, seq)
            .reshape(Shape::from_dims(&[1, 1, seq, seq]))?;
        let scores_masked = scores_scaled.broadcast_add(&mask)?;
        let attn = scores_masked.softmax_last_dim()?;
        let attn_v = attn.matmul(&v_full)?;

        let merged = attn_v.merge_heads()?;
        layer.attn_dense.apply_linear(&merged, cfg.hidden_size, cfg.hidden_size).add_optional_trailing_bias(layer.attn_dense_bias.as_ref())
    }

    fn mlp(
        &self,
        x_ln: &LazyTensor,
        layer: &FalconLayerWeights,
        _batch: usize,
        _seq: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let inter = 4 * cfg.hidden_size;
        let up = layer.mlp_up.apply_linear(x_ln, cfg.hidden_size, inter).add_optional_trailing_bias(layer.mlp_up_bias.as_ref())?;
        let up_act = up.gelu();
        layer.mlp_down.apply_linear(&up_act, inter, cfg.hidden_size).add_optional_trailing_bias(layer.mlp_down_bias.as_ref())
    }
}

// ---- HuggingFace safetensors loader ----------------------------------------

impl FalconWeights {
    /// Load Falcon weights from HF safetensors.
    ///
    /// HF naming (matches `tiiuae/falcon-7b`):
    ///   - `transformer.word_embeddings.weight`
    ///   - `transformer.h.{i}.input_layernorm.{weight,bias}`
    ///   - `transformer.h.{i}.self_attention.query_key_value.{weight,bias}`
    ///     — fused QKV, deinterleaved at load time.
    ///   - `transformer.h.{i}.self_attention.dense.{weight,bias}`
    ///   - `transformer.h.{i}.mlp.dense_h_to_4h.{weight,bias}`
    ///   - `transformer.h.{i}.mlp.dense_4h_to_h.{weight,bias}`
    ///   - `transformer.ln_f.{weight,bias}`
    ///   - `lm_head.weight` (or tied to word_embeddings)
    ///
    /// # Fused-QKV layout
    ///
    /// For `n_head_kv = 1` (Falcon-7B multi-query): the fused tensor
    /// is `((num_heads + 2) * head_dim) × hidden_size`. After
    /// transpose to `hidden_size × ((num_heads + 2) * head_dim)`, the
    /// last-dim columns are laid out `[Q_h0_d0..Q_hn_d0, K_d0, V_d0]`
    /// per `head_dim`-block. Q is the first `num_heads * head_dim`
    /// columns; K is the next `head_dim` columns; V is the last
    /// `head_dim` columns.
    ///
    /// For non-multi-query (`n_head_kv = num_heads`): the fused
    /// tensor is `(3 * hidden_size) × hidden_size` reshaped as
    /// `(num_heads, 3, head_dim) × hidden_size`. After transpose,
    /// pull out Q/K/V by stride-3 indexing.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &FalconConfig,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix};

        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.n_head_kv * head_dim;
        let q_dim = cfg.num_attention_heads * head_dim;

        let token_embedding = Arc::from(load_tensor_as_f32(
            st, "transformer.word_embeddings.weight",
        )?);

        let multi_query = cfg.n_head_kv == 1;
        let qkv_out_dim = if multi_query {
            h + 2 * head_dim
        } else {
            3 * h
        };

        let mut layers: Vec<FalconLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("transformer.h.{i}");

            let input_ln_gain = Arc::from(load_tensor_as_f32(st, &format!("{p}.input_layernorm.weight"))?);
            let input_ln_bias = Arc::from(load_tensor_as_f32(st, &format!("{p}.input_layernorm.bias"))?);
            // post_attn_ln only exists in the non-parallel variant.
            let post_attn_ln = if !cfg.parallel_attn {
                Some((
                    Arc::from(load_tensor_as_f32(st, &format!("{p}.post_attention_layernorm.weight"))?),
                    Arc::from(load_tensor_as_f32(st, &format!("{p}.post_attention_layernorm.bias"))?),
                ))
            } else {
                None
            };

            // Fused QKV: shape (qkv_out_dim, hidden_size) on disk.
            let qkv = load_transposed_matrix(
                st, &format!("{p}.self_attention.query_key_value.weight"),
                qkv_out_dim, h,
            )?;
            let (attn_q_flat, attn_k_flat, attn_v_flat) = split_fused_qkv(
                &qkv, h, cfg.num_attention_heads, head_dim, multi_query,
            );

            // Optional QKV bias.
            let (attn_q_bias, attn_k_bias, attn_v_bias) = if cfg.bias {
                let qkv_bias = load_tensor_as_f32(
                    st, &format!("{p}.self_attention.query_key_value.bias"),
                )?;
                let (qb, kb, vb) = split_fused_qkv_bias(
                    &qkv_bias, cfg.num_attention_heads, head_dim, multi_query,
                );
                (Some(Arc::from(qb)), Some(Arc::from(kb)), Some(Arc::from(vb)))
            } else {
                (None, None, None)
            };

            let attn_dense = crate::lazy::load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.self_attention.dense.weight"), h, q_dim,
            )?;
            let attn_dense_bias = if cfg.bias {
                Some(Arc::from(load_tensor_as_f32(st, &format!("{p}.self_attention.dense.bias"))?))
            } else {
                None
            };

            let inter = 4 * h;
            let mlp_up = crate::lazy::load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.mlp.dense_h_to_4h.weight"), inter, h,
            )?;
            let mlp_up_bias = if cfg.bias {
                Some(Arc::from(load_tensor_as_f32(st, &format!("{p}.mlp.dense_h_to_4h.bias"))?))
            } else {
                None
            };
            let mlp_down = crate::lazy::load_transposed_matrix_preserve_dtype(
                st, &format!("{p}.mlp.dense_4h_to_h.weight"), h, inter,
            )?;
            let mlp_down_bias = if cfg.bias {
                Some(Arc::from(load_tensor_as_f32(st, &format!("{p}.mlp.dense_4h_to_h.bias"))?))
            } else {
                None
            };

            layers.push(FalconLayerWeights {
                input_ln_gain, input_ln_bias, post_attn_ln,
                attn_q: WeightStorage::F32(Arc::from(attn_q_flat)),
                attn_q_bias,
                attn_k: WeightStorage::F32(Arc::from(attn_k_flat)),
                attn_k_bias,
                attn_v: WeightStorage::F32(Arc::from(attn_v_flat)),
                attn_v_bias,
                attn_dense, attn_dense_bias,
                mlp_up, mlp_up_bias,
                mlp_down, mlp_down_bias,
            });
        }

        let final_ln_gain = Arc::from(load_tensor_as_f32(st, "transformer.ln_f.weight")?);
        let final_ln_bias = Arc::from(load_tensor_as_f32(st, "transformer.ln_f.bias")?);
        // lm_head: tied if no lm_head.weight present, else load.
        let output = match crate::lazy::load_transposed_matrix_preserve_dtype(
            st, "lm_head.weight", cfg.vocab_size, h,
        ) {
            Ok(w) => w,
            Err(_) => {
                // Tied: transpose token_embedding (vocab, h) -> (h, vocab).
                crate::lazy_llama_full::tied_lm_head_from_embeddings(
                    &token_embedding, cfg.vocab_size, h,
                )
            }
        };

        Ok(Self {
            token_embedding,
            layers,
            final_ln_gain,
            final_ln_bias,
            output,
        })
    }
}

/// Split a transposed fused-QKV matrix (shape [hidden_size, qkv_out_dim],
/// row-major) into separate Q, K, V matrices.
///
/// Multi-query (n_head_kv=1): qkv_out_dim = (num_heads + 2) * head_dim.
/// Columns 0..num_heads*head_dim are Q, next head_dim cols are K, last
/// head_dim cols are V.
///
/// Non-multi-query: qkv_out_dim = 3 * num_heads * head_dim, with the
/// stride-3 head interleaved layout [Q_h0, K_h0, V_h0, Q_h1, K_h1, ...].
fn split_fused_qkv(
    transposed: &[f32],
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    multi_query: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_dim = num_heads * head_dim;
    let kv_dim = if multi_query { head_dim } else { q_dim };
    let qkv_out = if multi_query { q_dim + 2 * head_dim } else { 3 * q_dim };

    let mut q = vec![0.0_f32; hidden_size * q_dim];
    let mut k = vec![0.0_f32; hidden_size * kv_dim];
    let mut v = vec![0.0_f32; hidden_size * kv_dim];

    if multi_query {
        for row in 0..hidden_size {
            let src = &transposed[row * qkv_out..(row + 1) * qkv_out];
            q[row * q_dim..(row + 1) * q_dim].copy_from_slice(&src[0..q_dim]);
            k[row * head_dim..(row + 1) * head_dim].copy_from_slice(&src[q_dim..q_dim + head_dim]);
            v[row * head_dim..(row + 1) * head_dim].copy_from_slice(&src[q_dim + head_dim..]);
        }
    } else {
        // (num_heads, 3, head_dim) interleaved per head.
        for row in 0..hidden_size {
            let src = &transposed[row * qkv_out..(row + 1) * qkv_out];
            for h_i in 0..num_heads {
                let base = h_i * 3 * head_dim;
                q[row * q_dim + h_i * head_dim..row * q_dim + (h_i + 1) * head_dim]
                    .copy_from_slice(&src[base..base + head_dim]);
                k[row * kv_dim + h_i * head_dim..row * kv_dim + (h_i + 1) * head_dim]
                    .copy_from_slice(&src[base + head_dim..base + 2 * head_dim]);
                v[row * kv_dim + h_i * head_dim..row * kv_dim + (h_i + 1) * head_dim]
                    .copy_from_slice(&src[base + 2 * head_dim..base + 3 * head_dim]);
            }
        }
    }

    (q, k, v)
}

/// Split a fused-QKV bias vector. Bias has shape (qkv_out_dim,).
fn split_fused_qkv_bias(
    bias: &[f32],
    num_heads: usize,
    head_dim: usize,
    multi_query: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let q_dim = num_heads * head_dim;
    if multi_query {
        let q = bias[0..q_dim].to_vec();
        let k = bias[q_dim..q_dim + head_dim].to_vec();
        let v = bias[q_dim + head_dim..q_dim + 2 * head_dim].to_vec();
        (q, k, v)
    } else {
        let mut q = vec![0.0_f32; q_dim];
        let mut k = vec![0.0_f32; q_dim];
        let mut v = vec![0.0_f32; q_dim];
        for h_i in 0..num_heads {
            let base = h_i * 3 * head_dim;
            q[h_i * head_dim..(h_i + 1) * head_dim].copy_from_slice(&bias[base..base + head_dim]);
            k[h_i * head_dim..(h_i + 1) * head_dim].copy_from_slice(&bias[base + head_dim..base + 2 * head_dim]);
            v[h_i * head_dim..(h_i + 1) * head_dim].copy_from_slice(&bias[base + 2 * head_dim..base + 3 * head_dim]);
        }
        (q, k, v)
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_weights(cfg: &FalconConfig) -> FalconWeights {
        let mut s: u32 = 5555;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let vec_of = |n: usize, next: &mut dyn FnMut() -> f32| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let h = cfg.hidden_size;
        let kv = cfg.n_head_kv * cfg.head_dim();
        let inter = 4 * h;
        let mut next_box: Box<dyn FnMut() -> f32> = Box::new(next);
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *next_box);
        let layers: Vec<FalconLayerWeights> = (0..cfg.num_hidden_layers).map(|_| FalconLayerWeights {
            input_ln_gain: Arc::from(vec![1.0_f32; h]),
            input_ln_bias: Arc::from(vec![0.0_f32; h]),
            post_attn_ln: if cfg.parallel_attn {
                None
            } else {
                Some((Arc::from(vec![1.0_f32; h]), Arc::from(vec![0.0_f32; h])))
            },
            attn_q: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            attn_q_bias: if cfg.bias { Some(vec_of(h, &mut *next_box)) } else { None },
            attn_k: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_k_bias: if cfg.bias { Some(vec_of(kv, &mut *next_box)) } else { None },
            attn_v: WeightStorage::F32(vec_of(h * kv, &mut *next_box)),
            attn_v_bias: if cfg.bias { Some(vec_of(kv, &mut *next_box)) } else { None },
            attn_dense: WeightStorage::F32(vec_of(h * h, &mut *next_box)),
            attn_dense_bias: if cfg.bias { Some(vec_of(h, &mut *next_box)) } else { None },
            mlp_up: WeightStorage::F32(vec_of(h * inter, &mut *next_box)),
            mlp_up_bias: if cfg.bias { Some(vec_of(inter, &mut *next_box)) } else { None },
            mlp_down: WeightStorage::F32(vec_of(inter * h, &mut *next_box)),
            mlp_down_bias: if cfg.bias { Some(vec_of(h, &mut *next_box)) } else { None },
        }).collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *next_box));
        FalconWeights { token_embedding, layers, final_ln_gain, final_ln_bias, output }
    }

    #[test]
    fn forward_shape_and_finite_parallel_attn() {
        let cfg = FalconConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            n_head_kv: 1,
            layer_norm_epsilon: 1e-5,
            max_position_embeddings: 64,
            parallel_attn: true,
            bias: false,
            rope_theta: 10_000.0,
        };
        let model = FalconModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }
    }

    /// Serial mode (parallel_attn = false): exercises the
    /// post-attention LayerNorm path.
    #[test]
    fn forward_shape_and_finite_serial_attn() {
        let cfg = FalconConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            n_head_kv: 2,
            layer_norm_epsilon: 1e-5,
            max_position_embeddings: 32,
            parallel_attn: false,
            bias: true,
            rope_theta: 10_000.0,
        };
        let model = FalconModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![3, 1, 4, 1, 5];
        let logits = model.forward(&tokens, 0).unwrap();
        assert_eq!(logits.shape().dims(), &[1, tokens.len(), cfg.vocab_size]);
        let out = logits.realize_f32();
        for &v in out.iter() {
            assert!(v.is_finite());
        }
    }

    /// Parallel-attn output must differ from serial-attn output for
    /// the same weights — they're different computations.
    #[test]
    fn parallel_and_serial_attn_diverge() {
        let cfg_p = FalconConfig {
            vocab_size: 16,
            hidden_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            n_head_kv: 2,
            layer_norm_epsilon: 1e-5,
            max_position_embeddings: 16,
            parallel_attn: true,
            bias: false,
            rope_theta: 10_000.0,
        };
        let weights = tiny_weights(&cfg_p);
        let mut cfg_s = cfg_p.clone();
        cfg_s.parallel_attn = false;
        // For serial mode the tiny_weights doesn't add post_attn_ln
        // (it checks `parallel_attn` at the moment of construction);
        // build a serial-shaped weight set instead.
        let weights_s = {
            let mut w = weights.clone();
            for l in &mut w.layers {
                l.post_attn_ln = Some((
                    Arc::from(vec![1.0_f32; cfg_p.hidden_size]),
                    Arc::from(vec![0.0_f32; cfg_p.hidden_size]),
                ));
            }
            w
        };
        let out_p = FalconModel { config: cfg_p, weights }
            .forward(&[1, 2, 3, 4], 0).unwrap().realize_f32();
        let out_s = FalconModel { config: cfg_s, weights: weights_s }
            .forward(&[1, 2, 3, 4], 0).unwrap().realize_f32();
        let any_diff = out_p.iter().zip(out_s.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-5);
        assert!(any_diff, "parallel vs serial attention must diverge");
    }

    /// `forward_hidden` returns post-final-LN hidden states
    /// `(1, seq, hidden_size)` without the lm_head matmul.
    #[test]
    fn forward_hidden_shape_and_finite() {
        let cfg = FalconConfig {
            vocab_size: 32,
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            n_head_kv: 1,
            layer_norm_epsilon: 1e-5,
            max_position_embeddings: 64,
            parallel_attn: true,
            bias: false,
            rope_theta: 10_000.0,
        };
        let model = FalconModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3, 4];
        let hidden = model.forward_hidden(&tokens, 0).unwrap();
        assert_eq!(hidden.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        for &v in &hidden.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    fn forward_embeds_test_cfg() -> FalconConfig {
        FalconConfig {
            vocab_size: 32, hidden_size: 16,
            num_hidden_layers: 2, num_attention_heads: 4, n_head_kv: 1,
            layer_norm_epsilon: 1e-5, max_position_embeddings: 64,
            parallel_attn: true, bias: false, rope_theta: 10_000.0,
        }
    }

    #[test]
    fn forward_embeds_matches_forward_after_token_lookup() {
        let cfg = forward_embeds_test_cfg();
        let model = FalconModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![1, 2, 3];
        let logits_ref = model.forward(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let logits_via_embeds = model.forward_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = logits_ref.iter().zip(logits_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "Falcon forward vs forward_embeds must agree (max diff {max_diff})");
    }

    #[test]
    fn forward_embeds_rejects_bad_shape() {
        let cfg = forward_embeds_test_cfg();
        let model = FalconModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let bad = LazyTensor::from_f32(
            vec![0.0_f32; 3 * (cfg.hidden_size + 1)],
            Shape::from_dims(&[1, 3, cfg.hidden_size + 1]), &Device::cpu(),
        );
        assert!(model.forward_embeds(&bad, 0).is_err());
    }

    #[test]
    fn forward_hidden_embeds_matches_forward_hidden() {
        let cfg = forward_embeds_test_cfg();
        let model = FalconModel { config: cfg.clone(), weights: tiny_weights(&cfg) };
        let tokens: Vec<u32> = vec![5, 7];
        let h_ref = model.forward_hidden(&tokens, 0).unwrap().realize_f32();
        let anchor = LazyTensor::from_f32(
            vec![0.0_f32], Shape::from_dims(&[1]), &Device::cpu(),
        );
        let embeds = model.embed_tokens_anchored(&anchor, &tokens).unwrap();
        let h_via_embeds = model.forward_hidden_embeds(&embeds, 0).unwrap().realize_f32();
        let max_diff = h_ref.iter().zip(h_via_embeds.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0_f32, f32::max);
        assert!(max_diff < 1e-5,
            "Falcon forward_hidden vs forward_hidden_embeds must agree (max diff {max_diff})");
    }
}
