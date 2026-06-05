//! OpenCLIP text transformer — lazy port.
//!
//! Pre-LN transformer that encodes a text token sequence into
//! per-token hidden states `(1, seq, embed_dim)`. Used as the
//! text tower of OpenCLIP / MobileCLIP / MetaCLIP variants.
//!
//! Layer structure (Pre-LN):
//!   ln1 → self-attention → +residual
//!   ln2 → MLP (fc1 → GELU → fc2) → +residual
//!
//! Pooling: CLIP's standard pooling takes the hidden state at the
//! position of the EOT (end-of-text) token. EOT is, by
//! tokenization convention, the position with the maximum token
//! id. v1 exposes:
//!   - `forward(input_ids)` — full sequence hidden states.
//!   - `forward_pooled(input_ids, eot_pos)` — caller supplies the
//!     EOT position. The lazy graph doesn't compute argmax over
//!     U32 indices; the caller is expected to find EOT on the
//!     host side and pass it explicitly.
//!
//! v1 scope:
//!   - F32, batch == 1.
//!   - No causal mask (matches the eager port's current
//!     attention forward — see `Attention::forward` in
//!     `fuel-transformers/src/models/multimodal/openclip/text_model.rs`).
//!   - Pre-LN structure.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct OpenClipTextConfig {
    pub vocab_size: usize,
    pub embed_dim: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
}

impl OpenClipTextConfig {
    /// ViT-B/32 text encoder preset.
    pub fn vit_base_patch32() -> Self {
        Self {
            vocab_size: 49408,
            embed_dim: 512,
            intermediate_size: 2048,
            max_position_embeddings: 77,
            num_hidden_layers: 12,
            num_attention_heads: 8,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_attention_heads
    }
}

#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct OpenClipAttentionWeights {
    /// `[embed_dim, embed_dim]` each. PyTorch's `nn.MultiheadAttention`
    /// stores Q/K/V as a single `in_proj_weight` of shape
    /// `[3·embed_dim, embed_dim]`; the loader chunks it into Q/K/V.
    pub q_proj: WeightStorage,
    pub q_proj_bias: Arc<[f32]>,
    pub k_proj: WeightStorage,
    pub k_proj_bias: Arc<[f32]>,
    pub v_proj: WeightStorage,
    pub v_proj_bias: Arc<[f32]>,
    pub out_proj: WeightStorage,
    pub out_proj_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct MlpWeights {
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct OpenClipEncoderLayerWeights {
    pub ln1: LayerNormWeights,
    pub attn: OpenClipAttentionWeights,
    pub ln2: LayerNormWeights,
    pub mlp: MlpWeights,
}

#[derive(Debug, Clone)]
pub struct OpenClipTextWeights {
    /// `[vocab_size, embed_dim]`.
    pub token_embedding: Arc<[f32]>,
    /// `[max_position_embeddings, embed_dim]`.
    pub position_embedding: Arc<[f32]>,
    pub layers: Vec<OpenClipEncoderLayerWeights>,
    pub final_ln: LayerNormWeights,
}

#[derive(Debug, Clone)]
pub struct OpenClipTextModel {
    pub config: OpenClipTextConfig,
    pub weights: OpenClipTextWeights,
}

impl OpenClipTextModel {
    /// Run the text transformer and return per-token hidden states
    /// `(1, seq, embed_dim)`.
    pub fn forward(&self, input_ids: &[u32]) -> Result<LazyTensor> {
        self.run_backbone(input_ids)
    }

    /// Run the text transformer and return the EOT-pooled embedding
    /// `(1, embed_dim)`. `eot_pos` is the position of the
    /// end-of-text token in `input_ids` (CLIP's standard pooling
    /// picks the argmax-token-id position; v1 takes it as an
    /// explicit parameter so the lazy graph stays shape-static).
    pub fn forward_pooled(
        &self, input_ids: &[u32], eot_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        assert!(eot_pos < input_ids.len(),
            "eot_pos {eot_pos} must be < seq_len {}", input_ids.len());
        let hidden = self.run_backbone(input_ids)?;
        // (1, seq, embed_dim) → narrow seq to single position → squeeze → (1, embed_dim).
        let pooled = hidden
            .narrow(1_usize, eot_pos, 1)?
            .reshape(Shape::from_dims(&[1, cfg.embed_dim]))?;
        Ok(pooled)
    }

    fn run_backbone(&self, input_ids: &[u32]) -> Result<LazyTensor> {
        let cfg = &self.config;
        let w = &self.weights;
        let seq = input_ids.len();
        assert!(seq > 0, "input_ids must be non-empty");
        assert!(
            seq <= cfg.max_position_embeddings,
            "seq_len {seq} > max_position_embeddings {}", cfg.max_position_embeddings,
        );

        // Token embedding lookup.
        let token_table = LazyTensor::from_f32(
            Arc::clone(&w.token_embedding),
            Shape::from_dims(&[cfg.vocab_size, cfg.embed_dim]),
            &crate::Device::cpu(),
        );
        let ids = token_table.const_u32_like(
            input_ids.to_vec(), Shape::from_dims(&[seq]),
        );
        let tok = token_table
            .index_select(0_usize, &ids)?
            .reshape(Shape::from_dims(&[1, seq, cfg.embed_dim]))?;

        // Positional embedding: narrow the table to the first `seq` rows.
        let pos_table = token_table.const_f32_like(
            Arc::clone(&w.position_embedding),
            Shape::from_dims(&[cfg.max_position_embeddings, cfg.embed_dim]),
        );
        let pos = pos_table
            .narrow(0_usize, 0, seq)?
            .reshape(Shape::from_dims(&[1, seq, cfg.embed_dim]))?;

        let mut x = tok.add(&pos)?;
        for layer in &w.layers {
            x = apply_layer(&x, layer, cfg, &token_table)?;
        }
        Ok(apply_layer_norm(&x, &w.final_ln, cfg.embed_dim, 1e-5)?)
    }
}

fn apply_layer(
    x: &LazyTensor,
    w: &OpenClipEncoderLayerWeights,
    cfg: &OpenClipTextConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let residual = x.clone();
    let normed = apply_layer_norm(x, &w.ln1, cfg.embed_dim, 1e-5)?;
    let attn_out = apply_attention(&normed, &w.attn, cfg, anchor)?;
    let x = residual.add(&attn_out)?;

    let residual = x.clone();
    let normed = apply_layer_norm(&x, &w.ln2, cfg.embed_dim, 1e-5)?;
    let mlp_out = apply_mlp(&normed, &w.mlp, cfg.embed_dim, cfg.intermediate_size, anchor)?;
    residual.add(&mlp_out)
}

fn apply_attention(
    x: &LazyTensor,
    w: &OpenClipAttentionWeights,
    cfg: &OpenClipTextConfig,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let seq = dims[1];
    let embed = cfg.embed_dim;
    let n_heads = cfg.num_attention_heads;
    let head_dim = cfg.head_dim();
    let scale = 1.0_f64 / (head_dim as f64).sqrt();

    let q = w.q_proj.apply_linear_with_bias(x, embed, embed, std::sync::Arc::clone(&w.q_proj_bias))?;
    let k = w.k_proj.apply_linear_with_bias(x, embed, embed, std::sync::Arc::clone(&w.k_proj_bias))?;
    let v = w.v_proj.apply_linear_with_bias(x, embed, embed, std::sync::Arc::clone(&w.v_proj_bias))?;

    let q = q.mul_scalar(scale);

    // (B, seq, embed) → (B, n_heads, seq, head_dim)
    let _ = (b, seq, embed);
    let q = q.split_heads(n_heads, head_dim)?;
    let k = k.split_heads(n_heads, head_dim)?;
    let v = v.split_heads(n_heads, head_dim)?;

    let kt = k.permute([0, 1, 3, 2_usize])?;
    let scores = q.matmul(&kt)?;
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?.merge_heads()?;
    w.out_proj.apply_linear_with_bias(&ctx, embed, embed, std::sync::Arc::clone(&w.out_proj_bias))
}

fn apply_mlp(
    x: &LazyTensor,
    m: &MlpWeights,
    in_dim: usize,
    hidden_dim: usize,
    anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let h1 = m.fc1.apply_linear_with_bias(x, in_dim, hidden_dim, std::sync::Arc::clone(&m.fc1_bias))?;
    let h1 = h1.gelu_erf();
    m.fc2.apply_linear_with_bias(&h1, hidden_dim, in_dim, std::sync::Arc::clone(&m.fc2_bias))
}

fn apply_layer_norm(
    x: &LazyTensor,
    ln: &LayerNormWeights,
    hidden: usize,
    eps: f64,
) -> Result<LazyTensor> {
    let _ = hidden;
    x.layer_norm_affine(Arc::clone(&ln.gain), Arc::clone(&ln.bias), eps)
}


// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
    fn ln_w(c: usize) -> LayerNormWeights {
        LayerNormWeights {
            gain: Arc::from(vec![1.0_f32; c]),
            bias: Arc::from(vec![0.0_f32; c]),
        }
    }

    fn tiny_config() -> OpenClipTextConfig {
        OpenClipTextConfig {
            vocab_size: 32,
            embed_dim: 8,
            intermediate_size: 16,
            max_position_embeddings: 16,
            num_hidden_layers: 2,
            num_attention_heads: 2,
        }
    }

    fn tiny_weights(cfg: &OpenClipTextConfig) -> OpenClipTextWeights {
        let mut nb = rng_seed(2026);
        let e = cfg.embed_dim;
        let layers: Vec<OpenClipEncoderLayerWeights> = (0..cfg.num_hidden_layers).map(|_| {
            OpenClipEncoderLayerWeights {
                ln1: ln_w(e),
                attn: OpenClipAttentionWeights {
                    q_proj: ws(e * e, &mut nb), q_proj_bias: vec_of(e, &mut nb),
                    k_proj: ws(e * e, &mut nb), k_proj_bias: vec_of(e, &mut nb),
                    v_proj: ws(e * e, &mut nb), v_proj_bias: vec_of(e, &mut nb),
                    out_proj: ws(e * e, &mut nb), out_proj_bias: vec_of(e, &mut nb),
                },
                ln2: ln_w(e),
                mlp: MlpWeights {
                    fc1: ws(e * cfg.intermediate_size, &mut nb),
                    fc1_bias: vec_of(cfg.intermediate_size, &mut nb),
                    fc2: ws(cfg.intermediate_size * e, &mut nb),
                    fc2_bias: vec_of(e, &mut nb),
                },
            }
        }).collect();
        OpenClipTextWeights {
            token_embedding: vec_of(cfg.vocab_size * e, &mut nb),
            position_embedding: vec_of(cfg.max_position_embeddings * e, &mut nb),
            layers,
            final_ln: ln_w(e),
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = OpenClipTextModel { config: cfg.clone(), weights };
        let ids = vec![1_u32, 5, 10, 31];
        let out = model.forward(&ids).unwrap();
        assert_eq!(out.shape().dims(), &[1, ids.len(), cfg.embed_dim]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite hidden: {v}");
        }
    }

    #[test]
    fn forward_pooled_picks_eot_position() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = OpenClipTextModel { config: cfg.clone(), weights };
        // 4 tokens; EOT at position 2.
        let ids = vec![1_u32, 5, 31, 0];
        let full = model.forward(&ids).unwrap().realize_f32();
        let pooled = model.forward_pooled(&ids, 2).unwrap();
        assert_eq!(pooled.shape().dims(), &[1, cfg.embed_dim]);
        let pooled_data = pooled.realize_f32();
        for d in 0..cfg.embed_dim {
            let expected = full[2 * cfg.embed_dim + d];
            assert!((pooled_data[d] - expected).abs() < 1e-5,
                "pooled[{d}] = {} != full[2, {d}] = {expected}", pooled_data[d]);
        }
    }

    #[test]
    fn forward_responds_to_input() {
        let cfg = tiny_config();
        let weights = tiny_weights(&cfg);
        let model = OpenClipTextModel { config: cfg, weights };
        let a = model.forward(&[1_u32, 5, 10, 31]).unwrap().realize_f32();
        let b = model.forward(&[3_u32, 7, 20, 31]).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "text encoder must respond to token changes, max_diff = {max_diff}");
    }

    #[test]
    fn preset_constructs() {
        let cfg = OpenClipTextConfig::vit_base_patch32();
        assert_eq!(cfg.vocab_size, 49408);
        assert_eq!(cfg.embed_dim, 512);
        assert_eq!(cfg.max_position_embeddings, 77);
        assert_eq!(cfg.num_hidden_layers, 12);
    }
}
