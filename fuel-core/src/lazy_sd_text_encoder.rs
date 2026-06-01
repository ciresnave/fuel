//! Stable Diffusion 1.5 CLIP text encoder ported to the lazy-graph API.
//!
//! First component of Phase 6a anchor #6 (SD 1.5). SD 1.5's text
//! conditioning comes from OpenAI's CLIP-ViT-L/14 text encoder: a
//! 12-layer causal transformer that maps a 77-token prompt to a
//! `[1, 77, 768]` hidden state. The UNet cross-attends into that
//! tensor at every down/mid/up block during diffusion.
//!
//! Architecturally this is BERT-shaped with three twists:
//! - **Causal self-attention**, not bidirectional. Same mask we use in
//!   the Whisper decoder.
//! - **Pre-LayerNorm** block order: `x + sublayer(LN(x))`. Matches
//!   Whisper; opposite of BERT's post-LN.
//! - **QuickGELU activation**: `x * sigmoid(1.702 * x)`. A faster
//!   approximation to GELU that CLIP training baked in — swapping it
//!   for the standard GELU produces visibly different outputs at the
//!   same weights, so we match it exactly.
//!
//! SD 1.5 + 2.x + most OpenCLIP derivatives all use this same text
//! encoder shape (dim=768, 12 layers, 12 heads, max_pos=77), so once
//! this lands the UNet's cross-attention K/V source is a solved
//! problem.
//!
//! # Example
//!
//! ```no_run
//! use fuel_core::lazy_sd_text_encoder::{SdTextEncoder, SdTextTokenizer};
//! let model = SdTextEncoder::from_hub("stable-diffusion-v1-5/stable-diffusion-v1-5")?;
//! let tokenizer = SdTextTokenizer::from_hub("stable-diffusion-v1-5/stable-diffusion-v1-5")?;
//! let tokens = tokenizer.encode_padded("a photo of a cat")?;
//! let hidden = model.forward(&tokens);
//! let flat = hidden.realize_f32();
//! assert_eq!(flat.len(), 77 * 768);
//! # Ok::<(), fuel_core::Error>(())
//! ```

use crate::lazy::LazyTensor;
use fuel_core_types::Shape;
use serde::Deserialize;
use std::sync::Arc;

// ---- Config ----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct ClipTextConfig {
    pub vocab_size:              usize,
    pub hidden_size:             usize,
    pub num_hidden_layers:       usize,
    pub num_attention_heads:     usize,
    pub intermediate_size:       usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps:          f64,
    pub bos_token_id:            u32,
    pub eos_token_id:            u32,
    pub pad_token_id:            u32,
}

fn default_layer_norm_eps() -> f64 {
    1e-5
}

impl ClipTextConfig {
    /// SD 1.5 / SD 2.x text encoder shape. Same as OpenAI
    /// `clip-vit-large-patch14`.
    pub fn sd_v1() -> Self {
        Self {
            vocab_size:              49408,
            hidden_size:              768,
            num_hidden_layers:         12,
            num_attention_heads:       12,
            intermediate_size:       3072,
            max_position_embeddings:   77,
            layer_norm_eps:          1e-5,
            bos_token_id:               0,
            eos_token_id:               2,
            pad_token_id:               1,
        }
    }

    pub fn from_hf_json_str(s: &str) -> crate::Result<Self> {
        serde_json::from_str::<Self>(s)
            .map_err(|e| crate::Error::Msg(format!("parsing clip text config: {e}")).bt())
    }

    pub fn head_dim(&self) -> usize {
        assert_eq!(self.hidden_size % self.num_attention_heads, 0);
        self.hidden_size / self.num_attention_heads
    }
}

// ---- Weight storage --------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ClipLayerWeights {
    pub ln1_g: Arc<[f32]>,
    pub ln1_b: Arc<[f32]>,
    pub q_w: Arc<[f32]>,
    pub q_b: Arc<[f32]>,
    pub k_w: Arc<[f32]>,
    pub k_b: Arc<[f32]>,
    pub v_w: Arc<[f32]>,
    pub v_b: Arc<[f32]>,
    pub out_w: Arc<[f32]>,
    pub out_b: Arc<[f32]>,
    pub ln2_g: Arc<[f32]>,
    pub ln2_b: Arc<[f32]>,
    pub fc1_w: Arc<[f32]>,
    pub fc1_b: Arc<[f32]>,
    pub fc2_w: Arc<[f32]>,
    pub fc2_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ClipTextWeights {
    /// Shape `[vocab_size, hidden_size]`.
    pub token_embedding:    Arc<[f32]>,
    /// Shape `[max_position_embeddings, hidden_size]`.
    pub position_embedding: Arc<[f32]>,
    pub layers:             Vec<ClipLayerWeights>,
    pub final_ln_g:         Arc<[f32]>,
    pub final_ln_b:         Arc<[f32]>,
}

// ---- Model -----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SdTextEncoder {
    pub config:  ClipTextConfig,
    pub weights: ClipTextWeights,
}

impl SdTextEncoder {
    /// Run the forward pass on a batch of 1 × `max_position_embeddings`
    /// token IDs. The caller is responsible for padding to exactly
    /// `max_position_embeddings` — SD's UNet expects a fixed `[1, 77,
    /// 768]` conditioning, and producing a shorter sequence here is a
    /// common source of cryptic mismatches downstream.
    ///
    /// Returns `[1, seq, hidden_size]` hidden states.
    pub fn forward(&self, tokens: &[u32]) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        assert_eq!(
            tokens.len(), cfg.max_position_embeddings,
            "SdTextEncoder::forward: expected exactly {} tokens, got {}",
            cfg.max_position_embeddings, tokens.len(),
        );
        let seq = tokens.len();
        let h = cfg.hidden_size;

        // Anchor the graph on the token embedding matrix.
        let token_emb = LazyTensor::from_f32(
            self.weights.token_embedding.clone(),
            Shape::from_dims(&[cfg.vocab_size, h]),
            &crate::Device::cpu(),
        );
        let input_ids = token_emb.const_u32_like(tokens.to_vec(), Shape::from_dims(&[seq]));
        let pos_ids: Vec<u32> = (0..seq as u32).collect();
        let position_ids = token_emb.const_u32_like(pos_ids, Shape::from_dims(&[seq]));
        let pos_emb = token_emb.const_f32_like(
            self.weights.position_embedding.clone(),
            Shape::from_dims(&[cfg.max_position_embeddings, h]),
        );

        let w = token_emb.index_select(0, &input_ids);
        let p = pos_emb.index_select(0, &position_ids);
        let embeds = w.add(&p).reshape(Shape::from_dims(&[1, seq, h]));

        let mut x = embeds;
        for lw in &self.weights.layers {
            x = encoder_layer(&x, lw, cfg, seq);
        }
        Ok(layer_norm_affine(&x, &self.weights.final_ln_g, &self.weights.final_ln_b, cfg.layer_norm_eps, h, seq))
    }
}

/// One CLIP transformer block: causal self-attention + quick-GELU FFN,
/// each wrapped as `x + sublayer(LN(x))` (pre-norm).
fn encoder_layer(
    x: &LazyTensor,
    lw: &ClipLayerWeights,
    cfg: &ClipTextConfig,
    seq: usize,
) -> LazyTensor {
    let h = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let d_head = cfg.head_dim();

    // --- self-attention -------------
    let x_ln = layer_norm_affine(x, &lw.ln1_g, &lw.ln1_b, cfg.layer_norm_eps, h, seq);
    let q = linear(&x_ln, &lw.q_w, Some(&lw.q_b), h, h, seq);
    let k = linear(&x_ln, &lw.k_w, Some(&lw.k_b), h, h, seq);
    let v = linear(&x_ln, &lw.v_w, Some(&lw.v_b), h, h, seq);

    let q = q
        .reshape(Shape::from_dims(&[1, seq, n_heads, d_head]))
        .permute(&[0, 2, 1, 3]);
    let k = k
        .reshape(Shape::from_dims(&[1, seq, n_heads, d_head]))
        .permute(&[0, 2, 1, 3]);
    let v = v
        .reshape(Shape::from_dims(&[1, seq, n_heads, d_head]))
        .permute(&[0, 2, 1, 3]);
    let k_t = k.permute(&[0, 1, 3, 2]);
    let scale = 1.0_f64 / (d_head as f64).sqrt();
    let mut scores = q.matmul(&k_t).mul_scalar(scale);
    // Causal mask: -inf above the diagonal.
    let mut mask = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        for j in 0..seq {
            if j > i { mask[i * seq + j] = f32::NEG_INFINITY; }
        }
    }
    let mask_t = scores
        .const_f32_like(mask, Shape::from_dims(&[seq, seq]))
        .reshape(Shape::from_dims(&[1, 1, seq, seq]))
        .broadcast_to(Shape::from_dims(&[1, n_heads, seq, seq]));
    scores = scores.add(&mask_t);
    let probs = scores.softmax_last_dim();
    let ctx = probs
        .matmul(&v)
        .permute(&[0, 2, 1, 3])
        .reshape(Shape::from_dims(&[1, seq, h]));
    let attn_out = linear(&ctx, &lw.out_w, Some(&lw.out_b), h, h, seq);
    let x = x.add(&attn_out);

    // --- MLP with QuickGELU ---------
    let x_ln = layer_norm_affine(&x, &lw.ln2_g, &lw.ln2_b, cfg.layer_norm_eps, h, seq);
    let h_ff = cfg.intermediate_size;
    let mid = linear(&x_ln, &lw.fc1_w, Some(&lw.fc1_b), h, h_ff, seq);
    let mid = quick_gelu(&mid);
    let ffn = linear(&mid, &lw.fc2_w, Some(&lw.fc2_b), h_ff, h, seq);
    x.add(&ffn)
}

/// QuickGELU: `x * sigmoid(1.702 * x)`. CLIP's approximation to GELU;
/// the 1.702 constant is part of CLIP's trained baseline and swapping
/// in the exact GELU at inference produces visibly different outputs.
fn quick_gelu(x: &LazyTensor) -> LazyTensor {
    // sigmoid(y) = 1 / (1 + exp(-y))
    // sigmoid(1.702 * x) = 1 / (1 + exp(-1.702 * x))
    let scaled = x.mul_scalar(1.702);
    let neg = scaled.neg();
    let ex = neg.exp();
    let one_plus = ex.add_scalar(1.0);
    // Reciprocal via 1/one_plus. LazyTensor's div goes through the
    // Div op; use `div` with the numerator 1-tensor.
    let ones = x
        .const_f32_like(vec![1.0_f32; 1], Shape::from_dims(&[1]))
        .broadcast_to(one_plus.shape());
    let sig = ones.div(&one_plus);
    x.mul(&sig)
}

/// `y = LayerNorm(x) * gamma + beta`. Same pattern as BERT / Whisper.
fn layer_norm_affine(
    x: &LazyTensor,
    gamma: &Arc<[f32]>,
    beta: &Arc<[f32]>,
    eps: f64,
    hidden: usize,
    seq: usize,
) -> LazyTensor {
    let normed = x.layer_norm_last_dim(eps);
    let g = x
        .const_f32_like(gamma.clone(), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden]))
        .broadcast_to(Shape::from_dims(&[1, seq, hidden]));
    let b = x
        .const_f32_like(beta.clone(), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden]))
        .broadcast_to(Shape::from_dims(&[1, seq, hidden]));
    normed.mul(&g).add(&b)
}

fn linear(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: Option<&Arc<[f32]>>,
    in_f: usize,
    out_f: usize,
    seq: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[in_f, out_f]));
    let proj = x.matmul(&w_t);
    match b {
        Some(b) => {
            let bias = x
                .const_f32_like(b.clone(), Shape::from_dims(&[out_f]))
                .reshape(Shape::from_dims(&[1, 1, out_f]))
                .broadcast_to(Shape::from_dims(&[1, seq, out_f]));
            proj.add(&bias)
        }
        None => proj,
    }
}

// ---- Safetensors loader ----------------------------------------------------

impl ClipTextWeights {
    /// Load all CLIP text-encoder weights from a mmapped safetensors
    /// file using the HF naming convention
    /// (`text_model.embeddings.{token_embedding,position_embedding}.
    /// weight`, `text_model.encoder.layers.{i}.…`, `text_model.
    /// final_layer_norm.{weight,bias}`).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &ClipTextConfig,
    ) -> crate::Result<Self> {
        let h = cfg.hidden_size;
        let token_embedding =
            load_f32(st, "text_model.embeddings.token_embedding.weight")?;
        let position_embedding =
            load_f32(st, "text_model.embeddings.position_embedding.weight")?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("text_model.encoder.layers.{i}");
            let ln1_g = load_f32(st, &format!("{p}.layer_norm1.weight"))?;
            let ln1_b = load_f32(st, &format!("{p}.layer_norm1.bias"))?;
            let q_w = load_transposed(st, &format!("{p}.self_attn.q_proj.weight"), h, h)?;
            let q_b = load_f32(st, &format!("{p}.self_attn.q_proj.bias"))?;
            let k_w = load_transposed(st, &format!("{p}.self_attn.k_proj.weight"), h, h)?;
            let k_b = load_f32(st, &format!("{p}.self_attn.k_proj.bias"))?;
            let v_w = load_transposed(st, &format!("{p}.self_attn.v_proj.weight"), h, h)?;
            let v_b = load_f32(st, &format!("{p}.self_attn.v_proj.bias"))?;
            let out_w = load_transposed(st, &format!("{p}.self_attn.out_proj.weight"), h, h)?;
            let out_b = load_f32(st, &format!("{p}.self_attn.out_proj.bias"))?;
            let ln2_g = load_f32(st, &format!("{p}.layer_norm2.weight"))?;
            let ln2_b = load_f32(st, &format!("{p}.layer_norm2.bias"))?;
            let fc1_w = load_transposed(st, &format!("{p}.mlp.fc1.weight"), cfg.intermediate_size, h)?;
            let fc1_b = load_f32(st, &format!("{p}.mlp.fc1.bias"))?;
            let fc2_w = load_transposed(st, &format!("{p}.mlp.fc2.weight"), h, cfg.intermediate_size)?;
            let fc2_b = load_f32(st, &format!("{p}.mlp.fc2.bias"))?;
            layers.push(ClipLayerWeights {
                ln1_g: Arc::from(ln1_g), ln1_b: Arc::from(ln1_b),
                q_w: Arc::from(q_w), q_b: Arc::from(q_b),
                k_w: Arc::from(k_w), k_b: Arc::from(k_b),
                v_w: Arc::from(v_w), v_b: Arc::from(v_b),
                out_w: Arc::from(out_w), out_b: Arc::from(out_b),
                ln2_g: Arc::from(ln2_g), ln2_b: Arc::from(ln2_b),
                fc1_w: Arc::from(fc1_w), fc1_b: Arc::from(fc1_b),
                fc2_w: Arc::from(fc2_w), fc2_b: Arc::from(fc2_b),
            });
        }
        let final_ln_g = load_f32(st, "text_model.final_layer_norm.weight")?;
        let final_ln_b = load_f32(st, "text_model.final_layer_norm.bias")?;

        Ok(Self {
            token_embedding:    Arc::from(token_embedding),
            position_embedding: Arc::from(position_embedding),
            layers,
            final_ln_g: Arc::from(final_ln_g),
            final_ln_b: Arc::from(final_ln_b),
        })
    }
}

fn load_f32(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
) -> crate::Result<Vec<f32>> {
    use safetensors::Dtype;
    let view = st
        .get(name)
        .map_err(|e| crate::Error::Msg(format!("clip load_f32 {name:?}: {e}")).bt())?;
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::bf16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        other => crate::bail!("clip load_f32: unsupported dtype {other:?} for {name:?}"),
    }
}

fn load_transposed(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<f32>> {
    let flat = load_f32(st, name)?;
    if flat.len() != out_features * in_features {
        crate::bail!(
            "clip load_transposed: {name:?} has {} elements, expected {}",
            flat.len(), out_features * in_features,
        );
    }
    let mut out = vec![0.0_f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            out[j * out_features + i] = flat[i * in_features + j];
        }
    }
    Ok(out)
}

// ---- HuggingFace Hub integration -------------------------------------------

impl SdTextEncoder {
    /// Downloads an SD 1.5-style diffusers repo's `text_encoder/`
    /// subfolder + loads the weights. Defaults to the SD 1.5 text
    /// encoder config; override via `from_hub_with_config` for SD 2.x
    /// (different `hidden_size` + `num_hidden_layers`).
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        Self::from_hub_with_config(repo_id, ClipTextConfig::sd_v1())
    }

    pub fn from_hub_with_config(
        repo_id: &str,
        config: ClipTextConfig,
    ) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        // Diffusers repos ship one safetensors per subcomponent; SD 1.5
        // puts the text encoder at text_encoder/model.safetensors.
        let path = repo
            .get("text_encoder/model.safetensors")
            .map_err(|e| crate::Error::Msg(format!("hf-hub text_encoder/model.safetensors: {e}")))?;
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path) }?;
        let weights = ClipTextWeights::load_from_mmapped(&st, &config)?;
        Ok(Self { config, weights })
    }
}

// ---- Tokenizer -------------------------------------------------------------

/// CLIP text tokenizer wrapper. Adds the `encode_padded` helper that
/// pads to exactly `max_position_embeddings` — the shape SD's UNet
/// expects for cross-attention K/V.
pub struct SdTextTokenizer {
    inner: tokenizers::Tokenizer,
    pad_id: u32,
    eos_id: u32,
    max_len: usize,
}

impl SdTextTokenizer {
    pub fn from_file<P: AsRef<std::path::Path>>(
        path: P,
        cfg: &ClipTextConfig,
    ) -> crate::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| crate::Error::Msg(format!("clip tokenizer: {e}")))?;
        Ok(Self {
            inner,
            pad_id: cfg.pad_token_id,
            eos_id: cfg.eos_token_id,
            max_len: cfg.max_position_embeddings,
        })
    }

    /// Loads the CLIP tokenizer from HuggingFace. Because the diffusers
    /// SD 1.5 repo only ships the legacy `vocab.json` + `merges.txt`
    /// (no consolidated `tokenizer.json`), we pull `tokenizer.json`
    /// from a tokenizer-compatible CLIP-L repo that does publish the
    /// modern format. `laion/CLIP-ViT-L-14-laion2B-s32B-b82K` uses the
    /// same OpenAI CLIP vocabulary as SD 1.5 / 2.x and ships
    /// `tokenizer.json`.
    pub fn from_hub(_repo_id: &str) -> crate::Result<Self> {
        Self::from_hub_with_config(_repo_id, &ClipTextConfig::sd_v1())
    }

    pub fn from_hub_with_config(
        _repo_id: &str,
        cfg: &ClipTextConfig,
    ) -> crate::Result<Self> {
        // The _repo_id parameter is reserved for future support of
        // diffusers repos that do ship tokenizer/tokenizer.json. The
        // laion mirror is the default fallback — it's the canonical
        // CLIP-L tokenizer and interchangeable with the one SD was
        // trained against.
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model("laion/CLIP-ViT-L-14-laion2B-s32B-b82K".to_string());
        let path = repo
            .get("tokenizer.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub clip tokenizer: {e}")))?;
        Self::from_file(path, cfg)
    }

    /// Encode a prompt and pad/truncate to `max_position_embeddings`.
    /// Padding uses `pad_token_id`; sequences longer than the max are
    /// truncated and closed with EOS.
    pub fn encode_padded(&self, text: &str) -> crate::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, true)
            .map_err(|e| crate::Error::Msg(format!("clip encode: {e}")))?;
        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        if ids.len() >= self.max_len {
            ids.truncate(self.max_len - 1);
            ids.push(self.eos_id);
        } else {
            while ids.len() < self.max_len {
                ids.push(self.pad_id);
            }
        }
        Ok(ids)
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(v: Vec<f32>) -> Arc<[f32]> { Arc::from(v) }

    #[test]
    fn sd_v1_config_shape() {
        let cfg = ClipTextConfig::sd_v1();
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.head_dim(), 64);
        assert_eq!(cfg.max_position_embeddings, 77);
    }

    #[test]
    fn forward_shape_and_finite() {
        // Tiny synthetic config for a shape smoke test.
        let cfg = ClipTextConfig {
            vocab_size: 100, hidden_size: 16,
            num_hidden_layers: 2, num_attention_heads: 4,
            intermediate_size: 32, max_position_embeddings: 8,
            layer_norm_eps: 1e-5,
            bos_token_id: 0, eos_token_id: 2, pad_token_id: 1,
        };
        let h = cfg.hidden_size;
        let z = |n: usize| arc(vec![0.0_f32; n]);
        let o = |n: usize| arc(vec![1.0_f32; n]);
        let weights = ClipTextWeights {
            token_embedding: z(cfg.vocab_size * h),
            position_embedding: z(cfg.max_position_embeddings * h),
            layers: (0..cfg.num_hidden_layers)
                .map(|_| ClipLayerWeights {
                    ln1_g: o(h), ln1_b: z(h),
                    q_w: z(h * h), q_b: z(h),
                    k_w: z(h * h), k_b: z(h),
                    v_w: z(h * h), v_b: z(h),
                    out_w: z(h * h), out_b: z(h),
                    ln2_g: o(h), ln2_b: z(h),
                    fc1_w: z(h * cfg.intermediate_size),
                    fc1_b: z(cfg.intermediate_size),
                    fc2_w: z(cfg.intermediate_size * h),
                    fc2_b: z(h),
                }).collect(),
            final_ln_g: o(h), final_ln_b: z(h),
        };
        let model = SdTextEncoder { config: cfg.clone(), weights };
        let tokens: Vec<u32> = (0..cfg.max_position_embeddings as u32).collect();
        let hidden = model.forward(&tokens).unwrap();
        let flat = hidden.realize_f32();
        assert_eq!(flat.len(), cfg.max_position_embeddings * h);
        assert!(flat.iter().all(|v| v.is_finite()));

        // Phase 6a oracle gate.
        let flat_ref = hidden.realize_f32_reference();
        crate::test_utils::assert_allclose_f32(&flat, &flat_ref, 1e-4, 1e-3);
    }

    #[test]
    fn quick_gelu_matches_reference() {
        // QuickGELU(x) = x * sigmoid(1.702 * x)
        // We build a 1-element graph and compare against the closed form.
        let x_vals = [-2.0_f32, -0.5, 0.0, 0.5, 1.0, 2.0];
        for &v in &x_vals {
            let x = LazyTensor::from_f32(vec![v], Shape::from_dims(&[1]), &crate::Device::cpu());
            let y = quick_gelu(&x);
            let out = y.realize_f32()[0];
            let expected = v * (1.0 / (1.0 + (-1.702_f32 * v).exp()));
            assert!(
                (out - expected).abs() < 1e-6,
                "quick_gelu({v}) = {out}, expected {expected}",
            );
        }
    }
}
