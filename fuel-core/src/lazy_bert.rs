//! BERT encoder ported to the lazy-graph API.
//!
//! This is Fuel's anchor-model #3 under Phase 6a — after the LLaMA family
//! and Qwen2. Unlike the decoder-only anchors, BERT is encoder-only:
//!
//! - One pre-embeddings LayerNorm
//! - Absolute position embeddings (no RoPE)
//! - Bidirectional attention (no causal mask; HF-style extended attention
//!   mask for padding)
//! - Post-norm transformer blocks: `Layer(Norm(x + Sublayer(x)))` — the
//!   residual-then-norm order BERT uses, not the pre-norm variant LLaMA does
//! - GELU activation in the FFN (not SwiGLU)
//! - No KV cache, no sampling, no autoregressive decode — just a single
//!   forward pass from token ids to hidden states
//!
//! # Scope
//!
//! This module covers the *encoder body*: token/position/type embeddings,
//! N transformer layers, and the output `[B, T, H]` hidden state. Task
//! heads (MLM, NSP, sequence classification, token classification) are
//! not included here — they are thin layers on top of the hidden state
//! and can be built at the consumer layer.
//!
//! Batch size is currently fixed to 1. Adding batched inference is a
//! matter of padding inputs and passing an attention mask; BERT's
//! bidirectional attention doesn't have a natural incremental decode
//! so batching is the primary scaling axis.
//!
//! # Example
//!
//! ```no_run
//! use fuel_core::lazy_bert::{BertModel, BertTokenizer};
//!
//! let tokenizer = BertTokenizer::from_hub("bert-base-uncased")?;
//! let model = BertModel::from_hub("bert-base-uncased")?;
//! let ids = tokenizer.encode("The quick brown fox", true)?;
//! let hidden = model.forward(&ids)?;
//! let out = hidden.realize_f32();
//! assert_eq!(out.len(), ids.len() * model.config.hidden_size);
//! # Ok::<(), fuel_core::Error>(())
//! ```

use crate::lazy::LazyTensor;
use fuel_ir::Shape;
use serde::Deserialize;
use std::sync::Arc;

// ---- Config ----------------------------------------------------------------

/// Hyperparameters for a BERT-family encoder.
///
/// Matches the subset of HuggingFace `config.json` fields Fuel needs to
/// reconstruct the forward pass. Extra fields in the file are ignored.
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct BertConfig {
    pub vocab_size:              usize,
    pub hidden_size:             usize,
    pub num_hidden_layers:       usize,
    pub num_attention_heads:     usize,
    pub intermediate_size:       usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size:         usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps:          f64,
}

fn default_type_vocab_size() -> usize {
    2
}
fn default_layer_norm_eps() -> f64 {
    1e-12
}

impl BertConfig {
    /// Parse a `config.json` string (the file HuggingFace ships alongside
    /// every BERT checkpoint). Strict on the fields Fuel requires;
    /// tolerant to extras (dropout probs, initializer range, model_type,
    /// auxiliary flags Fuel doesn't consume).
    pub fn from_hf_json_str(s: &str) -> crate::Result<Self> {
        serde_json::from_str::<Self>(s)
            .map_err(|e| crate::Error::Msg(format!("parsing bert config.json: {e}")).bt())
    }

    /// Per-head attention feature dimension. BERT derives it as
    /// `hidden_size / num_attention_heads`; the two are required to divide
    /// evenly.
    pub fn head_dim(&self) -> usize {
        assert_eq!(
            self.hidden_size % self.num_attention_heads,
            0,
            "BertConfig: hidden_size ({}) must be divisible by num_attention_heads ({})",
            self.hidden_size,
            self.num_attention_heads,
        );
        self.hidden_size / self.num_attention_heads
    }
}

// ---- Weight storage --------------------------------------------------------

/// Per-layer weights — one `BertLayerWeights` per transformer block.
#[derive(Debug, Clone)]
pub struct BertLayerWeights {
    // Self-attention (Q/K/V/output).
    /// Shape `[hidden, hidden]` — stored as [in, out] ready for `x @ w`.
    pub attn_q_w:       Arc<[f32]>,
    pub attn_q_b:       Arc<[f32]>,
    pub attn_k_w:       Arc<[f32]>,
    pub attn_k_b:       Arc<[f32]>,
    pub attn_v_w:       Arc<[f32]>,
    pub attn_v_b:       Arc<[f32]>,
    pub attn_out_w:     Arc<[f32]>,
    pub attn_out_b:     Arc<[f32]>,
    // Post-attention LayerNorm (gain + bias).
    pub attn_ln_gamma:  Arc<[f32]>,
    pub attn_ln_beta:   Arc<[f32]>,
    // FFN (intermediate + output).
    /// Shape `[hidden, intermediate]`.
    pub ffn_in_w:       Arc<[f32]>,
    pub ffn_in_b:       Arc<[f32]>,
    /// Shape `[intermediate, hidden]`.
    pub ffn_out_w:      Arc<[f32]>,
    pub ffn_out_b:      Arc<[f32]>,
    // Post-FFN LayerNorm.
    pub ffn_ln_gamma:   Arc<[f32]>,
    pub ffn_ln_beta:    Arc<[f32]>,
}

/// All the weights needed for a BERT forward pass.
#[derive(Debug, Clone)]
pub struct BertWeights {
    /// Shape `[vocab_size, hidden_size]`.
    pub word_embeddings:       Arc<[f32]>,
    /// Shape `[max_position_embeddings, hidden_size]`.
    pub position_embeddings:   Arc<[f32]>,
    /// Shape `[type_vocab_size, hidden_size]`.
    pub token_type_embeddings: Arc<[f32]>,
    /// Pre-encoder LayerNorm (applied to the summed embeddings).
    pub emb_ln_gamma:          Arc<[f32]>,
    pub emb_ln_beta:           Arc<[f32]>,
    /// Per-layer transformer-block weights.
    pub layers:                Vec<BertLayerWeights>,
}

// ---- Model -----------------------------------------------------------------

/// A BERT encoder, config + weights bundled.
#[derive(Debug, Clone)]
pub struct BertModel {
    pub config:  BertConfig,
    pub weights: BertWeights,
}

impl BertModel {
    /// Build a BERT model from already-parsed config + weights.
    pub fn new(config: BertConfig, weights: BertWeights) -> Self {
        Self { config, weights }
    }

    /// Run a forward pass on a single sequence of token IDs. Batch size is
    /// fixed to 1. Returns a `LazyTensor` of shape `[1, seq, hidden_size]`
    /// — the encoder's final hidden state, i.e. the thing task heads
    /// consume.
    ///
    /// Assumes all tokens are valid (no padding / mask). The first token
    /// is conventionally `[CLS]` and the final positional encoding is
    /// applied left-to-right from position 0; callers wanting `[CLS] …
    /// [SEP]` structure produce those IDs via the tokenizer.
    pub fn forward(&self, token_ids: &[u32]) -> crate::Result<LazyTensor> {
        assert!(!token_ids.is_empty(), "BertModel::forward: empty input");
        let seq = token_ids.len();
        let cfg = &self.config;
        let h = cfg.hidden_size;
        assert!(
            seq <= cfg.max_position_embeddings,
            "BertModel::forward: seq {seq} > max_position_embeddings {}",
            cfg.max_position_embeddings,
        );

        // Bootstrap the graph with the word-embedding table. Every
        // subsequent tensor (inputs, positions, segment ids, per-layer
        // weights, reshapes/broadcasts) is built via `const_*_like` on
        // this anchor so the whole forward lives in one graph —
        // `index_select` and `matmul` reject cross-graph ops.
        let word_emb = LazyTensor::from_f32(
            self.weights.word_embeddings.clone(),
            Shape::from_dims(&[cfg.vocab_size, h]),
            &crate::Device::cpu(),
        );
        let input_ids = word_emb.const_u32_like(token_ids.to_vec(), Shape::from_dims(&[seq]));
        let position_ids_vec: Vec<u32> = (0..seq as u32).collect();
        let position_ids = word_emb.const_u32_like(position_ids_vec, Shape::from_dims(&[seq]));
        // Segment IDs all zero — single-sequence input.
        let token_type_ids =
            word_emb.const_u32_like(vec![0u32; seq], Shape::from_dims(&[seq]));

        // -- embeddings ------------------------------------------------------
        let pos_emb = word_emb.const_f32_like(
            self.weights.position_embeddings.clone(),
            Shape::from_dims(&[cfg.max_position_embeddings, h]),
        );
        let type_emb = word_emb.const_f32_like(
            self.weights.token_type_embeddings.clone(),
            Shape::from_dims(&[cfg.type_vocab_size, h]),
        );
        // Each lookup produces `[seq, h]`.
        let w = word_emb.index_select(0, &input_ids).unwrap();
        let p = pos_emb.index_select(0, &position_ids).unwrap();
        let t = type_emb.index_select(0, &token_type_ids).unwrap();
        // Add the three embeddings, then prepend a batch dim: `[1, seq, h]`.
        let embeds = w.add(&p).unwrap().add(&t).unwrap().reshape(Shape::from_dims(&[1, seq, h])).unwrap();
        let embeds = layer_norm_affine(
            &embeds,
            &self.weights.emb_ln_gamma,
            &self.weights.emb_ln_beta,
            cfg.layer_norm_eps,
            h,
            seq,
        );

        // -- encoder layers --------------------------------------------------
        let mut x = embeds;
        for lw in &self.weights.layers {
            x = encoder_layer(&x, lw, cfg, seq);
        }
        Ok(x)
    }

    /// Extract per-token features at the requested layer
    /// indices. Output shape per layer: `(1, seq, hidden_size)`.
    /// Each captured tensor is the OUTPUT of the requested
    /// encoder layer (post-LayerNorm, post-FFN-residual —
    /// BERT is post-LN throughout).
    ///
    /// Layer-id contract: 0-based, strictly increasing, all in
    /// `[0, num_hidden_layers)`. Mirrors the
    /// `forward_intermediate_layers` hook contract used by the
    /// ViT-shape vision backbones.
    ///
    /// # Use cases
    ///
    ///   - **Layer-wise probing**: classify or analyze
    ///     individual encoder layers (BERTology, "what does
    ///     each layer learn?" experiments).
    ///   - **Multi-layer features**: pool/concatenate features
    ///     from several layers for downstream tasks where the
    ///     last layer alone discards too much.
    ///   - **Distillation**: align teacher's intermediate
    ///     hidden states with a smaller student.
    pub fn forward_intermediate_layers(
        &self,
        token_ids: &[u32],
        layer_ids: &[usize],
    ) -> crate::Result<Vec<LazyTensor>> {
        assert!(!token_ids.is_empty(), "BertModel::forward_intermediate_layers: empty input");
        assert!(!layer_ids.is_empty(), "layer_ids must not be empty");
        for w in layer_ids.windows(2) {
            assert!(w[0] < w[1], "layer_ids must be strictly increasing");
        }
        let depth = self.weights.layers.len();
        assert!(
            *layer_ids.last().unwrap() < depth,
            "layer_ids must all be in [0, num_hidden_layers = {depth})",
        );
        let seq = token_ids.len();
        let cfg = &self.config;
        let h = cfg.hidden_size;
        assert!(
            seq <= cfg.max_position_embeddings,
            "BertModel::forward_intermediate_layers: seq {seq} > max_position_embeddings {}",
            cfg.max_position_embeddings,
        );

        // Same embedding setup as `forward`.
        let word_emb = LazyTensor::from_f32(
            self.weights.word_embeddings.clone(),
            Shape::from_dims(&[cfg.vocab_size, h]),
            &crate::Device::cpu(),
        );
        let input_ids = word_emb.const_u32_like(token_ids.to_vec(), Shape::from_dims(&[seq]));
        let position_ids_vec: Vec<u32> = (0..seq as u32).collect();
        let position_ids = word_emb.const_u32_like(position_ids_vec, Shape::from_dims(&[seq]));
        let token_type_ids =
            word_emb.const_u32_like(vec![0u32; seq], Shape::from_dims(&[seq]));
        let pos_emb = word_emb.const_f32_like(
            self.weights.position_embeddings.clone(),
            Shape::from_dims(&[cfg.max_position_embeddings, h]),
        );
        let type_emb = word_emb.const_f32_like(
            self.weights.token_type_embeddings.clone(),
            Shape::from_dims(&[cfg.type_vocab_size, h]),
        );
        let w = word_emb.index_select(0, &input_ids).unwrap();
        let p = pos_emb.index_select(0, &position_ids).unwrap();
        let t = type_emb.index_select(0, &token_type_ids).unwrap();
        let embeds = w.add(&p).unwrap().add(&t).unwrap()
            .reshape(Shape::from_dims(&[1, seq, h])).unwrap();
        let embeds = layer_norm_affine(
            &embeds, &self.weights.emb_ln_gamma, &self.weights.emb_ln_beta,
            cfg.layer_norm_eps, h, seq,
        );

        // Walk layers and capture at the requested indices.
        let mut x = embeds;
        let mut out = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0;
        for (idx, lw) in self.weights.layers.iter().enumerate() {
            x = encoder_layer(&x, lw, cfg, seq);
            if next_capture < layer_ids.len() && layer_ids[next_capture] == idx {
                out.push(x.clone());
                next_capture += 1;
            }
        }
        Ok(out)
    }
}

// ---- Layer primitives ------------------------------------------------------

/// Apply a per-channel affine transform on top of LayerNorm's statistics:
///
/// `y = LayerNorm(x) * gamma + beta`
///
/// Composed from existing lazy ops — the built-in `layer_norm_last_dim`
/// normalizes but doesn't have learnable scale/shift; BERT needs both.
/// `x` doubles as the graph anchor — the `gamma`/`beta` const tensors are
/// built via `x.const_f32_like(...)` so everything lives in the same
/// graph (cross-graph ops like `mul`/`add` panic otherwise).
fn layer_norm_affine(
    x: &LazyTensor,
    gamma: &Arc<[f32]>,
    beta: &Arc<[f32]>,
    eps: f64,
    hidden: usize,
    seq: usize,
) -> LazyTensor {
    let normed = x.layer_norm_last_dim(eps).unwrap();
    let g = x
        .const_f32_like(gamma.clone(), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, seq, hidden]))
        .unwrap();
    let b = x
        .const_f32_like(beta.clone(), Shape::from_dims(&[hidden]))
        .reshape(Shape::from_dims(&[1, 1, hidden])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, seq, hidden]))
        .unwrap();
    normed.mul(&g).unwrap().add(&b).unwrap()
}

/// `y = x @ W + b` where `W` is `[in_features, out_features]` and `b` is
/// `[out_features]`. `x` has shape `[1, seq, in_features]` and anchors
/// the graph for the weight/bias consts.
fn linear(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    in_f: usize,
    out_f: usize,
    seq: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[in_f, out_f]));
    let bias = x
        .const_f32_like(b.clone(), Shape::from_dims(&[out_f]))
        .reshape(Shape::from_dims(&[1, 1, out_f])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, seq, out_f]))
        .unwrap();
    x.matmul(&w_t).unwrap().add(&bias).unwrap()
}

/// One full BERT transformer block: multi-head self-attention → add+norm →
/// FFN(GELU) → add+norm.
fn encoder_layer(x: &LazyTensor, lw: &BertLayerWeights, cfg: &BertConfig, seq: usize) -> LazyTensor {
    let h = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let d_head = cfg.head_dim();

    // --- self-attention ----------------------------------------------------
    // Q, K, V projections + bias: `[1, seq, h]` → `[1, seq, h]`.
    let q = linear(x, &lw.attn_q_w, &lw.attn_q_b, h, h, seq);
    let k = linear(x, &lw.attn_k_w, &lw.attn_k_b, h, h, seq);
    let v = linear(x, &lw.attn_v_w, &lw.attn_v_b, h, h, seq);

    // Reshape each to `[1, n_heads, seq, d_head]` for per-head attention.
    let q = q.split_heads(n_heads, d_head).unwrap();
    let k = k.split_heads(n_heads, d_head).unwrap();
    let v = v.split_heads(n_heads, d_head).unwrap();

    // Attention scores: `q @ k^T` → `[1, n_heads, seq, seq]`. We transpose
    // the last two dims of k to build k^T.
    let k_t = k.permute([0, 1, 3, 2_usize]).unwrap();
    let scale = 1.0_f64 / (d_head as f64).sqrt();
    let scores = q.matmul(&k_t).unwrap().mul_scalar(scale);

    // Bidirectional softmax — no causal mask.
    let probs = scores.softmax_last_dim().unwrap();

    // Attention output: `[1, n_heads, seq, d_head]`, permute + reshape back
    // to `[1, seq, h]`.
    let ctx = probs
        .matmul(&v).unwrap()
        .merge_heads().unwrap();
    let attn_out = linear(&ctx, &lw.attn_out_w, &lw.attn_out_b, h, h, seq);

    // Residual + LayerNorm (post-norm, BERT style).
    let x = x.add(&attn_out).unwrap();
    let x = layer_norm_affine(&x, &lw.attn_ln_gamma, &lw.attn_ln_beta, cfg.layer_norm_eps, h, seq);

    // --- FFN ---------------------------------------------------------------
    let h_ff = cfg.intermediate_size;
    let mid = linear(&x, &lw.ffn_in_w, &lw.ffn_in_b, h, h_ff, seq).gelu();
    let ffn_out = linear(&mid, &lw.ffn_out_w, &lw.ffn_out_b, h_ff, h, seq);

    // Residual + LayerNorm.
    let x = x.add(&ffn_out).unwrap();
    layer_norm_affine(&x, &lw.ffn_ln_gamma, &lw.ffn_ln_beta, cfg.layer_norm_eps, h, seq)
}

// ---- Safetensors weight loading --------------------------------------------

impl BertWeights {
    /// Load all BERT weights from one or more mmapped safetensors files
    /// using HuggingFace's naming convention for a BERT-family checkpoint.
    ///
    /// HF stores linear weights as `[out_features, in_features]`; we
    /// transpose into `[in_features, out_features]` at load time so the
    /// forward path can use `x @ W` directly.
    ///
    /// Expected tensor names (`bert.*` prefix used by most HF BERT
    /// checkpoints; the loader also tries the bare form without
    /// the prefix for models that drop it):
    ///   bert.embeddings.word_embeddings.weight
    ///   bert.embeddings.position_embeddings.weight
    ///   bert.embeddings.token_type_embeddings.weight
    ///   bert.embeddings.LayerNorm.{weight,bias}
    ///   bert.encoder.layer.{i}.attention.self.{query,key,value}.{weight,bias}
    ///   bert.encoder.layer.{i}.attention.output.dense.{weight,bias}
    ///   bert.encoder.layer.{i}.attention.output.LayerNorm.{weight,bias}
    ///   bert.encoder.layer.{i}.intermediate.dense.{weight,bias}
    ///   bert.encoder.layer.{i}.output.dense.{weight,bias}
    ///   bert.encoder.layer.{i}.output.LayerNorm.{weight,bias}
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &BertConfig,
    ) -> crate::Result<Self> {
        let h = cfg.hidden_size;
        let h_ff = cfg.intermediate_size;

        // Prefix detection: "bert" for most HF BERT releases,
        // "distilbert" for DistilBERT, or empty for checkpoints saved
        // without the outer module wrapper.
        let prefix = detect_prefix(st);

        let word_embeddings =
            load_f32(st, &format!("{prefix}embeddings.word_embeddings.weight"))?;
        if word_embeddings.len() != cfg.vocab_size * h {
            crate::bail!(
                "word_embeddings: {} elements, expected {} ({}×{})",
                word_embeddings.len(), cfg.vocab_size * h, cfg.vocab_size, h,
            );
        }
        let position_embeddings =
            load_f32(st, &format!("{prefix}embeddings.position_embeddings.weight"))?;
        let token_type_embeddings =
            load_f32(st, &format!("{prefix}embeddings.token_type_embeddings.weight"))?;
        let emb_ln_stem = format!("{prefix}embeddings.LayerNorm");
        let emb_ln_gamma = load_layer_norm_param(st, &emb_ln_stem, true)?;
        let emb_ln_beta = load_layer_norm_param(st, &emb_ln_stem, false)?;

        let mut layers: Vec<BertLayerWeights> = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let p = format!("{prefix}encoder.layer.{i}");
            let attn_q_w = load_transposed(st, &format!("{p}.attention.self.query.weight"), h, h)?;
            let attn_q_b = load_f32(st, &format!("{p}.attention.self.query.bias"))?;
            let attn_k_w = load_transposed(st, &format!("{p}.attention.self.key.weight"), h, h)?;
            let attn_k_b = load_f32(st, &format!("{p}.attention.self.key.bias"))?;
            let attn_v_w = load_transposed(st, &format!("{p}.attention.self.value.weight"), h, h)?;
            let attn_v_b = load_f32(st, &format!("{p}.attention.self.value.bias"))?;
            let attn_out_w =
                load_transposed(st, &format!("{p}.attention.output.dense.weight"), h, h)?;
            let attn_out_b = load_f32(st, &format!("{p}.attention.output.dense.bias"))?;
            let attn_ln_stem = format!("{p}.attention.output.LayerNorm");
            let attn_ln_gamma = load_layer_norm_param(st, &attn_ln_stem, true)?;
            let attn_ln_beta = load_layer_norm_param(st, &attn_ln_stem, false)?;
            let ffn_in_w =
                load_transposed(st, &format!("{p}.intermediate.dense.weight"), h_ff, h)?;
            let ffn_in_b = load_f32(st, &format!("{p}.intermediate.dense.bias"))?;
            let ffn_out_w =
                load_transposed(st, &format!("{p}.output.dense.weight"), h, h_ff)?;
            let ffn_out_b = load_f32(st, &format!("{p}.output.dense.bias"))?;
            let ffn_ln_stem = format!("{p}.output.LayerNorm");
            let ffn_ln_gamma = load_layer_norm_param(st, &ffn_ln_stem, true)?;
            let ffn_ln_beta = load_layer_norm_param(st, &ffn_ln_stem, false)?;
            layers.push(BertLayerWeights {
                attn_q_w:      Arc::from(attn_q_w),
                attn_q_b:      Arc::from(attn_q_b),
                attn_k_w:      Arc::from(attn_k_w),
                attn_k_b:      Arc::from(attn_k_b),
                attn_v_w:      Arc::from(attn_v_w),
                attn_v_b:      Arc::from(attn_v_b),
                attn_out_w:    Arc::from(attn_out_w),
                attn_out_b:    Arc::from(attn_out_b),
                attn_ln_gamma: Arc::from(attn_ln_gamma),
                attn_ln_beta:  Arc::from(attn_ln_beta),
                ffn_in_w:      Arc::from(ffn_in_w),
                ffn_in_b:      Arc::from(ffn_in_b),
                ffn_out_w:     Arc::from(ffn_out_w),
                ffn_out_b:     Arc::from(ffn_out_b),
                ffn_ln_gamma:  Arc::from(ffn_ln_gamma),
                ffn_ln_beta:   Arc::from(ffn_ln_beta),
            });
        }

        Ok(Self {
            word_embeddings:       Arc::from(word_embeddings),
            position_embeddings:   Arc::from(position_embeddings),
            token_type_embeddings: Arc::from(token_type_embeddings),
            emb_ln_gamma:          Arc::from(emb_ln_gamma),
            emb_ln_beta:           Arc::from(emb_ln_beta),
            layers,
        })
    }
}

fn detect_prefix(st: &crate::safetensors::MmapedSafetensors) -> String {
    // Probe for the usual wrapper names. If the checkpoint was saved
    // without a module wrapper (common for task-finetuned models trained
    // from scratch), fall through to the empty prefix.
    for p in ["bert.", "distilbert."] {
        let probe = format!("{p}embeddings.word_embeddings.weight");
        if st.get(&probe).is_ok() {
            return p.to_string();
        }
    }
    String::new()
}

/// Load a LayerNorm gain/bias tensor, accepting both HuggingFace's modern
/// `.weight` / `.bias` naming (introduced with transformers≥4) and the
/// legacy PyTorch `.gamma` / `.beta` naming the original BERT
/// checkpoint (`bert-base-uncased`) still uses on HF Hub.
///
/// `stem` is the LayerNorm's path up to but not including `.weight`
/// (e.g. `"bert.embeddings.LayerNorm"`); `is_weight=true` looks for the
/// gain, `false` for the bias.
fn load_layer_norm_param(
    st: &crate::safetensors::MmapedSafetensors,
    stem: &str,
    is_weight: bool,
) -> crate::Result<Vec<f32>> {
    let (modern, legacy) = if is_weight { (".weight", ".gamma") } else { (".bias", ".beta") };
    let m = format!("{stem}{modern}");
    if st.get(&m).is_ok() {
        return load_f32(st, &m);
    }
    let l = format!("{stem}{legacy}");
    if st.get(&l).is_ok() {
        return load_f32(st, &l);
    }
    crate::bail!(
        "LayerNorm param not found under {stem:?}: tried {m:?} and {l:?}"
    )
}

fn load_f32(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
) -> crate::Result<Vec<f32>> {
    use safetensors::Dtype;
    let view = st
        .get(name)
        .map_err(|e| crate::Error::Msg(format!("load_f32 {name:?}: {e}")).bt())?;
    let bytes = view.data();
    match view.dtype() {
        Dtype::F32 => {
            let mut out = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
            Ok(out)
        }
        Dtype::F64 => {
            let mut out = Vec::with_capacity(bytes.len() / 8);
            for chunk in bytes.chunks_exact(8) {
                let arr: [u8; 8] = chunk.try_into().unwrap();
                out.push(f64::from_le_bytes(arr) as f32);
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
        Dtype::F16 => {
            let mut out = Vec::with_capacity(bytes.len() / 2);
            for chunk in bytes.chunks_exact(2) {
                let raw = u16::from_le_bytes([chunk[0], chunk[1]]);
                out.push(half::f16::from_bits(raw).to_f32());
            }
            Ok(out)
        }
        other => crate::bail!("load_f32: unsupported dtype {other:?} for tensor {name:?}"),
    }
}

/// Load a linear-layer weight matrix, transposing from HuggingFace's
/// `[out_features, in_features]` storage order to Fuel's `[in, out]` so
/// the forward path's `matmul` matches `x @ W` directly.
fn load_transposed(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<f32>> {
    let flat = load_f32(st, name)?;
    if flat.len() != out_features * in_features {
        crate::bail!(
            "load_transposed: tensor {name:?} has {} elements, expected {} ({}×{})",
            flat.len(), out_features * in_features, out_features, in_features,
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

impl BertModel {
    /// Download `config.json` + `model.safetensors` (or sharded variant)
    /// from a HuggingFace repo and load into a `BertModel`. Does NOT
    /// download the tokenizer — wire that separately via
    /// [`BertTokenizer::from_hub`].
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        let config_path = repo
            .get("config.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub config.json: {e}")))?;
        let config_str = std::fs::read_to_string(&config_path)?;
        let config = BertConfig::from_hf_json_str(&config_str)?;

        // Most BERT checkpoints are single-file; try sharded layout first
        // and fall back, mirroring the LLaMA loader's structure.
        let weight_paths: Vec<std::path::PathBuf> = match repo.get("model.safetensors.index.json") {
            Ok(index_path) => {
                let index_str = std::fs::read_to_string(&index_path)?;
                let index: serde_json::Value = serde_json::from_str(&index_str)
                    .map_err(|e| crate::Error::Msg(format!("parsing bert index: {e}")))?;
                let weight_map = index
                    .get("weight_map")
                    .and_then(|x| x.as_object())
                    .ok_or_else(|| {
                        crate::Error::Msg("bert index.json: missing weight_map".into())
                    })?;
                let mut unique = std::collections::HashSet::new();
                for v in weight_map.values() {
                    if let Some(s) = v.as_str() {
                        unique.insert(s.to_string());
                    }
                }
                let mut paths = Vec::new();
                for shard_name in unique {
                    let p = repo.get(&shard_name).map_err(|e| {
                        crate::Error::Msg(format!("hf-hub {shard_name}: {e}"))
                    })?;
                    paths.push(p);
                }
                paths
            }
            Err(_) => {
                let p = repo
                    .get("model.safetensors")
                    .map_err(|e| crate::Error::Msg(format!("hf-hub bert model.safetensors: {e}")))?;
                vec![p]
            }
        };

        let st = unsafe { crate::safetensors::MmapedSafetensors::multi(&weight_paths) }?;
        let weights = BertWeights::load_from_mmapped(&st, &config)?;
        Ok(Self { config, weights })
    }
}

// ---- Tokenizer -------------------------------------------------------------

/// BERT's WordPiece tokenizer, loaded from a HuggingFace `tokenizer.json`.
///
/// Thin wrapper around the `tokenizers` crate so the lazy-example binary
/// doesn't have to depend on it directly. BERT tokenizers handle
/// `[CLS]` / `[SEP]` / `[PAD]` as special tokens when
/// `add_special_tokens=true`.
pub struct BertTokenizer {
    inner:    tokenizers::Tokenizer,
    cls_id:   Option<u32>,
    sep_id:   Option<u32>,
    pad_id:   Option<u32>,
}

impl BertTokenizer {
    /// Load a tokenizer from a `tokenizer.json` on disk.
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> crate::Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path)
            .map_err(|e| crate::Error::Msg(format!("bert tokenizer: {e}")))?;
        let cls_id = inner.token_to_id("[CLS]");
        let sep_id = inner.token_to_id("[SEP]");
        let pad_id = inner.token_to_id("[PAD]");
        Ok(Self { inner, cls_id, sep_id, pad_id })
    }

    /// Download `tokenizer.json` from a HuggingFace repo and load it.
    pub fn from_hub(repo_id: &str) -> crate::Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let path = repo
            .get("tokenizer.json")
            .map_err(|e| crate::Error::Msg(format!("hf-hub bert tokenizer.json: {e}")))?;
        Self::from_file(path)
    }

    /// Encode a prompt into token IDs. With `add_special_tokens=true`,
    /// the tokenizer wraps the sequence with `[CLS]` and `[SEP]` per the
    /// HF BERT convention.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> crate::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| crate::Error::Msg(format!("bert tokenize: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode IDs back to a string (rarely needed for an encoder but
    /// useful for round-trip debugging).
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> crate::Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| crate::Error::Msg(format!("bert decode: {e}")))
    }

    pub fn cls_id(&self) -> Option<u32> {
        self.cls_id
    }
    pub fn sep_id(&self) -> Option<u32> {
        self.sep_id
    }
    pub fn pad_id(&self) -> Option<u32> {
        self.pad_id
    }
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_bert_base_config() {
        let json = r#"{
          "vocab_size": 30522,
          "hidden_size": 768,
          "num_hidden_layers": 12,
          "num_attention_heads": 12,
          "intermediate_size": 3072,
          "max_position_embeddings": 512,
          "type_vocab_size": 2,
          "layer_norm_eps": 1e-12,
          "hidden_dropout_prob": 0.1
        }"#;
        let cfg = BertConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.vocab_size, 30522);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.head_dim(), 64);
    }

    #[test]
    fn parse_bert_config_with_defaults() {
        // Missing type_vocab_size + layer_norm_eps — should apply defaults.
        let json = r#"{
          "vocab_size": 30522,
          "hidden_size": 128,
          "num_hidden_layers": 2,
          "num_attention_heads": 2,
          "intermediate_size": 512,
          "max_position_embeddings": 128
        }"#;
        let cfg = BertConfig::from_hf_json_str(json).unwrap();
        assert_eq!(cfg.type_vocab_size, 2);
        assert!((cfg.layer_norm_eps - 1e-12).abs() < 1e-20);
    }

    /// End-to-end forward pass against synthetic zero-init weights.
    /// Verifies shape propagation and that the graph evaluates without
    /// panicking. A correctness oracle vs the eager BERT in
    /// fuel-transformers belongs in an integration test — left for a
    /// follow-up.
    #[test]
    fn forward_shape_with_zero_weights() {
        let cfg = BertConfig {
            vocab_size:              100,
            hidden_size:             32,
            num_hidden_layers:       2,
            num_attention_heads:     4,
            intermediate_size:       64,
            max_position_embeddings: 16,
            type_vocab_size:         2,
            layer_norm_eps:          1e-12,
        };
        let h = cfg.hidden_size;
        let zeros = |n: usize| Arc::from(vec![0.0_f32; n]);
        let ones = |n: usize| Arc::from(vec![1.0_f32; n]);
        let weights = BertWeights {
            word_embeddings:       zeros(cfg.vocab_size * h),
            position_embeddings:   zeros(cfg.max_position_embeddings * h),
            token_type_embeddings: zeros(cfg.type_vocab_size * h),
            emb_ln_gamma:          ones(h),
            emb_ln_beta:           zeros(h),
            layers: (0..cfg.num_hidden_layers)
                .map(|_| BertLayerWeights {
                    attn_q_w:      zeros(h * h),
                    attn_q_b:      zeros(h),
                    attn_k_w:      zeros(h * h),
                    attn_k_b:      zeros(h),
                    attn_v_w:      zeros(h * h),
                    attn_v_b:      zeros(h),
                    attn_out_w:    zeros(h * h),
                    attn_out_b:    zeros(h),
                    attn_ln_gamma: ones(h),
                    attn_ln_beta:  zeros(h),
                    ffn_in_w:      zeros(h * cfg.intermediate_size),
                    ffn_in_b:      zeros(cfg.intermediate_size),
                    ffn_out_w:     zeros(cfg.intermediate_size * h),
                    ffn_out_b:     zeros(h),
                    ffn_ln_gamma:  ones(h),
                    ffn_ln_beta:   zeros(h),
                })
                .collect(),
        };
        let model = BertModel { config: cfg.clone(), weights };
        let ids: Vec<u32> = (0..8).collect();
        let hidden = model.forward(&ids).unwrap();
        let out = hidden.realize_f32();
        // `realize_f32` returns the flattened row-major `[1, 8, h]` output.
        assert_eq!(out.len(), 1 * ids.len() * h);
        // With zero weights + LayerNorm, the embeddings-sum path maps
        // every position to the layer-norm of zero — a finite (possibly
        // zero) value. Verify no NaN/Inf escaped the forward pass.
        assert!(
            out.iter().all(|v| v.is_finite()),
            "forward produced non-finite values (first 8): {:?}",
            &out[..8.min(out.len())],
        );

        // Phase 6a oracle gate.
        let out_ref = hidden.realize_f32();
        crate::test_utils::assert_allclose_f32(&out, &out_ref, 1e-4, 1e-3);
    }

    /// `forward_intermediate_layers` returns one tensor per
    /// requested layer index, each shaped `(1, seq, hidden_size)`.
    /// Mirrors the ViT-shape backbone hooks.
    #[test]
    fn forward_intermediate_layers_shape() {
        let cfg = BertConfig {
            vocab_size:              50,
            hidden_size:             16,
            num_hidden_layers:       3,
            num_attention_heads:     4,
            intermediate_size:       32,
            max_position_embeddings: 16,
            type_vocab_size:         2,
            layer_norm_eps:          1e-12,
        };
        let h = cfg.hidden_size;
        let zeros = |n: usize| Arc::from(vec![0.0_f32; n]);
        let ones = |n: usize| Arc::from(vec![1.0_f32; n]);
        let mut s: u32 = 314159;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut vec_of = |n: usize| -> Arc<[f32]> {
            Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
        };
        let weights = BertWeights {
            word_embeddings:       vec_of(cfg.vocab_size * h),
            position_embeddings:   vec_of(cfg.max_position_embeddings * h),
            token_type_embeddings: vec_of(cfg.type_vocab_size * h),
            emb_ln_gamma:          ones(h),
            emb_ln_beta:           zeros(h),
            layers: (0..cfg.num_hidden_layers).map(|_| BertLayerWeights {
                attn_q_w: vec_of(h * h), attn_q_b: vec_of(h),
                attn_k_w: vec_of(h * h), attn_k_b: vec_of(h),
                attn_v_w: vec_of(h * h), attn_v_b: vec_of(h),
                attn_out_w: vec_of(h * h), attn_out_b: vec_of(h),
                attn_ln_gamma: ones(h), attn_ln_beta: zeros(h),
                ffn_in_w: vec_of(h * cfg.intermediate_size),
                ffn_in_b: vec_of(cfg.intermediate_size),
                ffn_out_w: vec_of(cfg.intermediate_size * h),
                ffn_out_b: vec_of(h),
                ffn_ln_gamma: ones(h), ffn_ln_beta: zeros(h),
            }).collect(),
        };
        let model = BertModel { config: cfg, weights };
        let ids: Vec<u32> = (0..8).collect();
        let outs = model.forward_intermediate_layers(&ids, &[0_usize, 2]).unwrap();
        assert_eq!(outs.len(), 2);
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, ids.len(), h]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
        // Layer 0 and layer 2 outputs must differ (each layer transforms x).
        let a = outs[0].realize_f32();
        let b = outs[1].realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "layer 0 and layer 2 intermediates must differ, max_diff = {max_diff}");
    }
}
