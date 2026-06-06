//! MMDiT (Multimodal Diffusion Transformer) ported to the lazy-graph API.
//!
//! Substrate for Stable Diffusion 3 / SD 3.5 / Flux. A transformer that
//! processes joint text and image token streams with shared attention.
//!
//! - Paper: <https://arxiv.org/abs/2403.03206>.
//! - Eager: `fuel-transformers/src/models/diffusion/mmdit/*`.
//!
//! Two block types:
//!
//! - **DoubleStreamBlock** — text and image have separate Q/K/V
//!   projections + separate AdaLN modulation params, but the attention
//!   keys/values are concatenated across modalities so each token
//!   attends to both modalities. Output is split back to per-modality
//!   sub-sequences.
//! - **SingleStreamBlock** — text and image are concatenated and run
//!   through a single shared attention + MLP (post-DoubleStream
//!   join). One AdaLN modulation block; no per-modality split.
//!
//! Modulation: each block reads a `(B, n_mod_params * dim)` vector
//! from a timestep + label embedding, chunked into `(shift, scale,
//! gate)` triples. AdaLN applies `(1 + scale) * norm(x) + shift`,
//! and the gate scales the residual contribution.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1 supported (broadcast-compatible code
//! paths so larger batches also work), F32. The implementation
//! mirrors the SD3 (not SD3.5-MMDiT-X) shape: 6 modulation params per
//! DoubleStream block per stream. No QK-norm, no Flux-style parallel
//! attention.
//!
//! The `forward` entry point bypasses patchify / unpatchify (the
//! caller hands in image tokens already shaped `(B, S_image, dim)`)
//! and bypasses CLIP / T5 text encoders (text tokens already shaped
//! `(B, S_text, dim)`). It applies one DoubleStreamBlock layer per
//! `depth` followed by one SingleStreamBlock layer per `depth`, with
//! AdaLN conditioning derived from `timestep` + `y`.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

// ---- Config -----------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct MmDitConfig {
    /// Joint hidden dimension (text and image share it).
    pub dim: usize,
    pub num_heads: usize,
    /// Number of DoubleStreamBlock layers.
    pub depth_double: usize,
    /// Number of SingleStreamBlock layers. SD3 sets this to zero;
    /// Flux sets it to ~38 (after 19 double-stream blocks).
    pub depth_single: usize,
    pub mlp_ratio: usize,
    /// Layer norm epsilon (AdaLN's norm uses no learnable affine, only
    /// the modulation produced affine).
    pub eps: f64,
}

impl MmDitConfig {
    pub fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }
}

// ---- Weights ----------------------------------------------------------------

/// Per-stream Q/K/V projections + output projection + AdaLN-modulation
/// projection for one DoubleStreamBlock. SD3 has separate `text_*` and
/// `image_*` instances.
#[derive(Debug, Clone)]
pub struct StreamWeights {
    /// AdaLN modulation projection. `[dim, 6 * dim]` — produces the
    /// `(shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
    /// gate_mlp)` chunks. Applied after a SiLU on the conditioning
    /// vector `c`.
    pub adaln_proj: WeightStorage,
    pub adaln_bias: Arc<[f32]>,
    /// `[dim, 3 * dim]` fused QKV projection.
    pub qkv_proj: WeightStorage,
    pub qkv_bias: Arc<[f32]>,
    /// `[dim, dim]` attention output projection.
    pub out_proj: WeightStorage,
    pub out_bias: Arc<[f32]>,
    /// MLP fc1 `[dim, mlp_hidden]` + fc2 `[mlp_hidden, dim]`.
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

/// DoubleStreamBlock holds one `StreamWeights` per modality.
#[derive(Debug, Clone)]
pub struct DoubleStreamBlockWeights {
    pub text: StreamWeights,
    pub image: StreamWeights,
}

/// SingleStreamBlock applies one fused QKV + MLP across the concatenated
/// (text | image) sequence. Shape mirrors `StreamWeights` minus the
/// per-modality split.
#[derive(Debug, Clone)]
pub struct SingleStreamBlockWeights {
    pub adaln_proj: WeightStorage,
    pub adaln_bias: Arc<[f32]>,
    pub qkv_proj: WeightStorage,
    pub qkv_bias: Arc<[f32]>,
    pub out_proj: WeightStorage,
    pub out_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

/// Timestep + label embedding for the conditioning vector `c`.
///
/// Per the eager `TimestepEmbedder` + `VectorEmbedder` modules, both
/// are 2-layer SiLU MLPs. The two outputs are summed to form `c`.
#[derive(Debug, Clone)]
pub struct ConditioningWeights {
    /// `[freq_embed, dim]` first linear of the timestep MLP.
    pub t_fc1: WeightStorage,
    pub t_fc1_bias: Arc<[f32]>,
    /// `[dim, dim]` second linear of the timestep MLP.
    pub t_fc2: WeightStorage,
    pub t_fc2_bias: Arc<[f32]>,
    /// `[adm_in, dim]` first linear of the label MLP.
    pub y_fc1: WeightStorage,
    pub y_fc1_bias: Arc<[f32]>,
    /// `[dim, dim]` second linear of the label MLP.
    pub y_fc2: WeightStorage,
    pub y_fc2_bias: Arc<[f32]>,
    /// Frequency embedding dimension (the timestep sinusoidal feature
    /// length — must equal the t_fc1 input dim).
    pub frequency_embedding_size: usize,
    /// Label embedding input dimension (must equal y_fc1's input dim).
    pub adm_in_channels: usize,
}

#[derive(Debug, Clone)]
pub struct MmDitWeights {
    pub conditioning: ConditioningWeights,
    pub double_blocks: Vec<DoubleStreamBlockWeights>,
    pub single_blocks: Vec<SingleStreamBlockWeights>,
}

// ---- Model ------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MmDitModel {
    pub config: MmDitConfig,
    pub weights: MmDitWeights,
}

impl MmDitModel {
    /// Run the joint transformer over text + image token streams.
    ///
    /// - `img`: `(B, S_image, dim)` image token sequence (already
    ///   patch-embedded + positionally encoded by the caller).
    /// - `txt`: `(B, S_text, dim)` text token sequence (already
    ///   projected to the joint dim by the caller's CLIP / T5
    ///   projector).
    /// - `timestep`: `(B,)` diffusion step scalar per batch element.
    /// - `y`: `(B, adm_in_channels)` label / pooled-text conditioning.
    ///
    /// Returns the post-SingleStream image sequence,
    /// `(B, S_image, dim)`. The text sequence is updated through the
    /// DoubleStream layers but only the image sequence is returned
    /// (DiT's prediction target is the image stream).
    pub fn forward(
        &self,
        img: &LazyTensor,
        txt: &LazyTensor,
        timestep: &LazyTensor,
        y: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;

        // ---- Conditioning vector c = T_mlp(sin_embed(t)) + Y_mlp(y) -----
        let c = build_conditioning(&self.weights.conditioning, timestep, y, cfg.dim)?;

        // ---- DoubleStream stack ----------------------------------------
        let (mut txt_h, mut img_h) = (txt.clone(), img.clone());
        for blk in &self.weights.double_blocks {
            let (new_txt, new_img) =
                apply_double_stream(&txt_h, &img_h, &c, blk, cfg)?;
            txt_h = new_txt;
            img_h = new_img;
        }

        // ---- SingleStream stack (concat text+image, run unified) -------
        let s_text = txt_h.shape().dims()[1];
        let mut joined = txt_h.concat(&img_h, 1_usize)?;
        for blk in &self.weights.single_blocks {
            joined = apply_single_stream(&joined, &c, blk, cfg)?;
        }

        // Return just the image segment.
        let total = joined.shape().dims()[1];
        joined.narrow(1_usize, s_text, total - s_text)
    }
}

// ---- Conditioning -----------------------------------------------------------

/// Build the conditioning vector `c = MLP_t(sin(t)) + MLP_y(y)`. Both
/// MLPs are 2-layer SiLU-activated linears. Result shape: `(B, dim)`.
fn build_conditioning(
    w: &ConditioningWeights,
    timestep: &LazyTensor,
    y: &LazyTensor,
    dim: usize,
) -> Result<LazyTensor> {
    let dims_t = timestep.shape().dims().to_vec();
    if dims_t.len() != 1 {
        return Err(crate::Error::Msg(format!(
            "build_conditioning: timestep must be rank-1, got rank {}",
            dims_t.len()
        )).bt());
    }
    let batch = dims_t[0];

    let t_feat = timestep_sinusoidal_embed(timestep, w.frequency_embedding_size)?;
    let t1 = w.t_fc1.apply_linear(&t_feat, w.frequency_embedding_size, dim);
    let t1 = t1.add_trailing_bias(Arc::clone(&w.t_fc1_bias))?;
    let t1 = t1.silu();
    let t_emb = w.t_fc2.apply_linear(&t1, dim, dim);
    let t_emb = t_emb.add_trailing_bias(Arc::clone(&w.t_fc2_bias))?;

    let y1 = w.y_fc1.apply_linear(y, w.adm_in_channels, dim);
    let y1 = y1.add_trailing_bias(Arc::clone(&w.y_fc1_bias))?;
    let y1 = y1.silu();
    let y_emb = w.y_fc2.apply_linear(&y1, dim, dim);
    let y_emb = y_emb.add_trailing_bias(Arc::clone(&w.y_fc2_bias))?;

    let c = t_emb.add(&y_emb)?;
    let c_dims = c.shape().dims().to_vec();
    if c_dims != vec![batch, dim] {
        return Err(crate::Error::Msg(format!(
            "build_conditioning: c shape {:?} != expected ({}, {})",
            c_dims, batch, dim
        )).bt());
    }
    Ok(c)
}

/// Sinusoidal feature embedding of a scalar timestep tensor. Input
/// `(B,)`, output `(B, dim)` with first `dim/2` cosines followed by
/// `dim/2` sines, matching `SD`'s `flip_sin_to_cos = true` ordering.
fn timestep_sinusoidal_embed(t: &LazyTensor, dim: usize) -> Result<LazyTensor> {
    if !dim.is_multiple_of(2) {
        return Err(crate::Error::Msg(format!(
            "timestep_sinusoidal_embed: dim {dim} must be even"
        )).bt());
    }
    let half = dim / 2;
    let batch = t.shape().dims()[0];

    let max_period = 10_000.0_f32;
    let log_mp = max_period.ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-log_mp * (i as f32) / (half as f32)).exp())
        .collect();
    let freqs_t = t
        .const_f32_like(Arc::from(freqs), Shape::from_dims(&[half]))
        .reshape(Shape::from_dims(&[1, half]))?
        .broadcast_to(Shape::from_dims(&[batch, half]))?;

    let t_col = t
        .reshape(Shape::from_dims(&[batch, 1]))?
        .broadcast_to(Shape::from_dims(&[batch, half]))?;
    let args = t_col.mul(&freqs_t)?;

    let cosines = args.cos();
    let sines = args.sin();
    cosines.concat(&sines, 1_usize)
}

// ---- Modulation -------------------------------------------------------------

/// Container for the six per-block modulation tensors produced by the
/// AdaLN projection. Each tensor has shape `(B, dim)`.
#[derive(Debug, Clone)]
pub struct ModulationChunks {
    pub shift_msa: LazyTensor,
    pub scale_msa: LazyTensor,
    pub gate_msa: LazyTensor,
    pub shift_mlp: LazyTensor,
    pub scale_mlp: LazyTensor,
    pub gate_mlp: LazyTensor,
}

/// Compute `(1 + scale) * x + shift` along the trailing feature dim.
///
/// `x` is `(B, S, dim)` and `scale`/`shift` are `(B, dim)`. The latter
/// two are unsqueezed and broadcast across the sequence dimension to
/// match `x`.
pub fn apply_modulation(
    x: &LazyTensor,
    scale: &LazyTensor,
    shift: &LazyTensor,
) -> Result<LazyTensor> {
    let x_dims = x.shape().dims().to_vec();
    if x_dims.len() != 3 {
        return Err(crate::Error::Msg(format!(
            "apply_modulation: x must be rank-3 (B, S, dim), got rank {}",
            x_dims.len()
        )).bt());
    }
    let (b, s, dim) = (x_dims[0], x_dims[1], x_dims[2]);

    let scale_bc = scale
        .reshape(Shape::from_dims(&[b, 1, dim]))?
        .broadcast_to(Shape::from_dims(&[b, s, dim]))?;
    let shift_bc = shift
        .reshape(Shape::from_dims(&[b, 1, dim]))?
        .broadcast_to(Shape::from_dims(&[b, s, dim]))?;

    let one_plus_scale = scale_bc.add_scalar(1.0);
    let scaled = x.mul(&one_plus_scale)?;
    scaled.add(&shift_bc)
}

fn compute_modulation(
    c: &LazyTensor,
    adaln_proj: &WeightStorage,
    adaln_bias: &Arc<[f32]>,
    dim: usize,
) -> Result<ModulationChunks> {
    let c_act = c.silu();
    let m = adaln_proj.apply_linear(&c_act, dim, 6 * dim);
    let m = m.add_trailing_bias(Arc::clone(adaln_bias))?;
    let chunks = m.chunk(6, 1_usize)?;
    if chunks.len() != 6 {
        return Err(crate::Error::Msg(format!(
            "compute_modulation: expected 6 chunks, got {}",
            chunks.len()
        )).bt());
    }
    Ok(ModulationChunks {
        shift_msa: chunks[0].clone(),
        scale_msa: chunks[1].clone(),
        gate_msa: chunks[2].clone(),
        shift_mlp: chunks[3].clone(),
        scale_mlp: chunks[4].clone(),
        gate_mlp: chunks[5].clone(),
    })
}

// ---- Projections ------------------------------------------------------------

fn split_qkv(
    qkv: &LazyTensor,
    num_heads: usize,
    head_dim: usize,
) -> Result<(LazyTensor, LazyTensor, LazyTensor)> {
    let dims = qkv.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(crate::Error::Msg(format!(
            "split_qkv: input must be rank-3 (B, S, 3*dim), got rank {}",
            dims.len()
        )).bt());
    }
    let (b, s, three_dim) = (dims[0], dims[1], dims[2]);
    let dim = num_heads * head_dim;
    if three_dim != 3 * dim {
        return Err(crate::Error::Msg(format!(
            "split_qkv: last dim {three_dim} != 3 * num_heads ({num_heads}) * head_dim ({head_dim})"
        )).bt());
    }
    let q = qkv.narrow(2_usize, 0, dim)?;
    let k = qkv.narrow(2_usize, dim, dim)?;
    let v = qkv.narrow(2_usize, 2 * dim, dim)?;
    let q = q.split_heads(num_heads, head_dim)?;
    let k = k.split_heads(num_heads, head_dim)?;
    let v = v.split_heads(num_heads, head_dim)?;
    let _ = (b, s);
    Ok((q, k, v))
}

fn project_qkv(
    x_norm_mod: &LazyTensor,
    qkv_proj: &WeightStorage,
    qkv_bias: &Arc<[f32]>,
    num_heads: usize,
    head_dim: usize,
) -> Result<(LazyTensor, LazyTensor, LazyTensor)> {
    let dim = num_heads * head_dim;
    let qkv = qkv_proj.apply_linear(x_norm_mod, dim, 3 * dim);
    let qkv = qkv.add_trailing_bias(Arc::clone(qkv_bias))?;
    split_qkv(&qkv, num_heads, head_dim)
}

// ---- Joint scaled dot-product attention -------------------------------------

fn attention(q: &LazyTensor, k: &LazyTensor, v: &LazyTensor, head_dim: usize) -> Result<LazyTensor> {
    let k_t = k.transpose()?;
    let scale = 1.0_f64 / (head_dim as f64).sqrt();
    let scores = q.matmul(&k_t)?.mul_scalar(scale);
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(v)?;
    ctx.merge_heads()
}

// ---- DoubleStreamBlock ------------------------------------------------------

/// Apply one DoubleStreamBlock: separate per-modality AdaLN +
/// QKV-projection, then joint-attention with concat'd K/V/Q, split
/// back per modality, gated residual + per-modality MLP residual.
pub fn apply_double_stream(
    txt: &LazyTensor,
    img: &LazyTensor,
    c: &LazyTensor,
    weights: &DoubleStreamBlockWeights,
    cfg: &MmDitConfig,
) -> Result<(LazyTensor, LazyTensor)> {
    let dim = cfg.dim;
    let num_heads = cfg.num_heads;
    let head_dim = cfg.head_dim();

    let txt_mod = compute_modulation(c, &weights.text.adaln_proj, &weights.text.adaln_bias, dim)?;
    let img_mod = compute_modulation(c, &weights.image.adaln_proj, &weights.image.adaln_bias, dim)?;

    let txt_norm = txt.layer_norm_last_dim(cfg.eps)?;
    let img_norm = img.layer_norm_last_dim(cfg.eps)?;

    let txt_mod_x = apply_modulation(&txt_norm, &txt_mod.scale_msa, &txt_mod.shift_msa)?;
    let img_mod_x = apply_modulation(&img_norm, &img_mod.scale_msa, &img_mod.shift_msa)?;

    let (txt_q, txt_k, txt_v) = project_qkv(
        &txt_mod_x, &weights.text.qkv_proj, &weights.text.qkv_bias, num_heads, head_dim,
    )?;
    let (img_q, img_k, img_v) = project_qkv(
        &img_mod_x, &weights.image.qkv_proj, &weights.image.qkv_bias, num_heads, head_dim,
    )?;

    let s_txt = txt_q.shape().dims()[2];
    let s_img = img_q.shape().dims()[2];

    let q_all = txt_q.concat(&img_q, 2_usize)?;
    let k_all = txt_k.concat(&img_k, 2_usize)?;
    let v_all = txt_v.concat(&img_v, 2_usize)?;
    let attn_all = attention(&q_all, &k_all, &v_all, head_dim)?;
    let txt_attn = attn_all.narrow(1_usize, 0, s_txt)?;
    let img_attn = attn_all.narrow(1_usize, s_txt, s_img)?;

    let txt_attn_out = weights.text.out_proj.apply_linear(&txt_attn, dim, dim);
    let txt_attn_out = txt_attn_out.add_trailing_bias(Arc::clone(&weights.text.out_bias))?;
    let img_attn_out = weights.image.out_proj.apply_linear(&img_attn, dim, dim);
    let img_attn_out = img_attn_out.add_trailing_bias(Arc::clone(&weights.image.out_bias))?;

    let txt_h1 = gated_residual(txt, &txt_attn_out, &txt_mod.gate_msa)?;
    let img_h1 = gated_residual(img, &img_attn_out, &img_mod.gate_msa)?;

    let txt_out = mlp_residual(&txt_h1, &txt_mod, &weights.text, cfg)?;
    let img_out = mlp_residual(&img_h1, &img_mod, &weights.image, cfg)?;

    Ok((txt_out, img_out))
}

/// `x + gate.unsqueeze(1) * delta` along the sequence axis.
fn gated_residual(
    x: &LazyTensor,
    delta: &LazyTensor,
    gate: &LazyTensor,
) -> Result<LazyTensor> {
    let x_dims = x.shape().dims().to_vec();
    if x_dims.len() != 3 {
        return Err(crate::Error::Msg(format!(
            "gated_residual: x must be rank-3, got rank {}",
            x_dims.len()
        )).bt());
    }
    let (b, s, dim) = (x_dims[0], x_dims[1], x_dims[2]);
    let gate_bc = gate
        .reshape(Shape::from_dims(&[b, 1, dim]))?
        .broadcast_to(Shape::from_dims(&[b, s, dim]))?;
    let gated = delta.mul(&gate_bc)?;
    x.add(&gated)
}

fn mlp_residual(
    x: &LazyTensor,
    m: &ModulationChunks,
    weights: &StreamWeights,
    cfg: &MmDitConfig,
) -> Result<LazyTensor> {
    let dim = cfg.dim;
    let mlp_hidden = dim * cfg.mlp_ratio;
    let x_norm = x.layer_norm_last_dim(cfg.eps)?;
    let x_mod = apply_modulation(&x_norm, &m.scale_mlp, &m.shift_mlp)?;
    let h1 = weights.fc1.apply_linear(&x_mod, dim, mlp_hidden);
    let h1 = h1.add_trailing_bias(Arc::clone(&weights.fc1_bias))?;
    let h1 = h1.gelu();
    let h2 = weights.fc2.apply_linear(&h1, mlp_hidden, dim);
    let h2 = h2.add_trailing_bias(Arc::clone(&weights.fc2_bias))?;
    gated_residual(x, &h2, &m.gate_mlp)
}

// ---- SingleStreamBlock ------------------------------------------------------

/// Apply one SingleStreamBlock: shared AdaLN + QKV projection over the
/// concat'd (text | image) sequence, then attention, gated residual,
/// shared MLP with gated residual.
pub fn apply_single_stream(
    joined: &LazyTensor,
    c: &LazyTensor,
    weights: &SingleStreamBlockWeights,
    cfg: &MmDitConfig,
) -> Result<LazyTensor> {
    let dim = cfg.dim;
    let num_heads = cfg.num_heads;
    let head_dim = cfg.head_dim();

    let m = compute_modulation(c, &weights.adaln_proj, &weights.adaln_bias, dim)?;

    let x_norm = joined.layer_norm_last_dim(cfg.eps)?;
    let x_mod = apply_modulation(&x_norm, &m.scale_msa, &m.shift_msa)?;

    let (q, k, v) = project_qkv(
        &x_mod, &weights.qkv_proj, &weights.qkv_bias, num_heads, head_dim,
    )?;
    let attn = attention(&q, &k, &v, head_dim)?;
    let attn_out = weights.out_proj.apply_linear(&attn, dim, dim);
    let attn_out = attn_out.add_trailing_bias(Arc::clone(&weights.out_bias))?;
    let h1 = gated_residual(joined, &attn_out, &m.gate_msa)?;

    let mlp_hidden = dim * cfg.mlp_ratio;
    let h1_norm = h1.layer_norm_last_dim(cfg.eps)?;
    let h1_mod = apply_modulation(&h1_norm, &m.scale_mlp, &m.shift_mlp)?;
    let h2 = weights.fc1.apply_linear(&h1_mod, dim, mlp_hidden);
    let h2 = h2.add_trailing_bias(Arc::clone(&weights.fc1_bias))?;
    let h2 = h2.gelu();
    let h3 = weights.fc2.apply_linear(&h2, mlp_hidden, dim);
    let h3 = h3.add_trailing_bias(Arc::clone(&weights.fc2_bias))?;
    gated_residual(&h1, &h3, &m.gate_mlp)
}

// ---- Public block helpers (aliased struct names for the spec) --------------

#[derive(Debug, Clone)]
pub struct DoubleStreamBlock {
    pub config: MmDitConfig,
    pub weights: DoubleStreamBlockWeights,
}

impl DoubleStreamBlock {
    pub fn forward(
        &self,
        txt: &LazyTensor,
        img: &LazyTensor,
        c: &LazyTensor,
    ) -> Result<(LazyTensor, LazyTensor)> {
        apply_double_stream(txt, img, c, &self.weights, &self.config)
    }
}

#[derive(Debug, Clone)]
pub struct SingleStreamBlock {
    pub config: MmDitConfig,
    pub weights: SingleStreamBlockWeights,
}

impl SingleStreamBlock {
    pub fn forward(&self, joined: &LazyTensor, c: &LazyTensor) -> Result<LazyTensor> {
        apply_single_stream(joined, c, &self.weights, &self.config)
    }
}

// ---- Safetensors loader ----------------------------------------------------

fn load_stream_weights(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    dim: usize,
    mlp_hidden: usize,
) -> Result<StreamWeights> {
    use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
    Ok(StreamWeights {
        adaln_proj: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.adaLN_modulation.1.weight"), 6 * dim, dim,
        )?,
        adaln_bias: Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.adaLN_modulation.1.bias"),
        )?),
        qkv_proj: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.attn.qkv.weight"), 3 * dim, dim,
        )?,
        qkv_bias: Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.attn.qkv.bias"),
        )?),
        out_proj: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.attn.proj.weight"), dim, dim,
        )?,
        out_bias: Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.attn.proj.bias"),
        )?),
        fc1: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.mlp.fc1.weight"), mlp_hidden, dim,
        )?,
        fc1_bias: Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.mlp.fc1.bias"),
        )?),
        fc2: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.mlp.fc2.weight"), dim, mlp_hidden,
        )?,
        fc2_bias: Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.mlp.fc2.bias"),
        )?),
    })
}

fn load_single_stream_weights(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    dim: usize,
    mlp_hidden: usize,
) -> Result<SingleStreamBlockWeights> {
    use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
    Ok(SingleStreamBlockWeights {
        adaln_proj: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.adaLN_modulation.1.weight"), 6 * dim, dim,
        )?,
        adaln_bias: Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.adaLN_modulation.1.bias"),
        )?),
        qkv_proj: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.attn.qkv.weight"), 3 * dim, dim,
        )?,
        qkv_bias: Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.attn.qkv.bias"),
        )?),
        out_proj: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.attn.proj.weight"), dim, dim,
        )?,
        out_bias: Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.attn.proj.bias"),
        )?),
        fc1: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.mlp.fc1.weight"), mlp_hidden, dim,
        )?,
        fc1_bias: Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.mlp.fc1.bias"),
        )?),
        fc2: load_transposed_matrix_preserve_dtype(
            st, &format!("{prefix}.mlp.fc2.weight"), dim, mlp_hidden,
        )?,
        fc2_bias: Arc::from(load_tensor_as_f32(
            st, &format!("{prefix}.mlp.fc2.bias"),
        )?),
    })
}

impl MmDitWeights {
    /// Walk a HuggingFace MMDiT safetensors namespace and build a
    /// `MmDitWeights` bag.
    ///
    /// `adm_in_channels` is the label-embedding input dim (caller supplies
    /// it from the upstream config — eager SD3 uses 2048).
    /// `frequency_embedding_size` is the timestep sinusoidal feature
    /// length (eager default 256).
    ///
    /// Expected names (mirrors the eager `mmdit::DiffusionTransformer`
    /// `var_builder` calls):
    /// - `t_embedder.mlp.0.{weight,bias}` → conditioning `t_fc1`
    /// - `t_embedder.mlp.2.{weight,bias}` → conditioning `t_fc2`
    /// - `y_embedder.mlp.0.{weight,bias}` → conditioning `y_fc1`
    /// - `y_embedder.mlp.2.{weight,bias}` → conditioning `y_fc2`
    /// - `joint_blocks.{i}.context_block.*` → `double_blocks[i].text.*`
    /// - `joint_blocks.{i}.x_block.*` → `double_blocks[i].image.*`
    /// - `single_blocks.{i}.*` → `single_blocks[i].*`
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &MmDitConfig,
        adm_in_channels: usize,
        frequency_embedding_size: usize,
    ) -> Result<Self> {
        use crate::lazy::{load_tensor_as_f32, load_transposed_matrix_preserve_dtype};
        let dim = cfg.dim;
        let mlp_hidden = dim * cfg.mlp_ratio;
        let conditioning = ConditioningWeights {
            t_fc1: load_transposed_matrix_preserve_dtype(
                st, "t_embedder.mlp.0.weight", dim, frequency_embedding_size,
            )?,
            t_fc1_bias: Arc::from(load_tensor_as_f32(st, "t_embedder.mlp.0.bias")?),
            t_fc2: load_transposed_matrix_preserve_dtype(
                st, "t_embedder.mlp.2.weight", dim, dim,
            )?,
            t_fc2_bias: Arc::from(load_tensor_as_f32(st, "t_embedder.mlp.2.bias")?),
            y_fc1: load_transposed_matrix_preserve_dtype(
                st, "y_embedder.mlp.0.weight", dim, adm_in_channels,
            )?,
            y_fc1_bias: Arc::from(load_tensor_as_f32(st, "y_embedder.mlp.0.bias")?),
            y_fc2: load_transposed_matrix_preserve_dtype(
                st, "y_embedder.mlp.2.weight", dim, dim,
            )?,
            y_fc2_bias: Arc::from(load_tensor_as_f32(st, "y_embedder.mlp.2.bias")?),
            frequency_embedding_size,
            adm_in_channels,
        };
        let mut double_blocks = Vec::with_capacity(cfg.depth_double);
        for i in 0..cfg.depth_double {
            double_blocks.push(DoubleStreamBlockWeights {
                text:  load_stream_weights(st, &format!("joint_blocks.{i}.context_block"), dim, mlp_hidden)?,
                image: load_stream_weights(st, &format!("joint_blocks.{i}.x_block"),       dim, mlp_hidden)?,
            });
        }
        let mut single_blocks = Vec::with_capacity(cfg.depth_single);
        for i in 0..cfg.depth_single {
            single_blocks.push(load_single_stream_weights(
                st, &format!("single_blocks.{i}"), dim, mlp_hidden,
            )?);
        }
        Ok(MmDitWeights { conditioning, double_blocks, single_blocks })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn tiny_config() -> MmDitConfig {
        MmDitConfig {
            dim: 16,
            num_heads: 4,
            depth_double: 1,
            depth_single: 1,
            mlp_ratio: 2,
            eps: 1e-6,
        }
    }

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn make_rng() -> Box<dyn FnMut() -> f32> {
        let mut s: u32 = 0xC0FFEE;
        Box::new(move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        })
    }

    fn stream_weights(dim: usize, mlp_ratio: usize, nb: &mut Box<dyn FnMut() -> f32>) -> StreamWeights {
        let mlp_h = dim * mlp_ratio;
        StreamWeights {
            adaln_proj: WeightStorage::F32(vec_of(dim * 6 * dim, &mut **nb)),
            adaln_bias: vec_of(6 * dim, &mut **nb),
            qkv_proj: WeightStorage::F32(vec_of(dim * 3 * dim, &mut **nb)),
            qkv_bias: vec_of(3 * dim, &mut **nb),
            out_proj: WeightStorage::F32(vec_of(dim * dim, &mut **nb)),
            out_bias: vec_of(dim, &mut **nb),
            fc1: WeightStorage::F32(vec_of(dim * mlp_h, &mut **nb)),
            fc1_bias: vec_of(mlp_h, &mut **nb),
            fc2: WeightStorage::F32(vec_of(mlp_h * dim, &mut **nb)),
            fc2_bias: vec_of(dim, &mut **nb),
        }
    }

    fn single_weights(dim: usize, mlp_ratio: usize, nb: &mut Box<dyn FnMut() -> f32>) -> SingleStreamBlockWeights {
        let mlp_h = dim * mlp_ratio;
        SingleStreamBlockWeights {
            adaln_proj: WeightStorage::F32(vec_of(dim * 6 * dim, &mut **nb)),
            adaln_bias: vec_of(6 * dim, &mut **nb),
            qkv_proj: WeightStorage::F32(vec_of(dim * 3 * dim, &mut **nb)),
            qkv_bias: vec_of(3 * dim, &mut **nb),
            out_proj: WeightStorage::F32(vec_of(dim * dim, &mut **nb)),
            out_bias: vec_of(dim, &mut **nb),
            fc1: WeightStorage::F32(vec_of(dim * mlp_h, &mut **nb)),
            fc1_bias: vec_of(mlp_h, &mut **nb),
            fc2: WeightStorage::F32(vec_of(mlp_h * dim, &mut **nb)),
            fc2_bias: vec_of(dim, &mut **nb),
        }
    }

    fn tiny_weights(cfg: &MmDitConfig, adm_in: usize, freq_embed: usize) -> MmDitWeights {
        let mut nb: Box<dyn FnMut() -> f32> = make_rng();
        let conditioning = ConditioningWeights {
            t_fc1: WeightStorage::F32(vec_of(freq_embed * cfg.dim, &mut *nb)),
            t_fc1_bias: vec_of(cfg.dim, &mut *nb),
            t_fc2: WeightStorage::F32(vec_of(cfg.dim * cfg.dim, &mut *nb)),
            t_fc2_bias: vec_of(cfg.dim, &mut *nb),
            y_fc1: WeightStorage::F32(vec_of(adm_in * cfg.dim, &mut *nb)),
            y_fc1_bias: vec_of(cfg.dim, &mut *nb),
            y_fc2: WeightStorage::F32(vec_of(cfg.dim * cfg.dim, &mut *nb)),
            y_fc2_bias: vec_of(cfg.dim, &mut *nb),
            frequency_embedding_size: freq_embed,
            adm_in_channels: adm_in,
        };
        let double_blocks = (0..cfg.depth_double)
            .map(|_| DoubleStreamBlockWeights {
                text: stream_weights(cfg.dim, cfg.mlp_ratio, &mut nb),
                image: stream_weights(cfg.dim, cfg.mlp_ratio, &mut nb),
            })
            .collect();
        let single_blocks = (0..cfg.depth_single)
            .map(|_| single_weights(cfg.dim, cfg.mlp_ratio, &mut nb))
            .collect();
        MmDitWeights { conditioning, double_blocks, single_blocks }
    }

    fn tiny_inputs(
        cfg: &MmDitConfig, seq_text: usize, seq_image: usize, adm_in: usize,
    ) -> (LazyTensor, LazyTensor, LazyTensor, LazyTensor) {
        let dev = Device::cpu();
        let mut s: u32 = 0xBADF00D;
        let mut rng = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.5
        };
        let txt_data: Vec<f32> = (0..(1 * seq_text * cfg.dim)).map(|_| rng()).collect();
        let img_data: Vec<f32> = (0..(1 * seq_image * cfg.dim)).map(|_| rng()).collect();
        let y_data: Vec<f32> = (0..(1 * adm_in)).map(|_| rng()).collect();
        let t_data: Vec<f32> = vec![0.5_f32];
        let txt = LazyTensor::from_f32(Arc::from(txt_data), Shape::from_dims(&[1, seq_text, cfg.dim]), &dev);
        let img = txt.const_f32_like(Arc::from(img_data), Shape::from_dims(&[1, seq_image, cfg.dim]));
        let y = txt.const_f32_like(Arc::from(y_data), Shape::from_dims(&[1, adm_in]));
        let t = txt.const_f32_like(Arc::from(t_data), Shape::from_dims(&[1]));
        (txt, img, t, y)
    }

    #[test]
    fn forward_shape_and_finite_tiny() {
        let cfg = tiny_config();
        let adm_in = 32;
        let freq_embed = 16;
        let w = tiny_weights(&cfg, adm_in, freq_embed);
        let model = MmDitModel { config: cfg.clone(), weights: w };
        let (txt, img, t, y) = tiny_inputs(&cfg, 8, 16, adm_in);
        let out = model.forward(&img, &txt, &t, &y).unwrap();
        assert_eq!(out.shape().dims(), &[1, 16, cfg.dim]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn modulation_zero_scale_is_layernorm() {
        let dev = Device::cpu();
        let b = 1;
        let s = 4;
        let dim = 8;
        let data: Vec<f32> = (0..(b * s * dim))
            .map(|i| (i as f32 * 0.137).sin())
            .collect();
        let x = LazyTensor::from_f32(Arc::from(data), Shape::from_dims(&[b, s, dim]), &dev);
        let normed = x.layer_norm_last_dim(1e-6).unwrap();
        let zero = x.const_f32_like(
            Arc::from(vec![0.0_f32; b * dim]),
            Shape::from_dims(&[b, dim]),
        );
        let modulated = apply_modulation(&normed, &zero, &zero).unwrap();
        let a = normed.realize_f32();
        let bv = modulated.realize_f32();
        assert_eq!(a.len(), bv.len());
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(bv.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff < 1e-5, "zero-scale-and-shift modulation should equal plain norm, max_diff = {max_diff}");
    }

    #[test]
    fn modulation_zero_gate_is_residual() {
        let dev = Device::cpu();
        let b = 1;
        let s = 4;
        let dim = 8;
        let x_data: Vec<f32> = (0..(b * s * dim))
            .map(|i| (i as f32 * 0.19).cos())
            .collect();
        let delta_data: Vec<f32> = (0..(b * s * dim))
            .map(|i| (i as f32 * 0.07).sin())
            .collect();
        let x = LazyTensor::from_f32(Arc::from(x_data), Shape::from_dims(&[b, s, dim]), &dev);
        let delta = x.const_f32_like(Arc::from(delta_data), Shape::from_dims(&[b, s, dim]));
        let gate = x.const_f32_like(
            Arc::from(vec![0.0_f32; b * dim]),
            Shape::from_dims(&[b, dim]),
        );
        let out = gated_residual(&x, &delta, &gate).unwrap();
        let a = x.realize_f32();
        let bv = out.realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(bv.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff < 1e-6, "zero-gate residual should equal x, max_diff = {max_diff}");
    }

    // ---- Safetensors loader round-trip --------------------------------

    fn write_tmp_safetensors(
        tensors: &[(String, Vec<usize>, Vec<f32>)],
    ) -> std::path::PathBuf {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;
        let bytes_store: Vec<Vec<u8>> = tensors.iter()
            .map(|(_, _, data)| data.iter().flat_map(|f| f.to_le_bytes()).collect())
            .collect();
        let views: HashMap<String, TensorView<'_>> = tensors.iter()
            .zip(bytes_store.iter())
            .map(|((name, shape, _), bytes)| {
                let v = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("TensorView::new");
                (name.clone(), v)
            })
            .collect();
        let metadata: Option<HashMap<String, String>> = None;
        let bytes_out = safetensors::serialize(&views, metadata).unwrap();
        let path = std::env::temp_dir().join(format!(
            "fuel_lazy_mmdit_test_{}.safetensors",
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
        ));
        std::fs::write(&path, bytes_out).unwrap();
        path
    }

    fn linear_tensors(prefix: &str, in_f: usize, out_f: usize, seed: u32)
        -> Vec<(String, Vec<usize>, Vec<f32>)>
    {
        let mut s = seed;
        let mut next = || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let w_data: Vec<f32> = (0..in_f * out_f).map(|_| next()).collect();
        let b_data: Vec<f32> = (0..out_f).map(|_| next()).collect();
        vec![
            (format!("{prefix}.weight"), vec![out_f, in_f], w_data),
            (format!("{prefix}.bias"),   vec![out_f], b_data),
        ]
    }

    /// load_from_mmapped: round-trip a minimal MMDiT config through a
    /// synthesized safetensors file and confirm the loaded weights have
    /// the right shapes.
    #[test]
    fn load_from_mmapped_round_trip_tiny() {
        let cfg = MmDitConfig {
            dim: 8, num_heads: 2, depth_double: 1, depth_single: 0,
            mlp_ratio: 2, eps: 1e-6,
        };
        let dim = cfg.dim;
        let mlp_h = dim * cfg.mlp_ratio;
        let freq_embed = 16;
        let adm_in = 32;

        let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        // Conditioning.
        tensors.extend(linear_tensors("t_embedder.mlp.0", freq_embed, dim, 1));
        tensors.extend(linear_tensors("t_embedder.mlp.2", dim, dim, 2));
        tensors.extend(linear_tensors("y_embedder.mlp.0", adm_in, dim, 3));
        tensors.extend(linear_tensors("y_embedder.mlp.2", dim, dim, 4));
        // joint_blocks.0 (DoubleStreamBlock) — text=context_block, image=x_block.
        for (which, seed_base) in [("context_block", 10), ("x_block", 50)] {
            tensors.extend(linear_tensors(&format!("joint_blocks.0.{which}.adaLN_modulation.1"), dim, 6 * dim, seed_base));
            tensors.extend(linear_tensors(&format!("joint_blocks.0.{which}.attn.qkv"), dim, 3 * dim, seed_base + 1));
            tensors.extend(linear_tensors(&format!("joint_blocks.0.{which}.attn.proj"), dim, dim, seed_base + 2));
            tensors.extend(linear_tensors(&format!("joint_blocks.0.{which}.mlp.fc1"), dim, mlp_h, seed_base + 3));
            tensors.extend(linear_tensors(&format!("joint_blocks.0.{which}.mlp.fc2"), mlp_h, dim, seed_base + 4));
        }

        let path = write_tmp_safetensors(&tensors);
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path).unwrap() };
        let weights = MmDitWeights::load_from_mmapped(&st, &cfg, adm_in, freq_embed).unwrap();
        assert_eq!(weights.double_blocks.len(), 1);
        assert_eq!(weights.single_blocks.len(), 0);
        assert_eq!(weights.conditioning.frequency_embedding_size, freq_embed);
        assert_eq!(weights.conditioning.adm_in_channels, adm_in);

        let _ = std::fs::remove_file(&path);
    }
}
