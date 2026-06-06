//! Flux (Black Forest Labs) — rectified-flow MMDiT image diffusion,
//! ported to the lazy-graph API.
//!
//! Components:
//!
//! - [`FluxModel`] — the DiT itself. `depth` DoubleStreamBlocks process
//!   text and image streams with independent AdaLN modulation and a
//!   joint (concat-K/V) attention, followed by `depth_single`
//!   SingleStreamBlocks operating on the concatenated stream with a
//!   parallel attention + MLP merge.
//! - [`FluxVae`] — the 16-channel, 8x-downsampling autoencoder. Encoder
//!   maps `(B, 3, H, W)` images to `(B, 2*z_channels, H/8, W/8)` mean
//!   + log-var latents, then `(z - shift) * scale` scales them into the
//!   DiT's input range. Decoder reverses the process.
//! - [`FlowMatchScheduler`] — linear (or shifted-linear) flow-matching
//!   schedule. Pure host scalars; no graph ops.
//! - [`QuantizedFluxModel`] — Q4_0-quantized weight variant of
//!   [`FluxModel`]. Built by [`QuantizedFluxModel::from_f32_bake`].
//! - [`generate`] — end-to-end sampling driver: takes CLIP + T5 text
//!   embeddings + a noise latent and runs `num_steps` denoising
//!   iterations against a [`FluxModel`].
//!
//! Flux specializations vs [`crate::lazy_mmdit`]:
//!
//! - QK-Norm: per-head RMSNorm applied to Q and K inside attention.
//! - Parallel attention + MLP: SingleStream block computes attention
//!   and MLP from a single fused projection, then sums their outputs.
//! - Per-stream modulation: each DoubleStreamBlock has separate `img`
//!   and `txt` AdaLN projections (6 params each per block).
//! - N-dim RoPE: positional embedding is per-axis (e.g.
//!   `axes_dim = [16, 56, 56]` partitions the head dim across three
//!   independent rotary frequency bands).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

// ---- Config -----------------------------------------------------------------

/// Hyperparameters for the Flux DiT.
#[derive(Debug, Clone)]
pub struct FluxConfig {
    pub in_channels: usize,
    pub vec_in_dim: usize,
    pub context_in_dim: usize,
    pub hidden_size: usize,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    pub depth: usize,
    pub depth_single_blocks: usize,
    pub axes_dim: Vec<usize>,
    pub theta: usize,
    pub qkv_bias: bool,
    pub guidance_embed: bool,
    /// QK-Norm gain on Q/K inside attention. Always on for Flux; the
    /// flag is kept so tests can null it out and confirm the path is
    /// actually being exercised.
    pub qk_norm: bool,
}

impl FluxConfig {
    /// FLUX.1-dev (12B, guidance-distilled).
    pub fn dev() -> Self {
        Self {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 3072,
            mlp_ratio: 4.0,
            num_heads: 24,
            depth: 19,
            depth_single_blocks: 38,
            axes_dim: vec![16, 56, 56],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: true,
            qk_norm: true,
        }
    }

    /// FLUX.1-schnell (12B, distillation-free).
    pub fn schnell() -> Self {
        Self {
            in_channels: 64,
            vec_in_dim: 768,
            context_in_dim: 4096,
            hidden_size: 3072,
            mlp_ratio: 4.0,
            num_heads: 24,
            depth: 19,
            depth_single_blocks: 38,
            axes_dim: vec![16, 56, 56],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: false,
            qk_norm: true,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }

    pub fn mlp_hidden(&self) -> usize {
        (self.hidden_size as f64 * self.mlp_ratio) as usize
    }
}

// ---- Weight containers ------------------------------------------------------

/// Linear weight + optional bias. The weight is held as a generic
/// [`WeightStorage`] so the same struct serves both the F32 [`FluxModel`]
/// and the Q4_0-quantized [`QuantizedFluxModel`].
#[derive(Debug, Clone)]
pub struct FluxLinear {
    pub weight: WeightStorage,
    pub bias: Option<Arc<[f32]>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl FluxLinear {
    fn apply(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let y = self.weight.apply_linear(x, self.in_features, self.out_features);
        match &self.bias {
            Some(b) => Ok(y.add_trailing_bias(Arc::clone(b))?),
            None => Ok(y),
        }
    }
}

/// QK-Norm: per-head RMSNorm gains for Q and K. Each gain is
/// length-`head_dim`. The forward applies `rms_norm_affine` along the
/// last dim with eps `1e-6`.
#[derive(Debug, Clone)]
pub struct FluxQkNorm {
    pub query_gain: Arc<[f32]>,
    pub key_gain: Arc<[f32]>,
}

/// Two-layer SiLU MLP embedder for the time + label conditioning.
#[derive(Debug, Clone)]
pub struct FluxMlpEmbedder {
    pub in_layer: FluxLinear,
    pub out_layer: FluxLinear,
}

impl FluxMlpEmbedder {
    fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let h = self.in_layer.apply(x)?;
        let h = h.silu();
        self.out_layer.apply(&h)
    }
}

/// Self-attention block: fused QKV projection + QK-Norm + output proj.
#[derive(Debug, Clone)]
pub struct FluxSelfAttention {
    pub qkv: FluxLinear,
    pub qk_norm: FluxQkNorm,
    pub proj: FluxLinear,
    pub num_heads: usize,
    pub head_dim: usize,
}

/// 2-layer MLP (Linear -> GELU -> Linear) used inside DoubleStreamBlock.
#[derive(Debug, Clone)]
pub struct FluxMlp {
    pub fc1: FluxLinear,
    pub fc2: FluxLinear,
}

impl FluxMlp {
    fn forward(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let h = self.fc1.apply(x)?;
        let h = h.gelu();
        self.fc2.apply(&h)
    }
}

/// AdaLN modulation projection. `Modulation1` outputs a single
/// (shift, scale, gate) chunk; `Modulation2` outputs two. The eager
/// projection is `silu(c) @ lin` followed by chunk along the trailing
/// dim.
#[derive(Debug, Clone)]
pub struct FluxModulation {
    pub lin: FluxLinear,
    pub num_chunks: usize,
}

impl FluxModulation {
    fn forward(&self, vec_c: &LazyTensor) -> Result<Vec<ModulationOut>> {
        let y = self.lin.apply(&vec_c.silu())?;
        // y has shape (B, num_chunks * dim). Unsqueeze a sequence dim
        // and split into num_chunks (shift, scale, gate) triples.
        let dims = y.shape().dims().to_vec();
        if dims.len() != 2 {
            return Err(crate::Error::Msg(format!(
                "FluxModulation::forward: expected rank-2 output, got {dims:?}",
            )).bt());
        }
        let (b, total) = (dims[0], dims[1]);
        if total % (3 * self.num_chunks) != 0 {
            return Err(crate::Error::Msg(format!(
                "FluxModulation: trailing dim {total} not divisible by 3 * num_chunks ({})",
                3 * self.num_chunks,
            )).bt());
        }
        let dim = total / (3 * self.num_chunks);
        let y = y.reshape(Shape::from_dims(&[b, 1, 3 * self.num_chunks * dim]))?;
        let chunks = y.chunk(3 * self.num_chunks, 2_usize)?;
        let mut out = Vec::with_capacity(self.num_chunks);
        for i in 0..self.num_chunks {
            out.push(ModulationOut {
                shift: chunks[3 * i].clone(),
                scale: chunks[3 * i + 1].clone(),
                gate: chunks[3 * i + 2].clone(),
            });
        }
        Ok(out)
    }
}

/// One (shift, scale, gate) modulation triple. Each tensor has shape
/// `(B, 1, dim)`.
#[derive(Debug, Clone)]
pub struct ModulationOut {
    pub shift: LazyTensor,
    pub scale: LazyTensor,
    pub gate: LazyTensor,
}

impl ModulationOut {
    fn scale_shift(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let scale_plus_one = self.scale.add_scalar(1.0);
        let scaled = x.broadcast_mul(&scale_plus_one)?;
        scaled.broadcast_add(&self.shift)
    }

    fn gate_apply(&self, x: &LazyTensor) -> Result<LazyTensor> {
        self.gate.broadcast_mul(x)
    }
}

/// DoubleStreamBlock weights: independent image + text branches.
#[derive(Debug, Clone)]
pub struct FluxDoubleStreamBlockWeights {
    pub img_mod: FluxModulation, // 2 chunks
    pub img_attn: FluxSelfAttention,
    pub img_mlp: FluxMlp,
    pub txt_mod: FluxModulation, // 2 chunks
    pub txt_attn: FluxSelfAttention,
    pub txt_mlp: FluxMlp,
}

/// SingleStreamBlock weights: fused linear1 over `(qkv | mlp_hidden)`
/// and fused linear2 over `(attn | mlp_hidden)`.
#[derive(Debug, Clone)]
pub struct FluxSingleStreamBlockWeights {
    pub linear1: FluxLinear, // (h, 3h + mlp_hidden)
    pub linear2: FluxLinear, // (h + mlp_hidden, h)
    pub qk_norm: FluxQkNorm,
    pub modulation: FluxModulation, // 1 chunk
    pub num_heads: usize,
    pub head_dim: usize,
    pub mlp_hidden: usize,
}

/// Final projection layer. Applies an AdaLN (1 chunk: shift+scale only)
/// followed by a linear.
#[derive(Debug, Clone)]
pub struct FluxLastLayer {
    pub linear: FluxLinear,
    pub ada_ln_modulation: FluxLinear, // out=2*h
}

#[derive(Debug, Clone)]
pub struct FluxWeights {
    pub img_in: FluxLinear,
    pub txt_in: FluxLinear,
    pub time_in: FluxMlpEmbedder,
    pub vector_in: FluxMlpEmbedder,
    pub guidance_in: Option<FluxMlpEmbedder>,
    pub double_blocks: Vec<FluxDoubleStreamBlockWeights>,
    pub single_blocks: Vec<FluxSingleStreamBlockWeights>,
    pub final_layer: FluxLastLayer,
}

// ---- Helpers ---------------------------------------------------------------

/// Sinusoidal embedding matching the eager `timestep_embedding` (and the
/// `flip_sin_to_cos=false` convention used by Flux): `cos` half first,
/// then `sin` half. Input `(B,)` host-side scalars are interpreted as
/// `t * 1000`. Output `(B, dim)`.
fn timestep_embedding(t: &LazyTensor, dim: usize) -> Result<LazyTensor> {
    if dim % 2 != 0 {
        return Err(crate::Error::Msg(format!(
            "timestep_embedding: dim {dim} must be even",
        )).bt());
    }
    let time_factor = 1000.0_f64;
    let max_period = 10_000.0_f64;
    let half = dim / 2;
    let dims = t.shape().dims().to_vec();
    if dims.len() != 1 {
        return Err(crate::Error::Msg(format!(
            "timestep_embedding: t must be rank-1, got {dims:?}",
        )).bt());
    }
    let batch = dims[0];
    let t_scaled = t.mul_scalar(time_factor);
    let log_mp = max_period.ln();
    let freqs: Vec<f32> = (0..half)
        .map(|i| (-log_mp * (i as f64) / (half as f64)).exp() as f32)
        .collect();
    let freqs_t = t
        .const_f32_like(Arc::from(freqs), Shape::from_dims(&[half]))
        .reshape(Shape::from_dims(&[1, half]))?
        .broadcast_to(Shape::from_dims(&[batch, half]))?;
    let t_col = t_scaled
        .reshape(Shape::from_dims(&[batch, 1]))?
        .broadcast_to(Shape::from_dims(&[batch, half]))?;
    let args = t_col.mul(&freqs_t)?;
    let cosines = args.cos();
    let sines = args.sin();
    cosines.concat(&sines, 1_usize)
}

/// Build the 2-2 RoPE table for one axis: returns
/// `(b, n, dim/2, 2, 2)` where the trailing 2x2 is the rotation matrix
/// `[[cos, -sin], [sin, cos]]`.
fn rope_axis(pos: &LazyTensor, dim: usize, theta: usize) -> Result<LazyTensor> {
    if dim % 2 != 0 {
        return Err(crate::Error::Msg(format!(
            "rope_axis: dim {dim} must be even",
        )).bt());
    }
    let theta_f = theta as f64;
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (1.0 / theta_f.powf((2 * i) as f64 / dim as f64)) as f32)
        .collect();
    let pos_dims = pos.shape().dims().to_vec();
    if pos_dims.len() != 2 {
        return Err(crate::Error::Msg(format!(
            "rope_axis: pos must be rank-2 (B, N), got {pos_dims:?}",
        )).bt());
    }
    let (b, n) = (pos_dims[0], pos_dims[1]);
    let inv_freq_t = pos
        .const_f32_like(Arc::from(inv_freq), Shape::from_dims(&[half]))
        .reshape(Shape::from_dims(&[1, 1, half]))?
        .broadcast_to(Shape::from_dims(&[b, n, half]))?;
    let pos_bnh = pos
        .reshape(Shape::from_dims(&[b, n, 1]))?
        .broadcast_to(Shape::from_dims(&[b, n, half]))?;
    let freqs = pos_bnh.mul(&inv_freq_t)?;
    let cos = freqs.cos();
    let sin = freqs.sin();
    let neg_sin = sin.mul_scalar(-1.0);
    // Stack [cos, -sin, sin, cos] along a new dim, then reshape into
    // the (b, n, half, 2, 2) rotation-block layout.
    let stacked = LazyTensor::stack(&[&cos, &neg_sin, &sin, &cos], 3_usize)?;
    // stacked shape: (b, n, half, 4)
    stacked.reshape(Shape::from_dims(&[b, n, half, 2, 2]))
}

/// N-axis RoPE: for each axis index i in `ids[.., :, i]`, build a rope
/// table over the corresponding `axes_dim[i]`, then concat along the
/// half-dim. Returns the joint RoPE freq tensor with a head-dim of size
/// 1 unsqueezed at position 1 — shape
/// `(B, 1, N, sum(axes_dim)/2, 2, 2)`.
fn embed_nd(ids: &LazyTensor, axes_dim: &[usize], theta: usize) -> Result<LazyTensor> {
    let dims = ids.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(crate::Error::Msg(format!(
            "embed_nd: ids must be rank-3 (B, N, n_axes), got {dims:?}",
        )).bt());
    }
    let n_axes = dims[2];
    if n_axes != axes_dim.len() {
        return Err(crate::Error::Msg(format!(
            "embed_nd: ids trailing dim {n_axes} != axes_dim.len() {}",
            axes_dim.len(),
        )).bt());
    }
    let mut per_axis: Vec<LazyTensor> = Vec::with_capacity(n_axes);
    for i in 0..n_axes {
        let pos = ids.narrow(2_usize, i, 1)?.squeeze(2_usize)?;
        per_axis.push(rope_axis(&pos, axes_dim[i], theta)?);
    }
    // Concatenate along the `half` dim (dim 2 of each (b, n, half_i,
    // 2, 2) tensor).
    let mut emb = per_axis[0].clone();
    for next in per_axis.iter().skip(1) {
        emb = emb.concat(next, 2_usize)?;
    }
    // Unsqueeze head-dim at position 1: (b, 1, n, half_total, 2, 2).
    emb.unsqueeze(1_usize)
}

/// Apply RoPE to a query/key tensor of shape `(B, H, S, D)`.
/// `pe` has shape `(B, 1, S, D/2, 2, 2)`.
fn apply_rope(x: &LazyTensor, pe: &LazyTensor) -> Result<LazyTensor> {
    let x_dims = x.shape().dims().to_vec();
    if x_dims.len() != 4 {
        return Err(crate::Error::Msg(format!(
            "apply_rope: x must be rank-4 (B, H, S, D), got {x_dims:?}",
        )).bt());
    }
    let (b, h, s, d) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    if d % 2 != 0 {
        return Err(crate::Error::Msg(format!(
            "apply_rope: head dim {d} must be even",
        )).bt());
    }
    // x: (B, H, S, D/2, 2)
    let x5 = x.reshape(Shape::from_dims(&[b, h, s, d / 2, 2]))?;
    // narrow last dim to get the two "phases".
    let x0 = x5.narrow(4_usize, 0, 1)?; // (B, H, S, D/2, 1)
    let x1 = x5.narrow(4_usize, 1, 1)?; // (B, H, S, D/2, 1)
    // pe slices: get_on_dim(-1, k) drops the last dim.
    let pe_dims = pe.shape().dims().to_vec();
    if pe_dims.len() != 6 {
        return Err(crate::Error::Msg(format!(
            "apply_rope: pe must be rank-6 (B, 1, S, D/2, 2, 2), got {pe_dims:?}",
        )).bt());
    }
    let fr0 = pe.narrow(5_usize, 0, 1)?.squeeze(5_usize)?; // (B, 1, S, D/2, 2)
    let fr1 = pe.narrow(5_usize, 1, 1)?.squeeze(5_usize)?; // (B, 1, S, D/2, 2)
    // Broadcast pe (head=1) and x (head=h) to (B, H, S, D/2, 2).
    let target = Shape::from_dims(&[b, h, s, d / 2, 2]);
    let fr0_bc = fr0.broadcast_to(target.clone())?;
    let fr1_bc = fr1.broadcast_to(target.clone())?;
    // x0 / x1 are (B, H, S, D/2, 1); broadcast across the last 2.
    let x0_bc = x0.broadcast_to(target.clone())?;
    let x1_bc = x1.broadcast_to(target.clone())?;
    let out = fr0_bc.mul(&x0_bc)?.add(&fr1_bc.mul(&x1_bc)?)?;
    out.reshape(Shape::from_dims(&[b, h, s, d]))
}

/// Apply rope to q,k then run scaled-dot-product attention. Returns
/// `(B, S, H*D)`.
fn attention(
    q: &LazyTensor, k: &LazyTensor, v: &LazyTensor, pe: &LazyTensor, head_dim: usize,
) -> Result<LazyTensor> {
    let q = apply_rope(q, pe)?;
    let k = apply_rope(k, pe)?;
    let k_t = k.transpose()?;
    let scale = 1.0_f64 / (head_dim as f64).sqrt();
    let scores = q.matmul(&k_t)?.mul_scalar(scale);
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(v)?;
    // ctx: (B, H, S, D) → merge_heads → (B, S, H*D)
    ctx.merge_heads()
}

/// Split a fused QKV projection `(B, S, 3*dim)` into three head-major
/// tensors `(B, H, S, head_dim)` with QK-Norm applied to Q and K.
fn split_qkv_with_qknorm(
    qkv: &LazyTensor,
    num_heads: usize,
    head_dim: usize,
    qk_norm: &FluxQkNorm,
    qk_norm_enabled: bool,
) -> Result<(LazyTensor, LazyTensor, LazyTensor)> {
    let dims = qkv.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(crate::Error::Msg(format!(
            "split_qkv_with_qknorm: input must be rank-3, got {dims:?}",
        )).bt());
    }
    let dim = num_heads * head_dim;
    let q = qkv.narrow(2_usize, 0, dim)?.split_heads(num_heads, head_dim)?;
    let k = qkv.narrow(2_usize, dim, dim)?.split_heads(num_heads, head_dim)?;
    let v = qkv.narrow(2_usize, 2 * dim, dim)?.split_heads(num_heads, head_dim)?;
    let (q, k) = if qk_norm_enabled {
        let q = q.rms_norm_affine(Arc::clone(&qk_norm.query_gain), 1e-6)?;
        let k = k.rms_norm_affine(Arc::clone(&qk_norm.key_gain), 1e-6)?;
        (q, k)
    } else {
        (q, k)
    };
    Ok((q, k, v))
}

// ---- DoubleStreamBlock forward ---------------------------------------------

fn apply_double_stream(
    img: &LazyTensor,
    txt: &LazyTensor,
    vec_c: &LazyTensor,
    pe: &LazyTensor,
    blk: &FluxDoubleStreamBlockWeights,
    cfg: &FluxConfig,
) -> Result<(LazyTensor, LazyTensor)> {
    let img_mods = blk.img_mod.forward(vec_c)?;
    let txt_mods = blk.txt_mod.forward(vec_c)?;
    if img_mods.len() != 2 || txt_mods.len() != 2 {
        return Err(crate::Error::Msg(
            "apply_double_stream: each modulation must produce 2 chunks".into(),
        ).bt());
    }
    let (img_mod1, img_mod2) = (&img_mods[0], &img_mods[1]);
    let (txt_mod1, txt_mod2) = (&txt_mods[0], &txt_mods[1]);

    // Pre-attn norm + modulation + QKV projection per stream.
    let img_n = img.layer_norm_last_dim(1e-6)?;
    let img_n = img_mod1.scale_shift(&img_n)?;
    let img_qkv = blk.img_attn.qkv.apply(&img_n)?;
    let (img_q, img_k, img_v) = split_qkv_with_qknorm(
        &img_qkv, blk.img_attn.num_heads, blk.img_attn.head_dim,
        &blk.img_attn.qk_norm, cfg.qk_norm,
    )?;

    let txt_n = txt.layer_norm_last_dim(1e-6)?;
    let txt_n = txt_mod1.scale_shift(&txt_n)?;
    let txt_qkv = blk.txt_attn.qkv.apply(&txt_n)?;
    let (txt_q, txt_k, txt_v) = split_qkv_with_qknorm(
        &txt_qkv, blk.txt_attn.num_heads, blk.txt_attn.head_dim,
        &blk.txt_attn.qk_norm, cfg.qk_norm,
    )?;

    // Joint attention: cat over the sequence axis (dim 2 in the
    // head-major (B, H, S, D) layout).
    let q = txt_q.concat(&img_q, 2_usize)?;
    let k = txt_k.concat(&img_k, 2_usize)?;
    let v = txt_v.concat(&img_v, 2_usize)?;
    let attn_all = attention(&q, &k, &v, pe, blk.img_attn.head_dim)?;
    // attn_all: (B, S_total, H*D). Split back per stream.
    let s_txt = txt.shape().dims()[1];
    let total = attn_all.shape().dims()[1];
    let s_img = total - s_txt;
    let txt_attn = attn_all.narrow(1_usize, 0, s_txt)?;
    let img_attn = attn_all.narrow(1_usize, s_txt, s_img)?;

    // Attention output projection + gated residual + MLP residual.
    let img_attn_proj = blk.img_attn.proj.apply(&img_attn)?;
    let img_h1 = img.add(&img_mod1.gate_apply(&img_attn_proj)?)?;
    let img_n2 = img_h1.layer_norm_last_dim(1e-6)?;
    let img_n2 = img_mod2.scale_shift(&img_n2)?;
    let img_mlp = blk.img_mlp.forward(&img_n2)?;
    let img_out = img_h1.add(&img_mod2.gate_apply(&img_mlp)?)?;

    let txt_attn_proj = blk.txt_attn.proj.apply(&txt_attn)?;
    let txt_h1 = txt.add(&txt_mod1.gate_apply(&txt_attn_proj)?)?;
    let txt_n2 = txt_h1.layer_norm_last_dim(1e-6)?;
    let txt_n2 = txt_mod2.scale_shift(&txt_n2)?;
    let txt_mlp = blk.txt_mlp.forward(&txt_n2)?;
    let txt_out = txt_h1.add(&txt_mod2.gate_apply(&txt_mlp)?)?;

    Ok((img_out, txt_out))
}

// ---- SingleStreamBlock forward ---------------------------------------------

/// Parallel attention + MLP block: one linear projects to
/// `(qkv | mlp_hidden)`; the qkv part feeds attention while the
/// mlp_hidden part feeds a GELU. Their outputs are concatenated and
/// projected back by `linear2`, then added to the input with a gate.
fn apply_single_stream(
    xs: &LazyTensor,
    vec_c: &LazyTensor,
    pe: &LazyTensor,
    blk: &FluxSingleStreamBlockWeights,
    cfg: &FluxConfig,
) -> Result<LazyTensor> {
    let mods = blk.modulation.forward(vec_c)?;
    if mods.len() != 1 {
        return Err(crate::Error::Msg(
            "apply_single_stream: modulation must produce 1 chunk".into(),
        ).bt());
    }
    let m = &mods[0];
    let h = blk.num_heads * blk.head_dim;
    let x_norm = xs.layer_norm_last_dim(1e-6)?;
    let x_mod = m.scale_shift(&x_norm)?;
    let proj = blk.linear1.apply(&x_mod)?;
    let qkv = proj.narrow(2_usize, 0, 3 * h)?;
    let mlp_part = proj.narrow(2_usize, 3 * h, blk.mlp_hidden)?;
    let (q, k, v) = split_qkv_with_qknorm(
        &qkv, blk.num_heads, blk.head_dim, &blk.qk_norm, cfg.qk_norm,
    )?;
    let attn = attention(&q, &k, &v, pe, blk.head_dim)?;
    let mlp = mlp_part.gelu();
    let merged = attn.concat(&mlp, 2_usize)?;
    let out = blk.linear2.apply(&merged)?;
    xs.add(&m.gate_apply(&out)?)
}

// ---- LastLayer -------------------------------------------------------------

fn apply_last_layer(
    xs: &LazyTensor, vec_c: &LazyTensor, last: &FluxLastLayer,
) -> Result<LazyTensor> {
    let h = vec_c.silu();
    let proj = last.ada_ln_modulation.apply(&h)?;
    let chunks = proj.chunk(2, 1_usize)?;
    if chunks.len() != 2 {
        return Err(crate::Error::Msg(format!(
            "apply_last_layer: expected 2 chunks from adaLN, got {}",
            chunks.len(),
        )).bt());
    }
    let shift = chunks[0].unsqueeze(1_usize)?;
    let scale = chunks[1].unsqueeze(1_usize)?;
    let xn = xs.layer_norm_last_dim(1e-6)?;
    let scale_plus_one = scale.add_scalar(1.0);
    let xs = xn.broadcast_mul(&scale_plus_one)?.broadcast_add(&shift)?;
    last.linear.apply(&xs)
}

// ---- Model -----------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FluxModel {
    pub config: FluxConfig,
    pub weights: FluxWeights,
}

impl FluxModel {
    /// Forward over an already-packed image patch tensor `(B, S_image,
    /// in_channels)` plus text tokens `(B, S_text, context_in_dim)`,
    /// per-token RoPE id tensors `img_ids` / `txt_ids` of shape `(B,
    /// S_*, n_axes)`, a scalar timestep `(B,)`, a pooled-text vector
    /// `(B, vec_in_dim)`, and an optional guidance scalar `(B,)`.
    /// Returns `(B, S_image, in_channels)`.
    pub fn forward(
        &self,
        img: &LazyTensor,
        img_ids: &LazyTensor,
        txt: &LazyTensor,
        txt_ids: &LazyTensor,
        timesteps: &LazyTensor,
        y: &LazyTensor,
        guidance: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        if txt.rank() != 3 {
            return Err(crate::Error::Msg(format!(
                "FluxModel::forward: txt must be rank-3, got {}", txt.rank(),
            )).bt());
        }
        if img.rank() != 3 {
            return Err(crate::Error::Msg(format!(
                "FluxModel::forward: img must be rank-3, got {}", img.rank(),
            )).bt());
        }

        // Concatenate ids over the sequence axis (dim 1) → run rope.
        let ids = txt_ids.concat(img_ids, 1_usize)?;
        let pe = embed_nd(&ids, &cfg.axes_dim, cfg.theta)?;

        let mut txt_h = self.weights.txt_in.apply(txt)?;
        let mut img_h = self.weights.img_in.apply(img)?;

        let t_emb = timestep_embedding(timesteps, 256)?;
        let mut vec_c = self.weights.time_in.forward(&t_emb)?;
        if let (Some(gw), Some(g)) = (self.weights.guidance_in.as_ref(), guidance) {
            let g_emb = timestep_embedding(g, 256)?;
            vec_c = vec_c.add(&gw.forward(&g_emb)?)?;
        }
        vec_c = vec_c.add(&self.weights.vector_in.forward(y)?)?;

        for blk in &self.weights.double_blocks {
            let (new_img, new_txt) =
                apply_double_stream(&img_h, &txt_h, &vec_c, &pe, blk, cfg)?;
            img_h = new_img;
            txt_h = new_txt;
        }

        let s_txt = txt_h.shape().dims()[1];
        let mut joined = txt_h.concat(&img_h, 1_usize)?;
        for blk in &self.weights.single_blocks {
            joined = apply_single_stream(&joined, &vec_c, &pe, blk, cfg)?;
        }
        let total = joined.shape().dims()[1];
        let img_out = joined.narrow(1_usize, s_txt, total - s_txt)?;
        apply_last_layer(&img_out, &vec_c, &self.weights.final_layer)
    }
}

// ---- Quantized model -------------------------------------------------------

/// Q4_0-quantized variant of [`FluxModel`]. The shape of every weight
/// is identical; only [`FluxLinear::weight`] switches storage variant
/// for the matmul-heavy Linears. Biases, norm gains, and any non-Linear
/// tensors stay in F32 (mirrors the GGUF convention).
#[derive(Debug, Clone)]
pub struct QuantizedFluxModel {
    inner: FluxModel,
}

impl QuantizedFluxModel {
    /// Forward over text + image tokens with Q4_0 weights. Mirrors
    /// [`FluxModel::forward`] exactly.
    pub fn forward(
        &self,
        img: &LazyTensor,
        img_ids: &LazyTensor,
        txt: &LazyTensor,
        txt_ids: &LazyTensor,
        timesteps: &LazyTensor,
        y: &LazyTensor,
        guidance: Option<&LazyTensor>,
    ) -> Result<LazyTensor> {
        self.inner.forward(img, img_ids, txt, txt_ids, timesteps, y, guidance)
    }

    pub fn inner(&self) -> &FluxModel { &self.inner }

    /// Quantize every Linear weight in a source [`FluxModel`] to Q4_0
    /// and return the wrapped model. Linears whose `in_features` are
    /// not a multiple of 32 cannot be Q4_0-quantized (block size = 32);
    /// such Linears are passed through as F32 — matches llama.cpp's
    /// fallback policy for "small" weights inside diffusion models.
    pub fn from_f32_bake(model: FluxModel) -> Result<Self> {
        let mut model = model;
        bake_weights(&mut model.weights)?;
        Ok(Self { inner: model })
    }
}

fn bake_weights(w: &mut FluxWeights) -> Result<()> {
    bake_linear(&mut w.img_in)?;
    bake_linear(&mut w.txt_in)?;
    bake_mlp_embedder(&mut w.time_in)?;
    bake_mlp_embedder(&mut w.vector_in)?;
    if let Some(g) = w.guidance_in.as_mut() {
        bake_mlp_embedder(g)?;
    }
    for blk in &mut w.double_blocks {
        bake_linear(&mut blk.img_mod.lin)?;
        bake_linear(&mut blk.img_attn.qkv)?;
        bake_linear(&mut blk.img_attn.proj)?;
        bake_linear(&mut blk.img_mlp.fc1)?;
        bake_linear(&mut blk.img_mlp.fc2)?;
        bake_linear(&mut blk.txt_mod.lin)?;
        bake_linear(&mut blk.txt_attn.qkv)?;
        bake_linear(&mut blk.txt_attn.proj)?;
        bake_linear(&mut blk.txt_mlp.fc1)?;
        bake_linear(&mut blk.txt_mlp.fc2)?;
    }
    for blk in &mut w.single_blocks {
        bake_linear(&mut blk.linear1)?;
        bake_linear(&mut blk.linear2)?;
        bake_linear(&mut blk.modulation.lin)?;
    }
    bake_linear(&mut w.final_layer.linear)?;
    bake_linear(&mut w.final_layer.ada_ln_modulation)?;
    Ok(())
}

fn bake_mlp_embedder(m: &mut FluxMlpEmbedder) -> Result<()> {
    bake_linear(&mut m.in_layer)?;
    bake_linear(&mut m.out_layer)?;
    Ok(())
}

fn bake_linear(l: &mut FluxLinear) -> Result<()> {
    if l.in_features % 32 != 0 {
        return Ok(()); // leave as F32
    }
    let f32_in_out = match &l.weight {
        WeightStorage::F32(a) => a.to_vec(),
        WeightStorage::Q4_0 { .. } => return Ok(()),
        other => {
            let _ = other;
            return Err(crate::Error::Msg(
                "bake_linear: source weight must be F32 or already Q4_0".into(),
            ).bt());
        }
    };
    l.weight = quantize_in_out_to_q4_0(&f32_in_out, l.in_features, l.out_features)?;
    Ok(())
}

fn quantize_in_out_to_q4_0(
    f32_in_out: &[f32], in_features: usize, out_features: usize,
) -> Result<WeightStorage> {
    use fuel_quantized::{BlockQ4_0, GgmlType};
    const QK4_0: usize = 32;
    if in_features % QK4_0 != 0 {
        return Err(crate::Error::Msg(format!(
            "quantize_in_out_to_q4_0: in_features ({in_features}) must be divisible by {QK4_0}",
        )).bt());
    }
    let mut f32_out_in = vec![0.0_f32; out_features * in_features];
    for o in 0..out_features {
        for j in 0..in_features {
            f32_out_in[o * in_features + j] = f32_in_out[j * out_features + o];
        }
    }
    let n_blocks = out_features * in_features / QK4_0;
    let mut blocks: Vec<BlockQ4_0> = vec![BlockQ4_0::zeros(); n_blocks];
    BlockQ4_0::from_float(&f32_out_in, &mut blocks);
    let bytes_len = n_blocks * std::mem::size_of::<BlockQ4_0>();
    let byte_slice: &[u8] = unsafe {
        std::slice::from_raw_parts(blocks.as_ptr() as *const u8, bytes_len)
    };
    let padded_len = bytes_len.div_ceil(4) * 4;
    let mut padded = vec![0_u8; padded_len];
    padded[..bytes_len].copy_from_slice(byte_slice);
    let words: Vec<u32> = padded.chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(WeightStorage::Q4_0 {
        words: Arc::from(words),
        bytes_len,
        in_features,
        out_features,
    })
}

// ---- Autoencoder -----------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FluxVaeConfig {
    pub resolution: usize,
    pub in_channels: usize,
    pub ch: usize,
    pub out_ch: usize,
    pub ch_mult: Vec<usize>,
    pub num_res_blocks: usize,
    pub z_channels: usize,
    pub scale_factor: f64,
    pub shift_factor: f64,
    /// GroupNorm groups (always 32 in the Flux release).
    pub norm_num_groups: usize,
    pub norm_eps: f64,
}

impl FluxVaeConfig {
    /// FLUX.1-dev / schnell VAE config.
    pub fn dev() -> Self {
        Self {
            resolution: 256,
            in_channels: 3,
            ch: 128,
            out_ch: 3,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 2,
            z_channels: 16,
            scale_factor: 0.3611,
            shift_factor: 0.1159,
            norm_num_groups: 32,
            norm_eps: 1e-6,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VaeResnetWeights {
    pub n1_g: Arc<[f32]>, pub n1_b: Arc<[f32]>,
    pub c1_w: Arc<[f32]>, pub c1_b: Arc<[f32]>,
    pub n2_g: Arc<[f32]>, pub n2_b: Arc<[f32]>,
    pub c2_w: Arc<[f32]>, pub c2_b: Arc<[f32]>,
    pub shortcut_w: Option<Arc<[f32]>>,
    pub shortcut_b: Option<Arc<[f32]>>,
    pub in_channels: usize,
    pub out_channels: usize,
}

#[derive(Debug, Clone)]
pub struct VaeAttnWeights {
    pub gn_g: Arc<[f32]>, pub gn_b: Arc<[f32]>,
    pub q_w: Arc<[f32]>, pub q_b: Arc<[f32]>,
    pub k_w: Arc<[f32]>, pub k_b: Arc<[f32]>,
    pub v_w: Arc<[f32]>, pub v_b: Arc<[f32]>,
    pub out_w: Arc<[f32]>, pub out_b: Arc<[f32]>,
    pub channels: usize,
}

#[derive(Debug, Clone)]
pub struct VaeDownBlock {
    pub resnets: Vec<VaeResnetWeights>,
    pub downsample_conv: Option<(Arc<[f32]>, Arc<[f32]>)>,
}

#[derive(Debug, Clone)]
pub struct VaeUpBlock {
    pub resnets: Vec<VaeResnetWeights>,
    pub upsample_conv: Option<(Arc<[f32]>, Arc<[f32]>)>,
}

#[derive(Debug, Clone)]
pub struct FluxVaeEncoderWeights {
    pub conv_in_w: Arc<[f32]>, pub conv_in_b: Arc<[f32]>,
    pub down: Vec<VaeDownBlock>,
    pub mid1: VaeResnetWeights,
    pub mid_attn: VaeAttnWeights,
    pub mid2: VaeResnetWeights,
    pub norm_out_g: Arc<[f32]>, pub norm_out_b: Arc<[f32]>,
    pub conv_out_w: Arc<[f32]>, pub conv_out_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct FluxVaeDecoderWeights {
    pub conv_in_w: Arc<[f32]>, pub conv_in_b: Arc<[f32]>,
    pub mid1: VaeResnetWeights,
    pub mid_attn: VaeAttnWeights,
    pub mid2: VaeResnetWeights,
    pub up: Vec<VaeUpBlock>,
    pub norm_out_g: Arc<[f32]>, pub norm_out_b: Arc<[f32]>,
    pub conv_out_w: Arc<[f32]>, pub conv_out_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct FluxVae {
    pub config: FluxVaeConfig,
    pub encoder: FluxVaeEncoderWeights,
    pub decoder: FluxVaeDecoderWeights,
}

impl FluxVae {
    /// Encode an image `(B, in_channels, H, W)` to the Flux latent
    /// space `(B, z_channels, H/8, W/8)`. Takes the diagonal-Gaussian
    /// mean (deterministic — sampling is left as a follow-up if a
    /// caller needs stochastic encodes).
    pub fn encode(&self, x: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let w = &self.encoder;
        let mut h = conv2d_k3_s1_p1(x, &w.conv_in_w, &w.conv_in_b, cfg.in_channels, cfg.ch)?;
        let mut cur_c = cfg.ch;
        let last_idx = cfg.ch_mult.len() - 1;
        for (i_level, blk) in w.down.iter().enumerate() {
            let block_out = cfg.ch * cfg.ch_mult[i_level];
            for r in &blk.resnets {
                h = vae_resnet(&h, r, cfg)?;
            }
            cur_c = block_out;
            if let Some((cw, cb)) = &blk.downsample_conv {
                h = downsample_conv(&h, cw, cb, cur_c)?;
            }
            let _ = i_level == last_idx;
        }
        h = vae_resnet(&h, &w.mid1, cfg)?;
        h = vae_spatial_attention(&h, &w.mid_attn, cfg)?;
        h = vae_resnet(&h, &w.mid2, cfg)?;
        let h_dims = h.shape().dims().to_vec();
        let (b, c, hh, ww) = (h_dims[0], h_dims[1], h_dims[2], h_dims[3]);
        let _ = b;
        let h = group_norm(&h, &w.norm_out_g, &w.norm_out_b, cfg.norm_num_groups, cfg.norm_eps, c, hh, ww)?;
        let h = h.silu();
        let h = conv2d_k3_s1_p1(&h, &w.conv_out_w, &w.conv_out_b, c, 2 * cfg.z_channels)?;
        // Take the mean (first z_channels) — deterministic encode.
        let mean = h.narrow(1_usize, 0, cfg.z_channels)?;
        let shifted = mean.add_scalar(-cfg.shift_factor);
        Ok(shifted.mul_scalar(cfg.scale_factor))
    }

    /// Decode latents `(B, z_channels, H_lat, W_lat)` back to image
    /// space `(B, out_ch, 8 * H_lat, 8 * W_lat)`.
    pub fn decode(&self, z: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let w = &self.decoder;
        let z = z.mul_scalar(1.0 / cfg.scale_factor);
        let z = z.add_scalar(cfg.shift_factor);
        let block_in = cfg.ch * cfg.ch_mult.last().copied().unwrap_or(1);
        let mut h = conv2d_k3_s1_p1(&z, &w.conv_in_w, &w.conv_in_b, cfg.z_channels, block_in)?;
        h = vae_resnet(&h, &w.mid1, cfg)?;
        h = vae_spatial_attention(&h, &w.mid_attn, cfg)?;
        h = vae_resnet(&h, &w.mid2, cfg)?;
        // up_blocks are stored in DECODER order (level 0 = innermost,
        // matches the original Flux eager implementation's `up.reverse()`).
        // We iterate from highest level (matching cfg.ch_mult.last())
        // down to level 0 (matching cfg.ch_mult.first()).
        let mut cur_c = block_in;
        for blk in w.up.iter().rev() {
            for r in &blk.resnets {
                h = vae_resnet(&h, r, cfg)?;
            }
            cur_c = blk.resnets.last().map(|r| r.out_channels).unwrap_or(cur_c);
            if let Some((cw, cb)) = &blk.upsample_conv {
                h = upsample_conv(&h, cw, cb, cur_c)?;
            }
        }
        let h_dims = h.shape().dims().to_vec();
        let (b, c, hh, ww) = (h_dims[0], h_dims[1], h_dims[2], h_dims[3]);
        let _ = b;
        let h = group_norm(&h, &w.norm_out_g, &w.norm_out_b, cfg.norm_num_groups, cfg.norm_eps, c, hh, ww)?;
        let h = h.silu();
        conv2d_k3_s1_p1(&h, &w.conv_out_w, &w.conv_out_b, c, cfg.out_ch)
    }
}

// ---- VAE primitives --------------------------------------------------------

fn vae_resnet(
    x: &LazyTensor, rw: &VaeResnetWeights, cfg: &FluxVaeConfig,
) -> Result<LazyTensor> {
    let dims = x.shape().dims().to_vec();
    let (b, c_in, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let _ = b;
    let c_out = rw.out_channels;
    let h1 = group_norm(x, &rw.n1_g, &rw.n1_b, cfg.norm_num_groups, cfg.norm_eps, c_in, h, w)?;
    let h1 = h1.silu();
    let h1 = conv2d_k3_s1_p1(&h1, &rw.c1_w, &rw.c1_b, c_in, c_out)?;
    let h2 = group_norm(&h1, &rw.n2_g, &rw.n2_b, cfg.norm_num_groups, cfg.norm_eps, c_out, h, w)?;
    let h2 = h2.silu();
    let h2 = conv2d_k3_s1_p1(&h2, &rw.c2_w, &rw.c2_b, c_out, c_out)?;
    let shortcut = match (&rw.shortcut_w, &rw.shortcut_b) {
        (Some(sw), Some(sb)) => conv2d_k1_s1_p0(x, sw, sb, c_in, c_out)?,
        _ => x.clone(),
    };
    shortcut.add(&h2)
}

/// Self-attention over `H*W` positions with 1x1 conv projections.
fn vae_spatial_attention(
    x: &LazyTensor, aw: &VaeAttnWeights, cfg: &FluxVaeConfig,
) -> Result<LazyTensor> {
    let dims = x.shape().dims().to_vec();
    let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
    let xn = group_norm(x, &aw.gn_g, &aw.gn_b, cfg.norm_num_groups, cfg.norm_eps, c, h, w)?;
    let q = conv2d_k1_s1_p0(&xn, &aw.q_w, &aw.q_b, c, c)?;
    let k = conv2d_k1_s1_p0(&xn, &aw.k_w, &aw.k_b, c, c)?;
    let v = conv2d_k1_s1_p0(&xn, &aw.v_w, &aw.v_b, c, c)?;
    // (B, C, H, W) → (B, H*W, C)
    let n = h * w;
    let to_seq = |t: &LazyTensor| -> Result<LazyTensor> {
        Ok(t.reshape(Shape::from_dims(&[b, c, n]))?.permute([0, 2, 1_usize])?)
    };
    let q = to_seq(&q)?;
    let k = to_seq(&k)?;
    let v = to_seq(&v)?;
    let k_t = k.transpose()?;
    let scale = 1.0_f64 / (c as f64).sqrt();
    let scores = q.matmul(&k_t)?.mul_scalar(scale);
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v)?; // (B, N, C)
    // Back to (B, C, H, W)
    let ctx_chw = ctx.permute([0, 2, 1_usize])?.reshape(Shape::from_dims(&[b, c, h, w]))?;
    let proj = conv2d_k1_s1_p0(&ctx_chw, &aw.out_w, &aw.out_b, c, c)?;
    x.add(&proj)
}

/// Downsample: pad right + bottom by 1, then stride-2 3x3 conv. Eager
/// uses asymmetric padding to keep parity with the upstream Flux Python.
fn downsample_conv(
    x: &LazyTensor, w: &Arc<[f32]>, b: &Arc<[f32]>, c: usize,
) -> Result<LazyTensor> {
    let x = x.pad_with_zeros(3_usize, 0, 1)?;
    let x = x.pad_with_zeros(2_usize, 0, 1)?;
    let w_t = x.const_f32_like(Arc::clone(w), Shape::from_dims(&[c, c, 3, 3]));
    let b_t = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[c]));
    x.conv2d(&w_t, Some(&b_t), (2, 2), (0, 0), 1)
}

/// Upsample: 2x nearest then 3x3 stride-1 padding-1 conv.
fn upsample_conv(
    x: &LazyTensor, w: &Arc<[f32]>, b: &Arc<[f32]>, c: usize,
) -> Result<LazyTensor> {
    let x = x.upsample_nearest2d(2)?;
    let w_t = x.const_f32_like(Arc::clone(w), Shape::from_dims(&[c, c, 3, 3]));
    let b_t = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[c]));
    x.conv2d(&w_t, Some(&b_t), (1, 1), (1, 1), 1)
}

fn conv2d_k3_s1_p1(
    x: &LazyTensor, w: &Arc<[f32]>, b: &Arc<[f32]>, cin: usize, cout: usize,
) -> Result<LazyTensor> {
    let w_t = x.const_f32_like(Arc::clone(w), Shape::from_dims(&[cout, cin, 3, 3]));
    let b_t = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[cout]));
    x.conv2d(&w_t, Some(&b_t), (1, 1), (1, 1), 1)
}

fn conv2d_k1_s1_p0(
    x: &LazyTensor, w: &Arc<[f32]>, b: &Arc<[f32]>, cin: usize, cout: usize,
) -> Result<LazyTensor> {
    let w_t = x.const_f32_like(Arc::clone(w), Shape::from_dims(&[cout, cin, 1, 1]));
    let b_t = x.const_f32_like(Arc::clone(b), Shape::from_dims(&[cout]));
    x.conv2d(&w_t, Some(&b_t), (1, 1), (0, 0), 1)
}

fn group_norm(
    x: &LazyTensor, gamma: &Arc<[f32]>, beta: &Arc<[f32]>,
    groups: usize, eps: f64,
    c: usize, h: usize, w: usize,
) -> Result<LazyTensor> {
    if c % groups != 0 {
        return Err(crate::Error::Msg(format!(
            "group_norm: C={c} not divisible by groups={groups}",
        )).bt());
    }
    let dims = x.shape().dims().to_vec();
    let b = dims[0];
    let cpg = c / groups;
    let m = cpg * h * w;
    let x_flat = x.reshape(Shape::from_dims(&[b, groups, m]))?;
    let mean = x_flat.mean_dim(2_usize)?;
    let mean_bc = mean
        .reshape(Shape::from_dims(&[b, groups, 1]))?
        .broadcast_to(Shape::from_dims(&[b, groups, m]))?;
    let centered = x_flat.sub(&mean_bc)?;
    let sq = centered.mul(&centered)?;
    let var = sq.mean_dim(2_usize)?;
    let std = var.add_scalar(eps).sqrt();
    let std_bc = std
        .reshape(Shape::from_dims(&[b, groups, 1]))?
        .broadcast_to(Shape::from_dims(&[b, groups, m]))?;
    let normed = centered.div(&std_bc)?;
    let normed_chw = normed.reshape(Shape::from_dims(&[b, c, h, w]))?;
    let g = x
        .const_f32_like(Arc::clone(gamma), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1]))?
        .broadcast_to(Shape::from_dims(&[b, c, h, w]))?;
    let bb = x
        .const_f32_like(Arc::clone(beta), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1]))?
        .broadcast_to(Shape::from_dims(&[b, c, h, w]))?;
    Ok(normed_chw.mul(&g)?.add(&bb)?)
}

// ---- Scheduler -------------------------------------------------------------

/// Flow-matching schedule for Flux sampling.
///
/// Returns a descending list of timesteps from 1 (noise) to 0 (data),
/// optionally with a sigmoid-tilt that emphasises later steps. The
/// shift is parametrised by image sequence length and two `(y1, y2)`
/// anchors (256 / 4096 are the upstream defaults).
#[derive(Debug, Clone)]
pub struct FlowMatchScheduler {
    pub num_steps: usize,
    /// Optional `(image_seq_len, base_shift, max_shift)`. Linear (None)
    /// is fine for FLUX.1-schnell; FLUX.1-dev uses a shift.
    pub shift: Option<(usize, f64, f64)>,
}

impl FlowMatchScheduler {
    /// Construct a linear scheduler with no shift.
    pub fn linear(num_steps: usize) -> Self {
        Self { num_steps, shift: None }
    }

    /// Construct a shifted scheduler matching the upstream Flux config.
    pub fn shifted(num_steps: usize, image_seq_len: usize, base_shift: f64, max_shift: f64) -> Self {
        Self { num_steps, shift: Some((image_seq_len, base_shift, max_shift)) }
    }

    /// The schedule as a descending list of `num_steps + 1` timesteps
    /// (each in `[0, 1]`). Adjacent pairs are the `(t_curr, t_prev)`
    /// inputs to a single denoising step.
    pub fn timesteps(&self) -> Vec<f64> {
        let mut ts: Vec<f64> = (0..=self.num_steps)
            .map(|v| v as f64 / self.num_steps as f64)
            .rev()
            .collect();
        if let Some((image_seq_len, y1, y2)) = self.shift {
            let (x1, x2) = (256.0_f64, 4096.0_f64);
            let m = (y2 - y1) / (x2 - x1);
            let b = y1 - m * x1;
            let mu = m * image_seq_len as f64 + b;
            ts = ts.into_iter().map(|v| time_shift(mu, 1.0, v)).collect();
        }
        ts
    }

    /// One denoising step: `img + pred * (t_prev - t_curr)`.
    pub fn step(
        &self, img: &LazyTensor, pred: &LazyTensor, t_curr: f64, t_prev: f64,
    ) -> Result<LazyTensor> {
        let delta = pred.mul_scalar(t_prev - t_curr);
        img.add(&delta)
    }
}

fn time_shift(mu: f64, sigma: f64, t: f64) -> f64 {
    let e = mu.exp();
    e / (e + (1.0 / t - 1.0).powf(sigma))
}

// ---- generate --------------------------------------------------------------

/// End-to-end sampling driver. Runs `num_steps` denoising iterations
/// against `model`, integrating the predicted velocity field with the
/// supplied schedule.
///
/// - `text_clip` is the CLIP pooled vector `(B, vec_in_dim)`.
/// - `text_t5` is the T5 context `(B, S_text, context_in_dim)`.
/// - `img` is the initial packed-noise latent `(B, S_image, in_channels)`.
/// - `img_ids` / `txt_ids` are the per-token RoPE id tensors.
/// - `guidance` is the per-batch guidance scalar (only used when
///   `model.config.guidance_embed == true`).
#[allow(clippy::too_many_arguments)]
pub fn generate(
    model: &FluxModel,
    text_clip: &LazyTensor,
    text_t5: &LazyTensor,
    img: &LazyTensor,
    img_ids: &LazyTensor,
    txt_ids: &LazyTensor,
    scheduler: &FlowMatchScheduler,
    guidance: Option<&LazyTensor>,
) -> Result<LazyTensor> {
    let ts = scheduler.timesteps();
    let mut x = img.clone();
    let batch = x.shape().dims()[0];
    for window in ts.windows(2) {
        let t_curr = window[0];
        let t_prev = window[1];
        let t_vec_data: Vec<f32> = vec![t_curr as f32; batch];
        let t_vec = x.const_f32_like(Arc::from(t_vec_data), Shape::from_dims(&[batch]));
        let pred = model.forward(&x, img_ids, text_t5, txt_ids, &t_vec, text_clip, guidance)?;
        x = scheduler.step(&x, &pred, t_curr, t_prev)?;
    }
    Ok(x)
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn linear_rand(
        in_f: usize, out_f: usize, bias: bool, rng: &mut impl FnMut() -> f32,
    ) -> FluxLinear {
        let w: Vec<f32> = (0..in_f * out_f).map(|_| rng()).collect();
        let b = if bias {
            Some(Arc::<[f32]>::from((0..out_f).map(|_| rng()).collect::<Vec<_>>()))
        } else { None };
        FluxLinear {
            weight: WeightStorage::F32(Arc::from(w)),
            bias: b,
            in_features: in_f,
            out_features: out_f,
        }
    }

    fn make_rng(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.1
        }
    }

    fn tiny_cfg() -> FluxConfig {
        // hidden_size=16, num_heads=2 → head_dim=8. axes_dim=[2,2,4]
        // partitions the head dim across three axes; every entry is
        // even so the half-dim splits cleanly.
        FluxConfig {
            in_channels: 8,
            vec_in_dim: 12,
            context_in_dim: 16,
            hidden_size: 16,
            mlp_ratio: 2.0,
            num_heads: 2,
            depth: 2,
            depth_single_blocks: 1,
            axes_dim: vec![2, 2, 4],
            theta: 10_000,
            qkv_bias: true,
            guidance_embed: false,
            qk_norm: true,
        }
    }

    fn tiny_qk_norm(head_dim: usize, gain: f32) -> FluxQkNorm {
        FluxQkNorm {
            query_gain: Arc::from(vec![gain; head_dim]),
            key_gain: Arc::from(vec![gain; head_dim]),
        }
    }

    fn tiny_self_attention(
        cfg: &FluxConfig, rng: &mut impl FnMut() -> f32,
    ) -> FluxSelfAttention {
        let h = cfg.hidden_size;
        FluxSelfAttention {
            qkv: linear_rand(h, 3 * h, cfg.qkv_bias, rng),
            qk_norm: tiny_qk_norm(cfg.head_dim(), 1.0),
            proj: linear_rand(h, h, true, rng),
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
        }
    }

    fn tiny_mlp(cfg: &FluxConfig, rng: &mut impl FnMut() -> f32) -> FluxMlp {
        let h = cfg.hidden_size;
        let m = cfg.mlp_hidden();
        FluxMlp {
            fc1: linear_rand(h, m, true, rng),
            fc2: linear_rand(m, h, true, rng),
        }
    }

    fn tiny_modulation(
        h: usize, num_chunks: usize, rng: &mut impl FnMut() -> f32,
    ) -> FluxModulation {
        FluxModulation {
            lin: linear_rand(h, 3 * num_chunks * h, true, rng),
            num_chunks,
        }
    }

    fn tiny_double_block(
        cfg: &FluxConfig, rng: &mut impl FnMut() -> f32,
    ) -> FluxDoubleStreamBlockWeights {
        let h = cfg.hidden_size;
        FluxDoubleStreamBlockWeights {
            img_mod: tiny_modulation(h, 2, rng),
            img_attn: tiny_self_attention(cfg, rng),
            img_mlp: tiny_mlp(cfg, rng),
            txt_mod: tiny_modulation(h, 2, rng),
            txt_attn: tiny_self_attention(cfg, rng),
            txt_mlp: tiny_mlp(cfg, rng),
        }
    }

    fn tiny_single_block(
        cfg: &FluxConfig, rng: &mut impl FnMut() -> f32,
    ) -> FluxSingleStreamBlockWeights {
        let h = cfg.hidden_size;
        let m = cfg.mlp_hidden();
        FluxSingleStreamBlockWeights {
            linear1: linear_rand(h, 3 * h + m, true, rng),
            linear2: linear_rand(h + m, h, true, rng),
            qk_norm: tiny_qk_norm(cfg.head_dim(), 1.0),
            modulation: tiny_modulation(h, 1, rng),
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
            mlp_hidden: m,
        }
    }

    fn tiny_mlp_embedder(
        in_sz: usize, h: usize, rng: &mut impl FnMut() -> f32,
    ) -> FluxMlpEmbedder {
        FluxMlpEmbedder {
            in_layer: linear_rand(in_sz, h, true, rng),
            out_layer: linear_rand(h, h, true, rng),
        }
    }

    fn tiny_last_layer(
        h: usize, in_ch: usize, rng: &mut impl FnMut() -> f32,
    ) -> FluxLastLayer {
        FluxLastLayer {
            linear: linear_rand(h, in_ch, true, rng),
            ada_ln_modulation: linear_rand(h, 2 * h, true, rng),
        }
    }

    fn tiny_model(cfg: &FluxConfig) -> FluxModel {
        let mut rng = make_rng(0xC0FFEE);
        let h = cfg.hidden_size;
        let weights = FluxWeights {
            img_in: linear_rand(cfg.in_channels, h, true, &mut rng),
            txt_in: linear_rand(cfg.context_in_dim, h, true, &mut rng),
            time_in: tiny_mlp_embedder(256, h, &mut rng),
            vector_in: tiny_mlp_embedder(cfg.vec_in_dim, h, &mut rng),
            guidance_in: if cfg.guidance_embed {
                Some(tiny_mlp_embedder(256, h, &mut rng))
            } else { None },
            double_blocks: (0..cfg.depth).map(|_| tiny_double_block(cfg, &mut rng)).collect(),
            single_blocks: (0..cfg.depth_single_blocks)
                .map(|_| tiny_single_block(cfg, &mut rng)).collect(),
            final_layer: tiny_last_layer(h, cfg.in_channels, &mut rng),
        };
        FluxModel { config: cfg.clone(), weights }
    }

    fn tiny_inputs(
        cfg: &FluxConfig, seq_text: usize, seq_image: usize,
    ) -> (LazyTensor, LazyTensor, LazyTensor, LazyTensor, LazyTensor, LazyTensor) {
        let dev = Device::cpu();
        let mut rng = make_rng(0xBADF00D);
        let img_data: Vec<f32> = (0..(1 * seq_image * cfg.in_channels)).map(|_| rng()).collect();
        let img = LazyTensor::from_f32(
            Arc::from(img_data), Shape::from_dims(&[1, seq_image, cfg.in_channels]), &dev,
        );
        let txt_data: Vec<f32> = (0..(1 * seq_text * cfg.context_in_dim)).map(|_| rng()).collect();
        let txt = img.const_f32_like(
            Arc::from(txt_data), Shape::from_dims(&[1, seq_text, cfg.context_in_dim]),
        );
        let n_axes = cfg.axes_dim.len();
        let img_ids_data: Vec<f32> = (0..(seq_image * n_axes)).map(|i| (i % 4) as f32).collect();
        let img_ids = img.const_f32_like(
            Arc::from(img_ids_data), Shape::from_dims(&[1, seq_image, n_axes]),
        );
        let txt_ids_data: Vec<f32> = vec![0.0_f32; seq_text * n_axes];
        let txt_ids = img.const_f32_like(
            Arc::from(txt_ids_data), Shape::from_dims(&[1, seq_text, n_axes]),
        );
        let y_data: Vec<f32> = (0..cfg.vec_in_dim).map(|_| rng()).collect();
        let y = img.const_f32_like(Arc::from(y_data), Shape::from_dims(&[1, cfg.vec_in_dim]));
        let t = img.const_f32_like(Arc::from(vec![0.5_f32]), Shape::from_dims(&[1]));
        (img, img_ids, txt, txt_ids, t, y)
    }

    #[test]
    fn flux_dit_forward_shape_and_finite_tiny() {
        let cfg = tiny_cfg();
        let model = tiny_model(&cfg);
        let (img, img_ids, txt, txt_ids, t, y) = tiny_inputs(&cfg, 4, 8);
        let out = model.forward(&img, &img_ids, &txt, &txt_ids, &t, &y, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, 8, cfg.in_channels]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    #[test]
    fn qk_norm_inside_attention_changes_output() {
        // Compare two models that differ ONLY in cfg.qk_norm. With
        // qk_norm = true the rms-norm gain is unit (1.0); with
        // qk_norm = false the path is skipped entirely. The two
        // outputs must differ to confirm the qk-norm code path is
        // actually reached.
        let mut cfg_with = tiny_cfg();
        cfg_with.qk_norm = true;
        let mut cfg_without = tiny_cfg();
        cfg_without.qk_norm = false;
        let model_with = tiny_model(&cfg_with);
        let mut model_without = model_with.clone();
        model_without.config = cfg_without;
        let (img, img_ids, txt, txt_ids, t, y) = tiny_inputs(&cfg_with, 4, 8);
        let out_with = model_with.forward(&img, &img_ids, &txt, &txt_ids, &t, &y, None).unwrap().realize_f32();
        let out_without = model_without.forward(&img, &img_ids, &txt, &txt_ids, &t, &y, None).unwrap().realize_f32();
        let max_diff: f32 = out_with.iter().zip(out_without.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(
            max_diff > 1e-6,
            "qk_norm path must change output (max_diff = {max_diff})",
        );
    }

    // ---- VAE tiny config + tests --------------------------------------

    fn tiny_vae_cfg() -> FluxVaeConfig {
        FluxVaeConfig {
            resolution: 64,
            in_channels: 3,
            ch: 4,
            out_ch: 3,
            ch_mult: vec![1, 2, 4, 4],
            num_res_blocks: 1,
            z_channels: 4,
            scale_factor: 1.0,
            shift_factor: 0.0,
            norm_num_groups: 2,
            norm_eps: 1e-6,
        }
    }

    fn zeros(n: usize) -> Arc<[f32]> { Arc::from(vec![0.0_f32; n]) }
    fn ones(n: usize) -> Arc<[f32]> { Arc::from(vec![1.0_f32; n]) }

    fn zero_resnet(c_in: usize, c_out: usize) -> VaeResnetWeights {
        VaeResnetWeights {
            n1_g: ones(c_in), n1_b: zeros(c_in),
            c1_w: zeros(c_out * c_in * 9), c1_b: zeros(c_out),
            n2_g: ones(c_out), n2_b: zeros(c_out),
            c2_w: zeros(c_out * c_out * 9), c2_b: zeros(c_out),
            shortcut_w: if c_in != c_out { Some(zeros(c_out * c_in)) } else { None },
            shortcut_b: if c_in != c_out { Some(zeros(c_out)) } else { None },
            in_channels: c_in,
            out_channels: c_out,
        }
    }

    fn zero_attn(c: usize) -> VaeAttnWeights {
        VaeAttnWeights {
            gn_g: ones(c), gn_b: zeros(c),
            q_w: zeros(c * c), q_b: zeros(c),
            k_w: zeros(c * c), k_b: zeros(c),
            v_w: zeros(c * c), v_b: zeros(c),
            out_w: zeros(c * c), out_b: zeros(c),
            channels: c,
        }
    }

    fn tiny_vae() -> FluxVae {
        let cfg = tiny_vae_cfg();
        let ch = cfg.ch;
        let mults = &cfg.ch_mult;
        // Encoder
        let mut down = Vec::with_capacity(mults.len());
        for (i_level, &mult) in mults.iter().enumerate() {
            let block_out = ch * mult;
            let in_mult = if i_level == 0 { 1 } else { mults[i_level - 1] };
            let mut block_in = ch * in_mult;
            let mut resnets = Vec::with_capacity(cfg.num_res_blocks);
            for _ in 0..cfg.num_res_blocks {
                resnets.push(zero_resnet(block_in, block_out));
                block_in = block_out;
            }
            let downsample_conv = if i_level != mults.len() - 1 {
                Some((zeros(block_in * block_in * 9), zeros(block_in)))
            } else { None };
            down.push(VaeDownBlock { resnets, downsample_conv });
        }
        let block_in = ch * mults.last().copied().unwrap_or(1);
        let encoder = FluxVaeEncoderWeights {
            conv_in_w: zeros(ch * cfg.in_channels * 9),
            conv_in_b: zeros(ch),
            down,
            mid1: zero_resnet(block_in, block_in),
            mid_attn: zero_attn(block_in),
            mid2: zero_resnet(block_in, block_in),
            norm_out_g: ones(block_in), norm_out_b: zeros(block_in),
            conv_out_w: zeros(2 * cfg.z_channels * block_in * 9),
            conv_out_b: zeros(2 * cfg.z_channels),
        };
        // Decoder
        let mut up = Vec::with_capacity(mults.len());
        // up_blocks indexed 0..mults.len(), level 0 = innermost (smallest
        // channel count after the first stage); we build in the natural
        // order (level 0 first) and iterate in reverse during decode.
        let mut block_in_dec = ch * mults.last().copied().unwrap_or(1);
        for (i_level_rev, &mult) in mults.iter().enumerate().rev() {
            let block_out = ch * mult;
            let mut resnets = Vec::with_capacity(cfg.num_res_blocks + 1);
            for _ in 0..=cfg.num_res_blocks {
                resnets.push(zero_resnet(block_in_dec, block_out));
                block_in_dec = block_out;
            }
            let upsample_conv = if i_level_rev != 0 {
                Some((zeros(block_in_dec * block_in_dec * 9), zeros(block_in_dec)))
            } else { None };
            up.push(VaeUpBlock { resnets, upsample_conv });
        }
        up.reverse();
        let last_dec_c = ch * mults[0];
        let decoder = FluxVaeDecoderWeights {
            conv_in_w: zeros((ch * mults.last().copied().unwrap_or(1)) * cfg.z_channels * 9),
            conv_in_b: zeros(ch * mults.last().copied().unwrap_or(1)),
            mid1: zero_resnet(ch * mults.last().copied().unwrap_or(1), ch * mults.last().copied().unwrap_or(1)),
            mid_attn: zero_attn(ch * mults.last().copied().unwrap_or(1)),
            mid2: zero_resnet(ch * mults.last().copied().unwrap_or(1), ch * mults.last().copied().unwrap_or(1)),
            up,
            norm_out_g: ones(last_dec_c), norm_out_b: zeros(last_dec_c),
            conv_out_w: zeros(cfg.out_ch * last_dec_c * 9),
            conv_out_b: zeros(cfg.out_ch),
        };
        FluxVae { config: cfg, encoder, decoder }
    }

    #[test]
    fn flux_vae_round_trip_tiny() {
        let dev = Device::cpu();
        let vae = tiny_vae();
        let cfg = vae.config.clone();
        // (1, 3, 64, 64) image
        let h_in = 64;
        let w_in = 64;
        let n = cfg.in_channels * h_in * w_in;
        let data: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.013).sin()) * 0.5).collect();
        let img = LazyTensor::from_f32(
            Arc::from(data), Shape::from_dims(&[1, cfg.in_channels, h_in, w_in]), &dev,
        );
        let z = vae.encode(&img).unwrap();
        let z_dims = z.shape().dims().to_vec();
        // 3 downsamples → /8 spatial
        assert_eq!(z_dims, vec![1, cfg.z_channels, h_in / 8, w_in / 8]);
        for &v in &z.realize_f32() {
            assert!(v.is_finite(), "non-finite latent: {v}");
        }
        let dec = vae.decode(&z).unwrap();
        let dec_dims = dec.shape().dims().to_vec();
        assert_eq!(dec_dims, vec![1, cfg.out_ch, h_in, w_in]);
        for &v in &dec.realize_f32() {
            assert!(v.is_finite(), "non-finite decode: {v}");
        }
    }

    #[test]
    fn flow_match_scheduler_step_finite() {
        let dev = Device::cpu();
        let sched = FlowMatchScheduler::linear(4);
        let ts = sched.timesteps();
        assert_eq!(ts.len(), 5);
        // Descending from 1.0 to 0.0.
        assert!((ts[0] - 1.0).abs() < 1e-9);
        assert!(ts.last().copied().unwrap().abs() < 1e-9);
        for w in ts.windows(2) {
            assert!(w[0] > w[1]);
        }
        // One step: img + pred * (t_prev - t_curr).
        let img = LazyTensor::from_f32(
            Arc::from(vec![1.0_f32, 2.0, 3.0, 4.0]),
            Shape::from_dims(&[1, 2, 2]), &dev,
        );
        let pred = img.const_f32_like(
            Arc::from(vec![0.5_f32, 0.5, 0.5, 0.5]), Shape::from_dims(&[1, 2, 2]),
        );
        let out = sched.step(&img, &pred, ts[0], ts[1]).unwrap();
        let out_v = out.realize_f32();
        // dt = ts[1] - ts[0] = 0.75 - 1.0 = -0.25
        // img + 0.5 * (-0.25) = img - 0.125
        for (i, v) in out_v.iter().enumerate() {
            let expected = (i + 1) as f32 - 0.125;
            assert!((v - expected).abs() < 1e-5, "step output[{i}] = {v}, expected {expected}");
            assert!(v.is_finite());
        }

        // Shifted scheduler.
        let sched_shift = FlowMatchScheduler::shifted(2, 64, 0.5, 1.15);
        let ts2 = sched_shift.timesteps();
        assert_eq!(ts2.len(), 3);
        for v in &ts2 {
            assert!(v.is_finite(), "non-finite shifted timestep: {v}");
        }
    }

    #[test]
    fn generate_end_to_end_tiny() {
        let cfg = tiny_cfg();
        let model = tiny_model(&cfg);
        let (img, img_ids, txt, txt_ids, _t, y) = tiny_inputs(&cfg, 4, 8);
        let sched = FlowMatchScheduler::linear(2);
        let out = generate(&model, &y, &txt, &img, &img_ids, &txt_ids, &sched, None).unwrap();
        assert_eq!(out.shape().dims(), &[1, 8, cfg.in_channels]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite generate output: {v}");
        }
    }

    #[test]
    fn quantized_flux_model_q4_0_close_to_source() {
        // Bake the source weights to Q4_0 (only Linears with
        // in_features % 32 == 0 are actually quantized; the rest fall
        // back to F32 — matches the upstream convention for "small"
        // diffusion-model weights). The Q4_0 output must stay numerically
        // close to the F32 source for tiny random inputs.
        let cfg = tiny_cfg();
        let model = tiny_model(&cfg);
        let q = QuantizedFluxModel::from_f32_bake(model.clone()).unwrap();
        let (img, img_ids, txt, txt_ids, t, y) = tiny_inputs(&cfg, 4, 8);
        let a = model.forward(&img, &img_ids, &txt, &txt_ids, &t, &y, None).unwrap().realize_f32();
        let b = q.forward(&img, &img_ids, &txt, &txt_ids, &t, &y, None).unwrap().realize_f32();
        let max_diff: f32 = a.iter().zip(b.iter())
            .map(|(x, y)| (x - y).abs()).fold(0.0, f32::max);
        // Q4_0 round-trip noise is bounded by block-max / 8; with
        // small random weights the per-output drift stays well below 0.01.
        assert!(max_diff < 1e-2, "quantized Flux forward should stay close to F32 (max_diff = {max_diff})");
        for &v in &b {
            assert!(v.is_finite(), "non-finite quantized output: {v}");
        }
    }
}
