//! Z-Image text-to-image diffusion model ported to the lazy-graph API.
//!
//! Z-Image-Turbo is a Flow Matching T2I generator from Alibaba's
//! Tongyi-MAI lab. Pipeline:
//!
//! ```text
//! prompt  ---> ZImageTextEncoder (Qwen3 backbone, return layer[-2])
//!                   --> cap_feats   [1, T, 2560]
//! noise   ---> ZImageTransformer2DModel(noise, t, cap_feats, cap_mask)
//!                   --> predicted velocity, iterated by
//!                       FlowMatchEulerDiscreteScheduler
//!                   --> final latent [1, 16, H/16, W/16]
//! latent  ---> AutoEncoderKL.decode
//!                   --> RGB image  [1, 3, H, W]  in [-1, 1]
//! ```
//!
//! # Architectural firsts (vs prior shipped modules)
//!
//! - **3D RoPE with interleaved real/imag form**. Each token has a
//!   `(frame, height, width)` index triple. Per-axis cos/sin tables
//!   of `axes_dims[i]/2` columns are concatenated to `head_dim/2`
//!   total, then applied as a complex multiplication on consecutive
//!   `(x[..., 2i], x[..., 2i+1])` pairs. Different from Fuel's
//!   built-in `rope_with_tables`, which uses the half-split rotate
//!   convention (LLaMA-style).
//! - **AdaLN-Zero modulation** with `tanh`-gated scale/gate. A
//!   shared 256-d timestep embedding drives a per-block linear
//!   that emits 4 chunks (scale_msa, gate_msa, scale_mlp, gate_mlp).
//!   Gates pass through `tanh`; scales get +1 before broadcasting.
//! - **Image+text unified attention**: image tokens (after their
//!   own modulated "noise refiner") and text tokens (after their
//!   own un-modulated "context refiner") are concatenated; the main
//!   transformer stack runs over the joint `[img | txt]` sequence
//!   with shared 3D RoPE.
//! - **VAE-with-16-latent-channels**: AutoencoderKL with 4 down
//!   stages (channels `[128, 256, 512, 512]`), 2 ResNet blocks per
//!   stage, mid block has a single-head spatial attention. Same
//!   shape family as the SD 1.5 VAE but with 16 latent channels
//!   (vs 4) and 8× downsample (matching 8× upsample on the decoder).
//! - **Flow Matching Euler discrete scheduler**. Sigmas live in
//!   `[0, 1]`, the scheduler steps `x_{t-1} = x_t + dt * v_t`,
//!   and the noise schedule uses a static shift `shift=3.0` over
//!   the linear interpolation.
//!
//! # Scope (v1)
//!
//! Forward-only — no autograd for the diffusion stack. Text encoder
//! uses a Qwen3-shape decoder block but bakes the
//! "return layer[-2], skip final norm" behavior directly instead
//! of routing through [`crate::lazy_qwen3`] (so we don't have to
//! patch the existing port to expose a "stop at layer K" entry
//! point). VAE encode + decode both ported.
//!
//! # Tests
//!
//! All five required tests are tiny synthetic configurations:
//!
//! - `text_encoder_forward_shape_tiny` — 2-layer Qwen3-shape encoder,
//!   2 tokens; checks output shape & finiteness.
//! - `vae_round_trip_tiny` — tiny 8×8 image through encode → decode,
//!   shape and finiteness check.
//! - `transformer_forward_shape_tiny` — small DiT (1 noise_refiner +
//!   1 context_refiner + 1 main layer); checks output latent shape.
//! - `scheduler_step_finite` — exercises the pure-host scheduler.
//! - `generate_end_to_end_tiny` — full pipeline (skipping the text
//!   encoder for tractability) runs `transformer -> scheduler ->
//!   VAE.decode` to produce a finite image of the expected shape.

use crate::lazy::LazyTensor;
use fuel_core_types::Shape;
use std::sync::Arc;

// ============================================================================
// Constants
// ============================================================================

/// AdaLN embedding dimension shared across the transformer.
pub const ADALN_EMBED_DIM: usize = 256;
/// Sinusoidal frequency-embedding feature count for the timestep encoder.
pub const FREQUENCY_EMBEDDING_SIZE: usize = 256;
/// Maximum sinusoidal period (matches the canonical Vaswani formula).
pub const MAX_PERIOD: f64 = 10000.0;

// ============================================================================
// Transformer configuration
// ============================================================================

/// Z-Image transformer hyperparameters. The `z_image_turbo` constructor
/// matches the released checkpoint; the test suite instantiates much
/// smaller configs to keep CPU runtime manageable.
#[derive(Debug, Clone)]
pub struct ZImageConfig {
    pub patch_size: usize,
    pub f_patch_size: usize,
    pub in_channels: usize,
    pub dim: usize,
    pub n_layers: usize,
    pub n_refiner_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub norm_eps: f64,
    pub qk_norm: bool,
    pub cap_feat_dim: usize,
    pub rope_theta: f64,
    pub t_scale: f64,
    pub axes_dims: Vec<usize>,
    pub axes_lens: Vec<usize>,
}

impl ZImageConfig {
    /// Production-shape config for Z-Image-Turbo (24B params).
    pub fn z_image_turbo() -> Self {
        Self {
            patch_size: 2,
            f_patch_size: 1,
            in_channels: 16,
            dim: 3840,
            n_layers: 30,
            n_refiner_layers: 2,
            n_heads: 30,
            n_kv_heads: 30,
            norm_eps: 1e-5,
            qk_norm: true,
            cap_feat_dim: 2560,
            rope_theta: 256.0,
            t_scale: 1000.0,
            axes_dims: vec![32, 48, 48],
            axes_lens: vec![1536, 512, 512],
        }
    }

    pub fn head_dim(&self) -> usize {
        self.dim / self.n_heads
    }

    pub fn hidden_dim(&self) -> usize {
        (self.dim / 3) * 8
    }
}

// ============================================================================
// Transformer block weights
// ============================================================================

#[derive(Debug, Clone)]
pub struct ZImageAttnWeights {
    pub to_q_w: Arc<[f32]>,
    pub to_k_w: Arc<[f32]>,
    pub to_v_w: Arc<[f32]>,
    pub to_out_w: Arc<[f32]>,
    /// `[head_dim]`, present iff `qk_norm == true`.
    pub q_norm_gain: Option<Arc<[f32]>>,
    pub k_norm_gain: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct ZImageFFNWeights {
    pub w1: Arc<[f32]>,
    pub w2: Arc<[f32]>,
    pub w3: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct ZImageBlockWeights {
    pub attn: ZImageAttnWeights,
    pub ffn: ZImageFFNWeights,
    pub attn_norm1_gain: Arc<[f32]>,
    pub attn_norm2_gain: Arc<[f32]>,
    pub ffn_norm1_gain: Arc<[f32]>,
    pub ffn_norm2_gain: Arc<[f32]>,
    /// `[4*dim, adaln_dim]` linear + `[4*dim]` bias. `None` for
    /// blocks with modulation disabled (= context refiner).
    pub adaln_w: Option<Arc<[f32]>>,
    pub adaln_b: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct ZImageTransformerWeights {
    /// Timestep MLP: in 256-sinusoidal → mid 1024 → out 256 (adaln_dim).
    pub t_embed_w1: Arc<[f32]>,
    pub t_embed_b1: Arc<[f32]>,
    pub t_embed_w2: Arc<[f32]>,
    pub t_embed_b2: Arc<[f32]>,
    /// Caption RMSNorm gain `[cap_feat_dim]` + linear `[cap_feat_dim, dim]` + bias `[dim]`.
    pub cap_norm_gain: Arc<[f32]>,
    pub cap_linear_w: Arc<[f32]>,
    pub cap_linear_b: Arc<[f32]>,
    /// Image patch embedder linear: `[patch_dim, dim]` + bias `[dim]`.
    pub x_embed_w: Arc<[f32]>,
    pub x_embed_b: Arc<[f32]>,
    /// FinalLayer: AdaLN SiLU-projection (`[dim, adaln_dim]`) + linear (`[out_ch, dim]` + bias).
    pub final_adaln_w: Arc<[f32]>,
    pub final_adaln_b: Arc<[f32]>,
    pub final_linear_w: Arc<[f32]>,
    pub final_linear_b: Arc<[f32]>,
    pub noise_refiner: Vec<ZImageBlockWeights>,
    pub context_refiner: Vec<ZImageBlockWeights>,
    pub layers: Vec<ZImageBlockWeights>,
}

// ============================================================================
// Helpers
// ============================================================================

/// Build a `(B, C, F, H, W)` patch sequence `(B, num_patches, patch_dim)`.
///
/// For `F=1, f_patch=1` (the image-generation case Z-Image always
/// hits) this collapses to the standard 4D patchify.
fn patchify(x: &LazyTensor, patch_size: usize, f_patch_size: usize) -> LazyTensor {
    let dims = x.shape().dims().to_vec();
    let (b, c, f, h, w) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
    let ph = patch_size;
    let pw = patch_size;
    let pf = f_patch_size;
    let f_tokens = f / pf;
    let h_tokens = h / ph;
    let w_tokens = w / pw;
    let num_patches = f_tokens * h_tokens * w_tokens;
    let patch_dim = pf * ph * pw * c;

    if f == 1 && pf == 1 {
        let x = x.squeeze(2_usize).unwrap();
        let x = x.reshape(Shape::from_dims(&[b, c, h_tokens, ph, w_tokens, pw])).unwrap();
        let x = x.permute([0, 2, 4, 3, 5, 1_usize]).unwrap();
        x.reshape(Shape::from_dims(&[b, num_patches, patch_dim])).unwrap()
    } else {
        // General case used only for video; matches eager fallback.
        let x = x.permute([0, 2, 3, 4, 1_usize]).unwrap();
        let x = x.reshape(Shape::from_dims(&[b, f_tokens, pf, h_tokens, ph, w_tokens * pw * c])).unwrap();
        let x = x.permute([0, 1, 3, 5, 2, 4_usize]).unwrap();
        x.reshape(Shape::from_dims(&[b, num_patches, patch_dim])).unwrap()
    }
}

fn unpatchify(
    x: &LazyTensor,
    size: (usize, usize, usize),
    patch_size: usize,
    f_patch_size: usize,
    out_channels: usize,
) -> LazyTensor {
    let (f, h, w) = size;
    let ph = patch_size;
    let pw = patch_size;
    let pf = f_patch_size;
    let f_tokens = f / pf;
    let h_tokens = h / ph;
    let w_tokens = w / pw;
    let ori_len = f_tokens * h_tokens * w_tokens;
    let dims = x.shape().dims().to_vec();
    let b = dims[0];
    let x = x.narrow(1_usize, 0, ori_len).unwrap();

    if f == 1 && pf == 1 {
        let x = x.reshape(Shape::from_dims(&[b, h_tokens, w_tokens, ph, pw, out_channels])).unwrap();
        let x = x.permute([0, 5, 1, 3, 2, 4_usize]).unwrap();
        let x = x.reshape(Shape::from_dims(&[b, out_channels, h, w])).unwrap();
        x.unsqueeze(2_usize).unwrap()
    } else {
        let x = x.reshape(Shape::from_dims(&[b, f_tokens, h_tokens, w_tokens, pf * ph * pw * out_channels])).unwrap();
        let x = x.reshape(Shape::from_dims(&[b, f_tokens, h_tokens, w_tokens * pf, ph, pw * out_channels])).unwrap();
        let x = x.permute([0, 5, 1, 3, 2, 4_usize]).unwrap();
        x.reshape(Shape::from_dims(&[b, out_channels, f, h, w])).unwrap()
    }
}

/// `y = x @ W (+ b)`. `x` rank arbitrary, last dim = `in_f`. `W` shape
/// `[in_f, out_f]`. Optional bias `[out_f]` broadcast on trailing dim.
fn linear(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: Option<&Arc<[f32]>>,
    in_f: usize,
    out_f: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[in_f, out_f]));
    let proj = x.matmul(&w_t).unwrap();
    match b {
        None => proj,
        Some(bias) => proj.add_trailing_bias(bias.clone()).unwrap(),
    }
}

/// Sinusoidal timestep embedding `(B,) -> (B, frequency_embedding_size)`.
fn timestep_embedding(t: &LazyTensor, anchor: &LazyTensor) -> LazyTensor {
    let half = FREQUENCY_EMBEDDING_SIZE / 2;
    // freqs[i] = exp( -ln(MAX_PERIOD) * i / half )  for i in 0..half
    let freqs_data: Vec<f32> = (0..half)
        .map(|i| (-MAX_PERIOD.ln() * (i as f64) / (half as f64)).exp() as f32)
        .collect();
    let freqs = anchor.const_f32_like(Arc::from(freqs_data), Shape::from_dims(&[half]));
    // t: (B,) -> (B, 1)
    let b_size = t.shape().dims()[0];
    let t_col = t.reshape(Shape::from_dims(&[b_size, 1])).unwrap();
    let freqs_row = freqs.reshape(Shape::from_dims(&[1, half])).unwrap();
    let args = t_col.broadcast_mul(&freqs_row).unwrap();
    let cos_e = args.cos();
    let sin_e = args.sin();
    cos_e.concat(&sin_e, 1_usize).unwrap()
}

/// Apply RoPE in the **interleaved real/imag** form. `x` shape
/// `(B, L, H, head_dim)`, `cos`/`sin` shape `(L, head_dim/2)`.
fn apply_rotary_emb_interleaved(
    x: &LazyTensor,
    cos: &LazyTensor,
    sin: &LazyTensor,
) -> LazyTensor {
    let dims = x.shape().dims().to_vec();
    let (b, l, n, hd) = (dims[0], dims[1], dims[2], dims[3]);
    let half = hd / 2;
    // Reshape to (B, L, N, half, 2).
    let x5 = x.reshape(Shape::from_dims(&[b, l, n, half, 2])).unwrap();
    // Extract real / imag halves: each is (B, L, N, half).
    let x_real = x5.narrow(4_usize, 0, 1).unwrap().reshape(Shape::from_dims(&[b, l, n, half])).unwrap();
    let x_imag = x5.narrow(4_usize, 1, 1).unwrap().reshape(Shape::from_dims(&[b, l, n, half])).unwrap();
    // cos / sin: (L, half) -> (1, L, 1, half).
    let cos_e = cos
        .reshape(Shape::from_dims(&[1, l, 1, half])).unwrap()
        .broadcast_to(Shape::from_dims(&[b, l, n, half])).unwrap();
    let sin_e = sin
        .reshape(Shape::from_dims(&[1, l, 1, half])).unwrap()
        .broadcast_to(Shape::from_dims(&[b, l, n, half])).unwrap();
    // Complex mult.
    let y_real = x_real.mul(&cos_e).unwrap().sub(&x_imag.mul(&sin_e).unwrap()).unwrap();
    let y_imag = x_real.mul(&sin_e).unwrap().add(&x_imag.mul(&cos_e).unwrap()).unwrap();
    // Interleave back: reshape both to (B, L, N, half, 1), concat dim=4, reshape (B, L, N, hd).
    let yr = y_real.reshape(Shape::from_dims(&[b, l, n, half, 1])).unwrap();
    let yi = y_imag.reshape(Shape::from_dims(&[b, l, n, half, 1])).unwrap();
    let stacked = yr.concat(&yi, 4_usize).unwrap();
    stacked.reshape(Shape::from_dims(&[b, l, n, hd])).unwrap()
}

/// Build the 3D-RoPE cos/sin tables for a sequence of `(f, h, w)`
/// position triples. Returns tables of shape `(seq, head_dim/2)`
/// suitable for [`apply_rotary_emb_interleaved`].
fn build_3d_rope_tables(
    positions: &[(usize, usize, usize)],
    axes_dims: &[usize],
    axes_lens: &[usize],
    theta: f64,
) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(axes_dims.len(), 3);
    debug_assert_eq!(axes_lens.len(), 3);
    // Pre-build per-axis inv_freq tables.
    let half_dims: Vec<usize> = axes_dims.iter().map(|d| d / 2).collect();
    let inv_freqs: Vec<Vec<f32>> = axes_dims.iter().map(|d| {
        let half_d = d / 2;
        (0..half_d)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / *d as f32))
            .collect()
    }).collect();

    let seq = positions.len();
    let head_dim_half: usize = half_dims.iter().sum();
    let mut cos = vec![0.0_f32; seq * head_dim_half];
    let mut sin = vec![0.0_f32; seq * head_dim_half];

    for (token_idx, &(f, h, w)) in positions.iter().enumerate() {
        let pos = [f, h, w];
        let mut col = 0;
        for ax in 0..3 {
            let p = pos[ax] as f32;
            let _ = axes_lens[ax]; // not used directly; eager just builds tables up to axes_lens
            for &freq in &inv_freqs[ax] {
                let angle = p * freq;
                cos[token_idx * head_dim_half + col] = angle.cos();
                sin[token_idx * head_dim_half + col] = angle.sin();
                col += 1;
            }
        }
    }

    (cos, sin)
}

/// `LayerNorm` without learnable params (eager `LayerNormNoParams`).
/// Subtracts mean, divides by std (with eps), all along the last dim.
fn layer_norm_no_params(x: &LazyTensor, eps: f64) -> LazyTensor {
    // Build a no-affine layer norm via primitives. `layer_norm_last_dim`
    // already does the math we need with the same eps semantics.
    x.layer_norm_last_dim(eps).unwrap()
}

// ============================================================================
// Transformer attention
// ============================================================================

fn z_image_attention(
    x: &LazyTensor,
    attn_mask: Option<&LazyTensor>,
    cos: &LazyTensor,
    sin: &LazyTensor,
    aw: &ZImageAttnWeights,
    cfg: &ZImageConfig,
) -> LazyTensor {
    let dims = x.shape().dims().to_vec();
    let (b, l, dim) = (dims[0], dims[1], dims[2]);
    let n_heads = cfg.n_heads;
    let head_dim = cfg.head_dim();
    let _ = dim;

    let q = linear(x, &aw.to_q_w, None, cfg.dim, n_heads * head_dim);
    let k = linear(x, &aw.to_k_w, None, cfg.dim, cfg.n_kv_heads * head_dim);
    let v = linear(x, &aw.to_v_w, None, cfg.dim, cfg.n_kv_heads * head_dim);

    // (B, L, H, D)
    let q = q.reshape(Shape::from_dims(&[b, l, n_heads, head_dim])).unwrap();
    let k = k.reshape(Shape::from_dims(&[b, l, cfg.n_kv_heads, head_dim])).unwrap();
    let v = v.reshape(Shape::from_dims(&[b, l, cfg.n_kv_heads, head_dim])).unwrap();

    // QK norm per head (RMSNorm along head_dim).
    let (q, k) = match (&aw.q_norm_gain, &aw.k_norm_gain) {
        (Some(qg), Some(kg)) => {
            let q = q.rms_norm_affine(Arc::clone(qg), 1e-5).unwrap();
            let k = k.rms_norm_affine(Arc::clone(kg), 1e-5).unwrap();
            (q, k)
        }
        _ => (q, k),
    };

    // RoPE.
    let q = apply_rotary_emb_interleaved(&q, cos, sin);
    let k = apply_rotary_emb_interleaved(&k, cos, sin);

    // Transpose to (B, H, L, D).
    let q = q.permute([0, 2, 1, 3_usize]).unwrap();
    let k = k.permute([0, 2, 1, 3_usize]).unwrap();
    let v = v.permute([0, 2, 1, 3_usize]).unwrap();

    // Basic attention.
    let scale = 1.0 / (head_dim as f64).sqrt();
    let k_t = k.transpose_last_two().unwrap();
    let mut scores = q.matmul(&k_t).unwrap().mul_scalar(scale);

    if let Some(m) = attn_mask {
        // m: (B, L) F32 with 1.0 = valid, 0.0 = padding. Convert to
        // additive bias (0 / -inf-ish) and broadcast to (B, 1, 1, L).
        let m_b = m.reshape(Shape::from_dims(&[b, 1, 1, l])).unwrap();
        let m_b = m_b.broadcast_to(Shape::from_dims(&[b, n_heads, l, l])).unwrap();
        let m_neg = m_b.add_scalar(-1.0).mul_scalar(1e9);
        scores = scores.add(&m_neg).unwrap();
    }

    let probs = scores.softmax_last_dim().unwrap();
    let ctx = probs.matmul(&v).unwrap();
    let ctx = ctx.permute([0, 2, 1, 3_usize]).unwrap().reshape(Shape::from_dims(&[b, l, n_heads * head_dim])).unwrap();
    linear(&ctx, &aw.to_out_w, None, n_heads * head_dim, cfg.dim)
}

// ============================================================================
// Transformer block forward
// ============================================================================

fn z_image_block(
    x: &LazyTensor,
    attn_mask: Option<&LazyTensor>,
    cos: &LazyTensor,
    sin: &LazyTensor,
    adaln_input: Option<&LazyTensor>,
    bw: &ZImageBlockWeights,
    cfg: &ZImageConfig,
) -> LazyTensor {
    let dim = cfg.dim;
    let hidden_dim = cfg.hidden_dim();
    let adaln_dim = dim.min(ADALN_EMBED_DIM);

    if let (Some(aw), Some(_ab)) = (&bw.adaln_w, &bw.adaln_b) {
        let adaln_input = adaln_input.expect("adaln_input required for modulation blocks");
        // (B, adaln_dim) → (B, 4*dim) via linear(+bias).
        let mod_out = linear(adaln_input, aw, bw.adaln_b.as_ref(), adaln_dim, 4 * dim);
        // (B, 4*dim) → (B, 1, 4*dim) → chunk(4) on last dim.
        let mod_out = mod_out.unsqueeze(1_usize).unwrap();
        let chunks = mod_out.chunk(4, 2_usize).unwrap();
        let scale_msa = chunks[0].add_scalar(1.0);
        let gate_msa = chunks[1].tanh();
        let scale_mlp = chunks[2].add_scalar(1.0);
        let gate_mlp = chunks[3].tanh();

        // Attention block.
        let normed = x.rms_norm_affine(Arc::clone(&bw.attn_norm1_gain), cfg.norm_eps).unwrap();
        let scaled = normed.broadcast_mul(&scale_msa).unwrap();
        let attn_out = z_image_attention(&scaled, attn_mask, cos, sin, &bw.attn, cfg);
        let attn_out = attn_out
            .rms_norm_affine(Arc::clone(&bw.attn_norm2_gain), cfg.norm_eps)
            .unwrap();
        let x = x.add(&gate_msa.broadcast_mul(&attn_out).unwrap()).unwrap();

        // FFN.
        let normed = x.rms_norm_affine(Arc::clone(&bw.ffn_norm1_gain), cfg.norm_eps).unwrap();
        let scaled = normed.broadcast_mul(&scale_mlp).unwrap();
        let ffn_out = ffn_swiglu(&scaled, &bw.ffn, dim, hidden_dim);
        let ffn_out = ffn_out.rms_norm_affine(Arc::clone(&bw.ffn_norm2_gain), cfg.norm_eps).unwrap();
        x.add(&gate_mlp.broadcast_mul(&ffn_out).unwrap()).unwrap()
    } else {
        // No modulation (context refiner).
        let normed = x.rms_norm_affine(Arc::clone(&bw.attn_norm1_gain), cfg.norm_eps).unwrap();
        let attn_out = z_image_attention(&normed, attn_mask, cos, sin, &bw.attn, cfg);
        let attn_out = attn_out
            .rms_norm_affine(Arc::clone(&bw.attn_norm2_gain), cfg.norm_eps)
            .unwrap();
        let x = x.add(&attn_out).unwrap();
        let normed = x.rms_norm_affine(Arc::clone(&bw.ffn_norm1_gain), cfg.norm_eps).unwrap();
        let ffn_out = ffn_swiglu(&normed, &bw.ffn, dim, hidden_dim);
        let ffn_out = ffn_out
            .rms_norm_affine(Arc::clone(&bw.ffn_norm2_gain), cfg.norm_eps)
            .unwrap();
        x.add(&ffn_out).unwrap()
    }
}

fn ffn_swiglu(x: &LazyTensor, fw: &ZImageFFNWeights, dim: usize, hidden_dim: usize) -> LazyTensor {
    let x1 = linear(x, &fw.w1, None, dim, hidden_dim).silu();
    let x3 = linear(x, &fw.w3, None, dim, hidden_dim);
    let mid = x1.mul(&x3).unwrap();
    linear(&mid, &fw.w2, None, hidden_dim, dim)
}

// ============================================================================
// ZImageTransformer2DModel
// ============================================================================

#[derive(Debug, Clone)]
pub struct ZImageTransformer2DModel {
    pub config: ZImageConfig,
    pub weights: ZImageTransformerWeights,
}

impl ZImageTransformer2DModel {
    /// Forward pass.
    /// - `x`:        latent `(1, C, F, H, W)`
    /// - `t`:        timestep scalar `(1,)` in `[0, 1]`
    /// - `cap_feats`: caption features `(1, T, cap_feat_dim)`
    /// - `cap_mask`: caption attn mask `(1, T)` (F32, 1.0=valid, 0.0=pad)
    ///
    /// Returns predicted velocity `(1, C, F, H, W)` matching `x`.
    pub fn forward(
        &self,
        x: &LazyTensor,
        t: &LazyTensor,
        cap_feats: &LazyTensor,
        cap_mask: &LazyTensor,
    ) -> LazyTensor {
        let cfg = &self.config;
        let w = &self.weights;
        let dims = x.shape().dims().to_vec();
        let (b, _c, f, h, w_dim) = (dims[0], dims[1], dims[2], dims[3], dims[4]);
        let patch_size = cfg.patch_size;
        let f_patch_size = cfg.f_patch_size;
        let adaln_dim = cfg.dim.min(ADALN_EMBED_DIM);

        // 1. Timestep embedding.
        let t_scaled = t.mul_scalar(cfg.t_scale);
        let t_freq = timestep_embedding(&t_scaled, x);
        let t_mid = linear(&t_freq, &w.t_embed_w1, Some(&w.t_embed_b1), FREQUENCY_EMBEDDING_SIZE, 1024).silu();
        let adaln_input = linear(&t_mid, &w.t_embed_w2, Some(&w.t_embed_b2), 1024, adaln_dim);
        //  (B, adaln_dim)

        // 2. Patchify and embed image.
        let x_patches = patchify(x, patch_size, f_patch_size);
        let patch_dim = f_patch_size * patch_size * patch_size * cfg.in_channels;
        let mut x_seq = linear(&x_patches, &w.x_embed_w, Some(&w.x_embed_b), patch_dim, cfg.dim);
        let img_seq_len = x_seq.shape().dims()[1];

        let f_tokens = f / f_patch_size;
        let h_tokens = h / patch_size;
        let w_tokens = w_dim / patch_size;
        let text_len = cap_feats.shape().dims()[1];

        // 3. RoPE tables for image / caption / unified.
        let img_positions: Vec<(usize, usize, usize)> = (0..f_tokens).flat_map(|fi| {
            (0..h_tokens).flat_map(move |hi| (0..w_tokens).map(move |wi| (text_len + 1 + fi, hi, wi)))
        }).collect();
        let cap_positions: Vec<(usize, usize, usize)> = (0..text_len).map(|i| (1 + i, 0, 0)).collect();
        let mut unified_positions = img_positions.clone();
        unified_positions.extend(cap_positions.iter().copied());

        let head_dim = cfg.head_dim();
        let half_head = head_dim / 2;
        let (img_cos_v, img_sin_v) = build_3d_rope_tables(
            &img_positions, &cfg.axes_dims, &cfg.axes_lens, cfg.rope_theta,
        );
        let (cap_cos_v, cap_sin_v) = build_3d_rope_tables(
            &cap_positions, &cfg.axes_dims, &cfg.axes_lens, cfg.rope_theta,
        );
        let (uni_cos_v, uni_sin_v) = build_3d_rope_tables(
            &unified_positions, &cfg.axes_dims, &cfg.axes_lens, cfg.rope_theta,
        );

        let img_cos = x_seq.const_f32_like(Arc::from(img_cos_v), Shape::from_dims(&[img_seq_len, half_head]));
        let img_sin = x_seq.const_f32_like(Arc::from(img_sin_v), Shape::from_dims(&[img_seq_len, half_head]));
        let cap_cos = x_seq.const_f32_like(Arc::from(cap_cos_v), Shape::from_dims(&[text_len, half_head]));
        let cap_sin = x_seq.const_f32_like(Arc::from(cap_sin_v), Shape::from_dims(&[text_len, half_head]));
        let uni_cos = x_seq.const_f32_like(Arc::from(uni_cos_v), Shape::from_dims(&[img_seq_len + text_len, half_head]));
        let uni_sin = x_seq.const_f32_like(Arc::from(uni_sin_v), Shape::from_dims(&[img_seq_len + text_len, half_head]));

        // 4. Caption RMSNorm + linear.
        let cap_normed = cap_feats
            .rms_norm_affine(Arc::clone(&w.cap_norm_gain), cfg.norm_eps)
            .unwrap();
        let mut cap = linear(&cap_normed, &w.cap_linear_w, Some(&w.cap_linear_b), cfg.cap_feat_dim, cfg.dim);

        // 5. Attention masks (F32: 1.0 = valid, 0.0 = padding).
        let ones_v: Vec<f32> = vec![1.0; b * img_seq_len];
        let img_mask = x_seq.const_f32_like(Arc::from(ones_v), Shape::from_dims(&[b, img_seq_len]));

        // 6. Noise refiner (modulated image stack).
        for blk in &w.noise_refiner {
            x_seq = z_image_block(
                &x_seq, Some(&img_mask), &img_cos, &img_sin, Some(&adaln_input), blk, cfg,
            );
        }

        // 7. Context refiner (un-modulated text stack).
        for blk in &w.context_refiner {
            cap = z_image_block(&cap, Some(cap_mask), &cap_cos, &cap_sin, None, blk, cfg);
        }

        // 8. Concat [image, text] on seq dim.
        let mut unified = x_seq.concat(&cap, 1_usize).unwrap();
        let unified_mask = img_mask.concat(cap_mask, 1_usize).unwrap();

        // 9. Main layers (modulated).
        for blk in &w.layers {
            unified = z_image_block(
                &unified, Some(&unified_mask), &uni_cos, &uni_sin,
                Some(&adaln_input), blk, cfg,
            );
        }

        // 10. Take image portion.
        let x_out = unified.narrow(1_usize, 0, img_seq_len).unwrap();

        // 11. Final layer = layer_norm(x) * (1 + scale) -> linear.
        let scale = linear(
            &adaln_input.silu(),
            &w.final_adaln_w, Some(&w.final_adaln_b),
            adaln_dim, cfg.dim,
        ).add_scalar(1.0).unsqueeze(1_usize).unwrap();
        let normed = layer_norm_no_params(&x_out, 1e-6);
        let scaled = normed.broadcast_mul(&scale).unwrap();
        let out_channels = patch_size * patch_size * f_patch_size * cfg.in_channels;
        let x_out = linear(&scaled, &w.final_linear_w, Some(&w.final_linear_b), cfg.dim, out_channels);

        // 12. Unpatchify back to (B, C, F, H, W).
        unpatchify(&x_out, (f, h, w_dim), patch_size, f_patch_size, cfg.in_channels)
    }
}

// ============================================================================
// Z-Image Text Encoder (Qwen3-shape backbone returning hidden_states[-2])
// ============================================================================

#[derive(Debug, Clone)]
pub struct TextEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

impl TextEncoderConfig {
    pub fn z_image() -> Self {
        Self {
            vocab_size: 151_936,
            hidden_size: 2560,
            intermediate_size: 9728,
            num_hidden_layers: 36,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TextEncoderLayer {
    pub q_w: Arc<[f32]>,
    pub k_w: Arc<[f32]>,
    pub v_w: Arc<[f32]>,
    pub o_w: Arc<[f32]>,
    /// `[head_dim]` per-head QK-norm gains.
    pub q_norm_gain: Arc<[f32]>,
    pub k_norm_gain: Arc<[f32]>,
    pub gate_w: Arc<[f32]>,
    pub up_w: Arc<[f32]>,
    pub down_w: Arc<[f32]>,
    pub input_ln_gain: Arc<[f32]>,
    pub post_attn_ln_gain: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct TextEncoderWeights {
    pub token_embedding: Arc<[f32]>,
    pub layers: Vec<TextEncoderLayer>,
}

#[derive(Debug, Clone)]
pub struct ZImageTextEncoder {
    pub config: TextEncoderConfig,
    pub weights: TextEncoderWeights,
}

impl ZImageTextEncoder {
    /// Encode token IDs into `(1, seq, hidden_size)`. Returns the output
    /// of `layers[num_hidden_layers - 2]` BEFORE the final RMSNorm, as
    /// in the upstream Z-Image text encoder.
    pub fn forward(&self, tokens: &[u32]) -> crate::Result<LazyTensor> {
        let cfg = &self.config;
        if tokens.is_empty() {
            return Err(crate::Error::Msg("text encoder: tokens must be non-empty".into()).bt());
        }
        let seq = tokens.len();
        let mut h = LazyTensor::embed_tokens(
            cfg.vocab_size_arc_clone(&self.weights.token_embedding),
            cfg.vocab_size, cfg.hidden_size, tokens, &crate::Device::cpu(),
        )?;
        // Pre-build RoPE tables (LLaMA half-split style; Qwen3 uses that).
        let (rope_cos, rope_sin) = h.rope_tables_const(cfg.rope_theta, 0, seq, cfg.head_dim);

        let target = cfg.num_hidden_layers - 2;
        let causal = LazyTensor::additive_causal_mask_like(&h, seq);
        for (i, layer) in self.weights.layers.iter().enumerate() {
            h = apply_text_encoder_layer(&h, layer, &rope_cos, &rope_sin, &causal, cfg)?;
            if i == target {
                return Ok(h);
            }
        }
        Err(crate::Error::Msg(format!(
            "text encoder: target layer index {target} out of {}", self.weights.layers.len(),
        )).bt())
    }
}

impl TextEncoderConfig {
    /// Convenience that just clones an existing `Arc` (mirrors the
    /// implicit `Arc::clone` of caller-provided weight slabs).
    fn vocab_size_arc_clone(&self, a: &Arc<[f32]>) -> Arc<[f32]> { a.clone() }
}

fn apply_text_encoder_layer(
    x: &LazyTensor,
    layer: &TextEncoderLayer,
    rope_cos: &LazyTensor,
    rope_sin: &LazyTensor,
    causal: &LazyTensor,
    cfg: &TextEncoderConfig,
) -> crate::Result<LazyTensor> {
    let hidden = cfg.hidden_size;
    let n_heads = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim;
    let kv_dim = n_kv * head_dim;
    let dims = x.shape().dims().to_vec();
    let (b, l, _) = (dims[0], dims[1], dims[2]);

    let x_norm = x.rms_norm_affine(Arc::clone(&layer.input_ln_gain), cfg.rms_norm_eps)?;
    let q = linear(&x_norm, &layer.q_w, None, hidden, n_heads * head_dim);
    let k = linear(&x_norm, &layer.k_w, None, hidden, kv_dim);
    let v = linear(&x_norm, &layer.v_w, None, hidden, kv_dim);

    let q = q.reshape(Shape::from_dims(&[b, l, n_heads, head_dim]))?.permute([0, 2, 1, 3_usize])?;
    let k = k.reshape(Shape::from_dims(&[b, l, n_kv, head_dim]))?.permute([0, 2, 1, 3_usize])?;
    let v = v.reshape(Shape::from_dims(&[b, l, n_kv, head_dim]))?.permute([0, 2, 1, 3_usize])?;

    // Per-head RMSNorm along head_dim.
    let q = q.rms_norm_affine(Arc::clone(&layer.q_norm_gain), cfg.rms_norm_eps)?;
    let k = k.rms_norm_affine(Arc::clone(&layer.k_norm_gain), cfg.rms_norm_eps)?;

    // RoPE (LLaMA half-split form via Fuel's fused op).
    let q_r = q.rope_with_tables(rope_cos, rope_sin)?;
    let k_r = k.rope_with_tables(rope_cos, rope_sin)?;

    // GQA expand: repeat K/V along the head axis.
    let n_rep = n_heads / n_kv;
    let k_full = k_r.repeat_interleave(1_usize, n_rep)?;
    let v_full = v.repeat_interleave(1_usize, n_rep)?;

    let k_t = k_full.transpose_last_two()?;
    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = q_r.matmul(&k_t)?.mul_scalar(scale);
    let scores = scores.broadcast_add(causal)?;
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v_full)?;
    let ctx = ctx.permute([0, 2, 1, 3_usize])?.reshape(Shape::from_dims(&[b, l, n_heads * head_dim]))?;
    let attn_out = linear(&ctx, &layer.o_w, None, n_heads * head_dim, hidden);

    let h = x.add(&attn_out)?;
    let h_norm = h.rms_norm_affine(Arc::clone(&layer.post_attn_ln_gain), cfg.rms_norm_eps)?;
    let gate = linear(&h_norm, &layer.gate_w, None, hidden, cfg.intermediate_size).silu();
    let up = linear(&h_norm, &layer.up_w, None, hidden, cfg.intermediate_size);
    let mid = gate.mul(&up)?;
    let down = linear(&mid, &layer.down_w, None, cfg.intermediate_size, hidden);
    h.add(&down)
}

// ============================================================================
// VAE (AutoEncoderKL, diffusers format)
// ============================================================================

#[derive(Debug, Clone)]
pub struct VaeConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub scaling_factor: f64,
    pub shift_factor: f64,
    pub norm_num_groups: usize,
}

impl VaeConfig {
    pub fn z_image() -> Self {
        Self {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 16,
            block_out_channels: vec![128, 256, 512, 512],
            layers_per_block: 2,
            scaling_factor: 0.3611,
            shift_factor: 0.1159,
            norm_num_groups: 32,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VaeResnetWeights {
    pub n1_g: Arc<[f32]>,
    pub n1_b: Arc<[f32]>,
    pub c1_w: Arc<[f32]>,
    pub c1_b: Arc<[f32]>,
    pub n2_g: Arc<[f32]>,
    pub n2_b: Arc<[f32]>,
    pub c2_w: Arc<[f32]>,
    pub c2_b: Arc<[f32]>,
    pub shortcut_w: Option<Arc<[f32]>>,
    pub shortcut_b: Option<Arc<[f32]>>,
}

#[derive(Debug, Clone)]
pub struct VaeAttnWeights {
    pub gn_g: Arc<[f32]>,
    pub gn_b: Arc<[f32]>,
    /// Pre-transposed `[in, out]`.
    pub q_w: Arc<[f32]>,
    pub q_b: Arc<[f32]>,
    pub k_w: Arc<[f32]>,
    pub k_b: Arc<[f32]>,
    pub v_w: Arc<[f32]>,
    pub v_b: Arc<[f32]>,
    pub out_w: Arc<[f32]>,
    pub out_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct VaeDownBlockWeights {
    pub resnets: Vec<VaeResnetWeights>,
    /// `[c, c, 3, 3]` + bias. Optional: last down block has no downsampler.
    pub downsample_conv: Option<(Arc<[f32]>, Arc<[f32]>)>,
}

#[derive(Debug, Clone)]
pub struct VaeUpBlockWeights {
    pub resnets: Vec<VaeResnetWeights>,
    pub upsample_conv: Option<(Arc<[f32]>, Arc<[f32]>)>,
}

#[derive(Debug, Clone)]
pub struct VaeWeights {
    // encoder
    pub enc_conv_in_w: Arc<[f32]>,
    pub enc_conv_in_b: Arc<[f32]>,
    pub enc_down_blocks: Vec<VaeDownBlockWeights>,
    pub enc_mid_resnet_1: VaeResnetWeights,
    pub enc_mid_attn: VaeAttnWeights,
    pub enc_mid_resnet_2: VaeResnetWeights,
    pub enc_conv_norm_out_g: Arc<[f32]>,
    pub enc_conv_norm_out_b: Arc<[f32]>,
    /// `[2*latent_channels, mid_c, 3, 3]` (mean + log-var concat).
    pub enc_conv_out_w: Arc<[f32]>,
    pub enc_conv_out_b: Arc<[f32]>,
    // decoder
    pub dec_conv_in_w: Arc<[f32]>,
    pub dec_conv_in_b: Arc<[f32]>,
    pub dec_mid_resnet_1: VaeResnetWeights,
    pub dec_mid_attn: VaeAttnWeights,
    pub dec_mid_resnet_2: VaeResnetWeights,
    pub dec_up_blocks: Vec<VaeUpBlockWeights>,
    pub dec_conv_norm_out_g: Arc<[f32]>,
    pub dec_conv_norm_out_b: Arc<[f32]>,
    pub dec_conv_out_w: Arc<[f32]>,
    pub dec_conv_out_b: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct AutoEncoderKL {
    pub config: VaeConfig,
    pub weights: VaeWeights,
}

// ---- VAE primitives (mirroring lazy_sd_vae but parameterised on
//      norm_num_groups so the Z-Image tiny test config works) -------

fn vae_group_norm(
    x: &LazyTensor,
    gamma: &Arc<[f32]>,
    beta: &Arc<[f32]>,
    groups: usize,
    eps: f64,
    c: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let cpg = c / groups;
    let m = cpg * h * w;
    let x_flat = x.reshape(Shape::from_dims(&[1, groups, m])).unwrap();
    let mean = x_flat.mean_dim(2_usize).unwrap();
    let mean_bc = mean
        .reshape(Shape::from_dims(&[1, groups, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, groups, m])).unwrap();
    let centered = x_flat.sub(&mean_bc).unwrap();
    let sq = centered.mul(&centered).unwrap();
    let var = sq.mean_dim(2_usize).unwrap();
    let std = var.add_scalar(eps).sqrt();
    let std_bc = std
        .reshape(Shape::from_dims(&[1, groups, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, groups, m])).unwrap();
    let normed = centered.div(&std_bc).unwrap();
    let normed_chw = normed.reshape(Shape::from_dims(&[1, c, h, w])).unwrap();
    let g = x
        .const_f32_like(gamma.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    let b = x
        .const_f32_like(beta.clone(), Shape::from_dims(&[c]))
        .reshape(Shape::from_dims(&[1, c, 1, 1])).unwrap()
        .broadcast_to(Shape::from_dims(&[1, c, h, w])).unwrap();
    normed_chw.mul(&g).unwrap().add(&b).unwrap()
}

fn conv2d_k3_s1_p1(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    cin: usize,
    cout: usize,
) -> LazyTensor {
    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[cout, cin, 3, 3]));
    let b_t = x.const_f32_like(b.clone(), Shape::from_dims(&[cout]));
    x.conv2d(&w_t, Some(&b_t), (1, 1), (1, 1), 1).unwrap()
}

fn conv2d_k3_s2_p0_with_pad(
    x: &LazyTensor,
    w: &Arc<[f32]>,
    b: &Arc<[f32]>,
    cin: usize,
    cout: usize,
    h: usize,
    w_sz: usize,
) -> LazyTensor {
    // Match Python: pad_with_zeros (right=1, bottom=1) then stride-2 conv.
    // Implemented by manual reshape+concat zero padding on axes 2 and 3.
    let zeros_right_v: Vec<f32> = vec![0.0; cin * h * 1];
    let zeros_right = x
        .const_f32_like(Arc::from(zeros_right_v), Shape::from_dims(&[1, cin, h, 1]));
    let x_w = x.concat(&zeros_right, 3_usize).unwrap();
    let new_w = w_sz + 1;
    let zeros_bottom_v: Vec<f32> = vec![0.0; cin * 1 * new_w];
    let zeros_bottom = x
        .const_f32_like(Arc::from(zeros_bottom_v), Shape::from_dims(&[1, cin, 1, new_w]));
    let x_padded = x_w.concat(&zeros_bottom, 2_usize).unwrap();

    let w_t = x.const_f32_like(w.clone(), Shape::from_dims(&[cout, cin, 3, 3]));
    let b_t = x.const_f32_like(b.clone(), Shape::from_dims(&[cout]));
    x_padded.conv2d(&w_t, Some(&b_t), (2, 2), (0, 0), 1).unwrap()
}

fn upsample_nearest_2x(x: &LazyTensor, c: usize, h: usize, w: usize) -> LazyTensor {
    let x6 = x.reshape(Shape::from_dims(&[1, c, h, 1, w, 1])).unwrap();
    let x6 = x6.concat(&x6, 3_usize).unwrap();
    let x6 = x6.concat(&x6, 5_usize).unwrap();
    x6.reshape(Shape::from_dims(&[1, c, 2 * h, 2 * w])).unwrap()
}

fn vae_resnet(
    x: &LazyTensor,
    rw: &VaeResnetWeights,
    cfg: &VaeConfig,
    c_in: usize,
    c_out: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let h1 = vae_group_norm(x, &rw.n1_g, &rw.n1_b, cfg.norm_num_groups, 1e-6, c_in, h, w);
    let h1 = h1.silu();
    let h1 = conv2d_k3_s1_p1(&h1, &rw.c1_w, &rw.c1_b, c_in, c_out);
    let h2 = vae_group_norm(&h1, &rw.n2_g, &rw.n2_b, cfg.norm_num_groups, 1e-6, c_out, h, w);
    let h2 = h2.silu();
    let h2 = conv2d_k3_s1_p1(&h2, &rw.c2_w, &rw.c2_b, c_out, c_out);
    let shortcut = match (&rw.shortcut_w, &rw.shortcut_b) {
        (Some(sw), Some(sb)) => {
            let w_t = x.const_f32_like(sw.clone(), Shape::from_dims(&[c_out, c_in, 1, 1]));
            let b_t = x.const_f32_like(sb.clone(), Shape::from_dims(&[c_out]));
            x.conv2d(&w_t, Some(&b_t), (1, 1), (0, 0), 1).unwrap()
        }
        _ => x.clone(),
    };
    shortcut.add(&h2).unwrap()
}

fn vae_spatial_attention(
    x: &LazyTensor,
    aw: &VaeAttnWeights,
    cfg: &VaeConfig,
    c: usize,
    h: usize,
    w: usize,
) -> LazyTensor {
    let n = h * w;
    let x_norm = vae_group_norm(x, &aw.gn_g, &aw.gn_b, cfg.norm_num_groups, 1e-6, c, h, w);
    let xf = x_norm.permute([0, 2, 3, 1_usize]).unwrap().reshape(Shape::from_dims(&[1, n, c])).unwrap();
    let q = linear(&xf, &aw.q_w, Some(&aw.q_b), c, c);
    let k = linear(&xf, &aw.k_w, Some(&aw.k_b), c, c);
    let v = linear(&xf, &aw.v_w, Some(&aw.v_b), c, c);
    let k_t = k.permute([0, 2, 1_usize]).unwrap();
    let scores = q.matmul(&k_t).unwrap().mul_scalar(1.0 / (c as f64).sqrt());
    let probs = scores.softmax_last_dim().unwrap();
    let ctx = probs.matmul(&v).unwrap();
    let out = linear(&ctx, &aw.out_w, Some(&aw.out_b), c, c);
    let out_chw = out.reshape(Shape::from_dims(&[1, h, w, c])).unwrap().permute([0, 3, 1, 2_usize]).unwrap();
    x.add(&out_chw).unwrap()
}

impl AutoEncoderKL {
    /// Encode RGB image `(1, 3, H, W)` -> latent `(1, latent_ch, H/8, W/8)`.
    /// Returns the mean of the diagonal-Gaussian (sample = false), then
    /// applies the scale/shift convention `(z - shift) * scale`.
    pub fn encode(&self, x: &LazyTensor) -> LazyTensor {
        let cfg = &self.config;
        let w = &self.weights;
        let dims = x.shape().dims().to_vec();
        let (mut h, mut wd) = (dims[2], dims[3]);
        let lc = cfg.latent_channels;

        let mut feat = conv2d_k3_s1_p1(x, &w.enc_conv_in_w, &w.enc_conv_in_b, cfg.in_channels, cfg.block_out_channels[0]);
        let mut c = cfg.block_out_channels[0];

        for (i, &out_c) in cfg.block_out_channels.iter().enumerate() {
            let block = &w.enc_down_blocks[i];
            for (ri, rb) in block.resnets.iter().enumerate() {
                let in_c = if ri == 0 { c } else { out_c };
                feat = vae_resnet(&feat, rb, cfg, in_c, out_c, h, wd);
            }
            c = out_c;
            if let Some((dw, db)) = &block.downsample_conv {
                feat = conv2d_k3_s2_p0_with_pad(&feat, dw, db, c, c, h, wd);
                h /= 2;
                wd /= 2;
            }
        }

        // Mid block.
        feat = vae_resnet(&feat, &w.enc_mid_resnet_1, cfg, c, c, h, wd);
        feat = vae_spatial_attention(&feat, &w.enc_mid_attn, cfg, c, h, wd);
        feat = vae_resnet(&feat, &w.enc_mid_resnet_2, cfg, c, c, h, wd);

        // conv_norm_out -> SiLU -> conv_out (-> 2*latent_channels).
        let feat = vae_group_norm(&feat, &w.enc_conv_norm_out_g, &w.enc_conv_norm_out_b, cfg.norm_num_groups, 1e-6, c, h, wd);
        let feat = feat.silu();
        let feat = conv2d_k3_s1_p1(&feat, &w.enc_conv_out_w, &w.enc_conv_out_b, c, 2 * lc);

        // Diagonal Gaussian: keep mean only (deterministic encode for tests).
        let mean = feat.narrow(1_usize, 0, lc).unwrap();
        // (z - shift) * scale.
        mean.add_scalar(-cfg.shift_factor).mul_scalar(cfg.scaling_factor)
    }

    /// Decode latent `(1, latent_ch, H/8, W/8)` -> RGB image `(1, 3, H, W)`.
    pub fn decode(&self, z: &LazyTensor) -> LazyTensor {
        let cfg = &self.config;
        let w = &self.weights;
        let dims = z.shape().dims().to_vec();
        let (mut h, mut wd) = (dims[2], dims[3]);

        // (z / scale + shift).
        let z = z.mul_scalar(1.0 / cfg.scaling_factor).add_scalar(cfg.shift_factor);

        let d_mid = *cfg.block_out_channels.last().unwrap();
        let mut feat = conv2d_k3_s1_p1(&z, &w.dec_conv_in_w, &w.dec_conv_in_b, cfg.latent_channels, d_mid);

        feat = vae_resnet(&feat, &w.dec_mid_resnet_1, cfg, d_mid, d_mid, h, wd);
        feat = vae_spatial_attention(&feat, &w.dec_mid_attn, cfg, d_mid, h, wd);
        feat = vae_resnet(&feat, &w.dec_mid_resnet_2, cfg, d_mid, d_mid, h, wd);

        let reversed: Vec<usize> = cfg.block_out_channels.iter().rev().cloned().collect();
        let mut c = d_mid;
        for (si, block) in w.dec_up_blocks.iter().enumerate() {
            let out_c = reversed[si];
            for (ri, rb) in block.resnets.iter().enumerate() {
                let in_c = if ri == 0 { c } else { out_c };
                feat = vae_resnet(&feat, rb, cfg, in_c, out_c, h, wd);
            }
            c = out_c;
            if let Some((uw, ub)) = &block.upsample_conv {
                feat = upsample_nearest_2x(&feat, c, h, wd);
                h *= 2;
                wd *= 2;
                feat = conv2d_k3_s1_p1(&feat, uw, ub, c, c);
            }
        }

        let feat = vae_group_norm(&feat, &w.dec_conv_norm_out_g, &w.dec_conv_norm_out_b, cfg.norm_num_groups, 1e-6, c, h, wd);
        let feat = feat.silu();
        conv2d_k3_s1_p1(&feat, &w.dec_conv_out_w, &w.dec_conv_out_b, c, cfg.out_channels)
    }
}

// ============================================================================
// FlowMatch Euler discrete scheduler (pure host-side math)
// ============================================================================

/// Configuration for [`FlowMatchEulerDiscreteScheduler`].
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub num_train_timesteps: usize,
    pub shift: f64,
    pub use_dynamic_shifting: bool,
}

impl SchedulerConfig {
    pub fn z_image_turbo() -> Self {
        Self {
            num_train_timesteps: 1000,
            shift: 3.0,
            use_dynamic_shifting: false,
        }
    }
}

/// FlowMatch Euler discrete scheduler. All math is host-side `f64`; the
/// scheduler outputs the step delta which is then applied to the latent
/// via [`LazyTensor::mul_scalar`] / `add_scalar`.
#[derive(Debug, Clone)]
pub struct FlowMatchEulerDiscreteScheduler {
    pub config: SchedulerConfig,
    pub timesteps: Vec<f64>,
    pub sigmas: Vec<f64>,
    step_index: usize,
}

impl FlowMatchEulerDiscreteScheduler {
    /// Build a fresh scheduler. Call [`set_timesteps`] before stepping.
    pub fn new(config: SchedulerConfig) -> Self {
        let n = config.num_train_timesteps;
        let mut timesteps: Vec<f64> = (1..=n).rev().map(|t| t as f64).collect();
        let mut sigmas: Vec<f64> = timesteps.iter().map(|&t| t / n as f64).collect();
        if !config.use_dynamic_shifting {
            let s = config.shift;
            sigmas = sigmas.iter().map(|&x| s * x / (1.0 + (s - 1.0) * x)).collect();
            timesteps = sigmas.iter().map(|&x| x * n as f64).collect();
        }
        Self {
            config,
            timesteps,
            sigmas,
            step_index: 0,
        }
    }

    /// Recompute timesteps / sigmas for a target inference-step count.
    pub fn set_timesteps(&mut self, num_inference_steps: usize, mu: Option<f64>) {
        let sigma_max = self.sigmas[0];
        let sigma_min = *self.sigmas.last().unwrap_or(&0.0);
        let n_train = self.config.num_train_timesteps as f64;
        let timesteps: Vec<f64> = (0..num_inference_steps).map(|i| {
            let t = i as f64 / num_inference_steps as f64;
            (sigma_max * (1.0 - t) + sigma_min * t) * n_train
        }).collect();
        let mut sigmas: Vec<f64> = timesteps.iter().map(|&t| t / n_train).collect();
        if let Some(mu) = mu {
            if self.config.use_dynamic_shifting {
                sigmas = sigmas.iter().map(|&t| {
                    if t <= 0.0 { 0.0 } else {
                        let e_mu = mu.exp();
                        e_mu / (e_mu + (1.0 / t - 1.0))
                    }
                }).collect();
            }
        } else if !self.config.use_dynamic_shifting {
            let s = self.config.shift;
            sigmas = sigmas.iter().map(|&x| s * x / (1.0 + (s - 1.0) * x)).collect();
        }
        sigmas.push(0.0);
        self.timesteps = timesteps;
        self.sigmas = sigmas;
        self.step_index = 0;
    }

    pub fn current_sigma(&self) -> f64 {
        self.sigmas[self.step_index]
    }

    /// Convert scheduler timestep to model input form (Z-Image expects `(1000 - t)/1000`).
    pub fn current_timestep_normalized(&self) -> f64 {
        let t = self.timesteps.get(self.step_index).copied().unwrap_or(0.0);
        (1000.0 - t) / 1000.0
    }

    /// Euler step: `x_{t-1} = x_t + (σ_{i+1} - σ_i) * v_t`.
    pub fn step(&mut self, model_output: &LazyTensor, sample: &LazyTensor) -> LazyTensor {
        let sigma = self.sigmas[self.step_index];
        let sigma_next = self.sigmas[self.step_index + 1];
        let dt = sigma_next - sigma;
        let next = sample.add(&model_output.mul_scalar(dt)).unwrap();
        self.step_index += 1;
        next
    }

    pub fn num_inference_steps(&self) -> usize { self.timesteps.len() }
    pub fn step_index(&self) -> usize { self.step_index }
    pub fn is_complete(&self) -> bool { self.step_index >= self.timesteps.len() }
}

/// Static shift schedule helper. Mirrors the eager `calculate_shift`.
pub fn calculate_shift(
    image_seq_len: usize,
    base_seq_len: usize,
    max_seq_len: usize,
    base_shift: f64,
    max_shift: f64,
) -> f64 {
    let m = (max_shift - base_shift) / (max_seq_len - base_seq_len) as f64;
    let b = base_shift - m * base_seq_len as f64;
    image_seq_len as f64 * m + b
}

// ============================================================================
// ZImageModel (top-level composition)
// ============================================================================

/// Top-level Z-Image composition. Wraps the three trainable components
/// and the scheduler so callers can drive the full pipeline through one
/// surface.
#[derive(Debug, Clone)]
pub struct ZImageModel {
    pub text_encoder: ZImageTextEncoder,
    pub transformer: ZImageTransformer2DModel,
    pub vae: AutoEncoderKL,
}

impl ZImageModel {
    /// Run the full diffusion pipeline. `tokens` is the tokenized
    /// prompt; we encode it, sample initial Gaussian noise from a
    /// deterministic linear-congruential generator seeded with `seed`,
    /// run `num_steps` Flow Matching steps, and decode the final
    /// latent through the VAE. The output is `(1, 3, H, W)` in [-1, 1].
    pub fn generate(
        &self,
        tokens: &[u32],
        latent_h: usize,
        latent_w: usize,
        num_steps: usize,
        seed: u64,
    ) -> crate::Result<LazyTensor> {
        if num_steps == 0 {
            return Err(crate::Error::Msg("generate: num_steps must be > 0".into()).bt());
        }
        // 1. Text encode.
        let cap_feats = self.text_encoder.forward(tokens)?;
        let text_len = cap_feats.shape().dims()[1];
        // cap_mask: all-1 over text_len (no padding in single-prompt path).
        let mask_v: Vec<f32> = vec![1.0; text_len];
        let cap_mask = cap_feats.const_f32_like(Arc::from(mask_v), Shape::from_dims(&[1, text_len]));

        // 2. Initial noise via deterministic LCG (host-side, then ported into a const).
        let cfg = &self.transformer.config;
        let c = cfg.in_channels;
        let numel = c * latent_h * latent_w;
        let noise_v = lcg_noise(seed, numel);
        let noise = cap_feats.const_f32_like(
            Arc::from(noise_v),
            Shape::from_dims(&[1, c, 1, latent_h, latent_w]),
        );

        // 3. Scheduler.
        let mut sched = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        sched.set_timesteps(num_steps, None);

        // 4. Denoising loop.
        let mut latent = noise;
        for _ in 0..num_steps {
            let t_norm = sched.current_timestep_normalized();
            let t = cap_feats.const_f32_like(
                Arc::from(vec![t_norm as f32]),
                Shape::from_dims(&[1]),
            );
            let v = self.transformer.forward(&latent, &t, &cap_feats, &cap_mask);
            latent = sched.step(&v, &latent);
        }

        // 5. VAE decode. (Strip the leading `F=1` axis for the conv stack.)
        let latent4 = latent.squeeze(2_usize)?;
        Ok(self.vae.decode(&latent4))
    }
}

/// Deterministic Gaussian-ish noise via a simple LCG + Box-Muller.
/// Used by the end-to-end test so we don't need an `Op::Randn`.
fn lcg_noise(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut next_u32 = || -> u32 {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (state >> 32) as u32
    };
    let mut next_f32 = || -> f32 {
        // uniform in (0, 1).
        let u = (next_u32() as f64 / u32::MAX as f64).max(1e-9).min(1.0 - 1e-9);
        u as f32
    };
    let mut out = Vec::with_capacity(n);
    while out.len() + 1 < n {
        let u1 = next_f32();
        let u2 = next_f32();
        let r = ((-2.0 * (u1 as f64).ln()) as f64).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2 as f64;
        out.push((r * theta.cos()) as f32);
        out.push((r * theta.sin()) as f32);
    }
    if out.len() < n {
        let u1 = next_f32();
        let u2 = next_f32();
        let r = ((-2.0 * (u1 as f64).ln()) as f64).sqrt();
        out.push((r * (2.0 * std::f64::consts::PI * u2 as f64).cos()) as f32);
    }
    out
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn arc(v: Vec<f32>) -> Arc<[f32]> { Arc::from(v) }

    fn det_seed(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed.wrapping_mul(2654435761).wrapping_add(0xDEADBEEF);
        move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 32) as u32) as f32 / u32::MAX as f32 - 0.5) * 0.05
        }
    }

    fn vec_of(n: usize, rng: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| rng()).collect::<Vec<_>>())
    }

    // ------ Transformer config used for tiny tests ------------------------

    fn tiny_transformer_cfg() -> ZImageConfig {
        ZImageConfig {
            patch_size: 2,
            f_patch_size: 1,
            in_channels: 2,
            dim: 16,
            n_layers: 1,
            n_refiner_layers: 1,
            n_heads: 2,
            n_kv_heads: 2,
            norm_eps: 1e-5,
            qk_norm: true,
            cap_feat_dim: 8,
            rope_theta: 256.0,
            t_scale: 1000.0,
            // head_dim=8 → axes_dims must sum to 8 (sum of axes_dims = head_dim)
            axes_dims: vec![2, 2, 4],
            axes_lens: vec![64, 64, 64],
        }
    }

    fn tiny_transformer_weights(cfg: &ZImageConfig) -> ZImageTransformerWeights {
        let mut rng = det_seed(12345);
        let dim = cfg.dim;
        let hidden = cfg.hidden_dim();
        let adaln_dim = dim.min(ADALN_EMBED_DIM);
        let head_dim = cfg.head_dim();
        let kv = cfg.n_kv_heads * head_dim;
        let patch_dim = cfg.f_patch_size * cfg.patch_size * cfg.patch_size * cfg.in_channels;
        let out_ch = cfg.patch_size * cfg.patch_size * cfg.f_patch_size * cfg.in_channels;

        let make_block = |rng: &mut dyn FnMut() -> f32, modulation: bool| ZImageBlockWeights {
            attn: ZImageAttnWeights {
                to_q_w: vec_of(dim * (cfg.n_heads * head_dim), rng),
                to_k_w: vec_of(dim * kv, rng),
                to_v_w: vec_of(dim * kv, rng),
                to_out_w: vec_of((cfg.n_heads * head_dim) * dim, rng),
                q_norm_gain: Some(arc(vec![1.0; head_dim])),
                k_norm_gain: Some(arc(vec![1.0; head_dim])),
            },
            ffn: ZImageFFNWeights {
                w1: vec_of(dim * hidden, rng),
                w2: vec_of(hidden * dim, rng),
                w3: vec_of(dim * hidden, rng),
            },
            attn_norm1_gain: arc(vec![1.0; dim]),
            attn_norm2_gain: arc(vec![1.0; dim]),
            ffn_norm1_gain: arc(vec![1.0; dim]),
            ffn_norm2_gain: arc(vec![1.0; dim]),
            adaln_w: if modulation { Some(vec_of(adaln_dim * (4 * dim), rng)) } else { None },
            adaln_b: if modulation { Some(arc(vec![0.0; 4 * dim])) } else { None },
        };

        ZImageTransformerWeights {
            t_embed_w1: vec_of(FREQUENCY_EMBEDDING_SIZE * 1024, &mut rng),
            t_embed_b1: arc(vec![0.0; 1024]),
            t_embed_w2: vec_of(1024 * adaln_dim, &mut rng),
            t_embed_b2: arc(vec![0.0; adaln_dim]),
            cap_norm_gain: arc(vec![1.0; cfg.cap_feat_dim]),
            cap_linear_w: vec_of(cfg.cap_feat_dim * dim, &mut rng),
            cap_linear_b: arc(vec![0.0; dim]),
            x_embed_w: vec_of(patch_dim * dim, &mut rng),
            x_embed_b: arc(vec![0.0; dim]),
            final_adaln_w: vec_of(adaln_dim * dim, &mut rng),
            final_adaln_b: arc(vec![0.0; dim]),
            final_linear_w: vec_of(dim * out_ch, &mut rng),
            final_linear_b: arc(vec![0.0; out_ch]),
            noise_refiner: (0..cfg.n_refiner_layers).map(|_| make_block(&mut rng, true)).collect(),
            context_refiner: (0..cfg.n_refiner_layers).map(|_| make_block(&mut rng, false)).collect(),
            layers: (0..cfg.n_layers).map(|_| make_block(&mut rng, true)).collect(),
        }
    }

    #[test]
    fn transformer_forward_shape_tiny() {
        let cfg = tiny_transformer_cfg();
        let weights = tiny_transformer_weights(&cfg);
        let model = ZImageTransformer2DModel { config: cfg.clone(), weights };

        // Tiny latent: 1 x 2 x 1 x 4 x 4 (so num_patches = 4).
        let c = cfg.in_channels;
        let h = 4; let w = 4;
        let x = LazyTensor::from_f32(
            vec![0.1_f32; c * h * w],
            Shape::from_dims(&[1, c, 1, h, w]),
            &crate::Device::cpu(),
        );
        let t = x.const_f32_like(Arc::from(vec![0.5_f32]), Shape::from_dims(&[1]));
        let cap = x.const_f32_like(Arc::from(vec![0.1_f32; 1 * 3 * cfg.cap_feat_dim]), Shape::from_dims(&[1, 3, cfg.cap_feat_dim]));
        let cap_mask = x.const_f32_like(Arc::from(vec![1.0_f32; 3]), Shape::from_dims(&[1, 3]));

        let out = model.forward(&x, &t, &cap, &cap_mask);
        assert_eq!(out.shape().dims(), &[1, c, 1, h, w]);
        let flat = out.realize_f32();
        assert!(flat.iter().all(|v| v.is_finite()), "non-finite transformer output");
    }

    // ------ Text encoder ------------------------------------------------

    fn tiny_text_cfg() -> TextEncoderConfig {
        TextEncoderConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000.0,
        }
    }

    fn tiny_text_weights(cfg: &TextEncoderConfig) -> TextEncoderWeights {
        let mut rng = det_seed(99);
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let layers: Vec<TextEncoderLayer> = (0..cfg.num_hidden_layers).map(|_| TextEncoderLayer {
            q_w: vec_of(h * (cfg.num_attention_heads * cfg.head_dim), &mut rng),
            k_w: vec_of(h * kv, &mut rng),
            v_w: vec_of(h * kv, &mut rng),
            o_w: vec_of((cfg.num_attention_heads * cfg.head_dim) * h, &mut rng),
            q_norm_gain: arc(vec![1.0; cfg.head_dim]),
            k_norm_gain: arc(vec![1.0; cfg.head_dim]),
            gate_w: vec_of(h * i, &mut rng),
            up_w: vec_of(h * i, &mut rng),
            down_w: vec_of(i * h, &mut rng),
            input_ln_gain: arc(vec![1.0; h]),
            post_attn_ln_gain: arc(vec![1.0; h]),
        }).collect();
        TextEncoderWeights {
            token_embedding: vec_of(cfg.vocab_size * h, &mut rng),
            layers,
        }
    }

    #[test]
    fn text_encoder_forward_shape_tiny() {
        let cfg = tiny_text_cfg();
        let weights = tiny_text_weights(&cfg);
        let enc = ZImageTextEncoder { config: cfg.clone(), weights };
        let tokens = [3_u32, 5, 7];
        let out = enc.forward(&tokens).unwrap();
        assert_eq!(out.shape().dims(), &[1, tokens.len(), cfg.hidden_size]);
        let flat = out.realize_f32();
        assert!(flat.iter().all(|v| v.is_finite()), "non-finite text encoder output");
    }

    // ------ VAE -----------------------------------------------------------

    fn tiny_vae_cfg() -> VaeConfig {
        // 2 stages so 4×4 input passes through 1 downsample to 2×2 latent.
        VaeConfig {
            in_channels: 3,
            out_channels: 3,
            latent_channels: 2,
            block_out_channels: vec![4, 4],
            layers_per_block: 1,
            scaling_factor: 1.0,
            shift_factor: 0.0,
            norm_num_groups: 2,
        }
    }

    fn tiny_vae_weights(cfg: &VaeConfig) -> VaeWeights {
        let mut rng = det_seed(31415);
        let lc = cfg.latent_channels;
        let n_stages = cfg.block_out_channels.len();

        let make_resnet = |rng: &mut dyn FnMut() -> f32, c_in: usize, c_out: usize| VaeResnetWeights {
            n1_g: arc(vec![1.0; c_in]),
            n1_b: arc(vec![0.0; c_in]),
            c1_w: vec_of(c_out * c_in * 9, rng),
            c1_b: arc(vec![0.0; c_out]),
            n2_g: arc(vec![1.0; c_out]),
            n2_b: arc(vec![0.0; c_out]),
            c2_w: vec_of(c_out * c_out * 9, rng),
            c2_b: arc(vec![0.0; c_out]),
            shortcut_w: if c_in != c_out { Some(vec_of(c_out * c_in, rng)) } else { None },
            shortcut_b: if c_in != c_out { Some(arc(vec![0.0; c_out])) } else { None },
        };

        let make_attn = |rng: &mut dyn FnMut() -> f32, c: usize| VaeAttnWeights {
            gn_g: arc(vec![1.0; c]),
            gn_b: arc(vec![0.0; c]),
            q_w: vec_of(c * c, rng),
            q_b: arc(vec![0.0; c]),
            k_w: vec_of(c * c, rng),
            k_b: arc(vec![0.0; c]),
            v_w: vec_of(c * c, rng),
            v_b: arc(vec![0.0; c]),
            out_w: vec_of(c * c, rng),
            out_b: arc(vec![0.0; c]),
        };

        let c_first = cfg.block_out_channels[0];
        let c_last = *cfg.block_out_channels.last().unwrap();

        // Encoder down blocks.
        let mut enc_down_blocks = Vec::new();
        for (i, &out_c) in cfg.block_out_channels.iter().enumerate() {
            let in_c = if i == 0 { c_first } else { cfg.block_out_channels[i - 1] };
            let resnets: Vec<_> = (0..cfg.layers_per_block).map(|ri| {
                let c_in = if ri == 0 { in_c } else { out_c };
                make_resnet(&mut rng, c_in, out_c)
            }).collect();
            let downsample_conv = if i < n_stages - 1 {
                Some((vec_of(out_c * out_c * 9, &mut rng), arc(vec![0.0; out_c])))
            } else { None };
            enc_down_blocks.push(VaeDownBlockWeights { resnets, downsample_conv });
        }

        // Decoder up blocks (reversed channels).
        let reversed: Vec<usize> = cfg.block_out_channels.iter().rev().cloned().collect();
        let mut dec_up_blocks = Vec::new();
        for (i, &out_c) in reversed.iter().enumerate() {
            let in_c = if i == 0 { c_last } else { reversed[i - 1] };
            let resnets: Vec<_> = (0..=cfg.layers_per_block).map(|ri| {
                let c_in = if ri == 0 { in_c } else { out_c };
                make_resnet(&mut rng, c_in, out_c)
            }).collect();
            let upsample_conv = if i < reversed.len() - 1 {
                Some((vec_of(out_c * out_c * 9, &mut rng), arc(vec![0.0; out_c])))
            } else { None };
            dec_up_blocks.push(VaeUpBlockWeights { resnets, upsample_conv });
        }

        VaeWeights {
            enc_conv_in_w: vec_of(c_first * cfg.in_channels * 9, &mut rng),
            enc_conv_in_b: arc(vec![0.0; c_first]),
            enc_down_blocks,
            enc_mid_resnet_1: make_resnet(&mut rng, c_last, c_last),
            enc_mid_attn: make_attn(&mut rng, c_last),
            enc_mid_resnet_2: make_resnet(&mut rng, c_last, c_last),
            enc_conv_norm_out_g: arc(vec![1.0; c_last]),
            enc_conv_norm_out_b: arc(vec![0.0; c_last]),
            enc_conv_out_w: vec_of((2 * lc) * c_last * 9, &mut rng),
            enc_conv_out_b: arc(vec![0.0; 2 * lc]),
            dec_conv_in_w: vec_of(c_last * lc * 9, &mut rng),
            dec_conv_in_b: arc(vec![0.0; c_last]),
            dec_mid_resnet_1: make_resnet(&mut rng, c_last, c_last),
            dec_mid_attn: make_attn(&mut rng, c_last),
            dec_mid_resnet_2: make_resnet(&mut rng, c_last, c_last),
            dec_up_blocks,
            dec_conv_norm_out_g: arc(vec![1.0; reversed[reversed.len() - 1]]),
            dec_conv_norm_out_b: arc(vec![0.0; reversed[reversed.len() - 1]]),
            dec_conv_out_w: vec_of(cfg.out_channels * reversed[reversed.len() - 1] * 9, &mut rng),
            dec_conv_out_b: arc(vec![0.0; cfg.out_channels]),
        }
    }

    #[test]
    fn vae_round_trip_tiny() {
        let cfg = tiny_vae_cfg();
        let weights = tiny_vae_weights(&cfg);
        let vae = AutoEncoderKL { config: cfg.clone(), weights };

        // 4×4 RGB image — 1 downsample stage → 2×2 latent.
        let h = 4; let w = 4;
        let x = LazyTensor::from_f32(
            vec![0.1_f32; cfg.in_channels * h * w],
            Shape::from_dims(&[1, cfg.in_channels, h, w]),
            &crate::Device::cpu(),
        );
        let z = vae.encode(&x);
        assert_eq!(z.shape().dims(), &[1, cfg.latent_channels, h / 2, w / 2]);
        let img = vae.decode(&z);
        assert_eq!(img.shape().dims(), &[1, cfg.out_channels, h, w]);
        let flat = img.realize_f32();
        assert!(flat.iter().all(|v| v.is_finite()), "non-finite VAE output");
    }

    // ------ Scheduler -----------------------------------------------------

    #[test]
    fn scheduler_step_finite() {
        let cfg = SchedulerConfig::z_image_turbo();
        let mut sched = FlowMatchEulerDiscreteScheduler::new(cfg);
        sched.set_timesteps(8, None);
        assert_eq!(sched.num_inference_steps(), 8);

        let sample = LazyTensor::from_f32(
            vec![0.5_f32; 4],
            Shape::from_dims(&[1, 4]),
            &crate::Device::cpu(),
        );
        let v = sample.const_f32_like(Arc::from(vec![0.1_f32; 4]), Shape::from_dims(&[1, 4]));
        let mut latent = sample.clone();
        while !sched.is_complete() {
            latent = sched.step(&v, &latent);
        }
        let flat = latent.realize_f32();
        assert!(flat.iter().all(|x| x.is_finite()), "non-finite scheduler output");
        assert_eq!(sched.step_index(), 8);
    }

    // ------ End-to-end -----------------------------------------------------

    #[test]
    fn generate_end_to_end_tiny() {
        // We exercise the transformer + scheduler + VAE chain. The text
        // encoder is wired but exercised separately; here we feed a
        // pre-built tiny cap_feats vector to keep the test fast.
        let tcfg = tiny_transformer_cfg();
        let tw = tiny_transformer_weights(&tcfg);
        let transformer = ZImageTransformer2DModel { config: tcfg.clone(), weights: tw };

        let vcfg = tiny_vae_cfg();
        let vw = tiny_vae_weights(&vcfg);
        let vae = AutoEncoderKL { config: vcfg.clone(), weights: vw };

        // Tiny 4×4 latent (matches transformer config) — VAE up-samples
        // 1 stage to 8×8 RGB.
        let h_lat = 4; let w_lat = 4;
        let c = tcfg.in_channels;
        let noise = LazyTensor::from_f32(
            vec![0.1_f32; c * h_lat * w_lat],
            Shape::from_dims(&[1, c, 1, h_lat, w_lat]),
            &crate::Device::cpu(),
        );
        // We can't share the cfg between transformer (in_channels=2) and
        // VAE (latent_channels=2): both happen to be 2 here, so we can
        // hand the transformer output to the VAE directly.
        assert_eq!(tcfg.in_channels, vcfg.latent_channels);

        let cap = noise.const_f32_like(
            Arc::from(vec![0.05_f32; 1 * 2 * tcfg.cap_feat_dim]),
            Shape::from_dims(&[1, 2, tcfg.cap_feat_dim]),
        );
        let cap_mask = noise.const_f32_like(
            Arc::from(vec![1.0_f32; 2]),
            Shape::from_dims(&[1, 2]),
        );

        let mut sched = FlowMatchEulerDiscreteScheduler::new(SchedulerConfig::z_image_turbo());
        sched.set_timesteps(2, None);
        let mut latent = noise;
        for _ in 0..2 {
            let t_norm = sched.current_timestep_normalized() as f32;
            let t = latent.const_f32_like(Arc::from(vec![t_norm]), Shape::from_dims(&[1]));
            let v = transformer.forward(&latent, &t, &cap, &cap_mask);
            latent = sched.step(&v, &latent);
        }

        // Strip frame axis and decode.
        let latent4 = latent.squeeze(2_usize).unwrap();
        let img = vae.decode(&latent4);
        assert_eq!(img.shape().dims(), &[1, vcfg.out_channels, h_lat * 2, w_lat * 2]);
        let flat = img.realize_f32();
        assert!(flat.iter().all(|x| x.is_finite()), "non-finite end-to-end output");
    }
}
