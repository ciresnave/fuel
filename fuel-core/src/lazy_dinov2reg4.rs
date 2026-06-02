//! DINOv2-reg4 (Darcet et al. 2023, "Vision Transformers
//! Need Registers") ported to the lazy-graph API.
//!
//! DINOv2 + four learned **register tokens** between the CLS
//! token and patch tokens. The registers absorb high-norm
//! "artifact" patches that vanilla DINOv2 attention attaches
//! to in late layers, giving cleaner feature maps for
//! downstream tasks. The fuel-transformers eager port targets
//! the PlantCLEF2024 plant-species classifier (7806 classes,
//! ViT-Small/14 backbone) but the architecture is general.
//!
//! # Departures from `lazy_dinov2`
//!
//!   1. **4 register tokens** prepended after CLS, before
//!      patches. Sequence is `[cls, reg₁, reg₂, reg₃, reg₄,
//!      patch₁, …, patch_N]` with total length `5 + N`.
//!   2. **Position embedding only on patches.** Stored as
//!      `(1, num_patches, embed_dim)` and added BEFORE CLS /
//!      registers are concatenated. CLS and registers carry
//!      no positional information — only attention identity.
//!   3. **Classifier head on CLS only**, not on
//!      `cat(cls, mean(patches))`. Maps `embed_dim → num_classes`
//!      with bias. CLS slot is `seq[0]` after the encoder + LN.
//!
//! Everything else (fused Wqkv, LayerScale on attn + MLP,
//! Pre-LN, biased Q/K/V + projection, GELU MLP) matches
//! lazy_dinov2 verbatim.
//!
//! # Scope (v1)
//!
//! Forward-only, single fixed-size image, F32. Variable
//! input size + bicubic position-embedding interpolation
//! deferred (eager uses `upsample_nearest2d` as a workaround
//! anyway — both are out of scope for v1).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Dinov2Reg4Config {
    pub embed_dim: usize,
    pub depth: usize,
    pub num_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub mlp_ratio: usize,
    pub layer_norm_eps: f64,
    pub num_classes: usize,
    /// Number of register tokens. Canonical DINOv2-reg uses 4.
    pub num_register_tokens: usize,
}

impl Dinov2Reg4Config {
    pub fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }
    pub fn num_patches(&self) -> usize {
        let p = self.num_patches_per_side();
        p * p
    }
    pub fn seq_len(&self) -> usize {
        1 + self.num_register_tokens + self.num_patches()
    }
    pub fn mlp_hidden(&self) -> usize {
        self.embed_dim * self.mlp_ratio
    }
    /// DINOv2-reg4 ViT-Small/14 trained for PlantCLEF2024:
    /// 12 layers × 384 dim × 6 heads, 518×518 image with
    /// patch 14, 7806 classes.
    pub fn vit_small_plantclef() -> Self {
        Self {
            embed_dim: 384, depth: 12, num_heads: 6,
            num_channels: 3, image_size: 518, patch_size: 14,
            mlp_ratio: 4, layer_norm_eps: 1e-6,
            num_classes: 7806,
            num_register_tokens: 4,
        }
    }
    /// DINOv2-reg4 ViT-Base/14 backbone variant.
    pub fn vit_base() -> Self {
        Self {
            embed_dim: 768, depth: 12, num_heads: 12,
            num_channels: 3, image_size: 518, patch_size: 14,
            mlp_ratio: 4, layer_norm_eps: 1e-6,
            num_classes: 1000,
            num_register_tokens: 4,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Dinov2Reg4BlockWeights {
    pub norm1_gain: Arc<[f32]>,
    pub norm1_bias: Arc<[f32]>,
    /// Fused `[embed_dim, 3 * embed_dim]`.
    pub qkv: WeightStorage,
    pub qkv_bias: Arc<[f32]>,
    pub proj: WeightStorage,
    pub proj_bias: Arc<[f32]>,
    pub ls1_gamma: Arc<[f32]>,
    pub norm2_gain: Arc<[f32]>,
    pub norm2_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
    pub ls2_gamma: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Dinov2Reg4Weights {
    /// `[embed_dim, num_channels, patch, patch]`.
    pub patch_proj: Arc<[f32]>,
    pub patch_proj_bias: Arc<[f32]>,
    /// `[1, 1, embed_dim]`.
    pub cls_token: Arc<[f32]>,
    /// `[1, num_register_tokens, embed_dim]`.
    pub reg_token: Arc<[f32]>,
    /// `[1, num_patches, embed_dim]`. Added to patches only.
    pub pos_embed: Arc<[f32]>,
    pub blocks: Vec<Dinov2Reg4BlockWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    /// `[embed_dim, num_classes]` linear classifier on CLS.
    pub head: WeightStorage,
    pub head_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Dinov2Reg4Model {
    pub config: Dinov2Reg4Config,
    pub weights: Dinov2Reg4Weights,
}

impl Dinov2Reg4Model {
    /// Run image classification. `pixel_values` is `(1, 3, H, W)`
    /// with `H == W == cfg.image_size`. Returns `(1, num_classes)`.
    pub fn forward(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        let batch = dims[0];
        assert_eq!(batch, 1, "v1 supports batch == 1");
        assert_eq!(dims[1], cfg.num_channels);
        assert_eq!(dims[2], cfg.image_size,
            "input H must equal cfg.image_size (variable input deferred)");
        assert_eq!(dims[3], cfg.image_size,
            "input W must equal cfg.image_size (variable input deferred)");

        let h = cfg.embed_dim;
        let np = cfg.num_patches();
        let n_reg = cfg.num_register_tokens;

        // ---- Patch Conv2d → (B, embed_dim, P, P) → reshape → permute ------
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[h, cfg.num_channels, cfg.patch_size, cfg.patch_size]),
        );
        let conv_b = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj_bias),
            Shape::from_dims(&[h]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w,
            Some(&conv_b),
            (cfg.patch_size, cfg.patch_size),
            (0, 0),
            1,
        )?;
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, h, np]))?
            .permute([0, 2, 1_usize])?;

        // ---- Add learned position embedding (patches only) -----------------
        let pos = pixel_values.const_f32_like(
            Arc::clone(&weights.pos_embed),
            Shape::from_dims(&[np, h]),
        );
        let pos_bc = pos
            .reshape(Shape::from_dims(&[1, np, h]))?
            .broadcast_to(Shape::from_dims(&[batch, np, h]))?;
        let patches = patches.add(&pos_bc)?;

        // ---- Prepend CLS + register tokens ---------------------------------
        let cls = pixel_values.const_f32_like(
            Arc::clone(&weights.cls_token),
            Shape::from_dims(&[1, 1, h]),
        );
        let cls_bc = cls.broadcast_to(Shape::from_dims(&[batch, 1, h]))?;
        let reg = pixel_values.const_f32_like(
            Arc::clone(&weights.reg_token),
            Shape::from_dims(&[1, n_reg, h]),
        );
        let reg_bc = reg.broadcast_to(Shape::from_dims(&[batch, n_reg, h]))?;
        let cls_reg = cls_bc.concat(&reg_bc, 1_usize)?;
        let mut x = cls_reg.concat(&patches, 1_usize)?;
        assert_eq!(
            x.shape().dims(),
            &[batch, cfg.seq_len(), h],
            "post-cat sequence length mismatch",
        );

        // ---- Encoder blocks ------------------------------------------------
        for block in &weights.blocks {
            x = self.apply_block(&x, block)?;
        }

        // ---- Final LayerNorm + CLS classifier -----------------------------
        let x_norm = crate::lazy::apply_affine_layer_norm_pub(
            &x,
            &weights.final_ln_gain, &weights.final_ln_bias,
            h, cfg.layer_norm_eps,
        );
        let cls_feat = x_norm
            .slice(1_usize, 0, 1)?
            .reshape(Shape::from_dims(&[batch, h]))?;
        let logits = weights.head.apply_linear(&cls_feat, h, cfg.num_classes);
        let bias_t = pixel_values.const_f32_like(
            Arc::clone(&weights.head_bias),
            Shape::from_dims(&[cfg.num_classes]),
        );
        logits.broadcast_add(&bias_t)
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        block: &Dinov2Reg4BlockWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.embed_dim;
        let n_heads = cfg.num_heads;
        let head_dim = cfg.head_dim();

        // Attention sublayer with Pre-LN + LayerScale.
        let x_norm = crate::lazy::apply_affine_layer_norm_pub(
            x, &block.norm1_gain, &block.norm1_bias, h, cfg.layer_norm_eps,
        );
        let qkv_lin = block.qkv.apply_linear(&x_norm, h, 3 * h);
        let qkv_bias_t = x.const_f32_like(
            Arc::clone(&block.qkv_bias),
            Shape::from_dims(&[3 * h]),
        );
        let qkv = qkv_lin.broadcast_add(&qkv_bias_t)?;
        let q = qkv.slice(2_usize, 0, h)?;
        let k = qkv.slice(2_usize, h, h)?;
        let v = qkv.slice(2_usize, 2 * h, h)?;

        let q = q
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k = k
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v = v
            .reshape(Shape::from_dims(&[batch, seq, n_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose()?)?.mul_scalar(scale);
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx
            .permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[batch, seq, h]))?;
        let proj = block.proj.apply_linear(&merged, h, h);
        let proj_b_t = x.const_f32_like(
            Arc::clone(&block.proj_bias),
            Shape::from_dims(&[h]),
        );
        let attn_out = proj.broadcast_add(&proj_b_t)?;

        // LayerScale 1: per-channel gamma multiplier BEFORE residual.
        let ls1_t = x.const_f32_like(
            Arc::clone(&block.ls1_gamma),
            Shape::from_dims(&[h]),
        );
        let attn_scaled = attn_out.broadcast_mul(&ls1_t)?;
        let h1 = x.add(&attn_scaled)?;

        // MLP sublayer with Pre-LN + LayerScale.
        let h1_norm = crate::lazy::apply_affine_layer_norm_pub(
            &h1, &block.norm2_gain, &block.norm2_bias, h, cfg.layer_norm_eps,
        );
        let mlp_hidden = cfg.mlp_hidden();
        let fc1 = block.fc1.apply_linear(&h1_norm, h, mlp_hidden);
        let fc1_b_t = x.const_f32_like(
            Arc::clone(&block.fc1_bias),
            Shape::from_dims(&[mlp_hidden]),
        );
        let fc1 = fc1.broadcast_add(&fc1_b_t)?.gelu_erf();
        let fc2 = block.fc2.apply_linear(&fc1, mlp_hidden, h);
        let fc2_b_t = x.const_f32_like(
            Arc::clone(&block.fc2_bias),
            Shape::from_dims(&[h]),
        );
        let mlp_out = fc2.broadcast_add(&fc2_b_t)?;

        // LayerScale 2.
        let ls2_t = x.const_f32_like(
            Arc::clone(&block.ls2_gamma),
            Shape::from_dims(&[h]),
        );
        let mlp_scaled = mlp_out.broadcast_mul(&ls2_t)?;
        h1.add(&mlp_scaled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    fn tiny_cfg() -> Dinov2Reg4Config {
        Dinov2Reg4Config {
            embed_dim: 16, depth: 2, num_heads: 4,
            num_channels: 3, image_size: 28, patch_size: 14,
            mlp_ratio: 2, layer_norm_eps: 1e-6,
            num_classes: 5,
            num_register_tokens: 4,
        }
    }

    fn tiny_weights(cfg: &Dinov2Reg4Config, seed: u32) -> Dinov2Reg4Weights {
        let mut nb = rng_seed(seed);
        let h = cfg.embed_dim;
        let np = cfg.num_patches();
        let patch_proj = vec_of(h * cfg.num_channels * cfg.patch_size * cfg.patch_size, &mut nb);
        let patch_proj_bias = vec_of(h, &mut nb);
        let cls_token = vec_of(h, &mut nb);
        let reg_token = vec_of(cfg.num_register_tokens * h, &mut nb);
        let pos_embed = vec_of(np * h, &mut nb);

        let mlp_hidden = cfg.mlp_hidden();
        let blocks: Vec<Dinov2Reg4BlockWeights> = (0..cfg.depth)
            .map(|_| Dinov2Reg4BlockWeights {
                norm1_gain: Arc::from(vec![1.0_f32; h]),
                norm1_bias: Arc::from(vec![0.0_f32; h]),
                qkv: WeightStorage::F32(vec_of(h * 3 * h, &mut nb)),
                qkv_bias: vec_of(3 * h, &mut nb),
                proj: WeightStorage::F32(vec_of(h * h, &mut nb)),
                proj_bias: vec_of(h, &mut nb),
                ls1_gamma: Arc::from(vec![1.0_f32; h]),
                norm2_gain: Arc::from(vec![1.0_f32; h]),
                norm2_bias: Arc::from(vec![0.0_f32; h]),
                fc1: WeightStorage::F32(vec_of(h * mlp_hidden, &mut nb)),
                fc1_bias: vec_of(mlp_hidden, &mut nb),
                fc2: WeightStorage::F32(vec_of(mlp_hidden * h, &mut nb)),
                fc2_bias: vec_of(h, &mut nb),
                ls2_gamma: Arc::from(vec![1.0_f32; h]),
            })
            .collect();
        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let head = WeightStorage::F32(vec_of(h * cfg.num_classes, &mut nb));
        let head_bias = vec_of(cfg.num_classes, &mut nb);

        Dinov2Reg4Weights {
            patch_proj, patch_proj_bias,
            cls_token, reg_token, pos_embed,
            blocks, final_ln_gain, final_ln_bias,
            head, head_bias,
        }
    }

    fn tiny_image(cfg: &Dinov2Reg4Config) -> LazyTensor {
        let mut nb = rng_seed(7);
        let n = cfg.num_channels * cfg.image_size * cfg.image_size;
        let data: Arc<[f32]> = Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>());
        LazyTensor::from_f32(
            data,
            Shape::from_dims(&[1, cfg.num_channels, cfg.image_size, cfg.image_size]),
            &Device::cpu(),
        )
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = tiny_cfg();
        let model = Dinov2Reg4Model {
            config: cfg.clone(), weights: tiny_weights(&cfg, 11),
        };
        let img = tiny_image(&cfg);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, cfg.num_classes]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// Sequence length is `1 (cls) + 4 (reg) + num_patches`.
    /// For 28×28 input with patch 14, num_patches = 4 → seq = 9.
    #[test]
    fn seq_len_includes_cls_and_registers() {
        let cfg = tiny_cfg();
        assert_eq!(cfg.num_patches(), 4);
        assert_eq!(cfg.seq_len(), 9);
    }

    /// Register tokens are wired: changing any register token
    /// row must change the CLS output (the classifier feature
    /// is taken from CLS, and CLS attends across the full
    /// sequence including registers).
    #[test]
    fn register_tokens_affect_cls_output() {
        let cfg = tiny_cfg();
        let base = tiny_weights(&cfg, 99);
        let mut modified = base.clone();
        let h = cfg.embed_dim;
        let n_reg = cfg.num_register_tokens;
        // Zero out the register tokens; with non-trivial weights
        // elsewhere this must alter the CLS classifier output.
        modified.reg_token = Arc::from(vec![0.0_f32; n_reg * h]);

        let m_a = Dinov2Reg4Model { config: cfg.clone(), weights: base };
        let m_b = Dinov2Reg4Model { config: cfg.clone(), weights: modified };
        let img = tiny_image(&cfg);
        let a = m_a.forward(&img).unwrap().realize_f32();
        let b = m_b.forward(&img).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "zeroing register tokens must alter CLS classifier output, max_diff = {max_diff}");
    }

    /// Classifier head is on CLS (slot 0), not on
    /// `cat(cls, mean(patches))` like vanilla DINOv2.
    /// Verify by replacing pos_embed (which only adds to
    /// patches) with zeros — patch contributions remain
    /// non-zero but their pos-coded signal disappears.
    /// The CLS classifier output must still vary because
    /// CLS attends to (and is influenced by) the patches.
    #[test]
    fn pos_embed_only_on_patches() {
        let cfg = tiny_cfg();
        let base = tiny_weights(&cfg, 42);
        let mut modified = base.clone();
        let np = cfg.num_patches();
        let h = cfg.embed_dim;
        // Zero pos_embed → patches lose positional offsets but
        // still carry the patch-embed features. CLS classifier
        // output must change.
        modified.pos_embed = Arc::from(vec![0.0_f32; np * h]);

        let m_a = Dinov2Reg4Model { config: cfg.clone(), weights: base };
        let m_b = Dinov2Reg4Model { config: cfg.clone(), weights: modified };
        let img = tiny_image(&cfg);
        let a = m_a.forward(&img).unwrap().realize_f32();
        let b = m_b.forward(&img).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-7,
            "zeroing pos_embed must alter CLS output (pos signal influences CLS via attention), \
             max_diff = {max_diff}");
    }
}
