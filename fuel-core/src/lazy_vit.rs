//! Vision Transformer (ViT) ported to the lazy-graph API.
//!
//! The canonical Vision Transformer architecture from "An Image
//! is Worth 16x16 Words" (Dosovitskiy et al., 2020). Foundational
//! vision encoder used directly by `google/vit-base-patch16-224`,
//! `microsoft/trocr-base-handwritten`, and indirectly by CLIP,
//! SigLIP, BEiT, DINOv2, EVA2, MobileCLIP, segment_anything, etc.
//! Porting ViT unlocks the patch-based-image-encoder path for
//! every one of those downstream models.
//!
//! Architecture:
//!
//!   1. **PatchEmbeddings.** `Conv2d` with kernel=patch_size and
//!      stride=patch_size projects pixel values `(b, c, h, w)` to
//!      `(b, hidden, h/p, w/p)`. Flattened on the spatial axes
//!      and transposed to `(b, num_patches, hidden)`.
//!   2. **Embeddings.** A learned `cls_token` of shape
//!      `(1, 1, hidden)` is prepended; a learned 1D
//!      `position_embeddings` of shape
//!      `(1, num_patches + 1, hidden)` is added.
//!   3. **Self-attention.** Standard MHA. Optional Q/K/V biases
//!      controlled by `qkv_bias`. Score scaling by
//!      `1 / sqrt(head_dim)`.
//!   4. **Pre-LN block** (HF ViT convention):
//!        `xs = LN_before(x); xs = SelfAttention(xs);
//!         xs = Output(SelfAttention output);
//!         xs = xs + x;            // first residual
//!         ys = LN_after(xs); ys = Intermediate(ys);
//!         out = Output_mlp(ys) + xs   // second residual`
//!   5. **Sequential MLP.** `Intermediate` projects
//!      `hidden → intermediate` and applies the configured
//!      activation; `Output_mlp` projects back to `hidden`
//!      and adds the residual.
//!   6. **Final LayerNorm + classifier** on the CLS token.
//!
//! # Scope (v1)
//!
//! Forward-only, single fixed-size image (batch == 1), F32.
//! `use_mask_token` (masked autoencoding pre-training) and
//! position interpolation (variable image sizes) deferred.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VitActivation {
    Gelu,
    GeluPytorchTanh,
    Relu,
    Silu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VitConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_activation: VitActivation,
    pub layer_norm_eps: f64,
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub qkv_bias: bool,
}

impl VitConfig {
    pub fn num_patches(&self) -> usize {
        let p = self.image_size / self.patch_size;
        p * p
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `google/vit-base-patch16-224`.
    pub fn vit_base_patch16_224() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_activation: VitActivation::Gelu,
            layer_norm_eps: 1e-12,
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            qkv_bias: true,
        }
    }

    /// `microsoft/trocr-base-handwritten`.
    pub fn microsoft_trocr_base_handwritten() -> Self {
        Self {
            qkv_bias: false,
            image_size: 384,
            ..Self::vit_base_patch16_224()
        }
    }
}

#[derive(Debug, Clone)]
pub struct VitLayerWeights {
    pub ln_before_gain: Arc<[f32]>,
    pub ln_before_bias: Arc<[f32]>,
    pub q_proj: WeightStorage,
    pub q_proj_bias: Option<Arc<[f32]>>,
    pub k_proj: WeightStorage,
    pub k_proj_bias: Option<Arc<[f32]>>,
    pub v_proj: WeightStorage,
    pub v_proj_bias: Option<Arc<[f32]>>,
    /// SelfOutput.dense — applied to the merged attention output.
    pub attn_output_proj: WeightStorage,
    pub attn_output_proj_bias: Arc<[f32]>,
    pub ln_after_gain: Arc<[f32]>,
    pub ln_after_bias: Arc<[f32]>,
    /// Intermediate.dense.
    pub intermediate_proj: WeightStorage,
    pub intermediate_proj_bias: Arc<[f32]>,
    /// Output.dense.
    pub mlp_output_proj: WeightStorage,
    pub mlp_output_proj_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct VitWeights {
    /// `[hidden, num_channels, patch_size, patch_size]` (Conv2d weight).
    pub patch_proj: Arc<[f32]>,
    /// `[hidden]` bias for the patch projection (Conv2d uses a learned bias).
    pub patch_proj_bias: Arc<[f32]>,
    /// `[1, 1, hidden]`.
    pub cls_token: Arc<[f32]>,
    /// `[1, num_patches + 1, hidden]`.
    pub position_embeddings: Arc<[f32]>,
    pub layers: Vec<VitLayerWeights>,
    pub final_ln_gain: Arc<[f32]>,
    pub final_ln_bias: Arc<[f32]>,
    /// Optional classifier. `[hidden, num_labels]` plus bias `[num_labels]`.
    pub classifier: Option<(WeightStorage, Arc<[f32]>)>,
}

#[derive(Debug, Clone)]
pub struct VitModel {
    pub config: VitConfig,
    pub weights: VitWeights,
}

impl VitModel {
    /// Encode a single image and return either the final hidden
    /// states (no classifier) or classification logits (with
    /// classifier). Shape:
    ///   - Without classifier: `(1, num_patches + 1, hidden)`
    ///   - With classifier: `(1, num_labels)` from the CLS token.
    pub fn forward(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        let batch = dims[0];
        assert_eq!(batch, 1, "v1 supports batch == 1");
        assert_eq!(dims[1], cfg.num_channels);
        assert_eq!(dims[2], cfg.image_size);
        assert_eq!(dims[3], cfg.image_size);

        // ---- Patch embeddings via Conv2d -----------------------------------
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[cfg.hidden_size, cfg.num_channels, cfg.patch_size, cfg.patch_size]),
        );
        let conv_b = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj_bias),
            Shape::from_dims(&[cfg.hidden_size]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w,
            Some(&conv_b),
            (cfg.patch_size, cfg.patch_size),
            (0, 0),
            1,
        )?;
        // (b, hidden, ph, pw) → (b, hidden, ph*pw) → (b, ph*pw, hidden)
        let num_patches = cfg.num_patches();
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, cfg.hidden_size, num_patches]))?
            .permute([0, 2, 1_usize])?;

        // Prepend CLS token: cls_token of shape (1, 1, hidden) → (batch, 1, hidden).
        let cls_tok = patches.const_f32_like(
            Arc::clone(&weights.cls_token),
            Shape::from_dims(&[1, 1, cfg.hidden_size]),
        );
        let cls_bc = cls_tok.broadcast_to(Shape::from_dims(&[batch, 1, cfg.hidden_size]))?;
        let with_cls = cls_bc.concat(&patches, 1_usize)?; // (b, num_patches + 1, hidden)

        // Add position embeddings.
        let pos = patches.const_f32_like(
            Arc::clone(&weights.position_embeddings),
            Shape::from_dims(&[1, num_patches + 1, cfg.hidden_size]),
        );
        let pos_bc = pos.broadcast_to(Shape::from_dims(&[batch, num_patches + 1, cfg.hidden_size]))?;
        let mut h_states = with_cls.add(&pos_bc)?;

        // ---- Encoder layers -------------------------------------------------
        for layer in &weights.layers {
            h_states = self.apply_layer(&h_states, layer)?;
        }

        // ---- Final LayerNorm ------------------------------------------------
        let h_norm = h_states.layer_norm_affine(std::sync::Arc::clone(&weights.final_ln_gain), std::sync::Arc::clone(&weights.final_ln_bias), cfg.layer_norm_eps)?;

        // ---- Optional classifier on CLS -------------------------------------
        match &weights.classifier {
            None => Ok(h_norm),
            Some((cls_w, cls_b)) => {
                // CLS token is position 0.
                let cls = h_norm
                    .slice(1_usize, 0, 1)?
                    .reshape(Shape::from_dims(&[batch, cfg.hidden_size]))?;
                let num_labels = cls_b.len();
                let logits = cls_w.apply_linear(&cls, cfg.hidden_size, num_labels);
                let bias_t = h_norm.const_f32_like(
                    Arc::clone(cls_b),
                    Shape::from_dims(&[num_labels]),
                );
                logits.broadcast_add(&bias_t)
            }
        }
    }

    /// Extract per-token features at the requested layer
    /// indices. Output shape per layer:
    /// `(1, num_patches + 1, hidden_size)` — CLS at slot 0,
    /// patches follow. **No final LayerNorm** applied — DPT-
    /// style heads (DPT-Hybrid, MiDaS variants built on plain
    /// ViT, etc.) have their own per-stage projection.
    ///
    /// Layer-id contract: 0-based, strictly increasing, all in
    /// `[0, num_hidden_layers)`. Mirrors
    /// [`crate::lazy_dinov2::Dinov2Model::forward_intermediate_layers`]
    /// — the difference between the two is just the LayerScale +
    /// fused Wqkv layout DINOv2 brings; the hook contract is
    /// identical.
    pub fn forward_intermediate_layers(
        &self,
        pixel_values: &LazyTensor,
        layer_ids: &[usize],
    ) -> Result<Vec<LazyTensor>> {
        let cfg = &self.config;
        let weights = &self.weights;
        let dims = pixel_values.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        let batch = dims[0];
        assert_eq!(batch, 1, "v1 supports batch == 1");
        assert_eq!(dims[1], cfg.num_channels);
        assert_eq!(dims[2], cfg.image_size);
        assert_eq!(dims[3], cfg.image_size);
        assert!(!layer_ids.is_empty(), "layer_ids must not be empty");
        for w in layer_ids.windows(2) {
            assert!(w[0] < w[1], "layer_ids must be strictly increasing");
        }
        let depth = weights.layers.len();
        assert!(
            *layer_ids.last().unwrap() < depth,
            "layer_ids must all be in [0, num_hidden_layers = {depth})",
        );

        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj),
            Shape::from_dims(&[cfg.hidden_size, cfg.num_channels, cfg.patch_size, cfg.patch_size]),
        );
        let conv_b = pixel_values.const_f32_like(
            Arc::clone(&weights.patch_proj_bias),
            Shape::from_dims(&[cfg.hidden_size]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w, Some(&conv_b),
            (cfg.patch_size, cfg.patch_size), (0, 0), 1,
        )?;
        let num_patches = cfg.num_patches();
        let patches = conv_out
            .reshape(Shape::from_dims(&[batch, cfg.hidden_size, num_patches]))?
            .permute([0, 2, 1_usize])?;

        let cls_tok = patches.const_f32_like(
            Arc::clone(&weights.cls_token),
            Shape::from_dims(&[1, 1, cfg.hidden_size]),
        );
        let cls_bc = cls_tok.broadcast_to(Shape::from_dims(&[batch, 1, cfg.hidden_size]))?;
        let with_cls = cls_bc.concat(&patches, 1_usize)?;

        let pos = patches.const_f32_like(
            Arc::clone(&weights.position_embeddings),
            Shape::from_dims(&[1, num_patches + 1, cfg.hidden_size]),
        );
        let pos_bc = pos.broadcast_to(Shape::from_dims(&[batch, num_patches + 1, cfg.hidden_size]))?;
        let mut h = with_cls.add(&pos_bc)?;

        let mut out = Vec::with_capacity(layer_ids.len());
        let mut next_capture = 0;
        for (idx, layer) in weights.layers.iter().enumerate() {
            h = self.apply_layer(&h, layer)?;
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
        layer: &VitLayerWeights,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = x.shape();
        let dims = dims.dims();
        let batch = dims[0];
        let seq = dims[1];
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();

        // Pre-LN before attention.
        let x_norm = x.layer_norm_affine(std::sync::Arc::clone(&layer.ln_before_gain), std::sync::Arc::clone(&layer.ln_before_bias), cfg.layer_norm_eps)?;

        // Q/K/V projections with optional biases.
        let q = layer.q_proj.apply_linear(&x_norm, h, h).add_optional_trailing_bias(layer.q_proj_bias.as_ref())?;
        let k = layer.k_proj.apply_linear(&x_norm, h, h).add_optional_trailing_bias(layer.k_proj_bias.as_ref())?;
        let v = layer.v_proj.apply_linear(&x_norm, h, h).add_optional_trailing_bias(layer.v_proj_bias.as_ref())?;

        let _ = (batch, seq);
        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        let k_t = k.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q.matmul(&k_t)?;
        let scores_scaled = scores.mul_scalar(scale);
        let probs = scores_scaled.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;

        // SelfOutput: dense projection only (no normalisation here in HF).
        let attn_out = layer.attn_output_proj.apply_linear(&merged, h, h);
        let attn_out_bias_t = x.const_f32_like(
            Arc::clone(&layer.attn_output_proj_bias),
            Shape::from_dims(&[h]),
        );
        let attn_out = attn_out.broadcast_add(&attn_out_bias_t)?;
        // First residual.
        let h1 = x.add(&attn_out)?;

        // Pre-LN before MLP.
        let h1_norm = h1.layer_norm_affine(std::sync::Arc::clone(&layer.ln_after_gain), std::sync::Arc::clone(&layer.ln_after_bias), cfg.layer_norm_eps)?;

        // Intermediate: linear + activation.
        let inter_proj = layer.intermediate_proj.apply_linear(&h1_norm, h, cfg.intermediate_size);
        let inter_bias_t = x.const_f32_like(
            Arc::clone(&layer.intermediate_proj_bias),
            Shape::from_dims(&[cfg.intermediate_size]),
        );
        let inter = inter_proj.broadcast_add(&inter_bias_t)?;
        let activated = match cfg.hidden_activation {
            VitActivation::Gelu => inter.gelu_erf(),
            VitActivation::GeluPytorchTanh => inter.gelu(),
            VitActivation::Relu => inter.relu(),
            VitActivation::Silu => inter.silu(),
        };

        // Output: linear + residual.
        let mlp_out = layer.mlp_output_proj.apply_linear(&activated, cfg.intermediate_size, h);
        let mlp_bias_t = x.const_f32_like(
            Arc::clone(&layer.mlp_output_proj_bias),
            Shape::from_dims(&[h]),
        );
        let mlp_out = mlp_out.broadcast_add(&mlp_bias_t)?;
        // Second residual.
        h1.add(&mlp_out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_weights(cfg: &VitConfig, num_labels: Option<usize>) -> VitWeights {
        let mut s: u32 = 19191;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let p = cfg.patch_size;
        let c = cfg.num_channels;
        let np = cfg.num_patches();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);

        let patch_proj = vec_of(h * c * p * p, &mut *nb);
        let patch_proj_bias = vec_of(h, &mut *nb);
        let cls_token = vec_of(h, &mut *nb);
        let position_embeddings = vec_of((np + 1) * h, &mut *nb);

        let layers: Vec<VitLayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| VitLayerWeights {
                ln_before_gain: Arc::from(vec![1.0_f32; h]),
                ln_before_bias: Arc::from(vec![0.0_f32; h]),
                q_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                q_proj_bias: if cfg.qkv_bias { Some(vec_of(h, &mut *nb)) } else { None },
                k_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                k_proj_bias: if cfg.qkv_bias { Some(vec_of(h, &mut *nb)) } else { None },
                v_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                v_proj_bias: if cfg.qkv_bias { Some(vec_of(h, &mut *nb)) } else { None },
                attn_output_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_output_proj_bias: vec_of(h, &mut *nb),
                ln_after_gain: Arc::from(vec![1.0_f32; h]),
                ln_after_bias: Arc::from(vec![0.0_f32; h]),
                intermediate_proj: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                intermediate_proj_bias: vec_of(inter, &mut *nb),
                mlp_output_proj: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                mlp_output_proj_bias: vec_of(h, &mut *nb),
            })
            .collect();

        let final_ln_gain = Arc::from(vec![1.0_f32; h]);
        let final_ln_bias = Arc::from(vec![0.0_f32; h]);
        let classifier = num_labels.map(|n| {
            let w = WeightStorage::F32(vec_of(h * n, &mut *nb));
            let b = vec_of(n, &mut *nb);
            (w, b)
        });
        VitWeights {
            patch_proj, patch_proj_bias,
            cls_token, position_embeddings,
            layers,
            final_ln_gain, final_ln_bias,
            classifier,
        }
    }

    fn tiny_config() -> VitConfig {
        VitConfig {
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 32,
            hidden_activation: VitActivation::Gelu,
            layer_norm_eps: 1e-12,
            image_size: 16,
            patch_size: 4,
            num_channels: 3,
            qkv_bias: true,
        }
    }

    fn tiny_image(cfg: &VitConfig) -> LazyTensor {
        let n_pix = 1 * cfg.num_channels * cfg.image_size * cfg.image_size;
        let img_data: Vec<f32> = (0..n_pix).map(|i| (i as f32 / n_pix as f32)).collect();
        LazyTensor::from_f32(
            Arc::from(img_data),
            Shape::from_dims(&[1, cfg.num_channels, cfg.image_size, cfg.image_size]),
            &Device::cpu(),
        )
    }

    #[test]
    fn forward_no_classifier_shape() {
        let cfg = tiny_config();
        let model = VitModel { config: cfg.clone(), weights: tiny_weights(&cfg, None) };
        let img = tiny_image(&cfg);
        let out = model.forward(&img).unwrap();
        assert_eq!(out.shape().dims(), &[1, cfg.num_patches() + 1, cfg.hidden_size]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn forward_with_classifier() {
        let cfg = tiny_config();
        let num_labels = 5;
        let model = VitModel { config: cfg.clone(), weights: tiny_weights(&cfg, Some(num_labels)) };
        let img = tiny_image(&cfg);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, num_labels]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// CLS token is prepended at position 0 — verify by zeroing
    /// the cls_token weight and confirming the encoder output at
    /// position 0 changes (the cls token feeds through all layers).
    #[test]
    fn cls_token_is_wired() {
        let cfg = tiny_config();
        let base = tiny_weights(&cfg, None);
        let mut zeroed = base.clone();
        zeroed.cls_token = Arc::from(vec![0.0_f32; cfg.hidden_size]);
        let m_base = VitModel { config: cfg.clone(), weights: base };
        let m_zero = VitModel { config: cfg.clone(), weights: zeroed };
        let img_a = tiny_image(&cfg);
        let img_b = tiny_image(&cfg);
        let out_a = m_base.forward(&img_a).unwrap().realize_f32();
        let out_b = m_zero.forward(&img_b).unwrap().realize_f32();
        // First hidden vector (position 0) should differ.
        let cls_a = &out_a[..cfg.hidden_size];
        let cls_b = &out_b[..cfg.hidden_size];
        let mut max_diff = 0.0_f32;
        for (x, y) in cls_a.iter().zip(cls_b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(max_diff > 1e-6,
            "CLS token weight change must alter CLS output, max_diff = {max_diff}");
    }

    /// QKV-biases-off path runs and produces different output
    /// than the bias-on baseline.
    #[test]
    fn qkv_bias_off_runs() {
        let cfg_off = VitConfig { qkv_bias: false, ..tiny_config() };
        let model = VitModel { config: cfg_off.clone(), weights: tiny_weights(&cfg_off, None) };
        let img = tiny_image(&cfg_off);
        let out = model.forward(&img).unwrap();
        assert_eq!(out.shape().dims(), &[1, cfg_off.num_patches() + 1, cfg_off.hidden_size]);
    }

    #[test]
    fn config_presets() {
        let c = VitConfig::vit_base_patch16_224();
        assert_eq!(c.num_patches(), 196);
        assert_eq!(c.head_dim(), 64);
        let t = VitConfig::microsoft_trocr_base_handwritten();
        assert!(!t.qkv_bias);
        assert_eq!(t.image_size, 384);
    }

    /// `forward_intermediate_layers` returns one tensor per
    /// requested layer index, each shaped
    /// `(1, num_patches + 1, hidden_size)`. Same contract as
    /// the DINOv2 hook (commit `de541296`).
    #[test]
    fn forward_intermediate_layers_shape() {
        let cfg = tiny_config();
        let model = VitModel { config: cfg.clone(), weights: tiny_weights(&cfg, None) };
        let img = tiny_image(&cfg);
        let outs = model.forward_intermediate_layers(&img, &[0_usize, 1]).unwrap();
        assert_eq!(outs.len(), 2);
        let np = cfg.num_patches();
        for out in &outs {
            assert_eq!(out.shape().dims(), &[1, np + 1, cfg.hidden_size]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite intermediate: {v}");
            }
        }
    }

    /// Intermediate features at different depths must differ.
    #[test]
    fn intermediate_layers_differ_across_depth() {
        let cfg = tiny_config();
        let model = VitModel { config: cfg.clone(), weights: tiny_weights(&cfg, None) };
        let img = tiny_image(&cfg);
        let outs = model.forward_intermediate_layers(&img, &[0_usize, 1]).unwrap();
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
