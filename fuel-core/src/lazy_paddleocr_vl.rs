//! PaddleOCR-VL top-level composition: text + vision + tile glue.
//!
//! Sub-port 3 of three for PaddleOCR-VL. Bundles the already-shipped
//! [`PaddleOcrVlTextModel`] (ERNIE-style decoder with M-RoPE) and
//! [`PaddleOcrVlVisionModel`] (fixed-tile-grid ViT + projector) into a
//! single multimodal forward path.
//!
//! # Forward pipeline
//!
//! 1. If `image_pixels` is `None`, embed `text_tokens` through the
//!    text model directly — equivalent to a plain ERNIE forward.
//! 2. Otherwise inspect the image's `(C, H, W)` host shape and pick a
//!    `(rows, cols)` tile grid via [`aspect_ratio_chooser`]
//!    (re-exported from the vision sub-port).
//! 3. Host-side partition the image into `rows * cols` tiles via
//!    [`partition_image`]; nearest-neighbor resize each tile to the
//!    vision encoder's `(image_size, image_size)` working resolution
//!    so a single fixed-size ViT can process them.
//! 4. Run [`PaddleOcrVlVisionModel::forward`] on the stacked tiles to
//!    get vision features of shape
//!    `(num_tiles * patches_per_tile_merged, text_hidden_size)`.
//! 5. Scatter the vision features into the embedded text stream at
//!    positions where `text_tokens[i] == image_token_id`. Implemented
//!    as host-side run-segmentation + `slice` + `concat` so the graph
//!    stays purely functional (no in-place writes).
//! 6. Feed the assembled `(1, total_seq, hidden)` embedding through
//!    [`PaddleOcrVlTextModel::forward_embeds`] to produce logits.
//!
//! # Scope (v1)
//!
//! - Forward-only, single batch, F32, single contiguous image stream.
//! - Image partitioning uses nearest-neighbor resize to keep the
//!   host helper deterministic and dependency-free. The PaddleOCR
//!   eager preprocessor uses bilinear; matching that goes alongside
//!   the eager image loader, deferred to its own follow-up.
//! - Text uses 1D positions (M-RoPE collapses to 1D for text-only
//!   positions — see the text sub-port's deviation test). True 3D
//!   M-RoPE position assignment for vision tokens is left to the
//!   eager preprocessor pipeline; the lazy path keeps the in-graph
//!   shape identical regardless.

use crate::lazy::LazyTensor;
use crate::lazy_paddleocr_vl_text::{
    load_paddleocr_vl_text_weights_with_prefix,
    PaddleOcrVlTextConfig, PaddleOcrVlTextModel, PaddleOcrVlTextWeights,
};
use crate::lazy_paddleocr_vl_vision::{
    PaddleOcrVlVisionConfig, PaddleOcrVlVisionModel, PaddleOcrVlVisionWeights,
};
pub use crate::lazy_paddleocr_vl_vision::{aspect_ratio_chooser, partition_image};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

/// Bundled configuration for the full PaddleOCR-VL stack.
#[derive(Debug, Clone)]
pub struct PaddleOcrVlConfig {
    pub text: PaddleOcrVlTextConfig,
    pub vision: PaddleOcrVlVisionConfig,
    /// Cap on the per-side tile count chosen by
    /// [`aspect_ratio_chooser`]. Bounds vision encoder cost for very
    /// elongated documents.
    pub max_tiles_per_side: usize,
}

impl PaddleOcrVlConfig {
    /// Default preset matching the published PaddleOCR-VL release
    /// (text ERNIE-4.5-0.3B + 384px tile-grid ViT). `max_tiles_per_side`
    /// defaults to 4 (16 tiles max).
    pub fn paddleocr_vl_default() -> Self {
        Self {
            text: PaddleOcrVlTextConfig::paddleocr_vl_default(),
            vision: PaddleOcrVlVisionConfig::paddleocr_vl(),
            max_tiles_per_side: 4,
        }
    }
}

/// Bundled weight storage for the full PaddleOCR-VL stack.
#[derive(Debug, Clone)]
pub struct PaddleOcrVlWeights {
    pub text: PaddleOcrVlTextWeights,
    pub vision: PaddleOcrVlVisionWeights,
}

/// Top-level PaddleOCR-VL model. Composes [`PaddleOcrVlTextModel`]
/// and [`PaddleOcrVlVisionModel`].
#[derive(Debug, Clone)]
pub struct PaddleOcrVlModel {
    pub config: PaddleOcrVlConfig,
    pub weights: PaddleOcrVlWeights,
}

impl PaddleOcrVlModel {
    /// Run the full multimodal forward.
    ///
    /// `image_pixels` is an optional `(C, H, W)` host pixel tensor.
    /// `text_tokens` is the token sequence; any token equal to
    /// `image_token_id` is treated as a placeholder slot that will be
    /// filled by a vision feature. When `image_pixels` is `Some`, the
    /// number of placeholder slots must equal the total number of
    /// vision feature tokens produced by the chosen tile grid; when
    /// it is `None`, `text_tokens` must not contain any image tokens.
    ///
    /// Returns logits of shape `(1, text_tokens.len(), vocab_size)`.
    pub fn forward(
        &self,
        image_pixels: Option<&LazyTensor>,
        text_tokens: &[u32],
        image_token_id: u32,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        if text_tokens.is_empty() {
            return Err(crate::Error::Msg(
                "PaddleOcrVlModel: text_tokens must be non-empty".into(),
            ).bt());
        }

        match image_pixels {
            None => {
                if text_tokens.iter().any(|&t| t == image_token_id) {
                    return Err(crate::Error::Msg(
                        "PaddleOcrVlModel: text_tokens contain image_token_id but image_pixels is None".into(),
                    ).bt());
                }
                self.text_model().forward(text_tokens, start_pos)
            }
            Some(pixels) => self.forward_with_image(pixels, text_tokens, image_token_id, start_pos),
        }
    }

    fn forward_with_image(
        &self,
        image_pixels: &LazyTensor,
        text_tokens: &[u32],
        image_token_id: u32,
        start_pos: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let v_cfg = &cfg.vision;
        let t_cfg = &cfg.text;

        let dims = image_pixels.shape();
        let dims = dims.dims().to_vec();
        if dims.len() != 3 || dims[0] != v_cfg.num_channels {
            return Err(crate::Error::Msg(format!(
                "PaddleOcrVlModel: image_pixels must be (C={}, H, W), got {:?}",
                v_cfg.num_channels, dims,
            )).bt());
        }
        let channels = dims[0];
        let height = dims[1];
        let width = dims[2];
        if height == 0 || width == 0 {
            return Err(crate::Error::Msg(
                "PaddleOcrVlModel: image height/width must be > 0".into(),
            ).bt());
        }

        let (rows, cols) = aspect_ratio_chooser(height, width, cfg.max_tiles_per_side);
        let num_tiles = rows * cols;
        let tile_h = height / rows;
        let tile_w = width / cols;
        if tile_h == 0 || tile_w == 0 {
            return Err(crate::Error::Msg(format!(
                "PaddleOcrVlModel: tile dim collapsed (H={height}, W={width}, rows={rows}, cols={cols})",
            )).bt());
        }

        // Host-side: read image pixels, partition into tiles, resize each
        // tile to (image_size, image_size) via nearest neighbor.
        let pixel_data = image_pixels.realize_f32();
        if pixel_data.len() != channels * height * width {
            return Err(crate::Error::Msg(format!(
                "PaddleOcrVlModel: image_pixels realize length {} mismatch C*H*W = {}",
                pixel_data.len(),
                channels * height * width,
            )).bt());
        }
        let tiles = partition_image(&pixel_data, channels, height, width, rows, cols);

        let target = v_cfg.image_size;
        let mut tile_pixels: Vec<f32> = Vec::with_capacity(num_tiles * channels * target * target);
        for tile in &tiles {
            resize_nearest_chw(tile, channels, tile_h, tile_w, target, target, &mut tile_pixels);
        }

        let stacked = LazyTensor::from_f32(
            Arc::from(tile_pixels),
            Shape::from_dims(&[num_tiles, channels, target, target]),
            &Device::cpu(),
        );

        // Run the tile-grid vision encoder -> (N_vision, text_hidden).
        let vision_model = PaddleOcrVlVisionModel {
            config: v_cfg.clone(),
            text_hidden_size: t_cfg.hidden_size,
            weights: self.weights.vision.clone(),
        };
        let vision_features = vision_model.forward(&stacked, (rows, cols))?;

        let v_dims = vision_features.shape();
        let v_dims = v_dims.dims().to_vec();
        if v_dims.len() != 2 || v_dims[1] != t_cfg.hidden_size {
            return Err(crate::Error::Msg(format!(
                "PaddleOcrVlModel: vision features shape {:?} must be (N, text_hidden={})",
                v_dims, t_cfg.hidden_size,
            )).bt());
        }
        let num_vision_tokens = v_dims[0];

        // Count image placeholders in text_tokens; must equal vision token count.
        let num_placeholders = text_tokens.iter().filter(|&&t| t == image_token_id).count();
        if num_placeholders != num_vision_tokens {
            return Err(crate::Error::Msg(format!(
                "PaddleOcrVlModel: text_tokens have {num_placeholders} image placeholders but vision produced {num_vision_tokens} feature tokens",
            )).bt());
        }

        // Promote vision features to (1, N_vision, hidden) so concat
        // aligns with the text embedding axis layout.
        let vision_embeds = vision_features
            .reshape(Shape::from_dims(&[1, num_vision_tokens, t_cfg.hidden_size]))?;

        // Embed all text tokens, then splice in vision features by
        // walking host-side runs of (text-tokens vs image-tokens) and
        // selecting slices from each tensor in order. Anchor the
        // embedding-table constants on the vision graph so the two
        // streams share a graph and `concat` succeeds.
        let text_embeds = vision_embeds.embed_tokens_anchored(
            self.weights.text.token_embedding.clone(),
            t_cfg.vocab_size,
            t_cfg.hidden_size,
            text_tokens,
        )?;

        let combined = splice_image_slots(
            &text_embeds,
            &vision_embeds,
            text_tokens,
            image_token_id,
        )?;

        self.text_model().forward_embeds(&combined, start_pos)
    }

    fn text_model(&self) -> PaddleOcrVlTextModel {
        PaddleOcrVlTextModel {
            config: self.config.text.clone(),
            weights: self.weights.text.clone(),
        }
    }
}

/// Build the combined `(1, seq, hidden)` embedding by walking
/// `text_tokens` and emitting slices alternately from `text_embeds`
/// (for runs of non-image tokens) and `vision_embeds` (for runs of
/// image-placeholder tokens). Both inputs are expected as
/// `(1, *, hidden)` so concat along dim 1 reassembles them in order.
fn splice_image_slots(
    text_embeds: &LazyTensor,
    vision_embeds: &LazyTensor,
    text_tokens: &[u32],
    image_token_id: u32,
) -> Result<LazyTensor> {
    let seq = text_tokens.len();
    let mut segments: Vec<LazyTensor> = Vec::new();
    let mut vision_offset = 0_usize;

    let mut i = 0;
    while i < seq {
        let is_image = text_tokens[i] == image_token_id;
        let start = i;
        while i < seq && (text_tokens[i] == image_token_id) == is_image {
            i += 1;
        }
        let len = i - start;
        let segment = if is_image {
            let s = vision_embeds.slice(1_usize, vision_offset, len)?;
            vision_offset += len;
            s
        } else {
            text_embeds.slice(1_usize, start, len)?
        };
        segments.push(segment);
    }

    let mut acc = segments.remove(0);
    for next in segments.into_iter() {
        acc = acc.concat(&next, 1_usize)?;
    }
    Ok(acc)
}

/// Pure nearest-neighbor resize of a single CHW tile. Appends the
/// resized pixels (channel-major, row-major) to `out`. Pure host
/// arithmetic — no graph ops. Used by [`PaddleOcrVlModel::forward`]
/// to bring an arbitrary tile to the vision encoder's fixed working
/// resolution.
fn resize_nearest_chw(
    tile: &[f32],
    channels: usize,
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
    out: &mut Vec<f32>,
) {
    debug_assert_eq!(tile.len(), channels * src_h * src_w);
    for ch in 0..channels {
        let plane = ch * src_h * src_w;
        for y in 0..dst_h {
            let sy = (y * src_h) / dst_h;
            for x in 0..dst_w {
                let sx = (x * src_w) / dst_w;
                out.push(tile[plane + sy * src_w + sx]);
            }
        }
    }
}

// ---- Safetensors loader ----------------------------------------------------

impl PaddleOcrVlModel {
    /// Load a complete PaddleOCR-VL model (vision + text) from a
    /// HuggingFace safetensors file. HF naming for the published
    /// checkpoint:
    ///   - Vision: `visual.vision_model.*` (NaViT encoder +
    ///     `post_layernorm`).
    ///   - Projector: `mlp_AR.*` (pre-norm + 2-layer MLP into
    ///     `text.hidden_size`).
    ///   - Text: top-level `model.*` + `lm_head.weight` (ERNIE-style
    ///     decoder with LLaMA-shape weight layout).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &PaddleOcrVlConfig,
    ) -> Result<Self> {
        let vision = PaddleOcrVlVisionWeights::load_from_mmapped(
            st, &cfg.vision, cfg.text.hidden_size,
        )?;
        let text = load_paddleocr_vl_text_weights_with_prefix(st, &cfg.text, "")?;
        Ok(PaddleOcrVlModel {
            config: cfg.clone(),
            weights: PaddleOcrVlWeights { vision, text },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lazy::{LayerWeights, WeightStorage};
    use crate::lazy_paddleocr_vl_text::PaddleOcrVlTextConfig;
    use crate::lazy_paddleocr_vl_vision::{
        PaddleOcrVlVisionActivation, PaddleOcrVlVisionBlockWeights,
        PaddleOcrVlVisionProjectorWeights, PaddleOcrVlVisionWeights,
    };

    fn tiny_text_cfg() -> PaddleOcrVlTextConfig {
        PaddleOcrVlTextConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            use_bias: false,
            tie_word_embeddings: false,
            mrope_section: vec![1, 1],
        }
    }

    fn tiny_vision_cfg() -> PaddleOcrVlVisionConfig {
        PaddleOcrVlVisionConfig {
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_channels: 3,
            image_size: 8,
            patch_size: 4,
            hidden_activation: PaddleOcrVlVisionActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
            spatial_merge_size: 2,
            rope_theta: 10_000.0,
        }
    }

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_text_weights(cfg: &PaddleOcrVlTextConfig) -> PaddleOcrVlTextWeights {
        let mut s: u32 = 13579;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let kv = cfg.num_key_value_heads * cfg.head_dim;
        let token_embedding = vec_of(cfg.vocab_size * h, &mut *nb);
        let layers: Vec<LayerWeights> = (0..cfg.num_hidden_layers)
            .map(|_| LayerWeights {
                attn_q: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                attn_q_bias: None,
                attn_k: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_k_bias: None,
                attn_v: WeightStorage::F32(vec_of(h * kv, &mut *nb)),
                attn_v_bias: None,
                attn_o: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                ffn_gate: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_up: WeightStorage::F32(vec_of(h * i, &mut *nb)),
                ffn_down: WeightStorage::F32(vec_of(i * h, &mut *nb)),
                attn_norm_gain: Arc::from(vec![1.0_f32; h]),
                ffn_norm_gain: Arc::from(vec![1.0_f32; h]),
            })
            .collect();
        let final_norm_gain = Arc::from(vec![1.0_f32; h]);
        let output = WeightStorage::F32(vec_of(h * cfg.vocab_size, &mut *nb));
        PaddleOcrVlTextWeights {
            token_embedding,
            layers,
            final_norm_gain,
            output: Some(output),
        }
    }

    fn tiny_vision_weights(cfg: &PaddleOcrVlVisionConfig, text_hidden: usize) -> PaddleOcrVlVisionWeights {
        let mut s: u32 = 414141;
        let mut next = move || -> f32 {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        };
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let p = cfg.patch_size;
        let c = cfg.num_channels;
        let np = cfg.num_patches_per_tile();
        let m = cfg.spatial_merge_size;
        let merged_hidden = h * m * m;

        let patch_proj = vec_of(h * c * p * p, &mut *nb);
        let patch_proj_bias = vec_of(h, &mut *nb);
        let position_embedding = vec_of(np * h, &mut *nb);

        let blocks: Vec<PaddleOcrVlVisionBlockWeights> = (0..cfg.num_hidden_layers)
            .map(|_| PaddleOcrVlVisionBlockWeights {
                ln1_gain: Arc::from(vec![1.0_f32; h]),
                ln1_bias: Arc::from(vec![0.0_f32; h]),
                q_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                q_proj_bias: vec_of(h, &mut *nb),
                k_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                k_proj_bias: vec_of(h, &mut *nb),
                v_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                v_proj_bias: vec_of(h, &mut *nb),
                out_proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                out_proj_bias: vec_of(h, &mut *nb),
                ln2_gain: Arc::from(vec![1.0_f32; h]),
                ln2_bias: Arc::from(vec![0.0_f32; h]),
                fc1: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                fc1_bias: vec_of(inter, &mut *nb),
                fc2: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                fc2_bias: vec_of(h, &mut *nb),
            })
            .collect();

        let post_ln_gain = Arc::from(vec![1.0_f32; h]);
        let post_ln_bias = Arc::from(vec![0.0_f32; h]);

        let projector = PaddleOcrVlVisionProjectorWeights {
            pre_norm_gain: Arc::from(vec![1.0_f32; h]),
            pre_norm_bias: Arc::from(vec![0.0_f32; h]),
            linear_1: WeightStorage::F32(vec_of(merged_hidden * merged_hidden, &mut *nb)),
            linear_1_bias: vec_of(merged_hidden, &mut *nb),
            linear_2: WeightStorage::F32(vec_of(merged_hidden * text_hidden, &mut *nb)),
            linear_2_bias: vec_of(text_hidden, &mut *nb),
        };

        PaddleOcrVlVisionWeights {
            patch_proj,
            patch_proj_bias,
            position_embedding,
            blocks,
            post_ln_gain,
            post_ln_bias,
            projector,
        }
    }

    fn tiny_model() -> PaddleOcrVlModel {
        let text_cfg = tiny_text_cfg();
        let vision_cfg = tiny_vision_cfg();
        let weights = PaddleOcrVlWeights {
            text: tiny_text_weights(&text_cfg),
            vision: tiny_vision_weights(&vision_cfg, text_cfg.hidden_size),
        };
        PaddleOcrVlModel {
            config: PaddleOcrVlConfig {
                text: text_cfg,
                vision: vision_cfg,
                max_tiles_per_side: 2,
            },
            weights,
        }
    }

    fn tiny_image(model: &PaddleOcrVlModel, height: usize, width: usize) -> LazyTensor {
        let cfg = &model.config.vision;
        let n_pix = cfg.num_channels * height * width;
        let data: Vec<f32> = (0..n_pix).map(|i| (i as f32 / n_pix as f32)).collect();
        LazyTensor::from_f32(
            Arc::from(data),
            Shape::from_dims(&[cfg.num_channels, height, width]),
            &Device::cpu(),
        )
    }

    /// Text-only forward (`image_pixels = None`) must produce the same
    /// logits as calling the text sub-port's `forward` directly. This
    /// catches a composition layer that accidentally adds extra state
    /// (e.g. a residual from a "no image" code path).
    #[test]
    fn forward_text_only_matches_paddleocr_vl_text_forward() {
        let model = tiny_model();
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5];
        let image_token_id: u32 = 31;

        let composed = model.forward(None, &tokens, image_token_id, 0).unwrap().realize_f32();

        let text_model = PaddleOcrVlTextModel {
            config: model.config.text.clone(),
            weights: model.weights.text.clone(),
        };
        let direct = text_model.forward(&tokens, 0).unwrap().realize_f32();

        assert_eq!(composed.len(), direct.len(), "logits length mismatch");
        for (i, (a, b)) in composed.iter().zip(direct.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "logits[{i}]: composed={a} text_only={b}",
            );
        }
    }

    /// Changing the image content must change the produced logits when
    /// the text contains image-placeholder tokens. Demonstrates that
    /// the scatter actually wires vision features into the language
    /// model rather than discarding them.
    #[test]
    fn forward_with_image_changes_output() {
        let model = tiny_model();
        let image_token_id: u32 = 31;
        let v_cfg = &model.config.vision;
        let merge = v_cfg.spatial_merge_size;
        let per_tile_merged = v_cfg.num_patches_per_tile() / (merge * merge);
        // Square image -> 1x1 tile grid.
        let num_image_tokens = per_tile_merged;
        // Surround placeholders with two text tokens on each side.
        let mut tokens: Vec<u32> = vec![3, 5];
        tokens.extend(std::iter::repeat(image_token_id).take(num_image_tokens));
        tokens.extend_from_slice(&[7, 11]);

        let img_a = tiny_image(&model, v_cfg.image_size, v_cfg.image_size);
        // Reversed pixel values -> distinctly different image content.
        let cfg = v_cfg;
        let n_pix = cfg.num_channels * cfg.image_size * cfg.image_size;
        let data_b: Vec<f32> = (0..n_pix).rev().map(|i| (i as f32 / n_pix as f32)).collect();
        let img_b = LazyTensor::from_f32(
            Arc::from(data_b),
            Shape::from_dims(&[cfg.num_channels, cfg.image_size, cfg.image_size]),
            &Device::cpu(),
        );

        let out_a = model.forward(Some(&img_a), &tokens, image_token_id, 0).unwrap().realize_f32();
        let out_b = model.forward(Some(&img_b), &tokens, image_token_id, 0).unwrap().realize_f32();
        assert_eq!(out_a.len(), out_b.len());
        let any_diff = out_a.iter().zip(out_b.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(any_diff, "swapping image content must change logits");
        for &v in &out_a {
            assert!(v.is_finite(), "out_a non-finite: {v}");
        }
        for &v in &out_b {
            assert!(v.is_finite(), "out_b non-finite: {v}");
        }
    }

    /// Verify the host-side aspect-ratio chooser drives the tile count
    /// the way the composition expects: small landscape -> (1, cols),
    /// small portrait -> (rows, 1). Square stays single-tile so the
    /// `forward_with_image_changes_output` test above remains valid.
    #[test]
    fn aspect_ratio_chooser_drives_tile_count() {
        // max_tiles_per_side caps the chosen rows/cols.
        let max = 3;
        // 1:1 -> 1x1.
        assert_eq!(aspect_ratio_chooser(64, 64, max), (1, 1));
        // ~3:1 landscape -> 1x3 (capped at max).
        assert_eq!(aspect_ratio_chooser(64, 200, max), (1, 3));
        // ~1:3 portrait -> 3x1.
        assert_eq!(aspect_ratio_chooser(200, 64, max), (3, 1));
        // 5:1 ratio capped at max.
        assert_eq!(aspect_ratio_chooser(50, 250, max), (1, 3));
    }

    /// Tile partition produces row-major `rows * cols` tiles, each of
    /// shape `(channels, height/rows, width/cols)`. The composition
    /// layer relies on this exact layout to feed the vision encoder.
    #[test]
    fn tile_partition_round_trip_shape() {
        let channels = 3;
        let height = 8;
        let width = 12;
        let rows = 2;
        let cols = 3;
        let total = channels * height * width;
        let img: Vec<f32> = (0..total).map(|i| i as f32).collect();
        let tiles = partition_image(&img, channels, height, width, rows, cols);
        assert_eq!(tiles.len(), rows * cols);
        let tile_h = height / rows;
        let tile_w = width / cols;
        for (idx, tile) in tiles.iter().enumerate() {
            assert_eq!(
                tile.len(),
                channels * tile_h * tile_w,
                "tile {idx} has wrong size",
            );
        }
        // Spot-check the first tile's first channel matches the
        // top-left submatrix of the input image's first channel.
        let plane = 0;
        let plane_off = plane * height * width;
        for yy in 0..tile_h {
            for xx in 0..tile_w {
                let img_val = img[plane_off + yy * width + xx];
                let tile_val = tiles[0][plane * tile_h * tile_w + yy * tile_w + xx];
                assert_eq!(img_val, tile_val);
            }
        }
    }

    /// End-to-end forward with a non-trivial multi-tile image (matching
    /// a 1x2 landscape grid) plus a mixed text + image token stream.
    /// Asserts shape, finiteness, and that the scatter consumed all
    /// vision tokens.
    #[test]
    fn end_to_end_tiny_image_plus_text() {
        let model = tiny_model();
        let image_token_id: u32 = 31;
        let v_cfg = &model.config.vision;
        let merge = v_cfg.spatial_merge_size;
        let per_tile_merged = v_cfg.num_patches_per_tile() / (merge * merge);

        // 1x2 landscape image: H = image_size, W = 2 * image_size will
        // trigger a (1, 2) tile grid -> 2 tiles, so we need
        // 2 * per_tile_merged image placeholder tokens.
        let h = v_cfg.image_size;
        let w = 2 * v_cfg.image_size;
        let img = tiny_image(&model, h, w);
        let grid = aspect_ratio_chooser(h, w, model.config.max_tiles_per_side);
        assert_eq!(grid, (1, 2), "test relies on 1x2 grid choice");
        let num_image_tokens = 2 * per_tile_merged;

        let mut tokens: Vec<u32> = Vec::new();
        tokens.push(2);
        tokens.extend(std::iter::repeat(image_token_id).take(num_image_tokens));
        tokens.push(4);
        tokens.push(6);

        let logits = model
            .forward(Some(&img), &tokens, image_token_id, 0)
            .unwrap();
        assert_eq!(
            logits.shape().dims(),
            &[1, tokens.len(), model.config.text.vocab_size],
            "end-to-end logits shape",
        );
        let out = logits.realize_f32();
        assert_eq!(out.len(), tokens.len() * model.config.text.vocab_size);
        for (i, &v) in out.iter().enumerate() {
            assert!(v.is_finite(), "logits[{i}] = {v} not finite");
        }

        // Mismatched placeholder count surfaces a typed build-time error.
        let bad_tokens: Vec<u32> = vec![2, image_token_id, 3];
        let err = model.forward(Some(&img), &bad_tokens, image_token_id, 0);
        assert!(err.is_err(), "mismatched placeholder count must error");
    }

    mod load {
        use super::*;
        use safetensors::tensor::TensorView;
        use safetensors::Dtype;
        use std::collections::HashMap;

        fn put(
            map: &mut HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
            name: &str,
            shape: &[usize],
            data: &[f32],
        ) {
            let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
            map.insert(name.to_string(), (Dtype::F32, shape.to_vec(), bytes));
        }

        fn serialize_to_tempfile(
            map: &HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
        ) -> std::path::PathBuf {
            let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
            for (k, (dt, shape, data)) in map {
                let v = TensorView::new(*dt, shape.clone(), data).expect("TensorView");
                views.insert(k.clone(), v);
            }
            let bytes = safetensors::serialize(&views, None).expect("serialize");
            let path = std::env::temp_dir().join(format!(
                "lazy_paddleocr_vl_load_{}_{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            ));
            std::fs::write(&path, bytes).expect("write tempfile");
            path
        }

        fn build_tiny_safetensors(cfg: &PaddleOcrVlConfig) -> std::path::PathBuf {
            let mut map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
            let mut s: u32 = 6464;
            let mut nxt = || -> f32 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.01
            };
            let mut vec_n = |n: usize| -> Vec<f32> { (0..n).map(|_| nxt()).collect() };

            // Vision tower under `visual.vision_model.*` + projector under `mlp_AR.*`.
            let v_cfg = &cfg.vision;
            let text_hidden = cfg.text.hidden_size;
            let h = v_cfg.hidden_size;
            let inter = v_cfg.intermediate_size;
            let np = v_cfg.num_patches_per_tile();
            let m = v_cfg.spatial_merge_size;
            let merged_hidden = h * m * m;
            let vp = "visual.vision_model.";
            put(&mut map, &format!("{vp}embeddings.patch_embedding.weight"),
                &[h, v_cfg.num_channels, v_cfg.patch_size, v_cfg.patch_size],
                &vec_n(h * v_cfg.num_channels * v_cfg.patch_size * v_cfg.patch_size));
            put(&mut map, &format!("{vp}embeddings.patch_embedding.bias"),
                &[h], &vec_n(h));
            put(&mut map, &format!("{vp}embeddings.position_embedding.weight"),
                &[np, h], &vec_n(np * h));
            put(&mut map, &format!("{vp}post_layernorm.weight"), &[h], &vec_n(h));
            put(&mut map, &format!("{vp}post_layernorm.bias"), &[h], &vec_n(h));
            for i in 0..v_cfg.num_hidden_layers {
                let p = format!("{vp}encoder.layers.{i}");
                put(&mut map, &format!("{p}.layer_norm1.weight"), &[h], &vec_n(h));
                put(&mut map, &format!("{p}.layer_norm1.bias"), &[h], &vec_n(h));
                put(&mut map, &format!("{p}.layer_norm2.weight"), &[h], &vec_n(h));
                put(&mut map, &format!("{p}.layer_norm2.bias"), &[h], &vec_n(h));
                for proj in &["q_proj", "k_proj", "v_proj", "out_proj"] {
                    put(&mut map, &format!("{p}.self_attn.{proj}.weight"),
                        &[h, h], &vec_n(h * h));
                    put(&mut map, &format!("{p}.self_attn.{proj}.bias"),
                        &[h], &vec_n(h));
                }
                put(&mut map, &format!("{p}.mlp.fc1.weight"),
                    &[inter, h], &vec_n(inter * h));
                put(&mut map, &format!("{p}.mlp.fc1.bias"), &[inter], &vec_n(inter));
                put(&mut map, &format!("{p}.mlp.fc2.weight"),
                    &[h, inter], &vec_n(h * inter));
                put(&mut map, &format!("{p}.mlp.fc2.bias"), &[h], &vec_n(h));
            }
            let pp = "mlp_AR.";
            put(&mut map, &format!("{pp}pre_norm.weight"), &[h], &vec_n(h));
            put(&mut map, &format!("{pp}pre_norm.bias"), &[h], &vec_n(h));
            put(&mut map, &format!("{pp}linear_1.weight"),
                &[merged_hidden, merged_hidden],
                &vec_n(merged_hidden * merged_hidden));
            put(&mut map, &format!("{pp}linear_1.bias"),
                &[merged_hidden], &vec_n(merged_hidden));
            put(&mut map, &format!("{pp}linear_2.weight"),
                &[text_hidden, merged_hidden],
                &vec_n(text_hidden * merged_hidden));
            put(&mut map, &format!("{pp}linear_2.bias"),
                &[text_hidden], &vec_n(text_hidden));

            // Text model at top level (`model.*` + `lm_head.weight`).
            let t_cfg = &cfg.text;
            let d = t_cfg.hidden_size;
            let kv = t_cfg.num_key_value_heads * t_cfg.head_dim;
            put(&mut map, "model.embed_tokens.weight",
                &[t_cfg.vocab_size, d], &vec_n(t_cfg.vocab_size * d));
            for i in 0..t_cfg.num_hidden_layers {
                let p = format!("model.layers.{i}");
                put(&mut map, &format!("{p}.self_attn.q_proj.weight"),
                    &[d, d], &vec_n(d * d));
                put(&mut map, &format!("{p}.self_attn.k_proj.weight"),
                    &[kv, d], &vec_n(kv * d));
                put(&mut map, &format!("{p}.self_attn.v_proj.weight"),
                    &[kv, d], &vec_n(kv * d));
                put(&mut map, &format!("{p}.self_attn.o_proj.weight"),
                    &[d, d], &vec_n(d * d));
                put(&mut map, &format!("{p}.mlp.gate_proj.weight"),
                    &[t_cfg.intermediate_size, d], &vec_n(t_cfg.intermediate_size * d));
                put(&mut map, &format!("{p}.mlp.up_proj.weight"),
                    &[t_cfg.intermediate_size, d], &vec_n(t_cfg.intermediate_size * d));
                put(&mut map, &format!("{p}.mlp.down_proj.weight"),
                    &[d, t_cfg.intermediate_size], &vec_n(d * t_cfg.intermediate_size));
                put(&mut map, &format!("{p}.input_layernorm.weight"),
                    &[d], &vec_n(d));
                put(&mut map, &format!("{p}.post_attention_layernorm.weight"),
                    &[d], &vec_n(d));
            }
            put(&mut map, "model.norm.weight", &[d], &vec_n(d));
            if !t_cfg.tie_word_embeddings {
                put(&mut map, "lm_head.weight",
                    &[t_cfg.vocab_size, d], &vec_n(t_cfg.vocab_size * d));
            }

            serialize_to_tempfile(&map)
        }

        #[test]
        fn round_trip_synthetic_safetensors() {
            let text_cfg = tiny_text_cfg();
            let vision_cfg = tiny_vision_cfg();
            let cfg = PaddleOcrVlConfig {
                text: text_cfg.clone(),
                vision: vision_cfg.clone(),
                max_tiles_per_side: 2,
            };
            let path = build_tiny_safetensors(&cfg);
            let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path) }
                .expect("mmap safetensors");
            let model = PaddleOcrVlModel::load_from_mmapped(&st, &cfg)
                .expect("PaddleOcrVlModel::load_from_mmapped");
            assert_eq!(model.weights.text.layers.len(), text_cfg.num_hidden_layers);
            assert_eq!(model.weights.vision.blocks.len(), vision_cfg.num_hidden_layers);

            let tokens: Vec<u32> = vec![1, 2, 3, 4];
            let logits = model.forward(None, &tokens, 31, 0).unwrap().realize_f32();
            for &v in &logits {
                assert!(v.is_finite(), "non-finite logit");
            }
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        #[ignore]
        fn from_hub_smoke_paddleocr_vl() {
            // Canonical: PaddlePaddle/PaddleOCR-VL.
        }
    }
}
