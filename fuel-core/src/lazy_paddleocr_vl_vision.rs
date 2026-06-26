//! PaddleOCR-VL vision tower ported to the lazy-graph API.
//!
//! OCR-specialized vision encoder built around a fixed-tile-grid
//! ViT. The eager model is NaViT-style with per-image dynamic
//! resolution; for the lazy v1 we materialize the same encoder
//! topology against a fixed-size tile (every tile is
//! `(image_size, image_size)`) and process arbitrarily-many tiles
//! by running the same stack per tile and concatenating the tile
//! features along the sequence dim.
//!
//! Layout per block (HF Siglip/Ernie ViT convention):
//!
//!   - LayerNorm (gain + bias) pre-attention.
//!   - Q/K/V/out projections all with biases.
//!   - 2D RoPE: rows feed even inv-freq slots, cols feed odd
//!     slots; the standard split-half rotate convention applies.
//!   - LayerNorm pre-MLP, then `fc1 → activation → fc2` with
//!     both layers carrying bias.
//!
//! Post the encoder a final `post_layernorm` is applied and then
//! the Projector ("mlp_AR"): pre-norm + 2x2 spatial merge + 2-layer
//! MLP with GELU (tanh approximation) into the text hidden size.
//!
//! Two host-side helpers ship with the model — `aspect_ratio_chooser`
//! picks a (rows, cols) tile grid for a given (H, W); `partition_image`
//! crops an RGB plane into `rows * cols` tiles (and clamps an empty /
//! degenerate side to the input as a single 1x1 tile). Both are pure
//! `Vec<f32>` / `usize` arithmetic — no graph ops.
//!
//! # Scope (v1) — fixed-tile path
//!
//! Forward-only, F32. Multi-tile input is supported by laying tiles
//! along axis 0 of `pixel_values`; the tower runs each tile through
//! the same encoder and concatenates along the patch axis. The
//! per-tile position embedding is the learned 1D table at the
//! checkpoint's base resolution (`num_patches_per_tile`).
//!
//! # Scope (v2) — dynamic-resolution NaViT path
//!
//! [`PaddleOcrVlNaVitModel`] runs the same encoder topology on a
//! single image at its natural aspect ratio. The host pads the image
//! to `(H', W')` where both axes are multiples of `patch_size`; the
//! conv-embed emits `(H'/patch_size, W'/patch_size)` patches. The
//! base learned position embedding (a `base_grid_size × base_grid_size`
//! table) is bilinearly interpolated to the requested patch grid
//! (with an LFU cache keyed on `(target_h, target_w)`), and the 2D
//! RoPE tables are recomputed for the requested grid via
//! `build_2d_rope_tables_hw`. Both paths share the encoder block /
//! projector / loader machinery — the NaViT extension is additive,
//! not a rewrite.

use crate::lazy::{
    load_tensor_as_f32, load_transposed_matrix_preserve_dtype,
    LazyTensor, WeightStorage,
};
use crate::{Device, Result};
use fuel_ir::Shape;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddleOcrVlVisionActivation {
    Gelu,
    GeluPytorchTanh,
    Silu,
    Relu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PaddleOcrVlVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_activation: PaddleOcrVlVisionActivation,
    pub layer_norm_eps: f64,
    pub spatial_merge_size: usize,
    pub rope_theta: f64,
}

impl PaddleOcrVlVisionConfig {
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }
    pub fn num_patches_per_tile(&self) -> usize {
        let s = self.num_patches_per_side();
        s * s
    }
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Preset matching the published PaddleOCR-VL vision encoder
    /// (hidden=1152, layers=27, heads=16, patch=14, image=384).
    pub fn paddleocr_vl() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_channels: 3,
            image_size: 384,
            patch_size: 14,
            hidden_activation: PaddleOcrVlVisionActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
            spatial_merge_size: 2,
            rope_theta: 10_000.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PaddleOcrVlVisionBlockWeights {
    pub ln1_gain: Arc<[f32]>,
    pub ln1_bias: Arc<[f32]>,
    pub q_proj: WeightStorage,
    pub q_proj_bias: Arc<[f32]>,
    pub k_proj: WeightStorage,
    pub k_proj_bias: Arc<[f32]>,
    pub v_proj: WeightStorage,
    pub v_proj_bias: Arc<[f32]>,
    pub out_proj: WeightStorage,
    pub out_proj_bias: Arc<[f32]>,
    pub ln2_gain: Arc<[f32]>,
    pub ln2_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct PaddleOcrVlVisionProjectorWeights {
    pub pre_norm_gain: Arc<[f32]>,
    pub pre_norm_bias: Arc<[f32]>,
    pub linear_1: WeightStorage,
    pub linear_1_bias: Arc<[f32]>,
    pub linear_2: WeightStorage,
    pub linear_2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct PaddleOcrVlVisionWeights {
    /// Conv2d patch projection `[hidden, num_channels, patch, patch]`.
    pub patch_proj: Arc<[f32]>,
    pub patch_proj_bias: Arc<[f32]>,
    /// `[num_patches_per_tile, hidden]` (per-tile learned 1D table).
    pub position_embedding: Arc<[f32]>,
    pub blocks: Vec<PaddleOcrVlVisionBlockWeights>,
    pub post_ln_gain: Arc<[f32]>,
    pub post_ln_bias: Arc<[f32]>,
    pub projector: PaddleOcrVlVisionProjectorWeights,
}

#[derive(Debug, Clone)]
pub struct PaddleOcrVlVisionModel {
    pub config: PaddleOcrVlVisionConfig,
    pub text_hidden_size: usize,
    pub weights: PaddleOcrVlVisionWeights,
}

impl PaddleOcrVlVisionModel {
    /// Encode a batch of tiles.
    ///
    /// `pixels` must have shape `(num_tiles, num_channels, image_size,
    /// image_size)` where `num_tiles == tile_grid.0 * tile_grid.1`.
    /// Output shape is
    /// `((num_tiles * num_patches_per_tile) / spatial_merge_size^2,
    ///   text_hidden_size)`.
    pub fn forward(
        &self,
        pixels: &LazyTensor,
        tile_grid: (usize, usize),
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let (rows, cols) = tile_grid;
        let num_tiles = rows * cols;
        assert!(num_tiles > 0, "tile_grid must have rows*cols > 0");
        let dims = pixels.shape();
        let dims = dims.dims().to_vec();
        assert_eq!(dims.len(), 4, "pixels must be rank-4 (tiles, c, h, w)");
        assert_eq!(
            dims[0], num_tiles,
            "pixels axis 0 ({}) must equal tile_grid.0 * tile_grid.1 = {}",
            dims[0], num_tiles,
        );
        assert_eq!(dims[1], cfg.num_channels);
        assert_eq!(dims[2], cfg.image_size);
        assert_eq!(dims[3], cfg.image_size);

        let num_patches_per_tile = cfg.num_patches_per_tile();
        let merge = cfg.spatial_merge_size;
        assert!(merge >= 1);
        let patches_per_side = cfg.num_patches_per_side();
        assert_eq!(
            patches_per_side % merge,
            0,
            "patches_per_side ({}) must be a multiple of spatial_merge_size ({})",
            patches_per_side,
            merge,
        );

        let head_dim = cfg.head_dim();
        assert_eq!(head_dim % 2, 0, "head_dim must be even for split-half RoPE");

        // Pre-compute the per-tile 2D RoPE tables once and reuse for each tile.
        let (cos_data, sin_data) = build_2d_rope_tables(
            cfg.rope_theta,
            head_dim,
            patches_per_side,
        );
        let cos = pixels.const_f32_like(
            Arc::from(cos_data),
            Shape::from_dims(&[num_patches_per_tile, head_dim]),
        );
        let sin = pixels.const_f32_like(
            Arc::from(sin_data),
            Shape::from_dims(&[num_patches_per_tile, head_dim]),
        );

        // Per-tile position embedding.
        let pos = pixels.const_f32_like(
            Arc::clone(&self.weights.position_embedding),
            Shape::from_dims(&[1, num_patches_per_tile, cfg.hidden_size]),
        );

        let conv_w = pixels.const_f32_like(
            Arc::clone(&self.weights.patch_proj),
            Shape::from_dims(&[
                cfg.hidden_size,
                cfg.num_channels,
                cfg.patch_size,
                cfg.patch_size,
            ]),
        );
        let conv_b = pixels.const_f32_like(
            Arc::clone(&self.weights.patch_proj_bias),
            Shape::from_dims(&[cfg.hidden_size]),
        );

        // Encode each tile independently and collect the post-merged
        // projections, then concatenate them in row-major tile order.
        let mut tile_outputs: Vec<LazyTensor> = Vec::with_capacity(num_tiles);
        for ti in 0..num_tiles {
            // Slice out the tile (1, c, h, w).
            let tile = pixels.slice(0_usize, ti, 1)?;
            let conv_out = tile.conv2d(
                &conv_w,
                Some(&conv_b),
                (cfg.patch_size, cfg.patch_size),
                (0, 0),
                1,
            )?;
            // (1, hidden, ph, pw) -> (1, hidden, np) -> (1, np, hidden).
            let patches = conv_out
                .reshape(Shape::from_dims(&[
                    1,
                    cfg.hidden_size,
                    num_patches_per_tile,
                ]))?
                .permute([0, 2, 1_usize])?;
            let with_pos = patches.add(&pos)?;

            // Encoder.
            let mut h = with_pos;
            for block in &self.weights.blocks {
                h = self.apply_block(&h, block, &cos, &sin)?;
            }

            // Final LayerNorm.
            let h_norm = h.layer_norm_affine(
                Arc::clone(&self.weights.post_ln_gain),
                Arc::clone(&self.weights.post_ln_bias),
                cfg.layer_norm_eps,
            )?;

            // Projector: per-tile 2x2 (or m x m) spatial merge + 2-layer MLP.
            let projected = self.apply_projector(&h_norm)?;
            tile_outputs.push(projected);
        }

        // Concatenate tile outputs along the sequence axis (dim 0 of each
        // (merged_patches, text_hidden) tensor).
        let mut acc = tile_outputs.remove(0);
        for next in tile_outputs.into_iter() {
            acc = acc.concat(&next, 0_usize)?;
        }
        Ok(acc)
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        block: &PaddleOcrVlVisionBlockWeights,
        cos: &LazyTensor,
        sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();

        // ---- Self-attention ----
        let x_norm = x.layer_norm_affine(
            Arc::clone(&block.ln1_gain),
            Arc::clone(&block.ln1_bias),
            cfg.layer_norm_eps,
        )?;
        let q = block
            .q_proj
            .apply_linear(&x_norm, h, h)
            .add_trailing_bias(Arc::clone(&block.q_proj_bias))?;
        let k = block
            .k_proj
            .apply_linear(&x_norm, h, h)
            .add_trailing_bias(Arc::clone(&block.k_proj_bias))?;
        let v = block
            .v_proj
            .apply_linear(&x_norm, h, h)
            .add_trailing_bias(Arc::clone(&block.v_proj_bias))?;

        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        // Apply 2D RoPE to Q and K.
        let q_r = q.rope_with_tables(cos, sin)?;
        let k_r = k.rope_with_tables(cos, sin)?;

        let k_t = k_r.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?.mul_scalar(scale);
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        let attn_out = block
            .out_proj
            .apply_linear(&merged, h, h)
            .add_trailing_bias(Arc::clone(&block.out_proj_bias))?;
        let h1 = x.add(&attn_out)?;

        // ---- MLP ----
        let h1_norm = h1.layer_norm_affine(
            Arc::clone(&block.ln2_gain),
            Arc::clone(&block.ln2_bias),
            cfg.layer_norm_eps,
        )?;
        let fc1 = block
            .fc1
            .apply_linear(&h1_norm, h, cfg.intermediate_size)
            .add_trailing_bias(Arc::clone(&block.fc1_bias))?;
        let activated = match cfg.hidden_activation {
            PaddleOcrVlVisionActivation::Gelu => fc1.gelu_erf(),
            PaddleOcrVlVisionActivation::GeluPytorchTanh => fc1.gelu(),
            PaddleOcrVlVisionActivation::Silu => fc1.silu(),
            PaddleOcrVlVisionActivation::Relu => fc1.relu(),
        };
        let fc2 = block
            .fc2
            .apply_linear(&activated, cfg.intermediate_size, h)
            .add_trailing_bias(Arc::clone(&block.fc2_bias))?;
        h1.add(&fc2)
    }

    fn apply_projector(&self, h_norm: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights.projector;
        let hidden = cfg.hidden_size;
        let m = cfg.spatial_merge_size;
        let merged_hidden = hidden * m * m;

        // h_norm: (1, num_patches_per_tile, hidden). Pre-norm.
        let normed = h_norm.layer_norm_affine(
            Arc::clone(&weights.pre_norm_gain),
            Arc::clone(&weights.pre_norm_bias),
            cfg.layer_norm_eps,
        )?;

        // 2x2 (or m x m) spatial merge expressed as reshape + permute + reshape:
        //   (1, ph*pw, hidden)
        //   -> (h_merged, m, w_merged, m, hidden)
        //   -> (h_merged, w_merged, m, m, hidden)
        //   -> (h_merged * w_merged, m * m * hidden)
        // Drop the leading batch=1 with squeeze.
        let ph = cfg.num_patches_per_side();
        let pw = cfg.num_patches_per_side();
        let h_merged = ph / m;
        let w_merged = pw / m;
        let merged_count = h_merged * w_merged;

        let flat = normed.squeeze(0_usize)?; // (ph*pw, hidden)
        let grid = flat.reshape(Shape::from_dims(&[h_merged, m, w_merged, m, hidden]))?;
        let permuted = grid.permute([0, 2, 1, 3, 4_usize])?;
        let merged = permuted.reshape(Shape::from_dims(&[merged_count, merged_hidden]))?;

        let l1 = weights
            .linear_1
            .apply_linear(&merged, merged_hidden, merged_hidden)
            .add_trailing_bias(Arc::clone(&weights.linear_1_bias))?;
        // Projector activation is gelu_pytorch_tanh per eager.
        let activated = l1.gelu();
        let l2 = weights
            .linear_2
            .apply_linear(&activated, merged_hidden, self.text_hidden_size)
            .add_trailing_bias(Arc::clone(&weights.linear_2_bias))?;
        Ok(l2)
    }
}

/// Build the 2D split-half RoPE (cos, sin) tables for a single tile
/// of `num_patches_per_side × num_patches_per_side` patches.
///
/// Thin wrapper over [`build_2d_rope_tables_hw`] that passes the same
/// side length for both axes. Kept as the fixed-tile path's call site.
fn build_2d_rope_tables(
    theta: f64,
    head_dim: usize,
    num_patches_per_side: usize,
) -> (Vec<f32>, Vec<f32>) {
    build_2d_rope_tables_hw(theta, head_dim, num_patches_per_side, num_patches_per_side)
}

/// Build the 2D split-half RoPE (cos, sin) tables for a patch grid
/// of `h_patches × w_patches` patches (raster-scan order: position
/// `p = r * w_patches + c`).
///
/// Layout per patch `(r, c)`:
///   - first `qh` slots: `cos/sin(r * freq_h[i])` for the row-axis
///     frequencies (taken from the even indices of `inv_freq`)
///   - next  `qw` slots: `cos/sin(c * freq_w[i])` for the col-axis
///     frequencies (taken from the odd indices of `inv_freq`)
///   - second half of `head_dim` mirrors the first half (the standard
///     split-half rotation expected by `rope_with_tables`).
///
/// `inv_freq[2i]` (even) feeds rows, `inv_freq[2i+1]` (odd) feeds
/// columns — same convention as Pixtral / the eager NaViT.
fn build_2d_rope_tables_hw(
    theta: f64,
    head_dim: usize,
    h_patches: usize,
    w_patches: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dim = head_dim;
    let half = dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| (theta.powf(-2.0 * i as f64 / dim as f64)) as f32)
        .collect();
    let freqs_h: Vec<f32> = inv_freq.iter().step_by(2).copied().collect();
    let freqs_w: Vec<f32> = inv_freq.iter().skip(1).step_by(2).copied().collect();
    let qh = freqs_h.len();
    let qw = freqs_w.len();
    assert_eq!(qh + qw, half);

    let np = h_patches * w_patches;
    let mut cos = vec![0.0_f32; np * dim];
    let mut sin = vec![0.0_f32; np * dim];
    for r in 0..h_patches {
        for c in 0..w_patches {
            let p = r * w_patches + c;
            let off = p * dim;
            for i in 0..qh {
                let theta_val = r as f32 * freqs_h[i];
                cos[off + i] = theta_val.cos();
                sin[off + i] = theta_val.sin();
            }
            for i in 0..qw {
                let theta_val = c as f32 * freqs_w[i];
                cos[off + qh + i] = theta_val.cos();
                sin[off + qh + i] = theta_val.sin();
            }
            for i in 0..half {
                cos[off + half + i] = cos[off + i];
                sin[off + half + i] = sin[off + i];
            }
        }
    }
    (cos, sin)
}

/// Choose a tile grid `(rows, cols)` for an image of pixel
/// dimensions `(height, width)`. Aspect ratios near 1 collapse to
/// a 1x1 grid (no tiling); landscape images get more columns, and
/// portrait images get more rows. `max_tiles_per_side` caps each
/// axis so the encoder cost stays bounded.
///
/// The decision is the rounded aspect ratio, clamped to
/// `[1, max_tiles_per_side]`. Pure host arithmetic — no graph ops.
pub fn aspect_ratio_chooser(
    height: usize,
    width: usize,
    max_tiles_per_side: usize,
) -> (usize, usize) {
    assert!(max_tiles_per_side >= 1);
    if height == 0 || width == 0 {
        return (1, 1);
    }
    let cap = max_tiles_per_side as f64;
    let h = height as f64;
    let w = width as f64;
    let ratio = w / h;
    if (ratio - 1.0).abs() < 0.20 {
        return (1, 1);
    }
    if ratio >= 1.0 {
        // landscape: more cols than rows
        let cols = (ratio.round().max(1.0)).min(cap) as usize;
        (1, cols)
    } else {
        // portrait
        let rows = ((1.0 / ratio).round().max(1.0)).min(cap) as usize;
        (rows, 1)
    }
}

/// Crop an interleaved-by-channel RGB image into `rows * cols`
/// equal-size tiles in row-major order. Tile pixels are returned
/// in `(channels, tile_h, tile_w)` order to match the patch-embed
/// Conv2d's expected layout.
///
/// `image` is shape `(channels, height, width)` flattened row-
/// major (the standard CHW layout). The cropped tile size is
/// `(height / rows, width / cols)`; remainder rows/cols are
/// discarded (matches eager's center-discard behavior). Pure host
/// arithmetic — no graph ops.
pub fn partition_image(
    image: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    rows: usize,
    cols: usize,
) -> Vec<Vec<f32>> {
    assert!(rows >= 1 && cols >= 1);
    assert_eq!(image.len(), channels * height * width);
    let tile_h = height / rows;
    let tile_w = width / cols;
    assert!(
        tile_h > 0 && tile_w > 0,
        "partition_image: tile dim collapsed (height={height}, width={width}, rows={rows}, cols={cols})",
    );

    let mut tiles = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let mut tile = Vec::with_capacity(channels * tile_h * tile_w);
            let y0 = r * tile_h;
            let x0 = c * tile_w;
            for ch in 0..channels {
                let plane_off = ch * height * width;
                for yy in 0..tile_h {
                    let row_off = plane_off + (y0 + yy) * width + x0;
                    tile.extend_from_slice(&image[row_off..row_off + tile_w]);
                }
            }
            tiles.push(tile);
        }
    }
    tiles
}

// ---- Safetensors loader ----------------------------------------------------

/// Load PaddleOCR-VL vision-tower weights under the given HF
/// prefixes. `vision_prefix` is the prefix for the NaViT encoder
/// itself (typically `"visual.vision_model."`); `projector_prefix`
/// names the `mlp_AR` projector (typically `"mlp_AR."`). HF
/// tensor names:
///   - `{vision_prefix}embeddings.patch_embedding.{weight,bias}`
///   - `{vision_prefix}embeddings.position_embedding.weight`
///   - `{vision_prefix}encoder.layers.{i}.layer_norm{1,2}.{weight,bias}`
///   - `{vision_prefix}encoder.layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}`
///   - `{vision_prefix}encoder.layers.{i}.mlp.fc{1,2}.{weight,bias}`
///   - `{vision_prefix}post_layernorm.{weight,bias}`
///   - `{projector_prefix}pre_norm.{weight,bias}`
///   - `{projector_prefix}linear_{1,2}.{weight,bias}`
pub fn load_paddleocr_vl_vision_weights(
    st: &crate::safetensors::MmapedSafetensors,
    cfg: &PaddleOcrVlVisionConfig,
    text_hidden_size: usize,
    vision_prefix: &str,
    projector_prefix: &str,
) -> Result<PaddleOcrVlVisionWeights> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let np = cfg.num_patches_per_tile();
    let m = cfg.spatial_merge_size;
    let merged_hidden = h * m * m;

    let patch_proj = load_tensor_as_f32(
        st, &format!("{vision_prefix}embeddings.patch_embedding.weight"),
    )?;
    let expected_patch = h * cfg.num_channels * cfg.patch_size * cfg.patch_size;
    if patch_proj.len() != expected_patch {
        crate::bail!(
            "{vision_prefix}embeddings.patch_embedding.weight: {} elts, expected {}",
            patch_proj.len(), expected_patch,
        );
    }
    let patch_proj_bias = load_tensor_as_f32(
        st, &format!("{vision_prefix}embeddings.patch_embedding.bias"),
    )?;
    if patch_proj_bias.len() != h {
        crate::bail!(
            "{vision_prefix}embeddings.patch_embedding.bias: {} elts, expected {}",
            patch_proj_bias.len(), h,
        );
    }
    let position_embedding = load_tensor_as_f32(
        st, &format!("{vision_prefix}embeddings.position_embedding.weight"),
    )?;
    if position_embedding.len() != np * h {
        crate::bail!(
            "{vision_prefix}embeddings.position_embedding.weight: {} elts, expected {}",
            position_embedding.len(), np * h,
        );
    }

    let post_ln_gain = load_tensor_as_f32(
        st, &format!("{vision_prefix}post_layernorm.weight"),
    )?;
    let post_ln_bias = load_tensor_as_f32(
        st, &format!("{vision_prefix}post_layernorm.bias"),
    )?;

    let mut blocks: Vec<PaddleOcrVlVisionBlockWeights> =
        Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("{vision_prefix}encoder.layers.{i}");
        let ln1_gain = load_tensor_as_f32(st, &format!("{p}.layer_norm1.weight"))?;
        let ln1_bias = load_tensor_as_f32(st, &format!("{p}.layer_norm1.bias"))?;
        let ln2_gain = load_tensor_as_f32(st, &format!("{p}.layer_norm2.weight"))?;
        let ln2_bias = load_tensor_as_f32(st, &format!("{p}.layer_norm2.bias"))?;
        let q_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.q_proj.weight"), h, h,
        )?;
        let q_proj_bias = load_tensor_as_f32(st, &format!("{p}.self_attn.q_proj.bias"))?;
        let k_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.k_proj.weight"), h, h,
        )?;
        let k_proj_bias = load_tensor_as_f32(st, &format!("{p}.self_attn.k_proj.bias"))?;
        let v_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.v_proj.weight"), h, h,
        )?;
        let v_proj_bias = load_tensor_as_f32(st, &format!("{p}.self_attn.v_proj.bias"))?;
        let out_proj = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.self_attn.out_proj.weight"), h, h,
        )?;
        let out_proj_bias = load_tensor_as_f32(
            st, &format!("{p}.self_attn.out_proj.bias"),
        )?;
        let fc1 = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.mlp.fc1.weight"), inter, h,
        )?;
        let fc1_bias = load_tensor_as_f32(st, &format!("{p}.mlp.fc1.bias"))?;
        let fc2 = load_transposed_matrix_preserve_dtype(
            st, &format!("{p}.mlp.fc2.weight"), h, inter,
        )?;
        let fc2_bias = load_tensor_as_f32(st, &format!("{p}.mlp.fc2.bias"))?;
        blocks.push(PaddleOcrVlVisionBlockWeights {
            ln1_gain: Arc::from(ln1_gain),
            ln1_bias: Arc::from(ln1_bias),
            q_proj, q_proj_bias: Arc::from(q_proj_bias),
            k_proj, k_proj_bias: Arc::from(k_proj_bias),
            v_proj, v_proj_bias: Arc::from(v_proj_bias),
            out_proj, out_proj_bias: Arc::from(out_proj_bias),
            ln2_gain: Arc::from(ln2_gain),
            ln2_bias: Arc::from(ln2_bias),
            fc1, fc1_bias: Arc::from(fc1_bias),
            fc2, fc2_bias: Arc::from(fc2_bias),
        });
    }

    // Projector (`mlp_AR` in HF): pre_norm + 2-layer MLP.
    let pre_norm_gain = load_tensor_as_f32(
        st, &format!("{projector_prefix}pre_norm.weight"),
    )?;
    let pre_norm_bias = load_tensor_as_f32(
        st, &format!("{projector_prefix}pre_norm.bias"),
    )?;
    let linear_1 = load_transposed_matrix_preserve_dtype(
        st, &format!("{projector_prefix}linear_1.weight"),
        merged_hidden, merged_hidden,
    )?;
    let linear_1_bias = load_tensor_as_f32(
        st, &format!("{projector_prefix}linear_1.bias"),
    )?;
    let linear_2 = load_transposed_matrix_preserve_dtype(
        st, &format!("{projector_prefix}linear_2.weight"),
        text_hidden_size, merged_hidden,
    )?;
    let linear_2_bias = load_tensor_as_f32(
        st, &format!("{projector_prefix}linear_2.bias"),
    )?;

    let projector = PaddleOcrVlVisionProjectorWeights {
        pre_norm_gain: Arc::from(pre_norm_gain),
        pre_norm_bias: Arc::from(pre_norm_bias),
        linear_1,
        linear_1_bias: Arc::from(linear_1_bias),
        linear_2,
        linear_2_bias: Arc::from(linear_2_bias),
    };

    Ok(PaddleOcrVlVisionWeights {
        patch_proj: Arc::from(patch_proj),
        patch_proj_bias: Arc::from(patch_proj_bias),
        position_embedding: Arc::from(position_embedding),
        blocks,
        post_ln_gain: Arc::from(post_ln_gain),
        post_ln_bias: Arc::from(post_ln_bias),
        projector,
    })
}

impl PaddleOcrVlVisionWeights {
    /// Load PaddleOCR-VL vision weights from a HuggingFace
    /// safetensors file. The PaddleOCR-VL release puts the vision
    /// encoder under `visual.vision_model.*` and the projector
    /// under `mlp_AR.*` at the top level. `text_hidden_size` is the
    /// language model's hidden size — the projector's `linear_2`
    /// projects into it.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &PaddleOcrVlVisionConfig,
        text_hidden_size: usize,
    ) -> Result<Self> {
        load_paddleocr_vl_vision_weights(
            st, cfg, text_hidden_size, "visual.vision_model.", "mlp_AR.",
        )
    }
}

// ===========================================================================
// NaViT — dynamic-resolution variant
// ===========================================================================

/// Default LFU cache capacity for the interpolated position embedding.
/// Mirrors the eager `DEFAULT_POS_EMBED_CACHE_SIZE`.
const NAVIT_DEFAULT_POS_EMBED_CACHE_SIZE: usize = 16;

/// Configuration for the dynamic-resolution NaViT variant of the
/// PaddleOCR-VL vision tower. Same encoder topology as
/// [`PaddleOcrVlVisionConfig`] but without a hard `image_size` —
/// `base_image_size` is only used to recover the base position
/// embedding grid shape (`base_image_size / patch_size`), which the
/// model bilinearly interpolates to the actual patch grid at forward
/// time.
#[derive(Debug, Clone, PartialEq)]
pub struct PaddleOcrVlNaVitConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    /// Source resolution that defines the base position-embedding
    /// grid (`base_image_size / patch_size` per side). The published
    /// checkpoint trains at `image_size=384`, giving a 27×27 base
    /// grid for `patch_size=14`.
    pub base_image_size: usize,
    pub patch_size: usize,
    pub hidden_activation: PaddleOcrVlVisionActivation,
    pub layer_norm_eps: f64,
    pub spatial_merge_size: usize,
    pub rope_theta: f64,
    /// Upper bound on the rasterized pixel count of an input image
    /// (host-side smart-resize cap). Not enforced by `forward` —
    /// callers consult this when preprocessing.
    pub max_pixels: usize,
}

impl PaddleOcrVlNaVitConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Side length of the base position embedding grid.
    pub fn base_grid_size(&self) -> usize {
        self.base_image_size / self.patch_size
    }

    /// Total count of base positions (`base_grid_size^2`).
    pub fn base_num_positions(&self) -> usize {
        let s = self.base_grid_size();
        s * s
    }

    /// Preset matching the published PaddleOCR-VL vision encoder
    /// (hidden=1152, layers=27, heads=16, patch=14, base=384,
    /// base grid=27×27).
    pub fn paddleocr_vl() -> Self {
        Self {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            num_channels: 3,
            base_image_size: 384,
            patch_size: 14,
            hidden_activation: PaddleOcrVlVisionActivation::GeluPytorchTanh,
            layer_norm_eps: 1e-6,
            spatial_merge_size: 2,
            rope_theta: 10_000.0,
            // 2_822_400 = (1680 * 1680) — eager default
            // `image_processing_paddleocr_vl.py`.
            max_pixels: 2_822_400,
        }
    }
}

/// Encoder-block + projector weight set for the NaViT variant. Same
/// concrete tensor list as [`PaddleOcrVlVisionWeights`]; the only
/// semantic difference is that `position_embedding` is now the **base
/// grid** (shape `(base_num_positions, hidden)` = `(27*27, hidden)`
/// for the published checkpoint), interpreted as a 2D grid by the
/// bilinear interpolator at forward time.
#[derive(Debug, Clone)]
pub struct PaddleOcrVlNaVitWeights {
    /// Conv2d patch projection `[hidden, num_channels, patch, patch]`.
    pub patch_proj: Arc<[f32]>,
    pub patch_proj_bias: Arc<[f32]>,
    /// `[base_grid_size * base_grid_size, hidden]` — the learned base
    /// position embedding to bilinearly interpolate from.
    pub position_embedding: Arc<[f32]>,
    pub blocks: Vec<PaddleOcrVlVisionBlockWeights>,
    pub post_ln_gain: Arc<[f32]>,
    pub post_ln_bias: Arc<[f32]>,
    pub projector: PaddleOcrVlVisionProjectorWeights,
}

/// LFU cache for the bilinearly interpolated position embedding.
/// Keyed on `(h_patches, w_patches)`. Values are the host-side
/// interpolated tensor data (`Arc<[f32]>`) so that re-hits avoid
/// recomputing the bilinear stencil; the LazyTensor side rebuilds
/// the const node fresh against the active graph each call.
#[derive(Debug, Default)]
struct PosEmbedLfuCache {
    entries: HashMap<(usize, usize), (Arc<[f32]>, usize)>,
    max_size: usize,
}

impl PosEmbedLfuCache {
    fn new(max_size: usize) -> Self {
        Self { entries: HashMap::with_capacity(max_size.max(1)), max_size: max_size.max(1) }
    }

    fn get(&mut self, key: (usize, usize)) -> Option<Arc<[f32]>> {
        let (data, freq) = self.entries.get_mut(&key)?;
        *freq = freq.saturating_add(1);
        Some(Arc::clone(data))
    }

    fn insert(&mut self, key: (usize, usize), data: Arc<[f32]>) {
        if self.entries.contains_key(&key) {
            // Replace and bump frequency.
            let slot = self.entries.get_mut(&key).expect("just checked");
            slot.0 = data;
            slot.1 = slot.1.saturating_add(1);
            return;
        }
        if self.entries.len() >= self.max_size {
            // Evict the least frequently used entry.
            if let Some((&lfu_key, _)) =
                self.entries.iter().min_by_key(|(_, (_, freq))| *freq)
            {
                self.entries.remove(&lfu_key);
            }
        }
        self.entries.insert(key, (data, 1));
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Dynamic-resolution NaViT variant of the PaddleOCR-VL vision
/// tower. Processes a single image at its natural aspect ratio:
/// patch-embed → per-grid bilinear-interpolated position embed →
/// encoder stack with 2D RoPE built for the actual patch grid → post-
/// LayerNorm → projector. The patch grid is derived from the input
/// shape `(H/patch_size, W/patch_size)`; both axes must already be a
/// multiple of `patch_size` (the host preprocessor is responsible).
///
/// Single-image only (batch dim must be 1). Multi-image packing via
/// `cu_seqlens` and variable-length attention is a follow-up; see the
/// session prompt for the deferred surface.
#[derive(Debug, Clone)]
pub struct PaddleOcrVlNaVitModel {
    pub config: PaddleOcrVlNaVitConfig,
    pub text_hidden_size: usize,
    pub weights: PaddleOcrVlNaVitWeights,
    /// Shared LFU cache for the interpolated position embedding. The
    /// `Arc<Mutex<...>>` wrap keeps the model `Send + Sync` and gives
    /// shallow-clone semantics (clones share the cache).
    pos_embed_cache: Arc<Mutex<PosEmbedLfuCache>>,
}

impl PaddleOcrVlNaVitModel {
    /// Build a fresh NaViT model with a default-capacity position
    /// embedding cache (`NAVIT_DEFAULT_POS_EMBED_CACHE_SIZE` = 16,
    /// matching the eager port).
    pub fn new(
        config: PaddleOcrVlNaVitConfig,
        text_hidden_size: usize,
        weights: PaddleOcrVlNaVitWeights,
    ) -> Self {
        Self {
            config,
            text_hidden_size,
            weights,
            pos_embed_cache: Arc::new(Mutex::new(PosEmbedLfuCache::new(
                NAVIT_DEFAULT_POS_EMBED_CACHE_SIZE,
            ))),
        }
    }

    /// Build a NaViT model with an explicit position-embedding cache
    /// capacity. `0` is silently lifted to `1` (eviction always runs
    /// against a non-empty cache).
    pub fn with_pos_embed_cache_size(
        config: PaddleOcrVlNaVitConfig,
        text_hidden_size: usize,
        weights: PaddleOcrVlNaVitWeights,
        cache_size: usize,
    ) -> Self {
        Self {
            config,
            text_hidden_size,
            weights,
            pos_embed_cache: Arc::new(Mutex::new(PosEmbedLfuCache::new(cache_size))),
        }
    }

    /// Number of entries currently cached. Test-only — used to assert
    /// cache-hit behaviour without exposing the internal map.
    #[doc(hidden)]
    pub fn pos_embed_cache_len(&self) -> usize {
        self.pos_embed_cache.lock().expect("pos_embed_cache lock").len()
    }

    /// Forward pass on a single image.
    ///
    /// `pixel_values` must be rank-4 with shape `(1, num_channels, H,
    /// W)` where `H` and `W` are both multiples of `config.patch_size`.
    /// Misalignment panics with a clear message before any graph op is
    /// emitted (the input shape is known at build time).
    ///
    /// Returns a `(merged_patches, text_hidden_size)` tensor where
    /// `merged_patches = (H/patch_size * W/patch_size) /
    /// spatial_merge_size^2`.
    pub fn forward(&self, pixel_values: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let dims = pixel_values.shape();
        let dims = dims.dims().to_vec();
        assert_eq!(
            dims.len(),
            4,
            "PaddleOcrVlNaVitModel::forward: pixel_values must be rank-4 (1, C, H, W), got {dims:?}",
        );
        assert_eq!(
            dims[0], 1,
            "PaddleOcrVlNaVitModel::forward: only batch=1 is supported (v1); got batch={}",
            dims[0],
        );
        assert_eq!(
            dims[1], cfg.num_channels,
            "PaddleOcrVlNaVitModel::forward: channel dim {} != config.num_channels {}",
            dims[1], cfg.num_channels,
        );
        let h = dims[2];
        let w = dims[3];
        assert!(
            h > 0 && w > 0,
            "PaddleOcrVlNaVitModel::forward: H and W must be positive, got ({h}, {w})",
        );
        assert!(
            h % cfg.patch_size == 0,
            "PaddleOcrVlNaVitModel::forward: H ({h}) is not a multiple of patch_size ({}); \
             pad host-side before calling forward",
            cfg.patch_size,
        );
        assert!(
            w % cfg.patch_size == 0,
            "PaddleOcrVlNaVitModel::forward: W ({w}) is not a multiple of patch_size ({}); \
             pad host-side before calling forward",
            cfg.patch_size,
        );

        let h_patches = h / cfg.patch_size;
        let w_patches = w / cfg.patch_size;
        let num_patches = h_patches * w_patches;
        let merge = cfg.spatial_merge_size;
        assert!(merge >= 1, "spatial_merge_size must be >= 1, got {merge}");
        assert!(
            h_patches % merge == 0 && w_patches % merge == 0,
            "PaddleOcrVlNaVitModel::forward: patch grid ({h_patches}, {w_patches}) is not a \
             multiple of spatial_merge_size ({merge}) on both axes",
        );

        let head_dim = cfg.head_dim();
        assert_eq!(head_dim % 2, 0, "head_dim must be even for split-half RoPE");

        // 2D RoPE tables for this specific grid.
        let (cos_data, sin_data) = build_2d_rope_tables_hw(
            cfg.rope_theta,
            head_dim,
            h_patches,
            w_patches,
        );
        let cos = pixel_values.const_f32_like(
            Arc::from(cos_data),
            Shape::from_dims(&[num_patches, head_dim]),
        );
        let sin = pixel_values.const_f32_like(
            Arc::from(sin_data),
            Shape::from_dims(&[num_patches, head_dim]),
        );

        // Bilinear-interpolated position embedding for this grid.
        let pos_data = self.interpolated_position_embedding(h_patches, w_patches);
        let pos = pixel_values.const_f32_like(
            pos_data,
            Shape::from_dims(&[1, num_patches, cfg.hidden_size]),
        );

        // Conv2d patch embedding.
        let conv_w = pixel_values.const_f32_like(
            Arc::clone(&self.weights.patch_proj),
            Shape::from_dims(&[
                cfg.hidden_size,
                cfg.num_channels,
                cfg.patch_size,
                cfg.patch_size,
            ]),
        );
        let conv_b = pixel_values.const_f32_like(
            Arc::clone(&self.weights.patch_proj_bias),
            Shape::from_dims(&[cfg.hidden_size]),
        );
        let conv_out = pixel_values.conv2d(
            &conv_w,
            Some(&conv_b),
            (cfg.patch_size, cfg.patch_size),
            (0, 0),
            1,
        )?;
        // (1, hidden, h_patches, w_patches) -> (1, hidden, num_patches)
        //   -> (1, num_patches, hidden).
        let patches = conv_out
            .reshape(Shape::from_dims(&[1, cfg.hidden_size, num_patches]))?
            .permute([0, 2, 1_usize])?;
        let with_pos = patches.add(&pos)?;

        // Encoder.
        let mut hidden = with_pos;
        for block in &self.weights.blocks {
            hidden = self.apply_block(&hidden, block, &cos, &sin)?;
        }

        // Final LayerNorm.
        let h_norm = hidden.layer_norm_affine(
            Arc::clone(&self.weights.post_ln_gain),
            Arc::clone(&self.weights.post_ln_bias),
            cfg.layer_norm_eps,
        )?;

        // Projector with m×m spatial merge.
        self.apply_projector(&h_norm, h_patches, w_patches)
    }

    fn apply_block(
        &self,
        x: &LazyTensor,
        block: &PaddleOcrVlVisionBlockWeights,
        cos: &LazyTensor,
        sin: &LazyTensor,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();

        // ---- Self-attention ----
        let x_norm = x.layer_norm_affine(
            Arc::clone(&block.ln1_gain),
            Arc::clone(&block.ln1_bias),
            cfg.layer_norm_eps,
        )?;
        let q = block
            .q_proj
            .apply_linear(&x_norm, h, h)
            .add_trailing_bias(Arc::clone(&block.q_proj_bias))?;
        let k = block
            .k_proj
            .apply_linear(&x_norm, h, h)
            .add_trailing_bias(Arc::clone(&block.k_proj_bias))?;
        let v = block
            .v_proj
            .apply_linear(&x_norm, h, h)
            .add_trailing_bias(Arc::clone(&block.v_proj_bias))?;

        let q = q.split_heads(n_heads, head_dim)?;
        let k = k.split_heads(n_heads, head_dim)?;
        let v = v.split_heads(n_heads, head_dim)?;

        let q_r = q.rope_with_tables(cos, sin)?;
        let k_r = k.rope_with_tables(cos, sin)?;

        let k_t = k_r.transpose()?;
        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let scores = q_r.matmul(&k_t)?.mul_scalar(scale);
        let probs = scores.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        let merged = ctx.merge_heads()?;
        let attn_out = block
            .out_proj
            .apply_linear(&merged, h, h)
            .add_trailing_bias(Arc::clone(&block.out_proj_bias))?;
        let h1 = x.add(&attn_out)?;

        // ---- MLP ----
        let h1_norm = h1.layer_norm_affine(
            Arc::clone(&block.ln2_gain),
            Arc::clone(&block.ln2_bias),
            cfg.layer_norm_eps,
        )?;
        let fc1 = block
            .fc1
            .apply_linear(&h1_norm, h, cfg.intermediate_size)
            .add_trailing_bias(Arc::clone(&block.fc1_bias))?;
        let activated = match cfg.hidden_activation {
            PaddleOcrVlVisionActivation::Gelu => fc1.gelu_erf(),
            PaddleOcrVlVisionActivation::GeluPytorchTanh => fc1.gelu(),
            PaddleOcrVlVisionActivation::Silu => fc1.silu(),
            PaddleOcrVlVisionActivation::Relu => fc1.relu(),
        };
        let fc2 = block
            .fc2
            .apply_linear(&activated, cfg.intermediate_size, h)
            .add_trailing_bias(Arc::clone(&block.fc2_bias))?;
        h1.add(&fc2)
    }

    fn apply_projector(
        &self,
        h_norm: &LazyTensor,
        h_patches: usize,
        w_patches: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let weights = &self.weights.projector;
        let hidden = cfg.hidden_size;
        let m = cfg.spatial_merge_size;
        let merged_hidden = hidden * m * m;

        // h_norm: (1, num_patches, hidden). Pre-norm.
        let normed = h_norm.layer_norm_affine(
            Arc::clone(&weights.pre_norm_gain),
            Arc::clone(&weights.pre_norm_bias),
            cfg.layer_norm_eps,
        )?;

        let h_merged = h_patches / m;
        let w_merged = w_patches / m;
        let merged_count = h_merged * w_merged;

        let flat = normed.squeeze(0_usize)?; // (h_patches * w_patches, hidden)
        let grid = flat.reshape(Shape::from_dims(&[h_merged, m, w_merged, m, hidden]))?;
        let permuted = grid.permute([0, 2, 1, 3, 4_usize])?;
        let merged = permuted.reshape(Shape::from_dims(&[merged_count, merged_hidden]))?;

        let l1 = weights
            .linear_1
            .apply_linear(&merged, merged_hidden, merged_hidden)
            .add_trailing_bias(Arc::clone(&weights.linear_1_bias))?;
        let activated = l1.gelu();
        let l2 = weights
            .linear_2
            .apply_linear(&activated, merged_hidden, self.text_hidden_size)
            .add_trailing_bias(Arc::clone(&weights.linear_2_bias))?;
        Ok(l2)
    }

    /// Bilinearly interpolate the base position embedding to the
    /// requested `(target_h, target_w)` patch grid. Cached with LFU
    /// eviction (default capacity = 16). Matches PyTorch's
    /// `nn.functional.interpolate(mode='bilinear', align_corners=False)`.
    fn interpolated_position_embedding(
        &self,
        target_h: usize,
        target_w: usize,
    ) -> Arc<[f32]> {
        let key = (target_h, target_w);
        if let Some(cached) = self
            .pos_embed_cache
            .lock()
            .expect("pos_embed_cache lock")
            .get(key)
        {
            return cached;
        }
        let cfg = &self.config;
        let base = cfg.base_grid_size();
        let hidden = cfg.hidden_size;
        let data = bilinear_interpolate_position_embedding(
            &self.weights.position_embedding,
            base,
            base,
            hidden,
            target_h,
            target_w,
        );
        let arc: Arc<[f32]> = Arc::from(data);
        self.pos_embed_cache
            .lock()
            .expect("pos_embed_cache lock")
            .insert(key, Arc::clone(&arc));
        arc
    }
}

/// Bilinearly interpolate a `(base_h, base_w, hidden)` position
/// embedding to `(target_h, target_w, hidden)` and return the result
/// as a flat row-major `Vec<f32>` of length
/// `target_h * target_w * hidden`. Source layout is row-major over
/// (row, col, channel) — matches `position_embedding.reshape(base_h,
/// base_w, hidden)` in PyTorch.
///
/// `align_corners=False`: source coordinates are computed as
/// `(t + 0.5) * (base / target) - 0.5`, clamped to `[0, base - 1]`.
/// If `target == base` along both axes, returns the source verbatim
/// (a fast path matching the eager fast return).
fn bilinear_interpolate_position_embedding(
    src: &[f32],
    base_h: usize,
    base_w: usize,
    hidden: usize,
    target_h: usize,
    target_w: usize,
) -> Vec<f32> {
    assert_eq!(
        src.len(),
        base_h * base_w * hidden,
        "bilinear_interpolate_position_embedding: src len {} != base_h*base_w*hidden = {}",
        src.len(),
        base_h * base_w * hidden,
    );
    if target_h == base_h && target_w == base_w {
        return src.to_vec();
    }
    let scale_h = base_h as f64 / target_h as f64;
    let scale_w = base_w as f64 / target_w as f64;
    let mut out = Vec::with_capacity(target_h * target_w * hidden);
    for ty in 0..target_h {
        let sy = ((ty as f64 + 0.5) * scale_h - 0.5)
            .max(0.0)
            .min((base_h - 1) as f64);
        let sy0 = sy.floor() as usize;
        let sy1 = (sy0 + 1).min(base_h - 1);
        let fy = (sy - sy0 as f64) as f32;
        for tx in 0..target_w {
            let sx = ((tx as f64 + 0.5) * scale_w - 0.5)
                .max(0.0)
                .min((base_w - 1) as f64);
            let sx0 = sx.floor() as usize;
            let sx1 = (sx0 + 1).min(base_w - 1);
            let fx = (sx - sx0 as f64) as f32;

            let w00 = (1.0 - fy) * (1.0 - fx);
            let w01 = (1.0 - fy) * fx;
            let w10 = fy * (1.0 - fx);
            let w11 = fy * fx;

            let off00 = (sy0 * base_w + sx0) * hidden;
            let off01 = (sy0 * base_w + sx1) * hidden;
            let off10 = (sy1 * base_w + sx0) * hidden;
            let off11 = (sy1 * base_w + sx1) * hidden;
            for d in 0..hidden {
                let v = w00 * src[off00 + d]
                    + w01 * src[off01 + d]
                    + w10 * src[off10 + d]
                    + w11 * src[off11 + d];
                out.push(v);
            }
        }
    }
    out
}

/// Load PaddleOCR-VL NaViT weights under the given HF prefixes.
///
/// Tensor name surface is identical to the fixed-tile loader (same
/// vision encoder topology); the only difference is the semantic
/// interpretation of `embeddings.position_embedding.weight` — this
/// loader treats it as the `(base_num_positions, hidden)` base grid
/// that the model bilinearly interpolates at forward time. The
/// `packing_position_embedding` fallback at eager line 122-124 is
/// not used by the lazy port and is skipped.
pub fn load_paddleocr_vl_navit_weights(
    st: &crate::safetensors::MmapedSafetensors,
    cfg: &PaddleOcrVlNaVitConfig,
    text_hidden_size: usize,
    vision_prefix: &str,
    projector_prefix: &str,
) -> Result<PaddleOcrVlNaVitWeights> {
    let h = cfg.hidden_size;
    let inter = cfg.intermediate_size;
    let np = cfg.base_num_positions();
    let m = cfg.spatial_merge_size;
    let merged_hidden = h * m * m;

    let patch_proj = load_tensor_as_f32(
        st,
        &format!("{vision_prefix}embeddings.patch_embedding.weight"),
    )?;
    let expected_patch = h * cfg.num_channels * cfg.patch_size * cfg.patch_size;
    if patch_proj.len() != expected_patch {
        crate::bail!(
            "{vision_prefix}embeddings.patch_embedding.weight: {} elts, expected {}",
            patch_proj.len(),
            expected_patch,
        );
    }
    let patch_proj_bias = load_tensor_as_f32(
        st,
        &format!("{vision_prefix}embeddings.patch_embedding.bias"),
    )?;
    if patch_proj_bias.len() != h {
        crate::bail!(
            "{vision_prefix}embeddings.patch_embedding.bias: {} elts, expected {}",
            patch_proj_bias.len(),
            h,
        );
    }
    let position_embedding = load_tensor_as_f32(
        st,
        &format!("{vision_prefix}embeddings.position_embedding.weight"),
    )?;
    if position_embedding.len() != np * h {
        crate::bail!(
            "{vision_prefix}embeddings.position_embedding.weight: {} elts, expected {} \
             (base_grid_size={}, hidden={})",
            position_embedding.len(),
            np * h,
            cfg.base_grid_size(),
            h,
        );
    }

    let post_ln_gain =
        load_tensor_as_f32(st, &format!("{vision_prefix}post_layernorm.weight"))?;
    let post_ln_bias =
        load_tensor_as_f32(st, &format!("{vision_prefix}post_layernorm.bias"))?;

    let mut blocks: Vec<PaddleOcrVlVisionBlockWeights> =
        Vec::with_capacity(cfg.num_hidden_layers);
    for i in 0..cfg.num_hidden_layers {
        let p = format!("{vision_prefix}encoder.layers.{i}");
        let ln1_gain = load_tensor_as_f32(st, &format!("{p}.layer_norm1.weight"))?;
        let ln1_bias = load_tensor_as_f32(st, &format!("{p}.layer_norm1.bias"))?;
        let ln2_gain = load_tensor_as_f32(st, &format!("{p}.layer_norm2.weight"))?;
        let ln2_bias = load_tensor_as_f32(st, &format!("{p}.layer_norm2.bias"))?;
        let q_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{p}.self_attn.q_proj.weight"),
            h,
            h,
        )?;
        let q_proj_bias =
            load_tensor_as_f32(st, &format!("{p}.self_attn.q_proj.bias"))?;
        let k_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{p}.self_attn.k_proj.weight"),
            h,
            h,
        )?;
        let k_proj_bias =
            load_tensor_as_f32(st, &format!("{p}.self_attn.k_proj.bias"))?;
        let v_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{p}.self_attn.v_proj.weight"),
            h,
            h,
        )?;
        let v_proj_bias =
            load_tensor_as_f32(st, &format!("{p}.self_attn.v_proj.bias"))?;
        let out_proj = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{p}.self_attn.out_proj.weight"),
            h,
            h,
        )?;
        let out_proj_bias =
            load_tensor_as_f32(st, &format!("{p}.self_attn.out_proj.bias"))?;
        let fc1 = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{p}.mlp.fc1.weight"),
            inter,
            h,
        )?;
        let fc1_bias = load_tensor_as_f32(st, &format!("{p}.mlp.fc1.bias"))?;
        let fc2 = load_transposed_matrix_preserve_dtype(
            st,
            &format!("{p}.mlp.fc2.weight"),
            h,
            inter,
        )?;
        let fc2_bias = load_tensor_as_f32(st, &format!("{p}.mlp.fc2.bias"))?;
        blocks.push(PaddleOcrVlVisionBlockWeights {
            ln1_gain: Arc::from(ln1_gain),
            ln1_bias: Arc::from(ln1_bias),
            q_proj,
            q_proj_bias: Arc::from(q_proj_bias),
            k_proj,
            k_proj_bias: Arc::from(k_proj_bias),
            v_proj,
            v_proj_bias: Arc::from(v_proj_bias),
            out_proj,
            out_proj_bias: Arc::from(out_proj_bias),
            ln2_gain: Arc::from(ln2_gain),
            ln2_bias: Arc::from(ln2_bias),
            fc1,
            fc1_bias: Arc::from(fc1_bias),
            fc2,
            fc2_bias: Arc::from(fc2_bias),
        });
    }

    let pre_norm_gain =
        load_tensor_as_f32(st, &format!("{projector_prefix}pre_norm.weight"))?;
    let pre_norm_bias =
        load_tensor_as_f32(st, &format!("{projector_prefix}pre_norm.bias"))?;
    let linear_1 = load_transposed_matrix_preserve_dtype(
        st,
        &format!("{projector_prefix}linear_1.weight"),
        merged_hidden,
        merged_hidden,
    )?;
    let linear_1_bias =
        load_tensor_as_f32(st, &format!("{projector_prefix}linear_1.bias"))?;
    let linear_2 = load_transposed_matrix_preserve_dtype(
        st,
        &format!("{projector_prefix}linear_2.weight"),
        text_hidden_size,
        merged_hidden,
    )?;
    let linear_2_bias =
        load_tensor_as_f32(st, &format!("{projector_prefix}linear_2.bias"))?;

    let projector = PaddleOcrVlVisionProjectorWeights {
        pre_norm_gain: Arc::from(pre_norm_gain),
        pre_norm_bias: Arc::from(pre_norm_bias),
        linear_1,
        linear_1_bias: Arc::from(linear_1_bias),
        linear_2,
        linear_2_bias: Arc::from(linear_2_bias),
    };

    Ok(PaddleOcrVlNaVitWeights {
        patch_proj: Arc::from(patch_proj),
        patch_proj_bias: Arc::from(patch_proj_bias),
        position_embedding: Arc::from(position_embedding),
        blocks,
        post_ln_gain: Arc::from(post_ln_gain),
        post_ln_bias: Arc::from(post_ln_bias),
        projector,
    })
}

impl PaddleOcrVlNaVitWeights {
    /// Load PaddleOCR-VL NaViT weights from a HuggingFace safetensors
    /// file. The published release puts the vision encoder under
    /// `visual.vision_model.*` and the projector under `mlp_AR.*` at
    /// the top level; `text_hidden_size` is the language model's
    /// hidden size (projector `linear_2` projects into it).
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &PaddleOcrVlNaVitConfig,
        text_hidden_size: usize,
    ) -> Result<Self> {
        load_paddleocr_vl_navit_weights(
            st,
            cfg,
            text_hidden_size,
            "visual.vision_model.",
            "mlp_AR.",
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| next()).collect::<Vec<_>>())
    }

    fn tiny_cfg() -> PaddleOcrVlVisionConfig {
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

    fn tiny_weights(cfg: &PaddleOcrVlVisionConfig, text_hidden: usize) -> PaddleOcrVlVisionWeights {
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

    fn tiny_pixels(cfg: &PaddleOcrVlVisionConfig, num_tiles: usize) -> LazyTensor {
        let n_pix = num_tiles * cfg.num_channels * cfg.image_size * cfg.image_size;
        let data: Vec<f32> = (0..n_pix).map(|i| (i as f32 / n_pix as f32)).collect();
        LazyTensor::from_f32(
            Arc::from(data),
            Shape::from_dims(&[num_tiles, cfg.num_channels, cfg.image_size, cfg.image_size]),
            &Device::cpu(),
        )
    }

    #[test]
    fn forward_shape_single_tile() {
        let cfg = tiny_cfg();
        let text_hidden = 24;
        let weights = tiny_weights(&cfg, text_hidden);
        let model = PaddleOcrVlVisionModel {
            config: cfg.clone(),
            text_hidden_size: text_hidden,
            weights,
        };
        let pixels = tiny_pixels(&cfg, 1);
        let out = model.forward(&pixels, (1, 1)).unwrap();
        let merge = cfg.spatial_merge_size;
        let expected = cfg.num_patches_per_tile() / (merge * merge);
        assert_eq!(out.shape().dims(), &[expected, text_hidden]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite vision feature {v}");
        }
    }

    #[test]
    fn forward_shape_multi_tile() {
        let cfg = tiny_cfg();
        let text_hidden = 24;
        let weights = tiny_weights(&cfg, text_hidden);
        let model = PaddleOcrVlVisionModel {
            config: cfg.clone(),
            text_hidden_size: text_hidden,
            weights,
        };
        let grid = (2_usize, 2_usize);
        let num_tiles = grid.0 * grid.1;
        let pixels = tiny_pixels(&cfg, num_tiles);
        let out = model.forward(&pixels, grid).unwrap();
        let merge = cfg.spatial_merge_size;
        let per_tile_merged = cfg.num_patches_per_tile() / (merge * merge);
        let expected = num_tiles * per_tile_merged;
        assert_eq!(out.shape().dims(), &[expected, text_hidden]);
        for &v in &out.realize_f32() {
            assert!(v.is_finite(), "non-finite vision feature {v}");
        }
    }

    /// Hand-trace `aspect_ratio_chooser` for a few representative inputs.
    #[test]
    fn aspect_ratio_chooser_branches() {
        // Square -> 1x1.
        assert_eq!(aspect_ratio_chooser(384, 384, 4), (1, 1));
        // Near-square (ratio ~1.1) -> 1x1.
        assert_eq!(aspect_ratio_chooser(384, 400, 4), (1, 1));
        // 2:1 landscape -> 1x2.
        assert_eq!(aspect_ratio_chooser(384, 768, 4), (1, 2));
        // 1:2 portrait -> 2x1.
        assert_eq!(aspect_ratio_chooser(768, 384, 4), (2, 1));
        // Capped landscape.
        assert_eq!(aspect_ratio_chooser(100, 1000, 4), (1, 4));
        // Degenerate zero -> 1x1.
        assert_eq!(aspect_ratio_chooser(0, 384, 4), (1, 1));
    }

    /// Hand-trace `partition_image` matches what a manual crop would
    /// produce. Use a tiny channels=1 image so the byte layout is
    /// inspectable.
    #[test]
    fn tile_partition_host_helpers_match_eager_output() {
        // 1 channel, 4x4 image, values 0..16.
        let img: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let tiles = partition_image(&img, 1, 4, 4, 2, 2);
        assert_eq!(tiles.len(), 4);
        // Tile 0 (rows 0..2, cols 0..2): rows of the original image at
        //   row 0 cols 0..2 = [0, 1]
        //   row 1 cols 0..2 = [4, 5]
        assert_eq!(tiles[0], vec![0.0, 1.0, 4.0, 5.0]);
        // Tile 1 (rows 0..2, cols 2..4):
        //   row 0 cols 2..4 = [2, 3]
        //   row 1 cols 2..4 = [6, 7]
        assert_eq!(tiles[1], vec![2.0, 3.0, 6.0, 7.0]);
        // Tile 2 (rows 2..4, cols 0..2):
        //   row 2 cols 0..2 = [8, 9]
        //   row 3 cols 0..2 = [12, 13]
        assert_eq!(tiles[2], vec![8.0, 9.0, 12.0, 13.0]);
        // Tile 3 (rows 2..4, cols 2..4):
        //   row 2 cols 2..4 = [10, 11]
        //   row 3 cols 2..4 = [14, 15]
        assert_eq!(tiles[3], vec![10.0, 11.0, 14.0, 15.0]);

        // 2 channels, 2x4, values laid out per channel:
        //   ch0: [0..8]
        //   ch1: [8..16]
        // Split into 1x2 tiles (each 2x2). Tile 0 keeps the left half
        // of both channels, tile 1 keeps the right half.
        let mut img2 = Vec::with_capacity(16);
        for v in 0..8 { img2.push(v as f32); }
        for v in 8..16 { img2.push(v as f32); }
        let tiles2 = partition_image(&img2, 2, 2, 4, 1, 2);
        assert_eq!(tiles2.len(), 2);
        // Tile 0 channels concatenated:
        //   ch0 row0 cols 0..2 = [0, 1]
        //   ch0 row1 cols 0..2 = [4, 5]
        //   ch1 row0 cols 0..2 = [8, 9]
        //   ch1 row1 cols 0..2 = [12, 13]
        assert_eq!(tiles2[0], vec![0.0, 1.0, 4.0, 5.0, 8.0, 9.0, 12.0, 13.0]);
        // Tile 1:
        //   ch0 row0 cols 2..4 = [2, 3]
        //   ch0 row1 cols 2..4 = [6, 7]
        //   ch1 row0 cols 2..4 = [10, 11]
        //   ch1 row1 cols 2..4 = [14, 15]
        assert_eq!(tiles2[1], vec![2.0, 3.0, 6.0, 7.0, 10.0, 11.0, 14.0, 15.0]);
    }

    /// RoPE table position (0, 0) is the identity: cos == 1, sin == 0
    /// across all features (theta == 0 reduces to that).
    #[test]
    fn rope_tables_position_zero_is_identity() {
        let (cos, sin) = build_2d_rope_tables(10_000.0, 8, 4);
        for i in 0..8 {
            assert!((cos[i] - 1.0).abs() < 1e-6, "cos[0, {i}] = {} != 1", cos[i]);
            assert!(sin[i].abs() < 1e-6, "sin[0, {i}] = {} != 0", sin[i]);
        }
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
                "lazy_paddleocr_vl_vision_load_{}_{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            ));
            std::fs::write(&path, bytes).expect("write tempfile");
            path
        }

        fn build_tiny_safetensors(
            cfg: &PaddleOcrVlVisionConfig,
            text_hidden: usize,
        ) -> std::path::PathBuf {
            let mut map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
            let mut s: u32 = 5252;
            let mut nxt = || -> f32 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.01
            };
            let mut vec_n = |n: usize| -> Vec<f32> { (0..n).map(|_| nxt()).collect() };

            let h = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let np = cfg.num_patches_per_tile();
            let m = cfg.spatial_merge_size;
            let merged_hidden = h * m * m;

            let vp = "visual.vision_model.";
            put(&mut map, &format!("{vp}embeddings.patch_embedding.weight"),
                &[h, cfg.num_channels, cfg.patch_size, cfg.patch_size],
                &vec_n(h * cfg.num_channels * cfg.patch_size * cfg.patch_size));
            put(&mut map, &format!("{vp}embeddings.patch_embedding.bias"),
                &[h], &vec_n(h));
            put(&mut map, &format!("{vp}embeddings.position_embedding.weight"),
                &[np, h], &vec_n(np * h));
            put(&mut map, &format!("{vp}post_layernorm.weight"), &[h], &vec_n(h));
            put(&mut map, &format!("{vp}post_layernorm.bias"), &[h], &vec_n(h));
            for i in 0..cfg.num_hidden_layers {
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
                put(&mut map, &format!("{p}.mlp.fc1.bias"),
                    &[inter], &vec_n(inter));
                put(&mut map, &format!("{p}.mlp.fc2.weight"),
                    &[h, inter], &vec_n(h * inter));
                put(&mut map, &format!("{p}.mlp.fc2.bias"),
                    &[h], &vec_n(h));
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
            serialize_to_tempfile(&map)
        }

        #[test]
        fn round_trip_synthetic_safetensors() {
            let cfg = tiny_cfg();
            let text_hidden = 16;
            let path = build_tiny_safetensors(&cfg, text_hidden);
            let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path) }
                .expect("mmap safetensors");
            let weights = PaddleOcrVlVisionWeights::load_from_mmapped(
                &st, &cfg, text_hidden,
            ).expect("PaddleOcrVlVisionWeights::load_from_mmapped");
            assert_eq!(weights.blocks.len(), cfg.num_hidden_layers);
            assert_eq!(weights.patch_proj.len(),
                cfg.hidden_size * cfg.num_channels * cfg.patch_size * cfg.patch_size);
            assert_eq!(weights.position_embedding.len(),
                cfg.num_patches_per_tile() * cfg.hidden_size);

            let model = PaddleOcrVlVisionModel {
                config: cfg.clone(),
                text_hidden_size: text_hidden,
                weights,
            };
            let pixels = tiny_pixels(&cfg, 1);
            let out = model.forward(&pixels, (1, 1)).unwrap();
            let merge = cfg.spatial_merge_size;
            let expected = cfg.num_patches_per_tile() / (merge * merge);
            assert_eq!(out.shape().dims(), &[expected, text_hidden]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite vision feature");
            }
            let _ = std::fs::remove_file(&path);
        }

        #[test]
        #[ignore]
        fn from_hub_smoke_paddleocr_vl_vision() {
            // Canonical: PaddlePaddle/PaddleOCR-VL — vision encoder
            // lives under `visual.vision_model.*`, projector under
            // `mlp_AR.*` at the top level of the checkpoint.
        }
    }

    // -----------------------------------------------------------------------
    // NaViT (dynamic-resolution) tests
    // -----------------------------------------------------------------------

    mod navit {
        use super::*;

        fn navit_cfg() -> PaddleOcrVlNaVitConfig {
            // base_image_size / patch_size = 8 / 4 = 2 → base grid 2×2.
            // hidden_size 16, 2 layers, 4 heads, spatial_merge=2 so the
            // patch grid is always a multiple of 2.
            PaddleOcrVlNaVitConfig {
                hidden_size: 16,
                intermediate_size: 32,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                num_channels: 3,
                base_image_size: 8,
                patch_size: 4,
                hidden_activation: PaddleOcrVlVisionActivation::GeluPytorchTanh,
                layer_norm_eps: 1e-6,
                spatial_merge_size: 2,
                rope_theta: 10_000.0,
                max_pixels: 4 * 4 * 32 * 32,
            }
        }

        fn navit_weights(
            cfg: &PaddleOcrVlNaVitConfig,
            text_hidden: usize,
        ) -> PaddleOcrVlNaVitWeights {
            let mut s: u32 = 909090;
            let next = move || -> f32 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
            };
            let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);
            let h = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let p = cfg.patch_size;
            let c = cfg.num_channels;
            let np = cfg.base_num_positions();
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
                linear_1: WeightStorage::F32(vec_of(
                    merged_hidden * merged_hidden,
                    &mut *nb,
                )),
                linear_1_bias: vec_of(merged_hidden, &mut *nb),
                linear_2: WeightStorage::F32(vec_of(
                    merged_hidden * text_hidden,
                    &mut *nb,
                )),
                linear_2_bias: vec_of(text_hidden, &mut *nb),
            };

            PaddleOcrVlNaVitWeights {
                patch_proj,
                patch_proj_bias,
                position_embedding,
                blocks,
                post_ln_gain,
                post_ln_bias,
                projector,
            }
        }

        fn navit_pixels(
            cfg: &PaddleOcrVlNaVitConfig,
            h: usize,
            w: usize,
        ) -> LazyTensor {
            let n_pix = cfg.num_channels * h * w;
            let data: Vec<f32> = (0..n_pix).map(|i| i as f32 / n_pix as f32).collect();
            LazyTensor::from_f32(
                Arc::from(data),
                Shape::from_dims(&[1, cfg.num_channels, h, w]),
                &Device::cpu(),
            )
        }

        fn navit_model(text_hidden: usize) -> PaddleOcrVlNaVitModel {
            let cfg = navit_cfg();
            let weights = navit_weights(&cfg, text_hidden);
            PaddleOcrVlNaVitModel::new(cfg, text_hidden, weights)
        }

        /// (1) Forward at the base grid (8×8 = 2×2 patches) returns the
        /// expected `(merged_patches, text_hidden)` shape and all finite.
        #[test]
        fn forward_shape_base_grid() {
            let text_hidden = 24;
            let model = navit_model(text_hidden);
            let cfg = model.config.clone();
            let pixels = navit_pixels(&cfg, cfg.base_image_size, cfg.base_image_size);
            let out = model.forward(&pixels).expect("NaVit forward");
            let merge = cfg.spatial_merge_size;
            let base = cfg.base_grid_size();
            let expected = (base * base) / (merge * merge);
            assert_eq!(out.shape().dims(), &[expected, text_hidden]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite NaVit feature {v}");
            }
        }

        /// (2) Forward at a different (larger, non-square) grid scales
        /// the output sequence dim correctly.
        #[test]
        fn forward_shape_larger_grid() {
            let text_hidden = 24;
            let model = navit_model(text_hidden);
            let cfg = model.config.clone();
            // 4×6 patches (16×24 pixels at patch_size=4). Both axes are
            // multiples of spatial_merge_size=2.
            let h_patches = 4_usize;
            let w_patches = 6_usize;
            let pixels = navit_pixels(
                &cfg,
                h_patches * cfg.patch_size,
                w_patches * cfg.patch_size,
            );
            let out = model.forward(&pixels).expect("NaVit forward");
            let merge = cfg.spatial_merge_size;
            let expected = (h_patches * w_patches) / (merge * merge);
            assert_eq!(out.shape().dims(), &[expected, text_hidden]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite NaVit feature {v}");
            }
            assert_ne!(
                (cfg.base_grid_size(), cfg.base_grid_size()),
                (h_patches, w_patches),
                "grid must differ from base for an interpolation exercise",
            );
        }

        /// (3) Bilinear position-embedding interpolation matches a hand-
        /// computed result. Base 2×2 grid of hidden=4, target 3×3 grid:
        /// every interior point is a weighted sum of the 4 corners, with
        /// align_corners=False scale factors.
        #[test]
        fn position_embedding_interpolation_matches_hand_computed() {
            // Base 2×2 × hidden 4. Each row of the grid is filled with a
            // distinct ramp so any bilinear weight is observable.
            let base_h = 2_usize;
            let base_w = 2_usize;
            let hidden = 4_usize;
            // Row-major (row, col, channel):
            //   (0,0): [0,1,2,3]    (0,1): [10,11,12,13]
            //   (1,0): [100,101,102,103] (1,1): [110,111,112,113]
            let src: Vec<f32> = vec![
                0.0, 1.0, 2.0, 3.0,
                10.0, 11.0, 12.0, 13.0,
                100.0, 101.0, 102.0, 103.0,
                110.0, 111.0, 112.0, 113.0,
            ];

            let target_h = 3_usize;
            let target_w = 3_usize;
            let out = bilinear_interpolate_position_embedding(
                &src, base_h, base_w, hidden, target_h, target_w,
            );
            assert_eq!(out.len(), target_h * target_w * hidden);

            // Hand-compute the expected output via the same align_corners
            // =False stencil, then compare element-wise to within 1e-6.
            let scale_h = base_h as f64 / target_h as f64;
            let scale_w = base_w as f64 / target_w as f64;
            for ty in 0..target_h {
                let sy = ((ty as f64 + 0.5) * scale_h - 0.5)
                    .max(0.0)
                    .min((base_h - 1) as f64);
                let sy0 = sy.floor() as usize;
                let sy1 = (sy0 + 1).min(base_h - 1);
                let fy = (sy - sy0 as f64) as f32;
                for tx in 0..target_w {
                    let sx = ((tx as f64 + 0.5) * scale_w - 0.5)
                        .max(0.0)
                        .min((base_w - 1) as f64);
                    let sx0 = sx.floor() as usize;
                    let sx1 = (sx0 + 1).min(base_w - 1);
                    let fx = (sx - sx0 as f64) as f32;
                    let w00 = (1.0 - fy) * (1.0 - fx);
                    let w01 = (1.0 - fy) * fx;
                    let w10 = fy * (1.0 - fx);
                    let w11 = fy * fx;
                    for d in 0..hidden {
                        let e00 = src[(sy0 * base_w + sx0) * hidden + d];
                        let e01 = src[(sy0 * base_w + sx1) * hidden + d];
                        let e10 = src[(sy1 * base_w + sx0) * hidden + d];
                        let e11 = src[(sy1 * base_w + sx1) * hidden + d];
                        let expected = w00 * e00 + w01 * e01 + w10 * e10 + w11 * e11;
                        let got = out[(ty * target_w + tx) * hidden + d];
                        assert!(
                            (got - expected).abs() < 1e-6,
                            "interp mismatch at (ty={ty},tx={tx},d={d}): got {got}, expected {expected}",
                        );
                    }
                }
            }

            // Fast path: target == base must round-trip the source bytes.
            let identity =
                bilinear_interpolate_position_embedding(&src, base_h, base_w, hidden, 2, 2);
            assert_eq!(identity, src);
        }

        /// LFU cache hits — calling `forward` twice with the same grid
        /// fills the cache exactly once. A second `forward` with a
        /// different grid grows the cache.
        #[test]
        fn position_embedding_cache_hits() {
            let text_hidden = 16;
            let model = navit_model(text_hidden);
            let cfg = model.config.clone();
            assert_eq!(model.pos_embed_cache_len(), 0);

            let p = cfg.patch_size;
            let pixels_a = navit_pixels(&cfg, 4 * p, 4 * p);
            let _ = model.forward(&pixels_a).expect("forward grid A");
            assert_eq!(model.pos_embed_cache_len(), 1);
            // Same grid → still one entry.
            let _ = model.forward(&pixels_a).expect("forward grid A again");
            assert_eq!(model.pos_embed_cache_len(), 1);
            // New grid → cache grows.
            let pixels_b = navit_pixels(&cfg, 4 * p, 6 * p);
            let _ = model.forward(&pixels_b).expect("forward grid B");
            assert_eq!(model.pos_embed_cache_len(), 2);
        }

        /// (4) Misaligned input panics with a clear message naming
        /// `patch_size`. Both axes are checked; this test catches the
        /// `H % patch_size != 0` branch.
        #[test]
        #[should_panic(expected = "H (13) is not a multiple of patch_size (4)")]
        fn panic_on_misaligned_h() {
            let text_hidden = 16;
            let model = navit_model(text_hidden);
            let cfg = model.config.clone();
            // H=13 is not a multiple of patch_size=4. Build a valid
            // tensor of that shape via from_f32 and watch forward panic.
            let n_pix = cfg.num_channels * 13 * 12;
            let data: Vec<f32> = (0..n_pix).map(|i| i as f32).collect();
            let pixels = LazyTensor::from_f32(
                Arc::from(data),
                Shape::from_dims(&[1, cfg.num_channels, 13, 12]),
                &Device::cpu(),
            );
            let _ = model.forward(&pixels);
        }

        /// Sibling of (4) — the W-misalignment branch fires with the
        /// right message.
        #[test]
        #[should_panic(expected = "W (13) is not a multiple of patch_size (4)")]
        fn panic_on_misaligned_w() {
            let text_hidden = 16;
            let model = navit_model(text_hidden);
            let cfg = model.config.clone();
            let n_pix = cfg.num_channels * 12 * 13;
            let data: Vec<f32> = (0..n_pix).map(|i| i as f32).collect();
            let pixels = LazyTensor::from_f32(
                Arc::from(data),
                Shape::from_dims(&[1, cfg.num_channels, 12, 13]),
                &Device::cpu(),
            );
            let _ = model.forward(&pixels);
        }

        /// `build_2d_rope_tables_hw` reduces to the existing square
        /// helper when `h == w` — guards future refactors of the
        /// fixed-tile path.
        #[test]
        fn rope_tables_hw_matches_square_helper() {
            let theta = 10_000.0_f64;
            let head_dim = 8;
            let side = 4_usize;
            let (cos_a, sin_a) = build_2d_rope_tables(theta, head_dim, side);
            let (cos_b, sin_b) = build_2d_rope_tables_hw(theta, head_dim, side, side);
            assert_eq!(cos_a, cos_b);
            assert_eq!(sin_a, sin_b);
        }

        /// NaViT loader round-trips a synthetic safetensors blob — same
        /// fixture pattern as the fixed-tile loader test, with the base
        /// `(base_num_positions, hidden)` position-embedding shape.
        #[test]
        fn load_from_mmapped_round_trip() {
            use safetensors::tensor::TensorView;
            use safetensors::Dtype;
            use std::collections::HashMap;

            let cfg = navit_cfg();
            let text_hidden = 16_usize;

            let mut s: u32 = 7373;
            let mut nxt = || -> f32 {
                s = s.wrapping_mul(1103515245).wrapping_add(12345);
                ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.01
            };
            let mut vec_n = |n: usize| -> Vec<f32> { (0..n).map(|_| nxt()).collect() };

            let h = cfg.hidden_size;
            let inter = cfg.intermediate_size;
            let np = cfg.base_num_positions();
            let m = cfg.spatial_merge_size;
            let merged_hidden = h * m * m;

            let mut map: HashMap<String, (Dtype, Vec<usize>, Vec<u8>)> = HashMap::new();
            let put = |map: &mut HashMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
                       name: &str,
                       shape: &[usize],
                       data: &[f32]| {
                let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
                map.insert(name.to_string(), (Dtype::F32, shape.to_vec(), bytes));
            };
            let vp = "visual.vision_model.";
            put(&mut map, &format!("{vp}embeddings.patch_embedding.weight"),
                &[h, cfg.num_channels, cfg.patch_size, cfg.patch_size],
                &vec_n(h * cfg.num_channels * cfg.patch_size * cfg.patch_size));
            put(&mut map, &format!("{vp}embeddings.patch_embedding.bias"),
                &[h], &vec_n(h));
            put(&mut map, &format!("{vp}embeddings.position_embedding.weight"),
                &[np, h], &vec_n(np * h));
            put(&mut map, &format!("{vp}post_layernorm.weight"), &[h], &vec_n(h));
            put(&mut map, &format!("{vp}post_layernorm.bias"), &[h], &vec_n(h));
            for i in 0..cfg.num_hidden_layers {
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
                put(&mut map, &format!("{p}.mlp.fc1.bias"),
                    &[inter], &vec_n(inter));
                put(&mut map, &format!("{p}.mlp.fc2.weight"),
                    &[h, inter], &vec_n(h * inter));
                put(&mut map, &format!("{p}.mlp.fc2.bias"),
                    &[h], &vec_n(h));
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

            let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
            for (k, (dt, shape, data)) in &map {
                let v = TensorView::new(*dt, shape.clone(), data).expect("TensorView");
                views.insert(k.clone(), v);
            }
            let bytes = safetensors::serialize(&views, None).expect("serialize");
            let path = std::env::temp_dir().join(format!(
                "lazy_paddleocr_vl_navit_load_{}_{}.safetensors",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
            ));
            std::fs::write(&path, bytes).expect("write tempfile");

            let st = unsafe { crate::safetensors::MmapedSafetensors::new(&path) }
                .expect("mmap safetensors");
            let weights = PaddleOcrVlNaVitWeights::load_from_mmapped(
                &st, &cfg, text_hidden,
            )
            .expect("PaddleOcrVlNaVitWeights::load_from_mmapped");
            assert_eq!(weights.blocks.len(), cfg.num_hidden_layers);
            assert_eq!(
                weights.patch_proj.len(),
                cfg.hidden_size * cfg.num_channels * cfg.patch_size * cfg.patch_size,
            );
            assert_eq!(
                weights.position_embedding.len(),
                cfg.base_num_positions() * cfg.hidden_size,
            );

            let model = PaddleOcrVlNaVitModel::new(cfg.clone(), text_hidden, weights);
            let pixels = navit_pixels(&cfg, cfg.base_image_size, cfg.base_image_size);
            let out = model.forward(&pixels).expect("NaVit forward post-load");
            let merge = cfg.spatial_merge_size;
            let base = cfg.base_grid_size();
            let expected = (base * base) / (merge * merge);
            assert_eq!(out.shape().dims(), &[expected, text_hidden]);
            for &v in &out.realize_f32() {
                assert!(v.is_finite(), "non-finite NaVit feature");
            }
            let _ = std::fs::remove_file(&path);
        }
    }
}
