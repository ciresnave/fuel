//! Qwen3-VL vision tower (lazy-graph port).
//!
//! Sub-port 1 of `docs/session-prompts/port-qwen3-vl.md` — covers the
//! vision encoder only. The text decoder and the cross-modality
//! composition layer are separate sub-ports.
//!
//! ## Architecture
//!
//! 1. Conv3D temporal-patch-2 embedding (kernel = `(2, patch, patch)`,
//!    stride = `(2, patch, patch)`) — delegated to
//!    [`crate::lazy_conv3d::Conv3dTemporal2Weights`].
//! 2. Patch sequence `(N, hidden_size)` fed through a ViT-style
//!    transformer with pre-LayerNorm, multi-head self-attention, and
//!    a 2-layer MLP with GELU activation.
//! 3. `cu_seqlens` variable-length attention: patches belonging to the
//!    same "image" (a contiguous span between consecutive
//!    `cu_seqlens` boundaries) attend to one another only. The mask
//!    is materialized as a block-diagonal `(1, 1, N, N)` tensor of
//!    zeros inside each block and `-inf` across blocks, then
//!    broadcast-added to the pre-softmax attention scores.
//! 4. DeepStack residual injection — at each layer index listed in
//!    `deepstack_visual_indexes`, the post-block hidden states are
//!    projected through a learned `[hidden_size, out_hidden_size]`
//!    linear and collected. The caller (composition layer) is
//!    responsible for adding these into the text-side embeddings.
//!
//! ## Input contract
//!
//! `pixels` shape: `(N, in_channels, 2, patch_size, patch_size)` — the
//! caller has already extracted and flattened patches across all
//! images in the batch, matching the eager port's
//! `xs.reshape(((), C, T_p, H_p, W_p))` step. `cu_seqlens` is a
//! length-`(num_images + 1)` slice of cumulative patch counts; the
//! last entry must equal `N`.

use crate::lazy::{LazyTensor, WeightStorage};
use crate::lazy_conv3d::Conv3dTemporal2Weights;
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub layer_norm_eps: f64,
    /// Layer indices (0-based) at which a DeepStack projection is
    /// captured. Indices outside `[0, depth)` are ignored.
    pub deepstack_visual_indexes: Vec<usize>,
}

impl Qwen3VlVisionConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_heads
    }
}

#[derive(Debug, Clone)]
pub struct Qwen3VlVisionLayerWeights {
    pub norm1_gain: Arc<[f32]>,
    pub norm1_bias: Arc<[f32]>,
    pub norm2_gain: Arc<[f32]>,
    pub norm2_bias: Arc<[f32]>,
    /// `[hidden_size, 3 * hidden_size]` packed Q/K/V projection.
    pub qkv: WeightStorage,
    pub qkv_bias: Arc<[f32]>,
    pub proj: WeightStorage,
    pub proj_bias: Arc<[f32]>,
    pub fc1: WeightStorage,
    pub fc1_bias: Arc<[f32]>,
    pub fc2: WeightStorage,
    pub fc2_bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlVisionDeepStackProjection {
    /// Linear projection `[hidden_size, out_hidden_size]`.
    pub weight: WeightStorage,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlVisionWeights {
    pub patch_embed: Conv3dTemporal2Weights,
    /// `[hidden_size]` post-patch-embed bias (the eager port stores
    /// the Conv3d bias here — Conv3dTemporal2Weights itself does not
    /// model a bias, so we add it via `broadcast_add` after the conv).
    pub patch_embed_bias: Arc<[f32]>,
    pub layers: Vec<Qwen3VlVisionLayerWeights>,
    pub final_norm_gain: Arc<[f32]>,
    pub final_norm_bias: Arc<[f32]>,
    /// One projection per entry of `config.deepstack_visual_indexes`,
    /// in the same order.
    pub deepstack: Vec<Qwen3VlVisionDeepStackProjection>,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlVisionModel {
    pub config: Qwen3VlVisionConfig,
    pub weights: Qwen3VlVisionWeights,
}

#[derive(Debug, Clone)]
pub struct Qwen3VlVisionOutput {
    /// Final per-patch hidden states, shape `(N, hidden_size)`.
    pub embeddings: LazyTensor,
    /// Per-DeepStack projection, in `deepstack_visual_indexes` order.
    /// Each tensor has shape `(N, out_hidden_size)`.
    pub deepstack: Vec<LazyTensor>,
}

/// Build a block-diagonal additive attention mask of shape
/// `(1, 1, N, N)` where `mask[i, j] = 0` if patches `i` and `j` fall
/// inside the same `cu_seqlens` block and `-inf` otherwise.
pub fn build_cu_seqlens_mask(cu_seqlens: &[usize], n: usize) -> Result<Vec<f32>> {
    if cu_seqlens.len() < 2 {
        return Err(crate::Error::Msg(format!(
            "build_cu_seqlens_mask: cu_seqlens must contain at least \
             two entries (start and end), got {}",
            cu_seqlens.len(),
        ))
        .bt());
    }
    if cu_seqlens[0] != 0 {
        return Err(crate::Error::Msg(format!(
            "build_cu_seqlens_mask: cu_seqlens[0] must be 0, got {}",
            cu_seqlens[0],
        ))
        .bt());
    }
    if *cu_seqlens.last().unwrap() != n {
        return Err(crate::Error::Msg(format!(
            "build_cu_seqlens_mask: cu_seqlens.last ({}) must equal \
             total patch count N ({n})",
            cu_seqlens.last().unwrap(),
        ))
        .bt());
    }
    for w in cu_seqlens.windows(2) {
        if w[1] < w[0] {
            return Err(crate::Error::Msg(format!(
                "build_cu_seqlens_mask: cu_seqlens must be \
                 non-decreasing, got {:?}",
                cu_seqlens,
            ))
            .bt());
        }
    }
    let mut mask = vec![f32::NEG_INFINITY; n * n];
    for w in cu_seqlens.windows(2) {
        let start = w[0];
        let end = w[1];
        for i in start..end {
            for j in start..end {
                mask[i * n + j] = 0.0;
            }
        }
    }
    Ok(mask)
}

impl Qwen3VlVisionModel {
    /// Encode patches → `(embeddings, deepstack)`.
    pub fn forward(
        &self,
        pixels: &LazyTensor,
        cu_seqlens: &[usize],
    ) -> Result<Qwen3VlVisionOutput> {
        let cfg = &self.config;
        let weights = &self.weights;
        if cfg.hidden_size % cfg.num_heads != 0 {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlVisionModel: hidden_size ({}) must be divisible by num_heads ({})",
                cfg.hidden_size, cfg.num_heads,
            ))
            .bt());
        }
        if weights.layers.len() != cfg.depth {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlVisionModel: weights.layers ({}) must match config.depth ({})",
                weights.layers.len(),
                cfg.depth,
            ))
            .bt());
        }
        if weights.deepstack.len() != cfg.deepstack_visual_indexes.len() {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlVisionModel: weights.deepstack ({}) must match \
                 config.deepstack_visual_indexes ({})",
                weights.deepstack.len(),
                cfg.deepstack_visual_indexes.len(),
            ))
            .bt());
        }

        // ---- Conv3D patch embed → (N, hidden_size) -----------------
        let dims = pixels.shape();
        let dims = dims.dims();
        if dims.len() != 5 {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlVisionModel::forward: pixels must be rank 5 \
                 (N, C, T, H, W), got rank {}",
                dims.len(),
            ))
            .bt());
        }
        let n = dims[0];
        if dims[1] != cfg.in_channels {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlVisionModel::forward: pixels[1] ({}) must equal in_channels ({})",
                dims[1], cfg.in_channels,
            ))
            .bt());
        }
        if dims[2] != cfg.temporal_patch_size {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlVisionModel::forward: pixels[2] ({}) must equal temporal_patch_size ({})",
                dims[2], cfg.temporal_patch_size,
            ))
            .bt());
        }
        if dims[3] != cfg.patch_size || dims[4] != cfg.patch_size {
            return Err(crate::Error::Msg(format!(
                "Qwen3VlVisionModel::forward: pixels[3,4] ({},{}) must equal patch_size ({})",
                dims[3], dims[4], cfg.patch_size,
            ))
            .bt());
        }

        // (N, hidden, 1, 1, 1) from Conv3dTemporal2.apply, then reshape.
        let conv_out = weights.patch_embed.apply(pixels)?;
        let post_conv = conv_out.reshape(Shape::from_dims(&[n, cfg.hidden_size]))?;
        let bias = pixels.const_f32_like(
            Arc::clone(&weights.patch_embed_bias),
            Shape::from_dims(&[cfg.hidden_size]),
        );
        let mut hidden = post_conv.broadcast_add(&bias)?;

        // ---- Block-diagonal cu_seqlens mask, materialized once -----
        let mask_data = build_cu_seqlens_mask(cu_seqlens, n)?;
        let mask = pixels.const_f32_like(mask_data, Shape::from_dims(&[1, 1, n, n]));

        // ---- Transformer blocks + DeepStack capture ----------------
        let mut deepstack_outputs: Vec<LazyTensor> = Vec::new();
        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            hidden = self.apply_layer(&hidden, layer, &mask, n)?;
            for (ds_idx, &capture_at) in cfg.deepstack_visual_indexes.iter().enumerate() {
                if capture_at == layer_idx && capture_at < cfg.depth {
                    let projector = &weights.deepstack[ds_idx];
                    let projected = projector
                        .weight
                        .apply_linear(&hidden, cfg.hidden_size, cfg.out_hidden_size);
                    let bias_t = hidden.const_f32_like(
                        Arc::clone(&projector.bias),
                        Shape::from_dims(&[cfg.out_hidden_size]),
                    );
                    let with_bias = projected.broadcast_add(&bias_t)?;
                    deepstack_outputs.push(with_bias);
                }
            }
        }

        // ---- Final LayerNorm on (N, hidden) ------------------------
        let embeddings = hidden.layer_norm_affine(
            Arc::clone(&weights.final_norm_gain),
            Arc::clone(&weights.final_norm_bias),
            cfg.layer_norm_eps,
        )?;

        Ok(Qwen3VlVisionOutput {
            embeddings,
            deepstack: deepstack_outputs,
        })
    }

    fn apply_layer(
        &self,
        x: &LazyTensor,
        layer: &Qwen3VlVisionLayerWeights,
        mask: &LazyTensor,
        n: usize,
    ) -> Result<LazyTensor> {
        let cfg = &self.config;
        let h = cfg.hidden_size;
        let num_heads = cfg.num_heads;
        let head_dim = cfg.head_dim();

        // Pre-LN.
        let x_norm = x.layer_norm_affine(
            Arc::clone(&layer.norm1_gain),
            Arc::clone(&layer.norm1_bias),
            cfg.layer_norm_eps,
        )?;

        // Packed QKV: (N, hidden) → (N, 3 * hidden).
        let qkv = layer
            .qkv
            .apply_linear(&x_norm, h, 3 * h)
            .add_trailing_bias(Arc::clone(&layer.qkv_bias))?;
        // (N, 3, num_heads, head_dim) → (3, N, num_heads, head_dim).
        let qkv = qkv
            .reshape(Shape::from_dims(&[n, 3, num_heads, head_dim]))?
            .permute([1, 0, 2, 3_usize])?;
        let q = qkv.slice(0_usize, 0, 1)?.reshape(Shape::from_dims(&[n, num_heads, head_dim]))?;
        let k = qkv.slice(0_usize, 1, 1)?.reshape(Shape::from_dims(&[n, num_heads, head_dim]))?;
        let v = qkv.slice(0_usize, 2, 1)?.reshape(Shape::from_dims(&[n, num_heads, head_dim]))?;

        // (1, num_heads, N, head_dim) for matmul.
        let q = q.reshape(Shape::from_dims(&[1, n, num_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let k = k.reshape(Shape::from_dims(&[1, n, num_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;
        let v = v.reshape(Shape::from_dims(&[1, n, num_heads, head_dim]))?
            .permute([0, 2, 1, 3_usize])?;

        let scale = 1.0_f64 / (head_dim as f64).sqrt();
        let k_t = k.transpose_last_two()?;
        let scores = q.matmul(&k_t)?.mul_scalar(scale);
        let masked = scores.broadcast_add(mask)?;
        let probs = masked.softmax_last_dim()?;
        let ctx = probs.matmul(&v)?;
        // (1, num_heads, N, head_dim) → (1, N, num_heads * head_dim) → (N, hidden).
        let merged = ctx
            .permute([0, 2, 1, 3_usize])?
            .reshape(Shape::from_dims(&[n, h]))?;

        let attn_out = layer
            .proj
            .apply_linear(&merged, h, h)
            .add_trailing_bias(Arc::clone(&layer.proj_bias))?;
        let x_attn = x.add(&attn_out)?;

        // MLP block.
        let x_attn_norm = x_attn.layer_norm_affine(
            Arc::clone(&layer.norm2_gain),
            Arc::clone(&layer.norm2_bias),
            cfg.layer_norm_eps,
        )?;
        let fc1 = layer
            .fc1
            .apply_linear(&x_attn_norm, h, cfg.intermediate_size)
            .add_trailing_bias(Arc::clone(&layer.fc1_bias))?;
        let activated = fc1.gelu();
        let fc2 = layer
            .fc2
            .apply_linear(&activated, cfg.intermediate_size, h)
            .add_trailing_bias(Arc::clone(&layer.fc2_bias))?;
        x_attn.add(&fc2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn rng() -> impl FnMut() -> f32 {
        let mut s: u32 = 0xC0DE_F00D;
        move || {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    fn vec_of(n: usize, src: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| src()).collect::<Vec<_>>())
    }

    fn tiny_config(deepstack: Vec<usize>) -> Qwen3VlVisionConfig {
        Qwen3VlVisionConfig {
            depth: 2,
            hidden_size: 16,
            out_hidden_size: 16,
            intermediate_size: 32,
            num_heads: 4,
            in_channels: 3,
            patch_size: 14,
            temporal_patch_size: 2,
            layer_norm_eps: 1e-6,
            deepstack_visual_indexes: deepstack,
        }
    }

    fn tiny_weights(cfg: &Qwen3VlVisionConfig) -> Qwen3VlVisionWeights {
        use crate::lazy_conv3d::Conv3dTemporal2Config;
        let next = rng();
        let mut nb: Box<dyn FnMut() -> f32> = Box::new(next);

        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;

        // Conv3D weight: (out_channels=hidden, in_channels/groups=in_channels,
        //                  2, kH=patch, kW=patch).
        let conv_raw_len =
            cfg.hidden_size * cfg.in_channels * 2 * cfg.patch_size * cfg.patch_size;
        let conv_raw: Vec<f32> = (0..conv_raw_len).map(|_| (*nb)()).collect();
        let patch_embed = Conv3dTemporal2Weights::from_raw_weight(
            &conv_raw,
            cfg.hidden_size,
            cfg.in_channels,
            cfg.patch_size,
            cfg.patch_size,
            Conv3dTemporal2Config {
                stride: cfg.patch_size,
                ..Default::default()
            },
        )
        .unwrap();
        let patch_embed_bias = vec_of(h, &mut *nb);

        let layers: Vec<Qwen3VlVisionLayerWeights> = (0..cfg.depth)
            .map(|_| Qwen3VlVisionLayerWeights {
                norm1_gain: Arc::from(vec![1.0_f32; h]),
                norm1_bias: Arc::from(vec![0.0_f32; h]),
                norm2_gain: Arc::from(vec![1.0_f32; h]),
                norm2_bias: Arc::from(vec![0.0_f32; h]),
                qkv: WeightStorage::F32(vec_of(h * 3 * h, &mut *nb)),
                qkv_bias: vec_of(3 * h, &mut *nb),
                proj: WeightStorage::F32(vec_of(h * h, &mut *nb)),
                proj_bias: vec_of(h, &mut *nb),
                fc1: WeightStorage::F32(vec_of(h * inter, &mut *nb)),
                fc1_bias: vec_of(inter, &mut *nb),
                fc2: WeightStorage::F32(vec_of(inter * h, &mut *nb)),
                fc2_bias: vec_of(h, &mut *nb),
            })
            .collect();

        let deepstack: Vec<Qwen3VlVisionDeepStackProjection> = cfg
            .deepstack_visual_indexes
            .iter()
            .map(|_| Qwen3VlVisionDeepStackProjection {
                weight: WeightStorage::F32(vec_of(h * cfg.out_hidden_size, &mut *nb)),
                bias: vec_of(cfg.out_hidden_size, &mut *nb),
            })
            .collect();

        Qwen3VlVisionWeights {
            patch_embed,
            patch_embed_bias,
            layers,
            final_norm_gain: Arc::from(vec![1.0_f32; h]),
            final_norm_bias: Arc::from(vec![0.0_f32; h]),
            deepstack,
        }
    }

    /// Build a `(N, C, T, H, W)` flattened-patches tensor where the
    /// caller has already extracted patches across all images.
    /// `total_patches = num_images * (T_frames / temporal_patch_size) * (H/patch) * (W/patch)`.
    fn tiny_pixels(cfg: &Qwen3VlVisionConfig, n_patches: usize) -> LazyTensor {
        let numel = n_patches * cfg.in_channels * cfg.temporal_patch_size
            * cfg.patch_size * cfg.patch_size;
        let data: Vec<f32> = (0..numel).map(|i| (i as f32 / numel as f32) - 0.5).collect();
        LazyTensor::from_f32(
            Arc::from(data),
            Shape::from_dims(&[
                n_patches,
                cfg.in_channels,
                cfg.temporal_patch_size,
                cfg.patch_size,
                cfg.patch_size,
            ]),
            &Device::cpu(),
        )
    }

    /// T=4 frames, H=W=28, embed_dim=16, depth=2, num_heads=4, patch=(2,14,14).
    /// With patch=14 and H=W=28, spatial grid is 2×2 = 4 patches per
    /// frame-pair. With T=4 and temporal_patch=2, we have 2 frame-pairs.
    /// Total patches per single image = 2 * 4 = 8.
    #[test]
    fn forward_shape_and_finite_tiny() {
        let cfg = tiny_config(vec![]);
        let model = Qwen3VlVisionModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let n = 8_usize;
        let pixels = tiny_pixels(&cfg, n);
        let cu_seqlens = vec![0_usize, n];
        let out = model.forward(&pixels, &cu_seqlens).unwrap();
        assert_eq!(out.embeddings.shape().dims(), &[n, cfg.hidden_size]);
        for &v in &out.embeddings.realize_f32() {
            assert!(v.is_finite(), "non-finite embedding: {v}");
        }
        assert!(out.deepstack.is_empty());
    }

    #[test]
    fn cu_seqlens_mask_block_diagonal() {
        // cu_seqlens=[0, 4, 8], N=8 → two 4-patch blocks. Inside-block
        // entries are 0.0, cross-block entries are -inf.
        let n = 8_usize;
        let cu = vec![0_usize, 4, 8];
        let mask = build_cu_seqlens_mask(&cu, n).unwrap();
        for i in 0..n {
            for j in 0..n {
                let same_block = (i < 4 && j < 4) || (i >= 4 && j >= 4);
                let v = mask[i * n + j];
                if same_block {
                    assert_eq!(v, 0.0, "mask[{i},{j}] expected 0.0, got {v}");
                } else {
                    assert!(
                        v.is_infinite() && v < 0.0,
                        "mask[{i},{j}] expected -inf, got {v}",
                    );
                }
            }
        }
    }

    #[test]
    fn cu_seqlens_mask_rejects_bad_inputs() {
        assert!(build_cu_seqlens_mask(&[0], 0).is_err());
        // Doesn't start at 0.
        assert!(build_cu_seqlens_mask(&[1, 4, 8], 8).is_err());
        // Last entry doesn't equal N.
        assert!(build_cu_seqlens_mask(&[0, 4, 7], 8).is_err());
        // Non-monotone.
        assert!(build_cu_seqlens_mask(&[0, 5, 3, 8], 8).is_err());
    }

    /// DeepStack indices [0, 1] both inside `depth=2`: forward must
    /// emit two projection tensors, in deepstack_visual_indexes order,
    /// each shaped `(N, out_hidden_size)` and finite.
    #[test]
    fn deepstack_indexes_collect_correct_layers() {
        let cfg = tiny_config(vec![0, 1]);
        let model = Qwen3VlVisionModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let n = 8_usize;
        let pixels = tiny_pixels(&cfg, n);
        let cu_seqlens = vec![0_usize, n];
        let out = model.forward(&pixels, &cu_seqlens).unwrap();
        assert_eq!(out.deepstack.len(), 2);
        for ds in &out.deepstack {
            assert_eq!(ds.shape().dims(), &[n, cfg.out_hidden_size]);
            for &v in &ds.realize_f32() {
                assert!(v.is_finite(), "non-finite deepstack: {v}");
            }
        }
        // The two captures come from different layers and so must
        // differ numerically.
        let a = out.deepstack[0].realize_f32();
        let b = out.deepstack[1].realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        assert!(
            max_diff > 1e-7,
            "DeepStack captures at layers 0 and 1 must differ, max_diff = {max_diff}",
        );
    }

    /// Two-image packed batch: cu_seqlens=[0, 4, 8] with N=8. The
    /// block-diagonal mask must prevent any cross-image attention —
    /// verify by comparing logits when the second image's pixels
    /// change: the first image's output must stay identical.
    #[test]
    fn cu_seqlens_isolates_blocks_in_forward() {
        let cfg = tiny_config(vec![]);
        let model = Qwen3VlVisionModel {
            config: cfg.clone(),
            weights: tiny_weights(&cfg),
        };
        let n = 8_usize;
        let cu_seqlens = vec![0_usize, 4, 8];

        let mut pixels_a_data = Vec::<f32>::with_capacity(
            n * cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size,
        );
        let per_patch = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        for i in 0..(n * per_patch) {
            pixels_a_data.push((i as f32 / (n * per_patch) as f32) - 0.5);
        }
        let mut pixels_b_data = pixels_a_data.clone();
        // Perturb only the second image (patches 4..8).
        for k in (4 * per_patch)..(n * per_patch) {
            pixels_b_data[k] += 0.25;
        }
        let pixels_a = LazyTensor::from_f32(
            Arc::from(pixels_a_data),
            Shape::from_dims(&[
                n, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size,
            ]),
            &Device::cpu(),
        );
        let pixels_b = LazyTensor::from_f32(
            Arc::from(pixels_b_data),
            Shape::from_dims(&[
                n, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size,
            ]),
            &Device::cpu(),
        );

        let out_a = model.forward(&pixels_a, &cu_seqlens).unwrap();
        let out_b = model.forward(&pixels_b, &cu_seqlens).unwrap();
        let emb_a = out_a.embeddings.realize_f32();
        let emb_b = out_b.embeddings.realize_f32();

        // First image (patches 0..4) embeddings must match exactly —
        // no information from the perturbed second image can leak
        // across the block boundary.
        let first_block_len = 4 * cfg.hidden_size;
        let mut max_diff_first = 0.0_f32;
        for (x, y) in emb_a[..first_block_len].iter().zip(emb_b[..first_block_len].iter()) {
            max_diff_first = max_diff_first.max((x - y).abs());
        }
        assert!(
            max_diff_first < 1e-5,
            "first image must be unaffected by second image's pixels; max_diff = {max_diff_first}",
        );

        // Second image (patches 4..8) must differ between the two
        // perturbation runs — sanity check that the model itself is
        // sensitive to its input.
        let mut max_diff_second = 0.0_f32;
        for (x, y) in emb_a[first_block_len..].iter().zip(emb_b[first_block_len..].iter()) {
            max_diff_second = max_diff_second.max((x - y).abs());
        }
        assert!(
            max_diff_second > 1e-6,
            "second image embeddings must respond to perturbed pixels; max_diff = {max_diff_second}",
        );
    }
}
