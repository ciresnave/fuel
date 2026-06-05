//! Hiera — lazy port (hierarchical ViT without bells-and-whistles).
//!
//! Reference: timm `hiera.py`.
//!
//! Pipeline:
//!   image (B, 3, 224, 224)
//!     → patch-embed 7×7 stride 4 conv → (B, C, 56, 56) → flatten → (B, 3136, C)
//!     → add 1D pos-embed (1, 3136, C)
//!     → **unroll**: a fixed token permutation that nests three 2×2 splits;
//!       output is still (B, 3136, C) but tokens are reordered so q-stride
//!       windows become contiguous
//!     → 4 stages of `MaskUnitAttention` blocks, with channel doubling +
//!       q-stride 4 (token-count quartering) + window-size /= 4 at each
//!       stage boundary; stages 0-1 use windowed (mask) attention, stages
//!       2-3 use global attention
//!     → mean over tokens → LN → linear head
//!
//! v1 scope: F32, batch == 1, forward-only inference, fixed
//! 224×224 input (the unroll math assumes a 56×56 grid).

use crate::lazy::{LazyTensor, WeightStorage};
use crate::Result;
use fuel_core_types::Shape;
use std::sync::Arc;

const TOKEN_GRID: usize = 56;
const NUM_TOKENS: usize = TOKEN_GRID * TOKEN_GRID;

#[derive(Debug, Clone, PartialEq)]
pub struct HieraConfig {
    pub channels: usize,
    pub heads: usize,
    pub stages: [usize; 4],
    pub num_classes: Option<usize>,
}

impl HieraConfig {
    pub fn tiny() -> Self {
        Self { channels: 96, heads: 1, stages: [1, 2, 7, 2], num_classes: Some(1000) }
    }
    pub fn small() -> Self {
        Self { channels: 96, heads: 1, stages: [1, 2, 11, 2], num_classes: Some(1000) }
    }
    pub fn base() -> Self {
        Self { channels: 96, heads: 1, stages: [2, 3, 16, 3], num_classes: Some(1000) }
    }
    pub fn base_plus() -> Self {
        Self { channels: 112, heads: 2, stages: [2, 3, 16, 3], num_classes: Some(1000) }
    }
    pub fn large() -> Self {
        Self { channels: 144, heads: 2, stages: [2, 6, 36, 4], num_classes: Some(1000) }
    }
    pub fn huge() -> Self {
        Self { channels: 256, heads: 4, stages: [2, 6, 36, 4], num_classes: Some(1000) }
    }
}

// ---- Weight structures ------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LayerNormWeights {
    pub gain: Arc<[f32]>,
    pub bias: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct LinearWeights {
    pub w: WeightStorage,
    pub b: Arc<[f32]>,
    pub in_features: usize,
    pub out_features: usize,
}

#[derive(Debug, Clone)]
pub struct HieraBlockWeights {
    pub norm1: LayerNormWeights,
    pub norm2: LayerNormWeights,
    /// Channel-bump projection. Present iff `in_channels != out_channels`.
    pub proj: Option<LinearWeights>,
    pub qkv: LinearWeights,
    pub attn_proj: LinearWeights,
    pub mlp_fc1: LinearWeights,
    pub mlp_fc2: LinearWeights,
    /// Cached per-block params (derived from the stage table at build time).
    pub heads: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub q_stride: usize,
    pub window_size: usize,
    pub use_mask_attention: bool,
}

#[derive(Debug, Clone)]
pub struct HieraEmbeddingWeights {
    /// 7×7 stride-4 padding-3 conv: 3 → channels.
    pub conv_w: Arc<[f32]>,
    pub conv_b: Arc<[f32]>,
    /// Per-token learned embedding `(1, NUM_TOKENS, channels)`.
    pub pos_embed: Arc<[f32]>,
}

#[derive(Debug, Clone)]
pub struct HieraHeadWeights {
    pub norm: LayerNormWeights,
    pub fc: LinearWeights,
}

#[derive(Debug, Clone)]
pub struct HieraWeights {
    pub embed: HieraEmbeddingWeights,
    pub blocks: Vec<HieraBlockWeights>,
    pub head: Option<HieraHeadWeights>,
}

#[derive(Debug, Clone)]
pub struct HieraModel {
    pub config: HieraConfig,
    pub weights: HieraWeights,
}

// ---- Block plan ------------------------------------------------------------

/// Derive the per-block schedule from the stage table. Matches the eager
/// `hiera_blocks` loop exactly so weights can be enumerated in the same
/// order as the eager build.
pub fn block_schedule(cfg: &HieraConfig) -> Vec<HieraBlockSchedule> {
    let mut out = Vec::with_capacity(cfg.stages.iter().sum());
    let mut in_channels = cfg.channels;
    let mut out_channels = cfg.channels;
    let mut heads = cfg.heads;
    let mut q_stride = 1_usize;
    let mut window_size = 64_usize;

    for s in 0..4 {
        let use_mask_attention = s < 2;
        for _ in 0..cfg.stages[s] {
            out.push(HieraBlockSchedule {
                heads,
                in_channels,
                out_channels,
                q_stride,
                window_size,
                use_mask_attention,
            });
            in_channels = out_channels;
            q_stride = 1;
        }
        q_stride = 4;
        out_channels *= 2;
        heads *= 2;
        window_size /= 4;
    }
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HieraBlockSchedule {
    pub heads: usize,
    pub in_channels: usize,
    pub out_channels: usize,
    pub q_stride: usize,
    pub window_size: usize,
    pub use_mask_attention: bool,
}

// ---- Forward ---------------------------------------------------------------

impl HieraModel {
    /// Forward pass returning class logits (with head) or pooled features
    /// (without head).
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4);
        assert_eq!(dims[1], 3);
        assert_eq!(dims[2], 224);
        assert_eq!(dims[3], 224);
        let b = dims[0];
        let c = self.config.channels;

        // Patch embed.
        let w = image.const_f32_like(
            Arc::clone(&self.weights.embed.conv_w),
            Shape::from_dims(&[c, 3, 7, 7]),
        );
        let bias = image.const_f32_like(
            Arc::clone(&self.weights.embed.conv_b), Shape::from_dims(&[c]),
        );
        let x = image.conv2d(&w, Some(&bias), (4, 4), (3, 3), 1)?;
        // (B, C, 56, 56) → (B, C, 3136) → (B, 3136, C)
        let x = x.reshape(Shape::from_dims(&[b, c, NUM_TOKENS]))?
            .permute([0, 2, 1_usize])?;
        // Add pos_embed (broadcast 1 along the batch dim).
        let pos = image.const_f32_like(
            Arc::clone(&self.weights.embed.pos_embed),
            Shape::from_dims(&[1, NUM_TOKENS, c]),
        );
        let pos_b = pos.broadcast_to(Shape::from_dims(&[b, NUM_TOKENS, c]))?;
        let x = x.add(&pos_b)?;

        // Unroll: three nested 2×2 splits that reorder tokens so q-stride
        // windows become contiguous in the token dim.
        let mut x = unroll(&x, b, c)?;

        // Run blocks.
        for blk in &self.weights.blocks {
            x = apply_block(&x, blk, image)?;
        }

        // Mean over tokens, then optional head.
        let pooled = x.mean_dim(1_usize)?;
        match &self.weights.head {
            None => Ok(pooled),
            Some(head) => {
                let h = apply_layer_norm_last(&pooled, &head.norm, head.fc.in_features)?;
                head.fc.w.apply_linear_with_bias(&h, head.fc.in_features, head.fc.out_features, Arc::clone(&head.fc.b))
            }
        }
    }
}

fn unroll(x: &LazyTensor, b: usize, c: usize) -> Result<LazyTensor> {
    // (B, 3136, C) → (B, 56, 56, C)
    let mut xs = x.reshape(Shape::from_dims(&[b, TOKEN_GRID, TOKEN_GRID, c]))?;
    let mut b_cur = b;
    let mut size = TOKEN_GRID;
    for _ in 0..3 {
        size /= 2;
        xs = xs
            .reshape(Shape::from_dims(&[b_cur, size, 2, size, 2, c]))?
            .permute([0, 2, 4, 1, 3, 5_usize])?
            .reshape(Shape::from_dims(&[b_cur * 4, size, size, c]))?;
        b_cur *= 4;
    }
    // (64B, 7, 7, C) → (B, 3136, C)
    Ok(xs.reshape(Shape::from_dims(&[b, NUM_TOKENS, c]))?)
}

fn apply_block(
    x: &LazyTensor, blk: &HieraBlockWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let n_in = dims[1]; let c_in = dims[2];
    debug_assert_eq!(c_in, blk.in_channels);
    let c_out = blk.out_channels;

    // norm1, then either project + q-stride-pool the residual or carry x.
    let xs_norm = apply_layer_norm_last(x, &blk.norm1, c_in)?;
    let residual = match &blk.proj {
        None => x.clone(),
        Some(proj) => {
            let projected = proj.w.apply_linear_with_bias(&xs_norm, proj.in_features, proj.out_features, Arc::clone(&proj.b))?;
            // (B, N, C_out) → (B, q_stride=4, N/4, C_out) → max → (B, N/4, C_out)
            // The pool stride here is fixed at 4 in the eager port.
            let stride = 4;
            assert_eq!(n_in % stride, 0);
            let pooled = projected
                .reshape(Shape::from_dims(&[b, stride, n_in / stride, c_out]))?
                .max_dim(1_usize)?;
            pooled
        }
    };

    let attn_out = apply_attention(&xs_norm, blk, anchor)?;
    let after_attn = residual.add(&attn_out)?;

    let normed = apply_layer_norm_last(&after_attn, &blk.norm2, c_out)?;
    let mlp_out = {
        let h = blk.mlp_fc1.w.apply_linear_with_bias(&normed, blk.mlp_fc1.in_features, blk.mlp_fc1.out_features, Arc::clone(&blk.mlp_fc1.b))?.gelu();
        blk.mlp_fc2.w.apply_linear_with_bias(&h, blk.mlp_fc2.in_features, blk.mlp_fc2.out_features, Arc::clone(&blk.mlp_fc2.b))?
    };
    after_attn.add(&mlp_out)
}

fn apply_attention(
    x: &LazyTensor, blk: &HieraBlockWeights, anchor: &LazyTensor,
) -> Result<LazyTensor> {
    let dims = x.shape();
    let dims = dims.dims();
    let b = dims[0]; let n_in = dims[1];
    let c_out = blk.out_channels;
    let heads = blk.heads;
    let head_dim = c_out / heads;
    let scale = 1.0_f64 / (head_dim as f64).sqrt();

    let num_windows = if blk.use_mask_attention {
        n_in / (blk.q_stride * blk.window_size)
    } else {
        1
    };
    assert!(num_windows >= 1);
    let s_in = n_in / num_windows;

    // qkv: (B, N, 3*C_out)
    let qkv = blk.qkv.w.apply_linear_with_bias(x, blk.qkv.in_features, blk.qkv.out_features, Arc::clone(&blk.qkv.b))?;
    // Reshape: (B, s_in, num_windows, 3, heads, head_dim) → permute
    // (3, B, heads, num_windows, s_in, head_dim).
    let qkv = qkv
        .reshape(Shape::from_dims(&[b, s_in, num_windows, 3, heads, head_dim]))?
        .permute([3, 0, 4, 2, 1, 5_usize])?;
    let q = qkv.narrow(0_usize, 0, 1)?
        .reshape(Shape::from_dims(&[b, heads, num_windows, s_in, head_dim]))?;
    let k = qkv.narrow(0_usize, 1, 1)?
        .reshape(Shape::from_dims(&[b, heads, num_windows, s_in, head_dim]))?;
    let v = qkv.narrow(0_usize, 2, 1)?
        .reshape(Shape::from_dims(&[b, heads, num_windows, s_in, head_dim]))?;

    // Q-stride pooling on queries.
    let (q, s_q) = if blk.q_stride > 1 {
        let s_q = s_in / blk.q_stride;
        let q = q
            .reshape(Shape::from_dims(&[b, heads, num_windows, blk.q_stride, s_q, head_dim]))?
            .max_dim(3_usize)?;
        (q, s_q)
    } else {
        (q, s_in)
    };
    let q = q.mul_scalar(scale);

    // Collapse (B, heads, num_windows) → one batch dim for a plain 3D matmul.
    let batch = b * heads * num_windows;
    let q3 = q.reshape(Shape::from_dims(&[batch, s_q, head_dim]))?;
    let k3 = k.reshape(Shape::from_dims(&[batch, s_in, head_dim]))?;
    let v3 = v.reshape(Shape::from_dims(&[batch, s_in, head_dim]))?;
    let kt = k3.permute([0, 2, 1_usize])?;
    let scores = q3.matmul(&kt)?;
    let probs = scores.softmax_last_dim()?;
    let ctx = probs.matmul(&v3)?;
    // (batch, s_q, head_dim) → (B, heads, num_windows, s_q, head_dim)
    //   → permute back so token positions are contiguous in dim 1
    //   → (B, N_out, C_out)
    let ctx = ctx
        .reshape(Shape::from_dims(&[b, heads, num_windows, s_q, head_dim]))?
        // Match the eager `transpose(1, 3)` after `unsqueeze(0)` followed by
        // `reshape(b, -1, out_C)`: after permute, dim ordering is
        // (B, num_windows, s_q, heads, head_dim).
        .permute([0, 2, 3, 1, 4_usize])?
        .reshape(Shape::from_dims(&[b, num_windows * s_q, c_out]))?;
    blk.attn_proj.w.apply_linear_with_bias(&ctx, blk.attn_proj.in_features, blk.attn_proj.out_features, Arc::clone(&blk.attn_proj.b))
}

fn apply_layer_norm_last(
    x: &LazyTensor, ln: &LayerNormWeights, hidden: usize,
) -> Result<LazyTensor> {
    let _ = hidden;
    x.layer_norm_affine(Arc::clone(&ln.gain), Arc::clone(&ln.bias), 1e-6)
}


// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }
    fn vec_of(n: usize, nb: &mut dyn FnMut() -> f32) -> Arc<[f32]> {
        Arc::from((0..n).map(|_| nb()).collect::<Vec<_>>())
    }
    fn ws(n: usize, nb: &mut dyn FnMut() -> f32) -> WeightStorage {
        WeightStorage::F32(vec_of(n, nb))
    }

    fn ln_w(c: usize) -> LayerNormWeights {
        LayerNormWeights {
            gain: Arc::from(vec![1.0_f32; c]),
            bias: Arc::from(vec![0.0_f32; c]),
        }
    }

    fn linear_w(
        in_features: usize, out_features: usize, nb: &mut dyn FnMut() -> f32,
    ) -> LinearWeights {
        LinearWeights {
            w: ws(in_features * out_features, nb),
            b: vec_of(out_features, nb),
            in_features, out_features,
        }
    }

    fn build_block(
        sched: HieraBlockSchedule, nb: &mut dyn FnMut() -> f32,
    ) -> HieraBlockWeights {
        let proj = if sched.in_channels != sched.out_channels {
            Some(linear_w(sched.in_channels, sched.out_channels, nb))
        } else { None };
        HieraBlockWeights {
            norm1: ln_w(sched.in_channels),
            norm2: ln_w(sched.out_channels),
            proj,
            qkv: linear_w(sched.in_channels, sched.out_channels * 3, nb),
            attn_proj: linear_w(sched.out_channels, sched.out_channels, nb),
            mlp_fc1: linear_w(sched.out_channels, sched.out_channels * 4, nb),
            mlp_fc2: linear_w(sched.out_channels * 4, sched.out_channels, nb),
            heads: sched.heads,
            in_channels: sched.in_channels,
            out_channels: sched.out_channels,
            q_stride: sched.q_stride,
            window_size: sched.window_size,
            use_mask_attention: sched.use_mask_attention,
        }
    }

    fn tiny_weights(cfg: &HieraConfig) -> HieraWeights {
        let mut nb = rng_seed(2026);
        let c = cfg.channels;
        let embed = HieraEmbeddingWeights {
            conv_w: vec_of(c * 3 * 7 * 7, &mut nb),
            conv_b: vec_of(c, &mut nb),
            pos_embed: vec_of(NUM_TOKENS * c, &mut nb),
        };
        let blocks: Vec<HieraBlockWeights> = block_schedule(cfg)
            .into_iter()
            .map(|s| build_block(s, &mut nb))
            .collect();
        let head = cfg.num_classes.map(|n| HieraHeadWeights {
            norm: ln_w(c * 8),
            fc: linear_w(c * 8, n, &mut nb),
        });
        HieraWeights { embed, blocks, head }
    }

    #[test]
    fn block_schedule_matches_eager_stage_layout() {
        let cfg = HieraConfig::tiny();
        let sched = block_schedule(&cfg);
        assert_eq!(sched.len(), 1 + 2 + 7 + 2);
        // First block: in==out==96, q_stride=1, window=64, mask=true.
        assert_eq!(sched[0].in_channels, 96);
        assert_eq!(sched[0].out_channels, 96);
        assert_eq!(sched[0].q_stride, 1);
        assert_eq!(sched[0].window_size, 64);
        assert!(sched[0].use_mask_attention);
        // First block of stage 1: in=96, out=192, q_stride=4, window=16, mask=true.
        let stage1_start = sched[1];
        assert_eq!(stage1_start.in_channels, 96);
        assert_eq!(stage1_start.out_channels, 192);
        assert_eq!(stage1_start.q_stride, 4);
        assert_eq!(stage1_start.window_size, 16);
        assert!(stage1_start.use_mask_attention);
        // First block of stage 2: out=384, window=4, mask=false (global).
        let stage2_start = sched[1 + 2];
        assert_eq!(stage2_start.out_channels, 384);
        assert_eq!(stage2_start.window_size, 4);
        assert!(!stage2_start.use_mask_attention);
        // First block of stage 3: out=768.
        let stage3_start = sched[1 + 2 + 7];
        assert_eq!(stage3_start.out_channels, 768);
    }

    #[test]
    fn unroll_preserves_total_elements_and_shape() {
        let b = 1; let c = 4;
        let x = LazyTensor::from_f32(
            (0..(b * NUM_TOKENS * c)).map(|i| (i as f32) * 0.001).collect::<Vec<_>>(),
            Shape::from_dims(&[b, NUM_TOKENS, c]), &Device::cpu(),
        );
        let y = unroll(&x, b, c).unwrap();
        assert_eq!(y.shape().dims(), &[b, NUM_TOKENS, c]);
        let x_realized = x.realize_f32();
        let y_realized = y.realize_f32();
        // Same multiset of values — unroll is a permutation.
        let mut x_sorted = x_realized.clone();
        let mut y_sorted = y_realized.clone();
        x_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        y_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for (xv, yv) in x_sorted.iter().zip(y_sorted.iter()) {
            assert!((xv - yv).abs() < 1e-7, "unroll lost a value: {xv} vs {yv}");
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let cfg = HieraConfig::tiny();
        let weights = tiny_weights(&cfg);
        let model = HieraModel { config: cfg.clone(), weights };
        let img = LazyTensor::from_f32(
            (0..(3 * 224 * 224)).map(|i| (i as f32) * 0.001).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 224, 224]), &Device::cpu(),
        );
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 1000]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn forward_responds_to_input() {
        let cfg = HieraConfig::tiny();
        let weights = tiny_weights(&cfg);
        let model = HieraModel { config: cfg, weights };
        let img_a = LazyTensor::from_f32(
            (0..(3 * 224 * 224)).map(|i| (i as f32) * 0.001).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 224, 224]), &Device::cpu(),
        );
        let img_b = LazyTensor::from_f32(
            (0..(3 * 224 * 224)).map(|i| (i as f32) * 0.001 + 0.3).collect::<Vec<_>>(),
            Shape::from_dims(&[1, 3, 224, 224]), &Device::cpu(),
        );
        let a = model.forward(&img_a).unwrap().realize_f32();
        let b = model.forward(&img_b).unwrap().realize_f32();
        let mut max_diff = 0.0_f32;
        for (x, y) in a.iter().zip(b.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
        // Tiny weights + 12 attention blocks + global mean over 3136 tokens
        // heavily damp the signal.
        assert!(max_diff > 1e-10,
            "Hiera must respond to input changes, max_diff = {max_diff}");
    }

    #[test]
    fn presets_construct() {
        let tiny = HieraConfig::tiny();
        assert_eq!(tiny.channels, 96);
        let huge = HieraConfig::huge();
        assert_eq!(huge.channels, 256);
        assert_eq!(huge.heads, 4);
    }
}
