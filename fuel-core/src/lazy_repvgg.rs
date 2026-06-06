//! RepVGG (Ding et al. 2021, "RepVGG: Making VGG-style
//! ConvNets Great Again") ported to the lazy-graph API.
//!
//! RepVGG separates the training-time architecture (three
//! parallel branches per block: 3×3 conv, 1×1 conv, identity)
//! from the inference-time architecture (a single 3×3 conv +
//! bias). All three branches each have a BatchNorm, and the
//! inference-time fused conv is the analytic sum of the three
//! BN-fused branches. The math is purely a weight rewrite —
//! no runtime cost beyond the equivalent single conv.
//!
//! # Reparameterization at weight-load time
//!
//! For each block we have:
//!
//!   - Branch 3×3: `W_3, gamma_3, beta_3, mu_3, var_3`.
//!   - Branch 1×1: `W_1, gamma_1, beta_1, mu_1, var_1`.
//!   - Branch identity (optional, only when stride == 1 and
//!     `c_in == c_out`): `gamma_i, beta_i, mu_i, var_i`. The
//!     conv "weight" of the identity branch is a synthetic
//!     3×3 where the center is `1.0` on the diagonal and
//!     zero elsewhere (per group).
//!
//! Each branch contributes to the fused conv as
//! `(W * gamma / sqrt(var + eps), beta - mu * gamma / sqrt(var + eps))`.
//! The 1×1 conv weight is zero-padded to 3×3 (kernel center
//! holds the original value); the identity "conv weight" is
//! the synthetic delta kernel above. The fused 3×3 conv weight
//! is the sum across branches; same for bias.
//!
//! Captured by [`fuse_repvgg_block`]: takes raw branch
//! weights and produces the single fused `WeightStorage::F32`
//! conv + `Arc<[f32]>` bias to plug into the lazy model.
//!
//! # Inference-time architecture
//!
//!   - Stem: one RepVGG block, `3 → stem_dim`, stride=2.
//!   - Stages 1-4: per the config's `[n1, n2, n3, n4]`
//!     schedule. First layer of each stage downsamples
//!     (stride=2, NO identity branch); the rest carry
//!     identity branches (stride=1).
//!   - Output channels per stage are width-multiplier scaled:
//!     `stem = min(64, 64 * a)`, stage 1/2/3 = `64/128/256 * a`,
//!     stage 4 = `512 * b`. The `a` (small) and `b` (large)
//!     multipliers come from the config.
//!   - For `g4` variants, every odd-indexed layer (counting
//!     across stage boundaries) uses `groups = 4` instead of 1.
//!   - Head: global average pool → optional Linear classifier.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. Returns class logits
//! `(1, nclasses)` or pooled features `(1, last_channels)`
//! when the classifier is omitted. The fusion helper expects
//! raw branch weights; pre-fused safetensors (the "deploy"
//! checkpoint distributed by the RepVGG authors) can plug in
//! the same way without going through `fuse_repvgg_block`.

use crate::lazy::{load_tensor_as_f32, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

const CHANNELS_PER_STAGE: [usize; 5] = [64, 64, 128, 256, 512];

#[derive(Debug, Clone, PartialEq)]
pub struct RepVggConfig {
    pub a: f32,
    pub b: f32,
    /// 1 for the dense variants, 4 for the `g4` variants.
    pub groups: usize,
    pub stages: [usize; 4],
    pub nclasses: Option<usize>,
}

impl RepVggConfig {
    pub fn a0(nclasses: Option<usize>) -> Self {
        Self { a: 0.75, b: 2.5, groups: 1, stages: [2, 4, 14, 1], nclasses }
    }
    pub fn a1(nclasses: Option<usize>) -> Self {
        Self { a: 1.0, b: 2.5, groups: 1, stages: [2, 4, 14, 1], nclasses }
    }
    pub fn a2(nclasses: Option<usize>) -> Self {
        Self { a: 1.5, b: 2.75, groups: 1, stages: [2, 4, 14, 1], nclasses }
    }
    pub fn b0(nclasses: Option<usize>) -> Self {
        Self { a: 1.0, b: 2.5, groups: 1, stages: [4, 6, 16, 1], nclasses }
    }
    pub fn b1(nclasses: Option<usize>) -> Self {
        Self { a: 2.0, b: 4.0, groups: 1, stages: [4, 6, 16, 1], nclasses }
    }
    pub fn b1g4(nclasses: Option<usize>) -> Self {
        Self { a: 2.0, b: 4.0, groups: 4, stages: [4, 6, 16, 1], nclasses }
    }
    pub fn b2(nclasses: Option<usize>) -> Self {
        Self { a: 2.5, b: 5.0, groups: 1, stages: [4, 6, 16, 1], nclasses }
    }
    pub fn b2g4(nclasses: Option<usize>) -> Self {
        Self { a: 2.5, b: 5.0, groups: 4, stages: [4, 6, 16, 1], nclasses }
    }
    pub fn b3(nclasses: Option<usize>) -> Self {
        Self { a: 3.0, b: 5.0, groups: 1, stages: [4, 6, 16, 1], nclasses }
    }
    pub fn b3g4(nclasses: Option<usize>) -> Self {
        Self { a: 3.0, b: 5.0, groups: 4, stages: [4, 6, 16, 1], nclasses }
    }

    /// Output channels at a given stage (0 = stem, 1-4 = stages).
    pub fn channels_at(&self, stage: usize) -> usize {
        let base = CHANNELS_PER_STAGE[stage] as f32;
        match stage {
            0 => std::cmp::min(64, (base * self.a) as usize),
            4 => (base * self.b) as usize,
            _ => (base * self.a) as usize,
        }
    }
}

/// One fused 3×3 conv layer (post-reparameterization).
#[derive(Debug, Clone)]
pub struct RepVggLayerWeights {
    /// `[c_out, c_in / groups, 3, 3]`.
    pub conv_w: WeightStorage,
    pub conv_b: Arc<[f32]>,
    pub c_in: usize,
    pub c_out: usize,
    pub stride: usize,
    pub groups: usize,
}

#[derive(Debug, Clone)]
pub struct RepVggWeights {
    pub stem: RepVggLayerWeights,
    pub stages: [Vec<RepVggLayerWeights>; 4],
    /// Classifier head; present iff `cfg.nclasses.is_some()`.
    pub head: Option<(WeightStorage, Arc<[f32]>)>,
}

#[derive(Debug, Clone)]
pub struct RepVggModel {
    pub config: RepVggConfig,
    pub weights: RepVggWeights,
}

impl RepVggModel {
    /// Run a forward pass on `image` of shape `(1, 3, H, W)`.
    pub fn forward(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let cfg = &self.config;
        let x = self.run_backbone(image)?;
        let pooled = x.global_avg_pool_2d()?;
        match &self.weights.head {
            None => Ok(pooled),
            Some((w, b)) => {
                let n = cfg.nclasses.expect("head present but cfg.nclasses == None");
                let last_c = cfg.channels_at(4);
                let logits = w.apply_linear(&pooled, last_c, n);
                let bias_t = pooled.const_f32_like(
                    Arc::clone(b), Shape::from_dims(&[n]),
                );
                logits.broadcast_add(&bias_t)
            }
        }
    }

    /// Run the backbone (stem + 4 RepVGG stages, BN-fused into
    /// single Conv+bias+ReLU layers) and return the channels-
    /// first feature map BEFORE global avg pool and the
    /// classifier.
    pub fn forward_features(&self, image: &LazyTensor) -> Result<LazyTensor> {
        self.run_backbone(image)
    }

    fn run_backbone(&self, image: &LazyTensor) -> Result<LazyTensor> {
        let dims = image.shape();
        let dims = dims.dims();
        assert_eq!(dims.len(), 4, "image must be rank 4 [N, 3, H, W]");
        assert_eq!(dims[1], 3, "image must have 3 input channels");

        let mut x = self.apply_layer(image, &self.weights.stem)?;
        for stage in &self.weights.stages {
            for layer in stage {
                x = self.apply_layer(&x, layer)?;
            }
        }
        Ok(x)
    }

    fn apply_layer(&self, x: &LazyTensor, layer: &RepVggLayerWeights) -> Result<LazyTensor> {
        let w_shape = Shape::from_dims(&[layer.c_out, layer.c_in / layer.groups, 3, 3]);
        let w = layer.conv_w.const_like(x, w_shape)?;
        let conv_out = x.conv2d(
            &w, None,
            (layer.stride, layer.stride),
            (1, 1),
            layer.groups,
        )?;
        let bias_t = x
            .const_f32_like(Arc::clone(&layer.conv_b), Shape::from_dims(&[layer.c_out]))
            .reshape(Shape::from_dims(&[1, layer.c_out, 1, 1]))?;
        Ok(conv_out.broadcast_add(&bias_t)?.relu())
    }
}

/// Raw RepVGG block weights as they appear in a non-deploy
/// safetensors checkpoint: three parallel branches each with
/// a Conv2d (3×3 or 1×1) and a BatchNorm; one optional BN for
/// the identity branch.
#[derive(Debug, Clone)]
pub struct RepVggRawBlock {
    /// `[c_out, c_in / groups, 3, 3]`.
    pub conv_3x3_w: Vec<f32>,
    pub bn_3x3_gain: Vec<f32>,
    pub bn_3x3_bias: Vec<f32>,
    pub bn_3x3_mean: Vec<f32>,
    pub bn_3x3_var: Vec<f32>,
    /// `[c_out, c_in / groups, 1, 1]`.
    pub conv_1x1_w: Vec<f32>,
    pub bn_1x1_gain: Vec<f32>,
    pub bn_1x1_bias: Vec<f32>,
    pub bn_1x1_mean: Vec<f32>,
    pub bn_1x1_var: Vec<f32>,
    /// `Some` iff the block has the identity branch (stride == 1
    /// and c_in == c_out).
    pub identity_bn: Option<RepVggBn>,
    pub eps: f64,
    pub c_in: usize,
    pub c_out: usize,
    pub stride: usize,
    pub groups: usize,
}

#[derive(Debug, Clone)]
pub struct RepVggBn {
    pub gain: Vec<f32>,
    pub bias: Vec<f32>,
    pub mean: Vec<f32>,
    pub var: Vec<f32>,
}

/// Fuse a RepVGG block's three parallel branches (3×3, 1×1,
/// optional identity) into a single 3×3 conv + bias suitable
/// for inference. Returns `(fused_conv_3x3, fused_bias)`.
///
/// This is the math behind the RepVGG paper's "structural
/// reparameterization": each branch is `BN(conv(x))`, which
/// is equivalent to a `conv_with_bias(x)` because BN at
/// inference is affine. Summing three branches reduces to
/// summing the three equivalent convs, which since they
/// share input/output channel shapes can be a single 3×3
/// (with the 1×1 zero-padded and the identity expanded into
/// a per-channel delta kernel).
pub fn fuse_repvgg_block(b: &RepVggRawBlock) -> (Vec<f32>, Vec<f32>) {
    let c_out = b.c_out;
    let c_in_per_group = b.c_in / b.groups;
    let kernel_3x3 = 3 * 3;

    // Fuse 3×3 conv + BN.
    let (w3, b3) = fuse_conv_bn_kernel(
        &b.conv_3x3_w, &b.bn_3x3_gain, &b.bn_3x3_bias,
        &b.bn_3x3_mean, &b.bn_3x3_var, b.eps, c_out, c_in_per_group, kernel_3x3,
    );

    // Fuse 1×1 conv + BN, then zero-pad to 3×3 (center holds value).
    let (w1, b1) = fuse_conv_bn_kernel(
        &b.conv_1x1_w, &b.bn_1x1_gain, &b.bn_1x1_bias,
        &b.bn_1x1_mean, &b.bn_1x1_var, b.eps, c_out, c_in_per_group, 1,
    );
    let mut w1_3x3 = vec![0.0_f32; c_out * c_in_per_group * 9];
    for o in 0..c_out {
        for i in 0..c_in_per_group {
            // Pad the (o, i) 1×1 slot into the center of the 3×3 grid.
            let v = w1[o * c_in_per_group + i];
            w1_3x3[o * c_in_per_group * 9 + i * 9 + 4] = v;
        }
    }

    // Synthetic identity branch (when stride==1 and c_in==c_out).
    let (wid_3x3, bid) = match &b.identity_bn {
        None => (vec![0.0_f32; c_out * c_in_per_group * 9], vec![0.0_f32; c_out]),
        Some(idbn) => {
            // Build a delta 3×3 kernel: for each output channel o,
            // the input channel `o mod c_in_per_group` has center = 1.0.
            let mut delta = vec![0.0_f32; c_out * c_in_per_group * 9];
            for o in 0..c_out {
                let i = o % c_in_per_group;
                delta[o * c_in_per_group * 9 + i * 9 + 4] = 1.0;
            }
            let (w, b) = fuse_conv_bn_kernel(
                &delta, &idbn.gain, &idbn.bias, &idbn.mean, &idbn.var,
                b.eps, c_out, c_in_per_group, kernel_3x3,
            );
            (w, b)
        }
    };

    // Sum across the three branches.
    let n = c_out * c_in_per_group * 9;
    let mut fused_w = vec![0.0_f32; n];
    for i in 0..n {
        fused_w[i] = w3[i] + w1_3x3[i] + wid_3x3[i];
    }
    let mut fused_b = vec![0.0_f32; c_out];
    for c in 0..c_out {
        fused_b[c] = b3[c] + b1[c] + bid[c];
    }
    (fused_w, fused_b)
}

/// Common conv-BN fusion: `W' = W * (gamma / sqrt(var + eps))`
/// per-output-channel and `b' = beta - mu * gamma / sqrt(var + eps)`.
fn fuse_conv_bn_kernel(
    w: &[f32], gain: &[f32], bias: &[f32], mean: &[f32], var: &[f32],
    eps: f64, c_out: usize, c_in_per_group: usize, kernel_elems: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(w.len(), c_out * c_in_per_group * kernel_elems);
    assert_eq!(gain.len(), c_out);
    assert_eq!(bias.len(), c_out);
    assert_eq!(mean.len(), c_out);
    assert_eq!(var.len(), c_out);
    let mut w_out = vec![0.0_f32; w.len()];
    let mut b_out = vec![0.0_f32; c_out];
    for o in 0..c_out {
        let inv = 1.0_f32 / ((var[o] as f64 + eps) as f32).sqrt();
        let scale = gain[o] * inv;
        for i in 0..c_in_per_group {
            for k in 0..kernel_elems {
                let idx = o * c_in_per_group * kernel_elems + i * kernel_elems + k;
                w_out[idx] = w[idx] * scale;
            }
        }
        b_out[o] = bias[o] - mean[o] * scale;
    }
    (w_out, b_out)
}

// ---- HuggingFace safetensors loading ---------------------------------------

/// timm BatchNorm epsilon used everywhere in the RepVGG byobnet config.
const REPVGG_BN_EPS: f64 = 1e-5;

impl RepVggWeights {
    /// Load RepVGG weights from a timm-format `byobnet` safetensors checkpoint.
    /// Layout (top-level):
    ///
    /// - Stem: `stem.conv_kxk.conv.weight`, `stem.conv_kxk.bn.{weight,bias,running_mean,running_var}`,
    ///   `stem.conv_1x1.conv.weight`, `stem.conv_1x1.bn.*` (stem has no identity).
    /// - Per stage `s` in 0..=3, per layer `l`:
    ///   `stages.{s}.{l}.conv_kxk.conv.weight`, `stages.{s}.{l}.conv_kxk.bn.*`,
    ///   `stages.{s}.{l}.conv_1x1.conv.weight`, `stages.{s}.{l}.conv_1x1.bn.*`,
    ///   and optionally `stages.{s}.{l}.identity.*` (present iff
    ///   stride == 1 AND `c_in == c_out`).
    /// - Head: `head.fc.weight` (`[nclasses, last_channels]` row-major; we
    ///   transpose to `[in, out]` for `WeightStorage::apply_linear`),
    ///   `head.fc.bias`.
    ///
    /// Each RepVGG block fuses its three branches into a single 3×3 conv
    /// + bias at load time via [`fuse_repvgg_block`], following the
    /// "deploy-time" reparameterization from the paper.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &RepVggConfig,
    ) -> crate::Result<Self> {
        let stem_dim = cfg.channels_at(0);
        let stem = repvgg_load_layer(
            st,
            "stem",
            /* has_identity = */ false,
            3,
            stem_dim,
            /* stride = */ 2,
            /* groups = */ 1,
        )?;

        let mut stages: [Vec<RepVggLayerWeights>; 4] = Default::default();
        for stage_idx in 1..=4 {
            let nlayers = cfg.stages[stage_idx - 1];
            let prev_layers: usize = cfg.stages[..stage_idx - 1].iter().sum();
            let c_prev = cfg.channels_at(stage_idx - 1);
            let c_cur = cfg.channels_at(stage_idx);
            let mut layers = Vec::with_capacity(nlayers);
            for li in 0..nlayers {
                let (has_identity, stride, in_c) = if li == 0 {
                    (false, 2, c_prev)
                } else {
                    (true, 1, c_cur)
                };
                let groups = if (prev_layers + li) % 2 == 1 { cfg.groups } else { 1 };
                let prefix = format!("stages.{}.{}", stage_idx - 1, li);
                layers.push(repvgg_load_layer(
                    st, &prefix, has_identity, in_c, c_cur, stride, groups,
                )?);
            }
            stages[stage_idx - 1] = layers;
        }

        let head = if let Some(n) = cfg.nclasses {
            let last_c = cfg.channels_at(4);
            let fc_w_t = repvgg_load_transposed(st, "head.fc.weight", n, last_c)?;
            let fc_b = load_tensor_as_f32(st, "head.fc.bias")?;
            if fc_b.len() != n {
                return Err(crate::Error::Msg(format!(
                    "RepVGG head.fc.bias expected {n} entries, got {}",
                    fc_b.len(),
                ))
                .bt());
            }
            Some((WeightStorage::F32(Arc::from(fc_w_t)), Arc::from(fc_b)))
        } else {
            None
        };

        Ok(Self { stem, stages, head })
    }
}

/// Load one RepVGG inference-time layer from a raw, three-branch
/// checkpoint. Reads the kxk + 1×1 + (optional) identity BN tuples,
/// runs `fuse_repvgg_block`, and emits the fused 3×3 conv + bias.
fn repvgg_load_layer(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    has_identity: bool,
    c_in: usize,
    c_out: usize,
    stride: usize,
    groups: usize,
) -> crate::Result<RepVggLayerWeights> {
    let c_in_per_group = c_in / groups;
    let expected_3x3 = c_out * c_in_per_group * 3 * 3;
    let expected_1x1 = c_out * c_in_per_group;

    let conv_3x3_w = repvgg_load_check(st, &format!("{prefix}.conv_kxk.conv.weight"), expected_3x3)?;
    let bn_3x3_gain = repvgg_load_check(st, &format!("{prefix}.conv_kxk.bn.weight"), c_out)?;
    let bn_3x3_bias = repvgg_load_check(st, &format!("{prefix}.conv_kxk.bn.bias"), c_out)?;
    let bn_3x3_mean = repvgg_load_check(st, &format!("{prefix}.conv_kxk.bn.running_mean"), c_out)?;
    let bn_3x3_var  = repvgg_load_check(st, &format!("{prefix}.conv_kxk.bn.running_var"),  c_out)?;

    let conv_1x1_w = repvgg_load_check(st, &format!("{prefix}.conv_1x1.conv.weight"), expected_1x1)?;
    let bn_1x1_gain = repvgg_load_check(st, &format!("{prefix}.conv_1x1.bn.weight"), c_out)?;
    let bn_1x1_bias = repvgg_load_check(st, &format!("{prefix}.conv_1x1.bn.bias"), c_out)?;
    let bn_1x1_mean = repvgg_load_check(st, &format!("{prefix}.conv_1x1.bn.running_mean"), c_out)?;
    let bn_1x1_var  = repvgg_load_check(st, &format!("{prefix}.conv_1x1.bn.running_var"),  c_out)?;

    let identity_bn = if has_identity {
        Some(RepVggBn {
            gain: repvgg_load_check(st, &format!("{prefix}.identity.weight"), c_out)?,
            bias: repvgg_load_check(st, &format!("{prefix}.identity.bias"), c_out)?,
            mean: repvgg_load_check(st, &format!("{prefix}.identity.running_mean"), c_out)?,
            var:  repvgg_load_check(st, &format!("{prefix}.identity.running_var"),  c_out)?,
        })
    } else {
        None
    };

    let raw = RepVggRawBlock {
        conv_3x3_w, bn_3x3_gain, bn_3x3_bias, bn_3x3_mean, bn_3x3_var,
        conv_1x1_w, bn_1x1_gain, bn_1x1_bias, bn_1x1_mean, bn_1x1_var,
        identity_bn,
        eps: REPVGG_BN_EPS,
        c_in, c_out, stride, groups,
    };
    let (fused_w, fused_b) = fuse_repvgg_block(&raw);
    Ok(RepVggLayerWeights {
        conv_w: WeightStorage::F32(Arc::from(fused_w)),
        conv_b: Arc::from(fused_b),
        c_in, c_out, stride, groups,
    })
}

fn repvgg_load_check(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    expected_len: usize,
) -> crate::Result<Vec<f32>> {
    let v = load_tensor_as_f32(st, name)?;
    if v.len() != expected_len {
        return Err(crate::Error::Msg(format!(
            "RepVGG load {name:?}: got {} elements, expected {}",
            v.len(), expected_len,
        ))
        .bt());
    }
    Ok(v)
}

fn repvgg_load_transposed(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<f32>> {
    let flat = repvgg_load_check(st, name, out_features * in_features)?;
    let mut out = vec![0.0_f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            out[j * out_features + i] = flat[i * in_features + j];
        }
    }
    Ok(out)
}

impl RepVggModel {
    /// Download a timm-format RepVGG safetensors checkpoint from the Hub
    /// and load it into a model. Caller picks the config (variant). The
    /// canonical repo for timm checkpoints is `timm/repvgg_<variant>.<train>`,
    /// e.g. `timm/repvgg_a0.rvgg_in1k`.
    pub fn from_hub_with_config(repo_id: &str, config: RepVggConfig) -> Result<Self> {
        Self::from_hub_with_filename(repo_id, "model.safetensors", config)
    }

    /// Explicit-filename variant of [`Self::from_hub_with_config`].
    pub fn from_hub_with_filename(
        repo_id: &str,
        filename: &str,
        config: RepVggConfig,
    ) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let weights_path = repo
            .get(filename)
            .map_err(|e| crate::Error::Msg(format!("hf-hub repvgg safetensors: {e}")))?;
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&weights_path) }?;
        let weights = RepVggWeights::load_from_mmapped(&st, &config)?;
        Ok(Self { config, weights })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vec_of(n: usize, next: &mut dyn FnMut() -> f32) -> Vec<f32> {
        (0..n).map(|_| next()).collect()
    }

    fn rng_seed(seed: u32) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05
        }
    }

    fn build_layer(
        c_in: usize, c_out: usize, stride: usize, groups: usize,
        nb: &mut dyn FnMut() -> f32,
    ) -> RepVggLayerWeights {
        let n_w = c_out * (c_in / groups) * 3 * 3;
        RepVggLayerWeights {
            conv_w: WeightStorage::F32(Arc::from(vec_of(n_w, nb))),
            conv_b: Arc::from(vec_of(c_out, nb)),
            c_in, c_out, stride, groups,
        }
    }

    fn build_weights(cfg: &RepVggConfig, seed: u32) -> RepVggWeights {
        let mut nb = rng_seed(seed);
        let stem_dim = cfg.channels_at(0);
        let stem = build_layer(3, stem_dim, 2, 1, &mut nb);
        let mut stages: [Vec<RepVggLayerWeights>; 4] = Default::default();
        for stage_idx in 1..=4 {
            let mut layers = Vec::new();
            let nlayers = cfg.stages[stage_idx - 1];
            let prev_layers: usize = cfg.stages[..stage_idx - 1].iter().sum();
            let c_prev = cfg.channels_at(stage_idx - 1);
            let c_cur = cfg.channels_at(stage_idx);
            for li in 0..nlayers {
                let (stride, in_c) = if li == 0 { (2, c_prev) } else { (1, c_cur) };
                let groups = if (prev_layers + li) % 2 == 1 { cfg.groups } else { 1 };
                layers.push(build_layer(in_c, c_cur, stride, groups, &mut nb));
            }
            stages[stage_idx - 1] = layers;
        }
        let head = cfg.nclasses.map(|n| {
            let last_c = cfg.channels_at(4);
            (
                WeightStorage::F32(Arc::from(vec_of(last_c * n, &mut nb))),
                Arc::from(vec_of(n, &mut nb)),
            )
        });
        RepVggWeights { stem, stages, head }
    }

    fn tiny_image(h: usize) -> LazyTensor {
        let mut nb = rng_seed(99);
        let data: Arc<[f32]> = Arc::from((0..3 * h * h).map(|_| nb()).collect::<Vec<_>>());
        LazyTensor::from_f32(data, Shape::from_dims(&[1, 3, h, h]), &Device::cpu())
    }

    #[test]
    fn repvgg_a0_forward_shape() {
        let cfg = RepVggConfig::a0(Some(10));
        let weights = build_weights(&cfg, 11);
        let model = RepVggModel { config: cfg, weights };
        let img = tiny_image(32);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 10]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    #[test]
    fn repvgg_b1g4_uses_groups() {
        let cfg = RepVggConfig::b1g4(None);
        let weights = build_weights(&cfg, 22);
        let model = RepVggModel { config: cfg, weights };
        // Verify some layers use groups == 4.
        let mut group4_count = 0;
        for stage in &model.weights.stages {
            for layer in stage {
                if layer.groups == 4 { group4_count += 1; }
            }
        }
        assert!(group4_count > 0, "b1g4 must place groups=4 on odd layers");

        let img = tiny_image(32);
        let feats = model.forward(&img).unwrap();
        // No classifier head → returns features of shape (1, last_channels).
        let last_c = model.config.channels_at(4);
        assert_eq!(feats.shape().dims(), &[1, last_c]);
        for &v in &feats.realize_f32() {
            assert!(v.is_finite());
        }
    }

    /// Channel multipliers per variant follow the canonical RepVGG schedule.
    #[test]
    fn variant_channel_counts() {
        // A0: a=0.75, so stages 1/2/3 = 48/96/192; stage 4 = 512 * 2.5 = 1280; stem = min(64, 48) = 48.
        let a0 = RepVggConfig::a0(None);
        assert_eq!(a0.channels_at(0), 48);
        assert_eq!(a0.channels_at(1), 48);
        assert_eq!(a0.channels_at(2), 96);
        assert_eq!(a0.channels_at(3), 192);
        assert_eq!(a0.channels_at(4), 1280);
        // B0: a=1.0, b=2.5, stages 1/2/3 = 64/128/256; stage 4 = 1280; stem = min(64, 64) = 64.
        let b0 = RepVggConfig::b0(None);
        assert_eq!(b0.channels_at(0), 64);
        assert_eq!(b0.channels_at(1), 64);
        assert_eq!(b0.channels_at(2), 128);
        assert_eq!(b0.channels_at(3), 256);
        assert_eq!(b0.channels_at(4), 1280);
    }

    /// Identity-branch BN fusion with `gamma=1, beta=0, mean=0,
    /// var=1, eps=0` reduces the identity branch to a pure
    /// delta-kernel + zero-bias contribution. Verify the math.
    #[test]
    fn fuse_identity_bn_is_identity() {
        let c_out = 4;
        let c_in_per_group = 4;
        let mut delta = vec![0.0_f32; c_out * c_in_per_group * 9];
        for o in 0..c_out {
            delta[o * c_in_per_group * 9 + o * 9 + 4] = 1.0;
        }
        let gain = vec![1.0_f32; c_out];
        let bias = vec![0.0_f32; c_out];
        let mean = vec![0.0_f32; c_out];
        let var = vec![1.0_f32; c_out];
        let (w, b) = fuse_conv_bn_kernel(
            &delta, &gain, &bias, &mean, &var, 0.0, c_out, c_in_per_group, 9,
        );
        // Delta * (1/sqrt(1)) = delta unchanged; bias = 0 - 0 = 0.
        for i in 0..delta.len() {
            assert!((w[i] - delta[i]).abs() < 1e-7,
                "fused identity weight differs at {i}: expected {}, got {}", delta[i], w[i]);
        }
        for c in 0..c_out {
            assert!(b[c].abs() < 1e-7);
        }
    }

    /// Full block fusion with all-zero branches reduces to a
    /// zero conv (each branch contributes only its bias-from-BN
    /// term, and with `beta=0, mean=0` that term is zero too).
    #[test]
    fn fuse_full_block_zero_branches() {
        let c_in = 4;
        let c_out = 4;
        let raw = RepVggRawBlock {
            conv_3x3_w: vec![0.0; c_out * c_in * 9],
            bn_3x3_gain: vec![1.0; c_out],
            bn_3x3_bias: vec![0.0; c_out],
            bn_3x3_mean: vec![0.0; c_out],
            bn_3x3_var: vec![1.0; c_out],
            conv_1x1_w: vec![0.0; c_out * c_in],
            bn_1x1_gain: vec![1.0; c_out],
            bn_1x1_bias: vec![0.0; c_out],
            bn_1x1_mean: vec![0.0; c_out],
            bn_1x1_var: vec![1.0; c_out],
            identity_bn: Some(RepVggBn {
                gain: vec![1.0; c_out],
                bias: vec![0.0; c_out],
                mean: vec![0.0; c_out],
                var: vec![1.0; c_out],
            }),
            eps: 1e-5,
            c_in, c_out, stride: 1, groups: 1,
        };
        let (w, b) = fuse_repvgg_block(&raw);
        // 3×3 contributes 0, 1×1 contributes 0; identity contributes
        // its delta kernel scaled by 1/sqrt(var + eps) ≈ 1 - eps/2 for
        // small eps. So tolerance must absorb the BN epsilon.
        let bn_scale = 1.0_f32 / (1.0_f32 + 1e-5_f32).sqrt();
        for o in 0..c_out {
            for i in 0..c_in {
                for k in 0..9 {
                    let v = w[o * c_in * 9 + i * 9 + k];
                    let expected = if o == i && k == 4 { bn_scale } else { 0.0 };
                    assert!((v - expected).abs() < 1e-6,
                        "fused (o={o},i={i},k={k}) expected {expected}, got {v}");
                }
            }
        }
        for c in 0..c_out {
            assert!(b[c].abs() < 1e-7,
                "fused bias[{c}] expected 0, got {}", b[c]);
        }
    }

    #[test]
    fn forward_features_shape_and_finite() {
        let cfg = RepVggConfig::a0(Some(10));
        let weights = build_weights(&cfg, 33);
        let model = RepVggModel { config: cfg, weights };
        let img = tiny_image(32);
        let feats = model.forward_features(&img).unwrap();
        let shape = feats.shape();
        let dims = shape.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], model.config.channels_at(4));
        for &v in &feats.realize_f32() {
            assert!(v.is_finite(), "non-finite feature: {v}");
        }
    }

    // ---- load_from_mmapped round-trip ---------------------------------------

    /// Build raw f32 bytes for a tensor of `len` elements seeded by `seed`.
    fn raw_f32(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(len * 4);
        for _ in 0..len {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            // Range ~[-0.05, 0.05].
            let v = ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05;
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    /// All-ones BatchNorm gain/var bytes (so the BN affine is identity).
    fn raw_f32_const(len: usize, value: f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(len * 4);
        for _ in 0..len {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    /// Write a fully synthetic RepVGG-a0 (no classifier) safetensors blob,
    /// load it via `RepVggWeights::load_from_mmapped`, and verify the
    /// fused stem layer matches the analytic BN-fold of the raw branches.
    #[test]
    fn load_from_mmapped_round_trip_repvgg_a0_no_head() {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;

        let cfg = RepVggConfig::a0(None);
        let stem_dim = cfg.channels_at(0);
        let c_in = 3;
        let c_out = stem_dim;

        // Stem raw branch tensors. Conv 3x3: [c_out, c_in, 3, 3];
        // conv 1x1: [c_out, c_in, 1, 1]; BNs: [c_out] each.
        let conv_3x3_bytes = raw_f32(c_out * c_in * 3 * 3, 0xC0FFEE);
        let conv_1x1_bytes = raw_f32(c_out * c_in,         0xBEEF00);
        // gain=1, bias=0, mean=0, var=1 → BN reduces to ~identity (1/√(1+eps)).
        let bn_gain_bytes  = raw_f32_const(c_out, 1.0);
        let bn_bias_bytes  = raw_f32_const(c_out, 0.0);
        let bn_mean_bytes  = raw_f32_const(c_out, 0.0);
        let bn_var_bytes   = raw_f32_const(c_out, 1.0);

        // Helper: build identical BN-stat byte buffers per (slot) so we
        // don't borrow conflicts in the HashMap<&[u8]>.
        let mut owned_bytes: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
        owned_bytes.push((
            "stem.conv_kxk.conv.weight".into(),
            vec![c_out, c_in, 3, 3],
            conv_3x3_bytes,
        ));
        owned_bytes.push((
            "stem.conv_1x1.conv.weight".into(),
            vec![c_out, c_in, 1, 1],
            conv_1x1_bytes,
        ));
        for (suffix, raw) in [
            ("conv_kxk.bn.weight",       bn_gain_bytes.clone()),
            ("conv_kxk.bn.bias",         bn_bias_bytes.clone()),
            ("conv_kxk.bn.running_mean", bn_mean_bytes.clone()),
            ("conv_kxk.bn.running_var",  bn_var_bytes.clone()),
            ("conv_1x1.bn.weight",       bn_gain_bytes.clone()),
            ("conv_1x1.bn.bias",         bn_bias_bytes.clone()),
            ("conv_1x1.bn.running_mean", bn_mean_bytes.clone()),
            ("conv_1x1.bn.running_var",  bn_var_bytes.clone()),
        ] {
            owned_bytes.push((format!("stem.{suffix}"), vec![c_out], raw));
        }

        // Fill in every stage layer with the same zero-conv + identity BN
        // recipe so the test's structural walk hits every block path.
        for stage_idx in 1..=4_usize {
            let nlayers = cfg.stages[stage_idx - 1];
            let prev_layers: usize = cfg.stages[..stage_idx - 1].iter().sum();
            let c_prev = cfg.channels_at(stage_idx - 1);
            let c_cur  = cfg.channels_at(stage_idx);
            for li in 0..nlayers {
                let (has_identity, in_c) = if li == 0 {
                    (false, c_prev)
                } else {
                    (true, c_cur)
                };
                let groups = if (prev_layers + li) % 2 == 1 { cfg.groups } else { 1 };
                let c_in_per_group = in_c / groups;
                let prefix = format!("stages.{}.{}", stage_idx - 1, li);
                owned_bytes.push((
                    format!("{prefix}.conv_kxk.conv.weight"),
                    vec![c_cur, c_in_per_group, 3, 3],
                    raw_f32(c_cur * c_in_per_group * 9, (stage_idx * 100 + li) as u32),
                ));
                owned_bytes.push((
                    format!("{prefix}.conv_1x1.conv.weight"),
                    vec![c_cur, c_in_per_group, 1, 1],
                    raw_f32(c_cur * c_in_per_group, (stage_idx * 100 + li + 5000) as u32),
                ));
                for (suffix, raw) in [
                    ("conv_kxk.bn.weight",       raw_f32_const(c_cur, 1.0)),
                    ("conv_kxk.bn.bias",         raw_f32_const(c_cur, 0.0)),
                    ("conv_kxk.bn.running_mean", raw_f32_const(c_cur, 0.0)),
                    ("conv_kxk.bn.running_var",  raw_f32_const(c_cur, 1.0)),
                    ("conv_1x1.bn.weight",       raw_f32_const(c_cur, 1.0)),
                    ("conv_1x1.bn.bias",         raw_f32_const(c_cur, 0.0)),
                    ("conv_1x1.bn.running_mean", raw_f32_const(c_cur, 0.0)),
                    ("conv_1x1.bn.running_var",  raw_f32_const(c_cur, 1.0)),
                ] {
                    owned_bytes.push((
                        format!("{prefix}.{suffix}"),
                        vec![c_cur],
                        raw,
                    ));
                }
                if has_identity {
                    for (suffix, raw) in [
                        ("identity.weight",       raw_f32_const(c_cur, 1.0)),
                        ("identity.bias",         raw_f32_const(c_cur, 0.0)),
                        ("identity.running_mean", raw_f32_const(c_cur, 0.0)),
                        ("identity.running_var",  raw_f32_const(c_cur, 1.0)),
                    ] {
                        owned_bytes.push((
                            format!("{prefix}.{suffix}"),
                            vec![c_cur],
                            raw,
                        ));
                    }
                }
            }
        }

        // Build the TensorView map (borrowing the byte buffers).
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (name, shape, bytes) in &owned_bytes {
            let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .expect("TensorView::new");
            tensors.insert(name.clone(), view);
        }
        let metadata: Option<HashMap<String, String>> = None;
        let serialized = safetensors::serialize(&tensors, metadata)
            .expect("safetensors::serialize");

        // Write to a temp file (MmapedSafetensors requires a real file).
        let tmp = std::env::temp_dir().join(format!(
            "fuel_repvgg_load_test_{}.safetensors",
            std::process::id(),
        ));
        std::fs::write(&tmp, &serialized).expect("write tmp");
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&tmp) }
            .expect("MmapedSafetensors::new");
        let loaded = RepVggWeights::load_from_mmapped(&st, &cfg)
            .expect("RepVggWeights::load_from_mmapped");

        // Stem: with BN(gain=1, bias=0, mean=0, var=1), each branch's
        // fused conv is `W * (1/√(1+eps))` and fused bias is 0. The 1×1
        // branch is zero-padded into the 3×3 center. Sum across branches.
        let bn_scale = 1.0_f32 / (1.0_f32 + REPVGG_BN_EPS as f32).sqrt();
        let conv_3x3 = match &loaded.stem.conv_w {
            WeightStorage::F32(arc) => arc.clone(),
            other => panic!("expected F32 stem conv weights, got {other:?}"),
        };
        // Pull the same raw bytes we wrote and verify the per-element fusion.
        let raw_3x3 = &owned_bytes
            .iter()
            .find(|(name, _, _)| name == "stem.conv_kxk.conv.weight")
            .unwrap().2;
        let raw_1x1 = &owned_bytes
            .iter()
            .find(|(name, _, _)| name == "stem.conv_1x1.conv.weight")
            .unwrap().2;
        // Decode f32 elements from raw bytes.
        let mut raw_3x3_f: Vec<f32> = Vec::with_capacity(c_out * c_in * 9);
        for chunk in raw_3x3.chunks_exact(4) {
            raw_3x3_f.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        let mut raw_1x1_f: Vec<f32> = Vec::with_capacity(c_out * c_in);
        for chunk in raw_1x1.chunks_exact(4) {
            raw_1x1_f.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        // For each (o, i, kx, ky), expected = bn_scale * (W3 + delta_center * W1).
        // Stem has no identity branch, so no delta contribution.
        for o in 0..c_out {
            for i in 0..c_in {
                for k in 0..9 {
                    let w3 = raw_3x3_f[o * c_in * 9 + i * 9 + k];
                    let w1 = raw_1x1_f[o * c_in + i];
                    let expected = bn_scale * (w3 + if k == 4 { w1 } else { 0.0 });
                    let got = conv_3x3[o * c_in * 9 + i * 9 + k];
                    assert!((got - expected).abs() < 1e-6,
                        "stem fused (o={o},i={i},k={k}) expected {expected}, got {got}");
                }
            }
        }
        // Bias is exactly 0 because beta=0, mean=0 on every branch.
        for c in 0..c_out {
            assert!(loaded.stem.conv_b[c].abs() < 1e-6, "stem bias[{c}]");
        }

        // Sanity: structural shape of the loaded weights matches the spec.
        assert!(loaded.head.is_none());
        for stage_idx in 0..4 {
            assert_eq!(loaded.stages[stage_idx].len(), cfg.stages[stage_idx]);
        }

        // Forward through the loaded model — confirms shapes hooked up
        // through the fused conv path.
        let model = RepVggModel { config: cfg.clone(), weights: loaded };
        let img = tiny_image(32);
        let feats = model.forward(&img).unwrap();
        assert_eq!(feats.shape().dims(), &[1, cfg.channels_at(4)]);

        // Best-effort cleanup; ignore errors so a leftover file doesn't
        // fail the test.
        let _ = std::fs::remove_file(&tmp);
    }

    /// Smoke test that documents the canonical `from_hub_with_config`
    /// usage. Ignored by default because it hits the HF Hub.
    #[test]
    #[ignore]
    fn from_hub_smoke_repvgg_a0() {
        let cfg = RepVggConfig::a0(Some(1000));
        let model = RepVggModel::from_hub_with_config(
            "timm/repvgg_a0.rvgg_in1k", cfg,
        ).expect("from_hub_with_config");
        assert!(model.weights.head.is_some());
    }
}
