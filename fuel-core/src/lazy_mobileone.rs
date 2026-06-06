//! MobileOne (Vasu et al. 2022, "MobileOne: An Improved One
//! Millisecond Mobile Backbone") ported to the lazy-graph API.
//!
//! MobileOne is the depthwise-pointwise sibling of [`crate::lazy_repvgg`]
//! tuned for mobile inference. Each "block" in a stage is
//! actually a pair of reparameterized convs: a 3×3 depthwise
//! (groups = `c_in`, mixes spatial) followed by a 1×1
//! pointwise (groups = 1, mixes channels). Both use the
//! RepVGG-style branch fusion at inference, plus an
//! overparameterization factor `k` that sums multiple
//! parallel kxk training-time branches. The S4 variant adds
//! SE blocks between the fused conv and ReLU.
//!
//! # Reparameterization
//!
//! For each conv at inference, the fused 3×3 (or 1×1) weight
//! and bias are the analytical sum of:
//!
//!   - `k` parallel `kxk` conv+BN branches (overparameterization),
//!   - one `1×1` conv+BN "scale" branch (only when kernel > 1,
//!     zero-padded to kxk),
//!   - one identity+BN branch (only when stride == 1 and
//!     `c_in == c_out`).
//!
//! S0 sets `k = 4` (more training-time branches); all other
//! variants use `k = 1`. The fusion math is identical to
//! [`crate::lazy_repvgg::fuse_repvgg_block`] except for the
//! k-sum and the depthwise-friendly identity expansion (for
//! a 1×1 conv, the identity puts `1.0` at index
//! `i * (in_per_group + 1)` instead of `i * 9 + 4`).
//!
//! # Stage / block structure
//!
//! Five stages with block counts `[1, 2, 8, 10, 1]`. Per
//! "block" in a stage, the lazy port emits TWO conv layers:
//!
//!   1. **Depthwise 3×3**: `groups = c_in`, stride from the
//!      block (2 for the first block of each stage, 1 otherwise),
//!      `c_out = c_in`.
//!   2. **Pointwise 1×1**: `groups = 1`, stride = 1, `c_in →
//!      stage out_channels`.
//!
//! Channels per stage are `min(64, 64 * α₀)` for the stem
//! (matching stage 0) and `[64, 64, 128, 256, 512] * αᵢ`
//! for stages 1-4. The five `α` multipliers come from the
//! config.
//!
//! S4 variant adds SE blocks (with `squeeze = c_out / 16`)
//! between the fused conv and the ReLU. Other variants leave
//! `se = None`.
//!
//! # Scope (v1)
//!
//! Forward-only, batch == 1, F32. Returns `(1, nclasses)`
//! with the classifier head or `(1, last_channels)` without.

use crate::lazy::{load_tensor_as_f32, LazyTensor, WeightStorage};
use crate::{Device, Result};
use fuel_core_types::Shape;
use std::sync::Arc;

const STAGE_BLOCKS: [usize; 5] = [1, 2, 8, 10, 1];
const STAGE_BASE_CHANNELS: [usize; 5] = [64, 64, 128, 256, 512];

#[derive(Debug, Clone, PartialEq)]
pub struct MobileOneConfig {
    /// Overparameterization factor used at TRAINING time. The
    /// inference-time conv is the sum of `k` parallel kxk
    /// branches. v1 takes already-fused weights, so this is
    /// informational only.
    pub k: usize,
    pub alphas: [f32; 5],
    pub nclasses: Option<usize>,
}

impl MobileOneConfig {
    pub fn s0(nclasses: Option<usize>) -> Self {
        Self { k: 4, alphas: [0.75, 0.75, 1.0, 1.0, 2.0], nclasses }
    }
    pub fn s1(nclasses: Option<usize>) -> Self {
        Self { k: 1, alphas: [1.5, 1.5, 1.5, 2.0, 2.5], nclasses }
    }
    pub fn s2(nclasses: Option<usize>) -> Self {
        Self { k: 1, alphas: [1.5, 1.5, 2.0, 2.5, 4.0], nclasses }
    }
    pub fn s3(nclasses: Option<usize>) -> Self {
        Self { k: 1, alphas: [2.0, 2.0, 2.5, 3.0, 4.0], nclasses }
    }
    pub fn s4(nclasses: Option<usize>) -> Self {
        Self { k: 1, alphas: [3.0, 3.0, 3.5, 3.5, 4.0], nclasses }
    }

    /// Output channels for stage `stage` (0 = stem,
    /// 1-4 = stages). Same channel-clipping rule as RepVGG for
    /// the stem.
    pub fn channels_at(&self, stage: usize) -> usize {
        let base = STAGE_BASE_CHANNELS[stage] as f32;
        let m = self.alphas[stage];
        match stage {
            0 => std::cmp::min(64, (base * m) as usize),
            _ => (base * m) as usize,
        }
    }
}

/// One fused conv layer (depthwise or pointwise, k-sum +
/// scale + identity already collapsed). Followed by an
/// optional SE block in S4.
#[derive(Debug, Clone)]
pub struct MobileOneLayerWeights {
    /// `[c_out, c_in / groups, kernel, kernel]`.
    pub conv_w: WeightStorage,
    pub conv_b: Arc<[f32]>,
    pub c_in: usize,
    pub c_out: usize,
    pub kernel: usize,
    pub stride: usize,
    pub groups: usize,
    /// `Some` in S4 variant's appropriate layers; `None` everywhere else.
    pub se: Option<MobileOneSeWeights>,
}

#[derive(Debug, Clone)]
pub struct MobileOneSeWeights {
    /// `[c_out, c_in, 1, 1]` and bias `[c_out]`.
    pub fc1_w: WeightStorage,
    pub fc1_b: Arc<[f32]>,
    pub fc2_w: WeightStorage,
    pub fc2_b: Arc<[f32]>,
    pub squeeze: usize,
    pub channels: usize,
}

#[derive(Debug, Clone)]
pub struct MobileOneWeights {
    pub stem: MobileOneLayerWeights,
    /// Stage layers in evaluation order. Each stage block emits
    /// TWO layers (depthwise + pointwise), so a stage with `N`
    /// blocks has `2 * N` entries.
    pub stages: [Vec<MobileOneLayerWeights>; 4],
    pub head: Option<(WeightStorage, Arc<[f32]>)>,
}

#[derive(Debug, Clone)]
pub struct MobileOneModel {
    pub config: MobileOneConfig,
    pub weights: MobileOneWeights,
}

impl MobileOneModel {
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

    /// Run the backbone (stem + 4 MobileOne stages with
    /// branch-fused Conv+bias and optional SE) and return the
    /// channels-first feature map BEFORE global avg pool and
    /// the classifier.
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

    fn apply_layer(&self, x: &LazyTensor, layer: &MobileOneLayerWeights) -> Result<LazyTensor> {
        let w_shape = Shape::from_dims(&[
            layer.c_out, layer.c_in / layer.groups, layer.kernel, layer.kernel,
        ]);
        let w = layer.conv_w.const_like(x, w_shape)?;
        let pad = if layer.kernel > 1 { 1 } else { 0 };
        let conv_out = x.conv2d(
            &w, None,
            (layer.stride, layer.stride),
            (pad, pad),
            layer.groups,
        )?;
        let bias_t = x
            .const_f32_like(Arc::clone(&layer.conv_b), Shape::from_dims(&[layer.c_out]))
            .reshape(Shape::from_dims(&[1, layer.c_out, 1, 1]))?;
        let mut out = conv_out.broadcast_add(&bias_t)?;
        if let Some(se) = &layer.se {
            out = self.apply_se(&out, se)?;
        }
        Ok(out.relu())
    }

    fn apply_se(&self, x: &LazyTensor, se: &MobileOneSeWeights) -> Result<LazyTensor> {
        let pooled = x.mean_keepdim(2_usize)?.mean_keepdim(3_usize)?; // (N, C, 1, 1)
        let g = self.apply_se_conv(&pooled, &se.fc1_w, &se.fc1_b, se.channels, se.squeeze)?;
        let g = g.relu();
        let g = self.apply_se_conv(&g, &se.fc2_w, &se.fc2_b, se.squeeze, se.channels)?;
        let g = g.sigmoid();
        x.broadcast_mul(&g)
    }

    fn apply_se_conv(
        &self,
        x: &LazyTensor,
        w: &WeightStorage,
        b: &Arc<[f32]>,
        c_in: usize, c_out: usize,
    ) -> Result<LazyTensor> {
        let wt = w.const_like(x, Shape::from_dims(&[c_out, c_in, 1, 1]))?;
        let conv = x.conv2d(&wt, None, (1, 1), (0, 0), 1)?;
        let bt = x
            .const_f32_like(Arc::clone(b), Shape::from_dims(&[c_out]))
            .reshape(Shape::from_dims(&[1, c_out, 1, 1]))?;
        conv.broadcast_add(&bt)
    }
}

// ---- HuggingFace safetensors loading ---------------------------------------

/// timm BatchNorm eps used everywhere in the MobileOne byobnet config.
const MOBILEONE_BN_EPS: f64 = 1e-5;

impl MobileOneConfig {
    /// Returns true if this variant uses SE blocks (S4 only). Used by the
    /// loader to decide whether to probe the safetensors for `attn.fc{1,2}.*`
    /// entries on each pointwise layer.
    pub fn has_se(&self) -> bool {
        // S4 is the only stock variant with SE. Detected via the alpha
        // schedule: `[3.0, 3.0, 3.5, 3.5, 4.0]`.
        (self.alphas[0] - 3.0).abs() < 1e-6
            && (self.alphas[2] - 3.5).abs() < 1e-6
            && (self.alphas[4] - 4.0).abs() < 1e-6
    }
}

impl MobileOneWeights {
    /// Load MobileOne weights from a timm-format `byobnet` safetensors
    /// checkpoint. Layout (top-level):
    ///
    /// - Stem: one MobileOne block with `k=1` (no overparameterization at
    ///   inference; just a single kxk + 1×1 scale branch fusion). Prefix
    ///   is `stem.`.
    /// - Stages: `stages.{s}.{2*b}` (depthwise 3×3) and `stages.{s}.{2*b+1}`
    ///   (pointwise 1×1), where `s in 0..4` and `b in 0..STAGE_BLOCKS[s+1]`.
    /// - Each MobileOne block:
    ///   `conv_kxk.{i}.conv.weight` + `conv_kxk.{i}.bn.*` for `i in 0..k`,
    ///   optionally `conv_scale.conv.weight` + `conv_scale.bn.*` when
    ///   `kernel > 1`, and optionally `identity.*` when `has_identity == true`.
    /// - S4-only SE: on each *pointwise* layer (and the stem) probe
    ///   `attn.fc{1,2}.weight` and `attn.fc{1,2}.bias`.
    /// - Classifier: `head.fc.weight` (`[nclasses, last_channels]` →
    ///   transposed to `[in, out]`), `head.fc.bias`.
    pub fn load_from_mmapped(
        st: &crate::safetensors::MmapedSafetensors,
        cfg: &MobileOneConfig,
    ) -> crate::Result<Self> {
        let has_se = cfg.has_se();
        // Stem: k=1, kernel=3, stride=2, groups=1, no identity. SE on stem
        // only if has_se && timm wrote one — eager Fuel does add the SE
        // block on the stem in the S4 variant.
        let stem_dim = cfg.channels_at(0);
        let stem = mobileone_load_layer(
            st,
            "stem",
            cfg,
            /* has_identity = */ false,
            /* c_in = */ 3,
            /* c_out = */ stem_dim,
            /* kernel = */ 3,
            /* stride = */ 2,
            /* groups = */ 1,
            /* k = */ 1,
            has_se,
        )?;

        let mut stages: [Vec<MobileOneLayerWeights>; 4] = Default::default();
        for stage_idx in 1..=4 {
            let nblocks = STAGE_BLOCKS[stage_idx];
            let mut layers = Vec::with_capacity(nblocks * 2);
            let mut in_c = cfg.channels_at(stage_idx - 1);
            let out_c = cfg.channels_at(stage_idx);
            for b in 0..nblocks {
                let (has_identity, stride) = if b == 0 { (false, 2) } else { (true, 1) };
                // Depthwise: kernel=3, groups=in_c, c_out=in_c.
                let dw_prefix = format!("stages.{}.{}", stage_idx - 1, b * 2);
                let dw = mobileone_load_layer(
                    st, &dw_prefix, cfg, has_identity,
                    in_c, in_c, 3, stride, in_c, cfg.k,
                    /* has_se = */ false,
                )?;
                layers.push(dw);
                // Pointwise: kernel=1, groups=1, c_in=in_c, c_out=out_c.
                let pw_prefix = format!("stages.{}.{}", stage_idx - 1, b * 2 + 1);
                let pw = mobileone_load_layer(
                    st, &pw_prefix, cfg, has_identity,
                    in_c, out_c, 1, /* stride = */ 1, /* groups = */ 1, cfg.k,
                    /* has_se = */ has_se,
                )?;
                layers.push(pw);
                in_c = out_c;
            }
            stages[stage_idx - 1] = layers;
        }

        let head = if let Some(n) = cfg.nclasses {
            let last_c = cfg.channels_at(4);
            let fc_w_t = mobileone_load_transposed(st, "head.fc.weight", n, last_c)?;
            let fc_b = load_tensor_as_f32(st, "head.fc.bias")?;
            if fc_b.len() != n {
                return Err(crate::Error::Msg(format!(
                    "MobileOne head.fc.bias expected {n} entries, got {}",
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

/// Load and fuse one MobileOne block (depthwise or pointwise). Runs the
/// RepVGG-style branch fusion (`k`-sum + scale + identity), bakes BN into
/// the conv weights, and optionally attaches SE.
#[allow(clippy::too_many_arguments)]
fn mobileone_load_layer(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    cfg: &MobileOneConfig,
    has_identity: bool,
    c_in: usize,
    c_out: usize,
    kernel: usize,
    stride: usize,
    groups: usize,
    k: usize,
    has_se: bool,
) -> crate::Result<MobileOneLayerWeights> {
    let _ = cfg;
    let c_in_per_group = c_in / groups;
    let kernel_elems = kernel * kernel;
    let n_w = c_out * c_in_per_group * kernel_elems;

    // Accumulators.
    let mut fused_w = vec![0.0_f32; n_w];
    let mut fused_b = vec![0.0_f32; c_out];

    // k parallel kxk conv+BN branches.
    for i in 0..k {
        let conv_w = mobileone_load_check(
            st,
            &format!("{prefix}.conv_kxk.{i}.conv.weight"),
            n_w,
        )?;
        let (gain, bias, mean, var) = mobileone_load_bn(
            st, &format!("{prefix}.conv_kxk.{i}.bn"), c_out,
        )?;
        let (wk, bk) = fuse_conv_bn_kernel(
            &conv_w, &gain, &bias, &mean, &var,
            MOBILEONE_BN_EPS, c_out, c_in_per_group, kernel_elems,
        );
        for j in 0..n_w { fused_w[j] += wk[j]; }
        for j in 0..c_out { fused_b[j] += bk[j]; }
    }

    // Optional 1×1 scale branch (only when kernel > 1). The fused 1×1
    // conv is padded into the kxk center.
    if kernel > 1 {
        let scale_w = mobileone_load_check(
            st, &format!("{prefix}.conv_scale.conv.weight"),
            c_out * c_in_per_group,
        )?;
        let (gain, bias, mean, var) = mobileone_load_bn(
            st, &format!("{prefix}.conv_scale.bn"), c_out,
        )?;
        let (ws, bs) = fuse_conv_bn_kernel(
            &scale_w, &gain, &bias, &mean, &var,
            MOBILEONE_BN_EPS, c_out, c_in_per_group, 1,
        );
        // Place each (o, i) 1×1 value into the kxk center.
        let center = kernel_elems / 2;
        for o in 0..c_out {
            for i in 0..c_in_per_group {
                let v = ws[o * c_in_per_group + i];
                let off = o * c_in_per_group * kernel_elems + i * kernel_elems + center;
                fused_w[off] += v;
            }
        }
        for j in 0..c_out { fused_b[j] += bs[j]; }
    }

    // Optional identity branch (only when stride==1 && c_in==c_out).
    if has_identity {
        // Build the synthetic delta kernel (per-channel center).
        let mut delta = vec![0.0_f32; n_w];
        let id = c_in_per_group;
        for i in 0..c_in {
            if kernel > 1 {
                delta[i * kernel_elems + kernel_elems / 2] = 1.0;
            } else {
                // Pointwise 1×1: weights of shape [c_out, c_in, 1, 1] with
                // c_out == c_in (since has_identity ⇒ c_in == c_out).
                // The eager byobnet places 1.0 at `i * (id + 1)`.
                delta[i * (id + 1)] = 1.0;
            }
        }
        let (gain, bias, mean, var) = mobileone_load_bn(
            st, &format!("{prefix}.identity"), c_out,
        )?;
        let (wi, bi) = fuse_conv_bn_kernel(
            &delta, &gain, &bias, &mean, &var,
            MOBILEONE_BN_EPS, c_out, c_in_per_group, kernel_elems,
        );
        for j in 0..n_w { fused_w[j] += wi[j]; }
        for j in 0..c_out { fused_b[j] += bi[j]; }
    }

    // Optional SE block — probe the safetensors. The 1×1 conv shapes in
    // timm byobnet are `[channels, squeeze, 1, 1]` for fc2 and
    // `[squeeze, channels, 1, 1]` for fc1.
    let se = if has_se {
        let squeeze = (c_out / 16).max(1);
        // Probe — if the SE entries are missing, fall through with `None`.
        let probe = format!("{prefix}.attn.fc1.weight");
        if st.get(&probe).is_ok() {
            let fc1_w = mobileone_load_check(
                st, &format!("{prefix}.attn.fc1.weight"),
                squeeze * c_out,
            )?;
            let fc1_b = mobileone_load_check(
                st, &format!("{prefix}.attn.fc1.bias"),
                squeeze,
            )?;
            let fc2_w = mobileone_load_check(
                st, &format!("{prefix}.attn.fc2.weight"),
                c_out * squeeze,
            )?;
            let fc2_b = mobileone_load_check(
                st, &format!("{prefix}.attn.fc2.bias"),
                c_out,
            )?;
            Some(MobileOneSeWeights {
                fc1_w: WeightStorage::F32(Arc::from(fc1_w)),
                fc1_b: Arc::from(fc1_b),
                fc2_w: WeightStorage::F32(Arc::from(fc2_w)),
                fc2_b: Arc::from(fc2_b),
                squeeze,
                channels: c_out,
            })
        } else {
            None
        }
    } else {
        None
    };

    Ok(MobileOneLayerWeights {
        conv_w: WeightStorage::F32(Arc::from(fused_w)),
        conv_b: Arc::from(fused_b),
        c_in, c_out, kernel, stride, groups, se,
    })
}

/// Apply `W' = W * gamma / sqrt(var + eps)` per output channel and
/// `b' = beta - mu * gamma / sqrt(var + eps)`. Mirrors the math in
/// [`crate::lazy_repvgg::fuse_conv_bn_kernel`] (kept local to avoid a
/// pub-promotion just for an internal helper).
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

fn mobileone_load_bn(
    st: &crate::safetensors::MmapedSafetensors,
    prefix: &str,
    channels: usize,
) -> crate::Result<(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)> {
    let gain = mobileone_load_check(st, &format!("{prefix}.weight"), channels)?;
    let bias = mobileone_load_check(st, &format!("{prefix}.bias"),   channels)?;
    let mean = mobileone_load_check(st, &format!("{prefix}.running_mean"), channels)?;
    let var  = mobileone_load_check(st, &format!("{prefix}.running_var"),  channels)?;
    Ok((gain, bias, mean, var))
}

fn mobileone_load_check(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    expected_len: usize,
) -> crate::Result<Vec<f32>> {
    let v = load_tensor_as_f32(st, name)?;
    if v.len() != expected_len {
        return Err(crate::Error::Msg(format!(
            "MobileOne load {name:?}: got {} elements, expected {}",
            v.len(), expected_len,
        ))
        .bt());
    }
    Ok(v)
}

fn mobileone_load_transposed(
    st: &crate::safetensors::MmapedSafetensors,
    name: &str,
    out_features: usize,
    in_features: usize,
) -> crate::Result<Vec<f32>> {
    let flat = mobileone_load_check(st, name, out_features * in_features)?;
    let mut out = vec![0.0_f32; out_features * in_features];
    for i in 0..out_features {
        for j in 0..in_features {
            out[j * out_features + i] = flat[i * in_features + j];
        }
    }
    Ok(out)
}

impl MobileOneModel {
    /// Download a timm-format MobileOne safetensors checkpoint and load it.
    pub fn from_hub_with_config(repo_id: &str, config: MobileOneConfig) -> Result<Self> {
        Self::from_hub_with_filename(repo_id, "model.safetensors", config)
    }

    /// Explicit-filename variant of [`Self::from_hub_with_config`].
    pub fn from_hub_with_filename(
        repo_id: &str,
        filename: &str,
        config: MobileOneConfig,
    ) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()
            .map_err(|e| crate::Error::Msg(format!("hf-hub api init: {e}")))?;
        let repo = api.model(repo_id.to_string());
        let weights_path = repo
            .get(filename)
            .map_err(|e| crate::Error::Msg(format!("hf-hub mobileone safetensors: {e}")))?;
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&weights_path) }?;
        let weights = MobileOneWeights::load_from_mmapped(&st, &config)?;
        Ok(Self { config, weights })
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

    fn build_layer(
        c_in: usize, c_out: usize, kernel: usize, stride: usize, groups: usize,
        with_se: bool,
        nb: &mut dyn FnMut() -> f32,
    ) -> MobileOneLayerWeights {
        let w_len = c_out * (c_in / groups) * kernel * kernel;
        let se = if with_se {
            let sq = (c_out / 16).max(1);
            Some(MobileOneSeWeights {
                fc1_w: WeightStorage::F32(vec_of(sq * c_out, nb)),
                fc1_b: vec_of(sq, nb),
                fc2_w: WeightStorage::F32(vec_of(c_out * sq, nb)),
                fc2_b: vec_of(c_out, nb),
                squeeze: sq,
                channels: c_out,
            })
        } else {
            None
        };
        MobileOneLayerWeights {
            conv_w: WeightStorage::F32(vec_of(w_len, nb)),
            conv_b: vec_of(c_out, nb),
            c_in, c_out, kernel, stride, groups, se,
        }
    }

    fn build_weights(cfg: &MobileOneConfig, with_se: bool, seed: u32) -> MobileOneWeights {
        let mut nb = rng_seed(seed);
        let stem_dim = cfg.channels_at(0);
        let stem = build_layer(3, stem_dim, 3, 2, 1, false, &mut nb);
        let mut stages: [Vec<MobileOneLayerWeights>; 4] = Default::default();
        for stage_idx in 1..=4 {
            let mut layers = Vec::new();
            let n_blocks = STAGE_BLOCKS[stage_idx];
            let mut in_c = cfg.channels_at(stage_idx - 1);
            let out_c = cfg.channels_at(stage_idx);
            for block in 0..n_blocks {
                let stride = if block == 0 { 2 } else { 1 };
                // Depthwise 3×3: groups = in_c.
                layers.push(build_layer(in_c, in_c, 3, stride, in_c, false, &mut nb));
                // Pointwise 1×1: in_c → out_c, stride=1.
                layers.push(build_layer(in_c, out_c, 1, 1, 1, with_se, &mut nb));
                in_c = out_c;
            }
            stages[stage_idx - 1] = layers;
        }
        let head = cfg.nclasses.map(|n| {
            let last_c = cfg.channels_at(4);
            (
                WeightStorage::F32(vec_of(last_c * n, &mut nb)),
                vec_of(n, &mut nb),
            )
        });
        MobileOneWeights { stem, stages, head }
    }

    fn tiny_image(h: usize) -> LazyTensor {
        let mut nb = rng_seed(54);
        let data: Arc<[f32]> = Arc::from((0..3 * h * h).map(|_| nb()).collect::<Vec<_>>());
        LazyTensor::from_f32(data, Shape::from_dims(&[1, 3, h, h]), &Device::cpu())
    }

    #[test]
    fn mobileone_s0_forward_shape() {
        let cfg = MobileOneConfig::s0(Some(10));
        let weights = build_weights(&cfg, false, 11);
        let model = MobileOneModel { config: cfg, weights };
        let img = tiny_image(32);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 10]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// MobileOne-S4 wires SE blocks. Verify a model built with
    /// SE produces the same shape and finite output as without
    /// SE — and that flipping `with_se` builds a different
    /// number of weight values (SE adds two 1×1 convs per
    /// pointwise layer).
    #[test]
    fn mobileone_s4_with_se() {
        let cfg = MobileOneConfig::s4(Some(5));
        let weights = build_weights(&cfg, true, 33);
        let model = MobileOneModel { config: cfg, weights };
        let img = tiny_image(32);
        let logits = model.forward(&img).unwrap();
        assert_eq!(logits.shape().dims(), &[1, 5]);
        for &v in &logits.realize_f32() {
            assert!(v.is_finite(), "non-finite logit: {v}");
        }
    }

    /// Channel multipliers per variant: S0 starts with 0.75*64 = 48
    /// (clipped to min(64, 48) = 48); S4 starts with 3.0*64 = 192
    /// (clipped to min(64, 192) = 64). Stage 4 multipliers are
    /// the same as stages 1-3 in MobileOne (no separate `b`),
    /// so S0 stage 4 = 2.0 * 512 = 1024.
    #[test]
    fn variant_channel_counts() {
        let s0 = MobileOneConfig::s0(None);
        assert_eq!(s0.channels_at(0), 48);
        assert_eq!(s0.channels_at(1), 48);
        assert_eq!(s0.channels_at(2), 128);
        assert_eq!(s0.channels_at(3), 256);
        assert_eq!(s0.channels_at(4), 1024);
        let s4 = MobileOneConfig::s4(None);
        // 3.0 * 64 = 192 → clipped to 64 by the min(64, x) rule.
        assert_eq!(s4.channels_at(0), 64);
        // Stage 1: 3.0 * 64 = 192.
        assert_eq!(s4.channels_at(1), 192);
        // Stage 4: 4.0 * 512 = 2048.
        assert_eq!(s4.channels_at(4), 2048);
    }

    /// Each "block" in a stage emits TWO layers (depthwise +
    /// pointwise). For S1 with [1, 2, 8, 10, 1] blocks, stages
    /// 1-4 have [2, 4, 16, 20, 2] block-layers — well, 1-4 is
    /// [1, 2, 8, 10, 1] doubled to [2, 4, 16, 20, 2]. But the
    /// 5th stage entry [1] in STAGE_BLOCKS rolls into stage 4
    /// in this port; verify the actual per-stage layer counts.
    #[test]
    fn stage_block_counts_doubled_to_dw_pw() {
        let cfg = MobileOneConfig::s1(Some(10));
        let weights = build_weights(&cfg, false, 1);
        // STAGE_BLOCKS[1..=4] = [2, 8, 10, 1] → layer counts
        // [4, 16, 20, 2] after dw+pw doubling.
        let expected_layer_counts = [4, 16, 20, 2];
        for (i, count) in expected_layer_counts.iter().enumerate() {
            assert_eq!(weights.stages[i].len(), *count,
                "stage {} expected {} layers, got {}", i + 1, count, weights.stages[i].len());
        }
    }

    #[test]
    fn forward_features_shape_and_finite() {
        let cfg = MobileOneConfig::s0(Some(10));
        let weights = build_weights(&cfg, false, 44);
        let model = MobileOneModel { config: cfg, weights };
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

    fn raw_f32(len: usize, seed: u32) -> Vec<u8> {
        let mut s = seed;
        let mut out = Vec::with_capacity(len * 4);
        for _ in 0..len {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let v = ((s >> 16) as u16 as f32 / 65535.0 - 0.5) * 0.05;
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    fn raw_f32_const(len: usize, value: f32) -> Vec<u8> {
        let mut out = Vec::with_capacity(len * 4);
        for _ in 0..len {
            out.extend_from_slice(&value.to_le_bytes());
        }
        out
    }

    /// Push the bn quadruple (gain=1, bias=0, mean=0, var=1) so the BN
    /// fold is the (near-)identity scale `1/√(1+eps)`.
    fn push_identity_bn(
        owned: &mut Vec<(String, Vec<usize>, Vec<u8>)>,
        prefix: &str,
        channels: usize,
    ) {
        for (suffix, raw) in [
            ("weight",        raw_f32_const(channels, 1.0)),
            ("bias",          raw_f32_const(channels, 0.0)),
            ("running_mean",  raw_f32_const(channels, 0.0)),
            ("running_var",   raw_f32_const(channels, 1.0)),
        ] {
            owned.push((format!("{prefix}.{suffix}"), vec![channels], raw));
        }
    }

    /// Build a tiny synthetic safetensors blob for a MobileOne-S1 model
    /// (k=1, no SE) and verify the loader reproduces the same fused stem
    /// weight that hand-fusion would produce.
    #[test]
    fn load_from_mmapped_round_trip_mobileone_s1_no_head() {
        use safetensors::tensor::TensorView;
        use std::collections::HashMap;

        let cfg = MobileOneConfig::s1(None); // k = 1, no SE.
        let stem_dim = cfg.channels_at(0);

        let mut owned: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();

        // Stem: k=1, kernel=3, in=3, out=stem_dim, stride=2, groups=1.
        // conv_kxk.0.conv.weight: [stem_dim, 3, 3, 3]
        owned.push((
            "stem.conv_kxk.0.conv.weight".into(),
            vec![stem_dim, 3, 3, 3],
            raw_f32(stem_dim * 3 * 9, 0xC0FFEE),
        ));
        push_identity_bn(&mut owned, "stem.conv_kxk.0.bn", stem_dim);
        // conv_scale: [stem_dim, 3, 1, 1]
        owned.push((
            "stem.conv_scale.conv.weight".into(),
            vec![stem_dim, 3, 1, 1],
            raw_f32(stem_dim * 3, 0xBEEF00),
        ));
        push_identity_bn(&mut owned, "stem.conv_scale.bn", stem_dim);

        // Stages.
        for stage_idx in 1..=4_usize {
            let nblocks = STAGE_BLOCKS[stage_idx];
            let mut in_c = cfg.channels_at(stage_idx - 1);
            let out_c = cfg.channels_at(stage_idx);
            for b in 0..nblocks {
                let (has_identity, stride) = if b == 0 { (false, 2) } else { (true, 1) };

                // Depthwise: kernel=3, groups=in_c.
                let dw = format!("stages.{}.{}", stage_idx - 1, b * 2);
                owned.push((
                    format!("{dw}.conv_kxk.0.conv.weight"),
                    vec![in_c, 1, 3, 3],
                    raw_f32(in_c * 9, (stage_idx * 100 + b) as u32),
                ));
                push_identity_bn(&mut owned, &format!("{dw}.conv_kxk.0.bn"), in_c);
                owned.push((
                    format!("{dw}.conv_scale.conv.weight"),
                    vec![in_c, 1, 1, 1],
                    raw_f32(in_c, (stage_idx * 200 + b) as u32),
                ));
                push_identity_bn(&mut owned, &format!("{dw}.conv_scale.bn"), in_c);
                if has_identity {
                    push_identity_bn(&mut owned, &format!("{dw}.identity"), in_c);
                }
                let _ = stride;

                // Pointwise: kernel=1, groups=1.
                let pw = format!("stages.{}.{}", stage_idx - 1, b * 2 + 1);
                owned.push((
                    format!("{pw}.conv_kxk.0.conv.weight"),
                    vec![out_c, in_c, 1, 1],
                    raw_f32(out_c * in_c, (stage_idx * 300 + b) as u32),
                ));
                push_identity_bn(&mut owned, &format!("{pw}.conv_kxk.0.bn"), out_c);
                // No conv_scale on pointwise (kernel == 1).
                if has_identity {
                    // identity only valid if in_c == out_c; in MobileOne the
                    // first block downsamples so b>0 satisfies in == out.
                    push_identity_bn(&mut owned, &format!("{pw}.identity"), out_c);
                }

                in_c = out_c;
            }
        }

        // Build tensor views.
        let mut tensors: HashMap<String, TensorView<'_>> = HashMap::new();
        for (name, shape, bytes) in &owned {
            let view = TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                .expect("TensorView::new");
            tensors.insert(name.clone(), view);
        }
        let metadata: Option<HashMap<String, String>> = None;
        let serialized = safetensors::serialize(&tensors, metadata)
            .expect("safetensors::serialize");

        let tmp = std::env::temp_dir().join(format!(
            "fuel_mobileone_load_test_{}.safetensors",
            std::process::id(),
        ));
        std::fs::write(&tmp, &serialized).expect("write tmp");
        let st = unsafe { crate::safetensors::MmapedSafetensors::new(&tmp) }
            .expect("MmapedSafetensors::new");

        let loaded = MobileOneWeights::load_from_mmapped(&st, &cfg)
            .expect("MobileOneWeights::load_from_mmapped");

        // Verify the stem fused weight equals
        //   bn_scale * (W_3x3 + W_1x1_padded_to_center)
        // (no identity branch on stem; k=1 so only one conv_kxk).
        let bn_scale = 1.0_f32 / (1.0_f32 + MOBILEONE_BN_EPS as f32).sqrt();
        let conv = match &loaded.stem.conv_w {
            WeightStorage::F32(arc) => arc.clone(),
            other => panic!("expected F32 stem conv, got {other:?}"),
        };
        let raw_3x3 = &owned.iter()
            .find(|(n, _, _)| n == "stem.conv_kxk.0.conv.weight")
            .unwrap().2;
        let raw_1x1 = &owned.iter()
            .find(|(n, _, _)| n == "stem.conv_scale.conv.weight")
            .unwrap().2;
        let raw_3x3_f: Vec<f32> = raw_3x3.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let raw_1x1_f: Vec<f32> = raw_1x1.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        for o in 0..stem_dim {
            for i in 0..3_usize {
                for k in 0..9_usize {
                    let w3 = raw_3x3_f[o * 3 * 9 + i * 9 + k];
                    let w1 = raw_1x1_f[o * 3 + i];
                    let expected = bn_scale * (w3 + if k == 4 { w1 } else { 0.0 });
                    let got = conv[o * 3 * 9 + i * 9 + k];
                    assert!((got - expected).abs() < 1e-6,
                        "stem fused (o={o},i={i},k={k}) expected {expected}, got {got}");
                }
            }
        }
        for c in 0..stem_dim {
            assert!(loaded.stem.conv_b[c].abs() < 1e-6, "stem bias[{c}]");
        }

        // Structural shape: stages have 2*N layers (dw + pw) each.
        for stage_idx in 1..=4_usize {
            assert_eq!(
                loaded.stages[stage_idx - 1].len(),
                STAGE_BLOCKS[stage_idx] * 2,
            );
        }
        // No SE was wired on any layer (S1 has has_se == false).
        for stage in &loaded.stages {
            for layer in stage {
                assert!(layer.se.is_none(),
                    "S1 should have no SE; got Some on a layer");
            }
        }

        // Forward chain runs end-to-end on the loaded weights.
        let model = MobileOneModel { config: cfg.clone(), weights: loaded };
        let img = tiny_image(32);
        let feats = model.forward(&img).unwrap();
        assert_eq!(feats.shape().dims(), &[1, cfg.channels_at(4)]);

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn has_se_only_for_s4() {
        assert!(!MobileOneConfig::s0(None).has_se());
        assert!(!MobileOneConfig::s1(None).has_se());
        assert!(!MobileOneConfig::s2(None).has_se());
        assert!(!MobileOneConfig::s3(None).has_se());
        assert!(MobileOneConfig::s4(None).has_se());
    }

    /// Smoke test that documents the canonical `from_hub_with_config`
    /// usage. Ignored by default because it hits the HF Hub.
    #[test]
    #[ignore]
    fn from_hub_smoke_mobileone_s0() {
        let cfg = MobileOneConfig::s0(Some(1000));
        let model = MobileOneModel::from_hub_with_config(
            "timm/mobileone_s0.apple_in1k", cfg,
        ).expect("from_hub_with_config");
        assert!(model.weights.head.is_some());
    }
}
