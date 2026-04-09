//! Low-Rank Adaptation (LoRA) for linear layers.
//!
//! LoRA adds a trainable low-rank delta on top of a frozen base weight:
//!
//! ```text
//! y = base(x) + (x @ A^T @ B^T) * (alpha / rank)
//! ```
//!
//! where `A` is `(rank, in_features)` and `B` is `(out_features, rank)`.
//! `B` is initialized to zeros so the adapter starts as an identity
//! transformation; `A` is initialized with Kaiming normal.
//!
//! # Usage pattern
//!
//! Replace `candle_nn::linear` calls with [`lora_linear`] in any layer that
//! should be fine-tunable without updating the base weights. Train only the
//! `lora_A` and `lora_B` parameters by filtering them from the optimizer's
//! parameter list.
//!
//! # References
//!
//! Hu et al. (2021) — *LoRA: Low-Rank Adaptation of Large Language Models*
//! <https://arxiv.org/abs/2106.09685>

use candle::{Module, Result, Tensor};

use crate::{
    Linear, QuantizableLinear,
    init::{DEFAULT_KAIMING_NORMAL, ZERO},
};

/// A linear layer with a LoRA adapter.
///
/// The forward pass computes:
///
/// ```text
/// y = base(x) + (x @ lora_a^T @ lora_b^T) * scale
/// ```
///
/// where `scale = alpha / rank`.
///
/// # Example
///
/// ```rust,no_run
/// use candle::{Device, DType};
/// use candle_nn::{lora_linear, VarBuilder, VarMap, Module};
///
/// # fn main() -> candle::Result<()> {
/// # let device = Device::Cpu;
/// # let dtype = DType::F32;
/// # let in_dim = 128;
/// # let out_dim = 64;
/// # let rank = 8;
/// # let alpha = 16.0_f64;
/// let varmap = VarMap::new();
/// let vb = VarBuilder::from_varmap(&varmap, dtype, &device);
/// let layer = lora_linear(in_dim, out_dim, rank, alpha, vb)?;
///
/// let x = candle::Tensor::zeros((4, in_dim), dtype, &device)?;
/// let y = layer.forward(&x)?;  // shape: [4, out_dim]
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug)]
pub struct LoraLinear {
    base: QuantizableLinear,
    lora_a: Tensor,
    lora_b: Tensor,
    /// `alpha / rank`
    scale: f64,
}

impl LoraLinear {
    /// Create a `LoraLinear` directly from existing tensors.
    ///
    /// - `base` — the frozen base linear layer.
    /// - `lora_a` — shape `(rank, in_features)`.
    /// - `lora_b` — shape `(out_features, rank)`.
    /// - `alpha` — LoRA scaling numerator; scale = `alpha / rank`.
    pub fn new(base: QuantizableLinear, lora_a: Tensor, lora_b: Tensor, alpha: f64) -> Self {
        let rank = lora_a.dim(0).unwrap_or(1);
        let scale = alpha / rank as f64;
        Self {
            base,
            lora_a,
            lora_b,
            scale,
        }
    }

    /// Returns the scaling factor `alpha / rank`.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Returns a reference to the underlying base layer.
    pub fn base(&self) -> &QuantizableLinear {
        &self.base
    }

    /// Returns the `lora_a` matrix of shape `(rank, in_features)`.
    pub fn lora_a(&self) -> &Tensor {
        &self.lora_a
    }

    /// Returns the `lora_b` matrix of shape `(out_features, rank)`.
    pub fn lora_b(&self) -> &Tensor {
        &self.lora_b
    }

    /// Merge the LoRA delta into a plain [`Linear`] layer.
    ///
    /// Computes `merged_weight = base_weight + (lora_b @ lora_a) * scale`.
    /// The base bias (if any) is preserved unchanged.
    ///
    /// Returns a `Linear` layer with the adapted weight permanently baked in.
    /// This is useful before exporting a fine-tuned checkpoint so that the
    /// adapter overhead disappears at inference time.
    pub fn merge_weights(&self) -> Result<Linear> {
        // base_weight: (out_features, in_features)
        let base_weight = self.base.dequantized_weight()?;
        // lora_b @ lora_a: (out_features, rank) x (rank, in_features) = (out_features, in_features)
        let delta = self.lora_b.matmul(&self.lora_a)?;
        let merged = (base_weight + (delta * self.scale)?)?;
        let bias = match &self.base {
            QuantizableLinear::Float(l) => l.bias().cloned(),
            QuantizableLinear::Quantized(_) => None,
        };
        Ok(Linear::new(merged, bias))
    }
}

impl Module for LoraLinear {
    /// Run the adapted forward pass.
    ///
    /// For each input `x`:
    ///
    /// ```text
    /// y = base(x) + (x @ lora_a^T @ lora_b^T) * scale
    /// ```
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let base_out = self.base.forward(xs)?;
        // xs: (..., in_features)
        // xs @ lora_a^T: (..., rank)
        // (xs @ lora_a^T) @ lora_b^T: (..., out_features)
        let lora_out = xs.matmul(&self.lora_a.t()?)?.matmul(&self.lora_b.t()?)?;
        // scale and add
        (base_out + (lora_out * self.scale)?)?.to_dtype(xs.dtype()) // preserve input dtype
    }
}

// ─── Constructor functions ──────────────────────────────────────────────────

/// Build a `LoraLinear` layer, loading the base weight from `vb` and
/// allocating fresh (zero-initialised) LoRA adapter parameters.
///
/// The base weight is loaded from `vb` under the same keys used by
/// [`linear`](crate::linear::linear): `"weight"` and `"bias"`.
///
/// LoRA parameters are stored under:
/// - `vb.get("lora_A")` — shape `(rank, in_features)`, Kaiming normal init.
/// - `vb.get("lora_B")` — shape `(out_features, rank)`, zeros init.
///
/// # Arguments
///
/// - `in_dim` — input feature dimension.
/// - `out_dim` — output feature dimension.
/// - `rank` — LoRA rank `r`. Typical values: 4, 8, 16, 32.
/// - `alpha` — LoRA alpha. Scale factor = `alpha / rank`. Setting
///   `alpha = rank` gives `scale = 1.0`.
/// - `vb` — variable builder from which all tensors are loaded/created.
pub fn lora_linear(
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    alpha: f64,
    vb: crate::VarBuilder,
) -> Result<LoraLinear> {
    // Load the base linear layer (frozen — optimizer excludes these keys)
    let base_layer = crate::linear(in_dim, out_dim, vb.clone())?;
    _lora_from_base(
        QuantizableLinear::Float(base_layer),
        in_dim,
        out_dim,
        rank,
        alpha,
        vb,
    )
}

/// Build a `LoraLinear` wrapping a [`QuantizableLinear`] base layer.
///
/// Use this variant when the base layer was already loaded (e.g., from GGUF)
/// and you want to attach LoRA adapters. The LoRA parameters are allocated in
/// `vb` under `"lora_A"` and `"lora_B"`.
///
/// # Arguments
///
/// - `base` — the pre-loaded base layer (float or quantized).
/// - `in_dim` — input feature dimension (must match `base`).
/// - `out_dim` — output feature dimension (must match `base`).
/// - `rank` — LoRA rank.
/// - `alpha` — LoRA alpha.
/// - `vb` — variable builder for the LoRA parameters only.
pub fn lora_linear_with_base(
    base: QuantizableLinear,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    alpha: f64,
    vb: crate::VarBuilder,
) -> Result<LoraLinear> {
    _lora_from_base(base, in_dim, out_dim, rank, alpha, vb)
}

/// Build a `LoraLinear` that loads both the base weight and adapter parameters
/// from `vb` using the HuggingFace PEFT naming convention:
///
/// - `"base_layer.weight"` + `"base_layer.bias"` — base linear
/// - `"lora_A.weight"` — shape `(rank, in_features)`
/// - `"lora_B.weight"` — shape `(out_features, rank)`
///
/// This matches the file layout produced by `peft.LoraConfig` when saving
/// LoRA adapters to a HuggingFace repository.
pub fn lora_linear_peft(
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    alpha: f64,
    vb: crate::VarBuilder,
) -> Result<LoraLinear> {
    let base_layer = crate::linear(in_dim, out_dim, vb.pp("base_layer"))?;
    let lora_a =
        vb.pp("lora_A")
            .get_with_hints((rank, in_dim), "weight", DEFAULT_KAIMING_NORMAL)?;
    let lora_b = vb
        .pp("lora_B")
        .get_with_hints((out_dim, rank), "weight", ZERO)?;
    Ok(LoraLinear::new(
        QuantizableLinear::Float(base_layer),
        lora_a,
        lora_b,
        alpha,
    ))
}

// Internal helper shared by `lora_linear` and `lora_linear_with_base`.
fn _lora_from_base(
    base: QuantizableLinear,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
    alpha: f64,
    vb: crate::VarBuilder,
) -> Result<LoraLinear> {
    let lora_a = vb.get_with_hints((rank, in_dim), "lora_A", DEFAULT_KAIMING_NORMAL)?;
    let lora_b = vb.get_with_hints((out_dim, rank), "lora_B", ZERO)?;
    Ok(LoraLinear::new(base, lora_a, lora_b, alpha))
}
