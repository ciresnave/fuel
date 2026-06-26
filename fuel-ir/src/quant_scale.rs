//! Scale-granularity tagging for dynamic quantization formats.
//!
//! Some quant formats (FP8 E4M3 / E5M2, Int8 dynamic, Int4
//! dynamic) compute their scale factors at quantize time, leaving
//! the caller free to choose how coarse or fine-grained the
//! scales are. The standard granularities are:
//!
//! | Granularity   | Scale shape       | Use case                                                |
//! |---------------|-------------------|---------------------------------------------------------|
//! | `PerTensor`   | `f32[1]`          | Weights with narrow distribution (post-normalization).  |
//! | `PerToken`    | `f32[n_rows]`     | Activations (per-token / per-row distribution varies).  |
//! | `PerChannel`  | `f32[n_cols]`     | Weights (per-output-channel distribution varies).       |
//!
//! Per-tensor is the cheapest (one reduction, one scale) but
//! gives the worst accuracy on activations where row-to-row
//! distributions vary. Per-token gives much better activation
//! accuracy at the cost of `n_rows` extra reductions; per-channel
//! is the analog for weights. Production LLM inference stacks
//! (vLLM, TGI, TRT-LLM) typically pair per-token activations with
//! per-channel weights — that combination keeps the GEMM
//! accumulator's effective bit-width higher than any per-tensor
//! variant.
//!
//! **NOT applicable to static-quantization formats.** GGUF
//! (Q4_0 / Q4_K_M / Q5_0 / Q6_K / Q8_0), AWQ, Marlin, and NF4
//! have their scale layout baked into the binary format —
//! typically per-block (32 / 64 / 128 elements along K). Those
//! formats don't expose `ScaleGranularity` because there's no
//! free parameter to expose. Use this enum only on the dynamic-
//! quant API surface.
//!
//! See the 2026-05-29 architectural review in
//! `project_xn_audit_2026_05_29` for the broader rationale.

/// Where the scale factor lives relative to the tensor it
/// quantizes. See module docs for the per-format applicability.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScaleGranularity {
    /// One scalar scale for the entire tensor. Cheapest reduction;
    /// most lossy on tensors with heterogeneous per-row /
    /// per-column distributions.
    PerTensor,
    /// One scale per row (typically per-token in inference).
    /// Standard for activation quantization in autoregressive
    /// decode loops.
    PerToken,
    /// One scale per column (typically per-output-channel for
    /// weight matrices). Standard for weight quantization on
    /// linear / matmul ops.
    PerChannel,
}

impl ScaleGranularity {
    /// The number of `f32` scale values needed for a `[rows, cols]`
    /// tensor under this granularity.
    pub fn scale_count(&self, rows: usize, cols: usize) -> usize {
        match self {
            Self::PerTensor => 1,
            Self::PerToken => rows,
            Self::PerChannel => cols,
        }
    }

    /// Short stable identifier used in symbol names (e.g.
    /// `"per_tensor"`, `"per_token"`, `"per_channel"`) — matches
    /// the upstream baracuda / vLLM convention.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::PerTensor => "per_tensor",
            Self::PerToken => "per_token",
            Self::PerChannel => "per_channel",
        }
    }
}

impl std::fmt::Display for ScaleGranularity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Activation × weight scale-granularity pairing for a quantized
/// matmul / linear op. Used by dispatch code that picks between
/// per-(act, weight) kernel variants.
///
/// Naming convention follows baracuda's `<format>_matmul_act_<act>
/// _weight_<weight>_run` symbol family. Not every pair has a
/// dedicated kernel — typical builds ship the
/// `(PerToken, PerChannel)` pair as the "production decode" path,
/// `(PerTensor, PerTensor)` as the "simple smoke path", and may or
/// may not ship the cross pairings depending on whether anyone
/// has asked.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScalePair {
    pub activation: ScaleGranularity,
    pub weight: ScaleGranularity,
}

impl ScalePair {
    pub const fn new(activation: ScaleGranularity, weight: ScaleGranularity) -> Self {
        Self { activation, weight }
    }

    /// Production-decode default: per-token activations, per-channel
    /// weights. Best accuracy for LLM autoregressive decode.
    pub const PRODUCTION_DECODE: Self = Self {
        activation: ScaleGranularity::PerToken,
        weight: ScaleGranularity::PerChannel,
    };

    /// Per-tensor activations + per-tensor weights. Cheapest;
    /// matches CUTLASS's default FP8 GEMM path.
    pub const SIMPLE: Self = Self {
        activation: ScaleGranularity::PerTensor,
        weight: ScaleGranularity::PerTensor,
    };
}

impl std::fmt::Display for ScalePair {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "act_{}_weight_{}", self.activation, self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_count_matches_granularity() {
        let g = ScaleGranularity::PerTensor;
        assert_eq!(g.scale_count(100, 200), 1);
        let g = ScaleGranularity::PerToken;
        assert_eq!(g.scale_count(100, 200), 100);
        let g = ScaleGranularity::PerChannel;
        assert_eq!(g.scale_count(100, 200), 200);
    }

    #[test]
    fn display_strings_match_convention() {
        assert_eq!(ScaleGranularity::PerTensor.to_string(), "per_tensor");
        assert_eq!(ScaleGranularity::PerToken.to_string(), "per_token");
        assert_eq!(ScaleGranularity::PerChannel.to_string(), "per_channel");
    }

    #[test]
    fn scale_pair_display_matches_baracuda_symbol_convention() {
        let p = ScalePair::PRODUCTION_DECODE;
        assert_eq!(p.to_string(), "act_per_token_weight_per_channel");
        let p = ScalePair::SIMPLE;
        assert_eq!(p.to_string(), "act_per_tensor_weight_per_tensor");
    }

    #[test]
    fn presets_have_expected_pairings() {
        assert_eq!(ScalePair::PRODUCTION_DECODE.activation, ScaleGranularity::PerToken);
        assert_eq!(ScalePair::PRODUCTION_DECODE.weight, ScaleGranularity::PerChannel);
        assert_eq!(ScalePair::SIMPLE.activation, ScaleGranularity::PerTensor);
        assert_eq!(ScalePair::SIMPLE.weight, ScaleGranularity::PerTensor);
    }
}
