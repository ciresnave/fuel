//! Nf4Matmul — bitsandbytes-style 4-bit NormalFloat quantized matrix
//! multiply. Fifth FusedOpRegistry entry from the re-framed CPU
//! OpKind coverage plan; the only one whose mechanical shape diverges
//! from the FSCE / Mamba trio (new dtype-level quant format + new
//! 3-input fused-matmul shape).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   panicking `decompose`, stubbed pattern).
//!
//! Inputs: `[activations, w_packed, absmax]`.
//!   - `activations`: `[..., M, K]` — caller's dtype (F32/F16/BF16
//!     in v1).
//!   - `w_packed`:    `[N, K/2]` U8 — two NF4 codes per byte; `K`
//!     must be even. Lower nibble at column `k_byte` holds the code
//!     for `k = 2·k_byte`; upper nibble holds `k = 2·k_byte + 1`.
//!     This matches the bitsandbytes convention for the standard
//!     K-fastest packing.
//!   - `absmax`:      `[N, K/block_size]` F32 — per-output-row,
//!     per-block scale. `K` must be a multiple of `block_size`
//!     (typically 64 in bitsandbytes).
//!
//! Output: `[..., M, N]` matching the activations' dtype.
//!
//! ## NF4 NormalFloat lookup table
//!
//! The 16 NormalFloat values [-1, -0.696, …, +1] (the inverse-CDF
//! quantiles of the standard normal that minimize the expected
//! quantization error for N(0, 1)-distributed weights) are **baked
//! into the kernel** — not a runtime input. Modifying them would
//! mean a different quantization format entirely.
//!
//! ## Why a new fused op (not extending QMATMUL)
//!
//! [`super::qmatmul`] takes a single `w_q_bytes` input that holds a
//! self-contained block stream (per GGUF / llama.cpp's `BlockQ*`
//! convention: each block embeds its own scale). NF4 splits weight
//! and scales into **two separate tensors** (the packed codes and
//! the absmax scales), which doesn't fit QMATMUL's single-input
//! shape. Adding NF4 as a `QuantType` variant would require
//! special-casing the input count throughout the dispatch path —
//! more disruptive than just adding a sibling fused op.
//!
//! ## Architectural note — no primitive decomposition
//!
//! Same precedent as [`super::qmatmul`]: the fused dequant-in-kernel
//! design exists specifically to avoid the dequant + matmul DRAM
//! round-trip. Exposing that round-trip as a registry-layer
//! decomposition would defeat the point. [`decompose`] panics;
//! `cpu_fallback` handles backends without a native kernel.
//!
//! ## Why `BackwardKind::NotDifferentiable`
//!
//! NF4 is an inference format. The weight is frozen (the U8 byte
//! stream isn't a smooth function of any continuous parameter), and
//! the activation gradient via "dequantize then standard matmul" is
//! the wrong recipe (any caller wanting that should use F32 weights
//! to begin with). Mirrors QMATMUL's same decision.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_ir::{DType, Shape};

/// Metadata-side registry entry for Nf4Matmul.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::NF4_MATMUL,
        name:       "Nf4Matmul",
        family:     FusedOpFamily::Quantized,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Output shape rule: `[..., M, N]` where M is activations' second-
/// to-last dim and N is the weight's first dim (per
/// `w_packed: [N, K/2]`).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 3,
        "Nf4Matmul takes 3 inputs (activations, w_packed, absmax)",
    );
    let a_dims = input_shapes[0].dims();
    let w_dims = input_shapes[1].dims();
    debug_assert!(
        a_dims.len() >= 2,
        "Nf4Matmul: activations must be rank ≥ 2, got {a_dims:?}"
    );
    debug_assert_eq!(
        w_dims.len(), 2,
        "Nf4Matmul: w_packed must be rank 2 [N, K/2], got {w_dims:?}"
    );
    let n = w_dims[0];
    let mut out_dims: Vec<usize> = a_dims[..a_dims.len() - 1].to_vec();
    out_dims.push(n);
    Shape::from_dims(&out_dims)
}

/// Dtype rule: output dtype matches input 0 (activations). The
/// U8 w_packed and F32 absmax don't influence the output dtype.
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 3,
        "Nf4Matmul takes 3 inputs (activations, w_packed, absmax)",
    );
    input_dtypes[0]
}

/// See module preamble — Nf4Matmul deliberately has no primitive
/// decomposition (mirrors QMatMul's precedent). `cpu_fallback`
/// handles backends without a native kernel.
pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "nf4_matmul::decompose: Nf4Matmul has no registry-layer \
         decomposition. The fused dequant-in-kernel design exists \
         specifically to avoid the dequant + matmul DRAM round-trip; \
         exposing that round-trip as a registry-layer lowering would \
         defeat the point. cpu_fallback handles backends without a \
         native kernel. See qmatmul::decompose for the same precedent.",
    );
}

/// Matcher stub — Nf4Matmul nodes originate from the explicit
/// `Tensor::nf4_matmul` builder. There's no primitive subgraph to
/// recognize (the NF4 unpacking + lookup-table dequant doesn't
/// exist as fuel-graph primitives).
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
