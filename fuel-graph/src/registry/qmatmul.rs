//! QMatMul — quantized matrix multiply `C = A @ dequant(W_Q)`.
//! Phase 7.6 step 4 (continued — tenth op migrated; final step-4 op).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   panicking `decompose`, stubbed pattern).
//!
//! Inputs: `[a, w_q_bytes]`.
//!   - `a`:          `[..., M, K]` F32 activations
//!   - `w_q_bytes`:  U32-typed packed block stream for the `[N, K]`
//!     weight matrix (GGUF / llama.cpp convention).
//!
//! Output: `[..., M, N]` F32.
//!
//! ## Architectural note — frozen weights, non-differentiable
//!
//! QMatMul is the inference path for quantized model weights. The
//! weight tensor is frozen (the U32 byte stream isn't a smooth
//! function of any continuous parameter), and the activation gradient
//! isn't implemented today — matches the legacy `Op::QMatMul { .. }`
//! arm in `Tensor::backward`, which panics with a clear "use a
//! dequantize + standard matmul if you need gradients" message. The
//! registry entry's `BackwardKind::NotDifferentiable` reflects this.
//!
//! No primitive decomposition exposed at the registry layer — the
//! "dequantize then matmul" lowering would round-trip through F32 /
//! BF16 in DRAM (the very bandwidth waste QMatMul's fused
//! dequant-in-kernel design avoids). Backends without a native
//! QMatMul kernel fall back to the reference implementation through
//! `GraphExecutor::cpu_fallback`.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for QMatMul.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::QMATMUL,
        name:       "QMatMul",
        family:     FusedOpFamily::Quantized,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Shape rule: output is `[..., M, N]` where `M = a.shape[-2]` and
/// `N` is the weight's output dim from `FusedOpParams::QMatMul`.
fn shape_rule(input_shapes: &[Shape], params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 2,
        "QMatMul takes 2 inputs (a, w_q_bytes)",
    );
    let n = match params {
        FusedOpParams::QMatMul { n, .. } => *n,
        _ => panic!("qmatmul::shape_rule got non-QMatMul params: {params:?}"),
    };
    let a_dims = input_shapes[0].dims();
    let rank = a_dims.len();
    debug_assert!(rank >= 2, "QMatMul activations must be rank ≥ 2");
    let mut out_dims: Vec<usize> = a_dims[..rank - 1].to_vec();
    out_dims.push(n);
    Shape::from_dims(&out_dims)
}

/// Dtype rule: output dtype is F32 (the activations'). The U32 w_q
/// is opaque bytes — the dtype field on its node doesn't influence
/// QMatMul's output dtype.
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 2,
        "QMatMul takes 2 inputs",
    );
    input_dtypes[0]
}

/// QMatMul is a genuine **basis gap** (G2, 2026-06-20). Its primitive recipe
/// is `dequantize(w_q_bytes, quant_type) → matmul(a, ·)`, but the dequant needs
/// primitives Fuel lacks. The U32 `w_q_bytes` is an opaque GGUF/GGML `BlockQ*`
/// stream interleaving, per block: sub-byte-packed quants (4/5/2/3/6 bits) plus
/// f16 scale(s)/min(s) embedded as raw bytes (and, for the Q*_K super-blocks,
/// 6-bit-packed sub-block scales + high-bit masks). Expressing the dequant
/// needs (1) a **sub-byte bit-unpack** primitive (no BitShift/BitAnd/Unpack
/// exists; `Cast` converts whole dtypes only), (2) a **byte-reinterpret of an
/// offset slice** to recover the embedded f16 scales (`Cast` is value-
/// converting, not a bit-cast; `Slice` indexes logical typed elements, not byte
/// offsets), and (3) a GGML block/super-block layout op. `Gather`/`IndexSelect`
/// don't close the gap. Per G2 `decompose` is total + never-panic: with no
/// expressible recipe it returns **self** (a surfaced opaque-op gap), and
/// decomposes once those primitives land. `cpu_fallback` handles backends
/// without a native kernel. (The DRAM-round-trip-avoidance note is a kernel-
/// selection rationale — performance, not a totality argument.)
pub fn decompose(_graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    id
}

/// Matcher stub — QMatMul nodes originate from explicit quantized
/// builders, not user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
