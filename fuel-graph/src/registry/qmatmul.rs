//! QMatMul — quantized matrix multiply `C = A @ dequant(W_Q)`.
//! Phase 7.6 step 4 (continued — tenth op migrated; final step-4 op).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   self-returning `decompose` — a surfaced gap pending the
//!   decode-boundary-typing migration, see [`decompose`] — stubbed pattern).
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
use fuel_ir::{DType, Shape};

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

/// QMatMul is **not** a basis gap. Its primitive recipe is
/// `dequantize(w_q_bytes, quant_type) → matmul(a, ·)`, and the earlier claim
/// that the dequant needs three primitives Fuel lacks was **wrong** (scoped
/// 2026-07-25, cross-validated with KISS/kiss-ref/Baracuda). Correcting each:
///
/// - **Sub-byte quant unpack is power-of-2 arithmetic, not a bitwise
///   primitive.** The already-shipped `nf4_matmul` recipe unpacks packed
///   nibbles with `Cast`/`Floor`/`MulScalar`/`Sub` and **zero** bitwise ops;
///   Q4_0's `&0x0F`/`>>4` and the Q*_K 6-bit sub-scale unpacks are the same
///   power-of-2 arithmetic on integer fields `< 2^24` (F32-exact).
/// - **The GGML block / super-block layout is `Reshape` + `Slice`** — both
///   already in the basis.
/// - **The one byte-level step — recovering the embedded f16 block-scale from
///   its raw bytes — is hoisted to the DECODE BOUNDARY, not done in-graph.**
///   `w_q_bytes` is a **frozen loaded constant** (a GGUF weight, known at load;
///   never the output of a prior graph op), so per KISS §7.3-0002 the
///   reinterpret is structurally hoistable: the loader host-decodes each
///   block's f16 scale into a typed operand — exactly the `nf4_matmul`
///   `absmax`-as-separate-operand precedent — leaving the recipe pure
///   arithmetic. No in-graph byte-reinterpret, hence **no new primitive `Op`
///   (no `Bitcast`) is required.**
///
/// `decompose` still returns **self** today (a surfaced gap, never a crash)
/// because the recipe needs the node's inputs to include the host-decoded
/// scale operand — i.e. it awaits the **decode-boundary-typing migration**:
/// the `qmatmul` builder must bind the weight as typed operands (quant payload
/// + host-decoded f16 scales, `nf4_matmul`-style) so the recipe can be pure
/// `Reshape`/`Slice`/arithmetic + `matmul`. Until that lands, the single U32
/// buffer can't be split in-graph without the very byte-reinterpret we just
/// showed is unnecessary — so self-return is the correct interim posture.
/// `cpu_fallback` handles backends without a native kernel. (The
/// DRAM-round-trip-avoidance note below is a kernel-selection rationale —
/// performance, not a totality argument; the fused dequant-in-kernel arm stays
/// the cost-preferred cover, the decomposed form is the fallback + optimizer
/// lowering.)
pub fn decompose(_graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    id
}

/// Matcher stub — QMatMul nodes originate from explicit quantized
/// builders, not user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
