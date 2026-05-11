//! FlashAttn — multi-head scaled-dot-product attention with
//! FlashAttention-shaped kernel hooks. Phase 7.6 step 4 (continued —
//! eighth op migrated).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   panicking `decompose`, stubbed pattern).
//!
//! Inputs: `[q, k, v, optional alibi_slopes]`.
//!   - `q`: `[B, Hq, Sq, D]`
//!   - `k`: `[B, Hkv, Sk, D]`
//!   - `v`: `[B, Hkv, Sk, D]`
//!   - `alibi_slopes` (optional): `[Hq]`
//!
//! Output: same shape as `q` (`[B, Hq, Sq, D]`).
//!
//! ## Architectural note — no primitive decomposition (yet)
//!
//! Attention does have a primitive decomposition (`matmul → softmax →
//! matmul`, with masking + scaling), but FlashAttn's value is
//! specifically that it *avoids* materializing the `[B, Hq, Sq, Sk]`
//! attention matrix — a primitive lowering would defeat the purpose.
//! Backends without a flash-attention kernel route through
//! `GraphExecutor::cpu_fallback` to the reference naive-attention
//! implementation (which does decompose internally). A graph-level
//! `decompose` to a primitive subgraph would be a footgun: it would
//! either reproduce the very memory blowup FlashAttn exists to avoid,
//! or pretend the primitive form is equivalent when it isn't (the
//! tiled softmax in the kernel produces different numerics than the
//! naive form).
//!
//! Backward is not yet implemented (panic stub in `Tensor::backward`);
//! the FlashAttn-shaped backward is a separate algorithm (the
//! "recompute" variant in the FlashAttention paper) and lands when a
//! consumer needs differentiable attention. Today `BackwardKind::
//! NotDifferentiable` reflects runtime behavior.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for FlashAttn.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::FLASH_ATTN,
        name:       "FlashAttn",
        family:     FusedOpFamily::Attention,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
    }
}

/// Shape rule: output shape equals input 0 (`q`).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert!(
        input_shapes.len() == 4 || input_shapes.len() == 5,
        "FlashAttn takes 4 or 5 inputs (q, k, v, [softmax_lse], [alibi])",
    );
    input_shapes[0].clone()
}

/// Dtype rule: output dtype equals input 0 (`q`).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert!(
        input_dtypes.len() == 4 || input_dtypes.len() == 5,
        "FlashAttn takes 4 or 5 inputs",
    );
    input_dtypes[0]
}

/// See module preamble — FlashAttn deliberately has no primitive
/// decomposition exposed at the registry layer.
pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "flash_attn::decompose: FlashAttn has no registry-layer \
         decomposition. Its value is avoiding the materialized \
         [B, Hq, Sq, Sk] attention matrix that a primitive lowering \
         would reintroduce; the cpu_fallback path uses the reference \
         naive-attention implementation as the cross-backend safety \
         net. See module docs for the full rationale.",
    );
}

/// Matcher stub — FlashAttn nodes originate from
/// `Tensor::flash_attn`-style builders, not from user-decomposed
/// `matmul + softmax + matmul` patterns. Recognizing the latter as
/// fusion-into-FlashAttn would require careful tolerance handling
/// (the tiled-softmax numerics aren't bit-identical to the naive
/// form) and isn't on the step-4 critical path.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
