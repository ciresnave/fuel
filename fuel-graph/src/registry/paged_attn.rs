//! PagedAttn — paged-cache scaled-dot-product attention. Phase 7.6
//! step 4 (continued — ninth op migrated).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   panicking `decompose`, stubbed pattern).
//!
//! Inputs: `[q, k_cache, v_cache, block_table, context_lens, optional
//! alibi_slopes]`.
//!   - `q`:            `[B, Hq, Sq, D]`
//!   - `k_cache`:      `[num_blocks, block_size, Hkv, D]`
//!   - `v_cache`:      `[num_blocks, block_size, Hkv, D]`
//!   - `block_table`:  `[B, max_num_blocks_per_seq]` (u32)
//!   - `context_lens`: `[B]` (u32)
//!   - `alibi_slopes`: `[Hq]` (optional)
//!
//! Output: same shape as `q` (`[B, Hq, Sq, D]`).
//!
//! ## Architectural note — decode-only, non-differentiable
//!
//! PagedAttn is decode-side only by construction: the paged KV cache
//! has variable-length sequences and no training pass writes through
//! it. No gradient rule (matches the legacy `Op::PagedAttn { .. }`
//! arm in `Tensor::backward`, which panics). The registry entry's
//! `BackwardKind::NotDifferentiable` reflects this.
//!
//! No primitive decomposition exposed at the registry layer — same
//! rationale as FlashAttn (the paged-block traversal is the point of
//! the kernel; a "decompose to materialized k_cache + materialized
//! attention" lowering would defeat the design).

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for PagedAttn.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::PAGED_ATTN,
        name:       "PagedAttn",
        family:     FusedOpFamily::Attention,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: None,
    }
}

/// Shape rule: output shape equals input 0 (`q`).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert!(
        input_shapes.len() == 5 || input_shapes.len() == 6,
        "PagedAttn takes 5 or 6 inputs",
    );
    input_shapes[0].clone()
}

/// Dtype rule: output dtype equals input 0 (`q`).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert!(
        input_dtypes.len() == 5 || input_dtypes.len() == 6,
        "PagedAttn takes 5 or 6 inputs",
    );
    input_dtypes[0]
}

/// See module preamble — PagedAttn deliberately has no primitive
/// decomposition exposed at the registry layer.
pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "paged_attn::decompose: PagedAttn has no registry-layer \
         decomposition. The paged-block traversal is the kernel's \
         design point; a primitive lowering would defeat it. See \
         module docs.",
    );
}

/// Matcher stub — PagedAttn originates from explicit builders, not
/// user-decomposed forms.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
