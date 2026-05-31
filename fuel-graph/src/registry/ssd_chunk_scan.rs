//! SsdChunkScan — Mamba-2's State-Space Duality chunked scan
//! (forward). Fourth FusedOpRegistry entry added by the re-framed
//! CPU OpKind coverage plan; completes the Mamba-adjacent trio
//! (CausalConv1d + SelectiveScan + SsdChunkScan).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   panicking `decompose`, stubbed pattern).
//!
//! Inputs: `[x, dt, a, b, c]` — matches baracuda's
//! `ssd_chunk_scan_*_run` 5-input signature exactly (no optional
//! inputs in baracuda's API).
//!   - `x`:   `[batch, seqlen, heads, head_dim]` — multi-head input.
//!   - `dt`:  `[batch, seqlen, heads]` — per-step state update rate.
//!   - `a`:   `[heads]` — per-head scalar log A.
//!   - `b`:   `[batch, seqlen, heads, state_dim]` — selective input.
//!   - `c`:   `[batch, seqlen, heads, state_dim]` — selective output.
//!
//! Output: `y: [batch, seqlen, heads, head_dim]`. dtype matches input
//! dtype (uniform F32 in v1).
//!
//! ## v1 scope: single-chunk only (`chunk_size == seqlen`)
//!
//! The full chunked SSD algorithm splits the sequence into
//! non-overlapping blocks of `chunk_size` tokens, runs an intra-chunk
//! per-token recurrence, and propagates state across chunks via a
//! second recurrence over chunk-final states. fuel-transformers'
//! [mamba2.rs:408-512](../../../fuel-transformers/src/models/llm/mamba2.rs#L408-L512)
//! `ssd_chunked` is the textbook composition (~100 LOC of 6D
//! broadcast + permute + cumsum + segsum).
//!
//! When `chunk_size == seqlen`, the chunked algorithm degenerates to
//! a single intra-chunk recurrence — structurally identical to
//! [`super::selective_scan`] but with one extra leading dimension
//! (heads) and a per-head scalar `a` instead of `a: [dim, dstate]`.
//! v1's CPU kernel implements exactly this case; non-single-chunk
//! requests return a clear error. The multi-chunk inter-chunk-decay
//! algorithm is a follow-up — substantial enough to warrant its own
//! commit.
//!
//! ## v1 scope: y output only (no final_state)
//!
//! baracuda's `ssd_chunk_scan_*_run` signature ALREADY returns only
//! `y` (unlike `selective_scan_*_run` which returns y + last_state).
//! fuel-transformers' eager `ssd_chunked` wraps the bare scan with
//! `initial_state` input + `final_state` output for autoregressive
//! continuation. That wrapping is the caller's responsibility today;
//! v1 of the fused op mirrors baracuda's bare signature.
//!
//! ## Architectural note — no primitive decomposition
//!
//! Same precedent as [`super::selective_scan`] and
//! [`super::causal_conv1d`]: the recurrence is sequential, and
//! synthesizing it from primitives would yield `O(seqlen)` nodes
//! (or `O(seqlen × heads × head_dim × state_dim)` for the fully
//! unrolled form). [`decompose`] panics; backends without a native
//! kernel use the executor's `cpu_fallback` path.
//!
//! ## Why `BackwardKind::NotDifferentiable` for v1
//!
//! Mamba-2 inference is the only consumer surface today (and it's
//! on the eager Tensor path — see
//! `docs/session-prompts/mamba-eager-to-lazy-migration.md`).
//! baracuda's backward variant exists; wiring `SSD_CHUNK_SCAN_BACKWARD`
//! is mechanical when a training consumer materializes.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for SsdChunkScan.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::SSD_CHUNK_SCAN,
        name:       "SsdChunkScan",
        family:     FusedOpFamily::Forward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
    }
}

/// Output shape rule: `y: [batch, seqlen, heads, head_dim]` — same
/// as `x`'s shape (input 0).
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 5,
        "SsdChunkScan takes 5 inputs (x, dt, a, b, c)",
    );
    input_shapes[0].clone()
}

/// Dtype rule: output matches `x`'s dtype (input 0).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 5,
        "SsdChunkScan takes 5 inputs",
    );
    input_dtypes[0]
}

/// See module preamble — SsdChunkScan deliberately has no primitive
/// decomposition. The `cpu_fallback` path handles backends without
/// a native kernel.
pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "ssd_chunk_scan::decompose: SsdChunkScan has no registry-layer \
         decomposition. The chunked SSD recurrence is sequential at the \
         per-token level (and inter-chunk state passing adds another \
         sequential layer); synthesizing it from primitives would yield \
         O(seqlen) nodes minimum. Backends without a native SsdChunkScan \
         kernel use the executor's cpu_fallback path. See \
         selective_scan::decompose for the same precedent.",
    );
}

/// Matcher stub — SsdChunkScan nodes originate from the explicit
/// `Tensor::ssd_chunk_scan` builder. The 100+ primitive subgraph in
/// fuel-transformers' eager `ssd_chunked` is too complex to pattern-
/// match conservatively.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
