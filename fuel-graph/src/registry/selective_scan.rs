//! SelectiveScan — Mamba-1's selective state-space-model scan. Third
//! FusedOpRegistry entry added by the re-framed CPU OpKind coverage
//! plan (after FusedSoftmaxCrossEntropy + CausalConv1d).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   panicking `decompose`, stubbed pattern).
//!
//! Inputs: `[u, delta, a, b, c]` (5 required; the optional `d_skip`,
//! `z`, `delta_bias` from baracuda's full signature are deferred to a
//! later sibling — see "v1 scope" below).
//!   - `u`:     `[batch, seqlen, dim]` — input sequence.
//!   - `delta`: `[batch, seqlen, dim]` — per-step state update rate
//!     (the "selective" part).
//!   - `a`:     `[dim, dstate]` — recurrence matrix.
//!   - `b`:     `[batch, seqlen, dstate]` — selective input matrix.
//!   - `c`:     `[batch, seqlen, dstate]` — selective output matrix.
//!
//! Output: `y: [batch, seqlen, dim]`. dtype matches input dtype
//! (uniform F32 in v1).
//!
//! The forward recurrence (per `(batch, time, dim)`):
//!
//! ```text
//!   d = softplus(delta[b,t,i])  if delta_softplus else delta[b,t,i]
//!   for j in 0..dstate:
//!     h[b,i,j] = exp(d * a[i,j]) * h[b,i,j] + d * b[b,t,j] * u[b,t,i]
//!   y[b,t,i] = sum_j(h[b,i,j] * c[b,t,j])
//! ```
//!
//! `h` is a per-batch / per-dim / per-dstate hidden-state accumulator,
//! initialized to zero at the start of the scan and threaded across
//! timesteps. The kernel allocates it internally — it's NOT exposed
//! as an input or output in v1.
//!
//! ## v1 scope
//!
//! - **Required inputs only**: `u, delta, a, b, c`. baracuda's full
//!   signature also accepts optional `d_skip: [dim]` (skip-connection),
//!   `z: [batch, seqlen, dim]` (gating, multiplied by SiLU(z) at end),
//!   and `delta_bias: [dim]` (added to delta before softplus). These
//!   are mechanical extensions when a consumer needs them.
//! - **`y` output only**: baracuda also produces `last_state: [batch,
//!   dim, dstate]` for autoregressive resumption. Multi-output ops
//!   don't have a clean shape in fuel-graph's single-output-per-node
//!   model today; adding a sibling `SELECTIVE_SCAN_LAST_STATE` op
//!   (same inputs, returns the final h-state) is the path forward when
//!   a real consumer materializes.
//! - **F32 only**: per-dtype siblings follow the FSCE/CausalConv1d
//!   precedent.
//!
//! ## Architectural note — no primitive decomposition
//!
//! Like CausalConv1d and Conv2D, SelectiveScan has no clean primitive
//! decomposition in fuel-graph's current Op set. The textbook scan
//! is a sequential recurrence with per-timestep state updates — even
//! if we synthesized it from primitives (`MatMul + Exp + Add` chains
//! per step), the resulting graph would have `O(seqlen)` nodes and
//! defeat any optimization pass. [`decompose`] panics with a clear
//! pointer; backends without a native kernel use the executor's
//! `cpu_fallback` path.
//!
//! ## Why `BackwardKind::NotDifferentiable` for v1
//!
//! Mamba inference is the only consumer surface today (and it's on
//! the eager Tensor path — see
//! `docs/session-prompts/mamba-eager-to-lazy-migration.md`). Training
//! support requires a real Mamba training consumer to materialize
//! AND the migration to LazyTensor to land. The baracuda kernel
//! has a backward variant ready, so adding `SELECTIVE_SCAN_BACKWARD`
//! is mechanical when those preconditions are met.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern,
};
use crate::{Graph, NodeId};
use fuel_core_types::{DType, Shape};

/// Metadata-side registry entry for SelectiveScan.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:         FusedOps::SELECTIVE_SCAN,
        name:       "SelectiveScan",
        family:     FusedOpFamily::Forward,
        pattern:    SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:   BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
    }
}

/// Output shape rule: `y: [batch, seqlen, dim]` — same as `u`'s shape
/// (input 0). The recurrence preserves the input's leading dims; the
/// state is consumed internally.
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 5,
        "SelectiveScan takes 5 inputs (u, delta, a, b, c)",
    );
    input_shapes[0].clone()
}

/// Dtype rule: output matches `u`'s dtype (input 0). All 5 inputs
/// must agree at construction time (the builder validates).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 5,
        "SelectiveScan takes 5 inputs",
    );
    input_dtypes[0]
}

/// See module preamble — SelectiveScan deliberately has no primitive
/// decomposition. The `cpu_fallback` path handles backends without a
/// native kernel.
pub fn decompose(_graph: &mut Graph, _id: NodeId, _params: &FusedOpParams) -> NodeId {
    panic!(
        "selective_scan::decompose: SelectiveScan has no registry-layer \
         decomposition. The textbook scan is a sequential recurrence; \
         synthesizing it from primitives would yield O(seqlen) nodes \
         and defeat any optimization pass. Backends without a native \
         SelectiveScan kernel use the executor's cpu_fallback path. \
         See conv2d::decompose and causal_conv1d::decompose for the \
         same precedent.",
    );
}

/// Matcher stub — SelectiveScan nodes originate from the explicit
/// `Tensor::selective_scan` builder. The primitive subgraph that
/// Mamba's eager-Tensor inference code unrolls is a per-timestep
/// recurrence with mutable state — not a pattern that can be
/// auto-fused from a static graph walk.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
