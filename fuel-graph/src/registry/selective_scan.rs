//! SelectiveScan — Mamba-1's selective state-space-model scan. Third
//! FusedOpRegistry entry added by the re-framed CPU OpKind coverage
//! plan (after FusedSoftmaxCrossEntropy + CausalConv1d).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules, a
//!   self-returning `decompose` — a basis gap (needs a higher-order `Scan`
//!   primitive), per G2 — and a stubbed pattern).
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
//! ## Architectural note — a genuine basis gap (the SSM `Scan` primitive)
//!
//! SelectiveScan is the **canonical basis gap** the constitution names:
//! decisions-log G3 (2026-06-20) calls out "a higher-order `Scan` for SSMs"
//! as exactly the kind of primitive Fuel lacks and that must be closed by a
//! **build-time `Op`-enum extension**, not smuggled in at runtime. The
//! textbook scan is a sequential recurrence with per-timestep state; the two
//! ways to express it in *today's* basis are both rejected as recipes:
//!
//! - **Unroll `O(seqlen)` per-step chains** (`Exp/Mul/Add`). Total, but the
//!   node count is *shape-dependent and unbounded*, there is no finite
//!   `pattern` that re-fuses it, and it defeats the very optimization the base
//!   map exists for — not a recipe, an explosion.
//! - **Closed-form parallel scan via `CumSum`.** A diagonal SSM *does* have
//!   one: `h[t] = exp(a·D[t]) ⊙ cumsum_t(exp(−a·D[s]) ⊙ x[s])`, `D =
//!   cumsum_t(Δ)`. But Mamba's `a = −exp(a_log) < 0`, so `exp(−a·D[s]) =
//!   exp(|a|·D[s])` **overflows** for any realistic sequence — numerically
//!   invalid, i.e. *not* IEEE-equivalent to the fused kernel. This is exactly
//!   why Mamba's kernel uses a segmented/chunked scan, and why a stable
//!   decomposition needs the missing primitive (a `Scan` / associative-scan,
//!   or a chunked-scan op), not a cleverer rewrite.
//!
//! So per G2 [`decompose`] is total + **never-panic** by returning **self** —
//! the driver's fixpoint signal — leaving the node `Op::Fused` as a *surfaced
//! opaque-op gap* the inventory telemetry can find. Backends without a native
//! kernel use the executor's `cpu_fallback`. The precise ask to close it: add
//! a higher-order scan primitive to the `Op` basis (a Fuel-side build-time
//! extension per G3), then this decompose emits `cumsum`-scan-over-`Scan`.
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
use fuel_ir::storage::OutputViewSpec;
use fuel_ir::{DType, Layout, Shape};

/// Metadata-side registry entry for SelectiveScan. Multi-output (item
/// 3 consumer migration, 2026-06-01): slot 0 = `y`, slot 1 =
/// `last_state`. The `shape_rule` and `dtype_rule` report slot 0
/// (the primary, per the multi-output invariant); `output_views`
/// reports both slots' specs for the bundled allocator.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:           FusedOps::SELECTIVE_SCAN,
        name:         "SelectiveScan",
        family:       FusedOpFamily::Forward,
        pattern:      SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:     BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: Some(output_views),
    }
}

/// Output shape rule. Reports slot 0 (`y: [batch, seqlen, dim]`) —
/// the multi-output invariant requires `shape_rule` to equal
/// `output_views()[0].shape`. Slot 1 (`last_state`) is exposed via
/// `output_views` and reached through `Op::View { slot: 1 }`.
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 5,
        "SelectiveScan takes 5 inputs (u, delta, a, b, c)",
    );
    input_shapes[0].clone()
}

/// Multi-output authoring fn. Returns two slot specs:
/// - slot 0 = `y: [batch, seqlen, dim]`, same dtype as `u`.
/// - slot 1 = `last_state: [batch, dim, dstate]`, same dtype as `u`.
///
/// `batch`, `seqlen`, `dim` come from `u` (input 0); `dstate` comes
/// from `a` (input 2)'s second dim. Both slots are contiguous with
/// the default row-major layout — the bundled allocator computes
/// byte offsets via `compose_bundle`.
///
/// v1 keeps slot 1's dtype equal to slot 0's input dtype (matches the
/// kernel's `$T`-narrows-from-F64 contract). A future refinement
/// could pin slot 1 to F32 always, but that would force mixed-dtype
/// bundles for BF16/F16 callers and need a kernel-side split.
fn output_views(
    input_shapes: &[Shape],
    input_dtypes: &[DType],
    _params:      &FusedOpParams,
) -> Vec<OutputViewSpec> {
    debug_assert_eq!(
        input_shapes.len(), 5,
        "SelectiveScan output_views: takes 5 inputs (u, delta, a, b, c)",
    );
    debug_assert_eq!(
        input_dtypes.len(), 5,
        "SelectiveScan output_views: takes 5 input dtypes",
    );
    let u_dims = input_shapes[0].dims();
    let a_dims = input_shapes[2].dims();
    debug_assert!(
        u_dims.len() == 3 && a_dims.len() == 2,
        "SelectiveScan output_views: u rank=3, a rank=2 expected",
    );
    let batch  = u_dims[0];
    let seqlen = u_dims[1];
    let dim    = u_dims[2];
    let dstate = a_dims[1];
    let dtype  = input_dtypes[0];
    let y_shape = Shape::from_dims(&[batch, seqlen, dim]);
    let last_state_shape = Shape::from_dims(&[batch, dim, dstate]);
    vec![
        OutputViewSpec {
            dtype,
            shape:  y_shape.clone(),
            layout: Layout::contiguous(y_shape),
            name:   Some("y"),
        },
        OutputViewSpec {
            dtype,
            shape:  last_state_shape.clone(),
            layout: Layout::contiguous(last_state_shape),
            name:   Some("last_state"),
        },
    ]
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

/// SelectiveScan is the constitution's canonical **basis gap** (decisions-log
/// G3: "a higher-order `Scan` for SSMs"). No recipe over today's `Op` basis is
/// both *total* and *numerically valid* — the `O(seqlen)` unroll is an
/// unbounded, un-re-fusable explosion, and the `CumSum` closed-form overflows
/// for Mamba's `a < 0` regime (see the module note). Per G2 (2026-06-20)
/// `decompose` is therefore total and never panics by returning **self** — the
/// driver's fixpoint signal ("can't decompose further") — leaving the node
/// `Op::Fused` as a surfaced opaque-op gap for the inventory telemetry.
/// Closing it is a build-time `Op`-basis extension (a `Scan` / associative- or
/// chunked-scan primitive); backends without a native kernel use
/// `cpu_fallback`. `nf4_matmul_decompose_matches_kernel` and
/// `flash_attn_decompose_concrete_klen` (fuel-core) are the other two of the
/// original three panicking-decompose bugs — both now closed with real
/// recipes; this one remains a *documented* gap, not a bug.
pub fn decompose(_graph: &mut Graph, id: NodeId, _params: &FusedOpParams) -> NodeId {
    id
}

/// Matcher stub — SelectiveScan nodes originate from the explicit
/// `Tensor::selective_scan` builder. The primitive subgraph that
/// Mamba's eager-Tensor inference code unrolls is a per-timestep
/// recurrence with mutable state — not a pattern that can be
/// auto-fused from a static graph walk.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
