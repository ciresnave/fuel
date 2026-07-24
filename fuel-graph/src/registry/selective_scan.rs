//! SelectiveScan — Mamba-1's selective state-space-model scan. Third
//! FusedOpRegistry entry added by the re-framed CPU OpKind coverage
//! plan (after FusedSoftmaxCrossEntropy + CausalConv1d).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules, a
//!   total `decompose` to an `Op::Scan` recipe (G3 closed — see the
//!   architectural note below), and a stubbed pattern).
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
//! ## Architectural note — the SSM `Scan` primitive (G3 closed)
//!
//! SelectiveScan *was* the constitution's **canonical basis gap**: decisions-log
//! G3 (2026-06-20) named "a higher-order `Scan` for SSMs" as exactly the kind of
//! primitive Fuel lacked, to be closed by a **build-time `Op`-enum extension**,
//! not smuggled in at runtime. That primitive now exists ([`crate::Op::Scan`],
//! Phase 1), so [`decompose`] emits it — closing G3. Neither pre-`Scan` recipe
//! was admissible: the `O(seqlen)` unroll is an unbounded, un-re-fusable
//! explosion, and the `CumSum` closed-form (`h[t] = exp(a·D[t]) ⊙
//! cumsum_t(exp(−a·D[s]) ⊙ x[s])`, `D = cumsum_t(Δ)`) **overflows** for Mamba's
//! `a = −exp(a_log) < 0` (`exp(|a|·D[s])`), numerically invalid vs. the fused
//! kernel. `Op::Scan` sidesteps both — it *is* the sequential recurrence as a
//! bounded, sub-graph-carrying primitive.
//!
//! [`decompose`] lowers `Op::Fused(SELECTIVE_SCAN)` to an `Op::Scan` whose body
//! is the affine step `h ← exp(d·a)·h + d·b·u`; `y_t = Σ_dstate(h·c)`. `Op::Scan`
//! is a base-map terminal (no `LoweringRule` matches it), realized either by
//! `unroll_scan` (the numeric verify oracle / kernel-absent fallback) or —
//! the **executed production path** — by dispatching the *fused* op to its
//! CPU/CUDA kernel. The decompose recipe exists for the optimizer's base-map
//! cover-finding and the verify oracle; the fused kernel stays the fast path.
//!
//! **Phase-1 multi-output limitation.** The fused node is a 2-slot bundle
//! (slot 0 = `y`, slot 1 = `last_state`). The `Op::Scan` naturally stacks each
//! per-step `y_t` on axis 0 (`[seqlen,batch,dim]`); the decompose transposes it
//! back to `y [batch,seqlen,dim]` (slot 0, correct for general seqlen) and
//! re-attaches the `output_views` bundle contract so `Op::View(0)/(1)` keep
//! their shapes. Realizing `last_state` (slot 1) *through the decomposed graph*
//! fails loudly today (the kernel-less `Op::Scan` makes any realize-through-
//! decompose surface a typed error, not silent garbage) and would need a
//! bundle-composer op (`Op::ScatterIntoSlot`, currently kernel-less) to
//! re-unite the transposed `y` with the scan's final carry into one storage.
//! Because the executor runs the fused kernel (which writes both slots
//! correctly) and the oracle unrolls `ys` directly, this is an inert,
//! documented Phase-1 gap — not a live path. Wiring it (or dropping the
//! slot-1 claim) is a Phase-2 item that must land BEFORE `Op::Scan` gets a
//! native kernel: once it has one, an un-composed `view(1)` would read past
//! the slot-0 buffer instead of hitting today's typed dispatch error.
//!
//! ## Why `BackwardKind::Decompose` (Op::Scan Phase 2)
//!
//! Differentiability comes from the Phase-2 `lower_scans_for_backward`
//! pre-pass (`fuel-graph/src/lib.rs`), NOT from a bespoke
//! `SELECTIVE_SCAN_BACKWARD` fused op: `backward()` decomposes this node into
//! its `Op::Scan` recipe (below), `unroll_scan`s that to `bound` primitives,
//! and runs the node-general autograd over the unroll (truncated BPTT to the
//! static `bound`). `BackwardKind::Decompose` documents that intent — the
//! field is verified-dead metadata (the reverse walk never reads it; the
//! pre-pass replaces these nodes before the walk), so this is honesty about
//! the mechanism, not the mechanism itself.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps,
    PatternMatch, SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::storage::OutputViewSpec;
use fuel_ir::{DType, Layout, Shape};
use fuel_kernel_seam_types::shape_expr::{Dim, ShapeExpr};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode, SCAN_ROLE_CARRY, SCAN_ROLE_ELEM};
use std::sync::Arc;

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
        backward:     BackwardKind::Decompose,
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

/// SelectiveScan's `Op::Scan` recipe as portable `PatternNode` DATA (Increment
/// C, B2) — the structure-preserving migration of the pre-B2 imperative body
/// onto the B1 re-emit machinery. `Op::Scan` carries the affine Mamba step:
/// carry `h [batch,dim,dstate]` (zero-init), four per-step slices (`u_t`,
/// `d_t` = `[softplus]`-discretized `delta`, `b_t`, `c_t`), and `a` as the
/// single const; `bound = seqlen` (baked per call — the scan bound is a
/// shape-dependent structural param with no rel carrier), `emit = All`. The
/// body is `gate = exp(bc(d_t)*bc(a))`, `bu = bc(d_t*u_t)*bc(b_t)`,
/// `h_new = gate*h + bu`, `y_t = sum_dstate(h_new*bc(c_t))`.
///
/// Bind space: `0 = u`, `1 = delta`, `2 = a`, `3 = b`, `4 = c` — the fused
/// node's input order. Every interior shape rides a §6.20 `Dims`/`Extent`
/// target (`batch = u.0`, `dim = u.2`, `dstate = a.1`), so one recipe form (per
/// softplus variant) covers all `batch`/`dim`/`dstate` — and each
/// `ScanPlaceholder` carries its declared per-step shape via the same
/// `target_shape_rel` carrier (B2's emit extension), so the re-emitted body
/// matches the imperative one node-for-node (load-bearing: `unroll_scan` copies
/// stored body shapes). No open scalar slots — the `softplus` `AddScalar(1.0)`
/// and the zero-init `MulScalar(0.0)` are baked pattern constants.
///
/// The `delta_softplus` config toggles TWO recipe variants (the minimal
/// config-branch): the softplus arm prepends `Relu(d)+Log(1+Exp(-|d|))` to
/// `delta` before the per-step permute; the plain arm feeds `delta` directly.
fn recipe(seqlen: usize, delta_softplus: bool) -> PatternNode {
    use OpTag as T;
    let ext = |operand: u8, axis: u8| Dim::Extent { operand, axis };
    let one = || Dim::Const(1);
    let batch = || ext(0, 0);
    let dim = || ext(0, 2);
    let dstate = || ext(2, 1);
    let dims = |ds: Vec<Dim>| OpAttrs {
        target_shape_rel: Some(ShapeExpr::Dims(ds)),
        ..OpAttrs::default()
    };
    let op = |op, attrs, operands| PatternNode::Op { op, attrs, operands };
    let bind = |i| PatternNode::Bind { index: i };
    // Per-step + carry shapes and the reshape mids, as Dims over the bind space.
    let carry = || vec![batch(), dim(), dstate()];      // [b,d,s]
    let elem_ud = || vec![batch(), dim()];              // [b,d]
    let elem_bc = || vec![batch(), dstate()];           // [b,s]
    let bdi = || vec![batch(), dim(), one()];           // [b,d,1]
    let bis = || vec![batch(), one(), dstate()];        // [b,1,s]
    let ods = || vec![one(), dim(), dstate()];          // [1,d,s]
    let reshape = |src, ds: Vec<Dim>| op(T::Reshape, dims(ds), vec![src]);
    let bcast = |src, ds: Vec<Dim>| op(T::BroadcastTo, dims(ds), vec![src]);
    // bc([b,d]->[b,d,s]) and bc([b,s]->[b,d,s]) mirror the imperative bc_from_*.
    let bc_from_ud = |src| bcast(reshape(src, bdi()), carry());
    let bc_from_bc = |src| bcast(reshape(src, bis()), carry());
    // `a` broadcast to carry (used by BOTH init and the body gate — emit's
    // slot-free identity-share collapses the two occurrences to one node, as
    // the imperative body built two separate-but-identical copies).
    let a_to_carry = || bcast(reshape(bind(2), ods()), carry());
    let permute = |src| op(T::Permute, OpAttrs { perm: vec![1, 0, 2], ..OpAttrs::default() }, vec![src]);
    // A body hole carrying its DECLARED per-step shape (B2 carrier).
    let ph = |role: u8, index: u32, shape: Vec<Dim>| PatternNode::Op {
        op: T::ScanPlaceholder,
        attrs: OpAttrs {
            scan_role: Some(role),
            scan_index: Some(index),
            target_shape_rel: Some(ShapeExpr::Dims(shape)),
            ..OpAttrs::default()
        },
        operands: vec![],
    };

    // ---- 1. discretize delta BEFORE the scan (softplus has no recurrent
    // dependency). softplus(x) = Relu(x) + Log(1 + Exp(Neg(Abs(x)))).
    let d_full = if !delta_softplus {
        bind(1)
    } else {
        let ax = op(T::Abs, OpAttrs::default(), vec![bind(1)]);
        let nax = op(T::Neg, OpAttrs::default(), vec![ax]);
        let e = op(T::Exp, OpAttrs::default(), vec![nax]);
        let e1 = op(T::AddScalar, OpAttrs { scalars: vec![1.0], ..OpAttrs::default() }, vec![e]);
        let l = op(T::Log, OpAttrs::default(), vec![e1]);
        let r = op(T::Relu, OpAttrs::default(), vec![bind(1)]);
        op(T::Add, OpAttrs::default(), vec![r, l])
    };

    // ---- 2. per-step series (scan axis -> dim 0): Permute([1,0,2]).
    let u_ser = permute(bind(0));
    let d_ser = permute(d_full);
    let b_ser = permute(bind(3));
    let c_ser = permute(bind(4));

    // ---- 3. init_carry = MulScalar(0)(a-broadcast-to-carry) — zero carry
    // without a data-carrying Const (a recipe has no device handle).
    let init_carry = op(T::MulScalar, OpAttrs { scalars: vec![0.0], ..OpAttrs::default() }, vec![a_to_carry()]);

    // ---- 4. the body. Placeholders carry declared shapes; `a` is the const.
    let h = ph(SCAN_ROLE_CARRY, 0, carry());
    let u_t = ph(SCAN_ROLE_ELEM, 0, elem_ud());
    let d_t = ph(SCAN_ROLE_ELEM, 1, elem_ud());
    let b_t = ph(SCAN_ROLE_ELEM, 2, elem_bc());
    let c_t = ph(SCAN_ROLE_ELEM, 3, elem_bc());

    // gate = Exp( bc(d_t) * bc(a) )
    let da = op(T::Mul, OpAttrs::default(), vec![bc_from_ud(d_t.clone()), a_to_carry()]);
    let gate = op(T::Exp, OpAttrs::default(), vec![da]);
    // bu = bc(d_t * u_t) * bc(b_t)
    let du = op(T::Mul, OpAttrs::default(), vec![d_t, u_t]);
    let bu = op(T::Mul, OpAttrs::default(), vec![bc_from_ud(du), bc_from_bc(b_t)]);
    // h_new = gate * h + bu   (body_new_carry)
    let gh = op(T::Mul, OpAttrs::default(), vec![gate, h]);
    let h_new = op(T::Add, OpAttrs::default(), vec![gh, bu]);
    // y_t = Reshape([b,d])( ReduceSumTo([b,d,1])( h_new * bc(c_t) ) )   (body_y)
    let hc = op(T::Mul, OpAttrs::default(), vec![h_new.clone(), bc_from_bc(c_t)]);
    let y_keep = op(T::ReduceSumTo, dims(bdi()), vec![hc]);
    let y_t = op(T::Reshape, dims(elem_ud()), vec![y_keep]);

    // ---- 5. the Op::Scan node (n_xs=4, bound=seqlen, emit=All). Its operands
    // are the lax.scan encoding [init_carry, xs.., const(a), body_new_carry,
    // body_y]; emit attaches the 2-slot [stacked_ys, carry] bundle.
    let scan = PatternNode::Op {
        op: T::Scan,
        attrs: OpAttrs {
            scan_n_xs: Some(4),
            scan_bound: Some(seqlen as u32),
            scan_emit: Some(0), // ScanEmit::All
            scan_early_exit: None,
            ..OpAttrs::default()
        },
        operands: vec![init_carry, u_ser, d_ser, b_ser, c_ser, bind(2), h_new, y_t],
    };

    // ---- 6. project slot 0 (stacked ys [seqlen,batch,dim]) + transpose the
    // seqlen/batch axes back to y [batch,seqlen,dim].
    let ys_raw = op(T::View, OpAttrs { view_slot: Some(0), ..OpAttrs::default() }, vec![scan]);
    op(T::Permute, OpAttrs { perm: vec![1, 0, 2], ..OpAttrs::default() }, vec![ys_raw])
}

/// Total decomposition of SelectiveScan to an [`crate::Op::Scan`] recipe —
/// closing decisions-log G3 ("a higher-order `Scan` for SSMs"). Since Increment
/// C B2 a re-emit of [`recipe`]'s portable data through the
/// [`decompose_via_recipe`] bridge (structure-preserving: the emitted base map
/// is node-for-node identical to the pre-B2 imperative body — see the parity
/// test in `tests`). `seqlen` (the scan bound, a shape-dependent structural
/// param) and the `delta_softplus` variant are read here and baked into the
/// per-call recipe; all other extents ride §6.20 `Dims` rel-attrs.
///
/// Per G2 (2026-06-20) this is total + never-panic: the wrong-params /
/// malformed-shape guards and any bridge decline (bind-arity, rel-resolution,
/// …) return `id` (the driver's fixpoint signal, not a crash). The recipe is
/// the *math* the kernel computes; the fused CPU/CUDA kernel stays the executed
/// production path — the optimizer chooses by cost, and `unroll_scan` provides
/// the verify oracle. The returned node re-attaches the 2-slot `output_views`
/// bundle contract (slot 0 = `y [batch,seqlen,dim]`, slot 1 = `last_state`) —
/// see the module note on the Phase-1 `last_state`-realize limitation.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let delta_softplus = match params {
        FusedOpParams::SelectiveScan { delta_softplus } => *delta_softplus,
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    };

    // Read the 5 input shapes/dtype in a short borrow. `seqlen` (the scan
    // bound) is baked into the recipe; the shapes feed the bundle re-attach.
    let (u_shape, delta_shape, a_shape, b_shape, c_shape, dtype) = {
        let n = graph.node(id);
        // Defensive: a well-formed SELECTIVE_SCAN node has exactly 5 inputs.
        // Malformed → fixpoint self-return (never panic).
        if n.inputs.len() != 5 {
            return id;
        }
        (
            graph.node(n.inputs[0]).shape.clone(),
            graph.node(n.inputs[1]).shape.clone(),
            graph.node(n.inputs[2]).shape.clone(),
            graph.node(n.inputs[3]).shape.clone(),
            graph.node(n.inputs[4]).shape.clone(),
            n.dtype,
        )
    };
    let u_dims = u_shape.dims();
    let a_dims = a_shape.dims();
    // Defensive shape guards — malformed → fixpoint self-return.
    if u_dims.len() != 3 || a_dims.len() != 2 {
        return id;
    }
    let seqlen = u_dims[1];

    let recipe_node = recipe(seqlen, delta_softplus);
    // No open scalar slots (softplus 1.0 / zero-init 0.0 are baked constants).
    let root = decompose_via_recipe(graph, id, &recipe_node, Some(Vec::new()));
    if root == id {
        // Bridge declined (bind-arity, rel-resolution, …) — stay fused (G2).
        return id;
    }

    // Re-present the SelectiveScan 2-slot bundle contract on the returned
    // Permute so existing `Op::View(0)/(1)` consumers keep their shapes (the
    // recipe emit attaches the [ys,carry] bundle to the inner `Op::Scan`, not
    // to this outer node) — mirrors the imperative decompose's step 7.
    let input_shapes = [u_shape, delta_shape, a_shape, b_shape, c_shape];
    let input_dtypes = [dtype, dtype, dtype, dtype, dtype];
    let ss_specs = output_views(&input_shapes, &input_dtypes, params);
    if let Ok((_bytes, views)) = fuel_ir::storage::compose_bundle(&ss_specs) {
        let _ = graph.set_output_views(root, Arc::from(views.into_boxed_slice()));
    }
    root
}

/// Matcher stub — SelectiveScan nodes originate from the explicit
/// `Tensor::selective_scan` builder. The primitive subgraph that
/// Mamba's eager-Tensor inference code unrolls is a per-timestep
/// recurrence with mutable state — not a pattern that can be
/// auto-fused from a static graph walk.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::FusedOps;
    use crate::{Node, Op, ScanEmit, ScanRole};

    /// FROZEN copy of the pre-B2 imperative `selective_scan::decompose` body,
    /// verbatim (the `Op::Scan` + `ScanPlaceholder` + `View` + explicit-Shape
    /// spelling), before B2 replaced the live body with the data recipe. The
    /// B2 structure-preservation oracle: the migrated recipe re-emit must
    /// produce a graph structurally identical to this.
    fn frozen_legacy_decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
        let delta_softplus = match params {
            FusedOpParams::SelectiveScan { delta_softplus } => *delta_softplus,
            _ => return id,
        };
        let (u_id, delta_id, a_id, b_id, c_id, u_shape, a_shape, b_shape, c_shape, dtype) = {
            let n = graph.node(id);
            if n.inputs.len() != 5 {
                return id;
            }
            let u_shape = graph.node(n.inputs[0]).shape.clone();
            let a_shape = graph.node(n.inputs[2]).shape.clone();
            let b_shape = graph.node(n.inputs[3]).shape.clone();
            let c_shape = graph.node(n.inputs[4]).shape.clone();
            (
                n.inputs[0], n.inputs[1], n.inputs[2], n.inputs[3], n.inputs[4],
                u_shape, a_shape, b_shape, c_shape, n.dtype,
            )
        };
        let u_dims = u_shape.dims();
        let a_dims = a_shape.dims();
        if u_dims.len() != 3 || a_dims.len() != 2 {
            return id;
        }
        let batch = u_dims[0];
        let seqlen = u_dims[1];
        let dim = u_dims[2];
        let dstate = a_dims[1];

        let carry_shape = Shape::from_dims(&[batch, dim, dstate]);
        let elem_ud = Shape::from_dims(&[batch, dim]);

        let bc_from_ud = |graph: &mut Graph, src: NodeId| -> NodeId {
            let mid = Shape::from_dims(&[batch, dim, 1]);
            let full = Shape::from_dims(&[batch, dim, dstate]);
            let re = graph.push(Node { op: Op::Reshape(mid.clone()), inputs: vec![src], shape: mid, dtype });
            graph.push(Node { op: Op::BroadcastTo(full.clone()), inputs: vec![re], shape: full, dtype })
        };
        let bc_from_bc = |graph: &mut Graph, src: NodeId| -> NodeId {
            let mid = Shape::from_dims(&[batch, 1, dstate]);
            let full = Shape::from_dims(&[batch, dim, dstate]);
            let re = graph.push(Node { op: Op::Reshape(mid.clone()), inputs: vec![src], shape: mid, dtype });
            graph.push(Node { op: Op::BroadcastTo(full.clone()), inputs: vec![re], shape: full, dtype })
        };

        let delta_shape = Shape::from_dims(&[batch, seqlen, dim]);
        let d_full = if !delta_softplus {
            delta_id
        } else {
            let ax = graph.push(Node { op: Op::Abs, inputs: vec![delta_id], shape: delta_shape.clone(), dtype });
            let nax = graph.push(Node { op: Op::Neg, inputs: vec![ax], shape: delta_shape.clone(), dtype });
            let e = graph.push(Node { op: Op::Exp, inputs: vec![nax], shape: delta_shape.clone(), dtype });
            let e1 = graph.push(Node { op: Op::AddScalar(1.0), inputs: vec![e], shape: delta_shape.clone(), dtype });
            let l = graph.push(Node { op: Op::Log, inputs: vec![e1], shape: delta_shape.clone(), dtype });
            let r = graph.push(Node { op: Op::Relu, inputs: vec![delta_id], shape: delta_shape.clone(), dtype });
            graph.push(Node { op: Op::Add, inputs: vec![r, l], shape: delta_shape.clone(), dtype })
        };

        let ud_ser_shape = Shape::from_dims(&[seqlen, batch, dim]);
        let bc_ser_shape = Shape::from_dims(&[seqlen, batch, dstate]);
        let permute3 = |graph: &mut Graph, src: NodeId, out: Shape| -> NodeId {
            graph.push(Node { op: Op::Permute(vec![1, 0, 2]), inputs: vec![src], shape: out, dtype })
        };
        let u_ser = permute3(graph, u_id, ud_ser_shape.clone());
        let d_ser = permute3(graph, d_full, ud_ser_shape.clone());
        let b_ser = permute3(graph, b_id, bc_ser_shape.clone());
        let c_ser = permute3(graph, c_id, bc_ser_shape.clone());

        let a_re3 = graph.push(Node {
            op: Op::Reshape(Shape::from_dims(&[1, dim, dstate])), inputs: vec![a_id],
            shape: Shape::from_dims(&[1, dim, dstate]), dtype,
        });
        let a_bc = graph.push(Node {
            op: Op::BroadcastTo(carry_shape.clone()), inputs: vec![a_re3], shape: carry_shape.clone(), dtype,
        });
        let init_carry = graph.push(Node {
            op: Op::MulScalar(0.0), inputs: vec![a_bc], shape: carry_shape.clone(), dtype,
        });

        let h = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: carry_shape.clone(), dtype });
        let u_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 0 }, inputs: vec![], shape: elem_ud.clone(), dtype });
        let d_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 1 }, inputs: vec![], shape: elem_ud.clone(), dtype });
        let b_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 2 }, inputs: vec![], shape: Shape::from_dims(&[batch, dstate]), dtype });
        let c_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 3 }, inputs: vec![], shape: Shape::from_dims(&[batch, dstate]), dtype });

        let d_bc = bc_from_ud(graph, d_t);
        let a_body = {
            let mid = Shape::from_dims(&[1, dim, dstate]);
            let re = graph.push(Node { op: Op::Reshape(mid.clone()), inputs: vec![a_id], shape: mid, dtype });
            graph.push(Node { op: Op::BroadcastTo(carry_shape.clone()), inputs: vec![re], shape: carry_shape.clone(), dtype })
        };
        let da = graph.push(Node { op: Op::Mul, inputs: vec![d_bc, a_body], shape: carry_shape.clone(), dtype });
        let gate = graph.push(Node { op: Op::Exp, inputs: vec![da], shape: carry_shape.clone(), dtype });

        let du = graph.push(Node { op: Op::Mul, inputs: vec![d_t, u_t], shape: elem_ud.clone(), dtype });
        let du_bc = bc_from_ud(graph, du);
        let b_body = bc_from_bc(graph, b_t);
        let bu = graph.push(Node { op: Op::Mul, inputs: vec![du_bc, b_body], shape: carry_shape.clone(), dtype });

        let gh = graph.push(Node { op: Op::Mul, inputs: vec![gate, h], shape: carry_shape.clone(), dtype });
        let h_new = graph.push(Node { op: Op::Add, inputs: vec![gh, bu], shape: carry_shape.clone(), dtype });

        let c_body = bc_from_bc(graph, c_t);
        let hc = graph.push(Node { op: Op::Mul, inputs: vec![h_new, c_body], shape: carry_shape.clone(), dtype });
        let y_keep_shape = Shape::from_dims(&[batch, dim, 1]);
        let y_keep = graph.push(Node {
            op: Op::ReduceSumTo(y_keep_shape.clone()), inputs: vec![hc], shape: y_keep_shape, dtype,
        });
        let y_t = graph.push(Node {
            op: Op::Reshape(elem_ud.clone()), inputs: vec![y_keep], shape: elem_ud.clone(), dtype,
        });

        let ys_stacked_shape = Shape::from_dims(&[seqlen, batch, dim]);
        let scan = graph.push(Node {
            op: Op::Scan { n_xs: 4, bound: seqlen, emit: ScanEmit::All, early_exit: None },
            inputs: vec![init_carry, u_ser, d_ser, b_ser, c_ser, a_id, h_new, y_t],
            shape: ys_stacked_shape.clone(),
            dtype,
        });
        let scan_specs = vec![
            OutputViewSpec::contiguous(dtype, ys_stacked_shape.clone()),
            OutputViewSpec::contiguous(dtype, carry_shape.clone()),
        ];
        if let Ok((_bytes, views)) = fuel_ir::storage::compose_bundle(&scan_specs) {
            let _ = graph.set_output_views(scan, Arc::from(views.into_boxed_slice()));
        }

        let ys_raw = graph.push(Node {
            op: Op::View { slot: 0 }, inputs: vec![scan], shape: ys_stacked_shape, dtype,
        });
        let y_shape = Shape::from_dims(&[batch, seqlen, dim]);
        let y = graph.push(Node {
            op: Op::Permute(vec![1, 0, 2]), inputs: vec![ys_raw], shape: y_shape, dtype,
        });

        let input_shapes = [u_shape, delta_shape, a_shape, b_shape, c_shape];
        let input_dtypes = [dtype, dtype, dtype, dtype, dtype];
        let ss_specs = output_views(&input_shapes, &input_dtypes, params);
        if let Ok((_bytes, views)) = fuel_ir::storage::compose_bundle(&ss_specs) {
            let _ = graph.set_output_views(y, Arc::from(views.into_boxed_slice()));
        }
        y
    }

    /// Recursively assert two subgraphs are node-for-node identical (op, shape,
    /// dtype, arity, and recursively-equal inputs). A shared leaf (same NodeId
    /// — the bound external inputs) matches by identity. Shape-sensitive at
    /// EVERY node, so it catches the rank-0 placeholder / `du` regression the
    /// B1 `base_map_hash` oracle is blind to (op_key(ScanPlaceholder) folds no
    /// shape).
    fn assert_structural_eq(g: &Graph, a: NodeId, b: NodeId) {
        if a == b {
            return;
        }
        let na = g.node(a);
        let nb = g.node(b);
        assert_eq!(na.op, nb.op, "op mismatch: {:?} vs {:?}", na.op, nb.op);
        assert_eq!(na.shape, nb.shape, "shape mismatch at {:?}: {:?} vs {:?}", na.op, na.shape, nb.shape);
        assert_eq!(na.dtype, nb.dtype, "dtype mismatch at {:?}", na.op);
        assert_eq!(na.inputs.len(), nb.inputs.len(), "arity mismatch at {:?}", na.op);
        for (&ia, &ib) in na.inputs.iter().zip(nb.inputs.iter()) {
            assert_structural_eq(g, ia, ib);
        }
    }

    /// Build a fused SELECTIVE_SCAN node over `u,delta [b,seq,dim]`,
    /// `a [dim,dstate]`, `b,c [b,seq,dstate]`. Returns the fused NodeId.
    fn fused_node(g: &mut Graph, batch: usize, seqlen: usize, dim: usize, dstate: usize, softplus: bool) -> NodeId {
        let mk = |g: &mut Graph, dims: &[usize]| {
            g.push(Node { op: Op::Const, inputs: vec![], shape: Shape::from_dims(dims), dtype: DType::F32 })
        };
        let u = mk(g, &[batch, seqlen, dim]);
        let delta = mk(g, &[batch, seqlen, dim]);
        let a = mk(g, &[dim, dstate]);
        let b = mk(g, &[batch, seqlen, dstate]);
        let c = mk(g, &[batch, seqlen, dstate]);
        g.push(Node {
            op: Op::Fused(FusedOps::SELECTIVE_SCAN, FusedOpParams::SelectiveScan { delta_softplus: softplus }),
            inputs: vec![u, delta, a, b, c],
            shape: Shape::from_dims(&[batch, seqlen, dim]),
            dtype: DType::F32,
        })
    }

    /// B2: ONE recipe form (per softplus variant) decomposes at multiple
    /// `batch/seqlen/dim/dstate` shapes, and its emitted base map is
    /// node-for-node identical to the FROZEN pre-B2 imperative body. Born-red
    /// without B2's emit `ScanPlaceholder`-shape carrier (the placeholders +
    /// `du = Mul(d_t,u_t)` come out rank-0, tripping `assert_structural_eq`).
    #[test]
    fn selective_scan_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
        for softplus in [false, true] {
            for (batch, seqlen, dim, dstate) in [(1usize, 3, 2, 4), (2, 4, 3, 2)] {
                let params = FusedOpParams::SelectiveScan { delta_softplus: softplus };
                let mut g = Graph::new();
                let fused = fused_node(&mut g, batch, seqlen, dim, dstate, softplus);
                let out_sh = g.node(fused).shape.clone();

                let new_root = decompose(&mut g, fused, &params);
                assert_ne!(new_root, fused, "recipe decompose fires (softplus={softplus}, [{batch},{seqlen},{dim},{dstate}])");
                assert_eq!(g.node(new_root).shape, out_sh, "output y = [batch,seqlen,dim]");
                assert_eq!(g.node(new_root).dtype, DType::F32);
                // The re-attached 2-slot bundle carries slot 1 (last_state).
                assert_eq!(
                    g.output_views(new_root).map(|v| v.len()), Some(2),
                    "SelectiveScan 2-slot output_views re-attached on the recipe root",
                );

                let legacy_root = frozen_legacy_decompose(&mut g, fused, &params);
                assert_structural_eq(&g, new_root, legacy_root);
            }
        }
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint, never a
    /// crash, before any emission.
    #[test]
    fn selective_scan_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let fused = fused_node(&mut g, 1, 3, 2, 4, false);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}
