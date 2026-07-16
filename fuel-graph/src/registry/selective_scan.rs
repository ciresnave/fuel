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
use crate::{Graph, Node, NodeId, Op, ScanEmit, ScanRole};
use fuel_ir::storage::OutputViewSpec;
use fuel_ir::{DType, Layout, Shape};
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

/// Total decomposition of SelectiveScan to an [`crate::Op::Scan`] recipe —
/// closing decisions-log G3 ("a higher-order `Scan` for SSMs"), the last of the
/// original three self-returning decomposes (`nf4_matmul` + concrete-`k_len`
/// `flash_attn` were the other two). The `Op::Scan` carries the affine Mamba
/// step as its body: carry `h [batch,dim,dstate]` initialized to zero, four
/// per-step slices (`u_t`, `d_t` = `[softplus]`-discretized `delta`, `b_t`,
/// `c_t`), and `a` as the single const; `bound = seqlen`, `emit = All`. The
/// body is `gate = exp(bc(d_t)·a)`, `bu = bc(d_t·u_t)·bc(b_t)`,
/// `h_new = gate·h + bu`, `y_t = Σ_dstate(h_new·bc(c_t))`.
///
/// `softplus` (when set) is computed over the whole `delta` tensor *before* the
/// scan (no recurrent dependency). The per-step series are transposed to be
/// seqlen-major (scan axis 0). The returned node re-attaches the 2-slot
/// `output_views` bundle contract (slot 0 = `y`, transposed to
/// `[batch,seqlen,dim]`; slot 1 = `last_state`) — see the module note on the
/// Phase-1 `last_state`-realize limitation.
///
/// Per G2 (2026-06-20) this is total + never-panic: the only self-returns are
/// the belt-and-suspenders wrong-params / malformed-shape guards (structurally
/// impossible for a well-formed `SELECTIVE_SCAN` node), which are the driver's
/// fixpoint signal, not a crash. The recipe is the *math* the kernel computes;
/// the fused CPU/CUDA kernel stays the executed production path — the optimizer
/// chooses between them by cost, and `unroll_scan` provides the verify oracle.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    let delta_softplus = match params {
        FusedOpParams::SelectiveScan { delta_softplus } => *delta_softplus,
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    };

    // Read the 5 input NodeIds + shapes/dtype in a short borrow.
    let (u_id, delta_id, a_id, b_id, c_id, u_shape, a_shape, b_shape, c_shape, dtype) = {
        let n = graph.node(id);
        // Defensive: a well-formed SELECTIVE_SCAN node has exactly 5 inputs.
        // Malformed → fixpoint self-return (never panic).
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
    // Defensive shape guards — malformed → fixpoint self-return.
    if u_dims.len() != 3 || a_dims.len() != 2 {
        return id;
    }
    let batch = u_dims[0];
    let seqlen = u_dims[1];
    let dim = u_dims[2];
    let dstate = a_dims[1];

    let carry_shape = Shape::from_dims(&[batch, dim, dstate]);
    let elem_ud = Shape::from_dims(&[batch, dim]);
    let elem_bc = Shape::from_dims(&[batch, dstate]);

    // Broadcast [batch,dim] -> [batch,dim,1] -> [batch,dim,dstate].
    let bc_from_ud = |graph: &mut Graph, src: NodeId| -> NodeId {
        let mid = Shape::from_dims(&[batch, dim, 1]);
        let full = Shape::from_dims(&[batch, dim, dstate]);
        let re = graph.push(Node {
            op: Op::Reshape(mid.clone()), inputs: vec![src], shape: mid, dtype,
        });
        graph.push(Node {
            op: Op::BroadcastTo(full.clone()), inputs: vec![re], shape: full, dtype,
        })
    };
    // Broadcast [batch,dstate] -> [batch,1,dstate] -> [batch,dim,dstate].
    let bc_from_bc = |graph: &mut Graph, src: NodeId| -> NodeId {
        let mid = Shape::from_dims(&[batch, 1, dstate]);
        let full = Shape::from_dims(&[batch, dim, dstate]);
        let re = graph.push(Node {
            op: Op::Reshape(mid.clone()), inputs: vec![src], shape: mid, dtype,
        });
        graph.push(Node {
            op: Op::BroadcastTo(full.clone()), inputs: vec![re], shape: full, dtype,
        })
    };

    // ---- 1. discretize delta -> d [batch,seqlen,dim] BEFORE the scan (softplus
    // has no recurrent dependency). softplus(x) = max(x,0) + ln(1 + exp(-|x|))
    // = Relu(x) + Log(1 + Exp(Neg(Abs(x)))) — the byte-kernel's stable form.
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

    // ---- 2. per-step series with the scan (seqlen) axis moved to dim 0, so
    // `unroll_scan`'s `Slice{dim:0}` addresses each timestep: Permute([1,0,2]).
    let ud_ser_shape = Shape::from_dims(&[seqlen, batch, dim]);
    let bc_ser_shape = Shape::from_dims(&[seqlen, batch, dstate]);
    let permute3 = |graph: &mut Graph, src: NodeId, out: Shape| -> NodeId {
        graph.push(Node { op: Op::Permute(vec![1, 0, 2]), inputs: vec![src], shape: out, dtype })
    };
    let u_ser = permute3(graph, u_id, ud_ser_shape.clone());
    let d_ser = permute3(graph, d_full, ud_ser_shape.clone());
    let b_ser = permute3(graph, b_id, bc_ser_shape.clone());
    let c_ser = permute3(graph, c_id, bc_ser_shape.clone());

    // ---- 3. init_carry: zero [batch,dim,dstate], derived without a
    // data-carrying Const (a `decompose` fn has no device handle): broadcast
    // `a` up to the carry shape and multiply by 0.
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

    // ---- 4. the body: h ← exp(d·a)·h + d·b·u ; y = Σ_dstate(h·c). Carry h
    // and the per-step holes reference ScanPlaceholders; `a` is the const.
    let h = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: carry_shape.clone(), dtype });
    let u_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 0 }, inputs: vec![], shape: elem_ud.clone(), dtype });
    let d_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 1 }, inputs: vec![], shape: elem_ud.clone(), dtype });
    let b_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 2 }, inputs: vec![], shape: elem_bc.clone(), dtype });
    let c_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 3 }, inputs: vec![], shape: elem_bc.clone(), dtype });

    // gate = Exp( bc(d_t) ⊙ bc(a) )
    let d_bc = bc_from_ud(graph, d_t);
    let a_body = {
        let mid = Shape::from_dims(&[1, dim, dstate]);
        let re = graph.push(Node { op: Op::Reshape(mid.clone()), inputs: vec![a_id], shape: mid, dtype });
        graph.push(Node { op: Op::BroadcastTo(carry_shape.clone()), inputs: vec![re], shape: carry_shape.clone(), dtype })
    };
    let da = graph.push(Node { op: Op::Mul, inputs: vec![d_bc, a_body], shape: carry_shape.clone(), dtype });
    let gate = graph.push(Node { op: Op::Exp, inputs: vec![da], shape: carry_shape.clone(), dtype });

    // bu = bc(d_t ⊙ u_t) ⊙ bc(b_t)
    let du = graph.push(Node { op: Op::Mul, inputs: vec![d_t, u_t], shape: elem_ud.clone(), dtype });
    let du_bc = bc_from_ud(graph, du);
    let b_body = bc_from_bc(graph, b_t);
    let bu = graph.push(Node { op: Op::Mul, inputs: vec![du_bc, b_body], shape: carry_shape.clone(), dtype });

    // h_new = gate ⊙ h + bu   (body_new_carry)
    let gh = graph.push(Node { op: Op::Mul, inputs: vec![gate, h], shape: carry_shape.clone(), dtype });
    let h_new = graph.push(Node { op: Op::Add, inputs: vec![gh, bu], shape: carry_shape.clone(), dtype });

    // y_t = ReduceSumTo_dstate( h_new ⊙ bc(c_t) ) -> [batch,dim]   (body_y)
    let c_body = bc_from_bc(graph, c_t);
    let hc = graph.push(Node { op: Op::Mul, inputs: vec![h_new, c_body], shape: carry_shape.clone(), dtype });
    let y_keep_shape = Shape::from_dims(&[batch, dim, 1]);
    let y_keep = graph.push(Node {
        op: Op::ReduceSumTo(y_keep_shape.clone()), inputs: vec![hc], shape: y_keep_shape, dtype,
    });
    let y_t = graph.push(Node {
        op: Op::Reshape(elem_ud.clone()), inputs: vec![y_keep], shape: elem_ud.clone(), dtype,
    });

    // ---- 5. the Op::Scan node. Natural 2-slot bundle: slot 0 = stacked ys
    // [seqlen,batch,dim], slot 1 = final carry [batch,dim,dstate].
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

    // ---- 6. reconcile the axis order. `Op::Scan{emit=All}` stacks each y_t on
    // axis 0 -> [seqlen,batch,dim]; `y` must be [batch,seqlen,dim]. Project
    // slot 0 and transpose the seqlen/batch axes back.
    let ys_raw = graph.push(Node {
        op: Op::View { slot: 0 }, inputs: vec![scan], shape: ys_stacked_shape, dtype,
    });
    let y_shape = Shape::from_dims(&[batch, seqlen, dim]);
    let y = graph.push(Node {
        op: Op::Permute(vec![1, 0, 2]), inputs: vec![ys_raw], shape: y_shape, dtype,
    });

    // ---- 7. re-present the SelectiveScan 2-slot bundle contract (slot 0 = y
    // [batch,seqlen,dim], slot 1 = last_state [batch,dim,dstate]) on the return
    // node so the existing `Op::View(0)/(1)` consumers keep their shapes. See
    // the module note on the Phase-1 `last_state` realize gap (needs a bundle-
    // composer op; the executor runs the fused kernel, so this is inert today).
    let input_shapes = [u_shape, delta_shape, a_shape, b_shape, c_shape];
    let input_dtypes = [dtype, dtype, dtype, dtype, dtype];
    let ss_specs = output_views(&input_shapes, &input_dtypes, params);
    if let Ok((_bytes, views)) = fuel_ir::storage::compose_bundle(&ss_specs) {
        let _ = graph.set_output_views(y, Arc::from(views.into_boxed_slice()));
    }
    y
}

/// Matcher stub — SelectiveScan nodes originate from the explicit
/// `Tensor::selective_scan` builder. The primitive subgraph that
/// Mamba's eager-Tensor inference code unrolls is a per-timestep
/// recurrence with mutable state — not a pattern that can be
/// auto-fused from a static graph walk.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
