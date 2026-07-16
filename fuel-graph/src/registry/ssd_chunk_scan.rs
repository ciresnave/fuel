//! SsdChunkScan — Mamba-2's State-Space Duality chunked scan
//! (forward). Fourth FusedOpRegistry entry added by the re-framed
//! CPU OpKind coverage plan; completes the Mamba-adjacent trio
//! (CausalConv1d + SelectiveScan + SsdChunkScan).
//!
//! Provides:
//! - [`entry`] — the metadata-side `FusedOpEntry` (shape/dtype rules,
//!   a total `decompose` to an `Op::Scan` recipe (G3 closed — see the
//!   architectural note below), stubbed pattern).
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
//! ## On `chunk_size` and CPU dispatch
//!
//! `chunk_size` is the SSD block size — a GPU parallelization knob
//! that controls how many tokens are processed in parallel per
//! block. The Mamba-2 chunked algorithm rearranges the sequential
//! scan into block matrix ops (intra-chunk diagonal + inter-chunk
//! decay propagation) that GPUs can execute in parallel, but the
//! mathematical result is **identical** to a straight sequential
//! scan over all `seqlen` tokens.
//!
//! The CPU kernel runs the sequential scan directly (any
//! `chunk_size ∈ [1, seqlen]` that divides seqlen produces the same
//! answer). The GPU path (when wired) will use `chunk_size` for
//! parallelism granularity. Validation: `chunk_size > 0` and
//! `seqlen % chunk_size == 0`.
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
//! ## Architectural note — the SSM `Scan` primitive (G3 closed)
//!
//! Same precedent as [`super::selective_scan`]: SsdChunkScan *was* the twin of
//! the constitution's **canonical basis gap** (decisions-log G3, 2026-06-20 —
//! "a higher-order `Scan` for SSMs"). Synthesizing the recurrence from
//! pre-`Scan` primitives was inadmissible — the `O(seqlen)` unroll is an
//! unbounded, un-re-fusable explosion, and `CumSum` is only an *unweighted*
//! cumulative sum that cannot carry the per-step gating coefficient
//! `exp(dt·a)`. That primitive now exists ([`crate::Op::Scan`], Phase 1), so
//! [`decompose`] emits it — closing G3 (part 2).
//!
//! [`decompose`] lowers `Op::Fused(SSD_CHUNK_SCAN)` to an `Op::Scan` whose body
//! is the affine step `h ← exp(dt·a)·h + dt·b·x`; `y_t = Σ_state(h·c)`. The gate
//! `exp(dt·a_h)` is a per-head **scalar** (`a` is `[heads]`, broadcast across the
//! whole `head_dim×state_dim` block — simpler than selective_scan's per-`(dim,
//! dstate)` vector gate); there is **no** softplus. `Op::Scan` is a base-map
//! terminal (no `LoweringRule` matches it), realized either by `unroll_scan`
//! (the numeric verify oracle / kernel-absent fallback) or — the **executed
//! production path** — by dispatching the *fused* op to its CPU/CUDA kernel.
//! `chunk_size` is a documented CPU no-op: both the chunked GPU path and this
//! recipe re-decompose to the SAME sequential affine scan over all `seqlen`
//! tokens (the mathematical result is identical).
//!
//! **Phase-1 multi-output limitation.** The fused node is a 2-slot bundle
//! (slot 0 = `y`, slot 1 = `last_state`). The `Op::Scan` naturally stacks each
//! per-step `y_t` on axis 0 (`[seqlen,batch,heads,head_dim]`); the decompose
//! transposes it back to `y [batch,seqlen,heads,head_dim]` (slot 0, correct for
//! general seqlen) and re-attaches the `output_views` bundle contract so
//! `Op::View(0)/(1)` keep their shapes. Realizing `last_state` (slot 1) *via the
//! decomposed graph* fails loudly today (the kernel-less `Op::Scan` makes any
//! realize-through-decompose surface a typed error, not silent garbage); wiring
//! it is a Phase-2 item before `Op::Scan` gets a kernel. Because the executor
//! runs the fused kernel (which writes both slots) and the oracle unrolls `ys`
//! directly, this is an inert, documented Phase-1 gap — not a live path.
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
use crate::{Graph, Node, NodeId, Op, ScanEmit, ScanRole};
use fuel_ir::storage::OutputViewSpec;
use fuel_ir::{DType, Layout, Shape};
use std::sync::Arc;

/// Metadata-side registry entry for SsdChunkScan. Multi-output (item
/// 3 consumer migration, 2026-06-01): slot 0 = `y`, slot 1 =
/// `last_state`. Mirrors `selective_scan::entry`.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        id:           FusedOps::SSD_CHUNK_SCAN,
        name:         "SsdChunkScan",
        family:       FusedOpFamily::Forward,
        pattern:      SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward:     BackwardKind::NotDifferentiable,
        shape_rule,
        dtype_rule,
        output_views: Some(output_views),
    }
}

/// Output shape rule. Reports slot 0 (`y: [batch, seqlen, heads,
/// head_dim]`) — slot 1 (`last_state`) is exposed via `output_views`
/// and reached through `Op::View { slot: 1 }`.
fn shape_rule(input_shapes: &[Shape], _params: &FusedOpParams) -> Shape {
    debug_assert_eq!(
        input_shapes.len(), 5,
        "SsdChunkScan takes 5 inputs (x, dt, a, b, c)",
    );
    input_shapes[0].clone()
}

/// Multi-output authoring fn. Returns two slot specs:
/// - slot 0 = `y: [batch, seqlen, heads, head_dim]`, same dtype as `x`.
/// - slot 1 = `last_state: [batch, heads, head_dim, state_dim]`,
///   same dtype as `x`.
fn output_views(
    input_shapes: &[Shape],
    input_dtypes: &[DType],
    _params:      &FusedOpParams,
) -> Vec<OutputViewSpec> {
    debug_assert_eq!(
        input_shapes.len(), 5,
        "SsdChunkScan output_views: takes 5 inputs (x, dt, a, b, c)",
    );
    debug_assert_eq!(
        input_dtypes.len(), 5,
        "SsdChunkScan output_views: takes 5 input dtypes",
    );
    let x_dims = input_shapes[0].dims();
    let b_dims = input_shapes[3].dims();
    debug_assert!(
        x_dims.len() == 4 && b_dims.len() == 4,
        "SsdChunkScan output_views: x rank=4, b rank=4 expected",
    );
    let batch     = x_dims[0];
    let seqlen    = x_dims[1];
    let heads     = x_dims[2];
    let head_dim  = x_dims[3];
    let state_dim = b_dims[3];
    let dtype     = input_dtypes[0];
    let y_shape = Shape::from_dims(&[batch, seqlen, heads, head_dim]);
    let last_state_shape =
        Shape::from_dims(&[batch, heads, head_dim, state_dim]);
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

/// Dtype rule: output matches `x`'s dtype (input 0).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(
        input_dtypes.len(), 5,
        "SsdChunkScan takes 5 inputs",
    );
    input_dtypes[0]
}

/// Total decomposition of SsdChunkScan to an [`crate::Op::Scan`] recipe —
/// closing decisions-log G3 ("a higher-order `Scan` for SSMs"), part 2 (the twin
/// of `selective_scan`). The `Op::Scan` carries the affine Mamba-2 SSD step as
/// its body: carry `h [batch,heads,head_dim,state_dim]` initialized to zero,
/// four per-step slices (`x_t`, `dt_t`, `b_t`, `c_t`), and `a` as the single
/// const; `bound = seqlen`, `emit = All`, **no** softplus. The body is
/// `gate = exp(bc(dt_t)·bc(a))` — a per-head SCALAR gate broadcast across the
/// whole `head_dim×state_dim` block (`a` is `[heads]`) — `dbx = bc(dt_t)·bc(b_t)·
/// bc(x_t)`, `h_new = gate·h + dbx`, `y_t = Σ_state(h_new·bc(c_t))`.
///
/// The per-step series are transposed to be seqlen-major (scan axis 0) for
/// `unroll_scan`'s `Slice{dim:0}`. The returned node re-attaches the 2-slot
/// `output_views` bundle contract (slot 0 = `y`, transposed to
/// `[batch,seqlen,heads,head_dim]`; slot 1 = `last_state`) — see the module note
/// on the Phase-1 `last_state`-realize limitation. `chunk_size` is a documented
/// CPU no-op (both this and the chunked GPU path re-decompose to the SAME
/// sequential scan), so it does not appear in the recipe.
///
/// Per G2 (2026-06-20) this is total + never-panic: the only self-returns are
/// the belt-and-suspenders wrong-params / malformed-shape guards (structurally
/// impossible for a well-formed `SSD_CHUNK_SCAN` node), which are the driver's
/// fixpoint signal, not a crash. The recipe is the *math* the kernel computes;
/// the fused CPU/CUDA kernel stays the executed production path — the optimizer
/// chooses between them by cost, and `unroll_scan` provides the verify oracle.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    // Confirm this id really is an SSD_CHUNK_SCAN node. `chunk_size` is a
    // documented CPU no-op (same sequential scan regardless), so we don't read
    // its value into the recipe.
    match params {
        FusedOpParams::SsdChunkScan { .. } => {}
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    }

    // Read the 5 input NodeIds + shapes/dtype in a short borrow.
    let (x_id, dt_id, a_id, b_id, c_id, x_shape, dt_shape, a_shape, b_shape, c_shape, dtype) = {
        let n = graph.node(id);
        // Defensive: a well-formed SSD_CHUNK_SCAN node has exactly 5 inputs.
        // Malformed → fixpoint self-return (never panic).
        if n.inputs.len() != 5 {
            return id;
        }
        let x_shape = graph.node(n.inputs[0]).shape.clone();
        let dt_shape = graph.node(n.inputs[1]).shape.clone();
        let a_shape = graph.node(n.inputs[2]).shape.clone();
        let b_shape = graph.node(n.inputs[3]).shape.clone();
        let c_shape = graph.node(n.inputs[4]).shape.clone();
        (
            n.inputs[0], n.inputs[1], n.inputs[2], n.inputs[3], n.inputs[4],
            x_shape, dt_shape, a_shape, b_shape, c_shape, n.dtype,
        )
    };

    let x_dims = x_shape.dims();
    let b_dims = b_shape.dims();
    // Defensive shape guards — malformed → fixpoint self-return.
    if x_dims.len() != 4 || b_dims.len() != 4 {
        return id;
    }
    let batch = x_dims[0];
    let seqlen = x_dims[1];
    let heads = x_dims[2];
    let head_dim = x_dims[3];
    let state_dim = b_dims[3];

    let carry_shape = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
    let elem_x = Shape::from_dims(&[batch, heads, head_dim]); // x_t
    let elem_dt = Shape::from_dims(&[batch, heads]); // dt_t
    let elem_state = Shape::from_dims(&[batch, heads, state_dim]); // b_t / c_t

    // Broadcast helpers up to the carry shape [batch,heads,head_dim,state_dim].
    // Each captures only the Copy extents + dtype (no Shape borrow), building
    // the full target locally — mirrors selective_scan's `bc_from_*`.
    // dt_t [batch,heads] -> [batch,heads,1,1] -> carry.
    let bc_dt = |graph: &mut Graph, src: NodeId| -> NodeId {
        let mid = Shape::from_dims(&[batch, heads, 1, 1]);
        let full = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
        let re = graph.push(Node { op: Op::Reshape(mid.clone()), inputs: vec![src], shape: mid, dtype });
        graph.push(Node { op: Op::BroadcastTo(full.clone()), inputs: vec![re], shape: full, dtype })
    };
    // x_t [batch,heads,head_dim] -> [batch,heads,head_dim,1] -> carry.
    let bc_x = |graph: &mut Graph, src: NodeId| -> NodeId {
        let mid = Shape::from_dims(&[batch, heads, head_dim, 1]);
        let full = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
        let re = graph.push(Node { op: Op::Reshape(mid.clone()), inputs: vec![src], shape: mid, dtype });
        graph.push(Node { op: Op::BroadcastTo(full.clone()), inputs: vec![re], shape: full, dtype })
    };
    // {b_t,c_t} [batch,heads,state_dim] -> [batch,heads,1,state_dim] -> carry.
    let bc_state = |graph: &mut Graph, src: NodeId| -> NodeId {
        let mid = Shape::from_dims(&[batch, heads, 1, state_dim]);
        let full = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
        let re = graph.push(Node { op: Op::Reshape(mid.clone()), inputs: vec![src], shape: mid, dtype });
        graph.push(Node { op: Op::BroadcastTo(full.clone()), inputs: vec![re], shape: full, dtype })
    };
    // a [heads] -> [1,heads,1,1] -> carry (the scalar-per-head gate coefficient).
    let bc_a = |graph: &mut Graph, src: NodeId| -> NodeId {
        let mid = Shape::from_dims(&[1, heads, 1, 1]);
        let full = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
        let re = graph.push(Node { op: Op::Reshape(mid.clone()), inputs: vec![src], shape: mid, dtype });
        graph.push(Node { op: Op::BroadcastTo(full.clone()), inputs: vec![re], shape: full, dtype })
    };

    // ---- 1. per-step series with the scan (seqlen) axis moved to dim 0, so
    // `unroll_scan`'s `Slice{dim:0}` addresses each timestep.
    // x/b/c: Permute([1,0,2,3]); dt (rank 3): Permute([1,0,2]).
    let x_ser_shape = Shape::from_dims(&[seqlen, batch, heads, head_dim]);
    let dt_ser_shape = Shape::from_dims(&[seqlen, batch, heads]);
    let bc_ser_shape = Shape::from_dims(&[seqlen, batch, heads, state_dim]);
    let x_ser = graph.push(Node { op: Op::Permute(vec![1, 0, 2, 3]), inputs: vec![x_id], shape: x_ser_shape.clone(), dtype });
    let dt_ser = graph.push(Node { op: Op::Permute(vec![1, 0, 2]), inputs: vec![dt_id], shape: dt_ser_shape, dtype });
    let b_ser = graph.push(Node { op: Op::Permute(vec![1, 0, 2, 3]), inputs: vec![b_id], shape: bc_ser_shape.clone(), dtype });
    let c_ser = graph.push(Node { op: Op::Permute(vec![1, 0, 2, 3]), inputs: vec![c_id], shape: bc_ser_shape, dtype });

    // ---- 2. init_carry: zero [batch,heads,head_dim,state_dim], derived without
    // a data-carrying Const (a `decompose` fn has no device handle): broadcast
    // `a` up to the carry shape and multiply by 0.
    let a_bc0 = bc_a(graph, a_id);
    let init_carry = graph.push(Node {
        op: Op::MulScalar(0.0), inputs: vec![a_bc0], shape: carry_shape.clone(), dtype,
    });

    // ---- 3. the body: h ← exp(dt·a)·h + dt·b·x ; y = Σ_state(h·c). Carry h and
    // the per-step holes reference ScanPlaceholders; `a` is the const.
    let h = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Carry, index: 0 }, inputs: vec![], shape: carry_shape.clone(), dtype });
    let x_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 0 }, inputs: vec![], shape: elem_x.clone(), dtype });
    let dt_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 1 }, inputs: vec![], shape: elem_dt.clone(), dtype });
    let b_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 2 }, inputs: vec![], shape: elem_state.clone(), dtype });
    let c_t = graph.push(Node { op: Op::ScanPlaceholder { role: ScanRole::Elem, index: 3 }, inputs: vec![], shape: elem_state.clone(), dtype });

    // gate = Exp( bc(dt_t) ⊙ bc(a) ) — SCALAR-per-head gate (a is [heads])
    // broadcast across the whole head_dim×state_dim block.
    let dt_g = bc_dt(graph, dt_t);
    let a_g = bc_a(graph, a_id);
    let da = graph.push(Node { op: Op::Mul, inputs: vec![dt_g, a_g], shape: carry_shape.clone(), dtype });
    let gate = graph.push(Node { op: Op::Exp, inputs: vec![da], shape: carry_shape.clone(), dtype });

    // dbx = bc(dt_t) ⊙ bc(b_t) ⊙ bc(x_t)  (= dt · b · x, no softplus)
    let dt_b = bc_dt(graph, dt_t);
    let b_body = bc_state(graph, b_t);
    let x_body = bc_x(graph, x_t);
    let db = graph.push(Node { op: Op::Mul, inputs: vec![dt_b, b_body], shape: carry_shape.clone(), dtype });
    let dbx = graph.push(Node { op: Op::Mul, inputs: vec![db, x_body], shape: carry_shape.clone(), dtype });

    // h_new = gate ⊙ h + dbx   (body_new_carry)
    let gh = graph.push(Node { op: Op::Mul, inputs: vec![gate, h], shape: carry_shape.clone(), dtype });
    let h_new = graph.push(Node { op: Op::Add, inputs: vec![gh, dbx], shape: carry_shape.clone(), dtype });

    // y_t = ReduceSumTo_state( h_new ⊙ bc(c_t) ) -> [batch,heads,head_dim] (body_y)
    let c_body = bc_state(graph, c_t);
    let hc = graph.push(Node { op: Op::Mul, inputs: vec![h_new, c_body], shape: carry_shape.clone(), dtype });
    let y_keep_shape = Shape::from_dims(&[batch, heads, head_dim, 1]);
    let y_keep = graph.push(Node {
        op: Op::ReduceSumTo(y_keep_shape.clone()), inputs: vec![hc], shape: y_keep_shape, dtype,
    });
    let y_t = graph.push(Node {
        op: Op::Reshape(elem_x.clone()), inputs: vec![y_keep], shape: elem_x.clone(), dtype,
    });

    // ---- 4. the Op::Scan node. Natural 2-slot bundle: slot 0 = stacked ys
    // [seqlen,batch,heads,head_dim], slot 1 = final carry.
    let ys_stacked_shape = Shape::from_dims(&[seqlen, batch, heads, head_dim]);
    let scan = graph.push(Node {
        op: Op::Scan { n_xs: 4, bound: seqlen, emit: ScanEmit::All, early_exit: None },
        inputs: vec![init_carry, x_ser, dt_ser, b_ser, c_ser, a_id, h_new, y_t],
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

    // ---- 5. reconcile the axis order. `Op::Scan{emit=All}` stacks each y_t on
    // axis 0 -> [seqlen,batch,heads,head_dim]; `y` must be
    // [batch,seqlen,heads,head_dim]. Project slot 0 and transpose seqlen/batch.
    let ys_raw = graph.push(Node {
        op: Op::View { slot: 0 }, inputs: vec![scan], shape: ys_stacked_shape, dtype,
    });
    let y_shape = Shape::from_dims(&[batch, seqlen, heads, head_dim]);
    let y = graph.push(Node {
        op: Op::Permute(vec![1, 0, 2, 3]), inputs: vec![ys_raw], shape: y_shape, dtype,
    });

    // ---- 6. re-present the SsdChunkScan 2-slot bundle contract (slot 0 = y
    // [batch,seqlen,heads,head_dim], slot 1 = last_state [batch,heads,head_dim,
    // state_dim]) on the return node so the existing `Op::View(0)/(1)` consumers
    // keep their shapes. See the module note on the Phase-1 `last_state` realize
    // gap (kernel-less Op::Scan makes realize-through-decompose fail loudly).
    let input_shapes = [x_shape, dt_shape, a_shape, b_shape, c_shape];
    let input_dtypes = [dtype, dtype, dtype, dtype, dtype];
    let ss_specs = output_views(&input_shapes, &input_dtypes, params);
    if let Ok((_bytes, views)) = fuel_ir::storage::compose_bundle(&ss_specs) {
        let _ = graph.set_output_views(y, Arc::from(views.into_boxed_slice()));
    }
    y
}

/// Matcher stub — SsdChunkScan nodes originate from the explicit
/// `Tensor::ssd_chunk_scan` builder. The 100+ primitive subgraph in
/// fuel-transformers' eager `ssd_chunked` is too complex to pattern-
/// match conservatively.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}
