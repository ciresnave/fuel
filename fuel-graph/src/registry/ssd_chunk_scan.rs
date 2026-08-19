// SPDX-License-Identifier: MIT OR Apache-2.0
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
//! ## Why `BackwardKind::Decompose` (Op::Scan Phase 2)
//!
//! Differentiability comes from the Phase-2 `lower_scans_for_backward`
//! pre-pass (`fuel-graph/src/lib.rs`), NOT a bespoke `SSD_CHUNK_SCAN_BACKWARD`
//! fused op: `backward()` decomposes this node into its `Op::Scan` recipe,
//! `unroll_scan`s that to `bound` primitives, and runs the node-general
//! autograd over the unroll (truncated BPTT to the static `bound`).
//! `BackwardKind::Decompose` documents that intent — the field is
//! verified-dead metadata (the reverse walk never reads it; the pre-pass
//! replaces these nodes before the walk), so this is honesty about the
//! mechanism, not the mechanism itself.

use crate::registry::{
    BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternMatch,
    SubgraphPattern, decompose_via_recipe,
};
use crate::{Graph, NodeId};
use fuel_ir::storage::OutputViewSpec;
use fuel_ir::{DType, Layout, Shape};
use fuel_kernel_seam_types::shape_expr::{Dim, ShapeExpr};
use fuel_kernel_seam_types::{OpAttrs, OpTag, PatternNode, SCAN_ROLE_CARRY, SCAN_ROLE_ELEM};
use std::sync::Arc;

/// Metadata-side registry entry for SsdChunkScan. Multi-output (item
/// 3 consumer migration, 2026-06-01): slot 0 = `y`, slot 1 =
/// `last_state`. Mirrors `selective_scan::entry`.
pub fn entry() -> FusedOpEntry {
    FusedOpEntry {
        destructive_input: None,
        id: FusedOps::SSD_CHUNK_SCAN,
        name: "SsdChunkScan",
        family: FusedOpFamily::Forward,
        pattern: SubgraphPattern::Callable(canonical_pattern),
        decompose,
        backward: BackwardKind::Decompose,
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
        input_shapes.len(),
        5,
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
    _params: &FusedOpParams,
) -> Vec<OutputViewSpec> {
    debug_assert_eq!(
        input_shapes.len(),
        5,
        "SsdChunkScan output_views: takes 5 inputs (x, dt, a, b, c)",
    );
    debug_assert_eq!(
        input_dtypes.len(),
        5,
        "SsdChunkScan output_views: takes 5 input dtypes",
    );
    let x_dims = input_shapes[0].dims();
    let b_dims = input_shapes[3].dims();
    debug_assert!(
        x_dims.len() == 4 && b_dims.len() == 4,
        "SsdChunkScan output_views: x rank=4, b rank=4 expected",
    );
    let batch = x_dims[0];
    let seqlen = x_dims[1];
    let heads = x_dims[2];
    let head_dim = x_dims[3];
    let state_dim = b_dims[3];
    let dtype = input_dtypes[0];
    let y_shape = Shape::from_dims(&[batch, seqlen, heads, head_dim]);
    let last_state_shape = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
    vec![
        OutputViewSpec {
            dtype,
            shape: y_shape.clone(),
            layout: Layout::contiguous(y_shape),
            name: Some("y"),
        },
        OutputViewSpec {
            dtype,
            shape: last_state_shape.clone(),
            layout: Layout::contiguous(last_state_shape),
            name: Some("last_state"),
        },
    ]
}

/// Dtype rule: output matches `x`'s dtype (input 0).
fn dtype_rule(input_dtypes: &[DType], _params: &FusedOpParams) -> DType {
    debug_assert_eq!(input_dtypes.len(), 5, "SsdChunkScan takes 5 inputs",);
    input_dtypes[0]
}

/// SsdChunkScan's `Op::Scan` recipe as portable `PatternNode` DATA (Increment
/// C, B2) — the structure-preserving migration of the pre-B2 imperative body
/// onto the B1 re-emit machinery; the twin of `selective_scan::recipe` with NO
/// softplus and a per-head SCALAR gate. `Op::Scan` carries the affine Mamba-2
/// SSD step: carry `h [batch,heads,head_dim,state_dim]` (zero-init), four
/// per-step slices (`x_t`, `dt_t`, `b_t`, `c_t`), and `a` as the single const;
/// `bound = seqlen` (baked per call), `emit = All`. The body is
/// `gate = exp(bc(dt_t)*bc(a))` — `a` is `[heads]`, broadcast across the whole
/// `head_dim*state_dim` block — `dbx = bc(dt_t)*bc(b_t)*bc(x_t)`,
/// `h_new = gate*h + dbx`, `y_t = sum_state(h_new*bc(c_t))`.
///
/// Bind space: `0 = x`, `1 = dt`, `2 = a`, `3 = b`, `4 = c`. Every interior
/// shape rides a §6.20 `Dims`/`Extent` target (`batch = x.0`, `heads = x.2`,
/// `head_dim = x.3`, `state_dim = b.3`), and each `ScanPlaceholder` carries its
/// declared per-step shape via `target_shape_rel` (B2's emit extension), so the
/// re-emitted body matches the imperative one node-for-node. No open scalar
/// slots (the zero-init `MulScalar(0.0)` is a baked constant). `chunk_size` is a
/// documented CPU no-op (same sequential scan regardless), so it never enters
/// the recipe.
fn recipe(seqlen: usize) -> PatternNode {
    use OpTag as T;
    let ext = |operand: u8, axis: u8| Dim::Extent { operand, axis };
    let one = || Dim::Const(1);
    let batch = || ext(0, 0);
    let heads = || ext(0, 2);
    let head_dim = || ext(0, 3);
    let state_dim = || ext(3, 3);
    let dims = |ds: Vec<Dim>| OpAttrs {
        target_shape_rel: Some(ShapeExpr::Dims(ds)),
        ..OpAttrs::default()
    };
    let op = |op, attrs, operands| PatternNode::Op {
        op,
        attrs,
        operands,
    };
    let bind = |i| PatternNode::Bind { index: i };
    let carry = || vec![batch(), heads(), head_dim(), state_dim()]; // [b,h,hd,s]
    let elem_x = || vec![batch(), heads(), head_dim()]; // [b,h,hd]
    let elem_dt = || vec![batch(), heads()]; // [b,h]
    let elem_state = || vec![batch(), heads(), state_dim()]; // [b,h,s]
    let reshape = |src, ds: Vec<Dim>| op(T::Reshape, dims(ds), vec![src]);
    let bcast = |src, ds: Vec<Dim>| op(T::BroadcastTo, dims(ds), vec![src]);
    // Broadcast helpers up to carry [b,h,hd,s] — mirror the imperative bc_*.
    let bc_dt = |src| bcast(reshape(src, vec![batch(), heads(), one(), one()]), carry());
    let bc_x = |src| {
        bcast(
            reshape(src, vec![batch(), heads(), head_dim(), one()]),
            carry(),
        )
    };
    let bc_state = |src| {
        bcast(
            reshape(src, vec![batch(), heads(), one(), state_dim()]),
            carry(),
        )
    };
    // a [heads] -> [1,heads,1,1] -> carry (per-head scalar gate coefficient),
    // used by BOTH init and the gate — emit's identity-share collapses them.
    let bc_a = || {
        bcast(
            reshape(bind(2), vec![one(), heads(), one(), one()]),
            carry(),
        )
    };
    let permute4 = |src| {
        op(
            T::Permute,
            OpAttrs {
                perm: vec![1, 0, 2, 3],
                ..OpAttrs::default()
            },
            vec![src],
        )
    };
    let permute3 = |src| {
        op(
            T::Permute,
            OpAttrs {
                perm: vec![1, 0, 2],
                ..OpAttrs::default()
            },
            vec![src],
        )
    };
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

    // ---- 1. per-step series (scan axis -> dim 0). x/b/c: Permute([1,0,2,3]);
    // dt (rank 3): Permute([1,0,2]).
    let x_ser = permute4(bind(0));
    let dt_ser = permute3(bind(1));
    let b_ser = permute4(bind(3));
    let c_ser = permute4(bind(4));

    // ---- 2. init_carry = MulScalar(0)(a-broadcast-to-carry).
    let init_carry = op(
        T::MulScalar,
        OpAttrs {
            scalars: vec![0.0],
            ..OpAttrs::default()
        },
        vec![bc_a()],
    );

    // ---- 3. the body. Placeholders carry declared shapes; `a` is the const.
    let h = ph(SCAN_ROLE_CARRY, 0, carry());
    let x_t = ph(SCAN_ROLE_ELEM, 0, elem_x());
    let dt_t = ph(SCAN_ROLE_ELEM, 1, elem_dt());
    let b_t = ph(SCAN_ROLE_ELEM, 2, elem_state());
    let c_t = ph(SCAN_ROLE_ELEM, 3, elem_state());

    // gate = Exp( bc(dt_t) * bc(a) ) — per-head scalar gate.
    let da = op(
        T::Mul,
        OpAttrs::default(),
        vec![bc_dt(dt_t.clone()), bc_a()],
    );
    let gate = op(T::Exp, OpAttrs::default(), vec![da]);
    // dbx = bc(dt_t) * bc(b_t) * bc(x_t)  (= dt * b * x, no softplus)
    let db = op(T::Mul, OpAttrs::default(), vec![bc_dt(dt_t), bc_state(b_t)]);
    let dbx = op(T::Mul, OpAttrs::default(), vec![db, bc_x(x_t)]);
    // h_new = gate * h + dbx   (body_new_carry)
    let gh = op(T::Mul, OpAttrs::default(), vec![gate, h]);
    let h_new = op(T::Add, OpAttrs::default(), vec![gh, dbx]);
    // y_t = Reshape([b,h,hd])( ReduceSumTo([b,h,hd,1])( h_new * bc(c_t) ) )  (body_y)
    let hc = op(
        T::Mul,
        OpAttrs::default(),
        vec![h_new.clone(), bc_state(c_t)],
    );
    let y_keep = op(
        T::ReduceSumTo,
        dims(vec![batch(), heads(), head_dim(), one()]),
        vec![hc],
    );
    let y_t = op(T::Reshape, dims(elem_x()), vec![y_keep]);

    // ---- 4. the Op::Scan node (n_xs=4, bound=seqlen, emit=All).
    let scan = PatternNode::Op {
        op: T::Scan,
        attrs: OpAttrs {
            scan_n_xs: Some(4),
            scan_bound: Some(seqlen as u32),
            scan_emit: Some(0), // ScanEmit::All
            scan_early_exit: None,
            ..OpAttrs::default()
        },
        operands: vec![init_carry, x_ser, dt_ser, b_ser, c_ser, bind(2), h_new, y_t],
    };

    // ---- 5. project slot 0 (stacked ys [seqlen,batch,heads,head_dim]) +
    // transpose the seqlen/batch axes back to y [batch,seqlen,heads,head_dim].
    let ys_raw = op(
        T::View,
        OpAttrs {
            view_slot: Some(0),
            ..OpAttrs::default()
        },
        vec![scan],
    );
    op(
        T::Permute,
        OpAttrs {
            perm: vec![1, 0, 2, 3],
            ..OpAttrs::default()
        },
        vec![ys_raw],
    )
}

/// Total decomposition of SsdChunkScan to an [`crate::Op::Scan`] recipe —
/// closing decisions-log G3 ("a higher-order `Scan` for SSMs"), part 2 (the twin
/// of `selective_scan`). Since Increment C B2 a re-emit of [`recipe`]'s portable
/// data through the [`decompose_via_recipe`] bridge (structure-preserving: the
/// emitted base map is node-for-node identical to the pre-B2 imperative body).
/// `seqlen` (the scan bound, a shape-dependent structural param) is read here
/// and baked into the per-call recipe; all other extents ride §6.20 `Dims`
/// rel-attrs. `chunk_size` is a documented CPU no-op (both this and the chunked
/// GPU path re-decompose to the SAME sequential scan), so it never enters the
/// recipe.
///
/// Per G2 (2026-06-20) this is total + never-panic: the wrong-params /
/// malformed-shape guards and any bridge decline return `id` (the driver's
/// fixpoint signal, not a crash). The recipe is the *math* the kernel computes;
/// the fused CPU/CUDA kernel stays the executed production path — the optimizer
/// chooses by cost, and `unroll_scan` provides the verify oracle. The returned
/// node re-attaches the 2-slot `output_views` bundle contract (slot 0 = `y
/// [batch,seqlen,heads,head_dim]`, slot 1 = `last_state`) — see the module note
/// on the Phase-1 `last_state`-realize limitation.
pub fn decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
    // Confirm this id really is an SSD_CHUNK_SCAN node. `chunk_size` is a
    // documented CPU no-op (same sequential scan regardless), so we don't read
    // its value into the recipe.
    match params {
        FusedOpParams::SsdChunkScan { .. } => {}
        // Wrong params for this id — can't decompose; return self (fixpoint).
        _ => return id,
    }

    // Read the 5 input shapes/dtype in a short borrow. `seqlen` (the scan
    // bound) is baked into the recipe; the shapes feed the bundle re-attach.
    let (x_shape, dt_shape, a_shape, b_shape, c_shape, dtype) = {
        let n = graph.node(id);
        // Defensive: a well-formed SSD_CHUNK_SCAN node has exactly 5 inputs.
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
    let x_dims = x_shape.dims();
    let b_dims = b_shape.dims();
    // Defensive shape guards — malformed → fixpoint self-return.
    if x_dims.len() != 4 || b_dims.len() != 4 {
        return id;
    }
    let seqlen = x_dims[1];

    let recipe_node = recipe(seqlen);
    // No open scalar slots (zero-init 0.0 is a baked constant).
    let root = decompose_via_recipe(graph, id, &recipe_node, Some(Vec::new()));
    if root == id {
        // Bridge declined (bind-arity, rel-resolution, …) — stay fused (G2).
        return id;
    }

    // Re-present the SsdChunkScan 2-slot bundle contract on the returned Permute
    // so existing `Op::View(0)/(1)` consumers keep their shapes (the recipe emit
    // attaches the [ys,carry] bundle to the inner `Op::Scan`, not to this outer
    // node) — mirrors the imperative decompose's step 6.
    let input_shapes = [x_shape, dt_shape, a_shape, b_shape, c_shape];
    let input_dtypes = [dtype, dtype, dtype, dtype, dtype];
    let ss_specs = output_views(&input_shapes, &input_dtypes, params);
    if let Ok((_bytes, views)) = fuel_ir::storage::compose_bundle(&ss_specs) {
        let _ = graph.set_output_views(root, Arc::from(views.into_boxed_slice()));
    }
    root
}

/// Matcher stub — SsdChunkScan nodes originate from the explicit
/// `Tensor::ssd_chunk_scan` builder. The 100+ primitive subgraph in
/// fuel-transformers' eager `ssd_chunked` is too complex to pattern-
/// match conservatively.
pub fn canonical_pattern(_graph: &Graph, _root: NodeId) -> Option<PatternMatch> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::FusedOps;
    use crate::{Node, Op, ScanEmit, ScanRole};

    /// FROZEN copy of the pre-B2 imperative `ssd_chunk_scan::decompose` body,
    /// verbatim, before B2 replaced the live body with the data recipe — the
    /// B2 structure-preservation oracle.
    fn frozen_legacy_decompose(graph: &mut Graph, id: NodeId, params: &FusedOpParams) -> NodeId {
        match params {
            FusedOpParams::SsdChunkScan { .. } => {}
            _ => return id,
        }
        let (x_id, dt_id, a_id, b_id, c_id, x_shape, dt_shape, a_shape, b_shape, c_shape, dtype) = {
            let n = graph.node(id);
            if n.inputs.len() != 5 {
                return id;
            }
            let x_shape = graph.node(n.inputs[0]).shape.clone();
            let dt_shape = graph.node(n.inputs[1]).shape.clone();
            let a_shape = graph.node(n.inputs[2]).shape.clone();
            let b_shape = graph.node(n.inputs[3]).shape.clone();
            let c_shape = graph.node(n.inputs[4]).shape.clone();
            (
                n.inputs[0],
                n.inputs[1],
                n.inputs[2],
                n.inputs[3],
                n.inputs[4],
                x_shape,
                dt_shape,
                a_shape,
                b_shape,
                c_shape,
                n.dtype,
            )
        };
        let x_dims = x_shape.dims();
        let b_dims = b_shape.dims();
        if x_dims.len() != 4 || b_dims.len() != 4 {
            return id;
        }
        let batch = x_dims[0];
        let seqlen = x_dims[1];
        let heads = x_dims[2];
        let head_dim = x_dims[3];
        let state_dim = b_dims[3];

        let carry_shape = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
        let elem_x = Shape::from_dims(&[batch, heads, head_dim]);
        let elem_dt = Shape::from_dims(&[batch, heads]);
        let elem_state = Shape::from_dims(&[batch, heads, state_dim]);

        let bc_dt = |graph: &mut Graph, src: NodeId| -> NodeId {
            let mid = Shape::from_dims(&[batch, heads, 1, 1]);
            let full = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
            let re = graph.push(Node {
                op: Op::Reshape(mid.clone()),
                inputs: vec![src],
                shape: mid,
                dtype,
            });
            graph.push(Node {
                op: Op::BroadcastTo(full.clone()),
                inputs: vec![re],
                shape: full,
                dtype,
            })
        };
        let bc_x = |graph: &mut Graph, src: NodeId| -> NodeId {
            let mid = Shape::from_dims(&[batch, heads, head_dim, 1]);
            let full = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
            let re = graph.push(Node {
                op: Op::Reshape(mid.clone()),
                inputs: vec![src],
                shape: mid,
                dtype,
            });
            graph.push(Node {
                op: Op::BroadcastTo(full.clone()),
                inputs: vec![re],
                shape: full,
                dtype,
            })
        };
        let bc_state = |graph: &mut Graph, src: NodeId| -> NodeId {
            let mid = Shape::from_dims(&[batch, heads, 1, state_dim]);
            let full = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
            let re = graph.push(Node {
                op: Op::Reshape(mid.clone()),
                inputs: vec![src],
                shape: mid,
                dtype,
            });
            graph.push(Node {
                op: Op::BroadcastTo(full.clone()),
                inputs: vec![re],
                shape: full,
                dtype,
            })
        };
        let bc_a = |graph: &mut Graph, src: NodeId| -> NodeId {
            let mid = Shape::from_dims(&[1, heads, 1, 1]);
            let full = Shape::from_dims(&[batch, heads, head_dim, state_dim]);
            let re = graph.push(Node {
                op: Op::Reshape(mid.clone()),
                inputs: vec![src],
                shape: mid,
                dtype,
            });
            graph.push(Node {
                op: Op::BroadcastTo(full.clone()),
                inputs: vec![re],
                shape: full,
                dtype,
            })
        };

        let x_ser_shape = Shape::from_dims(&[seqlen, batch, heads, head_dim]);
        let dt_ser_shape = Shape::from_dims(&[seqlen, batch, heads]);
        let bc_ser_shape = Shape::from_dims(&[seqlen, batch, heads, state_dim]);
        let x_ser = graph.push(Node {
            op: Op::Permute(vec![1, 0, 2, 3]),
            inputs: vec![x_id],
            shape: x_ser_shape.clone(),
            dtype,
        });
        let dt_ser = graph.push(Node {
            op: Op::Permute(vec![1, 0, 2]),
            inputs: vec![dt_id],
            shape: dt_ser_shape,
            dtype,
        });
        let b_ser = graph.push(Node {
            op: Op::Permute(vec![1, 0, 2, 3]),
            inputs: vec![b_id],
            shape: bc_ser_shape.clone(),
            dtype,
        });
        let c_ser = graph.push(Node {
            op: Op::Permute(vec![1, 0, 2, 3]),
            inputs: vec![c_id],
            shape: bc_ser_shape,
            dtype,
        });

        let a_bc0 = bc_a(graph, a_id);
        let init_carry = graph.push(Node {
            op: Op::MulScalar(0.0),
            inputs: vec![a_bc0],
            shape: carry_shape.clone(),
            dtype,
        });

        let h = graph.push(Node {
            op: Op::ScanPlaceholder {
                role: ScanRole::Carry,
                index: 0,
            },
            inputs: vec![],
            shape: carry_shape.clone(),
            dtype,
        });
        let x_t = graph.push(Node {
            op: Op::ScanPlaceholder {
                role: ScanRole::Elem,
                index: 0,
            },
            inputs: vec![],
            shape: elem_x.clone(),
            dtype,
        });
        let dt_t = graph.push(Node {
            op: Op::ScanPlaceholder {
                role: ScanRole::Elem,
                index: 1,
            },
            inputs: vec![],
            shape: elem_dt.clone(),
            dtype,
        });
        let b_t = graph.push(Node {
            op: Op::ScanPlaceholder {
                role: ScanRole::Elem,
                index: 2,
            },
            inputs: vec![],
            shape: elem_state.clone(),
            dtype,
        });
        let c_t = graph.push(Node {
            op: Op::ScanPlaceholder {
                role: ScanRole::Elem,
                index: 3,
            },
            inputs: vec![],
            shape: elem_state.clone(),
            dtype,
        });

        let dt_g = bc_dt(graph, dt_t);
        let a_g = bc_a(graph, a_id);
        let da = graph.push(Node {
            op: Op::Mul,
            inputs: vec![dt_g, a_g],
            shape: carry_shape.clone(),
            dtype,
        });
        let gate = graph.push(Node {
            op: Op::Exp,
            inputs: vec![da],
            shape: carry_shape.clone(),
            dtype,
        });

        let dt_b = bc_dt(graph, dt_t);
        let b_body = bc_state(graph, b_t);
        let x_body = bc_x(graph, x_t);
        let db = graph.push(Node {
            op: Op::Mul,
            inputs: vec![dt_b, b_body],
            shape: carry_shape.clone(),
            dtype,
        });
        let dbx = graph.push(Node {
            op: Op::Mul,
            inputs: vec![db, x_body],
            shape: carry_shape.clone(),
            dtype,
        });

        let gh = graph.push(Node {
            op: Op::Mul,
            inputs: vec![gate, h],
            shape: carry_shape.clone(),
            dtype,
        });
        let h_new = graph.push(Node {
            op: Op::Add,
            inputs: vec![gh, dbx],
            shape: carry_shape.clone(),
            dtype,
        });

        let c_body = bc_state(graph, c_t);
        let hc = graph.push(Node {
            op: Op::Mul,
            inputs: vec![h_new, c_body],
            shape: carry_shape.clone(),
            dtype,
        });
        let y_keep_shape = Shape::from_dims(&[batch, heads, head_dim, 1]);
        let y_keep = graph.push(Node {
            op: Op::ReduceSumTo(y_keep_shape.clone()),
            inputs: vec![hc],
            shape: y_keep_shape,
            dtype,
        });
        let y_t = graph.push(Node {
            op: Op::Reshape(elem_x.clone()),
            inputs: vec![y_keep],
            shape: elem_x.clone(),
            dtype,
        });

        let ys_stacked_shape = Shape::from_dims(&[seqlen, batch, heads, head_dim]);
        let scan = graph.push(Node {
            op: Op::Scan {
                n_xs: 4,
                bound: seqlen,
                emit: ScanEmit::All,
                early_exit: None,
            },
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

        let ys_raw = graph.push(Node {
            op: Op::View { slot: 0 },
            inputs: vec![scan],
            shape: ys_stacked_shape,
            dtype,
        });
        let y_shape = Shape::from_dims(&[batch, seqlen, heads, head_dim]);
        let y = graph.push(Node {
            op: Op::Permute(vec![1, 0, 2, 3]),
            inputs: vec![ys_raw],
            shape: y_shape,
            dtype,
        });

        let input_shapes = [x_shape, dt_shape, a_shape, b_shape, c_shape];
        let input_dtypes = [dtype, dtype, dtype, dtype, dtype];
        let ss_specs = output_views(&input_shapes, &input_dtypes, params);
        if let Ok((_bytes, views)) = fuel_ir::storage::compose_bundle(&ss_specs) {
            let _ = graph.set_output_views(y, Arc::from(views.into_boxed_slice()));
        }
        y
    }

    /// Recursively assert two subgraphs are node-for-node identical (op, shape,
    /// dtype, arity, recursively-equal inputs). Shape-sensitive at every node.
    fn assert_structural_eq(g: &Graph, a: NodeId, b: NodeId) {
        if a == b {
            return;
        }
        let na = g.node(a);
        let nb = g.node(b);
        assert_eq!(na.op, nb.op, "op mismatch: {:?} vs {:?}", na.op, nb.op);
        assert_eq!(
            na.shape, nb.shape,
            "shape mismatch at {:?}: {:?} vs {:?}",
            na.op, na.shape, nb.shape
        );
        assert_eq!(na.dtype, nb.dtype, "dtype mismatch at {:?}", na.op);
        assert_eq!(
            na.inputs.len(),
            nb.inputs.len(),
            "arity mismatch at {:?}",
            na.op
        );
        for (&ia, &ib) in na.inputs.iter().zip(nb.inputs.iter()) {
            assert_structural_eq(g, ia, ib);
        }
    }

    /// Build a fused SSD_CHUNK_SCAN node over `x [b,seq,h,hd]`, `dt [b,seq,h]`,
    /// `a [h]`, `b,c [b,seq,h,s]`. Returns the fused NodeId.
    fn fused_node(
        g: &mut Graph,
        batch: usize,
        seqlen: usize,
        heads: usize,
        head_dim: usize,
        state_dim: usize,
    ) -> NodeId {
        let mk = |g: &mut Graph, dims: &[usize]| {
            g.push(Node {
                op: Op::Const,
                inputs: vec![],
                shape: Shape::from_dims(dims),
                dtype: DType::F32,
            })
        };
        let x = mk(g, &[batch, seqlen, heads, head_dim]);
        let dt = mk(g, &[batch, seqlen, heads]);
        let a = mk(g, &[heads]);
        let b = mk(g, &[batch, seqlen, heads, state_dim]);
        let c = mk(g, &[batch, seqlen, heads, state_dim]);
        g.push(Node {
            op: Op::Fused(
                FusedOps::SSD_CHUNK_SCAN,
                FusedOpParams::SsdChunkScan { chunk_size: 1 },
            ),
            inputs: vec![x, dt, a, b, c],
            shape: Shape::from_dims(&[batch, seqlen, heads, head_dim]),
            dtype: DType::F32,
        })
    }

    /// B2: ONE recipe form decomposes at multiple `batch/seqlen/heads/head_dim/
    /// state_dim` shapes, and its emitted base map is node-for-node identical to
    /// the FROZEN pre-B2 imperative body. Born-red without B2's emit
    /// `ScanPlaceholder`-shape carrier.
    #[test]
    fn ssd_chunk_scan_recipe_decompose_is_polymorphic_and_matches_frozen_legacy() {
        for (batch, seqlen, heads, head_dim, state_dim) in [(1usize, 3, 2, 2, 4), (2, 4, 1, 3, 2)] {
            let params = FusedOpParams::SsdChunkScan { chunk_size: 1 };
            let mut g = Graph::new();
            let fused = fused_node(&mut g, batch, seqlen, heads, head_dim, state_dim);
            let out_sh = g.node(fused).shape.clone();

            let new_root = decompose(&mut g, fused, &params);
            assert_ne!(
                new_root, fused,
                "recipe decompose fires ([{batch},{seqlen},{heads},{head_dim},{state_dim}])"
            );
            assert_eq!(
                g.node(new_root).shape,
                out_sh,
                "output y = [batch,seqlen,heads,head_dim]"
            );
            assert_eq!(g.node(new_root).dtype, DType::F32);
            assert_eq!(
                g.output_views(new_root).map(|v| v.len()),
                Some(2),
                "SsdChunkScan 2-slot output_views re-attached on the recipe root",
            );

            let legacy_root = frozen_legacy_decompose(&mut g, fused, &params);
            assert_structural_eq(&g, new_root, legacy_root);
        }
    }

    /// `chunk_size` is a CPU no-op: two different `chunk_size` values decompose
    /// to the SAME base map (it never enters the recipe).
    #[test]
    fn ssd_chunk_scan_recipe_ignores_chunk_size() {
        use crate::opt::base_map_hash;
        let mut g = Graph::new();
        let f1 = fused_node(&mut g, 1, 4, 1, 2, 2);
        // Rebuild the same fused node but with a different chunk_size param.
        let n = g.node(f1);
        let inputs = n.inputs.clone();
        let shape = n.shape.clone();
        let f2 = g.push(Node {
            op: Op::Fused(
                FusedOps::SSD_CHUNK_SCAN,
                FusedOpParams::SsdChunkScan { chunk_size: 4 },
            ),
            inputs,
            shape,
            dtype: DType::F32,
        });
        let r1 = decompose(&mut g, f1, &FusedOpParams::SsdChunkScan { chunk_size: 1 });
        let r2 = decompose(&mut g, f2, &FusedOpParams::SsdChunkScan { chunk_size: 4 });
        assert_eq!(
            base_map_hash(&g, r1),
            base_map_hash(&g, r2),
            "chunk_size does not change the recipe base map"
        );
    }

    /// Totality (G2): a wrong params payload declines to a fixpoint, never a
    /// crash, before any emission.
    #[test]
    fn ssd_chunk_scan_recipe_wrong_params_is_a_fixpoint_not_a_crash() {
        let mut g = Graph::new();
        let fused = fused_node(&mut g, 1, 3, 2, 2, 4);
        let before = g.len();
        let out = decompose(&mut g, fused, &FusedOpParams::Rope);
        assert_eq!(out, fused, "wrong params => typed decline => fixpoint");
        assert_eq!(g.len(), before, "declined before any emission");
    }
}
