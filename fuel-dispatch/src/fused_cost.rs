//! Fused-op Layer-1 cost **composed from its decomposition**
//! (`docs/session-prompts/fused-op-cost-from-decompose.md`).
//!
//! A fused / synthesized op with no declared or measured cost is
//! registered with the [`fused_unknown_cost`](crate::fkc::fused_unknown_cost)
//! sentinel → [`CostEstimate::default()`] → **zero**. Zero is the exact
//! mis-pricing Part A fixed for GPUs: a zero-priced candidate wins
//! spuriously. This module replaces that zero with a cost **composed from
//! the op's own decomposition**.
//!
//! The recipe principle guarantees every fused op carries a total
//! `decompose` (fused → equivalent primitive subgraph), and every
//! primitive has a Layer-1 cost fn
//! ([`default_cost_for_op_kind`](crate::cost::default_cost_for_op_kind)).
//! [`cost_from_decompose`] folds the two: it emits the decompose subgraph
//! into a scratch [`Graph`], sums the primitive nodes' FLOPs, and pairs
//! that with the FUSED op's own boundary I/O bytes and a single launch
//! overhead.
//!
//! ## Layering (spec §5)
//!
//! measured (Judge, Layer-2) › declared (contract, Task-F) › **composed
//! (this module)** › (never) zero. [`fused_layer1_cost`] is the accessor
//! that enforces it: a fused op WITH a real `cost:` fn is priced by that
//! fn unchanged; only the [`fused_unknown_cost`] sentinel derives its cost
//! from the recipe.

use std::collections::HashSet;

use fuel_ir::backend::BackendCapabilities;
use fuel_ir::{DType, Shape};
use fuel_graph::registry::{FusedOpId, FusedOpParams};
use fuel_graph::{Graph, Node, NodeId, Op};

use crate::fused::{BackendImpl, CostEstimate};
use crate::kernel::OpParams;

/// Guard against a pathological (mis-authored) decompose that never
/// reaches primitives — bounds the nested-fused lowering recursion. The
/// base map is `decompose`'s fixpoint, so a well-formed op terminates in
/// one shot; this only ever fires defensively (never-panic).
const MAX_DECOMPOSE_DEPTH: u32 = 16;

/// Element count of a (concrete) shape, as `u64`. Mirrors the
/// per-family cost fns' own element counting.
fn elem_count(shape: &Shape) -> u64 {
    shape.dims().iter().map(|&d| d as u64).product::<u64>()
}

/// Whether `cost` is the fused-op cost sentinel — the fused analog of the
/// primitive `unknown_cost` identity-compare that `fill_unset_*` already
/// does. A `true` here means "derive me from my recipe."
pub fn is_fused_cost_sentinel(
    cost: fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate,
) -> bool {
    cost as *const () as usize
        == crate::fkc::fused_unknown_cost as *const () as usize
}

/// The fused-op Layer-1 cost accessor (spec §4 shape A — the sentinel
/// fallback at the use site).
///
/// - A fused op whose registered `cost` is the [`fused_unknown_cost`]
///   sentinel gets [`cost_from_decompose`] (composed-from-recipe), never
///   the zero.
/// - A fused op WITH a declared/measured cost fn is priced by that fn,
///   **unchanged** — the default only fires for the sentinel, keeping the
///   layering measured › declared › composed › (never) zero.
///
/// [`fused_unknown_cost`]: crate::fkc::fused_unknown_cost
pub fn fused_layer1_cost(
    impl_: &BackendImpl,
    id: FusedOpId,
    input_shapes: &[Shape],
    input_dtypes: &[DType],
    params: &FusedOpParams,
    caps: &BackendCapabilities,
) -> CostEstimate {
    // FKC gap-closure Task 2.4 (§2.3): a declared cost AST outranks both the
    // hand-written `cost` fn and the decompose-derived fallback below — it's
    // the contract's own priced formula, layered above "composed" and below
    // "measured" (spec §5: measured › declared › composed › never zero).
    if let Some(expr) = impl_.cost_expr {
        if let Ok(est) = crate::fkc::cost_compile::fused_cost_estimate(expr, input_shapes, input_dtypes, params) {
            return est;
        }
    }
    if is_fused_cost_sentinel(impl_.cost) {
        cost_from_decompose(id, params, input_shapes, input_dtypes, caps)
    } else {
        (impl_.cost)(input_shapes, params, caps)
    }
}

/// Compose a fused op's Layer-1 [`CostEstimate`] from its decomposition
/// (spec §2, recommended v1 = *tight* estimate):
///
/// - **`flops`** = `Σ` over the decompose subgraph's primitive nodes of
///   `default_cost_for_op_kind(node.op)(node_shapes, node_dtypes, ..).flops`
///   — arithmetically exact for algebraic fusions.
/// - **`bytes_moved`** = the FUSED op's own **boundary I/O** (its declared
///   inputs + final output) — fusion elides the intermediates, so this is
///   the *tight* estimate, not `Σ` intermediate bytes.
/// - **`kernel_overhead_ns`** = **one** launch overhead (the `max` of the
///   component primitives' per-launch overheads — the fused kernel launches
///   once, its overhead ≈ its heaviest component's, not `Σ`).
///
/// Never-panic (spec §2): a degenerate / fixpoint / unregistered decompose
/// falls back to the sentinel-equivalent [`CostEstimate::default()`], never
/// a crash. Nested fused ops in the decompose recurse (bounded by
/// [`MAX_DECOMPOSE_DEPTH`]) — the base map is `decompose`'s fixpoint.
pub fn cost_from_decompose(
    id: FusedOpId,
    params: &FusedOpParams,
    input_shapes: &[Shape],
    input_dtypes: &[DType],
    caps: &BackendCapabilities,
) -> CostEstimate {
    cost_from_decompose_inner(id, params, input_shapes, input_dtypes, caps, 0)
}

fn cost_from_decompose_inner(
    id: FusedOpId,
    params: &FusedOpParams,
    input_shapes: &[Shape],
    input_dtypes: &[DType],
    caps: &BackendCapabilities,
    depth: u32,
) -> CostEstimate {
    // Never-panic guards (spec §2): a degenerate operand list has no
    // recipe to fold → fall back to the sentinel-equivalent zero.
    if input_shapes.is_empty() || input_shapes.len() != input_dtypes.len() {
        return CostEstimate::default();
    }

    // Emit `Op::Fused(id, params)` over fresh input leaves into a scratch
    // graph, then `decompose` it in place — no permanent graph mutation.
    let mut g = Graph::new();
    let mut leaves: HashSet<NodeId> = HashSet::with_capacity(input_shapes.len());
    let mut input_ids = Vec::with_capacity(input_shapes.len());
    for (s, d) in input_shapes.iter().zip(input_dtypes.iter()) {
        let nid = g.push(Node {
            op: Op::Const,
            inputs: vec![],
            shape: s.clone(),
            dtype: *d,
        });
        leaves.insert(nid);
        input_ids.push(nid);
    }
    // The fused node's own shape/dtype are not load-bearing for the fold —
    // `decompose` recomputes every interior shape from the inputs + params.
    // A placeholder (input[0]) keeps the node well-formed.
    let fused = g.push(Node {
        op: Op::Fused(id, params.clone()),
        inputs: input_ids,
        shape: input_shapes[0].clone(),
        dtype: input_dtypes[0],
    });

    // Decompose: the runtime sidecar re-emits its region; a static op uses
    // its registered `decompose`. Both are total + never-panic (recipe
    // principle). An unregistered id → no recipe → sentinel zero.
    let root = if id.is_runtime() {
        fuel_graph::runtime_fused::decompose_region(&mut g, fused)
    } else {
        match fuel_graph::registry::default_registry().entry(id) {
            Some(entry) => (entry.decompose)(&mut g, fused, params),
            None => return CostEstimate::default(),
        }
    };

    // A fixpoint (op with no recipe returns itself, G2) or a decompose that
    // resolves straight to an input leaf → no primitive subgraph to price →
    // sentinel zero, not a fabricated cost.
    if root == fused || leaves.contains(&root) {
        return CostEstimate::default();
    }

    // FLOPs = Σ over the decompose subgraph's primitive nodes; one launch
    // overhead = max of the component primitives' per-launch overheads.
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut max_overhead: u32 = 0;
    let flops = fold_subgraph(&g, root, &leaves, &mut visited, caps, &mut max_overhead, depth);

    // bytes_moved = the FUSED op's own boundary I/O (inputs + final output),
    // NOT Σ intermediate bytes — fusion elides the intermediates (spec §2).
    let mut bytes: u64 = 0;
    for (s, d) in input_shapes.iter().zip(input_dtypes.iter()) {
        bytes = bytes.saturating_add(elem_count(s).saturating_mul(crate::cost::dtype_bytes(*d)));
    }
    let out = g.node(root);
    bytes = bytes
        .saturating_add(elem_count(&out.shape).saturating_mul(crate::cost::dtype_bytes(out.dtype)));

    CostEstimate {
        flops,
        bytes_moved: bytes,
        kernel_overhead_ns: max_overhead,
    }
}

/// Sum the Layer-1 FLOPs of every primitive node reachable from `root`,
/// stopping at the fused op's input leaves. A nested `Op::Fused` in the
/// decompose recurses (lowered by its own recipe, bounded by
/// [`MAX_DECOMPOSE_DEPTH`]); the base map is `decompose`'s fixpoint so this
/// terminates. `max_overhead` accumulates the single-launch overhead as the
/// `max` of the component primitives' launch overheads.
fn fold_subgraph(
    graph: &Graph,
    node_id: NodeId,
    leaves: &HashSet<NodeId>,
    visited: &mut HashSet<NodeId>,
    caps: &BackendCapabilities,
    max_overhead: &mut u32,
    depth: u32,
) -> u64 {
    // An input leaf contributes no compute; a shared subexpression is
    // counted once.
    if leaves.contains(&node_id) || !visited.insert(node_id) {
        return 0;
    }
    let node = graph.node(node_id);
    match &node.op {
        // Leaves / pure-metadata producers: no compute.
        Op::Const | Op::Iota { .. } => 0,
        // Nested fused op: lower it by its own recipe (bounded recursion),
        // then also price the primitives feeding it in this graph.
        Op::Fused(nested_id, nested_params) => {
            let mut flops = 0u64;
            if depth < MAX_DECOMPOSE_DEPTH {
                let in_shapes: Vec<Shape> =
                    node.inputs.iter().map(|&i| graph.node(i).shape.clone()).collect();
                let in_dtypes: Vec<DType> =
                    node.inputs.iter().map(|&i| graph.node(i).dtype).collect();
                let sub = cost_from_decompose_inner(
                    *nested_id,
                    nested_params,
                    &in_shapes,
                    &in_dtypes,
                    caps,
                    depth + 1,
                );
                *max_overhead = (*max_overhead).max(sub.kernel_overhead_ns);
                flops = flops.saturating_add(sub.flops);
            }
            for &inp in &node.inputs {
                flops = flops.saturating_add(fold_subgraph(
                    graph, inp, leaves, visited, caps, max_overhead, depth,
                ));
            }
            flops
        }
        // A primitive node: price it via the same per-family cost machinery
        // the optimizer applies to any candidate path. Shapes follow the
        // binding-table convention `[input1, .., inputN, output]`; dtypes
        // likewise. `OpParams::None` is exact for the shape-driven
        // elementwise/scalar families (the algebraic-fusion case); a
        // param-carrying interior op prices at its shape-derivable floor
        // here and is refined by the Judge (spec §6).
        _ => {
            let mut flops = 0u64;
            if let Some(kind) = crate::pipelined::op_to_op_kind(&node.op) {
                let mut shapes: Vec<Shape> =
                    node.inputs.iter().map(|&i| graph.node(i).shape.clone()).collect();
                shapes.push(node.shape.clone());
                let mut dtypes: Vec<DType> =
                    node.inputs.iter().map(|&i| graph.node(i).dtype).collect();
                dtypes.push(node.dtype);
                let c = crate::cost::default_cost_for_op_kind(kind)(
                    &shapes,
                    &dtypes,
                    &OpParams::None,
                    caps,
                );
                flops = c.flops;
                *max_overhead = (*max_overhead).max(c.kernel_overhead_ns);
            }
            for &inp in &node.inputs {
                flops = flops.saturating_add(fold_subgraph(
                    graph, inp, leaves, visited, caps, max_overhead, depth,
                ));
            }
            flops
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fused::{KernelRevisionHash, PrecisionGuarantee};
    use crate::kernel::{KernelCaps, KernelRef};
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_graph::runtime_fused::register_runtime_fused;
    use fuel_ir::backend::{BackendCapabilities, SubstrateClass, TransferPath};
    use fuel_ir::probe::BackendId;
    use fuel_ir::{DeviceLocation, Layout};
    use std::collections::HashSet as StdHashSet;
    use std::sync::{Arc, RwLock};

    fn noop_kernel(
        _i: &[Arc<RwLock<fuel_memory::Storage>>],
        _o: &mut [Arc<RwLock<fuel_memory::Storage>>],
        _l: &[Layout],
        _p: &OpParams,
    ) -> fuel_ir::Result<()> {
        Ok(())
    }

    fn cpu_caps() -> BackendCapabilities {
        BackendCapabilities {
            backend_id: BackendId::Cpu,
            device_location: DeviceLocation::Cpu,
            op_dtype_support: StdHashSet::new(),
            required_alignment: 1,
            access_granularity_bits: 8,
            transfer_paths: vec![(DeviceLocation::Cpu, TransferPath::SameDevice)],
            storage_substrate: SubstrateClass::HostBytes,
            compute_throughput_flops_per_ns: 1.0,
            mem_bandwidth_bytes_per_ns: 4.0,
        }
    }

    /// `relu(add(x0, x1))` — a two-input, pure-elementwise runtime region.
    /// Its `decompose_region` re-emits `Add` then `Relu` over the two
    /// inputs, so its composed FLOPs are exactly `add(n) + relu(n) = 2n`.
    fn relu_add_region() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Relu,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Add,
                attrs: OpAttrs::default(),
                operands: vec![
                    PatternNode::Bind { index: 0 },
                    PatternNode::Bind { index: 1 },
                ],
            }],
        }
    }

    /// A non-sentinel declared cost fn — a FIXED nonzero cost, distinct
    /// from anything the decompose fold would produce.
    fn declared_cost(
        _s: &[Shape],
        _p: &FusedOpParams,
        _c: &BackendCapabilities,
    ) -> CostEstimate {
        CostEstimate { flops: 777, bytes_moved: 42, kernel_overhead_ns: 9 }
    }

    fn sentinel_impl() -> BackendImpl {
        BackendImpl {
            kernel: noop_kernel as KernelRef,
            dtypes: &[DType::F32, DType::F32, DType::F32],
            cost: crate::fkc::fused_unknown_cost,
            precision: PrecisionGuarantee::UNAUDITED,
            caps: KernelCaps::empty(),
            revision: KernelRevisionHash::UNTRACKED,
            cost_expr: None,
        }
    }

    // =====================================================================
    // FKC gap-closure Task 2.4: a declared cost AST reaches Layer-1 pricing
    // — not the sentinel/decompose fallback, even when `impl_.cost` IS the
    // sentinel (as `sentinel_impl()` sets it to be).
    // =====================================================================
    #[test]
    fn fused_declared_cost_reaches_layer1_not_sentinel() {
        // NOTE (deviation from brief): `BackendCapabilities` has no `Default`
        // impl in this tree (`#[derive(Debug, Clone)]` only, see
        // `fuel-ir/src/backend.rs`) — using the module's existing
        // `cpu_caps()` fixture in its place; the caps value doesn't matter
        // here since the declared-AST path never reads `caps`.
        use fuel_graph::registry::{FusedOps, FusedOpParams};
        let expr = crate::fkc::cost_compile::intern_cost_expr(&crate::fkc::cost_expr::compile_field(Some("n")).unwrap()).unwrap();
        let impl_ = BackendImpl { cost_expr: Some(expr), ..sentinel_impl() };
        let caps = cpu_caps();
        let est = fused_layer1_cost(&impl_, FusedOps::SOFTMAX_LAST_DIM, &[Shape::from_dims(&[8])], &[DType::F32], &FusedOpParams::SoftmaxLastDim, &caps);
        assert_eq!(est.flops, 8, "declared fused flops = n = 8; not the sentinel/decompose fallback");
    }

    // =====================================================================
    // BORN-RED: a fused op registered with the sentinel now gets a NONZERO
    // composed cost — where before it was CostEstimate::default() (zero).
    // =====================================================================
    #[test]
    fn sentinel_fused_op_gets_nonzero_composed_cost_from_its_decompose() {
        let n: u64 = 8;
        let shape = Shape::from_dims(&[n as usize]);
        let input_shapes = [shape.clone(), shape.clone()];
        let input_dtypes = [DType::F32, DType::F32];

        let id = register_runtime_fused("test::cost_from_decompose::relu_add", relu_add_region())
            .expect("relu(add) is a registrable region");
        let impl_ = sentinel_impl();
        let caps = cpu_caps();

        // (RED anchor) The sentinel cost fn itself prices at ZERO — the bug.
        let raw = (impl_.cost)(&input_shapes, &FusedOpParams::Runtime { scalars: vec![] }, &caps);
        assert_eq!(
            raw,
            CostEstimate::default(),
            "the fused_unknown_cost sentinel prices at zero — the mis-pricing this fixes",
        );

        // (GREEN target) The composed cost is NONZERO and equals the fold of
        // the decompose primitives: Add(n) + Relu(n) = 2n FLOPs.
        let composed = fused_layer1_cost(
            &impl_,
            id,
            &input_shapes,
            &input_dtypes,
            &FusedOpParams::Runtime { scalars: vec![] },
            &caps,
        );
        assert!(
            composed.flops > 0,
            "composed cost must be NONZERO (was {:?})",
            composed,
        );
        assert_eq!(
            composed.flops, 2 * n,
            "flops = Σ decompose primitives = add(n) + relu(n) = 2n",
        );
        // Boundary I/O bytes: 2 F32 inputs of n elems + 1 F32 output of n
        // elems = 3·n·4 (intermediates elided — the tight estimate).
        assert_eq!(
            composed.bytes_moved,
            3 * n * 4,
            "bytes = fused op's own boundary I/O (inputs + output), not Σ intermediates",
        );
        // One launch overhead (max of the elementwise components' 50 ns),
        // not Σ.
        assert_eq!(composed.kernel_overhead_ns, 50, "one launch overhead, not Σ");
    }

    // =====================================================================
    // GUARD: a fused op with a NON-sentinel (declared/measured) cost is
    // UNCHANGED — the composed default only fires for the sentinel.
    // =====================================================================
    #[test]
    fn declared_cost_fused_op_is_untouched_by_the_decompose_default() {
        let shape = Shape::from_dims(&[8]);
        let input_shapes = [shape.clone(), shape.clone()];
        let input_dtypes = [DType::F32, DType::F32];

        let id = register_runtime_fused("test::cost_from_decompose::declared", relu_add_region())
            .expect("region registers");
        let mut impl_ = sentinel_impl();
        impl_.cost = declared_cost; // a real declared cost, NOT the sentinel

        assert!(!is_fused_cost_sentinel(impl_.cost), "declared cost is not the sentinel");

        let got = fused_layer1_cost(
            &impl_,
            id,
            &input_shapes,
            &input_dtypes,
            &FusedOpParams::Runtime { scalars: vec![] },
            &cpu_caps(),
        );
        assert_eq!(
            got,
            CostEstimate { flops: 777, bytes_moved: 42, kernel_overhead_ns: 9 },
            "a declared cost fn is priced by that fn, unchanged — not composed",
        );
    }
}
