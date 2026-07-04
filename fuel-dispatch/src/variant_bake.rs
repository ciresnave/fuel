//! **Optimize-time kernel-variant baking** — the same-device arm selector.
//!
//! ## The gap this closes
//!
//! `optimize_graph` records `Op::Branch` decision points whose arms are
//! competing routes. Two *kinds* of arm coexist under one representation:
//!
//! - **Placement arms** — the same op on *different* `(backend, device)`
//!   pairs (CPU vs. CUDA). These stay live `Op::Branch` points, resolved at
//!   dispatch by the runtime route picker on live load / VRAM
//!   ([`crate::ranker::route_picker`], 06-runtime).
//! - **Kernel-variant arms** — *different implementations of the same op on
//!   the SAME device* (the decomposed attention region vs. a fused flash
//!   kernel, both on CUDA). Per 04-optimization these are **"largely baked
//!   at optimize time"**, NOT deferred to the runtime picker.
//!
//! The shipped runtime picker resolves only placement arms: it ranks arms by
//! their `(backend, device)` keyed on arm 0's op, so two same-device arms map
//! to the same candidate and always resolve to **arm 0** (the finding in
//! commit `c7bfc1b5`). A same-device fused variant — e.g. the CUDA flash
//! decode arm from [`crate::decode_flash::offer_decode_flash_arm`] — can be
//! *offered* but never *selected*.
//!
//! ## What this module does (the bake)
//!
//! [`bake_variant_branches`] runs at optimize time (Picker 1, where costs +
//! caps + the Judge oracle are in hand — NOT the executor). For every
//! reachable **same-device** variant branch it:
//!
//! 1. costs arm 0 (the decomposed-region oracle) and each variant arm;
//! 2. picks the **unique** variant that is **strictly cheaper** than arm 0;
//! 3. **collapses** the branch to that winner via
//!    [`fuel_graph::Graph::collapse_variant_branch`] (the merge is rewired to
//!    the winner; the branch becomes an inert single arm — "the Branch
//!    collapses" of 04-optimization).
//!
//! **Conservative defaults (the oracle stands):** a tie, an unknown/sentinel
//! cost, or a variant whose kernel is not available on the placed device all
//! resolve to **arm 0** — the decomposed correctness oracle. **Placement
//! branches (arms on ≥2 devices) are never touched** — they are left live for
//! the runtime picker. A graph with no same-device variant branch (every
//! CPU/Vulkan build today, since no CUDA flash arm is offered there) is a
//! **no-op**, so the byte-exact suites are unaffected by construction.
//!
//! ## Where it rides
//!
//! The bake mutates the graph **in place** during `optimize_graph`, so the
//! baked choice rides [`crate::optimize::OptimizedGraph`] (which derives its
//! dispatch order from the graph) with no extra state — plan-once persistent
//! decode bakes once per generation and every realize lowers the collapsed
//! graph.
//!
//! ## What the full version adds
//!
//! This is the smallest constitutional version. The full version adds:
//! **Judge-refined variant ranking** (a Layer-2 measurement overriding the
//! Layer-1 composed cost — plugs into the cost closure); **Pareto retention
//! of both arms** (keep the variant as a runtime-selectable path instead of
//! pruning it, once the picker can distinguish same-device variants); and, if
//! it ever becomes constitutional, **runtime load-adaptive variant
//! switching**. Today variant choice is baked and the loser is pruned.

use std::collections::HashSet;

use fuel_graph::registry::{FusedOpParams, FusedOps};
use fuel_graph::{branches_in_topo_order, Graph, NodeId, Op};
use fuel_ir::backend::{BackendCapabilities, SubstrateClass, TransferPath};
use fuel_ir::dispatch::SizeClass;
use fuel_ir::probe::BackendId;
use fuel_ir::{DType, DeviceLocation, DynScalar, Shape};

use crate::cost::{cost_flash_decoding_cuda, cost_matmul_cpu, default_cost_for_op_kind};
use crate::fused::CostEstimate;
use crate::kernel::OpParams;
use crate::pipelined::op_to_op_kind;
use crate::ranker::cost::{composite_ns, default_backend_rates};
use crate::ranker::judge::JudgeOracle;
use crate::runtime_fused_kernels::fused_kernel_available;

/// The composite-nanosecond cost of a branch arm, or `None` when it is
/// **unknown or inadmissible** — an unpriced/sentinel-cost op, or a variant
/// whose kernel is not available on the placed device. `None` means "this arm
/// cannot win", so the conservative arm-0 oracle stands.
///
/// Arguments: `(graph, branch, arm_index, arm_interior)` where `arm_interior`
/// is the arm's interior node set (the nodes on paths `diverge → exit`, arm 0's
/// being the decomposed region and a fused arm's being the single fused node).
pub type ArmCostFn<'a> =
    dyn Fn(&Graph, NodeId, usize, &[NodeId]) -> Option<u64> + 'a;

/// Bake same-device kernel-variant branches to their cost winner (see the
/// module docs). Returns the number of branches collapsed **to a variant arm**
/// (arm > 0) — a diagnostic the caller / tests assert on. Never panics: a
/// structural surprise from [`Graph::collapse_variant_branch`] is ignored
/// rather than crashing (never-panic on a production path).
pub fn bake_variant_branches(
    graph: &mut Graph,
    roots: &[NodeId],
    arm_cost: &ArmCostFn<'_>,
) -> usize {
    // Decide first (immutable analysis over the whole set), then mutate — a
    // collapse rewires edges, so we must not analyse against a half-mutated
    // graph. Topo order over `reconverge_at` is the picker's order; it does
    // not matter for independent same-device bakes but keeps parity.
    let branches = branches_in_topo_order(graph, roots);
    let mut decisions: Vec<(NodeId, usize)> = Vec::new();

    for branch in branches {
        let arms = graph.node(branch).inputs.clone();
        if arms.len() < 2 {
            continue; // already inert — no decision
        }
        // Same-device gate: every arm must resolve to the SAME backend. A
        // placement branch (arms on ≥2 backends) is left LIVE for the runtime
        // route picker; an unstamped arm (no `target_backend`) is not a
        // determinate variant branch, so skip it conservatively.
        let Some(backend0) = graph.target_backend(arms[0]) else {
            continue;
        };
        if !arms.iter().all(|&a| graph.target_backend(a) == Some(backend0)) {
            continue; // placement branch → runtime picker owns it
        }

        // Cost the oracle (arm 0). An unknown oracle cost ⇒ leave arm 0.
        let interiors = arm_interiors(graph, branch);
        let Some(c0) = arm_cost(graph, branch, 0, &interiors[0]) else {
            continue;
        };

        // The winner is the UNIQUE variant strictly cheaper than the oracle.
        // A tie at the cheapest cost, or no cheaper admissible variant, ⇒
        // arm 0 (the conservative oracle default).
        let mut best: Option<(usize, u64)> = None;
        let mut tie_at_best = false;
        for w in 1..arms.len() {
            let Some(cw) = arm_cost(graph, branch, w, &interiors[w]) else {
                continue; // unknown / capability-missing ⇒ cannot win
            };
            if cw >= c0 {
                continue; // not strictly cheaper than the oracle
            }
            match best {
                None => best = Some((w, cw)),
                Some((_, bc)) if cw < bc => {
                    best = Some((w, cw));
                    tie_at_best = false;
                }
                Some((_, bc)) if cw == bc => tie_at_best = true,
                _ => {}
            }
        }
        let winner = if tie_at_best {
            0
        } else {
            best.map(|(w, _)| w).unwrap_or(0)
        };
        decisions.push((branch, winner));
    }

    // Apply: collapse every same-device variant branch — to its winning
    // variant, else to arm 0 (dropping the offered-but-unpicked variant arms).
    // Either way no same-device variant branch survives optimize; only
    // placement branches remain for the runtime picker.
    let mut baked_to_variant = 0usize;
    for (branch, winner) in decisions {
        if winner != 0 {
            baked_to_variant += 1;
        }
        let _ = graph.collapse_variant_branch(branch, winner);
    }
    baked_to_variant
}

/// The per-arm **interior** node sets of a branch — the nodes on paths
/// `diverge → exit`, excluding the shared diverge prefix. Mirrors
/// `fuel_graph::run`'s cone/shared-prefix derivation (the op carries only
/// `reconverge_at`, so the diverge is recovered as the intersection of the
/// arm exits' backward cones). Index `i` is arm `i`'s interior; a fused arm's
/// interior is typically the single fused node, arm 0's is the decomposed
/// region.
fn arm_interiors(graph: &Graph, branch: NodeId) -> Vec<Vec<NodeId>> {
    let arm_exits = graph.node(branch).inputs.clone();
    let cones: Vec<HashSet<NodeId>> =
        arm_exits.iter().map(|&e| backward_cone(graph, e)).collect();
    if cones.is_empty() {
        return Vec::new();
    }
    let mut shared: HashSet<NodeId> = cones[0].clone();
    for c in &cones[1..] {
        shared = shared.intersection(c).copied().collect();
    }
    cones
        .iter()
        .map(|cone| cone.iter().copied().filter(|n| !shared.contains(n)).collect())
        .collect()
}

/// Backward-reachable cone of `from` (the node and all transitive inputs).
fn backward_cone(graph: &Graph, from: NodeId) -> HashSet<NodeId> {
    let mut seen: HashSet<NodeId> = HashSet::new();
    let mut stack = vec![from];
    while let Some(n) = stack.pop() {
        if !seen.insert(n) {
            continue;
        }
        for &inp in &graph.node(n).inputs {
            if !seen.contains(&inp) {
                stack.push(inp);
            }
        }
    }
    seen
}

/// The **production cost provider** (04-optimization "cost-from-decompose" +
/// the region composed-Layer-1 sum). Folds a Layer-1 [`CostEstimate`] over the
/// arm's interior and converts it to a composite-ns figure on the placed
/// `backend`'s throughput roofline. This is the defensible v1 the session
/// prompt blesses: **compare the fused arm's declared cost against the region's
/// composed Layer-1 sum over its nodes.**
///
/// - a `Op::Fused(FLASH_ATTN, …)` arm is priced by [`cost_flash_decoding_cuda`]
///   (its declared decode cost — work scales with the live prefix `k_len`,
///   not the physical capacity), gated on the CUDA flash kernel being bound
///   (capability: absent ⇒ `None` ⇒ the oracle stands);
/// - an `Op::MatMul` interior node (the dominant decode-region cost) is priced
///   with its geometry derived from operand shapes;
/// - any other op with an [`op_to_op_kind`] mapping is priced by its Layer-1
///   family at `OpParams::None` (the shape-derivable floor — the same
///   documented approximation `fused_cost` uses; a param-carrying interior op
///   under-prices here and is refined by the Judge, biasing conservatively
///   toward the oracle);
/// - the region's arms are summed as **sequential** launches
///   (`Σ composite_ns`), so the fused arm's single-launch win over an N-launch
///   region is visible.
///
/// Symbolic `k_len` (persistent decode): the flash arm is priced at the KV
/// **capacity** as the representative prefix. Both the flash cost and the
/// decomposed region scale ~linearly in the prefix, so the winner is stable
/// across the concrete prefix — the capacity is a sound representative extent
/// for the plan-once bake.
///
/// Returns `None` (⇒ arm 0) when the arm prices to zero (nothing costable —
/// an unknown/sentinel region) so a zero never spuriously "wins".
pub fn decode_arm_composite_ns(
    graph: &Graph,
    _branch: NodeId,
    _arm_idx: usize,
    interior: &[NodeId],
    backend: BackendId,
) -> Option<u64> {
    let (compute_rate, mem_bw) = default_backend_rates(backend);
    let caps = neutral_caps(backend, compute_rate, mem_bw);
    let mut total_ns: u64 = 0;
    let mut any = false;
    for &nid in interior {
        let Some(cost) = node_layer1_cost(graph, nid, backend, &caps) else {
            continue;
        };
        any = true;
        total_ns = total_ns.saturating_add(composite_ns(&cost, compute_rate, mem_bw));
    }
    if !any || total_ns == 0 {
        return None; // nothing costable ⇒ oracle stands (no spurious zero win)
    }
    Some(total_ns)
}

/// The **Layer-2-aware** cost provider — the slice-4 payoff. Consults the Judge
/// oracle (empirical measured latency, 06-optimization Layer-2) **first** per
/// interior node, falling back to the [`decode_arm_composite_ns`] Layer-1
/// composite for any node the Judge hasn't measured. Summed per arm exactly as
/// the Layer-1 provider does, so the comparison stays apples-to-apples:
///
/// - **arm 0** (the decomposed region) = `Σ` over its primitives of
///   `measured(op, dtype, size_class, backend) ?? layer1(op)`;
/// - **arm 1** (the fused flash node) = `measured(FlashAttn, …) ?? layer1(flash)`.
///
/// A measured latency is used **directly** (it is already a wall-clock ns, the
/// same convention [`crate::ranker::cost::compute_static_costs`] uses when it
/// folds the Judge cell into `kernel_overhead_ns`), so flash's real
/// algorithm-changing win (no materialized attention matrix, one launch —
/// smaller measured latency than the region's `N` measured launches) becomes
/// visible where Layer-1 alone saw a FLOP-for-FLOP tie and the conservative
/// bake kept arm 0.
///
/// **A Judge measurement is evidence of capability.** A cell exists only
/// because the kernel was actually profiled (it ran), so a measured flash cell
/// is used even when the Layer-1 capability gate ([`fused_kernel_available`])
/// would decline it — the gate only guards the *fallback* Layer-1 path (no
/// measurement + unbound kernel ⇒ `None` ⇒ the oracle stands, unchanged).
///
/// **Hybrid sum.** When some of an arm's nodes are Judge-covered and others are
/// not, measured cells and Layer-1 estimates are summed together — directionally
/// better than an all-Layer-1 fold, and the conservative tie-break in
/// [`bake_variant_branches`] (strictly-cheaper-or-arm-0) still protects against
/// a bad partial.
///
/// **`kernel_source`.** The bake has no per-node candidate in hand, so it
/// queries the legacy / default (`""`) cell — where a single-impl flash/matmul
/// kernel registers. Threading a per-arm `kernel_source` is a follow-up; `""`
/// keeps the born-red and single-impl production apples-to-apples.
///
/// With `judge == None` this is **byte-identical** to [`decode_arm_composite_ns`]
/// (it delegates), so the no-Judge production path and the persistent
/// byte-exact suites are unaffected by construction.
pub fn decode_arm_composite_ns_judged(
    graph: &Graph,
    branch: NodeId,
    arm_idx: usize,
    interior: &[NodeId],
    backend: BackendId,
    judge: Option<&dyn JudgeOracle>,
) -> Option<u64> {
    // No oracle ⇒ the Layer-1-only provider, byte-identical to today. The
    // no-Judge production path + the persistent byte-exact suites are
    // unaffected by construction.
    let Some(judge) = judge else {
        return decode_arm_composite_ns(graph, branch, arm_idx, interior, backend);
    };

    // Layer-2 FIRST, per interior node, with a Layer-1 fallback (the hybrid
    // sum): a measured wall-clock latency is used directly (same convention as
    // `compute_static_costs`'s `kernel_overhead_ns` fold); an unmeasured node
    // falls back to its Layer-1 composite. Summed as sequential launches so the
    // fused arm's single measured launch is compared against the region's `N`
    // measured/Layer-1 launches apples-to-apples.
    let (compute_rate, mem_bw) = default_backend_rates(backend);
    let caps = neutral_caps(backend, compute_rate, mem_bw);
    let mut total_ns: u64 = 0;
    let mut any = false;
    for &nid in interior {
        // A Judge measurement is evidence of capability (a cell exists only
        // because the kernel ran), so a measured node is admissible even if the
        // Layer-1 capability gate would decline it. The gate guards only the
        // fallback below.
        let ns = match node_measured_ns(graph, nid, backend, judge) {
            Some(measured) => Some(measured),
            None => node_layer1_cost(graph, nid, backend, &caps)
                .map(|cost| composite_ns(&cost, compute_rate, mem_bw)),
        };
        if let Some(ns) = ns {
            any = true;
            total_ns = total_ns.saturating_add(ns);
        }
    }
    if !any || total_ns == 0 {
        return None; // nothing costable ⇒ oracle stands (no spurious zero win)
    }
    Some(total_ns)
}

/// The measured (Layer-2) latency for one interior node's Judge cell, keyed the
/// SAME way the ranker + Judge producer key it: `(op, principal_dtype,
/// SizeClass::for_op(op, input_shapes), backend, "")`. `None` when the node has
/// no [`op_to_op_kind`] mapping (a view/leaf) or the Judge hasn't measured that
/// cell. The principal dtype is the first operand's dtype (the
/// [`crate::ranker::cost::compute_static_costs`] convention), so a matmul /
/// flash cell the Judge profiled is found by the SAME `for_op` key here.
fn node_measured_ns(
    graph: &Graph,
    nid: NodeId,
    backend: BackendId,
    judge: &dyn JudgeOracle,
) -> Option<u64> {
    let node = graph.node(nid);
    let kind = op_to_op_kind(&node.op)?;
    let principal_dtype = match node.inputs.first() {
        Some(&i) => graph.node(i).dtype,
        None => node.dtype, // nullary — has no OpKind mapping anyway
    };
    let input_shapes: Vec<Shape> =
        node.inputs.iter().map(|&i| graph.node(i).shape.clone()).collect();
    let size_class = SizeClass::for_op(kind, &input_shapes);
    judge.measured_latency_ns(kind, principal_dtype, size_class, backend, "")
}

/// Layer-1 [`CostEstimate`] for one interior node (see [`decode_arm_composite_ns`]).
/// `None` for a node that carries no priceable compute (a view/leaf, or a
/// flash arm whose kernel is not bound — the capability gate).
fn node_layer1_cost(
    graph: &Graph,
    nid: NodeId,
    backend: BackendId,
    caps: &BackendCapabilities,
) -> Option<CostEstimate> {
    let node = graph.node(nid);
    match &node.op {
        // The fused flash decode arm: priced by its declared decode cost,
        // gated on the CUDA flash kernel being bound (capability).
        Op::Fused(fid, params) if *fid == FusedOps::FLASH_ATTN => {
            if !fused_kernel_available(FusedOps::FLASH_ATTN, backend) {
                return None; // capability-missing ⇒ arm cannot win
            }
            let op_params = flash_op_params(graph, node, params)?;
            let dtypes = [node.dtype, node.dtype];
            Some(cost_flash_decoding_cuda(&[], &dtypes, &op_params, caps))
        }
        // MatMul — the dominant decode-region cost; geometry from shapes.
        Op::MatMul => {
            let op_params = matmul_op_params(graph, node)?;
            let dtypes = [node.dtype, node.dtype, node.dtype];
            Some(cost_matmul_cpu(&[], &dtypes, &op_params, caps))
        }
        // Any other primitive with an OpKind mapping: its Layer-1 family at
        // the shape-derivable floor (OpParams::None), per the fused_cost
        // precedent. Shapes follow `[input1, .., inputN, output]`.
        other => {
            let kind = op_to_op_kind(other)?;
            let mut shapes: Vec<Shape> =
                node.inputs.iter().map(|&i| graph.node(i).shape.clone()).collect();
            shapes.push(node.shape.clone());
            let mut dtypes: Vec<DType> =
                node.inputs.iter().map(|&i| graph.node(i).dtype).collect();
            dtypes.push(node.dtype);
            Some(default_cost_for_op_kind(kind)(&shapes, &dtypes, &OpParams::None, caps))
        }
    }
}

/// Derive `OpParams::Matmul` geometry from a `MatMul` node's operand shapes:
/// `A[.., m, k] · B[.., k, n] → [.., m, n]`. `None` if either operand is
/// rank < 2 (nothing to cost).
fn matmul_op_params(graph: &Graph, node: &fuel_graph::Node) -> Option<OpParams> {
    let a = graph.node(*node.inputs.first()?).shape.clone();
    let b = graph.node(*node.inputs.get(1)?).shape.clone();
    let ad = a.dims();
    let bd = b.dims();
    if ad.len() < 2 || bd.len() < 2 {
        return None;
    }
    let m = ad[ad.len() - 2];
    let k = ad[ad.len() - 1];
    let n = bd[bd.len() - 1];
    let lhs_batch_dims: Vec<usize> = ad[..ad.len() - 2].to_vec();
    let rhs_batch_dims: Vec<usize> = bd[..bd.len() - 2].to_vec();
    Some(OpParams::Matmul { lhs_batch_dims, rhs_batch_dims, m, n, k })
}

/// Derive `OpParams::FlashAttn` for the decode cost from the flash node's
/// `[q, k, v, ..]` inputs + its `FusedOpParams::FlashAttn`. `q = [b, hq, sq, d]`,
/// `k = [b, hkv, sk, d]`. A symbolic `k_len` prices at the KV capacity `sk`
/// (the representative prefix — see [`decode_arm_composite_ns`]).
fn flash_op_params(
    graph: &Graph,
    node: &fuel_graph::Node,
    params: &FusedOpParams,
) -> Option<OpParams> {
    let FusedOpParams::FlashAttn {
        softmax_scale,
        causal,
        window_size_left,
        window_size_right,
        softcap,
        k_len,
    } = params
    else {
        return None;
    };
    let q = graph.node(*node.inputs.first()?).shape.clone();
    let kv = graph.node(*node.inputs.get(1)?).shape.clone();
    let qd = q.dims();
    let kvd = kv.dims();
    if qd.len() != 4 || kvd.len() != 4 {
        return None;
    }
    let (b, hq, sq, d) = (qd[0], qd[1], qd[2], qd[3]);
    let hkv = kvd[1];
    let sk = kvd[2];
    // Symbolic prefix ⇒ price at the capacity (representative extent).
    let k_len_val = match k_len {
        Some(DynScalar::Concrete(v)) => *v,
        _ => sk,
    };
    Some(OpParams::FlashAttn {
        b,
        hq,
        hkv,
        sq,
        sk,
        d,
        k_len: k_len_val,
        softmax_scale: *softmax_scale,
        causal: *causal,
        window_size_left: *window_size_left,
        window_size_right: *window_size_right,
        softcap: *softcap,
    })
}

/// A minimal [`BackendCapabilities`] carrying only the backend id + its
/// throughput rates — the Layer-1 cost families ignore the rest of the struct;
/// the roofline rates flow through [`composite_ns`] separately.
fn neutral_caps(backend: BackendId, compute: f64, mem_bw: f64) -> BackendCapabilities {
    BackendCapabilities {
        backend_id: backend,
        device_location: DeviceLocation::Cpu,
        op_dtype_support: HashSet::new(),
        required_alignment: 1,
        access_granularity_bits: 8,
        transfer_paths: vec![(DeviceLocation::Cpu, TransferPath::SameDevice)],
        storage_substrate: SubstrateClass::HostBytes,
        compute_throughput_flops_per_ns: compute,
        mem_bandwidth_bytes_per_ns: mem_bw,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ranker::judge::HashMapJudge;
    use fuel_graph::{lower_runs_arm0, Node};
    use fuel_ir::dispatch::OpKind;
    use fuel_ir::Shape;

    fn node(g: &mut Graph, op: Op, inputs: Vec<NodeId>, dims: &[usize], dt: DType) -> NodeId {
        g.push(Node { op, inputs, shape: Shape::from_dims(dims), dtype: dt })
    }

    /// Build a same-device 2-arm variant branch mirroring the decode flash
    /// shape: arm 0 is a 2-node "decomposed region" (Relu→Silu over the
    /// diverge), arm 1 is a single "fused variant" node (Gelu). Both arms are
    /// stamped `backend`. Returns `(graph, branch, arm0_exit, arm1, reconverge, post)`.
    fn variant_diamond(
        backend: BackendId,
    ) -> (Graph, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let dt = DType::F32;
        let mut g = Graph::new();
        let pre = node(&mut g, Op::Const, vec![], &[4], dt);
        let diverge = node(&mut g, Op::Relu, vec![pre], &[4], dt);
        // arm 0 = a 2-node decomposed region.
        let a0_mid = node(&mut g, Op::Relu, vec![diverge], &[4], dt);
        let arm0 = node(&mut g, Op::Silu, vec![a0_mid], &[4], dt);
        // arm 1 = a single fused-variant node.
        let arm1 = node(&mut g, Op::Gelu, vec![diverge], &[4], dt);
        g.set_target_backend(a0_mid, backend);
        g.set_target_backend(arm0, backend);
        g.set_target_backend(arm1, backend);
        let reconverge = node(&mut g, Op::Relu, vec![arm0], &[4], dt);
        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let branch = b
            .finalize_branches(&mut g, reconverge)
            .expect("well-formed 2-arm branch")
            .expect("2 arms survive");
        let post = node(&mut g, Op::Tanh, vec![reconverge], &[4], dt);
        (g, branch, arm0, arm1, reconverge, post)
    }

    /// A cost closure that returns fixed per-arm costs by arm index.
    fn fixed_costs(costs: Vec<Option<u64>>) -> impl Fn(&Graph, NodeId, usize, &[NodeId]) -> Option<u64> {
        move |_g, _b, arm, _interior| costs.get(arm).copied().flatten()
    }

    // ===================================================================
    // BORN-RED: a same-device variant arm is NEVER selected today (the
    // runtime picker resolves same-device arms to arm 0). The bake makes
    // the cheaper variant SELECTABLE by collapsing the branch to it.
    // ===================================================================
    #[test]
    fn cheaper_variant_is_baked_and_becomes_the_route() {
        let (mut g, branch, arm0, arm1, reconverge, post) =
            variant_diamond(BackendId::Cuda);

        // (RED anchor) Before the bake, the default lowering realizes arm 0 —
        // the same-device variant (arm 1) is offered but never selected.
        let before = lower_runs_arm0(&g, &[post]);
        assert!(
            before.contains(&arm0) && !before.contains(&arm1),
            "before the bake the same-device variant is never selected (arm 0 wins)",
        );

        // (GREEN) The variant (arm 1) is strictly cheaper ⇒ the bake collapses
        // the branch to it.
        let cost = fixed_costs(vec![Some(1000), Some(400)]); // arm0=1000, arm1=400
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 1, "one variant branch baked to its winner");

        // The merge now reads the winner; the branch is collapsed.
        assert!(g.node(reconverge).inputs.contains(&arm1), "merge rewired to the variant");
        assert_eq!(g.node(branch).inputs, vec![arm1], "branch collapsed to the winner");

        // The default lowering now follows the baked variant (arm 1) and prunes
        // arm 0's region — selectable with NO runtime pick.
        let after = lower_runs_arm0(&g, &[post]);
        assert!(after.contains(&arm1), "after the bake the variant is realized; order={after:?}");
        assert!(!after.contains(&arm0), "arm 0's region is pruned; order={after:?}");
    }

    /// GUARD: a costlier variant ⇒ arm 0 (the oracle). The branch collapses to
    /// arm 0 (variant pruned), realize is byte-identical to today.
    #[test]
    fn costlier_variant_keeps_arm0() {
        let (mut g, branch, arm0, arm1, _recon, post) = variant_diamond(BackendId::Cuda);
        let cost = fixed_costs(vec![Some(400), Some(1000)]); // arm1 costlier
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 0, "no variant wins ⇒ nothing baked to a variant");
        assert_eq!(g.node(branch).inputs, vec![arm0], "branch collapsed to arm 0");
        let after = lower_runs_arm0(&g, &[post]);
        assert!(after.contains(&arm0) && !after.contains(&arm1), "arm 0 realized (oracle)");
    }

    /// GUARD: an unknown / sentinel variant cost ⇒ arm 0. `None` cannot win.
    #[test]
    fn unknown_variant_cost_keeps_arm0() {
        let (mut g, _branch, arm0, arm1, _recon, post) = variant_diamond(BackendId::Cuda);
        let cost = fixed_costs(vec![Some(1000), None]); // arm1 unknown
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 0, "unknown variant cost ⇒ arm 0 (conservative)");
        let after = lower_runs_arm0(&g, &[post]);
        assert!(after.contains(&arm0) && !after.contains(&arm1), "arm 0 realized");
    }

    /// GUARD: a tie between the oracle and the variant ⇒ arm 0.
    #[test]
    fn tie_keeps_arm0() {
        let (mut g, _branch, arm0, arm1, _recon, post) = variant_diamond(BackendId::Cuda);
        let cost = fixed_costs(vec![Some(500), Some(500)]); // equal ⇒ not strictly cheaper
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 0, "tie ⇒ arm 0 (variant not strictly cheaper)");
        let after = lower_runs_arm0(&g, &[post]);
        assert!(after.contains(&arm0) && !after.contains(&arm1), "arm 0 realized");
    }

    /// GUARD: capability-missing (modelled as a `None` variant cost, the way
    /// [`decode_arm_composite_ns`] surfaces an unbound flash kernel) ⇒ arm 0.
    #[test]
    fn capability_missing_keeps_arm0() {
        let (mut g, _branch, arm0, arm1, _recon, post) = variant_diamond(BackendId::Cuda);
        // A provider that declines the variant arm (capability gate) but prices
        // the oracle — exactly the None the flash capability gate returns.
        let cost = |_g: &Graph, _b: NodeId, arm: usize, _i: &[NodeId]| -> Option<u64> {
            if arm == 0 { Some(1000) } else { None }
        };
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 0, "capability-missing variant ⇒ arm 0");
        let after = lower_runs_arm0(&g, &[post]);
        assert!(after.contains(&arm0) && !after.contains(&arm1), "arm 0 realized");
    }

    /// GUARD: a **placement** branch (arms on DIFFERENT backends) is UNTOUCHED
    /// — it stays a live 2-arm decision point for the runtime picker, even
    /// when the variant-shaped cost would prefer arm 1. The bake only ever
    /// collapses same-device branches.
    #[test]
    fn placement_branch_is_untouched() {
        let (mut g, branch, arm0, arm1, _recon, post) = variant_diamond(BackendId::Cuda);
        // Re-stamp arm 1 onto a DIFFERENT backend ⇒ placement branch.
        g.set_target_backend(arm1, BackendId::Cpu);
        let cost = fixed_costs(vec![Some(1000), Some(1)]); // arm1 "cheaper", but placement
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 0, "a placement branch is never baked");
        assert_eq!(
            g.node(branch).inputs,
            vec![arm0, arm1],
            "placement branch stays a live 2-arm decision point (runtime picker owns it)",
        );
    }

    /// GUARD: an unstamped branch (arms with no `target_backend`) is skipped —
    /// not a determinate same-device variant branch.
    #[test]
    fn unstamped_branch_is_skipped() {
        let dt = DType::F32;
        let mut g = Graph::new();
        let pre = node(&mut g, Op::Const, vec![], &[4], dt);
        let diverge = node(&mut g, Op::Relu, vec![pre], &[4], dt);
        let a0_mid = node(&mut g, Op::Relu, vec![diverge], &[4], dt);
        let arm0 = node(&mut g, Op::Silu, vec![a0_mid], &[4], dt);
        let arm1 = node(&mut g, Op::Gelu, vec![diverge], &[4], dt);
        // NO target_backend stamps on the arms.
        let reconverge = node(&mut g, Op::Relu, vec![arm0], &[4], dt);
        let mut b = g.open_branch(diverge);
        b.add_arm(arm0);
        b.add_arm(arm1);
        let branch = b.finalize_branches(&mut g, reconverge).unwrap().unwrap();
        let post = node(&mut g, Op::Tanh, vec![reconverge], &[4], dt);
        let cost = fixed_costs(vec![Some(1000), Some(1)]);
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 0, "unstamped branch is skipped");
        assert_eq!(g.node(branch).inputs, vec![arm0, arm1], "branch untouched");
    }

    // ===================================================================
    // The PRODUCTION cost provider: a decomposed attention region vs. the
    // fused flash arm — flash WINS on the region-vs-fused Layer-1 fold.
    // ===================================================================

    /// The **region-vs-fused costing approach**, priced honestly. The
    /// production provider folds a real Layer-1 cost over BOTH arm kinds — the
    /// decomposed region (matmul-dominated fold) and the fused flash arm
    /// (declared decode cost from derived `OpParams`). This proves the costing
    /// path end-to-end; it deliberately does NOT force flash to "win" on
    /// Layer-1: `cost_flash_decoding_cuda` gives flash the SAME FLOPs as the
    /// region's two matmuls (they compute the same attention), so on the
    /// coarse Layer-1 model the two are ~cost-equal and the conservative bake
    /// keeps the oracle (arm 0). Flash's real win is an *algorithm-changing*
    /// fusion (no materialized attention matrix, one launch) that the Judge's
    /// Layer-2 measurement corrects downward — "a missed win until measured,
    /// not an optimistic irreversible commit" (04-optimization).
    #[test]
    fn region_and_flash_arms_are_both_priced_by_the_provider() {
        let (b, hq, d, sk) = (1usize, 8, 128, 512);
        let dt = DType::F16;
        let mut g = Graph::new();
        let q = node(&mut g, Op::Const, vec![], &[b, hq, 1, d], dt);
        let k = node(&mut g, Op::Const, vec![], &[b, hq, sk, d], dt);
        let v = node(&mut g, Op::Const, vec![], &[b, hq, sk, d], dt);

        // Decomposed region interior (two matmuls dominate).
        let kt = node(&mut g, Op::Permute(vec![0, 1, 3, 2]), vec![k], &[b, hq, d, sk], dt);
        let scores = node(&mut g, Op::MatMul, vec![q, kt], &[b, hq, 1, sk], dt);
        let attn = node(&mut g, Op::MatMul, vec![scores, v], &[b, hq, 1, d], dt);
        let region: Vec<NodeId> = vec![kt, scores, attn];

        // (1) The region folds to a real, nonzero composite cost — no binding
        // needed (pure shape/FLOP fold).
        let region_ns = decode_arm_composite_ns(&g, NodeId(0), 0, &region, BackendId::Cuda)
            .expect("the decomposed region has costable compute (its two matmuls)");
        assert!(region_ns > 0, "region prices to a real nonzero cost");

        // (2) The fused flash arm's DECLARED decode cost, priced directly from
        // derived OpParams (the provider's capability gate is bypassed here so
        // the costing path is exercised on any build, incl. no-CUDA).
        let flash_node = Node {
            op: Op::Fused(
                FusedOps::FLASH_ATTN,
                FusedOpParams::FlashAttn {
                    softmax_scale: 0.1,
                    causal: true,
                    window_size_left: None,
                    window_size_right: None,
                    softcap: None,
                    k_len: Some(DynScalar::Concrete(sk)),
                },
            ),
            inputs: vec![q, k, v],
            shape: Shape::from_dims(&[b, hq, 1, d]),
            dtype: dt,
        };
        let (cr, bw) = default_backend_rates(BackendId::Cuda);
        let caps = neutral_caps(BackendId::Cuda, cr, bw);
        let fp_full = decode_flash_cost_at(&g, &flash_node, sk, &caps, cr, bw);
        let fp_small = decode_flash_cost_at(&g, &flash_node, 32, &caps, cr, bw);
        assert!(fp_full > 0 && fp_small > 0, "flash decode cost is real + nonzero");
        assert!(
            fp_full > fp_small,
            "the flash decode cost scales with the live prefix k_len \
             ({fp_full} @512 > {fp_small} @32) — decode work is O(prefix)",
        );

        // (3) The provider's capability gate: without a bound CUDA flash kernel
        // the flash ARM is inadmissible (None) ⇒ the oracle stands. (In a live
        // CUDA build the kernel is bound and the arm prices instead.)
        let flash = node(
            &mut g,
            Op::Fused(
                FusedOps::FLASH_ATTN,
                FusedOpParams::FlashAttn {
                    softmax_scale: 0.1,
                    causal: true,
                    window_size_left: None,
                    window_size_right: None,
                    softcap: None,
                    k_len: Some(DynScalar::Concrete(sk)),
                },
            ),
            vec![q, k, v],
            &[b, hq, 1, d],
            dt,
        );
        let gated = decode_arm_composite_ns(&g, NodeId(0), 1, &[flash], BackendId::Cuda);
        if fused_kernel_available(FusedOps::FLASH_ATTN, BackendId::Cuda) {
            assert!(gated.is_some(), "bound flash kernel ⇒ arm priced");
        } else {
            assert!(gated.is_none(), "unbound flash kernel ⇒ arm inadmissible (oracle stands)");
        }
    }

    /// Helper: the flash decode composite-ns at a given prefix (bypasses the
    /// capability gate to exercise the declared-cost path on any build).
    fn decode_flash_cost_at(
        g: &Graph,
        flash_node: &Node,
        k_len: usize,
        caps: &BackendCapabilities,
        cr: f64,
        bw: f64,
    ) -> u64 {
        let params = FusedOpParams::FlashAttn {
            softmax_scale: 0.1,
            causal: true,
            window_size_left: None,
            window_size_right: None,
            softcap: None,
            k_len: Some(DynScalar::Concrete(k_len)),
        };
        let op_params = flash_op_params(g, flash_node, &params).expect("derivable");
        let dtypes = [flash_node.dtype, flash_node.dtype];
        let cost = cost_flash_decoding_cuda(&[], &dtypes, &op_params, caps);
        composite_ns(&cost, cr, bw)
    }

    /// The flash `OpParams` derivation prices a symbolic `k_len` at the KV
    /// capacity (the representative prefix), so a plan-once persistent-decode
    /// bake still produces a concrete, stable cost.
    #[test]
    fn symbolic_k_len_prices_at_capacity() {
        let (b, hq, d, sk) = (1usize, 4, 64, 256);
        let dt = DType::F16;
        let mut g = Graph::new();
        let q = node(&mut g, Op::Const, vec![], &[b, hq, 1, d], dt);
        let k = node(&mut g, Op::Const, vec![], &[b, hq, sk, d], dt);
        let v = node(&mut g, Op::Const, vec![], &[b, hq, sk, d], dt);
        let flash = node(
            &mut g,
            Op::Fused(
                FusedOps::FLASH_ATTN,
                FusedOpParams::FlashAttn {
                    softmax_scale: 0.1,
                    causal: true,
                    window_size_left: None,
                    window_size_right: None,
                    softcap: None,
                    // Symbolic prefix (persistent decode).
                    k_len: Some(DynScalar::Sym(fuel_ir::symbol::SymId(0))),
                },
            ),
            vec![q, k, v],
            &[b, hq, 1, d],
            dt,
        );
        let flash_node = g.node(flash).clone();
        let params_in = match &flash_node.op {
            Op::Fused(_, p) => p.clone(),
            _ => unreachable!(),
        };
        let params = flash_op_params(&g, &flash_node, &params_in).expect("derivable");
        match params {
            OpParams::FlashAttn { k_len, sk: sk_out, sq, .. } => {
                assert_eq!(k_len, sk, "symbolic k_len prices at the capacity");
                assert_eq!(sk_out, sk);
                assert_eq!(sq, 1, "decode sq == 1");
            }
            _ => panic!("expected FlashAttn params"),
        }
    }

    // ===================================================================
    // SLICE 4 (THE PAYOFF): the variant bake consults the Judge's MEASURED
    // latency (Layer-2) per arm, so the CUDA flash arm WINS the
    // decode-flash-vs-decomposed bake where it merely TIES on Layer-1. With
    // an injected oracle this proves the SELECTION logic on a CPU build (no
    // live GPU): the born-red flips arm 0 → arm 1 the moment the provider
    // reads the Judge.
    // ===================================================================

    /// A same-device (CUDA) 2-arm DECODE variant branch mirroring the real
    /// decode-flash bake: arm 0 is the decomposed attention region
    /// (`Permute` → `MatMul` QKᵀ → `MatMul` PV), arm 1 is the fused
    /// `FlashAttn` node. `q` is a `Relu` over a const so a single Op::Branch
    /// diverge point dominates both arm exits; both arm exits are stamped
    /// CUDA (same-device). Returns
    /// `(graph, branch, region_exit, flash_exit, reconverge, post)`.
    fn flash_variant_diamond() -> (Graph, NodeId, NodeId, NodeId, NodeId, NodeId) {
        let (b, hq, d, sk) = (1usize, 8, 128, 512);
        let dt = DType::F16;
        let mut g = Graph::new();
        let qc = node(&mut g, Op::Const, vec![], &[b, hq, 1, d], dt);
        let k = node(&mut g, Op::Const, vec![], &[b, hq, sk, d], dt);
        let v = node(&mut g, Op::Const, vec![], &[b, hq, sk, d], dt);
        // The single DIVERGE point both arms read as their `q` operand.
        let q = node(&mut g, Op::Relu, vec![qc], &[b, hq, 1, d], dt);
        // arm 0 — the decomposed attention region (two matmuls dominate).
        let kt = node(&mut g, Op::Permute(vec![0, 1, 3, 2]), vec![k], &[b, hq, d, sk], dt);
        let scores = node(&mut g, Op::MatMul, vec![q, kt], &[b, hq, 1, sk], dt);
        let attn = node(&mut g, Op::MatMul, vec![scores, v], &[b, hq, 1, d], dt);
        // arm 1 — the fused flash node (same exit shape & dtype as the region).
        let flash = node(
            &mut g,
            Op::Fused(
                FusedOps::FLASH_ATTN,
                FusedOpParams::FlashAttn {
                    softmax_scale: 0.1,
                    causal: true,
                    window_size_left: None,
                    window_size_right: None,
                    softcap: None,
                    k_len: Some(DynScalar::Concrete(sk)),
                },
            ),
            vec![q, k, v],
            &[b, hq, 1, d],
            dt,
        );
        // Same-device stamps — the bake's same-device gate reads the arm exits.
        g.set_target_backend(kt, BackendId::Cuda);
        g.set_target_backend(scores, BackendId::Cuda);
        g.set_target_backend(attn, BackendId::Cuda);
        g.set_target_backend(flash, BackendId::Cuda);
        let reconverge = node(&mut g, Op::Relu, vec![attn], &[b, hq, 1, d], dt);
        let mut br = g.open_branch(q);
        br.add_arm(attn);
        br.add_arm(flash);
        let branch = br
            .finalize_branches(&mut g, reconverge)
            .expect("well-formed 2-arm decode branch")
            .expect("2 arms survive");
        let post = node(&mut g, Op::Tanh, vec![reconverge], &[b, hq, 1, d], dt);
        (g, branch, attn, flash, reconverge, post)
    }

    /// The decode-shape Judge keys, single-sourced through the SAME helpers
    /// the bake's `for_op` derivation and the fuel-core Judge producer use:
    /// scores QKᵀ = `matmul(m=1, n=sk, k=d)`, attn PV = `matmul(m=1, n=d,
    /// k=sk)`, flash = `attention(hq, k_len=sk, d)`.
    fn decode_keys() -> (SizeClass, SizeClass, SizeClass) {
        let (hq, d, sk) = (8usize, 128, 512);
        (
            SizeClass::matmul(1, sk, d), // scores
            SizeClass::matmul(1, d, sk), // attn
            SizeClass::attention(hq, sk, d),
        )
    }

    /// **BORN-RED (the arc headline).** A measured flash cell that is strictly
    /// cheaper than the region's summed measured primitives flips the bake from
    /// arm 0 to arm 1. RED before the Layer-2 wiring (the flash arm ties / is
    /// gated on Layer-1 ⇒ arm 0 kept); GREEN after (measured flash wins).
    #[test]
    fn judge_measured_flash_wins_the_decode_bake() {
        let (mut g, branch, attn, flash, reconverge, post) = flash_variant_diamond();
        let dt = DType::F16;
        let (k_scores, k_attn, k_flash) = decode_keys();

        // (RED anchor) the default lowering realizes the decomposed region.
        let before = lower_runs_arm0(&g, &[post]);
        assert!(
            before.contains(&attn) && !before.contains(&flash),
            "before the bake the decomposed region wins (arm 0); order={before:?}",
        );

        // Injected Judge: flash measures CHEAPER (400 ns) than the region's two
        // measured matmuls (500 + 500 = 1000 ns) — flash's algorithm-changing
        // win Layer-1 can't see (FLOP tie / CPU-build capability gate).
        let mut judge = HashMapJudge::new();
        judge.insert(OpKind::MatMul, dt, k_scores, BackendId::Cuda, "", 500);
        judge.insert(OpKind::MatMul, dt, k_attn, BackendId::Cuda, "", 500);
        judge.insert(OpKind::FlashAttn, dt, k_flash, BackendId::Cuda, "", 400);

        let cost = |g: &Graph, b: NodeId, arm: usize, i: &[NodeId]| -> Option<u64> {
            decode_arm_composite_ns_judged(g, b, arm, i, BackendId::Cuda, Some(&judge))
        };
        let baked = bake_variant_branches(&mut g, &[post], &cost);

        assert_eq!(baked, 1, "measured flash is strictly cheaper ⇒ the branch bakes to it");
        assert_eq!(g.node(branch).inputs, vec![flash], "branch collapsed to the flash arm");
        assert!(g.node(reconverge).inputs.contains(&flash), "merge rewired to flash");

        let after = lower_runs_arm0(&g, &[post]);
        assert!(after.contains(&flash), "flash realized after the bake; order={after:?}");
        assert!(!after.contains(&attn), "the decomposed region is pruned; order={after:?}");
    }

    /// GUARD: the Judge says flash is SLOWER than the region ⇒ arm 0 (the
    /// conservative oracle default holds — a measurement never optimistically
    /// commits a losing variant).
    #[test]
    fn judge_measured_flash_slower_keeps_the_region() {
        let (mut g, branch, attn, flash, _recon, post) = flash_variant_diamond();
        let dt = DType::F16;
        let (k_scores, k_attn, k_flash) = decode_keys();
        let mut judge = HashMapJudge::new();
        judge.insert(OpKind::MatMul, dt, k_scores, BackendId::Cuda, "", 500);
        judge.insert(OpKind::MatMul, dt, k_attn, BackendId::Cuda, "", 500);
        judge.insert(OpKind::FlashAttn, dt, k_flash, BackendId::Cuda, "", 2000);
        let cost = |g: &Graph, b: NodeId, arm: usize, i: &[NodeId]| -> Option<u64> {
            decode_arm_composite_ns_judged(g, b, arm, i, BackendId::Cuda, Some(&judge))
        };
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 0, "measured flash is slower ⇒ arm 0 (the region) stands");
        assert_eq!(g.node(branch).inputs, vec![attn], "branch collapsed to arm 0");
        let after = lower_runs_arm0(&g, &[post]);
        assert!(after.contains(&attn) && !after.contains(&flash), "region realized (oracle)");
    }

    /// GUARD: with `judge == None` the judged provider is **byte-identical** to
    /// the Layer-1-only provider per arm (the no-Judge production path + the
    /// persistent byte-exact suites stay untouched by construction), and the
    /// bake keeps arm 0.
    #[test]
    fn no_oracle_is_byte_identical_to_layer1() {
        let (mut g, branch, attn, flash, _recon, post) = flash_variant_diamond();
        let interiors = arm_interiors(&g, branch);
        for (arm, interior) in interiors.iter().enumerate() {
            assert_eq!(
                decode_arm_composite_ns_judged(&g, branch, arm, interior, BackendId::Cuda, None),
                decode_arm_composite_ns(&g, branch, arm, interior, BackendId::Cuda),
                "judged(None) must equal the Layer-1 provider for arm {arm}",
            );
        }
        let cost = |g: &Graph, b: NodeId, arm: usize, i: &[NodeId]| -> Option<u64> {
            decode_arm_composite_ns_judged(g, b, arm, i, BackendId::Cuda, None)
        };
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 0, "no oracle ⇒ Layer-1 path ⇒ arm 0 (flash unbound on a CPU build)");
        assert_eq!(g.node(branch).inputs, vec![attn]);
        let _ = flash;
    }

    /// GUARD (capability-missing, via the hybrid fallback): the Judge measured
    /// the region's matmuls but NOT the flash cell, so the flash arm falls back
    /// to Layer-1 where the capability gate (no bound CUDA flash kernel on a CPU
    /// build) makes it inadmissible ⇒ arm 0. A measurement is required to make
    /// an unbound-on-this-build variant admissible.
    #[test]
    fn judge_without_flash_cell_falls_to_capability_gate() {
        let (mut g, branch, attn, flash, _recon, post) = flash_variant_diamond();
        let dt = DType::F16;
        let (k_scores, k_attn, _k_flash) = decode_keys();
        let mut judge = HashMapJudge::new();
        judge.insert(OpKind::MatMul, dt, k_scores, BackendId::Cuda, "", 1);
        judge.insert(OpKind::MatMul, dt, k_attn, BackendId::Cuda, "", 1);
        let cost = |g: &Graph, b: NodeId, arm: usize, i: &[NodeId]| -> Option<u64> {
            decode_arm_composite_ns_judged(g, b, arm, i, BackendId::Cuda, Some(&judge))
        };
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        if !fused_kernel_available(FusedOps::FLASH_ATTN, BackendId::Cuda) {
            assert_eq!(baked, 0, "no flash measurement + unbound kernel ⇒ arm 0");
            assert_eq!(g.node(branch).inputs, vec![attn]);
        }
        let _ = flash;
    }

    /// GUARD (the hybrid sum): when only SOME of an arm's nodes are
    /// Judge-covered, measured cells and Layer-1 estimates are summed together.
    /// Here only the `scores` matmul is measured (500 ns); `attn` falls back to
    /// Layer-1. The region cost is exactly `500 + Layer-1(attn)`, and a cheaper
    /// measured flash still wins.
    #[test]
    fn hybrid_sum_mixes_measured_and_layer1_for_the_region() {
        let (mut g, branch, attn, flash, _recon, post) = flash_variant_diamond();
        let dt = DType::F16;
        let (k_scores, _k_attn, k_flash) = decode_keys();
        let mut judge = HashMapJudge::new();
        judge.insert(OpKind::MatMul, dt, k_scores, BackendId::Cuda, "", 500); // scores only
        judge.insert(OpKind::FlashAttn, dt, k_flash, BackendId::Cuda, "", 400);

        // The region arm's hybrid cost == measured scores (500) + Layer-1 attn.
        let region = arm_interiors(&g, branch)[0].clone();
        let hybrid = decode_arm_composite_ns_judged(
            &g, branch, 0, &region, BackendId::Cuda, Some(&judge),
        )
        .expect("region prices");
        let layer1_attn = decode_arm_composite_ns(&g, branch, 0, &[attn], BackendId::Cuda)
            .expect("attn prices on Layer-1");
        assert_eq!(
            hybrid,
            500 + layer1_attn,
            "hybrid region = measured scores (500) + Layer-1 attn ({layer1_attn})",
        );

        // Flash (measured 400) is cheaper than the hybrid region ⇒ it wins.
        let cost = |g: &Graph, b: NodeId, arm: usize, i: &[NodeId]| -> Option<u64> {
            decode_arm_composite_ns_judged(g, b, arm, i, BackendId::Cuda, Some(&judge))
        };
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 1, "measured flash beats the hybrid region ⇒ bakes to flash");
        assert_eq!(g.node(branch).inputs, vec![flash]);
    }

    /// GUARD: a **placement** branch (arm exits on DIFFERENT backends) is never
    /// touched by the bake — even with a flash-loving Judge — because the
    /// same-device gate in `bake_variant_branches` skips it BEFORE the cost
    /// provider (and thus the Judge) is ever consulted. It stays a live 2-arm
    /// decision point for the runtime route picker.
    #[test]
    fn judge_does_not_touch_a_placement_branch() {
        let (mut g, branch, attn, flash, _recon, post) = flash_variant_diamond();
        // Re-stamp the flash arm onto a DIFFERENT backend ⇒ placement branch.
        g.set_target_backend(flash, BackendId::Cpu);
        let dt = DType::F16;
        let (_ks, _ka, k_flash) = decode_keys();
        let mut judge = HashMapJudge::new();
        judge.insert(OpKind::FlashAttn, dt, k_flash, BackendId::Cpu, "", 1);
        let cost = |g: &Graph, b: NodeId, arm: usize, i: &[NodeId]| -> Option<u64> {
            let backend = g.target_backend(g.node(b).inputs.first().copied()?)?;
            decode_arm_composite_ns_judged(g, b, arm, i, backend, Some(&judge))
        };
        let baked = bake_variant_branches(&mut g, &[post], &cost);
        assert_eq!(baked, 0, "a placement branch is never baked, even with a flash-loving judge");
        assert_eq!(
            g.node(branch).inputs,
            vec![attn, flash],
            "placement branch stays a live 2-arm decision point (runtime picker owns it)",
        );
    }
}
