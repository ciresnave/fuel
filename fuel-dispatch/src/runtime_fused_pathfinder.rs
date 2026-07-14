//! Runtime-fused-op arm **pathfinder** — the match-finder
//! [`crate::runtime_fused_arm`]'s module doc names as its remaining half:
//! "the pathfinder that finds matches + splices reconverge points registers
//! it into `optimize_graph`." This module is that pathfinder.
//!
//! For every op [adopted](fuel_graph::runtime_fused::runtime_entries) at
//! runtime (JIT-synthesized or import-time), it scans the base map for
//! subgraphs whose shape matches the op's region
//! ([`fuel_graph::jit::match_region`]) and offers each match as a gated
//! `Op::Branch` via [`offer_runtime_fused_arm`] — arm 0 stays the region's
//! existing primitive subgraph (the correctness oracle), arm 1 is the
//! synthesized kernel, offered only when
//! [`fused_kernel_available_in`](crate::runtime_fused_kernels::fused_kernel_available_in)
//! gates it open (reading the `OptimizationContext`-threaded binding table,
//! never re-acquiring the global lock). Same constitutional posture as
//! [`crate::decode_flash`]: the
//! optimizer emits/prunes arms, backends never decide, and a graph with no
//! adopted runtime op (or no structural match) is left byte-identical.
//!
//! v1 scope mirrors `fuel_graph::runtime_fused`'s own v1 note: every offered
//! arm is pinned to [`BackendId::Cpu`]. A runtime op can be adopted for
//! multiple backends (`adopt_runtime_fused` is called once per backend), so
//! picking the *right* one — and potentially offering more than one fused
//! arm per match — is a follow-up; see the module-level "unsure" note in the
//! session report.

use std::collections::HashMap;

use fuel_graph::jit::match_region_extract;
use fuel_graph::registry::FusedOpId;
use fuel_graph::runtime_fused::runtime_entries;
use fuel_graph::{Graph, NodeId};
use fuel_ir::Result;
use fuel_ir::probe::BackendId;

use crate::driver::{OptimizationContext, Pathfinder};
use crate::kernel::KernelBindingTable;
use crate::runtime_fused_arm::{RuntimeFusedSpec, offer_runtime_fused_arm};

/// A structural match of a registered runtime op's region against the base
/// map, ready to become a [`RuntimeFusedSpec`] once collection is done (kept
/// separate from the spec so the collection pass stays a read-only borrow of
/// `graph`, mirroring [`crate::driver::PlacementForkPathfinder`]'s two-phase
/// collect-then-mutate shape).
struct RegionMatch {
    runtime_id: FusedOpId,
    inputs: Vec<NodeId>,
    /// Live `extract:` values from the matched region's open scalar slots,
    /// pattern pre-order (stamped into the arm's `Runtime { scalars }`).
    scalars: Vec<f64>,
    primitive_sink: NodeId,
    reconverge: NodeId,
}

/// Scan `graph` for subgraphs matching an adopted runtime fused op's region
/// and offer each as a gated `Op::Branch` (see [`offer_runtime_fused_arm`]).
/// Returns the number of arms emitted — `0` for a graph with no adopted
/// runtime op, or none whose region structurally matches anywhere in it.
///
/// For each registered runtime op, every node in `graph` is tried as the
/// region's `root` (arm 0's exit / [`match_region`]'s sink). A match binds
/// the region's external inputs; the candidate is only a genuine branch
/// point when `root` has **exactly one** consumer (the sole reconverge —
/// zero consumers is a dead end, ≥2 is ordinary fan-out neither arm can
/// safely replace). Matches are collected first (read-only over `graph`),
/// then spliced (mutating), exactly like the placement-fork pathfinder.
///
/// `table` is the binding table each match's capability gate reads —
/// threaded straight through to [`offer_runtime_fused_arm`] (see its doc for
/// why this is the non-nesting form of the gate).
pub fn emit_runtime_fused_arms(graph: &mut Graph, table: &KernelBindingTable) -> Result<usize> {
    let entries = runtime_entries();
    if entries.is_empty() {
        return Ok(0);
    }

    // Consumer counts over the WHOLE graph — a runtime region can sit
    // anywhere in the base map, not just along a particular dispatch order
    // (mirrors `match_region`'s own tests' `consumer_counts` helper).
    let mut consumer_count: HashMap<NodeId, usize> = HashMap::new();
    let mut sole_consumer: HashMap<NodeId, NodeId> = HashMap::new();
    let len = graph.len();
    for i in 0..len {
        let id = NodeId(i);
        for &input in &graph.node(id).inputs {
            *consumer_count.entry(input).or_insert(0) += 1;
            sole_consumer.insert(input, id);
        }
    }
    let consumers = |n: NodeId| consumer_count.get(&n).copied().unwrap_or(0);

    // A `root` claimed by one entry's match is not offered again for another
    // entry: two runtime ops can register structurally-identical regions
    // (e.g. the same fusion synthesized twice under different hashes), and a
    // second `open_branch` at an already-branched diverge point is a
    // malformed splice (the first branch's arm-0 already reads `root`, so the
    // second attempt trips the "arm interior read from outside the branch"
    // guard). First registered entry to match a given `root` wins it; this is
    // a dedup, not a priority order — a graph rarely contains a genuine
    // ambiguity like this, and when it does, offering one arm is still
    // correct (arm 0 stays the primitive fallback either way).
    let mut claimed: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    let mut matches: Vec<RegionMatch> = Vec::new();
    for entry in &entries {
        for i in 0..len {
            let root = NodeId(i);
            if claimed.contains(&root) {
                continue;
            }
            let Some((inputs, scalars)) =
                match_region_extract(graph, root, &entry.region, &consumers)
            else {
                continue;
            };
            // Exactly one consumer of `root` ⇒ a genuine reconverge point.
            if consumers(root) != 1 {
                continue;
            }
            claimed.insert(root);
            matches.push(RegionMatch {
                runtime_id: entry.id,
                inputs,
                scalars,
                primitive_sink: root,
                reconverge: sole_consumer[&root],
            });
        }
    }

    let mut emitted = 0usize;
    for m in matches {
        let spec = RuntimeFusedSpec {
            runtime_id: m.runtime_id,
            inputs: m.inputs,
            primitive_sink: m.primitive_sink,
            reconverge: m.reconverge,
            // v1: pinned to Cpu — see the module doc's "unsure" note.
            backend: BackendId::Cpu,
            // The live `extract:` values (pattern pre-order) — stamped into the
            // arm's `Runtime { scalars }`, launched as the trailing `p{i}` args.
            scalars: m.scalars,
        };
        if offer_runtime_fused_arm(graph, &spec, table)?.is_some() {
            emitted += 1;
        }
    }
    Ok(emitted)
}

/// The registered [`Pathfinder`] wrapper around [`emit_runtime_fused_arms`].
/// Ignores `ctx`'s ranked plan / dispatch order: a runtime-op region match is
/// a purely structural question over the whole base map, independent of
/// placement — the same reason [`crate::decode_flash`]'s arm is emitted by a
/// dedicated call site rather than reading `ctx.plan`.
pub struct RuntimeFusedArmPathfinder;

impl Pathfinder for RuntimeFusedArmPathfinder {
    fn name(&self) -> &'static str {
        "RuntimeFusedArm"
    }

    fn propose(&self, graph: &mut Graph, ctx: &OptimizationContext<'_>) -> Result<()> {
        emit_runtime_fused_arms(graph, ctx.bindings())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_fused_kernels::adopt_runtime_fused;
    use fuel_graph::jit::{OpAttrs, OpTag, PatternNode};
    use fuel_graph::{Node, Op};
    use fuel_ir::{DType, Layout, Shape};
    use std::sync::{Arc, RwLock as StdRwLock};

    fn noop_kernel(
        _inputs: &[Arc<StdRwLock<fuel_memory::Storage>>],
        _outputs: &mut [Arc<StdRwLock<fuel_memory::Storage>>],
        _layouts: &[Layout],
        _params: &crate::kernel::OpParams,
    ) -> fuel_ir::Result<()> {
        Ok(())
    }

    /// tanh(mul(a, b)) as a PatternNode region. Deliberately NOT the
    /// `relu(add(a, b))` shape every other adopted-op test in this crate
    /// uses (`fused_cost`, `jit_adopt`, `runtime_fused_arm`,
    /// `runtime_fused_kernels`) — `runtime_entries()` is a process-global
    /// sidecar shared by every test in this binary, so a region matching
    /// one of those adopted shapes would non-deterministically match
    /// whichever of them happened to register first, defeating the
    /// `rid`-specific assertions below (and, before the `claimed`-root dedup
    /// this pathfinder now has, would double-branch the same diverge point
    /// and panic).
    fn tanh_mul_region() -> PatternNode {
        PatternNode::Op {
            op: OpTag::Tanh,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Mul,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        }
    }

    /// tanh(mul(a, b)) plus a downstream `neg` consumer (the reconverge).
    fn graph_with_region() -> (Graph, NodeId) {
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let leaf = |g: &mut Graph| {
            g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 })
        };
        let a = leaf(&mut g);
        let b = leaf(&mut g);
        let mul = g.push(Node { op: Op::Mul, inputs: vec![a, b], shape: s.clone(), dtype: DType::F32 });
        let tanh = g.push(Node { op: Op::Tanh, inputs: vec![mul], shape: s.clone(), dtype: DType::F32 });
        let _neg = g.push(Node { op: Op::Neg, inputs: vec![tanh], shape: s.clone(), dtype: DType::F32 });
        (g, tanh)
    }

    #[test]
    fn emits_a_fused_arm_for_an_adopted_region_match() {
        let rid = adopt_runtime_fused(
            "test::pathfinder::tanh_mul",
            tanh_mul_region(),
            noop_kernel as crate::kernel::KernelRef,
            vec![DType::F32, DType::F32, DType::F32],
            BackendId::Cpu,
        )
        .expect("registrable region");

        let (mut g, tanh) = graph_with_region();
        let before = g.len();

        // adopt_runtime_fused binds into the GLOBAL table; pass it explicitly
        // as the threaded table this test exercises the pathfinder against.
        let emitted = emit_runtime_fused_arms(&mut g, &crate::dispatch::global_bindings())
            .expect("no build-time error");
        assert_eq!(emitted, 1, "exactly one match ⇒ one arm emitted");
        assert!(g.len() > before, "the fused node + branch were appended");

        // A Branch was recorded at (or after) `tanh`'s old position, and a
        // Fused(rid, ..) node exists among the new nodes.
        let mut saw_branch = false;
        let mut saw_fused = false;
        for i in before..g.len() {
            match &g.node(NodeId(i)).op {
                Op::Branch { .. } => saw_branch = true,
                Op::Fused(id, _) if *id == rid => saw_fused = true,
                _ => {}
            }
        }
        assert!(saw_branch, "an Op::Branch decision point was emitted");
        assert!(saw_fused, "an Op::Fused(rid, ..) arm was emitted");
        // `tanh` (arm 0 / the primitive sink) is untouched — still the
        // region's Tanh output the correctness oracle relies on.
        assert!(matches!(g.node(tanh).op, Op::Tanh));
    }

    #[test]
    fn no_adopted_op_emits_nothing() {
        // No `adopt_runtime_fused` call in this test — `runtime_entries()`
        // may still carry ops other tests adopted (a process-global sidecar),
        // but NONE of them match a region this graph doesn't structurally
        // contain: a bare `neg(a)` has no relu/add/tanh/mul anywhere.
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let a = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let _n = g.push(Node { op: Op::Neg, inputs: vec![a], shape: s.clone(), dtype: DType::F32 });
        let before = g.len();

        // The production table (any op other tests adopted lives here too —
        // irrelevant, since none structurally matches this graph).
        let emitted = emit_runtime_fused_arms(&mut g, &crate::dispatch::global_bindings())
            .expect("no build-time error");
        assert_eq!(emitted, 0, "no structural match ⇒ no arm emitted");
        assert_eq!(g.len(), before, "graph untouched");
    }

    /// A slot-template region (`sigmoid(add_scalar(x))`, value open): the
    /// pathfinder extracts the LIVE scalar from the matched graph node and
    /// stamps it into the emitted arm's `Runtime { scalars }` — the `extract:`
    /// round-trip at the arm level. Region shape unique to this test (the
    /// sidecar is process-global; see `tanh_mul_region`'s rationale).
    #[test]
    fn emitted_arm_carries_the_extracted_live_scalar() {
        let region = PatternNode::Op {
            op: OpTag::Sigmoid,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::AddScalar,
                attrs: OpAttrs::default(), // empty = open slot
                operands: vec![PatternNode::Bind { index: 0 }],
            }],
        };
        let rid = adopt_runtime_fused(
            "test::pathfinder::sigmoid_add_scalar",
            region,
            noop_kernel as crate::kernel::KernelRef,
            vec![DType::F32, DType::F32],
            BackendId::Cpu,
        )
        .expect("slot template is registrable");

        // sigmoid(add_scalar(x, 7.25)) + a downstream consumer.
        let mut g = Graph::new();
        let s = Shape::from_dims(&[4]);
        let x = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: DType::F32 });
        let asn = g.push(Node {
            op: Op::AddScalar(7.25),
            inputs: vec![x],
            shape: s.clone(),
            dtype: DType::F32,
        });
        let sig = g.push(Node {
            op: Op::Sigmoid, inputs: vec![asn], shape: s.clone(), dtype: DType::F32,
        });
        let _neg = g.push(Node {
            op: Op::Neg, inputs: vec![sig], shape: s.clone(), dtype: DType::F32,
        });
        let before = g.len();

        // adopt_runtime_fused binds into the GLOBAL table; pass it explicitly
        // as the threaded table this test exercises the pathfinder against.
        let emitted = emit_runtime_fused_arms(&mut g, &crate::dispatch::global_bindings())
            .expect("no build-time error");
        assert_eq!(emitted, 1, "the slot region matched once");

        let mut found = false;
        for i in before..g.len() {
            if let Op::Fused(fid, fuel_graph::registry::FusedOpParams::Runtime { scalars }) =
                &g.node(NodeId(i)).op
            {
                if *fid == rid {
                    assert_eq!(scalars, &vec![7.25], "the LIVE value rode into the arm");
                    found = true;
                }
            }
        }
        assert!(found, "an Op::Fused(rid, Runtime) arm was emitted");
    }
}
