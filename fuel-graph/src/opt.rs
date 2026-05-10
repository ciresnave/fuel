//! Graph-level, backend-agnostic optimization passes.
//!
//! Transforms a `fuel-graph` computation graph before execution. Every
//! backend benefits from these passes because they operate purely on
//! the abstract op graph — no backend-specific knowledge required.
//!
//! **Passes today:**
//!
//! - **CSE** (common subexpression elimination): if the graph contains
//!   two structurally-identical nodes — same op, same inputs (after
//!   prior simplification), same shape/dtype — consumers of the
//!   duplicate are redirected to the canonical node. The duplicate
//!   becomes unreferenced and is silently dropped by the executor's
//!   topo-walk. Commutative ops (`Add`, `Mul`, `Maximum`, `Minimum`)
//!   are keyed on sorted input IDs so `a + b` and `b + a` fold to the
//!   same canonical node.
//!
//! - **Algebraic simplification**: a handful of identity/zero rules
//!   that eliminate no-op ops:
//!   - `AddScalar(0.0)(x)` → `x`
//!   - `MulScalar(1.0)(x)` → `x`
//!   - `Neg(Neg(x))` → `x`
//!   - `Reshape(Reshape(x, _), s)` → `Reshape(x, s)`
//!   These rarely appear in hand-written user code but show up
//!   routinely in autograd-generated backward graphs and in
//!   generic transformer building blocks that sometimes collapse.
//!
//! **Design:** graphs are append-only, so rewrites don't mutate
//! existing nodes. Instead, the pass walks topologically, builds a
//! remap `HashMap<NodeId, NodeId>` of old → canonical, and appends
//! newly-canonicalized nodes to the same graph. Unreferenced originals
//! remain in the arena but are never visited during realize. This is
//! effectively a combined CSE + simplification + free DCE pass.
//!
//! **Return value:** the function takes the roots the caller cares
//! about and returns the rewritten roots. Callers use these to update
//! their `Tensor` handles.

use crate::registry::{
    default_registry, FusedOpEntry, FusedOpId, FusedOpParams, FusedOps, SubgraphPattern,
};
use crate::{topo_order_multi, Graph, Node, NodeId, Op, SharedGraph};
use fuel_core_types::{DeviceLocation, DType, Shape};
use std::collections::HashMap;

// ---- Rule registry framework ----------------------------------------------
//
// Optimization is a pipeline of rule-driven graph rewrites. Two rule
// families:
//
// - `RuleFamily::Lowering` — high-level op → primitive subgraph. Exposes
//   fusion opportunities and lets backends without a fused kernel run
//   the composition through whichever primitives they DO have.
// - `RuleFamily::Fusion` — recognized primitive subgraph → fused op.
//   Recovers (or improves on) the original-flavour kernel where one
//   exists.
//
// A `Rule` is `(matcher, rewriter)`. The driver in
// [`RuleRegistry::optimize_to_fixpoint`] runs lowering rules to fixpoint
// first, then fusion rules to fixpoint. End state is post-fusion: for
// SoftmaxLastDim (whose lower and fuse rules are inverses of each
// other) the round-trip is identity. For lowering rules that ship
// without a fusion partner — RmsNorm, LayerNorm, Affine, Clamp — the
// post-pipeline graph is the lowered form.
//
// PR 3 ships the framework + a single (lower, fuse) rule pair for
// `Op::SoftmaxLastDim`. The locked design (see ROADMAP.md → Phase 7.5)
// also covers transactional snapshots, in-flight switching, and budget
// modes; those are subsequent PRs that wrap this synchronous loop
// without rewriting it.

/// Which family a [`Rule`] belongs to. Drives the phase ordering in
/// [`RuleRegistry::optimize_to_fixpoint`]: lowering runs to fixpoint
/// first, then fusion. This ordering ensures inverse rule pairs
/// converge cleanly — fusion has the last word, and its output is
/// what the executor runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleFamily {
    /// High-level op → primitive subgraph. Always grows the graph.
    Lowering,
    /// Recognized primitive subgraph → fused op. Always shrinks the
    /// graph (the fused op replaces the canonical subgraph's output;
    /// the orphaned subgraph nodes remain in the arena but are
    /// unreachable from the user's roots after consumer rewiring).
    Fusion,
}

/// One graph-rewrite rule. A rule pairs a structural matcher with a
/// rewriter that emits replacement nodes and records the
/// `(old_id → new_id)` remapping so the driver can rewrite consumer
/// input edges to point at the replacement.
///
/// Rules don't need to update the layout side-table for view nodes
/// they insert: [`Graph::push`] auto-populates layout for view ops
/// from the input's layout.
///
/// Rules don't express ordering edges for destructive ops they emit
/// (e.g. `Op::Release`); [`derive_ordering`] does that automatically
/// at execution-plan time.
pub trait Rule: Send + Sync {
    /// Stable, human-readable name. Shows up in panic messages and
    /// debug traces.
    fn name(&self) -> &'static str;

    /// Which family this rule belongs to. Determines pass ordering.
    fn family(&self) -> RuleFamily;

    /// Whether this rule can rewrite the node at `id`. Should be
    /// pure — same inputs, same answer. Called once per node per
    /// pass; the first matching rule wins.
    fn matches(&self, graph: &Graph, id: NodeId) -> bool;

    /// Apply the rewrite: append fresh nodes to the graph, declare
    /// any side-effect roots if the rewrite emits side-effecting
    /// ops, and write `remap.insert(id, replacement_id)` so the
    /// driver can redirect consumer edges. Calling this method
    /// implies [`matches`](Self::matches) returned `true` for the
    /// same `(graph, id)` — rewriters can rely on the structure
    /// the matcher checked for.
    fn rewrite(&self, graph: &mut Graph, id: NodeId, remap: &mut HashMap<NodeId, NodeId>);
}

/// A registry of [`Rule`]s. Drives optimize-to-fixpoint via
/// [`Self::optimize_to_fixpoint`]: lowering rules run to fixpoint,
/// then fusion rules run to fixpoint. The locked design (ROADMAP.md
/// Phase 7.5) intends this entry point to be wrapped in transactions
/// in subsequent PRs without rewriting the loop body — keep this API
/// stable for that future evolution.
#[derive(Default)]
pub struct RuleRegistry {
    rules: Vec<Box<dyn Rule>>,
}

impl RuleRegistry {
    /// Empty registry. Use [`Self::with_rule`] to add rules, or
    /// [`Self::default_rules`] / [`Self::lowering_only`] for the
    /// shipped configurations.
    pub fn new() -> Self { Self { rules: Vec::new() } }

    /// Append a rule. Returns self for builder-style chaining.
    pub fn with_rule(mut self, rule: Box<dyn Rule>) -> Self {
        self.rules.push(rule);
        self
    }

    /// Lower + fuse rules for every fused op registered in the
    /// process-wide [`default_registry`]. Round-trip is identity for
    /// canonical SoftmaxLastDim graphs (the lowered subgraph is
    /// structurally re-fused). Use this in production realize paths.
    ///
    /// Phase 7.6 step 3: PR 3's hand-written
    /// `SoftmaxLastDimLowerRule`/`SoftmaxLastDimFuseRule` are deleted;
    /// equivalent behavior is auto-generated from the SoftmaxLastDim
    /// registry entry. Subsequent fused-op migrations in step 4 add
    /// their entries to the same registry; `default_rules` picks them
    /// up automatically.
    pub fn default_rules() -> Self {
        let registry = default_registry();
        let mut r = Self::new();
        for entry in registry.entries_iter() {
            r = r
                .with_rule(Box::new(LoweringRule::from_entry(entry)))
                .with_rule(Box::new(FusionRule::from_entry(entry)));
        }
        r
    }

    /// Lowering rules only — no fusion. Use this when the test or
    /// caller wants the lowered form to be the post-pipeline state
    /// (e.g. equivalence-testing the composed-math path on a backend
    /// that has the primitives but not the fused kernel).
    pub fn lowering_only() -> Self {
        let registry = default_registry();
        let mut r = Self::new();
        for entry in registry.entries_iter() {
            r = r.with_rule(Box::new(LoweringRule::from_entry(entry)));
        }
        r
    }

    /// Number of registered rules.
    pub fn len(&self) -> usize { self.rules.len() }

    /// Whether the registry has any rules.
    pub fn is_empty(&self) -> bool { self.rules.is_empty() }

    /// Run lowering rules to fixpoint, then fusion rules to fixpoint.
    /// Returns the (possibly-rewritten) roots — callers use these to
    /// replace their `Tensor` handles.
    ///
    /// The two-phase ordering is important: rules that are inverses
    /// of each other (e.g. SoftmaxLastDim's lower + fuse pair)
    /// would oscillate in a single mixed-family fixpoint loop.
    /// Phase ordering breaks the oscillation and gives fusion the
    /// last word.
    pub fn optimize_to_fixpoint(
        &self,
        graph: &SharedGraph,
        roots: &[NodeId],
    ) -> Vec<NodeId> {
        let mut roots = roots.to_vec();
        if self.rules.is_empty() { return roots; }
        // Phase 1: lowering to fixpoint.
        loop {
            let any = self.run_pass(RuleFamily::Lowering, graph, &mut roots);
            if !any { break; }
        }
        // Phase 2: fusion to fixpoint.
        loop {
            let any = self.run_pass(RuleFamily::Fusion, graph, &mut roots);
            if !any { break; }
        }
        roots
    }

    /// One pass over the graph for a single rule family. Returns
    /// `true` if any rule fired during the pass — caller loops until
    /// `false` for fixpoint.
    fn run_pass(
        &self,
        family: RuleFamily,
        graph: &SharedGraph,
        roots: &mut Vec<NodeId>,
    ) -> bool {
        let order = {
            let g = graph.read().unwrap();
            topo_order_multi(&g, roots)
        };

        // Iterate this pass's family-rules in the order they were
        // registered. First match wins per node.
        let family_rules: Vec<&Box<dyn Rule>> = self
            .rules
            .iter()
            .filter(|r| r.family() == family)
            .collect();
        if family_rules.is_empty() { return false; }

        let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
        let mut any_fired = false;

        for id in order {
            // Skip nodes that an earlier rule in this pass already
            // rewrote — the consumer-edge rewrite happens after the
            // pass, so the original node's structure still looks
            // matchable mid-pass; we use `remap` as a "this pass
            // already handled it" marker.
            if remap.contains_key(&id) { continue; }

            let firing_rule_idx = {
                let g = graph.read().unwrap();
                family_rules.iter().position(|r| r.matches(&g, id))
            };
            if let Some(idx) = firing_rule_idx {
                let mut g = graph.write().unwrap();
                family_rules[idx].rewrite(&mut g, id, &mut remap);
                any_fired = true;
            }
        }

        if !any_fired { return false; }

        // Apply remap to every consumer's input list, in place. This
        // is the moment the rewritten nodes become the live ones —
        // any node whose inputs referenced a remapped id now reads
        // from its replacement.
        {
            let mut g = graph.write().unwrap();
            let n_nodes = g.len();
            for nid in 0..n_nodes {
                let inputs_len = g.node(NodeId(nid)).inputs.len();
                for i in 0..inputs_len {
                    let cur = g.node(NodeId(nid)).inputs[i];
                    if let Some(&new) = remap.get(&cur) {
                        g.rewrite_input(NodeId(nid), cur, new);
                    }
                }
            }
        }

        // And update the user's roots so the next phase / the caller
        // sees the canonical post-rewrite ids.
        for r in roots.iter_mut() {
            if let Some(&new) = remap.get(r) { *r = new; }
        }

        true
    }
}

// ---- Auto-generated lowering + fusion rules from FusedOpRegistry ----------
//
// Phase 7.6 step 3: PR 3's hand-written `SoftmaxLastDimLowerRule` and
// `SoftmaxLastDimFuseRule` are replaced by [`LoweringRule`] +
// [`FusionRule`] generic abstractions that read the per-fused-op
// metadata (decompose function, pattern matcher) from a
// [`FusedOpEntry`] in `crate::registry::default_registry`. Each rule
// fires on `Op::Fused(id, _)` for lowering or on the pattern's
// canonical primitive subgraph for fusion.
//
// Lowered form (7 nodes for SoftmaxLastDim, symmetric across max/sum):
//
//   m   = ReduceMaxTo([..., 1])(x)   # max-keepdim in one node
//   mb  = BroadcastTo([..., last])(m)
//   s   = Sub(x, mb)                 # numerically-stable shift
//   e   = Exp(s)
//   d   = ReduceSumTo([..., 1])(e)   # sum-keepdim in one node
//   db  = BroadcastTo([..., last])(d)
//   out = Div(e, db)
//
// The compile_one auto-Contiguize at the executor catches any backend
// that doesn't support the strided BroadcastTo output as a kernel
// input. PR 2-wide lets CUDA binary F32 kernels consume the broadcast
// strides directly without materialization.

/// Auto-generated lowering rule from a [`FusedOpEntry`]. Matches
/// `Op::Fused(id, _)` and decomposes via the entry's `decompose`
/// function. Each rule corresponds to one fused-op id in the registry.
///
/// Step 3 also fires on the legacy per-fused-op `Op` variants (e.g.
/// `Op::SoftmaxLastDim`) so emission sites that haven't yet migrated
/// to `Op::Fused` continue to lower correctly. Step 5 deletes those
/// variants and the legacy match arm with them.
pub struct LoweringRule {
    id:        FusedOpId,
    decompose: fn(&mut Graph, NodeId, &FusedOpParams) -> NodeId,
}

impl LoweringRule {
    pub fn from_entry(entry: &FusedOpEntry) -> Self {
        Self { id: entry.id, decompose: entry.decompose }
    }
}

impl Rule for LoweringRule {
    fn name(&self) -> &'static str {
        // Step 3 ships only SoftmaxLastDim; step 4 generalizes per-id.
        // Until then, all auto-generated lowering rules share one name —
        // the registry id distinguishes them at the structural level.
        "FusedOpLowering"
    }

    fn family(&self) -> RuleFamily { RuleFamily::Lowering }

    fn matches(&self, graph: &Graph, id: NodeId) -> bool {
        let node = graph.node(id);
        // Single-input shape check — every fused op registered in step
        // 3 is a single-input op (SoftmaxLastDim). Generalize when an
        // op with arity > 1 lands.
        if node.inputs.len() != 1 || node.shape.rank() < 1 {
            return false;
        }
        match &node.op {
            Op::Fused(fid, _) if *fid == self.id => true,
            // Legacy variant fallthrough: the SoftmaxLastDim entry's
            // lowering rule also fires on `Op::SoftmaxLastDim` so that
            // emission sites that haven't migrated to the
            // `Op::Fused(SOFTMAX_LAST_DIM, _)` form (e.g. the
            // pipelined-executor test that constructs the node
            // directly) keep lowering correctly. Step 5 drops the
            // legacy variant; this arm goes with it.
            Op::SoftmaxLastDim if self.id == FusedOps::SOFTMAX_LAST_DIM => true,
            _ => false,
        }
    }

    fn rewrite(&self, graph: &mut Graph, id: NodeId, remap: &mut HashMap<NodeId, NodeId>) {
        // Synthesize the params from the matched node's variant.
        // SoftmaxLastDim has no per-instance payload so this is just a
        // canonical FusedOpParams::SoftmaxLastDim. Step 4's per-fused-op
        // migrations extend this match.
        let params = match &graph.node(id).op {
            Op::Fused(_, p) => p.clone(),
            Op::SoftmaxLastDim => FusedOpParams::SoftmaxLastDim,
            other => unreachable!(
                "LoweringRule::rewrite reached with non-matching op {other:?} \
                 — matcher contract violated"
            ),
        };
        let new_id = (self.decompose)(graph, id, &params);
        remap.insert(id, new_id);
    }
}

/// Auto-generated fusion rule from a [`FusedOpEntry`]. Matches the
/// canonical primitive-subgraph pattern carried by the entry's
/// [`SubgraphPattern`] and emits a single `Op::Fused(id, params)`
/// node. Round-trips with [`LoweringRule`] for the same id.
///
/// Currently params are reconstructed by-id: SoftmaxLastDim's variant
/// is parameterless, so the fusion rule emits
/// `FusedOpParams::SoftmaxLastDim`. Step 4 extends [`PatternMatch`]
/// (or adds a sibling `extract_params` field on `FusedOpEntry`) so
/// param-bearing fused ops can recover their parameters from the
/// matched subgraph.
pub struct FusionRule {
    id:      FusedOpId,
    pattern: PatternKind,
}

/// Internal: the runtime-flavor of a registry entry's pattern. We
/// keep a function-pointer copy here rather than holding a reference
/// into the (process-wide) registry so the rule is `'static`.
enum PatternKind {
    Callable(fn(&Graph, NodeId) -> Option<crate::registry::PatternMatch>),
    /// Step 4 wires the declarative form. Until then, declarative
    /// patterns from the registry don't fuse — the fusion rule is
    /// constructed but `matches` always returns false.
    Declarative,
}

impl FusionRule {
    pub fn from_entry(entry: &FusedOpEntry) -> Self {
        let pattern = match &entry.pattern {
            SubgraphPattern::Callable(f) => PatternKind::Callable(*f),
            SubgraphPattern::Declarative(_) => PatternKind::Declarative,
        };
        Self { id: entry.id, pattern }
    }
}

impl Rule for FusionRule {
    fn name(&self) -> &'static str { "FusedOpFusion" }
    fn family(&self) -> RuleFamily { RuleFamily::Fusion }

    fn matches(&self, graph: &Graph, id: NodeId) -> bool {
        match self.pattern {
            PatternKind::Callable(f) => f(graph, id).is_some(),
            PatternKind::Declarative => false,
        }
    }

    fn rewrite(&self, graph: &mut Graph, id: NodeId, remap: &mut HashMap<NodeId, NodeId>) {
        let pattern_match = match self.pattern {
            PatternKind::Callable(f) => f(graph, id)
                .expect("rewrite called with non-matching id — matcher contract violated"),
            PatternKind::Declarative => unreachable!(
                "Declarative fusion patterns aren't fired in step 3"
            ),
        };
        // Bindings convention: index 0 is the "input x" for SoftmaxLastDim.
        // Step 4 generalizes per-fused-op when arity > 1 lands.
        let x_id = pattern_match
            .bindings
            .iter()
            .find(|(idx, _)| *idx == 0)
            .map(|(_, n)| *n)
            .expect("FusionRule expects a binding at index 0 (the input x)");
        // Reconstruct params by-id. Step 4 extends this match with the
        // remaining 12 fused ops (some of which carry payloads).
        let params = if self.id == FusedOps::SOFTMAX_LAST_DIM {
            FusedOpParams::SoftmaxLastDim
        } else {
            unreachable!(
                "FusionRule for id {:?} has no params reconstructor — \
                 step 4 extends this match",
                self.id,
            );
        };
        let dtype = graph.node(id).dtype;
        let shape = graph.node(id).shape.clone();
        let new_id = graph.push(Node {
            op:     Op::Fused(self.id, params),
            inputs: vec![x_id],
            shape,
            dtype,
        });
        remap.insert(id, new_id);
    }
}

/// A HashMap-friendly encoding of `Op`. Needed because `Op` carries
/// `f64` (in `AddScalar`, `MulScalar`, `Clamp`, `LayerNormLastDim`,
/// `LayerNormLastDimBackward`) and `ConstData` (in `Const`), neither
/// of which is `Hash + Eq`. We keep `Const` out of CSE entirely
/// (identity-dedup happens via the executor's Arc-pointer const pool
/// already) and encode scalar payloads as their bit patterns.
#[derive(Hash, PartialEq, Eq)]
struct OpKey {
    tag: u16,
    ints: Vec<i64>,
    bits: Vec<u64>,
    // Sparse payloads we don't need to index into for equality are
    // serialized via their Debug repr as a fallback. Cheap and
    // correct; not used on the hot path outside simplification.
    dims: Vec<usize>,
    shape: Option<Vec<usize>>,
    dtype: Option<u32>,
}

fn op_key(op: &Op) -> Option<OpKey> {
    // We deliberately refuse to CSE `Const`. Rationale above.
    let (tag, ints, bits, dims, shape, dtype) = match op {
        Op::Const => return None,

        Op::Add => (1, vec![], vec![], vec![], None, None),
        Op::Sub => (2, vec![], vec![], vec![], None, None),
        Op::Mul => (3, vec![], vec![], vec![], None, None),
        Op::Div => (4, vec![], vec![], vec![], None, None),

        Op::Neg => (10, vec![], vec![], vec![], None, None),
        Op::Sqr => (11, vec![], vec![], vec![], None, None),
        Op::Sqrt => (12, vec![], vec![], vec![], None, None),
        Op::Exp => (13, vec![], vec![], vec![], None, None),
        Op::Log => (14, vec![], vec![], vec![], None, None),
        Op::Sin => (15, vec![], vec![], vec![], None, None),
        Op::Cos => (16, vec![], vec![], vec![], None, None),
        Op::Tanh => (17, vec![], vec![], vec![], None, None),
        Op::Sigmoid => (18, vec![], vec![], vec![], None, None),
        Op::Silu => (19, vec![], vec![], vec![], None, None),
        Op::Gelu => (20, vec![], vec![], vec![], None, None),
        Op::Relu => (21, vec![], vec![], vec![], None, None),
        Op::Step => (22, vec![], vec![], vec![], None, None),
        Op::Recip => (23, vec![], vec![], vec![], None, None),
        Op::Abs => (24, vec![], vec![], vec![], None, None),

        // --- comparison family (output dtype is U8) ---
        Op::Equal => (25, vec![], vec![], vec![], None, None),
        Op::Ne => (26, vec![], vec![], vec![], None, None),
        Op::Lt => (27, vec![], vec![], vec![], None, None),
        Op::Le => (28, vec![], vec![], vec![], None, None),
        Op::Gt => (29, vec![], vec![], vec![], None, None),
        // Tag 30 already taken by Op::MatMul below; the comparison
        // family wraps to 33 (the next free slot above linear-algebra)
        // for Op::Ge. The family sits at 25-29 + 33 — a small
        // contiguity break tolerated rather than renumbering existing
        // ops opportunistically.
        Op::Ge => (33, vec![], vec![], vec![], None, None),

        // Ternary select. Slot 34 (next free after the comparison
        // family's wrap to 33).
        Op::Where => (34, vec![], vec![], vec![], None, None),

        // --- rounding family (non-differentiable) ---
        Op::Floor => (35, vec![], vec![], vec![], None, None),
        Op::Ceil => (36, vec![], vec![], vec![], None, None),
        Op::Round => (37, vec![], vec![], vec![], None, None),
        Op::Sign => (38, vec![], vec![], vec![], None, None),
        Op::Erf => (39, vec![], vec![], vec![], None, None),
        // Tag 40 was Op::Cast; the unary fanout wraps to 47 (next
        // free slot above the 40-46 cast/shape/reduce cluster).
        Op::GeluErf => (47, vec![], vec![], vec![], None, None),
        Op::Pow => (49, vec![], vec![], vec![], None, None),
        Op::Rsqrt => (54, vec![], vec![], vec![], None, None),
        Op::Rem => (55, vec![], vec![], vec![], None, None),

        Op::MatMul => (30, vec![], vec![], vec![], None, None),
        Op::Transpose => (31, vec![], vec![], vec![], None, None),
        Op::Permute(axes) => (32, vec![], vec![], axes.clone(), None, None),

        Op::Cast(dt) => (40, vec![], vec![], vec![], None, Some(dtype_key(*dt))),
        Op::BroadcastTo(s) => (41, vec![], vec![], vec![], Some(s.dims().to_vec()), None),
        Op::Reshape(s) => (42, vec![], vec![], vec![], Some(s.dims().to_vec()), None),
        Op::Unsqueeze { dim } => (45, vec![*dim as i64], vec![], vec![], None, None),
        Op::Squeeze { dim } => (48, vec![*dim as i64], vec![], vec![], None, None),
        Op::ReduceSumTo(s) => (43, vec![], vec![], vec![], Some(s.dims().to_vec()), None),
        Op::ReduceMaxTo(s) => (44, vec![], vec![], vec![], Some(s.dims().to_vec()), None),
        Op::ReduceMaxToBackward => (46, vec![], vec![], vec![], None, None),

        Op::SumAll => (50, vec![], vec![], vec![], None, None),
        Op::MaxAll => (51, vec![], vec![], vec![], None, None),
        Op::MinAll => (52, vec![], vec![], vec![], None, None),
        Op::MeanAll => (53, vec![], vec![], vec![], None, None),

        Op::SumDim(d) => (60, vec![*d as i64], vec![], vec![], None, None),
        Op::MaxDim(d) => (61, vec![*d as i64], vec![], vec![], None, None),
        Op::MinDim(d) => (62, vec![*d as i64], vec![], vec![], None, None),
        Op::MeanDim(d) => (63, vec![*d as i64], vec![], vec![], None, None),
        Op::ArgMaxDim(d) => (64, vec![*d as i64], vec![], vec![], None, None),
        Op::ArgMinDim(d) => (65, vec![*d as i64], vec![], vec![], None, None),

        Op::SoftmaxLastDim => (70, vec![], vec![], vec![], None, None),
        Op::LayerNormLastDim { eps } => (71, vec![], vec![eps.to_bits()], vec![], None, None),
        Op::RmsNormLastDim { eps } => (74, vec![], vec![eps.to_bits()], vec![], None, None),
        Op::Rope => (75, vec![], vec![], vec![], None, None),
        Op::RmsNormLastDimBackward { eps } => (76, vec![], vec![eps.to_bits()], vec![], None, None),
        Op::SoftmaxLastDimBackward => (72, vec![], vec![], vec![], None, None),
        Op::LayerNormLastDimBackward { eps } => {
            (73, vec![], vec![eps.to_bits()], vec![], None, None)
        }

        Op::Concat { dim } => (80, vec![*dim as i64], vec![], vec![], None, None),
        Op::Slice { dim, start, len } => (
            81,
            vec![*dim as i64, *start as i64, *len as i64],
            vec![],
            vec![],
            None,
            None,
        ),

        Op::AddScalar(c) => (90, vec![], vec![c.to_bits()], vec![], None, None),
        Op::MulScalar(c) => (91, vec![], vec![c.to_bits()], vec![], None, None),
        Op::PowI(n) => (92, vec![*n as i64], vec![], vec![], None, None),
        Op::Clamp { min, max } => (93, vec![], vec![min.to_bits(), max.to_bits()], vec![], None, None),

        Op::Maximum => (100, vec![], vec![], vec![], None, None),
        Op::Minimum => (101, vec![], vec![], vec![], None, None),

        // Phase 7.6 step 2: registry-extended fused ops. CSE folds two
        // Op::Fused nodes with identical (id, params) to one. Encoding:
        // base tag 200; ints = [id.0, params.tag, ...params.ints]; bits
        // = params.bits. Mirrors the FusedOpParamsKey shape from
        // crate::registry so the encoding tracks any future param
        // variants without rewriting this arm.
        Op::Fused(fid, fparams) => {
            let pk = fparams.key();
            let mut ints: Vec<i64> = Vec::with_capacity(2 + pk.ints.len());
            ints.push(fid.0 as i64);
            ints.push(pk.tag as i64);
            ints.extend_from_slice(&pk.ints);
            (200, ints, pk.bits, vec![], None, None)
        }

        // Indexing and anything else we haven't explicitly listed:
        // fall back to a unique tag that includes a structural
        // discriminant. These ops rarely appear more than once with
        // identical inputs, so we just mark them non-CSE-able by
        // returning None. Conservative; safe.
        _ => return None,
    };
    Some(OpKey { tag, ints, bits, dims, shape, dtype })
}

fn dtype_key(dt: DType) -> u32 {
    // Cheap injection: the Debug form is stable.
    // For tiny enums this compiles to a jump table.
    format!("{dt:?}").as_bytes().iter().fold(0u32, |acc, b| acc.wrapping_mul(31).wrapping_add(*b as u32))
}

fn is_commutative(op: &Op) -> bool {
    matches!(op, Op::Add | Op::Mul | Op::Maximum | Op::Minimum)
}

/// Run CSE + algebraic simplification on the graph reachable from
/// `roots`. Returns the (possibly-rewritten) roots the caller should
/// use afterward.
///
/// The pass runs to a fixed point on a single topological walk: each
/// node is visited once, canonicalized (inputs remapped, constants
/// folded where trivial), then either matched to an existing canonical
/// node (CSE) or appended as fresh. No multi-pass iteration — the
/// simplification rules are chosen so one forward pass suffices.
pub fn optimize(graph: &SharedGraph, roots: &[NodeId]) -> Vec<NodeId> {
    let order = {
        let g = graph.read().unwrap();
        topo_order_multi(&g, roots)
    };

    let mut g = graph.write().unwrap();
    let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
    let mut cse: HashMap<(OpKey, Vec<NodeId>), NodeId> = HashMap::new();

    for id in order {
        let (op, inputs, shape, dtype) = {
            let node = g.node(id);
            (node.op.clone(), node.inputs.clone(), node.shape.clone(), node.dtype)
        };
        let mapped_inputs: Vec<NodeId> = inputs
            .iter()
            .map(|input_id| *remap.get(input_id).unwrap_or(input_id))
            .collect();
        let inputs_unchanged = mapped_inputs == inputs;

        // 1. Algebraic simplifications that produce an alias (no new
        //    node). If a rule fires, we remap this id to an existing
        //    node and skip CSE for it entirely.
        if let Some(aliased) = try_simplify(&op, &mapped_inputs) {
            remap.insert(id, aliased);
            continue;
        }

        // 2. CSE: if a structurally-identical canonical node already
        //    exists, reuse it. Commutative ops use sorted inputs as
        //    the key so `a+b` and `b+a` match. If no match exists and
        //    nothing about this node needs rewriting (inputs unchanged,
        //    no simplification fired), keep the original node in place
        //    to avoid polluting the arena with identical copies.
        if let Some(key) = op_key(&op) {
            let key_inputs = if is_commutative(&op) {
                let mut v = mapped_inputs.clone();
                v.sort();
                v
            } else {
                mapped_inputs.clone()
            };
            let full_key = (key, key_inputs);
            if let Some(&existing) = cse.get(&full_key) {
                remap.insert(id, existing);
                continue;
            }
            let canonical_id = if inputs_unchanged {
                id
            } else {
                g.push(Node {
                    op: op.clone(),
                    inputs: mapped_inputs.clone(),
                    shape: shape.clone(),
                    dtype,
                })
            };
            cse.insert(full_key, canonical_id);
            remap.insert(id, canonical_id);
        } else {
            // Const or other non-CSE-able op. Keep the original if
            // unchanged; otherwise append a rewritten copy (Const
            // never has inputs, so this branch effectively keeps
            // Const originals).
            let canonical_id = if inputs_unchanged {
                id
            } else {
                g.push(Node {
                    op: op.clone(),
                    inputs: mapped_inputs,
                    shape,
                    dtype,
                })
            };
            remap.insert(id, canonical_id);
        }
    }

    roots.iter().map(|r| *remap.get(r).unwrap_or(r)).collect()
}

/// Insert `Op::Copy` nodes so every op's inputs live on the same
/// device as the op itself. Returns the rewritten roots.
///
/// This is the Phase-3.5 pass that lifts the Router from "explicit
/// copies only" to "auto-insert where needed."
///
/// ## Device inference rules
///
/// For each node N, walked in topological order:
///
/// 1. **Target device** = first match wins:
///    - If N has a placement hint via `graph.set_placement(n, d)`, use `d`.
///    - Else, inherit from N's first input's inferred device.
///    - Else (N is a Const or the first input has no inferred device
///      either), the target is `None` — N is "placeless" and will be
///      placed by its consumer's demands.
///
/// 2. **Input reconciliation**: for each input I of N, if I's inferred
///    device differs from N's target device, insert an
///    `Op::Copy { target }` node and redirect N's input edge to the
///    new Copy.
///
/// ## What it doesn't do
///
/// - **Backward ops:** the pass currently treats all ops the same
///   way — no special reasoning about what device a backward node
///   should run on. For pure inference this is fine; for training
///   through the Router we'd want a reverse-mode pass that mirrors
///   forward placement. Tracked as a follow-up TODO.
/// - **Const placement lowering:** if a Const flows into a single
///   consumer on device D, we emit `Copy(Const, D)` instead of
///   tagging the Const itself as on-device-D. A future cost-lowering
///   pass can fuse the two — tracked as a TODO alongside Phase 4's
///   scheduler.
/// - **Redundant copy elision:** `Copy(X, A) -> Copy(_, A)` when X is
///   already on A is dropped via the idempotent check in step 2, but
///   there's no pass that merges `Copy(X, A) -> Copy(_, B) -> Copy(_, A)`
///   (cross-device round-trips that cancel). Also future.
pub fn insert_copies(graph: &SharedGraph, roots: &[NodeId]) -> Vec<NodeId> {
    let order = {
        let g = graph.read().unwrap();
        topo_order_multi(&g, roots)
    };

    // Inferred output device per node. `None` = placeless (e.g., an
    // unplaced Const). Read on the input side to decide whether a
    // Copy is needed.
    let mut inferred: HashMap<NodeId, Option<DeviceLocation>> = HashMap::new();
    // Rewritten (old → new) node ID map. A node gets rewritten when
    // any of its inputs needed a Copy interposed.
    let mut remap: HashMap<NodeId, NodeId> = HashMap::new();

    let mut g = graph.write().unwrap();

    for id in order {
        // Snapshot the node — all subsequent reads need the old id's
        // metadata, not whatever we're about to rewrite.
        let (op, inputs, shape, dtype) = {
            let node = g.node(id);
            (node.op.clone(), node.inputs.clone(), node.shape.clone(), node.dtype)
        };
        let placement_hint = g.placement(id);

        // Target device for this node: explicit hint > first input's
        // inferred device > None (placeless).
        let target_device: Option<DeviceLocation> = match placement_hint {
            Some(d) => Some(d),
            None => inputs.first()
                .and_then(|i| {
                    let mapped = *remap.get(i).unwrap_or(i);
                    inferred.get(&mapped).copied().flatten()
                }),
        };

        // Walk inputs, inserting Copy where needed.
        let mut new_inputs: Vec<NodeId> = Vec::with_capacity(inputs.len());
        let mut any_changed = false;
        for input_id in &inputs {
            let mapped_in = *remap.get(input_id).unwrap_or(input_id);
            let in_device = inferred.get(&mapped_in).copied().flatten();

            let needs_copy = match (in_device, target_device) {
                // Both known and disagree → copy.
                (Some(src), Some(tgt)) if src != tgt => true,
                // Input is placeless and we have a target → copy
                // (Const gets Copy'd onto the target device; the
                // future cost-lowering pass can fuse this).
                (None, Some(_)) => true,
                // Everything else (match, or no target) → no copy.
                _ => false,
            };

            if needs_copy {
                let tgt = target_device.expect("needs_copy implies target is Some");
                // A Copy preserves the INPUT's shape/dtype, not the
                // outer consumer's. Read them from the source node.
                let (in_shape, in_dtype) = {
                    let n = g.node(mapped_in);
                    (n.shape.clone(), n.dtype)
                };
                let copy_id = g.push(Node {
                    op: Op::Copy { target: tgt },
                    inputs: vec![mapped_in],
                    shape: in_shape,
                    dtype: in_dtype,
                });
                inferred.insert(copy_id, Some(tgt));
                new_inputs.push(copy_id);
                any_changed = true;
            } else {
                if mapped_in != *input_id { any_changed = true; }
                new_inputs.push(mapped_in);
            }
        }

        // If nothing changed and this isn't a node whose inference
        // output device we need to record fresh, keep the original.
        let canonical_id = if any_changed {
            let new_id = g.push(Node {
                op: op.clone(),
                inputs: new_inputs,
                shape,
                dtype,
            });
            // Carry placement hint to the rewritten node so downstream
            // consumers see the same target.
            if let Some(d) = placement_hint {
                g.set_placement(new_id, d);
            }
            new_id
        } else {
            id
        };

        // Record this node's inferred output device:
        //   - Op::Copy / Op::Move: output is the transfer target.
        //   - Else if we have a target_device, that's the output
        //     device (it matches all inputs post-reconciliation).
        //   - Else None (placeless forward).
        let out_device = match &op {
            Op::Copy { target } | Op::Move { target } => Some(*target),
            _ => target_device,
        };
        inferred.insert(canonical_id, out_device);
        remap.insert(id, canonical_id);
    }

    roots.iter().map(|r| *remap.get(r).unwrap_or(r)).collect()
}

/// Lower `Const` placements: if every consumer of an unplaced Const
/// has the same placement hint, tag the Const with that placement.
///
/// This is a pre-pass for [`insert_copies`]. Without it, a model's
/// weight Consts (unplaced at graph-build time) flow into ops tagged
/// for a specific device; insert_copies then emits a Copy for each
/// such Const every forward. Lowering the Const's placement directly
/// tells the Router to upload it straight to the target device, and
/// insert_copies skips the Copy.
///
/// Conservative: a Const whose consumers disagree on device stays
/// unplaced. A future replication pass could clone the Const to each
/// target, but that needs to weigh replication cost against transfer
/// cost — scheduler territory (Phase 4).
///
/// Returns the number of Const placements set. Mutates `graph`
/// placements in place.
pub fn lower_const_placement(graph: &SharedGraph, roots: &[NodeId]) -> usize {
    // Reverse edges: for each node, list its consumers. Only walks
    // nodes reachable from `roots`, so unused Consts don't waste work.
    let order = {
        let g = graph.read().unwrap();
        topo_order_multi(&g, roots)
    };

    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    {
        let g = graph.read().unwrap();
        for &nid in &order {
            let node = g.node(nid);
            for &input in &node.inputs {
                consumers.entry(input).or_default().push(nid);
            }
        }
    }

    let mut lowered = 0;
    let mut g = graph.write().unwrap();
    for &nid in &order {
        // Only consider Const nodes without an explicit placement.
        let is_const = matches!(g.node(nid).op, Op::Const);
        if !is_const || g.placement(nid).is_some() {
            continue;
        }
        let Some(cs) = consumers.get(&nid) else {
            continue; // Const is a root — no consumers to infer from.
        };
        // Walk consumers; collect their placements. If all are
        // Some and agree, adopt that device.
        let mut target: Option<DeviceLocation> = None;
        let mut unanimous = true;
        for &c in cs {
            match g.placement(c) {
                Some(d) => match target {
                    None => target = Some(d),
                    Some(prev) if prev == d => {}
                    Some(_) => { unanimous = false; break; }
                },
                None => { unanimous = false; break; }
            }
        }
        if unanimous {
            if let Some(d) = target {
                g.set_placement(nid, d);
                lowered += 1;
            }
        }
    }
    lowered
}

/// Try to produce an aliasing simplification: "this op is equivalent
/// to one of its inputs." Returns the aliased NodeId if so.
fn try_simplify(op: &Op, inputs: &[NodeId]) -> Option<NodeId> {
    match op {
        // AddScalar(0) and MulScalar(1) are no-ops. Route consumers
        // straight to the input and drop the op.
        Op::AddScalar(c) if *c == 0.0 => Some(inputs[0]),
        Op::MulScalar(c) if *c == 1.0 => Some(inputs[0]),
        // PowI(1) is a no-op. PowI(0) would be "1" but that needs a
        // new const node with the right shape — skip here, let the
        // executor handle or another rule add it.
        Op::PowI(1) => Some(inputs[0]),
        _ => None,
    }
}

// ---- Ordering analysis for destructive ops ---------------------------------
//
// A destructive op (one whose `Op::destructive_input()` is `Some(i)`)
// invalidates its i-th input when it runs. For correctness, every
// non-destructive reader of that input must complete BEFORE the
// destructive op runs. The graph's data-flow edges don't express this
// — they'd let a backend schedule the destructive op any time after
// its input is produced, including AHEAD of sibling readers.
//
// `derive_ordering` analyzes the graph and returns a map of
// destructive-op → sibling-readers. The [`execution_plan`] below
// consumes this map alongside the data graph in a Kahn's-algorithm
// walk to produce an execution order that respects both constraint
// kinds.

/// Derived ordering edges beyond the data-flow graph. Each entry
/// `(nid, deps)` means `nid` must run after every node in `deps`.
///
/// Produced by [`derive_ordering`] from destructive-op metadata
/// ([`Op::destructive_input`]). Consumed by [`execution_plan`].
#[derive(Debug, Clone, Default)]
pub struct OrderingEdges(pub HashMap<NodeId, Vec<NodeId>>);

impl OrderingEdges {
    pub fn new() -> Self { Self(HashMap::new()) }

    /// True if the map is empty — no destructive ops in the analyzed
    /// subgraph, so the execution plan is just the plain topo order.
    pub fn is_empty(&self) -> bool { self.0.is_empty() }

    /// Get the set of must-run-before nodes for `nid`, if any.
    pub fn deps_of(&self, nid: NodeId) -> &[NodeId] {
        self.0.get(&nid).map(|v| v.as_slice()).unwrap_or(&[])
    }
}

/// Fuse `MatMul → Add(rank-1 bias)` patterns into `Op::FusedLinear`
/// (Phase 6d Track 3). Walks the graph; for each `Add` whose LHS is
/// a `MatMul` and whose RHS is a rank-1 bias whose length equals the
/// matmul output's last dim, emits a fresh `FusedLinear` node and
/// remaps consumers of the `Add` to it.
///
/// Conservative: only fires when the `Add` is the sole consumer of
/// the `MatMul`. Otherwise we'd be creating a duplicate matmul
/// computation. CSE doesn't help here because `MatMul` and
/// `MatMul-inside-FusedLinear` aren't structurally equal at the IR
/// level — backends with truly fused kernels are the ones that
/// benefit, so we'd rather skip the fusion than waste the work.
///
/// Returns the count of fusions applied.
pub fn fuse_linear(graph: &SharedGraph, roots: &[NodeId]) -> usize {
    let order = {
        let g = graph.read().unwrap();
        topo_order_multi(&g, roots)
    };
    // Count consumers of each node (so we can guard "single consumer of matmul").
    let mut consumer_count: HashMap<NodeId, usize> = HashMap::new();
    {
        let g = graph.read().unwrap();
        for &nid in &order {
            for &input in &g.node(nid).inputs {
                *consumer_count.entry(input).or_insert(0) += 1;
            }
        }
        // Also count root references — a root is implicitly a consumer.
        for &r in roots {
            *consumer_count.entry(r).or_insert(0) += 1;
        }
    }

    let mut g = graph.write().unwrap();
    let mut remap: HashMap<NodeId, NodeId> = HashMap::new();
    let mut fused = 0usize;

    for nid in order {
        // Apply already-known remappings to inputs.
        let (op, inputs, shape, dtype) = {
            let n = g.node(nid);
            (n.op.clone(), n.inputs.clone(), n.shape.clone(), n.dtype)
        };
        let mapped: Vec<NodeId> = inputs.iter().map(|i| *remap.get(i).unwrap_or(i)).collect();
        // Pattern: Op::Add { inputs[0]=matmul_node, inputs[1]=rank-1 bias }.
        if !matches!(op, Op::Add) || mapped.len() != 2 {
            continue;
        }
        let lhs = mapped[0];
        let rhs = mapped[1];
        // LHS must be a MatMul.
        let lhs_op = g.node(lhs).op.clone();
        if !matches!(lhs_op, Op::MatMul) {
            continue;
        }
        // LHS matmul must have only THIS Add as a consumer (otherwise
        // fusing would duplicate the matmul computation).
        // Note: consumer_count counts pre-remapping references; remap
        // only happens for skipped nodes here so it's still valid.
        if consumer_count.get(&lhs).copied().unwrap_or(0) != 1 {
            continue;
        }
        // RHS must be a BroadcastTo of a rank-1 bias whose length
        // equals the matmul output's last dim. The build-time `Add`
        // requires same-shape inputs, so user code typically does:
        //     bias[N].broadcast_to([..., M, N]).add(matmul_out)
        // Walk through that BroadcastTo to find the rank-1 source.
        let mm_dims = g.node(lhs).shape.dims().to_vec();
        if mm_dims.is_empty() { continue; }
        let last_dim = mm_dims[mm_dims.len() - 1];
        let rhs_node = g.node(rhs);
        let bias_src_id = if matches!(rhs_node.op, Op::BroadcastTo(_)) && rhs_node.inputs.len() == 1 {
            *remap.get(&rhs_node.inputs[0]).unwrap_or(&rhs_node.inputs[0])
        } else {
            // Bias broadcast may also have been pre-shaped; allow rank-1
            // direct (rare with build-time shape checks but cheap to
            // recognize).
            rhs
        };
        let bias_dims = g.node(bias_src_id).shape.dims().to_vec();
        if bias_dims.len() != 1 || bias_dims[0] != last_dim {
            continue;
        }
        // Pull the matmul's a, b inputs (apply remap to those too).
        let mm_inputs = g.node(lhs).inputs.clone();
        if mm_inputs.len() != 2 {
            continue;
        }
        let a = *remap.get(&mm_inputs[0]).unwrap_or(&mm_inputs[0]);
        let b = *remap.get(&mm_inputs[1]).unwrap_or(&mm_inputs[1]);
        let new_id = g.push(Node {
            op: Op::FusedLinear,
            // FusedLinear takes the *original* rank-1 bias, not the
            // BroadcastTo'd one — the executor's arm broadcasts it
            // internally to the matmul output shape.
            inputs: vec![a, b, bias_src_id],
            shape,
            dtype,
        });
        remap.insert(nid, new_id);
        fused += 1;
    }

    // Apply remap by rewriting any consumer that still references an
    // un-fused Add. Using the existing `rewrite_input` helper.
    if !remap.is_empty() {
        // Collect all node ids; iterate to update inputs that point at
        // remapped nodes. We need the mutable borrow; clone the set of
        // nodes to iterate without borrow conflicts.
        let n_nodes = g.nodes.len();
        for nid in 0..n_nodes {
            let node = &mut g.nodes[nid];
            for input in node.inputs.iter_mut() {
                if let Some(&new) = remap.get(input) {
                    *input = new;
                }
            }
        }
    }
    fused
}

/// Derive ordering edges for every destructive op reachable from
/// `roots`. Result: `nid` → list of other readers of `nid`'s
/// destroyed input.
///
/// O(V + E). Does not mutate the graph.
pub fn derive_ordering(graph: &crate::Graph, roots: &[NodeId]) -> OrderingEdges {
    let order = topo_order_multi(graph, roots);

    // Consumer index: tensor NodeId → list of consumers of that tensor.
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for &nid in &order {
        for &input in &graph.node(nid).inputs {
            consumers.entry(input).or_default().push(nid);
        }
    }

    let mut ordering: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for &nid in &order {
        let node = graph.node(nid);
        let Some(d_idx) = node.op.destructive_input() else { continue };
        if d_idx >= node.inputs.len() { continue }
        let destroyed = node.inputs[d_idx];
        let Some(readers) = consumers.get(&destroyed) else { continue };
        for &reader in readers {
            if reader != nid {
                ordering.entry(nid).or_default().push(reader);
            }
        }
    }
    OrderingEdges(ordering)
}

/// Build an execution plan that respects both data-flow edges (via
/// [`topo_order_multi`]) and ordering edges (via [`derive_ordering`]).
/// Returns a `Vec<NodeId>` in an order the executor can walk linearly,
/// evaluating each node's dependencies before it.
///
/// **Fast path:** when the graph has no destructive ops, this returns
/// the same order `topo_order_multi` does — no extra cost beyond the
/// analysis pass.
///
/// **Cycle detection:** if the combined (data + ordering) graph has a
/// cycle — the residency rule emitted a destructive op that
/// transitively depends on itself — this panics with a clear error
/// message. That panic indicates a bug in the rule that emitted the
/// destructive op, not user code.
///
/// **Stability:** among nodes that are concurrently ready, this
/// picks the one whose `topo_order_multi` index is smallest. Result:
/// when there are no ordering edges, the output matches
/// `topo_order_multi` exactly.
pub fn execution_plan(graph: &crate::Graph, roots: &[NodeId]) -> Vec<NodeId> {
    let base_order = topo_order_multi(graph, roots);
    let ordering = derive_ordering(graph, roots);
    if ordering.is_empty() {
        return base_order;
    }

    // Position of each node in base_order — used as the stable tiebreaker.
    let pos: HashMap<NodeId, usize> = base_order.iter().enumerate()
        .map(|(i, &n)| (n, i)).collect();
    let node_set: std::collections::HashSet<NodeId> =
        base_order.iter().copied().collect();

    // Build in-degree + reverse adjacency for Kahn's.
    let mut in_degree: HashMap<NodeId, usize> = HashMap::with_capacity(base_order.len());
    let mut forward: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for &nid in &base_order {
        let node = graph.node(nid);
        let mut d = 0usize;
        for &input in &node.inputs {
            if node_set.contains(&input) {
                d += 1;
                forward.entry(input).or_default().push(nid);
            }
        }
        for &dep in ordering.deps_of(nid) {
            if node_set.contains(&dep) {
                d += 1;
                forward.entry(dep).or_default().push(nid);
            }
        }
        in_degree.insert(nid, d);
    }

    // Stable ready-set keyed by base_order position. BTreeSet pops the
    // smallest → output matches topo order when ordering edges allow it.
    let mut ready: std::collections::BTreeSet<(usize, NodeId)> = base_order.iter()
        .copied()
        .filter(|n| in_degree[n] == 0)
        .map(|n| (pos[&n], n))
        .collect();

    let mut plan = Vec::with_capacity(base_order.len());
    while let Some(&(_, n)) = ready.iter().next() {
        ready.remove(&(pos[&n], n));
        plan.push(n);
        if let Some(succs) = forward.get(&n) {
            for &s in succs {
                let d = in_degree.get_mut(&s).expect("in_degree covers all nodes");
                *d -= 1;
                if *d == 0 {
                    ready.insert((pos[&s], s));
                }
            }
        }
    }

    if plan.len() != base_order.len() {
        panic!(
            "execution_plan: cycle in ordering edges (plan={}, base={}) — \
             a destructive op transitively depends on itself. This is a \
             bug in whatever rule emitted the destructive op.",
            plan.len(), base_order.len(),
        );
    }
    plan
}

// ---- Evict + reload graph surgery -----------------------------------------

/// Insert an evict-chain around `candidate` to free its device storage
/// during a gap between uses, with automatic reload before the post-gap
/// consumers. Returns `(cpu_copy_id, release_id, reload_id)`.
///
/// ## What it does
///
/// Inserts three new nodes:
/// 1. `cpu_copy = Op::Copy { target: Cpu }` reading `candidate` — stages
///    the data to host memory.
/// 2. `release = Op::Release` reading `candidate` — destructive; frees
///    `candidate`'s device-resident storage once it runs. The
///    [`derive_ordering`] pass pins this to run AFTER every
///    non-destructive reader of `candidate` (including `cpu_copy` and
///    whatever pre-gap consumers the caller left untouched).
/// 3. `reload = Op::Copy { target: src_device }` reading `cpu_copy` —
///    restages the data to the device right before the post-gap
///    consumers need it.
///
/// Then rewrites every `post_gap_consumer`'s input edge from `candidate`
/// to `reload`. Pre-gap consumers keep reading `candidate` directly.
///
/// ## Caller's responsibility
///
/// - `candidate` is a NodeId currently in the graph whose device
///   residency is `src_device`.
/// - `post_gap_consumers` are NodeIds currently in the graph that each
///   have `candidate` in their `inputs`. Typically these come from the
///   residency analyzer's gap-positioning logic.
/// - The caller will update the `Placement` map afterward to place
///   `cpu_copy` on Cpu and `reload` on `src_device`.
pub fn insert_evict_reload(
    graph: &SharedGraph,
    candidate: NodeId,
    src_device: DeviceLocation,
    post_gap_consumers: &[NodeId],
) -> (NodeId, NodeId, NodeId) {
    let mut g = graph.write().unwrap();
    let (shape, dtype) = {
        let n = g.node(candidate);
        (n.shape.clone(), n.dtype)
    };

    let cpu_copy_id = g.push(Node {
        op:     Op::Copy { target: DeviceLocation::Cpu },
        inputs: vec![candidate],
        shape:  shape.clone(),
        dtype,
    });
    let release_id = g.push(Node {
        op:     Op::Release,
        inputs: vec![candidate],
        shape:  Shape::from_dims(&[0]),
        dtype:  DType::F32,
    });
    // Release's zero-element output has no consumer, so it wouldn't be
    // reachable from the user's roots. Register it as a side-effect
    // root so the executor still walks + runs it (freeing `candidate`'s
    // device memory).
    g.add_side_effect_root(release_id);

    let reload_id = g.push(Node {
        op:     Op::Copy { target: src_device },
        inputs: vec![cpu_copy_id],
        shape,
        dtype,
    });

    // Rewrite each post-gap consumer's `candidate` input to `reload_id`.
    // Pre-gap consumers (not in this list) continue reading `candidate`
    // directly, which is why derive_ordering needs `release_id` to run
    // after them.
    for &consumer in post_gap_consumers {
        g.rewrite_input(consumer, candidate, reload_id);
    }

    (cpu_copy_id, release_id, reload_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;
    use fuel_core_types::{DeviceLocation, Shape};
    use std::sync::Arc;

    /// Phase 7.5 G2: tests need a real device for slot-populating
    /// constructors. Singleton CpuBackendDevice via OnceLock.
    fn cpu_dev() -> &'static Arc<dyn fuel_core_types::DynBackendDevice> {
        static D: std::sync::OnceLock<Arc<dyn fuel_core_types::DynBackendDevice>>
            = std::sync::OnceLock::new();
        D.get_or_init(|| Arc::new(fuel_cpu_backend::dyn_impl::CpuBackendDevice))
    }

    fn make_scalar_graph() -> (SharedGraph, Tensor) {
        let t = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
        (t.graph().clone(), t)
    }

    fn count_copy_nodes(graph: &SharedGraph) -> usize {
        let g = graph.read().unwrap();
        (0..g.len()).filter(|i| matches!(g.node(NodeId(*i)).op, Op::Copy { .. })).count()
    }

    #[test]
    fn insert_copies_no_placement_no_copies() {
        // Graph with no placement hints: pass should be a no-op, no
        // Copies inserted, roots unchanged.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.add(&b);
        let graph = c.graph().clone();
        let before = count_copy_nodes(&graph);
        let new_roots = insert_copies(&graph, &[c.id()]);
        assert_eq!(new_roots, vec![c.id()]);
        assert_eq!(count_copy_nodes(&graph), before);
    }

    #[test]
    fn insert_copies_tagged_node_pulls_inputs_to_its_device() {
        // Const a, Const b, Add(a, b) placed on Vulkan.
        // Expected: two Copy(a, Vulkan) and Copy(b, Vulkan) inserted,
        // Add's inputs rewritten to reference the Copies.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.add(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let graph = c.graph().clone();

        let before_copies = count_copy_nodes(&graph);
        let new_roots = insert_copies(&graph, &[c.id()]);

        // Should be exactly 2 new Copies (one per input).
        assert_eq!(count_copy_nodes(&graph) - before_copies, 2);

        // Rewritten Add should reference two Copy nodes.
        let g = graph.read().unwrap();
        let new_add = g.node(new_roots[0]);
        assert_eq!(new_add.inputs.len(), 2);
        for input in &new_add.inputs {
            let node = g.node(*input);
            assert!(matches!(
                node.op,
                Op::Copy { target: DeviceLocation::Vulkan { gpu_id: 0 } }
            ));
        }
    }

    #[test]
    fn insert_copies_matching_device_no_copies_inserted() {
        // Const a and Add both placed on Vulkan. Const flows into
        // Add, both want Vulkan — no Copy needed.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev())
            .on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]))
            .on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let c = a.add(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let graph = c.graph().clone();

        let before = count_copy_nodes(&graph);
        insert_copies(&graph, &[c.id()]);
        assert_eq!(count_copy_nodes(&graph), before);
    }

    #[test]
    fn insert_copies_handles_backward_graph() {
        // Forward: y = sum(x * x) with x placed on Vulkan.
        // Call backward — this appends gradient nodes to the same graph.
        // Then run insert_copies on BOTH the forward root and the
        // gradient root. Every backward op should end up on Vulkan
        // (inherited from its inputs, which trace back to Vulkan-placed x).
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev())
            .on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let sq = x.mul(&x).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let y = sq.sum_all().on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let graph = y.graph().clone();

        let grads = y.backward();
        let grad_x = grads.get(&x).expect("dL/dx should exist");

        // Run insert_copies on both roots.
        let before_copies = count_copy_nodes(&graph);
        let _new = insert_copies(&graph, &[y.id(), grad_x.id()]);
        let after_copies = count_copy_nodes(&graph);

        // With everything on Vulkan, no Copies should be needed —
        // this verifies the pass doesn't spuriously insert Copies into
        // the backward graph when placements are consistent.
        assert_eq!(
            after_copies, before_copies,
            "insert_copies should leave a uniformly-Vulkan forward+backward graph alone"
        );
    }

    #[test]
    fn insert_copies_backward_inherits_forward_device() {
        // Forward: y = sum(x * x) with NO explicit placement.
        // Compute backward. Then place JUST the forward root on Vulkan
        // and re-run insert_copies. The Copies that get inserted should
        // pull x to Vulkan; the backward path should follow suit when
        // we ask for both roots.
        let x = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
        let sq = x.mul(&x);
        let y = sq.sum_all().on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let graph = y.graph().clone();

        let grads = y.backward();
        let grad_x = grads.get(&x).expect("dL/dx should exist");

        // Before: only y is tagged. The backward graph has no tags.
        // After insert_copies([y, grad_x]): y pulls its inputs (sq)
        // toward Vulkan → sq pulls x → Copies inserted.
        // Backward inherits device from its inputs transitively.
        let new_roots = insert_copies(&graph, &[y.id(), grad_x.id()]);
        assert_eq!(new_roots.len(), 2);
        // At least one Copy should have been inserted (the unplaced
        // forward consts need to get to Vulkan).
        assert!(
            count_copy_nodes(&graph) > 0,
            "expected Copies to be inserted for unplaced Const inputs"
        );
    }

    #[test]
    fn lower_const_placement_single_vulkan_consumer() {
        // Const a → Add(a, b) placed on Vulkan. lower_const_placement
        // should tag a with Vulkan since Add is its only consumer.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.add(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let graph = c.graph().clone();

        let lowered = lower_const_placement(&graph, &[c.id()]);
        assert_eq!(lowered, 2); // both a and b tagged
        assert_eq!(graph.read().unwrap().placement(a.id()), Some(DeviceLocation::Vulkan { gpu_id: 0 }));
        assert_eq!(graph.read().unwrap().placement(b.id()), Some(DeviceLocation::Vulkan { gpu_id: 0 }));

        // After lowering, insert_copies should emit NO Copies (the
        // Consts are now on the target device).
        let before_copies = count_copy_nodes(&graph);
        insert_copies(&graph, &[c.id()]);
        assert_eq!(count_copy_nodes(&graph), before_copies);
    }

    #[test]
    fn lower_const_placement_consumers_disagree_stays_unplaced() {
        // Const a flows into two consumers on different devices.
        // Without replication support, lowering has to leave a unplaced.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let cpu_sum = a.add(&b).on_device(DeviceLocation::Cpu);
        let vulkan_sum = a.add(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let graph = a.graph().clone();

        lower_const_placement(&graph, &[cpu_sum.id(), vulkan_sum.id()]);
        assert_eq!(graph.read().unwrap().placement(a.id()), None, "const with disagreeing consumers stays unplaced");
        assert_eq!(graph.read().unwrap().placement(b.id()), None);
    }

    #[test]
    fn lower_const_placement_skips_already_placed() {
        // An explicitly-placed Const should not be overridden.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev())
            .on_device(DeviceLocation::Cpu);
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.add(&b).on_device(DeviceLocation::Vulkan { gpu_id: 0 });
        let graph = c.graph().clone();

        lower_const_placement(&graph, &[c.id()]);
        // a keeps its explicit Cpu placement even though its consumer is Vulkan.
        assert_eq!(graph.read().unwrap().placement(a.id()), Some(DeviceLocation::Cpu));
        // b had no hint; it gets lowered to Vulkan.
        assert_eq!(graph.read().unwrap().placement(b.id()), Some(DeviceLocation::Vulkan { gpu_id: 0 }));
    }

    #[test]
    fn insert_copies_idempotent() {
        let a = Tensor::from_f32(vec![1.0], Shape::from_dims(&[1]), cpu_dev());
        let b = a.const_f32_like(vec![2.0], Shape::from_dims(&[1]));
        let c = a.add(&b).on_device(DeviceLocation::Cpu);
        let graph = c.graph().clone();

        let roots1 = insert_copies(&graph, &[c.id()]);
        let after_first = count_copy_nodes(&graph);
        let _roots2 = insert_copies(&graph, &roots1);
        assert_eq!(
            count_copy_nodes(&graph), after_first,
            "insert_copies should be idempotent on already-reconciled graphs"
        );
    }

    #[test]
    fn cse_folds_identical_add() {
        let (graph, a) = make_scalar_graph();
        let b = a.add(&a);
        let c = a.add(&a);
        let pre_len = graph.read().unwrap().len();
        let new_roots = optimize(&graph, &[b.id(), c.id()]);
        assert_eq!(new_roots[0], new_roots[1], "CSE should map both to same node");
        assert!(graph.read().unwrap().len() >= pre_len);
    }

    #[test]
    fn cse_folds_commutative() {
        // Build two tensors inside the same graph so `a + b` and
        // `b + a` share NodeIds for the inputs. Use add_scalar on `a`
        // as a simple way to get a second tensor handle sharing a's graph.
        let (_graph, a) = make_scalar_graph();
        let b = a.add_scalar(5.0);
        let ab = a.add(&b);
        let ba = b.add(&a);
        let graph = a.graph().clone();
        let new_roots = optimize(&graph, &[ab.id(), ba.id()]);
        assert_eq!(
            new_roots[0], new_roots[1],
            "commutative CSE should fold a+b and b+a to one node"
        );
    }

    #[test]
    fn simplifies_add_scalar_zero() {
        let (graph, a) = make_scalar_graph();
        let b = a.add_scalar(0.0);
        let new_roots = optimize(&graph, &[b.id()]);
        assert_eq!(new_roots[0], a.id(), "AddScalar(0) should alias to input");
    }

    #[test]
    fn simplifies_mul_scalar_one() {
        let (graph, a) = make_scalar_graph();
        let b = a.mul_scalar(1.0);
        let new_roots = optimize(&graph, &[b.id()]);
        assert_eq!(new_roots[0], a.id(), "MulScalar(1) should alias to input");
    }

    #[test]
    fn cse_does_not_fold_distinct_ops() {
        // Add and Mul on the same inputs are not equivalent.
        let (graph, a) = make_scalar_graph();
        let sum = a.add(&a);
        let prod = a.mul(&a);
        let new_roots = optimize(&graph, &[sum.id(), prod.id()]);
        assert_ne!(new_roots[0], new_roots[1], "Add and Mul must stay distinct");
    }

    #[test]
    fn cse_does_not_fold_distinct_scalars() {
        let (graph, a) = make_scalar_graph();
        let b = a.add_scalar(1.0);
        let c = a.add_scalar(2.0);
        let new_roots = optimize(&graph, &[b.id(), c.id()]);
        assert_ne!(new_roots[0], new_roots[1], "AddScalar with different c must stay distinct");
    }

    #[test]
    fn cse_nested_chain_deduplicates() {
        // (a + a) + a  appearing twice should dedupe both subexpressions.
        let (graph, a) = make_scalar_graph();
        let p1 = a.add(&a);
        let p2 = a.add(&a); // duplicates p1
        let q1 = p1.add(&a);
        let q2 = p2.add(&a); // structurally identical to q1 via CSE
        let new_roots = optimize(&graph, &[q1.id(), q2.id()]);
        assert_eq!(new_roots[0], new_roots[1], "nested duplicates must fold");
    }

    // ---- derive_ordering + execution_plan -----------------------------------

    #[test]
    fn derive_ordering_empty_for_non_destructive_graph() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.add(&b);
        let ord = derive_ordering(&c.graph().read().unwrap(), &[c.id()]);
        assert!(ord.is_empty(), "graph without destructive ops → no ordering edges");
    }

    #[test]
    fn derive_ordering_release_must_run_after_sibling_readers() {
        // Graph:
        //   a       (producer)
        //   b = relu(a)   (non-destructive reader of a)
        //   r = release(a) (destructive reader of a)
        // Expected ordering: r must run after b.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.relu();
        let r = a.release();
        let ord = derive_ordering(&a.graph().read().unwrap(), &[b.id(), r.id()]);
        let deps = ord.deps_of(r.id());
        assert_eq!(deps.len(), 1, "release should have one ordering dep (the relu)");
        assert_eq!(deps[0], b.id(), "release must run after relu");
    }

    #[test]
    fn derive_ordering_release_of_multi_reader_input() {
        // a read by relu AND neg, then released. Both relu and neg
        // must precede the release.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.relu();
        let c = a.neg();
        let r = a.release();
        let ord = derive_ordering(&a.graph().read().unwrap(), &[b.id(), c.id(), r.id()]);
        let mut deps = ord.deps_of(r.id()).to_vec();
        deps.sort_by_key(|n| n.0);
        let mut expected = vec![b.id(), c.id()];
        expected.sort_by_key(|n| n.0);
        assert_eq!(deps, expected);
    }

    #[test]
    fn execution_plan_matches_topo_when_no_destructive_ops() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.add(&b);
        let graph = c.graph().read().unwrap();
        let plan = execution_plan(&graph, &[c.id()]);
        let topo = topo_order_multi(&graph, &[c.id()]);
        assert_eq!(plan, topo);
    }

    #[test]
    fn execution_plan_pins_release_after_sibling_reader() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.relu();
        let r = a.release();
        let plan = execution_plan(&a.graph().read().unwrap(), &[b.id(), r.id()]);
        let b_pos = plan.iter().position(|&n| n == b.id()).unwrap();
        let r_pos = plan.iter().position(|&n| n == r.id()).unwrap();
        assert!(b_pos < r_pos, "expected relu@{b_pos} to precede release@{r_pos}: {plan:?}");
    }

    #[test]
    fn execution_plan_handles_chain_of_destructive_ops() {
        // a -> relu -> b
        // a -> neg -> c
        // a -> release
        // b and c must come before release.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.relu();
        let c = a.neg();
        let r = a.release();
        let plan = execution_plan(&a.graph().read().unwrap(), &[b.id(), c.id(), r.id()]);
        let b_pos = plan.iter().position(|&n| n == b.id()).unwrap();
        let c_pos = plan.iter().position(|&n| n == c.id()).unwrap();
        let r_pos = plan.iter().position(|&n| n == r.id()).unwrap();
        assert!(b_pos < r_pos, "relu@{b_pos} must precede release@{r_pos}: {plan:?}");
        assert!(c_pos < r_pos, "neg@{c_pos} must precede release@{r_pos}: {plan:?}");
    }

    #[test]
    fn insert_evict_reload_creates_expected_chain() {
        // Graph:
        //   a         (producer, device=Cpu default)
        //   b = relu(a)   (pre-gap consumer — stays wired to a)
        //   c = neg(a)    (post-gap consumer — should be rewired to reload)
        // After insert_evict_reload(a, Cpu, &[c]):
        //   - 3 new nodes (cpu_copy, release, reload) appended
        //   - c's input list now has `reload_id` instead of `a.id()`
        //   - b's input list STILL has `a.id()` (unchanged)
        let a = Tensor::from_f32(vec![1.0_f32, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.relu();
        let c = a.neg();

        let (cpu_copy_id, release_id, reload_id) = insert_evict_reload(
            a.graph(), a.id(), DeviceLocation::Cpu, &[c.id()],
        );

        let g = a.graph().read().unwrap();
        // cpu_copy is an Op::Copy{Cpu} reading a
        match &g.node(cpu_copy_id).op {
            Op::Copy { target: DeviceLocation::Cpu } => {},
            other => panic!("expected Op::Copy{{Cpu}}, got {other:?}"),
        }
        assert_eq!(g.node(cpu_copy_id).inputs, vec![a.id()]);
        // release is Op::Release reading a
        assert!(matches!(g.node(release_id).op, Op::Release));
        assert_eq!(g.node(release_id).inputs, vec![a.id()]);
        // reload is an Op::Copy{Cpu} (src_device passed in) reading cpu_copy
        assert!(matches!(g.node(reload_id).op, Op::Copy { .. }));
        assert_eq!(g.node(reload_id).inputs, vec![cpu_copy_id]);
        // c's input was rewired
        assert_eq!(g.node(c.id()).inputs, vec![reload_id],
            "post-gap consumer should read from reload, not candidate directly");
        // b's input stays
        assert_eq!(g.node(b.id()).inputs, vec![a.id()],
            "pre-gap consumer should still read candidate directly");
    }

    #[test]
    fn execution_plan_respects_transitive_data_deps_after_release() {
        // Graph:
        //   a
        //   b = relu(a)
        //   r = release(a)  (destructive; runs after b)
        //   sum = sum_all(b) (data-dependent on b, not on r)
        // Plan must have: b before r, b before sum; b before both is enough.
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.relu();
        let sum = b.sum_all();
        let r = a.release();
        let plan = execution_plan(&a.graph().read().unwrap(), &[sum.id(), r.id()]);
        let b_pos = plan.iter().position(|&n| n == b.id()).unwrap();
        let sum_pos = plan.iter().position(|&n| n == sum.id()).unwrap();
        let r_pos = plan.iter().position(|&n| n == r.id()).unwrap();
        assert!(b_pos < sum_pos);
        assert!(b_pos < r_pos);
    }

    #[test]
    fn fuse_linear_collapses_matmul_plus_rank1_bias() {
        // Build [batch=1, m=2, k=3] @ [k=3, n=4] + bias[4].
        let a = crate::Tensor::from_f32(
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0],
            crate::Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let b = a.const_f32_like(
            (0..12).map(|i| (i as f32) * 0.1).collect::<Vec<f32>>(),
            crate::Shape::from_dims(&[3, 4]));
        let bias = a.const_f32_like(
            vec![0.5_f32, -0.5, 1.0, -1.0],
            crate::Shape::from_dims(&[4]));
        let mm = a.matmul(&b);
        let bias_b = bias.broadcast_to(crate::Shape::from_dims(&[2, 4]));
        let out = mm.add(&bias_b);
        // Note: real users would call broadcast_to first, then Add.
        // The fusion pass looks for `Add(MatMul, Const-shape-1-N)`
        // so we need to update it to also recognize the
        // BroadcastTo-then-Add pattern.
        let n_fused = fuse_linear(out.graph(), &[out.id()]);
        assert_eq!(n_fused, 1, "exactly one MatMul→Add(bias[N]) should fuse");

        // The fused node should now be reachable as the canonical
        // root after remap. Its op is FusedLinear with three inputs.
        let g = out.graph().read().unwrap();
        // Walk consumers of the original Add: any leftover Add should
        // be unreferenced; the new FusedLinear should be present.
        let any_fused = g.nodes.iter().any(|n| matches!(n.op, Op::FusedLinear));
        assert!(any_fused, "graph should contain a FusedLinear node");
    }

    #[test]
    fn fuse_linear_skips_when_matmul_has_other_consumers() {
        // If the matmul is consumed by both Add and something else,
        // fusing would duplicate the matmul. Pass should skip.
        let a = crate::Tensor::from_f32(
            vec![1.0_f32; 6],
            crate::Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let b = a.const_f32_like(
            vec![1.0_f32; 12],
            crate::Shape::from_dims(&[3, 4]));
        let bias = a.const_f32_like(
            vec![1.0_f32; 4],
            crate::Shape::from_dims(&[4]));
        let mm = a.matmul(&b);
        let bias_b = bias.broadcast_to(crate::Shape::from_dims(&[2, 4]));
        let with_bias = mm.add(&bias_b);
        let also_used = mm.relu();        // second consumer of mm
        // Both with_bias and also_used are roots.
        let n_fused = fuse_linear(a.graph(), &[with_bias.id(), also_used.id()]);
        assert_eq!(n_fused, 0, "MatMul has 2 consumers — fusion would duplicate work");
    }

    // ---- Rule registry: SoftmaxLastDim lower / fuse / round-trip ------------

    fn count_op_in_reachable<F: Fn(&Op) -> bool>(
        graph: &SharedGraph,
        roots: &[NodeId],
        pred: F,
    ) -> usize {
        let g = graph.read().unwrap();
        topo_order_multi(&g, roots)
            .into_iter()
            .filter(|id| pred(&g.node(*id).op))
            .count()
    }

    /// The lower rule, applied to a graph with one `Op::SoftmaxLastDim`,
    /// produces the canonical 7-node primitive subgraph reachable from
    /// the rewritten root, with auto-populated layout entries on the
    /// inserted BroadcastTo nodes.
    #[test]
    fn softmax_last_dim_lower_rule_produces_canonical_subgraph() {
        let x = Tensor::from_f32(
            vec![1.0_f32; 12],
            Shape::from_dims(&[2, 3, 2]),
            cpu_dev(),
        );
        let sm = x.softmax_last_dim();
        let graph = sm.graph().clone();

        let registry = RuleRegistry::lowering_only();
        let new_roots = registry.optimize_to_fixpoint(&graph, &[sm.id()]);
        assert_eq!(new_roots.len(), 1);
        let new_root = new_roots[0];

        // Reachable subgraph from the new root should contain exactly
        // the 7 lowered nodes (plus the `x` Const leaf — that's 8).
        // Op composition: 1 ReduceMaxTo, 2 BroadcastTo, 1 Sub, 1 Exp,
        // 1 ReduceSumTo, 1 Div, 1 Const = 8 reachable.
        let g = graph.read().unwrap();
        let reachable = topo_order_multi(&g, &[new_root]);
        assert_eq!(reachable.len(), 8, "lowered subgraph: 7 ops + 1 Const = 8 reachable nodes");
        drop(g);

        // Op-shape sanity checks.
        let n_reduce_max  = count_op_in_reachable(&graph, &[new_root], |op| matches!(op, Op::ReduceMaxTo(_)));
        let n_max_dim     = count_op_in_reachable(&graph, &[new_root], |op| matches!(op, Op::MaxDim(_)));
        let n_reshape     = count_op_in_reachable(&graph, &[new_root], |op| matches!(op, Op::Reshape(_)));
        let n_broadcast   = count_op_in_reachable(&graph, &[new_root], |op| matches!(op, Op::BroadcastTo(_)));
        let n_sub         = count_op_in_reachable(&graph, &[new_root], |op| matches!(op, Op::Sub));
        let n_exp         = count_op_in_reachable(&graph, &[new_root], |op| matches!(op, Op::Exp));
        let n_reduce_sum  = count_op_in_reachable(&graph, &[new_root], |op| matches!(op, Op::ReduceSumTo(_)));
        let n_div         = count_op_in_reachable(&graph, &[new_root], |op| matches!(op, Op::Div));
        // Phase 7.6 step 3: post-migration the builder emits
        // `Op::Fused(SOFTMAX_LAST_DIM, _)`. The lowering rule fires on
        // either form; the assertion checks both shapes are gone.
        let n_softmax     = count_op_in_reachable(&graph, &[new_root], |op| {
            matches!(op, Op::SoftmaxLastDim)
                || matches!(op, Op::Fused(fid, _) if *fid == FusedOps::SOFTMAX_LAST_DIM)
        });
        assert_eq!(n_reduce_max, 1);
        assert_eq!(n_max_dim, 0, "PR 3.5 max side uses ReduceMaxTo, not MaxDim+Reshape");
        assert_eq!(n_reshape, 0, "PR 3.5 max side drops the Reshape (ReduceMaxTo carries keepdim)");
        assert_eq!(n_broadcast, 2);
        assert_eq!(n_sub, 1);
        assert_eq!(n_exp, 1);
        assert_eq!(n_reduce_sum, 1);
        assert_eq!(n_div, 1);
        assert_eq!(n_softmax, 0, "no SoftmaxLastDim/Fused(SOFTMAX_LAST_DIM) should remain reachable post-lowering");

        // Layout side-table entries on the inserted BroadcastTo nodes
        // must be populated and have the expected broadcast strides.
        let g = graph.read().unwrap();
        for id in topo_order_multi(&g, &[new_root]) {
            if let Op::BroadcastTo(_) = g.node(id).op {
                assert!(
                    g.has_explicit_layout(id),
                    "lowered BroadcastTo at {id:?} missing explicit Layout entry — \
                     Graph::push auto-populate should have set it",
                );
                // The inserted BroadcastTo broadcasts shape [2,3,1] -> [2,3,2]
                // with last-dim stride 0.
                let l = g.layout(id);
                assert_eq!(l.shape().dims(), &[2, 3, 2]);
                assert_eq!(l.stride().last().copied(), Some(0),
                    "BroadcastTo from [2,3,1] to [2,3,2] should have stride 0 on last dim");
            }
        }
    }

    /// The fuse rule, applied to the canonical 7-node lowered subgraph,
    /// collapses it back to a single `Op::SoftmaxLastDim` reachable
    /// from the rewritten root.
    #[test]
    fn softmax_last_dim_fuse_rule_collapses_canonical_subgraph() {
        // Build the canonical subgraph by-hand — start from a Const
        // and apply the same op sequence the lower rule would emit.
        let x = Tensor::from_f32(
            vec![1.0_f32; 6],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let m = x.reduce_max_to(Shape::from_dims(&[2, 1]));
        let mb = m.broadcast_to(Shape::from_dims(&[2, 3]));
        let s = x.sub(&mb);
        let e = s.exp();
        let d = e.reduce_sum_to(Shape::from_dims(&[2, 1]));
        let db = d.broadcast_to(Shape::from_dims(&[2, 3]));
        let out = e.div(&db);
        let graph = out.graph().clone();

        // Phase 7.6 step 3: fuse rule comes from the registry entry.
        let registry = RuleRegistry::new()
            .with_rule(Box::new(FusionRule::from_entry(
                &crate::registry::softmax_last_dim::entry(),
            )));
        let new_roots = registry.optimize_to_fixpoint(&graph, &[out.id()]);
        assert_eq!(new_roots.len(), 1);
        let new_root = new_roots[0];

        // Reachable from the new root: 1 fused-softmax + 1 Const x = 2.
        let g = graph.read().unwrap();
        let reachable = topo_order_multi(&g, &[new_root]);
        assert_eq!(reachable.len(), 2,
            "fused subgraph: 1 Fused(SOFTMAX_LAST_DIM) + 1 Const = 2 reachable nodes");
        assert!(
            matches!(
                g.node(new_root).op,
                Op::Fused(fid, _) if fid == FusedOps::SOFTMAX_LAST_DIM
            ),
            "post-migration fusion produces Op::Fused(SOFTMAX_LAST_DIM, _)",
        );
        // Fused-softmax's input should be the original Const x.
        assert_eq!(g.node(new_root).inputs, vec![x.id()]);
    }

    /// Lower then fuse to fixpoint: a graph with one `Op::SoftmaxLastDim`
    /// returns to a graph that's structurally identical to the input
    /// (modulo a re-canonicalized root NodeId — the original
    /// SoftmaxLastDim node remains in the arena but is unreachable
    /// from the rewritten root).
    #[test]
    fn softmax_last_dim_lower_then_fuse_round_trips() {
        let x = Tensor::from_f32(
            vec![1.0_f32; 6],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let sm = x.softmax_last_dim();
        let graph = sm.graph().clone();
        let original_root = sm.id();

        let registry = RuleRegistry::default_rules();
        let new_roots = registry.optimize_to_fixpoint(&graph, &[original_root]);
        assert_eq!(new_roots.len(), 1);
        let new_root = new_roots[0];

        // Reachable from the new root must look like the original
        // graph: one fused-softmax consuming one Const. Phase 7.6
        // step 3: the post-migration builder + fuse cycle produces
        // `Op::Fused(SOFTMAX_LAST_DIM, _)`, not `Op::SoftmaxLastDim`.
        let g = graph.read().unwrap();
        let reachable = topo_order_multi(&g, &[new_root]);
        assert_eq!(reachable.len(), 2,
            "round-tripped graph: 1 Fused(SOFTMAX_LAST_DIM) + 1 Const = 2 reachable nodes");
        assert!(
            matches!(
                g.node(new_root).op,
                Op::Fused(fid, _) if fid == FusedOps::SOFTMAX_LAST_DIM
            ),
            "round-tripped node should be Op::Fused(SOFTMAX_LAST_DIM, _)",
        );
        assert_eq!(g.node(new_root).inputs, vec![x.id()],
            "round-tripped Fused(SOFTMAX_LAST_DIM) should consume the original input");
        // Shape and dtype preserved.
        assert_eq!(g.node(new_root).shape, g.node(original_root).shape);
        assert_eq!(g.node(new_root).dtype, g.node(original_root).dtype);
    }

    /// Empty registry is a no-op: roots come back unchanged.
    #[test]
    fn empty_registry_is_noop() {
        let x = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        let y = x.add(&x);
        let graph = y.graph().clone();
        let pre_len = graph.read().unwrap().len();
        let new_roots = RuleRegistry::new().optimize_to_fixpoint(&graph, &[y.id()]);
        assert_eq!(new_roots, vec![y.id()]);
        assert_eq!(graph.read().unwrap().len(), pre_len,
            "empty registry should add no nodes");
    }

    /// Lowering rule does not fire on a graph that has no
    /// SoftmaxLastDim nodes — the registry returns the input
    /// roots unchanged.
    #[test]
    fn lower_rule_does_not_fire_without_softmax() {
        let x = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        let y = x.relu();
        let graph = y.graph().clone();
        let pre_len = graph.read().unwrap().len();
        let new_roots = RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[y.id()]);
        assert_eq!(new_roots, vec![y.id()]);
        assert_eq!(graph.read().unwrap().len(), pre_len);
    }

    /// Fuse rule does not fire on a non-canonical Div pattern (e.g.,
    /// a plain `a / b` with no upstream softmax structure).
    #[test]
    fn fuse_rule_does_not_fire_on_plain_div() {
        let a = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        let b = a.const_f32_like(vec![2.0_f32; 4], Shape::from_dims(&[4]));
        let c = a.div(&b);
        let graph = c.graph().clone();
        let pre_len = graph.read().unwrap().len();
        let new_roots = RuleRegistry::new()
            .with_rule(Box::new(FusionRule::from_entry(
                &crate::registry::softmax_last_dim::entry(),
            )))
            .optimize_to_fixpoint(&graph, &[c.id()]);
        assert_eq!(new_roots, vec![c.id()]);
        assert_eq!(graph.read().unwrap().len(), pre_len);
    }
}
