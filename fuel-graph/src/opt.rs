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
    /// Algebraic identity rewrites that don't fit the lowering or
    /// fusion shapes: cast elimination (`Cast(src→dst) → Op(dst)`
    /// becomes `Op(src)` when the consumer supports the source
    /// dtype), constant folding, identity elimination, common-
    /// subexpression hoisting. Runs after fusion so cast-elimination
    /// can't break fusion-pattern dtype expectations.
    Algebraic,
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
        Self::capability_gated_rules(|_| true).0
    }

    /// Build lower + fuse rules, but **fuse** a fused op only when
    /// `has_kernel(id)` holds — the capability gate
    /// (`runtime-fused-op-registration.md` §6). Every fused op (static +
    /// runtime) still gets a *lowering* rule (so any already-fused node
    /// decomposes); an op WITHOUT a kernel gets **no fusion rule**, so a
    /// kernel-absent fused node never forms in the first place — the miss is
    /// caught at the pattern-match step, not repaired afterward.
    ///
    /// Returns the rule set plus the ids that were **gated out of fusion** — the
    /// miss candidates feeding the JIT work-order signal. The dispatch layer
    /// supplies `has_kernel` from its `FusedKernelRegistry` + the target
    /// backend; [`Self::default_rules`] is the all-available case.
    pub fn capability_gated_rules(
        has_kernel: impl Fn(FusedOpId) -> bool,
    ) -> (Self, Vec<FusedOpId>) {
        let registry = default_registry();
        let mut r = Self::new();
        let mut gated_out = Vec::new();
        for entry in registry.entries_iter() {
            r = r.with_rule(Box::new(LoweringRule::from_entry(entry)));
            if has_kernel(entry.id) {
                r = r.with_rule(Box::new(FusionRule::from_entry(entry)));
            } else {
                gated_out.push(entry.id);
            }
        }
        // Runtime-registered fused ops (Tier-2), same gate. A runtime op with no
        // kernel yet (JIT cold start) is gated out + reported as a miss — the
        // synthesize work-order — and its region stays primitive this pass.
        for e in crate::runtime_fused::runtime_entries() {
            r = r.with_rule(Box::new(LoweringRule::runtime(e.id)));
            if has_kernel(e.id) {
                r = r.with_rule(Box::new(FusionRule::declarative(
                    e.id,
                    crate::registry::PatternTree {
                        root:   e.region,
                        params: FusedOpParams::Runtime { scalars: Vec::new() },
                    },
                )));
            } else {
                gated_out.push(e.id);
            }
        }
        (r, gated_out)
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
        // Runtime ops decompose to their region here too — this is the
        // kernel-absent path (a backend without the synthesized kernel runs the
        // primitive recipe).
        for e in crate::runtime_fused::runtime_entries() {
            r = r.with_rule(Box::new(LoweringRule::runtime(e.id)));
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
        // Phase 3: algebraic identities to fixpoint. Runs last so it
        // can't disturb fusion patterns by eliminating casts that
        // fusion was about to match on.
        loop {
            let any = self.run_pass(RuleFamily::Algebraic, graph, &mut roots);
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
                let before = remap.len();
                family_rules[idx].rewrite(&mut g, id, &mut remap);
                // A rule that matched but recorded no remap entry made no
                // progress — e.g. a `decompose` that returned self (G2's
                // fixpoint signal: this op can't decompose further). Only
                // count real rewrites, so the fixpoint loop terminates
                // instead of spinning on a no-op match.
                if remap.len() != before {
                    any_fired = true;
                }
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

    /// Lowering rule for a runtime-registered fused op: decompose by re-emitting
    /// its `runtime_fused` sidecar region as primitives. One per runtime id.
    pub fn runtime(id: FusedOpId) -> Self {
        Self { id, decompose: crate::runtime_fused::runtime_lowering_decompose }
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
        // Arity is per-fused-op; the decompose function is the
        // authority. We don't gate on `node.inputs.len()` here.
        match &node.op {
            Op::Fused(fid, _) if *fid == self.id => true,
            // Phase 7.6 step 5 (2026-05-11): legacy-variant
            // fallthroughs have been dropped — every migrated fused
            // op now flows through `Op::Fused(id, _)`. Step 5 dropped
            // the per-fused-op `Op` variants together with these
            // matcher arms.
            _ => false,
        }
    }

    fn rewrite(&self, graph: &mut Graph, id: NodeId, remap: &mut HashMap<NodeId, NodeId>) {
        // Read params from the matched `Op::Fused(_, p)` node; the
        // legacy primitive-variant fallthroughs were dropped in
        // step 5.
        let params = match &graph.node(id).op {
            Op::Fused(_, p) => p.clone(),
            other => unreachable!(
                "LoweringRule::rewrite reached with non-matching op {other:?} \
                 — matcher contract violated"
            ),
        };
        let new_id = (self.decompose)(graph, id, &params);
        // G2 (2026-06-20): a `decompose` that returns the SAME node is at
        // its fixpoint — the op can't decompose further (a primitive, or a
        // coarse op whose recipe/primitives don't exist yet, which returns
        // self). That is not a rewrite, so we record no remap entry; the
        // driver (`run_pass`) then sees the unchanged remap and does not
        // count it as progress, so the lowering fixpoint terminates instead
        // of oscillating. The node is left as `Op::Fused` — a surfaced
        // opaque-op gap that an inventory pass can find.
        if new_id != id {
            remap.insert(id, new_id);
        }
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
    /// The declarative form (fkc-fusion-patterns §3): carries the owned
    /// `PatternTree` (the §3 grammar root + the params to stamp) so the rule is
    /// `'static`. `matches`/`rewrite` walk it via [`crate::jit::match_region`].
    Declarative(crate::registry::PatternTree),
}

impl FusionRule {
    pub fn from_entry(entry: &FusedOpEntry) -> Self {
        let pattern = match &entry.pattern {
            SubgraphPattern::Callable(f) => PatternKind::Callable(*f),
            SubgraphPattern::Declarative(tree) => PatternKind::Declarative(tree.clone()),
        };
        Self { id: entry.id, pattern }
    }

    /// Fusion rule from a declarative pattern (the runtime path): match the
    /// region and stamp `tree.params` on the emitted `Op::Fused(id, _)`.
    pub fn declarative(id: FusedOpId, tree: crate::registry::PatternTree) -> Self {
        Self { id, pattern: PatternKind::Declarative(tree) }
    }
}

impl Rule for FusionRule {
    fn name(&self) -> &'static str { "FusedOpFusion" }
    fn family(&self) -> RuleFamily { RuleFamily::Fusion }

    fn matches(&self, graph: &Graph, id: NodeId) -> bool {
        match &self.pattern {
            PatternKind::Callable(f) => (*f)(graph, id).is_some(),
            PatternKind::Declarative(tree) => {
                let consumers = |n: NodeId| {
                    (0..graph.len())
                        .filter(|&i| graph.node(NodeId(i)).inputs.contains(&n))
                        .count()
                };
                crate::jit::match_region(graph, id, &tree.root, &consumers).is_some()
            }
        }
    }

    fn rewrite(&self, graph: &mut Graph, id: NodeId, remap: &mut HashMap<NodeId, NodeId>) {
        let pattern_match = match &self.pattern {
            PatternKind::Callable(f) => (*f)(graph, id)
                .expect("rewrite called with non-matching id — matcher contract violated"),
            PatternKind::Declarative(tree) => {
                // match_region returns the region's external inputs in
                // bind-index order; wrap them (+ the params template) as a
                // PatternMatch so the shared reconstruction path below applies.
                let consumers = |n: NodeId| {
                    (0..graph.len())
                        .filter(|&i| graph.node(NodeId(i)).inputs.contains(&n))
                        .count()
                };
                let inputs = crate::jit::match_region(graph, id, &tree.root, &consumers)
                    .expect("rewrite called with non-matching declarative pattern");
                crate::registry::PatternMatch {
                    bindings: inputs.iter().enumerate().map(|(i, n)| (i, *n)).collect(),
                    params: tree.params.clone(),
                }
            }
        };
        // General-arity input reconstruction: bindings are (index, NodeId)
        // pairs; sorting by index yields the emitted node's inputs in
        // canonical order. SoftmaxLastDim binds {0 → x_id}; FusedLinear
        // binds {0 → a, 1 → b, 2 → bias}. Missing index in the [0, N)
        // range is a matcher contract violation.
        let mut sorted = pattern_match.bindings.clone();
        sorted.sort_by_key(|(idx, _)| *idx);
        for (expected_idx, (got_idx, _)) in sorted.iter().enumerate() {
            debug_assert_eq!(
                *got_idx, expected_idx,
                "FusionRule: bindings must be dense [0..N); pattern for id \
                 {:?} returned non-contiguous indices",
                self.id,
            );
        }
        let inputs: Vec<NodeId> = sorted.iter().map(|(_, n)| *n).collect();
        // Params come from the matcher — it's authoritative on what
        // variant the recognized subgraph represents.
        let params = pattern_match.params;
        let dtype = graph.node(id).dtype;
        let shape = graph.node(id).shape.clone();
        let new_id = graph.push(Node {
            op:     Op::Fused(self.id, params),
            inputs,
            shape,
            dtype,
        });
        remap.insert(id, new_id);
    }
}

// ---- Cast-fusion algebraic rule ------------------------------------------
//
// `CastFusionRule` fires on `Cast(src→dst) → Op(dst, ...)` chains
// where:
//   - the Cast (or chain of casts) has exactly one consumer at
//     each link of the chain,
//   - the consumer is a **type-preserving op** (its output dtype
//     equals the cast-fed input dtype — true for unary/binary
//     arithmetic, reductions, MatMul; *false* for compare ops,
//     Where, Cast),
//   - the consumer has a kernel for the cast's *source* dtype on
//     all input slots (capability predicate says yes).
//
// On fire: a new consumer node is appended with the cast input
// replaced by the cast's source. **The new consumer's output dtype
// is the cast-source dtype, not the original output dtype.** Any
// downstream consumers reading from the rewritten node now see the
// new dtype. This is by design — the rule fires aggressively under
// the assumption that downstream consumers either (a) are
// type-tolerant or (b) will have their own cast-fusion applied,
// collapsing the chain.
//
// **Caveat — semantic change**: in graphs where the downstream
// expects the original dst_dtype (e.g. a sink that returns BF16),
// aggressive cast-fusion silently changes the dtype the user
// observes. Wire this rule opt-in until a downstream-tolerance
// framework (precision filter, op-tolerance metadata) is in place
// to guard against unsafe rewrites.
//
// Chain cancellation is the always-safe sub-case: when the chain
// is a net no-op (`Cast(F32→BF16) → Cast(BF16→F32) → Op(F32)`),
// the final source dtype already matches the consumer's input
// slot, so neither output dtype nor kernel selection changes.
//
// Architecture v1.0: this is the first "algebraic" family rule.
// Future algebraic rewrites (constant folding, identity
// elimination, CSE hoisting) join the same family and run in the
// same Phase 3 loop.
//
// Decoupling: `fuel-graph::opt` doesn't know about binding tables,
// PrecisionGuarantee, or BackendId. The capability check is
// injected at rule construction via [`CapabilityPredicate`]; the
// consumer (typically the route-picker layer in fuel-storage or a
// higher crate) is responsible for closing over a binding-table
// reference and answering the predicate.

use std::sync::Arc;

/// Predicate answering "is there a registered kernel for this op
/// with these input/output dtypes?" The dtypes vector is the
/// consumer's full input-dtype list (after the cast-source
/// substitution) followed by the consumer's output dtype. Format
/// matches the binding-table key shape used by `fuel-storage`.
///
/// Returns `true` if at least one backend has a kernel registered
/// for the proposed (op, dtypes) combination — the route picker
/// later decides which backend to use.
pub type CapabilityPredicate =
    Arc<dyn Fn(&Op, &[DType]) -> bool + Send + Sync>;

/// `Cast(src→dst) → Op(dst) ≡ Op(src)` when the consumer has a
/// kernel for `src`.
///
/// Only fires when the cast has exactly one consumer (MVP: avoids
/// the DAG complication of partial fan-out elimination). Multi-
/// consumer Cast handling is a follow-up: rewrite one consumer and
/// keep the cast alive for the others.
pub struct CastFusionRule {
    capabilities: CapabilityPredicate,
}

impl CastFusionRule {
    /// Build the rule with a capability predicate. The predicate
    /// closes over whatever capability source the caller has (a
    /// binding table, a route picker, a static manifest). The rule
    /// itself stays decoupled from those internals.
    pub fn new(capabilities: CapabilityPredicate) -> Self {
        Self { capabilities }
    }

    /// Walk `node.inputs` looking for an edge ending at a Cast (or
    /// a chain of Casts) where every link in the chain has exactly
    /// one consumer (the next link). Returns `Some((input_index,
    /// final_src_id, consumer_dtypes))` where `final_src_id` is the
    /// deepest reachable source after walking through the chain.
    /// `consumer_dtypes` is the dtypes vector to ask the predicate
    /// about. Returns `None` if no input qualifies.
    ///
    /// Walking through chained casts in one pass avoids a subtle
    /// fixpoint trap: after the outer Cast is eliminated, the inner
    /// Cast's other consumer (the now-orphan outer Cast) is still
    /// in the arena and would make `count_consumers != 1` on the
    /// next iteration. Collapsing the whole chain at once sidesteps
    /// this because we never re-examine intermediate links.
    ///
    /// Shared by `matches` (which discards the index) and `rewrite`
    /// (which uses the index to point the new consumer's input edge
    /// at the deepest source). Computed twice intentionally — the
    /// alternative is mutable state across `matches`/`rewrite`,
    /// which conflicts with the trait's pure-matcher contract.
    fn find_eligible_cast_input(
        &self,
        graph: &Graph,
        id: NodeId,
    ) -> Option<(usize, NodeId, Vec<fuel_core_types::DType>)> {
        let node = graph.node(id);
        // Don't fire on a Cast node itself — fire on the consumer.
        if matches!(node.op, Op::Cast(_)) { return None; }
        for (idx, &input_id) in node.inputs.iter().enumerate() {
            // Walk through chained casts. `current` is the cast (or
            // chain of casts) currently being examined; `prev` is
            // the consumer at this link of the chain (initially `id`,
            // then the outer-most cast as we descend).
            if !matches!(graph.node(input_id).op, Op::Cast(_)) { continue; }
            // Type-preserving filter: the consumer's output dtype
            // must equal the cast'd input slot's dtype. This is the
            // family of ops where rewriting the input dtype is
            // equivalent to rewriting the output dtype — Neg/Add/
            // Mul/Div, reductions, MatMul, etc. Excludes Compare
            // (T→U8), Where (U8+T+T→T), and Cast itself (which we
            // already filtered above). For non-type-preserving
            // consumers, swapping the input dtype would produce an
            // output we can't predict without per-op knowledge.
            if node.dtype != graph.node(input_id).dtype { continue; }
            let mut prev = id;
            let mut current = input_id;
            let final_src_id = loop {
                let current_node = graph.node(current);
                if !matches!(current_node.op, Op::Cast(_)) {
                    break current;
                }
                // Single-consumer constraint at this link: the cast
                // at `current` must feed only `prev`. Otherwise we
                // can't eliminate it (would orphan the other consumer's
                // input edge).
                if !is_only_consumer(graph, current, prev) {
                    // The chain stops here; we treat the current node
                    // (a Cast) as the source. The consumer takes the
                    // dst dtype of this Cast.
                    break current;
                }
                let Some(&next) = current_node.inputs.first() else {
                    break current;
                };
                prev = current;
                current = next;
            };
            // If the walk didn't move past the first cast (e.g.
            // because the first cast has multiple consumers), there's
            // nothing to fuse for this input.
            if final_src_id == input_id { continue; }
            let final_src_dtype = graph.node(final_src_id).dtype;
            // Build the proposed input-dtypes + output-dtype vector.
            // Type-preserving filter above guarantees output dtype
            // tracks the cast-fed input, so the new output dtype is
            // `final_src_dtype` (not the original `node.dtype`).
            let mut dtypes: Vec<fuel_core_types::DType> = node
                .inputs
                .iter()
                .map(|&iid| if iid == input_id {
                    final_src_dtype
                } else {
                    graph.node(iid).dtype
                })
                .collect();
            dtypes.push(final_src_dtype);
            if (self.capabilities)(&node.op, &dtypes) {
                return Some((idx, final_src_id, dtypes));
            }
        }
        None
    }
}

/// Returns `true` if `expected_consumer` is the only consumer of
/// `target` in the arena. Used to guard chained-cast elimination.
fn is_only_consumer(graph: &Graph, target: NodeId, expected_consumer: NodeId) -> bool {
    let mut found_others = false;
    for i in 0..graph.len() {
        let nid = NodeId(i);
        if nid == expected_consumer { continue; }
        if graph.node(nid).inputs.contains(&target) {
            found_others = true;
            break;
        }
    }
    !found_others
}

impl Rule for CastFusionRule {
    fn name(&self) -> &'static str { "CastFusion" }
    fn family(&self) -> RuleFamily { RuleFamily::Algebraic }

    fn matches(&self, graph: &Graph, id: NodeId) -> bool {
        self.find_eligible_cast_input(graph, id).is_some()
    }

    fn rewrite(&self, graph: &mut Graph, id: NodeId, remap: &mut HashMap<NodeId, NodeId>) {
        let (idx, final_src_id, _dtypes) = self
            .find_eligible_cast_input(graph, id)
            .expect("CastFusionRule::rewrite called on non-matching node");
        let node = graph.node(id).clone();
        let final_src_dtype = graph.node(final_src_id).dtype;
        let mut new_inputs = node.inputs.clone();
        new_inputs[idx] = final_src_id;
        let new_id = graph.push(Node {
            op:     node.op.clone(),
            inputs: new_inputs,
            shape:  node.shape.clone(),
            // Type-preserving op (matcher's invariant): the new
            // output dtype tracks the new input dtype. Downstream
            // consumers seeing this node may observe a different
            // dtype than before the rewrite — see the module
            // docstring on the aggressive-semantics caveat.
            dtype:  final_src_dtype,
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
        Op::Flip { dim } => (56, vec![*dim as i64], vec![], vec![], None, None),
        Op::Roll { dim, shift } => (57, vec![*dim as i64, *shift], vec![], vec![], None, None),
        Op::CumSum { dim } => (58, vec![*dim as i64], vec![], vec![], None, None),
        Op::PadBackward { in_shape, padding, mode } => {
            let mode_tag = match mode {
                crate::PadMode::Constant => 0_i64,
                crate::PadMode::Reflect => 1,
                crate::PadMode::Replicate => 2,
            };
            let mut ints: Vec<i64> = Vec::with_capacity(padding.len() * 2 + 1);
            for &(b, a) in padding {
                ints.push(b as i64);
                ints.push(a as i64);
            }
            ints.push(mode_tag);
            (60, ints, vec![], vec![], Some(in_shape.dims().to_vec()), None)
        }
        Op::Pad { padding, mode, value } => {
            let mode_tag = match mode {
                crate::PadMode::Constant => 0_i64,
                crate::PadMode::Reflect => 1,
                crate::PadMode::Replicate => 2,
            };
            // Flatten padding pairs into [b0, a0, b1, a1, ...] + the
            // mode tag. Multi-dim padding keys uniquely off this vec.
            let mut ints: Vec<i64> = Vec::with_capacity(padding.len() * 2 + 1);
            for &(b, a) in padding {
                ints.push(b as i64);
                ints.push(a as i64);
            }
            ints.push(mode_tag);
            (59, ints, vec![value.to_bits()], vec![], None, None)
        }

        Op::MatMul => (30, vec![], vec![], vec![], None, None),
        Op::Transpose => (31, vec![], vec![], vec![], None, None),
        Op::Permute(axes) => (32, vec![], vec![], axes.clone(), None, None),

        Op::Cast(dt) => (40, vec![], vec![], vec![], None, Some(dtype_key(*dt))),
        Op::BroadcastTo(s) => (41, vec![], vec![], vec![], Some(s.dims().to_vec()), None),
        Op::Reshape(s) => (42, vec![], vec![], vec![], Some(s.dims().to_vec()), None),
        Op::Contiguize => (46, vec![], vec![], vec![], None, None),
        Op::Unsqueeze { dim } => (45, vec![*dim as i64], vec![], vec![], None, None),
        Op::Squeeze { dim } => (48, vec![*dim as i64], vec![], vec![], None, None),
        Op::ReduceSumTo(s) => (43, vec![], vec![], vec![], Some(s.dims().to_vec()), None),
        Op::ReduceMaxTo(s) => (44, vec![], vec![], vec![], Some(s.dims().to_vec()), None),
        // Phase 7.6 step 5 (2026-05-11): tags 46/70/71/72/73/74/75/76
        // formerly assigned to SoftmaxLastDim, LayerNormLastDim,
        // RmsNormLastDim, Rope and the four backward helpers have
        // been retired with their `Op` variants. Those ops now flow
        // through the `Op::Fused(fid, fparams)` arm below (tag 200,
        // with id + params encoded into the int/bit slots) and CSE
        // dedupes correctly because each entry produces a distinct
        // FusedOpId.

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
        //
        //    Identity Cast(d) on a dtype-d input is a no-op: dispatch keys
        //    casts on [src, dst] and intentionally registers no [d, d]
        //    kernel, so this elision (not a kernel) is what makes a same-dtype
        //    cast disappear before realize. Needs the input node's dtype, so
        //    it lives here rather than in `try_simplify` (which sees only
        //    op + inputs).
        if let Op::Cast(target) = &op {
            if g.node(mapped_inputs[0]).dtype == *target {
                remap.insert(id, mapped_inputs[0]);
                continue;
            }
        }
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
        // Phase 7.6 step 4: emit the registry-extended shape. The
        // executor's `Op::Fused(FUSED_LINEAR, _)` arm dispatches to
        // the same fused-linear kernel as the legacy variant. Step 5
        // drops the legacy `Op::FusedLinear` variant entirely.
        let new_id = g.push(Node {
            op: Op::Fused(
                crate::registry::FusedOps::FUSED_LINEAR,
                crate::registry::FusedOpParams::FusedLinear,
            ),
            // FusedLinear takes the *original* rank-1 bias, not the
            // BroadcastTo'd one — the kernel broadcasts it internally
            // to the matmul output shape.
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
/// **Bundle-aware aliasing** (Option C, Session 3): the alias-set
/// computation treats `Op::View { slot }` as alias-extending — every
/// View of a multi-output producer shares the bundle's storage Arc, so
/// a destructive op on the producer (or on any sibling View) must
/// run after every reader of every other View of the same producer.
/// `Op::ViewOwned` is NOT alias-extending: after its forward memcpy,
/// the ViewOwned's output is an independent Storage; it is still
/// pinned-after destructive ops on the producer via the regular
/// data-dependency edge (`inputs[0] == producer`), which falls out
/// of the standard reader analysis below without needing the
/// alias-set extension. See [`collect_alias_set`] for the full
/// alias-extension rule.
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
        // The alias set of `destroyed`: the destructive op writes
        // through `destroyed`'s Storage Arc, and every view-op node
        // transitively derived from `destroyed` shares that same Arc
        // (executor's view-op materialization wraps the source's Arc
        // in a fresh Layout — see `is_view_op` in `lib.rs`). So a
        // reader of any alias-set member sees `destroyed`'s bytes
        // and must run BEFORE the destructive op. Without this
        // walk, `y = x.transpose(); z = relu(y); x.relu_inplace()`
        // would not pin `relu(y)` before `relu_inplace(x)`, even
        // though they share storage.
        let alias_set = collect_alias_set(graph, destroyed, &consumers, &order);
        for &alias in &alias_set {
            let Some(readers) = consumers.get(&alias) else { continue };
            for &reader in readers {
                // Skip the destructive op itself + readers that are
                // themselves alias members (those are view ops; their
                // own consumers are what we actually need to pin, and
                // they show up in subsequent iterations).
                if reader != nid && !alias_set.contains(&reader) {
                    ordering.entry(nid).or_default().push(reader);
                }
            }
        }
    }
    OrderingEdges(ordering)
}

/// Collect the alias set of `root`: every node whose realized Storage
/// is the same `Arc<RwLock<Storage>>` as `root`'s.
///
/// Two families of alias-extending op share the input's Storage Arc
/// at realize time:
///
/// 1. **Single-input view ops** ([`Op::is_view_op`]): the executor
///    wraps the source's Arc with a fresh Layout — no copy. Examples:
///    `Op::Transpose`, `Op::Permute`, `Op::Slice`, `Op::Unsqueeze`,
///    `Op::Squeeze`, `Op::BroadcastTo`, `Op::Flip`.
/// 2. **Multi-output projection** (`Op::View { slot }`): the
///    realized View clones the producer's bundled storage Arc and
///    exposes one slot's window — bytes are shared, the bundle's Arc
///    refcount tracks the consumer's lifetime (Session 1 of the
///    multi-output Option C design — see
///    [`fuel_core_types::storage::OutputView`]).
///
/// `Op::ViewOwned` is explicitly NOT alias-extending: at execution
/// time it allocates a fresh standalone Storage and memcpys the
/// slot's bytes in. After the memcpy runs, no Arc handle on the
/// producer's bundle remains in the ViewOwned's chain. ViewOwned is
/// still pinned-after destructive readers via the regular
/// data-dependency edge (its `inputs[0]` is the producer), which the
/// caller already accounts for outside this function.
///
/// **Sibling-bundle rule**: when `root` is itself an `Op::View`,
/// every other `Op::View` of the same producer shares the same
/// bundle Arc. To capture them, the function pre-seeds the producer
/// into the alias set so the forward walk picks up siblings the same
/// way it picks up downstream `is_view_op` derivatives.
///
/// Implementation: O(|order|) forward walk after a constant-time
/// pre-seed. A node enters the alias set iff its op is in the
/// alias-extending union AND its `inputs[0]` is already in the set.
fn collect_alias_set(
    graph: &crate::Graph,
    root: NodeId,
    _consumers: &HashMap<NodeId, Vec<NodeId>>,
    order: &[NodeId],
) -> std::collections::HashSet<NodeId> {
    use crate::Op;

    let mut alias: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
    alias.insert(root);

    // Sibling-bundle pre-seed: if root is an Op::View of producer P,
    // every other Op::View of P shares the same bundle Arc. Add P
    // here so the forward walk reaches sibling Views via their
    // `inputs[0] == P` membership in the alias set.
    if let Op::View { .. } = graph.node(root).op {
        if let Some(&producer) = graph.node(root).inputs.first() {
            alias.insert(producer);
        }
    }

    for &nid in order {
        if alias.contains(&nid) { continue }
        let node = graph.node(nid);
        // Op::View shares the producer's bundle Arc — it extends the
        // alias set even though it isn't an `is_view_op` in the
        // narrower single-storage-strided-view sense.
        let extends_alias = node.op.is_view_op()
            || matches!(node.op, Op::View { .. });
        if !extends_alias { continue }
        if let Some(&inp) = node.inputs.first() {
            if alias.contains(&inp) {
                alias.insert(nid);
            }
        }
    }
    alias
}

/// Insert defensive `Op::Copy` snapshots ahead of destructive ops
/// that would otherwise produce a data-flow ↔ ordering-edge cycle.
///
/// Phase 5 of the in-place ops infrastructure
/// (`docs/session-prompts/in-place-ops-infrastructure.md`). The
/// view-aware [`derive_ordering`] (Phase 4a) handles the common case
/// where a destructive op's target is read only by upstream consumers
/// — pinning the destructive op after those reads is enough. The
/// case Phase 5 addresses is when a single op reads BOTH the
/// in-place target X AND a downstream consumer of the in-place op
/// (e.g., residual connections: `y = x.relu_inplace(); z = y + x`).
///
/// In that case, the data-flow says the reader (`z`) must run after
/// the destructive op (because it depends on `y`), and the ordering
/// edge says the destructive op must run after the reader (because
/// the reader reads `x`). Without intervention, [`execution_plan`]
/// detects the cycle and panics.
///
/// ## Conflict predicate (dependency-based, NOT topo-position-based)
///
/// A reader `R` of the destructive op `D`'s target needs a safety
/// copy iff `R` is not **provably ordered before** `D`. The proof
/// obligation is grounded in the executor's actual contract:
/// [`execution_plan`] emits a linear extension of the *combined*
/// precedence graph — data edges (`input → consumer`) plus
/// [`derive_ordering`]'s ordering edges (`reader → destructive op`) —
/// and the executor dispatches work items strictly sequentially in
/// that order. Therefore:
///
/// - `R` has a path to `D` in the combined graph AND `D` has no
///   path back to `R` → every valid plan runs `R` before `D`'s
///   mutation. Safe; **no copy**. (This covers every reader
///   `derive_ordering` pins directly, plus view-op readers whose
///   consumers are pinned — the pre-gap readers of an evict-chain
///   `Op::Move`, for example.)
/// - `D` has a path to `R` (e.g., the residual pattern: `R`
///   consumes `D`'s output while also reading `D`'s target) → the
///   pin edge `R → D` would close a cycle; the executor cannot
///   order `R` first. **Copy needed.**
/// - Neither direction → no ordering guarantee exists (possible for
///   alias-member view readers, which `derive_ordering` deliberately
///   does not pin). Safety cannot be proven, so **copy needed** —
///   the conservative default: a spurious copy costs bytes, a
///   missing copy is silent data corruption.
///
/// Historical note: this pass originally inferred conflicts from DFS
/// topo-order *position* (any reader appearing after `D` in
/// `topo_order_multi` order was treated as conflicting). Position in
/// a DFS walk carries no dependency information — an independent
/// pre-gap reader of a `Op::Move`d tensor could land after the Move
/// purely because of root visit order, earning a spurious copy
/// stamped onto the destructive op's device (live-GPU failure,
/// 2026-06-11).
///
/// This pass rewrites each conflict by:
/// 1. Inserting an `Op::Copy { target: same_device }(X) → X_safe` —
///    a same-device byte snapshot independent of `X`'s storage.
/// 2. Rewiring the conflicting reader's input from `X` to `X_safe`.
///
/// After the rewrite, the reader reads the snapshot (pre-mutation
/// bytes) while the destructive op proceeds on `X`. No cycle.
///
/// Scope: detects DIRECT conflicts (reader's input IS the
/// destructive op's target). View-mediated conflicts (reader reads
/// `view(X)`) fall through and produce the cycle panic from
/// `execution_plan` — a clear-but-not-friendly error pointing the
/// user at the pattern. View-mediated auto-resolution is a follow-up
/// (would need to insert `Copy(X)` then re-derive the view from the
/// snapshot for each conflicting reader).
///
/// Idempotent: calling this twice on the same graph inserts copies
/// at most once per conflict (the rewrite removes the conflict by
/// redirecting the reader's input edge to the copy).
///
/// Returns the number of safety copies inserted (for telemetry /
/// testing).
pub fn insert_safety_copies(graph: &mut crate::Graph, roots: &[NodeId]) -> usize {
    let order = topo_order_multi(graph, roots);

    // Combined precedence graph: data edges (input → consumer) plus
    // derive_ordering's edges (reader → destructive op). This is
    // exactly the edge set execution_plan linearizes, so path
    // queries here answer "does every valid plan order A before B?".
    let ordering = derive_ordering(graph, roots);
    let node_set: std::collections::HashSet<NodeId> = order.iter().copied().collect();
    let mut succ: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    let mut pred: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for &nid in &order {
        for &input in &graph.node(nid).inputs {
            if node_set.contains(&input) {
                succ.entry(input).or_default().push(nid);
                pred.entry(nid).or_default().push(input);
            }
        }
        for &dep in ordering.deps_of(nid) {
            if node_set.contains(&dep) {
                succ.entry(dep).or_default().push(nid);
                pred.entry(nid).or_default().push(dep);
            }
        }
    }

    // All nodes reachable from `start` over `adj` (excluding `start`
    // itself unless it sits on a cycle). Iterative DFS; the visited
    // set terminates cyclic combined graphs (the very cycles this
    // pass exists to break).
    fn reach(
        start: NodeId,
        adj: &HashMap<NodeId, Vec<NodeId>>,
    ) -> std::collections::HashSet<NodeId> {
        let mut seen: std::collections::HashSet<NodeId> = std::collections::HashSet::new();
        let mut stack = vec![start];
        while let Some(n) = stack.pop() {
            if let Some(next) = adj.get(&n) {
                for &m in next {
                    if seen.insert(m) {
                        stack.push(m);
                    }
                }
            }
        }
        seen
    }

    // Snapshot conflicts before mutating the graph: collect
    // (destructive_op_id, target, conflicting_readers) tuples.
    struct Conflict {
        destructive_nid: NodeId,
        target: NodeId,
        target_shape: crate::Shape,
        target_dtype: DType,
        conflicting_readers: Vec<NodeId>,
    }
    let mut conflicts: Vec<Conflict> = Vec::new();

    for &nid in &order {
        let node = graph.node(nid);
        let Some(d_idx) = node.op.destructive_input() else { continue };
        if d_idx >= node.inputs.len() { continue }
        let target = node.inputs[d_idx];

        // `forced_before`: nodes with a combined-graph path TO the
        // destructive op. `forced_after`: nodes the destructive op
        // has a combined-graph path to. A reader is provably safe
        // iff it is forced-before AND not forced-after (the latter
        // would mean its pin edge closes a cycle the executor cannot
        // satisfy). Anything else — cycle members, descendants, or
        // readers with no ordering guarantee in either direction —
        // gets a safety copy. False positives cost a snapshot;
        // false negatives are silent data corruption.
        let forced_before = reach(nid, &pred);
        let forced_after = reach(nid, &succ);

        let mut readers: Vec<NodeId> = Vec::new();
        for &maybe_reader in &order {
            if maybe_reader == nid { continue }
            if !graph.node(maybe_reader).inputs.contains(&target) { continue }
            let provably_before = forced_before.contains(&maybe_reader)
                && !forced_after.contains(&maybe_reader);
            if !provably_before {
                readers.push(maybe_reader);
            }
        }
        if readers.is_empty() { continue }

        let target_node = graph.node(target);
        conflicts.push(Conflict {
            destructive_nid: nid,
            target,
            target_shape: target_node.shape.clone(),
            target_dtype: target_node.dtype,
            conflicting_readers: readers,
        });
    }

    let inserted = conflicts.len();
    for Conflict { destructive_nid, target, target_shape, target_dtype, conflicting_readers } in conflicts {
        // Pick the copy's target_location. Prefer the target's own
        // placement (if any) — that's the device the data lives on.
        // Fall back to the destructive op's placement, then to Cpu.
        // The same-device Op::Copy semantic produces a fresh storage
        // on `target_location` (executor's WorkItemKind::Copy arm
        // allocates + the wrapper memcpys bytes).
        let target_location = graph.placement(target)
            .or_else(|| graph.placement(destructive_nid))
            .unwrap_or(DeviceLocation::Cpu);
        let copy_id = graph.push(crate::Node {
            op: crate::Op::Copy { target: target_location },
            inputs: vec![target],
            shape: target_shape,
            dtype: target_dtype,
        });
        // Propagate target_backend so the executor can look up the
        // copy's wrapper. Inherit from the target (the source of the
        // copy), falling back to the destructive op's target_backend.
        if let Some(backend) = graph.target_backend(target)
            .or_else(|| graph.target_backend(destructive_nid))
        {
            graph.set_target_backend(copy_id, backend);
        }
        for reader_id in conflicting_readers {
            graph.rewrite_input(reader_id, target, copy_id);
        }
    }
    inserted
}

/// Walk every multi-output producer in the graph and promote some of
/// its `Op::View` consumers to `Op::ViewOwned` when slot lifetimes
/// are sufficiently asymmetric that keeping the whole bundle alive
/// for the longest-lived slot is wasteful.
///
/// v1 heuristic (Option C, Session 2): for each multi-output producer
/// P, compute the per-slot "last use" position in topo order (the
/// max position over each slot's Views and their transitive
/// consumers). Slots whose last use is strictly later than at least
/// one other slot's last use get all their Views promoted to
/// `Op::ViewOwned`. The remaining short-lived slots stay as
/// `Op::View` so the bundle drops naturally when their last consumer
/// finishes.
///
/// Concrete example: SelectiveScan's `y` (slot 0) consumed at the
/// next layer; `last_state` (slot 1) retained across a barrier into
/// the next autoregressive step. `last_state`'s last use is much
/// later → it gets promoted to `ViewOwned`, the bundle drops once
/// `y`'s consumer finishes, and `last_state`'s standalone copy
/// survives across the gap.
///
/// Implementation: builds a consumer index over the topo order,
/// computes `last_use[N] = max(pos[N], last_use[consumers])` by
/// reverse-iterating, then per producer makes a per-slot rollup
/// and rewrites edges from the promoted Views to fresh
/// `Op::ViewOwned` nodes. Returns the number of promotions made.
/// Idempotent on already-promoted Views (an `Op::ViewOwned` is a
/// fixpoint).
///
/// **Where this plugs in:** today the multi-output infra has no
/// production consumer (Session 2 ships the authoring contract and
/// this pass; consumers light up in the
/// `selective-scan-ssd-chunk-multi-output-followup` session). The
/// pass is a no-op on graphs without any `Op::View` consumers of a
/// multi-output producer — the common case until consumers migrate.
/// Long-term it likely runs after the ranker/picker pass and before
/// `compile_plan`; placement is the picker session's call.
pub fn promote_views_for_liveness(
    graph:  &mut crate::Graph,
    roots:  &[NodeId],
) -> usize {
    use crate::Op;

    let order = topo_order_multi(graph, roots);
    if order.is_empty() {
        return 0;
    }
    // Forward consumer index: producer NodeId → list of consumer NodeIds.
    let mut consumers: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for &nid in &order {
        for &input in &graph.node(nid).inputs {
            consumers.entry(input).or_default().push(nid);
        }
    }

    // `depth[N]`: longest-input-chain length feeding into N. Two
    // sibling consumers in the same conceptual layer get the same
    // depth, so the lifetime comparison below treats them symmetrically
    // even when topo *position* puts them in different slots of the
    // walk (which depends on roots[] order and traversal direction —
    // unstable for our purposes).
    let mut depth: HashMap<NodeId, usize> = HashMap::with_capacity(order.len());
    for &nid in &order {
        let mut d = 0usize;
        for &inp in &graph.node(nid).inputs {
            if let Some(&di) = depth.get(&inp) {
                if di + 1 > d { d = di + 1; }
            }
        }
        depth.insert(nid, d);
    }

    // `last_use[N]`: max downstream depth reachable from N (including
    // N itself). Computed by reverse-iterating the topo order and
    // propagating each node's depth back to its inputs.
    let mut last_use: HashMap<NodeId, usize> = HashMap::with_capacity(order.len());
    for &nid in order.iter().rev() {
        let mut lu = depth[&nid];
        if let Some(downstream) = consumers.get(&nid) {
            for &c in downstream {
                if let Some(&cl) = last_use.get(&c) {
                    if cl > lu { lu = cl; }
                }
            }
        }
        last_use.insert(nid, lu);
    }

    // Find every multi-output producer that has any View / ViewOwned
    // consumers in this graph. Per producer: bucket the Views by slot
    // and compute the per-slot last-use position.
    struct ViewEntry {
        view_id: NodeId,
        slot:    u32,
        is_owned: bool,
    }
    let mut by_producer: HashMap<NodeId, Vec<ViewEntry>> = HashMap::new();
    for &nid in &order {
        let node = graph.node(nid);
        let (slot, is_owned) = match node.op {
            Op::View      { slot } => (slot, false),
            Op::ViewOwned { slot } => (slot, true),
            _ => continue,
        };
        // View / ViewOwned have a single input (the producer).
        let Some(&producer) = node.inputs.first() else { continue };
        // Only meaningful if producer was declared multi-output;
        // otherwise it's an authoring bug that the Tensor::view
        // builder already rejected. Defensive skip.
        if !graph.is_multi_output(producer) { continue }
        by_producer
            .entry(producer)
            .or_default()
            .push(ViewEntry { view_id: nid, slot, is_owned });
    }

    // For each producer, decide which slots get promoted and collect
    // the per-View promotions. Defer the actual graph mutations to a
    // second pass so we can iterate the analysis with `&Graph`.
    struct Promotion {
        view_id:    NodeId,
        slot:       u32,
        view_shape: crate::Shape,
        view_dtype: DType,
        // Consumers of `view_id` at analysis time — every edge gets
        // rewired to the new ViewOwned node.
        view_consumers: Vec<NodeId>,
    }
    let mut promotions: Vec<Promotion> = Vec::new();
    for (_producer, views) in &by_producer {
        // Per-slot last use rollup.
        let mut slot_last_use: HashMap<u32, usize> = HashMap::new();
        for v in views {
            let lu = last_use[&v.view_id];
            slot_last_use
                .entry(v.slot)
                .and_modify(|cur| { if lu > *cur { *cur = lu; } })
                .or_insert(lu);
        }
        // If every slot in this producer has the same last_use, the
        // bundle's natural drop position is fine — no promotions.
        let min_lu = match slot_last_use.values().copied().min() {
            Some(v) => v,
            None    => continue,
        };
        for v in views {
            if v.is_owned {
                // Already owned — nothing to do (idempotent).
                continue;
            }
            let lu = slot_last_use[&v.slot];
            if lu <= min_lu {
                // This slot is among the shortest-lived → stays View.
                continue;
            }
            let v_node = graph.node(v.view_id);
            promotions.push(Promotion {
                view_id:        v.view_id,
                slot:           v.slot,
                view_shape:     v_node.shape.clone(),
                view_dtype:     v_node.dtype,
                view_consumers: consumers.get(&v.view_id).cloned().unwrap_or_default(),
            });
        }
    }

    // Apply mutations: each promoted Op::View becomes a fresh
    // Op::ViewOwned that consumers point at instead. The old Op::View
    // node stays in the arena (no in-place op mutation) but is
    // unreachable from the roots — execution_plan + topo skip it
    // naturally. This mirrors the insert_safety_copies pattern of
    // "add new nodes + rewrite_input" rather than mutating ops in
    // place.
    let count = promotions.len();
    for p in promotions {
        let producer = graph.node(p.view_id).inputs[0];
        let owned_id = graph.push(crate::Node {
            op:     Op::ViewOwned { slot: p.slot },
            inputs: vec![producer],
            shape:  p.view_shape,
            dtype:  p.view_dtype,
        });
        // Inherit target_backend from the original View when set, so
        // the executor can look up ViewOwned's wrapper without a
        // re-resolution pass.
        if let Some(backend) = graph.target_backend(p.view_id) {
            graph.set_target_backend(owned_id, backend);
        }
        for consumer in p.view_consumers {
            graph.rewrite_input(consumer, p.view_id, owned_id);
        }
    }
    count
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
/// consumers. Returns `(move_id, reload_id)`.
///
/// ## What it does
///
/// Inserts two new nodes:
/// 1. `mv = Op::Move { target: Cpu }` reading `candidate` — stages the
///    data to host memory AND destructively releases `candidate`'s
///    device-resident storage once it runs ([`Op::Move`] is the fused
///    Copy + Release: `destructive_input() == Some(0)`). The
///    [`derive_ordering`] pass pins it to run AFTER every
///    non-destructive reader of `candidate` (the pre-gap consumers the
///    caller left untouched).
/// 2. `reload = Op::Copy { target: src_device }` reading `mv` —
///    restages the data to the device for the post-gap consumers.
///
/// Then rewrites every `post_gap_consumer`'s input edge from `candidate`
/// to `reload`. Pre-gap consumers keep reading `candidate` directly.
///
/// Historical note: the original (legacy-executor era) chain was three
/// nodes — `Op::Copy{Cpu}` + side-effect-root `Op::Release` +
/// `Op::Copy{device}`. `Op::Move` collapsed the first two; the fused
/// node's output is consumed by `reload`, so no side-effect root is
/// needed.
///
/// ## Caller's responsibility
///
/// - `candidate` is a NodeId currently in the graph whose device
///   residency is `src_device`.
/// - `post_gap_consumers` are NodeIds currently in the graph that each
///   have `candidate` in their `inputs`. Typically these come from the
///   residency analyzer's gap-positioning logic.
/// - The caller stamps placement/backend metadata on the new nodes
///   (`mv` runs on `src_device`'s backend, `reload` on Cpu — the
///   staged copy's residency) before realizing on the pipelined
///   executor. `fuel-dispatch::residency::insert_residency_evictions`
///   does this.
pub fn insert_evict_reload(
    graph: &SharedGraph,
    candidate: NodeId,
    src_device: DeviceLocation,
    post_gap_consumers: &[NodeId],
) -> (NodeId, NodeId) {
    let mut g = graph.write().unwrap();
    let (shape, dtype) = {
        let n = g.node(candidate);
        (n.shape.clone(), n.dtype)
    };

    let move_id = g.push(Node {
        op:     Op::Move { target: DeviceLocation::Cpu },
        inputs: vec![candidate],
        shape:  shape.clone(),
        dtype,
    });

    let reload_id = g.push(Node {
        op:     Op::Copy { target: src_device },
        inputs: vec![move_id],
        shape,
        dtype,
    });

    // Rewrite each post-gap consumer's `candidate` input to `reload_id`.
    // Pre-gap consumers (not in this list) continue reading `candidate`
    // directly, which is why derive_ordering needs `move_id` to run
    // after them.
    for &consumer in post_gap_consumers {
        g.rewrite_input(consumer, candidate, reload_id);
    }

    (move_id, reload_id)
}

// ===========================================================================
// Phase 2.1 — cross-device `Op::Copy` insertion
// ===========================================================================
//
// When the picker (or a user pin) commits a kernel-bearing node to a
// `DeviceLocation` that doesn't share a storage substrate with one of
// its inputs' resident `DeviceLocation`, the data needs to actually
// move. The executor handles the FINAL output via `pipelined_bridge`'s
// realize-root splicing; cross-device edges INSIDE the graph are
// handled here.
//
// `insert_cross_device_copies` is the pre-execute pass that handles
// internal cross-device edges. Architecture v1.0 §04 names the
// optimizer as the owner of transfer-op insertion — this is that
// owner. Wired into `fuel-core::pipelined_bridge::prepare()` (picker
// arc step 2): the bridge derives `placement_for` from the monolithic
// `target_backend` pinning + the StorageCache's resident locations,
// and `shares_storage` from `SystemTopology`.
//
// The pass deliberately doesn't decide placements itself — callers
// (the optimizer ranker / picker) commit to placements first, then
// hand the pass `(graph, placement_for_node, shares_storage)` and
// it materializes the implied transfers. Same pattern as the
// JudgeOracle / CapabilitiesLookup callbacks in
// `fuel-dispatch::ranker`: keep `fuel-graph` ignorant of how
// placements / topology are derived; the closure is the seam.

/// Insert `Op::Copy { target }` nodes on every edge where the
/// consumer's intended `DeviceLocation` doesn't share a storage
/// substrate with the producer's. Returns a remap of original
/// `NodeId`s to their post-rewrite equivalents (most NodeIds map to
/// themselves; only consumers of cross-device edges acquire new
/// inputs in-place via `rewrite_input`).
///
/// # Arguments
///
/// - `graph` — the graph to rewrite, mutably. New `Op::Copy` nodes
///   are appended via `Graph::push`; consumer edges are rewired via
///   the internal `rewrite_input` helper.
/// - `roots` — the realize roots the caller cares about. The pass
///   walks `topo_order_multi(roots)` and considers every node it
///   visits.
/// - `placement_for` — `Fn(NodeId) -> Option<DeviceLocation>`.
///   Returns the picker's committed placement for the node, or
///   `None` if the node has no placement (view ops, structural ops,
///   user-unpinned ops in a single-backend realize). When either
///   the producer or consumer has no placement, the pass leaves
///   the edge alone — the executor falls back to its existing
///   handling (auto_contiguize or the realize-root splice).
/// - `shares_storage` — `Fn(DeviceLocation, DeviceLocation) -> bool`.
///   The topology query. Typical implementation is
///   `SystemTopology::shares_storage((b1, dev1), (b2, dev2))` after
///   the caller resolves backends to devices; we pass just the
///   `DeviceLocation` pair here because the substrate question is
///   device-level (`Cpu` shares with `Cpu`, `Cuda{0}` shares with
///   `Cuda{0}` but not `Cuda{1}`, etc.). Same-device queries should
///   return `true`.
///
/// # Returns
///
/// The `NodeId`s of the inserted `Op::Copy` nodes, in insertion
/// order. Callers typically need these to stamp executor metadata
/// on the new nodes — e.g. `fuel-core::pipelined_bridge` sets
/// `target_backend` = the SOURCE backend (the pipelined executor's
/// Op::Copy kernel-lookup convention: the transfer kernel runs on
/// the backend the bytes come FROM). `.len()` is the transfer-count
/// metric the optimizer reports when deciding whether to pursue
/// alternative placements that minimize transfers.
///
/// # Idempotence
///
/// The pass is idempotent — re-running on an already-rewritten
/// graph adds zero new copies (the inserted `Op::Copy` nodes
/// carry `placement` matching the consumer's, so subsequent
/// passes see same-device edges from `Op::Copy` to consumer).
///
/// # CSE on inserted copies
///
/// When two consumers share an input AND both need to bring it to
/// the same target device, the pass deduplicates: one `Op::Copy`
/// node serves both consumers. This matches the existing CSE
/// pattern in `optimize_to_fixpoint`.
pub fn insert_cross_device_copies<P, S>(
    graph: &mut Graph,
    roots: &[NodeId],
    placement_for: P,
    shares_storage: S,
) -> Vec<NodeId>
where
    P: Fn(NodeId) -> Option<DeviceLocation>,
    S: Fn(DeviceLocation, DeviceLocation) -> bool,
{
    let order = topo_order_multi(graph, roots);
    // CSE on inserted copies: `(producer, target_device) → Op::Copy NodeId`.
    let mut copy_cache: HashMap<(NodeId, DeviceLocation), NodeId> = HashMap::new();
    let mut inserted: Vec<NodeId> = Vec::new();

    // Two passes: first compute the rewires we want, then apply
    // them. Splitting avoids borrow issues with graph mutation
    // mid-iteration and keeps the iteration order stable.
    let mut rewires: Vec<(NodeId, NodeId, NodeId)> = Vec::new();

    for &consumer_id in &order {
        let consumer_placement = match placement_for(consumer_id) {
            Some(p) => p,
            None => continue,
        };
        // Skip ops that are themselves transfers — `Op::Copy` /
        // `Op::Move` exist precisely TO bridge devices; inserting
        // another copy on their input would be infinite-regress.
        let consumer_op = &graph.node(consumer_id).op;
        if matches!(consumer_op, Op::Copy { .. } | Op::Move { .. }) {
            continue;
        }

        // Snapshot inputs so we don't reborrow during the loop.
        let inputs: Vec<NodeId> = graph.node(consumer_id).inputs.clone();
        for producer_id in inputs {
            let producer_placement = match placement_for(producer_id) {
                Some(p) => p,
                None => continue,
            };
            if shares_storage(producer_placement, consumer_placement) {
                continue;
            }

            let copy_id = match copy_cache.get(&(producer_id, consumer_placement)) {
                Some(&id) => id,
                None => {
                    let producer = graph.node(producer_id);
                    let shape = producer.shape.clone();
                    let dtype = producer.dtype;
                    let id = graph.push(Node {
                        op: Op::Copy { target: consumer_placement },
                        inputs: vec![producer_id],
                        shape,
                        dtype,
                    });
                    // The inserted copy itself sits on the
                    // consumer's device — its output is what the
                    // consumer reads.
                    graph.set_placement(id, consumer_placement);
                    copy_cache.insert((producer_id, consumer_placement), id);
                    inserted.push(id);
                    id
                }
            };
            rewires.push((consumer_id, producer_id, copy_id));
        }
    }

    for (consumer, old_input, new_input) in rewires {
        graph.rewrite_input(consumer, old_input, new_input);
    }

    inserted
}

/// Phase 2.2 of the picker arc — insert `Op::Contiguize` before any
/// kernel-bearing node whose chosen kernel doesn't accept strided
/// inputs but whose live input layout is non-contiguous (strided,
/// offset, or both).
///
/// # When to call
///
/// After Phase 2.1's `insert_cross_device_copies` and after the
/// picker has committed to per-node kernel choices. The picker's
/// commitment determines the `KernelCaps::strided_input` flag for
/// each node; this pass reads that flag via the `kernel_accepts_strided`
/// callback and inserts layout-fixups where the flag is false and
/// the input is strided.
///
/// # Callback contract
///
/// `kernel_accepts_strided(consumer_id) -> bool` returns `true` if
/// the kernel resolved for that node tolerates strided / offset
/// inputs (i.e. its `KernelCaps::strided_input == true`). The
/// callback is opaque to this pass — typical wiring reads the
/// picker's commit-to-winner alternative set and queries the
/// winner's caps. When `false`, this pass inserts `Op::Contiguize`
/// on every non-contiguous input.
///
/// # Skips
///
/// - View ops (`Transpose`, `Permute`, `BroadcastTo`, `Slice`,
///   `Unsqueeze`, `Squeeze`, `Flip`) — they preserve strided
///   layouts intentionally.
/// - Structural ops (`Const`, `Release`, `Reshape`, `Contiguize`,
///   `Copy`, `Move`, `Alloc`, `ZeroFill`, `WriteSlice`) — their
///   executor arms handle strided naturally or are not kernel-
///   bearing.
///
/// # CSE
///
/// When two consumers share a non-contiguous input AND both kernels
/// reject strided, one `Op::Contiguize` node serves both. The
/// per-pass cache is keyed on producer NodeId (no target-device
/// dimension — Contiguize is device-inheriting).
///
/// # Idempotence
///
/// Running this pass on an already-rewritten graph adds zero new
/// nodes: an inserted `Op::Contiguize` has a contiguous + zero-offset
/// output layout, so the subsequent fixup check (`is_contiguous &&
/// start_offset == 0`) succeeds and the input is skipped.
///
/// Returns the number of `Op::Contiguize` nodes inserted.
pub fn insert_layout_fixups<F>(
    graph: &mut Graph,
    roots: &[NodeId],
    kernel_accepts_strided: F,
) -> usize
where
    F: Fn(NodeId) -> bool,
{
    let order = topo_order_multi(graph, roots);
    // CSE: producer NodeId → Contiguize NodeId. No device dim since
    // Contiguize inherits the producer's residency.
    let mut fixup_cache: HashMap<NodeId, NodeId> = HashMap::new();
    let mut inserted = 0usize;
    let mut rewires: Vec<(NodeId, NodeId, NodeId)> = Vec::new();

    for &consumer_id in &order {
        // Snapshot the consumer to release the borrow before
        // mutation.
        let (consumer_op, consumer_inputs) = {
            let node = graph.node(consumer_id);
            (node.op.clone(), node.inputs.clone())
        };

        // Skip non-kernel-bearing nodes — their executor arms
        // either preserve strided layouts (view ops, Reshape) or
        // are structural with no kernel to mismatch on (Const,
        // Release, Copy, Move, Alloc, ZeroFill, WriteSlice,
        // Contiguize).
        if consumer_op.is_view_op()
            || matches!(
                consumer_op,
                Op::Const
                    | Op::Release
                    | Op::Reshape(_)
                    | Op::Contiguize
                    | Op::Copy { .. }
                    | Op::Move { .. }
                    | Op::Alloc { .. }
                    | Op::ZeroFill
                    | Op::WriteSlice { .. }
            )
        {
            continue;
        }

        // The kernel accepts strided inputs directly — no fixup.
        if kernel_accepts_strided(consumer_id) {
            continue;
        }

        for input_id in consumer_inputs {
            let input_layout = graph.layout(input_id);
            if input_layout.is_contiguous() && input_layout.start_offset() == 0 {
                continue;
            }

            let fixup_id = match fixup_cache.get(&input_id) {
                Some(&id) => id,
                None => {
                    let (shape, dtype) = {
                        let p = graph.node(input_id);
                        (p.shape.clone(), p.dtype)
                    };
                    let id = graph.push(Node {
                        op: Op::Contiguize,
                        inputs: vec![input_id],
                        shape,
                        dtype,
                    });
                    fixup_cache.insert(input_id, id);
                    inserted += 1;
                    id
                }
            };
            rewires.push((consumer_id, input_id, fixup_id));
        }
    }

    for (consumer, old_input, new_input) in rewires {
        graph.rewrite_input(consumer, old_input, new_input);
    }

    inserted
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
    fn simplifies_identity_cast() {
        let (graph, a) = make_scalar_graph();
        // Cast to the same dtype is a no-op; the optimizer aliases it away.
        // (Dispatch keys casts on [src, dst] and registers no [d, d] kernel,
        // so this elision is what removes a same-dtype cast before realize.)
        let same = a.cast(a.dtype());
        let new_roots = optimize(&graph, &[same.id()]);
        assert_eq!(
            new_roots[0],
            a.id(),
            "Cast(d) on a dtype-d input should alias to input",
        );
    }

    #[test]
    fn keeps_real_cast() {
        let (graph, a) = make_scalar_graph();
        // A genuine dtype change (F32→F64) is NOT elided.
        let cast = a.cast(fuel_core_types::DType::F64);
        let new_roots = optimize(&graph, &[cast.id()]);
        assert_ne!(new_roots[0], a.id(), "a real Cast must survive optimization");
    }

    #[test]
    fn declarative_pattern_matches_and_fuses_relu_add() {
        use crate::jit::{OpAttrs, OpTag, PatternNode};
        use crate::registry::{
            BackwardKind, FusedOpEntry, FusedOpFamily, FusedOpParams, FusedOps, PatternTree,
            SubgraphPattern,
        };
        fn shape_rule(s: &[Shape], _: &FusedOpParams) -> Shape {
            s[0].clone()
        }
        fn dtype_rule(d: &[DType], _: &FusedOpParams) -> DType {
            d[0]
        }
        fn no_decompose(_: &mut Graph, id: NodeId, _: &FusedOpParams) -> NodeId {
            id
        }
        // A declaratively-registered fused op for relu(add(a, b)).
        let entry = FusedOpEntry {
            id: FusedOps::SOFTMAX_LAST_DIM, // a real parameterless id (plumbing test)
            name: "test_declarative_relu_add",
            family: FusedOpFamily::Forward,
            pattern: SubgraphPattern::Declarative(PatternTree {
                root: PatternNode::Op {
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
                },
                params: FusedOpParams::SoftmaxLastDim,
            }),
            decompose: no_decompose,
            backward: BackwardKind::NotDifferentiable,
            shape_rule,
            dtype_rule,
            output_views: None,
        };
        let rule = FusionRule::from_entry(&entry);

        // Build relu(add(a, b)).
        let mut g = Graph::new();
        let s = Shape::from_dims(&[2]);
        let f32 = DType::F32;
        let a = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: f32 });
        let b = g.push(Node { op: Op::Const, inputs: vec![], shape: s.clone(), dtype: f32 });
        let sum = g.push(Node { op: Op::Add, inputs: vec![a, b], shape: s.clone(), dtype: f32 });
        let r = g.push(Node { op: Op::Relu, inputs: vec![sum], shape: s.clone(), dtype: f32 });

        // The declarative rule fires at the relu sink and fuses to Op::Fused.
        assert!(rule.matches(&g, r), "declarative pattern matches relu(add)");
        let mut remap = HashMap::new();
        rule.rewrite(&mut g, r, &mut remap);
        let fused_id = remap[&r];
        match &g.node(fused_id).op {
            Op::Fused(id, FusedOpParams::SoftmaxLastDim) => {
                assert_eq!(*id, FusedOps::SOFTMAX_LAST_DIM)
            }
            other => panic!("expected Op::Fused(SOFTMAX_LAST_DIM, _), got {other:?}"),
        }
        assert_eq!(
            g.node(fused_id).inputs,
            vec![a, b],
            "fused inputs = the region's bound external inputs",
        );
    }

    #[test]
    fn runtime_op_fuses_and_round_trips_through_the_pass() {
        use crate::jit::{OpAttrs, OpTag, PatternNode};
        use crate::runtime_fused::register_runtime_fused;
        // A region no other test registers — tanh(sub(a, b)) — so the fuse is
        // deterministic (only our runtime rule matches it; sub is non-commutative).
        let region = PatternNode::Op {
            op: OpTag::Tanh,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Sub,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        };
        let rid = register_runtime_fused("test::tanh_sub", region).unwrap();

        // Build tanh(sub(a, b)) on primitives (b shares a's graph).
        let a = Tensor::from_f32(vec![0.5, -0.5, 1.0, -1.0], Shape::from_dims(&[4]), cpu_dev());
        let b = a.const_f32_like(vec![0.1, 0.2, 0.3, 0.4], Shape::from_dims(&[4]));
        let y = a.sub(&b).tanh();
        let graph = y.graph().clone();

        // default_rules: lower (no-op here) then fuse — the runtime declarative
        // FusionRule folds tanh(sub) → Op::Fused(rid, Runtime).
        let fused = RuleRegistry::default_rules().optimize_to_fixpoint(&graph, &[y.id()]);
        {
            let g = graph.read().unwrap();
            match &g.node(fused[0]).op {
                Op::Fused(fid, FusedOpParams::Runtime { scalars }) => {
                    assert_eq!(*fid, rid, "fused to our runtime op");
                    assert!(fid.is_runtime());
                    assert!(scalars.is_empty(), "parameterless region");
                }
                other => panic!("expected runtime Op::Fused, got {other:?}"),
            }
        }

        // lowering_only: the kernel-absent path — decompose the runtime op back
        // to its region on primitives (tanh(sub(a, b))).
        let lowered = RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &fused);
        let g = graph.read().unwrap();
        assert!(matches!(g.node(lowered[0]).op, Op::Tanh), "re-emitted sink is Tanh");
        let sub_id = g.node(lowered[0]).inputs[0];
        assert!(matches!(g.node(sub_id).op, Op::Sub), "Tanh's input is Sub");
        assert_eq!(g.node(sub_id).inputs.len(), 2, "Sub over the two bound inputs");
    }

    #[test]
    fn capability_gate_blocks_fusion_without_a_kernel() {
        use crate::jit::{OpAttrs, OpTag, PatternNode};
        use crate::runtime_fused::register_runtime_fused;
        // Unique region — sigmoid(div(a, b)) — so the gate decision is ours alone.
        let region = PatternNode::Op {
            op: OpTag::Sigmoid,
            attrs: OpAttrs::default(),
            operands: vec![PatternNode::Op {
                op: OpTag::Div,
                attrs: OpAttrs::default(),
                operands: vec![PatternNode::Bind { index: 0 }, PatternNode::Bind { index: 1 }],
            }],
        };
        let rid = register_runtime_fused("test::sigmoid_div", region).unwrap();

        let a = Tensor::from_f32(vec![1.0, 2.0, 3.0, 4.0], Shape::from_dims(&[4]), cpu_dev());
        let b = a.const_f32_like(vec![2.0, 2.0, 2.0, 2.0], Shape::from_dims(&[4]));
        let y = a.div(&b).sigmoid();
        let graph = y.graph().clone();

        // No kernel for our op → it is gated out of fusion (reported) and its
        // region stays primitive — the miss is caught at match time, not after.
        let (rules, gated_out) = RuleRegistry::capability_gated_rules(|id| id != rid);
        assert!(gated_out.contains(&rid), "kernel-absent op is reported as a miss candidate");
        let roots = rules.optimize_to_fixpoint(&graph, &[y.id()]);
        {
            let g = graph.read().unwrap();
            assert!(
                matches!(g.node(roots[0]).op, Op::Sigmoid),
                "kernel-absent runtime op stays primitive (Sigmoid sink), never fuses",
            );
        }

        // With a kernel available, the same op fuses.
        let (rules2, _) = RuleRegistry::capability_gated_rules(|_| true);
        let roots2 = rules2.optimize_to_fixpoint(&graph, &[roots[0]]);
        let g = graph.read().unwrap();
        assert!(
            matches!(g.node(roots2[0]).op, Op::Fused(fid, _) if fid == rid),
            "with a kernel, the runtime op fuses",
        );
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
    fn derive_ordering_follows_view_chains_for_destructive_op() {
        // Phase 4a of in-place ops infrastructure: a destructive op
        // on `x` must run after any reader of any view-of-x, because
        // views share `x`'s Storage Arc at realize time. Without the
        // alias-set walk in derive_ordering, this case would not pin
        // `reader(view)` before `destructive(x)`.
        //
        // Graph:
        //   x       (producer; rank-2)
        //   v       = x.transpose()    (view of x)
        //   z       = v.relu()         (reader of v, transitively reads x's bytes)
        //   r       = x.release()      (destructive on x)
        // Expected: r must run after z (not just after v).
        let x = Tensor::from_f32(
            vec![1.0_f32, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), cpu_dev(),
        );
        let v = x.transpose();
        let z = v.relu();
        let r = x.release();
        let ord = derive_ordering(&x.graph().read().unwrap(), &[z.id(), r.id()]);
        let deps = ord.deps_of(r.id());
        assert!(
            deps.contains(&z.id()),
            "release(x) must be pinned after relu(transpose(x)) (view-aware alias set); got deps = {deps:?}",
        );
    }

    #[test]
    fn derive_ordering_inplace_unary_through_view_chain() {
        // Specifically for the in-place ops Phase 4 use case:
        // `y = x.transpose(); z = y.relu(); x.relu_inplace()` —
        // ReluInplace must be pinned after the read through the
        // transpose view chain.
        let x = Tensor::from_f32(
            vec![-1.0_f32, 2.0, -3.0, 4.0], Shape::from_dims(&[2, 2]), cpu_dev(),
        );
        let y = x.transpose();
        let z = y.relu();
        // Capture x's shape + dtype + id BEFORE acquiring the write
        // lock — otherwise the args evaluate inside the locked region
        // and self-deadlock (RwLock doesn't grant a read lock to the
        // writer-holding thread).
        let x_shape = x.shape();
        let x_dtype = x.dtype();
        let x_id = x.id();
        let inplace_id = x.graph().write().unwrap().push(crate::Node {
            op: crate::Op::ReluInplace,
            inputs: vec![x_id],
            shape: x_shape,
            dtype: x_dtype,
        });
        let ord = derive_ordering(&x.graph().read().unwrap(), &[z.id(), inplace_id]);
        let deps = ord.deps_of(inplace_id);
        assert!(
            deps.contains(&z.id()),
            "ReluInplace(x) must be pinned after Relu(Transpose(x)) via view-aware alias set; got deps = {deps:?}",
        );
    }

    // ---- Phase 5: insert_safety_copies ----

    #[test]
    fn insert_safety_copies_residual_connection_breaks_cycle() {
        // Canonical residual pattern: `y = x.relu_inplace(); z = y + x`.
        // Before Phase 5: derive_ordering pins ReluInplace after Add
        // (Add reads x), but data flow pins Add after ReluInplace (Add
        // reads y, y depends on ReluInplace) → cycle. After Phase 5:
        // a Copy(x) → x_safe is inserted and Add's x input rewires to
        // x_safe.
        let x = Tensor::from_f32(
            vec![1.0_f32, -2.0, 3.0, -4.0], Shape::from_dims(&[4]), cpu_dev(),
        );
        let x_shape = x.shape();
        let x_dtype = x.dtype();
        let x_id = x.id();

        // Manually build: y = ReluInplace(x); z = Add(y, x).
        let (y_id, z_id) = {
            let mut g = x.graph().write().unwrap();
            let y_id = g.push(crate::Node {
                op: crate::Op::ReluInplace,
                inputs: vec![x_id],
                shape: x_shape.clone(),
                dtype: x_dtype,
            });
            let z_id = g.push(crate::Node {
                op: crate::Op::Add,
                inputs: vec![y_id, x_id],
                shape: x_shape,
                dtype: x_dtype,
            });
            (y_id, z_id)
        };

        // Before the pass: cycle would exist. Run insert_safety_copies.
        let inserted = insert_safety_copies(
            &mut x.graph().write().unwrap(),
            &[z_id],
        );
        assert_eq!(inserted, 1, "should insert exactly one safety copy");

        // Verify Add's inputs were rewired: still [y_id, <something>],
        // but the second input is NOT x_id anymore — it's the new
        // Op::Copy node.
        let copy_id = {
            let g = x.graph().read().unwrap();
            let z_node = g.node(z_id);
            assert_eq!(z_node.inputs.len(), 2, "Add still has 2 inputs");
            assert_eq!(z_node.inputs[0], y_id, "Add's y input unchanged");
            assert_ne!(z_node.inputs[1], x_id, "Add's x input was rewired");
            let copy_node_id = z_node.inputs[1];
            let copy_node = g.node(copy_node_id);
            assert!(matches!(copy_node.op, crate::Op::Copy { .. }),
                "rewired input is Op::Copy; got {:?}", copy_node.op);
            assert_eq!(copy_node.inputs, vec![x_id], "Op::Copy reads x");
            copy_node_id
        };

        // Verify execution_plan succeeds (no cycle).
        let plan = execution_plan(&x.graph().read().unwrap(), &[z_id]);
        // Plan must contain x, the copy, ReluInplace, and Add.
        // The copy must come before ReluInplace (it's a reader of x).
        let pos: HashMap<NodeId, usize> = plan.iter().enumerate().map(|(i, n)| (*n, i)).collect();
        let copy_pos = pos[&copy_id];
        let inplace_pos = pos[&y_id];
        assert!(copy_pos < inplace_pos,
            "Copy must run before ReluInplace; copy={copy_pos} inplace={inplace_pos}");
    }

    #[test]
    fn insert_safety_copies_noop_when_no_destructive_conflicts() {
        // Pure functional graph — no in-place ops. The pass should
        // insert zero copies.
        let x = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let y = x.relu();
        let z = y.sum_all();
        let inserted = insert_safety_copies(
            &mut x.graph().write().unwrap(),
            &[z.id()],
        );
        assert_eq!(inserted, 0);
    }

    #[test]
    fn insert_safety_copies_noop_when_inplace_target_only_consumed_via_alias_path() {
        // `y = x.relu_inplace(); loss = y.sum_all()` — the only
        // downstream consumer of x reads it through y (the in-place
        // node), not x directly. derive_ordering pins ReluInplace
        // after... only ReluInplace itself (no other readers of x).
        // No conflict, no copy needed.
        let x = Tensor::from_f32(vec![1.0_f32, -1.0], Shape::from_dims(&[2]), cpu_dev());
        let x_shape = x.shape();
        let x_dtype = x.dtype();
        let x_id = x.id();
        let y_id = x.graph().write().unwrap().push(crate::Node {
            op: crate::Op::ReluInplace,
            inputs: vec![x_id],
            shape: x_shape,
            dtype: x_dtype,
        });
        // Build a chain that only reads y (not x).
        let inserted = insert_safety_copies(
            &mut x.graph().write().unwrap(),
            &[y_id],
        );
        assert_eq!(inserted, 0);
    }

    /// Regression for the 2026-06-11 live-GPU failure: a reader with
    /// NO dependency relationship to a destructive `Op::Move` must
    /// not get a safety copy just because DFS visit order happened to
    /// place it after the Move.
    ///
    /// Graph: `a; b = relu(a); m = move(a)`. derive_ordering pins
    /// `m` after `b` (reader-before-mutator), so every execution_plan
    /// runs `b` before `m`'s release — no copy is needed in EITHER
    /// roots order. The old positional predicate inserted a spurious
    /// copy when roots ordering made `b` land after `m` in
    /// `topo_order_multi` (which visits the LAST root's subtree
    /// first), and that copy then inherited the Move's device.
    #[test]
    fn insert_safety_copies_no_spurious_copy_for_independent_reader_of_move() {
        for flip_roots in [false, true] {
            let a = Tensor::from_f32(
                vec![1.0_f32, -2.0], Shape::from_dims(&[2]), cpu_dev(),
            );
            let b = a.relu();
            let a_shape = a.shape();
            let a_dtype = a.dtype();
            let a_id = a.id();
            let m_id = a.graph().write().unwrap().push(crate::Node {
                op: crate::Op::Move { target: DeviceLocation::Cpu },
                inputs: vec![a_id],
                shape: a_shape,
                dtype: a_dtype,
            });
            let roots = if flip_roots { [m_id, b.id()] } else { [b.id(), m_id] };
            // Sanity: in the `[b, m]` roots order, DFS visits m's
            // subtree first, so b lands positionally AFTER the Move —
            // the exact shape the old predicate misclassified.
            let inserted = insert_safety_copies(
                &mut a.graph().write().unwrap(),
                &roots,
            );
            assert_eq!(
                inserted, 0,
                "relu(a) is pinned before move(a) by derive_ordering; \
                 no copy regardless of roots order (flip_roots={flip_roots})",
            );
            let g = a.graph().read().unwrap();
            assert_eq!(
                g.node(b.id()).inputs, vec![a_id],
                "reader must keep reading `a` directly",
            );
            drop(g);
            // The plan must still order the reader before the Move.
            let plan = execution_plan(&a.graph().read().unwrap(), &roots);
            let b_pos = plan.iter().position(|&n| n == b.id()).unwrap();
            let m_pos = plan.iter().position(|&n| n == m_id).unwrap();
            assert!(b_pos < m_pos, "relu must precede move: {plan:?}");
        }
    }

    /// A reader that TRANSITIVELY depends on the destructive op (not
    /// just via a direct input) still conflicts: the pin edge would
    /// close a cycle through the intermediate node. The
    /// dependency-based predicate must use reachability, not direct
    /// input inspection.
    ///
    /// Graph: `y = relu_inplace(x); w = relu(y); z = add(w, x)`.
    #[test]
    fn insert_safety_copies_transitive_descendant_reader_gets_copy() {
        let x = Tensor::from_f32(
            vec![1.0_f32, -2.0, 3.0, -4.0], Shape::from_dims(&[4]), cpu_dev(),
        );
        let x_shape = x.shape();
        let x_dtype = x.dtype();
        let x_id = x.id();
        let (y_id, z_id) = {
            let mut g = x.graph().write().unwrap();
            let y_id = g.push(crate::Node {
                op: crate::Op::ReluInplace,
                inputs: vec![x_id],
                shape: x_shape.clone(),
                dtype: x_dtype,
            });
            let w_id = g.push(crate::Node {
                op: crate::Op::Relu,
                inputs: vec![y_id],
                shape: x_shape.clone(),
                dtype: x_dtype,
            });
            let z_id = g.push(crate::Node {
                op: crate::Op::Add,
                inputs: vec![w_id, x_id],
                shape: x_shape,
                dtype: x_dtype,
            });
            (y_id, z_id)
        };
        let inserted = insert_safety_copies(
            &mut x.graph().write().unwrap(),
            &[z_id],
        );
        assert_eq!(inserted, 1, "transitive residual cycle needs one copy");
        let g = x.graph().read().unwrap();
        let z_node = g.node(z_id);
        assert_ne!(z_node.inputs[1], x_id, "z's x input must be rewired");
        assert!(matches!(g.node(z_node.inputs[1]).op, crate::Op::Copy { .. }));
        drop(g);
        let plan = execution_plan(&x.graph().read().unwrap(), &[z_id]);
        assert!(plan.contains(&y_id), "plan still contains the in-place op");
    }

    /// A genuinely-parallel reader — no ordering guarantee in either
    /// direction — keeps its safety copy (conservative default).
    ///
    /// The only readers derive_ordering does NOT pin before the
    /// mutator are alias-set members (view ops of the target). A view
    /// with consumers is transitively forced before the mutator via
    /// its pinned consumers; a view that is itself a realize root has
    /// no ordering guarantee at all. Safety cannot be proven, so the
    /// pass must copy. Note the old positional predicate got this
    /// case WRONG in the `[m, v]` roots order (v lands positionally
    /// before m → no copy; only the plan's stable tiebreak saved it).
    #[test]
    fn insert_safety_copies_parallel_view_root_reader_keeps_copy() {
        let x = Tensor::from_f32(
            vec![1.0_f32, 2.0, 3.0, 4.0], Shape::from_dims(&[2, 2]), cpu_dev(),
        );
        let v = x.transpose();
        let x_shape = x.shape();
        let x_dtype = x.dtype();
        let x_id = x.id();
        let m_id = x.graph().write().unwrap().push(crate::Node {
            op: crate::Op::Move { target: DeviceLocation::Cpu },
            inputs: vec![x_id],
            shape: x_shape,
            dtype: x_dtype,
        });
        // Roots [m, v]: DFS visits v's subtree first, so v sits
        // positionally BEFORE the Move — the shape the positional
        // predicate silently skipped.
        let inserted = insert_safety_copies(
            &mut x.graph().write().unwrap(),
            &[m_id, v.id()],
        );
        assert_eq!(
            inserted, 1,
            "unpinned view-root reader has no ordering guarantee → copy",
        );
        let g = x.graph().read().unwrap();
        let v_node = g.node(v.id());
        assert_ne!(v_node.inputs[0], x_id, "view must be rewired to the snapshot");
        assert!(matches!(g.node(v_node.inputs[0]).op, crate::Op::Copy { .. }));
    }

    /// Cycles through ANOTHER destructive op's ordering edge are
    /// still conflicts — the predicate must walk the combined
    /// (data + ordering) graph, not just data edges.
    ///
    /// Graph:
    ///   m1 = move(t1); m2 = move(t2)
    ///   x  = add(m1, t2)   (pin x→m2; data m1→x)
    ///   r  = add(m2, t1)   (pin r→m1; data m2→r)
    /// Combined cycle: m1 → x → m2 → r → m1. Neither x nor r is a
    /// pure-data descendant of the op whose target it reads, yet
    /// both pins are unsatisfiable. Both need copies; afterwards
    /// execution_plan must succeed.
    #[test]
    fn insert_safety_copies_breaks_cycle_through_other_ordering_edges() {
        let t1 = Tensor::from_f32(vec![1.0_f32, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let t2 = t1.const_f32_like(vec![3.0_f32, 4.0], Shape::from_dims(&[2]));
        let shape = t1.shape();
        let dtype = t1.dtype();
        let (t1_id, t2_id) = (t1.id(), t2.id());
        let (x_id, r_id) = {
            let mut g = t1.graph().write().unwrap();
            let m1 = g.push(crate::Node {
                op: crate::Op::Move { target: DeviceLocation::Cpu },
                inputs: vec![t1_id],
                shape: shape.clone(), dtype,
            });
            let m2 = g.push(crate::Node {
                op: crate::Op::Move { target: DeviceLocation::Cpu },
                inputs: vec![t2_id],
                shape: shape.clone(), dtype,
            });
            let x = g.push(crate::Node {
                op: crate::Op::Add,
                inputs: vec![m1, t2_id],
                shape: shape.clone(), dtype,
            });
            let r = g.push(crate::Node {
                op: crate::Op::Add,
                inputs: vec![m2, t1_id],
                shape, dtype,
            });
            (x, r)
        };
        let inserted = insert_safety_copies(
            &mut t1.graph().write().unwrap(),
            &[x_id, r_id],
        );
        assert_eq!(inserted, 2, "both cross-readers sit on the combined cycle");
        // The rewritten graph must be plannable (no cycle panic).
        let plan = execution_plan(&t1.graph().read().unwrap(), &[x_id, r_id]);
        assert!(plan.contains(&x_id) && plan.contains(&r_id));
    }

    /// Idempotence under the dependency-based predicate: a second run
    /// sees the inserted Copy as a pinned, acyclic reader of the
    /// target and inserts nothing new.
    #[test]
    fn insert_safety_copies_idempotent_after_rewrite() {
        let x = Tensor::from_f32(
            vec![1.0_f32, -2.0], Shape::from_dims(&[2]), cpu_dev(),
        );
        let x_shape = x.shape();
        let x_dtype = x.dtype();
        let x_id = x.id();
        let z_id = {
            let mut g = x.graph().write().unwrap();
            let y_id = g.push(crate::Node {
                op: crate::Op::ReluInplace,
                inputs: vec![x_id],
                shape: x_shape.clone(),
                dtype: x_dtype,
            });
            g.push(crate::Node {
                op: crate::Op::Add,
                inputs: vec![y_id, x_id],
                shape: x_shape,
                dtype: x_dtype,
            })
        };
        let first = insert_safety_copies(&mut x.graph().write().unwrap(), &[z_id]);
        assert_eq!(first, 1);
        let second = insert_safety_copies(&mut x.graph().write().unwrap(), &[z_id]);
        assert_eq!(second, 0, "second run must be a no-op");
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
        //   - 2 new nodes (move, reload) appended
        //   - c's input list now has `reload_id` instead of `a.id()`
        //   - b's input list STILL has `a.id()` (unchanged)
        let a = Tensor::from_f32(vec![1.0_f32, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.relu();
        let c = a.neg();

        let (move_id, reload_id) = insert_evict_reload(
            a.graph(), a.id(), DeviceLocation::Cpu, &[c.id()],
        );

        let g = a.graph().read().unwrap();
        // mv is an Op::Move{Cpu} reading a — the fused stage-to-host +
        // destructive release of a's device storage.
        match &g.node(move_id).op {
            Op::Move { target: DeviceLocation::Cpu } => {},
            other => panic!("expected Op::Move{{Cpu}}, got {other:?}"),
        }
        assert_eq!(g.node(move_id).inputs, vec![a.id()]);
        assert_eq!(
            g.node(move_id).op.destructive_input(), Some(0),
            "Move must destroy its source so the device storage frees",
        );
        // reload is an Op::Copy{Cpu} (src_device passed in) reading mv
        assert!(matches!(g.node(reload_id).op, Op::Copy { .. }));
        assert_eq!(g.node(reload_id).inputs, vec![move_id]);
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
        // Phase 7.6 step 4: emission is the registry-extended shape.
        let any_fused = g.nodes.iter().any(|n| matches!(
            n.op,
            Op::Fused(fid, _) if fid == crate::registry::FusedOps::FUSED_LINEAR,
        ));
        assert!(any_fused, "graph should contain an Op::Fused(FUSED_LINEAR) node");
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
        // Phase 7.6 step 5: post-migration the builder emits
        // `Op::Fused(SOFTMAX_LAST_DIM, _)` only — the legacy
        // `Op::SoftmaxLastDim` variant was retired.
        let n_softmax     = count_op_in_reachable(&graph, &[new_root], |op| {
            matches!(op, Op::Fused(fid, _) if *fid == FusedOps::SOFTMAX_LAST_DIM)
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

    /// G2 (2026-06-20): FlashAttn IS decomposable — the vanilla (non-causal,
    /// MHA) case lowers to its materialized scaled-dot-product-attention
    /// subgraph. Born-red before the fix (`flash_attn::decompose` panicked).
    #[test]
    fn flash_attn_vanilla_decomposes_to_sdpa() {
        let s = Shape::from_dims(&[1, 1, 1, 1]);
        let q = Tensor::from_f32(vec![0.0_f32; 1], s.clone(), cpu_dev());
        let k = q.const_f32_like(vec![0.0_f32; 1], s.clone());
        let v = q.const_f32_like(vec![0.0_f32; 1], s.clone());
        // Vanilla: non-causal, Hq==Hkv==1, no alibi/window/softcap.
        let attn = q.flash_attn(&k, &v, None, 1.0_f32, false, None, None, None);
        let graph = attn.graph().clone();

        let new_roots =
            RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[attn.id()]);
        assert_eq!(new_roots.len(), 1);
        let root = new_roots[0];
        let g = graph.read().unwrap();
        // The fused FlashAttn is gone from the root, replaced by the SDPA
        // subgraph — which contains MatMul nodes.
        assert!(
            !matches!(g.node(root).op, Op::Fused(fid, _) if fid == crate::registry::FusedOps::FLASH_ATTN),
            "vanilla FlashAttn must decompose, not stay fused",
        );
        let has_matmul = (0..g.len()).any(|i| matches!(g.node(NodeId(i)).op, Op::MatMul));
        assert!(has_matmul, "decomposed FlashAttn must contain a MatMul");
    }

    /// G2: causal FlashAttn decomposes too — the causal mask is an additive
    /// `-inf` `Triu` band (no new primitive needed), so the node lowers
    /// rather than staying fused. (The decomposed *numerics* are verified
    /// separately by a parity test vs. the reference attention.)
    #[test]
    fn flash_attn_causal_decomposes_with_triu_mask() {
        let s = Shape::from_dims(&[1, 1, 2, 1]); // Sq=Sk=2 so the mask is non-trivial
        let q = Tensor::from_f32(vec![0.0_f32; 2], s.clone(), cpu_dev());
        let k = q.const_f32_like(vec![0.0_f32; 2], s.clone());
        let v = q.const_f32_like(vec![0.0_f32; 2], s.clone());
        let attn = q.flash_attn(&k, &v, None, 1.0_f32, true, None, None, None); // causal
        let graph = attn.graph().clone();

        let new_roots =
            RuleRegistry::lowering_only().optimize_to_fixpoint(&graph, &[attn.id()]);
        assert_eq!(new_roots.len(), 1);
        let root = new_roots[0];
        let g = graph.read().unwrap();
        assert!(
            !matches!(g.node(root).op, Op::Fused(fid, _) if fid == crate::registry::FusedOps::FLASH_ATTN),
            "causal FlashAttn must decompose (Triu mask), not stay fused",
        );
        let has_triu = (0..g.len()).any(|i| matches!(g.node(NodeId(i)).op, Op::Triu { .. }));
        assert!(has_triu, "causal decomposition must contain a Triu mask band");
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

    // -------- Cast-fusion rule -------------------------------------------

    /// Build a "yes to everything" capability predicate. Useful when
    /// the test wants to confirm the matcher/rewriter mechanics work;
    /// orthogonal tests cover the predicate-says-no path.
    fn allow_all_predicate() -> CapabilityPredicate {
        Arc::new(|_op: &Op, _dtypes: &[fuel_core_types::DType]| true)
    }

    /// Predicate that says no for everything. Useful for confirming
    /// the rule respects the capability gate.
    fn deny_all_predicate() -> CapabilityPredicate {
        Arc::new(|_op: &Op, _dtypes: &[fuel_core_types::DType]| false)
    }

    /// Happy path: `x:f32 → Cast(BF16) → Neg(BF16)` collapses to
    /// `x:f32 → Neg(f32)` when the predicate says Neg supports f32.
    #[test]
    fn cast_fusion_drops_cast_when_consumer_supports_source_dtype() {
        let x = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        let xc = x.cast(fuel_core_types::DType::BF16);
        let y = xc.neg();
        let graph = y.graph().clone();
        let pre_len = graph.read().unwrap().len();

        let registry = RuleRegistry::new()
            .with_rule(Box::new(CastFusionRule::new(allow_all_predicate())));
        let new_roots = registry.optimize_to_fixpoint(&graph, &[y.id()]);
        assert_eq!(new_roots.len(), 1);
        let new_root = new_roots[0];

        let g = graph.read().unwrap();
        // The new root is a fresh Neg whose input is the original
        // f32 Const x, not the Cast.
        assert!(matches!(g.node(new_root).op, Op::Neg));
        assert_eq!(g.node(new_root).inputs, vec![x.id()]);
        // Cast node remains in the arena but is no longer reachable
        // from the new root.
        let reachable = topo_order_multi(&g, &[new_root]);
        assert!(!reachable.iter().any(|&n| matches!(g.node(n).op, Op::Cast(_))),
            "no Cast node should be reachable post-fusion");
        // One new Neg node was appended; the original Neg + Cast are
        // unreachable but still in the arena.
        assert!(g.len() > pre_len);
    }

    /// Capability gate: when the predicate says no, the rule must
    /// not fire. Graph structure is unchanged.
    #[test]
    fn cast_fusion_no_fire_when_predicate_denies() {
        let x = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        let xc = x.cast(fuel_core_types::DType::BF16);
        let y = xc.neg();
        let graph = y.graph().clone();
        let pre_len = graph.read().unwrap().len();

        let registry = RuleRegistry::new()
            .with_rule(Box::new(CastFusionRule::new(deny_all_predicate())));
        let new_roots = registry.optimize_to_fixpoint(&graph, &[y.id()]);
        assert_eq!(new_roots, vec![y.id()]);
        assert_eq!(graph.read().unwrap().len(), pre_len,
            "predicate-says-no should append no nodes");
        // The original Cast → Neg chain is intact.
        let g = graph.read().unwrap();
        assert!(matches!(g.node(y.id()).op, Op::Neg));
        assert_eq!(g.node(y.id()).inputs, vec![xc.id()]);
        assert!(matches!(g.node(xc.id()).op, Op::Cast(_)));
    }

    /// Single-consumer constraint: if the Cast feeds two consumers,
    /// the rule must not fire (eliminating the cast for one consumer
    /// would orphan it for the other).
    #[test]
    fn cast_fusion_no_fire_when_cast_has_multiple_consumers() {
        let x = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        let xc = x.cast(fuel_core_types::DType::BF16);
        // Two consumers of the same Cast.
        let y1 = xc.neg();
        let y2 = xc.relu();
        let graph = y1.graph().clone();
        let pre_len = graph.read().unwrap().len();

        let registry = RuleRegistry::new()
            .with_rule(Box::new(CastFusionRule::new(allow_all_predicate())));
        let new_roots = registry.optimize_to_fixpoint(&graph, &[y1.id(), y2.id()]);
        // Roots unchanged; cast survives because rule didn't fire.
        assert_eq!(new_roots, vec![y1.id(), y2.id()]);
        assert_eq!(graph.read().unwrap().len(), pre_len);
        let g = graph.read().unwrap();
        assert!(matches!(g.node(xc.id()).op, Op::Cast(_)));
    }

    /// Output dtype tracks the new input: in aggressive mode the
    /// rewritten consumer's output dtype is the cast-source dtype,
    /// not the original consumer's output dtype. Downstream
    /// consumers reading from the rewritten node will see the new
    /// dtype — by design (see module docstring's
    /// aggressive-semantics caveat).
    #[test]
    fn cast_fusion_rewrites_output_dtype_to_source_dtype() {
        let x = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        let xc = x.cast(fuel_core_types::DType::BF16);
        let y = xc.neg();
        // Pre-fusion the original Neg produces BF16.
        assert_eq!(y.dtype(), fuel_core_types::DType::BF16);

        let graph = y.graph().clone();
        let registry = RuleRegistry::new()
            .with_rule(Box::new(CastFusionRule::new(allow_all_predicate())));
        let new_roots = registry.optimize_to_fixpoint(&graph, &[y.id()]);
        let new_root = new_roots[0];

        let g = graph.read().unwrap();
        // The rewritten Neg consumes the F32 source directly and
        // produces F32. This differs from the original BF16
        // output; the rule's aggressive semantics commit to this.
        assert_eq!(g.node(new_root).dtype, fuel_core_types::DType::F32);
    }

    /// Multi-input op: only the qualifying input fuses; the others
    /// are left alone.
    #[test]
    fn cast_fusion_multi_input_op_fuses_only_qualifying_edge() {
        let a = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        // a:f32 → Cast(BF16) → Add(bf16, b:bf16) where b is already bf16.
        let ac = a.cast(fuel_core_types::DType::BF16);
        let b = a.const_bf16_like(
            vec![half::bf16::from_f32(2.0); 4],
            Shape::from_dims(&[4]),
        );
        let sum = ac.add(&b);
        let graph = sum.graph().clone();

        // Build a predicate that says yes only for Add[f32, bf16, bf16].
        // (Realistic kernels typically wouldn't have mixed-dtype Add;
        // this isolates the matcher's per-input behavior. If the
        // predicate said no, the rule wouldn't fire and `b` would
        // stay paired with the Cast.)
        let predicate: CapabilityPredicate = Arc::new(|op: &Op, _dtypes: &[fuel_core_types::DType]| {
            matches!(op, Op::Add)
        });

        let registry = RuleRegistry::new()
            .with_rule(Box::new(CastFusionRule::new(predicate)));
        let new_roots = registry.optimize_to_fixpoint(&graph, &[sum.id()]);
        let new_root = new_roots[0];

        let g = graph.read().unwrap();
        assert!(matches!(g.node(new_root).op, Op::Add));
        // First input of new Add: the original f32 const (cast eliminated).
        // Second input: still `b` (a bf16 const, no cast was there).
        assert_eq!(g.node(new_root).inputs, vec![a.id(), b.id()]);
    }

    /// Chained casts: `x:f32 → Cast(BF16) → Cast(F32) → Op` should
    /// reduce to `x:f32 → Op` via the fixpoint loop firing the rule
    /// twice (once per consumer-side cast).
    #[test]
    fn cast_fusion_handles_chained_casts_via_fixpoint() {
        let x = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        let x_bf16 = x.cast(fuel_core_types::DType::BF16);
        let x_f32 = x_bf16.cast(fuel_core_types::DType::F32);
        let y = x_f32.neg();
        let graph = y.graph().clone();

        let registry = RuleRegistry::new()
            .with_rule(Box::new(CastFusionRule::new(allow_all_predicate())));
        let new_roots = registry.optimize_to_fixpoint(&graph, &[y.id()]);
        let new_root = new_roots[0];

        let g = graph.read().unwrap();
        // After two iterations: the outer Cast is gone (Neg consumes
        // x_bf16 directly), then the inner Cast is gone (Neg consumes
        // x directly). No Cast nodes are reachable.
        let reachable = topo_order_multi(&g, &[new_root]);
        let cast_count = reachable.iter()
            .filter(|&&n| matches!(g.node(n).op, Op::Cast(_)))
            .count();
        assert_eq!(cast_count, 0,
            "fixpoint should eliminate both casts in the chain");
        assert!(matches!(g.node(new_root).op, Op::Neg));
        assert_eq!(g.node(new_root).inputs, vec![x.id()]);
    }

    /// The rule must not fire on the Cast node itself — only on the
    /// consumer of a Cast.
    #[test]
    fn cast_fusion_does_not_fire_on_cast_node_itself() {
        let x = Tensor::from_f32(vec![1.0_f32; 4], Shape::from_dims(&[4]), cpu_dev());
        let xc = x.cast(fuel_core_types::DType::BF16);
        let graph = xc.graph().clone();
        let pre_len = graph.read().unwrap().len();

        let registry = RuleRegistry::new()
            .with_rule(Box::new(CastFusionRule::new(allow_all_predicate())));
        // Root the optimizer at the Cast itself. The rule should look
        // at the Cast and decline (it's not the consumer's role).
        let new_roots = registry.optimize_to_fixpoint(&graph, &[xc.id()]);
        assert_eq!(new_roots, vec![xc.id()]);
        assert_eq!(graph.read().unwrap().len(), pre_len);
    }

    // ===== Phase 2.1: insert_cross_device_copies tests =====

    use std::collections::HashMap as StdHashMap;

    fn add_node(g: &mut Graph, op: Op, inputs: Vec<NodeId>) -> NodeId {
        g.push(Node {
            op,
            inputs,
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        })
    }

    /// Snapshot every node's `Graph::placement` into an external
    /// map. Needed because `&mut Graph` and `|id| g.placement(id)`
    /// closure can't co-exist (overlapping borrows). The closure-
    /// based oracle is the production shape (the picker hands in a
    /// `Fn(NodeId) -> Option<DeviceLocation>` derived from its own
    /// commit map); tests just precompute the snapshot from the
    /// graph.
    fn snapshot_placements(g: &Graph) -> StdHashMap<NodeId, DeviceLocation> {
        let mut m = StdHashMap::new();
        for id in 0..g.len() {
            let nid = NodeId(id);
            if let Some(loc) = g.placement(nid) {
                m.insert(nid, loc);
            }
        }
        m
    }

    fn build_two_node_graph_with_placements(
        a: DeviceLocation,
        b: DeviceLocation,
    ) -> (Graph, NodeId, NodeId) {
        let mut g = Graph::new();
        let n1 = add_node(&mut g, Op::Const, vec![]);
        let n2 = add_node(&mut g, Op::Neg, vec![n1]);
        g.set_placement(n1, a);
        g.set_placement(n2, b);
        (g, n1, n2)
    }

    fn shares_storage_cpu_devices_only(
        a: DeviceLocation,
        b: DeviceLocation,
    ) -> bool {
        // Test stub topology: CPU shares with CPU only (the AOCL/MKL/
        // portable-CPU substrate story). CUDA shares with same-gpu_id
        // CUDA. Vulkan shares with same-gpu_id Vulkan. Cross device
        // is never shared.
        match (a, b) {
            (DeviceLocation::Cpu, DeviceLocation::Cpu) => true,
            (DeviceLocation::Cuda { gpu_id: x }, DeviceLocation::Cuda { gpu_id: y }) => x == y,
            (DeviceLocation::Vulkan { gpu_id: x }, DeviceLocation::Vulkan { gpu_id: y }) => x == y,
            (DeviceLocation::Metal { gpu_id: x }, DeviceLocation::Metal { gpu_id: y }) => x == y,
            _ => false,
        }
    }

    /// No cross-device edges → no copies inserted.
    #[test]
    fn insert_copies_noop_when_all_same_device() {
        let (mut g, _n1, n2) = build_two_node_graph_with_placements(
            DeviceLocation::Cpu,
            DeviceLocation::Cpu,
        );
        let pre_len = g.len();
        let placements = snapshot_placements(&g);
        let inserted = insert_cross_device_copies(
            &mut g, &[n2],
            |id| placements.get(&id).copied(),
            shares_storage_cpu_devices_only,
        );
        assert_eq!(inserted.len(), 0, "same-device edge → no copy needed");
        assert_eq!(g.len(), pre_len, "graph unchanged");
    }

    /// Cross-device edge (CPU → CUDA) → one `Op::Copy { target: Cuda }`
    /// inserted; consumer's input rewired to point at it.
    #[test]
    fn insert_copies_cuda_consumer_of_cpu_input() {
        let cuda = DeviceLocation::Cuda { gpu_id: 0 };
        let (mut g, n1, n2) = build_two_node_graph_with_placements(
            DeviceLocation::Cpu, cuda,
        );
        let pre_len = g.len();
        let placements = snapshot_placements(&g);
        let inserted = insert_cross_device_copies(
            &mut g, &[n2],
            |id| placements.get(&id).copied(),
            shares_storage_cpu_devices_only,
        );
        assert_eq!(inserted.len(), 1);
        assert_eq!(g.len(), pre_len + 1, "one new Op::Copy node appended");
        // Consumer's input now points at the new Op::Copy (not n1).
        let consumer_inputs = &g.node(n2).inputs;
        assert_eq!(consumer_inputs.len(), 1);
        let copy_id = consumer_inputs[0];
        assert_ne!(copy_id, n1, "consumer's input was rewired");
        let copy_node = g.node(copy_id);
        assert!(
            matches!(copy_node.op, Op::Copy { target } if target == cuda),
            "rewired input is Op::Copy targeting CUDA; got {:?}",
            copy_node.op,
        );
        assert_eq!(copy_node.inputs, vec![n1], "Op::Copy reads from n1");
        assert_eq!(g.placement(copy_id), Some(cuda), "Op::Copy lands on CUDA");
    }

    /// Two consumers share an input AND both need it on the same target
    /// device → ONE Op::Copy serves both (CSE on inserted copies).
    #[test]
    fn insert_copies_dedupes_when_two_consumers_share_input_and_target() {
        let cuda = DeviceLocation::Cuda { gpu_id: 0 };
        let mut g = Graph::new();
        let cpu_src = add_node(&mut g, Op::Const, vec![]);
        g.set_placement(cpu_src, DeviceLocation::Cpu);
        let cuda_a = add_node(&mut g, Op::Neg, vec![cpu_src]);
        let cuda_b = add_node(&mut g, Op::Sqr, vec![cpu_src]);
        g.set_placement(cuda_a, cuda);
        g.set_placement(cuda_b, cuda);

        let pre_len = g.len();
        let placements = snapshot_placements(&g);
        let inserted = insert_cross_device_copies(
            &mut g, &[cuda_a, cuda_b],
            |id| placements.get(&id).copied(),
            shares_storage_cpu_devices_only,
        );
        assert_eq!(inserted.len(), 1, "CSE — one Op::Copy serves both consumers");
        assert_eq!(g.len(), pre_len + 1);
        // Both consumers point at the SAME Op::Copy node.
        let a_input = g.node(cuda_a).inputs[0];
        let b_input = g.node(cuda_b).inputs[0];
        assert_eq!(a_input, b_input, "both consumers share the rewired input");
    }

    /// Two consumers share an input but want it on DIFFERENT target
    /// devices → TWO distinct Op::Copy nodes inserted.
    #[test]
    fn insert_copies_does_not_dedupe_across_distinct_targets() {
        let cuda = DeviceLocation::Cuda { gpu_id: 0 };
        let vulkan = DeviceLocation::Vulkan { gpu_id: 0 };
        let mut g = Graph::new();
        let cpu_src = add_node(&mut g, Op::Const, vec![]);
        g.set_placement(cpu_src, DeviceLocation::Cpu);
        let cuda_c = add_node(&mut g, Op::Neg, vec![cpu_src]);
        let vk_c = add_node(&mut g, Op::Sqr, vec![cpu_src]);
        g.set_placement(cuda_c, cuda);
        g.set_placement(vk_c, vulkan);

        let placements = snapshot_placements(&g);

        let inserted = insert_cross_device_copies(
            &mut g, &[cuda_c, vk_c],
            |id| placements.get(&id).copied(),
            shares_storage_cpu_devices_only,
        );
        assert_eq!(inserted.len(), 2, "distinct targets → distinct copies");
        let cuda_input = g.node(cuda_c).inputs[0];
        let vk_input = g.node(vk_c).inputs[0];
        assert_ne!(cuda_input, vk_input);
    }

    /// Producer with no placement → pass leaves the edge alone (the
    /// executor handles via fallback). Same for consumer with no
    /// placement.
    #[test]
    fn insert_copies_skips_when_placement_absent() {
        let mut g = Graph::new();
        let n1 = add_node(&mut g, Op::Const, vec![]);
        let n2 = add_node(&mut g, Op::Neg, vec![n1]);
        // Only consumer has placement; producer doesn't.
        g.set_placement(n2, DeviceLocation::Cuda { gpu_id: 0 });
        let placements = snapshot_placements(&g);
        let inserted = insert_cross_device_copies(
            &mut g, &[n2],
            |id| placements.get(&id).copied(),
            shares_storage_cpu_devices_only,
        );
        assert_eq!(inserted.len(), 0, "producer with no placement → skip");
    }

    /// `Op::Copy` and `Op::Move` are themselves transfers — the pass
    /// must not try to insert ANOTHER copy on their inputs (would
    /// infinite-regress).
    #[test]
    fn insert_copies_skips_transfer_ops_themselves() {
        let cuda = DeviceLocation::Cuda { gpu_id: 0 };
        let mut g = Graph::new();
        let cpu_src = add_node(&mut g, Op::Const, vec![]);
        g.set_placement(cpu_src, DeviceLocation::Cpu);
        // A user-authored Op::Copy from CPU to CUDA. Should not be
        // re-wrapped.
        let copy = g.push(Node {
            op: Op::Copy { target: cuda },
            inputs: vec![cpu_src],
            shape: Shape::from_dims(&[4]),
            dtype: DType::F32,
        });
        g.set_placement(copy, cuda);

        let pre_len = g.len();
        let placements = snapshot_placements(&g);
        let inserted = insert_cross_device_copies(
            &mut g, &[copy],
            |id| placements.get(&id).copied(),
            shares_storage_cpu_devices_only,
        );
        assert_eq!(inserted.len(), 0, "the existing Op::Copy is not re-wrapped");
        assert_eq!(g.len(), pre_len);
    }

    /// Pass is idempotent — re-running on a graph already rewritten
    /// adds zero new copies. Phase 2.1 variant.
    #[test]
    fn insert_cross_device_copies_idempotent() {
        let cuda = DeviceLocation::Cuda { gpu_id: 0 };
        let (mut g, _n1, n2) = build_two_node_graph_with_placements(
            DeviceLocation::Cpu, cuda,
        );
        let placements_before = snapshot_placements(&g);
        let first = insert_cross_device_copies(
            &mut g, &[n2],
            |id| placements_before.get(&id).copied(),
            shares_storage_cpu_devices_only,
        );
        let placements_after = snapshot_placements(&g);
        let second = insert_cross_device_copies(
            &mut g, &[n2],
            |id| placements_after.get(&id).copied(),
            shares_storage_cpu_devices_only,
        );
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 0, "re-run sees inserted-Copy as on-target; no new copies");
    }

    /// Cross-GPU edge (CUDA gpu_id 0 → CUDA gpu_id 1) → copy inserted
    /// (devices don't share storage even though both are CUDA).
    #[test]
    fn insert_copies_across_gpu_ids_within_one_backend() {
        let cuda0 = DeviceLocation::Cuda { gpu_id: 0 };
        let cuda1 = DeviceLocation::Cuda { gpu_id: 1 };
        let (mut g, _n1, n2) = build_two_node_graph_with_placements(cuda0, cuda1);
        let placements = snapshot_placements(&g);
        let inserted = insert_cross_device_copies(
            &mut g, &[n2],
            |id| placements.get(&id).copied(),
            shares_storage_cpu_devices_only,
        );
        assert_eq!(inserted.len(), 1, "cross-gpu_id is cross-device too");
        let copy_id = g.node(n2).inputs[0];
        let copy_node = g.node(copy_id);
        assert!(matches!(copy_node.op, Op::Copy { target } if target == cuda1));
    }

    /// Per-node `placement_for` can be backed by an external map, not
    /// just `Graph::placement`. Smoke test of the closure shape.
    #[test]
    fn insert_copies_takes_external_placement_oracle() {
        let cuda = DeviceLocation::Cuda { gpu_id: 0 };
        let (mut g, n1, n2) = build_two_node_graph_with_placements(
            DeviceLocation::Cpu, cuda,
        );
        // Clear graph-side placements; back the oracle with an
        // external map instead.
        let mut external: StdHashMap<NodeId, DeviceLocation> = StdHashMap::new();
        external.insert(n1, DeviceLocation::Cpu);
        external.insert(n2, cuda);
        // Strip graph-side placements so we PROVE the closure is the
        // authoritative source.
        // (No public setter to clear; instead override by leaving
        // graph.placement untouched and using only the external map.)
        let placements = snapshot_placements(&g);
        let inserted = insert_cross_device_copies(
            &mut g, &[n2],
            |id| external.get(&id).copied(),
            shares_storage_cpu_devices_only,
        );
        assert_eq!(inserted.len(), 1);
    }

    // ===== Phase 2.2 — insert_layout_fixups =====

    fn count_contiguize_nodes(graph: &SharedGraph) -> usize {
        let g = graph.read().unwrap();
        (0..g.len())
            .filter(|i| matches!(g.node(NodeId(*i)).op, Op::Contiguize))
            .count()
    }

    /// All-contiguous-input graph: pass adds nothing regardless of
    /// the kernel_accepts_strided predicate.
    #[test]
    fn insert_fixups_no_strided_inputs_no_fixups() {
        let a = Tensor::from_f32(vec![1.0, 2.0], Shape::from_dims(&[2]), cpu_dev());
        let b = a.const_f32_like(vec![3.0, 4.0], Shape::from_dims(&[2]));
        let c = a.add(&b);
        let graph = c.graph().clone();

        let before = count_contiguize_nodes(&graph);
        let inserted = {
            let mut g = graph.write().unwrap();
            insert_layout_fixups(&mut g, &[c.id()], |_| false)
        };
        assert_eq!(inserted, 0);
        assert_eq!(count_contiguize_nodes(&graph), before);
    }

    /// Strided input (Transpose) feeding a kernel that rejects
    /// strided: one Contiguize gets inserted between them and the
    /// consumer's input gets rewired.
    #[test]
    fn insert_fixups_strided_input_rejecting_kernel_inserts_contiguize() {
        // 2x3 const, transpose to 3x2, then "consumer" reads the
        // transposed view. The consumer kernel claims it can't
        // handle strided.
        let a = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        // Transpose produces a strided view via the layout side-table.
        let at = a.transpose();
        // A binary op as our "consumer" (Op::Add is kernel-bearing,
        // not a view op).
        let b = at.const_f32_like(vec![0.0; 6], Shape::from_dims(&[3, 2]));
        let c = at.add(&b);
        let graph = c.graph().clone();

        let before = count_contiguize_nodes(&graph);
        let inserted = {
            let mut g = graph.write().unwrap();
            // Kernel rejects strided for every node.
            insert_layout_fixups(&mut g, &[c.id()], |_| false)
        };
        // One Contiguize for the transposed input of `c`. (`b` is a
        // contiguous Const → no fixup.)
        assert_eq!(inserted, 1);
        assert_eq!(count_contiguize_nodes(&graph) - before, 1);

        // c's first input was `at`; after rewrite it should point
        // to the new Op::Contiguize node.
        let g = graph.read().unwrap();
        let c_node = g.node(c.id());
        let new_input = g.node(c_node.inputs[0]);
        assert!(
            matches!(new_input.op, Op::Contiguize),
            "consumer's strided input must be rewired through Op::Contiguize, \
             got {:?}",
            new_input.op,
        );
        // The Contiguize's input is the original transposed view.
        assert_eq!(new_input.inputs, vec![at.id()]);
    }

    /// Strided input feeding a kernel that accepts strided: pass
    /// skips the input, no Contiguize inserted.
    #[test]
    fn insert_fixups_strided_input_accepting_kernel_no_fixup() {
        let a = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let at = a.transpose();
        let b = at.const_f32_like(vec![0.0; 6], Shape::from_dims(&[3, 2]));
        let c = at.add(&b);
        let graph = c.graph().clone();

        let before = count_contiguize_nodes(&graph);
        let inserted = {
            let mut g = graph.write().unwrap();
            // Every kernel accepts strided — no fixup.
            insert_layout_fixups(&mut g, &[c.id()], |_| true)
        };
        assert_eq!(inserted, 0);
        assert_eq!(count_contiguize_nodes(&graph), before);
    }

    /// CSE: two non-strided-tolerant consumers reading the same
    /// transposed view share one inserted Contiguize node.
    #[test]
    fn insert_fixups_cse_dedupes_shared_strided_input() {
        let a = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let at = a.transpose();
        // Two consumers of the same transposed view.
        let b = at.const_f32_like(vec![0.0; 6], Shape::from_dims(&[3, 2]));
        let c1 = at.add(&b);
        let c2 = at.mul(&b);
        let graph = c1.graph().clone();

        let inserted = {
            let mut g = graph.write().unwrap();
            insert_layout_fixups(&mut g, &[c1.id(), c2.id()], |_| false)
        };
        // Exactly ONE Contiguize, shared between the two consumers.
        assert_eq!(inserted, 1);

        let g = graph.read().unwrap();
        let c1_input = g.node(c1.id()).inputs[0];
        let c2_input = g.node(c2.id()).inputs[0];
        assert_eq!(
            c1_input, c2_input,
            "CSE: both consumers must share the same Contiguize node",
        );
        assert!(matches!(g.node(c1_input).op, Op::Contiguize));
    }

    /// Idempotency: running the pass twice on the same graph adds
    /// 0 new nodes the second time.
    #[test]
    fn insert_fixups_idempotent() {
        let a = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let at = a.transpose();
        let b = at.const_f32_like(vec![0.0; 6], Shape::from_dims(&[3, 2]));
        let c = at.add(&b);
        let graph = c.graph().clone();

        let first = {
            let mut g = graph.write().unwrap();
            insert_layout_fixups(&mut g, &[c.id()], |_| false)
        };
        assert_eq!(first, 1, "first run inserts one Contiguize");

        let second = {
            let mut g = graph.write().unwrap();
            insert_layout_fixups(&mut g, &[c.id()], |_| false)
        };
        assert_eq!(second, 0, "second run is a no-op");
    }

    /// View ops in the consumer position do NOT get Contiguize
    /// inserted before them — they preserve strided layouts
    /// intentionally as part of their semantics.
    #[test]
    fn insert_fixups_skips_view_op_consumers() {
        let a = Tensor::from_f32(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            Shape::from_dims(&[2, 3]),
            cpu_dev(),
        );
        let at = a.transpose();
        // Permute on a transposed view: both are view ops, the
        // strided layout flows through unchanged.
        let ap = at.permute(&[1, 0]);
        let graph = ap.graph().clone();

        let inserted = {
            let mut g = graph.write().unwrap();
            insert_layout_fixups(&mut g, &[ap.id()], |_| false)
        };
        assert_eq!(
            inserted, 0,
            "view ops preserve strided layouts; no Contiguize should be inserted",
        );
    }
}
