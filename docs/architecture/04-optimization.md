# Optimization

**Status**: v0.3 (draft, 2026-05-09). v0.3 changes: (1) cost-model layer-1 (static annotations) is explicitly composed of static priors PLUS community-aggregated empirical data when available, not pure static; (2) a precision-filter pass runs before cost ranking, pruning alternatives whose `PrecisionGuarantee` doesn't meet the user's per-call precision requirement. v0.2 changes: (a) Op identity is the single `Op` enum with primitive variants and one `Op::Fused(id, params)` arm — no `NodeKind` discriminator; (b) the top-N model is per-decision-point alternatives with coupled cost adjustments, not N complete graphs; (c) the cost model accounts for parallelism (wall-clock under expected concurrency, not strict-serial sum); (d) diversity-of-routes phrasing clarified.

How the base map becomes the optimized form. The DecompositionMap and OptimizationMap, the rule engine that drives both, per-decision-point alternative preservation, the sliding window that allows optimization and execution to overlap, and the cost model that ranks candidate plans.

This is the longest section. The optimizer is where most of fuel's leverage lives — the architectural commitments here determine which competitive edges are reachable and which aren't.

---

## The two-stage transformation

Optimization in fuel is two distinct stages, each with its own machinery:

1. **Decomposition.** User-facing form → base map. Every `Op::Fused(id, params)` node is replaced by its primitive decomposition, recursively. Output: a graph containing only primitive `Op` variants — no `Op::Fused` arms. Deterministic, one-shot, retained as a permanent artifact (see [03-ir](03-ir.md#the-base-map-fully-decomposed-primitive-dag-permanently-retained)).
2. **Optimization.** Base map → optimized form. The optimizer searches for transformations (fusions, algebraic rewrites, dtype lowerings, placement decisions, transfer-op insertions, layout fixups) and produces an annotated DAG where decision points carry up to N alternatives ranked by cost. Output: per-decision-point alternative sets with pre-resolved kernels and coupled cost adjustments.

Both stages are driven by the same rule-engine machinery. The difference is the rule sets they apply and the structural promises they make: decomposition's rules are exact and exhaustive (everything that *can* be decomposed *is* decomposed; no residual fused ops in the output); optimization's rules are heuristic and search-driven (the optimizer picks among many possible transformations, ranking by cost).

## DecompositionMap

The DecompositionMap is a mapping from fused-op identity to a function that produces the primitive subgraph equivalent:

- **Keys**: `FusedOpId` (registry-assigned identifiers from [03-ir](03-ir.md)). Every fused-op entry in the registry contributes one key.
- **Values**: a decomposition function `fn(graph, node_id, params) -> NodeId` that, given a fused-op node, appends its primitive subgraph to the graph and returns the subgraph's output node.

Decomposition runs as a fixpoint: walk the graph, find any node whose `op` is `Op::Fused(id, _)`, look up the entry's decomposition function, replace the node. Repeat until no `Op::Fused` arms remain. Termination is guaranteed because decomposition is *strictly contracting*: each step reduces the count of `Op::Fused` nodes by one, increasing primitive count by some bounded amount; there's no cycle.

The DecompositionMap derives one-to-one from the fused-op registry. There's no DecompositionMap entry for a fused op without a registry entry, and no registry entry without a DecompositionMap entry — they're the same data viewed differently. In practice the DecompositionMap may not be a literal `HashMap` data structure; it's the conceptual surface the optimizer sees.

Decomposition rules carry no error annotation: every fused-op decomposition is, by registry contract, mathematically equivalent to the fused form (modulo IEEE float-rounding determinism, where the fused form may actually be more numerically stable in some cases — see [07-tolerance](07-tolerance.md#direction-of-error-and-one-sided-budgets)). Decomposition produces a strict-equivalent reference; the optimizer can later choose non-equivalent rewrites under tolerance budgets.

## OptimizationMap

The OptimizationMap is the registry of transformations the optimizer can apply to the base map. Each entry is a `(matcher, rewriter)` pair:

- **Matcher**: examines a node and its surrounding subgraph; returns `Some(match)` if the rule applies, `None` otherwise. Matches identify the specific subgraph instance that will be rewritten.
- **Rewriter**: given a match, emits the replacement subgraph and records the consumer-edge remapping so the optimizer can rewire downstream nodes.

Two complementary engines for matchers, used uniformly through the same rule trait:

**Declarative patterns with variables.** A rule's matcher is expressed as a pattern: a tree of op names with hole-variables that bind to subtrees. Example: `(Mul (Add ?a ?b) (Sub ?a ?b)) → (Sub (Sqr ?a) (Sqr ?b))` recognizes `(a+b)*(a-b)` and rewrites to `a² - b²`. Declarative patterns compile to fast pattern-matchers; their structure is analyzable (the optimizer can reason about whether two rules conflict, whether a rule is shape-preserving, etc.); they're auto-generatable from fused-op registry entries via the entry's canonical pattern.

**Callable functions.** A rule's matcher is a Rust function `fn(&Graph, NodeId) -> Option<Match>`. Used when the pattern can't be expressed declaratively — e.g., a rule that needs to count global consumer references (PR 3's SoftmaxLastDimFuseRule does exactly this), check shape relationships across non-adjacent nodes, or apply some computation the declarative form can't reach.

The optimizer treats both engine types uniformly through a common `Rule` trait. Most rules are declarative; callable functions are the escape hatch. This avoids the rigidity of a single-engine design while keeping the common case analyzable.

A rule's *family*, *cost contribution*, and *frontier compatibility* are part of its identity:

- **Family**: lowering (high-level → primitive subgraph; runs in decomposition), fusion (primitive subgraph → high-level; runs in optimization), algebraic (primitive ↔ primitive equivalence-preserving rewrite), tolerance-gated (rewrite that requires non-zero tolerance budget).
- **Cost contribution**: how the optimizer should score a route that includes this rule's rewrite. Static annotation (FLOPs delta, bytes-moved delta, kernel-overhead delta) plus error contribution (for tolerance gating). Refined by empirical Judge data over time.
- **Frontier compatibility**: whether this rule can fire safely while concurrent optimize-and-execute is in progress (see [the sliding window section](#the-sliding-window-optimization-and-execution-overlap) below). Two values:
  - `Concurrent` — the rule's matches are local enough to fit within the optimizer's lookahead window. The rule never needs to rewrite a node that's already been finalized. Safe under concurrent execution.
  - `WholeGraph` — the rule's matches may span arbitrary distances and can target nodes whose finalization status varies. Common for rules like cross-graph CSE, global re-association, or whole-program dead-code elimination. Cannot fire under concurrent execution; must run before the optimization frontier starts advancing.

Most rules are `Concurrent`. The classification is conservative: when in doubt, declare `WholeGraph`. The optimizer reads the declaration and decides per-route whether concurrent execution is feasible (see below).

## Per-decision-point alternatives

The optimizer's output is not N parallel complete graphs. It is the base map annotated with **decision points** — places in the DAG where ≥2 viable alternatives exist — and at each decision point, up to N alternatives are preserved (default N=3, configurable). The runtime route picker resolves per-decision-point alternatives at dispatch time, reading current backend telemetry.

This is strictly more flexible than "top-N complete plans": with M decision points, per-decision-point alternatives expose N^M reachable execution combinations to the route picker, of which N complete plans would only cover a small subset. The picker can mix and match — choose alternative X1 at decision point A, alternative Y2 at decision point B — based on whatever telemetry says about current device load.

### What counts as a decision point

A decision point is any location in the optimized DAG where the optimizer found ≥2 viable alternatives during its search. The most common kinds:

- **Kernel-variant choices**: same op, different kernel implementations (e.g., cuBLAS vs custom matmul; different tile shapes; different precision-tradeoff variants for the same backend).
- **Placement choices**: same op, different (backend, device) assignments.
- **Fusion-vs-decomposition choices**: a subgraph that can be left as primitives OR fused into a single registered fused op.
- **Algebraic-rewrite choices**: a subgraph that has equivalent algebraic forms with different op counts or operand traffic.
- **Tolerance-trade-off choices**: a region that can run strict OR with a non-zero tolerance budget for a faster approximate variant.

### Storage and structural-distinctness

- Alternatives within a decision point are stored as a small set attached to that point in the DAG. Storage scales as N × M (per-decision-point sets), not as N complete graph copies. For typical inference graphs this is well within budget.
- Subgraphs not at decision points are shared — they're regions where the optimizer found exactly one viable alternative.
- Alternatives within a decision point are **structurally distinct by construction** — the optimizer deduplicates during search. Two technically-different rule applications producing the same subgraph collapse to one alternative.
- What is *not* enforced is meaningful diversity beyond structural distinctness: two distinct alternatives might differ only trivially (same kernel choices, same placements, slightly different intermediate node ordering) and so present the runtime picker with redundant options. Future refinement could enforce explicit diversity (e.g., "the N alternatives at a decision point must differ on at least one placement or one tolerance trade-off") if shallow distinctness turns out to be common in practice.

### Coupling between decisions

Decisions aren't always independent. Placement choices couple: putting two adjacent ops on the same device avoids a transfer; splitting them across devices adds transfer cost. The optimizer encodes these couplings as **conditional cost adjustments** attached to alternatives:

```text
Decision point A — branch_A's kernel choice:
  Alternative X1 (CUDA cuBLAS):  base 8ms,  +2ms if downstream join is on CPU.
  Alternative X2 (CPU OpenBLAS): base 12ms, no downstream-placement penalty.
  Alternative X3 (Vulkan):       base 10ms, +3ms if downstream join is on CPU.

Decision point B — branch_B's kernel choice:
  Alternative Y1 (CUDA):  base 6ms.
  Alternative Y2 (CPU):   base 9ms.

Decision point C — join's kernel choice:
  Alternative Z1 (CUDA):  base 4ms, requires both branches' outputs on CUDA.
  Alternative Z2 (CPU):   base 7ms, requires both branches' outputs on CPU.
  Alternative Z3 (Vulkan): base 5ms, requires both branches' outputs on Vulkan.
```

The runtime picker reads alternatives + couplings + current telemetry, then resolves to a coherent plan. Locally-greedy resolution (pick each decision point's locally-best alternative independently, ignoring couplings) is usually fine; rare adversarial cases where greedy is bad get caught by a small lookahead in the picker.

### What makes an alternative admissible at a decision point

- It must satisfy all hard constraints at its location: tolerance budget not exceeded along any path through this decision point, viable backend assignment, all transfers feasible, all layouts reachable via fixup ops.
- Among alternatives that satisfy hard constraints, the top N by local cost (with conditional adjustments treated as worst-case for ranking) are retained.
- N is per-decision-point, not global. A decision point with only one viable alternative stores one; a decision point with twelve stores up to N (default 3).

### Default N and configurability

Default N=3 per decision point gives the runtime picker meaningful flexibility (typically a strict-fast alternative, a strict-memory-conserving alternative, and a tolerance-aware fast alternative) without exploding storage or search cost. Configurable up to roughly 10 in practice; beyond that diminishing returns set in (alternatives differ only marginally; runtime picker rarely picks the lower-ranked ones).

The cost of N>1 per decision point:

- **Optimizer search cost** scales with N per decision point — more search per location to find more competitive alternatives.
- **Storage cost** scales as N × M (alternatives per decision point × number of decision points). Linear in both factors; manageable for typical inference graphs.
- **Runtime picker cost per realize** scales with M (a decision per location). Mitigated by caching: decisions stable across realizes if telemetry is stable; only re-decide when telemetry shifts meaningfully.

## The sliding window: optimization and execution overlap

Fuel supports **concurrent optimize-and-execute** as the default execution model when the active rules permit it. The optimizer runs as a producer; the executor runs as a consumer; the boundary between them — the **optimization frontier** — slides through the DAG over time. Once the frontier has passed a node, that node is *finalized* and eligible to execute (subject to its inputs being ready).

Whether concurrent execution is actually used for a given realize depends on the rules that fired in the route being executed. Each rule self-declares its frontier compatibility (see [Rule properties above](#optimizationmap)); the optimizer uses these declarations to decide per-route whether concurrency is feasible.

The architectural rules that make concurrent execution safe:

- **Frontier advance is monotonic.** The frontier never moves backward. Once a region is finalized, no future optimization can touch it. This is what makes execution-eligible-from-frontier safe: an executing node knows its op, kernel, inputs, and outputs are bit-stable.
- **`Concurrent`-classified rules only fire ahead of the frontier.** A `Concurrent` rule whose match would touch a finalized node is deferred; if the node has already executed by the time the rule's match is considered, the match is rejected (the node has the value it had; rewriting it is meaningless). The classification's contract is that this rejection is rare — `Concurrent` rules are designed to operate within the optimizer's lookahead window where the frontier hasn't caught up yet.
- **`WholeGraph`-classified rules disable concurrency wherever they fire.** A decision-point alternative produced by a `WholeGraph` rule cannot be picked under concurrent execute. The optimizer must complete optimization of the whole graph before execution starts when any `WholeGraph` rule has produced an alternative the picker might use. Such alternatives are marked `concurrent-incompatible`; the runtime picker treats them accordingly.
- **Concurrent and whole-graph alternatives coexist at decision points.** Within one decision point, some alternatives may have been produced exclusively by `Concurrent` rules (concurrent-compatible) and others may include `WholeGraph` rule applications (concurrent-incompatible). The picker reads per-realize concurrency policy to select among compatible alternatives at each decision point.
- **Per-decision-point commit at the frontier.** As the frontier passes a decision point that's still ahead of execution, the optimizer must commit: one of the concurrent-compatible alternatives wins for that location. The runtime picker is constrained to the committed alternative from there forward. (Decision points whose alternatives all require `WholeGraph` rules don't participate in concurrent execute at all; the picker chooses among them only after the whole graph is optimized.) This means the per-decision-point alternative set can shrink to one as the frontier passes, never grow.

The route picker's per-realize concurrency policy:

- `Concurrency::Auto` (default) — picker prefers concurrent-compatible alternatives when they're competitive on cost. Falls back to whole-graph alternatives only if concurrent ones are demonstrably worse.
- `Concurrency::Required` — only concurrent-compatible alternatives are eligible at every decision point. If a decision point has no concurrent-compatible alternative, the realize fails or surfaces an error.
- `Concurrency::Forbidden` — only whole-graph alternatives are eligible. Useful for one-shot batch computations where the latency-to-first-output benefit doesn't justify the complexity.

What concurrent execution enables:

- **Latency to first output is bounded by execution time of the early portion of the graph, not by optimization time of the whole graph.** For large models, optimization of the deepest layers can run in parallel with execution of the shallowest. Meaningful for inference TTFT.
- **Graphs that mutate (autoregressive decoding extending the same graph repeatedly) can have execution and optimization interleaved naturally.** As the user appends nodes for the next decode step, the optimizer works on them; nodes from previous steps are already executing or done.

The cost — for routes that use it:

- **Implementation complexity is high.** Frontier tracking, top-N collapse mechanics, atomic-swap of route assignments at the frontier — each is non-trivial. Race conditions are subtle. Tests and invariants must be airtight.
- **Cost-model accuracy matters more.** When the frontier advances, the optimizer commits irreversibly. Bad cost estimates → bad commits. Empirical Judge data is more important under concurrent execution than under whole-graph execution, where errors could be corrected before execution starts.
- **Some optimization opportunities only show up in `WholeGraph` rules.** A workload that benefits substantially from cross-graph CSE or global re-association will favor whole-graph routes; concurrent execution leaves those wins on the table. The optimizer can produce both kinds and let the route picker compare costs honestly.

This concurrent model is more aggressive than what's currently implemented. PR 3's optimizer is synchronous, single-graph, no transactions. Migrating to the concurrent model is itself a multi-phase project (transactions first, then frontier tracking, then per-decision-point alternative tracking, then `Concurrent`/`WholeGraph` rule classification, then per-realize concurrency policy). The architectural commitment is the destination; the migration is phase work.

## Cost model: static annotations refined by empirical Judge data, accounting for parallelism

Each rule and each backend's kernel implementation contributes per-node cost estimates. The optimizer composes them over a candidate plan to produce a *wall-clock cost*, then ranks alternatives at each decision point by that cost.

**Wall-clock, not strict-serial.** Independent subgraphs in the DAG can execute concurrently across backends and across same-backend slots (see [06-runtime](06-runtime.md) for the dispatch model). Plan cost is therefore composed as `wall_clock ≈ max(parallel_branches) + serial_remainder`, not as the sum of per-node costs. A plan that exposes two long parallel branches has wall-clock cost ≈ max(branch_a, branch_b), not branch_a + branch_b. Plans that maximize useful parallelism rank higher than serially-equivalent plans even when their summed cost is identical.

**Three layers of cost data, composed:**

**Layer 1 — static annotations, optionally refined by community-aggregated empirical priors.** Each rule annotates its cost contribution: FLOPs delta, bytes-moved delta, kernel-overhead delta. Each backend-impl annotates its cost as a function `cost(shapes, params, capabilities) -> CostEstimate { flops, bytes_moved, kernel_overhead_ns }`. Static annotations are pessimistic upper bounds; conservative on uncertainty. Plans ranked by static cost alone give a reasonable first-order ordering but miss bandwidth saturation, queue contention, hardware quirks, and parallelism interactions.

When community-aggregated empirical data is available for the user's hardware fingerprint (per [08-pattern-harvest](08-pattern-harvest.md#shared-infrastructure-with-tolerance-recipes) and [11-persistence](11-persistence.md#cache-generation-and-distribution)), layer-1 is refined: per-cell community medians replace the FLOP-counting estimate where data exists, with confidence intervals tied to sample count. Cells without community data fall back to FLOP-counting. The cache-generation tool uses this same refined layer-1 when producing distributable caches; users on common hardware get caches whose static cost ranking is calibrated against actual measured behavior on similar hardware, not just theoretical bounds.

**Layer 2 — empirical Judge data.** The Judge measures actual per-(op, dtype, size_class, backend, device) latency and accumulates a profile report. Static cost estimates are *advisory*; Judge data *overrides* them when available. Alternatives that look good on static cost but bad on empirical measurement get demoted; vice versa. As Judge coverage grows, the optimizer's rankings improve.

**Layer 3 — current telemetry.** The route picker (not the optimizer) reads current backend telemetry — memory pressure, slot availability, queue depth, currently-resident weights — at dispatch time. Decisions that are competitive on layers 1+2 are picked among per-decision-point based on layer 3. This is how the "fast under low load, memory-conserving under contention" property emerges. See [06-runtime](06-runtime.md).

**Parallelism budgets enter at layers 1 and 3.** The optimizer (layer 1+2) accounts for total parallelism capacity advertised by backends — slot counts per device — and penalizes plans that exceed advertised capacity OR that exceed peak-memory budgets when in-flight activations are summed across parallel branches. The runtime picker (layer 3) reads currently-available slot counts, not just totals, and picks plans that fit current capacity.

**The three layers compose**: layer 1 provides structural ordering, layer 2 corrects it with measurements, layer 3 adapts at runtime. The optimizer commits a per-decision-point alternative set based on layers 1+2; the route picker resolves alternatives at dispatch time using layer 3.

## Precision-filter pass: runs before cost ranking

Before cost-based ranking happens, alternatives at each decision point are filtered by their `PrecisionGuarantee` (per [05-backend-contract §Per-kernel precision guarantees](05-backend-contract.md#per-kernel-precision-guarantees)) against two requirements:

- **The user's per-call precision requirement.** If the user has specified a precision floor (e.g., "≤ 0.1% relative error per element"), alternatives whose `max_relative` exceeds it are pruned regardless of how cheap they are.
- **The cumulative tolerance budget.** Even if no per-call precision floor is set, the route's cumulative error along every path must stay within the tolerance budget (per [Tolerance budgets gate which rules fire](#tolerance-budgets-gate-which-rules-fire) below). Alternatives whose contribution would push cumulative error over budget are pruned.

The filter runs first; cost ranking ranks the survivors. This order matters: a fast kernel that doesn't meet the precision requirement isn't a candidate, period. Cost is a tiebreaker among admissible alternatives, not a way to admit non-admissible ones.

When the user hasn't specified a per-call precision floor and the tolerance budget is `Strict` (the default), the precision filter admits only alternatives with strict-equivalent precision (kernels with `bit_stable_on_same_hardware: true` or with bounded error guaranteed within IEEE-rounding tolerance). Most kernels qualify here; the filter is restrictive only under tighter user requirements.

## Tolerance budgets gate which rules fire

The optimizer reads the tolerance specification (graph default → subgraph override → per-op override) when evaluating a candidate route. Routes that violate the tolerance budget along any path are pruned. See [07-tolerance](07-tolerance.md) for the full model; here, the architectural integration:

- Each rule declares its error contribution (zero for strict-equivalence rewrites, non-zero for approximate ones).
- The optimizer tracks cumulative error along each candidate route's path through the graph. Cumulative tracking respects the rule's compositional behavior (additive for most ops, multiplicative for some, nonlinear for a few; details in 07).
- A rule may not fire on a node if the resulting cumulative error along *any* output path through that node would exceed the budget.
- Routes with zero non-strict rewrites are always admissible (under any non-negative tolerance). Routes with non-zero rewrites are admissible only under the corresponding tolerance budget.

The optimizer also produces routes that span *different tolerance strategies* — one strict route, one tolerance-aware route — when both are competitive. The runtime route picker honors per-call tolerance overrides by selecting the route that's tightest within the per-call budget.

## Cross-cutting transformations the optimizer is responsible for

The OptimizationMap doesn't only contain "rewrite subgraph A to subgraph B" rules. Three cross-cutting concerns are folded into the optimizer's responsibility:

**Layout fixups.** When a candidate route would feed a kernel an input layout the kernel doesn't accept (the kernel's `KernelCaps` says no), the optimizer inserts an explicit `Op::Contiguize` (or its equivalent) node. The fixup's cost is included in the route's cost. The optimizer is the only place layout materialization is decided; the executor never inserts layout fixups.

**Transfer-op insertions.** When a candidate route assigns adjacent ops to different (backend, device) instances, the optimizer inserts `Op::Copy` (and possibly `Op::Move`/`Op::Release` for residency control) nodes between them. Transfer cost is read from `BackendCapabilities.transfer_paths` and included in the route's total cost. Routes that cross devices unnecessarily are penalized; routes that cross only when the destination's per-op cost win exceeds the transfer cost are admissible. Cross-device routes that aren't reachable (no transfer path advertised) are pruned.

**Dtype changes.** Mixed-precision routes (under tolerance budgets) require explicit `Op::Cast` insertions where the dtype changes. The optimizer inserts them; the cost includes the cast op's cost.

In all three cases, the principle is the same: the cost is *visible to the optimizer*, not hidden inside the kernel or backend. Routes are ranked on cost-with-fixups-included. A route that looks great on raw matmul cost but requires three contiguize ops + a host-staging transfer is correctly-priced.

## Relationship to PR 3's existing rule registry

PR 3 (commit `3d7ca325`, 2026-05-07) shipped a rule registry framework: `Rule` trait, `RuleFamily::{Lowering, Fusion}`, `RuleRegistry`, fixpoint driver. One concrete rule pair: `SoftmaxLastDimLowerRule` and `SoftmaxLastDimFuseRule`.

PR 3 is the substrate this section's optimizer builds on. The architectural relationships:

- **PR 3's `Rule` trait extends to declarative + callable engines uniformly.** Today PR 3's rules are all callable. The architecture's commitment is that the trait can host either kind without re-design.
- **Decomposition rules are PR 3's lowering family.** Today PR 3's `SoftmaxLastDimLowerRule` is hand-written; under the FusedOpRegistry refactor, it's auto-generated from the registry entry's decomposition function. The mechanics are unchanged.
- **OptimizationMap rules are PR 3's fusion family + new algebraic-rewrite rules + tolerance-gated rules.** Today PR 3's `SoftmaxLastDimFuseRule` is the only fusion rule; the architecture admits dozens more, plus algebraic rewrites that aren't fusion at all.
- **Per-decision-point alternatives are not in PR 3.** PR 3's driver picks one transformation per match (first-match-wins). The architecture's per-decision-point model requires extending the driver to track multiple alternatives at each location and emit them as a structured set on the optimized DAG. The Rule trait surface doesn't change; the driver and the output-graph annotation surface do.
- **Concurrent optimize-and-execute is not in PR 3.** PR 3 runs synchronously to fixpoint, then execution starts. Concurrent execution requires frontier tracking and atomic-swap commit semantics. ROADMAP §"Phase 7.5 graph optimizer architecture" sketches the transactional snapshot model that this builds on.

The progression from PR 3 to the architecture in this section is incremental: each piece (declarative engine, registry-auto-generation, top-N, frontier, concurrent execution) is a separable phase. PR 3's API is preserved; the driver evolves.

## What this rules out

A few non-features called out explicitly:

- **No e-graph saturation as the primary engine.** E-graphs (egg, equality saturation) are appealing for algebraic rewrites but their performance characteristics make them unsuitable for the per-realize hot path fuel needs. They may show up later as an offline rule-discovery tool (find new algebraic equivalences in the harvested workload data) feeding the OptimizationMap, but the optimizer itself stays on the rule-driver model.
- **No autotuning-search-style optimizer.** TVM-style autotuner search produces excellent results but is operationally heavy. Fuel's optimizer is heuristic + cost-driven, not search-driven. Empirical Judge data fills the gap autotuning would otherwise fill.
- **No user-installable rules at runtime.** The OptimizationMap is populated at startup, frozen thereafter. Runtime rule extension would let users hot-load optimizations but introduces a security and stability surface fuel doesn't need to take on.
- **No global optimization passes that aren't rule-based.** Every transformation goes through the rule machinery. "Special pass that's not a rule" is forbidden because it makes the optimizer's behavior unanalyzable and its commits unreproducible.

---

## See also

- [01-identity](01-identity.md) — the algebraic-rewrites and top-N edges this section implements.
- [03-ir](03-ir.md) — the data structures the optimizer transforms (DAG, base map, optimized form, side-tables).
- [05-backend-contract](05-backend-contract.md) — what backends advertise that the optimizer reads (kernels, costs, transfer paths).
- [06-runtime](06-runtime.md) — the route picker, telemetry-driven runtime route selection, frontier handoff to executor.
- [07-tolerance](07-tolerance.md) — how tolerance budgets prune candidate routes.
- [08-pattern-harvest](08-pattern-harvest.md) — telemetry that feeds back into rule discovery.
- ROADMAP §"Phase 7.5 graph optimizer architecture" — the transactional / frontier model this section builds on.
- `docs/architecture-audit.md` §"Q-A" through §"Q-E" — the cross-cutting questions this section's commitments resolve.
