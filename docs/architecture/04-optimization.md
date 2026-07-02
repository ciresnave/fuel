# Optimization

**Status**: v0.8 (draft, 2026-07-02). **v0.8 adds the fused-op Layer-1 *cost-from-decompose* default** (see [10-decisions-log](10-decisions-log.md)): a fused/synthesized op with no declared or measured cost no longer prices at the zero sentinel — its Layer-1 cost is **composed from its own `decompose`** (`flops = Σ` decompose primitives, `bytes_moved =` the fused op's boundary I/O, one launch overhead), fired only for the sentinel so the layering stays measured › declared › composed › (never) zero (§Three layers of cost data — "Fused-op Layer-1 default"). This closes the same zero-cost landmine Part A closed for GPUs, now for every FKC-imported and runtime-adopted fused op. **v0.7 realizes the deferred per-backend throughput in Layer-1 cost** (FKC cost unification Part C — see [10-decisions-log](10-decisions-log.md)): the composite nanosecond figure is now a **roofline over each candidate backend's registered throughput** (`BackendCapabilities.compute_throughput_flops_per_ns` / `mem_bandwidth_bytes_per_ns`), replacing the backend-agnostic 1-FLOP/ns prior (§Three layers of cost data). This makes **cross-device placement honest** — a GPU op is priced on GPU throughput, so the placement DP prefers a GPU for large parallel work (throughput dwarfs transfer) while small ops stay local. Pairs with Part A (registering GPU `BackendCapabilities` + cost fns so an unseeded GPU is no longer priced at zero). **v0.6 adds the 2026-06-20 adaptive-runtime-fusion decision** ([10-decisions-log](10-decisions-log.md)): the **recipe principle** (every fused op carries `decompose` + `pattern`, both mandatory — §The two-stage transformation / DecompositionMap), the **`decompose` totality + never-panic + primitive→self** invariant (the base map is the fixpoint of `decompose`; a non-decomposing non-primitive is a surfaced opaque-op gap, not a crash), and the re-scoping of "No user-installable rules at runtime" (§What this rules out) to *untrusted-user* rules only — trusted, Fuel-orchestrated, cost-gated runtime fused-op registration is now in-bounds. **v0.6 also corrects the implemented-vs-intended status of already-stated commitments**: the per-decision-point alternatives, the lock-step driver, the bounded per-device Pareto frontier, and the runtime route picker have all landed (PR-A/B/C); what remains future is the sliding-window concurrent optimize-and-execute. The two stale "not in PR 3" framings (§Relationship-to-PR-3, §The sliding window) are updated; no core-claim change. **v0.5 implemented the 2026-06-14 "plan IS the graph" redirection** (see [10-decisions-log](10-decisions-log.md)): `optimize_graph` is reframed as **pathfinders + rankers + optimizers** that transform the base map **in place into the multi-path graph** — the plan IS the graph, not a separate annotation. The surviving alternatives are a **bounded per-device Pareto frontier** (kept small by the right axes — one central time metric, memory as tiers, discrete precision/accuracy — with a crowding-distance cap as the hard backstop; prototype-validated at ~10² paths). Timing optimizes on a **single central metric** (`t_min` dropped as a selection axis); memory is a **per-tier vector**; ties break precision → accuracy → memory. Most of v0.4 stands (decomposition, the rule engine, the carry-forward placement DP, plan-fragment memoization, the sliding window) — reframed and bounded, not replaced. v0.4 changes (per the 2026-06-11 load-time-planning decision): (1) the optimization producer starts at **model load** — planning begins as soon as graph nodes exist, not when `realize()` is entered; `realize()` reduces to wait-for-plan-coverage + dispatch; (2) new §Load-time incremental planning + cost-based placement: numeric transfer pricing, the (position × device) carry-forward placement DP with bounded fused-jump lookback, plan-fragment memoization by structural hash, and the plan as the weight-prefetch schedule; (3) horizontal fusion / similarity scheduling named as a future rule family. v0.3 changes (preserved): (1) cost-model layer-1 (static annotations) is explicitly composed of static priors PLUS community-aggregated empirical data when available, not pure static; (2) a precision-filter pass runs before cost ranking, pruning alternatives whose `PrecisionGuarantee` doesn't meet the user's per-call precision requirement. v0.2 changes: (a) Op identity is the single `Op` enum with primitive variants and one `Op::Fused(id, params)` arm — no `NodeKind` discriminator; (b) the top-N model is per-decision-point alternatives with coupled cost adjustments, not N complete graphs; (c) the cost model accounts for parallelism (wall-clock under expected concurrency, not strict-serial sum); (d) diversity-of-routes phrasing clarified.

How the base map becomes the optimized form. The DecompositionMap and OptimizationMap, the rule engine that drives both, per-decision-point alternative preservation, the sliding window that allows optimization and execution to overlap, and the cost model that ranks candidate plans.

This is the longest section. The optimizer is where most of fuel's leverage lives — the architectural commitments here determine which competitive edges are reachable and which aren't.

---

## The two-stage transformation

Optimization in fuel is two distinct stages, each with its own machinery:

1. **Decomposition.** User-facing form → base map. Every `Op::Fused(id, params)` node is replaced by its primitive decomposition, recursively. Output: a graph containing only primitive `Op` variants — no `Op::Fused` arms. Deterministic, one-shot, retained as a permanent artifact (see [03-ir](03-ir.md#the-base-map-fully-decomposed-primitive-dag-permanently-retained)).
2. **Optimization (`optimize_graph`).** Base map → optimized multi-path graph, **in place** — the plan IS the graph (see [03-ir](03-ir.md#the-optimized-form-the-multi-path-graph-the-plan-is-the-graph)). `optimize_graph` runs three kinds of pass **lock-step** (prune as you go, never explode-then-extract): **pathfinders** add candidate paths (fusions, algebraic rewrites, dtype lowerings under tolerance, placement/transfer choices, layout fixups); **rankers** measure each path (timing, precision, accuracy, per-tier memory); **optimizers** merge/discard paths (precision-filter, duplicate-path convergence, path-timing). Output: a multi-path graph carrying a **bounded per-device Pareto frontier** of routes with **decision points at branches** and optionally pre-resolved kernels — see [Bounding the frontier](#bounding-the-frontier-pareto-per-device--crowding-cap).

Both stages are driven by the same rule-engine machinery. The difference is the rule sets they apply and the structural promises they make: decomposition's rules are exact and exhaustive (everything that *can* be decomposed *is* decomposed; no residual fused ops in the output); optimization's rules are heuristic and search-driven (the optimizer picks among many possible transformations, ranking by cost).

## DecompositionMap

The DecompositionMap is a mapping from fused-op identity to a function that produces the primitive subgraph equivalent:

- **Keys**: `FusedOpId` (registry-assigned identifiers from [03-ir](03-ir.md)). Every fused-op entry in the registry contributes one key.
- **Values**: a decomposition function `fn(graph, node_id, params) -> NodeId` that, given a fused-op node, appends its primitive subgraph to the graph and returns the subgraph's output node.

Decomposition runs as a fixpoint: walk the graph, find any node whose `op` is `Op::Fused(id, _)`, look up the entry's decomposition function, replace the node. Repeat until no `Op::Fused` arms remain. Termination is guaranteed because decomposition is *strictly contracting*: each step reduces the count of `Op::Fused` nodes by one, increasing primitive count by some bounded amount; there's no cycle.

The DecompositionMap derives one-to-one from the fused-op registry. There's no DecompositionMap entry for a fused op without a registry entry, and no registry entry without a DecompositionMap entry — they're the same data viewed differently. In practice the DecompositionMap may not be a literal `HashMap` data structure; it's the conceptual surface the optimizer sees.

Decomposition rules carry no error annotation: every fused-op decomposition is, by registry contract, mathematically equivalent to the fused form (modulo IEEE float-rounding determinism, where the fused form may actually be more numerically stable in some cases — see [07-tolerance](07-tolerance.md#direction-of-error-and-one-sided-budgets)). Decomposition produces a strict-equivalent reference; the optimizer can later choose non-equivalent rewrites under tolerance budgets.

### `decompose` is total, never-panics, and is the base map's fixpoint

(Per the 2026-06-20 adaptive-fusion decision, [10-decisions-log](10-decisions-log.md) G1/G2.) The recipe principle has two inverse halves and **both are mandatory** for every fused op: a `decompose` (the *break-down* — fused → primitive subgraph; this section's lowering) and a `pattern` (the *build-up* — recognize that primitive subgraph and re-fuse; the [OptimizationMap](#optimizationmap) below). A fused op missing either half is an **opaque island**: invisible to base-map analysis and impossible to re-fuse. They are the same data viewed in opposite directions — the DecompositionMap and OptimizationMap derive one-to-one from the registry.

`decompose` is therefore a **total function that never `panic!`s** (the never-panic constitution rule):

- A **primitive decomposes to itself** — the recursion's fixpoint, already the identity form `decompose = |_g, id, _p| id` at `fuel-graph/src/registry.rs:823`. A primitive is simply a node no lowering rule fires on.
- The **base map is the fixpoint of `decompose` over every node** (lower until `decompose(x) == x` everywhere) — exactly this section's `optimize_graph` rewrite-to-fixpoint model.
- A `panic!` in `decompose` is **always a bug**: the op is either a true primitive (must return self) or a non-primitive whose recipe is missing (a bug / basis gap). The two are distinguished by **basis membership, never by the return value**. A non-basis op that fails to decompose is a **surfaced opaque-op gap** — flagged on the base map and fed to the missing-fusion / inventory telemetry ([08-pattern-harvest](08-pattern-harvest.md)) — never a crash and never silently masquerading as a primitive.

This is **load-bearing for optimization itself, not only for JIT/telemetry**: because optimization *is* lower-to-base-map-then-find-the-best-cover, an op that won't decompose **breaks the optimizer** (it cannot be lowered, so it cannot participate in re-fusion or cost comparison). A fused op's recipe therefore **always ships with the op** — never deferred "until intermediates fit," which would strand an opaque island in the base map. The three current panicking decomposes (`nf4_matmul.rs:120`, `flash_attn`, `selective_scan`) are bugs to fix under this contract, not a permanent category. (The recipe is the *math* definition; the kernel is a faster, numerically-close implementation governed by the FKC `precision` tolerance — e.g. flash-attn's online softmax vs its materialized `softmax(QKᵀ)·V` recipe.)

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

The optimizer's output is not N parallel complete graphs, nor a per-node side-table. It is the base map transformed **in place** into the multi-path graph: alternatives are **paths** that diverge at **decision points** (branch points — places where ≥2 viable alternatives exist) and reconverge later, with a fixed **run** between decision points. At any single branch a small number of local alternatives are preserved; globally these compose into whole paths, and the optimizer keeps a **bounded per-device Pareto frontier** of those paths (see [Bounding the frontier](#bounding-the-frontier-pareto-per-device--crowding-cap)) — not an unbounded product. The runtime route picker resolves the surviving paths at decision points at dispatch time, reading current backend telemetry.

This is strictly more flexible than "top-N complete plans": the reachable space across M decision points is far larger than the N complete plans a fixed top-N would cover, and the picker can mix and match — choose one alternative at decision point A, another at decision point B — based on whatever telemetry says about current device load. The bounded Pareto frontier is what survives of that space (next).

### Bounding the frontier: Pareto per device + crowding cap

The per-decision-point alternatives compose into whole **paths**, and the optimizer must bound how many paths survive or the search explodes — a 32-layer graph with a few alternatives per region blows past thousands of surviving paths within ~14 layers (the prototype behind [10-decisions-log §2026-06-14](10-decisions-log.md) demonstrated exactly this). Two complementary mechanisms keep it bounded, and they are why "the plan is the graph" is tractable rather than a combinatorial trap:

- **The right axes keep the natural frontier small and lossless.** The optimizer keeps, **per ending device**, the Pareto-optimal paths over the ranker vector. Because the rankers are kept low-dimensional — **one central time metric** (not a min/max pair), **memory as discrete tiers** (not continuous bytes), **discrete precision/accuracy levels** — the Pareto frontier stays naturally small (order 10² across a deep model) and is **lossless**: a path dominated on the same ending device can never beat its dominator downstream (futures are identical, costs additive/monotone), so dropping it loses no reachable optimum. In the prototype, this alone held the frontier flat at ~100 paths over 128 regions with no cap.

- **A crowding-distance cap is the hard backstop.** When a device bucket's frontier exceeds a configured `keep`, it is reduced NSGA-II-style by crowding distance — retaining the per-axis extremes and the best frontier-*spanners*, **not** the top-N on a single metric (which would discard whole tradeoff dimensions). The prototype found `keep ≈ 32/device` matches the no-cap optimum on every runtime query (fastest / most-precise / least-memory / balanced) and bounds even adversarial continuous-axis cases.

This is why **multiple paths survive per device deliberately** (a fast-but-memory-heavy path and a slow-but-light one are both Pareto-optimal; the runtime picks by current memory pressure), why **a path may span devices** (colocation at one node ≠ mergeable), and why convergence merges only **forward-identical** paths and never strands the last path for a (device, backend). The lock-step discipline — run the merge/discard optimizers *after each pathfinder step*, not once at the end — keeps the working set bounded throughout, which (with mmap-backed graph storage) also keeps `optimize_graph` within memory even for very large models, provided its traversal stays local.

### What counts as a decision point

A decision point is any location in the optimized DAG where the optimizer found ≥2 viable alternatives during its search. The most common kinds:

- **Kernel-variant choices**: same op, different kernel implementations (e.g., cuBLAS vs custom matmul; different tile shapes; different precision-tradeoff variants for the same backend).
- **Placement choices**: same op, different (backend, device) assignments.
- **Fusion-vs-decomposition choices**: a subgraph that can be left as primitives OR fused into a single registered fused op.
- **Algebraic-rewrite choices**: a subgraph that has equivalent algebraic forms with different op counts or operand traffic.
- **Tolerance-trade-off choices**: a region that can run strict OR with a non-zero tolerance budget for a faster approximate variant.

### Storage and structural-distinctness

- Alternatives are paths in the multi-path graph, not copies of whole graphs; shared regions are shared. Total storage scales with the bounded per-device frontier (order 10² paths), not with N complete graph copies — well within budget for typical inference graphs.
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

### What makes an alternative admissible

- It must satisfy all hard constraints: tolerance budget not exceeded along any path through it, viable backend assignment, all transfers feasible, all layouts reachable via fixup ops.
- Among admissible alternatives, survival is decided **globally** by the per-device Pareto frontier + crowding cap (see [Bounding the frontier](#bounding-the-frontier-pareto-per-device--crowding-cap)), not by a fixed per-branch count. How many alternatives a single branch shows is *emergent* — whatever the surviving paths imply there — not a separately-tuned N.

### The tunable: `keep` per device (not a per-decision-point N)

The real knob is **`keep` — the crowding-cap size per ending device** (≈32 in the prototype). It bounds the whole surviving frontier (≤ `keep` × devices), so cost scales with the *bounded frontier*, not an unbounded N × M:

- **Search + storage** scale with the bounded per-device frontier (order 10² paths total with the right axes — single central time, tiered memory, discrete precision/accuracy), because lock-step pruning keeps the working set bounded throughout, not just at the end.
- **Runtime picker cost per realize** scales with the number of decision points actually hit, mitigated by caching (decisions are stable while telemetry is stable; re-decide only on a meaningful shift).

(A *single* small N "across all devices" — the old "default N=3" framing — is explicitly **not** the model: it would strand slow devices, the scalar-top-N failure the per-device frontier avoids. See 03-ir [Bounding the frontier](03-ir.md#bounding-the-frontier-pareto-per-device--crowding-cap).)

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

This concurrent model is more aggressive than what's currently implemented. The optimizer is now multi-path (per-decision-point alternatives + the bounded per-device frontier) and the runtime route picker resolves arms by telemetry, but optimization still runs synchronously to completion before execution starts — there is no sliding frontier yet. Migrating to the concurrent model is itself a multi-phase project (transactions first, then frontier tracking, then `Concurrent`/`WholeGraph` rule classification, then per-realize concurrency policy). The architectural commitment is the destination; the migration is phase work.

## Load-time incremental planning + cost-based placement (v0.4)

Decided 2026-06-11. This section gives the concurrent model above its
driver and its placement algorithm. Session prompt:
`docs/session-prompts/load-time-incremental-planner.md`.

### Planning starts at model load

The optimization producer starts when graph nodes start existing —
at model load for loader-built graphs, at op-construction time for
user-built graphs — not when `realize()` is entered. `realize()`
becomes: *wait until the plan frontier covers the requested roots,
then dispatch against the committed plan*. Dispatch never waits for
global optimality: the first op launches on the best placement known
at that moment (usually the device the input data is resident on),
and the planner — working in microseconds while kernels work in
milliseconds and weight pages fault in from disk — maps far ahead of
the execution frontier. Everything behind the frontier is sunk;
everything ahead stays fluid (per-decision-point atomic swap, as
above).

With mmap'd weights this pipelines three activities: page-in of a
layer's weights, planning of downstream layers, and execution of
upstream layers all overlap. The plan itself is the **prefetch
schedule**: it states which weights are needed on which device in
what order, so residency management becomes planned prefetch instead
of demand faulting (see [06-runtime](06-runtime.md)).

### Cost-based placement: the carry-forward DP

Device placement is decided by cost, not locality policy. The
objective is end-to-end wall-clock of the sequence — which makes
placement non-local: a locally-fast kernel (fused or not) on device A
loses if it strands residency that a later segment needs on device B.
Per-node greedy choice cannot see that; window enumeration over long
sequences is unnecessary. The mechanism is a carry-forward dynamic
program over the topo order:

- **State**: for each frontier node, `best[d]` = lowest accumulated
  cost to arrive here with output resident on device `d`. O(devices)
  state per node — the accumulated state *summarizes* the entire
  prefix, which is what makes unbounded segments checkable without
  re-examining them.
- **Extension**: when node N arrives, for each device `d`:
  `best_N[d] = min over d' of (best_{N-1}[d'] + transfer(d'→d,
  boundary bytes) + cost(N on d))`, where `cost` is the layered cost
  model below and `transfer` uses calibrated numeric estimates
  (bytes ÷ measured bandwidth + per-transfer latency) per
  `SystemTopology` path.
- **Fused jumps**: fusion candidates enter the same recurrence as
  multi-node edges — `best_N[d]` also considers
  `best_{N-k-1}[d'] + transfer + fused_cost(N-k..N on d)` for every
  registered pattern matching the trailing subgraph. Fusion lookback
  is **bounded** by the longest registered pattern (~5 ops), so the
  per-node work is O(devices² + patterns·devices), and a fused op
  that would strand residency is out-competed naturally: the DP
  never finalizes a choice until downstream extensions roll its
  consequences in.
- **Commit**: choices finalize only at the execution frontier
  (backtracking from the best current end-state); ahead of the
  frontier the plan revises freely as the table extends.
- **DAG branches**: the chain recurrence extends over a topo
  linearization; at joins the state merge considers both producers'
  residencies. Model trunks are chain-dominated; branch handling may
  start heuristic (merge to the cheaper producer device) and tighten
  later.

Fusion *pattern matching* remains subgraph-shaped, not window-shaped
(per the DecompositionMap/OptimizationMap rules): when node N
arrives, patterns are re-probed against the neighborhood within the
longest-pattern distance of N. Linear windows would both test
non-adjacent pairs and miss diamond patterns (FlashAttention's
matmul→softmax→matmul).

### Plan-fragment memoization (repeated structure)

Models repeat structure (32 identical transformer layers). Plan
fragments memoize by **structural hash** (op sequence + shapes +
dtypes + params): the planner maps the first instance honestly and
stamps the rest. The same hash keys three further reuses:

- **Persisted-plan cache** fragments (per
  [11-persistence](11-persistence.md)) — a second load of the same
  or a structurally-similar model skips planning for matched
  fragments.
- **Record-once-replay execution** — a repeated fragment is the
  natural capture unit for CUDA Graphs / pre-recorded Vulkan command
  buffers, replayed per instance with rebased weight pointers.
  Kernel *binaries* are already cached per (kernel, device); what
  repetition adds is amortizing launch orchestration.
- **Activation-pool sizing** — identical fragments have identical
  intermediate shapes; one fragment-sized buffer ring serves all
  instances instead of per-instance allocations. (Weights do NOT
  dedupe — instances share structure, not parameter values.)

### Horizontal fusion / similarity scheduling (future rule family)

When repeated subgraphs are mutually *independent* (no path between
them — per-head projections, MoE experts, batched branches; NOT
sequential layer stacks, which are data-dependent), the scheduler may
group same-shaped ops adjacently and a horizontal-fusion rule may
merge them into one batched kernel launch. The win is launch-count
and occupancy, not kernel loading (binaries are already cached); the
cost is higher peak activation memory (more intermediates live
simultaneously), which the residency planner arbitrates. This is a
rule family + scheduling freedom inside `derive_ordering`, gated
entirely on graph-derived independence.

## Rankers and the cost model: a per-path cost vector, ranked

A path is ranked on a **cost vector**, not a scalar — the dimensions the **rankers** produce: **time** (a single central metric — median/average for throughput, a tail percentile for latency-SLA; the Judge measures the full distribution, but the optimizer optimizes on *one* mode-selected metric — `t_min` / "fastest best case" is deliberately *not* a selection axis), **precision** (digits), **accuracy** (ULP / rounding / monotonicity), and **memory** as a **per-tier vector** (host-RAM and device-VRAM footprints tracked separately, since which tier binds depends on the target machine). The per-device Pareto frontier (above) is dominance over this whole vector; ties break **precision → accuracy → memory**. Keeping time to one axis and memory/precision/accuracy discrete is what makes that frontier small (see [Bounding the frontier](#bounding-the-frontier-pareto-per-device--crowding-cap)).

Each rule and each backend's kernel implementation contributes per-node cost estimates. The optimizer composes them over a candidate path to produce its cost vector — the *time* dimension as a *wall-clock* cost — then ranks paths by Pareto dominance over the vector.

**Wall-clock, not strict-serial.** Independent subgraphs in the DAG can execute concurrently across backends and across same-backend slots (see [06-runtime](06-runtime.md) for the dispatch model). Plan cost is therefore composed as `wall_clock ≈ max(parallel_branches) + serial_remainder`, not as the sum of per-node costs. A plan that exposes two long parallel branches has wall-clock cost ≈ max(branch_a, branch_b), not branch_a + branch_b. Plans that maximize useful parallelism rank higher than serially-equivalent plans even when their summed cost is identical.

**Three layers of cost data, composed:**

**Layer 1 — static annotations, optionally refined by community-aggregated empirical priors.** Each rule annotates its cost contribution: FLOPs delta, bytes-moved delta, kernel-overhead delta. Each backend-impl annotates its cost as a function `cost(shapes, params, capabilities) -> CostEstimate { flops, bytes_moved, kernel_overhead_ns }`. Static annotations are pessimistic upper bounds; conservative on uncertainty. Plans ranked by static cost alone give a reasonable first-order ordering but miss bandwidth saturation, queue contention, hardware quirks, and parallelism interactions.

**Per-backend throughput closes the FLOP → time gap (the composite figure).** The `CostEstimate` counts raw FLOPs and bytes; converting those to a sortable nanosecond figure is a **roofline** parameterized by the *candidate backend's* throughput: `composite_ns ≈ max(flops ÷ compute_throughput, bytes_moved ÷ mem_bandwidth) + kernel_overhead_ns`, where `compute_throughput` (FLOPs/ns) and `mem_bandwidth` (bytes/ns) come from that backend's `BackendCapabilities`. This is what makes **cross-device placement honest**: the same op priced on a GPU's throughput yields fewer nanoseconds than on a CPU's, so the priced placement DP prefers a GPU for large parallel work (the throughput win dwarfs the inbound-transfer cost) while a small op stays local (transfer dominates). A backend's rates are the authoritative registered figure where caps are in hand (the placement DP), and a matching per-backend prior where they aren't (the candidate rank, which sees only the backend id); the two are derived from the same constants so they agree until the Judge refines a cell. The rates are deliberately conservative, directionally-correct priors — CPU is the historical neutral baseline (1 FLOP/ns, 4 bytes/ns), GPUs a higher tier — and `kernel_overhead_ns` is added serially and *never* scaled, so Layer 2's measured latency (packed into the overhead term) passes through unchanged. This realizes the throughput refinement the earlier drafts deferred to "Phase 1.5"; it is directionally correct, not calibrated — Layer 2 supplies calibration.

**Fused-op Layer-1 default — cost composed from the op's decomposition (no zeros).** A fused or synthesized op with no declared or measured cost previously registered a zero-cost sentinel (`fused_unknown_cost`), and a zero-priced candidate wins spuriously — the exact mis-pricing Part A fixed for GPUs. The **recipe principle** guarantees every fused op carries a total `decompose` (fused → equivalent primitive subgraph), and every primitive already has a Layer-1 `cost` fn. So a fused op's Layer-1 cost is **composed from its decomposition**: `flops = Σ` over the decompose subgraph's primitive nodes of their per-node FLOPs (arithmetically exact for algebraic fusions — gelu/silu/rmsnorm/softmax); `bytes_moved =` the fused op's own **boundary I/O** (its declared inputs + final output — fusion elides the intermediates, so this is the *tight* estimate, not `Σ` intermediate bytes); `kernel_overhead_ns =` **one** launch overhead (the fused kernel launches once, not `N` times). This is an **optimizer-level default** computed where the graph + `decompose` are both in hand (`fuel_dispatch::fused_cost::{cost_from_decompose, fused_layer1_cost}`) — NOT inside the bare `cost(shapes, params, caps)` fn pointer, which has no graph. It fires **only for the sentinel**: a fused op WITH a declared cost (a contract `cost:` expression) or a Judge measurement is priced by that instead, unchanged — preserving the degrade-gracefully layering **measured › declared › composed-from-recipe › (never) zero**. The composition is available immediately (no Judge data required) and directionally safe: for algorithm-changing fusions (flash-attention) the decompose FLOPs are an approximation the Judge later corrects downward — a missed win until measured, not an optimistic irreversible commit. Never-panic: a degenerate / fixpoint / unregistered decompose falls back to the sentinel-equivalent zero rather than crashing; a nested fused op in the decompose recurses (the base map is `decompose`'s fixpoint). See [fused-op-cost-from-decompose](../session-prompts/fused-op-cost-from-decompose.md).

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
- **Per-decision-point alternatives are now landed.** The driver tracks the bounded per-device Pareto frontier at each location and records its surviving paths as `Op::Branch` decision points on the optimized graph — the per-device frontier + crowding cap replaced PR 3's first-match-wins single transformation. The `Rule` trait surface is unchanged, as committed.
- **Concurrent optimize-and-execute is not yet landed.** Production optimization runs synchronously to completion inside the realize bridge, then execution proceeds; the sliding-window model (frontier tracking + atomic-swap commit) remains the destination. The runtime *route picker* that selects an arm per branch by live telemetry **is** landed (it resolves the surviving paths at dispatch time); what remains is the monotonic optimization frontier that lets optimization and execution overlap.

The progression from PR 3 to the architecture in this section is incremental: each piece (declarative engine, registry-auto-generation, top-N, frontier, concurrent execution) is a separable phase. PR 3's API is preserved; the driver evolves.

## What this rules out

A few non-features called out explicitly:

- **No e-graph saturation on the per-realize hot path.** E-graphs (egg, equality saturation) are unsuitable for hot-path optimization. But the offline multi-path path-search `optimize_graph` itself performs is e-graph-*adjacent*, and that is **in-bounds**: `optimize_graph` runs at load/import, not per realize. E-graph techniques may also feed the OptimizationMap as an offline rule-discovery tool. What stays out is e-graph saturation *on the per-realize hot path* (see [09-non-goals](09-non-goals.md)).
- **No autotuning-search-style optimizer.** TVM-style autotuner search produces excellent results but is operationally heavy. Fuel's optimizer is heuristic + cost-driven, not search-driven. Empirical Judge data fills the gap autotuning would otherwise fill.
- **No *untrusted* user-installable rules at runtime.** Letting an end user hot-load arbitrary optimization code introduces a security and stability surface fuel doesn't take on (see [09-non-goals](09-non-goals.md)). **This bounds trust + provenance, not "runtime" per se** (reconciled 2026-06-20, [10-decisions-log](10-decisions-log.md)): the OptimizationMap is populated at startup for *untrusted* sources, but **trusted, Fuel-orchestrated, cost-gated runtime registration of new fused-op identities** (Tier 2) is in-bounds. There, Fuel chooses the sub-base-map region to fuse (strategy stays in the optimizer), a trusted backend synthesizes the kernel, the result arrives as a **declarative recipe** (`decompose` + `pattern` over existing primitives + an FKC `PrecisionGuarantee`), and the route picker cost-gates adoption — no untrusted code, no new primitive. The implementation mechanism is the declarative-pattern engine (today stubbed, `PatternKind::Declarative => false` at `fuel-graph/src/opt.rs`), which is the prerequisite for that runtime registration. The kernel binding table (implementations, distinct from rules) is already runtime-extensible (`extend_global_bindings`).
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
- [10-decisions-log §2026-05-09](10-decisions-log.md) — records the cross-cutting questions (Q-A through Q-E, from the now-removed `architecture-audit.md`) that this section's commitments resolved.
