# Runtime

**Status**: v1.0 (2026-05-09). v1.0 changes: (1) atomic-swap commit is per-decision-point, not whole-DAG — re-optimization preserves cached alternatives and merges them with locally-discovered ones, atomically swapping each decision point's alternative set as the optimizer reaches it; (2) re-optimization scope is computed from the trigger using per-decision-point dependency records — most triggers (device removed, kernel updated, profile data refined) affect a small subset of decision points, not the whole graph; (3) optimization cache is mmap'd at startup with lazy KernelRef resolution — startup is near-instant for cache hits, only the cache header is touched before the first realize. v0.2 added "Background re-optimization" and "Local Judge baseline initialization" sections.

How fuel goes from "the optimizer has produced an optimized form" to "outputs are computed." The route picker, telemetry-driven decision making, dispatch lookahead, data parallelism scheduling, and the executor's interaction with backends.

The runtime is the consumer of everything earlier sections produce: it reads the optimized DAG (with per-decision-point alternatives from [04-optimization](04-optimization.md)), the backends' static capabilities and dynamic telemetry (from [05-backend-contract](05-backend-contract.md)), and the user's per-call configuration (tolerance overrides, concurrency policy, route preferences). It commits decisions as needed at dispatch time and produces outputs.

---

## The runtime's responsibilities

The runtime owns four concerns:

1. **Per-decision-point alternative resolution.** When the optimized DAG has alternatives at a decision point, the runtime picks among them based on current telemetry.
2. **Dispatch scheduling.** Tracking which nodes are ready to execute (inputs available + finalized by the optimization frontier), assigning them to available backend slots, dispatching them.
3. **Data-parallel execution.** Dispatching independent subgraphs concurrently when slot capacity and memory permit.
4. **Synchronization at join points.** Waiting on outputs from multiple parallel branches before proceeding to a join op.

Three things the runtime does *not* own:

- **Strategic decisions** (placement, fusion, kernel-variant choice across alternatives) — those are the optimizer's. The runtime executes the alternative the picker chooses.
- **Kernel implementation** — that's the backend's. The runtime calls `KernelRef` function pointers; what they do is opaque.
- **Cache management** — that's [11-persistence](11-persistence.md). The runtime can load a cached optimized form into the route picker but doesn't decide cache policy.

## Route picker: per-decision-point resolution

The route picker is the runtime's reasoning surface. Per realize, it walks the optimized DAG and resolves alternatives at each decision point:

```text
For each decision point in the optimized DAG:
    Read the per-decision-point alternative set.
    Read current backend telemetry (slot availability, memory pressure, queue depth).
    Read per-call configuration (tolerance override, concurrency policy).
    Filter alternatives by hard constraints:
        - tolerance budget admissible
        - concurrency policy compatible
        - target backend has live slot capacity
    Among remaining alternatives, pick the one whose cost (with conditional
        adjustments from already-resolved upstream decisions) is lowest.
```

The picker's output: a coherent execution plan — one alternative chosen per decision point, with all conditional cost adjustments resolved into a single total cost estimate.

### Telemetry caching for picker speed

Naive per-realize re-resolution is expensive: M decision points × telemetry read × cost compare per realize. For a 30-layer transformer with 4 decision points per layer, that's 120 decisions per realize, each requiring telemetry plus cost lookups.

The runtime caches the resolved plan. Per realize:

1. Check whether telemetry has changed meaningfully since the last realize (memory pressure delta > threshold, slot availability delta > threshold, etc.).
2. If yes, re-resolve all decision points and update the cached plan.
3. If no, reuse the cached plan.

In steady state (realize-after-realize on a stable system), telemetry doesn't change much; the cached plan is reused. Re-resolution happens on transitions (memory pressure rising, a new device becoming available, the user changing concurrency policy).

### Resolution order matters when decisions are coupled

Decisions with conditional cost adjustments (placement choices that affect transfer costs at downstream joins) need to be resolved in topological order — upstream first. The picker walks decision points in topological order so that by the time it reaches a downstream point, its upstream decisions are already committed and the conditional adjustments evaluate to concrete numbers.

Locally-greedy resolution at each point (pick the locally-best alternative given upstream commitments) is the default. Rare adversarial cases where greedy is bad get caught by a small lookahead — the picker considers the next K decision points jointly, re-decides upstream if a downstream constraint forces it. K is bounded (default 3); deeper lookahead has diminishing returns.

## Dispatch lookahead: the commit boundary between decisions and execution

The route picker decides; the executor dispatches. The interface between them is the **dispatch lookahead window**.

Three policies, each with a clear use case:

- **Shallow (just-in-time)**: dispatch one op at a time, decide just-in-time when a slot becomes available. Maximum adaptivity. Risk: backends idle while the picker thinks. Useful for: very small ops where decision overhead dominates execution.
- **Deep**: queue many ops to each backend's slot in advance. Backends always busy. Risk: by execute-time, telemetry has changed and the decision is stale. Useful for: very large ops where execution dominates and decision freshness barely matters.
- **Bounded (default)**: queue exactly enough ops to keep advertised available slots full. If backend X advertises 4 currently-available streams, queue 4 ops to it; queue more only when one finishes. Balances adaptivity vs throughput; backends never idle, decisions stay current.

Bounded lookahead is the architectural commitment as the default. The bound = sum of all backends' currently-available slot counts. The runtime watches slot availability via telemetry; as a slot frees, it dispatches one more op to that backend's queue. Each new dispatch reads current telemetry and re-resolves its decision point if necessary.

### Cancellation is not supported

Most backends don't support cleanly canceling already-queued work. CUDA streams are FIFO; once submitted, kernels run to completion. Vulkan command buffers are roughly the same. CPU work submitted to a thread pool can be revoked if not yet started but reliably canceling in-flight work is hard.

The architectural commitment: **dispatched ops are committed**. Revision happens by *not dispatching* the next op yet, not by pulling back already-dispatched ones. Bounded lookahead keeps the staleness window small (bounded by execution time of currently-queued ops); if telemetry shifts during that window, the next dispatch reflects the new state, but the in-flight ops complete.

## Data parallelism: independent subgraphs run concurrently

Two subgraphs in the DAG with no shared input paths are independent and may execute concurrently. The runtime exploits this:

- The ready-set tracks all nodes whose inputs are available and frontier-finalized.
- Each ready node is dispatched to an available slot on its assigned backend.
- Across multiple backends, multiple nodes are simultaneously in flight.
- Within one backend, multiple slots dispatch concurrently to the device's parallelism primitives (CUDA streams, Vulkan queues, CPU sub-pools).

Backends own intra-kernel concurrency (per [05-backend-contract](05-backend-contract.md)); the runtime owns inter-kernel parallelism. Slot assignment is the runtime's mechanism — backends advertise slot capacity, runtime allocates work to slots, backends execute on the named slot.

### Cross-device transfers under parallelism

If subgraph A on CUDA and subgraph B on CPU both produce inputs needed by a join node on Vulkan, the optimizer planned for this with explicit transfer ops (per [04-optimization §Cross-cutting transformations](04-optimization.md#cross-cutting-transformations-the-optimizer-is-responsible-for)). The runtime dispatches transfers as soon as their source data is ready — concurrent transfers overlap with concurrent execution. No "wait for current step to finish" stall.

### Memory pressure as the parallelism limit

Parallel execution doubles peak activation memory (sum of in-flight activations across parallel branches). The runtime watches memory pressure; if pressure approaches device limits, it serializes additional dispatches even when slots are nominally available. The route picker's cost model already penalized routes with high parallel memory pressure, so this case should be rare; when it happens, the runtime's behavior is "throttle, don't fail."

### Determinism

Parallel execution introduces scheduling non-determinism (which of several ready ops dispatches first depends on slot availability). Outputs are still bit-deterministic per op; total wall-clock time is non-deterministic because scheduling order varies.

For inference: this is fine. For training (where reduction operations need ordered accumulation for bit-reproducibility): the optimizer can place ordering constraints (forcing a serial reduction order). Default is parallel; constraint flags trigger serial.

## Synchronization at join points

When a node depends on multiple upstream nodes, it can't dispatch until all upstreams have completed. The runtime tracks input-readiness per-node:

- Each node has a count of unresolved inputs.
- When an upstream completes, downstreams' counts decrement.
- Nodes whose count reaches zero (all inputs ready + frontier-finalized) move to the ready set.

This is standard Kahn-style scheduling, applied to the optimized DAG. The DAG already encodes the synchronization structure; the runtime just tracks readiness against it.

## Background re-optimization

When the runtime loads a cached optimization plan (per [11-persistence](11-persistence.md)) — particularly one downloaded from a remote source whose static cost annotations may not perfectly match local empirical reality — the cached plan becomes the *active* plan immediately. TTFT is fast; the user gets a runnable graph.

In parallel, a background optimizer thread runs the optimization pipeline using the local Judge's empirical data, not the static-only data the cache producer used. The re-optimization works **per decision point with merged alternative sets**, not as a whole-graph candidate-vs-active comparison:

1. The cached DAG is the working state.
2. The optimizer walks decision points (typically topological order).
3. For each decision point:
   - Take the union of (cached alternatives) and (alternatives the local optimizer can produce by re-applying rules with local empirical data).
   - Re-rank against local empirical cost.
   - Deduplicate structurally-equivalent alternatives.
   - Keep top N (default 3).
   - Atomic-swap the decision point's alternative set in place.

This gives both the merge property (cached alternatives that were structurally good aren't lost just because re-optimization happens — they get re-ranked alongside new candidates) and the incremental-obsoletion property (improvements become usable as soon as the next decision point's swap commits, not at the end of all optimization). Layers 1-5 might benefit from refined alternatives within seconds while layers 6-32 are still being processed.

**Trigger policy.** Re-optimization runs when there's new information to incorporate, not on a clock:

- **First load after downloading a cache** (the empirical-vs-static gap is largest here).
- **Judge data accumulation crosses a meaningful threshold** (enough new profile entries that one of the active plan's decisions could plausibly flip).
- **Backend telemetry shifts meaningfully** (new device added; previous fast device removed; sustained memory pressure shift).
- **Format-version migration** (per [11-persistence](11-persistence.md#format-memory-mappable-sibling-file-schema-versioned)): when newer fuel reads an older-format cache, background re-optimization opportunistically migrates the cache to the current format as a side effect of producing the refined plan. No separate migration pass needed.

**Per-decision-point atomic swap.** The architecture's commit mechanism is per-decision-point, not whole-graph. Each decision point's alternative set is held in an `Arc`-shared slot; commit is an atomic `Arc` swap of that slot. The route picker holds whichever Arc it loaded for a particular decision point at decision time; writers swap a new Arc into the slot; the old Arc lives until all readers release it. No locks on the hot path; consistent reads always.

This same per-decision-point swap mechanism is used by concurrent optimize-and-execute (per [04-optimization §The sliding window](04-optimization.md#the-sliding-window-optimization-and-execution-overlap)) when the optimization frontier passes a decision point and commits one of its alternatives. Background re-optimization and concurrent execute share the commit primitive; they differ only in what triggers a swap and what alternatives populate the post-swap set.

**Composition with concurrent execute.** The two are complementary. Concurrent execute helps the *first-ever* realize on a graph fuel hasn't seen before (optimization frontier slides during execution; per-decision-point commits as the frontier passes each point). Background re-optimization helps the *second-and-subsequent* realizes after a cached plan is in use (per-decision-point swap as refined alternatives are discovered). Together they cover both cold-start and steady-state TTFT improvements.

## Scoped re-optimization

Most triggers don't require touching every decision point in the graph. The runtime computes the **affected scope** from the trigger and runs re-optimization only on that scope:

| Trigger | Affected decision points |
| --- | --- |
| Device removed | Decision points with at least one alternative placed on that device. Alternatives using the removed device are pruned; if a decision point ends up empty, the optimizer re-runs to find replacements. |
| Backend kernel-revision hash changed | Decision points whose alternatives reference the changed kernel. Re-cost (and possibly re-generate) just those. |
| Profile data refined for cells `(op, dtype, size_class, backend, device)` | Decision points whose cost estimates depend on the refined cells. Often just re-rank existing alternatives without re-running rules. |
| Tolerance configuration changed | All decision points (the precision-filter pass needs to re-evaluate admissibility per [04-optimization §Precision-filter pass](04-optimization.md#precision-filter-pass-runs-before-cost-ranking)). |
| New device added | All decision points (the new device might be a better placement target for any of them). Genuinely global. |
| Backend feature-flag toggled (CUDA enabled/disabled) | Same scope as device removed/added. |

Most triggers are localized; only "new device added" and "tolerance config changed" require touching every decision point. For localized triggers, the savings are substantial — a partial re-optimization affecting 20% of decision points runs ~5× faster than full re-optimization.

**Mechanism.** Each decision point keeps a small dependency record — which kernels its alternatives reference, which devices, which cells of profile data its costs depend on. When a trigger fires, the runtime intersects the trigger with these records to compute the affected set. Re-optimization scopes to that set; unaffected decision points keep their cached alternatives untouched.

This composes cleanly with per-decision-point swap: scoped re-optimization replaces only the affected decision points' alternative sets via the same per-point swap primitive used elsewhere.

## Local Judge baseline initialization

The empirical Judge accumulates per-(op, dtype, size_class, backend, device) latency measurements during normal execution. By default, the Judge starts empty on a fresh installation; measurements accumulate over time as ops run.

Optionally, the Judge can initialize from a **community-aggregated baseline profile** for the user's hardware fingerprint (per [08-pattern-harvest §Shared infrastructure with tolerance recipes](08-pattern-harvest.md#shared-infrastructure-with-tolerance-recipes)). The framework downloads the latest community-aggregated summary statistics for the user's fingerprint, populates the local Judge with these as starting values, and then refines them with local measurements as they accumulate.

The benefit: cold-start cost-model accuracy goes from "static FLOP-counting only" to "community baseline plus local refinement." The route picker's first-realize decisions are calibrated against actual measured behavior on similar hardware, before the user has run anything locally to measure.

This is opt-in (user must enable community-telemetry to participate; baseline download is a one-time fetch). It's also coarse — community medians per cell aren't perfect predictors of individual hardware; local measurements override the baseline as confidence accumulates. Architecturally it's the same trajectory as the cache-generation tool's empirical-priors integration: community data refines starting estimates; local data eventually dominates.

## Concurrent optimize-and-execute interaction

When the optimizer is running concurrently with execution (per [04-optimization §The sliding window](04-optimization.md#the-sliding-window-optimization-and-execution-overlap)), the optimization frontier moves through the DAG. The runtime's ready-set tracking is unchanged — it still gates dispatch on input-readiness — but it adds a *finalization* check: a node is only dispatched if it's also frontier-finalized.

The optimizer commits per-decision-point alternatives at the frontier (collapsing the alternative set to one as the frontier passes). The runtime sees the committed alternative as the only choice from that point forward; alternatives upstream of the frontier are already-decided.

## What the runtime persists

The runtime is largely stateless across realizes — it dispatches against the optimized DAG and returns. Two cross-realize state items:

- **The cached resolved plan** (from the route picker's telemetry-caching optimization). Reused while telemetry is stable.
- **The optimization cache file** (from [11-persistence](11-persistence.md)). **Memory-mapped at process startup**, not read into memory. Only the cache header (a few hundred bytes) and the per-decision-point index get touched immediately — enough for the route picker to start work. Pages for individual node data load on first access via the OS page cache; pages for never-picked alternatives may never load at all. Cache files are designed mmap-friendly (relative offsets, no process-absolute pointers; per [11-persistence §Format](11-persistence.md#format-memory-mappable-sibling-file-schema-versioned)).

Beyond these, every realize starts fresh. No carry-over scheduling state.

### Lazy KernelRef resolution pairs with mmap

The pre-resolved `KernelRef` per node (per [03-ir §The optimized form](03-ir.md#the-optimized-form-top-n-routes-with-pre-resolved-kernels)) is *resolved lazily*: when the route picker chooses an alternative at a decision point, the runtime resolves `KernelRef`s for nodes in that alternative just-in-time via `binding_table.lookup(op_kind, dtypes, backend)`. Trivial cost (~100 ns per lookup), amortized over node execution.

Combined with mmap, this means startup is essentially instant for cache hits — only the cache header and per-decision-point index are touched before the first realize. Alternatives the route picker never chooses never get their KernelRefs resolved; their pages stay in the OS page cache, possibly never paged in.

### Cache updates: write-new-file-and-swap

When background re-optimization commits a refined plan or per-decision-point alternative set update (per [Background re-optimization](#background-re-optimization)), the cache file is updated via **write-new-file-and-swap**: the runtime writes the refined plan to a sibling temp file, fsyncs, atomically renames to replace the original, then mmaps the new file. The old mmap's pages drop from cache naturally as the OS reclaims memory.

This avoids in-place writable-mmap modification, which is harder to do safely (a crash mid-write corrupts the cache), platform-dependent (Windows mmap-write semantics differ from Unix), and forces the file to be writable in environments where it might be read-only.

### Mmap fallback

Some embedded platforms and certain WASM environments don't have mmap. The runtime detects mmap support at startup; uses it where available, falls back to read-into-memory mode where not. This adds one capability check at startup and a slow-path read implementation; no architectural cost.

## What this rules out

- **No runtime kernel selection across alternatives the optimizer didn't preserve.** If a kernel variant isn't in the per-decision-point alternative set, the runtime can't reach it. The optimizer is responsible for surfacing competitive alternatives; the runtime picks among what's available.
- **No silent runtime fallback to a different op.** If the chosen kernel fails (OOM, hardware fault), the runtime surfaces the error. It doesn't transparently switch to a different alternative without telling the user. Future feature: "fallback alternatives" registered as runtime-fallback paths, but not v1.
- **No dynamic graph extension during execution.** The DAG is fixed before execution begins. New ops can be added between realizes (e.g., autoregressive decoding extending the graph for the next token), but not mid-realize.
- **No runtime profile data accumulation that influences this realize.** The Judge measures realize-by-realize and updates its profile; the route picker reads the latest profile for the next realize. The current realize doesn't adapt to within-realize observations.

## Where this lives in code

The runtime as architectural concept maps to several existing crates:

- **fuel-graph-executor**: the core dispatch loop, ready-set tracking, slot assignment.
- **fuel-graph-router**: the route picker and per-decision-point resolution logic. Reads BackendCapabilities and dynamic telemetry.
- **Per-backend executors** (within backend crates): handle the actual KernelRef invocation, manage same-backend slot semantics.

Implementation detail (specific data structures, threading model, IPC if any) is not architectural and lives in the relevant crates' design docs.

---

## See also

- [04-optimization](04-optimization.md) — produces the optimized DAG with per-decision-point alternatives.
- [05-backend-contract](05-backend-contract.md) — backends advertise slot capacity and dynamic telemetry the runtime consumes.
- [07-tolerance](07-tolerance.md) — per-call tolerance overrides honored by the route picker.
- [11-persistence](11-persistence.md) — optimization-cache loading at startup.
- ROADMAP §"Phase 6" — lazy execution and autonomous scheduling. The runtime is where most Phase 6 commitments live.
