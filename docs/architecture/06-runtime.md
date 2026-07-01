# Runtime

**Status**: v1.5 (2026-06-28). **v1.5 scopes Step E (live-load arm re-picking) as a program with prerequisites.** Investigation found it is not a bounded step: execution is **synchronous** (a kernel blocks until the device finishes — no varying queue depth to react to within a realize) and there is **no load telemetry**. Sequenced **A (async/concurrent execution foundation — the real gate, fuel-internal, large) → B (queue-depth signal: B1 a fuel-internal per-device in-flight counter is primary; B2 optional baracuda/vulkane cross-process telemetry) → C (streaming run-walk + `DeviceLoadSelector` — the per-decision-point re-pick)**. The dispatch-core cleanup's "plan IS the graph" core (Steps A–D) is complete; E awaits async-execution design review before code. Design + asks: [`step-e-async-execution.md`](../session-prompts/step-e-async-execution.md). Core claim unchanged; no decisions-log entry. **v1.5 also reconciles to the 2026-06-20 adaptive-runtime-fusion decision** ([10-decisions-log](10-decisions-log.md)): the "No silent runtime fallback" rule is distinguished from **Fuel-orchestrated JIT registration** — JIT-ing a kernel for an *existing* op identity (Tier 1, the already-extensible binding table) is a v1 reality, only *silent op substitution mid-realize* is the non-goal; **idle-time / whole-machine-resource-aware JIT fusion** is added to the background-re-optimization triggers (Fuel as the strategist, G7); the megakernel is positioned as the extreme, narrow, non-monotonic end of the run-size axis (G6); and missing-fusion telemetry is cross-referenced (G5). **v1.4 moves the route pick from the bridge into the executor** (dispatch-core cleanup Step C): `PipelinedExecutor::realize_with_optimized_picking_env` runs `pick_route` at dispatch; the bridge (`pipelined_bridge.rs`) only builds the Device/Judge-derived selector + live-memory lookup and hands them over — it no longer pre-computes the route (`resolve_runtime_route` deleted). Still once-per-realize; per-decision-point re-picking by **live queue depth** remains the near-term step (Step E). Core claim unchanged; no decisions-log entry. **v1.3 refines landed-vs-intent and salvages two durable facts**: the route picker today resolves one arm per branch once per realize keyed on live per-tier free memory (+ Judge rank + static cost), with live-device-load arm selection (queue depth / stream utilization) and executor-side re-picking named as the not-yet-built near-term step; multi-process / tensor parallelism is flagged as an open lazy-first-class-vs-orchestrate-above design fork; and the decode causal mask is documented as having no dedicated op (CUDA flash `is_causal`; CPU/Vulkan host mask re-bound per pass). Core claim unchanged; no decisions-log entry. **v1.2 implemented the 2026-06-14 "plan IS the graph" redirection** (see [10-decisions-log](10-decisions-log.md)): the runtime consumes the **multi-path graph** (not a per-node alternative side-table); the route picker is the **runtime selector ("Picker 2")** that, at the **few decision points (branch points)**, picks among the surviving per-device Pareto paths by live telemetry including **per-tier free memory**; dispatch is by **run** (the fixed op-sequence between two decision points dispatches as a unit), the executor's per-node lowering being the **work-item producer**; the plan doubles as the **cross-tier prefetch schedule** (disk→RAM for larger-than-RAM, RAM→VRAM for larger-than-VRAM); the local Judge baseline is **bundled in-package** (2026-06-13 decision), not an opt-in download; and the code now lives in **fuel-dispatch** (`PipelinedExecutor` + ranker selectors), the former fuel-graph-router / fuel-graph-executor having been retired. Most of v1.1 stands (load-time planning, background + scoped re-optimization, mmap'd cache, per-decision-point atomic swap) — reframed onto the multi-path graph. v1.1 (2026-06-11): the optimization producer starts at model load; `realize()` = wait-for-plan-coverage + dispatch; the plan is the weight-prefetch schedule. v1.0: per-decision-point atomic swap; scoped re-optimization via dependency records; mmap'd cache + lazy KernelRef resolution.

How fuel goes from "the optimizer has produced the multi-path graph" to "outputs are computed." The route picker, telemetry-driven decisions, run dispatch, data parallelism, cross-tier prefetch, and the executor's interaction with backends.

The runtime is the consumer of everything earlier sections produce: it reads the optimized multi-path graph (paths + decision points from [04-optimization](04-optimization.md)), the backends' static capabilities and dynamic telemetry (from [05-backend-contract](05-backend-contract.md)), and the user's per-call configuration (tolerance overrides, concurrency policy, route preferences). It picks paths at decision points and produces outputs.

---

## The runtime's responsibilities

The runtime owns four concerns:

1. **Path resolution at decision points.** At the few decision points (branch points) of the multi-path graph, the runtime selector picks among the surviving per-device Pareto paths by current telemetry (device load, per-tier free memory). Between decision points the path is fixed.
2. **Run dispatch.** Dispatching the fixed **run** between two decision points as a unit (a pre-recorded command sequence where the backend supports it), not op-by-op — assigning runs to available backend slots as their inputs become ready.
3. **Data-parallel execution.** Dispatching independent runs concurrently when slot capacity and memory permit.
4. **Synchronization at join points.** Waiting on outputs from multiple parallel branches before a join op.

Three things the runtime does *not* own:

- **Strategic decisions** (placement, fusion, kernel-variant choice) — the optimizer's. The runtime executes the path the picker chooses.
- **Kernel implementation** — the backend's. The runtime calls `KernelRef` function pointers; what they do is opaque.
- **Cache management** — [11-persistence](11-persistence.md)'s. The runtime loads the persisted graph and writes refined plans, but doesn't decide cache policy.

## Route picker (the runtime selector / "Picker 2")

The route picker — the runtime selector, **"Picker 2"** (the plan-time ranker in [04-optimization](04-optimization.md) is "Picker 1") — is the runtime's reasoning surface. Per realize it walks the multi-path graph and, at each **decision point** (branch), picks among the surviving paths:

```text
For each decision point (branch) in the multi-path graph, in topological order:
    Read the surviving paths diverging here.
    Read current backend telemetry (slot availability, per-tier free memory, queue depth).
    Read per-call configuration (tolerance override, concurrency policy).
    Filter paths by hard constraints:
        - tolerance budget admissible
        - concurrency policy compatible
        - target device has live slot capacity and the path's memory fits the binding tier
    Among survivors, pick the one whose cost (with conditional adjustments from
        already-resolved upstream decisions) is lowest.
```

Decision points are **few** — most of the graph is a single agreed run — so the picker decides rarely, not per op. Its output: a coherent route through the multi-path graph, one path chosen at each branch, all conditional cost adjustments resolved.

What is **landed today**: the picker resolves one arm per branch, once per realize, keyed on **live per-tier free memory** (a VRAM-pressure guard), the Judge-measured rank, and the static cost winner; under no pressure it degrades to arm-0 and realize is unchanged. **The pick now runs inside the executor** — `PipelinedExecutor::realize_with_optimized_picking_env` calls `pick_route` at dispatch; the bridge only builds the Device/Judge-derived selector + live-memory lookup and hands them in (dispatch-core cleanup Step C, 2026-06-28), no longer pre-computing the route. What is **the architecture's intent but not yet built**: selecting arms by **live device load — queue depth and stream utilization** — and making the pick **per-decision-point during the dispatch walk** so it re-picks as load shifts (rather than the current once-per-realize computation). This is **Step E, now scoped as a program** ([`step-e-async-execution.md`](../session-prompts/step-e-async-execution.md)) — gated on async/concurrent execution (the synchronous executor has no varying queue depth) + a load signal (a fuel-internal per-device in-flight counter, with optional sibling cross-process telemetry). The `slot availability, queue depth` telemetry named in the resolution sketch above is the target signal set. The free-memory-pressure path is the one live runtime signal the picker consumes today.

**Step E Phase A–C landed (2026-06-30).** The async/concurrent foundation (A1 completion-handle seam, A2 Vulkan lazy-flush, A3 CUDA stream-ordered alloc/free, A4 multi-device placement + A4b the CUDA+Vulkan cross-device CONCURRENCY mechanism — eager Vulkan submit + in-flight handles), plus Phase C: **B1** a process-wide per-device in-flight counter behind the `BackendStreams` Tier-2 trait (the [05-backend-contract](05-backend-contract.md) seam); **C1** a streaming run-walk (`walk_picked_route_streaming`) where the compiler thread resolves each `Op::Branch` arm lazily at the frontier (so the pick reads load current to that decision point); **C2** the `DeviceLoadSelector` — a `load_tier` leg folded into the `ChainedSelector` key between the VRAM guard and the latency rank, so an arm on a busy device is demoted (no-load ⇒ byte-identical to the VRAM-only chain); and **C3** the **automatic cross-device overlap reorder** (`fuel_graph::device_alternating_order`) — a critical-path topological reorder of the run dispatch order that emits each device's heaviest compute chunk before the host-blocking cross-device drain, so independent device sub-DAGs dispatch so they overlap, automatically, for an arbitrary graph (previously A4b overlapped only for a hand-arranged dispatch order). The reorder is a scheduling **hint**: it is a valid topological reordering, gated to multi-device run lists (single-device ⇒ identity ⇒ byte-identical), so the realized result is byte-identical regardless of the reorder. **The remaining gap** is the full `06-runtime` end-state: the bounded-lookahead Kahn ready-set scheduler (below — "Data parallelism" + "Synchronization at join points") that dispatches ready runs to least-loaded slots with run-ahead work-item production, replacing the layered streaming-walk + reorder-pass form; and making the wall-clock overlap of an arbitrary reconverge fully independent of the graph's operand order (a residual in the executor's async-submit timing that the static run-reorder does not fully control).

### Telemetry caching for picker speed

The runtime caches the resolved route. Per realize: (1) check whether telemetry changed meaningfully since the last realize (per-tier memory-pressure delta > threshold, slot-availability delta > threshold); (2) if yes, re-resolve the decision points and update the cached route; (3) if no, reuse it. In steady state (realize-after-realize on a stable system) the cached route is reused; re-resolution happens on transitions (memory pressure rising, a device appearing, the user changing concurrency policy). With few decision points the per-realize picker cost is small to begin with.

### Resolution order matters when decisions are coupled

Decisions with conditional cost adjustments (placement choices that affect transfer cost at downstream joins) are resolved in **topological order** — upstream first — so that by the time the picker reaches a downstream branch its upstream decisions are committed and the adjustments evaluate to concrete numbers. Locally-greedy resolution is the default; rare adversarial cases get caught by a small bounded lookahead (default K=3 branches considered jointly).

## Dispatch: runs, not single ops

The route picker decides at branches; the executor dispatches the **runs** between them. A run is the fixed op-sequence between two decision points (most of the graph), and it dispatches as a **unit** — ideally a pre-recorded command sequence (a CUDA Graph / a Vulkan command buffer) replayed with rebased operands — not op-by-op. The executor's per-node lowering of a run into concrete kernel invocations + operand bindings is the **work-item producer** (see [14-lifecycle](14-lifecycle.md)); it runs ahead of execution, preparing the next run while the current one executes.

Dispatching whole runs is what makes lazy pay off over eager: the backend receives a long sequence (amortizing per-op submission overhead — ~5,000 ns/op on Vulkan) while the planner prepares the next run concurrently, instead of paying decision + submission cost per op. Narrowing dispatch to one op at a time would forfeit exactly this advantage.

**The run is a size axis, and the megakernel is its extreme end.** A run dispatches as a unit; how *much* of the graph that unit covers is a continuum — from a single op, through a pre-recorded command sequence (a CUDA Graph / Vulkan command buffer, the common case), up to a single backend-synthesized **megakernel** that fuses a whole region's compute into one launch. That far end is real but **narrow, the last / highest-risk target, never the default** (per the 2026-06-20 decision G6, [10-decisions-log](10-decisions-log.md)): a megakernel wins *something* even over an ideal captured run (it removes inter-kernel scheduling bubbles and enables cross-boundary software pipelining), but the benefit curve **turns over** — fixed launch geometry and kernel-global register allocation impose the worst region's footprint everywhere (sub-kernels do not fix this), and per-shape specialization combinatorics grow. "Bigger fusion = better" is **not monotonic**. Captured-run replay is the cheaper step below it that already captures most of the launch-overhead win without fusing compute. So the default run is a captured command sequence; the megakernel is a profiling-justified exception the optimizer reaches for last.

The **lookahead window** governs how far ahead of execution the picker commits runs:

- **Shallow (just-in-time)**: commit the next run only as a slot frees. Maximum adaptivity; risk of backend idle while the picker thinks. For very small runs where decision overhead dominates.
- **Deep**: queue many runs ahead. Backends always busy; risk that telemetry has shifted by execute-time. For very large runs where execution dominates.
- **Bounded (default)**: keep exactly enough runs queued to fill currently-available slots; queue more only as one finishes. Balances adaptivity vs throughput — backends never idle, decisions stay current. The bound = sum of backends' currently-available slot counts.

### Cancellation is not supported

Most backends can't cleanly cancel queued work (CUDA streams and Vulkan command buffers are FIFO; once submitted, kernels run to completion). The architectural commitment: **dispatched runs are committed.** Revision happens by *not dispatching* the next run yet, not by pulling back dispatched ones. Bounded lookahead keeps the staleness window small (bounded by the execution time of currently-queued runs); a telemetry shift during that window is reflected in the next dispatch, while in-flight runs complete.

## Data parallelism: independent runs run concurrently

Two runs with no shared input path are independent and may execute concurrently:

- The ready-set tracks all runs whose inputs are available and frontier-finalized.
- Each ready run is dispatched to an available slot on its assigned backend.
- Across backends, multiple runs are simultaneously in flight; within one backend, multiple slots dispatch to the device's parallelism primitives (CUDA streams, Vulkan queues, CPU sub-pools).

Backends own intra-kernel concurrency (per [05-backend-contract](05-backend-contract.md)); the runtime owns inter-run parallelism via slot assignment.

### Cross-device transfers under parallelism

When parallel runs on different devices feed a join, the optimizer planned explicit transfer ops (`Op::Copy`, per [04-optimization](04-optimization.md#cross-cutting-transformations-the-optimizer-is-responsible-for)). The runtime dispatches a transfer as soon as its source is ready — concurrent transfers overlap concurrent execution; no "wait for current step" stall.

### Memory pressure as the parallelism limit

Parallel execution sums in-flight activation memory across branches. The runtime watches **per-tier** memory pressure; if a tier approaches its limit it serializes additional dispatches even when slots are free. The picker's cost model already priced per-tier memory into path selection, so this should be rare; when it happens the behavior is "throttle, don't fail." This is also where the per-tier memory ranking pays off: under VRAM pressure the picker prefers a path whose binding tier has headroom (e.g. a CPU/host-RAM path), which is exactly why multiple paths per device are retained.

### Determinism

Parallel scheduling is non-deterministic in *order* (which ready run dispatches first depends on slot availability); outputs are still bit-deterministic per op, only wall-clock varies. For inference this is fine. For training (where reductions need ordered accumulation for bit-reproducibility) the optimizer can place ordering constraints forcing serial reduction order. Default parallel; constraint flags trigger serial.

### Multi-process / tensor parallelism is an open design fork

Single-process inter-run parallelism (above) is the runtime's job. Multi-device-across-processes (tensor/pipeline parallel) currently exists only as scaffolding that orchestrates synchronous collectives *above* the lazy graph — the collective is a blocking call outside realize, not a graph node. Whether the lazy substrate should make collectives **first-class** (collective `Op` variants carrying explicit comm dependencies the optimizer can schedule and overlap with compute) or continue to **orchestrate eager collectives above** an otherwise single-process graph is undecided. The lazy-first-class route is the one consistent with this document's premise (every decision lives in the DAG); it is not yet chosen and not yet built.

## Synchronization at join points

A node depending on multiple upstreams can't dispatch until all have completed. The runtime tracks input-readiness (a per-node unresolved-input count; an upstream completing decrements its downstreams; a node reaching zero, and frontier-finalized, joins the ready set). This is standard Kahn-style scheduling against the graph, which already encodes the synchronization structure.

## Cross-tier prefetch: the plan is the schedule

Because the plan states which weights are needed on which device in what order, **residency management is planned prefetch, not demand faulting**, and it spans the whole memory hierarchy (per [03-ir Storage classes](03-ir.md#storage-classes-and-sessions)):

- **Disk → host RAM** (larger-than-RAM): the plan issues `madvise(WILLNEED)` / page-touch for the mmap'd weights of upcoming runs ahead of the execution frontier, so cold pages fault in while earlier runs execute. Without this, an out-of-core model thrashes; with it, it streams.
- **Host RAM → device VRAM** (larger-than-VRAM): the plan issues H2D for upcoming runs ahead of the frontier and evicts (Move/Release) device-resident buffers a later run no longer needs, bounded by the device's free-memory budget from `BackendRuntime`.

One mechanism — plan-driven prefetch ahead of the frontier — serves both boundaries. It pipelines against planning and execution: page-in of a layer's weights, planning of downstream layers, and execution of upstream layers all overlap, so nothing waits for "the model to finish loading." Larger-than-RAM and larger-than-VRAM are usable precisely because the plan makes access local and prefetched.

## Background re-optimization

When the runtime loads a persisted graph (per [11-persistence](11-persistence.md)) — or a freshly-imported model with static-only costs — the loaded paths become the *active* plan immediately; TTFT is fast. In parallel, a background optimizer thread re-runs `optimize_graph`'s rankers against the local Judge's empirical data, **per decision point with merged path sets**:

1. The loaded multi-path graph is the working state.
2. The optimizer walks decision points (topological order).
3. At each: take the union of the loaded paths and any the local optimizer can produce with local empirical data; re-rank by local cost; converge structurally-equivalent paths; **keep the per-device Pareto frontier (crowding-capped)**; atomic-swap that decision point's path set in place.

This gives both the **merge** property (loaded paths that are still good are re-ranked, not discarded) and the **incremental** property (improvements become usable as soon as the next decision point's swap commits, not at the end). Early layers can benefit within seconds while later layers are still processing.

**Trigger policy** (run on new information, not a clock): first load of a graph whose costs were static-only; Judge-data accumulation crossing a meaningful threshold; backend telemetry shifting meaningfully (device added/removed, sustained per-tier pressure shift); format-version migration (opportunistic, as a side effect of producing the refined plan).

**Idle-time, resource-aware JIT fusion** rides this same background producer (per the 2026-06-20 decision G7, [10-decisions-log](10-decisions-log.md)). Fuel is the **strategist**: it is the only layer that sees the *whole machine* (host plus every device), so it chooses *which* sub-base-map regions to ask a backend to JIT-synthesize a kernel for, controls *when* (during idle time, gated on host- and all-device resource availability so synthesis never steals cycles from live work), sends only a **partial** base map (never the whole graph — strategy stays in the optimizer), and makes the cost-gated **adopt / reject** call when the kernel comes back. The backend (Baracuda first) is the **synthesizer** — it builds the best kernel for the Fuel-*chosen* region and returns it with its FKC contract (incl. `PrecisionGuarantee`); Fuel choosing the region *is* the fusion decision, so there is **no backend-side opportunity-finding** (the constitution holds; this is not backend-internal fusion, per [09-non-goals](09-non-goals.md)). Which regions to try first is ordered by the missing-fusion / co-occurrence telemetry (the exploration prior, see below); empirical *winning* (a kernel entering an optimized plan under cost-gated selection) is the exploit posterior, and win-rate flattening is the STOP signal that bounds JIT requests. An adopted kernel for a **new** fused-op identity lands as a Tier-2 declarative registration; a kernel for an **existing** identity lands in the already-extensible binding table (Tier 1) — see [What this rules out](#what-this-rules-out).

**Per-decision-point atomic swap.** Each decision point's path set is an `Arc`-shared slot; commit is an atomic `Arc` swap. The picker holds whichever Arc it loaded; writers swap a new Arc in; the old lives until readers release it. No hot-path locks. The same primitive serves concurrent optimize-and-execute (the frontier passing a decision point) and background re-optimization; they differ only in trigger and post-swap contents.

## Scoped re-optimization

Most triggers touch only a few decision points; the runtime computes the **affected scope** and re-optimizes only that. This is also exactly what makes **load-time validation** of a persisted graph cheap (per [03-ir Persisting the unified graph](03-ir.md#persisting-the-unified-graph-base-map--optimized-paths)): on load, validate the persisted paths and scope re-optimization to whatever changed.

| Trigger | Affected decision points |
| --- | --- |
| Device removed | Points with an alternative on that device; prune, re-run if a point empties. |
| Backend kernel-revision hash changed | Points whose paths reference the changed kernel; re-cost / re-generate just those. |
| Profile data refined for cells `(op, dtype, size_class, backend, device)` | Points whose costs depend on those cells; often just re-rank, no rule re-run. |
| Tolerance configuration changed | All points (the precision-filter re-evaluates admissibility). |
| New device added | All points (the new device may be a better target anywhere). Genuinely global. |
| Loading a `.fuel` on changed hardware | Points whose paths fail validation (stale kernels / absent devices); the rest are reused as-is. |

Most triggers are localized (a partial re-optimization over ~20% of points runs ~5× faster than full); only "new device added" and "tolerance config changed" are global. **Mechanism**: each decision point keeps a small dependency record (kernels referenced, devices used, profile cells its costs depend on); the runtime intersects a trigger with these records to compute the affected set, re-optimizes that set via the per-point swap, and leaves the rest untouched.

## Local Judge baseline: bundled in-package

The empirical Judge accumulates per-`(op, dtype, size_class, backend, device)` measurements during execution. So a fresh install is not cold, fuel **ships a bundled baseline Judge dataset in-package** (2026-06-13 decision, [10-decisions-log](10-decisions-log.md); supersedes the earlier opt-in *download* of a community baseline). On first run the local Judge initializes from the bundled baseline for the nearest hardware class and refines it with local measurements as they accumulate; a near-miss hardware class is still a better prior than static FLOP-counting, and the data degrades gracefully (it seeds, it does not lock).

This requires **no network** — important for offline / limited-connectivity deployments. Telemetry *upload* remains strictly opt-in (unchanged); the bundle is only what ships *down* in the box. (The online/idle-time measurement path — the expected-vs-real dispatch check feeding background re-optimization — further refines costs on the user's exact hardware; see ROADMAP.)

## Concurrent optimize-and-execute interaction

When the optimizer runs concurrently with execution (per [04-optimization §The sliding window](04-optimization.md#the-sliding-window-optimization-and-execution-overlap)), the optimization frontier moves through the graph. The runtime's ready-set tracking is unchanged but adds a *finalization* check: a run dispatches only when it is also frontier-finalized. As the frontier passes a decision point the optimizer commits one path there (via the per-point swap); the runtime sees the committed path as the only choice from that point forward.

## What the runtime persists

The runtime is largely stateless across realizes. Two cross-realize items:

- **The cached resolved route** (from the picker's telemetry-caching). Reused while telemetry is stable.
- **The persisted graph / optimization cache** (per [11-persistence](11-persistence.md)). **Memory-mapped at startup**, not read into memory: only the header and the decision-point index are touched before the first realize; node-data pages load on first access via the OS page cache; pages for never-taken paths may never load. Cache files are mmap-friendly (relative offsets, no process-absolute pointers).

### Kernel resolution: optional pre-resolve, else lazy

Per [03-ir §The optimized form](03-ir.md#the-optimized-form-the-multi-path-graph-the-plan-is-the-graph), kernel binding is optional. A throughput-first deployment pre-resolves all `KernelRef`s up front (lookup off the hot path); a TTFT-first one resolves **lazily**: when the picker takes a path, the runtime resolves its nodes' `KernelRef`s just-in-time via `binding_table.lookup(op_kind, dtypes, backend, kernel_source)` (~100 ns each, amortized over execution). Combined with mmap, lazy resolution makes startup near-instant for cache hits — paths the picker never takes never get resolved, their pages possibly never faulting in.

### Cache updates: write-new-file-and-swap

When background re-optimization commits a refined plan, the cache file is updated by **write-new-file-and-swap**: write to a sibling temp file, fsync, atomically rename, re-mmap. The old mmap's pages drop as the OS reclaims memory. This avoids in-place writable-mmap modification (crash-unsafe, platform-dependent, requires a writable file).

### Mmap fallback

Some embedded / WASM environments lack mmap. The runtime detects support at startup and falls back to read-into-memory mode where absent — one capability check plus a slow-path read; no architectural cost.

## What this rules out

- **No runtime kernel selection across paths the optimizer didn't keep.** If a variant isn't in the surviving per-device frontier, the runtime can't reach it. The optimizer surfaces competitive paths; the runtime picks among them.
- **No silent runtime fallback to a different op.** A chosen kernel failing (OOM, hardware fault) surfaces the error; the runtime doesn't transparently switch paths without telling the user. (A future "fallback path" feature could register runtime-fallbacks, but not v1.) **What this does *not* rule out** (reconciled 2026-06-20, [10-decisions-log](10-decisions-log.md) G4): **Fuel-orchestrated runtime kernel registration** is a v1 reality, not a post-v1 feature. The kernel **binding table is already runtime-extensible** (`extend_global_bindings` — append-only, multi-sibling, with route-cache invalidation; Tier 1), so JIT-ing a kernel for an **existing op identity** and ingesting it lands today; registering a **new fused-op identity** (Tier 2) is the in-flight goal, append-only with stable ids. The genuine non-goal is narrower than "runtime registration" — it is **silent op *substitution* mid-realize**: swapping the *math* a chosen path computes without the optimizer's cost-gated, telemetry-visible decision. JIT-registered kernels are cost-gated by the route picker and surface through telemetry; a kernel that doesn't win is never used.
- **No graph rebuild per input or per token.** The graph is the input-independent model (per [03-ir](03-ir.md)); autoregressive decode **reuses** the persistent decode-step graph with advancing per-session state, it does not rebuild or extend the graph each token. Structural changes happen at load/import + optimize, not mid-loop.

  The decode causal mask is one place this shows. There is **no dedicated mask op**: the mask is lower-triangular with its diagonal at the cached prefix length (a runtime offset). On CUDA, production decode passes a flash-attention `is_causal` flag — no mask tensor exists. On CPU/Vulkan (the decomposed reference path) the mask is an additive (0 / −∞) host-built constant **re-bound each pass** like the KV buffers, so the graph structure stays fixed across tokens. The intended end state is a fixed-capacity mask buffer re-bound by the per-token offset (keeping the persistent decode graph byte-stable); current decode rebuilds an exact-length mask each pass, which is correct but not yet the input-independent form.
- **Within-realize observations don't change the current realize.** Online measurements (the expected-vs-real dispatch check) feed **background re-optimization** for *subsequent* realizes; the in-flight realize runs on the route it started with.

## Where this lives in code

- **fuel-dispatch** — `PipelinedExecutor` (the single executor on every realize entry: the work-item producer + the executor loop), the ranker/selector chain (the runtime selector / Picker 2), `PlanStore`, dispatch. (The former `fuel-graph-router` and `fuel-graph-executor` are retired — see the executor-unification program.)
- **Per-backend crates** — actual `KernelRef` invocation; same-backend slot semantics.
- **fuel-core** — `pipelined_bridge` (realize entry / prep) and the Judge.

Implementation detail (data structures, threading model) is not architectural and lives in the relevant crates' design docs.

---

## See also

- [03-ir](03-ir.md) — the multi-path graph, decision points, storage classes, the persisted unified graph.
- [04-optimization](04-optimization.md) — produces the multi-path graph (pathfinders/rankers/optimizers, the bounded frontier).
- [05-backend-contract](05-backend-contract.md) — slot capacity and dynamic telemetry the runtime consumes.
- [07-tolerance](07-tolerance.md) — per-call tolerance overrides honored by the picker.
- [08-pattern-harvest](08-pattern-harvest.md) — the missing-fusion / co-occurrence telemetry that orders which regions the idle-time JIT loop tries first.
- [11-persistence](11-persistence.md) — persisting/loading the unified graph; scoped re-optimization; the kernel-cache pruning policy and captured-run replay artifacts.
- [14-lifecycle](14-lifecycle.md) — realize, the work-item producer, the run/executor split end to end.
- [10-decisions-log §2026-06-14](10-decisions-log.md) — the "plan IS the graph" redirection this version's v1.2 implements.
- [10-decisions-log §2026-06-20](10-decisions-log.md) — the adaptive-runtime-fusion decision this version's v1.3 reconciles to (Tier-1/Tier-2 extensibility, the strategist/synthesizer JIT loop, the megakernel curve).
