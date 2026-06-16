# Master plan — the "plan IS the graph" rebuild

> **Status: program spec, opened 2026-06-15.** This is the umbrella for rebuilding fuel's
> dispatch/optimization spine to the 2026-06-14 *"the plan is the graph"* redirection
> ([`../architecture/10-decisions-log.md`](../architecture/10-decisions-log.md) §2026-06-14).
> The architecture constitution already describes the destination (03-ir v0.4, 04-optimization
> v0.5, 06-runtime v1.2, 14-lifecycle v0.2); the as-built code does **not** yet realize it.
> This prompt is the path from one to the other. It **supersedes the Stage 4+ shape of**
> [`load-time-incremental-planner.md`](load-time-incremental-planner.md) (the planner-as-producer
> / Arc-keyed PlanStore / coverage-wait staging post) while **carrying forward that prompt's
> landed numeric substrate** (Stages 1–3) and memory-planning addendum unchanged.

---

## What this program is

The redirection names the built optimizer/executor — *"every kernel-bearing node gets an
`AlternativeSet`, resolved per node, dispatched per WorkItem"* — as **the drift to undo**
("almost eager with a queue"). The destination is one structure, not two: the optimized form
is the **same graph transformed in place** into a bounded multi-path shape, with decisions only
at the **few branch points**, retained per device by a **Pareto frontier + crowding cap**
(never a fixed top-N), dispatched as **runs** (op-sequences between decision points), built
**at load** (input-independent), serving many sessions via **storage classes keyed by
`SessionId`**, and persisted as one unified `.fuel`.

This is **not a refactor** — it is a near-total rebuild of the dispatch spine (five coupled
large surfaces; see [Code surface](#code-surface-what-the-rebuild-touches)). The leaf primitives
(`Candidate`, enumerate/filter/cost/placement-DP, selector *semantics*) survive as the components
`optimize_graph` composes; their callers and the data structures they populate all change.

**Hard discipline (the staged-migration mandate):** the system stays runnable at every phase
boundary. The production path has **no fallback executor** once `fuel-graph-executor` is gone
(see [Dependency](#dependency-do-this-first)), so each phase ships green against the existing
suites (`fuel-core --lib`, `fuel-dispatch --lib`, `fuel-graph --lib`, and the `#[ignore]`'d live
GPU suites run one at a time). **TDD is the default**: write the born-red test first, watch it
fail, make it green. The session may stop cleanly after any phase.

---

## Read first (in this order)

1. [`../architecture/10-decisions-log.md`](../architecture/10-decisions-log.md) — the
   **2026-06-14 entry** is the anchor; it wins over any not-yet-revised text elsewhere.
2. [`../architecture/03-ir.md`](../architecture/03-ir.md) v0.4 — §"The DAG" (input-independent,
   append-only-with-compaction), §"Storage classes and sessions", §"The optimized form: the
   multi-path graph", §"Bounding the frontier", §"Persisting the unified graph".
3. [`../architecture/04-optimization.md`](../architecture/04-optimization.md) v0.5 — §"The
   two-stage transformation", §"Rankers and the cost model: a per-path cost vector, ranked",
   §"Bounding the frontier: Pareto per device + crowding cap", §"Relationship to PR 3".
4. [`../architecture/06-runtime.md`](../architecture/06-runtime.md) v1.2 — §"Dispatch: runs, not
   single ops", §"Route picker (the runtime selector / Picker 2)", §"Cross-tier prefetch: the
   plan is the schedule".
5. [`../architecture/14-lifecycle.md`](../architecture/14-lifecycle.md) v0.2 — the per-stage
   **Today vs Intended** notes are the precise diff this program closes.
6. [`load-time-incremental-planner.md`](load-time-incremental-planner.md) — the carry-forward
   substrate (Stages 1–3 LANDED) and the memory-planning addendum (preserved verbatim into the
   Intended design). Its Stage 4+ is the superseded shape this program replaces.
7. **Frontier-pruning prototype** (`C:/Projects/frontier-prototype`, outside the workspace) —
   the standalone simulation that validated the retention policy stays bounded (~10² paths,
   lossless, `keep ≈ 32`/device matches the no-cap optimum). Evidence the Phase B math is sound
   before committing to the IR.

---

## Dependency: do this first

**Executor-unification Session 7** ([`executor-unification-reaudit-2026-06-11.md`](executor-unification-reaudit-2026-06-11.md))
retires the 6 surviving `GraphBackend` impls, the ~2147-LOC `fuel-graph-executor` crate, and the
`fuel-graph-cpu::realize_any` third evaluator. That crate is already off the production realize
path (every `fuel-core` mention is a doc comment, not a call), so it is independent at the
call-graph level — **but land Session 7's deletion first** so this rebuild targets the
post-Session-7 `PipelinedExecutor` and not a doomed crate. Session 7 gap-5 also explicitly hands
off full lowering / fusion-registry-on-the-default-realize-path *to this program*; gaps 7
(const-pool byte budget) and 13 (`Op::Move` executor arm, needed for residency eviction) are
dependencies Phase B/E will consume.

**Delete, do not migrate:** the Stage 4b revision surface in `plan_store.rs`
(`submit_revision` / `PlanRevisionWatch::poll`, commit `303ae8ca`) is **verified dead code** — it
references `RevisionState` / `realize_with_plan_revisions` that exist nowhere in the tree (only in
its own doc-comments). Drop it.

---

## Carry forward (reuse by reference — do not re-spec)

From [`load-time-incremental-planner.md`](load-time-incremental-planner.md), **landed and
model-agnostic**, reused wholesale as the components `optimize_graph` composes:

- **Stage 1** — `TransferEstimate` per topology path (calibration probe).
- **Stage 2** — inbound-transfer cost term + always-enumerate-priced admission.
- **Stage 3** — the (position × device) carry-forward placement DP, O(devices) state, fused
  multi-node edges, subgraph-shaped fusion matching.
- **Memory-planning addendum (2026-06-13)** — plan-time capacity-aware DP + activation-pool
  sizing + realize-time pressure eviction. Preserved verbatim into the Intended design; it
  becomes the memory tier of the Phase B cost vector and the Phase E prefetch/eviction schedule.

What does **not** carry: Stage 4's whole shape (planner-as-producer driven by graph-construction
events, PlanStore keyed by graph `Arc` identity + topology generation, `realize()` coverage-wait
against a moving execution frontier). The new model builds the multi-path graph **once at load**,
so there is no construction-event driver, no identity-keyed plan store, and no coverage-wait.

---

## The phases

The build order is bottom-up so each capability has a working substrate (capability numbers
[1]–[13] match the constitution audit). **Phases C and D may overlap once A/B land** (runtime vs
load-time-build touch largely different crates — `fuel-dispatch` vs `fuel-core`/`fuel-graph`),
but D-[10] waits on D-[8]/[9], and E is last.

### Phase A — In-graph multi-path substrate *(foundation; nothing else is honest without it)*

> **STATUS: LANDED 2026-06-15/16 (PR-A0…A4).** `Op::Branch` is an arena fact; `optimize_graph`
> is the sole realize-path optimizer; `ExecutionPlan` demoted to a transient view, `PlanStore`
> deleted. See [14-lifecycle](../architecture/14-lifecycle.md) Stage 2/3.
>
> **Representation — DECIDED 2026-06-15 (design panel + architect ruling).** The multi-path
> structure is an **arena fact, not an overlay**: a new `Op::Branch` (phi/merge) node whose
> inputs *are* the divergent routes, carrying an explicit `reconverge_at`. Chosen for
> persistence, compaction, graph-walking, path-filtering, and readability — the overlay's only
> real edge was avoiding the closed-enum edit, with **no inference-time benefit** (once a route
> is picked and lowered to runs, the representation is invisible to the hot path).
> **Accepted, manageable costs:** (a) a new arm in the exhaustively-matched `Op` enum — one-time,
> mechanical (short_name/describe, shape-inference, autograd, optimizer, executor match sites);
> (b) *"immutable nodes"* is read as *immutable once finalized* — pathfinders add `Op::Branch`
> arms during the `optimize_graph` phase (the arena is already append-with-compaction there), and
> a Branch is emitted with its arms known; (c) *"one concrete dtype"* is preserved by
> **cast-to-uniform at `reconverge_at`** — dtype-lowered alternatives live *inside* an arm whose
> exit casts back, so the merge has one dtype. A graph with **zero `Op::Branch` nodes is exactly
> today's single-route graph**, which keeps the suite green across every phase boundary.
>
> **Phase A build order (each born-red-test-first; PR-A3 retargets the post-Session-7 executor):**
>
> - **PR-A0** (fuel-graph): add the `Op::Branch { reconverge_at, … }` arm, handled as an inert
>   passthrough at every exhaustive `Op` match site. Test: full suite green; a zero-Branch graph
>   is unchanged.
> - **PR-A1** (fuel-graph): `open_branch`/`add_arm`/`finalize_branches` builders with build-time
>   validation returning `Result` (reconverge must be a descendant of diverge; arms internally
>   disjoint; cast-to-uniform at reconverge; single-arm branches dropped). Test: a 2-arm diamond
>   round-trips; a non-descendant reconverge and a non-disjoint arm each return a typed `Error` —
>   never panic.
> - **PR-A2** (fuel-graph): run extraction + the transient `lower_run` view (runs delimited by
>   Branch boundaries + residency seams; each run a single-device contiguous chain). Test:
>   straight-line → one run; a 2-arm branch → {pre, arm0, arm1, post}; a residency change starts a
>   new run; **plus the fewness gate** — a per-layer-branch graph passes branches/nodes < ~5%, a
>   synthetic per-op-branch graph fails it (locks the granularity crux before Phase B).
> - **PR-A3** (fuel-dispatch): `compile_plan` → `optimize_graph(&mut Graph)` writing Branch nodes
>   in place; `ExecutionPlan` demoted to the transient `lower_run` view; **delete `PlanStore` +
>   the dead `303ae8ca` revision surface**. Equivalence gate: a no-competing-routes graph
>   optimizes to zero Branch nodes and `lower_run` reproduces today's exact dispatch order against
>   the executor.
> - **PR-A4** (fuel-dispatch): the first real pathfinder emits a Branch via the `PlacementDp`
>   (CPU vs CUDA matmul) — **deliberate-fork seed** (not per-device-multiplicity; Phase B's Pareto
>   bound introduces device-multiplicity arms later). Test: exactly one 2-arm Branch; an ordinary
>   DAG fan-out in the same graph is *not* flagged; the graph realizes on arm 0; no `DEFAULT_MAX_N`
>   anywhere.

- **[1] In-place multi-path graph — retire the separate `ExecutionPlan`.** The optimized form
  lives **in the graph** as alternative routes that diverge and reconverge: a path/branch
  representation on the arena, a convergence merge that fuses only forward-identical paths, and
  retirement of the standalone `ExecutionPlan` as the source of truth (it may remain a transient
  lowering *view*). *03-ir §"The optimized form"; decisions #1/#2.*
  *Today:* `ExecutionPlan { order, alternatives: HashMap<NodeId, AlternativeSet>, generation }`
  (`fuel-dispatch/src/plan.rs:56`), a separate object memoized in `PlanStore` by graph identity.
- **[2] Branch-point decision model — not per-node alternatives.** Alternatives attach only to
  decision points (branch points where ≥2 viable routes exist and reconverge — these are FEW);
  everything between two decision points is a single fixed run. Identify branch points during
  search, emit a structured per-branch alternative set on the graph, collapse straight-line spans
  into shared single-route regions. *04-optimization §"Per-decision-point alternatives" +
  §"Coupling between decisions".*
  *Today:* `compile_plan` builds an `AlternativeSet` for **every** kernel-bearing node, and the
  selector may fire at any node — exactly the drift to undo.

### Phase B — A correct, bounded frontier *(turns the substrate into a real optimizer)*

> **STATUS: LANDED 2026-06-16 (PR-B1…B4).** Cost **vector** + Pareto dominance, per-ending-device
> Pareto frontier + crowding cap (`KEEP_PER_DEVICE`, fixed top-N retired), lock-step
> pathfinder/ranker/optimizer driver, and arena compaction — all Today.

- **[4] Per-path cost VECTOR, one central time metric.** Rankers produce a vector: **one** central
  time metric (median/avg for throughput, p99 for latency-SLA — `t_min` is explicitly **dropped**
  as a selection axis), memory as a per-tier vector (disk/host/device), precision (digits),
  accuracy (ULP/rounding/monotonicity). Pareto dominance over the whole vector; ties break
  precision → accuracy → memory. *04-optimization §"Rankers and the cost model".*
  *Today:* a single scalar `composite_ns` sort; no memory tier, precision, or accuracy axis; no
  dominance.
- **[3] Per-ending-device Pareto frontier + crowding cap — retire fixed top-N.** Per ending
  device, keep the Pareto-optimal paths over the cost vector; backstop with an NSGA-II
  crowding-distance cap of `keep`/device (prototype `keep ≈ 32`). Invariants: ≥1 path per device
  survives, total ≤ `keep × devices`, **never strand the last path for a `(device, backend)`**.
  *04-optimization §"Bounding the frontier"; decision #8.*
  *Today:* `DEFAULT_MAX_N = 3` (`ranker/alternative_set.rs:53`) truncation on a scalar, per node,
  no per-device bucketing — the scalar-top-N failure that strands slow devices.
- **[5] `optimize_graph` = lock-step pathfinders + rankers + optimizers.** A driver running three
  pass kinds interleaved (prune-as-you-go, **never** explode-then-extract): *pathfinders* ADD
  candidate paths (fusion, algebraic, dtype-lowering under tolerance, placement/transfer, layout
  fixups); *rankers* MEASURE each path (the [4] vector); *optimizers* MERGE/DISCARD (precision
  filter, duplicate-path convergence, path-timing) without stranding the last `(device, backend)`
  path or touching a path in an active cycle. Builds on PR3's `Rule`/`RuleRegistry`/fixpoint
  driver, extended to multi-alternative tracking + a declarative-pattern engine. *04-optimization
  §"The two-stage transformation", §"Relationship to PR 3"; decision #7.* *Today:* no
  `optimize_graph` symbol exists; `compile_plan` is per-node, single-pass, first-match-wins.
- **[11] Required compaction of the append-only arena.** A pass that drops unreachable nodes (the
  orphan debris pathfinders leave), keeping only the base map + surviving multi-path graph.
  **Required** before finalize-to-disk; optionally between optimization rounds to bound the
  working set. Only becomes necessary once [1]/[5] rewrite the arena in place. *03-ir §"The DAG"
  property 3.*

### Phase C — Runtime catches up to the new shape

> **STATUS: LANDED 2026-06-16 (PR-C1, C2a, C2b).** [7] route picker (Picker 2) is wired into
> realize — `pick_route` chooses arms at branch points by live per-tier free memory, consulted
> per branch not per node (PR-C1). For [6]: the **picking** half is done (the picker fires only at
> branch points; realize lowers the picked route), and a **run-capture capability** is built +
> GPU-proven on both GPU backends — CUDA graphs (PR-C2a, `fuel-cuda-backend/src/capture.rs`) and
> reusable Vulkan command buffers (PR-C2b, `fuel-vulkan-backend/src/capture.rs`) — for
> capture/replay/rebind of a run's launches. **Deferred to Phase D:** *wiring* run-capture into
> the executor (it still dispatches per `WorkItem`); a captured run amortizes only over repeated
> replay of the same run, which needs Phase D's persistent cross-realize graph.

- **[6] Run-as-dispatch-unit.** Define a **run** = the fixed op-sequence between two decision
  points and dispatch it as a UNIT (ideally a pre-recorded CUDA Graph / Vulkan command buffer
  replayed with rebased operands). The per-node lowering becomes the **work-item producer**
  running ahead; collapse straight-line spans into runs so the picker is consulted a handful of
  times, not per node. *06-runtime §"Dispatch: runs, not single ops"; 14-lifecycle Stage 4.*
  *Today:* the executor dispatches per `WorkItem` (per node) (`pipelined.rs:521`); no `Run` type.
- **[7] Runtime selector (Picker 2) at branch points.** The route picker walks the multi-path
  graph and, at the few decision points only, picks among surviving per-device Pareto paths by
  live telemetry — crucially **per-tier free memory** (so under VRAM pressure it prefers a
  host-RAM path). Resolve coupled decisions in topo order; cache the route and re-resolve only on
  a meaningful telemetry delta; bounded lookahead K=3 for adversarial coupling. *06-runtime
  §"Route picker"; decision #9.* *Today:* `ChainedSelector` fires per kernel-bearing node over a
  top-N set; no per-tier-free-memory axis, no run boundary to decide at.

### Phase D — Move the build earlier + add sessions *(the largest single Today→Intended shift)*

- **[8] Load-time, input-independent graph build.** Build (and ideally optimize) the graph when
  the model is **loaded**, before the first input — the graph is a property of the model.
  `realize()` reduces to wait-until-plan-covers-roots + dispatch; inputs/session state flow
  through the persistent graph; nodes need not own storage to exist. *03-ir §"The DAG" property 1;
  14-lifecycle Stage 2/3 + §"Where the boundary sits"; decision #3.* *Today:* no graph until
  `forward` runs; planning is per-forward at realize time.
- **[9] Storage classes keyed by `SessionId`.** Each node carries a class inferred from its op
  with explicit override: **shared** (`Op::Const` weights, one copy across sessions, side-table by
  `NodeId`), **session-state** (KV-cache + explicit cache-write targets, by `(NodeId, SessionId)`),
  **transient** (activations/scratch, never persisted, may cross devices D2D mid-realize). One
  optimized graph serves many concurrent sessions via `SessionId`. *03-ir §"Storage classes and
  sessions"; decision #6.* *Today:* no `StorageClass` / `SessionId` types; KV-caches are `Arc`s
  held outside any graph, re-bound as fresh `Op::Const` per token.
- **[10] Persistent decode graph reused across tokens.** Autoregressive decode reuses the single
  input-independent decode-step graph; only session-class storage (KV-cache, by `SessionId`)
  advances via the `cached_len` host scalar. Plan reuse falls out of the graph being the same
  object — no structural-hash plan cache bolted onto fresh-every-step graphs. **Falls out of
  [8]+[9] by construction** and closes the production-decode re-plan gap (the headline 1.8×/token
  win does not reach today's fresh-graph-per-token decode). *06-runtime §"What this rules out";
  14-lifecycle Stage 5.*

### Phase E — Persistence + out-of-core *(capstone; depends on the unified graph being stable)*

- **[13] Mmap-backed zero-copy `Storage` + plan-as-prefetch-schedule.** `Storage` must support
  mmap-backed zero-copy paged views (not only owned buffers) — the prerequisite for
  larger-than-RAM and the native format. The plan **is** the cross-tier prefetch schedule:
  disk→RAM (`madvise WILLNEED` / page-touch ahead of the execution frontier) for larger-than-RAM;
  RAM→VRAM (H2D ahead of frontier + `Op::Move`/release eviction) for larger-than-VRAM. One
  mechanism serves both boundaries; fall back to read-into-memory where mmap is unsupported.
  *06-runtime §"Cross-tier prefetch"; decisions #10/#11.* *Today:* eager-copy load defeats mmap;
  residency is demand-driven, not planned.
- **[12] LOAD vs IMPORT split + unified `.fuel`.** Separate **LOAD** (`map_from_file` on a
  finalized native `.fuel` — reconstructs base map + storage + this machine's optimized paths
  directly, mmap'd, no conversion) from **IMPORT** (`from_gguf`/`from_safetensors`/HF — convert
  once into the base map). The `.fuel` holds base map + optimized paths; finalize-to-disk is the
  default run mode. On load, validate persisted paths by revision-hash and scope re-optimization
  to what changed; foreign-hardware paths drop and rebuild from the portable base map. Ship a
  **stripper** that removes all but the base map. *03-ir §"Persisting the unified graph";
  14-lifecycle Stage 1/2; decisions #4/#5.* *Today:* no `map_from_file`, no native format, no
  load/import distinction; loading eagerly copies every tensor.

---

## Code surface (what the rebuild touches)

| Component | Action | Size |
|---|---|---|
| `fuel-dispatch/src/plan.rs` — `struct ExecutionPlan` (56–96) | rewrite (becomes a transient lowering view, not the source of truth) | large |
| `fuel-dispatch/src/plan.rs` — `compile_plan` + helpers (488–1352) | rewrite → `optimize_graph` over the in-graph form | large |
| `fuel-dispatch/src/plan.rs` — `PlanOptions` + builders (201–406) | rewrite | medium |
| `fuel-dispatch/src/ranker/alternative_set.rs` — `AlternativeSet` + `DEFAULT_MAX_N=3` | rewrite → per-branch set + Pareto/crowding retention | medium |
| `fuel-dispatch/src/ranker/candidate.rs` — `Candidate` + `CouplingAdjustment` | keep | small |
| `fuel-dispatch/src/ranker/{enumerate,filter,filters,cost,placement_dp}.rs` | extend (cost → vector; reuse the DP) | large |
| `fuel-dispatch/src/plan_store.rs` — `PlanStore`/`StoredPlan`/warm latches | replace (no identity-keyed store under load-time build) | large |
| `fuel-dispatch/src/plan_store.rs` — `submit_revision`/`PlanRevisionWatch` (303ae8ca) | **delete** (dead code) | medium |
| `fuel-dispatch/src/pipelined.rs` — `PipelinedExecutor` + `realize_inner`/`compiler_thread_body` (381–757) | rewrite (work-item producer + run dispatch) | large |
| `fuel-dispatch/src/pipelined.rs` — `compile_one` + `WorkItem`/`WorkItemKind` (62–300, 841+) | rewrite | large |
| `fuel-dispatch/src/pipelined.rs` — `resolve_compiled` (791–825) + selector plumbing | rewrite (decide at branches) | medium |
| `fuel-dispatch/src/ranker/{chained,runtime,judge_aware,vram_pressure}_selector.rs` | extend (per-tier free memory; per-branch) | medium |
| `fuel-core/src/pipelined_bridge.rs` — `build_execution_plan`/`dispatch_*_with_plan_retry`/`stamp_plan_backends`/`prepare` (~2371 LOC) | **rewrite — riskiest** | large |
| `fuel-graph-executor/src/lib.rs` — `GraphExecutor<B>` + `GraphBackend` | **delete** (via Session 7) | medium |

**Riskiest surface — `fuel-core/src/pipelined_bridge.rs`.** It embeds all four inversions the
redirection targets *simultaneously* (plan built inside `forward()` not at load; plan as a
separate retry-able artifact not the graph; LOAD vs IMPORT undifferentiated; per-node stamping
instead of branch-point/run structure) **and** carries the load-bearing correctness machinery
(`TopologyChanged` retry loop, residency stitching, layout fixups, const-cache upload,
realize-split). Every model run goes through it, with **no fallback executor** left to catch a
regression. Its `stamp_plan_backends` / `insert_resident_input_copies` / `apply_layout_fixups`
passes — which write decisions back onto the graph's `target_backend` side-table — are precisely
the in-place transform that should be centralized into `optimize_graph` ([1]/[5]).

---

## What this program must NOT do

- **Don't keep the per-node `AlternativeSet` "just for now."** It is the central drift; Phase A
  exists to remove it. Building Phase B's frontier on top of per-node alternatives would be the
  wrong substrate, rewritten later.
- **Don't reintroduce a fixed top-N anywhere** (no `DEFAULT_MAX_N`). The bound is per-device
  Pareto + crowding cap. Do not rank on a scalar — rank on the cost vector with dominance.
- **Don't optimize on `t_min`.** One central time metric (mode-selected), per decision #7/#10.
- **Don't explode-then-extract.** Pathfinders/rankers/optimizers run lock-step; the working set
  stays bounded by construction (Phase B + [11] compaction).
- **Don't build against `fuel-graph-executor`.** It is being deleted (Session 7); target the
  single `PipelinedExecutor`.
- **Don't leave the production path red across a phase boundary.** Each phase ships green on the
  `--lib` suites + live GPU suites (one at a time). Born-red tests first.
- **Don't push to remote** unless the user asks. WIP/unverified work goes on a branch.

---

## Open questions / caveats

- **Fused-op kernel-side registry placement** (`BackendImpl` payloads / `FusedKernelRegistry`):
  `fuel-memory` vs `fuel-dispatch` is an explicit **open** question
  ([`../architecture/02-layers.md`](../architecture/02-layers.md)); moving it to `fuel-dispatch`
  is acceptable if it fits better. Phase B's `optimize_graph` selects over this surface — settle
  the placement when you touch it.
- **The autoregressive barrier** ([`data-dependent-shapes-design.md`](data-dependent-shapes-design.md)):
  "one DAG built once at load" cannot cover everything — each decode token depends on the previous
  token's *sampled* output (a genuine host decision). The input-independent decode-step graph
  ([10]) is reused across tokens, but the host-side sample/argmax is the legitimate boundary; data-
  dependent output shapes use the capacity-shape + valid-count pattern. Phase D must respect this.
- **Branch-point granularity (was the Phase A crux) — RESOLVED 2026-06-15.** The risk: too many
  branch points and the model drifts back toward per-node; too few and real alternatives get
  stranded inside a run. Resolved by (a) the **deliberate-fork seed** — Phase A emits a branch only
  where a pathfinder deliberately forks, not wherever the DP sees ≥2 viable devices — and (b) the
  **fewness-gate born-red test** (PR-A2: branches/nodes < ~5% on a real decode graph) that locks
  the invariant before Phase B builds on it. The frontier prototype validated the *retention* math;
  the fewness gate guards the *detection* side.

---

## Relationship to other prompts

- **Supersedes** the Stage 4+ shape of [`load-time-incremental-planner.md`](load-time-incremental-planner.md)
  (a superseded-by pointer will be added there); **reuses** its Stages 1–3 + memory addendum.
- **Depends on** [`executor-unification-reaudit-2026-06-11.md`](executor-unification-reaudit-2026-06-11.md)
  Session 7 (do first).
- **Consumed by** [`baracuda-cutlass-alpha-13-integration.md`](baracuda-cutlass-alpha-13-integration.md)
  (CUTLASS alternatives register at the branch points this builds) and
  [`model-interchange-import-export-plan.md`](model-interchange-import-export-plan.md) (the `.fuel`
  load/persistence half of "graph built at load").
- ROADMAP's transaction-model framework (`ROADMAP.md` ~1650–1729) is the closest pre-redirection
  design seed (optimize the single graph in place via working-copy transactions + Arc-swap commit)
  but lacks branch points / Pareto frontier / runs / `optimize_graph`; reconcile or retire it when
  ROADMAP is updated for the redirection.

## See also

- [`../architecture/03-ir.md`](../architecture/03-ir.md), [`../architecture/04-optimization.md`](../architecture/04-optimization.md), [`../architecture/06-runtime.md`](../architecture/06-runtime.md), [`../architecture/14-lifecycle.md`](../architecture/14-lifecycle.md), [`../architecture/10-decisions-log.md`](../architecture/10-decisions-log.md) (2026-06-14).
