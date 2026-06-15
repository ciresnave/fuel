# Session prompt — load-time incremental planner + cost-based placement

> **Reconciled 2026-06-15 against the 2026-06-14 redirection + current git: live spec — Stages 1–4a landed, 4b WIP, Stages 5–7 + memory-planning unbuilt; the "incremental driver inside forward" is now reframed as a staging post toward load-time build (see note below).**
>
> **Today vs Intended (per [14-lifecycle](../architecture/14-lifecycle.md)).** This program is **not retired** — 3+ stages remain unbuilt and the constitution preserves its substance. But the 2026-06-14 *"the plan is the graph"* redirection ([10-decisions-log](../architecture/10-decisions-log.md) 2026-06-14 entry) reframes what this program is building toward:
>
> - The optimized form is **the same graph transformed in place**, not a separate `ExecutionPlan` artifact; alternatives attach to **branch points** as a bounded Pareto frontier (crowding-capped), not per-node top-N; runtime dispatches **runs** between decision points and the route picker chooses only at branch points.
> - That graph is **built at LOAD and input-independent** — a property of the model, not of a request — so planning is load-time, not a driver that runs from graph-construction events inside `forward()`.
> - Accordingly, the "Target behavior" formulation below and **Stage 4**'s incremental-driver/plan-store/coverage-wait shape are the **as-built (Today) path** — a *staging post*, conservatively correct, that the redirection **supersedes**: once the graph is built once at load and reused across steps, the planner does not need to run from construction events ahead of an execution frontier. Stages 1–3 (DP + transfer pricing), Stages 6–7, and the memory-planning addendum are preserved by the constitution and are read directly into the Intended design.
>
> Status against git: Stage 1 `70018b9c`, Stage 2 `30e35c26`, Stage 3 `0a786821`, Stage 4a `240af759`/`99e26dea` landed; Stage 4b WIP at `303ae8ca` (UNVERIFIED); Stages 5–7 + memory planning unbuilt.

Decision (2026-06-11, Fuel author): **planning moves out of
`realize()`, and soon** — Fuel should build in this direction before
more executor work accretes around the realize-time-planning shape.
Architecture anchors: [04-optimization §Load-time incremental
planning + cost-based placement
(v0.4)](../architecture/04-optimization.md) and
[06-runtime v1.1](../architecture/06-runtime.md) — both updated
2026-06-11 to record this decision. The concurrent
optimize-and-execute model (optimizer as producer, executor as
consumer, frontier between them) was already the committed
architecture; this program builds its driver, its placement
algorithm, and its memoization layer.

## Target behavior (the author's formulation)

- Planning starts as soon as Fuel is told to load a model, seeded
  with: the first op, the speed-ranked implementations of that op
  (today's `AlternativeSet`), and where the incoming data is
  resident.
- The first op launches wherever its data lives — usually CPU —
  without waiting for any global plan. While it executes, the
  planner maps ahead and discovers (by cost) that migrating to a GPU
  pays; when the CPU ops finish, the frontier dispatches the rest of
  the sequence on the GPU.
- The plan grows incrementally as ops are added: best single impl
  for each new op, fusion checks against the trailing neighborhood,
  placement re-evaluated as accumulated cost state extends.
- `realize()` reduces to *wait until the plan frontier covers the
  requested roots, then dispatch against the committed plan*. The
  plan is fluid ahead of the execution frontier, sunk behind it.
- A running realize finds optimized paths ahead of it already
  prepared, because planning (µs/op) outruns both execution (ms/op)
  and weight page-in.

## Design pillars (agreed in review, 2026-06-11)

1. **Never block first dispatch.** Planning completeness is never a
   precondition for executing the frontier op with the best plan
   known *now*.
2. **mmap pipelining.** Weights fault in lazily; the first op needs
   only its own pages. The plan doubles as the prefetch schedule:
   it knows which weights are needed on which device in what order,
   so H2D uploads and page-ins stream ahead of the execution
   frontier instead of blocking at load.
3. **Placement = carry-forward DP, not window enumeration.** State
   per frontier node: `best[d]` = cheapest accumulated cost arriving
   here with output on device `d`. Extension per new node considers
   every (previous device → this device) pair, pricing the boundary
   transfer numerically. The accumulated state *summarizes* the
   unbounded prefix — extremely long segments are "checked" by
   extending O(devices) state, never by re-examining windows.
4. **Fusion = bounded jumps in the same DP.** Registered fused
   patterns enter the recurrence as multi-node edges (lookback
   bounded by the longest pattern, ~5 ops). A locally-fast fused op
   that strands residency for a later segment loses on accumulated
   cost automatically — the DP finalizes nothing until downstream
   extensions roll in (up to the commit horizon).
5. **Fusion matching is subgraph-shaped.** Probe patterns against
   the neighborhood within longest-pattern distance of each new
   node — NOT linear topo-order windows (they test non-adjacent
   pairs and miss diamonds like FlashAttention).
6. **Commit horizon = execution frontier.** Behind it: sunk. Ahead
   of it: per-decision-point atomic swap (the existing concurrent
   model + the Phase 4.3 generation-check machinery is the
   revalidation seam).
7. **Memoize plan fragments by structural hash** (op sequence +
   shapes + dtypes + params). Map the first transformer layer
   honestly, stamp the other 31. Same hash keys the persisted-plan
   cache fragments (11-persistence), record-once-replay capture
   units (CUDA Graphs / Vulkan command buffers, weight pointers
   rebased per instance), and fragment-sized activation buffer
   rings. Weights do not dedupe (same structure, different values).
   *(Redirection note, 2026-06-15: this structural-hash memo — together
   with its Stage 5 build below — is the **documented near-term decode
   fix** flagged in [ROADMAP.md:85](../../ROADMAP.md): production decode
   mints a fresh `Graph` per token so every step is a `PlanStore` miss,
   and a structural-hash key beside the Arc-identity path turns decode
   steps back into fragment hits. Under the 2026-06-14 redirection
   ([14-lifecycle](../architecture/14-lifecycle.md)) this fix becomes
   **redundant by construction**: the input-independent graph is built
   once at load and reused across steps, so plan reuse falls out of the
   graph being the same object rather than needing a structural-hash
   cache bolted onto fresh-every-step graphs. Keep the pillar — it is
   still the bridge fix until the load-time graph lands, and the
   persisted-fragment / record-replay / activation-ring keying it
   enables remains in the Intended design.)*
8. **Horizontal fusion / similarity scheduling (later phase).**
   Independent repeated subgraphs (per-head projections, MoE
   experts — NOT sequential layers) may be scheduled adjacently and
   merged into batched launches. Win = launch count + occupancy;
   cost = peak activation memory; arbiter = residency planner;
   gate = graph-derived independence (no path between).

## Build order (staged; each stage lands green independently)

Stages 1–3 are pure additions to today's picker substrate and are
prerequisites for everything later. Stage 4 is the architectural
move. Estimated 6–9 sessions total.

### Stage 1 — transfer calibration probe (~half session) — LANDED `70018b9c`

Numeric `TransferEstimate` per `SystemTopology` transfer path:
measure H2D/D2H/D2D at a few sizes per path (Judge-style, once per
topology generation), fit `bytes/bandwidth + latency`, store on the
topology snapshot. Pure Fuel-side (`cuMemcpy` timing; Vulkan staging
equivalents). Un-parks TDP-4 option B. Tests: probe runs on CPU-only
hosts (zero paths) and on the RTX 4070 box (live, `#[ignore]`).

### Stage 2 — residency plumbing + greedy inbound pricing (~1 session) — LANDED `30e35c26`

Thread committed producer placements through `compile_plan`'s walk
(topo order ⇒ producers decided before consumers). Extend the cost
composer with the inbound-transfer term for every input not on the
candidate's device. Relax the off-device admission policy from
"missing-impl only" to "always enumerate, priced." Greedy is
conservative-correct: it never makes an unjustified move, it only
misses globally-justified ones (the first op of a beneficial
migration must pay the crossing alone). Parity tests: single-device
systems produce identical plans; priced systems never regress a
locality plan unless the numbers genuinely favor a move.

### Stage 3 — carry-forward placement DP + fused jumps (~1–2 sessions) — LANDED `0a786821`

Replace greedy per-node commits with the `(position × device)` DP
(design pillar 3+4) inside the plan walk. Chain-first: linearize the
trunk, handle joins by heuristic state-merge (cheaper producer
device), document the branch-handling debt. Backtracked plan stamps
`target_backend` per node exactly as Phase 4a does today. Tests:
synthetic graphs where (a) a mid-sequence GPU segment beats
all-CPU/all-GPU, (b) a fused op must LOSE because it strands
residency, (c) DP plan == greedy plan when transfers dominate.

### Stage 4 — the incremental driver + commit horizon (~2 sessions) — Stage 4a LANDED `240af759`/`99e26dea`; Stage 4b WIP `303ae8ca` (UNVERIFIED)

> **Redirection note (2026-06-15).** This stage's framing — a planner
> *producer running from graph-construction events*, a plan store keyed
> by graph identity, and a `realize()` coverage-wait against an
> execution frontier — is the **as-built (Today) path**, and it is the
> shape the 2026-06-14 redirection **supersedes**. Under "the plan is
> the graph," the graph (with its optimized in-place multi-path form) is
> built **once at load and input-independent**, so there is no driver
> chasing construction events ahead of a moving frontier and no separate
> plan store to cover: the optimized graph simply *is* present when
> realize starts. Keep this stage as the conservatively-correct staging
> post that gets us there incrementally; do not read it as the steady
> state. (Stage 4a landed at `240af759`/`99e26dea`; Stage 4b — the
> plan-store revision/latch surface — is WIP at `303ae8ca`, UNVERIFIED.)

The planner becomes a producer running from graph-construction
events (model-loader graphs first; user-built graphs subscribe the
same way). `prepare()`'s planning half moves behind a plan store
keyed by graph + topology generation; `realize()` waits for coverage
of its roots and dispatches. Commit horizon tracks the executor's
dispatch frontier; ahead-of-horizon revisions use per-decision-point
swap; behind it, sunk. Re-plan triggers reuse the Phase 4.3
generation check. Tests: realize during in-progress planning gets
correct results (coverage wait), plan revisions ahead of the
frontier are observed by later chunks, revision behind the frontier
is rejected.

### Stage 5 — structural-hash memoization + stamping (~1 session)

> **Redirection note (2026-06-15).** This is the **documented near-term
> decode fix** ([ROADMAP.md:85](../../ROADMAP.md)) for the live miss:
> `forward_with_kv_context` mints a fresh `Graph` (new `Arc`) per token
> and `PlanStore` keys on `Arc::as_ptr`, so every decode token is a full
> replan today; decode-step graphs are structurally identical (only the
> host scalar `cached_len` changes — KV writes land in pre-allocated
> fixed-capacity buffers via `Op::WriteSlice`), so a structural-hash key
> beside the Arc-identity fast path turns decode steps into fragment
> hits. **The acceptance test MUST use fresh `Graph` Arcs per step (the
> kv-context shape), not one growing graph, and must settle the
> sequence-length policy** (the hash includes exact shapes ⇒ zero reuse
> across prompt lengths without bucketing/capacity). Under the 2026-06-14
> redirection ([14-lifecycle](../architecture/14-lifecycle.md)) this stage
> becomes **redundant by construction**: the graph is built once at load
> and reused across steps, so the plan persists because it is the *same
> graph object*, not because a structural-hash cache rematched a
> fresh-every-step graph. Build it now as the bridge fix; the
> persisted-fragment keying it aligns (11-persistence) stays Intended.

Fragment hash (ops + shapes + dtypes + params over a bounded
subgraph), planner-level memo table, stamp-on-match. Validate on a
real LLM graph: layer 2..N planning cost ≈ 0. Wire the same hash as
the persisted-plan fragment key (11-persistence alignment, header
fields unchanged).

### Stage 6 — plan-driven weight prefetch (~1 session)

The committed plan emits a prefetch schedule: (weight tensor →
device, ordered by plan position). A background streamer issues
page-touch + H2D ahead of the frontier, throttled by VramPressure
(the Phase 5.2 selector's `BackendRuntime` reads). TTFT test: first
token on a cold mmap'd model must not wait for full-model upload.

### Stage 7 — record-replay + horizontal fusion (later, separate prompts)

CUDA-Graph / command-buffer capture per memoized fragment;
similarity scheduling + horizontal-fusion rule family gated on
independence. Each is its own session prompt when reached.

## Code touchpoints

- `fuel-dispatch/src/plan.rs` — `compile_plan`, `PlanOptions`,
  enumeration policy (stages 2–3).
- `fuel-dispatch/src/ranker/cost.rs` — inbound-transfer term
  (stage 2).
- `fuel-core/src/topology.rs` — `TransferEstimate` storage
  (stage 1).
- `fuel-core/src/pipelined_bridge.rs` — prepare() split: planning
  half → plan store; realize = coverage-wait + dispatch (stage 4).
- `fuel-dispatch/src/pipelined.rs` — commit-horizon tracking beside
  the existing generation check (stage 4).
- `fuel-graph` — graph-construction event hook for the driver
  (stage 4); fragment hashing (stage 5).
- FusedOpRegistry — pattern-indexed neighborhood probe (stage 3).

## Constraints (standing)

- Result-returning everywhere; no panics on production paths.
- Every check that can run at plan time must run at plan time.
- Parity gates: each stage proves single-device plans unchanged
  before enabling new freedom.
- Live RTX 4070 sweep after executor-touching stages.
- Baracuda asks (if any kernel gaps surface) are proposed to the
  author first, never assumed.

## Memory planning (must-do for larger-than-VRAM) — added 2026-06-13

Larger-than-VRAM support cannot be claimed until BOTH halves exist. Today
neither is wired into the production realize path (`insert_residency_evictions`
is called only from tests; the placement DP prices residency only at Stage 2's
first cut). Both are required, not optional:

1. **Plan-time, liveness/capacity-aware (part of the DP + optimization).** The
   planner sees the whole graph, so it must price device-memory *capacity* into
   placement, size the activation pool from liveness, and refuse/relocate
   placements that would exceed a device's budget. Extends Stage 2's
   residency-priced placement and the Stage 5 activation-pool sizing note.
2. **Realize-time, pressure-driven eviction/paging.** When live device-memory
   usage crosses a `BackendRuntime` byte budget during realize, evict/page
   (Move to host/disk, re-stage on next use) — analogous to the retired legacy
   const-pool LRU but driven by the plan's residency annotations. Wire
   `insert_residency_evictions` (or its planner-priced successor) into the
   pipelined executor.

Gate: a synthetic graph whose working set exceeds a configured budget must
complete on the live GPU via Move/Copy chains (the larger-than-VRAM test).
Until both land, `06-runtime` should state plainly that larger-than-VRAM models
are unsupported on the pipelined executor.
