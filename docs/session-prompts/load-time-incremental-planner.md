# Session prompt — load-time incremental planner + cost-based placement

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

### Stage 1 — transfer calibration probe (~half session)

Numeric `TransferEstimate` per `SystemTopology` transfer path:
measure H2D/D2H/D2D at a few sizes per path (Judge-style, once per
topology generation), fit `bytes/bandwidth + latency`, store on the
topology snapshot. Pure Fuel-side (`cuMemcpy` timing; Vulkan staging
equivalents). Un-parks TDP-4 option B. Tests: probe runs on CPU-only
hosts (zero paths) and on the RTX 4070 box (live, `#[ignore]`).

### Stage 2 — residency plumbing + greedy inbound pricing (~1 session)

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

### Stage 3 — carry-forward placement DP + fused jumps (~1–2 sessions)

Replace greedy per-node commits with the `(position × device)` DP
(design pillar 3+4) inside the plan walk. Chain-first: linearize the
trunk, handle joins by heuristic state-merge (cheaper producer
device), document the branch-handling debt. Backtracked plan stamps
`target_backend` per node exactly as Phase 4a does today. Tests:
synthetic graphs where (a) a mid-sequence GPU segment beats
all-CPU/all-GPU, (b) a fused op must LOSE because it strands
residency, (c) DP plan == greedy plan when transfers dominate.

### Stage 4 — the incremental driver + commit horizon (~2 sessions)

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
