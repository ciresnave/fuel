# Step E A4 — Multi-device realize (concurrent CUDA + Vulkan)

**Status:** design (2026-06-29). Foundation **C** (`DeviceLoadSelector`) builds on this.

> **UPDATE 2026-06-29 — A4a is ALREADY BUILT (verified live).** A fresh max-effort Opus agent went to
> implement A4a and, reading the *prune code* (not just the doc comment), found the premise below was
> wrong: plan.rs:187-192 is NOT a graph-global "one device" constraint — it's a **per-node** prune (each
> node's *alternative set* prunes to *that node's* winner device), which is exactly the per-node
> multi-device placement we want, plus a safety invariant `Op::Branch` arm-selection needs. **No code
> change was required.** A live test (`fuel-core/tests/cuda_multidevice_realize_live.rs`) realizes a
> mixed CPU+CUDA graph in ONE pass — per-node placement honored, cross-device `Op::Copy` auto-inserted
> both directions, byte-exact `[22,86,192,340]` — verified green on the RTX 4070 (independently re-run).
> The only edit was correcting the misleading plan.rs doc comment. So A4a-1 (mechanism) + likely A4a-2
> (the bridge already does cost-based auto-placement; observed but not yet independently confirmed) are
> DONE. **Remaining A4 = A4b (does a mixed realize actually OVERLAP the devices — concurrency, not just
> correctness?) + A4c (dual-GPU benchmark).** The "feature, not a benchmark" framing below was itself
> half-wrong — the placement *mechanism* existed; only the *concurrency* question (A4b) is open. Lesson:
> test, don't read — two read-only probes + I all misread the doc comment.

## Goal

One `realize()` whose graph spans **multiple physical devices**, so independent sub-DAGs placed
on different backends **progress concurrently**. The async foundation already shipped (A1
completion-handle seam, A2 Vulkan async, A3 CUDA stream-ordered) makes the overlap **emergent
once the graph is mixed-backend**: the single executor thread enqueues device-A's op (non-blocking,
A2/A3), enqueues device-B's, and both streams fill. The missing piece is everything that *produces*
a mixed-backend graph and lets the executor hold + dispatch more than one backend in a pass.

## Why this is a feature, not a benchmark (the finding that reframed A4)

A 3-reader feasibility probe (2026-06-29) found the "emergent, measure-first" framing from the
earlier scheduling investigation was **over-optimistic** — it read the `Run`/graph *structure* as
ready but never checked whether any realize path *produces* multi-device work. It does not:

- **Entry pins one device.** `realize_f32_cuda/_vulkan/_one_as/_many_as` (fuel-core/src/lazy.rs:1376,
  pipelined_bridge.rs:117) each take ONE `Device`, threaded as `PlanOptions.pinned_device`
  (plan.rs:147). The planner enumerates per-node candidates only at that device and **"the surviving
  set lives on ONE device"** (plan.rs:187-192) — each node's alternatives prune to a single device
  before residency stitching. `stamp_plan_backends` then writes the pinned backend onto every node
  without an explicit `AlternativeSet` (optimize.rs:294-316).
- **Executor enforces single-backend.** The cache is keyed by NodeId, not backend (pipelined.rs:91);
  `execute_work_item` *errors* if a kernel node's `target_backend` ≠ its input storage backend
  ("mixed-backend kernels require an explicit Op::Copy first", pipelined.rs:4314-4361); the realize
  loop is one sequential walk into one cache.

So the concurrency is a structural truth with an **unmet precondition** — nothing creates the
mixed-backend graph, and the executor would error on one lacking bridge copies.

## What already exists (the foundation — do NOT rebuild)

- **Async dispatch** (A1/A2/A3): per-device streams/queues pipeline; per-device flush/sync guards
  (`force_flush_vulkan`, CUDA stream-ordered free) don't cross-stall other devices.
- **Cross-device `Op::Copy`/`Op::Move`**: implemented for CPU↔CUDA and CPU↔Vulkan (pipelined.rs:3961-4145),
  inserted by `insert_residency_copies`→`insert_cross_device_copies` (optimize.rs:319-392), the copy
  kernel runs on the SOURCE backend and produces output on the target. Proven end-to-end live
  (`residency_eviction_live.rs` — D2H Move + H2D Copy round-trip on CUDA).
- **Per-NodeId cache** holds any backend's `Storage` simultaneously (pipelined.rs:91) — already
  multi-backend-capable as a container.
- **Per-node dispatch**: `execute_work_item` already routes each node to its own `target_backend`;
  it only errors on an *un-bridged* mixed edge — with residency copies present, mixed dispatch is fine.
- **Device selection**: `CudaDevice::new(idx)` + `VulkanBackend::with_selection(DeviceSelection::{Index,ByName,PreferDiscrete})`
  (fuel-vulkan-backend/src/lib.rs:370) — can bind the AMD iGPU for Vulkan while CUDA uses the RTX 4070.
- **`Op::Branch` scaffold** + `target_backend` side-table + `graph.set_target_backend` (fuel-graph/src/lib.rs:1451).

## The build

### A4a — multi-device entry + cross-device placement (the bulk; the planner change)

- **Entry**: a realize variant whose target is a **device set**, not one pinned device — e.g.
  `realize_multi(targets, &[Device])` / a `PlanOptions.device_set: Vec<DeviceLocation>` replacing the
  single `pinned_device`. CPU stays the fallback/host.
- **Placement (per-node, multi-device — the constitutional design, confirmed by CireSnave 2026-06-29).**
  Device placement is **per-node**; the surviving graph is **almost never single-device**, and even a
  path between decision points will *occasionally* span devices. The planner's "the surviving set lives
  on ONE device" (plan.rs:187-192) is **incorrect/scaffolding and is REMOVED**, not worked around.
  Build the *mechanism* first, then the *policy*:
  - **A4a-1 (mechanism): remove the one-device constraint + per-node placement + copies.** Delete/relax
    plan.rs:187-192 so each node keeps its own device; the residency pass inserts cross-device `Op::Copy`
    at boundary edges; the executor dispatches per-node. **Validate with EXPLICIT placement** (the caller
    stamps independent sub-DAGs via `graph.set_placement`/`set_target_backend`, which already has priority
    over `pinned_device`, plan.rs:147-159) — deterministic + testable, exercises the mechanism without
    needing the auto-placement policy yet.
  - **A4a-2 (policy): cost-based auto-placement.** Extend the placement DP so per-node candidate sets span
    the device set; the DP decides each node's device by cost/balance (and, with B1, live load → this is
    where **C**'s `DeviceLoadSelector` rides). Reuses `PlacementForkPathfinder` (driver.rs:245). Layered
    on A4a-1's mechanism once it + A4b are proven.
  - **`Run` seam (constitution update).** Today `Run` = "maximal straight-line SINGLE-device segment
    between decision points" with one `device` field (run.rs). Since a path can now span devices, adopt
    **(a): a cross-device `Op::Copy` is a run boundary** — each `Run` stays a single-device dispatch unit,
    but an inter-decision-point path may be several device-runs stitched by copies. (`Run.device` stays
    meaningful; least churn; the copy is already a node.) Alternative (b) — `Run` spans devices, drops
    `device`, executor dispatches per-node within a run — is feasible but rewrites every `Run.device`
    consumer; not chosen. Either way `docs/architecture/` Run definition must be updated.
- **Residency**: `insert_residency_copies` already inserts copies from placement facts — confirm it
  fires for arbitrary cross-device boundaries (not just eviction), and that CUDA↔Vulkan goes via host
  staging (D2H then H2D through CPU) since there's no direct D2D copy kernel between those backends yet.

### A4b — executor concurrent multi-backend dispatch (likely small; mostly verify + targeted fixes)

- The cache + per-node dispatch + async + per-device guards should already overlap independent
  sub-DAGs. Audit for serialization points: (1) the `TopologyChanged` chunk-boundary check
  (pipelined.rs:690-708) keys on `target_backend` changes — confirm it doesn't force a drain at every
  backend switch (which would serialize the two devices); (2) per-device in-flight tracking is
  independent; (3) the cross-device copy's source-drain (D2H syncs the source device) doesn't stall
  the *other* device. Fix only what an audit/benchmark shows serializes.

### A4c — verification (the original "A4-minimal", now reachable)

- Live dual-GPU: CUDA on RTX 4070 + Vulkan on the AMD iGPU (`DeviceSelection::ByName("AMD")`).
- **Correctness**: a graph with two independent sub-DAGs (one per device) + a CPU reconverge; assert
  byte-exact vs the CPU oracle (exercises the cross-device copies).
- **Concurrency**: heavy independent sub-DAGs (e.g. sizable matmuls), `std::time::Instant` around
  realize; assert wall-clock materially < sum of the two sequential single-device realizes. If it does
  NOT overlap, A4b has a serialization point to fix (the benchmark is the measurement that decides).

## Cross-device correctness

- Cross-device edges self-synchronize via the copy's source-drain (A2 `download_bytes`/`force_flush`,
  A3 `to_cpu_bytes`/sync) — confirmed by the A4 sync probe.
- Per-device flush/sync granularity (A2/A3) means draining one device never stalls another → preserves
  concurrency.
- A3's single-origin-stream precondition holds per device (one stream each); the host-staged copy is
  the cross-device handoff, so no buffer is used on a foreign stream.

## Open questions (for review before code)

1. Entry API shape: a dedicated `realize_multi(&[Device])` vs a `PlanOptions.device_set` that the
   existing entries thread? (Prefer the latter — one planner path.)
2. ~~Is plan.rs:187-192's prune per-node or global?~~ **RESOLVED (CireSnave 2026-06-29):** the
   one-device constraint is incorrect/scaffolding regardless of its granularity — REMOVE it; placement
   is per-node and the surviving graph is almost never single-device. `Run` seam = interpretation (a)
   (cross-device `Op::Copy` is a run boundary) unless changed to (b).
3. CUDA↔Vulkan transfer: host-staged (D2H→H2D, available now) for the first cut; a direct D2D path is a
   later optimization (likely needs sibling support).
4. Does **C** (`DeviceLoadSelector`) ride directly on A4a-2's auto-placement (load-aware = the DP
   ranks by live per-device queue depth)? If so, A4a-2 + B1 (in-flight counter) + C converge — the
   convergence the earlier note flagged, now correctly placed AFTER the multi-device path exists.

## Phasing & risk

**A4a-1 (mechanism) — DONE 2026-06-29** (no code change; the mechanism existed — see the UPDATE banner;
`cuda_multidevice_realize_live.rs` is the live guard). **A4a-2 (auto-placement) — likely DONE** (the
bridge already wires `transfer_estimator` + `fallback_placements_for` + the priced placement DP; an
un-pinned probe split CPU/CUDA on its own — confirm with a no-explicit-placement test). Remaining:
**A4b** — does a mixed realize actually OVERLAP the devices? The A4a test proves *correctness*, not
*concurrency*; audit whether the CPU sub-DAG (synchronous) and the CUDA sub-DAG (async) actually
progress in parallel, and whether the `TopologyChanged` chunk-boundary drain (pipelined.rs:690-708)
serializes at every backend switch. → **A4c** — the dual-GPU concurrency benchmark (RTX 4070 CUDA + AMD
iGPU Vulkan; assert wall-clock < sum-of-sequential), now that two GPU backends can be mixed in one
realize. → then **C** (`DeviceLoadSelector`). Risk is low for what shipped (A4a was doc-only + a test;
single-device realize provably byte-identical — the prune was untouched); the open risk is whether A4b
needs real executor work to get overlap (TBD by measurement). A pre-existing gap noted by the agent: a
CPU-*pinned* mixed realize would need a CUDA device-seed anchor the bridge doesn't add today (the A4a
test pins CUDA, which seeds the device handle the H2D copies need).
