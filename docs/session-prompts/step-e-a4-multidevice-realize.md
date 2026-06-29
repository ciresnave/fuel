# Step E A4 — Multi-device realize (concurrent CUDA + Vulkan)

**Status:** design (2026-06-29), for review before code. Chosen over the "validation slice"
and "pause" options: build the full multi-device realize path. This is the largest remaining
Step E piece and the foundation **C** (`DeviceLoadSelector`) also requires.

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
- **Placement**: relax the planner's "surviving set lives on ONE device" to a **per-node device
  assignment across the set**. Two sub-options:
  - **A4a-1 (first cut, recommended): explicit placement.** The caller stamps independent sub-DAGs to
    devices (`graph.set_placement`/`set_target_backend`), which already has **priority over
    `pinned_device`** in resolution (plan.rs:147-159). The planner respects those, the residency pass
    inserts cross-device copies at boundary edges, the executor dispatches per-node. Deterministic +
    testable; the smallest change that unlocks a real mixed-backend realize. **Verify** the
    "one device" prune (plan.rs:187-192) is per-node-set (honors explicit per-node placement) vs
    global — if global, that prune is the specific code to relax.
  - **A4a-2 (follow-up): cost-based auto-placement.** Extend the placement DP so per-node candidate
    sets span the device set; partition independent sub-DAGs across devices by cost/balance. Reuses
    `PlacementForkPathfinder` (driver.rs:245). Heavier; defer until A4a-1 + A4b prove the path.
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
2. A4a-1 explicit-placement: is plan.rs:187-192's prune per-node or global? (Determines whether A4a-1
   is "just use existing explicit placement" or needs a planner relax.)
3. CUDA↔Vulkan transfer: host-staged (D2H→H2D, available now) for the first cut; a direct D2D path is a
   later optimization (likely needs sibling support).
4. Does **C** (`DeviceLoadSelector`) ride directly on A4a-2's auto-placement (load-aware = the DP
   ranks by live per-device queue depth)? If so, A4a-2 + B1 (in-flight counter) + C converge — the
   convergence the earlier note flagged, now correctly placed AFTER the multi-device path exists.

## Phasing & risk
A4a-1 (entry + explicit placement + residency) → A4b (executor multi-backend audit/fix) → A4c
(dual-GPU benchmark + correctness) → [A4a-2 auto-placement → C `DeviceLoadSelector`]. Biggest risk is
A4a's planner change (the "one device" invariant is correctness-critical — relax it carefully, keep
single-device realize byte-identical) and dual-GPU test flakiness. Nothing is byte-affecting for
existing single-device realizes (the multi-device path is additive, gated on a device-set entry).
Each phase ships with its test; A4c needs the AMD iGPU + RTX 4070 both live.
