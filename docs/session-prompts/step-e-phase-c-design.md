# Step E Phase C — live-load arm re-pick + automatic cross-device overlap (DESIGN)

**Status: SHIPPED 2026-06-30.** All PRs landed + verified (CPU + live-GPU on RTX 4070 + AMD
iGPU): C-0 residency-all-arms fix (`5cf57516`) → B1 in-flight counter + `BackendStreams` trait
(`d3d21e20`) → C1 streaming run-walk (`aed217d7`) → C2 `DeviceLoadSelector` / live-load arm
re-pick (`337a4b77`) → C3 auto-overlap reorder (`1ad3c32d`). The **live-load arm re-pick is
proven** (a 2-device branched graph picks the unloaded arm under load; VRAM still outranks
load); single-device byte-identical throughout. This is the culmination of the "dispatch-core
cleanup / plan IS the graph" program.

**Follow-ons resolved (2026-06-30):** ① The suspected "operand-order auto-overlap residual" was
a MISDIAGNOSIS — a max-effort investigation (`c48ddfa3`) proved it was MEASUREMENT NOISE, not a
code path. C3's reorder sorts runs by downstream compute weight (operand-position-independent),
so both reconverge operand orders (`xc.add(&xv)` / `xv.add(&xc)`) lower to a **byte-identical
(op,device) dispatch order**, and the executor already eager-submits all Vulkan before any wait —
i.e. the executor is already operand-insensitive. The earlier "0.39 vs 0.0" flipped run-to-run on
identical code (thermal + CUDA mem-pool growth + iGPU OOM ceiling). Now proven deterministically by
`c3_reorder_is_operand_order_invariant` (CPU) + the both-orders benchmark. The full ready-set Kahn
pump remains a *deferred* end-state for bounded-lookahead scheduling / CUDA-graph replay — **not**
for operand-independence. ② The error-path UAF (`SubmittedBatch::Drop` didn't fence-wait) is FIXED
(`1eb3f515`, self-wait-on-Drop). ③ A2.1 (Vulkan deferred-deletion, throughput-only) in flight.

(Originally: design / scoping 2026-06-30 — read-only audit; implemented PR-by-PR via sub-agents
with per-PR review + live verify.)

**Builds on (all SHIPPED + live-verified):** A1 (`CompletionHandle` seam,
`compiled.rs:235`), A2 (Vulkan lazy-flush), A3 (CUDA stream-ordered alloc/free),
A4a (per-node multi-device placement mechanism), A4c-prereq (CUDA+Vulkan correctness),
A4b-1..5 (the cross-device CONCURRENCY mechanism + proof, ~0.50–0.66 overlap
efficiency). See [`step-e-a4b-async-completion.md`](step-e-a4b-async-completion.md),
[`step-e-a4-multidevice-realize.md`](step-e-a4-multidevice-realize.md),
[`step-e-async-execution.md`](step-e-async-execution.md).

---

## 0. TL;DR + the two goals, and how they relate

Phase C has two intertwined goals, both named in `06-runtime` as architecture-intent-
not-yet-built:

1. **Live-load arm re-pick (the original Step E).** The picker re-picks `Op::Branch`
   arms **per decision-point by live device load** — "which path drains the device
   queues fastest right now" (`06-runtime:45` "selecting arms by live device load —
   queue depth and stream utilization … per-decision-point during the dispatch walk").
   Today the picker (`route_picker::pick_route`, `route_picker.rs:224`) runs **once
   per realize, before dispatch**, keyed only on **VRAM free-memory pressure**
   (`ChainedSelector`). Load is not a signal because (a) there is no per-device load
   counter, and (b) the route is one-shot — there is no "during the walk" to re-pick in.

2. **Automatic cross-device overlap (the A4b caveat).** A4b built + proved the
   CUDA+Vulkan concurrency MECHANISM, but **only for a hand-constructed dispatch
   order**. The order comes from `topo_order_multi` (`fuel-graph/src/lib.rs:90`), a
   fixed DFS; the overlap benchmark (`cuda_vulkan_overlap_bench_live.rs:26-29`,
   `203-228`) carefully arranges the graph (CPU-primary pinning + `xv.add(&xc)` input
   ordering so the DFS pops the CUDA chain first) so the eager-submit fires at the one
   backend switch that overlaps. For an **arbitrary** graph the DFS interleaves
   independent device sub-DAGs however it falls out, and the executor emits each
   chunk's D2H right after its producer — serializing them. Phase C must make the
   scheduler dispatch independent sub-DAGs **adjacently** so both devices fill.

**The central question this design resolves: are these ONE load-aware scheduler or
TWO concerns?** Answer (derived from the constitution, §4 below): **one substrate,
two distinct decisions.** `06-runtime`'s "Data parallelism" + "Synchronization at join
points" describe a **single Kahn-style ready-set scheduler** (`06-runtime:75` "The
ready-set tracks all runs whose inputs are available and frontier-finalized. Each
ready run is dispatched to an available slot on its assigned backend"; `:99` "standard
Kahn-style scheduling against the graph"). Within that one scheduler:

- **Arm-pick (C2)** decides, *at a branch*, WHICH arm (= which device-placement) a run
  takes. It is a choice among the optimizer's surviving alternatives.
- **Run-ordering (auto-overlap)** decides, among *independent ready runs already
  committed to their devices*, WHICH dispatches next so both devices stay busy. It is
  not a choice of placement — placement is fixed by the time a run is ready.

They share the B1 load signal and the same scheduler loop, but they are separable: a
graph with no branches still needs auto-overlap (C2 is a no-op), and a graph realized
on one device still needs arm-pick under VRAM/load pressure (auto-overlap is a no-op).
**Recommendation: build them as two PRs against one streaming-walk substrate (C1),
NOT one monolithic scheduler.** §6 sequences this.

**Scope honesty.** The full `06-runtime` "bounded lookahead, slot-filling, run-ahead
work-item producer" scheduler is larger than Phase C should attempt. Phase C delivers
the **minimum that makes both goals real and tested**: a per-device load counter (B1),
a streaming walk that pauses at branches (C1), a load-aware selector (C2), and a
single-pass independent-sub-DAG interleave for auto-overlap. The full bounded-lookahead
ready-set scheduler with CUDA-Graph run replay is follow-on work (§9). This is called
out so review can right-size expectations.

---

## 1. The current one-shot model (worktree-verified)

Confirmed exactly as the prompt frames it. The route is decided ONCE, fully flattened,
then walked linearly:

1. **Pick (once, before dispatch).** `realize_with_optimized_picking_env`
   (`pipelined.rs:576`) calls `pick_route_for` (`:602`) → `route_picker::pick_route`
   (`route_picker.rs:224`). `pick_route` walks **all** branches in topo order
   (`branches_in_topo_order`, `run.rs:244`) and, at each, builds a per-arm
   `AlternativeSet` and consults the `ChainedSelector` (`chained_selector.rs:175`),
   producing a complete `PickedRoute` (`HashMap<NodeId, usize>`, `run.rs:48`). Telemetry
   is sampled ONCE (`RouteCache`/`TelemetryFingerprint`, `route_picker.rs:88-188`). No
   branches ⇒ `None` ⇒ arm-0 fast path (`pick_route_for:614-618`).

2. **Flatten (once).** `order_for` (`pipelined.rs:1204`) → `lower_picked_route`
   (`run.rs:403`) flattens the chosen route to a static `Vec<NodeId>` (`order`,
   `realize_inner:649`). A run = a maximal straight-line single-device segment
   (`extract_runs_multi`, `run.rs:92`); `lower_picked_route` concatenates the runs of
   the chosen arms, skipping non-chosen arms' runs (`non_chosen_arm_nodes`, `:439`).
   The `Op::Branch` node itself is never an executable member (`run.rs:136`); arms are
   SKIPPED, not decision points at dispatch.

3. **Walk (linear).** `realize_inner` spawns `compiler_thread_body` (`:1255`) which
   walks the flat `order`, `compile_one`'s each node, and pushes `WorkItem`s on an
   `mpsc` channel. The executor thread consumes `for item in rx` (`:716`), dispatching
   each via `execute_work_item` (`:3673`) → `execute_compiled` (`compiled.rs:145`). The
   A4b in-flight tracking (`handles`, `inflight_vulkan`, `multi_backend`,
   `current_chunk_backend`) lives in this loop (`:691-833`).

**Consequence for Phase C:** there is no "decision point during the walk" — by the time
the executor runs, every arm is already chosen and the order is frozen. C1 must reopen
this: keep the per-run granularity (`extract_runs_multi` already gives it) but resolve
the next branch's arm *as the frontier reaches it*, reading B1 load at that moment.

**The ordering lever for auto-overlap.** `topo_order_multi` (`fuel-graph/src/lib.rs:90`)
is the SOLE order source (both `lower_picked_route` and `OptimizedGraph::dispatch_order`
flatten its run partition). It is a deterministic DFS: push node, push its inputs, revisit
— so children pop in reverse-input order. Independent sub-DAGs are emitted in whatever
contiguous blocks the DFS produces; nothing interleaves them by device. Auto-overlap is
fundamentally a **reordering of this output** (or a ready-set replacement, §4).

---

## 2. B1 — the per-device live-load signal

### 2.1 What it measures

**In-flight async submission count per device** — the number of GPU operations the
executor has submitted but not yet observed completed, per `DeviceLocation`. This is the
"queue depth / slot utilization" signal `06-runtime:33,45` names. It is a *fuel-internal*
count derived from A4b's existing handle tracking, NOT a device-API query (B2 sibling
telemetry is optional/cross-process, §8). Rationale for choosing op-count over byte-count
or wall-time:

- **Op-count** falls directly out of A4b's submit/drain points (one `+1` per
  `produce_pending` Pending, one `-1` per handle `wait`/drain), is monotone and cheap, and
  is exactly what "queue depth" means. **Chosen.**
- Byte-count (bytes in flight) would better reflect transfer pressure but needs the
  output size threaded to the increment site; defer as a refinement (the selector can
  weight op-count by a static cost estimate it already has — §3).
- Wall-time-since-last-drain is noisy and needs a clock read on the hot path. Rejected.

### 2.2 Where it is exposed — the seam (the key design call)

**Use the Tier-2 `BackendStreams` trait the constitution ALREADY specifies** — do NOT
add load to the base `BackendRuntime`. `05-backend-contract.md:346-362` defines:

```rust
pub trait BackendStreams: BackendRuntime {
    fn pending_work_count(&self) -> Option<u32>;   // submitted-but-not-finished slots
    fn slot_capacity(&self) -> u32;                // advertised concurrency
    fn flush(&self) -> Result<()>;                 // barrier
}
```

with "Implemented by: CUDA (streams), Vulkan (queues / command buffers). Not implemented
by: CPU (synchronous), Reference." This is the constitutional home for B1 and it does not
yet exist in code (grep: `BackendStreams`/`pending_work_count` appear ONLY in the doc).
The `BackendRuntime` base trait (`backend.rs:114`) stays exactly as-is (the honesty
contract `:100-105` is preserved: `Option<u32>` lets a backend say "no queue concept").

**But there is a subtlety the constitution's framing under-specifies, and it drives the
implementation:** the in-flight count the *selector* needs is the executor's own
submitted-not-drained count, which lives in the `handles`/`inflight_vulkan` executor
locals (`pipelined.rs:699,709`) — NOT something a backend-device handle can observe (the
device doesn't know what the executor has chosen to defer-wait). A bare
`CudaDevice::pending_work_count()` could only ask the driver `cuStreamQuery` (busy/idle, a
bool, not a depth) — coarse and process-wide-stream-only.

**Resolution — a process-wide per-device atomic counter, the same idiom as
`TOPOLOGY_GENERATION` (`dispatch.rs:4830`):**

```rust
// fuel-dispatch/src/dispatch.rs (beside TOPOLOGY_GENERATION)
// Keyed by DeviceLocation; small fixed map or a per-(backend,gpu_id) slot array.
static DEVICE_INFLIGHT: OnceLock<DeviceInflightTable> = OnceLock::new();

pub fn inflight_inc(loc: DeviceLocation);     // +1 on async submit
pub fn inflight_dec(loc: DeviceLocation);     // -1 on observed completion
pub fn inflight_count(loc: DeviceLocation) -> u32;   // read for the selector
```

`DeviceInflightTable` = a small `RwLock<HashMap<DeviceLocation, AtomicU32>>` initialized
lazily, or (faster) a fixed `[AtomicU32; N]` indexed by a `(BackendId, gpu_id)` hash with
a fallback map for unusual ordinals. Either way the hot path is a single relaxed
`fetch_add`/`fetch_sub` (no lock once the slot exists) — the same negligible cost the
A4b notes measured for `cuEventRecord`. `Relaxed` ordering suffices: the counter is a
*hint* for scheduling, never a correctness gate (correctness rests entirely on the A4b
waits, §5), so it does not need to synchronize memory.

**`BackendStreams::pending_work_count` then reads this counter** (the device handle knows
its own `DeviceLocation`), so the constitutional trait surface is honored AND the
selector can read load uniformly through the existing `BackendRuntimeLookup`
(`vram_pressure_selector.rs:75`) by downcasting the handle to `BackendStreams` (the Tier-2
pattern `05-backend-contract.md:338` describes: "Selectors check at runtime whether the
backend implements the trait"). The lookup the bridge builds
(`backend_runtime_lookup_for`, `pipelined_bridge.rs:1179`) returns `DeviceRuntimeHandle`
(`:1144`) — extend it to also implement `BackendStreams` by reading `inflight_count`. This
keeps the *one load source serves all backends* shape and means the selector never holds
executor internals — it reads the process-wide counter through the handle, exactly as it
reads free memory today.

### 2.3 Where incremented / decremented (rides on A4b's points)

The counter mirrors the A4b handle lifecycle 1:1 — every place a handle is produced
increments, every place one is waited/dropped decrements:

- **CUDA submit** — `produce_pending` (`compiled.rs:182`) returns
  `Pending(CudaCompletion)` after `record_completion_event`. **`inflight_inc(cuda_loc)`
  here** (the one site that knows a CUDA op was just enqueued). The `_backend` param +
  the output storage's `DeviceLocation` give the key.
- **CUDA drain** — `wait_producer_handle` (`pipelined.rs:4777`), `drain_handles`
  (`:4801`), and the `handles.remove(&destroyed)` eviction (`:829`). **`inflight_dec`**
  at each — i.e. exactly when a `CompletionHandle::Pending` is consumed or dropped.
  Cleanest: decrement inside `CudaCompletion::wait` / a `Drop` on the handle so a single
  site covers wait-or-drop (avoids missing the eviction-drop path).
- **Vulkan submit** — `eager_submit_all_vulkan` (`pipelined.rs:4956`) and the realize-end
  `drain_vulkan_pending` (`:5020`) push `VulkanCompletion`s. **`inflight_inc(vk_loc)` per
  submitted batch.** Vulkan's count is per-*batch* (coarser than per-op — the A4b
  fence-per-batch granularity, `a4b §1.3`), which is fine: it still tells the selector
  "this iGPU has N submitted batches outstanding."
- **Vulkan drain** — `drain_inflight_vulkan` (`:5007`) / `VulkanCompletion::into_wait`
  (`:4920`). **`inflight_dec(vk_loc)`** per batch waited.

**Hot-path cost:** one relaxed atomic add/sub per GPU op-submit and per drain. Negligible
versus the `cuEventRecord` / `vkQueueSubmit` it accompanies. CPU never touches it
(synchronous, no Pending). **Single-device safety:** a single-device realize still
increments/decrements its own device's counter, but nothing *reads* it to make a
different choice (C2 only diverges from arm-0 under cross-device contention, §3), so
behavior is byte-identical — the counter is pure observation.

### 2.4 Counter scope (open question 5 from the A-phase, resolved)

**Process-wide, not per-realize.** A process-wide overlay is what makes *inter-run*
parallelism legible (`06-runtime:45` "single-process inter-run parallelism"): two
concurrent `realize()` calls (data-parallel batches) each see the other's load and steer
around it. A per-realize counter would miss this and is no simpler. The process-wide
atomic is the natural shape and matches `TOPOLOGY_GENERATION`'s precedent.

---

## 3. C2 — the `DeviceLoadSelector`

### 3.1 What it is

A `RuntimeSelector` (`runtime_selector.rs:88`) that, at each `Op::Branch`, ranks the
viable arms by **live load (B1) composed with the existing static/VRAM/Judge ranking**,
demoting arms on busy devices. It is the concrete selector the `runtime_selector.rs`
module sketch (`:49`) names ("`DeviceLoadSelector` — queue-depth + stream-utilization
probing") and says "needs telemetry infrastructure that doesn't exist yet" — B1 IS that
infrastructure.

### 3.2 Composition with the optimizer + the VRAM/Judge chain (the critical constraint)

The optimizer must **NOT pre-bake the load choice** (`plan-is-the-graph` memory: "The
optimizer's job is to emit viable arms and prune impossible ones … it must NOT pre-bake
the load-based choice"). The optimizer already does its part: it emits the arms (per-device
placements, arm-0 = static winner) and prunes by capability. C2 adds a *runtime* tie-break
layer ON TOP of the existing `ChainedSelector` key, it does not replace static ranking.

The `ChainedSelector` (`chained_selector.rs:186-197`) computes a sort key
`(pressure_tier, latency_ns, original_index)` and picks the minimum. **C2 extends this to
`(pressure_tier, load_tier, latency_ns, original_index)`** — load slots BETWEEN the
hard VRAM guard and the latency rank:

1. **`pressure_tier`** (unchanged, the hard guard): `WontFit` skipped; `Tight`=1;
   `Comfortable`/`Unknown`=0. VRAM is a *correctness-adjacent* limit (OOM), so it outranks
   load — never pick a busy-but-fitting device over… no: never pick a device that
   **won't fit** to balance load. Load reorders only *within* a fit tier.
2. **`load_tier`** (NEW): bucket `inflight_count(arm.device)` (read via `BackendStreams`
   through the lookup) into coarse tiers — e.g. `0` (idle / below a small threshold),
   `1` (moderate), `2` (saturated, ≥ `slot_capacity`). Coarse bucketing (not raw count)
   mirrors `free_bytes_bucket` (`route_picker.rs:128`) so jitter doesn't thrash the pick
   and the `RouteCache` fingerprint stays stable. A busy device's arm sorts after an idle
   device's arm of the same VRAM tier — "which path drains the queues fastest right now".
   `Unknown` (a device with no `BackendStreams` / CPU) = tier 0 (no signal, honest — same
   as the VRAM Unknown semantics, `chained_selector.rs:30`).
3. **`latency_ns`** (unchanged): Judge-measured or static composite + inbound transfer.
4. **`original_index`** (unchanged): ties → static winner → determinism.

**Degenerate-fallback guarantee preserved.** With no load signal (single device, or no
`BackendStreams` handle) every arm's `load_tier` is 0, so the key reduces to the current
`(pressure_tier, latency_ns, idx)` — **byte-identical to today's `ChainedSelector`**, which
itself reduces to `WinnerSelector` = arm-0 with no signals (`chained_selector.rs:53-60`).
This is the no-load contract: load changes nothing unless devices genuinely contend.

### 3.3 The `runtime_selector.rs` seam + where it's built

Two implementation shapes; **recommend (a):**

- **(a) Fold load into `ChainedSelector` as a third leg.** Add an optional
  `load_lookup: Option<BackendRuntimeLookup>` (or reuse the same lookup, downcasting to
  `BackendStreams`) and the `load_tier` term to its key. One selector, one composition,
  matches how VRAM + Judge already compose in `ChainedSelector` (which exists precisely
  because standalone selectors "cannot be composed through `RuntimeSelector::select`",
  `chained_selector.rs:6-13`). The production bridge (`production_selector_for`,
  `pipelined_bridge.rs:1123`) constructs it; just pass the load-aware lookup.
- **(b) A standalone `DeviceLoadSelector` + a chaining combinator.** More modular but
  reintroduces the composition problem `ChainedSelector` was built to solve. Rejected for
  the production path; a standalone `DeviceLoadSelector` is still worth shipping as the
  *unit-testable* core (a pure `fn load_tier(count, capacity) -> u8` + a selector wrapper
  for isolated tests), then folded into `ChainedSelector` for production.

**Build site:** `fuel-dispatch/src/ranker/` (new `device_load.rs` for the `load_tier`
helper + standalone selector; extend `chained_selector.rs` for the production leg). The
bridge wires the load lookup beside the VRAM lookup in `backend_runtime_lookup_for`
(`pipelined_bridge.rs:1179`) — the `DeviceRuntimeHandle` (`:1144`) gains a `BackendStreams`
impl reading `inflight_count`. **Never routes through `ExecutionPlan`** (deleted; the
selector reads the graph + the process-wide counter — the constitutional shape).

---

## 4. C1 — the streaming run-walk + the auto-overlap scheduler

This is the structural heart. Two facets on one substrate, per §0.

### 4.1 The streaming walk (makes arm-pick per-decision-point)

**Today:** `pick_route` resolves ALL branches up front (`route_picker.rs:242`), then
`lower_picked_route` flattens (`run.rs:403`), then the executor walks the frozen `Vec`.

**Phase C:** make the picker resolve branches **lazily as the frontier reaches them**,
reading B1 load at that moment. Concretely:

- **`extract_runs_multi` stays** (`run.rs:92`) — it already partitions the graph into
  runs at every boundary including arm-entries and reconverge points (`run.rs:14-30`).
  The run partition is the streaming unit; nothing about run extraction changes.
- **`lower_picked_route` becomes incremental.** Instead of building the whole `PickedRoute`
  then flattening once, the walk processes runs in topo order and, **when it reaches a
  run whose entry is a branch's arm-entry**, it resolves *that branch's* arm via the
  selector (reading current B1 load), records the pick, and emits only the chosen arm's
  runs — skipping the others. The branches-in-topo-order list (`run.rs:244`) gives the
  resolution order; the change is *when* `pick_arm` (`route_picker.rs:265`) is called
  (at the frontier, not all up front). Upstream-first topo order is already the contract
  (`route_picker.rs:240`, `06-runtime:51-53`) so coupled decisions stay consistent.
- **The realize loop pauses at branch resolution.** The cleanest seam: keep the
  compiler-thread / executor-thread split (`pipelined.rs:672-676`) but make the compiler
  thread call the selector at each branch boundary rather than walking a pre-flattened
  `Vec`. The compiler thread already holds the graph read-lock and walks topologically;
  it gains the selector + lookup (passed in, like `sym_env`) and, at an arm-entry,
  resolves the branch before continuing to compile that arm's runs. The executor thread
  is unchanged — it still consumes `WorkItem`s; it just receives the chosen-arm items as
  they're resolved. **B1 load is current at resolution** because the executor has been
  draining/submitting (and updating the counter) while the compiler ran ahead.

**`pick_route` / `pick_route_for` / `order_for` changes:**

- `pick_route` (`route_picker.rs:224`): split into a **per-branch** `resolve_branch(graph,
  branch, &mut picked, bindings, selector, lookup)` (the body of the existing loop,
  `:242-255`) that the streaming walk calls one branch at a time. The whole-route
  `pick_route` stays as the *eager* path for the `RouteCache` fast-lane and tests.
- `pick_route_for` (`pipelined.rs:602`): on a branchless graph still returns `None` (fast
  path unchanged). On a branched graph it no longer pre-computes the route; it hands the
  selector+lookup to the streaming walk.
- `order_for` (`pipelined.rs:1204`): the `OrderSource::Optimized { route: Some }` arm
  (`:1223`) that calls `lower_picked_route` becomes a **streaming lowering** that yields
  runs incrementally with per-branch resolution. The `route: None` arm
  (`OptimizedGraph::dispatch_order`, `:1228`) — the no-selector / branchless path — is
  **untouched**, preserving byte-identity.

**Single-device / no-branch fast path stays byte-identical.** A branchless graph: `has_branch`
false ⇒ `pick_route_for` returns `None` ⇒ the streaming machinery is never entered ⇒
`OptimizedGraph::dispatch_order` flattens `topo_order_multi` exactly as today. A branched
graph under no load/pressure: every branch resolves to arm-0 (the degenerate fallback,
§3.2) ⇒ the emitted order equals `lower_runs_arm0` = today. **The streaming change is
observable only when a branch's pick differs from arm-0**, which only happens under
genuine VRAM or load contention.

### 4.2 The auto-overlap scheduler (makes A4b automatic)

This is the **distinct** concern (§0): given runs already committed to their devices
(arm-pick done, placements fixed), order *independent* runs so both devices fill.

**The problem, precisely.** `topo_order_multi` (`fuel-graph/src/lib.rs:90`) is a fixed
DFS. For two independent sub-DAGs A (CUDA) and B (Vulkan) that reconverge, the DFS emits
one fully then the other (in reverse-input order of the join). The executor then runs A's
runs (CUDA streaming, A3), hits the backend switch, eager-submits Vulkan only when it
*leaves* a Vulkan chunk (`pipelined.rs:746`), and — critically — if A's reconverge needs a
D2H of A's result *before* B is even recorded, that D2H serializes (the A4b caveat). The
benchmark dodges this by hand-ordering so the Vulkan chunk is recorded-then-submitted while
the already-enqueued CUDA chunk runs, with NO intervening CUDA D2H
(`cuda_vulkan_overlap_bench_live.rs:216-228`).

**The fix: a load-aware run-ordering pass that interleaves independent device sub-DAGs so a
chunk on device X is submitted/recorded before device X's results are needed, letting the
other device's already-submitted chunk run concurrently.** Two design options:

- **(A) A reordering pass over the run list (recommended for Phase C).** After
  `extract_runs_multi`, before flattening, reorder runs so that when an independent
  sub-DAG exists on a *different* device, its entry run is scheduled *adjacent to* (just
  before) the current device's run that would otherwise block on a cross-device join.
  Concretely: a stable topological reordering that, among ready runs (all inputs emitted),
  prefers emitting a run on a device with **lower B1 load** AND prefers keeping a device's
  chunk contiguous up to a backend-switch, so the eager-submit at the switch
  (`pipelined.rs:746`) hands a full chunk to the idle device. This is a *run-list
  transform*, lives beside `extract_runs_multi`/`lower_picked_route` in `fuel-graph` (or
  `fuel-dispatch` if it needs the live counter), and is gated to multi-device graphs (a
  single-device run list has nothing to interleave ⇒ identity ⇒ byte-identical).
- **(B) A true Kahn ready-set scheduler replacing the linear walk (the full
  `06-runtime` shape, follow-on).** Maintain a per-run unresolved-input count
  (`06-runtime:99`); when a run's inputs are all complete it joins the ready set; dispatch
  the ready run on the **least-loaded available device** (B1). This is the architecture's
  end state but is a larger executor rewrite (the `for item in rx` loop becomes a ready-set
  pump with multiple in-flight runs and slot accounting). **Defer to follow-on (§9).**

**Recommendation: (A) for Phase C.** It makes overlap automatic for arbitrary graphs (the
stated goal) without the full ready-set rewrite, composes with the existing A4b
eager-submit + in-flight handles unchanged (it only changes the *order* runs reach the
executor — the executor's chunk-boundary eager-submit and cross-device waits are exactly
the mechanism that then overlaps), and degrades to identity on single-device. The full
ready-set scheduler (B) is the documented end state and should be the explicit follow-on
once (A) proves the interleave heuristic and B1 is trusted.

### 4.3 Composition: one scheduler, two decisions (the resolved central question)

- **Arm-pick (C2)** runs in the **streaming walk (C1, §4.1)** — at a branch, before the
  arm's runs are committed.
- **Run-ordering (auto-overlap, §4.2)** runs over the **committed runs** — after arm-pick,
  ordering independent same-realize runs across devices.

They are **the same scheduling loop reading the same B1 counter**, but at different
moments: C2 chooses a *placement* among optimizer alternatives; auto-overlap chooses a
*dispatch order* among fixed placements. In option (A) they layer cleanly: the streaming
walk resolves arms (C2), the reordering pass interleaves the resulting runs (overlap). In
the eventual option (B) they fuse into one ready-set pump (the ready run's device is
either fixed (non-branch) or chosen by the selector (branch arm-entry), and the pump
dispatches to the least-loaded slot). Phase C ships the layered form; the constitution's
single-scheduler form is the convergence point (§9).

**Composition with A4b eager-submit (unchanged + load-bearing).** Auto-overlap (A) only
reorders the runs feeding the executor. The executor's A4b machinery —
`current_chunk_backend` switch detection (`pipelined.rs:724`), `eager_submit_all_vulkan`
at a Vulkan-chunk exit (`:746`) and before a cross-device copy (`:776`), the in-flight
Vulkan drain before a Vulkan-source read / eviction / realize-end (`:763,787,813,847`),
and the finer CUDA `wait_producer_handle` (`:794`) — is **exactly the mechanism that
turns the better order into overlap**. Reordering to put an independent Vulkan chunk
adjacent to a CUDA chunk means the eager-submit fires productively (the iGPU runs the
submitted Vulkan batch while CUDA streams), which is what the benchmark hand-arranges.
Nothing in A4b changes; it just stops depending on a hand-built order.

---

## 5. Race-safety + correctness

**The invariant: B1 + the streaming walk + reordering are all SCHEDULING HINTS;
correctness rests entirely on A4b's waits/handles, which are untouched.**

1. **The A4b waits/copies are preserved verbatim.** Reordering (A) changes only the
   *sequence* runs reach the executor; every cross-device `Op::Copy` is still a node, still
   a run boundary (`run.rs:153`), still triggers `wait_producer_handle` on its source
   (`pipelined.rs:778-795`) and the in-flight Vulkan drain when its source is Vulkan
   (`:787`). The realize-end `drain_handles` + `drain_inflight_vulkan` + `drain_vulkan_pending`
   (`:843-853`) are unchanged. **No reorder can drop a wait** because the waits key on the
   *node's inputs*, not on dispatch position.

2. **The streaming walk preserves topo order.** Arm-pick at the frontier still resolves
   branches upstream-first (`branches_in_topo_order`, `run.rs:244`) and a run still only
   dispatches after its inputs (the compiler thread walks topologically). The reordering
   pass (A) must be a *valid topological reordering* (a run is emitted only after all its
   inputs' runs) — this is the one hard correctness obligation on the pass, and it is
   checkable (assert the emitted order respects the DAG). Within that constraint it is free
   to interleave devices.

3. **Arm-safety invariant (A4a) — the per-node placement guard MUST hold.** A4a's prune
   (`plan.rs:187-192`, the "per-node winner device" prune that the `plan-is-the-graph`
   memory flags as a **load-bearing safety invariant, do NOT delete**) ensures every node's
   inputs were copied to *its* device. **C2 must not pick an arm whose inputs were never
   copied to that arm's device.** Today this holds because the optimizer's residency pass
   inserted copies for the arms it kept, and the selector picks only among kept arms
   (`route_picker.rs:282` iterates `arms` = the branch's `inputs`, each a real placed
   sub-DAG). The streaming re-pick does not change *which arms exist* — it only changes
   *which kept arm is chosen and when* — so the residency copies for every arm are already
   in the graph (the optimizer emitted them at plan time for all surviving arms). **The
   design obligation: confirm `insert_residency_copies` stitches copies for ALL arms of a
   branch, not only arm-0** (the prompt's A4a note + `route_picker` test
   `vram_pressure_picks_host_ram_arm` exercising a CPU arm suggests it does, but this is the
   one thing C2 must verify before it can re-pick arm-1+ at runtime). If an arm's inbound
   copies were pruned because arm-0 was assumed, that is a prerequisite bug to fix in C2's
   first PR (see §7 risks).

4. **Determinism (`06-runtime:89`).** Load-based ordering is non-deterministic in *order*
   (which independent run dispatches first depends on live load) but every op is
   bit-deterministic, so outputs are invariant — only wall-clock varies. The arm-pick is
   value-preserving by construction (`run.rs:399` "every arm is a valid kernel for the same
   op (cast-to-uniform at `reconverge_at`)"). The reorder-invariance test (§7) is the guard:
   the same graph realized under injected/varying load produces byte-identical output.

5. **Within-realize observations don't revise committed work (`06-runtime:175`,
   "Cancellation is not supported" `:67`).** The streaming re-pick reads load at an
   *upcoming* branch the frontier hasn't reached — it never pulls back a dispatched run.
   "Dispatched runs are committed" (`:69`). This reconciles "re-pick during the walk" with
   "the in-flight realize runs on the route it started with": the route is resolved
   *progressively forward*, never revised backward.

6. **Single-device + no-load = byte-identical (the hard gate).** Branchless ⇒ streaming
   machinery never entered (§4.1). Single-device branched under no contention ⇒ arm-0
   everywhere + identity reorder ⇒ today's order. B1 counter increments but is never read
   to diverge. The `multi_backend` gate (`pipelined.rs:715`) keeps A4b's eager path
   unreachable on single-device, unchanged. **This is the same contract every Step E phase
   held and the primary regression gate.**

**Failure propagation** is unchanged from A4b §4: an async fault surfaces at the next
handle `wait` (CUDA sticky error / Vulkan `VK_ERROR_DEVICE_LOST`), `?`-propagated out of
the realize loop, latest at the realize-end drain. The B1 counter on an error path: decrement
must still fire (put it in `CudaCompletion::wait`/`Drop` so an error-propagated drain still
decrements — else the counter leaks and biases future scheduling; harmless to correctness,
but fix it for hygiene).

---

## 6. PR breakdown (sequenced, each independently testable, single-device-safe)

Ordered B1 → C1-walk → C2 → auto-overlap, because each builds on the last and the load
signal must exist before the selector can read it, and the streaming walk must exist before
per-decision re-pick has a place to happen. Auto-overlap is sequenced LAST as the distinct
concern that needs B1 + the multi-device order machinery.

- **C-0 — arm residency-copy audit (prerequisite, may be a no-op).** Confirm
  `insert_residency_copies` stitches inbound copies for **every** surviving arm of a branch,
  not just arm-0, so C2 can legally re-pick arm-1+ at runtime (§5.3). A pure read +
  born-red test: build a 2-device diamond, optimize, assert both arms' device inputs have
  their `Op::Copy`. If it fails, fix here before C2. **Gate:** the test + existing
  multi-device live suites stay byte-exact.

- **B1 — per-device in-flight counter + `BackendStreams` seam.** Add `DEVICE_INFLIGHT` +
  `inflight_inc/dec/count` (`dispatch.rs`, beside `TOPOLOGY_GENERATION`); wire inc/dec into
  the A4b submit/drain points (`compiled.rs:182`, `pipelined.rs:4777/4801/4956/5007/5020`,
  ideally via `CudaCompletion`/`VulkanCompletion` so wait-or-drop both decrement); add the
  `BackendStreams` trait (`fuel-backend-contract`, per `05-backend-contract.md:346`) with
  CUDA + Vulkan impls reading the counter; extend `DeviceRuntimeHandle`
  (`pipelined_bridge.rs:1144`) to expose it. **Behavior-preserving: nothing reads the
  counter yet.** **Gate:** counter unit tests (inc/dec balance across a realize, empty at
  end — mirror the A4b `handles`-empty assert); existing CPU/CUDA/Vulkan suites byte-exact;
  a live assertion that the counter is non-zero on both devices mid-overlap-benchmark
  (`a4b §6.6` already proposed this).

- **C1 — streaming run-walk (arm resolution at the frontier).** Split `pick_route` into
  per-branch `resolve_branch` (`route_picker.rs:242`); make the `OrderSource::Optimized {
  route: Some }` lowering (`pipelined.rs:1223`) stream runs with per-branch resolution;
  thread the selector+lookup into the compiler thread. **Still uses the existing
  `ChainedSelector` (VRAM only) — no load yet — so under no pressure the streamed order
  equals `lower_runs_arm0` byte-for-byte.** **Gate:** the `run.rs`/`route_picker.rs` unit
  tests still pass (the streamed route equals the one-shot route on every existing
  fixture); `lower_picked_route_follows_chosen_arm_and_skips_others` (`run.rs:700`) holds
  incrementally; branchless + single-device live suites byte-exact.

- **C2 — `DeviceLoadSelector` (load-aware arm-pick).** Add the `load_tier` helper +
  standalone selector (`ranker/device_load.rs`); fold the `load_tier` leg into
  `ChainedSelector` (`chained_selector.rs:186`); wire the load lookup in the bridge. Now the
  streaming walk re-picks by live load. **Gate (the headline verification):** a 2-device
  branched graph picks the UNLOADED arm under *simulated* load — a fake `BackendStreams`
  lookup reporting high `pending_work_count` on one device flips the pick to the other
  device's arm; output byte-identical; the chosen node lands on the right device (assert
  via `target_backend` of the realized nodes). No-load ⇒ arm-0 ⇒ byte-identical (the
  degenerate-fallback test, mirror `no_telemetry_picks_arm0_empty_route`,
  `route_picker.rs:530`). Unit-test the tiering with a `MockBackendStreams`.

- **C3 — auto-overlap run-ordering pass (option A).** Add the load-aware
  topological-reorder pass over the run list, gated to multi-device graphs; identity on
  single-device. **Gate (the second headline verification):** extend the A4b dual-GPU
  benchmark (`cuda_vulkan_overlap_bench_live.rs`) to build an arbitrary independent-sub-DAG
  graph **WITHOUT** the hand-constructed `xv.add(&xc)` ordering / CPU-primary trick — e.g.
  reconverge with inputs in the "wrong" order, or two unrelated outputs — and assert overlap
  efficiency ≥ 0.4 anyway (the pass found the interleave the benchmark used to hand-build).
  Add a **reorder-invariance** test: realize the same multi-device graph under
  artificially varied load and assert byte-identical output (proves the reorder is a
  hint, not a correctness lever). Single-device + branchless byte-exact.

**Why this order over alternatives.** B1 before C2 is forced (selector reads the signal).
C1 before C2 is forced (re-pick needs a place to happen). Auto-overlap (C3) is last because
it is the separable concern (§0) and benefits from B1 being trusted + the multi-device
machinery exercised by C0/C2. C0 is first as a cheap de-risking read. An alternative —
auto-overlap (C3) *before* the load selector (C2) — is defensible (auto-overlap delivers
the more universally-applicable win and doesn't strictly need C2), and **if review wants
the overlap win soonest, C3 can move ahead of C1+C2** (it needs only B1 + a static
reorder heuristic, with load as a refinement). Flagged as a legitimate resequencing.

---

## 7. Verification (the two headline gates + the regression floor)

- **Gate 1 — load-based arm-pick (C2).** A 2-device branched graph (the
  `route_picker.rs:459` diamond, arm-0 CUDA / arm-1 Vulkan-or-CPU) realized with a fake
  `BackendRuntimeLookup` whose `BackendStreams::pending_work_count` reports the CUDA device
  saturated: the picker takes the *unloaded* arm; output byte-identical to the loaded-arm
  result (value-preserving); the realized node's `target_backend` is the unloaded device.
  Mirror the existing `vram_pressure_picks_host_ram_arm` test shape (`route_picker.rs:557`)
  but with a load signal instead of a memory signal. No-load ⇒ arm-0 ⇒ empty route ⇒
  byte-identical.
- **Gate 2 — automatic overlap of arbitrary sub-DAGs (C3).** Extend
  `cuda_vulkan_overlap_bench_live.rs` with a variant that does NOT hand-construct the
  overlap-friendly order (`:203-228`) and assert `overlap efficiency ≥ 0.4` and
  `combined < sequential sum` (the same hard gates, `:321,333`). The pass must recover the
  overlap the benchmark currently arranges by hand. Pair with the reorder-invariance
  byte-equality test.
- **Regression floor (every PR).** The live suites stay byte-exact:
  `cuda_async_realize_live`, `vulkan_bridge_realize_live`, `cuda_multidevice_realize_live`,
  `cuda_vulkan_multidevice_realize_live` (the `[22,86,192,340]` oracle), the existing
  overlap benchmark. CPU suites unaffected. One live-GPU suite at a time (12 GB GPU,
  `--test-threads=1`). The single-device byte-identity gate is the contract.
- **B1 hygiene.** Counter inc/dec balanced (empty per device at realize-end, a debug-assert
  mirroring `drain_handles`' `:4810`); non-zero on both devices simultaneously during the
  heavy mixed realize (direct evidence of concurrent in-flight work).

---

## 8. B2 / sibling telemetry (out of scope, confirmed optional)

B2 (device-native cross-process queue depth) is RESOLVED + OPTIONAL per the program notes:
CUDA has `Stream::is_complete` + NVML utilization (baracuda alpha.69/70); Vulkan has
`device_identity()` as a join-key to an out-of-band source. **Phase C does NOT gate on
siblings** — the fuel-internal B1 counter (§2) is the primary, sufficient signal for
single-process inter-run + intra-realize load. B2 would later widen
`BackendStreams::pending_work_count` to *also* fold in cross-process device utilization
(an identity-keyed GPU-load crate, `step-e-async-execution.md:136`), read through the same
`BackendStreams` seam — purely additive, no Phase C dependency.

---

## 9. Open questions + the follow-on

1. **Load tiering thresholds.** What `pending_work_count` buckets map to load tiers 0/1/2,
   and against `slot_capacity` or absolute counts? Propose: tier by `count / slot_capacity`
   (idle <0.25, moderate <1.0, saturated ≥1.0) so it's device-relative. Needs the Gate-2
   benchmark to tune, like the A4b eager-submit threshold (`a4b §8.3`).
2. **Vulkan per-batch vs per-op counter granularity.** B1's Vulkan count is per-*batch*
   (the A4b fence granularity), so a long Vulkan chunk counts as "1 in flight" while an
   equivalent CUDA chunk counts as N. Does the selector need to normalize (weight CUDA
   op-count down, or count Vulkan ops within a batch)? Likely fine coarse (the tiering is
   relative), but flag for the benchmark.
3. **Reorder pass location — `fuel-graph` or `fuel-dispatch`?** If the auto-overlap pass
   (A) reads the *live* B1 counter it must live in `fuel-dispatch` (the counter's crate);
   if it's a static structural interleave (device-alternating topo order) it can live in
   `fuel-graph` beside `extract_runs_multi`. Recommend: static structural interleave in
   `fuel-graph` for C3 (simpler, deterministic, testable), with live-load refinement as a
   `fuel-dispatch` follow-on — this also makes Gate-2 a non-live unit test of the ordering.
4. **Multi-GPU CUDA (≥2 CUDA devices).** B1's counter is keyed by `DeviceLocation`
   (handles gpu_id), but `find_cuda_device_in_cache` matches the first CUDA storage
   regardless of gpu_id (`pipelined.rs:4836` "single-GPU setups always match"). Two CUDA
   GPUs need the lookup + counter to distinguish ordinals. Out of scope for Phase C (one
   CUDA + one Vulkan), flagged (also A4b open Q5).
5. **Does C2's arm-pick subsume A4a-2's cost-DP auto-placement?** `06-runtime:45` +
   `a4-multidevice` open-Q4 note the convergence: load-aware placement could be the DP
   ranking by live queue depth. Phase C does NOT touch the optimizer DP (it only re-picks
   among arms the DP emitted); whether the DP itself becomes load-aware is a later
   optimizer question, deliberately deferred.
6. **The full ready-set scheduler (option B, §4.2) — the explicit follow-on.** The
   constitution's end state (`06-runtime:71-99`) is a bounded-lookahead Kahn ready-set pump
   dispatching runs to least-loaded slots with run-ahead work-item production and CUDA-Graph
   run replay (`:57`). Phase C ships the layered streaming-walk + reorder-pass form; the
   ready-set rewrite is the next program. **The doc obligation:** Phase C must update
   `06-runtime`'s "what is landed today" (`:45`) to reflect live-load arm-pick + auto-overlap
   landing, and note the ready-set scheduler as the remaining gap — keeping the constitution
   in step with code per CLAUDE.md.

---

## 10. Summary judgement (engage-critically)

- **The two goals ARE separable and should be two PRs (C2, C3) on one streaming substrate
  (C1) + one signal (B1)**, not one monolithic load-aware scheduler. The constitution's
  single-ready-set-scheduler is the *convergence point* (follow-on option B), not Phase C's
  shape. Phase C is the layered, testable, single-device-safe path to both wins.
- **B1's home is the constitution's pre-specified `BackendStreams` Tier-2 trait** backed by
  a process-wide atomic (the `TOPOLOGY_GENERATION` idiom), read by the selector through the
  existing `BackendRuntimeLookup` — no new executor-internal leak, no base-trait change.
- **C2 is a fourth sort-key leg on `ChainedSelector`**, not a new composition — it slots
  between the VRAM guard and the latency rank and degrades to byte-identical with no load.
- **Auto-overlap (C3) is a topological-reorder pass, not an A4b change** — A4b's
  eager-submit + waits are exactly the mechanism that turns the better order into overlap;
  C3 just stops depending on a hand-built order. Recommend a *static structural* interleave
  for the first cut (deterministic, unit-testable), live-load refinement later.
- **The one real prerequisite to de-risk first (C-0):** confirm the optimizer's residency
  pass stitches inbound copies for ALL arms, not just arm-0 — C2 cannot legally re-pick
  arm-1+ at runtime otherwise (the A4a per-node placement safety invariant). Cheap to check,
  potentially a small fix.
- **Right-sizing:** the framing's "scheduler" is bigger than Phase C should swallow whole;
  ship the four-PR layered form, defer the full ready-set pump. Get the seam (B1 via
  `BackendStreams`, the streaming walk reusing `extract_runs_multi`, the reorder as a
  run-list transform) right and the end-state scheduler is a clean follow-on.
