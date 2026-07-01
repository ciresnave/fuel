# Step E — Async execution foundation (Phase A) + live-load arm selection (B/C)

**Status:** design / scoping — *no executor code until Phase A is reviewed.*
**Owner program:** dispatch-core cleanup, Step E (the last item; Steps A–D shipped to `main`,
`d6a596f7`→`29fe586f`, 2026-06-28).
**Prereq map:** A (async execution) → B1 (in-flight counter) → C (streaming walk + `DeviceLoadSelector`);
B2 (sibling queue-depth telemetry) is an optional cross-process refinement.

---

## Why

Step E wants the executor to re-pick `Op::Branch` arms *per decision-point* by **live device load**
— "which path drains the device queues fastest right now" (`06-runtime` §Resolution). Investigation
found the blocker is not the selector (the `RuntimeSelector` seam + a sketched `DeviceLoadSelector`
already exist) but the **execution model**: dispatch is synchronous, so there is no varying queue
depth to react to, and the executor maintains no per-device load signal. Async execution is the gate
— and it is independently valuable (compute/transfer overlap, inter-run parallelism, the
command-buffer replay the runtime doc already calls for).

## Current model (worktree-verified)

`realize_inner` / `realize_many_inner` (`fuel-dispatch/src/pipelined.rs`):
- A **compiler thread** walks the flat dispatch order (`compiler_thread_body` → `compile_one` per
  node) and pushes `WorkItem`s on an `mpsc` channel.
- The **executor thread** consumes the channel: `for item in rx { execute_work_item(&item, …) }`,
  calling the kernel via `execute_compiled` (`compiled.rs`).
- `KernelRef = fn(&[Arc<RwLock<Storage>>], &mut [Arc<RwLock<Storage>>], &[Layout], &OpParams) -> Result<()>`
  — **synchronous**: it returns only after the device finishes. CPU runs in-process; Vulkan records
  into a batched command buffer but `flush_batch` does `fence.wait(u64::MAX)` (effectively per-op);
  CUDA submits to its `Stream` then synchronizes. No two devices' kernels are ever in flight at once.

Consequence: by the time a kernel returns, its device is idle. There is no live queue depth, and no
per-device in-flight notion for a `DeviceLoadSelector` to read.

## Target model (Phase A)

Decouple *submit* from *complete*:
- GPU dispatch **enqueues** to the device's stream/queue and returns a **completion handle** (CUDA
  event, Vulkan fence/timeline value). CPU kernels stay synchronous (their "handle" is already-ready).
- The executor keeps a **per-node completion handle map** and a **per-device in-flight counter**
  (increment on submit, decrement when a handle signals). It **waits only at dependency boundaries**:
  a node submits once its inputs' producer-handles have signalled (or inserts a stream wait); the
  realize waits at the end (D2H / resident-result extraction) and at any host-read (`Op::Copy{Cpu}`).
- Independent sub-DAGs on different devices then progress concurrently → real, varying per-device load.

### The submit/handle surface
Add an async path beside the sync `KernelRef`. Two candidate shapes (decide in review):
1. **A `submit` trait method on the backend** returning a `CompletionHandle` (an enum: `Ready` for
   CPU, `CudaEvent`, `VulkanFence`/timeline). The executor calls `submit` instead of the blocking
   kernel and tracks the handle. `KernelRef` stays for backends/ops without an async path (fallback =
   submit-then-immediately-wait, i.e. today's behavior).
2. **Keep `KernelRef` but make the *executor* own stream/event management** per backend via the
   `DynBackendDevice` (record into the device stream, return an event). Less per-kernel churn; more
   executor-side backend knowledge.
Recommendation to evaluate: (1) — it keeps the executor backend-agnostic (it tracks opaque handles),
matching the "backends advertise, the executor decides" principle.

### Completion tracking + dependency model
- A `NodeId → CompletionHandle` map (executor-local). Before submitting node N, for each input
  produced this realize, either (a) enqueue a stream-side wait on the producer's handle (same device:
  cheap, no host sync), or (b) host-wait if crossing devices / reading on the host.
- The **in-flight counter** (Phase B1) is `HashMap<DeviceLocation, AtomicU64>` (or per-realize +
  a process-wide overlay for inter-run): `+1` on submit, `-1` when a handle is observed signalled
  (polled at submit time + drained at realize end). This *is* the queue-depth signal `DeviceLoadSelector`
  reads — no device API needed.

### The Vulkan change — A2 (turnkey plan from the 2026-06-28 investigation)

**Confirmed:** all Vulkan compute ops go to ONE compute queue (`fuel-vulkan-backend/src/lib.rs:529`),
so submission order = execution order → same-queue producer→consumer deps need **no** per-op wait.
`download_bytes` (the D2H host-read) already calls `flush_pending()` (`lib.rs:1144`); the residency pass
splices `Op::Copy{Cpu}` on every Vulkan→CPU edge (`optimize.rs:319`), so all host reads funnel through it.

**The synchrony today:** every kernel wrapper calls `record_dispatch_batched()` then `flush_pending()`
(`lib.rs:~1822`), and `flush_batch()` (`recorder.rs:204`) does `queue.submit(Some(&fence))` then
`fence.wait(u64::MAX)`. The `BATCH_LIMIT=500` batching never accumulates because of the per-op flush.

**⚠️ The hazard (decisive):** `Recorder::record_batch_dispatch` (`recorder.rs:104`) retains only
`batch_transients` (param/shape uniforms) + `batch_descs` — it tracks the input/output **data** buffers
as raw `u64` handles (`dirty_buffers`, for barriers), NOT as Arcs. So a naive "defer the flush" =
**use-after-free**: a destructively-evicted input (`cache.remove`, `pipelined.rs:709`) or the realize-end
cache drop frees a data buffer while a recorded-but-unsubmitted command still references it.

**Safe shape (avoids touching every wrapper + the recorder's buffer model):**
1. **Backend (one place):** make `flush_pending()` LAZY — flush only when `should_flush()` (batch full);
   add `force_flush()` that always submits + waits. The per-op wrapper calls (now `flush_pending`) thus
   DEFER; the batch accumulates + auto-flushes at `BATCH_LIMIT` (bounds TDR).
2. **Repoint host reads:** `download_bytes` (+ any other `flush_pending` caller that precedes a host
   read — AUDIT all callers) → `force_flush()`.
3. **Executor buffer-lifetime guard (the UAF fix, no recorder change):** in `realize_inner` +
   `realize_many_inner`, `force_flush` the Vulkan backend (via the existing
   `find_vulkan_backend_in_cache`, `pipelined.rs:~4491`) **before every destructive eviction**
   (`cache.remove`) **and at realize-end** (before the cache drops / results return). Same-queue ordering
   covers all non-evicting, non-host-read deps → real intra-device pipelining for compute runs;
   destructive/in-place ops flush first (safe, slightly less pipelining).
4. Leave the `CompletionHandle` (A1) as-is for now — the fence is per-BATCH not per-op, and `KernelRef`
   can't carry it back, so A2 uses the backend-internal lazy-flush model; A4 (concurrent multi-device)
   is where the executor tracks per-device completion explicitly.

**Exact edits (audit complete, `fuel-vulkan-backend/src/lib.rs`):** rename the current `flush_pending`
(`:1491`, always submit+wait) → `pub fn force_flush`; add a new lazy `flush_pending` =
`if should_flush() { force_flush() }`. The ~35 per-op compute wrappers keep calling `flush_pending`
(now lazy → defer). **Repoint exactly these GPU→host sync points to `force_flush`** (the complete
host-read set — a miss = silent stale data): `synchronize_pending` (`:603`), `download_bytes`
(`:1144`), `download_slice` (`:1252`), `download_raw_bytes` (`:9994` — add a `force_flush` if it has
none today), and the batch-full auto-flush in `record_dispatch_batched` (`:1466`). `fill_bytes_zero`
(`:1059`) is a GPU write → stays lazy. **Executor (`pipelined.rs`):** before each destructive eviction
(`cache.remove`, `:~709`) and at realize-end (`:~717` + realize_many `:~946`), `force_flush` the Vulkan
backend via `find_vulkan_backend_in_cache` (`:~4491` — confirm it yields a handle on which `force_flush`
is callable; make `force_flush` reachable, e.g. `pub` + via the storage's `backend()`).

**Verification (mandatory before commit):** CPU suites unaffected (no Vulkan recorder); `cargo check
--features vulkan`; then **live-GPU**: run the `#[ignore]`'d Vulkan suites (one suite at a time, 12 GB
GPU) over BOTH non-destructive (elementwise chains) AND destructive/in-place + Vulkan→CPU graphs, and
diff outputs against the synchronous baseline. The design is race-free by construction (single-queue
order + buffers retained until a forced flush at every host-read/eviction/realize-end), so passing
live-GPU outputs is strong evidence; a failure means a missed flush point.

## Phase B — the signal

- **B1 (fuel-internal, primary):** surface the in-flight counter via
  `BackendRuntime::pending_work() -> Option<u64>` (default `None`) **or** through the existing
  `BackendRuntimeLookup` the selector already consults. Covers single-process inter-run parallelism
  (the runtime's job per `06-runtime` §Data parallelism). No sibling change.
- **B2 (sibling, optional):** device-native queue depth for *cross-process* GPU sharing.
  **CUDA: UNBLOCKED (2026-06-28)** — baracuda **alpha.69** already ships `Stream::is_complete()`
  (cuStreamQuery, this-process stream idle/busy) + `baracuda_nvml::Device::utilization()` /
  `gpu_utilization_percent() -> Option<u8>` (cross-process, alpha.70 alias); NVML is crate-split
  (`baracuda-nvml`), gate it behind a Fuel `cuda-telemetry` feature so the default build stays clean.
  Wire at `fuel-cuda-backend`'s `as_backend_runtime()` → `BackendRuntime::pending_work()`. See
  [`../outreach/baracuda-queue-depth-response.md`](../outreach/baracuda-queue-depth-response.md).
  **Vulkan: RESOLVED (2026-06-28)** — Vulkan has no compute-load query (API boundary, not a Vulkane
  gap); instead Vulkane shipped `PhysicalDevice::device_identity()` (UUID / LUID / PCI), the join-key
  to an out-of-band telemetry source. See [`../outreach/vulkane-queue-depth-response.md`](../outreach/vulkane-queue-depth-response.md).
  **Synthesis:** B2 = an **API-agnostic, identity-keyed GPU-load crate (Fuel-side)** that takes a
  device identity and returns `Option<load>` from the matching vendor/OS backend — NVML (via
  `baracuda-nvml`, matched by UUID) for NVIDIA (CUDA *or* Vulkan), amdgpu sysfs (PCI) for AMD-Vulkan,
  PDH/D3DKMT (LUID) for Windows — read through `BackendRuntime`. One load source serves all backends;
  no per-backend duplication. Neither sibling is required for the single-process win (B1).

## Phase C — streaming walk + `DeviceLoadSelector` (the actual Step E)

- Today `order_for` (pipelined.rs) → `lower_picked_route` (`fuel-graph/src/run.rs`) flattens the whole
  route to a static `Vec<NodeId>`; `compiler_thread_body` walks it; `Op::Branch` nodes are *skipped*.
- Phase C: walk **runs** (`Run` = straight-line segment between decision points; already the unit
  `lower_picked_route` is built from). At each `Op::Branch`, call a re-pick hook that consults the
  Phase-B signal (via `pick_arm`'s existing selector path), choose the arm, then dispatch that run's
  nodes. `pick_route` becomes incremental rather than pre-flattened.
- Implement **`DeviceLoadSelector`** (the `runtime_selector.rs` sketch) reading the Phase-B signal to
  demote arms on loaded devices. It composes into the existing `ChainedSelector` chain (VRAM tier →
  load tier → Judge latency → static cost).
- **Behavior contract:** single-device / no-load realize stays byte-identical to today (the load tier
  is flat → falls through to the current ordering); the change is observable only under genuine
  multi-device contention.

## Suggested PR breakdown (each its own plan + verify)
- **A1 — SHIPPED (`06cf3fbf`).** `CompletionHandle` type at `execute_compiled`; CPU = `Ready`;
  behavior-identical. (Refined: handle originates at `execute_compiled`'s return, not an
  executor-held backend submit; the 390 sync kernels untouched.)
- **A2 — SHIPPED + live-GPU-verified (2026-06-28).** Vulkan async via the lazy-`flush_pending` +
  `force_flush` model (NOT a per-op fence handle — the fence is per-batch). Backend: `flush_pending`
  lazy (BATCH_LIMIT cap), `pub force_flush`, host reads (`download_*`, `synchronize_pending`,
  auto-flush) → `force_flush`. Executor: `force_flush_vulkan` before destructive eviction +
  `force_flush_all_vulkan` at realize-end. Verified: CPU 382/1282 (behavior-identical), vulkan+cuda
  compile, live RTX 4070 (`byte_storage_live` 4; `vulkan_bridge_realize_live` 2 incl. a deep 4-op
  fan-out chain). Same-queue submission order = execution order carries intra-realize deps.
- **A3** — CUDA async. **SHIPPED 2026-06-29 (option 3, stream-ordered alloc/free).** Baracuda
  **alpha.72** made `new_async`/`zeros_async` buffers free STREAM-ORDERED on `Drop` (`cuMemFreeAsync`
  on the retained origin stream) — answering `../outreach/baracuda-stream-ordered-alloc-ask.md` (Q1/Q2
  = yes). So A3 was the small, pure-Fuel path, no retention pool, no executor force-sync guards:
  1. `device.rs` `alloc_zeros` → `DeviceBuffer::zeros_async(ctx, len, &self.stream)` — all 56 op
     allocations (kernel outputs + `Workspace`/scratch, all of which route through `alloc_zeros`) now
     free stream-ordered, so an intermediate dropped while its kernel is in flight frees *after* the
     kernel. UAF eliminated by construction.
  2. Removed all 59 per-op `device.synchronize()?` from the 28 `baracuda/*.rs` compute ops → kernels
     pipeline on the single per-device stream (submission order = execution order carries deps).
  3. `new_with_stream` raises the default mem-pool release threshold to `u64::MAX` (retain freed blocks
     for reuse vs trim-to-OS each sync; non-owning handle). Follow-up: optional `trim_to` at realize-end.
  4. KEPT byte_storage.rs's 5 D2H/H2D syncs — correctness now rests on these host-boundary syncs
     (`to_cpu_bytes` etc.); audit confirmed no op does a raw (non-self-syncing) host readback.
  Single-origin-stream precondition (alpha.72) holds trivially (one stream per device).
  **Verified** RTX 4070 (driver API): `cuda_async_realize_live` 3/3 (mul_add, deep_chain fan-out,
  long_chain 32-op pool-reuse stress), `recip_abs` 2/2, `phase_c_rotating_kv` 3/3 (in-place
  WriteSliceRotating) — 8/8, byte-exact vs the CPU/Vulkan references. cuda compile clean (16m, alpha.72
  rebuild).
- **A2.1 (optional Vulkan refinement, not done).** A2's data-buffer eviction force-flushes the whole
  batch before a destructive `cache.remove` (a pipeline drain on in-place-heavy graphs). Vulkan has NO
  driver-side stream-ordered free (vkAllocate/FreeMemory are host-side, not queue-ordered), so the
  idiomatic analog of option 3 is **manual deferred-deletion**: move the evicted buffer into the
  recorder's `batch_transients` (retain-until-fence) instead of force-flushing — letting in-place
  Vulkan graphs pipeline without a per-eviction drain. A2 already uses this retain-until-fence idiom
  for transients; extending it to evicted data buffers is the optional follow-up. Correctness is
  unaffected (A2 is shipped + verified); this is a throughput refinement for in-place-heavy graphs.
- **A4** — concurrent multi-device scheduling. **A FEATURE, not a benchmark — design in
  [`step-e-a4-multidevice-realize.md`](step-e-a4-multidevice-realize.md) (chosen 2026-06-29).**
  A first scheduling investigation called this "emergent, measure-first," but a deeper feasibility
  probe (2026-06-29) corrected that: it was a *structural* truth with an unmet precondition. Every
  realize entry pins ONE device (`PlanOptions.pinned_device`; "the surviving set lives on ONE device,"
  plan.rs:187-192), and the executor errors on an un-bridged mixed-backend edge (pipelined.rs:4314) —
  so **no realize ever produces the mixed-backend graph the overlap would need.** The async foundation
  (A1/A2/A3) + per-NodeId cache + cross-device `Op::Copy` (CPU↔CUDA/Vulkan, tested) + device selection
  are the ready *pieces*; what's missing is a multi-device entry + cross-device placement (relaxing the
  one-device prune) + executor multi-backend dispatch. Phasing: A4a (device-set entry + explicit
  placement + residency copies — the planner change) → A4b (executor multi-backend audit/fix) → A4c
  (dual-GPU benchmark + correctness, RTX 4070 CUDA + AMD iGPU Vulkan). This is also the gate **C**
  (`DeviceLoadSelector`) needs. **A2.1** (Vulkan deferred-deletion refinement) is queued after the CUDA
  work, independent of A4.
- **B1** — in-flight counter + `pending_work()` seam. (Subsumed by A4-escalation's per-device slot
  state if that path is taken; otherwise a standalone counter incremented on async submit / decremented
  on drain.)
- **C1** — streaming run-walk; **C2** — `DeviceLoadSelector` + per-decision-point re-pick.
- **B2** — sibling telemetry (after their APIs land).

## Open questions (for review before A1)
1. Submit/handle surface: new backend `submit` method (recommended) vs executor-owned stream mgmt?
2. Dependency waits: stream-side waits (preferred, no host sync) vs host-side join — per backend?
3. Storage lifetime across async: `Arc<RwLock<Storage>>` already pins inputs; confirm output buffers
   aren't reused before the producing kernel signals (the safety-copy pass interaction).
4. Failure propagation: a kernel that errors asynchronously — how/when surfaced (poll at next
   dependency wait + at realize end)? `TopologyChanged` retry interaction.
5. Counter scope: per-realize vs process-wide overlay for inter-run load.

## Verification (per phase)
A1–A4: existing realize suites must stay green at each step (behavior-preserving until A4 enables
concurrency); add async-completion-ordering tests; live-GPU `#[ignore]` suites after A2/A3 (one suite
at a time — 12 GB GPU). C: a 2-device branched graph under *simulated* load (a fake `BackendRuntimeLookup`
reporting high in-flight on one device) picks the unloaded arm; no-load stays byte-identical.
