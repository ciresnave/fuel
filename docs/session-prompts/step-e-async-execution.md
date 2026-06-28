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

### The Vulkan change
The recorder already batches (`BATCH_LIMIT` ops/command-buffer). Today it `fence.wait`s per flush.
Phase A: flush + **return the fence as the handle** instead of waiting; wait only when a dependent op
on another device (or a host read) needs the result. This alone unlocks intra-device pipelining.

## Phase B — the signal

- **B1 (fuel-internal, primary):** surface the in-flight counter via
  `BackendRuntime::pending_work() -> Option<u64>` (default `None`) **or** through the existing
  `BackendRuntimeLookup` the selector already consults. Covers single-process inter-run parallelism
  (the runtime's job per `06-runtime` §Data parallelism). No sibling change.
- **B2 (sibling, optional):** device-native queue depth for *cross-process* GPU sharing — drafted in
  `docs/outreach/baracuda-queue-depth-ask.md` + `vulkane-queue-depth-ask.md`. Read-only telemetry
  additions; gated on CireSnave's approval. Not required for the single-process win.

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
- **A1** — handle type + the submit/complete trait surface; CPU path (handle = Ready); executor
  tracks handles but still waits at every node (behavior-identical; pure plumbing).
- **A2** — Vulkan async (flush→fence handle, dependency-boundary waits); intra-device pipelining.
- **A3** — CUDA async (stream events, cross-stream waits).
- **A4** — concurrent multi-device scheduling (independent sub-DAGs progress in parallel).
- **B1** — in-flight counter + `pending_work()` seam.
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
