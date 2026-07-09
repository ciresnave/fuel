# Step E A4b — Executor-orchestrated cross-device concurrency (async completion handles)

**Status: SHIPPED 2026-06-29.** All 5 PRs landed + verified (byte-exact + live-GPU on RTX 4070 + AMD
iGPU): A4b-1 CUDA `Pending(Event)` (`51d43e93`) → A4b-2 Vulkan `Pending`/`submit_batch` (`056ca786`) →
A4b-3 finer cross-device wait (`4150ad7a`) → A4b-4 eager Vulkan submit + A4b-5 overlap benchmark
(`16aefc28`). **Real concurrency confirmed: ~0.50–0.66 overlap efficiency** (combined ~1.10s vs
sequential sum ~1.26s on the dual-GPU benchmark). Single-device byte-identical throughout (the
`multi_backend` gate keeps the eager path unreachable on single-device graphs).

> **IMPORTANT caveat (carry into Phase C):** A4b-4 is the overlap *enabler* + proof; it does NOT yet
> overlap *arbitrary* independent sub-DAGs automatically. The topo scheduler emits each chunk's D2H
> copy right after its producer, which serializes them, so the A4b-5 benchmark *constructs* the
> dispatch order the mechanism overlaps (CUDA-side reconverge, no intermediate cross-device D2H).
> **General automatic overlap = the Phase C `DeviceLoadSelector`/scheduler frontier** (dispatch
> independent sub-DAGs adjacently). A4b supplies the mechanism; C makes it automatic. The §6.5
> "wall-clock < sum" framing below should be read with this in mind.

(Originally: design 2026-06-29 — read-only audit against pinned `baracuda-driver 0.0.1-alpha.72` +
the live `vulkane` checkout; implemented PR-by-PR via sub-agents with per-PR review + live verify.)

**Builds on:** A1 (`CompletionHandle`/`Completion` seam — shipped), A2 (Vulkan lazy-flush — shipped),
A3 (CUDA stream-ordered alloc/free — shipped), A4a + A4c-prereq (per-node multi-device placement +
mixed CUDA+Vulkan realize CORRECTNESS — shipped, `54e7043b`,
`cuda_vulkan_multidevice_realize_live.rs` green). See
[`step-e-async-execution.md`](step-e-async-execution.md) and
[`step-e-a4-multidevice-realize.md`](step-e-a4-multidevice-realize.md).

**Goal:** make independent sub-DAGs placed on different devices (CUDA + Vulkan) actually PROGRESS IN
PARALLEL during one `realize()`. Today they do not (except incidentally) — that is the A4b gap.

---

## 0. TL;DR + the critical re-framing of the audit

The audit framing is **correct that there are zero `CompletionHandle::Pending` producers**
(`compiled.rs:156` always returns `Ready`; the only `Pending` is in a unit test, `compiled.rs:262`),
and correct that the cross-device `Op::Copy` D2H drain serializes at dependency edges. But three
findings sharpen (and partly *simplify*) the build:

1. **The overlap enabler is NOT the `CompletionHandle`. It is Vulkan submit timing.** CUDA already
   submits on launch (A3: kernels enqueue on the per-device stream, no per-op sync — `device.rs:917`
   `synchronize()` is stream-level and now only called at host boundaries). So if a CUDA sub-DAG and a
   Vulkan sub-DAG are both *submitted*, they already overlap on the two devices with zero handle
   machinery — the executor is single-threaded but both devices are async. The blocker is that
   **Vulkan does not submit until `force_flush`** (`lib.rs:1498`), and the executor's only
   `force_flush` triggers are destructive eviction (`pipelined.rs:717`/`955`) and realize-end
   (`pipelined.rs:732`/`970`). On a typical mixed graph the Vulkan sub-DAG sits in the recorder's
   batch, unsubmitted, until realize-end — so the GPU never even *starts* it concurrently. **A4b's
   load-bearing change is eager Vulkan submission at sub-DAG boundaries, not the completion handle.**

2. **The `CompletionHandle` is what makes eager submission SAFE and makes the cross-device edge
   FINER than today's full source-device drain.** Once Vulkan submits early, the recorder's
   submit-then-wait (`recorder.rs:225-226`) must split into submit-now / wait-later, and "later" needs
   a place to store the fence → that is `Pending(fence)`. Likewise the CUDA→CPU copy can wait a single
   recorded *event* (the producer's) instead of draining the whole source stream. So the handle is
   real and needed; it is just downstream of the submit-timing change in importance.

3. **A stream-level / batch-level handle SUFFICES; we do NOT need per-op events or timeline
   semaphores.** baracuda exposes per-op `Event` (`event.rs`: `record(&Stream)`, `synchronize()`,
   `is_complete()`), but with **one stream per device** (`device.rs:146`) an event recorded *after* a
   node's launch signals exactly when that node (and everything before it on the stream) is done —
   which is all the executor needs, because the stream already serializes same-device deps. Vulkan has
   **one compute queue** (`lib.rs:286`) and submits whole batches; the batch `Fence` (`recorder.rs:224`)
   is the natural handle. vulkane *has* timeline semaphores (`sync.rs:120-130`) and
   `submit_with_sync(wait, signal)`, but we do **not** need GPU↔GPU device-side waits: CUDA↔Vulkan has
   no direct D2D path (it goes host-staged, two-hop, per A4c-prereq), so every cross-device handoff
   already passes through a host buffer where a host-side wait is correct and cheap relative to the PCIe
   copy. **Recommendation: per-node CUDA `Event` + per-batch Vulkan `Fence`, host-side waits only.
   Timeline semaphores are a later optimization (direct D2D), explicitly out of scope.**

Net: A4b is **moderate**, not large. The hard part is the eager-submit + defer-wait *policy* and its
race-safety proof, not new FFI.

---

## 1. Pending PRODUCTION per backend

### 1.1 The seam (unchanged from A1)

`execute_compiled` (`compiled.rs:143`) returns `Result<CompletionHandle>`; `CompletionHandle` is
`Ready | Pending(Box<dyn Completion>)` (`compiled.rs:170`); `Completion: Send` with
`wait(self: Box<Self>) -> Result<()>` (`compiled.rs:194`). The 5 producing call sites in the executor
(`pipelined.rs:3706` WriteSlice, `3803` WriteSliceRotating, `4139` Copy/Move, `4398` Kernel, `4484`
in-place) currently do `execute_compiled(...)?.wait()` inline. A4b stops the inline `.wait()` and
threads the returned handle into the executor's handle map (§2).

**Problem: `execute_compiled` cannot today see the backend to record a signal.** It calls
`(compiled.kernel)(...)` — a `KernelRef` fn pointer — then returns `Ready`. The kernel reaches the
device via the input `Storage`, not via an executor-held backend (`compiled.rs:150-155` comment). The
handle must therefore be produced **from the storage**, after the kernel returns, inside
`execute_compiled` (or a thin wrapper the executor calls). The output `Storage` carries the device
(CUDA: `CudaStorageBytes::device()` → `CudaDevice` with its `stream`; Vulkan:
`VulkanStorageBytes::backend()` → `Arc<VulkanBackend>` with its `recorder`+`queue`). So:

```text
execute_compiled(compiled, inputs, outputs, layouts):
    (compiled.kernel)(inputs, outputs, layouts, &params)?   // enqueues; does NOT sync (A2/A3)
    handle = produce_pending(compiled.backend, outputs)?     // NEW
    Ok(handle)
```

`produce_pending` matches on `compiled.backend` (already on `CompiledNode`, `compiled.rs:54`) and
inspects `outputs[0]`'s `BackendStorage` to reach the device handle. CPU → `Ready`.

### 1.2 CUDA — a recorded `Event`

After the kernel's launch is enqueued on the device stream, record an event on that same stream:

```rust
// in produce_pending, CUDA arm (outputs[0] is BackendStorage::Cuda(s))
let dev: &CudaDevice = s.device();
let ev = baracuda_driver::Event::no_timing(dev.context_ref())?;  // DISABLE_TIMING (sync-only)
ev.record(dev.stream())?;                                        // signals after all prior stream work
Ok(CompletionHandle::Pending(Box::new(CudaCompletion { ev })))
```

`Event` is `Clone + Send + Sync` (Arc-inner, `event.rs:14-26`), created against the device `Context`
(`Event::no_timing(ctx)`, `event.rs:49`; reachable via `CudaDevice::context_ref()`, `device.rs:433`),
recorded on `&Stream` (reachable via `CudaDevice::stream()`, `device.rs:424`).

```rust
struct CudaCompletion { ev: baracuda_driver::Event }
impl Completion for CudaCompletion {
    fn wait(self: Box<Self>) -> Result<()> { self.ev.synchronize().map_err(...) }  // event.rs:85 cuEventSynchronize
}
```

Non-blocking poll (for the in-flight counter / opportunistic drain) is `ev.is_complete()`
(`event.rs:92`, cuEventQuery → `Ok(true)`/`Ok(false)`). **A stream-level handle is sufficient given one
stream per device** — the recorded event is a marker on the single stream; waiting it waits exactly the
node's completion plus its (already-ordered) predecessors. We do **not** need a per-op event for
correctness; we record one per node only because that is the unit the executor tracks (and it is cheap:
`cuEventRecord` is a stream marker, no allocation beyond the small `Event`). Note `Event::new`/
`with_flags` call `context.set_current()` (`event.rs:55`) — single-threaded executor on a thread that
has touched this context, so this is a no-op push; multi-GPU future work must ensure event creation
happens on a thread bound to the right context (it does, since we create from the output storage's own
device).

### 1.3 Vulkan — the batch `Fence`, produced by SUBMITTING

A2's recorder *records* into a batch and `force_flush` does submit+wait atomically
(`recorder.rs:204-238`: `queue.submit(&[&cmd], Some(&fence)); fence.wait(u64::MAX)`). To produce a
`Pending(fence)` we must **submit without waiting**. This is the one new primitive on the Vulkan side.

Add to `Recorder` a `flush_batch_async` that ends recording, submits with a fresh fence, and **returns
the fence + the retired transient resources** instead of waiting:

```rust
// recorder.rs — sibling of flush_batch, returns the fence instead of waiting
pub fn submit_batch(&mut self, device, queue, queue_family)
    -> Result<Option<SubmittedBatch>>
{
    let Some(cmd) = self.batch_cb.take() else { return Ok(None) };   // empty batch → None
    // ... vkEndCommandBuffer (same as flush_batch) ...
    let fence = Fence::new(device)?;
    queue.submit(&[&cmd], Some(&fence))?;                            // ASYNC — returns immediately
    // Move (not drop) the transient/desc/CB resources into the returned struct:
    Ok(Some(SubmittedBatch {
        fence,
        cmd,                                  // keep CB alive until fence signals
        transients: std::mem::take(&mut self.batch_transients),
        descs:      std::mem::take(&mut self.batch_descs),
        retired_pool: std::mem::replace(&mut self.pool, CommandPool::new(device, queue_family)?),
    }));
    // reset counters/dirty exactly as flush_batch does
}
```

`SubmittedBatch` owns everything the in-flight CB references (CB, descriptor sets, param/uniform
transient buffers, the retired pool). Its `Drop` is empty; resources free when the struct drops —
which the executor does only *after* `fence.wait()`. The Vulkan `Completion`:

```rust
struct VulkanCompletion { backend: Arc<VulkanBackend>, batch: SubmittedBatch }
impl Completion for VulkanCompletion {
    fn wait(self: Box<Self>) -> Result<()> {
        self.batch.fence.wait(u64::MAX).map_err(...)?;     // sync.rs:80
        self.backend.retire_pools_post_drain_pub();        // mirror force_flush's pipelines.retire_pools_post_drain()
        Ok(())  // self.batch drops here → CB/descs/transients/pool free, post-fence (safe)
    }
}
```

The backend grows a `pub fn submit_pending(&self) -> Result<Option<SubmittedBatch>>` that locks the
recorder and calls `submit_batch` (parallel to `force_flush` at `lib.rs:1498`, which stays as the
blocking variant for the destructive/host-read paths that want submit+wait in one call).

**Important Vulkan subtlety — the fence is per-BATCH, not per-node.** A single `submit_pending` flushes
*all* recorded ops since the last submit (could be several nodes from the same Vulkan sub-DAG). So the
executor does **not** get one Vulkan handle per node; it gets one handle per *submission*, and that
handle covers a contiguous run of Vulkan nodes. This is fine and even desirable (fewer fences, larger
batches = the A2 batching win preserved). The executor maps **every Vulkan node in the just-submitted
batch to the same shared `Pending` handle** (Arc the `SubmittedBatch`, or — simpler — key the handle by
"the most recent Vulkan submission" and have all those nodes point at it). See §2.

**No `vkGetFenceStatus` wrapper exists in vulkane** (only `Fence::wait`, `sync.rs:80`). Non-blocking
poll = `fence.wait(0)` (zero timeout → `VK_TIMEOUT`/`VK_SUCCESS`); the `check()` maps `VK_TIMEOUT` to an
error today, so the in-flight counter's Vulkan poll either (a) treats any non-`SUCCESS` as "still
pending" by catching the timeout, or (b) asks vulkane to add a `Fence::status()` wrapper (a small
sibling ask, deferrable — B1 can ship CUDA-only poll first). Correctness never depends on the poll;
only the load *signal* does.

---

## 2. Executor handle tracking

**Where:** `realize_inner` (`pipelined.rs:~660-748`) and `realize_many_inner` (`~872-995`) own the
single executor loop `for item in rx { ... }`. They already hold `cache: StorageCache`
(`HashMap<NodeId, Arc<RwLock<Storage>>>`) and `layout_cache`. Add a third executor-local map:

```rust
let mut handles: HashMap<NodeId, InFlight> = HashMap::new();
enum InFlight {
    Cuda(CudaCompletion),                      // per-node event
    Vulkan(Arc<VulkanCompletion>),             // shared per-batch fence (Arc: many nodes → one batch)
}
```

CUDA handles are 1:1 (one event per node). Vulkan handles are N:1 (every node in a submitted batch
shares one `Arc<VulkanCompletion>`); the executor tracks the "current open Vulkan batch's node list"
and, on submit, installs the same `Arc` for all of them. (Simplest impl: a `Vec<NodeId>`
`open_vulkan_nodes` accumulated as Vulkan kernels dispatch; on submit, drain it into `handles` all
pointing at the new `Arc<VulkanCompletion>`.)

**Lifetime / drop:** a handle is dropped the moment its `wait()` is called (it is consumed by value;
`Completion::wait(self: Box<Self>)`). After waiting node N's handle the executor `handles.remove(&N)`.
For the Vulkan shared-Arc case, `wait` runs once (when the last reader forces it) and the remaining map
entries become "already-signalled" — model this by removing all of a batch's NodeIds together on the
first wait, or by an `Option<...>` that becomes `None` post-wait so repeat lookups are no-ops. Realize-
end drains everything still in `handles` (§3, the realize-end rule).

**The handle map is purely executor-local** (never crosses the realize boundary), matching A1's
"handle originates at `execute_compiled`'s return" and the constitution's "backends advertise, the
executor decides" — the executor tracks opaque `Completion`s and decides when to wait.

---

## 3. DEFER-WAIT POLICY (the crux)

Rule, per boundary, with the safety reason. Let P = a producer node with handle H(P); C = a consumer
about to read P's output buffer.

| Boundary | Wait H(P)? | Why |
|---|---|---|
| **Same-device consumer** (C and P on the same backend) | **NO** | One stream/queue per device serializes submission order = execution order. P was submitted before C (topo order, single executor thread); the device runs them in order. The CUDA stream (`device.rs:146`) and the Vulkan compute queue (`lib.rs:286`) both guarantee this. Inserting a host wait here would needlessly drain the pipeline — this is exactly the A2/A3 intra-device win. |
| **Cross-device consumer via `Op::Copy`/`Op::Move`** (residency boundary) | **YES — wait H(P) (the source producer) before the D2H read; then wait the H2D's own handle is unnecessary on the source side** | The Copy kernel's D2H (`copy_from_cuda_wrapper` → `to_cpu_bytes`, `byte_storage.rs:332`) reads P's *device* buffer on the host. If P is still in flight, the read races. Today `to_cpu_bytes` calls `device.synchronize()` (`byte_storage.rs:341`) — a **whole-stream** drain. A4b replaces that implicit full-device drain with **waiting H(P) specifically** before the copy: finer-grained (only P's completion, not the whole device's other in-flight work for the *other* sub-DAG sharing... — see note). Concretely: the executor, before dispatching a `Copy`/`Move` WorkItem, waits the handle of its single input (`item.inputs[0]`). |
| **Host-read** (`to_cpu_bytes` at realize root / explicit) | **YES — wait the source node's handle** | Same race as above; the realize root D2H is itself an `Op::Copy{target:Cpu}` spliced by residency (`optimize.rs:319`), so it is covered by the cross-device rule. The realize-end barrier (below) is the backstop. |
| **Destructive eviction** (`cache.remove`, `pipelined.rs:709`/`951`) | **YES — wait H(evicted) AND any handle whose buffer might be the same allocation** | Freeing a buffer a still-in-flight kernel reads/writes is UAF. For CUDA this is already covered by A3 stream-ordered free (the free is enqueued *after* the consuming kernel on the same stream — `device.rs:177-182`), so CUDA eviction needs **no** host wait. For Vulkan, A2 force-flushes (drains) before eviction (`force_flush_vulkan`, `pipelined.rs:717`); A4b keeps that as **wait H(evicted)** if a handle exists, else the existing `force_flush` (submit+wait). |
| **Realize-end** (before `cache.remove(&target)` / results return, `pipelined.rs:732`/`970`) | **YES — wait ALL handles still in `handles`** | The returned `Storage` must be fully computed before the caller (or a host read) touches it, and the cache is about to drop, freeing every intermediate. Drain every outstanding handle, then clear the map. This subsumes A2's `force_flush_all_vulkan`. |

**The finer-than-today claim, stated precisely.** Today `to_cpu_bytes`'s `device.synchronize()` drains
the *entire* source stream. With one stream per device and the CUDA sub-DAG being independent of the
Vulkan one, that drain blocks the host until *all* CUDA work is done — including CUDA work that the
*other* output does not depend on. Waiting H(P) (a single recorded event) instead lets the host proceed
as soon as P specifically is done. In practice, for the cross-device *reconverge* this is a wash (the
copy's result is needed immediately), but it matters for the **A2/A3 interaction** (§5): we want the
Vulkan submission to have happened *before* we block on the CUDA copy, so both run while the host waits
on one of them.

**Why same-device needs no wait — the load-bearing invariant.** Submission order on a single
stream/queue equals execution order, and the single-threaded executor submits in topological order. So
for any same-device edge P→C, P is submitted before C and the device will not start C's kernel until
P's has retired (CUDA stream ordering; Vulkan: same queue + the recorder's dependency-aware barrier,
`recorder.rs:138`, inserts a `vkCmdPipelineBarrier` exactly when C reads a buffer P wrote). This is the
property A2/A3 already rely on and verified live; A4b does not weaken it.

---

## 4. RACE-SAFETY argument

Claim: **no buffer is read or freed before its producer's handle has signalled.**

Define: every buffer B in the cache was produced by exactly one node P(B) (the node at whose slot it was
`cache.insert`ed), with handle H(P(B)) in `handles` until waited. A buffer is *consumed* either (a) as a
same-device kernel input, (b) as a cross-device/host copy source, or *freed* either (c) by destructive
eviction or (d) by realize-end cache drop.

- **(a) same-device read.** P(B) and the consumer are on the same single stream/queue; submission order
  = execution order ⇒ the device executes the read after the write. No host wait needed, no race. ∎
- **(b) cross-device/host copy source.** Policy waits H(P(B)) before dispatching the copy (§3 row 2/3).
  `Completion::wait` blocks until the device signal fires (CUDA: `cuEventSynchronize`, fires after the
  whole stream up to and including P(B); Vulkan: `vkWaitForFences`, fires after the whole submitted
  batch containing P(B)). So B is fully written before the D2H reads it. ∎
- **(c) destructive eviction.** CUDA: the free is stream-ordered (`zeros_async`/`new_async` Drop →
  `cuMemFreeAsync` on the origin stream, `device.rs:170-182`), enqueued after every consuming kernel on
  that stream ⇒ the free executes last; no host wait needed. Vulkan: policy waits H(evicted) (or
  `force_flush`) before `cache.remove`; the batch holding any command that references the buffer has
  signalled, so the GPU is done with it before the host frees it. ∎
- **(d) realize-end.** Policy drains all handles before the cache drops ⇒ every device buffer is idle
  before free. ∎

**Subtle cases:**

1. **CSE'd copies / the two-hop CPU intermediate.** A4c-prereq shares one CPU intermediate node for a
   CUDA↔Vulkan edge (Vulkan→CPU→CUDA; `cuda_vulkan_multidevice_realize_live.rs:202-257`). The first hop
   (Vulkan→CPU) is an `Op::Copy{target:Cpu}` consuming the Vulkan producer → covered by (b): wait the
   Vulkan producer's batch fence before the D2H. Its *output* is a CPU buffer (synchronous, `Ready`
   handle). The second hop (CPU→CUDA H2D) consumes that CPU buffer (already on host, no wait) and
   produces a CUDA buffer; the H2D `write_from_host` syncs internally (`byte_storage.rs:287`) so the
   CUDA buffer is ready when its (CUDA) consumer reads it same-device (a). The CSE means the CPU
   intermediate has multiple readers, but they are all host-side reads of a host buffer → no device
   race. ∎ The single risk is if we *removed* the H2D's internal sync in a later optimization without
   adding a handle wait — out of scope; A4b keeps the host-boundary syncs (mirrors A3's "KEPT
   byte_storage's 5 D2H/H2D syncs").
2. **In-place ops** (`WriteSlice` 3706, `WriteSliceRotating` 3803, in-place kernels 4484). These adopt
   the destination buffer Arc at the node's slot and mutate it in place. The producer of the
   *destination* is a prior node; the in-place op is a same-device consumer of it ⇒ (a) covers
   ordering. The in-place op's *own* handle then represents the mutated buffer; downstream same-device
   readers are ordered after it on the stream. The destructive eviction of the *source* slot (the old
   NodeId) is (c). The only hazard would be a cross-device reader of an in-place result — that goes
   through `Op::Copy` ⇒ (b). Safety-copy insertion (`insert_safety_copies`, `pipelined.rs:890`) already
   breaks residual cycles before this, so an in-place op never aliases a buffer with a *live concurrent*
   other reader on another device without a Copy between them. ∎
3. **Output buffer reuse before producing kernel signals** (open question 3 from the Phase-A sketch).
   The executor never reuses an output buffer: each kernel `alloc`s a fresh output on its backend
   (`pipelined.rs:4301-4378`), inserted at a new NodeId. The mem-pool *recycles freed blocks* (CUDA,
   `device.rs:454-455` release threshold `u64::MAX`), but a block only returns to the pool on Drop,
   which for an in-flight buffer is stream-ordered ⇒ it cannot be handed out to a new `alloc` until the
   prior kernel retired. So no reuse race. ∎

**Failure propagation.** An async kernel that faults surfaces at the next `wait()` of a handle on that
stream/device (CUDA: `cuEventSynchronize` returns the sticky error; Vulkan: `fence.wait` /
`vkQueueSubmit` returns `VK_ERROR_DEVICE_LOST`). The executor `?`-propagates it out of the realize loop
exactly as a synchronous kernel error today; the realize-end drain is the latest it can surface. The
`TopologyChanged` retry (`pipelined.rs:699`) is unaffected — it fires *before* dispatch at a chunk
boundary, never mid-flight.

---

## 5. A2/A3 INTERACTION — the overlap enabler

**This is where concurrency is actually won or lost.** The three pieces:

- **CUDA (A3): submits on launch.** Each kernel enqueues on the device's single stream and returns; no
  per-op sync (the 59 per-op `synchronize()` were removed). So a CUDA sub-DAG is *streaming* onto the
  GPU as the executor walks it. Good — nothing to change for CUDA submit timing.
- **Vulkan (A2): defers submission.** Kernels `record` into the batch; nothing reaches the GPU until
  `force_flush` (host-read / eviction / realize-end). **This is the overlap killer:** in the mixed
  graph, the executor walks (say) the CUDA sub-DAG, then the Vulkan sub-DAG, then the reconverge. The
  Vulkan ops only *record*; the GPU sits idle on them until the realize-end flush — by which point the
  CUDA work is long done. **They never overlap.**

**The A4b change: eager Vulkan submission at sub-DAG / dependency boundaries.** The executor must
`submit_pending` the Vulkan backend at the points where (a) a sizable independent batch has accumulated
and (b) we are about to block the host on *another* device — so the Vulkan GPU is busy while the host
waits on CUDA (and vice-versa). Concretely, submit Vulkan's open batch:

1. **Before waiting any CUDA handle** (the cross-device copy boundary, §3 row 2, and realize-end). This
   is the key one: if the reconverge needs the CUDA copy of the Vulkan result, we (i) `submit_pending`
   Vulkan → its batch starts on the AMD iGPU, (ii) wait the Vulkan batch fence (it is now running, not
   merely recorded), (iii) do the Vulkan→CPU copy. Meanwhile any independent CUDA work submitted earlier
   ran concurrently. For two *independent* outputs (no reconverge), the realize-end drain submits Vulkan
   then waits both — but to get overlap *during* the walk we also submit at (2).
2. **At a backend-switch boundary in the dispatch order** — when the executor transitions from emitting
   Vulkan nodes to emitting CUDA nodes (the dispatch order is chunked by backend; see the
   `current_chunk_backend` tracking, `pipelined.rs:691-706`). Submitting Vulkan's batch when we leave a
   Vulkan chunk means the iGPU starts that chunk while the executor records/submits the CUDA chunk onto
   the NVIDIA stream → genuine concurrency. **This reuses the existing chunk-boundary hook** (which today
   only does the `TopologyChanged` generation check) — add a Vulkan `submit_pending` there, gated on
   "we are leaving a Vulkan chunk."
3. **At the existing TDR cap** (`should_flush`/`BATCH_LIMIT=500`, `recorder.rs:199`) — unchanged, but
   now `submit_pending` (async) rather than `force_flush` (sync) where the executor can defer the wait.

**Critical: this must NOT regress single-device.** For a pure-Vulkan realize there is no backend switch
and no cross-device copy, so boundaries (1) and (2) never fire mid-walk; the batch accumulates exactly
as in A2 and flushes at realize-end. **Byte-identical to A2, same batching, same throughput.** For
pure-CUDA, nothing changes (CUDA already submits on launch; the handle is recorded but waited only at
realize-end / host read — equivalent to today's stream sync at the boundary). The eager-submit logic is
**only reachable on a multi-backend graph** (a backend switch or a cross-vendor copy must exist), which
is exactly the A4b target and never the single-device path.

**Composition with the per-device guards.** A2's `force_flush_vulkan` (eviction) and
`force_flush_all_vulkan` (realize-end) become: prefer `wait H(node)` when a handle exists; the
realize-end `force_flush_all_vulkan` is replaced by "drain all handles" (which includes submitting any
still-open Vulkan batch then waiting it). A3's stream-ordered free and per-device stream are untouched —
they already don't cross-stall the other device (draining the CUDA stream via an event wait does not
touch the Vulkan queue and vice-versa), which is what preserves concurrency.

**Do we need timeline semaphores?** No. They would let the *GPU* wait on another submission's
completion device-side (no host round-trip) — valuable only for a direct CUDA↔Vulkan D2D path, which
does not exist (A4c-prereq is host-staged two-hop). Every cross-device handoff passes through a CPU
buffer, where a host-side `wait` is both correct and not on the critical path (the PCIe copy dominates).
Revisit timeline semaphores (`sync.rs`, `submit_with_sync`) only if/when a direct D2D copy lands.

---

## 6. TESTING without a sanitizer

compute-sanitizer is not installed. Confidence comes from layered behavioral evidence:

1. **Regression — existing live suites stay green, byte-exact.** `cuda_async_realize_live` (8/8),
   `vulkan_bridge_realize_live`, `cuda_multidevice_realize_live`, and especially
   `cuda_vulkan_multidevice_realize_live` (`[22,86,192,340]`) must stay byte-identical. These exercise
   the cross-device copy, two-hop bridge, in-place WriteSliceRotating, and deep fan-out chains — the
   exact paths the defer-wait policy touches. A wrong wait point shows up as a wrong or nondeterministic
   byte.
2. **Stress — deep cross-device chains × many iterations.** New `#[ignore]` live test: a graph with a
   long alternating CUDA→(copy)→Vulkan→(copy)→CUDA chain (10–20 hops), realized N×100 times in a loop,
   asserting byte-exact every iteration. A missing wait manifests probabilistically as a stale-read on
   some iterations; 100× amplifies a 1% race to ~63% detection per run, and the loop runs cheaply.
   Run one suite at a time (12 GB GPU).
3. **Determinism.** Realize the same mixed graph 1000× and assert all outputs identical. Non-
   determinism = a race (a read landing before/after a write depending on timing).
4. **Deliberate delay/reorder injection (the strongest cheap signal).** Behind a test-only cfg, insert
   an artificial device-side delay on the *producer* of a cross-device edge (CUDA: a long dummy kernel
   on the stream before the event record; Vulkan: a spin in a dummy dispatch before submit). If the
   defer-wait policy is correct the result is unchanged; if a wait is missing, the consumer reads stale
   data and the byte assert fails *deterministically*. This converts a latent race into a hard failure
   — it is the sanitizer substitute. Also inject the inverse: artificially *reorder* independent
   sub-DAG submission and confirm the result is invariant (proves the sub-DAGs really are independent).
5. **Concurrency proof — the A4c benchmark.** Two heavy independent sub-DAGs (sizable matmuls), one per
   device, `std::time::Instant` around realize; assert wall-clock materially `< sum` of the two
   sequential single-device realizes. This is the positive proof that overlap *happens* (not just that
   it is safe). If it does not overlap, a submit-timing boundary (§5) is missing. Pair it with a
   per-device "first-submit timestamp" log to confirm the Vulkan batch is submitted *before* the CUDA
   drain (the §5.1 ordering).
6. **In-flight counter sanity (B1 dependency).** Assert the counter is non-zero on both devices
   simultaneously at some point during the heavy mixed realize (sample from a host thread) — direct
   evidence of concurrent in-flight work.

The design is race-free *by construction* (single stream/queue order + waits at every cross-device /
free / realize-end boundary), so green stress + green delay-injection + the wall-clock proof is strong
evidence; a failure points at a specific missing boundary.

---

## 7. PR breakdown

Each PR is independently testable; the first lands single-device-safe.

- **A4b-1 — CUDA Pending production + executor handle map; defer-wait at realize-end only.**
  `produce_pending` CUDA arm (record `Event`), `CudaCompletion`, the `handles` map, and the realize-end
  "drain all handles" (replacing the implicit per-call wait for CUDA). Same-device + cross-device still
  wait *conservatively* (keep the inline `to_cpu_bytes` sync) — so this PR is **behavior-preserving**
  and single-device-safe: CUDA realizes still produce identical bytes; the only change is *where* the
  final sync happens. Gate: `cuda_async_realize_live` + `recip_abs` + `phase_c_rotating_kv` byte-exact.
- **A4b-2 — Vulkan Pending production (`submit_batch`/`SubmittedBatch`/`submit_pending` +
  `VulkanCompletion`); realize-end drains via handles.** No eager-submit-during-walk yet (realize-end
  only), so pure-Vulkan stays byte-identical to A2 (batch accumulates, submitted once at end).
  Gate: `vulkan_bridge_realize_live` byte-exact, including the deep fan-out chain.
- **A4b-3 — finer cross-device wait.** Replace the cross-device `Op::Copy` source-drain
  (`to_cpu_bytes`'s full `synchronize`) with `wait H(producer)` before the copy WorkItem; same for the
  Vulkan eviction guard (`wait H(evicted)` instead of `force_flush`). Still single-device-identical
  (no cross-device edge on single-device graphs). Gate: `cuda_vulkan_multidevice_realize_live` +
  `residency_eviction_live` byte-exact.
- **A4b-4 — eager Vulkan submission at backend-switch + pre-CUDA-wait boundaries (the overlap
  enabler, §5).** Add `submit_pending` at the chunk-boundary hook (leaving a Vulkan chunk) and before
  waiting a CUDA handle at a cross-device copy. **Guarded to fire only on multi-backend graphs** →
  single-device byte-identical and throughput-neutral. Gate: stress + determinism + delay-injection
  tests (§6.2–6.4) all green and byte-exact.
- **A4b-5 — the concurrency benchmark (A4c).** Dual-GPU wall-clock < sum-of-sequential, with
  first-submit-timestamp + simultaneous-in-flight assertions (§6.5–6.6). This is the proof A4b
  delivered concurrency, and the gate for **C** (`DeviceLoadSelector`).
- **(later) B1 — in-flight counter** via `Event::is_complete` / `Fence::status` poll; rides on the
  handle map. **(later, optional) timeline-semaphore D2D** — out of scope unless a direct CUDA↔Vulkan
  copy lands.

---

## 8. Open questions (for CireSnave / review before implementation)

1. **Vulkan handle granularity.** Per-batch fence shared by N nodes (proposed) vs forcing one-node
   batches to get 1:1 handles. The shared-batch keeps A2's batching win but makes "wait node N" wait the
   whole batch N is in. Is the coarser wait acceptable? (I believe yes — same-device deps don't wait at
   all, and cross-device waits want the whole batch done anyway.)
2. **`Fence::status()` ask to vulkane.** B1's Vulkan load poll wants a non-blocking fence query; vulkane
   exposes only `Fence::wait` (no `vkGetFenceStatus` wrapper). Add `Fence::status()` to vulkane (small,
   in-house — vulkane is ours), or ship B1 CUDA-only first and treat Vulkan load as "batch open = busy"?
3. **Eager-submit aggressiveness (§5.2).** Submit Vulkan at *every* backend-switch, or only when the
   open batch exceeds a size threshold (avoid tiny submissions that lose the batching win)? Needs the
   A4b-5 benchmark to tune; propose a threshold (e.g. submit if `batch_count >= 16` OR we are about to
   wait a foreign-device handle).
4. **CUDA event cost at scale.** One `Event` per node (`cuEventRecord` + a small `Event` alloc) across
   thousands of nodes in a transformer — is the overhead negligible vs the removed per-op syncs? (Almost
   certainly yes; `cuEventRecord` is a stream marker. But measure on the long-chain stress.) Alternative:
   record an event only on nodes that are cross-device-copy *sources* or realize roots (the only nodes
   actually waited mid-walk), and leave same-device-only nodes handle-less (`Ready`-equivalent). This is
   a strict optimization of the proposed design and may be the better default — **recommend evaluating
   "event only where waited" in A4b-1.**

   **RESOLVED (2026-07-08) — "event only where waited" implemented.** For the materialized order
   sources (`OrderSource::Default` / `Optimized`) the executor pre-scans the frozen dispatch order for
   `Op::Copy`/`Op::Move` producers (`pipelined::build_wait_set` — exactly the set
   `wait_producer_handle` can ever consult mid-walk) and passes a per-node `will_be_waited` bit into
   `compiled::execute_compiled_with_wait_hint`; a CUDA node NOT in the set returns
   `CompletionHandle::Ready` with no `cuEventCreate`/`cuEventRecord`/`cuEventSynchronize`/
   `cuEventDestroy` and no B1 in-flight inc/dec (the counter now tracks only waited CUDA work on these
   sources; `Streaming` — the only source the load-aware picker reads the counter through — never
   elides). The realize-end `drain_handles` (which now holds only unconsumed wait-set handles) is
   complemented by ONE `cuStreamSynchronize` per CUDA device still holding live cache storage
   (`pipelined::sync_active_cuda_devices`) — sufficient because one-stream-per-device order makes every
   elided same-device ancestor complete no later than the live node's own kernel, and evicted
   intermediates are A3 stream-ordered-free safe. `OrderSource::Streaming` (lazy order — cannot be
   pre-scanned without defeating streaming) falls back to event-every-node, byte-identical to the prior
   behavior. Correctness belt-and-suspenders: cross-device D2H stays race-free independent of any event
   via `to_cpu_bytes_finer`'s legacy-default-stream ordering (`cuMemcpyDtoH_v2` vs `CU_STREAM_DEFAULT`).
5. **Multi-GPU CUDA (future).** `find_cuda_device_in_cache` matches the first CUDA storage regardless of
   `gpu_id` (`pipelined.rs:4522` comment "single-GPU setups always match"). The per-node event is
   recorded on the *output storage's own* device/stream, so it is correct for multi-GPU; but the
   in-flight counter keyed by `DeviceLocation` must distinguish gpu_ids. Out of scope for A4b (one CUDA +
   one Vulkan), flagging for B1/C.
6. **`#[must_use]` enforcement.** The `CompletionHandle` is `#[must_use]` (`compiled.rs:169`). Once the
   executor stores handles instead of waiting inline, the compiler can no longer catch a dropped handle
   at the `execute_compiled` call site (it is moved into the map). Add a debug-assert at realize-end that
   `handles` is empty after the drain, so a leaked (never-waited) handle is caught in tests.
