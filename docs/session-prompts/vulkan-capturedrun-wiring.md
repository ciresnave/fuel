# Vulkan CapturedRun executor wiring — design + increment plan

**Status (2026-07-10):** investigation complete; implementation not started. Parallel track to the
CUDA CapturedRun build-out (Increments 1–4a on `main` @ `e1117b54`), chosen because CUDA's
Increment 4b is **blocked on external Baracuda kernels** (capture-safe GEMV + gather) while Vulkan's
kernels are **Fuel-internal Slang** (`fuel-vulkan-kernels`) — no external blocker. Goal: bring
Vulkan to true parity on the capture-replay decode-latency lever, and possibly to a *working*
full-decode capture ahead of CUDA.

> ⚠️ **AMENDED 2026-08-19 — HALF OF THIS STATUS IS WRONG, AND ONLY HALF.**
> "Implementation not started" is false: **`fuel-vulkan-backend/src/capture.rs`
> exists** — 231 lines, 5 public items, `pub struct CapturedRun` with `replay`
> and `rebind`, labelled in its own module docs as *Phase C PR-C2b*.
>
> **The WIRING claim, which is what this document is actually about, appears to
> survive — but I could not measure it cleanly, and say so rather than ship a
> tidy number.** Both backends define `pub struct CapturedRun` and both
> re-export it as `pub use capture::CapturedRun`, and **no file imports it by
> `use` at all**, so a name-based query cannot separate Vulkan's consumers from
> CUDA's. A first attempt returned "0 outside consumers"; the same query shape
> returned **31** when widened, and its control (`fuel_cuda_backend::capture`)
> **also** returned 0 — so the zero was uninformative, not evidence.
>
> What IS measurable is the asymmetry: files referencing `CapturedRun` are
> **fuel-cuda-backend 15, fuel-core 7, fuel-dispatch 6, fuel-vulkan-backend 2**
> (its own module plus `lib.rs`). Vulkan's capture is implemented and thin on
> consumers; **"not wired" is consistent with that and not established by it.**
>
> Correcting only the demonstrably false half deliberately: replacing "not
> started" with "complete" would be a second wrong status, and the more durable
> kind, because nobody re-checks a completion.
>
> **Re-derive:** `git show HEAD:fuel-vulkan-backend/src/capture.rs | wc -l` → ~231;
> `git grep -l CapturedRun -- '*.rs' | sed 's|/src/.*||;s|/tests/.*||' | sort | uniq -c | sort -rn`
> → the per-crate distribution above.
> **Do not** use `git grep -l CapturedRun | grep -v fuel-vulkan-backend` as a
> consumer count — it counts CUDA's type too.

## Where parity stands (verified)

- **Persistent decode** (plan-once, re-bind per token): AT PARITY. `bench_persistent_decode_real_model_{cpu,vulkan,cuda,cuda_bf16}` all exist + pass (`fuel-core/src/lazy.rs:12208+`).
- **Capture primitive**: AT PARITY. Vulkan has `CapturedRun` (`fuel-vulkan-backend/src/capture.rs`, "Phase C PR-C2b"), the analog of CUDA's — a re-submittable `VkCommandBuffer`, `replay()` = re-submit, `rebind()` = re-record. GPU-tested (`capture_replay_rebind_affine`). CPU: N/A (no launch overhead to amortize).
- **Capture executor wiring** (`capture_decode` + persistent-output mode + `CapturedDecodeSession`): **CUDA-ONLY** — this session's work. Vulkan lacks it. THIS is the gap.

## The architectural difference from CUDA (the crux)

CUDA capture is **transparent**: kernels launch on `device.stream()`, which `capture_run` puts into
capture mode, so `execute_work_item`'s kernels are recorded automatically. The persistent-output
executor mode + `capture_decode` just run the dispatch loop inside `CudaDevice::capture_run` and it
"just works."

Vulkan is **not** transparent. `VulkanBackend::capture_run(|rec: &mut CommandBufferRecording| ...)`
hands you an *explicit* recorder that dispatches must be recorded ONTO. But the executor's Vulkan
compute ops record into the backend's *internal* `recorder: Mutex<Recorder>` batch
(`record_dispatch_batched` → `Recorder::record_batch_dispatch`, `fuel-vulkan-backend/src/lib.rs:1444`),
NOT a caller-provided recorder. So you cannot just run the executor loop inside `capture_run`.

**Key enabling fact:** the backend's batch CB is ALREADY reusable — vulkane's `begin()` uses default
flags (NOT `ONE_TIME_SUBMIT`), per `capture.rs:6-9`. So the executor's normal Vulkan realize already
builds a re-submittable command buffer. **Capture = retain that batch CB (+ its transients +
descriptor sets + the persistent I/O buffers) instead of `submit_batch`-ing it.**

## Recommended approach: "retain the batch" (reuses the executor's recording path)

The `Recorder` (`fuel-vulkan-backend/src/recorder.rs:213`) holds `batch_cb: Option<CommandBuffer>`,
`batch_transients: Vec<(Buffer, Allocation)>`, and per-dispatch descriptor sets. `submit_batch`
(`:407`) ends the CB, submits it, and MOVES the resources into a `SubmittedBatch` that frees them
post-fence. For capture we want to **end the CB and MOVE the resources into a retained `CapturedRun`
WITHOUT submitting** — then replay = re-submit that CB.

### Increments

1. **Recorder capture-mode + extract.** Add a `capture` flag to `Recorder` that makes `should_flush()`
   return `false` (whole run in ONE batch — no mid-run auto-flush), and a
   `end_and_take_batch() -> Result<(CommandBuffer, Vec<(Buffer,Allocation)>, <descs>)>` that ends the
   CB recording and moves out the batch resources without submitting. `VulkanBackend::capture_from_batch(record_closure)`
   drives it. Minimal backend-level test: record 2 affine dispatches through the batch path, extract,
   replay, verify bit-exact. *(self-contained; proves the retain-the-batch mechanism)*
2. **Extend the Vulkan `CapturedRun`** (or a sibling) to own `(CommandBuffer, pool, transients, descs,
   retained I/O buffers)` so re-submit is valid + all referenced resources stay alive (the capture.rs
   contract: keep pipeline + descriptor sets + buffers alive for the CapturedRun's life).
3. **`capture_decode` Vulkan arm.** Generalize the executor's persistent-output capture (the
   `PersistentOutputs` Record/Reuse mode in `pipelined.rs` is ALREADY backend-agnostic; only
   `capture_decode` is CUDA-gated). Add a Vulkan path: warm-record (populate persistent buffers) →
   run the reuse pass with the backend `Recorder` in capture mode → `end_and_take_batch` → a
   `VulkanCapturedDecode`. A Vulkan `op_kind_is_capture_writeinto` predicate (Vulkan kernels write
   into descriptor-bound output buffers — **write-into by nature, so likely NO per-kernel `_into`
   refactor needed**, unlike CUDA's ~30 variants; verify per family).
4. **`CapturedDecodeSession` Vulkan analog** (or generalize the existing one over a backend trait):
   fixed per-token input buffers + `replay_token` (H2D-update in place → `CapturedRun::replay` →
   read output). GPU test: multi-token synthetic decode-shaped session, bit-exact per token.
5. **Full-decode capture attempt** — because there's no Baracuda blocker, wire `fuel-core`'s Vulkan
   `DecodeSession` to the Vulkan capture session and try a REAL decode step end-to-end. This is where
   Vulkan could get *ahead* of CUDA. Verify each Vulkan compute dispatch replays correctly under
   re-submit (the analog of the CUDA index_select/gemm capture bugs — but Vulkan kernels are simpler
   Fuel-internal Slang; the affine primitive already replays clean).

## Risks / open questions

- **Descriptor-set lifetime under re-submit.** The batch's descriptor sets point at the persistent
  I/O buffers; they + the buffers must outlive every replay. The `retain the batch` extract must move
  them into the CapturedRun (analogous to `SubmittedBatch` owning them until its fence).
- **Multi-batch runs.** If a decode run exceeds one batch even with flush disabled (unlikely for
  decode-sized graphs), capture must retain a *sequence* of CBs re-submitted in order.
- **Replay correctness per kernel** — the Vulkan analog of "does this op replay bit-exact?" Verify on
  a synthetic multi-family graph before claiming parity (learned twice on CUDA: cuBLAS + index_select).
- **Build/test:** `fuel-vulkan-backend` + `fuel-dispatch --features vulkan`; live tests need the Vulkan
  SDK + GPU (installed). One live-GPU suite at a time (don't run concurrently with a CUDA live suite).

## Why this is worth doing now

CUDA 4b waits on Baracuda (external, unbounded). Vulkan has no such blocker AND likely needs no
write-into kernel refactor (descriptor-bound outputs), so the wiring may be *lighter* than CUDA's and
reach a working full-decode capture first — turning "Vulkan lags on the executor-wiring lever" into
"Vulkan is the first backend with working decode capture." The persistent-output executor mode +
`CapturedDecodeSession` orchestration are already built + backend-agnostic; this is the Vulkan-side
capture-primitive integration + the retain-the-batch Recorder change.
