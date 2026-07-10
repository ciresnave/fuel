# CapturedRun executor build-out — design + increment plan

**Status (2026-07-10):** Phases 1, 2a, 4 landed on `main`. This doc scopes the remaining
persistent-output executor capability (the CapturedRun "capture wiring"), which the original banked
plan under-scoped. User directive: **full build-out**.

## The problem (adversarially verified, 3/3 probes)

CUDA-graph replay (`fuel-cuda-backend/src/capture.rs`, `CapturedRun::replay`) bakes buffer
**addresses** into the graph and replays against the *same* addresses with fresh contents. But:

- The pipelined executor's `WorkItemKind::Kernel` arm allocates a **fresh** output buffer for every
  compute node on every realize (`pipelined.rs:5485`, `CudaStorageBytes::alloc`; doc: "always
  allocates output").
- The kernel dispatch ABI is **allocate-and-return**: every fuel-cuda-backend kernel fn has shape
  `(src*, layout*) -> Result<CudaStorageBytes>` (allocates its output internally), and the dispatch
  wrappers do `*out = result` (replace the output storage's buffer with the fresh one). ~30 wrapper
  variants follow this pattern.

So a compute node's output device **address changes token-to-token** → a captured graph replays
against stale memory. This gap is acknowledged in `docs/architecture/14-lifecycle.md:33-38`
("wiring run-capture into the executor … arrives with Phase D's persistent graph").

## Why it's tractable (not a redesign, not a baracuda ask)

- The **baracuda FFI already takes an output pointer `y` for every op** (verified: UnaryContigRun,
  RmsNormRun, SoftmaxRun, gemm_dense, RopeRun, FlashDecodingRun all take `y: *mut c_void`). Writing
  into a caller-provided output is a **Fuel-side wrapper** change only.
- **Precedent exists in-tree**: `unary_inplace_run` (`elementwise.rs:176`) and
  `WorkItemKind::InplaceKernel` already pass a fixed buffer's pointer into a kernel; `flash_decoding`
  already reuses a persistent per-device workspace across calls (fixed-address reuse).
- Phase 1 (`Op::WriteSliceDoff`) already solved the *offset*-freezing sub-problem (device-resident
  KV-write start, updated per token by a fixed-address H2D memcpy — capture-tolerant).

## Architecture

A **persistent-output-buffer map** (`NodeId -> Arc<RwLock<Storage>>`) owned by the `DecodeSession`:

- **Warm run** (uncaptured): realize normally, recording each compute node's output Arc into the map
  (buffers `v1`, fixed addresses).
- **Capture run**: re-issue the decode launches through the executor in **persistent-output mode**
  (each node writes into its `v1` buffer via a **write-into-output** kernel path — no realloc, no
  sync), inside `CudaDevice::capture_run` → a `CapturedRun` stored on the session.
- **Replay** (token 3+): H2D-update the fixed per-token input buffers (token-ids/rope/mask/offset)
  in place, `CapturedRun::replay()`, then D2H the logits **outside** the capture.

### Executor changes (`fuel-dispatch/src/pipelined.rs`)
- `PersistentOutputs { map: HashMap<NodeId, Arc<RwLock<Storage>>> }`.
- Thread `Option<&mut PersistentOutputs>` through `realize_inner` → `execute_work_item` (None for all
  existing callers ⇒ unchanged behavior, zero regression).
- Kernel arm: if persistent-mode and the node's buffer is present, pass it to the kernel's
  **write-into** path; else allocate + record.
- A capture-aware realize entry that wraps the dispatch loop in `capture_run` and keeps the D2H
  `Op::Copy` outside the capture scope (no alloc/sync inside capture).

### Kernel changes (`fuel-cuda-backend` + `fuel-dispatch/src/baracuda_dispatch.rs`)
Write-into-output variants for the decode-critical kernel families (the FFI `y` pointer is already
there — the variant just skips the internal alloc and uses the provided buffer):
- binary elementwise (add/mul — residual + gate), unary (silu), matmul (gemm_dense),
  rmsnorm, softmax, rope, index_select (embed). (KV writes + in-place unary already write-into.)

### DecodeSession changes (`fuel-core`)
- Fixed per-token input buffers (token-ids/rope/mask/offset) allocated once, H2D-updated in place
  (currently fresh Arcs each token).
- Warm→capture→replay state machine + the `PersistentOutputs` map.

## Increment plan (each a verified commit; GPU-gated where noted)
1. **Foundation**: `PersistentOutputs` + executor persistent-output mode + capture entry + write-into
   for ONE family (binary elementwise). GPU test: capture a small elementwise chain through the
   executor, replay, bit-exact vs uncaptured. *(proves the whole mechanism)*
2. **Kernel coverage**: write-into variants for the remaining decode families (parallelizable).
3. **DecodeSession orchestration**: fixed input buffers + warm/capture/replay state machine.
4. **Validation**: bit-exact captured-vs-uncaptured full-decode test + re-bench on the stable
   protocol (median of >=8 same-phase runs, discard cold starts, log nvidia-smi).

## Invariants / risks
- No device alloc or sync inside a capture scope (capture.rs hard rule).
- `realize_inner` is shared by ALL callers — the persistent-output param MUST default to None with
  byte-identical behavior; gate every change behind it.
- Capture only pays off for f32 decode today (no flash arm). bf16/flash decode capture additionally
  needs a device-resident attended-length (`k_len`) carrier — out of scope here.
