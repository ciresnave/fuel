# Baracuda ask — stream-ordered allocation/free for async dispatch (Step E A3)

**From:** Fuel dispatch-core cleanup, Step E **Phase A3** (CUDA async dispatch).
**Status:** question — likely already mostly satisfied by `DeviceBuffer::new_async`; one free-side
detail to confirm, and possibly one small additive change.
**Nature:** allocation-lifetime semantics; no hot-path behavior change requested, just confirmation +
(maybe) a stream-ordered `Drop`.

## What Fuel is doing (the why)

Step E A3 makes CUDA dispatch **pipeline on the single per-device stream** instead of
`stream.synchronize()`-ing after every op. Same-stream submission order = execution order, so
producer→consumer deps need no inline sync — we only sync at host reads (D2H), and at realize-end.

The blocker is **buffer lifetime**. Each fuel-cuda-backend op today does
`launch(stream) → device.synchronize() → return`, and that per-op synchronize is what keeps the op's
buffers — the `Workspace`/scratch (`baracuda/scratch.rs` → `device.alloc_zeros`) **and** the output
buffer — alive until the kernel finishes. If we just drop the synchronize, those buffers free (Rust
`Drop`) while the kernel is still pending on the stream → **use-after-free**.

The clean fix is **stream-ordered allocation + free**: if a buffer allocated on the stream also
*frees* on the stream (`cudaFreeAsync`), the free is enqueued *after* the kernel that used it, so it's
safe-by-construction — no host sync, no Rust-side retention pool, and (if it covers data buffers too)
no executor-side lifetime guards. The CUDA driver's stream-ordered mem-pool also **reuses** freed
blocks across a chain, so peak VRAM ≈ one workspace and there's no repeated real `cuMemAlloc`.

## What we already see

`fuel-cuda-backend` already calls `DeviceBuffer::new_async(&ctx, len, &stream)` (device.rs:163) —
stream-ordered *allocation*. So the alloc half looks present.

## Questions

1. **Free semantics:** does a `DeviceBuffer` allocated via `new_async(ctx, len, stream)` free via
   **`cudaFreeAsync` (stream-ordered)** on `Drop`, or via synchronous `cuMemFree`? (Does the buffer
   retain its stream so `Drop` can enqueue the free on it?)
2. **Coverage:** do the *non*-`new_async` paths Fuel uses — `alloc_zeros` (output buffers) and any
   plain `DeviceBuffer::new` — also allocate/free stream-ordered, or only `new_async`? We'd want the
   **output/data buffers** stream-ordered too (so eviction is safe without executor guards), not just
   workspaces.
3. **Mem pool:** is the stream-ordered pool (default device pool / `cudaMemPool`) enabled, so frees
   feed reuse rather than returning to the OS each time? Any release-threshold knob we should set?
4. **Concurrency:** safe to have many `new_async` allocs/frees outstanding on one stream across a
   realize (hundreds of ops) — any pool-growth or fragmentation caveat at that scale?

## If the free is synchronous today (the only possible ask)

If `Drop` frees synchronously, the additive ask is: **a stream-ordered free path** — either make
`new_async`-allocated buffers `Drop` via `cudaFreeAsync` on their origin stream, or expose an explicit
`free_async(stream)` / a `StreamOrdered` buffer variant Fuel opts into for op scratch + outputs. Same
contract discipline as elsewhere: no behavior change for existing sync callers, opt-in for the async
path.

## What Fuel does meanwhile

If stream-ordered free is already there (Q1/Q2 yes): A3 is small + pure-Fuel — switch `Workspace` +
output allocs to the stream-ordered path, defer the per-op `synchronize`, keep the D2H sync in
`to_cpu_bytes`. If not: A3 falls back to a Fuel-side retention pool for workspaces + executor
`force_synchronize` guards for data buffers (heavier, more peak VRAM) until the async free lands —
so the answer determines which path we build.

## Pointer

Design context: `fuel/docs/session-prompts/step-e-async-execution.md` (Phase A3). Consumers:
`fuel-cuda-backend/src/baracuda/*.rs` (per-op `Workspace` + output allocs) +
`fuel-cuda-backend/src/device.rs` (`alloc_zeros`, `new_async`).
