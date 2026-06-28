# Baracuda ask — device-load telemetry (Step E, Phase B2) — DRAFT / OPTIONAL

**Status:** draft for CireSnave's review; **not yet sent**. Lower priority than the fuel-internal
signal (see "Why optional").
**From:** fuel dispatch-core cleanup, Step E (live-load `Op::Branch` arm selection).
**Nature:** read-only telemetry addition — no behavior change, no allocation, no hot-path impact.

## What fuel wants

A cheap, read-only way to ask "how busy is this CUDA device / stream right now?", consumed by
fuel's `DeviceLoadSelector` to steer arm selection toward the device that will drain fastest. Two
levels, either/both:

1. **This-process stream idle/busy** — a `cuStreamQuery` wrapper: `stream.is_idle() -> bool`
   (CUDA_SUCCESS ⇒ idle, CUDA_ERROR_NOT_READY ⇒ work pending). Cheap; reflects only *this* process's
   submissions to that stream.
2. **Cross-process device utilization** — an NVML wrapper:
   `device.utilization() -> Option<u8>` (GPU busy %, via `nvmlDeviceGetUtilizationRates`), and
   optionally `device.compute_processes()`/free-vs-used SM proxy. This is the signal that captures
   *other processes* sharing the GPU — the only thing fuel can't measure itself.

Shape is baracuda's call; fuel needs only a `Send + Sync`, allocation-free, `Option`-returning read
(honest `None` when unavailable, never fabricated — same contract as the existing `cuMemGetInfo`
VRAM query fuel already consumes).

## Why optional (and what fuel does without it)

Fuel's primary load signal is **fuel-internal**: once execution is async (Step E Phase A), the
executor maintains a per-device in-flight-work counter (work it submitted but hasn't seen complete).
That fully covers the single-process inter-run-parallelism case, which is the runtime's documented
job. This ask only adds **cross-process** visibility (another job hammering the same GPU). So:
- If easy: (1) `cuStreamQuery` is a tiny, obviously-safe add; (2) NVML utilization is the higher-value
  one for shared-GPU scheduling.
- If not: fuel ships Step E on the internal counter alone; this stays a backlog refinement.

## Constraints / non-asks
- Read-only; must not synchronize the stream or perturb scheduling.
- `Option`/`Result` return; never panic; honest "no signal" when the driver/NVML can't answer.
- No new required dependency for fuel's default build — gate NVML behind a baracuda feature if it
  pulls libnvidia-ml.

## Pointer
Design context: `fuel/docs/session-prompts/step-e-async-execution.md` (Phase B). Consumed via
`fuel-backend-contract::BackendRuntime` (a new `pending_work()`/utilization accessor) → the CUDA
`DynBackendDevice::as_backend_runtime()` impl in `fuel-cuda-backend`.
