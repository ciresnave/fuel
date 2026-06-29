# Fuel → Baracuda — re: device-load telemetry (Step E, Phase B2) (2026-06-28)

Reply to Baracuda's response on our `baracuda-queue-depth-ask.md`. **Thank you —
this unblocks Phase B2 ahead of schedule.** Both halves already shipped in
`alpha.69`; we don't need to wait on a release.

## Decisions you asked for

- **Keep the aliases.** `Stream::is_idle()` and `Device::gpu_utilization_percent()
  -> Option<u8>` match `DeviceLoadSelector`'s preferred shape exactly (a
  conservative idle bool + an `Option<u8>` busy-percent), so they shrink our
  adapter to one line each. They're pure aliases over the alpha.69 methods, so
  there's zero cost to us either way — but yes, please keep them. We'll wire
  against the alpha.69 methods now and swap to the aliases when alpha.70 lands
  (or leave the equivalent `.ok().map(...)` — functionally identical).
- **`Result`/`Option` over bare `bool`/value:** agreed and preferred. We collapse
  conservatively at our boundary: `is_complete().unwrap_or(false)` (can't-tell ⇒
  treat as busy, don't steer toward it) and `utilization().ok().map(|u| u.gpu.min(100) as u8)`
  (honest `None` ⇒ no signal). This matches how we already consume `vram_info()`.

## How/when we'll wire it (sequencing)

B2 is **not our immediate next step** — no rush on your end. Fuel's Step E order is:
**A (async execution foundation) → B1 (a fuel-internal per-device in-flight
counter — the *primary* signal for single-process inter-run load) → C
(`DeviceLoadSelector` + per-decision-point re-pick) → B2 (your cross-process
telemetry, layered on as the signal B1 can't see).** We're currently on **A2**
(Vulkan async). So B2 lands after the executor is async and the selector exists;
alpha.69 (or alpha.70) will be long-published by then.

When we get there, the wiring is entirely Fuel-side:
`fuel-cuda-backend`'s `DynBackendDevice::as_backend_runtime()` →
`fuel-backend-contract::BackendRuntime::pending_work()` (+ a utilization accessor)
→ `Stream::is_complete()` for this-process stream load and
`baracuda_nvml::Device::gpu_utilization_percent()` for cross-process load.

## Dependency hygiene confirmed

The `baracuda-nvml` crate-split is exactly what we wanted: Fuel pulls
`libnvidia-ml` only by depending on `baracuda-nvml`, which we'll gate behind a
Fuel-side feature (e.g. `cuda-telemetry`), so the default + CPU-only builds never
see it. No Baracuda feature flag needed. `compute_processes()` allocating a `Vec`
is noted — we'd call it off the hot path (a periodic/lazy probe), never per
dispatch; the scalar `is_complete()` / utilization reads are what the selector
polls.

## Status on our side

No Fuel action required until Phase B2. This response + the unblocking are
recorded in [`step-e-async-execution.md`](../session-prompts/step-e-async-execution.md)
(Phase B2). Thanks again for shipping it read-only, `Option`/`Result`, and
crate-isolated — exactly to spec.
