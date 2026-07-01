# Vulkane ask — device-load telemetry (Step E, Phase B2) — RESOLVED

**Status:** sent + **answered (2026-06-28)** — see [`vulkane-queue-depth-response.md`](vulkane-queue-depth-response.md).
**Outcome:** Vulkan has no cross-process (or in-process) compute-load query — confirmed not a Vulkane
gap but a Vulkan-API boundary; `pending_submissions` declined (no submission state to expose → use the
fuel-internal in-flight counter B1). Instead Vulkane shipped `PhysicalDevice::device_identity()`
(device_uuid / driver_uuid / device_luid / pci) — the **join-key** to correlate the device with an
out-of-band, API-agnostic load source. Pairs with Baracuda's NVML (UUID match). Fuel owns the
identity-keyed load-lookup crate (B2, after A/B1/C). Original ask below for record.
**From:** fuel dispatch-core cleanup, Step E (live-load `Op::Branch` arm selection).
**Nature:** read-only telemetry — no behavior change, no hot-path impact.

## What fuel wants

A cheap, read-only "how busy is this Vulkan device right now?" signal for fuel's `DeviceLoadSelector`
to steer arm selection. Candidate levels (whatever is feasible):

1. **This-process queue/fence depth** — vulkane already batches command buffers and waits on a fence
   per flush. Once fuel goes async (Step E Phase A), fuel will track its own per-device in-flight
   count, so vulkane needn't expose this — *unless* vulkane internally queues submissions fuel can't
   see, in which case a `pending_submissions() -> usize` read would help.
2. **Cross-process device utilization** — the genuinely-useful-but-hard one. Vulkan core has no
   portable GPU-utilization query. Feasible only via vendor/OS paths:
   `VK_EXT_memory_budget` (already used for VRAM) has no compute-load analog; options are
   vendor extensions, NVML (NVIDIA), or OS perf counters. The ask is: **is any cross-process
   busy signal reachable from vulkane on the target platforms?** If not, that's a fine answer.

Same contract as the existing VRAM-budget query fuel consumes: `Send + Sync`, allocation-free,
`Option`-returning, honest `None` when unavailable.

## Why optional (and what fuel does without it)

Fuel's primary load signal is **fuel-internal** (the executor's async in-flight counter, Step E
Phase A) — sufficient for single-process inter-run parallelism. This ask only adds cross-process
visibility. For Vulkan specifically, if no portable cross-process signal exists, fuel ships Step E on
the internal counter alone for the Vulkan backend; this stays a backlog item.

## Constraints / non-asks
- Read-only; must not stall the queue or force a submit/flush.
- `Option`/`Result`; never panic; honest "no signal."
- No new required dependency for fuel's default Vulkan build.

## Pointer
Design context: `fuel/docs/session-prompts/step-e-async-execution.md` (Phase B). Consumed via
`fuel-backend-contract::BackendRuntime` → the Vulkan `DynBackendDevice::as_backend_runtime()` impl
in `fuel-vulkan-backend` (which already implements the VRAM-budget half).
