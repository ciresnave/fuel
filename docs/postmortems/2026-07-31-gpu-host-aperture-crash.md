# Post-mortem: GPU host-aperture kernel bugcheck (2026-07-31)

**Status:** investigated, mitigations landed / in progress. No root cause *proven*; the
most consistent read is a driver-internal fault under unserialized multi-session GPU
pressure, with several real latent defects fixed on their merits.

## What happened

While several concurrent agent sessions were doing GPU-adjacent development on this ML
stack, the machine hard-crashed with a kernel bugcheck:

> `VIDEO_MEMORY_MANAGEMENT_INTERNAL` (0x10E), subcode `0x2D` —
> "Call to `DdiMapCpuHostAperture` failed, but was expected to succeed."

The display driver failed to map video memory into the CPU's PCIe BAR window (the
*host aperture*), and the OS video-memory manager treats that failure as unrecoverable.

## The underlying problem

Several distinct issues combined; **none alone is proven to be the trigger.**

1. **The CPU host aperture is a separate budget from VRAM.** The aperture is the PCIe BAR
   window the CPU uses to address video memory — a distinct, smaller, fragmentation-
   sensitive resource, *not* total VRAM. A mapping can fail for want of contiguous
   aperture even when free VRAM looks comfortable. It is also invisible to ordinary VRAM
   budget queries (`cuMemGetInfo`, Vulkan memory-budget) — those report VRAM. We had **no
   instrument for the aperture at all**. (Note: with Resizable BAR enabled the aperture
   spans the whole framebuffer, so this is not a tiny-legacy-window problem — the window
   is already as large as the hardware allows.)

2. **Concurrent GPU work was blind and unserialized.** Multiple sessions ran GPU-touching
   work at once, each unable to see the others' allocations. A "one GPU run at a time"
   rule existed only as a convention in a docs file, enforced by nothing — and with
   enough sessions it did not hold. *A rule that depends on every participant having read
   the same file and voluntarily complied is not a control.*

3. **Latent allocator defects could ratchet or mis-route under pressure.** The shared
   Vulkan allocator had (a) a host-visible memory-type selection that, on some driver
   memory-type layouts, could resolve CPU-staging buffers to an aperture-backed heap
   instead of system RAM, and (b) an error path where a *failed* host mapping never freed
   its (large) block — turning a recoverable failure into progressive exhaustion. A
   redundant, dormant CUDA virtual-memory crate in our own tree carried its own
   unreleased-mapping bug.

4. **A wrong hardware assumption.** Config/docs assumed a larger VRAM capacity than the
   actual (smaller, laptop-class) GPU, so any capacity math over-committed.

## What we investigated — and did NOT conclude

- **The GPU driver was rolled back — MANUALLY by the operator, not by the OS — so it carries
  no attribution.** The crash-time NVIDIA driver was 610.88 (`32.0.16.1088`); the machine now
  reports 610.74 (`.1074`), confirmed by two independent reads. The operator rolled it back
  *deliberately, as a precaution*, on the hypothesis that the .1088 driver might have been the
  cause — it was **not** a Windows-initiated post-bugcheck rollback. (An earlier draft of this
  note inferred "the OS attributed the fault to the driver" from the version change alone,
  without verifying *who* caused it — the same assert-from-a-datum-without-a-positive-control
  defect this file's process exists to catch, and the one place all night the evidence seemed
  to point our way, which is exactly when the inference was weakest.) So there is no OS verdict
  here; there is an operator's reasonable suspicion. What *does* hold, and matters
  operationally: **every result after the rollback runs on .1074 and says nothing about
  .1088** — the clean post-crash test run and all later validation are evidence that .1074
  works, not that the .1088 which crashed was fine. A reproduction attempt from here runs on a
  different driver, so a non-reproduction means "the .1088 conditions no longer exist on this
  machine," NOT "unreproducible, probably not our code." (A control assertion did confirm the
  driver-identity change is real, not an artifact of the same-night Vulkan instance-version
  bump.) **This rollback is an active mitigation, not a restoration — the machine is now in a
  MODIFIED state pinned to .1074. A future driver update could silently reinstall .1088 and
  bring back the pre-crash conditions with no one connecting the two events. Treat the
  pinned-back driver as an ongoing condition to watch, not a closed action.**
- **No single session was the proximate trigger.** At the crash moment every agent
  session was doing CPU-only or compile-only work. Sessions that had done live GPU work
  earlier had finished and freed well before the crash.
- **A session census cannot clear "the machine."** The processes that dominate GPU memory
  at idle — the desktop compositor, terminals, browser/webview processes — are outside
  any session's view and collectively far outweigh anything this ML stack holds.
- **With a working per-process instrument (OS GPU performance counters), the crashed GPU
  sat far below capacity at idle, and no measurement showed our workloads approaching
  exhaustion.** We could **not** positively confirm volume exhaustion.
- **Retiring volume exhaustion does not exonerate allocation patterns.** The subcode
  ("expected to succeed") is the driver reporting an *internal invariant violation*, which
  fragmentation of the aperture or a driver bug can cause with no unusual volume from
  anyone. A small-but-badly-shaped mapping pattern is excluded by nothing we can measure.

**Instrument note (a mistake worth recording):** resident-VRAM counters measure the wrong
quantity — the aperture cares about CPU-*mapped* bytes, not resident VRAM. An experiment
scoped to resident usage would have come back clean *without testing the hypothesis*. The
right instrument for our own footprint is a peak counter over the host-visible **mapped**
allocations we ourselves request (see mitigation 5).

## What we changed (to reduce the chance of recurrence)

1. **Deleted the redundant, dormant, leaky CUDA VMM crate.** Its mechanism already lives
   in the CUDA driver library; any future VMM support will be a **backend-agnostic**
   interface implemented uniformly per backend, not a per-backend crate.
2. **Fixed the shared Vulkan allocator:** host-visible staging now *prefers* a
   non-device-local (system-RAM) memory type, falling back only if none exists; the
   failed-mapping error path now frees its block on every exit; readback is documented to
   use a cached custom pool.
3. **Machine-wide GPU serialization as a seam, not a convention:** a `gpu-run` wrapper that
   acquires a lockfile carrying held-by metadata, reclaims a stale lock only by
   PID-liveness (failing loud on ambiguity, never a silent timeout heuristic), and logs
   contention — scoped to device-creating runs (on-device tests, capture/replay,
   sanitizers, GPU BLAS, Vulkan enumeration). **The harness takes the lock, so forgetting
   is impossible** rather than merely discouraged.
4. **Bounded our own GPU footprint:** cap concurrent CUDA contexts in tests (reuse one
   shared context rather than one per test); bound the device memory pool's retention;
   route readback through a cached pool plus a defensive check that staging never resolves
   to a device-local (aperture) memory type.
5. **A self-monitoring aperture counter:** an in-process peak counter over host-visible
   *mapped* allocation bytes — the aperture-relevant quantity — so we can see our own
   aperture footprint going forward (partially closing the "nothing tracks the aperture"
   gap). **Landed + live-verified:** a `MappedByteMeter` (saturating current+peak) plus an
   RAII `MappedGuard` wired into the two host-visible H2D staging sites (`upload_bytes`,
   `write_bytes`) — the guard releases on every exit including a `?` return, so a failed
   mapping can't ratchet the accounting (mitigation-3b's leak mode). A live 4 MiB upload on
   the RTX 4070 lifted the peak by exactly 4 MiB and released back to zero
   (`fuel-vulkan-backend/tests/mapped_meter_live.rs`, run under the GPU lock). It is a
   *partial* instrument by design — it sees our own mapped footprint, not the machine's
   total aperture pressure — and the D2H download-staging path is not yet wired (follow-up).
6. **Corrected the VRAM-capacity assumption** to the real hardware.

## Lessons

- **Aperture ≠ VRAM.** Track and reason about the host aperture as its own budget; don't
  assume free VRAM implies free aperture.
- **Serialize shared hardware with a seam, not a convention.** A rule enforced by
  everyone-remembering is not enforced. The harness holds the lock, not the human/agent.
- **Measure the right quantity.** Resident VRAM is not mapped (aperture) memory; a clean
  reading of the wrong counter is a false negative that looks like a result.
- **An "expected to succeed" driver failure can be driver-internal.** Fix the latent
  defects as hygiene; do not over-attribute the crash to any one allocation.
- **Verify the committed state, not the working tree.** (Meta: the commit that deleted the
  offending crate initially left a dangling workspace member because only part of the
  change was staged — `cargo` was checked against the working tree, which had both edits.
  Verify what you actually shipped.)
