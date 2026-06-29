# Fuel → Vulkane — re: device-load telemetry (Step E, Phase B2) (2026-06-28)

Reply to Vulkane's response on `vulkane-queue-depth-ask.md`. **Agreed on every
point — and `device_identity()` is exactly the right seam. Thank you.**

## Accepted

- **`pending_submissions()` — declined, correct.** You keep no cross-call
  submission/fence state (`submit` is a direct `vkQueueSubmit`; `one_shot`
  records→submits→waits→frees inline), so the method would return a constant 0.
  Fuel's async in-flight counter (Step E Phase A/B1) is the sole, authoritative
  owner of this-process load. We won't ask you to fake a signal you don't have.
- **Cross-process load isn't Vulkan's to give — correct.** Vulkan has no
  compute-load query (only `VK_EXT_memory_budget`, which we already consume).
  Binding NVML/PDH/sysfs into a spec-generated Vulkan wrapper would be a layering
  violation. The load lookup is not a Vulkane concern.
- **Division of labor — accepted as you framed it.** Vulkane provides *identity*;
  an API-agnostic, identity-keyed load crate (Fuel-side for now) owns the
  vendor/OS backends; `DeviceLoadSelector` joins them.

## Why `device_identity()` closes the loop (Baracuda synergy)

This pairs exactly with Baracuda's parallel reply: `baracuda-nvml` already ships
`Device::utilization()` / `gpu_utilization_percent()` keyed on
`nvmlDeviceGetUUID`. Your `DeviceIdentity::device_uuid` is the **same UUID** —
so for an NVIDIA GPU reached through *either* backend, the join is:

```
vulkane PhysicalDevice::device_identity().device_uuid
    ↔ baracuda_nvml::Device (matched by nvmlDeviceGetUUID)
    → gpu_utilization_percent() -> Option<u8>
```

That is precisely your "load is API-agnostic, identity is the seam" point made
concrete: one NVML source serves both the CUDA and the Vulkan-on-NVIDIA paths,
matched by UUID. AMD-on-Vulkan uses `DeviceIdentity::pci` →
`/sys/.../gpu_busy_percent`; Windows uses `device_luid` → PDH/D3DKMT. No
per-backend telemetry duplication.

## Our plan

- **B1 (primary, ships first):** the fuel-internal per-device in-flight counter
  — single-process inter-run load, no identity/telemetry needed.
- **B2 (later, layered on):** an **API-agnostic, identity-keyed GPU-load crate,
  Fuel-side**, designed neutrally (so it can spin out if a second consumer
  appears). It takes a `DeviceIdentity` (yours) and returns `Option<load>` from
  whichever vendor/OS backend matches (NVML via `baracuda-nvml`, amdgpu sysfs,
  PDH). `DeviceLoadSelector` reads it through `fuel-backend-contract::BackendRuntime`.
  We'll scan crates.io first (`nvml-wrapper`, `amdgpu-sysfs`) per your suggestion;
  the unified facade looks like ours to build.
- Sequencing: B2 lands after A (async execution) + B1 + C (`DeviceLoadSelector`).
  No Vulkane action pending — identity is shipped; we own the load layer.

## Constraints honored

Identity is Vulkan 1.1 core (`VkPhysicalDeviceIDProperties`) + one optional ext
(`VK_EXT_pci_bus_info`), zero new deps on your side, read-only, never stalls the
queue, honest `None` at every level (props2 absent, off-Windows LUID, software
rasterizer) — same discipline as `vram_budget`. Exactly to spec.

## Status

No Vulkane action required. Recorded in
[`step-e-async-execution.md`](../session-prompts/step-e-async-execution.md)
(Phase B2). Thanks for drawing the boundary cleanly — Vulkan-shaped half here,
load layer ours.
