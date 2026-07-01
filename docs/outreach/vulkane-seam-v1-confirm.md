# To the Vulkane team — Kernel-Seam Interop Contract (Profile v1), confirmed + wording fixes applied

**Status: DRAFT — not sent.** Round-2 note for `docs/specs/kernel-seam-interop.md` (Profile v1). The owner
sends.

---

Thank you — **Profile v1 conformance confirmed at the named BDA surface**, and all three of your wording
fixes (plus the alignment-ownership clarification) are **applied to the contract**. They were right: each
keeps the contract accurate and future-proof rather than leaning on a silent assumption. Summary of what
changed:

1. **The BDA "subset" is now stated as three producer-side preconditions, not a Vulkane auto-guarantee
   (§7.2).** Vulkane never auto-applies `SHADER_DEVICE_ADDRESS`; for `Buffer::device_address()` to be valid,
   `fuel-vulkan-backend` must satisfy all three together — (1) allocator via `new_with_options(..,
   buffer_device_address: true)`, (2) **per-buffer** `BufferUsage::SHADER_DEVICE_ADDRESS` on each
   buffer-table buffer, (3) the `bufferDeviceAddress` device feature
   (`DeviceFeatures::with_buffer_device_address()`, else `device_address()` returns `Error::MissingFunction`).
   Listed as a producer-side obligation under the "never silent coercion" discipline.
2. **§3.5 no longer misattributes the handshake to Vulkane.** It now reads that **`fuel-vulkan-backend`
   advertises the FDX version, derived from the linked `vulkane` crate version** — nothing on the wire
   originates from Vulkane, which exposes no `BackendCapabilities`/`BackendProbe` (those are Fuel-side FDX
   abstractions). As you noted, that's *better* than the spec implied: Vulkane does literally nothing, so
   there's nothing on your side to version or break.
3. **The Vulkane contract is pinned to behavior + named surface, not the crate version (§7.2).** Normative:
   `data = VkDeviceAddress` on `kDLVulkan`, `byte_offset` folded at dispatch, via
   `AllocatorOptions::buffer_device_address` / `Buffer::device_address` + per-buffer
   `SHADER_DEVICE_ADDRESS`. `vulkane 0.8.2` is the **informative** "first version exposing it" pointer — **a
   Vulkane major bump triggers a re-check of that named surface, not a silent pass**.
4. **Final-address alignment is named as Fuel's to own (§7.2).** Vulkane honors
   `VkMemoryRequirements.alignment` (often 16–256), not a guaranteed 256-aligned base address; since
   `byte_offset` is folded at dispatch and Fuel owns it, only the final `data + byte_offset` must meet the
   kernel's load alignment, and **`fuel-vulkan-backend` ensures it**. (§4.2 footnote² — Vulkane role-agnostic
   — stands, as you confirmed.)

## Net

Vulkane is conformant to **Profile v1** at the named BDA surface (`vulkane 0.8.2` being its first carrier).
We're clear to ratify alongside Baracuda once they re-verify their generator against fusion-patterns rev 4
(their two conditions are resolved in the same bundle). Thanks again — the BDA design is doing exactly what
we hoped at the seam.
