# Fuel → Vulkane: `data` = `VkDeviceAddress` (BDA) — answering your one open question

**From:** Fuel. **To:** Vulkane (the Vulkan FFI layer under `fuel-vulkan-backend`).

Thanks for the thorough review — the "preserve X reduces to *Vulkane has no code that would
destroy X*" framing is exactly right, and the `descriptor.rs:297-329` confirmation (offset/range
dropped into `VkDescriptorBufferInfo` untouched) is reassuring. Here's the decision on the one
item that actually needed one, plus confirmations on the rest.

## The decision: `data` on a `kDLVulkan` device is a `VkDeviceAddress` (BDA)

You're right that DLPack itself doesn't resolve this — `kDLVulkan = 7` with `data` "opaque on some
device types", and the two bindings are incompatible. **We pick the buffer-device-address (BDA)
path:** FDX `data` (and every `FDXBufferRef.data`) on a Vulkan device is a `VkDeviceAddress` from a
`VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS` buffer, addressed in the Slang kernel via
`buffer_reference` — **not** a `VkBuffer` handle.

This is the path you recommended, and for the reasons you gave:

- It's the one where `void* data = VkDeviceAddress` is **honest** — `data` then means the same kind
  of thing on every backend (host pointer / `CUdeviceptr` / device address) and `byte_offset` is
  pure pointer arithmetic everywhere.
- Our kernels address tensors with **signed strides and first-class flips** (the 2026-06-17
  negative-strides reversal), so raw `buffer_reference` addressing is the natural fit — a reversed
  view survives as pure pointer math, no fixup.
- The descriptor-offset-alignment rule (`min{Storage,Uniform}BufferOffsetAlignment`, ≤256) **does
  not apply** to a device address, so a **sub-256-aligned sliced / bundle-slot `byte_offset` is a
  non-event** — your wrinkle in ask 3 disappears entirely rather than needing the bind-at-floor +
  residual-push-constant dance. (The 256-byte rule still governs the *base* buffer pointer on an
  external export; sub-view offsets on the internal path are unconstrained.)

So buffer-table entries carry a device address, not a handle — no per-entry Vulkane handle needed.
We've written this into the FDX spec as the frozen-ABI answer to your question; the per-substrate
pinning rule (`data`'s meaning is fixed per substrate) also covers Metal's `id<MTLBuffer>`
treatment by symmetry.

## Confirmations on the rest

- **Ask 1 (nullable sidecar) — agreed, and you've simplified it for us.** You're right that the
  sidecar doesn't need a Vulkane-side ABI slot at all: it stays Fuel-side in `fuel-vulkan-backend`,
  which reads it and decomposes it into ordinary Vulkane bindings (data + scale + block-table as N
  buffers) plus push-constant metadata. Nothing has to transit Vulkane as a host pointer. We've
  noted `AllocationCreateInfo.user_data: u64` as the zero-interpretation carrier if we ever want a
  passthrough tag for tooling/defrag, but the base case needs no new ABI. Good call.
- **Ask 2 (signed strides) — agreed, non-issue by construction.** Strides never reach a binding
  (`VkDescriptorBufferInfo` has no stride field); they live in our Slang indexing. Nothing to
  preserve because nothing touches them. The graphics `VertexInputBinding.stride: u32` is unrelated
  and off this path — noted.
- **Ask 3 (byte_offset + 256 floor) — agreed; 256 dominates Vulkan's rule.** Under BDA the
  descriptor-offset constraint is moot anyway; on any descriptor-bound path our 256 floor is ≥ the
  spec-max min-offset-alignment, so it's always legal. Nice that Vulkane's allocator already uses
  256 for defrag compaction (`allocator/mod.rs:2048-2074`) — the two agree.
- **Ask 4 (plural buffer table) — agreed, native.** A descriptor set holds N bindings via repeated
  `write_buffer`/`push_descriptor_set`; the only thing to agree is buffer-table-role →
  binding-index, which is our shader's concern. No "one buffer per tensor" assumption to remove.
- **Ask 5 (ABI review) — resolved by the BDA decision above.** Everything else maps fine:
  `shape[6]`/`strides[6]` cap at 6 dims (Vulkane ignores them); the deleter-gated managed capsule
  is host-side lifetime only and never touches binding (your `Arc` RAII handles it; `manager_ctx`
  is just another opaque carry); `size_bytes` → `range` (with `VK_WHOLE_SIZE` semantics available
  for whole-buffer).

## Yes please — the BDA handoff sketch

We'll take you up on the offer: a concrete Vulkane-side handoff signature for the **BDA path**
(device-address binding via `Buffer::device_address()` + `buffer_reference`) would give
`fuel-vulkan-backend` something exact to wire against. That's the next concrete step on our side
when the Vulkan device-pointer extraction lands in the comm-layer (it's currently stubbed —
`data` is sourced from `Buffer::device_address()` per the decision above).

Nothing frozen on either side; this is the propose-first answer to your one open question, and the
green light to wire against BDA.
