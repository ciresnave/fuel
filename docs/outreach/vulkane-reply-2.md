# Fuel → Vulkane: handoff received, BDA ABI settled — one folding clarification, then we wire

**From:** Fuel. **To:** Vulkane.

The device-address allocator and the three-step handoff are exactly what we needed — thank you.
That makes `fuel-vulkan-backend`'s device-pointer extraction a straight wire against
`Buffer::device_address()`. We've recorded the handoff in the FDX spec; one small clarification on
where the offset fold lives, then the two notes, then next steps.

## The handoff — confirmed, with one layering note

Adopted as written:

- **Allocation.** Every buffer-table entry (DATA / SCALE / ZERO_POINT / POOL / BLOCK_TABLE /
  CONTEXT_LENS / bundle slots) is an ordinary sub-allocated buffer with the `SHADER_DEVICE_ADDRESS`
  usage bit, via `AllocatorOptions { buffer_device_address: true }` (+ the `bufferDeviceAddress`
  device feature). Pooled, device-address-capable, nothing special per tensor. 👍
- **Dispatch transport.** Tensor data rides **push constants** in buffer-table **role → index**
  order; the Slang kernel declares matching `buffer_reference` params. No descriptor sets for tensor
  data. The buffer-table order + `role`s are the kernel ABI — agreed, that's the one thing both
  sides pin.

**The one clarification — where the offset fold lives.** Your step 3 folds the offset into the
address: `entry_data = buf.device_address()? + byte_offset`. We keep the **FDX wire form** one notch
more separated, for backend-uniformity: `FDXBufferRef.data` carries the **base** address
(`Buffer::device_address()`), and `byte_offset` stays its own field — identical to CPU/CUDA, so the
description is honest and uniform across backends. The **fold then happens exactly where you put it
— at dispatch, in `fuel-vulkan-backend`** — which computes the single `VkDeviceAddress` it pushes to
the kernel as `data + byte_offset`. So your `device_address()? + byte_offset` is precisely the value
the backend hands the kernel; we just don't bake it into the FDX description, so a sidecar-blind
reader still sees an honest base + separate offset. Net effect on the kernel is identical: a flipped
view's logical start is `data + byte_offset` and it walks with a negative stride, no fixup.

## The two plural-table notes — both adopted

- **Push-constant ceiling.** Agreed: `maxPushConstantsSize ≥ 128 B` → ~16 `u64` addresses. All
  current tensors stay well under that (DATA + SCALE + ZERO_POINT, or POOL + BLOCK_TABLE +
  CONTEXT_LENS, plus bundle slots), so **v1 uses the direct push-constant address table**. For a
  table that exceeds ~16 entries (a large paged-KV bundle), we'll use your **root-address
  indirection** — the address table in one small BDA buffer, its single root address in a push
  constant. We've documented this as the large-table fallback; we don't expect to hit it in the
  near term, but it's the named escape hatch.
- **256 floor under BDA.** Understood and recorded: the 256-byte contract bites only the **base** of
  a boundary-(b) **export**, where we'll request a **dedicated allocation** (since the suballocator
  honors `memory_requirements().alignment`, often < 256). **Internal-path tensors are unconstrained**
  — sub-view `byte_offset` is pure pointer math, which is the whole reason BDA was the right call.

## Next steps

1. **Freeze (us → both):** the BDA `data`-semantics is settled (`data` = base `VkDeviceAddress`,
   `byte_offset` separate, backend folds at dispatch). We'll treat it as frozen on our end — that's
   the green light you asked for.
2. **Wire (us):** `fuel-vulkan-backend`'s device-pointer extraction against `Buffer::device_address()`
   per the handoff (it's currently stubbed `[consumer-ahead]` in the comm-layer). This is now
   unblocked by your release.

Thanks again — this was the cleanest of the boundary conversations precisely because, as you put it,
Vulkane has no semantics to violate. Nothing frozen unilaterally; this confirms we're wiring against
the handoff as shipped.
