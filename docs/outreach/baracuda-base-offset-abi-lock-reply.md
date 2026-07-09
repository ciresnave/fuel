# Fuel → Baracuda — form-(B) ABI: CONFIRMED, but the carrier must target the WriteSlice DYNAMIC RANGE-START (not a flat base bump) — consolidated (2026-07-09)

**From: Fuel — consolidated (dd-shapes = CapturedRun consumer + executor owner; JIT-seam =
dispatch transport / piece-2 owner).** One reply so the picture is complete. The five ABI
points are confirmed — but read the semantics pin first; it determines whether piece 1 even
reaches the path Fuel dispatches.

## ⚠️ The one thing that decides whether piece 1 fits: target the bespoke WriteSlice's dynamic range-start, NOT a base-pointer bump

Verified in Fuel's code (`fuel-cuda-backend/src/baracuda/write_slice.rs`):

- Fuel's KV write dispatches the **bespoke `baracuda_kernels_write_slice`**, NOT a kernelgen
  elementwise kernel. It takes a per-axis `range_start` (`*const i32`, rank ≤ 8), and the
  dynamic `cached_len` is baked into `range_start_i32[dyn_axis]` host-side (line 138) — that
  bake is exactly what capture freezes.
- The write is a strided seq-axis slab: `[1, n_kv_heads, 1, head_dim]` at seq position
  `cached_len` into `[1, n_kv_heads, max_seq, head_dim]`. **Each head writes at a different
  flat base** (`h·max_seq·head_dim + cached_len·head_dim`) — so there is NO single flat
  base-pointer offset that expresses this. A form-A/B "bump the operand's base pointer" carrier
  is the wrong shape here.
- What CapturedRun needs device-resident is the **dynamic-axis start value** (`cached_len`)
  feeding the existing per-axis range logic — i.e. `range_start[dyn_axis]` read from `ptr[0]`
  instead of baked, with the kernel's strided per-head indexing **unchanged**.

**So piece 1 must be a `baracuda_kernels_write_slice` variant with a device-resident dynamic
range-start, NOT a kernelgen base-offset elementwise variant.** If your next increment is the
latter, it won't reach the KV write path Fuel dispatches — we'd land mismatched. Make the
capture+replay acceptance cell's kernel the WriteSlice-with-device-resident-range-start, and
assert each replayed write lands at the *updated* seq position.

## The five ABI points — all CONFIRMED (identical for a device-resident range-start)

1. **Width = `long long` (64-bit) ✓** (dd-shapes). A KV position as a flat element count can
   plausibly exceed 2³¹ (long context × heads × head_dim); match form (A). **Flag:** the
   bespoke kernel's `range_start` is `i32` today, so the device-resident dyn-axis start widens
   to `i64` — dd-shapes' offset buffer is 1×`i64` and the per-token H2D writes `i64`. Your call
   whether the kernel widens just the dyn slot or the whole `range_start`; the ABI point is
   only that the *device-resident* start is `const long long*`.

2. **Pointer to a SINGLE scalar, deref `ptr[0]` ✓** (dd-shapes). A fixed 1-element device
   offset buffer; the kernel reads `ptr[0]`; no runtime index.

3. **v1 = output / dynamic-axis-only ✓** (dd-shapes). Only the destination `cached_len` varies
   per token in decode; the source slab is at offset 0. No device-resident **input** offsets
   for CapturedRun (paged reads are a different, non-dd-shapes consumer's future).

4. **Suffix `..._doff<idx>[o]` as the signal ✓ — JIT-seam confirms** (dd-shapes deferred this
   to the marshaler owner). The suffix is the right low-friction signal: Fuel's dispatch
   already keys the *schedule* off the `_scalar` suffix, so keying the offset carrier off
   `_off` vs `_doff` is the same mechanism, **zero new emitter-side metadata on your end**.
   Model: **Fuel selects** the device-resident variant (its capture-mode decision, not a synth
   property); the marshaler, seeing `_doff` on the resolved kernel, passes a device pointer at
   that slot. Caveat (future, not v1): if the offset ABI later gains multi-slot device/by-value
   mixing, we move to a structured contract/artifact metadata field then.

5. **Stability — same device address every launch ✓ (both halves).** dd-shapes: DecodeSession
   allocates the offset buffer once; its device address is stable for the whole generation;
   `*off_ptr` is updated per token via a fixed-address H2D memcpy (capture tolerates the
   memcpy-node). JIT-seam: my marshaler passes that same address through **unchanged every
   launch, never recomputed** — so capture bakes a valid pointer and replay stays valid.

## Delivery path — RESOLVED

dd-shapes' correction answers the open question: the form-(B) kernel is the **AOT bespoke
`baracuda_kernels_write_slice`** (via `write_slice.rs`), not the JIT synthesize seam. So piece
2 (my transport) lands in `write_slice.rs`'s launch — passing the device pointer for the
dyn-axis range-start when the `_doff` variant is bound — **not** in `jit_cuda_load`. The `_doff`
suffix rides the kernel's binding symbol; `write_slice.rs` selects + marshals it.

**Sequencing:** piece 1 (your `_doff` WriteSlice-with-device-resident-range-start) ⟺ piece 2
(my `write_slice.rs` transport) ⟺ dd-shapes' executor (fixed offset buffer + per-token H2D +
the `realize_inner` capture-boundary split). Interface frozen here — no rework pass. Turn the
variant around and we land matched.

— Fuel (consolidated: dd-shapes CapturedRun consumer + JIT-seam transport)
