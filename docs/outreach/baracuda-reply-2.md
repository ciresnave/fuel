# Fuel → Baracuda: confirmations before freeze (reply to your 2026-06-19 acceptance)

**From:** Fuel. **To:** Baracuda kernel-specialization / AOT-matrix team.

Thanks — we're aligned. You accepted the join-token model and all six asks; this closes the four
items you flagged before freeze. Everything below is **done or committed as Fuel's position**;
nothing is frozen unilaterally.

## The four items you raised — all settled

1. **`structure_key` takes a minimal operand-description projection, not the whole sidecar —
   CONFIRMED.** That's exactly what we'll feed. Fuel will build a small `FdxOperandDesc` carrying
   only the five facts the key reads — **strides** (→ contiguity / broadcast-from-stride-0 / flipped
   from sign), **dtype** (base code + the sub-byte/quant facts), **alignment**, **quant**
   (family + block geometry), and **symbolic extent** — and call your shipped
   `structure_key(op_class, operands, arch)` with it. We never derive the key. Your
   `From<FDX>` / `From<DLTensor>` / `From<TensorRef>` adapters are the right shape and keep the
   shared dependency small; we'll consume whichever the call site has.

2. **`ImplId` wire fields stay separable, never one opaque hash — CONFIRMED and LOCKED.** The wire
   form serializes the five basis fields independently — `backend`, `op`, `dtypes`,
   `kernel_source`, `kernel_revision_hash` — so you can group by
   `(backend, op, dtypes, kernel_source)` for matrix ranking and use the full tuple for exact
   re-resolution. We added a test that asserts the JSONL carries all five by name and round-trips
   losslessly, so a future change can't silently collapse them into a hash. **Ready to freeze on
   the basis tuple.**

3. **Pre-freeze dtype reconciliation — DONE.**
   - **Added** three FDX sidecar logical-dtype codes (the `0x01xx` low-bit family):
     `I4 = 0x0102` (packed 4-bit signed int, 2/byte, `DENSE_SUBBYTE`, `bit_width 4`),
     `U4 = 0x0103` (packed 4-bit unsigned int, same packing), `B1 = 0x0104` (bitpacked binary,
     8/byte, `DENSE_SUBBYTE`, `bit_width 1`). They have no Fuel `DType` (so the reverse
     `fdx_to_dtype` returns `None`, like the `GENERIC_LOW_BIT_*` escapes) — they exist purely so a
     producer/consumer can name packed `S4`/`U4`/`Bin` in the sidecar. Dedicated codes (vs. the
     generic escape with a flag in `reserved[0]`) keep your structure-key dtype axis clean:
     `I4`/`U4`/`U8` are distinct codes. Both the Rust constants and the vendored C header carry them.
   - **Base-DLPack passthrough — CONFIRMED.** The `FDX_DTYPE_*` namespace is **sidecar-only**; it
     never constrains the base `DLTensor.dtype`. So every type DLPack v1.3 can name rides the base
     honestly with **no FDX code and no sidecar**: `Fp8E5M2` → `kDLFloat8_e5m2`;
     `Complex32`/`Complex64` → `kDLComplex` (`bits` = **total**: 64 / 128 — we adopt DLPack's
     total-bit convention, so your per-component `Complex32`=2×f32 maps to numpy `complex64`
     unambiguously); `Bool` → `kDLBool`; *unpacked* 4-bit int → `kDLInt`/`kDLUInt`, `bits = 4`. A
     sidecar `FDXDTypeExt` is required only for the *packed* sub-byte cases (base = `uint8`
     stand-in). The `COMPLEX64`/`BOOL` rows in our table are reserved placeholders only; in
     practice complex and bool ride the base and you emit no `FDXDTypeExt` for them.
   - **`F32Strict`** — agreed, it's a precision *mode* over f32 storage, surfaced in the FKC
     `precision` block (bit-stability / accumulate width), never a wire dtype. It never reaches the
     FDX dtype namespace.

4. **FKC as a *generated* projection of your `KernelSku` / `PrecisionGuarantee` / OP-MATRIX —
   CONFIRMED, and it's the better model.** It changes nothing in our importer — we parse the same
   ` ```fkc ` blocks and resolve the same `link_registry`; you just author them mechanically. The
   property it buys is exactly the one that makes our miss signal honest: a generated contract's
   admissibility predicate *is* the structure key by construction, so it can't under- or
   over-declare admissibility. Hand-authoring stays available to other providers, never required.

## On our side

- **Emission layer — underway.** We've started building it over the confirmed retention: the
  JSONL wire schema (`DispatchRecord` / `Candidate` / `MissRecord` / `ImplId` /
  `StructureKeyToken`) and the `ImplId::classify()` → `{Baracuda|Vendor|FuelNative}` projection
  are done; the miss-signal read, the per-impl `candidates[]` fill from the oracle, the opt-in
  `Off`/`Coarse`/`Detailed` flag, and the JSONL sink follow.
- **Coverage gating — aligned (your §7).** v1 ranks the matrix by miss `count` first (coverage-
  independent); `candidates[]` densifies automatically as the Judge's matrix grows, no format
  change. The vendor-exclusion gate from `candidates[]` lights up then.

## Next steps (matching your §8)

1. **Done (us):** the two sidecar dtype codes added; base-DLPack passthrough confirmed.
2. **Both:** freeze the `ImplId` wire encoding — basis settled, fields separable, our schema
   locked. Say the word and we'll treat it as frozen.
3. **In progress (us):** the emission layer over the confirmed retention.
4. **You:** ship `structure_key` + the FKC/`link_registry` generator with the elementwise pilot.
   When your callable lands we wire Fuel's `structure_key` trampoline to it (Fuel ships a no-op
   provider until then, so the miss histogram and `candidates[]` work coverage-agnostically in the
   meantime).

Nothing here is frozen on either side; this is the propose-first confirmation that the four
pre-freeze items are resolved.
