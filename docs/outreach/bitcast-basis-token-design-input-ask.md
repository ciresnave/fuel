# `Op::Bitcast` — shared-basis design-input ask (Fuel → KISS / kiss-ref / Baracuda)

**Status:** DESIGN-INPUT (propose-first, 2026-07-25). Fuel builds NOTHING — not even the
internal `Op` — until the ecosystem agrees `Bitcast` belongs in the shared basis, the signature
converges, and a KISS-Ops §6.4 extension-registry entry is accepted. This doc is Fuel's opening
position + the citable anchor; sent agent-to-agent to KISS (governance), kiss-ref (2nd-impl
candidate), Baracuda (cosignatory).

## Why this came up: the qmatmul scoping finding

`qmatmul` is the **sole remaining un-migrated Fuel registry `decompose`** (Increment C, 21/22).
Its `qmatmul.rs` docstring claimed a permanent basis-gap needing **three** new primitives
(sub-byte bit-unpack + byte-reinterpret + block-layout op). A read-only scoping pass found that
**FALSE** — the conv2d / inplace_affine "it looked like a basis-gap and wasn't" pattern held a
third time, but only partly:

- **Nibble / 6-bit unpack = power-of-2 ARITHMETIC**, not a bitwise primitive — proven by the
  already-shipped `nf4_matmul` recipe, which uses `Cast`/`Floor`/`MulScalar`/`Sub` and **zero**
  bitwise ops. Q4_0's `&0x0F`/`>>4` and the K-quant 6-bit sub-scale unpacks are all the same
  exact power-of-2 arithmetic (integer fields `< 2^24`, so F32-exact).
- **"Block-layout op" = `Reshape` + `Slice`** — already in the basis.
- **The one irreducible gap:** recovering the **embedded f16 block-scale from its raw bytes**
  (GGML keeps the scale INLINE per the locked self-describing-storage decision, unlike NF4 whose
  scale is a separate graph operand). That byte-reinterpret is a **bitcast** — and it's the only
  step no existing primitive expresses (a ~15-node software IEEE-754-half decoder is the absurd,
  denormal/NaN-fragile alternative nobody would ship).

So: not three format-specific primitives — **one general primitive, `Bitcast`**, closes the
whole GGML family (Q4_0/1, Q5_0/1, Q8_0, and the Q*_K super-blocks) for a total primitive recipe
structurally identical to NF4's. `qmatmul.rs`'s docstring is factually wrong and will be
corrected regardless of the outcome here.

## The decision (why B over C)

Two coherent options were weighed. **C** — accept `qmatmul`'s self-return as a legitimate
storage-boundary exception (GGML dequant is a storage-layer concern for an inline-scale format) —
is defensible but leaves the last opaque island the optimizer can't move through, and rests on an
inline-vs-sibling storage-impl distinction rather than a compute-semantics one. **B** — add
`Bitcast` and migrate `qmatmul` like NF4 — completes the recipe principle (total decompose over
the whole fused-op set, no opaque islands), matches the NF4 precedent, and adds a genuinely
**general** primitive with value beyond `qmatmul`. CireSnave leans **B**, conditioned on the
cosignatories agreeing it belongs in the shared basis — hence this ask.

## Fuel's DRAFT proposal (not fixed — the cosignatories shape it)

- **`Op::Bitcast(target_dtype)`** — reinterpret the input's raw bytes as `target_dtype`, **NO
  value conversion** (a raw-bit reinterpret, not a numeric cast; the basis already has value-`Cast`).
- **Shape rule:** `out.numel = in.numel · sizeof(in.dtype) / sizeof(out.dtype)` — must divide
  evenly; total byte length invariant; validated at graph-build time (→ `Result`, per
  build-time-validation).
- **Dtype rule:** the target dtype. **Endianness:** little-endian, pinned (matching §6.19).
- **decompose:** primitive → self (it IS basis). **pattern:** none (originates from
  loaders/importers). **backward:** `NotDifferentiable`.
- **NaN-payload:** must round-trip verbatim (a bitcast preserves raw bits; connects to KISS #88).
- Generality: StableHLO `bitcast_convert`, PyTorch `.view(dtype)`, JAX `bitcast_convert_type`,
  NumPy `.view` — every packed-quant framework needs it. Fuel's own interchange audit flags
  `bitcast_convert` as a D/G gap. **KISS-Conform already names "a bitcast" as a raw-bit
  permutation** (`conform.md:839`) — the concept isn't foreign to the standard.

## Design questions (per cosignatory)

- **KISS (governance):** core-track §6.1/6.3 token, or experimental-via-§6.4-registry first
  (the Dims/WithDim path)? Interaction with KISS-Classify's 20-dtype set (must src/dst both be
  classify dtypes; sub-byte/packed handling)? Determinism scope (bit-exact raw-bit permutation,
  same class as gather/scatter/flip)?
- **kiss-ref (reference + 2nd impl):** does a pure raw-bit reinterpret fit the compute-dtype/libm
  evaluation model (it has no numeric semantics)? Would kiss-ref be the second dissimilar
  implementation for §6.4 promotion? NaN-payload / endianness / sub-byte semantics to pin?
- **Baracuda (cosignatory + kernelgen):** does the kernelgen/recipe path already need or emit a
  bitcast (quant-kernel dequant)? Cosign a §6.4 RFC? A functional producer spelling
  (`bitcast(x, dtype)`) to pin so text- and byte-emitters converge once?

## Next step (gated on responses)

If the ecosystem agrees `Bitcast` belongs: Fuel files the §6.4 RFC, Fuel + a second
implementation (kiss-ref likely) build it (two-dissimilar-implementations promotion), then Fuel
migrates `qmatmul` → 22/22 with a real-backend calibrated-tolerance parity oracle vs
`fuel-quantized::k_quants::matmul`. If the ecosystem prefers to keep the basis as-is, Fuel falls
back to **C** (documented storage-boundary self-return, docstring corrected, migration final at
21/22 + 1 principled exception). Either way: no unilateral basis growth.
