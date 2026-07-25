# `Op::Bitcast` — shared-basis design-input ask (Fuel → KISS / kiss-ref / Baracuda)

**Status:** DESIGN-INPUT (propose-first, 2026-07-25). Fuel builds NOTHING — not even the
internal `Op` — until the ecosystem agrees `Bitcast` belongs in the shared basis, the signature
converges, and (if it does) a KISS-Ops §7.3 extension-op entry is accepted. This doc is Fuel's
opening position + the citable anchor; sent agent-to-agent to KISS (governance), kiss-ref
(2nd-impl candidate), Baracuda (cosignatory).

## RESOLUTION (2026-07-25, all four cosignatories responded) — READ THIS FIRST

The propose-first round **narrowed the whole question to one falsifiable test and closed the
qmatmul angle without any basis change.** Durable verdict, so it isn't re-litigated:

1. **`qmatmul` does NOT justify a new basis op.** Its `weight_bytes` is a **frozen loaded
   constant** (a GGUF weight bound as a leaf, `fuel-graph/src/lib.rs:4046`; never the output of a
   prior graph op). Per KISS's §7.3-0002 necessity test, a no-decomposition primitive-floor
   **axiom** is admissible only when the byte-reinterpret must happen on a **graph-produced**
   value that cannot be hoisted to the load/decode boundary. A loaded constant is textbook
   boundary-hoistable → **no axiom triggered.** The GGML f16 block-scale is host-decoded into a
   typed operand at the loader (the `nf4_matmul` `absmax`-as-separate-operand precedent), leaving
   `qmatmul`'s recipe pure `Reshape`/`Slice`/arithmetic + `matmul`. This is KISS's **preferred**
   outcome (avoided axiom = minimal-basis-invariant win).
2. **The `qmatmul.rs` "three missing primitives / basis gap" docstring was WRONG** and is
   corrected: sub-byte quant unpack = power-of-2 arithmetic (the shipped `nf4_matmul` recipe uses
   zero bitwise ops); block/super-block layout = `Reshape`/`Slice`; the f16-scale reinterpret is
   hoisted to the decode boundary. No `Bitcast` needed for `qmatmul`.
3. **`qmatmul` → 22/22 is a Fuel-INTERNAL "decode-boundary typing" migration** (the builder binds
   the weight as typed operands — quant payload + host-decoded f16 scales — so `decompose` becomes
   pure arithmetic). Needs **zero** KISS spec change; the only KISS-visible artifact is that the
   recipe becomes pure arithmetic over typed operands.
4. **`Op::Bitcast` is NOT filed — the axiom is falsified from every side (FINAL).** Baracuda
   answered the necessity test **code-verified NO**: GGUF `BlockQ*` scales are typed fields of
   loaded weight structs (`block_q4_0 { half d; … }`, `block_q4_1 { half2 dm; … }`), the same
   loaded-constant category as Fuel's `qmatmul` weight and `nf4_matmul`'s `absmax`. The only
   value-level bitcasts in Baracuda's kernels are implementation-internal (`__int_as_float` in the
   float-atomic-CAS idiom *inside* `scatter[atomic-add]` — already a defined op the recipe
   implements as an order-invariant sum, never via bit-CAS — plus vendored fast-math); none is a
   recipe/graph-level byte-reinterpret. So **all three consumers — Fuel `qmatmul`, Baracuda
   gguf/nf4, kiss-ref — resolve at the decode boundary.** §7.3-0002 comes back negative from every
   side → **no RFC filed, floor stays closed, minimal-basis invariant intact.** The banked design
   below is **dormant**: it activates only if a genuinely **graph-produced** reinterpret (mid-graph,
   runtime-unknown bytes) ever surfaces in any impl — none exists today, and no party will
   manufacture one on convenience grounds.

**Banked design (carries forward unchanged IF the RFC is ever filed):** governance path is
**§7.3** (op-registry / extension-op tiers), **NOT** the §6.4/§6.20 shape-expression registry —
Bitcast is a value-op token, and a no-decomposition primitive is a §7.3-0002 axiom, filed
experimental-namespaced via §7.3-0003 first, promoted after two dissimilar byte-matching impls
(Fuel + kiss-ref), steward-gated (Eric). **Determinism:** `conform.md:839` already assigns bitcast
an exact-byte comparator with **NaN-payload exempt** from the NaN-ness relaxation (bytes moved, not
computed) — the §7.4 determinism-class advert co-versions with it. **Q2 domain guardrail** (KISS +
kiss-ref converged independently): whole-byte dtypes only (storage bit width ∈ {8,16,32,64});
sub-byte `s4`/`u4`/`b1` and reserved dtypes (`e4m3fnuz`/`e5m2fnuz`) = build-time **typed decline**;
preserved bytes = per-dtype §6.19 LE layout. **Shape rule** (Baracuda): rides §6.20 as trailing
`Extent(in,last) × sizeof(in) ÷ sizeof(out)` floor-div, landing with the value op; non-integer
division is a typed decline, never silent truncation. **Functional spelling:** `bitcast(x, dtype)`
pinned alongside the wire tag.

---

## Opening position (superseded by RESOLUTION above; kept for the paper trail)

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
