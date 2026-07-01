# Fuel → Baracuda — JIT wire types: reconciliation round 2 (2026-06-21)

To the Baracuda synthesizer team, on your *reconcile + enumerations* reply (2026-06-21).
**Your §1 catch is correct and we accept it** — `FdxOperandDesc` carrying a pre-classified
`LayoutFlags` did re-introduce Fuel computing the key, exactly the ratified-division violation
you flagged. That's the one item you'd block the freeze on, so it's first and it's settled (§1).
§2 confirmed, the enumerations reconciled (§3 — with one naming note you'll want), §4/§5 agreed.
Net: nothing left to block the freeze; what remains is **us cutting the two frozen types** (§6).

---

## 1. `FdxOperandDesc` = the raw ratified projection. ACCEPTED — our error, reverted.

You're right, and the reasoning is airtight: a per-operand `LayoutFlags` on the wire is either
lossy (can't derive `vec_width`/`idx_width`/`inner_div`/`flipped` without raw strides + alignment)
or it *is* Fuel computing the key (the desync-on-disagreement failure you named). The five flags are
the **output** of `structure_key`, never an input. We revert to **exactly your
`baracuda_kernels_types::OperandDesc` projection**, unchanged:

```rust
pub struct OperandDesc {
    pub rank:        u8,
    pub shape:       [i64; 8],   // logical extents; symbolic axes carry capacity
    pub strides:     [i64; 8],   // signed element strides (0 = broadcast, <0 = flipped)
    pub dtype:       DTypeTag,
    pub align_bytes: u32,        // base-pointer alignment — drives vec width
    pub quant:       Option<QuantFacts>,   // carried; v1 doesn't key on it
    pub symbolic:    Option<SymExtent>,    // live-vs-capacity, attention-class
}
```

Fuel builds this from its `FDXSidecar` / tensor (we already have the raw strides + `align_bytes` +
extents there) and passes it **verbatim**; **you classify; you return the `StructureKey`.** No
`LayoutFlags` crosses the wire. If we ever want a display projection of the classification, it's a
pure function of the key you return — derived on our side, never recomputed. This is the same
single-classifier discipline FDX uses, applied to the schedule key. Settled.

## 2. Scalar `attrs` = the slot, value param-ized. CONFIRMED — that's the intent.

Exactly as you describe, and it matches 5.3: `attrs` carries the **scalar slot** (the `operand(j)…value`
target the emitted `extract:` points at) plus the non-scalar load-bearing attributes (`Reduction.axis`,
`Clamp.min/.max`, …). The concrete `AddScalar.value` is **not** baked in increment 1 — it becomes
`op_params.param{i}`, and our matcher re-reads the live value from the matched graph node via the
`extract:` path at match time. Round-trip confirmed: region `attrs` slot → your `Param` → emitted
`extract:` path → our `op_params` binding. Specialization-by-baking stays a future, budget-gated
option, not the default — agreed.

## 3. Enumerations

**`OpTag` — Fuel's primitive-`Op` vocabulary is the canonical list; your increment-1 subset maps 1:1.**
Our canonical set is the primitive `Op` enum (`fuel-graph/src/lib.rs`, the `op_short_name` table —
the source of truth we'll freeze `OpTag` from). Reconciled against your increment-1 synthesizer
coverage:
- **binary `Add`/`Sub`/`Mul`/`Div`** → `Op::{Add,Sub,Mul,Div}` — 1:1.
- **scalar-param `AddScalar`/`MulScalar`** → `Op::{AddScalar,MulScalar}` — 1:1.
- **unary `Neg`/`Abs`/`Sqr`/`Sqrt`/`Rsqrt`/`Recip`/`Exp`/`Log`/`Tanh`/`Sigmoid`/`Relu`/`Erf`/`Silu`**
  → `Op::{Neg,Abs,Sqr,Sqrt,Rsqrt,Recip,Exp,Log,Tanh,Sigmoid,Relu,Erf,Silu}` — 1:1.
- **`GeluErf` (exact erf)** → `Op::GeluErf` — 1:1, **and your "bare Gelu/tanh is not synthesized" is
  exactly right**: in our vocabulary `Op::Gelu` is the *tanh approximation* (a distinct variant), and
  `Op::GeluErf` is the exact-erf one. So synthesizing **only `GeluErf`** is the correct, non-lossy
  choice — a region with `Op::Gelu` (tanh-approx) is genuinely a different op and an honest
  `UnsupportedOp` miss until you add it, which is the behavior we want.

Two deltas to record (neither blocks — they're "won't appear in a region"):
- **In-place variants** (`Op::ReluInplace`, `…Inplace`) are **never in a JIT region** — a region is the
  *functional* primitive subgraph; in-place is a Fuel-side scheduling rewrite. `OpTag` covers the
  functional ops only.
- **Structural ops** (`Op::Const`, `Release`, `ZeroFill`, `Contiguize`, `Move`) are graph-bookkeeping,
  not synthesizable region ops — also excluded from `OpTag`.

The full §4.1 set (`Maximum`/`Minimum`/`Pow`/`Where`/`MatMul`/reductions/shape-layout/…) is the shared
enum; you miss on the ones outside increment-1 IR coverage. We'll publish the frozen `OpTag` enum
(functional ops only) and you confirm 1:1 / send the delta.

**`DTypeTag` — FDX §5 base table; your increment-1 subset is clean.** Our dtype set is the FDX §5
base (`F32`/`F64`/`F16`/`BF16`/`F8E4M3` + the integer family); your increment-1 `F32`/`F16`/`BF16`/
`F64`/`I32`/`I64` is a strict subset — confirmed, `F32Strict→F32`, sub-byte rides the sidecar (not a
base dtype), `F8E5M2`/complex have no §5 slot → honest miss. The `OperandDesc.dtype: DTypeTag` is this
shared table.

**`OpCategory` (yours, opaque to us) — acknowledged.** Our `JitRequest` constructor sets one of your
variants; for an elementwise-epilogue region it's `UnaryElementwise` / `BinaryElementwise` /
`TernaryElementwise` / `GatedActivation`, and we pass it to `structure_key` untouched. `#[non_exhaustive]`
noted.

**`ArchSku` (yours, we derive) — acknowledged.** `Sm80`/`Sm89`/`Sm90a`; we derive from the device and
pass it. We'll flag if we need a SKU you don't list (Baracuda-side build-time add, understood).

## 4. One node form, direction-specific fields. CONFIRMED.

One `PatternNode` type, two directions: a **region** (Fuel→Baracuda) populates `op`/`operands`/`attrs`;
an emitted **`pattern:`** (Baracuda→Fuel, in the contract) populates `op`/`operands`/`consumers`/`extract`.
`see_through`/`any` are matcher-only and never appear in a concrete region. When we cut the frozen type
(§6) it carries the union `{op: OpTag, operands, attrs, consumers, extract}` with the region direction
leaving `consumers`/`extract` empty — you align your internal `{op, operands, consumers, extract}` to it
by adding `attrs` + taking `op: OpTag`. One type, both the JIT region and `pattern:` matching, as agreed.

## 5. Transport + the two foundations. AGREED.

Direct-Rust for increment 1, C-ABI deferred, handshake stays C-ABI — yes. And confirmed: our two §D
foundations (the frozen `PatternNode` enum + the `OperandDesc` projection) gate the **first live call**,
not the wire shape. Yours is built and on-device-validated; the moment we cut them you reconcile (§1, §4)
and we call across.

## 6. What we're cutting next (the unblock is entirely on us)

With §1 settled and §3 reconciled, nothing on the wire is open. The remaining work is **Fuel cutting and
freezing the two types**, which is also our declarative-pattern-engine foundation (one `PatternNode`
serves the JIT region, `pattern:` matching, and a synthesized op's `decompose`):
1. **`OperandDesc`** — your §1 projection, verbatim, built from our `FDXSidecar`.
2. **`PatternNode`** — the §3 grammar as a concrete enum, `OpTag`-keyed (functional ops only), with
   `attrs` for the scalar-slot/`extract` round-trip.

We freeze and send both; you turn them around fast and we advertise **`SeamCapJitOnRequest`**. No
remaining disagreement — just our two types to cut.

— Fuel
