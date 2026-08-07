# Scalar / DType Completion — Design Spec (GAP-002)

| | |
|---|---|
| **Date** | 2026-08-07 |
| **Branch** | `feat/scalar-dtype-completion` (off `origin/main` @ 56843728) |
| **Gap** | GAP-002 (Tier A — never-panic violation) |
| **Scope choice** | "Scales real, packed honest" (approved by CireSnave) |
| **Status** | Design — awaiting user review before implementation |

---

## Problem

`fuel_ir::Scalar::{zero, one, from_f64}` **panic** on 5 of the 16 `DType`
variants:

```rust
DType::F6E2M3 | DType::F6E3M2 | DType::F4 | DType::F8E8M0 | DType::F8E6M2 => {
    panic!("Cannot create zero scalar for dummy type {dtype:?}")
}
```
(`fuel-ir/src/scalar.rs:37, 57, 84`)

This is a standing violation of the never-panic-on-production-paths invariant.
Adding `F8E6M2` surfaced it but did not create it — the hole has existed since
the sub-byte dummy dtypes were introduced. Two of the three call sites reach
these constructors with a dtype derived from a live tensor node, so the panic is
*latent-but-reachable*, not merely theoretical.

## Background: `DType` is a superset of `Scalar`

Two sets that are currently conflated:

- **`DType` (16 variants)** — the *token space*: every format Fuel can name,
  parse, serialize, and route at the type level. Includes packed/scaled formats
  that exist so Fuel can speak to safetensors / ONNX / GGUF and carry
  `F4-elements + F8E8M0-scale` metadata through the DAG (see
  `fuel-ir/src/dummy_dtype.rs` and the self-describing-storage decision).
- **`Scalar` (11 variants)** — the subset Fuel can hold a *live arithmetic
  value* of.

The 5 missing dtypes are **not homogeneous**, which is why a blanket "add 5
variants" would be wrong:

| DType | Kind | Has a real single value? | Backing Rust type today |
|---|---|---|---|
| `F8E8M0` | block-**scale** | Yes — a positive power of two | none (raw `u8`) |
| `F8E6M2` | block-**scale** (finer) | Yes — 6-exp/2-mantissa unsigned | none (raw `u8`) |
| `F4` | packed **element** | Only as a bit pattern | none |
| `F6E2M3` | packed **element** | Only as a bit pattern | none |
| `F6E3M2` | packed **element** | Only as a bit pattern | none |

The scales are genuinely scalar (a per-block scale *is* one number). The packed
element formats have no Rust type and no single-value compute story in Fuel —
tensors in these formats are always dequantized *whole* to F32/BF16 before any
elementwise math; a lone `F4` value is never constructed.

## Decisions

### D1 — Constructor signatures become `Result`

`Scalar::{zero, one, from_f64}` change to:

```rust
pub fn zero(dtype: DType) -> Result<Self, ScalarError>;
pub fn one(dtype: DType) -> Result<Self, ScalarError>;
pub fn from_f64(v: f64, dtype: DType) -> Result<Self, ScalarError>;
```

Two independent forces require this — it is not merely to avoid the panic:

1. **Never-panic from day one.** The honest failure form is `Err`, not `unwrap`.
2. **Scales have no exact zero.** `F8E8M0`/`F8E6M2` encode `2^(x−bias)`; there is
   no bit pattern for `0`. So even *with* real scale variants, `zero(F8E8M0)`
   has no exact answer and must return `Err(ScalarError::NoZero { dtype })`.
   An infallible signature would force a fabricated value here — exactly the
   dishonesty this spec removes.

`ScalarError` (new, in `fuel-ir`) is a small enum:
`NoZero { dtype }`, `NoOne { dtype }`, `Unrepresentable { dtype, value: f64 }`,
`PackedElementHasNoScalar { dtype }`.

### D2 — Scales get real variants (`F8E8M0`, `F8E6M2`)

Add two variants holding the raw byte:

```rust
Scalar::F8E8M0(u8),   // OCP MX scale: value = 2^(x - 127); x=255 => NaN
Scalar::F8E6M2(u8),   // sk4 unsigned scale: 6 exp, 2 mantissa, no sign
```

with correct decode/encode (numeric appendix below). Because `Scalar` is **not**
`#[non_exhaustive]`, adding these two variants is a compiler-driven sweep: every
match on `Scalar` (`dtype()`, `to_f64()`, `PartialEq`, the `WithDType` glue,
plus any downstream) gains two arms. That is the intended mechanism — the build
breaks until each is handled deliberately.

- `to_f64` — decode per the appendix (exact for `F8E8M0`; finite non-NaN for
  in-range `F8E6M2`).
- `from_f64` — round to nearest representable; `Err(Unrepresentable)` on
  negative / NaN / out-of-range (both formats are unsigned, no ±inf as a value).
- `one` — the byte encoding `2^0`.
- `zero` — `Err(NoZero)` (no exact zero; see D1).

### D3 — Packed elements return `Err`, no new variant

`F4`, `F6E2M3`, `F6E3M2` get **no** `Scalar` variant. The three constructors
return `Err(PackedElementHasNoScalar { dtype })` for them.

Rationale (chosen over an opaque-bytes variant): a `Scalar::Packed { dtype,
bits }` whose `to_f64` still cannot yield a number merely *relocates* the panic
from the constructor to the accessor. No consumer round-trips a *lone* packed
scalar — packed values live inside tensors and are serialized as tensor bytes,
never as a standalone `Scalar` — so the opaque variant buys nothing and enlarges
the type. `Err` models the truth: "dequantize the tensor first."

The match stays **exhaustive**: 11 real → `Ok`, 2 scales → `Ok`/`Err(NoZero)`,
3 packed → `Err`. All 16 dtypes handled, no wildcard.

### D4 — Exhaustiveness is enforced, not assumed

Two mechanisms:

1. **No wildcard `_ =>` arms** in any `Scalar` method that matches on `DType` or
   on `Scalar`. Every dtype is named, so the next dtype addition *forces* a
   conscious `Ok`/`Err` decision at compile time (this is precisely how the
   `F8E6M2` addition was caught).
2. **A never-panic sweep over every `DType`**, using `catch_unwind` to prove
   `Ok`-or-`Err` (never unwind) for all three constructors:

   ```rust
   #[test]
   fn scalar_ctors_never_unwind_over_all_dtypes() {
       use std::panic::{catch_unwind, AssertUnwindSafe};
       for &dt in DType::ALL {
           // AssertUnwindSafe: the closures capture `dt` (Copy); the whole point
           // is to observe whether the call unwinds, so opting out is correct.
           assert!(catch_unwind(AssertUnwindSafe(|| Scalar::zero(dt))).is_ok(),
                   "zero() unwound for {dt:?}");
           assert!(catch_unwind(AssertUnwindSafe(|| Scalar::one(dt))).is_ok(),
                   "one() unwound for {dt:?}");
           assert!(catch_unwind(AssertUnwindSafe(|| Scalar::from_f64(1.0, dt))).is_ok(),
                   "from_f64() unwound for {dt:?}");
       }
   }
   ```

   **The sweep is only as honest as `DType::ALL` is complete** — and an
   `assert_eq!(DType::ALL.len(), 16)` guard does NOT establish that. If `ALL`
   and the `16` both trace to the same forgotten-to-update source, the assertion
   is a tautology that passes while a variant goes unswept (credit: peer
   refinement — this is the same vacuous-instrument class the exhaustiveness work
   exists to kill; I had written exactly this tautology into an earlier draft).
   The count needs an **independent, compiler-enforced** source. Two acceptable
   mechanisms, in preference order:

   - **Derive `ALL` by reflection** (`strum::EnumIter` or equivalent) so the list
     is *generated from the enum* and cannot omit a variant. Preferred — it
     removes the hand-maintained list entirely. Cost: a proc-macro dep on
     `fuel-ir`; evaluate that first.
   - **Generate the enum + `ALL` from one declarative macro**, so both share a
     single variant list.

   Absent either, the residual guarantee is mechanism (1): because every
   constructor match is wildcard-free, adding a `DType` variant fails to compile
   at each constructor until handled — which drags the author to the one place
   `ALL` is defined. That is a tripwire, not a proof; the reflection-derived
   `ALL` is the actual fix and is the recommendation.

### D5 — Build-time validation of `MaskedFill` (in-scope)

Per "validate at graph-build time," `MaskedFill` (and its backward) reject a
packed/scale element dtype at graph-build with a proper `Result` error, so the
constructor `Err` is a *backstop*, not the only guard. This closes the two
currently-unguarded call sites (`runtime_fused.rs:2699`, `lib.rs:8434`) at the
level where the mistake is expressible, rather than deep in a scalar
constructor. A fill/mask over a packed-quant tensor is a build-time bug; it
should be named as one.

### D6 — Docs

`DType`/`Scalar` semantics touch a constitution claim (never-panic + the
DType-vs-Scalar model). Update the relevant `docs/architecture/` section and add
a `10-decisions-log.md` entry (the scale-vs-element split, and `Result`
constructors). Coordinate the scale classification wording with the sk4 RFC so
the two do not diverge.

## Caller ripple (small)

Three call sites, all in `fuel-graph`:

| Site | Current | After |
|---|---|---|
| `runtime_fused.rs:510` `masked_fill_scalar` | already guards the 5 dummies → `None` | `Scalar::from_f64(..).ok()` (the guard collapses into the `Result`) |
| `runtime_fused.rs:2699` `Op::MaskedFill { value: Scalar::one(dtype) }` | unguarded | `?` (or `.map_err`) — **confirm enclosing fn is `Result`-returning during TDD**; add D5 build-time guard upstream |
| `lib.rs:8434` `Scalar::zero(dtype)` (MaskedFill backward) | unguarded | same as above |

`reduce_max_to_backward.rs:86` is a doc comment only. If either autograd site is
*not* in a `Result` context, the fix threads `Result` locally rather than
introducing an `.expect` (which would reintroduce a panic path). This is the one
mechanical unknown; it is resolved by reading the two enclosing signatures at
implementation time, not by guessing here.

## Numeric appendix (decode/encode)

**`F8E8M0` — definitive (OCP Microscaling spec).**
Value `= 2^(X − 127)` for `X ∈ [0, 254]`; `X = 255` ⇒ NaN. No sign, no zero, no
subnormals, no ±inf-as-value. `one` ⇒ `X = 127`. `from_f64(v)` for `v > 0` and
finite ⇒ nearest `X`; else `Err`.

**`F8E6M2` — per sk4 ruling (unsigned 6-exp / 2-mantissa, no sign).**
Proposed IEEE-style unsigned interpretation: bias `= 2^(6−1) − 1 = 31`; normals
`2^(E−31) · (1 + M/4)` for `E ∈ [1, 62]`; subnormals at `E = 0`; `E = 63`
reserved (NaN/inf per the reference). **The exact special-value handling (E=63,
subnormal rounding) is pinned against the sk4 / kiss-ref reference during TDD —
the RED tests are derived from that reference, NOT from this paragraph.** This
doc states the shape; the oracle states the bits.

## Testing strategy (TDD)

- **Born-red.** Each numeric behavior gets a failing test first, watched red,
  then green. No test written after the code it exercises.
- **Reference oracle.** `F8E8M0` decode is cross-checked against the OCP spec
  values; `F8E6M2` against the sk4 / kiss-ref reference vectors. Round-trip
  (`from_f64(to_f64(x)) == x` for representable `x`) is a property test over all
  256 byte values per scale format.
- **Sabotage-calibrated tolerances.** Any epsilon is validated by a confirmed
  sabotage run (perturb the decode, watch the test go red *after recompilation*)
  — a passing sabotage run without confirmed recompilation is invalid.
- **Exhaustiveness (D4)** — the never-panic `DType::ALL` sweep, over a
  reflection-derived (not hand-listed) `ALL`.
- **Build-time guard (D5)** — a graph-build test that a `MaskedFill` over a
  packed/scale dtype returns `Err` at build, not at realize.

## Out of scope / deferred

- **Full value-decode for packed elements** (`F4`/`F6*` bit-pattern ↔ f64).
  No single-scalar consumer; tensors dequant whole. Revisit behind a consumer.
- **An opaque-bytes `Scalar` variant.** Deferred unless a lone-packed-scalar
  interchange consumer appears (D3).
- **In-graph monitoring/probe ops.** Separate, larger design (its own spec).

## Open items (resolved during implementation, not blocking design approval)

1. Enclosing-fn `Result`-ness of the two autograd call sites (caller ripple).
2. Whether `DType::ALL` already exists; if not, add it with its own
   exhaustiveness guard.
3. Exact `F8E6M2` special-value bits from the sk4 / kiss-ref reference.
