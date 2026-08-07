# Scalar / DType Completion Implementation Plan (GAP-002)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `fuel_ir::Scalar` total and never-panicking over all 16 `DType`
variants — real arithmetic for the two block-scale dtypes, honest `Err` for the
three packed element dtypes — closing GAP-002.

**Architecture:** `Scalar::{zero,one,from_f64}` become `Result`-returning
(never-panic; scales have no exact zero). Two new `Scalar` variants
(`F8E8M0(u8)`, `F8E6M2(u8)`) carry the raw scale byte with OCP-MX / sk4
decode. Packed element dtypes return `Err`. A build-time guard rejects
`MaskedFill` on packed/scale dtypes so the two infallible autograd call sites'
`Err` branches are provably unreachable. Exhaustiveness is enforced by
wildcard-free matches plus a reflection-derived `DType::ALL` never-panic sweep.

**Tech Stack:** Rust (edition 2024), `thiserror` (already in `fuel-ir`),
`strum` (new — see Global Constraints), `half`/`float8` (already used by
`Scalar`).

**Design spec:** `docs/superpowers/specs/2026-08-07-scalar-dtype-completion-design.md`

## Global Constraints

- **Build per-crate only:** `cargo build -p fuel-ir`, `cargo build -p fuel-graph`,
  `cargo test -p fuel-ir`. Never workspace-wide. One cargo invocation at a time.
- **Extend the existing error type.** Add variants to `fuel_ir::Error`
  (`fuel-ir/src/error.rs`) — do NOT introduce a separate `ScalarError`. The
  constructors return `fuel_ir::Result<Scalar>` (= `Result<Scalar, Error>`).
  This is a deliberate deviation from the spec's `ScalarError`, to match the
  crate's single-`Error` convention.
- **One dependency decision:** `DType::ALL` is derived via `strum::EnumIter`
  (adds `strum` to `fuel-ir`). If that dependency is rejected at review, the
  fallback is a declarative macro generating the enum + `ALL` from one list
  (Task 4 notes both). Everything else in the plan is dep-neutral.
- **No wildcard `_ =>` arms** in any `Scalar`/`DType` match in `scalar.rs`.
- **TDD, born-red.** Every behavior gets a failing test watched red first. MX
  numeric expectations come from the reference oracle, never from this document.
- **Numeric authority:** `F8E8M0` per OCP Microscaling spec (definitive);
  `F8E6M2` per the sk4 / kiss-ref reference (its exact reserved-code and
  subnormal bits are pinned by the RED tests, not by this plan's formulas).

---

## File Structure

- `fuel-ir/src/error.rs` — add 4 scalar error variants to `Error`.
- `fuel-ir/src/scalar.rs` — `Result` signatures; 2 new variants; MX decode/encode;
  all match arms; unit tests.
- `fuel-ir/src/dtype.rs` — `#[derive(EnumIter)]`; `DType::ALL`; wildcard-free
  witness for the completeness test.
- `fuel-ir/Cargo.toml` — add `strum` (feature `derive`).
- `fuel-graph/src/runtime_fused.rs` — fix `masked_fill_scalar` (`.ok()`) and the
  `frozen_legacy_reduce_max_to_backward_decompose` call site (self-return on Err).
- `fuel-graph/src/lib.rs` — fix the `MaskedFill` backward site (`debug_assert!` +
  skip on the D5-unreachable Err).
- `fuel-graph/src/…` (graph-build validation module) — D5 `MaskedFill` dtype guard.
- `docs/architecture/…` + `docs/architecture/10-decisions-log.md` — doc update.

---

## Task 1: `Result` constructors + never-panic (closes the violation)

**Files:**
- Modify: `fuel-ir/src/error.rs` (add variants)
- Modify: `fuel-ir/src/scalar.rs:22-88` (three constructors)
- Modify: `fuel-graph/src/runtime_fused.rs:510-517` (`masked_fill_scalar`)
- Modify: `fuel-graph/src/runtime_fused.rs:2699` (frozen-legacy decompose)
- Modify: `fuel-graph/src/lib.rs:8434` (MaskedFill backward)
- Test: `fuel-ir/src/scalar.rs` (`#[cfg(test)]` module)

**Interfaces:**
- Produces: `Scalar::zero(DType) -> fuel_ir::Result<Scalar>`,
  `Scalar::one(DType) -> fuel_ir::Result<Scalar>`,
  `Scalar::from_f64(f64, DType) -> fuel_ir::Result<Scalar>`.
- Produces: `Error::{NoZeroScalar(DType), NoOneScalar(DType),
  ScalarUnrepresentable(DType, f64), PackedElementHasNoScalar(DType)}`.
- Consumes: nothing new.

- [ ] **Step 1: Write the failing test** (`fuel-ir/src/scalar.rs`)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ctors_return_err_not_panic_for_subbyte_dtypes() {
        for dt in [DType::F6E2M3, DType::F6E3M2, DType::F4] {
            assert!(matches!(Scalar::zero(dt), Err(_)), "{dt:?} zero");
            assert!(matches!(Scalar::one(dt), Err(_)), "{dt:?} one");
            assert!(matches!(Scalar::from_f64(1.0, dt), Err(_)), "{dt:?} from_f64");
        }
    }

    #[test]
    fn ctors_ok_for_real_dtypes() {
        assert_eq!(Scalar::zero(DType::F32).unwrap(), Scalar::F32(0.0));
        assert_eq!(Scalar::one(DType::I64).unwrap(), Scalar::I64(1));
        assert_eq!(Scalar::from_f64(-1.0, DType::F16).unwrap(),
                   Scalar::F16(f16::from_f64(-1.0)));
    }
}
```

- [ ] **Step 2: Run to verify it fails** — `cargo test -p fuel-ir scalar::tests`
  Expected: FAIL to compile (constructors still return `Self`, not `Result`).

- [ ] **Step 3: Add the error variants** (`fuel-ir/src/error.rs`, inside `enum Error`, in the `=== DType Errors ===` group):

```rust
    #[error("no zero scalar for dtype {0:?} (scale format has no exact zero)")]
    NoZeroScalar(DType),
    #[error("no one scalar for dtype {0:?}")]
    NoOneScalar(DType),
    #[error("value {1} not representable as a scalar of dtype {0:?}")]
    ScalarUnrepresentable(DType, f64),
    #[error("dtype {0:?} is a packed element format with no scalar representation")]
    PackedElementHasNoScalar(DType),
```

- [ ] **Step 4: Change the three constructors to `Result`** (`fuel-ir/src/scalar.rs`). For now BOTH scales and packed dtypes return `Err` (scales get real values in Tasks 2–3). Real dtypes wrap in `Ok`. Example for `zero` (mirror for `one`/`from_f64`):

```rust
    pub fn zero(dtype: DType) -> crate::Result<Self> {
        Ok(match dtype {
            DType::U8 => Scalar::U8(0),
            DType::I8 => Scalar::I8(0),
            DType::U32 => Scalar::U32(0),
            DType::I16 => Scalar::I16(0),
            DType::I32 => Scalar::I32(0),
            DType::I64 => Scalar::I64(0),
            DType::BF16 => Scalar::BF16(bf16::ZERO),
            DType::F16 => Scalar::F16(f16::ZERO),
            DType::F32 => Scalar::F32(0.0),
            DType::F64 => Scalar::F64(0.0),
            DType::F8E4M3 => Scalar::F8E4M3(f8e4m3::ZERO),
            // Scales: no exact zero (Task 2/3 keep this Err for zero()).
            DType::F8E8M0 | DType::F8E6M2 => {
                return Err(crate::Error::NoZeroScalar(dtype))
            }
            // Packed element formats: no scalar representation.
            DType::F6E2M3 | DType::F6E3M2 | DType::F4 => {
                return Err(crate::Error::PackedElementHasNoScalar(dtype))
            }
        })
    }
```
For `from_f64`, the scale arms also return `Err(PackedElementHasNoScalar)` is
WRONG — use `Err(crate::Error::NoOneScalar(dtype))`? No: for `from_f64` the
interim scale arm returns `Err(crate::Error::ScalarUnrepresentable(dtype, v))`
(Task 2/3 replace it with real rounding). Packed arms →
`Err(PackedElementHasNoScalar(dtype))` in all three. For `one`, scale arm interim
→ `Err(NoOneScalar(dtype))` (Task 2/3 replace with the real `2^0` byte).

- [ ] **Step 5: Fix `masked_fill_scalar`** (`fuel-graph/src/runtime_fused.rs:510-517`):

```rust
fn masked_fill_scalar(value: f64, dtype: fuel_ir::DType) -> Option<fuel_ir::Scalar> {
    // Honest miss for any dtype without a scalar rep; the Result collapses
    // the old hand-written dummy-dtype guard.
    fuel_ir::Scalar::from_f64(value, dtype).ok()
}
```

- [ ] **Step 6: Fix the frozen-legacy decompose** (`fuel-graph/src/runtime_fused.rs`, at the `Op::MaskedFill { value: Scalar::one(dtype) }` construction ~line 2699). This fn returns `NodeId`; on `Err` self-return `id` (decompose-fixpoint / surfaced-gap convention — a reduce-max backward over a non-real dtype is not a thing this frozen oracle handles):

```rust
        let fill = match Scalar::one(dtype) {
            Ok(s) => s,
            Err(_) => return id, // non-real dtype: return node unchanged (opaque)
        };
        // ... use `fill` in Op::MaskedFill { value: fill } ...
```

- [ ] **Step 7: Fix the MaskedFill backward** (`fuel-graph/src/lib.rs:8434`, inside `pub fn backward(&self) -> GradMap`). This fn is infallible; the `Err` branch is made unreachable by Task 5 (D5). Handle without panicking in release:

```rust
                Op::MaskedFill { value: _ } => {
                    let x = inputs[0];
                    let mask = inputs[1];
                    let x_shape = node_shape(&graph_handle, x);
                    let dtype = node_dtype(&graph_handle, x);
                    let zero = match fuel_ir::Scalar::zero(dtype) {
                        Ok(z) => z,
                        Err(_) => {
                            // D5 (Task 5) forbids MaskedFill on packed/scale dtypes,
                            // so this is unreachable. debug_assert catches a D5 gap
                            // in debug; release skips the gradient (never-panic).
                            debug_assert!(false, "MaskedFill backward on non-real dtype {dtype:?} — D5 gap");
                            continue;
                        }
                    };
                    let grad_x = push_node(
                        &graph_handle, Op::MaskedFill { value: zero },
                        vec![up_id, mask], x_shape, dtype,
                    );
                    accumulate_grad(&mut upstream, x, grad_x, &graph_handle);
                }
```
(If the backward loop is not a `for`/`while` where `continue` is valid, use an
`if let ... else { debug_assert!; <skip accumulate> }` block — confirm the loop
form when editing.)

- [ ] **Step 8: Run tests + builds**
  - `cargo test -p fuel-ir scalar::tests` → PASS
  - `cargo build -p fuel-graph` → compiles (all call sites fixed)

- [ ] **Step 9: Commit**

```bash
git add fuel-ir/src/error.rs fuel-ir/src/scalar.rs fuel-graph/src/runtime_fused.rs fuel-graph/src/lib.rs
git commit -m "fix(scalar)!: Result constructors, never-panic over all dtypes (GAP-002 T1)"
```

---

## Task 2: Real `F8E8M0` scale variant (OCP-MX)

**Files:**
- Modify: `fuel-ir/src/scalar.rs` (variant, arms, decode/encode, tests)

**Interfaces:**
- Produces: `Scalar::F8E8M0(u8)`; `to_f64`/`dtype`/`PartialEq` arms; real
  `one(F8E8M0)`, `from_f64(_, F8E8M0)`; `zero(F8E8M0)` stays `Err(NoZeroScalar)`.

- [ ] **Step 1: Write the failing test** (reference values from the OCP MX spec):

```rust
    #[test]
    fn f8e8m0_decode_matches_ocp() {
        // value = 2^(X - 127); X = 255 => NaN. No zero, no negatives.
        assert_eq!(Scalar::F8E8M0(127).to_f64(), 1.0);          // 2^0
        assert_eq!(Scalar::F8E8M0(128).to_f64(), 2.0);          // 2^1
        assert_eq!(Scalar::F8E8M0(126).to_f64(), 0.5);          // 2^-1
        assert!(Scalar::F8E8M0(255).to_f64().is_nan());
    }

    #[test]
    fn f8e8m0_roundtrip_all_finite_bytes() {
        for x in 0u8..=254 {
            let v = Scalar::F8E8M0(x).to_f64();
            assert_eq!(Scalar::from_f64(v, DType::F8E8M0).unwrap(),
                       Scalar::F8E8M0(x), "byte {x}");
        }
    }

    #[test]
    fn f8e8m0_one_and_no_zero() {
        assert_eq!(Scalar::one(DType::F8E8M0).unwrap(), Scalar::F8E8M0(127));
        assert!(matches!(Scalar::zero(DType::F8E8M0),
                         Err(crate::Error::NoZeroScalar(DType::F8E8M0))));
    }
```

- [ ] **Step 2: Run to verify it fails** — `cargo test -p fuel-ir scalar::tests::f8e8m0`
  Expected: FAIL to compile (`Scalar::F8E8M0` does not exist).

- [ ] **Step 3: Add the variant + arms.** In `enum Scalar` add `F8E8M0(u8),`.
  The compiler now errors on every non-exhaustive `Scalar` match — add arms:
  - `dtype()` → `Scalar::F8E8M0(_) => DType::F8E8M0`
  - `to_f64()` → `Scalar::F8E8M0(x) => if *x == 255 { f64::NAN } else { 2f64.powi(*x as i32 - 127) }`

- [ ] **Step 4: Real constructors for `F8E8M0`.** In `one`: `DType::F8E8M0 => Scalar::F8E8M0(127)`. In `from_f64`, replace the scale interim `Err` for `F8E8M0` with nearest-power-of-two encode:

```rust
    DType::F8E8M0 => {
        if !v.is_finite() || v <= 0.0 {
            return Err(crate::Error::ScalarUnrepresentable(dtype, v));
        }
        let x = v.log2().round() as i32 + 127;
        if !(0..=254).contains(&x) {
            return Err(crate::Error::ScalarUnrepresentable(dtype, v));
        }
        Scalar::F8E8M0(x as u8)
    }
```
Leave `zero(F8E8M0)` as `Err(NoZeroScalar)`.

- [ ] **Step 5: Sabotage-check the tolerance.** Temporarily change `- 127` to
  `- 126` in `to_f64`; `cargo test -p fuel-ir scalar::tests::f8e8m0` MUST go red
  AFTER recompilation. Revert.

- [ ] **Step 6: Run tests** — `cargo test -p fuel-ir scalar::tests::f8e8m0` → PASS

- [ ] **Step 7: Commit**

```bash
git add fuel-ir/src/scalar.rs
git commit -m "feat(scalar): real F8E8M0 scale variant with OCP-MX decode (GAP-002 T2)"
```

---

## Task 3: Real `F8E6M2` scale variant (sk4 reference)

**Files:**
- Modify: `fuel-ir/src/scalar.rs` (variant, arms, decode/encode, tests)

**Interfaces:**
- Produces: `Scalar::F8E6M2(u8)`; arms; real `one`/`from_f64`; `zero` stays `Err`.

- [ ] **Step 1: Write the failing test — expectations FROM THE REFERENCE.**
  Fetch the sk4 / kiss-ref `F8E6M2` reference vectors (unsigned, 6 exp / 2
  mantissa, bias 31). Encode 4–6 exact reference pairs `(byte, value)` and the
  reserved-code (`E=63`) behavior AS THE REFERENCE DEFINES THEM — not from the
  formula below:

```rust
    #[test]
    fn f8e6m2_decode_matches_sk4_reference() {
        // (byte, expected) pairs copied from the sk4/kiss-ref reference vectors.
        // Fill from the reference at implementation time — do NOT derive here.
        let cases: &[(u8, f64)] = &[ /* e.g. (0x_?, 1.0), ... */ ];
        for &(b, want) in cases {
            assert_eq!(Scalar::F8E6M2(b).to_f64(), want, "byte {b:#x}");
        }
    }

    #[test]
    fn f8e6m2_roundtrip_all_finite_bytes() {
        for b in 0u8..=255 {
            let v = Scalar::F8E6M2(b).to_f64();
            if v.is_finite() {
                assert_eq!(Scalar::from_f64(v, DType::F8E6M2).unwrap(),
                           Scalar::F8E6M2(b), "byte {b:#x}");
            }
        }
    }
```

- [ ] **Step 2: Run to verify it fails** — expected: FAIL to compile.

- [ ] **Step 3: Add the variant + arms** (`F8E6M2(u8)`, `dtype()`, `to_f64()`).
  Proposed decode shape (CONFIRM/adjust against the reference in Step 1):
  bias 31; normals `E∈1..=62 → 2^(E-31)·(1 + M/4)`; subnormal `E=0 → 2^-30·(M/4)`;
  `E=63` reserved (NaN/Inf per reference). Bit layout: `[eeeeee mm]`.

- [ ] **Step 4: Real constructors** — `one(F8E6M2)` = the byte for `2^0`
  (`E=31, M=0`); `from_f64` = nearest representable, `Err(ScalarUnrepresentable)`
  on negative / NaN / out-of-range; `zero(F8E6M2)` stays `Err(NoZeroScalar)`.

- [ ] **Step 5: Sabotage-check** — perturb the mantissa scale (`M/4` → `M/2`);
  the decode test MUST go red after recompilation. Revert.

- [ ] **Step 6: Run tests** — `cargo test -p fuel-ir scalar::tests::f8e6m2` → PASS

- [ ] **Step 7: Commit**

```bash
git add fuel-ir/src/scalar.rs
git commit -m "feat(scalar): real F8E6M2 scale variant per sk4 reference (GAP-002 T3)"
```

---

## Task 4: `DType::ALL` + reflection-checked never-panic sweep

**Files:**
- Modify: `fuel-ir/Cargo.toml` (add `strum`)
- Modify: `fuel-ir/src/dtype.rs` (`EnumIter`, `ALL`, witness)
- Test: `fuel-ir/src/scalar.rs` (the sweep)

**Interfaces:**
- Produces: `DType::ALL: &'static [DType]` (reflection-derived).

- [ ] **Step 1: Write the failing test** (`fuel-ir/src/scalar.rs`):

```rust
    #[test]
    fn scalar_ctors_never_unwind_over_all_dtypes() {
        use std::panic::{catch_unwind, AssertUnwindSafe};
        for &dt in DType::ALL {
            assert!(catch_unwind(AssertUnwindSafe(|| Scalar::zero(dt))).is_ok(),
                    "zero() unwound for {dt:?}");
            assert!(catch_unwind(AssertUnwindSafe(|| Scalar::one(dt))).is_ok(),
                    "one() unwound for {dt:?}");
            assert!(catch_unwind(AssertUnwindSafe(|| Scalar::from_f64(1.0, dt))).is_ok(),
                    "from_f64() unwound for {dt:?}");
        }
    }
```

- [ ] **Step 2: Run to verify it fails** — expected: FAIL to compile
  (`DType::ALL` does not exist).

- [ ] **Step 3: Derive `ALL` by reflection.** In `fuel-ir/Cargo.toml`:
  `strum = { version = "0.26", features = ["derive"] }`. In `dtype.rs`:

```rust
    use strum::IntoEnumIterator;
    #[derive(strum::EnumIter, /* existing derives */ ...)]
    pub enum DType { /* unchanged */ }

    impl DType {
        /// Every DType variant, reflection-derived (cannot silently omit one).
        pub fn all() -> impl Iterator<Item = DType> { DType::iter() }
        pub const ALL_LEN: usize = /* see witness below */;
    }
```
  Because `const` cannot call the iterator, expose `ALL` as a function or a
  `once_cell`/`LazyLock` slice, OR keep the sweep iterating `DType::iter()`
  directly and drop the `&[DType]` form. Simplest: change the test to
  `for dt in DType::iter()`. Adjust the interface line accordingly.

  **Fallback if `strum` is rejected:** define `pub const ALL: &[DType] = &[..16..]`
  by hand AND add a wildcard-free witness whose exhaustiveness the compiler
  enforces:

```rust
    #[cfg(test)]
    fn _dtype_witness(dt: DType) { match dt {
        DType::U8|DType::I8|DType::U32|DType::I16|DType::I32|DType::I64
        |DType::BF16|DType::F16|DType::F32|DType::F64|DType::F8E4M3
        |DType::F8E8M0|DType::F8E6M2|DType::F4|DType::F6E2M3|DType::F6E3M2 => {}
        // no wildcard: adding a variant fails to compile here
    }}
```
  and a test asserting every `ALL` entry is covered and `ALL.len()` equals the
  count the witness enumerates. (Reflection is preferred precisely because it
  removes this hand list.)

- [ ] **Step 4: Run tests** — `cargo test -p fuel-ir scalar::tests` → PASS

- [ ] **Step 5: Commit**

```bash
git add fuel-ir/Cargo.toml fuel-ir/src/dtype.rs fuel-ir/src/scalar.rs
git commit -m "test(scalar): reflection-derived DType enumeration + never-panic sweep (GAP-002 T4)"
```

---

## Task 5: Build-time `MaskedFill` dtype guard (D5)

**Files:**
- Modify: `fuel-graph/src/…` (the graph-build validation path — locate the
  existing build-time validator; MaskedFill node construction/validation)
- Test: alongside the validator

**Interfaces:**
- Produces: a graph-build `Result` error when a `MaskedFill` node's dtype is a
  packed/scale format (`F4`/`F6E2M3`/`F6E3M2`/`F8E8M0`/`F8E6M2`).

- [ ] **Step 1: Find the validator.** Locate where ops are validated at
  graph-build (search `fuel-graph` for the build-time validation entry point that
  already returns `fuel_ir::Result`). MaskedFill is validated there.

- [ ] **Step 2: Write the failing test** — building a `MaskedFill` node whose
  dtype is `DType::F4` returns `Err`; a `DType::F32` MaskedFill returns `Ok`.

- [ ] **Step 3: Run to verify it fails** — the F4 case currently builds `Ok`.

- [ ] **Step 4: Implement the guard** — in MaskedFill validation, reject a
  packed/scale dtype with a clear `Error` (`UnsupportedDTypeForOp(dtype, "MaskedFill")`).

- [ ] **Step 5: Run tests** — validator tests PASS.

- [ ] **Step 6: Tighten the backward comment** — update the `debug_assert!` note
  at `lib.rs:8434` and the self-return at `runtime_fused.rs:2699` to cite this
  guard as the reachability proof.

- [ ] **Step 7: Commit**

```bash
git add fuel-graph/src
git commit -m "feat(graph): build-time guard rejecting MaskedFill on packed/scale dtypes (GAP-002 T5/D5)"
```

---

## Task 6: Docs (architecture + decisions log)

**Files:**
- Modify: the relevant `docs/architecture/` section (DType/Scalar model + never-panic)
- Modify: `docs/architecture/10-decisions-log.md`

- [ ] **Step 1:** Record the DType-superset-of-Scalar model, the scale-vs-packed
  split, and `Result` constructors. Bump the section version; add a decisions-log
  entry. Coordinate the scale-classification wording with the sk4 RFC so the two
  do not diverge.

- [ ] **Step 2: Commit**

```bash
git add docs/architecture
git commit -m "docs(architecture): Scalar/DType completion + never-panic (GAP-002 T6)"
```

---

## Self-Review

- **Spec coverage:** D1 (T1), D2 (T2+T3), D3 (T1 packed `Err`), D4 (T4), D5 (T5),
  D6 (T6). All six decisions have a task.
- **Placeholders:** the only intentionally-deferred content is the `F8E6M2`
  reference vectors (Task 3 Step 1) and the exact validator location (Task 5
  Step 1) — both are "read the authoritative source at implementation time,"
  not invented values, which the spec mandates.
- **Type consistency:** constructors return `fuel_ir::Result<Scalar>` throughout;
  error variants named consistently (`NoZeroScalar`, `NoOneScalar`,
  `ScalarUnrepresentable`, `PackedElementHasNoScalar`); `Scalar::F8E8M0(u8)` /
  `F8E6M2(u8)` used identically in Tasks 2–4.
- **Ordering:** T1 first (compiles + closes never-panic). T2/T3 independent, both
  after T1. T4 after T1. T5 after T1 (references its call sites). T6 last.
