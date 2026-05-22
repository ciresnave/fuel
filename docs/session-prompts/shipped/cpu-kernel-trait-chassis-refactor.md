# Session prompt — CPU kernel trait-chassis refactor

## What this session is for

Refactor the CPU kernel surface in `fuel-cpu-backend/src/byte_kernels.rs` from its current shape — concrete per-(op, dtype) functions plus a handful of textual macros (`sum_reduce_half!`, `reduce_half_extremum!`, `mean_reduce_half!`) — onto a trait-chassis pattern. The chassis carries the shape/stride/loop logic once per kernel family; op-specific traits carry the math (~3–5 lines each); per-dtype impls only carry the bits that actually change per dtype (accumulator type for low-precision floats, integer-overflow policy, etc.).

The point isn't aesthetics — it's structural drift reduction and a path to scaling native (op × dtype) coverage cheaply. With the chassis in place, adding a new dtype to a kernel family is "implement these 3 trait methods"; adding a new op is "implement this trait + register the public entry points." Today both are multi-line copy-paste from a sibling kernel, which is how drift creeps in.

This is the foundation that lets the broader "native kernel for every (op × dtype × backend) where the device supports the dtype" goal scale without combinatorial maintenance burden. Pairs with the cast-fusion optimizer rule (see `docs/session-prompts/cast-fusion-optimizer-rule.md`) — together they cover both the high-leverage native cells and the long tail.

This session is **parallel-safe with any session that adds new ops**, since refactor is at the same call site (`sum_reduce_f32` etc.) and existing entry-point signatures don't change. It is **NOT parallel-safe with another session that refactors the same kernel family** for obvious reasons.

## Read first (in this order)

1. **`docs/architecture/05-backend-contract.md`** — particularly the `PrecisionGuarantee` per kernel and the always-built `bit_stable` coverage commitment. The refactor preserves both; what changes is how the kernels are *written*, not what they *guarantee*.
2. **`docs/architecture/10-decisions-log.md`** — decision #11 (precision per kernel) and the dtype-coverage philosophy. The chassis encodes those decisions instead of relying on per-(op, dtype) authors getting them right.
3. **`fuel-cpu-backend/src/byte_kernels.rs`** §reduction kernels (lines ~4700–5200 at session-start) — the area to refactor first. Read the existing `reduce_f64_generic` helper (line ~5044) and `sum_reduce_half!` macro (line ~5121); both are already moving toward the chassis pattern in different ways.
4. **`fuel-storage/src/dispatch.rs`** §reduction registration (lines ~2380–2410, ~4180–4188) — the binding-table entries that call into byte_kernels. **These must keep working unchanged.** The public function names + signatures are the API contract.
5. **`fuel-reference-backend/src/ops.rs`** §`sum_all`, `mean_all`, etc. — already generic-over-T with a `Float` bound. This is the pattern direction for fuel-cpu-backend, but at the byte-kernel layer instead of at the typed `RefTensor<T>` layer.
6. Memory entry `project_judge_coverage_expansion_shipped.md` — current Judge probe coverage. Per-(op, dtype) tests for accumulating/lossy kernels stay; the chassis reduces structural drift, not numerical drift.
7. Memory entry `feedback_architectural_cleanness_over_local_pragmatism.md` — applies here. The refactor is non-trivial; resist the urge to "just leave the existing macros as-is, they work" if you find a structurally cleaner path.

## What this session must NOT do

- **Don't change public function signatures.** `pub fn sum_reduce_f32(input: &CpuStorageBytes, output: &mut CpuStorageBytes, input_shape: &[usize], reduce_dims: &[usize]) -> Result<()>` is the API. The binding-table wrappers in `fuel-storage/src/dispatch.rs` call these by name; changing signatures cascades into the dispatch crate. The chassis lives *behind* these functions.
- **Don't break BF16/F16 accumulator promotion.** The current half-float kernels accumulate in `f32` (deliberately — bf16's 7-mantissa-bit truncation catastrophically loses precision past ~128 elements summed naively). The chassis must preserve this — encode it as the `Sum<bf16>` impl's `Acc = f32` associated type, not "the loop happens to use f32." Document the invariant on the trait.
- **Don't refactor MatMul / Conv2D / attention kernels.** Those are specialized with hand-tuned inner loops; the chassis pattern doesn't help them. Stay in elementwise + reduction territory.
- **Don't introduce a new crate** (`fuel-cpu-kernel-chassis`, etc.) in this session. The chassis lives in `fuel-cpu-backend/src/chassis/` (or a single new module file). If AOCL/MKL backends later want to share it, that's a follow-up extraction; doing it now is speculative coupling.
- **Don't drop the existing per-(op, dtype) tests.** Numerical-correctness tests stay — the chassis reduces *structural* drift, not *numerical* drift. BF16 sum-of-10K-elements still needs its own assert; only the *implementation* it tests is centralized.
- **Don't push to remote.**

## Branch and starting state

- **Current branch**: whatever the user is on at session start. Verify with `git status` + `git log --oneline -5`. Confirm there are no in-flight uncommitted changes touching the same files; if there are, coordinate with the user before starting.
- **Coordination**: parallel sessions adding new ops (`Op::Pow`, `Op::CumSum`, etc.) keep working — they add new entry-point functions following the existing pattern; this session migrates the *existing* functions to call the chassis. Merge conflicts in dispatch.rs are unlikely (registration entries are unchanged); in byte_kernels.rs they're mechanical if both sessions touch the same area.

## Concrete work — sequenced by leverage and risk

The refactor is ~5 PRs. Each is a separable commit (or small series). Start with reductions (highest leverage — 4 op families × 4 dtypes = 16 cells, already partially macro-driven, so the win is biggest). Other families follow once the chassis pattern is validated on reductions.

### PR 1 — Reduction chassis

Design the chassis trait. Suggested shape (refine as you go):

```rust
// fuel-cpu-backend/src/chassis/reduction.rs

/// One reduction operation (sum / max / min / mean / product / ...).
/// Generic over the input element type `T`. The associated `Acc`
/// type is the accumulator — usually `T` itself for f32/f64, but
/// `f32` for bf16/f16 to preserve precision (see the impl for
/// `Sum<bf16>` for the invariant).
pub trait ReduceOp<T: Copy> {
    type Acc: Copy;
    fn init() -> Self::Acc;
    fn fold(acc: Self::Acc, x: T) -> Self::Acc;
    fn finalize(acc: Self::Acc, count: usize) -> T;
}

/// Op-specific markers. These are zero-sized; the trait impls below
/// carry the math.
pub struct Sum;
pub struct Max;
pub struct Min;
pub struct Mean;

// Generic-over-T impls — work for any numeric T satisfying the bounds.
impl<T> ReduceOp<T> for Sum where T: Copy + std::ops::Add<Output = T> + num_traits::Zero {
    type Acc = T;
    fn init() -> T { T::zero() }
    fn fold(acc: T, x: T) -> T { acc + x }
    fn finalize(acc: T, _: usize) -> T { acc }
}

// Per-dtype specialization where dtype changes semantics:
impl ReduceOp<bf16> for Sum {
    type Acc = f32;  // accumulator promotion — invariant per architecture v1.0
    fn init() -> f32 { 0.0 }
    fn fold(acc: f32, x: bf16) -> f32 { acc + x.to_f32() }
    fn finalize(acc: f32, _: usize) -> bf16 { bf16::from_f32(acc) }
}
// ... and same for f16.

// Chassis function — the loop / shape decode / output index lives ONCE.
pub fn reduce<T, R>(
    name: &str,
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
    input_shape: &[usize],
    reduce_dims: &[usize],
) -> Result<()>
where
    T: Copy + Pod,
    R: ReduceOp<T>,
{
    // Existing loop logic from sum_reduce_f64_generic / sum_reduce_half!,
    // factored to call R::init / R::fold / R::finalize. Inputs decoded
    // from CpuStorageBytes via input.as_slice::<T>(); outputs via
    // output.as_slice_mut::<T>(). Shape/stride math unchanged.
}
```

Public entry points become 1-line thunks:

```rust
pub fn sum_reduce_f32(input, output, shape, dims) -> Result<()> {
    reduce::<f32, Sum>("sum_reduce_f32", input, output, shape, dims)
}
pub fn sum_reduce_bf16(input, output, shape, dims) -> Result<()> {
    reduce::<bf16, Sum>("sum_reduce_bf16", input, output, shape, dims)
}
// etc.
```

**Files**:
- New: `fuel-cpu-backend/src/chassis/mod.rs` and `fuel-cpu-backend/src/chassis/reduction.rs`.
- `fuel-cpu-backend/src/byte_kernels.rs` — replace the 16 per-dtype reduce functions with 1-line thunks calling `reduce::<T, OpMarker>`. Delete the `sum_reduce_half!` / `mean_reduce_half!` / `reduce_half_extremum!` / `reduce_f32_generic` / `reduce_f64_generic` helpers — the chassis replaces them.
- `fuel-cpu-backend/src/lib.rs` — declare the `chassis` module.

**Tests**: keep every existing reduction test green (16+ per-dtype numerical tests in `byte_kernels.rs::tests::*`). Add ONE structural test per trait: `reduce_op_sum_f32_zero_init_then_fold_matches_iter_sum`, asserting the trait methods compose correctly. The structural tests prove "if the trait is right, every kernel built on it is right"; the existing per-(op, dtype) tests prove "the trait was right for this dtype."

**Watch for**:
- The existing macros use `concat!(stringify!($name))` for error messages; the chassis takes the name as a parameter — preserve the error-message format so panic outputs don't drift.
- `as_slice::<T>()` on `CpuStorageBytes` must already exist (check `fuel-core-types::cpu_storage`) or the chassis needs to use a different accessor.
- `num_traits::Zero` may not be a current dependency; if not, define a local `Zero` trait with f32/f64/bf16/f16/integer impls or use a different bound that's already satisfied.

**Commit**: `refactor(cpu-backend): introduce reduction chassis trait; ~16 kernels become thunks`.

### PR 2 — ReduceSumTo / ReduceMaxTo chassis

The reduce-to-broadcast-target kernels in `byte_kernels.rs` follow the same pattern as the per-axis reductions but with a different output-shape derivation. After PR 1, this is mechanical: same `ReduceOp<T>` trait, different chassis function (`reduce_to<T, R>`).

8 per-dtype kernels (4 dtypes × 2 ops) collapse to 8 thunks + 2 chassis functions.

**Commit**: `refactor(cpu-backend): ReduceSumTo + ReduceMaxTo on reduction chassis`.

### PR 3 — Elementwise unary chassis

The harder one. Unary ops are dtype-generic on the *math* (`Neg`, `Sqr`) but dtype-specific on *transcendentals* (`Sin`, `Cos`, `Exp`, `Log`, `Tanh`, `Sigmoid` — different intrinsics per dtype). The chassis splits into two layers:

```rust
pub trait UnaryOp<T: Copy> {
    fn apply(x: T) -> T;
}

pub fn unary<T: Copy + Pod, U: UnaryOp<T>>(
    name: &str,
    input: &CpuStorageBytes,
    output: &mut CpuStorageBytes,
) -> Result<()> { /* element-wise loop calling U::apply */ }

// Generic impls for math that's identical across dtypes:
impl<T: Neg<Output = T> + Copy> UnaryOp<T> for NegOp {
    fn apply(x: T) -> T { -x }
}

// Per-dtype specializations for transcendentals:
impl UnaryOp<f32> for SinOp { fn apply(x: f32) -> f32 { x.sin() } }
impl UnaryOp<f64> for SinOp { fn apply(x: f64) -> f64 { x.sin() } }
impl UnaryOp<bf16> for SinOp { fn apply(x: bf16) -> bf16 { bf16::from_f32(x.to_f32().sin()) } }
impl UnaryOp<f16> for SinOp  { fn apply(x: f16) -> f16 { f16::from_f32(x.to_f32().sin()) } }
```

This pattern lets dtype-neutral ops (Neg, Sqr, Recip — assuming `T: Div + One`) have ONE impl covering 4 dtypes. Transcendentals (~10 ops) need 4 impls each but they're each one line. Net: ~15 ops × 4 dtypes = 60 cells collapse to ~30 trait impls (the dtype-neutral ones get one impl total).

**Commit**: `refactor(cpu-backend): elementwise unary chassis; transcendental specializations`.

### PR 4 — Elementwise binary chassis

Same pattern as unary, with a 2-input apply method. Add/Sub/Mul/Div/Max/Min on the math-neutral path; Pow if/when it lands needs per-dtype.

```rust
pub trait BinaryOp<T: Copy> {
    fn apply(a: T, b: T) -> T;
}
```

**Commit**: `refactor(cpu-backend): elementwise binary chassis`.

### PR 5 — Binary-compare chassis (separate from binary because output dtype differs)

Comparison ops emit `U8` regardless of input dtype. They need their own chassis with two type parameters:

```rust
pub trait CompareOp<T: Copy> {
    fn apply(a: T, b: T) -> u8;
}
```

The chassis function handles the F32×F32→U8 etc. shape contract. NaN-unordered policy lives in the per-(op, dtype) impl (or in a default with override).

**Commit**: `refactor(cpu-backend): binary-compare chassis (T → U8 output)`.

## Stretch: integer dtype coverage

Once the chassis lands for any kernel family, **adding integer dtypes to that family is trivial**:

- Sum/Max/Min over `u8` / `u32` / `i16` / `i32` / `i64` → `impl ReduceOp<u8> for Sum { type Acc = u64; ... }` (with deliberate accumulator promotion for u8/i8/i16 to avoid overflow).
- Mean over integers → needs a policy decision (truncate? round? promote to f64?) — flag back to user; don't ship without explicit call. Existing default is "no mean on integers; user must cast first."
- Neg / Abs / Sign on signed integers → `impl UnaryOp<i32> for NegOp` etc.

If session has time after PR 1, ship `u8` sum/max/min on the reduction chassis as the demonstration that the pattern works for integers — unblocks `(a == b).sum_all()` directly without a Cast.

## Test commands

After each PR:

```bash
cargo test -p fuel-cpu-backend --lib
cargo test -p fuel-storage --lib
cargo test -p fuel-core --lib judge_
```

End-of-session sweep:

```bash
cargo check --workspace
cargo test -p fuel-cpu-backend -p fuel-storage -p fuel-core --lib
```

If any judge probe regresses numerically by more than expected (Sum/Mean BF16 in particular is the canary — accumulator promotion is the load-bearing invariant), surface it immediately; don't paper over with looser tolerance.

## Operating principles

- **Engage critically.** If you find the chassis trait shape above isn't quite right for the reality of the existing kernels (e.g., reductions need access to input-shape info inside `fold`, or the `Pod` bound conflicts with `bf16`'s non-`Pod` status), adjust the trait shape and document why. The sketch above is a starting point, not a spec.
- **Bit-stable behavior preserved.** Every numerical assertion in the existing tests must still pass. If a refactor would change behavior even subtly (e.g., reduction order changes alter floating-point rounding), call it out — don't ship.
- **Accumulator-promotion invariant is non-negotiable.** BF16 / F16 sum/mean must accumulate in F32. Encode it as the trait's associated type, not "the implementation happens to do it." A future contributor adding `Sum<i8>` should be unable to forget the accumulator promotion because the trait makes the choice explicit.
- **Don't over-genericize.** If three ops fit a chassis cleanly, ship that chassis; if a fourth would require warping the trait shape, give it its own chassis (or no chassis). Premature unification is worse than well-scoped duplication.
- **Memory updates per PR landed.** A short entry (`project_cpu_reduction_chassis_shipped.md` etc.) capturing what migrated, where the chassis lives, and any landmines (traits that didn't quite fit, etc.) makes future-session pickup trivial.
- **Don't push to remote unless asked.**

## End-of-session deliverable

At minimum (PR 1):

- `fuel-cpu-backend/src/chassis/reduction.rs` exists with `ReduceOp` trait + `Sum`/`Max`/`Min`/`Mean` markers + `reduce<T, R>` chassis function.
- 16 reduction entry-points in `byte_kernels.rs` are 1-line thunks; no more `*_half!` macros or `*_f32_generic`/`*_f64_generic` helpers.
- All existing reduction tests green; one new structural test per trait.
- Binding-table registration unchanged.

Stretch (PRs 2–5):

- Reduce-to chassis (PR 2).
- Unary chassis (PR 3).
- Binary chassis (PR 4).
- Compare chassis (PR 5).
- Integer-dtype coverage on at least one chassis as proof of concept (most likely U8 sum/max/min on the reduction chassis).

## Coordination notes

- **No conflict with op-adding sessions** — those add new entry-point functions following the existing pattern; the chassis lives behind those functions. If a new op lands in `byte_kernels.rs` after PR 1 but before PR 3, it'll be in pre-chassis form; PR 3 picks it up during the unary/binary migration.
- **Pairs with the cast-fusion optimizer rule session** (`docs/session-prompts/cast-fusion-optimizer-rule.md`). Both reduce the maintenance pressure of "native everywhere it matters" — the chassis makes adding cells cheap; the cast-fusion rule means we don't need cells in the long tail. Land independently in either order.
- **GPU backends not in scope.** Slang's interfaces+generics support the same pattern; CUDA `.cu` files support C++ templates. A follow-up session can apply the same chassis pattern to Slang shaders and `.cu` files once the CPU pattern is validated. Hand-tuned PTX kernels (matmul family) are exempt.
