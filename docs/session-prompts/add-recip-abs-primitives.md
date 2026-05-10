# Session prompt — Add `Op::Recip` and `Op::Abs` as primitive ops

## What this session is for

Two primitive elementwise ops are declared in `OpKind` (the dispatch-table key enum) but have no `Op` enum variant, no graph-builder method, and no kernel anywhere: `OpKind::RecipElementwise` (`1/x`) and `OpKind::AbsElementwise` (`|x|`). Today, models that need either of these have to express them as workarounds: `recip(x)` as `const_1 / x` (one extra div + materialized constant), `abs(x)` as `maximum(x, -x)` (two ops, both touching memory). Both are real primitives — every modern hardware has dedicated reciprocal and abs instructions, and ML frameworks expose them as first-class — and exposing them as `Op` variants unlocks single-instruction kernels and cleaner gradients.

This session adds both ops end-to-end across the stack: `Op` variant + `Tensor` method + reference impl + CPU kernel + binding-table registration + `LazyTensor` wrapper + Judge `PROFILED_OPS` entry. CUDA / Vulkan / AOCL / MKL kernels are stretch goals — the architecture's `bit_stable` coverage commitment only requires one always-built backend (cpu+reference); GPU kernels can land in follow-up sessions.

This session is parallel-safe with Phase 7.6, CUDA Tier 1 work, and Judge dtype expansion. It touches the `Op` enum (so will conflict mechanically with any other session that's adding new `Op` variants), but no two such sessions should be running at once anyway.

## Read first (in this order)

1. **`docs/architecture/00-index.md`** + **`docs/architecture/01-identity.md`** — grounding.
2. **`docs/architecture/03-ir.md`** — particularly the `Op` enum's role and the rule that "every primitive op has a single canonical `Op` variant; fused ops live in the registry." Recip and Abs are primitives by every reasonable definition; they belong in `Op`.
3. **`docs/architecture/05-backend-contract.md`** — `PrecisionGuarantee` per kernel + binding-table registration shape. Recip / Abs are bit-stable on every backend that does the obvious thing (no transcendental approximation needed).
4. **`fuel-graph/src/lib.rs`** §`pub enum Op` — read the existing unary entries (Neg, Sqr, Sqrt, Exp, Log, Sin, Cos, Tanh, Sigmoid, Silu, Gelu, Relu, Step) for the pattern. Recip + Abs are added the same way.
5. **`fuel-graph/src/lib.rs`** §`unary_op` and the unary `Tensor` methods (e.g. `pub fn neg`, `pub fn sqr`) — same pattern repeats for recip + abs.
6. **`fuel-graph/src/lib.rs`** §`Tensor::backward` — read the backward arms for `Op::Sqr`, `Op::Sqrt`, `Op::Log` for examples of how unary backward rules thread through. Backward rules:
   - `d/dx(1/x) = -1/x²`. Implementable as `let g_x = -upstream * recip(x).sqr()` (or `recip(x) * recip(x)`). Forward output `1/x` is also useful: `d(1/x)/dx = -(1/x)² = -fwd_out²`. Either formulation is fine; the latter saves one recip recomputation if the forward output is in scope.
   - `d/dx(|x|) = sign(x)`, with `sign(0)` taken as 0 by convention (subgradient). Implementable as `step(x) - step(-x)` (uses existing `Op::Step`); also expressible as `Op::Cast`-then-multiply schemes if a `sign` primitive is preferred. Don't introduce `Op::Sign` in this session — the `step(x) - step(-x)` form is fine.
7. **`fuel-graph/src/opt.rs`** §`fn op_key` — each new `Op` variant gets a unique tag for CSE keying. Pick the next available number after the current largest tag.
8. **`fuel-reference-backend/src/exec.rs`** — find existing unary arms (e.g. `Op::Neg => out.iter_mut().for_each(...)`); add `Op::Recip` and `Op::Abs` at the same site.
9. **`fuel-graph-cpu/src/...`** — find the CPU kernel registration sites for unary ops; add recip + abs the same way.
10. **`fuel-storage/src/dispatch.rs`** — find the binding-table entries for an existing unary op (e.g. `Neg` → `neg_f32_cpu_wrapper`); register Recip and Abs the same way. Each will be one `KernelRef` per `(OpKind, [DType], BackendId)` tuple.
11. **`fuel-core/src/lazy.rs`** — read the existing unary `LazyTensor` methods; add `recip()` and `abs()` matching that pattern.
12. **`fuel-core/src/judge.rs`** — append `OpKind::RecipElementwise` and `OpKind::AbsElementwise` to `PROFILED_OPS`, add to the elementwise-unary `size_plan` arm, add to the `is_unary_elementwise` matcher, add to the `apply_unary` match. Recip's input needs to be shifted away from zero (use the same `+ 1.5` trick as `sqrt`/`log`).

## What this session must NOT do

- **Don't introduce `Op::Sign`.** Abs's backward uses `step(x) - step(-x)`; `Op::Step` already exists. A standalone `Op::Sign` is its own scope.
- **Don't replace existing workarounds in `fuel-transformers` or anywhere else automatically.** This session ships the primitives; migrations to use them are separate. Files that currently express abs as `maximum(x, -x)` keep working as-is.
- **Don't ship a recip kernel that handles divide-by-zero by silently producing `inf` differently from `1.0/x`'s IEEE-correct behavior.** The reference impl is `1.0 / x` (or whatever the obvious f32 expression is); per architecture v1.0's `bit_stable` commitment, the CPU backend matches that exactly.
- **Don't push to remote.**

## Branch and starting state

- **Current branch (at session start)**: `feature/storage-unification`. Verify with `git log --oneline -5`. The tip will be later than `6a279eb5` (the LazyTensor wrapper cleanup commit).
- **Coordination**: parallel session may be running CUDA Tier 1 fanout, Phase 7.6 step 4+ work, or Judge dtype expansion. None of those touch the `Op` enum's primitive list at the same site recip+abs do; merge conflicts are unlikely.

## Concrete work — minimum viable scope

One commit per layer, in dependency order:

### Commit 1: `Op::Recip` and `Op::Abs` enum variants + Tensor methods

In `fuel-graph/src/lib.rs`:

1. Add `Op::Recip` and `Op::Abs` to the `pub enum Op` — slot them next to other unary ops (after `Step`, alphabetically ordered or in the family-grouped pattern of the existing entries; match the existing convention).
2. Add `Tensor::recip()` and `Tensor::abs()` methods — same pattern as `Tensor::neg`:
   ```rust
   pub fn recip(&self) -> Tensor {
       self.unary_op(Op::Recip)
   }
   pub fn abs(&self) -> Tensor {
       self.unary_op(Op::Abs)
   }
   ```
3. Update `op_short_name` to include `"Recip"` and `"Abs"`.
4. Update `Tensor::backward` to emit gradients:
   - Recip: `g_x = -upstream * recip(x).sqr()` (or `-upstream * fwd_out * fwd_out` if forward output is available).
   - Abs: `g_x = upstream * (step(x) - step(-x))` — both `Op::Step` and `Op::Sub` already exist.
5. Update `fuel-graph/src/opt.rs`'s `op_key` to assign Recip and Abs unique tags (next available numbers).

Test: existing fuel-graph unit tests should keep passing. Add explicit unit tests:
- `recip_forward_returns_inverse` — `recip(2.0) == 0.5`.
- `recip_backward_matches_minus_x_squared` — pick a value, compare against `-1/x²` analytical.
- `abs_forward_returns_magnitude` — `abs(-3.0) == 3.0` and `abs(3.0) == 3.0`.
- `abs_backward_matches_sign` — gradient at `x = -2` is `-1`, at `x = 2` is `+1`, at `x = 0` is `0` (convention).

### Commit 2: reference + CPU backend kernels

1. `fuel-reference-backend/src/exec.rs` — add `Op::Recip` and `Op::Abs` arms in the unary-op dispatch match.
2. `fuel-graph-cpu/src/...` — find the CPU unary kernels file and add the wrappers + binding-table registration.
3. `fuel-storage/src/dispatch.rs` — register `(RecipElementwise, [F32], Cpu)` and `(AbsElementwise, [F32], Cpu)` (+ same for `[F64]`, `[BF16]`, `[F16]` if those exist for other unary ops on CPU).

Test: `cargo test -p fuel-graph` and `cargo test -p fuel-graph-cpu` and `cargo test -p fuel-reference-backend`. Add an integration test in fuel-core or fuel-graph-cpu that realizes a small `recip` graph and an `abs` graph through both the reference and the CPU backend, confirming bit-equivalence.

### Commit 3: LazyTensor wrappers + Judge coverage

1. `fuel-core/src/lazy.rs` — add `LazyTensor::recip` and `LazyTensor::abs` (one-line wrappers).
2. `fuel-core/src/judge.rs`:
   - Append `OpKind::RecipElementwise` and `OpKind::AbsElementwise` to `PROFILED_OPS`.
   - Add them to the elementwise-unary fanout group in `size_plan`.
   - Add them to the `is_unary_elementwise` matcher.
   - Add them to `apply_unary`.
   - Recip input needs to be away from zero — extend the `needs_positive` predicate (or rename to `needs_nonzero`) so `RecipElementwise` gets the `+1.5` shift too. Abs takes any f32 (no shift needed).
3. Extend `judge_profiles_all_unary_elementwise_ops` to include the two new kinds.

Test: `cargo test -p fuel-core --lib judge_`. All judge tests stay green; the unary fanout test now covers 15 ops instead of 13.

### Commit 4 (optional, stretch): GPU kernels

If the session has remaining time and a GPU backend's unary kernel set is registered in this repo (fuel-cuda-backend, fuel-vulkan-kernels), add Recip and Abs there too. Each is a one-line PTX/SPIR-V snippet plus a binding-table registration entry. Live tests via `cargo test -p fuel-core --features cuda --lib judge_cuda_ -- --ignored` (per `project_dev_environment.md`).

## Test commands

After each commit:

```bash
cargo test -p fuel-graph --lib
cargo test -p fuel-graph-cpu --lib
cargo test -p fuel-reference-backend --lib
cargo test -p fuel-core --lib
```

Live GPU tests after commit 4 (if attempted):

```bash
cargo test -p fuel-core --features cuda --lib judge_cuda_ -- --ignored
```

## Operating principles

- **Engage critically.** If the backward rule for Abs at `x = 0` matters for a real consumer, surface it; the convention-of-0 is fine but not load-bearing — flag if anyone's training on a model where the choice changes results.
- **No production panics.** Recip's runtime behavior on `x = 0` is `1.0 / 0.0 = inf` (IEEE-correct), not panic. Abs is total. Neither needs error returns.
- **Parity over performance.** Bit-stable on cpu+reference is the architecture commitment; GPU performance is opportunistic. If a CUDA recip kernel uses `__frcp_rn` and produces 1-ULP-off results vs reference, that's `PrecisionGuarantee::Approx { max_relative: 1e-7 }` not a bug — but the cpu+reference must be exact.
- **Don't push to remote unless asked.**

## End-of-session deliverable

At minimum (Commits 1–3):

- `Op::Recip` and `Op::Abs` exist as `Op` enum variants.
- `Tensor::recip()` / `Tensor::abs()` build graph nodes.
- Reference + CPU kernels green; backward rules tested.
- Binding-table entries registered.
- `LazyTensor::recip` / `abs` available.
- Judge `PROFILED_OPS` includes both; unary fanout test covers 15 kinds.

Stretch (Commit 4): CUDA recip + abs kernels with live-device parity tests.

## Coordination notes

- **Op enum extension** — only one session should be adding `Op` variants at a time to avoid awkward merge conflicts in the enum definition + `op_short_name` + `op_key`. If the user has another session adding `Op` variants, sequence this one after.
- **Phase 7.6** — fully orthogonal. This session adds primitives to the closed `Op` enum; Phase 7.6 governs the open fused-op registry. No interaction.
- **Judge dtype expansion** — orthogonal. Each can land independently; merge conflict in `judge.rs` is mechanical (separate match arms).
- **CUDA Tier 1 fanout resume** — orthogonal as long as Tier 1 is not in the middle of touching unary registration. If it is, sequence after.
