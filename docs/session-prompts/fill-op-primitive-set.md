# Session prompt — Fill the `Op` primitive set + seed `FusedOpRegistry`

## What this session is for

The `Op` enum in `fuel-graph` and the `OpKind` dispatch key in `fuel-core-types::dispatch` are partially populated. Audit (post-Recip/Abs landing, tip `896f97da`) found three tiers of missing ops, all surfaced because `OpKind`/`UnaryOp`/`CmpOp` already declare slots for many of them but no graph-level surface exists. This session adds them all — comparison family + `Where` first (because it unblocks everything else), then the mechanical Tier-1 unary fanout, then the irreducible Tier-2 primitives, then a batch of Tier-3 fused ops registered through `FusedOpRegistry` (Phase 7.6's architectural target).

End goal: every standard ML primitive a transformer / vision / SSM model needs has a clean graph-level expression, with bit-stable cpu+reference kernels for everything Tier-1/Tier-2 and decomposed-or-native kernels for Tier-3. Closes the "missing primitive → workaround in user code → kernel-launch tax" pattern across the board.

This session is **NOT parallel-safe** with anything else that touches the `Op` enum (a CUDA Tier-1 fanout for an op already in the enum is fine; another session adding new variants is not). Stage cleanly behind any in-flight `Op`-extending sessions.

## Read first (in this order)

1. **`docs/architecture/03-ir.md`** — particularly the closed-`Op` + open-`FusedOpRegistry` split. This session lives at the boundary: Tier 1+2 grow `Op`; Tier 3 grows the registry.
2. **`docs/architecture/05-backend-contract.md`** — `PrecisionGuarantee` + the always-built `bit_stable` coverage commitment. Every primitive added here must have a `bit_stable` cpu kernel.
3. **`docs/session-prompts/add-recip-abs-primitives.md`** + the four commits at branch tip (`23febb87`, `3cd970d8`, `c34f61d7`, `896f97da`) — this is the *exact* shape every Tier-1 unary follows. Read the commits to see how mechanical the wiring is when the byte-kernel infrastructure already exists.
4. **`fuel-graph/src/lib.rs`** §`pub enum Op` (line ~185) — the canonical primitive set you're extending.
5. **`fuel-core-types/src/op.rs`** — `UnaryOp`, `BinaryOp`, `CmpOp`, `ReduceOp`. Several Tier-1 ops are already declared here; you're filling the `Op`-side gap.
6. **`fuel-core-types/src/dispatch.rs`** §`pub enum OpKind` — extends to track new families. Mark with `#[non_exhaustive]` (already is) so persisted profiles parse forward.
7. **`fuel-graph/src/registry.rs`** — `FusedOpRegistry` skeleton from Phase 7.6 step 1. Tier 3 entries register here; cross-reference `docs/fused-op-registry.md`.
8. **`fuel-storage/src/dispatch.rs`** — the `KernelBindingTable::register` calls + `cpu_unary_wrapper!`/`cpu_binary_wrapper!` macros. New kernels register here.
9. **`fuel-cpu-backend/src/byte_kernels.rs`** — where new CPU byte kernels go (use the `unary_f32_kernel!` / `binary_f32_kernel!` macros + their `_f64` / `_bf16` / `_f16` siblings).
10. **`fuel-storage/src/pipelined.rs`** §`fn op_to_op_kind` — the `Op → OpKind` translation map; every new primitive gets an entry.

## What this session must NOT do

- **Don't ship without bit-stable cpu+reference kernels.** Architecture v1.0 commits the always-built backend covers every primitive at `bit_stable_on_same_hardware: true`. Each new `Op` variant must come with a CPU kernel that matches reference exactly (modulo intrinsic-different transcendentals like `tanh`, where bounded-error is acceptable).
- **Don't add Tier-3 ops to the `Op` enum.** They go in `FusedOpRegistry`. Any `Op` variant addition is a multi-file edit + every exhaustive consumer reviewed; the registry was built precisely to avoid this for compositions.
- **Don't introduce GPU kernels speculatively.** CPU + reference is the floor. CUDA / Vulkan kernels for new ops are stretch goals at the end of each PR; if they require non-trivial PTX/SPIR-V work, defer to follow-up sessions.
- **Don't break the closed-Op invariant.** Every variant must be a true primitive (irreducible OR with dedicated hardware support OR appears as a single instruction across multiple backends). If you find yourself wanting an `Op` for something that decomposes to 3 existing ops without precision loss, it belongs in the registry.
- **Don't push to remote unless asked.**

## Branch and starting state

- **Current branch (at session start)**: whichever branch the user is on. Verify with `git status`. Almost certainly `feature/storage-unification` continuing from tip `896f97da` (the Recip/Abs CUDA stretch commit).
- **Parallel work coordination**: none of this conflicts with Phase 7.6 step 4+ (registry-side work) as long as you only *read* the registry until your Tier-3 PRs. Conflicts with any other session adding `Op` variants.
- **Memory to consult**: `project_op_recip_abs_shipped.md` (recent precedent), `project_phase_7_6_step_3_shipped.md` (latest registry milestone), `feedback_no_panics_in_production.md`, `feedback_engage_critically.md`, `feedback_architectural_cleanness_over_local_pragmatism.md`.

## Concrete work — sequenced by leverage and dependency

The session is ~12 PRs, grouped into 4 batches. Batches are dependency-ordered; PRs within a batch are parallel-safe with each other modulo the shared `Op` enum / `op_short_name` / `op_key` files (which need merge resolution if multiple PRs touch them, but the conflicts are mechanical).

### Batch A: Comparison family + `Where` (UNBLOCKS EVERYTHING ELSE)

This is the highest-leverage block. Without comparison ops + `Where`, dozens of compositions in Batches B–D can't be expressed cleanly.

#### PR A1 — `Op::Equal` / `Ne` / `Lt` / `Le` / `Gt` / `Ge` (6 binary comparison ops)

- **Shape rule**: Output shape == input shape (both inputs broadcast-compatible; for the MVP require identical shapes per the existing binary-op convention).
- **Output dtype**: `U8` (mask: `0` or `1`). Per-`OpKind` dispatch keys: `EqualElementwise`, `NotEqualElementwise`, `LessElementwise`, `LessEqualElementwise`, `GreaterElementwise`, `GreaterEqualElementwise`.
- **CSE tags in `op_key`**: 25–30 (next free after Op::Abs's 24).
- **Backward**: zero gradient through both inputs (boolean-output ops are non-differentiable). Implementation: register `GradientRule` that returns `vec![None, None]`, OR rely on the legacy match's panic-on-unknown-op behavior. Prefer the `None`-returning rule so the autograd traversal terminates cleanly when a comparison is on a non-loss path.
- **Files**:
  - `fuel-graph/src/lib.rs`: 6 enum variants, 6 `Tensor` builder methods (`eq`, `ne`, `lt`, `le`, `gt`, `ge`), 6 `op_short_name` arms, 6 backward arms (no-op).
  - `fuel-graph/src/opt.rs`: 6 `op_key` tags.
  - `fuel-core-types/src/dispatch.rs`: 6 `OpKind` variants + `as_str` arms.
  - `fuel-storage/src/pipelined.rs`: `op_to_op_kind` arms.
  - `fuel-storage/src/dispatch.rs`: 6 wrappers per dtype (F32 minimum; F64/BF16/F16 follow same dispatch key but compare-and-emit-U8 — output dtype differs from input).
  - `fuel-cpu-backend/src/byte_kernels.rs`: per-dtype kernels emitting `U8`. The existing `binary_f32_kernel!` macro produces same-dtype output; you'll need a new macro variant for `f32 × f32 → u8`. Call it `binary_compare_f32_kernel!` or similar.
  - `fuel-reference-backend/src/exec.rs`: 6 `Op::Eq => binary!(...)` arms; the `ops::eq` etc. functions need to exist (add them in `fuel-reference-backend/src/ops.rs` — single-line maps over input slices).
  - `fuel-graph-cpu/src/lib.rs`: 6 arms.
  - `fuel-core/src/lazy.rs`: 6 `LazyTensor` wrappers.
- **Tests**: structural in fuel-graph (one per op asserting output dtype is U8); numerical equivalence in fuel-graph-cpu (one mass test that spot-checks all 6).
- **Watch for**: `OpParams` for binary may already assume same-dtype; the U8-output case may force a new params arm (or reuse none of them). Audit `op_to_op_params` carefully.

#### PR A2 — `Op::Where` (ternary select)

- **Inputs**: `(cond, a, b)`. `cond` is `U8`; `a` and `b` are same dtype. Output: same shape and dtype as `a`/`b`. `out[i] = if cond[i] != 0 { a[i] } else { b[i] }`.
- **CSE tag in `op_key`**: 31.
- **Backward**:
  - `cond`: zero gradient (non-differentiable).
  - `a`: upstream * `cast(cond, a.dtype)` — only the picked positions receive the upstream signal.
  - `b`: upstream * `cast(1 - cond, b.dtype)`.
  - Implement via `Op::Cast` + `Op::Mul` + `Op::Sub` (1 - cond requires `AddScalar(-1).neg()` or `SubScalar` if introduced; for now use the longhand: `cast(cond_f32) → cond_f32; ones_like - cond_f32; mul; ...`).
- **Files**: same shape as A1 plus `OpKind::Where`, `OpKind::WhereSelect` (pick a name), and a 3-input wrapper macro.
- **Tests**: forward picks correctly; backward routes upstream to `a`-positions vs `b`-positions; `cond.dtype == U8` assertion in builder.
- **Watch for**: `unary_op` and `binary_op` helpers in `fuel-graph/src/lib.rs` won't fit; you'll need a `ternary_op` helper following the same pattern.

### Batch B: Tier-1 unary fanout (mechanical, post-Recip/Abs precedent)

After A lands, B is a string of nearly-identical PRs. Each follows `add-recip-abs-primitives.md` exactly. Aim for one commit per op per layer (graph IR; reference + CPU; LazyTensor + Judge), 3 commits per op.

#### PR B1 — `Op::Floor` / `Op::Ceil` / `Op::Round`

- All three already have `UnaryOp::{Floor, Ceil, Round}` and `ops::{floor, ceil}` in fuel-reference-backend (round may need adding).
- Backward: all three have zero gradient almost everywhere (like `Op::Step`). Implement as silent no-op in the backward match.
- Round convention: **banker's rounding** (round-half-to-even, matching IEEE 754 `roundeven`); document and pin in the kernel.
- CSE tags 32, 33, 34.

#### PR B2 — `Op::Sign`

- Already has `UnaryOp::Sign` and `ops::sign` in fuel-reference-backend.
- Backward: zero (subgradient at 0; non-differentiable elsewhere except trivially).
- **Optional cleanup**: simplify the `Op::Abs` backward to `upstream * sign(x)` once this lands. Do this in a separate trailing commit so it's bisectable.
- CSE tag 35.

#### PR B3 — `Op::Erf` and `Op::GeluErf`

- `UnaryOp::Erf` and `UnaryOp::GeluErf` already exist. `ops::erf` likely needs adding (use `libm::erff` or the f64 path); `ops::gelu_erf` follows.
- Backward of `Erf`: `d/dx erf(x) = (2/√π) * exp(-x²)`. Decomposes to `Sqr` → `Neg` → `Exp` → `MulScalar(2/√π)`.
- Backward of `GeluErf`: closed form is `0.5 * (1 + erf(x/√2)) + x * exp(-x²/2) / √(2π)`. Express via existing primitives; document the chain.
- CSE tags 36, 37.

### Batch C: Tier-2 irreducible primitives

These have varied shapes. Order by leverage.

#### PR C1 — `Op::Squeeze { dim }` (metadata-only view)

- **Inverse of `Op::Unsqueeze`** (which already exists and is a view op, see `is_view_op`).
- Joins `is_view_op`, gets a layout transform via `Layout::squeeze`, no kernel needed (Layout side-table handles it).
- **Constraint**: panics at build time if the requested dim's size is not 1 (mirror Unsqueeze's bounds check).
- Backward: `Op::Unsqueeze { dim }` (the dim re-inserts at the same position).
- Files mostly in `fuel-graph/src/lib.rs` (variant + builder + view-op set + `derive_view_output_layout` arm). No kernel work.
- CSE tag 38. `OpKind` not strictly needed (view ops are layout-only) but add for symmetry.

#### PR C2 — `Op::Pow` (binary, real exponent)

- Inputs: `(base, exponent)`, same shape, same dtype. Output: same.
- **Distinct from `Op::PowI(i32)`**: that's scalar integer exponent (no second input). Op::Pow is full element-wise binary `pow(x[i], y[i])`.
- Backward: `d/da a^b = b * a^(b-1)`; `d/db a^b = a^b * ln(a)`. Standard.
- CPU kernel: `f32::powf` per element. CUDA: PTX `powf` intrinsic.
- CSE tag 39. New `OpKind::PowElementwise`.

#### PR C3 — `Op::Rsqrt`

- Element-wise `1/sqrt(x)`. Single op (vs `sqrt` then `recip` losing precision).
- Backward: `d/dx (1/√x) = -0.5 * x^(-3/2)`. Reuse forward output: `grad_x = -0.5 * upstream * fwd_out / x`.
- CPU: `1.0 / x.sqrt()`. CUDA: `__frsqrt_rn`.
- Strong RMSNorm relevance: today RMSNorm decomposes to `sqr → mean_dim → add_scalar → sqrt → reciprocal → broadcast → mul`. With Rsqrt the chain shortens.
- CSE tag 40. New `OpKind::RsqrtElementwise`.

#### PR C4 — `Op::Rem` (element-wise modulo)

- `a % b` element-wise, same shape, same dtype. Convention: IEEE 754 `remainder` (Rust's `f32::rem_euclid` vs `%` differ on negative — pick `f32::rem_euclid` to match PyTorch's `torch.remainder`).
- Backward: `d/da (a mod b) = 1`, `d/db (a mod b) = -floor(a / b)`. Floor exists after B1.
- CSE tag 41. New `OpKind::RemElementwise`.

#### PR C5 — `Op::Flip { dim }` and `Op::Roll { dim, shift }`

- Both can be view ops via clever stride manipulation (Flip uses negative strides; Roll uses an offset + wrap), but the existing `Layout` may not support negative strides. **Engage critically here**: check `Layout`'s capability before committing. If negative strides aren't supported, ship as materializing ops (CPU kernel that reorders bytes); add layout support in a follow-up.
- Backward of Flip: another Flip with the same dim. Backward of Roll: Roll with negated shift.
- CSE tags 42, 43.

#### PR C6 — `Op::CumSum { dim }`

- Running sum along `dim`. Output same shape as input. `out[..., i, ...] = sum(in[..., 0..=i, ...])`.
- Backward: reverse cumsum (cumsum from the other end). Express as `Flip → CumSum → Flip` (post-C5).
- CPU: serial scan per row. CUDA: prefix-scan kernel (existing in `fuel-cuda-kernels` likely; verify).
- Critical for Mamba/SSM. CSE tag 44.

#### PR C7 — `Op::Pad { dim, before, after, mode, value }`

- Modes: `Constant(f64)`, `Reflect`, `Replicate`. (Skip `Circular` for the MVP.)
- Output shape: input shape with `dim` extended by `before + after`.
- Backward: slice the gradient to remove the padded regions (gradient through `Constant` mode); `Reflect`/`Replicate` accumulate gradients at the reflected/replicated positions.
- This is a real kernel, not a view op. CSE tag 45. New `OpKind::Pad`.
- **Heads-up**: `Conv2D` already takes symmetric padding inline. Don't replace that — `Op::Pad` is for the explicit asymmetric / mode-aware case.

### Batch D: `FusedOpRegistry` seeding (Tier 3)

Each entry is one `FusedOpEntry` registration in `fuel-graph/src/registry.rs` with: canonical decomposition (mandatory; backends without a fused kernel use this), backward identity, shape + dtype rules, optional per-backend kernel implementations. **This session ships ZERO native kernels for Tier-3** — that's follow-up work. Decomposition is the deliverable.

Group by family. Each PR registers ~5 entries, exercises the decomposition through cpu+reference, asserts numerical correctness.

#### PR D1 — Activation compositions

`LeakyRelu(negative_slope)`, `Elu(alpha)`, `Selu`, `Softplus(beta, threshold)`, `Mish`, `HardSigmoid`, `HardSwish`, `LogSigmoid`. Each decomposes into 2–6 existing primitives. Document each formula in the registry entry's doc comment.

#### PR D2 — Stable-math compositions

`Expm1`, `Log1p`, `Log2`, `Log10`, `Exp2`, `LogSumExp(dim)`, `LogSoftmaxLastDim`. The `LogSumExp` family needs the max-subtract trick for stability — ship the stable form, not naive `log(sum(exp))`.

#### PR D3 — Trig & hyperbolic compositions

`Tan` (= `Sin/Cos`), `Asin`, `Acos`, `Atan`, `Atan2(y, x)` (binary), `Sinh`, `Cosh`, `Asinh`, `Acosh`, `Atanh`. Most decompose with primitives the agent will already have, but `Asin`/`Acos`/`Atan` may need approximation polynomials — defer to follow-up if no exact decomposition exists.

#### PR D4 — Boolean reductions and predicates

Now that Batch A landed: `Any { dim }`, `All { dim }`, `IsNaN`, `IsInf`, `IsFinite`. Each is `cast → reduce` or a single primitive emitting U8.

#### PR D5 — Shape compositions

`Stack { dim }` (= `Unsqueeze` + `Concat`), `Split { dim, sizes }` (= N `Slice`s), `Tile { reps }` (= broadcast + reshape, materializing), `Tril`, `Triu`, `Diag`, `OneHot`, `Eye`, `Arange`, `Linspace`. Some of these (Eye, Arange, Linspace) are pure constructors and may belong as `Tensor::from_*` builders rather than registry entries; engage critically.

#### PR D6 — Pooling family (vision)

`MaxPool2D`, `AvgPool2D`, `AdaptiveAvgPool2D`, `AdaptiveMaxPool2D`, `Upsample` (nearest + bilinear). Decomposition via reshape + reduce works for fixed kernel sizes; adaptive pooling has dynamic kernel size, so its decomposition is more involved — flag if difficult.

#### PR D7 — Normalization variants

`BatchNorm`, `GroupNorm`, `InstanceNorm`. All three decompose via `mean_dim`/`var_dim` along different axes; share most of the implementation. Same shape as the existing `RmsNormLastDim`/`LayerNormLastDim` registry entries.

#### PR D8 — Sampling and search

`TopK { dim, k }`, `Sort { dim, descending }`, `Multinomial`. Each emits a U32 index tensor (`TopK` may emit a tuple — defer that until tuples are first-class in the IR; for now ship `TopKValues` + `TopKIndices` as separate ops).

#### PR D9 — Masked-write and predicate compositions

`MaskedFill(value)` (= `Where(cond, broadcast(value), x)`), `IndexPut { dim }` (= `IndexAdd` after subtracting the existing slot — verify this composition is correct), `Multinomial` if not in D8.

## Test commands

After each PR:

```bash
cargo test -p fuel-graph --lib
cargo test -p fuel-graph-cpu --lib
cargo test -p fuel-reference-backend --lib
cargo test -p fuel-core --lib
```

For Batch C (which adds reductions + view ops + new shape rules), also:

```bash
cargo test -p fuel-storage --lib
```

For each PR's CUDA stretch arm (after Batch B at minimum):

```bash
cargo test -p fuel-cuda-backend -- --ignored --nocapture
cargo test -p fuel-core --features cuda --lib judge_ -- --nocapture
```

End-of-session full sweep:

```bash
cargo check --workspace
cargo test -p fuel-graph -p fuel-graph-cpu -p fuel-reference-backend -p fuel-core -p fuel-storage --lib
```

## Operating principles

- **Engage critically.** Several Tier-2/Tier-3 design decisions are sketches above, not final calls. Surface concerns before silently picking. Examples: should `Op::Squeeze` panic or no-op when the dim isn't 1? Should `Op::Pow`'s dtype handling promote? Should `Op::Where`'s cond accept f32 (truthy != 0) or strictly U8? Default to the **architecturally-cleanest** form even if it's slightly more user-facing work.
- **Bit-stable cpu+reference is non-negotiable.** Every new `Op` ships a CPU kernel that matches reference exactly, modulo well-known transcendental wobble (`erf`/`tanh` etc.). Document tolerance in the kernel's `PrecisionGuarantee`.
- **No production panics.** New backward arms that hit non-differentiable inputs (comparisons, cast-to-int, etc.) return `None` from their `GradientRule` rather than panicking. Comparisons on a loss path are user error; signal it via missing-gradient downstream, not panic.
- **One commit per logical layer per PR.** Mirrors the Recip/Abs precedent: graph IR + minimal exec arms in one commit; integration test in another; LazyTensor + Judge in a third; CUDA stretch in a fourth (when it lands).
- **Sequencing matters.** Batch A absolutely first; everything else depends on `Where` + comparisons existing for backward rules / decompositions to be expressible. Batch B and C can interleave freely after A. Batch D can start once at least one Tier-1/Tier-2 op the decomposition needs is in place.
- **Don't refactor existing ops opportunistically.** If you notice that `Op::Abs`'s backward could simplify after `Op::Sign` lands (PR B2), do it — but in a separate trailing commit, not folded into B2 itself. Keeps history bisectable.
- **Update memory after each batch lands.** A short memory file per batch (e.g., `project_op_primitive_set_batch_a_shipped.md`) makes the future-session pickup story trivial.
- **Don't push to remote unless asked.**

## End-of-session deliverable

Branch tip should advance by ~30–40 commits across ~15–20 PRs. Concretely:

- **`Op` enum** grew by ~16 variants (6 comparison + Where + 6 unary fanout + Squeeze + Pow + Rsqrt + Rem + Flip + Roll + CumSum + Pad).
- **`OpKind`** grew by the same set + a few more for the new kernel shapes.
- **`FusedOpRegistry`** seeded with ~30+ entries across activations, stable-math, trig, boolean reductions, shape compositions, pooling, normalizations, sampling, masking.
- **Every new primitive** has bit-stable cpu+reference kernels and at least one numerical correctness test.
- **`LazyTensor`** has builder methods for every new primitive and every Tier-3 fused op.
- **Judge `PROFILED_OPS`** extends to cover every new elementwise primitive in the unary/binary fanouts.
- **Memory** has fresh entries summarizing each batch's commits.

Stretch goal (deferred to follow-up sessions if not reached): native CUDA / Vulkan / AOCL / MKL kernels for any subset of the new ops where existing PTX/SPIR-V/vendor-BLAS intrinsics provide a single-instruction win.

## Coordination notes

- **`Op` enum extension** — exclusive lock during this session; no other session should be adding `Op` variants in parallel. CUDA Tier-1 fanout sessions for already-existing `Op` variants are fine.
- **`FusedOpRegistry`** — after Phase 7.6 step 3 the registry is the canonical home for compositions. This session is the first major user of that capability; document patterns thoroughly so future fused-op-adding sessions are mechanical.
- **Backward rules** — every new differentiable primitive registers a `GradientRule` in `fuel-graph/src/grad.rs` (Phase 6d Track 2 dispatcher). Inline backward arms in `Tensor::backward`'s legacy match are also acceptable for ops landing during this session, but prefer GradientRule registration as the cleaner long-term shape.
- **Tier 3 backward identities** — registry entries declare a backward identity (either "another registry entry" or "decompose first then differentiate"). For activations and stable-math ops, "decompose first then differentiate" is universally fine; document it once in the family-level docs and don't repeat per entry.
