# Session prompt — Baracuda CUTLASS alpha.13 integration

> **Reconciled 2026-06-15 against the 2026-06-14 redirection + current git:** Track A (alpha.7 → .13 version bump) shipped and was superseded by alpha.67; B1–B4 (bf16/f16 CUTLASS Rrr matmul) shipped (`dc282be1`, `a2d4bcc3`); this doc is now the backlog for the still-pending B5–B8, re-anchored on the branch-point + Pareto-frontier model and the `register_fused!` / `default_kernel_registry` surface.

## What this session is for

Baracuda 0.0.1-alpha.7 → 0.0.1-alpha.13 ships every Tier-1 and Tier-2 ask Fuel filed in the 2026-05-10 critique (`project_baracuda_cutlass_critique.md`), plus four preemptive Tier-3 extensions. Headline additions:

- **`DeviceBuffer<u8>::view_as<T>` + `DeviceSlice::from_raw_parts`** — safely construct typed `DeviceSlice<bf16>` from `Arc<DeviceBuffer<u8>>`. This was the Tier-1 blocker that prevented Fuel from calling `baracuda-cutlass` at all from the byte-storage substrate.
- **`LayoutSku::Rrr` (f16/bf16)** — row-major × row-major → row-major GEMM. Matches `Op::MatMul`'s natural shape and eliminates the RHS transpose pass.
- **`GemmPlan::<f32>::select` → TF32 path** — f32 input through Ampere TF32 tensor cores. First real alternative to cuBLAS's `Compute32FFastTF32` at the `(MatMul, F32, Cuda)` decision point.
- **`BatchedGemmPlan`** — uniform-shape batched GEMM with `batch_count` + `stride_{a,b,c,d}`. Drop-in for the equal-batch fast path.
- **`EpilogueKind::Bias` + `BiasRelu` / `BiasGelu` / `BiasSilu`** — fused bias and bias+activation epilogues. Collapses `MatMul → Add → activation` into one kernel pass with one memory pass.
- **`GemmPlan::precision_guarantee()`** — per-kernel `PrecisionGuarantee` accessor matching Fuel's own schema.

**Track A is retired.** The mechanical alpha.7 → alpha.13 version bump shipped long ago and the workspace pin has since moved on to `0.0.1-alpha.67` (see CLAUDE.md / `Cargo.toml`); the version-pin and `cargo update` instructions that used to live here are stale and have been struck. **B1–B4 also shipped** — the CUTLASS bf16 Rrr matmul landed in `dc282be1` and the f16 mirror in `a2d4bcc3`, both registered as **siblings** to the cuBLAS path. What remains is **Track B steps B5–B8**, the still-pending CUTLASS surface:

- **B5** — CUTLASS TF32 matmul (f32 input → tensor cores, Rcr layout).
- **B6** — `BatchedGemmPlan` for the uniform-batch fast path.
- **B7** — CUTLASS `FusedLinear` via the Bias epilogue.
- **B8** — `FusedLinearActivation` via the Bias+activation epilogue.

The goal is to wire CUTLASS as **alternative kernels** at the relevant branch points (`Op::MatMul` and `Op::FusedLinear`). Per the 2026-06-14 redirection, alternatives attach to **branch points**, not every node; CUTLASS registrations join the cuBLAS path on a bounded per-device Pareto frontier (no fixed top-N). This is the first real exercise of that framing on CUDA.

Each B-step is sized for its own commit. The session may stop after any step.

This session is **parallel-safe** with the fused-kernel registry work (which restructures the `register_fused!` / `default_kernel_registry` surface) modulo the shared `BackendImpl.precision: PrecisionGuarantee` field. Coordinate at the dispatch-registration site if that work has progressed.

## Read first (in this order)

1. **`docs/architecture/00-index.md`** + **`docs/architecture/01-identity.md`** — grounding. **`docs/architecture/10-decisions-log.md`** (2026-06-14 entry) is the current anchor; it wins over any not-yet-revised section.
2. **`docs/architecture/04-optimization.md`** §"Bounding the frontier: Pareto per device + crowding cap" — CUTLASS-backed matmul registers as one alternative; existing cuBLAS-backed matmul registers as another. Alternatives attach to **branch points**, not every node; survival is decided by the per-device Pareto frontier + crowding cap (no fixed top-N — the old "default N=3" framing is retired). The optimizer ranks them by cost + telemetry and the route picker chooses among the survivors at branch points. **Don't replace the cuBLAS path; add CUTLASS as a sibling on the frontier.**
3. **`docs/architecture/05-backend-contract.md`** §`PrecisionGuarantee` and §"always-built `bit_stable` coverage commitment." CUTLASS kernels are typically *not* bit-stable (TF32, fast-math intrinsics). Their `PrecisionGuarantee` declares this; the always-built CPU backend's `bit_stable` coverage upholds the architecture's correctness anchor. CUTLASS kernels are about throughput.
4. **Memory entry `project_baracuda_cutlass_critique.md`** — what was missing in alpha.7 and why. Every Tier-1/Tier-2 ask in that critique is now shipped on crates.io.
5. **The fused-kernel registration surface** — `register_fused!` and `default_kernel_registry()` in `fuel-dispatch/src/fused.rs`, with the bf16/f16 CUTLASS precedent already in `fuel-cuda-backend/src/cutlass.rs`. CUTLASS-backed kernels register the same way; the `BackendImpl.precision` field absorbs the `GemmPlan::precision_guarantee()` output.
6. **Memory entry `project_dev_environment.md`** — this Windows host has working CUDA on RTX 4070 (sm_86, Ampere). The TF32 + Rrr SKUs are sm_80; both run on this hardware. Run `#[ignore]`d live-CUDA tests after every kernel-touching commit; don't defer to "the user's other box."
7. **`fuel-cuda-backend/src/cutlass.rs`** — the shipped B1–B4 bridge (bf16/f16 Rrr matmul). The module docstring records that raw `Op::MatMul` still rides the single-impl binding table while `Op::FusedLinear` exercises the append-on-register sibling surface; the B5–B8 work extends this same file.

## What this session must NOT do

- **Don't replace the cuBLAS matmul paths.** The architecture commits to alternatives at branch points, ranked on the per-device Pareto frontier. The existing `gemm_strided_batched_{f32,f16,bf16,f64}` functions (storage.rs:3021/3071/3130/3189) stay in place. CUTLASS becomes a sibling registration alongside cuBLAS.
- **Don't fold all alpha.13 capabilities into one PR.** Each capability (Rrr GEMM, TF32 GEMM, BatchedGemmPlan, Bias epilogue, Bias+activation epilogue) is one step. Each step is one commit minimum and ideally one test commit alongside.
- **Don't fuse new optimizer rules speculatively.** `Op::FusedLinear` already exists (lib.rs:742) and `fuse_linear` already runs (opt.rs:1032). The Bias+activation epilogue work requires extending `fuse_linear` to also fuse `MatMul → Add → activation`, *or* introducing a new `Op::FusedLinearActivation` primitive, *or* registering through the `register_fused!` / `default_kernel_registry` surface in `fuel-dispatch/src/fused.rs`. Engage critically — pick the architecturally-cleanest home and document why before writing the rule.
- **Don't transmute around `view_as`.** The existing `as_raw().0` raw-pointer casts in `gemm_strided_batched_*` work for cuBLAS but are not the model for new CUTLASS code. New cutlass call sites use `view_as::<T>()` on `Arc<DeviceBuffer<u8>>`. The cuBLAS call sites are not retrofitted in this session — they're not broken, just not idiomatic.
- **Don't push to remote.**

## Branch and starting state

- **Start from `main`** and branch for the work. (The original prompt named `feature/storage-unification` / tip `3d1c6fbe`; that branch and the B1–B4 commits below have long since merged.)
- **Coordination**: conflicts would arise only if another session is also editing `fuel-cuda-backend/src/cutlass.rs` or the matmul/fused dispatch registration simultaneously.

## ~~Track A — Version bump~~ (RETIRED — shipped, then superseded)

> ~~Mechanical alpha.7 → alpha.13 version bump.~~ **Done and obsolete.** The bump shipped, and the workspace pin has since advanced to `baracuda 0.0.1-alpha.67` (the current pin per CLAUDE.md / `Cargo.toml`). The old `cargo update -p baracuda-*` list and the `Cargo.toml` lines-164–178 edit instructions are stale — do **not** follow them. There is nothing to do here. Skip to the still-pending B5–B8.

## Track B — Wire CUTLASS as alternative kernels

### ~~Step B1: Add `baracuda-cutlass` dependency~~ — DONE

The dependency, the `fuel-cuda-backend/src/cutlass.rs` bridge module, and `mod cutlass;` in `lib.rs` all landed (and the pin is now alpha.67). The bridge module is the integration seam the remaining steps extend.

### ~~Step B2: View-as adoption at byte-storage boundary~~ — DONE (in `dc282be1`)

> Landed alongside the bf16 matmul: `cutlass_matmul_rrr` in `fuel-cuda-backend/src/cutlass.rs` consumes typed `DeviceSlice`s from the byte-storage substrate; the cuBLAS paths were left byte-pointer-shaped as intended. Original spec retained below for reference.

The matmul callsite ([storage.rs:3038-3040](../../fuel-cuda-backend/src/storage.rs#L3038)) currently does `let a_ptr = a.as_raw().0 as *const _;`. This works for cuBLAS but not for cutlass. The clean fix:

1. In `fuel-cuda-backend/src/byte_storage.rs`, add a typed-view helper:
   ```rust
   pub fn view_as<T: baracuda_driver::DeviceRepr>(&self) -> Result<baracuda_driver::DeviceSlice<'_, T>, CudaError>
   ```
   Implementation: call `Arc::as_ref(&self.buf).view_as::<T>()` (alpha.8 API). Return `CudaError::InvalidDtypeBoundary` if `view_as` rejects (byte-divisibility failure).
2. Use it from one cutlass call site (added in B3 below). **Don't** retrofit the existing cuBLAS paths in this session — they work; the cuBLAS surface is byte-pointer-shaped on purpose.

This commit lands the helper but no behavior change unless B3 also lands. If B3 won't land this session, the helper is dead code — that's fine, it's the seam.

Commit: `feat(cuda): add CudaStorageBytes::view_as<T> helper for safe typed views`.

### ~~Step B3: Register CUTLASS bf16 matmul as alternative~~ — DONE (`dc282be1`)

> Shipped: `cutlass_matmul_rrr::<bf16>` with `LayoutSku::Rrr`, registered as a sibling to the cuBLAS bf16 path, plus the live-CUDA parity test. Original spec retained below for reference.

Goal: at the `(MatMul, [BF16, BF16, BF16], Cuda)` decision point, register two `BackendImpl`s — the existing cuBLAS path and a new CUTLASS Rrr path. Run a live-CUDA equivalence test that calls both and confirms parity within bf16 tolerance.

1. In `fuel-cuda-backend/src/cutlass.rs`, write a `cutlass_matmul_bf16` function:
   - Takes the same inputs as `gemm_strided_batched_bf16` (StridedBatchedConfig + three byte-storage slices).
   - Uses `view_as::<bf16>()` from B2 to get typed `DeviceSlice<'_, bf16>` for A, B, C.
   - Builds `GemmDescriptor { layout: LayoutSku::Rrr, kind: ElementKind::Bf16, epilogue: EpilogueKind::Identity }`.
   - `GemmPlan::<bf16>::select(&descriptor)?.run(stream, &args)`.
   - Args carry `bias: None`.
2. In the dispatch registration (`register_*_kernels` in `fuel-cuda-backend/src/backend.rs` or wherever the matmul kernel registers today), add a second registration for `(MatMul, [BF16, BF16, BF16], Cuda)` with the CUTLASS impl. **Both** registrations live in the table; the binding-table key includes a `BackendImpl` slot per alternative.
3. Fill `BackendImpl.precision` from `GemmPlan::precision_guarantee()` (alpha.8 accessor — bf16 Rrr is `bit_stable_on_same_hardware: true`, `accumulator: F32`, `math_precision: Bf16`, `deterministic: true`).
4. Add a live-CUDA test in `fuel-storage/tests/cuda_dispatch_live.rs`: call both impls on identical input, assert max-relative within bf16 tolerance (~5e-3).

**Watch for**: the existing cuBLAS path includes Op::Op transposition for non-row-major layouts. CUTLASS's Rrr expects A and B both row-major. If `Op::MatMul`'s shape convention is column-major somewhere internally, you'll see a 0-bit match fail — verify against `Op::MatMul`'s docstring + an actual gather of A and B layouts at the dispatch site.

Commits:
- `feat(cuda): CUTLASS bf16 matmul via baracuda-cutlass alpha.13 Rrr layout`
- `test(cuda): bf16 matmul parity — cuBLAS vs CUTLASS (live, ignored)`

### ~~Step B4: Mirror f16 matmul~~ — DONE (`a2d4bcc3`)

> Shipped: the f16 mirror fell out as the generic `cutlass_matmul_rrr<T>` over `T: CutlassElement` (now `Element`, see B5 note), instantiated for both `f16` and `bf16`. Original spec retained below for reference.

Identical to B3 but for `(MatMul, [F16, F16, F16], Cuda)`. The Rrr layout is shipped for f16 too. Reuse `cutlass_matmul_<T>` as a generic over the CUTLASS element type if it falls out cleanly; otherwise duplicate.

Commit: `feat(cuda): CUTLASS f16 matmul mirror of bf16 Rrr path`.

### Step B5: CUTLASS TF32 matmul (f32 input → tensor cores) — PENDING (next up)

`(MatMul, [F32, F32, F32], Cuda)` currently has one impl (cuBLAS with `Compute32FFastTF32`). Add a CUTLASS sibling using `GemmPlan::<f32>::select` (routes f32 input through TF32 tensor cores in Rcr layout). **Note**: Rrr × F32 is *not* shipped; use Rcr. This means an RHS transpose pass is needed for f32 — measure whether the perf gain still beats cuBLAS Fast-TF32 after the transpose tax. If it doesn't, register it anyway (it survives or is pruned on the per-device Pareto frontier by empirical cost — the route picker won't choose it if cuBLAS dominates) and let the data settle.

**API note (alpha.67):** the trait the B1–B4 generic is written against, `CutlassElement`, was renamed to `Element` in alpha.26, gaining a `Scalar` associated type for the epilogue's α/β compute precision (f16/bf16/f32-input kernels all use `Scalar = f32`; only f64 uses `Scalar = f64`). The shipped bridge already pins this via `T: CutlassElement<Scalar = f32>` — see `fuel-cuda-backend/src/cutlass.rs:53-58`. New f32 code follows the same `Scalar = f32` constraint.

Fill `BackendImpl.precision`:
- `bit_stable_on_same_hardware: false`
- `accumulator: F32`
- `math_precision: Tf32`
- `max_relative: ~1e-3`

Commit: `feat(cuda): CUTLASS f32 matmul via TF32 tensor cores (Rcr layout)`.

### Step B6: BatchedGemmPlan for equal-batch fast path — PENDING

The existing `matmul()` path at storage.rs:2697 uses `StridedBatchedConfig` uniformly. For uniform-batch cases (the common transformer attention shape), `BatchedGemmPlan` is the direct cutlass equivalent. Register as a third alternative at the bf16/f16 MatMul branch points (only when the batch dims are uniform) — it joins the cuBLAS and Rrr siblings on the frontier and is selected empirically per shape class.

`BatchedGemmPlan` is `Identity`-only (no bias). That's fine for raw `Op::MatMul`; bias-fused matmul still routes through the single-GEMM `GemmPlan` path.

Commit: `feat(cuda): CUTLASS BatchedGemmPlan as alternative for uniform-batch matmul`.

### Step B7: CUTLASS FusedLinear via Bias epilogue — PENDING

`Op::FusedLinear` at [lib.rs:742](../../fuel-graph/src/lib.rs#L742) is emitted by [`fuse_linear`](../../fuel-graph/src/opt.rs#L1032) but has *no* CUDA kernel today (falls back to decomposed matmul + add). Register a CUTLASS-backed kernel through the `register_fused!` / `default_kernel_registry` surface (`fuel-dispatch/src/fused.rs`) — the same append-on-register sibling path the bf16/f16 fused entries use:

1. In `cutlass.rs`, add `cutlass_fused_linear_bf16` (mirror f16):
   - `EpilogueKind::Bias`
   - `GemmArgs { bias: Some(VectorRef::from(bias_slice)), ... }`
   - Bias is rank-1, `[N]`-shaped. Use `view_as::<bf16>()` to get the typed slice.
2. Register `(FusedLinear, [BF16, BF16, BF16, BF16], Cuda)` → CUTLASS kernel. Four input dtypes (a, b, bias, out).
3. Live-CUDA parity test: decomposed (MatMul + AddBias) vs fused. Within bf16 tolerance.

**Engage critically**: `Op::FusedLinear`'s current backward rule (lib.rs:5973) treats it as `(a @ b) + bias`. Verify the backward still works correctly with the CUTLASS-fused forward — the gradient flow doesn't change, but the cached values might. Specifically: today's decomposed path holds the matmul-output tensor for backward; the fused path doesn't materialize it. If backward needs that intermediate, the fused path needs to recompute (cheap) or hold it (memory tax). Document the choice.

Commits:
- `feat(cuda): CUTLASS fused-linear (Op::FusedLinear) via Bias epilogue`
- `test(cuda): fused-linear parity — decomposed vs CUTLASS Bias epilogue`

### Step B8: CUTLASS FusedLinearActivation via Bias+activation epilogue — PENDING

CUTLASS ships `EpilogueKind::{BiasRelu, BiasGelu, BiasSilu}`. This collapses three ops (`MatMul → Add → activation`) into one kernel. Fuel doesn't yet have an `Op::FusedLinearActivation` primitive; engage critically on the architecturally-clean shape:

**Option (a)**: Extend `Op::FusedLinear` to carry an `Option<ActivationKind>` field. Pro: one variant, minimum IR churn. Con: changes Op::FusedLinear's shape, which is widely consumed.

**Option (b)**: Add `Op::FusedLinearActivation { activation: ActivationKind }` as a new primitive. Pro: orthogonal to existing `Op::FusedLinear`. Con: new IR variant, every exhaustive consumer updates, decomposes through the same chain so it's redundant at the graph level.

**Option (c)**: Register a fused composition via `register_fused!` into `default_kernel_registry()` (`fuel-dispatch/src/fused.rs`) — the home for compositions. Pro: cross-backend visibility (Vulkan/CPU decompose; CUDA fires the fused kernel); aligned with what the registry exists to do, and matches how the bf16/f16 fused entries already register. Con: needs the rule that fuses `MatMul → Add → activation` into the registry entry.

**Recommendation**: option (c). The registry was built precisely for this. Steps:

1. Add a `FusedLinearActivation` entry to the registry with:
   - Decomposition: `MatMul → AddBias → activation` (each is an existing primitive).
   - Per-backend kernels: CUDA via CUTLASS Bias+activation epilogue; CPU + reference use the decomposition (no fused kernel needed).
   - Backward identity: same as `Op::FusedLinear`'s backward + activation backward, chained.
2. Extend `fuse_linear` (or add a parallel `fuse_linear_activation` rule) to detect `MatMul → Add → activation` and emit the registry entry's `Op::Fused(id, params)` form.
3. Live-CUDA test: decomposed (matmul + bias + activation as three separate ops) vs fused (one CUTLASS kernel). Parity within tolerance.

Activations: ReLU, GELU (alpha.11 ships exact erf-based GELU matching PyTorch default), SiLU. Three FusedLinear* permutations; ship as three commits or one if the registry shape factors them.

Commits:
- `feat(graph): FusedLinearActivation registry entry + fusion rule`
- `feat(cuda): CUTLASS FusedLinearActivation kernel (BiasRelu/BiasGelu/BiasSilu)`
- `test(cuda): fused-linear-activation parity — decomposed vs CUTLASS`

## Test commands

After each step:

```bash
cargo check --workspace --features cuda
cargo test -p fuel-cuda-backend --features cuda --lib
cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored --nocapture
```

After steps B7/B8 (graph-level changes):

```bash
cargo test -p fuel-graph --lib
cargo test -p fuel-graph -p fuel-graph-cpu -p fuel-reference-backend -p fuel-core --lib
```

End-of-session sweep:

```bash
cargo check --workspace --features "cuda cudnn nccl"
cargo test --workspace --lib
```

## Verification

1. ~~**`view_as` round-trip**~~ — DONE in B2/B3: typed views from `CudaStorageBytes` feed `cutlass_matmul_rrr` and were validated against the cuBLAS baseline.
2. ~~**Rrr matmul**~~ — DONE in B3/B4: `LayoutSku::Rrr` for the bf16/f16 `Op::MatMul` path, output matches cuBLAS.
3. **Bias-fused Linear** — route `Op::FusedLinear` through `EpilogueKind::Bias`. Single-kernel-launch confirmation via `nsys` (or `cuda-gdb`) is the qualitative win. Lands in B7.
4. **Bias + activation** — same as above for `Op::FusedLinearActivation`. Three kernel launches collapse to one. Lands in B8.
5. **`precision_guarantee()` mapping** — extract the four fields per `(layout, dtype, epilogue)` and map to Fuel's `PrecisionGuarantee` type. Stable per SKU — safe to cache at registration time. Lands incrementally across B3–B8.

## Operating principles

- **Bit-stable cpu+reference remains the correctness anchor.** Every primitive Fuel ships has bit-stable kernels in fuel-cpu-backend + fuel-reference-backend. CUTLASS kernels do not replace those; they're throughput alternatives. Their `PrecisionGuarantee.bit_stable_on_same_hardware` is what the architecture's precision-filter pass consults.
- **No production panics.** Result-returning everywhere (consistent with `feedback_no_panics_in_production`). The `view_as` helper returns `Result`; the cutlass call sites surface `baracuda_cutlass::Error` through the existing `CudaError` enum.
- **Engage critically.** Specifically for: (a) Op::FusedLinear backward semantics with the fused forward (step B7), (b) the FusedLinearActivation home choice (step B8 — registry vs Op enum), (c) whether to retrofit cuBLAS call sites to use `view_as` opportunistically (default: no, separate session).
- **One commit per logical step.** Each B-step is self-contained and bisectable.
- **Live-test on this host after every kernel-touching commit.** Per `project_dev_environment.md`. RTX 4070 supports sm_86; the sm_80 CUTLASS kernels run here. Don't defer to "the other box."
- **Update memory after each step lands.** Short topic file (e.g., `project_cutlass_bf16_matmul_shipped.md`, `project_cutlass_fused_linear_shipped.md`) plus a one-line MEMORY.md index entry. Keep MEMORY.md under the 24.4KB limit (it's at the limit today — prune older entries if needed).
- **Don't push to remote unless asked.**

## End-of-session deliverable

Already shipped (B1–B4): CUTLASS bf16/f16 Rrr matmul registered as siblings to cuBLAS at the MatMul branch points (`dc282be1`, `a2d4bcc3`). The first exercise of the branch-point alternatives framing on CUDA.

If through step B8: full CUTLASS surface integrated — Rrr matmul (bf16/f16, done), TF32 matmul (f32), BatchedGemmPlan (uniform batch), FusedLinear bias, FusedLinearActivation (3 activations). ~8–11 more commits. The CUDA matmul + Linear surface then carries several alternatives per branch point, which survive or are pruned on the per-device Pareto frontier and are chosen by the route picker empirically.

## Coordination notes

- **`Op` enum** — only step B8 *might* extend it (option b in the design decision). Default to option c (registry) — no Op enum changes.
- **`baracuda-cutlass`** — already a dependency of `fuel-cuda-backend` (the B1 bridge); the pin rides the workspace `baracuda 0.0.1-alpha.67`.
- **`fuel-cublaslt`** — separate crate, exposes cuBLASLt-backed bias+activation already. Out of scope for this session; both will eventually compete at the FusedLinear branch points and survive/prune on the same per-device Pareto frontier.
- **Fused-kernel registration surface** — register CUTLASS siblings through `register_fused!` into `default_kernel_registry()` (`fuel-dispatch/src/fused.rs`), the append-on-register path the shipped bf16/f16 fused entries already use.
- **Memory entry to write at the end**: `project_cutlass_integration_session_<date>.md` summarizing which steps landed. Mention the CUTLASS capability matrix ("Kernel SKU coverage") so a future session knows exactly what's available without re-deriving it.
