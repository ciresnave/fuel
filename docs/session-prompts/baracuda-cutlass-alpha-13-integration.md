# Session prompt — Baracuda CUTLASS alpha.13 integration

## What this session is for

Baracuda 0.0.1-alpha.7 → 0.0.1-alpha.13 ships every Tier-1 and Tier-2 ask Fuel filed in the 2026-05-10 critique (`project_baracuda_cutlass_critique.md`), plus four preemptive Tier-3 extensions. Headline additions:

- **`DeviceBuffer<u8>::view_as<T>` + `DeviceSlice::from_raw_parts`** — safely construct typed `DeviceSlice<bf16>` from `Arc<DeviceBuffer<u8>>`. This was the Tier-1 blocker that prevented Fuel from calling `baracuda-cutlass` at all from the byte-storage substrate.
- **`LayoutSku::Rrr` (f16/bf16)** — row-major × row-major → row-major GEMM. Matches `Op::MatMul`'s natural shape and eliminates the RHS transpose pass.
- **`GemmPlan::<f32>::select` → TF32 path** — f32 input through Ampere TF32 tensor cores. First real alternative to cuBLAS's `Compute32FFastTF32` at the `(MatMul, F32, Cuda)` decision point.
- **`BatchedGemmPlan`** — uniform-shape batched GEMM with `batch_count` + `stride_{a,b,c,d}`. Drop-in for the equal-batch fast path.
- **`EpilogueKind::Bias` + `BiasRelu` / `BiasGelu` / `BiasSilu`** — fused bias and bias+activation epilogues. Collapses `MatMul → Add → activation` into one kernel pass with one memory pass.
- **`GemmPlan::precision_guarantee()`** — per-kernel `PrecisionGuarantee` accessor matching Fuel's own schema.

Two related but separable tracks:

- **Track A** — mechanical alpha.7 → alpha.13 version bump (no behavior change, pulls in bug fixes).
- **Track B** — add `baracuda-cutlass = "0.0.1-alpha.13"` and start wiring CUTLASS as **alternative kernels** at relevant decision points. Per architecture v1.0, this is the first real exercise of the per-decision-point alternatives framing on CUDA (`Op::MatMul` and `Op::FusedLinear`).

Track A is ~30 min, low risk. Track B is multi-PR; each step (a)–(e) sized for its own commit. The session may stop after any step.

This session is **parallel-safe** with Phase 7.6 (which restructures the FusedOpRegistry surface) modulo the shared `BackendImpl.precision: PrecisionGuarantee` field. Coordinate at the dispatch-registration site if Phase 7.6 has progressed.

## Read first (in this order)

1. **`docs/architecture/00-index.md`** + **`docs/architecture/01-identity.md`** — grounding. Architecture v1.0 is the source of truth.
2. **`docs/architecture/04-optimization.md`** §"Per-decision-point alternatives" — CUTLASS-backed matmul registers as one alternative; existing cuBLAS-backed matmul registers as another. Same decision point, both populated. The optimizer + route-picker rank them by cost + telemetry. **Don't replace the cuBLAS path; add CUTLASS as a sibling.**
3. **`docs/architecture/05-backend-contract.md`** §`PrecisionGuarantee` and §"always-built `bit_stable` coverage commitment." CUTLASS kernels are typically *not* bit-stable (TF32, fast-math intrinsics). Their `PrecisionGuarantee` declares this; the always-built CPU backend's `bit_stable` coverage upholds the architecture's correctness anchor. CUTLASS kernels are about throughput.
4. **Memory entry `project_baracuda_cutlass_critique.md`** — what was missing in alpha.7 and why. Every Tier-1/Tier-2 ask in that critique is now shipped on crates.io.
5. **Memory entry `project_op_recip_abs_shipped.md`** + the registration pattern at `fuel-storage/src/dispatch.rs` — how kernels register today. CUTLASS-backed kernels register the same way; the `BackendImpl.precision` field absorbs the alpha.8 `GemmPlan::precision_guarantee()` output.
6. **Memory entry `project_dev_environment.md`** — this Windows host has working CUDA on RTX 4070 (sm_86, Ampere). Alpha.13's TF32 + Rrr SKUs are sm_80; both run on this hardware. Run `#[ignore]`d live-CUDA tests after every kernel-touching commit; don't defer to "the user's other box."
7. **Memory entry `project_phase_7_6_design_v2_ready.md`** — Phase 7.6 status. If step 6 (`BackendImpl`-shaped registration) has landed, use it; if not, use the existing binding-table path.

## What this session must NOT do

- **Don't replace the cuBLAS matmul paths.** Architecture v1.0 commits to alternatives at decision points. The existing `gemm_strided_batched_{f32,f16,bf16,f64}` functions (storage.rs:3021/3071/3130/3189) stay in place. CUTLASS becomes a sibling registration in the binding table.
- **Don't fold all alpha.13 capabilities into one PR.** Each capability (Rrr GEMM, TF32 GEMM, BatchedGemmPlan, Bias epilogue, Bias+activation epilogue) is one step. Each step is one commit minimum and ideally one test commit alongside.
- **Don't fuse new optimizer rules speculatively.** `Op::FusedLinear` already exists (lib.rs:742) and `fuse_linear` already runs (opt.rs:1032). The Bias+activation epilogue work requires extending `fuse_linear` to also fuse `MatMul → Add → activation`, *or* introducing a new `Op::FusedLinearActivation` primitive, *or* registering through `FusedOpRegistry`. Engage critically — pick the architecturally-cleanest home and document why before writing the rule.
- **Don't transmute around `view_as`.** The existing `as_raw().0` raw-pointer casts in `gemm_strided_batched_*` work for cuBLAS but are not the model for new CUTLASS code. New cutlass call sites use `view_as::<T>()` on `Arc<DeviceBuffer<u8>>`. The cuBLAS call sites are not retrofitted in this session — they're not broken, just not idiomatic.
- **Don't push to remote.**

## Branch and starting state

- **Current branch**: `feature/storage-unification`. Verify the tip: `git log --oneline -5`. At the time of this prompt (2026-05-11), tip is `3d1c6fbe` (Op::Flip-as-view).
- **Coordination notes**: parallel-safe with the in-flight session prompts in `docs/session-prompts/` (cast-fusion, cpu-kernel-trait-chassis, fill-op-primitive-set, phase-7-6-steps-1-3). Conflicts would arise only if another session is also editing `fuel-cuda-backend/src/storage.rs` matmul paths simultaneously.

## Track A — Version bump (single commit, ~30 min)

### Step A1: Bump workspace pins

Edit `Cargo.toml` (root workspace) lines 164–178: change every `baracuda-* = { version = "0.0.1-alpha.7", … }` to `"0.0.1-alpha.13"`. **Don't** add `baracuda-cutlass` here yet — that's Track B step B1.

### Step A2: Verify and test

```bash
cargo update -p baracuda-types -p baracuda-core -p baracuda-cuda-sys -p baracuda-driver -p baracuda-runtime
cargo check --workspace --features cuda
cargo check --workspace --features "cuda cudnn"
cargo check --workspace --features "cuda cudnn nccl"
cargo test -p fuel-cuda-backend --features cuda --lib
cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored --nocapture
```

The alpha.7 → alpha.13 jump traverses six minor releases of unrelated bug fixes; no breaking changes affect Fuel directly because `baracuda-cutlass` isn't yet a dependency. **Watch for**:

- Any `cargo update` regressions on transitive crates (half, num-traits, etc.).
- Any new lints from baracuda's macros that now-tighter rustc rules flag.
- The original alpha.4/.5 cuDNN search-dir fix on Windows is still present (verify `cudnn` feature compiles + the yolov8 anchor still loads on this host if convenient).

Commit message: `chore(deps): bump baracuda 0.0.1-alpha.7 → 0.0.1-alpha.13`. Body: short list of what alpha.13 enables (cutlass surface) without claiming any of it is wired up.

**End of Track A.** Track B steps below are optional and sequenced.

## Track B — Wire CUTLASS as alternative kernels

### Step B1: Add `baracuda-cutlass` dependency (single commit)

1. Workspace `Cargo.toml`: add `baracuda-cutlass = { version = "0.0.1-alpha.13" }` to the baracuda block (~line 178, after `baracuda-nccl`).
2. `fuel-cuda-backend/Cargo.toml`: add `baracuda-cutlass = { workspace = true }` to `[dependencies]`.
3. Add `use baracuda_cutlass::*;` to a new file `fuel-cuda-backend/src/cutlass.rs` — empty module for now, just `pub use` to confirm the crate compiles.
4. Wire `mod cutlass;` in `fuel-cuda-backend/src/lib.rs`.
5. `cargo check -p fuel-cuda-backend --features cuda`.
6. `cargo build -p fuel-cuda-backend --features cuda` — confirms linking against the cutlass-kernels-sys static lib succeeds on this host (expect ~80s build for the kernel .cu compilation if it's the first build).

Commit: `feat(cuda): add baracuda-cutlass dep + empty bridge module`. The bridge module is the integration seam for steps B2–B5.

### Step B2: View-as adoption at byte-storage boundary (single commit)

The matmul callsite ([storage.rs:3038-3040](../../fuel-cuda-backend/src/storage.rs#L3038)) currently does `let a_ptr = a.as_raw().0 as *const _;`. This works for cuBLAS but not for cutlass. The clean fix:

1. In `fuel-cuda-backend/src/byte_storage.rs`, add a typed-view helper:
   ```rust
   pub fn view_as<T: baracuda_driver::DeviceRepr>(&self) -> Result<baracuda_driver::DeviceSlice<'_, T>, CudaError>
   ```
   Implementation: call `Arc::as_ref(&self.buf).view_as::<T>()` (alpha.8 API). Return `CudaError::InvalidDtypeBoundary` if `view_as` rejects (byte-divisibility failure).
2. Use it from one cutlass call site (added in B3 below). **Don't** retrofit the existing cuBLAS paths in this session — they work; the cuBLAS surface is byte-pointer-shaped on purpose.

This commit lands the helper but no behavior change unless B3 also lands. If B3 won't land this session, the helper is dead code — that's fine, it's the seam.

Commit: `feat(cuda): add CudaStorageBytes::view_as<T> helper for safe typed views`.

### Step B3: Register CUTLASS bf16 matmul as alternative (1–2 commits)

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

### Step B4: Mirror f16 matmul (single commit)

Identical to B3 but for `(MatMul, [F16, F16, F16], Cuda)`. The Rrr layout is shipped for f16 too (alpha.9). Reuse `cutlass_matmul_<T>` as a generic over `T: CutlassElement` if it falls out cleanly; otherwise duplicate.

Commit: `feat(cuda): CUTLASS f16 matmul mirror of bf16 Rrr path`.

### Step B5: TF32 matmul (f32 input → tensor cores) (single commit)

`(MatMul, [F32, F32, F32], Cuda)` currently has one impl (cuBLAS with `Compute32FFastTF32`). Add a CUTLASS sibling using `GemmPlan::<f32>::select` (alpha.9 routes f32 input through TF32 tensor cores in Rcr layout). **Note**: Rrr × F32 is *not* shipped (alpha.13's "Things that aren't in alpha.13" list); use Rcr. This means an RHS transpose pass is needed for f32 — measure whether the perf gain still beats cuBLAS Fast-TF32 after the transpose tax. If it doesn't, register it anyway (the Judge will rank cuBLAS higher and route there empirically) and let the data settle.

Fill `BackendImpl.precision`:
- `bit_stable_on_same_hardware: false`
- `accumulator: F32`
- `math_precision: Tf32`
- `max_relative: ~1e-3`

Commit: `feat(cuda): CUTLASS f32 matmul via TF32 tensor cores (Rcr layout)`.

### Step B6: BatchedGemmPlan for equal-batch fast path (single commit)

The existing `matmul()` path at storage.rs:2697 uses `StridedBatchedConfig` uniformly. For uniform-batch cases (the common transformer attention shape), `BatchedGemmPlan` is the direct cutlass equivalent. Register as a third alternative at the bf16/f16 MatMul decision points (only when the batch dims are uniform — Judge selects empirically per shape class).

`BatchedGemmPlan` is `Identity`-only in alpha.13 (no bias). That's fine for raw `Op::MatMul`; bias-fused matmul still routes through the single-GEMM `GemmPlan` path.

Commit: `feat(cuda): CUTLASS BatchedGemmPlan as alternative for uniform-batch matmul`.

### Step B7: Fused-Linear via Bias epilogue (1–2 commits)

`Op::FusedLinear` at [lib.rs:742](../../fuel-graph/src/lib.rs#L742) is emitted by [`fuse_linear`](../../fuel-graph/src/opt.rs#L1032) but has *no* CUDA kernel today (falls back to decomposed matmul + add). Register a CUTLASS-backed kernel:

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

### Step B8: Fused-Linear-Activation via Bias+activation epilogue (2–3 commits)

Alpha.11/.12 ship `EpilogueKind::{BiasRelu, BiasGelu, BiasSilu}`. This collapses three ops (`MatMul → Add → activation`) into one kernel. Fuel doesn't yet have an `Op::FusedLinearActivation` primitive; engage critically on the architecturally-clean shape:

**Option (a)**: Extend `Op::FusedLinear` to carry an `Option<ActivationKind>` field. Pro: one variant, minimum IR churn. Con: changes Op::FusedLinear's shape, which is widely consumed.

**Option (b)**: Add `Op::FusedLinearActivation { activation: ActivationKind }` as a new primitive. Pro: orthogonal to existing `Op::FusedLinear`. Con: new IR variant, every exhaustive consumer updates, decomposes through the same chain so it's redundant at the graph level.

**Option (c)**: Register via `FusedOpRegistry` (Phase 7.6 home for compositions). Pro: cross-backend visibility (Vulkan/CPU decompose; CUDA fires the fused kernel); architecturally aligned with Phase 7.6's purpose. Con: requires Phase 7.6 step 3+ infrastructure (which is shipped, per `project_phase_7_6_step_3_shipped.md`) and the rule that fuses `MatMul → Add → activation` into the registry entry.

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

## Verification (per the alpha.13 changelog's "should investigate" list)

1. **`view_as` round-trip** — pull a `DeviceBuffer<u8>` from `CudaStorageBytes`, call `view_as::<bf16>()`, hand to `MatrixRef`, run a known-good GEMM, compare against the typed-buffer baseline. **The single most important regression check.** Lands in B2/B3.
2. **Rrr matmul** — replace the RHS-transpose pass with `LayoutSku::Rrr` for the `Op::MatMul` path and verify the output matches cuBLAS. Lands in B3.
3. **Bias-fused Linear** — route `Op::FusedLinear` through `EpilogueKind::Bias`. Single-kernel-launch confirmation via `nsys` (or `cuda-gdb`) is the qualitative win. Lands in B7.
4. **Bias + activation** — same as above for `Op::FusedLinearActivation`. Three kernel launches collapse to one. Lands in B8.
5. **`precision_guarantee()` mapping** — extract the four fields per `(layout, dtype, epilogue)` and map to Fuel's `PrecisionGuarantee` type. Stable per SKU — safe to cache at registration time. Lands incrementally across B3–B8.

## Operating principles

- **Bit-stable cpu+reference remains the correctness anchor.** Every primitive Fuel ships has bit-stable kernels in fuel-cpu-backend + fuel-reference-backend. CUTLASS kernels do not replace those; they're throughput alternatives. Their `PrecisionGuarantee.bit_stable_on_same_hardware` is what the architecture's precision-filter pass consults.
- **No production panics.** Result-returning everywhere (consistent with `feedback_no_panics_in_production`). The `view_as` helper returns `Result`; the cutlass call sites surface `baracuda_cutlass::Error` through the existing `CudaError` enum.
- **Engage critically.** Specifically for: (a) Op::FusedLinear backward semantics with the fused forward (step B7), (b) the FusedLinearActivation home choice (step B8 — registry vs Op enum), (c) whether to retrofit cuBLAS call sites to use `view_as` opportunistically (default: no, separate session).
- **One commit per logical step.** Each B-step is self-contained and bisectable.
- **Live-test on this host after every kernel-touching commit.** Per `project_dev_environment.md`. RTX 4070 supports sm_86; alpha.13's sm_80 kernels run here. Don't defer to "the other box."
- **Update memory after each step lands.** Short topic file (e.g., `project_cutlass_bf16_matmul_shipped.md`, `project_cutlass_fused_linear_shipped.md`) plus a one-line MEMORY.md index entry. Keep MEMORY.md under the 24.4KB limit (it's at the limit today — prune older entries if needed).
- **Don't push to remote unless asked.**

## End-of-session deliverable

If only Track A lands: workspace pinned to alpha.13, one commit, all tests green. ~30 min.

If through step B3: CUTLASS bf16 matmul registered as a sibling alternative to cuBLAS at one decision point. ~4–6 commits. First exercise of architecture v1.0's per-decision-point alternatives on CUDA.

If through step B8: full alpha.13 surface integrated — Rrr matmul (bf16/f16), TF32 matmul (f32), BatchedGemmPlan (uniform batch), FusedLinear bias, FusedLinearActivation (3 activations). ~12–15 commits. The CUDA matmul + Linear surface now has 3–4 alternatives per decision point, ranked empirically.

## Coordination notes

- **`Op` enum** — only step B8 *might* extend it (option b in the design decision). Default to option c (registry) — no Op enum changes.
- **`baracuda-cutlass`** — net-new dependency. No other crate currently depends on it; this session is its first consumer.
- **`fuel-cublaslt`** — separate crate, exposes cuBLASLt-backed bias+activation already. Out of scope for this session; both will eventually compete at the FusedLinear decision points and the Judge ranks them.
- **Phase 7.6** — if step 6 (`BackendImpl`-shaped registration) lands before this session starts, use that registration shape. If not, use the existing binding-table path; a later session migrates the cutlass registrations to `BackendImpl` form alongside the rest.
- **Memory entry to write at the end**: `project_cutlass_integration_session_<date>.md` summarizing which steps landed. Mention the alpha.13 capability matrix (the "Kernel SKU coverage" table from the baracuda team's note) so a future session knows exactly what's available without re-deriving it.
