# Session prompt — CUDA Tier 1 fanout resume + baracuda CUTLASS integration

## What this session is for

Two related but separable tracks of CUDA backend work:

- **Track A**: Resume CUDA Tier 1 fanout — finish the remaining mechanical Tier 1 ops that were paused for the SoftmaxLastDim foundation work + architecture set establishment.
- **Track B**: Integrate the new baracuda version with safe CUTLASS support — update the dependency, expose CUTLASS-backed kernel alternatives at relevant decision points (matmul first, others as the CUTLASS surface allows).

Both tracks are CUDA-backend focused; both are architecture-aligned (per architecture v1.0); they can interleave in one session because they touch the same crate (fuel-cuda-backend).

## Read first (in this order)

1. **`docs/architecture/00-index.md`** + **`docs/architecture/01-identity.md`** — grounding. Architecture v1.0 is the source of truth.
2. **`docs/architecture/05-backend-contract.md`** — what backends advertise. Particularly: `PrecisionGuarantee` per kernel; cost-estimate per `BackendImpl`; slot capacity. CUTLASS-backed kernels register the same way as hand-rolled ones; `PrecisionGuarantee` may differ (CUTLASS often uses TF32 and other approximations).
3. **`docs/architecture/04-optimization.md`** — particularly the per-decision-point alternatives section. CUTLASS-backed matmul registers as one alternative; existing hand-rolled matmul registers as another. The optimizer's per-decision-point alternative set carries both; the runtime route picker chooses based on cost + telemetry. **Don't replace the existing matmul; add CUTLASS as an alternative.**
4. **Memory entry `project_cuda_depth_migration_roadmap.md`** — the 3-tier CUDA migration plan. Tier 1 mechanical fanout. Tier 2 medium-effort kernels. Tier 3 wiring existing crates.
5. **Memory entries `project_cuda_*_shipped.md` series** — per-tier-step shipped state. Most recent: `project_cuda_matmul_affine_shipped.md` (MatMul + Affine), `project_cuda_cast_and_multidtype_key_shipped.md` (Cast + multi-dtype binding-table key). After Cast was the natural next op (per the memory: "SoftmaxLastDim or IndexSelect next"), but the SoftmaxLastDim foundation work paused Tier 1.
6. **Memory entry `project_baracuda_cutlass_update.md`** — what's known about the new baracuda version with safe CUTLASS support. The agent (you) needs to confirm the version with the user; details are partial.
7. **Memory entry `project_cudarc_to_baracuda_migration.md`** — the historical baracuda migration. Today fuel depends on baracuda alpha.2; the new version with CUTLASS is presumed to be a later release.
8. **Memory entry `project_dev_environment.md`** — this Windows host has working CUDA + Vulkan; run `#[ignore]`d live-GPU tests after every kernel-touching commit.
9. **Memory entry `project_phase_7_6_design_v2_ready.md`** — Phase 7.6 is queued separately. **Coordinate**: if Phase 7.6 has been started in another session, this session's CUDA work needs to use the registry-aware kernel registration shape (per architecture v1.0). If Phase 7.6 hasn't started yet, this session uses the existing binding-table registration; a later session migrates these CUDA registrations to `BackendImpl` form per Phase 7.6 step 6.

## What this session must NOT do

- **Don't refactor the kernel registration surface.** That's Phase 7.6 step 6's job. This session adds new kernels (CUTLASS-backed alternatives, remaining Tier 1 ops) using whatever registration shape exists when this session starts.
- **Don't replace existing kernels with CUTLASS-only versions.** Architecture v1.0 commits to alternatives at decision points. CUTLASS-backed matmul is added as *another* alternative; the existing hand-rolled matmul stays. The optimizer/route-picker chooses based on cost + telemetry.
- **Don't break parity with the always-built backend's `bit_stable` coverage.** CUTLASS kernels are typically NOT bit-stable (TF32, fast-math). Their `PrecisionGuarantee` reflects this. The architecture's coverage commitment ("at least one bit_stable kernel per primitive op") is upheld by the always-built backend (fuel-cpu-backend); CUDA kernels are about performance, not correctness anchors.
- **Don't push to remote.** Branch stays `feature/storage-unification` (or wherever Phase 7.6 has progressed to).

## Branch and starting state

- **Current branch (at this session start)**: `feature/storage-unification`. Verify the tip: `git log --oneline -5`. If Phase 7.6 has shipped in another session, the tip will be later than `35b1d038`.
- **Coordination with Phase 7.6**: this session is parallel-safe with Phase 7.6 because Tier 1 ops are *primitives* (Op enum primitive variants); Phase 7.6 changes how *fused* ops are tracked. CUTLASS work for matmul (a primitive) doesn't conflict with Phase 7.6's fused-op registry work. **However**: if Phase 7.6 step 6 (backend-registrations adopt BackendImpl shape) has shipped, this session's new kernel registrations should use the `BackendImpl` shape. Verify by checking what's in `fuel-storage::dispatch` at session start.

## Track A: CUDA Tier 1 fanout resume

### Current Tier 1 state (verify at session start)

Per memory, Tier 1 has shipped:

- Binary fanout (5 ops): Sub, Mul, Div, Maximum, Minimum.
- Unary fanout (15 ops): Relu, Neg, Sqr, Sqrt, Recip, Abs, Tanh, Exp, Log, Sin, Cos, Sigmoid, Silu, Gelu, Step.
- Reductions (4 ops): SumReduce (SumDim/SumAll), MaxReduce, MinReduce, MeanReduce.
- MatMul (cuBLAS-backed; equal-batch fast path + GQA per-batch loop).
- Affine (AddScalar / MulScalar / PowI promotion).
- Cast (with multi-dtype binding-table key).
- ReduceSumTo + ReduceMaxTo (PR 3.5 follow-up — native CUDA so lowered SoftmaxLastDim runs GPU-resident).

Plus PR 3.5 follow-up: SoftmaxLastDim runs end-to-end on CUDA via the lowered path.

### What's NOT yet on CUDA (per the original Tier 1 list and memory)

- IndexSelect (gather along a dim using a U32 index tensor).
- Gather (N-dimensional gather along a dim).
- Concat (concatenate N inputs along one dim — may be metadata-only, depends on layout).
- Slice (subset of a tensor — metadata-only with the Layout side-table).
- ArgMaxDim / ArgMinDim (integer-index-producing reductions).
- Clamp (elementwise; min/max from OpParams::Clamp).
- ReduceSumTo / ReduceMaxTo are shipped per PR 3.5; verify they still work post-architecture-set.

Plus possibly: Permute (likely metadata-only via Layout side-table; verify).

### Concrete work for Track A

1. **Confirm what's actually missing** with `cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored 2>&1 | tail -50` — failing/skipped tests reveal which ops fall back to CPU.
2. **For each missing op, ship one commit per op**: write the CUDA kernel (or wire to baracuda's kernel-name dispatch); register through `register_*_kernels`; add a live-CUDA equivalence test. Per the existing pattern in the memory entries `project_cuda_*_shipped.md`.
3. **Live-test each op on the dev machine**: per `project_dev_environment.md`, this Windows host has working CUDA. Run `cargo test --features cuda --test cuda_dispatch_live -- --ignored <test_name>` after each kernel commit.

## Track B: Baracuda CUTLASS integration

### Step B0: confirm the baracuda version with the user

The new baracuda version with safe CUTLASS support is mentioned in `project_baracuda_cutlass_update.md` but the specific version number isn't recorded. **Ask the user**:

- What's the version number? (alpha.3? beta.1? stable?)
- Is there documentation for the safe CUTLASS API surface?
- Any breaking changes from baracuda alpha.2 (the version fuel currently depends on)?

If the user can't answer immediately, check baracuda's repo (likely under the user's GitHub account or a known location) for the latest release.

### Step B1: update Cargo.toml

Update fuel-cuda-backend's baracuda dependency to the new version. Run `cargo update -p baracuda`. Run `cargo build --features cuda` to surface compile errors from breaking changes; fix incrementally.

### Step B2: investigate the safe CUTLASS surface

Read baracuda's CUTLASS-related modules. Identify:

- Which CUTLASS surfaces are exposed in safe form (matmul? conv? attention? specific shapes only?).
- The Rust API shape (function signatures, error types).
- Documented `PrecisionGuarantee` characteristics (TF32? FP16 with FP32 accumulator? Specific ULP bounds?).
- Performance characteristics (which shapes/dtypes CUTLASS wins on).

### Step B3: add CUTLASS-backed matmul as an alternative

Start with matmul because:
- Matmul is the highest-impact CUDA kernel (transformer inference is matmul-bound).
- Existing CUDA matmul (cuBLAS) is the comparison baseline; CUTLASS-backed matmul should win on at least some shape regimes.
- Pattern is straightforward: register a second kernel for `(MatMul, [F32, F32, F32], BackendId::Cuda)` alongside the existing one.

The architectural model (per architecture v1.0):

- Both matmul kernels register; both become alternatives at decision points where matmul appears.
- The optimizer's cost-model layer 1 (static annotations) ranks them; layer 2 (Judge profile data) refines once measurements accumulate; layer 3 (runtime telemetry) adapts at dispatch.
- `PrecisionGuarantee` differs: existing matmul → `bit_stable_on_same_hardware: false; max_ulp: ?` (cuBLAS isn't bit-stable cross-driver). CUTLASS matmul → `bit_stable_on_same_hardware: false; max_ulp: <tighter or looser based on CUTLASS docs>`.

If Phase 7.6 step 6 has not yet shipped (`BackendImpl` registration shape), use the existing `register_*_kernels` form. Add the `PrecisionGuarantee` and cost-estimate metadata to the kernel's registration site (or as TODO comments on the registration site to fill in when Phase 7.6 step 6 ships).

### Step B4: add CUTLASS-backed alternatives for other surfaces (as time and CUTLASS surface allow)

In rough order of impact:

- **Conv2D** if CUTLASS exposes implicit-GEMM convolution — matters for ConvNeXt / ResNet / image-tower models.
- **FlashAttention-style attention** if baracuda's CUTLASS surface includes it — matters for transformer inference; major perf win if available.
- **MatMul variants** for non-F32 dtypes (BF16, F16) if exposed.

Each addition is one commit + live-CUDA test verifying parity (within `PrecisionGuarantee` bounds) against the existing kernel.

### Step B5: live tests for all CUTLASS-backed kernels

Per `project_dev_environment.md`, run on the dev machine:

```bash
cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored matmul
cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored conv
# ... etc per added kernel
```

Both the existing kernel and the new CUTLASS-backed alternative should pass; the equivalence test compares against the always-built backend's `bit_stable` reference within `PrecisionGuarantee` bounds.

## Test commands

After each commit:

```bash
cargo build --features cuda
cargo test -p fuel-cuda-backend --features cuda
cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored
cargo test -p fuel-core --features cuda --lib
```

Pre-existing failures (unrelated): `pipelined_realize_cast_f32_to_f64` and `pipelined_realize_cast_round_trip_via_bf16` in fuel-storage lib.

## Operating principles

- **Engage critically.** If baracuda's CUTLASS surface doesn't quite match what the user described, surface that and ask. Don't pretend something exists if you can't find it in the API.
- **Architectural cleanness over local pragmatism.** If a CUTLASS kernel is genuinely strictly better than the existing one (no shape regime where the existing one wins), still add CUTLASS as a *new* kernel — let the optimizer's cost model demote the loser, don't manually replace.
- **Live-test on the dev machine.** Per `project_dev_environment.md`, kernel-touching commits get exercised locally before being declared done.
- **No production panics.** Result-returning throughout. CUTLASS errors propagate as `Result::Err`, not panics.
- **Don't push to remote unless asked.**
- **Per-commit memory updates.** Each shipped op (Tier 1 or CUTLASS-backed) gets a memory entry capturing what shipped + landmines, per the `project_cuda_*_shipped.md` precedent.

## End-of-session deliverable

At minimum:

- 2-3 of the missing Tier 1 ops shipped (whichever order makes sense; IndexSelect or Gather are good first picks because they're broadly useful and not trivially metadata-only).
- Baracuda dependency updated to the new version with CUTLASS support.
- CUTLASS-backed matmul registered as an alternative kernel; live-CUDA equivalence test green.
- Memory entries updated for each shipped item.

Stretch:

- All remaining Tier 1 ops shipped.
- CUTLASS-backed Conv2D and/or attention as additional alternatives.
- Cost-estimate functions populated for the new kernels (informs the cost-model layer-1 ranking).

## Coordination notes for the user

Two cross-cutting concerns the user should be aware of:

1. **If Phase 7.6 lands first**: this session's kernel registrations may need to migrate to `BackendImpl` form per Phase 7.6 step 6. The migration is mechanical; not a blocker. Document the migration debt in commit messages so a later session knows what to update.

2. **If Phase 7.6 hasn't started**: this session's work is fully parallel-safe. Phase 7.6 will pick up the new kernels naturally during its step 6 backend-registration migration.
