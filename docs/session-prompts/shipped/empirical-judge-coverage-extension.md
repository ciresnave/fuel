# Session prompt — Empirical Judge coverage extension

## What this session is for

Extend the empirical Judge (`fuel-core/src/judge.rs`) to profile every primitive `OpKind`, not just the 2 it covers today (`MatMul` + `AddElementwise`). The Judge architecture is in place; this session populates its coverage so the cost model's layer-2 (empirical Judge data) actually has something to feed the optimizer's per-decision-point alternative ranking.

Per architecture v1.0 [§04 cost model](../architecture/04-optimization.md#cost-model-static-annotations-refined-by-empirical-judge-data-accounting-for-parallelism), layer-2 is the load-bearing refinement of the static cost annotations. Without coverage, layer-2 is hollow — every alternative's cost is whatever its static annotation said, and runtime telemetry never finds opportunities to demote a static-best plan that's empirically slower than its alternatives.

This session is parallel-safe with both Phase 7.6 and CUDA Tier 1 work. The Judge's profiling expansion is mechanical addition of probe sites + match arms; it doesn't conflict with structural refactors in other parts of the tree.

## Read first (in this order)

1. **`docs/architecture/00-index.md`** + **`docs/architecture/01-identity.md`** — grounding.
2. **`docs/architecture/04-optimization.md` §"Cost model"** — particularly layers 1+2+3 and how they compose. The Judge feeds layer 2.
3. **`docs/architecture/05-backend-contract.md` §"Dynamic telemetry"** — backends advertise per-(op, dtype, size_class, backend, device) measurements; the Judge accumulates these. Note the opt-in upload of locally-aggregated summary statistics for community sharing — that's a separate downstream concern; this session focuses on local Judge coverage.
4. **`fuel-core/src/judge.rs`** — read end-to-end. The current state: probes for MatMul + AddElementwise; size ladder defined per op family; F32 only.
5. **`fuel-core-types/src/dispatch.rs`** — `OpKind` enum (47+ variants) + `ProfileEntry` + `ProfileReport` + `DispatchTable`. The Judge writes profile entries; DispatchTable consumes them.
6. **Memory entry `project_phase7b_aocl_shipped.md`** — historical context. AOCL Router empirical dispatch shipped using Judge data for matmul; same pattern extends to other ops.
7. **Memory entry `project_dev_environment.md`** — this Windows host has working CUDA + Vulkan; Judge probes can run on real hardware locally.

## What this session must NOT do

- **Don't extend `OpKind` to include fused ops.** Fused-op profiling is a Phase 7.6 follow-on (per ROADMAP Phase 7.6 step 8 + future Phase 7.6B). This session covers primitive `OpKind` variants only.
- **Don't change the Judge's architecture.** The probe/measure/serialize pipeline works; this session adds match arms + probe runners, doesn't restructure.
- **Don't push to remote.**
- **Don't ship a probe with broken parity against the reference.** Each probe runs the op on every backend that supports it AND on the reference (per architecture v1.0's `bit_stable` coverage commitment); the comparison must be within `PrecisionGuarantee` bounds. If parity fails, that's a kernel bug to flag, not a Judge bug.

## Branch and starting state

- **Current branch (at session start)**: `feature/storage-unification`. Verify the tip with `git log --oneline -5`. If Phase 7.6 has progressed, the tip will be later; this session's work is parallel-safe with Phase 7.6 anyway.
- **Coordination**: this session adds match arms in `fuel-core/src/judge.rs` and probe runners. If Phase 7.6 step 6 has shipped (`BackendImpl` registration shape with cost-estimate function pointers), this session can also wire the Judge's measured medians into the cost-estimate layer's "empirical refinement" path — but that wiring may already be in place or may be a future session. Investigate at session start.

## Concrete work

### Phase 1: Survey current Judge coverage

```bash
grep -n "OpKind::" fuel-core/src/judge.rs
```

This shows which `OpKind` variants the Judge currently profiles. Per memory, just `MatMul` and `AddElementwise`. Confirm.

```bash
grep -nE "OpKind::[A-Z]" fuel-core-types/src/dispatch.rs | sort -u
```

This shows the full set of `OpKind` variants. The gap between this and the Judge coverage is the work surface.

### Phase 2: Decide the per-op-family probe shapes

Different op families benefit from different size ladders and dtype coverage:

| Op family | Probe shapes | Dtypes |
|---|---|---|
| Elementwise unary (Relu/Neg/Sqr/Sqrt/Recip/Abs/Tanh/Exp/Log/Sin/Cos/Sigmoid/Silu/Gelu/Step) | 2^10, 2^16, 2^20 elements | F32 (initially); BF16 / F16 / F64 follow |
| Elementwise binary (Sub/Mul/Div) | Same as unary | Same |
| Comparison (when added per Phase 7.6 step 10) | Same as unary | Same |
| Reductions (SumReduce/MaxReduce/MinReduce/MeanReduce) | 1024, 16K, 1M-element inputs reducing to {full, last-dim, first-dim} | F32 |
| MatMul | Already covered: 64×64×64, 256×256×256, 1024×1024×1024 | F32 (BF16 / F16 follow) |
| ReduceSumTo / ReduceMaxTo | Various target shapes | F32 |
| Conv2D | Small (32×32 spatial), medium (224×224), large (512×512); few channel counts | F32 |
| ConvTranspose2D | Similar to Conv2D | F32 |
| Cast | Per (src, dst) dtype pair | All combinations |
| IndexSelect / Gather | Various tensor sizes; index counts | F32 |
| Concat | Various input counts and dim sizes | F32 |
| Slice | Various source sizes; various slice ranges | F32 |
| Affine | Same as elementwise unary | F32 |
| Clamp | Same as elementwise unary | F32 |
| PowI | Same as elementwise unary; few different exponents | F32 |
| Softmax / RmsNorm / LayerNorm / Rope | Various last-dim sizes | F32 (these are *fused-op* profiling; defer per "what this session must NOT do") |

For dtype expansion: F32 first; once F32 coverage lands, BF16 / F16 / F64 are mechanical extensions.

### Phase 3: Implement probes per family, one commit per family

For each op family:

1. Add a probe-runner function (`probe_<family>_<dtype>`) in `judge.rs`.
2. Add the family to the Judge's `match` arm or dispatch table.
3. Run the probe locally on the dev machine; verify the resulting `ProfileEntry`s populate correctly.
4. Verify pre-existing tests still pass.
5. Commit. Suggested message format: `feat(judge): profile <FamilyName> across F32 size ladder`.

Per-family work is ~30 minutes - 2 hours each depending on family complexity. Elementwise families are quick (one probe pattern covers all 15+ unary or 5 binary ops); reductions are slightly more involved (multiple reduction modes); Conv is the largest (multi-dimensional shape ladder).

### Phase 4: Wire Judge measurements into cost-model layer 2

If Phase 7.6 step 6 has shipped (`BackendImpl` carries cost-estimate function pointers), the Judge can update those function pointers' "empirical refinement" path with measured medians. This is the bridge from measurement → cost ranking.

If Phase 7.6 step 6 hasn't shipped, document this as TODO in the relevant `BackendImpl` registration sites (or wherever cost estimates live today); a later session does the wiring.

### Phase 5: Update DispatchTable build to use the expanded profile

The `DispatchTable::build` function in `fuel-core-types/src/dispatch.rs` builds an O(1) dispatch table from a `ProfileReport`. Verify it correctly handles the expanded `OpKind` coverage; size-class indexing and Criterion ranking should "just work" since the data shape is unchanged — just more entries.

Run a smoke test: build a `DispatchTable` from a `ProfileReport` with the expanded coverage; confirm it answers `pick(op, dtype, size_class, criterion)` for representative queries.

### Phase 6: Document the new coverage

Update `fuel-core/src/judge.rs`'s file-level doc comment to list which `OpKind` variants are now profiled. Update the `OpKind` enum's doc comments in `fuel-core-types/src/dispatch.rs` to mark which are profiled vs not.

Optional: update memory entry `project_phase7b_aocl_shipped.md` (or write a new one) to reflect the expanded Judge coverage and what optimizer wins it unblocks.

## Test commands

After each commit:

```bash
cargo test -p fuel-core --lib
cargo test -p fuel-core --features cuda --lib   # if CUDA judge probes are added
cargo test -p fuel-core-types --lib
```

The Judge probes are typically `#[ignore]`d (they take real wall-clock time); run explicitly:

```bash
cargo test -p fuel-core --lib judge_ -- --ignored
```

If your dev environment has CUDA (per `project_dev_environment.md`, this Windows host does):

```bash
cargo test -p fuel-core --features cuda --lib judge_cuda_ -- --ignored
```

## Operating principles

- **Engage critically.** If a probe consistently produces inconsistent measurements (high variance, thermal throttling visible in the data), surface it. Don't quietly normalize bad data — the Judge's value depends on the data being trustworthy.
- **Outlier rejection in aggregation later.** Per architecture v1.0 (community-aggregated empirical data per [§11-persistence](../architecture/11-persistence.md#cache-generation-and-distribution)), the upstream aggregation pipeline does outlier rejection. This session's job is to *gather* the local measurements; later sessions handle aggregation policy.
- **No production panics.** Probes return `Result`; failures are logged and skipped, not panicked.
- **Memory updates per family shipped.** Capture what was added + any landmines (probes that turned out flaky on this hardware, ops where the size ladder needed adjustment, etc.).
- **Don't push to remote unless asked.**

## End-of-session deliverable

At minimum:

- All elementwise unary + binary + comparison probes shipped (one commit per family).
- All reduction probes shipped.
- F32 coverage complete for the above families.
- All probes run locally; ProfileReport contains the expanded entries.

Stretch:

- BF16 + F16 + F64 dtype coverage for the above families.
- Convolution probes (Conv2D + ConvTranspose2D).
- IndexSelect / Gather / Concat / Slice probes.
- DispatchTable smoke test demonstrating the optimizer's runtime route picker has more options to choose from.
- Cost-model layer-2 wiring (if Phase 7.6 step 6 has shipped).

## Coordination notes

This session is fully parallel-safe with:

- **Phase 7.6**: doesn't touch the Op enum or fused-op machinery.
- **CUDA Tier 1 resume + baracuda CUTLASS**: the Judge profiles whatever CUDA kernels are registered; new CUDA kernels added in the parallel session get profiled automatically once they're available. There's a mild ordering preference: the CUDA session shipping new kernels first means the Judge probes have more to measure; but if Judge probes ship first, they correctly skip ops that don't yet have CUDA kernels (the probe respects backend support).

If multiple sessions land changes simultaneously, expect minor merge conflicts in `fuel-core/src/judge.rs` (separate match arms) and `fuel-core-types/src/dispatch.rs` (no expected conflicts; the OpKind enum is `#[non_exhaustive]`).
