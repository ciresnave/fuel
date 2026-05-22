# Session prompt — Phase 7.6 steps 1-3 (registry skeleton + Op enum extension + SoftmaxLastDim end-to-end)

## What this session is for

Implement the foundational portion of Phase 7.6: establish the FusedOpRegistry types, extend the `Op` enum with the `Op::Fused(id, params)` arm, and migrate one fused op (SoftmaxLastDim) end-to-end as proof of concept. After this session, the v1.0 architecture's central refactor is real in code; one fused op flows through the registry path and the live CUDA equivalence test still passes.

This is the literal bottom of Fuel's outstanding architectural work — everything in v1.0 stacks on what these three steps establish.

## Read first (in this order)

1. **`docs/architecture/00-index.md`** — table of contents, reading order, cross-link map.
2. **`docs/architecture/01-identity.md`** — five competitive edges + the bet. Grounds *why* Phase 7.6 matters.
3. **`docs/architecture/03-ir.md`** — the IR. Op enum (closed primitives + `Op::Fused` arm), Node, base map vs optimized form, pre-resolved KernelRef. **The most load-bearing section for this session.**
4. **`docs/architecture/04-optimization.md`** — DecompositionMap, OptimizationMap, per-decision-point alternatives. Phase 7.6's auto-generated lowering+fusion rules feed this.
5. **`docs/architecture/05-backend-contract.md`** — what backends advertise, `PrecisionGuarantee`, BackendImpl. Step 7 of Phase 7.6 will populate `PrecisionGuarantee`; for steps 1-3, just reference the structure.
6. **`docs/fused-op-registry.md` v2** — the implementation guide. End-to-end. Type shapes + 11-step migration path + open implementation questions. **This is your work plan.**
7. **Memory entry `project_phase_7_6_design_v2_ready.md`** — quick-recall summary.
8. **Memory entries `project_pr3_rule_registry_shipped.md` and `project_pr3_5_*` series** — what's already shipped that this builds on. PR 3's Rule trait + RuleRegistry is the substrate; PR 3.5's Op::ReduceMaxTo + Op::Unsqueeze + Op::ReduceMaxToBackward are primitives the SoftmaxLastDim decomposition uses.
9. **Memory entry `project_architecture_doc_set_v1_0.md`** — the architecture set's establishment summary; 24 captured decisions.

## What this session must NOT do

- **Don't re-litigate the architecture.** Op-shape A is locked; per-decision-point alternatives are locked; pre-resolved KernelRef is locked; PrecisionGuarantee replaces OracleGrade. If you find a gap, surface it; don't quietly redesign. The architecture set in `docs/architecture/` is the constitutional document.
- **Don't go past step 3.** Steps 4-11 are sequenced after this session; they need their own sessions or follow-ups. Stopping at step 3 is a clean state — one fused op flows through the new architecture; tests green; rest of fused ops still on legacy variants. That's the natural pause point.
- **Don't restructure unrelated code.** This session adds the registry skeleton + extends Op + migrates one op. It doesn't touch CUDA Tier 1 fanout, doesn't touch storage unification phases B/D, doesn't touch autograd's GradientRule migration.
- **Don't write Op variants for fused ops.** `Op::Fused(FusedOpId, FusedOpParams)` is the *only* path for fused ops post-migration. Resist the muscle memory of adding `Op::SoftmaxLastDim` etc. — those exist legacy-style for now and get dropped in step 5 (later session).
- **Don't introduce a `NodeKind` discriminator.** Architecture v1.0 explicitly chose Op-shape A over the original NodeKind framing. The single `Op` enum carries everything.

## Branch and starting state

- **Current branch**: `feature/storage-unification`.
- **Code tip (last committed)**: `35b1d038` — "docs(roadmap): Phase 7.6 FusedOpRegistry — design doc + ROADMAP entry".
- **Uncommitted changes you'll find on disk**: ROADMAP.md, docs/fused-op-registry.md, docs/storage-unification.md, GUIDE.md, README.md (all updated to align with architecture v1.0); docs/architecture/ (new directory, 11 sections); docs/architecture-audit.md (postscript added).

**First action**: commit the uncommitted architecture-doc work as a single doc-only commit (no code changes in it). Suggested commit message:

```text
docs(architecture): establish architecture set v1.0 + sync surrounding docs

- Add docs/architecture/ (11 sections, v1.0): identity, layers, IR, optimization,
  backend-contract, runtime, tolerance, pattern-harvest, non-goals, decisions-log,
  persistence. Constitutional document for fuel.
- Update ROADMAP.md: preamble pointing at architecture set; Phase 7.6 entry
  rewritten to v2 against v1.0.
- Update docs/fused-op-registry.md to v2 (anchored to architecture v1.0 type shapes).
- Update docs/storage-unification.md and docs/architecture-audit.md with
  postscripts noting architecture v1.0 is now source of truth.
- Add architecture pointers to README.md and GUIDE.md.

24 architectural decisions recorded in docs/architecture/10-decisions-log.md.
```

After that commit, the code state is clean and you can start step 1 with no ambiguity about what's doc vs code.

## The three steps

### Step 1: registry skeleton (~1-2 days)

Add the types. No callers; types compile; no behavior change. Tree green.

In `fuel-graph` (metadata side):

```rust
pub struct FusedOpId(pub u16);

pub struct FusedOpRegistry {
    entries:         Vec<FusedOpEntry>,
    by_name:         HashMap<&'static str, FusedOpId>,
    by_pattern_hash: HashMap<PatternHash, FusedOpId>,
}

pub struct FusedOpEntry {
    pub id:            FusedOpId,
    pub name:          &'static str,
    pub family:        FusedOpFamily,
    pub pattern:       SubgraphPattern,
    pub decompose:     fn(&mut Graph, NodeId, &FusedOpParams) -> NodeId,
    pub backward:      BackwardKind,
    pub backend_impls: SmallVec<[(BackendId, BackendImpl); 4]>,
    pub shape_rule:    fn(&[Shape], &FusedOpParams) -> Shape,
    pub dtype_rule:    fn(&[DType], &FusedOpParams) -> DType,
}

pub enum FusedOpParams {
    SoftmaxLastDim,
    // (this session adds only this one; step 4 of Phase 7.6 — later session — adds the others)
}

pub enum FusedOpFamily { Forward, Backward, Quantized, Attention, Norm }

pub enum BackwardKind {
    Fused(FusedOpId),
    Decompose,
    NotDifferentiable,
}

pub enum SubgraphPattern {
    Declarative(PatternTree),
    Callable(fn(&Graph, NodeId) -> Option<Match>),
}
```

In `fuel-storage` (BackendImpl payload side):

```rust
pub struct BackendImpl {
    pub kernel:    KernelRef,
    pub cost:      fn(&[Shape], &FusedOpParams, &BackendCapabilities) -> CostEstimate,
    pub precision: PrecisionGuarantee,
    pub caps:      KernelCaps,
    pub revision:  KernelRevisionHash,
}

pub struct CostEstimate {
    pub flops:              u64,
    pub bytes_moved:        u64,
    pub kernel_overhead_ns: u32,
}

pub struct PrecisionGuarantee {
    pub bit_stable_on_same_hardware: bool,
    pub max_ulp:      Option<u32>,
    pub max_relative: Option<f64>,
    pub max_absolute: Option<f64>,
    pub notes:        &'static str,
}

pub struct KernelRevisionHash(pub u64);  // newtype; computed from kernel source + version
```

Don't yet wire these into anything. Compile, run all existing tests, commit.

Suggested commit message: `feat(graph,storage): Phase 7.6 step 1 — FusedOpRegistry skeleton (types only)`.

### Step 2: extend Op enum with `Op::Fused(FusedOpId, FusedOpParams)` arm (~1-2 days)

Add the variant to `Op`. Existing variants (`Op::SoftmaxLastDim`, `Op::FlashAttn`, etc.) coexist with the new arm during migration. Update exhaustive consumers:

- `op_short_name` (in fuel-graph) — add an arm for `Op::Fused(id, _)` that returns the registry entry's name.
- `op_key` (in fuel-graph/src/opt.rs) — add an arm for `Op::Fused(id, params)` that derives a hashable key from `(id, hash(params))`. CSE depends on this; two `Op::Fused` nodes with identical id + params should CSE to one node.
- Autograd's match-on-Op (in fuel-graph/src/lib.rs around the `Tensor::backward` 600-line match) — add an arm for `Op::Fused(id, _)` that's initially `unreachable!()` (no fused-op nodes are emitted as `Op::Fused` yet; the arm gets implemented in step 3).
- Anywhere else that exhaustively matches on `Op` (the legacy executor, any printer / Debug impl, etc.) — add either a delegation to the registry or an `unreachable!()` initially. **Search the codebase**: `grep -rn "match.*Op\|match.*node.op" fuel-*/src/` to find every exhaustive consumer.

Tree compiles green; no behavior change yet (no nodes use `Op::Fused`). All existing tests pass.

Suggested commit message: `feat(graph): Phase 7.6 step 2 — Op enum gains Op::Fused(FusedOpId, FusedOpParams) arm`.

### Step 3: migrate first fused op (SoftmaxLastDim) end-to-end (~2-3 days)

The proof-of-concept commit. After this step, one fused op flows through the registry path; the others use the legacy variants.

Sub-steps:

1. **Create the SoftmaxLastDim registry entry** in fuel-graph. Fields:
   - `id`: const `FusedOps::SOFTMAX_LAST_DIM = FusedOpId(1)`.
   - `name`: `"SoftmaxLastDim"`.
   - `family`: `FusedOpFamily::Forward`.
   - `pattern`: the canonical 7-node primitive subgraph (per `fuel-graph/src/opt.rs::canonical_softmax_pattern` — port to `SubgraphPattern::Callable` initially; declarative-pattern-tree form is an open question per `docs/fused-op-registry.md` §Q1).
   - `decompose`: function that emits the 7-node primitive subgraph (per the existing `SoftmaxLastDimLowerRule::rewrite` in opt.rs).
   - `backward`: `BackwardKind::Fused(SOFTMAX_LAST_DIM_BACKWARD_ID)` — but the backward fused-op isn't migrated this session. Two options: (a) leave a stub `BackwardKind::Decompose` that uses the primitive decomposition's backward, or (b) wire to the legacy `Op::SoftmaxLastDimBackward` variant via a temporary bridge. Option (a) is cleaner for this session; revisit when migrating backward helpers in step 4.
   - `backend_impls`: at minimum the existing CPU kernel (in fuel-cpu-backend); CUDA can either be added now or in step 4.
   - `shape_rule`, `dtype_rule`: SoftmaxLastDim preserves shape and dtype.

2. **Teach the executor to dispatch `Op::Fused(SOFTMAX_LAST_DIM_ID, _)`** through the registry. In fuel-graph-executor's eval_node (or wherever per-node dispatch happens), add an arm for `Op::Fused(id, params)` that:
   - Looks up the registry entry by `id`.
   - Resolves the appropriate `BackendImpl` for the chosen backend (per architecture v1.0 §"Pre-resolved KernelRef": ideally pre-resolved at planning time; for this session a runtime lookup is acceptable since step 9 does the planning-time refactor).
   - Calls `backend_impl.kernel(inputs, outputs, layouts, params)`.

3. **Update `Tensor::softmax_last_dim()` builder** to emit `Op::Fused(FusedOps::SOFTMAX_LAST_DIM, FusedOpParams::SoftmaxLastDim)` instead of `Op::SoftmaxLastDim`. The existing `Op::SoftmaxLastDim` variant stays in the enum for now; nothing emits it after this session.

4. **Auto-generate the SoftmaxLastDim lowering and fusion rules from the registry entry**:
   - `LoweringRule { id: SOFTMAX_LAST_DIM, decompose: entry.decompose }`.
   - `FusionRule { id: SOFTMAX_LAST_DIM, pattern: entry.pattern }`.
   - Register them with the existing `RuleRegistry`.

5. **Delete PR 3's hand-written rules**: `SoftmaxLastDimLowerRule` and `SoftmaxLastDimFuseRule` (in fuel-graph/src/opt.rs around lines 295-498 per memory).

6. **Verify**: tree compiles green; all existing tests pass; the live CUDA equivalence test (`cuda_executor_matches_cpu_on_softmax_via_lowering` in fuel-core/src/lazy.rs) still passes — the lowered subgraph still runs natively on CUDA via the registry-dispatched path.

Suggested commit message: `feat(graph,storage): Phase 7.6 step 3 — SoftmaxLastDim migrates to FusedOpRegistry end-to-end`.

## Test commands

Run after each step's commit. All must stay green:

```bash
cargo test -p fuel-graph --lib
cargo test -p fuel-cpu-backend --lib
cargo test -p fuel-reference-backend --lib
cargo test -p fuel-graph-router --lib
cargo test -p fuel-core --features cuda --lib
cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored
```

**Pre-existing failures unrelated to this session**: `pipelined_realize_cast_f32_to_f64` and `pipelined_realize_cast_round_trip_via_bf16` in fuel-storage lib are cross-dtype Cast gaps that pre-date PR 3 and are not this session's concern.

The dev environment has working CUDA + Vulkan; per memory `project_dev_environment.md`, run `#[ignore]`d live-GPU tests locally after every kernel-touching commit.

## Operating principles

- **Engage critically; don't defer.** If the architecture v1.0 design has a concrete gap that surfaces during implementation, flag it and propose the fix. The architecture is iterable; v1.0 → v1.1 is fine if the fix is real. Don't quietly work around a gap.
- **Architectural cleanness over local pragmatism.** If migration step N reveals that step N+1's design is suboptimal, raise the question before forging ahead.
- **No production panics.** Result-returning throughout; the `unreachable!()` arms in step 2 should become `Err(...)` returns by step 3 (where the actual dispatch is implemented).
- **Don't push to remote unless asked.** Branch tip stays `feature/storage-unification` when done.
- **Save memory entries after each step.** Per-step commit + per-step memory entry capturing what landed and any landmines. Future-session pickup depends on these being current.

## End-of-session deliverable

At minimum, all three steps shipped: skeleton committed, Op enum extension committed, SoftmaxLastDim migrated end-to-end, live CUDA equivalence test still green.

Stretch: if time allows, migrate one more fused op (RmsNormLastDim is the natural second — same shape as SoftmaxLastDim; fewer surprises).

After session completion: the v1.0 architecture is real in code for at least one fused op. Subsequent sessions migrate the remaining 12 fused ops (step 4) without architectural risk — the pattern is established.
