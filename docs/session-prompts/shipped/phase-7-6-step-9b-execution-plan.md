# Session prompt — Phase 7.6 step 9b: ExecutionPlan + lazy KernelRef pre-resolution

## What this session is for

Step 9a (commit pending review, 2026-05-12) shipped `KernelBindingTable` multi-impl alternatives per `(OpKind, dtypes, BackendId)` decision point — siblings now coexist where previously `HashMap::insert` overwrote. That unblocks baracuda-cutlass B3/B4 (CUTLASS bf16/f16 matmul as cuBLAS siblings).

Step 9b is the **second** of the three moves the design doc collapses into "step 9":

1. **9a — multi-impl registration** (shipped). `KernelBindingTable` stores `SmallVec<[BindingEntry; 2]>` per key; route picker absent, first-wins on legacy lookup.
2. **9b — planning-time route resolution** (this session). Introduce `ExecutionPlan` + `compile_plan()` + a trivial route picker. `NodeKernelBinding::kernel` starts as `None` (lazy); the first executor query through the plan resolves it via `lookup_alternatives()` and caches the chosen `KernelRef`.
3. **9c — executor migration** (separate session). `fuel-graph-executor`'s per-`Op` dispatch arms (`eval_node` at [`fuel-graph-executor/src/lib.rs:1216`](../../fuel-graph-executor/src/lib.rs#L1216)) stop calling `GraphBackend::matmul()` / `.conv2d()` / etc. and instead invoke the pre-resolved `KernelRef` function pointer the plan carries. This is the large, mechanical, every-backend touch.

**Critical engagement: read this before scoping.** 9b alone ships an `ExecutionPlan` type and a `compile_plan()` pass that have **zero callers** in the executor. That's the "type-level seam without migration" anti-pattern flagged when 9a was scoped — we've been bitten by this before. The architecturally honest move is to bundle 9b + 9c into one session so 9b's seam gets consumed immediately. Track B of this prompt does that.

If the session must stop early (time-boxed, blocked on a 9c hazard, etc.), Track A leaves 9b alone — but only as a deliberate fallback, not the default.

This session is **parallel-safe** with the baracuda-cutlass alpha.13 integration (it adds CUTLASS siblings via 9a's append-on-register; doesn't touch the executor or planning surface). Modest conflict risk with any session that edits `fuel-graph-executor/src/lib.rs:1216`'s dispatch arms — coordinate.

## Read first (in this order)

1. **`docs/architecture/00-index.md`** + **`docs/architecture/04-optimization.md`** §"Per-decision-point alternatives." Architecture v1.0 names the route picker explicitly: each decision point carries alternatives; the picker resolves one at first-use, caching for the rest of the realize call.
2. **`docs/architecture/11-persistence.md`** §"Re-resolution on use (lazy, not at load)." This is the durability story 9b's lazy resolution feeds. Cache work itself is downstream — 9b doesn't write to disk.
3. **`docs/architecture/05-backend-contract.md`** §`PrecisionGuarantee`. The route picker's v1 filter is precision-policy-driven; understand what bit-stable / max_ulp / max_relative mean before writing the filter.
4. **`docs/fused-op-registry.md`** §"Step 9: binding-table planning-time refactor" + the `DecisionPointAlternative` / `NodeKernelBinding` type sketch at lines 230–247. The doc names the shape; 9b implements it (minus 9c's executor consumption).
5. **Memory entry `project_phase_7_6_step_4_in_progress.md`** (updated post-9a; expected to include the 9a-shipped note with the new `BindingEntry` + `lookup_alternatives` API). Confirms what 9a delivered and what 9b inherits.
6. **The 9a code** (after merge): [`fuel-storage/src/kernel.rs`](../../fuel-storage/src/kernel.rs) — `BindingEntry` struct, `lookup_alternatives()`, append-on-register, duplicate-panic. 9b's route picker is the primary consumer of `lookup_alternatives()`.
7. **Memory entry `project_phase6b_probe_judge_dispatch.md`** — Phase 6b's empirical Judge. **Out of scope for 9b** but the Judge is the eventual route-picker brain; 9b's v1 picker is a placeholder the Judge replaces in a later phase.
8. **Memory entry `project_dev_environment.md`** — RTX 4070 + working CUDA. After 9c's executor migration (Track B), every kernel-touching commit needs a live-CUDA test pass on this host.

## What this session must NOT do

- **Don't build the empirical Judge integration.** Phase 6b's Judge is the long-term home for route selection. 9b's picker is a v1 heuristic: "prefer bit-stable; otherwise first registered." The Judge plugs in later as a richer policy.
- **Don't compute `KernelRevisionHash` for real.** It stays `KernelRevisionHash::UNTRACKED` everywhere in 9b. Real hashing lands when the persistence cache work begins — separate phase.
- **Don't retrofit `fuel-storage/src/compiled.rs::compile_node` to use `lookup_alternatives`.** That's a parallel dispatch surface (byte-storage pipelined path); touching it expands scope. The pipelined path can opt in later.
- **Don't expand the `OpKind` enum** or add new `OpParams` variants. 9b is purely about plan structure; no IR change.
- **Don't introduce a Plan-level optimizer pass.** Decomposition-vs-fused alternatives at the same decision point (per architecture §04) are a Plan responsibility, but adding them requires graph-rewrite plumbing — Phase 7.6 step 10 or later.
- **Don't push to remote.**
- **For Track A specifically: don't migrate executor dispatch arms.** If the session is Track A only, `eval_node` stays untouched. Track B does that work.

## Branch and starting state

- **Current branch**: `feature/storage-unification`. Verify tip after the 9a commit lands. Without 9a merged, this session has nothing to consume (`lookup_alternatives()` doesn't exist yet).
- **Coordination notes**: parallel-safe with the cutlass session (cutlass adds CUTLASS siblings via 9a's append; doesn't touch planning). Conflicts arise only if another session is also editing `fuel-storage::kernel`, `fuel-graph::Graph`, or — for Track B — `fuel-graph-executor/src/lib.rs:1216` simultaneously.

---

## Track A — 9b alone (type-level seam, ~4 commits)

**Default: do not pick this track. Use only if Track B's executor migration is blocked or time-boxed out.**

### Step A1: define `NodeKernelBinding` + `ExecutionPlan` types

In a new file [`fuel-storage/src/plan.rs`](../../fuel-storage/src/plan.rs) (fuel-storage owns the binding table; the plan lives next to it):

```rust
/// One node's lazy kernel resolution. Per architecture v1.0:
/// `kernel` starts as `None`; the route picker fills it on first
/// use, caching for the rest of the realize call.
pub struct NodeKernelBinding {
    pub node:             NodeId,
    pub op_kind:          OpKind,
    pub dtypes:           SmallVec<[DType; 8]>,
    pub backend:          BackendId,
    pub device:           DeviceLocation,
    pub kernel:           Option<KernelRef>,          // lazy: None until first use
    pub kernel_revision:  KernelRevisionHash,         // stays UNTRACKED in 9b
}

/// Execution plan for one realize call. Built once by `compile_plan`;
/// consumed by the executor (Track B) or by tests (Track A).
pub struct ExecutionPlan {
    pub order:    Vec<NodeId>,                        // topological
    pub bindings: HashMap<NodeId, NodeKernelBinding>, // sparse — view-only ops + Const + Slot-adopted nodes have no binding
}
```

Engage critically on:

- **`bindings: HashMap` vs `Vec` indexed by topo position.** HashMap is simpler when bindings are sparse (view ops have no kernel); Vec needs a sentinel. Default to HashMap; if profiling shows the realize hot path eats HashMap lookups in 9c, revisit.
- **`device: DeviceLocation` field — is it derivable from `backend`?** Multi-device CUDA has `Cuda { gpu_id }`; the binding needs to remember which. Yes, keep the field.
- **`SmallVec<[DType; 8]>` capacity** — matches `KernelDTypes` from `kernel.rs`. PagedAttn at 7 entries is the worst case; 8 is right.

Commit: `feat(storage): NodeKernelBinding + ExecutionPlan types (Phase 7.6 step 9b)`.

### Step A2: implement `compile_plan(graph, order, target_backend, bindings_table)`

In [`fuel-storage/src/plan.rs`](../../fuel-storage/src/plan.rs):

```rust
pub fn compile_plan(
    graph: &Graph,
    order: &[NodeId],
    bindings_table: &KernelBindingTable,
) -> Result<ExecutionPlan> {
    // For each NodeId in `order`:
    //   - Skip if `op_to_op_kind(&node.op)` is None (view-only ops,
    //     Const, ops the binding table doesn't index).
    //   - Compute `target_backend = graph.target_backend(id)` (already
    //     populated by op-builder methods per Phase 7.5 B3).
    //   - Build `dtypes` via the same shape as
    //     `pipelined::build_lookup_dtypes`.
    //   - Insert NodeKernelBinding { kernel: None, ... } into bindings.
    // Don't call `lookup_alternatives` yet — that's lazy.
    // Verify against the binding table that AT LEAST ONE alternative
    // exists for (op, dtypes, backend); if not, surface
    // Error::NoBackendForOp at compile_plan time (fail-fast at plan
    // time beats failing at first-use time).
}
```

Note the early `assert ≥1 alternative` is **not** lazy — it's the fail-fast guard. Resolution (which specific `KernelRef`) is lazy; existence is checked eagerly.

Commit: `feat(storage): compile_plan + fail-fast missing-binding check (Phase 7.6 step 9b)`.

### Step A3: implement the v1 route picker

In [`fuel-storage/src/plan.rs`](../../fuel-storage/src/plan.rs):

```rust
pub fn resolve_kernel(
    binding: &mut NodeKernelBinding,
    bindings_table: &KernelBindingTable,
    policy: TolerancePolicy,
) -> Result<KernelRef> {
    if let Some(k) = binding.kernel { return Ok(k); }
    let alts = bindings_table.lookup_alternatives(
        binding.op_kind, &binding.dtypes, binding.backend,
    );
    let chosen = match policy {
        TolerancePolicy::BitStableFirst => alts.iter().find(|e| {
            e.precision.bit_stable_on_same_hardware
        }).or_else(|| alts.first()),
        TolerancePolicy::FirstAlternative => alts.first(),
    }.ok_or_else(|| ...NoBackendForOp...)?;
    binding.kernel = Some(chosen.kernel);
    // Update kernel_revision once revision hashing exists; UNTRACKED
    // for now.
    Ok(chosen.kernel)
}

pub enum TolerancePolicy {
    /// Default. Prefer an alternative whose `bit_stable_on_same_hardware`
    /// is true. If none exists, fall back to first-registered.
    BitStableFirst,
    /// First registered alternative wins. Used by tests that
    /// deliberately exercise non-bit-stable kernels.
    FirstAlternative,
}
```

Engage critically on:

- **Should `TolerancePolicy` carry the actual `PrecisionGuarantee` bounds** (`max_ulp_threshold: u32`, `max_relative_threshold: f64`) **instead of a discrete enum?** The architecture's long-term shape is a per-op tolerance budget that allows non-bit-stable kernels if their bounds are still tighter than what the user/test/calibration cares about. A discrete enum is the v1 placeholder; document that the real shape is per-op-budget-driven and lands when the calibration framework does.
- **Is the picker a free function or a method on a `RoutePicker` struct?** Free function is fine for v1; the picker is stateless. A struct becomes valuable when the Judge integration adds memoization / hot-path caching across realize calls.

Commit: `feat(storage): v1 route picker — BitStableFirst tolerance policy (Phase 7.6 step 9b)`.

### Step A4: tests for compile_plan + resolve_kernel

In [`fuel-storage/src/plan.rs`](../../fuel-storage/src/plan.rs) `mod tests`:

1. **`compile_plan_walks_graph_and_skips_view_ops`** — build a small graph with a few primitives + a Reshape (view op); assert `plan.bindings` has the primitives but not the Reshape.
2. **`compile_plan_fails_fast_on_missing_binding`** — register no kernels for `(MatMul, [F32, F32, F32], Cpu)`; build a graph with a MatMul; `compile_plan` returns `Err(NoBackendForOp)` before any node executes.
3. **`resolve_kernel_lazy_caches_first_resolution`** — call `resolve_kernel` twice on the same binding; second call returns the same `KernelRef` without touching the table (assert via a counter wrapped around `lookup_alternatives` if practical — or just assert idempotency).
4. **`resolve_kernel_bitstable_first_picks_bitstable_alternative_when_present`** — register two alternatives for one key (first non-bit-stable, second bit-stable); assert `BitStableFirst` returns the second.
5. **`resolve_kernel_bitstable_first_falls_back_to_first_when_no_bitstable`** — register two alternatives, both non-bit-stable; assert `BitStableFirst` returns the first (registration-order).
6. **`resolve_kernel_first_alternative_policy_returns_first`** — same setup as the bit-stable test; assert `FirstAlternative` returns the first regardless of precision.

Commit: `test(storage): compile_plan + resolve_kernel coverage (Phase 7.6 step 9b)`.

**End of Track A.** If the session stops here, the work is bisectable, tested, and unused. Track B is the natural follow-up.

---

## Track B — extend into 9c (executor migration, ~8 commits)

**Default scope. Track A is the prerequisite (steps A1–A4) — run those first, then continue here.**

### Step B1: thread `ExecutionPlan` through the executor's realize entry points

Touch [`fuel-graph-executor/src/lib.rs`](../../fuel-graph-executor/src/lib.rs):

- `realize_f32` (and `realize_f64`, `realize_bf16`, `realize_f16`, etc. — all the typed entry points) currently does:
  ```rust
  let order = execution_plan(&graph, &effective_roots);
  for id in order { ... eval_node ... }
  ```
- Change to:
  ```rust
  let order = execution_plan(&graph, &effective_roots);
  let mut plan = compile_plan(&graph, &order, self.bindings_table())?;
  for id in order { ... eval_node_with_plan(&mut plan, ...) ... }
  ```
- `bindings_table()` accessor: where does the executor get the table from today? Probably via a Lazy/OnceLock singleton in `fuel-storage::dispatch` — confirm. The executor takes a reference at realize-call time; the table itself is process-wide-immutable post-init.

`eval_node` continues to dispatch via `GraphBackend::matmul()` etc. unchanged in this step. The plan is built; nothing consumes the bindings yet.

Commit: `feat(executor): build ExecutionPlan at realize() entry (Phase 7.6 step 9c.1)`.

### Step B2: migrate `Op::MatMul` arm to use the pre-resolved KernelRef

In `eval_node` at [`fuel-graph-executor/src/lib.rs:1216`](../../fuel-graph-executor/src/lib.rs#L1216), the `Op::MatMul` arm currently calls `self.backend.matmul(...)`. Replace with:

```rust
Op::MatMul => {
    let binding = plan.bindings.get_mut(&node_id).expect("plan");
    let kernel = resolve_kernel(binding, bindings_table, TolerancePolicy::BitStableFirst)?;
    // Wrap (a, b) storages as &[Arc<RwLock<Storage>>] expected by KernelRef.
    // Wrap output as &mut [Arc<RwLock<Storage>>].
    // Call kernel(inputs, outputs, layouts, &OpParams::Matmul { ... })?;
}
```

**Critical hazard 1: Storage shape mismatch.** Today's executor works on `B::Storage` (the backend's owned type — `CudaStorage`, `CpuStorageBytes`, etc.). The byte-storage `KernelRef` signature takes `&[Arc<RwLock<Storage>>]` (the dispatch-erased `BackendStorage` enum from fuel-storage). Bridging these requires wrapping owned storages into the dispatch-erased `Storage` type. Investigate at session start whether the executor's `Storage` type is already convertible or if 9c needs a wrapper conversion at each call site. **This is the single biggest unknown in Track B.** If the conversion is non-trivial (e.g., requires a moved `Arc` that breaks borrow semantics), pause Track B and discuss before committing.

**Critical hazard 2: layout extraction.** The current arm extracts `a.layout()` + `b.layout()` from the cache. The `KernelRef` signature takes a `&[Layout]` slice — order matches the dtype list. Pass `&[a_layout, b_layout, output_layout]` (matching `OpKind::MatMul`'s 3-operand shape).

**Critical hazard 3: fallback semantics.** The pre-migration arm panics with `.expect("MatMul")` on error. The post-migration arm returns `Err`. The existing `Op::Fused(QMATMUL, _)` arm at [`fuel-graph-executor/src/lib.rs:1278`](../../fuel-graph-executor/src/lib.rs#L1278) handles `Err` by falling back to CPU. Decide: does `Op::MatMul` need the same fallback? If yes, the route picker should expose a "next alternative" method (`resolve_next_kernel(binding, ...) -> Result<KernelRef>`) for the executor to call after `Err`. If no, the executor surfaces `Err` directly and the user sees a typed error.

Live-CUDA test after this commit:
```bash
cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored --nocapture
```

Commits:
- `feat(executor): Op::MatMul via pre-resolved KernelRef (Phase 7.6 step 9c.2)`
- `test(executor): live-CUDA MatMul parity — pre vs post 9c migration` (if a parity test makes sense; otherwise smoke-test the existing live-CUDA suite stays green).

### Step B3: migrate the rest of the primitive `Op` arms

Mechanical extension of B2 to every `eval_node` arm. Each commit covers one op family:

- B3.1: elementwise unary (`Op::Relu`, `Op::Neg`, `Op::Sqr`, ..., ~15 arms — one commit)
- B3.2: elementwise binary (`Op::Add`, `Op::Sub`, `Op::Mul`, `Op::Div`, `Op::Pow`, ~10 arms)
- B3.3: reductions (`Op::SumAll`, `Op::MaxAll`, ..., `Op::ReduceSumTo`, `Op::ReduceMaxTo`)
- B3.4: compare family (`Op::Equal`, `Op::Lt`, `Op::Where`, ...)
- B3.5: indexing + scatter (`Op::IndexSelect`, `Op::Gather`, `Op::IndexAdd`, `Op::ScatterAdd`, `Op::MaskedFill`)
- B3.6: shape + misc (`Op::Concat`, `Op::Cast`, `Op::Flip`, `Op::Roll`, `Op::Pad`, `Op::Triu`, `Op::Tril`, `Op::CumSum`)
- B3.7: argindex (`Op::ArgMaxDim`, `Op::ArgMinDim`)

Each commit: identical migration shape to B2. Live-CUDA test after each commit. The total mechanical surface is ~70 arms across all op families.

### Step B4: migrate `Op::Fused` arms

Three arms today: QMATMUL ([fuel-graph-executor/src/lib.rs:1257](../../fuel-graph-executor/src/lib.rs#L1257)), CONV2D ([:1313](../../fuel-graph-executor/src/lib.rs#L1313)), CONV_TRANSPOSE2D ([~:1365](../../fuel-graph-executor/src/lib.rs#L1365)).

Each routes today through `FusedKernelRegistry` (per fuel-storage/src/fused.rs:255) rather than `KernelBindingTable`. **Decision required**: do `Op::Fused` arms use a parallel `compile_plan_fused` + `resolve_kernel_fused` path against `FusedKernelRegistry`, or does the route picker accept both registries as inputs?

Recommendation: **two parallel functions in v1.** `compile_plan` builds NodeKernelBindings for primitives via `KernelBindingTable`; a sibling `compile_plan_fused_node` (called from inside `compile_plan` when `node.op` is `Op::Fused`) builds NodeKernelBindings for fused ops via `FusedKernelRegistry`. The `BindingEntry` shape is the same on both sides post-step-6; the storage is different. Don't unify the registries — they have different keying schemes (primitive: `(OpKind, dtypes, BackendId)`; fused: `(FusedOpId, BackendId, dtypes)`).

Commits:
- `feat(executor): Op::Fused(QMATMUL) via pre-resolved KernelRef + fused-registry route picker`
- `feat(executor): Op::Fused(CONV2D) via pre-resolved KernelRef`
- `feat(executor): Op::Fused(CONV_TRANSPOSE2D) via pre-resolved KernelRef`

### Step B5: drop dead `GraphBackend` trait methods

After every `eval_node` arm has migrated, the `GraphBackend::matmul()` / `.conv2d()` / `.unary()` / `.binary()` / etc. methods are unused **from the executor**. They're still used by `fuel-graph-router::Router::matmul` (placement decision — picks a backend, then calls the trait method). Decision:

- **Keep `GraphBackend::matmul` etc.** as a parallel surface used only by Phase 6b's empirical Judge (per `project_phase6b_probe_judge_dispatch.md`). The Judge profiles via the trait method; the executor dispatches via the KernelRef. Two surfaces, one purpose each.
- **Or delete `GraphBackend::matmul` etc.** and re-point the Judge to profile via the binding table directly. This is cleaner long-term but expands Track B's surface significantly (Judge integration is its own complexity).

Recommendation: **keep both surfaces.** Document the split in a memory entry. Deleting the trait surface is its own session.

Commit: `docs(architecture): GraphBackend trait surface stays as Judge-profiling path post-9c`.

### Step B6: live-CUDA + cross-backend full sweep

After all primitive + fused arms migrated:

```bash
cargo test --workspace --lib --features cuda
cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored --nocapture
cargo test --workspace --lib --features "cuda cudnn"
cargo test --workspace --lib --features "cuda cudnn nccl"
```

Watch for: any test that bypassed the executor (e.g., direct `backend.matmul()` calls in test helpers) — those keep working because the trait surface is preserved. The migration only changes the *executor's* path, not the trait's call sites elsewhere.

Commit: `test(executor): full-workspace sweep post-step-9c migration (no failures)`.

---

## Test commands

After each Track A step:
```bash
cargo check -p fuel-storage
cargo test -p fuel-storage --lib
```

After each Track B step:
```bash
cargo check --workspace --features cuda
cargo test -p fuel-storage --lib
cargo test -p fuel-graph-executor --lib
cargo test -p fuel-storage --features cuda --test cuda_dispatch_live -- --ignored --nocapture
```

End-of-session sweep:
```bash
cargo check --workspace --features "cuda cudnn nccl"
cargo test --workspace --lib
```

## Verification (per architecture v1.0 §04 commitments)

1. **Lazy-resolution caching** — call `resolve_kernel` twice on one binding; second call returns same KernelRef without touching table. (Step A3 + a parity test.)
2. **Fail-fast on missing binding** — `compile_plan` surfaces `NoBackendForOp` at plan time, not at first-use time. (Step A2 + the matching test.)
3. **Multi-impl precedence: bit-stable wins** — register two alternatives, only second is bit-stable; route picker returns the second. (Step A4.)
4. **Multi-impl precedence: first-when-no-bit-stable** — register two non-bit-stable alternatives; picker returns first (registration-order). (Step A4.)
5. **Executor parity post-migration** — every existing test (CPU lib + CUDA live) keeps passing after Track B. The migration changes the dispatch path; result correctness is unchanged. (Step B6.)
6. **CUTLASS alternative selection** (if the cutlass session has shipped B3 by then) — at `(MatMul, [BF16, BF16, BF16], Cuda)`, both cuBLAS and CUTLASS alternatives are registered. With `BitStableFirst`, cuBLAS wins (it's `bit_stable_on_same_hardware: true`); CUTLASS gets picked only when a future tolerance policy explicitly allows non-bit-stable. **This is the architectural payoff** — the cutlass session can register CUTLASS without changing user behavior, and a later session enables it via a tolerance-policy switch.

## Operating principles

- **Bit-stable CPU + reference remain the correctness anchor.** Architecture v1.0 §05's commitment doesn't change; 9b/9c just move *where* the executor consults it (from runtime trait dispatch to planning-time route resolution).
- **No production panics.** `compile_plan` and `resolve_kernel` return `Result`. The only acceptable panic is at the `bindings.get_mut(&node_id).expect("plan")` site in B2+ — and that's debatable; an `unreachable!()` reads as "internal invariant" but a typed error reads as "production-correct." Engage critically per-call-site.
- **Engage critically.** Specifically on: (a) Storage shape bridging in B2 (the single biggest unknown), (b) tolerance-policy representation in A3 (discrete enum vs per-op-budget), (c) parallel vs unified registry path for `Op::Fused` arms in B4, (d) `GraphBackend` trait deletion vs retention in B5.
- **One commit per logical step.** Track A is 4 commits; Track B is ~8–10 commits. Each bisectable.
- **Live-test on this host after every kernel-touching commit.** RTX 4070 supports sm_86; per `project_dev_environment.md`. Don't defer.
- **Update memory after each track lands.** Track A: short note that 9b types + route picker shipped. Track B: short note that 9c executor migration shipped; document the dual-surface decision (Judge profiles via trait, executor via KernelRef).
- **Don't push to remote unless asked.**

## End-of-session deliverable

If only Track A lands: `ExecutionPlan` + `compile_plan` + `resolve_kernel` + 6 unit tests, ~4 commits, ~600 LOC. Unused by the executor — a deliberate dead-code seam awaiting Track B. Document the gap explicitly in the memory entry so the next session knows the seam is half-shipped.

If through Track B step B2: MatMul migrated; rest of the executor still on trait dispatch. ~6 commits. Tests should still pass (the executor calls KernelRef for MatMul, trait methods for everything else).

If through Track B step B6: full executor migration. ~12–14 commits across both tracks. The CUDA matmul + every primitive + every fused-op decision point now has its kernel resolved at plan time. The CUTLASS alternatives (if registered) participate in the picker.

## Coordination notes

- **`fuel-storage::compiled::compile_node`** — a parallel "compile" function that exists today but isn't called by the executor (only by tests). Out of scope; do not retrofit. The plan-based compilation 9b ships is the architecture-target path forward.
- **`fuel-graph-router::Router`** — currently does Phase 6b empirical placement (picks a backend per op). With 9c's `BitStableFirst` route picker as the default, the Router's placement output flows into `compile_plan`'s `target_backend` resolution; the route picker then picks among *that backend's* alternatives. Router stays in place; this is the two-level "backend then kernel" decomposition the architecture commits to.
- **Phase 6b Judge integration** — 9b's v1 picker is a stand-in. A later session replaces `BitStableFirst` with a Judge-driven policy that consults runtime telemetry. The `TolerancePolicy` enum's `Manual` variant (or similar) becomes the seam for that integration. Don't pre-build the Judge plumbing in 9b.
- **`fuel-storage/src/dispatch.rs::register_cpu_kernels`** — touches 335 register sites; the 9a fill-pass pattern means none need editing for 9b. The cost lint + precision lint stay green.
- **Memory entry to write at the end**: `project_phase_7_6_step_9b_shipped.md` (Track A) and/or `project_phase_7_6_step_9c_shipped.md` (Track B). Update `MEMORY.md` with one-line index entries. Update `project_phase_7_6_step_4_in_progress.md`'s "steps remaining" list to reflect what landed.
