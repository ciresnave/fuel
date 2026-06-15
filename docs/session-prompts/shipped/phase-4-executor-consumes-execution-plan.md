# Session prompt — Phase 4: PipelinedExecutor consumes ExecutionPlan

> **Archived 2026-06-15 — shipped** (commits `6ac8e065` / `a033f1dd` / `8d849fa2`, 2026-06-07; the as-built `ExecutionPlan` / `compile_plan` / `resolve_compiled` path). Accurate as a record. Its per-node `ExecutionPlan` model is the **staging post superseded by the 2026-06-14 "plan is the graph" redirection** ([`../../architecture/10-decisions-log.md`](../../architecture/10-decisions-log.md), [`../../architecture/14-lifecycle.md`](../../architecture/14-lifecycle.md)). The staged migration will rewrite this — useful here as provenance, not as a live queue item.

## What this session is for

Migrate `fuel_dispatch::pipelined::PipelinedExecutor` from the
legacy first-registered binding-table dispatch to consuming
`fuel_dispatch::plan::ExecutionPlan` produced by the optimizer
ranker (Phases 1.1–1.5 + 3). This is the **load-bearing executor
gate** named in the picker-work audit; until it lands, the
picker substrate ships infrastructure-only.

The session has three sub-phases. They form one architectural
arc but can be broken into separate commits.

## Sub-phases at a glance

| Sub-phase | Scope                                                                       | Risk      |
|-----------|-----------------------------------------------------------------------------|-----------|
| **4.1**   | `compile_one` consults `ExecutionPlan` for kernel resolution; new `realize_with_plan` / `realize_many_with_plan` public APIs | Low — additive |
| **4.2**   | Loosen `pipelined_bridge::prepare()`'s monolithic `target_backend` pinning   | Medium — touches cross-crate API |
| **4.3**   | Dispatch chunk boundaries + per-chunk SystemTopology generation check       | Medium — new executor concept |

**Recommended order: 4.1 first, alone, ship + verify. Then 4.2, then
4.3.** Each ships as its own commit. Don't bundle.

## Background — what's in place

The picker arc shipped this substrate (all committed, all live):

- **`fuel_dispatch::ranker`** (Phases 1.1–1.4 + 3) — `Candidate`,
  `AlternativeSet`, `apply_filter_chain`, `enumerate_candidates`,
  `default_chain`, `compute_static_costs` with optional `JudgeOracle`
  refinement, `rank_by_composite_cost`.
- **`fuel_dispatch::plan`** (Phase 1.5) — `ExecutionPlan` carries
  `alternatives: HashMap<NodeId, AlternativeSet>`; `compile_plan`
  orchestrates enumeration → filter → cost rank → truncate;
  `PlanOptions` with `placements_for_device`, `capabilities_for`,
  `judge`, `precision_requirement`, `max_alternatives_per_node`.
- **`fuel_graph::opt::insert_cross_device_copies`** (Phase 2.1) —
  graph-rewrite pass that inserts `Op::Copy` on cross-substrate
  edges; CSE-deduplicated, idempotent, callback-driven.
- **`fuel_core::topology::SystemTopology`** (Phase 0.1) — single
  source of truth for backend/device/substrate/transfer-path
  knowledge; lookups via `backends_for(dev)`, `shares_storage((b1,d1),
  (b2,d2))`, `transfer_path(src,dst)`, `capabilities(b)`.

**What's missing**: the wire from `compile_plan`'s output to the
executor's dispatch. Today `PipelinedExecutor` runs `compile_node`
per node → first-registered binding-table lookup, ignoring whatever
the picker said. Phase 4 closes that loop.

## Phase 4.1 — additive plan-aware compile_one

### Goal

`PipelinedExecutor` gains two new public methods:

```rust
impl PipelinedExecutor {
    pub fn realize_with_plan(
        graph: Arc<RwLock<Graph>>,
        target: NodeId,
        inputs: StorageCache,
        plan: Arc<ExecutionPlan>,
    ) -> Result<(Arc<RwLock<Storage>>, Layout)>;

    pub fn realize_many_with_plan(
        graph: Arc<RwLock<Graph>>,
        targets: &[NodeId],
        inputs: StorageCache,
        plan: Arc<ExecutionPlan>,
    ) -> Result<Vec<(Arc<RwLock<Storage>>, Layout)>>;
}
```

Both delegate to `realize_inner` / `realize_many_inner` that take
`Option<Arc<ExecutionPlan>>`. The existing `realize` and `realize_many`
become thin wrappers passing `None` — zero behavior change for the
existing call surface.

### Threading the plan

1. `compiler_thread_body` gains a `plan: Option<Arc<ExecutionPlan>>`
   parameter and passes `plan.as_deref()` to `compile_one`.
2. `compile_one` gains `plan: Option<&ExecutionPlan>`. Every
   `compile_node(op_kind, &dtypes, target_backend, op_params, bindings)?`
   call site (~4-5 in `compile_one`) is replaced with a new helper:

```rust
fn resolve_compiled(
    id: NodeId,
    op_kind: OpKind,
    dtypes: &[DType],
    target_backend: BackendId,
    op_params: OpParams,
    bindings: &KernelBindingTable,
    plan: Option<&ExecutionPlan>,
) -> Result<CompiledNode> {
    if let Some(p) = plan {
        if let Some(set) = p.alternatives(id) {
            if let Some(winner) = set.winner() {
                return Ok(CompiledNode {
                    op: op_kind,
                    dtypes: KernelDTypes::from_slice(dtypes),
                    backend: winner.backend,
                    kernel: winner.kernel,
                    caps: winner.caps,
                    op_params,
                });
            }
        }
    }
    compile_node(op_kind, dtypes, target_backend, op_params, bindings)
}
```

Plan absent or no entry for the node → fall through to legacy. This
is the load-bearing additive contract.

### Why `op_params` always comes from the executor

The plan-time `Candidate::op_params` is a placeholder
(`OpParams::None`; see `candidate_default_op_params` in
`fuel-dispatch/src/plan.rs`). The live OpParams shape — reduce dims,
conv geometry, etc. — comes from `op_to_op_params(graph, node,
layout_cache)` which the executor already computes. The plan's job
is `(kernel, caps, backend)`; the executor owns op-params derivation.

### Test plan (4.1)

In `fuel-dispatch/src/pipelined.rs::tests`:

1. **Plan-resolved kernel wins**: build a graph with one Add node;
   register two CPU kernels at `(AddElementwise, [F32×3], Cpu)` —
   `kernel_a` (first-registered) and `kernel_b`. Construct an
   `ExecutionPlan` whose `AlternativeSet` for the Add node has
   `kernel_b` as winner. `realize_with_plan` should dispatch
   `kernel_b`.
2. **Fallback when plan has no entry**: same graph, but the
   `ExecutionPlan::alternatives` map is empty. Execution falls
   through to legacy `compile_node` → `kernel_a` wins.
3. **Existing `realize` path unchanged**: identical workload through
   the no-plan `realize` API still picks `kernel_a` (no behavior
   drift).
4. **Plan with cross-backend winner**: enumerate Cpu + Aocl
   alternatives, plan picks Aocl. `realize_with_plan` dispatches the
   Aocl kernel even though the graph node's `target_backend` is Cpu.

### Files touched (4.1)

- `fuel-dispatch/src/pipelined.rs` — add `resolve_compiled` helper,
  thread `plan: Option<&ExecutionPlan>` through `compile_one`, add
  `realize_with_plan` + `realize_many_with_plan` + `_inner` siblings,
  modify `compiler_thread_body` signature.
- `fuel-dispatch/src/lib.rs` — no changes (the new APIs go on the
  existing `PipelinedExecutor` struct).

### Risks (4.1)

- **`compile_one` is ~500 LOC** with multiple `compile_node` arms
  (in-place, WriteSlice, Copy, others). Easy to miss one.
- **Parallel session activity on `compile_one`** — the
  SelectiveScan/SsdChunkScan `return_state` plumbing recently
  touched this file. Coordinate via `git stash` if uncommitted
  parallel work conflicts.

### Discipline (4.1)

After EACH edit to `pipelined.rs`, immediately run `git diff
fuel-dispatch/src/pipelined.rs | head` to verify the edit
persisted. An earlier 2026-06-01 attempt at 4.1 saw edits silently
fail to persist — don't trust the Edit-tool's success response
alone; verify with `git diff`.

## Phase 4.2 — loosen target_backend pinning in pipelined_bridge

### Goal

`fuel-core::pipelined_bridge::prepare()` currently does:

```rust
for &id in &order {
    let node = g.node(id);
    if matches!(node.op, Op::Const | Op::Release) || node.op.is_view_op() {
        continue;
    }
    g.set_target_backend(id, backend_id);  // MONOLITHIC — every node gets the same backend
}
```

This pins every reachable kernel-bearing node to the device the user
realized on. Phase 4.2 makes this conditional: only set
`target_backend` on nodes the user explicitly pinned; let the picker
decide for the rest.

### New shape

```rust
for &id in &order {
    let node = g.node(id);
    if matches!(node.op, Op::Const | Op::Release) || node.op.is_view_op() {
        continue;
    }
    if g.target_backend(id).is_none() {
        // User didn't pin; leave for the picker.
        continue;
    }
    // User explicitly pinned this node (rare today) — honor it.
}
```

`pipelined_bridge::realize_one_as` then builds the plan via
`compile_plan` with `PlanOptions::placements_for_device` wired to
`SystemTopology::backends_for(...)`, and dispatches via
`PipelinedExecutor::realize_with_plan`.

### Files touched (4.2)

- `fuel-core/src/pipelined_bridge.rs` — modify `prepare()` and the
  `realize_*` family to build + use plans.
- `fuel-core/src/judge` (or a new adapter module) — implement
  `JudgeOracle for ProfileReport` so the cached profile data flows
  into the plan's cost composer.

### Risks (4.2)

- **fuel-core has active parallel work** (Phase A.8 lazy migration,
  inplace ops, FlashAttn). Coordinate via stash.
- **Behavioral change is real**: today's "realize on CUDA" pins
  every node to CUDA. After 4.2, the picker may schedule some nodes
  on CPU when SystemTopology allows it and the cost composer
  prefers it. Tests need to verify the user's `realize_f32_cuda(&dev)`
  intent is still honored where it matters (cross-device fallback
  isn't silent + unexpected).

### Test plan (4.2)

- **Existing tests still pass**: every `realize_f32_cuda` /
  `realize_many_as` test that worked before still works.
- **Cross-co-located backends compete**: build a `Router::add_cpu()
  .add_aocl()` topology; realize a matmul; verify the picker picks
  Aocl over Cpu when the cost composer says so.
- **User pin honored**: explicitly setting `target_backend(node) =
  Cpu` on a node ensures it runs on CPU even when Aocl is cheaper.

## Phase 4.3 — dispatch chunks + per-chunk generation check

### Goal

Per the SystemTopology design (Phase 0.1's TDP-5 lifecycle):

> The runtime selector (Picker 2) checks SystemTopology generation
> immediately before submitting a chunk; if the backend invalidates
> after submission but before the kernel runs, that's the backend's
> failure-handling problem, not the planner's.

Phase 4.3 introduces "dispatch chunks" — groups of consecutive ops
going to the same `(backend, device)` — and a generation check
between chunks. If the topology generation has advanced since the
plan was built, the executor returns `Error::TopologyChanged` and
the realize layer re-builds the plan + retries.

### Sketch

```rust
fn execute_work_items_chunked(
    rx: Receiver<Result<WorkItem>>,
    cache: &mut StorageCache,
    layout_cache: &mut HashMap<NodeId, Layout>,
    plan_generation: u64,                  // captured at compile_plan time
) -> Result<()> {
    let mut current_chunk_backend: Option<BackendId> = None;
    for item in rx {
        let item = item?;
        // Chunk boundary: backend changed → check generation.
        if Some(item.target_backend) != current_chunk_backend {
            if fuel_dispatch::dispatch::topology_generation() != plan_generation {
                return Err(Error::TopologyChanged { ... });
            }
            current_chunk_backend = Some(item.target_backend);
        }
        execute_work_item(&item, cache, layout_cache)?;
        // ... destructive cleanup ...
    }
    Ok(())
}
```

### Error type

```rust
// In fuel-core-types::Error
TopologyChanged {
    plan_generation: u64,
    current_generation: u64,
}
```

### Test plan (4.3)

- **Generation stable → no retry**: realize a small graph; no
  topology change between plan and execute; succeeds first try.
- **Generation bumps mid-realize → plan rebuild + retry**: realize
  a graph; between `compile_plan` and the first chunk submission,
  call `bump_topology_generation()`; verify the realize layer
  catches `TopologyChanged`, rebuilds the plan against the new
  topology, retries, succeeds.
- **Multiple bumps don't infinite-loop**: a guard limits retries
  to N (default 3); after that, surface the error.

### Files touched (4.3)

- `fuel-core-types/src/error.rs` — add `Error::TopologyChanged`.
- `fuel-dispatch/src/pipelined.rs` — chunk boundary detection +
  generation check in the executor loop.
- `fuel-core/src/pipelined_bridge.rs` — wrap the
  `realize_*_with_plan` call in a retry loop that catches
  `TopologyChanged`.

## What's NOT in scope (any sub-phase)

- **Op::Copy / Op::Move materialization from picker decisions** —
  Phase 2.1's `insert_cross_device_copies` is the mechanism; the
  caller wires placements + topology. Already shipped; Phase 4.2
  invokes it from `pipelined_bridge` if the picker's committed
  alternatives cross substrates.
- **Runtime selector / Picker 2** — Phase 5. Layer-3 telemetry pick
  among top-N. Phase 4 ships winner-only dispatch.
- **fuel-cublaslt / tensor-tools / fuel-lazy-examples binaries
  pre-existing breakage** — unrelated, documented in
  [[project-dispatch-crate-extracted]]. Not part of Phase 4.
- **Op::Contiguize layout-fixup insertion** — Phase 2.2, gated on
  `Op::Contiguize` graph IR design. Not Phase 4.

## Coordination with parallel sessions

The fuel-dispatch + fuel-core area has been hot with parallel work
through May 2026 (in-place ops, FlashAttn backward, SelectiveScan
return_state, multi-output infra, lazy-tensor migration A.8.x). The
"churn should have slowed substantially" prerequisite for Phase 4
means:

- `git log --since="1 day" fuel-dispatch/` is light or empty
- `fuel-dispatch/src/pipelined.rs` and `fuel-core/src/pipelined_bridge.rs`
  have no uncommitted modifications in another session's workspace
- The `multi-output-nodes-option-c.md` infra has either landed (Op::View
  / Op::ViewOwned variants exist in fuel-graph::Op) or been reverted
  (so fuel-graph builds clean)

If those conditions aren't met, defer Phase 4 to a quieter day. The
substrate is in place; it's safe to wait.

## Discipline notes (apply to every sub-phase)

1. **Verify every Edit persisted.** After each `Edit` call, run
   `git diff <file>` immediately. The 2026-06-01 first attempt at
   4.1 saw edits silently roll back; verify don't trust.
2. **Commit one sub-phase at a time.** No bundling. 4.1 ships → green
   → 4.2 → green → 4.3.
3. **Use `git add <explicit paths>` only.** Never `git add .` or
   `-A`. The parallel-session work pattern of accidentally absorbing
   other people's uncommitted churn into your commit (see
   `d4a9efad` — Phase 2.1 absorbed 188 lines of multi-output infra
   work) is a real failure mode.
4. **Stash before building if other sessions are uncommitted.**
   `git stash push -m "phase4-isolation" -- <foreign paths>` keeps
   their work intact while you verify yours compiles.
5. **Pre-existing failures stay pre-existing.** Don't try to fix
   `precision_guarantee_lint_bit_stable_cpu_coverage` (FlashAttn
   backward FusedOpIds), fuel-cublaslt (Result destructuring),
   tensor-tools (Device::Cpu), lazy-examples binaries
   (generate_streaming). Document them in the commit message,
   move on.

## Deliverables (full Phase 4)

1. **3 commits** on `feature/storage-unification`:
   - `feat(executor): Phase 4.1 — plan-aware compile_one`
   - `feat(executor): Phase 4.2 — loosen target_backend pinning`
   - `feat(executor): Phase 4.3 — dispatch chunks + generation check`
2. **Memory entries** per sub-phase + index entry in MEMORY.md.
3. **Test sweep** clean across feature flags:
   - `cargo test -p fuel-dispatch --lib`
   - `cargo test -p fuel-core --lib`
   - `cargo test -p fuel-dispatch --lib --features cuda`
   - `cargo test -p fuel-dispatch --lib --features vulkan`
   - `cargo test -p fuel-dispatch --lib --features cuda,vulkan`
4. **Live-GPU sanity** (RTX 4070): re-run the audit harness to
   confirm coverage map unchanged; spot-check 2-3 live tests pass.

## Scope estimate

- Phase 4.1: ~2 hours focused work (5 callsite flips, 2 new public
  APIs, 4 new tests). 1 commit.
- Phase 4.2: ~3 hours (cross-crate change, careful test sweep).
  1 commit.
- Phase 4.3: ~3 hours (new executor concept + retry loop).
  1 commit.
- **Total: 1 session if everything goes smoothly; 2 sessions if
  any sub-phase surfaces an unexpected refactor.**

## After Phase 4 — what's next

- **Phase 5** — runtime selector (Picker 2) for layer-3 telemetry.
  Becomes meaningful once the executor is reading
  `AlternativeSet`s; until 4 lands, Picker 2 has nothing to select
  among.
- **Phase 2.2 / 2.3** — Op::Contiguize insertion + coupled-cost
  adjustments. Independent of Phase 4 but architecturally cleaner
  to do AFTER the executor migration proves out the basic shape.
- **Retire `compile_node`** — once Phase 4 fully migrates,
  `compile_node`'s callers are all in tests + `resolve_compiled`'s
  fallback path. Either retire the fallback (require every realize
  to supply a plan) or keep it as a defensive escape hatch.

## Pointers

- Picker-work arc: [[project-judge-alternatives-audit]]
- Substrate: [[project-phase-1-1-ranker-substrate-shipped]],
  [[project-phase-1-picker-substrate-complete]],
  [[project-phase-3-judge-integration-shipped]],
  [[project-phase-2-1-op-copy-insertion-shipped]]
- Prerequisite: [[project-system-topology-shipped]]
- Architecture: `docs/architecture/04-optimization.md`,
  `docs/architecture/06-runtime.md`
- Files this session will touch:
  - `fuel-dispatch/src/pipelined.rs` (~5 callsite flips,
    2 new public APIs, ~30 new test LOC)
  - `fuel-core/src/pipelined_bridge.rs` (~30 LOC change in
    `prepare`, new plan-building in `realize_*` family)
  - `fuel-core/src/judge/mod.rs` (new `JudgeOracle` adapter impl)
  - `fuel-core-types/src/error.rs` (new `TopologyChanged` variant)
- Reference for prior parallel-conflict handling:
  [[project-phase-1-picker-substrate-complete]]'s "Discipline notes"
  section describes the stash-commit-explicit-paths pattern used
  throughout Phases 1.2–1.5.
