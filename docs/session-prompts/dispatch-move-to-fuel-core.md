# Session prompt — P0.2: Move dispatch + executor from fuel-storage to new `fuel-dispatch` crate

## What this session is for

Move the **dispatch infrastructure** (`KernelBindingTable`,
registration wrappers, cost functions, fused-kernel registry, plan +
picker types) AND the **`PipelinedExecutor`** from `fuel-storage` to a
new dedicated crate, `fuel-dispatch`. **Purely structural refactor —
no semantic changes.** Every test that passes today must pass after.
Every public symbol consumers import from `fuel_storage::*` either
moves to a new path under `fuel-dispatch` or stays accessible via a
deprecated re-export through P0.2c.

This is **session P0.2** of the picker-work phasing established
2026-05-30 in [`judge-alternatives-picking-audit-results.md`](
./judge-alternatives-picking-audit-results.md). The original
"P0.2a then P0.2b" split has been collapsed into a single session
— see [Why a new crate, not fuel-core](#why-a-new-crate-not-fuel-core)
below for the dep-cycle reasoning that forced the collapse.

Subsequent sessions:

- **P0.2c** — Move `BackendStorage` enum to `fuel-core-types`; retire
  `fuel-storage` entirely.
- **P1.x** — Optimizer ranker (the actual picker work).

P0.2 is the unblock. Phase 1's optimizer ranker will touch dispatch
code extensively; doing the move *first* avoids re-doing every diff
across the boundary later.

## Why this session exists

Today `fuel-storage` is three concerns mashed together:

1. **Storage substrate** — `BackendStorage` enum, the discriminated
   union of per-backend storage variants. Stays for P0.2; moves to
   `fuel-core-types` in P0.2c.
2. **Dispatch infrastructure** — `KernelBindingTable`, registration
   wrappers per backend, `CompiledNode`, `compile_node`, `compile_plan`,
   `resolve_kernel`, `TolerancePolicy`, `FusedKernelRegistry`, cost
   functions. **Moves to `fuel-dispatch` in P0.2.**
3. **Executor** — `PipelinedExecutor`. **Moves to `fuel-dispatch` in
   P0.2** (forced by the dep-cycle constraint; see below).

The crate name `fuel-storage` describes only the first concern. The
dispatch infrastructure and executor logically belong in their own
home — they're the "given a graph, run it on registered backends"
layer.

## Why a new crate, not fuel-core

A natural-seeming option would be to move dispatch + executor into
`fuel-core`. That's the wrong call for two reasons:

**Reason 1: dep cycle.** If dispatch moves to `fuel-core` but the
executor stays in `fuel-storage`, then `fuel-storage::pipelined`
needs to import `fuel_core::*`. But `fuel-core` already depends on
`fuel-storage` (for `BackendStorage` enum). That's a cycle. The only
way out is to move the executor along — which means the executor is
in `fuel-core` too.

That gets the executor working, but raises Reason 2.

**Reason 2: `fuel-core` is already a catch-all.** Quick `ls
fuel-core/src/`:

- Tensor surface (tensor.rs, lazy.rs, variable.rs)
- Backend bridges (cpu_backend/, cuda_backend/, vulkan_backend/,
  metal_backend/)
- Lazy *model implementations* (lazy_bert.rs, lazy_yolov8.rs,
  lazy_qwen2_moe.rs, lazy_sd_unet.rs — these are full architectures!)
- Training (train.rs, backprop.rs, sampling.rs)
- Inference orchestration (inference_context.rs, kv_cache.rs,
  generate paths)
- Serialization (safetensors.rs, pickle.rs, npy.rs, quantized/)
- Probe / topology / Judge
- Conv ops, sort, scalar, indexer, transfer_cost…

The "core" name should mean "everything every Fuel user needs to
start a graph." It currently means "everything that doesn't yet
have a dedicated crate." Adding dispatch + executor would make
the problem worse.

**The right shape: a dedicated `fuel-dispatch` crate.** The dep
direction stays clean:

```text
fuel-core-types        base types (Storage, DType, OpKind, BackendId, DeviceLocation, ...)
   ↑
fuel-storage           BackendStorage enum (retired in P0.2c)
   ↑
fuel-graph             Op, NodeId, graph IR
   ↑
backend crates         typed kernels per backend; depend on fuel-core-types
   ↑
fuel-dispatch (NEW)    binding table + wrappers + executor + planner
   ↑
fuel-core              tensor surface + bridges + Judge + topology + … (leaner over time)
```

The cycle that motivated the original "two sub-sessions" split
disappears because `PipelinedExecutor` comes along with dispatch
— there's no leftover consumer of moved symbols inside
`fuel-storage`. Post-move `fuel-storage` contains only the
`BackendStorage` enum + variant types, with no upward imports.

The broader `fuel-core` decomposition (extract Judge, models,
serialization, training to their own crates) is a real architectural
concern but a **separate body of work, not in scope for this
session.**

## Background — what's in fuel-storage today

**Files that move to `fuel-dispatch`:**

| File                                    | Lines | What it contains                                |
|-----------------------------------------|-------|-------------------------------------------------|
| `fuel-storage/src/kernel.rs`            | ~1100 | binding-table types + register/lookup methods   |
| `fuel-storage/src/dispatch.rs`          | ~6100 | CPU wrappers + `global_bindings()` + lints      |
| `fuel-storage/src/baracuda_dispatch.rs` | ?     | baracuda CUDA registration wrappers (cuda-gated)|
| `fuel-storage/src/vulkan_dispatch.rs`   | ?     | Vulkan registration wrappers (vulkan-gated)     |
| `fuel-storage/src/compiled.rs`          | ~180  | `CompiledNode`, `compile_node`, exec helper     |
| `fuel-storage/src/cost.rs`              | ~660  | `default_cost_for_op_kind` + family cost fns    |
| `fuel-storage/src/fused.rs`             | ?     | fused registry + revision hash + precision      |
| `fuel-storage/src/plan.rs`              | ~880  | ExecutionPlan + compile_plan + resolve_kernel   |
| `fuel-storage/src/cast_fusion.rs`       | ?     | cast-fusion rule                                |
| `fuel-storage/src/pipelined.rs`         | ~7700 | `PipelinedExecutor` + work-item dispatch        |

Detail per file:

- `kernel.rs` — `KernelBindingTable`, `BindingEntry`, `KernelRef`,
  `KernelCaps`, `OpParams`, `CostFn`, register/lookup methods.
- `dispatch.rs` — `register_cpu_kernels` (~335 registrations), CPU
  dispatch wrappers, `global_bindings()`, `register_optional_backends`,
  `CapabilityRegistry`, the audit harness test, coverage lints.
- `compiled.rs` — `CompiledNode`, `compile_node`, `execute_compiled`.
- `fused.rs` — `FusedKernelRegistry`, `KernelRevisionHash`,
  `PrecisionGuarantee`, fused-op cost helpers.
- `plan.rs` — `ExecutionPlan`, `NodeKernelBinding`, `compile_plan`,
  `resolve_kernel`, `TolerancePolicy`.
- `pipelined.rs` — `PipelinedExecutor`, `realize`, `realize_many`,
  `compile_one`, `execute_work_item`.

**File that stays in fuel-storage** (moves in P0.2c):

| File                      | Why it stays                            |
|---------------------------|-----------------------------------------|
| `fuel-storage/src/lib.rs` | `BackendStorage` enum — moves in P0.2c  |

**External consumers to update:**

- `fuel-core/src/lib.rs` — re-exports from `fuel-storage::*`
- `fuel-core/src/pipelined_bridge.rs` — uses dispatch + executor APIs
  heavily
- `fuel-core/src/topology.rs` — reads `global_bindings()`,
  `iter_keys()`, `bump_topology_generation`, etc.
- `fuel-core/src/judge.rs` — possible
- `fuel-core/src/inference_context.rs` — uses `realize_many_as_*`
  helpers
- `fuel-core/src/factories.rs` — uses dispatch
- `fuel-graph-router/src/lib.rs` — verify
- Backend crates — should NOT import dispatch directly; verify
- Tests across the workspace
- `fuel-lazy-examples` binaries

After the move, all of these flip from `fuel_storage::*` to
`fuel_dispatch::*` for the moved symbols. `fuel_storage::BackendStorage`
remains the only thing under the old crate path.

## Module structure within `fuel-dispatch`

Recommended layout:

```text
fuel-dispatch/
├── Cargo.toml                          # new crate manifest
└── src/
    ├── lib.rs                          # re-exports + module docs
    ├── binding/                        # KernelBindingTable et al.
    │   ├── mod.rs                      # was kernel.rs's top half
    │   ├── table.rs                    # KernelBindingTable, BindingEntry
    │   ├── kernel_ref.rs               # KernelRef typedef + OpParams + KernelCaps
    │   ├── caps.rs                     # KernelCaps
    │   └── capability.rs               # CapabilityRegistry, global_registry()
    ├── register/                       # backend registration wrappers
    │   ├── mod.rs                      # global_bindings(), register_optional_backends()
    │   ├── cpu.rs                      # was the CPU portion of dispatch.rs
    │   ├── cuda.rs                     # was baracuda_dispatch.rs
    │   └── vulkan.rs                   # was vulkan_dispatch.rs
    ├── compiled.rs                     # CompiledNode, compile_node, execute_compiled
    ├── cost.rs                         # was cost.rs
    ├── fused.rs                        # FusedKernelRegistry, KernelRevisionHash, PrecisionGuarantee
    ├── cast_fusion.rs                  # cast-fusion rule
    ├── plan.rs                         # ExecutionPlan, NodeKernelBinding, compile_plan, resolve_kernel, TolerancePolicy (doomed in Phase 1)
    ├── executor/                       # PipelinedExecutor
    │   ├── mod.rs                      # PipelinedExecutor public surface
    │   ├── compile.rs                  # compile_one + compiler_thread_body
    │   ├── execute.rs                  # execute_work_item + work item kinds
    │   └── helpers.rs                  # build_lookup_dtypes, op_to_op_kind, etc.
    └── tests/                          # integration tests (if any)
```

The session's TDP-B1 below covers refinements; this is the starting
sketch.

## Architectural decisions to surface

### TDP-B1: Module substructure within `fuel-dispatch`

The 7700-line `pipelined.rs` is unwieldy as a single file. Two options:

- **A) Move as-is, defer split.** Keep `executor.rs` as one file
  matching the source; split in a later session.
- **B) Split during the move.** Use the `executor/` submodule structure
  above. Clean break; more cognitive load on this session.

**Recommendation: A.** This session is about moving, not
restructuring. Split `pipelined.rs` in a follow-up if it's painful;
don't bundle.

Same recommendation applies to the 6100-line `dispatch.rs` — move
as one file under `register/cpu.rs` or `register.rs`; split later
if needed.

### TDP-B2: Rename `fuel_core::dispatch`

`fuel_core::dispatch` (235 lines today) hosts the **Judge's
DispatchTable cache** — `cached()`, `populate_dispatch_table`,
`invalidate`. It's not "dispatch" in the sense of "what runs the
graph"; it's the cached output of the Judge.

Name options after `fuel-dispatch` exists:

- **A) Merge into `fuel_core::judge`.** The cache lifecycle is owned
  by the Judge; the merger reads cleanly. `cached()` becomes
  `fuel_core::judge::cached_dispatch_table()` or similar.
- **B) Rename to `fuel_core::dispatch_cache`.** Standalone module,
  describes contents.
- **C) Rename to `fuel_core::judge_cache`.** Slight reframing — the
  cache, owned by Judge.

**Recommendation: A.** The Judge owns this; keeping them together
matches the conceptual hierarchy.

### TDP-B3: `BackendStorage` enum location

The enum stays in `fuel-storage::lib.rs` for P0.2. It moves to
`fuel-core-types` in P0.2c.

Post-P0.2 deps: `fuel-dispatch` depends on `fuel-storage` (for
`BackendStorage`) + backend crates (for variant types).

### TDP-B4: `CapabilityRegistry` placement

`CapabilityRegistry` + `global_registry()` is tightly coupled to the
registration pipeline — `register_backend_capabilities` bumps the
topology generation and registers into the global. Move alongside
the rest of dispatch.

**Decision: moves to `fuel-dispatch::binding::capability`.**

### TDP-B5: Cargo features

Today `fuel-storage` has feature flags `cuda`, `vulkan`, `metal`,
`aocl`, `onemkl`. These gate the backend-specific dispatch wrapper
registrations.

`fuel-dispatch`'s new `Cargo.toml` declares the same features, with
matching feature-gated deps on backend crates. `fuel-core` (and any
other consumer) propagates feature flags through to `fuel-dispatch`:

```toml
[features]
cuda = ["fuel-dispatch/cuda"]
vulkan = ["fuel-dispatch/vulkan"]
# ... etc
```

The session should verify each feature chain — `--features cuda`
builds the CUDA wrappers in their new home; same for Vulkan, etc.

**No new features should be added.** Existing flags transfer
unchanged.

### TDP-B6: Test reorganization

Lots of tests live inline in the moving files. They follow the
files.

The tests that live in `fuel-storage/tests/*.rs`:

- `fuel-storage/tests/baracuda_*.rs` (live-GPU dispatch wrapper tests)
  → move to `fuel-dispatch/tests/`.
- `fuel-storage/tests/vulkan_dispatch_live.rs` → move to
  `fuel-dispatch/tests/`.
- Executor integration tests → move to `fuel-dispatch/tests/`.

**Pragmatic rule:** each test moves with the code it primarily
exercises.

### TDP-B7: Public API surface — re-exports or hard break?

`fuel-storage` currently re-exports a fair bit of dispatch surface
via `pub use kernel::*` etc. External callers import via
`fuel_storage::*`.

After the move, fuel-storage's surface shrinks dramatically (just
`BackendStorage` and friends remain). External callers must update
imports.

Options:

- **A) Hard break — delete re-exports, force callers to update.**
  Surface all consumers explicitly. Larger diff.
- **B) Soft transition — keep `pub use` re-exports in `fuel-storage`
  that point at `fuel_dispatch::*`.** Requires `fuel-storage` to
  depend on `fuel-dispatch` — that's the wrong direction (cycle).

**Recommendation: A.** Hard break. The dep direction must stay
clean; soft re-exports require an upward dep that defeats the
point. The diff is bounded (consumer list above) and grep-able.

### TDP-B8: Workspace `Cargo.toml` updates

The workspace root `Cargo.toml` needs:

- New member: `"fuel-dispatch"`.
- New workspace dep entry: `fuel-dispatch = { path = "fuel-dispatch", version = "X.Y" }`.

Version: align with the workspace version, currently consistent
across all crates per `version.workspace = true`.

### TDP-B9: Phase 1 will need SystemTopology + Judge accessible from `fuel-dispatch`

**Flagged for Phase 1, not this session.** The optimizer ranker
(Phase 1) needs SystemTopology and Judge data. Both currently live
in `fuel-core`. If `fuel-dispatch` consumes them via direct import,
it depends on `fuel-core` — that's the cycle direction again.

Three resolutions when Phase 1 starts:

- Move `SystemTopology` + `Judge` + `ProbeReport` out of `fuel-core`
  too (they're dispatch-adjacent — they describe and measure what
  dispatch consumes).
- Abstract via traits in `fuel-dispatch`; `fuel-core` provides the
  impl.
- Move just the data types (e.g. `DispatchTable`) lower in the
  stack; keep the collector / build logic in `fuel-core`.

The Phase 1 prompt resolves this. **Not P0.2's concern**, but
mention it in the memory entry so future sessions don't get
surprised.

## Scope of work

### Step 1 — create the new crate

```text
fuel-dispatch/
├── Cargo.toml      # new manifest, version.workspace = true, features = {cuda, vulkan, metal, aocl, onemkl}
└── src/
    └── lib.rs      # empty for now
```

Add to workspace `Cargo.toml` `members`. Verify `cargo build -p fuel-dispatch` produces an empty lib.

### Step 2 — `git mv` files in dependency-respecting order

Files with the fewest internal deps move first. Suggested order:

1. `kernel.rs` → `fuel-dispatch/src/binding/{table,kernel_ref,caps,capability}.rs` (split or as-is; TDP-B1)
2. `cost.rs` → `fuel-dispatch/src/cost.rs`
3. `fused.rs` → `fuel-dispatch/src/fused.rs`
4. `compiled.rs` → `fuel-dispatch/src/compiled.rs`
5. `cast_fusion.rs` → `fuel-dispatch/src/cast_fusion.rs`
6. `plan.rs` → `fuel-dispatch/src/plan.rs`
7. `dispatch.rs` → `fuel-dispatch/src/register/{cpu.rs,mod.rs}` (split CPU wrappers vs global_bindings + register_optional_backends)
8. `baracuda_dispatch.rs` → `fuel-dispatch/src/register/cuda.rs`
9. `vulkan_dispatch.rs` → `fuel-dispatch/src/register/vulkan.rs`
10. `pipelined.rs` → `fuel-dispatch/src/executor.rs` (or `executor/` per TDP-B1)

Use `git mv` to preserve history. One commit per file moved (or
grouped tightly), with build-passing checkpoints.

### Step 3 — update imports in moved files

Each file's `use crate::*` paths need updating. Cross-references
(e.g. `register::cpu` references `KernelBindingTable` from
`binding::table`) become explicit paths within `fuel-dispatch`.

Pay attention to feature gates: `#[cfg(feature = "cuda")]` blocks
need to follow the wrappers.

### Step 4 — handle `fuel_core::dispatch` rename

Per TDP-B2: merge into `fuel_core::judge`. The 235-line
`fuel-core/src/dispatch.rs` becomes a submodule under
`fuel-core/src/judge/` (or its contents fold directly into
`fuel-core/src/judge.rs`).

- Move `cached()`, `populate_dispatch_table()`, `invalidate()`,
  `try_load_persisted()` into `fuel-core::judge`.
- Re-export at the old path with `#[deprecated]` annotations for one
  cycle, or hard-break.

This is a small ancillary refactor in the same session — it removes
the name collision and clarifies ownership.

### Step 5 — update fuel-core consumers

- `fuel-core/src/lib.rs` — old `pub use fuel_storage::*` re-exports
  point at moved symbols. Flip to `pub use fuel_dispatch::*`.
- `fuel-core/src/pipelined_bridge.rs` — flip imports.
- `fuel-core/src/topology.rs` — flip `fuel_storage::dispatch::*` /
  `fuel_storage::global_bindings` to `fuel_dispatch::*`.
- `fuel-core/src/inference_context.rs` — flip imports.
- `fuel-core/src/factories.rs` — flip imports.
- Any other `fuel_storage::` import for moved symbols.

### Step 6 — update external consumers

Grep across the workspace for `fuel_storage::*` imports of moved
symbols. Each import flips to the new `fuel_dispatch::*` location.

Likely call sites:

- `fuel-lazy-examples/src/bin/*.rs`
- `fuel-graph-router/src/lib.rs` (if it touches dispatch)
- Tests in `fuel-graph-cpu`, `fuel-cuda-backend`, `fuel-vulkan-backend`,
  etc. if any reach for dispatch directly

### Step 7 — update `Cargo.toml` deps across the workspace

- **`fuel-dispatch/Cargo.toml`** (new): deps on `fuel-core-types`,
  `fuel-storage`, `fuel-graph`, backend crates (feature-gated),
  workspace deps for `smallvec`, `bytemuck`, etc.
- **`fuel-storage/Cargo.toml`**: drop deps that were only needed for
  moved code (likely the backend crate deps, depending on what
  `BackendStorage` enum's variant types need vs what the dispatch
  wrappers needed). Drop the feature flags that gated dispatch-only
  things; keep the ones the enum needs.
- **`fuel-core/Cargo.toml`**: add `fuel-dispatch` as a dep; propagate
  feature flags; verify backend crate deps that were only needed
  transitively through fuel-storage are still satisfied.
- **`fuel-graph-router/Cargo.toml`**: add `fuel-dispatch` if it consumes
  any moved symbols.
- **`fuel-lazy-examples/Cargo.toml`**: same.
- **Workspace root `Cargo.toml`**: add `fuel-dispatch` to `members`
  and `workspace.dependencies`.

### Step 8 — sweep

```powershell
cargo build -p fuel-dispatch
cargo build -p fuel-dispatch --features cuda
cargo build -p fuel-dispatch --features vulkan
cargo build -p fuel-dispatch --features cuda,vulkan

cargo build -p fuel-core
cargo build -p fuel-core --features cuda
cargo build -p fuel-core --features vulkan
cargo build -p fuel-core --features cuda,vulkan

cargo test -p fuel-dispatch --lib
cargo test -p fuel-dispatch --lib --features cuda
cargo test -p fuel-dispatch --lib --features vulkan
cargo test -p fuel-dispatch --lib --features cuda,vulkan

cargo test -p fuel-core --lib
cargo test -p fuel-core --lib --features cuda
cargo test -p fuel-core --lib --features vulkan
cargo test -p fuel-core --lib --features cuda,vulkan

cargo test -p fuel-storage --lib    # mostly empty now; sanity check
cargo test --workspace
```

Every test that passes today must pass after. Live-GPU tests
(`--ignored`) on RTX 4070:

```powershell
cargo test -p fuel-dispatch --features cuda,vulkan `
  audit_multi_backend_coverage -- --ignored --nocapture
```

Verify the output matches what's already captured in
`scripts/audit_output.txt` (modulo the test's home crate path).

### Step 9 — memory + audit doc updates

- New memory entry:
  `project_dispatch_crate_extracted.md` capturing what landed +
  TDP resolutions + the Phase 1 SystemTopology+Judge flag.
- Update `MEMORY.md` index.
- Update `project_judge_alternatives_audit.md` noting P0.2 shipped
  and the new crate's role.
- Update `project_system_topology_shipped.md` if topology code
  imports flipped paths.

## What's NOT in scope

- **`BackendStorage` enum move to `fuel-core-types`.** That's P0.2c.
- **Retiring `fuel-storage` entirely.** Also P0.2c.
- **`SystemTopology` / `Judge` / `ProbeReport` relocation.** Phase 1's
  problem (per TDP-B9).
- **Broader `fuel-core` decomposition** (extracting Judge, models,
  serialization, training to their own crates). Separate body of work.
- **Any picker semantics change.** No new `AlternativeFilter`, no
  filter-chain reshape, no enumeration via SystemTopology in
  `compile_plan`. All of that is Phase 1.
- **Optimizer ranker.** Phase 1.
- **Op::Copy planner pass.** Phase 2.
- **Splitting `pipelined.rs` or `dispatch.rs`'s 6000+ lines.** Per
  TDP-B1, move as-is.
- **Retiring `Router : GraphBackend`.** Phase 3 + the 9c retirement
  trajectory.
- **Changing any sibling project** (vulkane, lightbulb, mlmf, baracuda).
  Per feedback memory.
- **New features.** Existing feature flags transfer unchanged.

## Deliverables

1. New `fuel-dispatch` crate containing:
   - `binding/` — `KernelBindingTable`, `BindingEntry`, `KernelRef`,
     `KernelCaps`, `OpParams`, `CapabilityRegistry`
   - `register/` — `register_cpu_kernels`, `register_optional_backends`,
     `global_bindings()`, `extend_global_bindings`, backend-specific
     wrappers (cpu/cuda/vulkan)
   - `compiled.rs` — `CompiledNode`, `compile_node`, `execute_compiled`
   - `cost.rs` — `default_cost_for_op_kind` + family cost functions
   - `fused.rs` — `FusedKernelRegistry`, `KernelRevisionHash`,
     `PrecisionGuarantee`
   - `cast_fusion.rs`
   - `plan.rs` — `ExecutionPlan`, `compile_plan`, `resolve_kernel`,
     `TolerancePolicy` (Phase 1 will replace these)
   - `executor.rs` — `PipelinedExecutor`, `realize`, `realize_many`,
     work-item dispatch
2. `fuel-storage` reduced to just `BackendStorage` enum + variant
   glue (gets retired in P0.2c).
3. `fuel_core::dispatch` merged into `fuel_core::judge` (TDP-B2).
4. All workspace tests passing across CPU / CUDA / Vulkan / combined
   feature sweeps.
5. Live-GPU `--ignored` tests re-run on the dev box (RTX 4070)
   confirming no regression.
6. Audit harness rerun produces identical multi-backend coverage
   table.
7. Memory entry + audit-doc update + MEMORY.md index entry.

## Scope estimate

**1-2 focused sessions, 8-12 commits.** Mostly mechanical:

- `git mv` files
- Update imports across the workspace
- Workspace + per-crate `Cargo.toml` edits
- Verify feature matrix
- Verify live-GPU tests

Biggest risk surfaces:

- **Missed feature gates** — a `#[cfg(feature = "cuda")]` block that
  imports from the wrong path; only surfaces under specific feature
  combinations.
- **Stray imports** — a file that compiled because of a transitive
  re-export that disappears with the move.
- **Cargo dep cycles** — verify `cargo tree -p fuel-dispatch` and
  `cargo tree -p fuel-storage` don't reveal cycles.
- **External-binary breakage** — `fuel-lazy-examples` binaries each
  need their imports updated; one stale `use fuel_storage::pipelined::*`
  will compile-fail.

## Why this session, this scope, this order

Phase 1's optimizer ranker is the next architecturally meaningful
work. It will:

- Replace `compile_node` + `lookup_with_caps` with candidate
  enumeration via SystemTopology.
- Replace `TolerancePolicy` enum with the filter-chain shape.
- Reshape `NodeKernelBinding` into the per-decision-point
  alternative-set shape.
- Insert Op::Copy / layout-fixup ops as part of optimization.

Every one of those touches the dispatch code that currently lives
in `fuel-storage`. Doing those changes *in place* in `fuel-storage`
and then moving the rewritten code later is materially worse than
moving the boilerplate first and changing semantics in its new home.

The dispatch + executor extraction is also a clarity win
independent of the picker work — `fuel-storage`'s name finally
matches its remaining contents (a thin storage substrate, then gone
in P0.2c), and `fuel-dispatch` clearly owns "the layer that runs
graphs on backends."

## Pointers

- Phasing source: [`judge-alternatives-picking-audit-results.md`](
  ./judge-alternatives-picking-audit-results.md).
- Sister sessions:
  - P0.1 SystemTopology (shipped):
    `project_system_topology_shipped.md`
  - P0.2c: `BackendStorage` enum to `fuel-core-types`; retire
    `fuel-storage`.
  - P1.x: optimizer ranker work (depends on P0.2 landing).
- Files to move: as enumerated in the table above.
- Pattern reference: this is a pure-mechanical extract; no
  architectural shape changes. Compare to Phase 7.5 work item G
  (Storage → fuel-core-types) for a similar successful precedent.
