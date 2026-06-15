---
name: Judge / picker alternatives audit — results
description: 2026-05-30 scoping audit of where the Phase 7.6 step 9b picker, the Phase 6b Judge, and the PipelinedExecutor's first-registered dispatch sit relative to each other. Produces the working set + decision points for the follow-up sessions that wire empirical picking into the production executor.
date: 2026-05-30
session_type: audit / scoping
status: shipped — no implementation in this session
---

# Judge / picker alternatives audit — results

> **Archived 2026-06-15 — superseded design record.** This documents the v1-picker design (three pick surfaces, `resolve_kernel` / `TolerancePolicy`, a separate `ExecutionPlan`) that was **not** the path taken: the live design is the `ranker/` + Picker 2 (runtime selector) subsystem, and the 2026-06-14 "plan is the graph" redirection ([`../../architecture/10-decisions-log.md`](../../architecture/10-decisions-log.md), [`../../architecture/04-optimization.md`](../../architecture/04-optimization.md)) retired the per-node / top-N model entirely. Read as history; do not treat Sessions A–E or the decision-point recommendations as a live plan.

## Status updates since this audit shipped

- **2026-05-30 SystemTopology service shipped** as Session 0
  prerequisite. See `project_system_topology_shipped` memory.
- **2026-05-31 dispatch + executor extracted to `fuel-dispatch`
  crate** (P0.2 of the picker-work phasing). Paths below have been
  updated from `fuel-storage/src/*.rs` to `fuel-dispatch/src/*.rs`.
  `fuel_core::dispatch` (Judge cache) renamed to
  `fuel_core::judge::cache` to free the namespace. See
  `project_dispatch_crate_extracted` memory.
- **Phase 1 (optimizer ranker) is the next major work** — the
  binding-table dispatch path now lives in a dedicated crate where
  it can grow without further migrations.
- **Harness numbers have grown since the audit baseline.** Re-running
  on 2026-05-31 against the post-move tree reports **579 unique keys
  / 1137 alternatives / 365 multi-backend keys / 214 single-alt keys**
  (was 494 / 972 / 285 / 209 on 2026-05-30). The growth is the
  in-place ops dtype expansion (commits `41792de8`, `283037ba`)
  adding registrations across more dtypes. The audit's shape and
  conclusions are unchanged; the headline numbers below describe
  the original baseline.

## TL;DR

1. **Three pick surfaces coexist, none integrated.**
   - **`Router::pick_for_op`** (Phase 6b) consults the
     `DispatchTable` and picks a `BackendId`. **Only the legacy
     `GraphExecutor<B>` flow consults it** (Router : GraphBackend).
     PipelinedExecutor doesn't go through Router at all.
   - **`plan::resolve_kernel` + `TolerancePolicy`** (Phase 7.6
     step 9b) picks an alternative within one `(op, dtypes,
     backend)` decision-point. **Shipped, but unreferenced** —
     only `lib.rs` re-exports it and `plan.rs` tests it. No
     executor calls it.
   - **`compile_node` → `lookup_with_caps`** (PipelinedExecutor,
     today) takes the **first-registered alternative** for the
     `(op, dtypes, target_backend)` key. `target_backend` is
     pre-pinned by `pipelined_bridge::prepare()` to the
     user-passed device's backend.
2. **Multi-backend coverage is dense.** 494 unique `(op, dtypes)`
   keys; **285 of them (58%)** have ≥2 backends registered —
   most have all three of CPU / CUDA / Vulkan. **Zero** keys
   currently have multiple alternatives within a single backend
   (cuBLAS + CUTLASS-as-sibling isn't registered yet; per the
   memory `project_phase_7_6_step_4_in_progress` it's queued
   behind tolerance-policy work and a baracuda alpha bump).
3. **The picker integration story is gated on the executor-side
   "which backend?" decision migrating from Router → graph
   side-table.** Today the PipelinedExecutor never sees
   alternatives because `target_backend` is pinned monolithically
   per realize call. Until that is per-op (or per-decision-point),
   wiring Judge into `resolve_kernel` won't change executor
   behavior on the production path.

## Audit step 1 — multi-backend coverage today

### Harness

`fuel-dispatch/src/dispatch.rs::tests::audit_multi_backend_coverage`
— an `#[ignore]` test that iterates `global_bindings()`,
groups by `(op_kind, dtypes)`, and prints any key with `>1`
alternative.

Re-run with:

```powershell
cargo test -p fuel-dispatch --lib --features cuda,vulkan `
  audit_multi_backend_coverage -- --ignored --nocapture
```

Full output captured in [`scripts/audit_output.txt`](../../scripts/audit_output.txt).

### Headline numbers (build: `--features cuda,vulkan`, RTX 4070 host)

| Metric                                              | Count |
|-----------------------------------------------------|-------|
| Total unique `(op, dtypes)` keys                    | 494   |
| Total alternatives across all backends              | 972   |
| **Multi-backend keys (≥2 backends)**                | **285** |
| Single-alternative keys (no pick to make)           | 209   |
| Multi-impl single-backend keys (cuBLAS+CUTLASS-style)| 0     |
| Keys with 3 alts (CPU + CUDA + Vulkan)              | 193   |
| Keys with 2 alts                                    | 92    |

### Per-family coverage shape

Multi-backend coverage exists across **66 distinct OpKinds**.
The non-obvious patterns:

| Family                                | Coverage shape                                   |
|---------------------------------------|--------------------------------------------------|
| Elementwise unary/binary (f32/f16/bf16/f64) | 3-alt almost everywhere; the working core    |
| `MatMul`                              | 3-alt for {F32,F16,BF16}; {I8,U8} CUDA+CPU only |
| `Conv2D`                              | CPU+CUDA+Vulkan **f32-only**; {F16,BF16,F64} CPU-only |
| `QMatMul [F32,U32,F32]`               | CPU+Vulkan only (no CUDA registration yet)       |
| `*Inplace` (Relu/Silu/Gelu/Tanh/Sigmoid) | CPU+CUDA only (Vulkan inplace deferred)       |
| `InplaceAffine`                       | F32+F64 CPU+CUDA; BF16/F16 CPU only              |
| `LogSoftmaxLastDim`, `Pad`, `PadBackward`, `*Backward` | CUDA+CPU only (no Vulkan)        |
| `Rope`                                | CPU+Vulkan only (CUDA RoPE backward shelved per memory) |
| `ArgMaxDim`/`ArgMinDim`               | 3-alt across f32/f16/bf16/f64                    |
| `Triu`/`Tril`/`Flip`/`Roll`/`WriteSlice` | 3-alt fp dtypes; integer dtypes CUDA+Vulkan only (CPU registers fewer int variants) |

### What the working set means

**285 keys are the "Judge could plausibly help here" set.** Each
one is a decision point where:

- The user's hardware can execute on multiple backends.
- The pipelined executor currently runs **whichever happens to be
  registered first** at the chosen `target_backend` — but only the
  pinned backend's alternatives are even considered.

For the **executor** to actually consult the alternatives, two
things must change (in this order):

1. The "which backend runs this op?" decision has to move out of
   the monolithic `pipelined_bridge::prepare()` pinning and into
   a per-op (or per-decision-point) consultation.
2. `resolve_kernel` (or its successor) has to be invoked at that
   per-op point with a policy that can consult Judge data.

Today neither step exists for the pipelined path. The first step
is **architecturally significant**: it's effectively "build a
graph-aware Router for the binding-table world." Phase 7.6 step
9b's `compile_plan` is the right seam, but the executor doesn't
consume `ExecutionPlan` (Track B was deferred).

## Audit step 2 — picker flow trace

### Today's PipelinedExecutor dispatch path

```
fuel_core::pipelined_bridge::realize_one_as
  ├─ prepare()                                 fuel-core/src/pipelined_bridge.rs:236
  │   ├─ ensure_target_backends(graph, dev)   ── overwrites target_backend per node
  │   │                                          (graph.set_target_backend(id, dev_backend))
  │   └─ build_const_cache(...)               ── transient-graph Op::Copy chain
  └─ PipelinedExecutor::realize                fuel-dispatch/src/pipelined.rs:329
      ├─ compiler_thread_body                  fuel-dispatch/src/pipelined.rs:523
      │   └─ compile_one(graph, id, ...)       fuel-dispatch/src/pipelined.rs:571
      │       └─ compile_node(op, dtypes,      fuel-dispatch/src/compiled.rs:86
      │                       target_backend,
      │                       op_params, bindings)
      │           └─ bindings.lookup_with_caps(op, dtypes, backend)
      │                                       fuel-dispatch/src/kernel.rs:977
      │               └─ self.bindings.get(&key).and_then(|alts| alts.first())
      │                                                       ^^^^^^^^^^^^^^^^^
      │                                                       FIRST WINS — no policy
      └─ execute_work_item(...)                fuel-dispatch/src/pipelined.rs:2438
          └─ (compiled.kernel)(inputs, outputs, layouts, &params)
```

**Key observation:** `compile_node` does **not** consult `policy`
or `lookup_alternatives`. Multiple alternatives are silently
collapsed to the first-registered.

### The Phase 7.6 step 9b picker (UNUSED today)

```
fuel-dispatch/src/plan.rs
  - compile_plan(graph, order, table)           builds ExecutionPlan
  - resolve_kernel(binding, table, policy)      v1 picker — lazy cache
  - TolerancePolicy {BitStableFirst, FirstAlternative}
```

**Callers (search of the workspace):**

| Caller                            | Type                |
|-----------------------------------|---------------------|
| `fuel-dispatch/src/lib.rs:16`      | re-export only      |
| `fuel-dispatch/src/plan.rs` tests  | unit tests          |
| `fuel-dispatch/src/pipelined.rs`   | only doc-comments mention it; no calls |
| Any executor                      | **none**            |
| Any backend, router, fuel-core    | **none**            |

The picker is **shipped infrastructure that no production code
calls**. It's the seam for the future migration; the wire-up is
the work this audit unblocks.

### The Phase 6b Judge → Router path (legacy executor only)

```
fuel-core::dispatch::cached()                   returns Option<Arc<DispatchTable>>
   ↓
Router::with_dispatch_table(table)              fuel-graph-router/src/lib.rs:671
   ↓
Router::pick_for_op(op, dtype, n_elem, target)  fuel-graph-router/src/lib.rs:849
   ├─ table.pick_nearest(op, dtype, size_class, criterion)
   │     ↓
   │     pick → Pick { backend, device_index }
   ├─ if Some(b) = backend_for_id(pick.backend, target) → return b
   └─ else: fall through to backend_for(target)         (silent advisory)
   ↓
b.matmul(...) / b.unary(...) / ...               (GraphBackend trait)
```

The Judge data **is** consulted, but only via `Router :
GraphBackend`. The `GraphExecutor<B>` calls `B::matmul` etc;
when `B` is `Router`, the dispatch table picks among
backends-at-the-same-device-location (e.g. CpuBackend vs
AoclBackend vs MklBackend at `DeviceLocation::Cpu`).

Per `project_phase_7_6_step_9c_parity_audit`, `Router :
GraphBackend` is on the retirement path; Phase G of the 9c
migration is "rewire Router to use pipelined + KernelRef." Until
that lands, the Judge data is effectively pinned to the legacy
flow.

### The three surfaces summarised

| Surface                              | Picks                  | Status                  | Consumer                 |
|--------------------------------------|------------------------|-------------------------|--------------------------|
| `Router::pick_for_op` (Phase 6b)     | BackendId per op       | Live                    | `GraphExecutor<Router>`  |
| `resolve_kernel` (Phase 7.6 step 9b) | Alternative within key | Shipped, **unused**     | none                     |
| `compile_node`/`lookup_with_caps`    | First-registered       | Live                    | `PipelinedExecutor`      |

The architectural endpoint is **one** picker — empirical
Judge-driven, consulted at plan time (via `compile_plan`), with
output cached on `NodeKernelBinding`. Getting there means:

1. Migrate the executor to consume `ExecutionPlan` (9c Track B).
2. Replace `TolerancePolicy::BitStableFirst` with a real picker
   that consults Judge data when available, falls back to
   bit-stable + first-registered otherwise.
3. Move "which backend at this decision point?" into the picker
   (today it's pre-pinned via `prepare()`).
4. Retire `Router : GraphBackend`.

## Audit step 3 — decision points

Each item below is a choice a follow-up session must make. The
audit enumerates options + a recommendation; the deciding
session resolves and ships.

### DP-1: Picker scope — within-backend vs cross-backend

**Question:** When the picker has alternatives, does it choose
within one backend (today's `resolve_kernel` shape, e.g.
cuBLAS vs CUTLASS at `(MatMul, [BF16×3], Cuda)`) or across
backends (e.g. CPU vs CUDA vs Vulkan at `(MatMul, [F32×3], ?)`)?

**Options:**

- **A) Within-backend only.** Cleaner — `target_backend` stays
  the monolithic per-realize pinning; the picker only chooses
  among siblings registered against that backend. Today's 285
  multi-backend keys go un-picked. The single-backend
  alternatives (currently 0 registered) become the working set.
- **B) Cross-backend.** The picker chooses both the backend and
  the alternative within it. `target_backend` becomes a *hint*
  (or goes away entirely on kernel-bearing nodes). Op::Copy
  edges must be inserted when the picker crosses a device
  boundary. Couples to DP-3 (cross-device dispatch).
- **C) Two-tier.** Phase 1 = within-backend (unblocks CUTLASS
  registration). Phase 2 = cross-backend after Op::Copy
  injection infrastructure lands.

**Recommendation:** **C — two-tier.** Within-backend picking
ships value immediately (cuBLAS+CUTLASS, the original 9b
architectural payoff). Cross-backend picking is the deeper
architectural change; let it wait until the in-flight bridge-
retirement Phase 4 (cross-device dispatch) lands. Doing both at
once couples two large concerns.

### DP-2: Judge consultation policy

**Question:** When the picker has N alternatives at a decision
point and Judge data may or may not exist, what's the rule?

**Options:**

- **A) Always consult Judge.** If Judge has data → pick fastest.
  If not → fall back to `TolerancePolicy::BitStableFirst`.
- **B) Consult only when alternatives' static costs are within a
  band.** If `cost_a.flops / cost_b.flops ∈ [0.9, 1.1]` →
  Judge breaks the tie; otherwise pick statically. Saves
  Judge runtime cost on obvious wins.
- **C) Consult when policy demands.** Add a
  `TolerancePolicy::EmpiricalFastest` arm; only that arm
  consults Judge. `BitStableFirst` stays the default and skips
  Judge entirely.

**Recommendation:** **A — always consult, fall back when
absent.** The Judge lookup is `pick_nearest` on a small in-memory
table — sub-microsecond, well below a single kernel launch's
overhead. Adding policy-level gates (B / C) optimises for a
non-bottleneck and creates a configuration matrix users have to
reason about. The simpler rule: Judge is the truth when
available; static metadata is the fallback.

### DP-3: Destructive ops profiling

**Question:** The Judge loop runs each candidate N times against
the same input. For destructive ops (`ReluInplace`,
`InplaceAffine`, `WriteSlice`, `ZeroFill`, `Op::Copy`,
`Op::Release`, future fused-backward ops), iteration 2+ sees
post-mutation input → timings meaningless / errors.

**Options:**

- **A) Skip destructive ops entirely from Judge.** Document the
  gap. Picker falls back to first-registered for them.
- **B) Clone target before each iteration.** Correct, but adds
  memcpy cost to the measurement (and a clone may not even be
  cheap — e.g. cloning a 2GiB CUDA tensor 7 times).
- **C) Skip destructive ops AND emit "no Judge data" annotation.**
  Same as A but the dispatch table can flag the gap so the
  picker doesn't repeatedly look for data that won't exist.
- **D) Allow a single iteration for destructive ops.** Drops
  statistical confidence but at least produces *some* data.
  Compose with B for the multi-iteration case if needed.

**Recommendation:** **A initially, C if Judge starts logging
attempts.** The destructive op list is small (≤20 OpKinds today
per the in-place + structural-op set). First-registered is fine
for them — there's no perf-critical multi-impl alternative on
the horizon for inplace ops. Revisit if a CUTLASS-style sibling
emerges for a destructive family.

**Coupling note:** This DP is the audit prompt's explicit
"why audit first, don't implement" — destructive-ops semantics
intersect with `project_pipelined_executor_ordering_shipped`
(2026-05-30). Now that ordering integration has shipped, this
DP can be decided without further upstream work.

### DP-4: Cross-device dispatch (only if DP-1 = B or C-phase-2)

**Question:** If Judge says "Vulkan is fastest for this op" but
the input lives on CUDA, what happens?

**Options:**

- **A) Inject `Op::Copy { target: Vulkan }`.** Honest, but the
  copy cost may dominate. Picker needs cost-aware decisions
  (Judge gives the kernel time; the copy cost has to come from
  the static cost model or a separate copy-time profile).
- **B) Run on the input's backend; ignore the cross-device
  suggestion.** Pragmatic; locality often wins. Loses some
  potential gains.
- **C) Skip Judge's cross-device suggestions entirely.** Judge
  only ranks alternatives at the same (op, dtype, backend) key.
  Cross-device migration becomes a separate optimizer pass.

**Recommendation:** **DEFER until DP-1 Phase 2.** This DP is
only relevant if DP-1 takes the cross-backend path. The
within-backend-only Phase 1 doesn't surface this. If/when DP-1
Phase 2 lands, prefer **C** (separate the concern: cross-device
migration is graph-optimization, not picker concern).

### DP-5: Tolerance-budget integration order

**Question:** Step 9b's `TolerancePolicy` filters alternatives
by precision (e.g. only bit-stable ones). Does the Judge picker
run BEFORE the policy filter (pick fastest, drop if precision
fails) or AFTER (pick fastest among precision-acceptable)?

**Recommendation:** **AFTER — pre-filter by precision, then pick
empirically among survivors.** This is the architecturally clean
default that architecture v1.0 §04 names: bit-stability is a
correctness anchor, not something to be traded against speed
silently. The decision rule:

```
candidates = alternatives.filter(by policy)
chosen     = if judge_has_data(op, dtypes) then judge.pick_fastest(candidates)
             else BitStableFirst.pick(candidates)
```

**Implementation note:** The current `TolerancePolicy::FirstAlternative`
arm is the "tolerance budget already applied upstream" exit, which
suggests a future caller path that did its own precision filter.
That composes naturally with AFTER.

### DP-6: Fallback when Judge has no data

**Question:** Cold start, new shape, `--no-judge` flag. What
does the picker do?

**Options:**

- **A) First-registered.** Today's behaviour; safest.
- **B) Static cost-rank (use the `CostFn` already attached to
  each `BindingEntry`).** Layer-1 cost model is FLOPs +
  bandwidth — better than nothing for cold start.
- **C) Binding-table caps + precision rank.** Pick by
  `bit_stable_on_same_hardware` first, then by `caps.strided_input`
  (avoid an auto-contiguize step), then by registration order.

**Recommendation:** **C, with B as a future refinement.** The
current `BitStableFirst` already implements C's first tier;
expanding it to also prefer `strided_input: true` when the input
is non-contiguous is a small additive change that captures real
perf signal. Static cost-rank (B) is appealing but the cost
functions today are coarse — Layer-1's "FLOPs ± kernel_overhead"
under-predicts the Vulkan command-buffer submission cost (per
the V.3 fan-out memory). Refining the cost model is its own
session; don't gate the picker on it.

### DP-7: Persistence + invalidation

**Question:** Judge reports persist to disk via
`PROFILE_REPORT_VERSION`. When does the picker invalidate?

**Options:**

- **A) Never (manual flush).** Current behaviour — `invalidate()`
  is an explicit API call.
- **B) On detected hardware diff.** Already implemented in
  `try_load_persisted()` — `now_probe.diff(&prior).needs_rejudge()`.
- **C) On kernel-binding-table revision change.** Hash the
  binding-table's KernelRef function pointers; if the hash
  differs from the persisted profile, re-run Judge.
- **D) On driver/library version change.** Surface this via
  the probe; persist driver versions, diff on load.

**Recommendation:** **B is already done. Add C in a later
session.** The `KernelRevisionHash::UNTRACKED` slot on
`NodeKernelBinding` is the seam for C; until persistence-of-
picked-kernels lands, there's nothing to invalidate. D is a
nice-to-have but driver upgrades on dev machines are rare;
the cost of a stale profile is a slower-than-optimal kernel,
not incorrect results. C is more important than D.

### DP-8: Measurement granularity — per shape vs per shape-class

**Question:** Profile per exact shape, per log2 size-class
(today's `SizeClass`), or per-op only?

**Options:**

- **A) Status quo (log2 size-class).** `SizeClass::from_elem_count`
  buckets shapes into log2 buckets. Coarse but cheap.
- **B) Per exact shape.** Most accurate; explodes the table
  size and warmup cost. Many shapes appear once.
- **C) Per shape-class with nearest-neighbour fallback.** Already
  implemented in `pick_nearest`. Today's de-facto default.
- **D) Per shape-class with per-bucket linear interpolation.**
  More work, marginal gain.

**Recommendation:** **C — keep status quo.** The per-(op, dtype,
size_class) granularity has been validated in the original
Phase 6b shipping (crossover at 2^16 elements for matmul). The
table size stays small (28 OpKinds × 5 size classes × N
backends ≈ 5KB). Per-exact-shape is over-engineering; revisit
only if a perf-critical op shows mis-routing inside one size
class.

## Scope estimate — follow-up sessions

Each is its own session; do NOT bundle.

> **Update 2026-05-30 — Session 0 prerequisite shipped.** The
> `SystemTopology` service (`fuel-core::topology::SystemTopology`)
> is now the source-of-truth for which backends share storage on the
> same device, what transfer paths connect devices, and what
> `(op, dtype)` pairs each backend has kernels for. Sessions A/B/C/D
> below should consume `SystemTopology::current()` instead of
> walking `global_bindings()` / `global_registry()` / `ProbeReport`
> directly. See [`system-topology-service.md`](
> ./system-topology-service.md) and memory
> `project_system_topology_shipped` for the predicates + TDP
> resolutions. No consumers are wired yet — that's the next session's
> job, per session-split discipline.

### Session A — wire `compile_plan` + `resolve_kernel` into PipelinedExecutor (Track B Phase 1)

**Goal:** PipelinedExecutor consumes `ExecutionPlan`; per-node
dispatch calls `resolve_kernel` instead of `compile_node`.

**Scope:**
- `compiler_thread_body` builds an `ExecutionPlan` via
  `compile_plan` at the start of the realize loop.
- `compile_one` looks up the binding from the plan and calls
  `resolve_kernel(binding, &bindings, TolerancePolicy::default())`.
- Today's `compile_node` either becomes a thin shim
  (`resolve_kernel` + assemble `CompiledNode`) or is retired.
- Tests: existing pipelined regression sweep + a new test
  proving picker policy round-trips (register a non-bit-stable
  first and a bit-stable second; assert the bit-stable wins
  on the executor path).

**Risk:** Low — the picker reduces to first-registered under
`BitStableFirst` when only one alternative exists or all are
UNAUDITED (early lints would have caught this). No behaviour
change expected on today's binding table.

**Effort:** ~1 session, 2-3 commits, ~150 LOC.

### Session B — register CUTLASS bf16/f16 matmul as cuBLAS siblings

**Goal:** Validate the picker path with the original Phase 7.6
step 9a/9b architectural payoff. Register CUTLASS kernels at
`(MatMul, [BF16×3], Cuda)` alongside cuBLAS; assert under
`TolerancePolicy::BitStableFirst` that cuBLAS still wins; switch
the policy to `FirstAlternative` and assert CUTLASS now wins.

**Dependencies:** Session A. Needs baracuda CUTLASS bindings
(per `project_baracuda_cutlass_critique` blocked on
`DeviceSlice::from_raw_parts`; verify status at session start).

**Effort:** ~1 session if baracuda surface is ready; deferred
otherwise.

### Session C — graph-aware Router for the binding-table world

**Goal:** Move "which backend runs this op?" out of
`pipelined_bridge::prepare`'s monolithic pinning into a per-op
decision consulted at plan time.

**Scope:**
- New surface in `compile_plan` (or a `route_plan` companion):
  per-node, given the input devices + the binding-table's
  multi-backend coverage + Judge data, pick the backend.
- Op::Copy edge insertion when the picked backend differs from
  the input's residency.
- Retain a "pin everything to the same backend" mode for the
  current test pattern (`realize_f32_cuda(&dev)`).

**Risk:** HIGH — this is the architectural shift that unlocks
the 285 multi-backend keys for empirical picking. Couples to
DP-3, DP-4. The bridge-retirement Phase 4 work (cross-device
dispatch graph-level) is the natural home.

**Effort:** ~3-5 sessions split across phases (per-op decision,
Op::Copy insertion, Op::Move insertion, Router-as-GraphBackend
retirement).

### Session D — wire Judge data into `resolve_kernel`

**Goal:** Picker consults `fuel_core::dispatch::cached()` when
available. The DP-2 + DP-5 + DP-6 decisions implemented.

**Scope:**
- New `TolerancePolicy` arm (or a richer Policy type) carrying
  a Judge handle.
- `resolve_kernel` body: if Judge has data for the binding's
  op + size, pick fastest among precision-acceptable
  alternatives; otherwise BitStableFirst.
- Tests: register two alternatives + populate a synthetic
  `DispatchTable` claiming alt B is 10× faster; assert alt B
  wins regardless of registration order.

**Dependencies:** Sessions A + C (or just A, if DP-1 = C
Phase 1 = within-backend-only).

**Effort:** ~1-2 sessions, ~250-400 LOC including tests.

### Session E (optional) — Judge coverage for destructive ops

**Goal:** Per DP-3, the audit recommends skip-for-now. If a
destructive op family ever gets multi-impl alternatives,
revisit here.

**Trigger condition:** A backend lands an inplace-op alternative
that diverges meaningfully in perf from the default. Probably
never; capture as a tripwire memory entry if it happens.

## Recommended ordering

```
A (executor consumes ExecutionPlan)
   ├─ B (CUTLASS sibling validates within-backend picking)
   └─ D (Judge consultation in resolve_kernel) — composes with C
C (graph-aware Router for cross-backend picking)
   └─ D fully unlocks (cross-backend Judge picks)
```

Sessions A and C are independent and can run in parallel. B
gates on A; D gates on A (or A + C for cross-backend Judge
picks). E is a tripwire, not a planned session.

## What this audit did NOT do

Per the prompt:
- No picker code was written.
- No infrastructure for any DP was built.
- No cross-backend kernel writing (no "add Vulkan InplaceAffine").
- No `populate_dispatch_table` or Judge runner touched.

The only working-tree change is the `--ignored` audit harness
test in `fuel-dispatch/src/dispatch.rs` plus its captured output
in `scripts/audit_output.txt`. The test is durable: future
sessions can re-run it after registration changes to track the
multi-backend coverage growth.

## Pointers

- Harness: `fuel-dispatch/src/dispatch.rs::tests::audit_multi_backend_coverage`
- Output: `scripts/audit_output.txt`
- Picker: `fuel-dispatch/src/plan.rs` (compile_plan + resolve_kernel + TolerancePolicy)
- Today's dispatch: `fuel-dispatch/src/compiled.rs::compile_node`,
  `fuel-dispatch/src/kernel.rs::KernelBindingTable::lookup_with_caps`
- Judge: `fuel-core/src/judge.rs`, `fuel-core/src/dispatch.rs`
- Legacy picker (Router): `fuel-graph-router/src/lib.rs::Router::pick_for_op`
- Coverage memory: `project_phase_7_6_step_4_in_progress`,
  `project_phase_7_6_step_9c_parity_audit`,
  `project_pipelined_executor_ordering_shipped`
