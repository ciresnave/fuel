# Architectural audit: in-flight threads in fuel

**Status**: 2026-05-08. Branch `feature/storage-unification` at `35b1d038`.

> **2026-05-09 update — superseded by architecture v1.0**: this audit triggered the establishment of the architecture set in [`docs/architecture/`](architecture/00-index.md) (shipped at v1.0 on 2026-05-09). The five cross-cutting questions surfaced here (Q-A through Q-E in §4 below) have been resolved as follows:
>
> - **Q-A** (binding-table catalog vs runtime lookup): **resolved** — binding table is a planning-time catalog only; the executor calls pre-resolved `KernelRef` function pointers directly. See [03-ir §The optimized form](architecture/03-ir.md#the-optimized-form-top-n-routes-with-pre-resolved-kernels) and [11-persistence §Re-resolution on use](architecture/11-persistence.md#re-resolution-on-use-lazy-not-at-load).
> - **Q-B** (static cost vs empirical cost composition): **resolved** — three-layer composition (static priors with community refinement, empirical Judge data, runtime telemetry) per [04-optimization §Cost model](architecture/04-optimization.md#cost-model-static-annotations-refined-by-empirical-judge-data-accounting-for-parallelism).
> - **Q-C** (op identity post-fusion): **resolved** — Op-shape A: single `Op` enum with primitive variants + one `Op::Fused(id, params)` arm. Per [03-ir §How nodes carry their op identity](architecture/03-ir.md#how-nodes-carry-their-op-identity).
> - **Q-D** (cross-cutting types' home post-fission): **deferred** — revisit when fuel-core fission begins (Phase 7.5 work item E).
> - **Q-E** (foundation-first vs feature-first sequencing): **resolved** — the architecture set is the foundation; subsequent phase work anchors to it.
>
> The recommended sequence in §5 below has been incorporated into ROADMAP.md (Phase 7.6 entry rewritten to v2 against architecture v1.0; Phase 7.6 design doc at `docs/fused-op-registry.md` revised). The 24 architectural decisions made during the v0.x → v1.0 drafting period are recorded in [10-decisions-log.md](architecture/10-decisions-log.md).
>
> This audit document is preserved as the historical artifact that triggered the architecture-set establishment. Read it for context on why fuel's architecture is what it is now; **read the architecture set** for what fuel commits to going forward.

---

This document is a snapshot, not a proposal. It maps every in-flight architectural thread, identifies where they couple, surfaces cross-cutting design questions whose answers affect more than one thread, and recommends a sequencing for the next ~6-10 weeks. No code changes accompany this document.

The audit was triggered when the Phase 7.6 (FusedOpRegistry) implementation kickoff surfaced design questions that turned out to span three other unfinished threads. Rather than answer those questions one at a time mid-implementation, we paused to map the picture.

---

## 1. Purpose, method, and what this audit is not

**Purpose.** Decide, with information rather than vibes, whether to:
- Proceed with Phase 7.6 in its currently-designed scope.
- Broaden Phase 7.6 to absorb adjacent cleanups (Judge extension, binding-table cleanup).
- Pause Phase 7.6 to land foundation work that the registry depends on.
- Re-sequence multiple in-flight threads.

**Method.** For each in-flight thread: read the current code, the design doc (if one exists), the relevant memory entries, and the ROADMAP entry. Record what's shipped, what's pending, what it depends on, and what depends on it. Then identify cross-thread coupling by inspection — places where two threads make assumptions about each other.

**What this is not.** Not a redesign. Not a commitment to any particular sequence. Not a prescription. The output is a map; the user makes the routing decision.

**Honest caveats.** The threads listed here are the architectural ones — the work that changes the *shape* of fuel. Routine kernel-coverage work (e.g. "more Vulkan ops"), routine binding fanout (e.g. "more dtypes for IndexSelect"), and routine bug-fixing are not threads in this sense. They proceed regardless.

---

## 2. The eight threads

Each thread has a one-line summary, a status sheet (shipped / partial / pending / in-flux / blocked), what it owns, what it depends on, what depends on it, and the open questions that remain.

### Thread T1 — Storage unification (Phase 7.5 work item A and beyond)

**Summary.** Replace per-backend storage variants with a uniform `Storage { bytes, dtype, device }` substrate. Per-backend `BackendStorage` trait. Capability advertisement via `BackendCapabilities`. Source: `docs/storage-unification.md`.

**Status.** Phase A (substrate) **shipped** — `Storage`, `BackendStorage` trait, `BackendCapabilities`, `TransferPath`. Phase C CPU op surface **shipped** — F32/F64/BF16/F16 across the full inference + serving op surface plus 11 GGML quant types (per memory `project_phase_c_unary_binary_shipped`). Phase B (`.realize()` stubs) **partial** — B1 shipped (a8e192ff), B2-B6 plan exists (`project_phase_7_5_work_item_b_plan`), B3 *paused 2026-05-03* in favor of foundation work. Phase D (cleanup of legacy types) **pending**.

**Owns.** `Storage` substrate in fuel-core-types. `BackendCapabilities` shape. The seam between fuel-graph (handles) and fuel-storage (bytes + dispatch).

**Depends on.** Nothing (it's the foundation).

**Depended on by.** Every kernel call — Phase C op surface migrated through this. Backend depth migrations (T5) cross this seam. Layout-on-Node (T7) is half-completed because Phase A landed. FusedOpRegistry (T2) needs the BackendImpl payload type to live somewhere; the natural home is alongside Storage's substrate.

**Open questions.** Does B3 resume with the current shape or does the unification's promise of Storage-as-substrate change what B3 should do? B3 was paused because the CoW + invalidation design tangled with the storage seam. Resolution path: revisit B3 design only after FusedOpRegistry lands, since B3 will run against post-7.6 graph shape.

---

### Thread T2 — FusedOpRegistry (Phase 7.6)

**Summary.** Split `Op` into closed primitive enum + open registry of fused ops. Each registry entry encodes pattern, decomposition, per-backend kernel + cost, backward identity, shape/dtype rules. Source: `docs/fused-op-registry.md`. ROADMAP §Phase 7.6.

**Status.** Design **complete** (commit 35b1d038, 2026-05-07). Implementation **not started**. Today's session was meant to be Phase 7.6 step 1 (skeleton) but pivoted to this audit when cross-cutting questions surfaced.

**Owns.** `FusedOpRegistry`, `FusedOpEntry`, `FusedOpId`, `FusedOpParams`. `NodeKind::{Primitive(Op), Fused {id, params}}`. The auto-generation of lowering and fusion rules from registry entries (replacing PR 3's hand-written `SoftmaxLastDimLowerRule` / `SoftmaxLastDimFuseRule`).

**Depends on.** PR 3's rule registry (T6, shipped). Op enum — to drop the hybrid form. KernelRef shape from fuel-storage's binding table (T8). The way autograd looks up backward rules (T3).

**Depended on by.** Cost-based scheduler (Phase 4, future). Cross-backend fusion visibility for the Router. Phase 6b empirical Judge if it ever profiles fused ops (T4).

**Open questions.**
- Where does the registry crate live? Investigation says fuel-graph for the metadata half, fuel-storage for the `BackendImpl` payload (since it carries `KernelRef`).
- `FusedOpParams` shape — confirmed enum.
- Pattern representation — closure + helper module.
- Cost estimate shape — user picked static-cost-only `{ flops, bytes_moved, kernel_overhead_ns }`.
- Binding-table key naming under primitive+fused unification — unresolved pending decision on T8 cleanup.
- Are Conv2D/QMatMul registry entries — user confirmed yes.

---

### Thread T3 — Autograd: GradientRule trait migration (Phase 6d Track 2)

**Summary.** Replace the inline match-on-`Op` in `Tensor::backward` (~600 LOC, lib.rs:3700-4100) with per-op `GradientRule` impls registered in a dispatcher. Source: `fuel-graph/src/grad.rs` head comment.

**Status.** Scaffolding **shipped** — `GradientRule` trait, `dispatch_gradient` entry. Migrated ops: **3** (Add, Mul, Relu). Inline match arms remaining: **~40+** (every other Op variant including all fused ops and all backward helpers). Migration was started, never completed. No current memory entry tracks active progress.

**Owns.** Per-op backward rules. The rules registry (or trait dispatch table) that consumes them. Higher-order gradient story (today: panics on second-order through fused-backward helpers).

**Depends on.** Op enum's current shape — every rule pattern-matches on `&Op`. After T2 ships, rules will pattern-match on `&NodeKind` instead.

**Depended on by.** Anyone using `.backward()`. The unified-forward-and-backward graph design (Phase 7.5 work item C) — autograd-as-graph-rewrite presumes a clean per-op rule surface, not a giant inline match.

**Open questions.**
- Is migration paused intentionally, or did it stall? (No recent commit activity around `grad.rs`.)
- After T2 lands, the rule signature changes from `&Op` to `&NodeKind`. Should T3 migration continue with the old signature and re-migrate after T2, or wait for T2?
- The four fused-backward helpers (`SoftmaxLastDimBackward` etc.) currently panic on second-order gradients. Does T2's "registry entry has a `backward: BackwardKind` field" subsume the GradientRule trait, or do they coexist?

---

### Thread T4 — Empirical Judge / DispatchTable (Phase 6b)

**Summary.** Profile every (op, dtype, size_class) on every (backend, device); persist to a `ProfileReport`; build a `DispatchTable` for O(1) runtime "which backend wins for this op at this size?" picks. Source: `fuel-core/src/judge.rs`, `fuel-core-types/src/dispatch.rs`.

**Status.** Architecture **shipped** for primitive ops. Profile coverage **partial** — only `MatMul` and `AddElementwise` are actively profiled today; F32 only. The OpKind enum has 47+ variants but the Judge's match arm covers 2. Vulkan is detected by probe but not yet profiled (no `realize_f32_vulkan` helper).

**Owns.** The `Judge` (probes, runs measurements, emits `ProfileReport`). The `DispatchTable` (consumes reports, answers `pick(op, dtype, size_class, criterion) -> Pick`). The empirical-cost half of "which backend should this op run on?"

**Depends on.** OpKind enum (T8). Backends being callable through a uniform realize path (Router, T6/T8). The size-class taxonomy (`SizeClass`, log2-bucketed).

**Depended on by.** Router's dispatch policy. The (future) cost-based scheduler that combines static cost (from T2 registry entries) with empirical cost (from T4 profile data) for placement decisions. AOCL Router empirical dispatch already integrates Judge results for matmul (per memory `project_phase7b_aocl_shipped`).

**Open questions.**
- The Judge is primitive-only today. After T2 ships, fused ops also need profiling so the scheduler can compare "fused matmul+bias+relu" vs "decomposition" empirically. The Judge's match arm and `OpKind` need to grow accordingly. The empirical `DispatchKey` would also need to gain a fused-op axis (or `OpKind` needs to absorb `FusedOpId`).
- Does the Judge live in fuel-core forever, or does it move to a sibling crate when fuel-core fissions per Phase 7.5 work item E?
- Static cost (T2) vs empirical cost (T4): when scheduling, when does the planner trust which? Design doc for T2 says cost estimates are *advisory* and Judge is *authoritative*; the policy that combines them is Phase 4 work, not yet specified.

---

### Thread T5 — CUDA depth migration / Tier 1 fanout

**Summary.** Native CUDA kernels for every op, unblocking real models on real GPUs without falling back to the CPU through the legacy executor. Source: memory entries `project_cuda_depth_migration_roadmap`, `project_cuda_*_shipped`.

**Status.** Tier 1 mechanical fanout **partial**. Shipped per memory: binary fanout (5 ops: Sub/Mul/Div/Maximum/Minimum), unary fanout (15 ops: Relu→Step), reductions (4 ops: Sum/Max/Min/Mean), MatMul + Affine, Cast (with multi-dtype binding-table key). Native CUDA `ReduceSumTo` + `ReduceMaxTo` shipped (commit dd3694e3) so lowered SoftmaxLastDim runs GPU-resident end-to-end. Tier 1 was **paused** mid-fanout for SoftmaxLastDim foundation work that produced PR 3 / 3.5 / 3.5-followups.

**Owns.** The CUDA backend's per-op kernel coverage. Backend-side parity test infrastructure. The first published example of "a backend lives in its own crate, gets registered through fuel-storage's dispatch table."

**Depends on.** OpKind / binding table (T8). Storage substrate (T1). KernelRef ABI (mostly stable, layout-on-KernelRef partial — see T7).

**Depended on by.** Real model E2E performance on CUDA. Phase 6b Judge's CUDA arm (it can't profile what isn't kernelized). Multi-backend Router decisions that assume parity coverage.

**Open questions.**
- Does Tier 1 resume now, in parallel with T2, or after? Pre-T2, every fused-op CUDA kernel registered via the existing dispatch path; post-T2, fused-op kernels register via `BackendImpl` against a `FusedOpId`. Backend authors writing CUDA Tier 2 kernels would have to re-encode if T2 lands first.
- The pause-for-foundation pattern is recurring. Tier 1 was started, paused for SoftmaxLastDim foundation, paused again now for this audit. The audit's job is to surface whether the foundation work is the right thing to keep doing or whether Tier 1 should resume.

---

### Thread T6 — Scheduler-driven residency (PRs #1-#4) + graph optimizer framework (PR 3 / PR 3.5)

**Summary.** Two related initiatives that both rewrite the graph. T6a: residency primitives (`Op::Copy`, `Op::Move`, `Op::Release` with `destructive_input` metadata) + `derive_ordering` + `execution_plan` + `GraphMutatingSchedulerRule` trait + `ResidencyEvictionRule`. T6b: PR 3's rule registry framework (`Rule` trait, `RuleFamily::{Lowering, Fusion}`, `RuleRegistry`) + first rule pair (SoftmaxLastDim lower/fuse) + PR 3.5's primitives (`Op::ReduceMaxTo`, `Op::Unsqueeze`, `Op::ReduceMaxToBackward`).

**Status.** **Shipped.** PR 3 framework + lower/fuse pair. PR 3.5 primitives. PR 3.5 follow-ups (native CUDA reduce-to, fused max backward, metadata-only unsqueeze). All on `feature/storage-unification`.

**Owns.** Rule trait + registry. Optimize-to-fixpoint driver. Residency eviction rule and the destructive-op + ordering-edges machinery underneath it. The first cross-backend equivalence test that proves a lowered subgraph runs natively on CUDA.

**Depends on.** Op enum, Layout side-table (T7), Storage substrate (T1).

**Depended on by.** Phase 7.6 (T2) — the registry replaces hand-written SoftmaxLastDim rules with auto-generated ones; T2 is meant to be a thin layer ON TOP of the rule registry, not a replacement. Cost-based scheduler (Phase 4) consumes residency primitives + registry for placement and eviction. Future autograd-as-graph-rewrite (T3 + Phase 7.5 work item C) reuses the rule machinery for backward emission.

**Open questions.**
- The transactional snapshot / in-flight switching model is designed (ROADMAP §Phase 7.5 graph optimizer architecture) but not implemented. PR 3 is synchronous, single-graph, no transactions. Does Phase 7.6 implementation surface a need for transactions sooner, or can transactions stay deferred?
- The cost-based scheduler that consumes T6's residency primitives is Phase 4 work, not yet specified end-to-end.

---

### Thread T7 — Layout-on-Node + auto-Contiguize + KernelCaps

**Summary.** Tensor layout (strides, broadcast, slice) is a side-table on the graph (`Graph.layouts`), not metadata on Storage. View ops (Transpose, Permute, BroadcastTo, Slice, Reshape, Unsqueeze) are zero-copy and metadata-only. The executor's auto-Contiguize gate materializes bytes only when a kernel doesn't claim `KernelCaps::strided_input`. Source: memory entries `project_layout_on_node_complete`, `project_pr2_binary_broadcast_shipped`.

**Status.** **Shipped end-to-end** (tip 6beaf89d, 2026-05-04). PR 2 added `KernelCaps::strided_input` and migrated 6 binary F32 CUDA wrappers to consume layouts. PR 3's `Op::is_view_op()` + `derive_view_output_layout` completed the side-table migration; `Graph::push` auto-populates layouts for view ops. PR 3.5's `Op::Unsqueeze` joined the view-op set.

**Owns.** The Layout side-table. The auto-Contiguize gate (in `compile_one` / executor). The KernelCaps capability bit.

**Depends on.** Storage substrate (T1) — layouts are graph-side, not storage-side, so they decouple from Phase A's Storage seam.

**Depended on by.** Every kernel that consumes non-contiguous input. Future fused-op kernels that operate on strided tensors. Phase 7.6 (T2) — fused-op nodes need their own "is this a layout-affecting op?" answer.

**Open questions.**
- After T2 lands, `NodeKind::Fused` nodes need to either (a) skip the auto-populate-layout-on-push path (none of the current 13-14 fused ops are view ops), or (b) declare per-entry whether they're layout-preserving. Trivial in step 2 of T2's migration but worth listing.
- Does `KernelCaps::strided_input` extend to fused-op kernels post-T2, or do per-entry `BackendImpl`s carry their own caps?

---

### Thread T8 — Binding table / OpKind / Kernel registration

**Summary.** Every per-backend kernel is a `KernelRef` function pointer registered in `KernelBindingTable` keyed by `(OpKind, KernelDTypes, BackendId)`. Backends call `register_*_kernels` at startup to populate. Source: `fuel-storage/src/kernel.rs`, `fuel-storage/src/dispatch.rs`.

**Status.** **Shipped and stable**. Binding table is the load-bearing kernel-dispatch surface. ~50 register call sites across backends. Multi-dtype key (per memory `project_cuda_cast_and_multidtype_key_shipped`) added recently for Cast.

**Owns.** `KernelRef` ABI. `OpParams` enum (kernel-side extras bag). `KernelBindingTable` lookup API. The boundary between fuel-storage (dispatch wrappers) and backend crates (typed kernels).

**Depends on.** OpKind (T4 / T8). Storage substrate (T1).

**Depended on by.** Every backend. Every kernel call at execution time. Phase 6b Judge dispatches through this. Phase 7.6 (T2) needs to extend it to support `FusedOpId`.

**Open questions** — this thread is where most of the cross-cutting tension lives:
- **Lookup at execution time vs at planning time.** Today the executor calls `binding_table.lookup(op, dtypes, backend) -> Result<KernelRef>` per kernel call. This treats the binding table as a runtime decision point. Architecturally, it's a *catalog* — by the time the optimizer has chosen a backend, the kernel-fn-pointer is determined. A cleaner design pre-resolves `KernelRef` at planning time and stores it on each node; the executor never looks up anything. This isn't shipped, isn't designed in detail, and isn't currently scoped to any phase.
- **Naming under Phase 7.6.** The proposed extension `BindingOp { Primitive(OpKind), Fused(FusedOpId) }` collides with the existing private `DispatchKey` in `fuel-core-types/src/dispatch.rs`. The two "dispatch keys" answer different questions but share the word.
- **Two parallel registration surfaces during T2 migration.** Until step 10, primitive kernels register through `table.register(OpKind, ...)` and fused kernels register through `registry.attach_backend_impl(FusedOpId, ...)`. The two surfaces need to coexist for the migration window.

---

## 3. Coupling map

The threads are not independent. Below is the practical coupling graph; an arrow from A → B means "A's design assumes a property of B" or "changes to B affect A."

```
T1 (Storage)  ──┬─→ T7 (Layout-on-Node)
                ├─→ T8 (Binding table — KernelRef takes Storage)
                └─→ T5 (CUDA) — every kernel reads/writes through Storage

T6 (Optimizer + residency) ──┬─→ T2 (FusedOpRegistry — auto-generates rules)
                              ├─→ T7 (rule-inserted view nodes auto-populate layouts)
                              └─→ T8 (rule-inserted destructive ops use ordering edges)

T2 (FusedOpRegistry)  ──┬─→ T8 (binding table extension — BindingOp)
                        ├─→ T3 (autograd backward emission via registry)
                        ├─→ T7 (NodeKind::Fused interaction with auto-Contiguize)
                        ├─→ T4 (Judge eventually profiles fused ops)
                        └─→ T5 (CUDA fused-op kernels register through registry, not table)

T4 (Judge / DispatchTable) ──┬─→ T8 (DispatchKey collision; needs FusedOpId axis post-T2)
                              └─→ T5 (Judge can only profile what's kernelized)

T3 (GradientRule)  ──→ T2 (post-T2, rule signatures change Op → NodeKind)

T5 (CUDA)  ──┬─→ T8 (registers through binding table)
              ├─→ T1 (Storage CUDA variant)
              └─→ T7 (some kernels claim strided_input, others don't)
```

**Critical mutual coupling pairs:**

- **T2 ↔ T8.** The binding table extension to support fused ops is the most concrete cross-thread interface. Naming and step-10-vs-step-1 timing are open.
- **T2 ↔ T6.** T2 is meant to ride on top of T6's rule registry; T6 shipped two weeks before T2's design even completed. T2's "auto-generated rules" assumes T6's rule trait stays stable.
- **T4 ↔ T2.** Empirical profiling of fused ops can't happen until both T2 has landed AND T4 has grown its `match` arm. The user's question that triggered this audit was specifically "the Judge should be profiling all ops" — i.e. T4's primitive-only scope is a gap.
- **T3 ↔ T2.** Migrating autograd to GradientRule under the current Op enum means re-migrating after T2. Stalling T3 until T2 lands is logical but increases T2's review surface.
- **T5 ↔ T2.** Each unmigrated CUDA Tier 2 kernel that lands pre-T2 is technical debt — it'll be re-registered through the registry post-T2.

---

## 4. Cross-cutting design questions

Five questions span more than one thread. None has a current consensus answer. Each needs a decision before the threads it touches can stabilize.

### Q-A. Binding table: planning-time catalog or runtime lookup?

Today the executor calls `binding_table.lookup(...)` per kernel invocation. This treats the table as a runtime decision point even though the actual decision (which backend, which kernel variant) was made upstream by the optimizer.

The cleaner alternative: optimizer pre-resolves `KernelRef` per node and stores it on the node; executor never looks up anything; binding table becomes a planning-time catalog only.

**Threads affected:** T8 (definitionally), T2 (registration shape), T4 (Judge's relationship to the catalog), T5 (kernel registration surface), T6 (rule-inserted nodes need their KernelRef resolved at insertion time).

**Cost of the cleaner design:** ~3-5 days. Adds a `kernel: KernelRef` field to `Node` (or to a parallel side-table). Migrates the executor to call `node.kernel(inputs, ...)` instead of looking up. The optimizer (and rules that insert nodes) populate the field.

**Cost of leaving as-is:** the runtime lookup hot path stays. Functional, just architecturally not where the decisions live.

### Q-B. Static cost vs empirical cost: how do they combine?

T2 specifies static cost functions per `BackendImpl` returning `{ flops, bytes_moved, kernel_overhead_ns }`. T4 measures actual latency per profile entry. The Phase 4 cost-based scheduler (unbuilt) will combine them, but the policy is unspecified.

Concrete options:
- **Empirical wins, static is fallback.** Use Judge data when it exists; fall back to static when profile data is absent for this `(op, dtype, size_class, backend)` cell.
- **Static is plan-time, empirical is run-time adaptation.** Plan with static; rerun the planner if observed latency diverges from prediction by more than threshold X.
- **Linear combination.** weighted blend, weights tuned by Criterion.

**Threads affected:** T2 (cost shape), T4 (profile granularity), T6 (which the rule pipeline trusts).

**Recommendation deferred — this is genuinely Phase 4 work.** But T2 shouldn't ship cost functions without at least a sketch of how they'll be consumed; otherwise the cost surface drifts from what the scheduler will actually want.

### Q-C. Op identity post-fusion: how do downstream consumers identify ops?

Today consumers pattern-match on `Op` (autograd, executor, op_short_name, op_key, dispatch wrappers, rule matchers). Post-T2, `Op` no longer has fused variants; consumers must match on `NodeKind::Fused { id, .. }` and look up the registry entry.

There's a subtle question: do consumers ever need to ask "what kind of fused op is this?" by id-comparison (`if id == SOFTMAX_LAST_DIM_ID`), or do they always go through the registry's polymorphism (`registry.get(id).backward(...)`)? The design doc leans on the latter, but autograd's "I need to emit a specific backward" behavior is the former.

**Threads affected:** T2 (identity model), T3 (autograd dispatch), T6 (rule matchers).

**Recommendation:** mixed. Polymorphism for runtime dispatch (executor); id-comparison for autograd's symbolic backward emission and for rule pattern-matching. Both are legitimate; the design doc should make it explicit.

### Q-D. fuel-core-types as the home of cross-cutting types

Several types have ambiguous homes:
- `OpKind` (T4) lives in fuel-core-types/dispatch.rs.
- `Storage`, `BackendCapabilities` (T1) live in fuel-core-types/backend.rs.
- `KernelRef`, `OpParams`, `KernelBindingTable` (T8) live in fuel-storage.
- `Op` (T2/T6) lives in fuel-graph.

Phase 7.5 work item E proposes fissioning fuel-core into fuel-tensor / fuel-autograd / fuel-formats / fuel-loaders. Where will the cross-cutting registry types end up? T2's investigation answered "fuel-graph for the metadata, fuel-storage for the BackendImpl" — but that's only Phase 7.6's split. The bigger question (where do `Storage`, `OpKind`, `Op` ultimately live after the fission?) isn't answered.

**Threads affected:** T1, T2, T4, T8, plus Phase 7.5 work item E.

**Cost of leaving open:** every new cross-cutting type creates a re-home decision later. T2's registry will need re-homing during the fission.

### Q-E. Sequencing: foundation-first vs feature-first

The recurring pattern in 2026-05: start a feature, find a foundation gap, pause for foundation, resume feature, find another foundation gap, pause again. CUDA Tier 1 was started, paused for SoftmaxLastDim foundation. Phase 7.5 work item B was started, paused for storage unification. Phase 7.6 was about to start, paused for this audit.

This pattern has been productive — each pause has produced a real foundation improvement (PR 3 rule registry, storage substrate, this audit). But it's unsustainable as a deliberate strategy: every feature would queue behind every foundation.

**Threads affected:** all 8.

**The implicit decision:** whether to land remaining T2 / T3 / T4 cleanups before resuming Tier 1 (T5) or any other feature work. Either answer is defensible; the choice affects the next 6-8 weeks of sequencing.

---

## 5. Sequencing recommendation

The concrete sequence below is shaped by two principles:
1. **Foundation-first only when a feature would surface foundation work mid-flight.** When the foundation isn't blocking, ship the feature.
2. **Each phase ends at a stable, shippable, tested state.** No half-migrated states left in the trunk for more than a week.

### Recommended order

**Phase A (1 session, ~1 day): close out Q-A.**
Decide whether the binding table moves to planning-time-catalog or stays as runtime-lookup. This is the single decision that affects the most other threads (T2, T4, T5, T6, T8). The decision can be quick if framed clearly; once made, the answer drives T2's design choices.

**Phase B (1 week): finish Phase 7.6 design v2.**
Update `docs/fused-op-registry.md` to v2 with: cost-estimate shape locked (static-only), naming locked (post-Q-A), registry crate location locked (fuel-graph + fuel-storage split), Conv2D/QMatMul classification confirmed (registry). Resolve Q-C (op identity post-fusion). Defer Q-B and Q-D explicitly to follow-up phases.

**Phase C (2-3 weeks): Phase 7.6 implementation, narrow scope.**
Per the existing 11-step migration plan, with the Phase A/B decisions baked in. Each fused-op migration is its own commit. Step 1-3 ships the proof-of-concept (skeleton + parallel field + SoftmaxLastDim end-to-end). Steps 4-9 finish the migration.

**Phase D (1 week): Phase 7.6B — extend Judge to profile fused ops.**
After T2 ships, T4's `OpKind` needs to grow (or a `FusedOpId` axis added). Bump `ProfileReport` schema version. Add Judge probe sites for the 13-14 fused ops. ~3-5 days.

**Phase E (parallel with above, 2 weeks): resume CUDA Tier 1 fanout (T5).**
Tier 1 is mechanical. It can run in parallel with T2 implementation if Tier 2 ops aren't started until T2 finishes. Tier 1 ops register through the existing binding table — that surface stays stable until T2 step 10.

**Phase F (defer): T3 (autograd GradientRule migration).**
Best done after T2 lands so rule signatures don't change mid-migration. Sized at ~1-2 weeks once started. Not blocking anything else; can wait.

**Phase G (defer): cost-aware scheduler (Phase 4) consuming T2 + T4 cost surfaces.**
Sized at ~3-4 weeks. Genuinely Phase 4 territory. After T2 + T4 (Phase D) ship, the scheduler can be designed and built against a stable substrate.

**Phase H (defer): fuel-core fission (Phase 7.5 work item E).**
Touches every consumer. Best done after T2/T3 stabilize so the moved types don't shift again.

### What this doesn't do

- Doesn't propose a full architectural revamp. The threads are mostly mutually consistent; T2 + Q-A unblock most of the rest.
- Doesn't unify the binding table and the empirical Judge into a single "kernel catalog + profile data" abstraction. That unification is appealing but bigger; defer until after T2/T4 stabilize and a real consumer (Phase 4 scheduler) needs it.
- Doesn't reopen settled phases (Phase A storage substrate, Phase C CPU op surface, PR 3 rule registry, layout-on-Node).

### Estimated total

~6-8 weeks of dedicated work to reach Phase E completion. Phase F and G are independent follow-ons; Phase H is its own multi-week effort.

---

## 6. Decision points

This audit produces no automatic next step. The user decides:

1. **Q-A** (binding-table layer): catalog-only or runtime-lookup? *Most leverage; affects most threads.*
2. **Q-C** (op identity post-fusion): polymorphism + id-comparison both, or polymorphism only?
3. **Phase B scope**: redo design doc to v2 with the Q-A/C decisions, or proceed to implementation against the current design?
4. **Phase E timing**: resume CUDA Tier 1 in parallel with Phase 7.6, or after?
5. **Q-B and Q-D** can stay deferred without losing forward progress.

If the user picks "Q-A unresolved, proceed narrow scope," that's also defensible — Phase 7.6 narrow can ship without Q-A, leaving the binding-table-layer-cleanup as future work. The cost is one more architectural debt entry.

---

## Appendix: what's NOT in flux

For symmetry, the parts of fuel that are stable and don't appear in the eight threads above:

- Backend extensibility model (Cargo features only, no DLL plug-ins, locked design per `project_backend_design_principles`).
- Hardware FFI layering (vulkane, baracuda, aocl-blas, onemkl outside fuel; per same memory).
- DType set (f32, f64, bf16, f16, plus 11 GGML quant types in storage substrate).
- The lazy-execution model (Phase 6, shipped, stable).
- Multi-backend Router (`fuel-graph-router`, shipped).
- KVCache park / unpark, ResidencyFile, weight tiering const_pool LRU (Phase 6 / P5 step 2, shipped).
- Phase 6c CUDA depth wins (rmsnorm/layernorm shared-mem fix, matmul stride matcher fix, depthwise via cuDNN).
- Backend-agnostic core refactor (15-step plan, complete 2026-05-01).

These are mentioned to keep the scope of this audit honest: most of fuel is stable. The eight threads above are the live work.
