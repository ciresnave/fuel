# Session prompt — Phase 1.1: `AlternativeSet` + filter-chain infrastructure

## What this session is for

Build the **types and filter-application machinery** the
optimizer ranker (Picker 1) will consume. Pure infrastructure —
no consumers, no SystemTopology integration, no executor change,
no graph annotation. This session ships the substrate everything
else in Phase 1 plugs into.

This is **Phase 1.1** of the picker-work arc. Phase 1 overview:

| Sub-phase | Scope                                                                    |
|-----------|--------------------------------------------------------------------------|
| **1.1**   | `AlternativeSet` + filter trait + filter-chain pipeline (this session)   |
| 1.2       | Candidate enumeration via SystemTopology (cross co-located backends)     |
| 1.3       | Precision + tolerance hard filters (using existing `PrecisionGuarantee`) |
| 1.4       | Cost ranking — Layer-1 static composition (defer Judge to its own phase) |
| 1.5       | `ExecutionPlan` carries `AlternativeSet`s; integration with `compile_plan` |

Subsequent phases:

- **Phase 2** — `Op::Copy` / `Op::Move` / layout-fixup insertion in the optimizer
- **Phase 3** — Judge consultation in the ranker (Layer-2 refinement)
- **Phase 4** — Executor consumes `AlternativeSet` + dispatch-chunk generation check
- **Phase 5** — Runtime selector (Picker 2) for layer-3 telemetry

This session does Phase 1.1 only. Resist scope creep.

## Why 1.1 first, alone

The Picker 1 / Picker 2 split that fell out of the architecture
conversation needs concrete data structures before any meaningful
work can land:

- **What's an "alternative"?** A `(KernelRef, KernelCaps, BackendId,
  DeviceLocation, PrecisionGuarantee, CostEstimate, coupling info)`
  bundle. Today's `BindingEntry` is the per-key cousin; alternatives
  are the per-decision-point set that survives filtering + ranking.
- **What's an "alternative set"?** A bounded collection at one
  decision point, preserving up to N (default 3) entries after
  filtering and cost ranking.
- **How do filters compose?** Hard filters (user precision floor)
  may filter to zero — that surfaces as a hard error. Soft filters
  (caps preference, empirical refinement) may only filter if at
  least N≥1 remains.

These have to exist as types before Phase 1.2 can enumerate
candidates into them, before 1.3 can implement precision/tolerance
filters, before 1.4 can compose costs. They're the substrate.

## Background reading

- [`judge-alternatives-picking-audit-results.md`](
  ./judge-alternatives-picking-audit-results.md) — the audit that
  motivated this whole arc; specifically the "two pickers" split.
- `docs/architecture/04-optimization.md` — per-decision-point
  alternatives, precision-filter-before-cost-rank, top-N preservation
  (default 3), coupling-via-conditional-cost-adjustments.
- `docs/architecture/07-tolerance.md` — tolerance budget shape
  (what hard filters consume).
- `fuel-dispatch/src/kernel.rs` — current `BindingEntry`,
  `KernelCaps`, `KernelRef`, `OpParams`, `CostFn`. The new
  `Candidate` type composes from these.
- `fuel-dispatch/src/fused.rs` — `PrecisionGuarantee`,
  `CostEstimate`. The bounds the ranker filters/ranks against.
- `fuel-dispatch/src/plan.rs` — the doomed-in-Phase-1.5 picker.
  `TolerancePolicy` enum dies; `NodeKernelBinding` reshapes.
- Memory: `project_dispatch_crate_extracted` (most recent),
  `project_system_topology_shipped`,
  `project_judge_alternatives_audit`.

## Architectural decisions to resolve

These are TDPs the session must answer up-front. Their resolution
shapes Phase 1.2+.

### TDP-1.1-A: Where does the ranker live?

The optimizer ranker (Picker 1) sits between the graph optimizer
(which decides which subgraph rewrites to apply) and the executor
(which runs the resolved kernel). Three candidate homes:

- **A) `fuel-dispatch::ranker`** (sibling of `plan.rs`).
  Pragmatic — the binding-table, KernelRef typedef, and current
  picker (`resolve_kernel`) all already live here. Same dep graph
  as today.
- **B) `fuel-graph::ranker`** — graph-side annotation lives near
  the graph. But fuel-graph doesn't depend on fuel-dispatch
  (would cycle if it did, since fuel-dispatch depends on
  fuel-graph for `Op`/`NodeId`). To put alternatives on the
  graph, `KernelRef` + `KernelCaps` + `OpParams` would need to
  move to `fuel-core-types` first.
- **C) New `fuel-optimizer` crate** between fuel-graph and
  fuel-dispatch. Cleaner separation of concerns ("the layer that
  picks") but adds workspace surface.

**Recommendation: A** for Phase 1.1. The types live next to
their substrate (KernelRef + BindingEntry). If Phase 1.5 needs
graph-side annotation, that's where we revisit (likely by
moving the type definitions down to fuel-core-types, leaving the
machinery in fuel-dispatch). Surface this as a "watch this
boundary" item but don't pre-emptively restructure.

### TDP-1.1-B: `AlternativeFilter` trait shape

Two competing pulls:

- **Trait-object dynamism** — `Vec<Box<dyn AlternativeFilter>>`.
  Filters compose at runtime, easy to add new ones. Allocation cost
  per call.
- **Static generic** — filters parameterized at compile time. Fast
  but the chain composition becomes ugly (nested generics).

Sketch (trait-object form):

```rust
pub trait AlternativeFilter: Send + Sync {
    /// Apply this filter, returning indices into `alts` to keep.
    fn filter(&self, alts: &[Candidate], ctx: &FilterContext) -> Vec<usize>;
    fn classification(&self) -> FilterClass;
    fn name(&self) -> &'static str; // for diagnostics
}

pub enum FilterClass {
    /// May filter to zero. Caller propagates the empty result as
    /// an error (user precision floor unmet → fail, don't
    /// silently substitute).
    Hard,
    /// Must leave at least `min_remaining` alternatives. If the
    /// filter would drop below this threshold, it's a no-op for
    /// this call (logged as "filter saturated").
    Soft { min_remaining: usize },
}
```

**Recommendation: trait object.** The cost is one heap allocation
per filter per decision point at plan time; plan time isn't a hot
loop in any meaningful sense. The flexibility lets Phase 3 add
a Judge-driven filter without re-instantiating every consumer's
generic chain.

### TDP-1.1-C: `Candidate` shape — copy from `BindingEntry`?

`BindingEntry` is what the binding table stores: `(KernelRef,
KernelCaps, PrecisionGuarantee, CostFn)`. A `Candidate` needs
more — placement (which backend × device), the input shapes the
cost was computed against, and possibly coupling info (cost
adjustments contingent on adjacent placements).

Sketch:

```rust
pub struct Candidate {
    // Identity
    pub kernel: KernelRef,
    pub caps: KernelCaps,
    pub backend: BackendId,
    pub device: DeviceLocation,
    // Static metadata
    pub precision: PrecisionGuarantee,
    pub static_cost: CostEstimate,
    // Phase 1.2+ populates these
    pub op_params: OpParams,
    pub coupling: Vec<CouplingAdjustment>,  // empty in 1.1
}
```

`CouplingAdjustment` can be a stub in 1.1 (empty `Vec`); Phase 2
populates it when transfer-op cost coupling lands.

### TDP-1.1-D: `AlternativeSet` shape — bare `Vec` or newtype?

A newtype gives us a place to hang invariants:

```rust
pub struct AlternativeSet {
    candidates: SmallVec<[Candidate; 4]>,
    max_n: usize,                          // default 3 per arch §04
}

impl AlternativeSet {
    pub fn from_candidates(c: Vec<Candidate>, max_n: usize) -> Self;
    pub fn apply_filter(&mut self, f: &dyn AlternativeFilter, ctx: &FilterContext);
    pub fn rank_by_cost(&mut self);        // Phase 1.4 fills body
    pub fn truncate_to_top_n(&mut self);
    pub fn winner(&self) -> Option<&Candidate>;  // for Phase 4's executor
    pub fn alternatives(&self) -> &[Candidate];  // for the runtime selector (Picker 2)
}
```

**Recommendation:** newtype. Future work (cost-rank, top-N
truncation, coupling resolution) wants a focused API surface.

### TDP-1.1-E: `FilterContext` — what does the chain see?

The hard filter "user requires bit-stable" needs the user's
tolerance setting. The soft filter "prefer strided_input when
input is non-contiguous" needs the input's layout. The (future)
Judge filter needs the op + dtypes + size class.

Sketch:

```rust
pub struct FilterContext<'a> {
    pub op: OpKind,
    pub dtypes: &'a [DType],
    pub input_layouts: &'a [Layout],       // for caps decisions
    pub tolerance: Option<TolerancePolicy>, // user's per-call floor
    // Phase 3+ adds: judge: Option<&'a dyn JudgeOracle>
}
```

Most filters will only read a few fields. Wide context is fine
at plan time.

### TDP-1.1-F: Default filter chain

Document but don't necessarily build the default chain in 1.1.
Probably ends up as:

1. **Hard**: precision floor (if user set one) — `PrecisionFloor`
2. **Hard**: tolerance budget — `ToleranceBudget`
3. **Soft**: prefer `caps.strided_input` if input non-contiguous —
   `StridedInputPreference` (min_remaining=1)
4. **Soft**: bit-stable preference — `BitStablePreference`
   (min_remaining=1)
5. **(Phase 3)** Soft: Judge empirical — `JudgeFastest`
   (min_remaining=1)

Phase 1.1 ships the chain infrastructure. The actual filters
land in 1.3 (precision/tolerance hard) and naturally in later
phases. **Don't build a filter for which the data doesn't exist yet.**

## Scope of work

### Step 1 — types

New module `fuel-dispatch/src/ranker/` (or `fuel-dispatch/src/picker/`):

- `mod.rs` — re-exports
- `candidate.rs` — `Candidate`, `CouplingAdjustment` (stub)
- `alternative_set.rs` — `AlternativeSet`
- `filter.rs` — `AlternativeFilter` trait, `FilterClass`,
  `FilterContext`
- `chain.rs` — the application pipeline (`apply_filter_chain`)

Public surface:

```rust
pub use candidate::{Candidate, CouplingAdjustment};
pub use alternative_set::{AlternativeSet, DEFAULT_MAX_N};
pub use filter::{AlternativeFilter, FilterClass, FilterContext};
pub use chain::apply_filter_chain;
```

### Step 2 — chain semantics

The pipeline:

```rust
pub fn apply_filter_chain(
    set: &mut AlternativeSet,
    filters: &[Box<dyn AlternativeFilter>],
    ctx: &FilterContext,
) -> Result<()> {
    for filter in filters {
        let keep_indices = filter.filter(set.alternatives(), ctx);
        let kept = keep_indices.len();
        match filter.classification() {
            FilterClass::Hard if kept == 0 => {
                return Err(Error::FilterRejected {
                    filter: filter.name(),
                    ctx_summary: ctx.summary(),
                }.bt());
            }
            FilterClass::Soft { min_remaining } if kept < min_remaining => {
                // Filter would over-restrict; skip it as a no-op.
                tracing::debug!(
                    filter = filter.name(),
                    kept,
                    min_remaining,
                    "soft filter saturated; skipping",
                );
                continue;
            }
            _ => {}
        }
        set.retain_indices(&keep_indices);
    }
    Ok(())
}
```

Decision visible: hard filters surface emptyness as an error;
soft filters silently skip when they would over-filter. Both
log enough to diagnose at the call site.

### Step 3 — tests

Inline unit tests in each module:

- `candidate.rs` — construction smoke test; `CouplingAdjustment`
  default is empty.
- `alternative_set.rs`:
  - empty set has no winner
  - `truncate_to_top_n` respects `max_n`
  - `alternatives()` returns full list before truncation
- `filter.rs` — trait object dyn dispatch works (smoke test with
  a `MockFilter` impl).
- `chain.rs` — the meat:
  - empty filter list → all candidates pass through
  - single hard filter to zero → returns `FilterRejected`
  - single soft filter that would drop to zero → no-op, all
    candidates remain
  - single soft filter with `min_remaining=2` that would leave 1
    → no-op
  - chain of hard + soft: hard fails early; soft skipped if it
    saturates after the hard pass
  - filter ORDER matters: a hard filter after a soft filter only
    sees the soft-filtered set

### Step 4 — `Error::FilterRejected` variant

Add to `fuel-core-types::Error`:

```rust
FilterRejected {
    filter: &'static str,
    ctx_summary: String,
    available_alternatives: usize,
}
```

Plus a `FilterContext::summary()` helper that produces a short
diagnostic string (op kind, dtypes, device).

### Step 5 — module wired into `fuel-dispatch/src/lib.rs`

Add `pub mod ranker;` and re-export the public surface alongside
the existing `compile_plan` / `resolve_kernel` re-exports.

### Step 6 — memory + docs

- New memory entry
  `project_phase_1_1_alternative_set_filter_chain_shipped.md` —
  captures the TDP resolutions + the public surface + what Phase
  1.2 picks up.
- `MEMORY.md` index entry.
- Note in the entry which `AlternativeFilter` impls are stubbed
  vs implemented (1.1 ships none of them; 1.3 lands hard
  precision/tolerance; later phases land the rest).

## What's NOT in scope

- **Any `AlternativeFilter` implementation.** The trait + chain
  ship; filters land in 1.3+. (Tests use mock impls.)
- **SystemTopology integration.** Phase 1.2.
- **Candidate enumeration logic.** Phase 1.2.
- **Cost ranking / Layer-1 composition.** Phase 1.4.
- **Cost ranking with Judge.** Phase 3.
- **`ExecutionPlan` / `compile_plan` integration.** Phase 1.5.
- **Executor changes.** Phase 4.
- **Op::Copy / layout-fixup insertion.** Phase 2.
- **`TolerancePolicy` removal.** Phase 1.5 (when `resolve_kernel`
  retires alongside).
- **`KernelRef` migration to fuel-core-types.** Defer; if Phase
  1.5 needs it for graph-side annotation, do it then.
- **Runtime selector (Picker 2).** Phase 5.
- **Any change to `compile_node` / `lookup_with_caps` callers.**
- **Cross-project changes** (vulkane / lightbulb / baracuda).
  Per the feedback memory.

## Deliverables

1. New module `fuel-dispatch/src/ranker/` with `Candidate`,
   `AlternativeSet`, `AlternativeFilter`, `FilterClass`,
   `FilterContext`, `apply_filter_chain`.
2. `Error::FilterRejected` variant in fuel-core-types.
3. Module re-exported from `fuel-dispatch/src/lib.rs`.
4. Inline unit tests covering the chain semantics in Step 3.
5. Memory entry + index update.
6. Workspace `cargo check` + `cargo test` clean (CPU; CUDA/Vulkan
   tests unaffected by infrastructure-only changes but should be
   verified clean).

## Scope estimate

- Step 1 (types): ~30 min
- Step 2 (chain): ~30 min
- Step 3 (tests): ~60 min — the chain semantics deserve thorough
  coverage
- Step 4 (Error variant): ~15 min
- Step 5 (lib.rs): ~10 min
- Step 6 (memory + docs): ~30 min

**Total: 1 focused session, 1–3 commits.** Mechanically simple;
the substance is the TDP resolutions surfaced up-front.

## Why this session, this scope, this order

Phase 1 has been the architectural-discussion focus for the
last several sessions. Phase 1.1 ships the *types* that
everything subsequent — candidate enumeration (1.2), precision
filtering (1.3), cost composition (1.4), plan integration (1.5),
executor consumption (Phase 4) — operates against. Doing the
types alone, with no consumer, forces the API to be shaped by
the predicates rather than by the first caller's accidents.
This is the same session-split discipline that worked for
SystemTopology.

The temptation will be to "just sketch in one filter while
we're here." **Don't.** Filters land in 1.3 (precision/tolerance)
where their semantics get real review. A mock filter for tests is
fine; a production filter is the next session's call.

## Pointers

- Audit doc: [`judge-alternatives-picking-audit-results.md`](
  ./judge-alternatives-picking-audit-results.md)
- Architecture: [`docs/architecture/04-optimization.md`](
  ../architecture/04-optimization.md)
- Current picker (doomed in Phase 1.5):
  `fuel-dispatch/src/plan.rs::{compile_plan, resolve_kernel,
  TolerancePolicy, NodeKernelBinding}`
- Binding-entry shape (input to Candidate):
  `fuel-dispatch/src/kernel.rs::BindingEntry`
- Precision + cost types: `fuel-dispatch/src/fused.rs`
- Memory: `project_dispatch_crate_extracted`,
  `project_system_topology_shipped`,
  `project_judge_alternatives_audit`
