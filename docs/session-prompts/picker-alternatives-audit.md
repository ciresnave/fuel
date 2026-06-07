# Session prompt — Picker alternatives audit

## What this session is for

**Scoping session, NOT implementation.** Phase 6b's promise —
empirical per-op dispatch — is split into two roles:

- The **Judge** rates ops (measures op × dtype × size × backend
  timings, produces a `DispatchTable` of ratings).
- The **picker** (`compile_node` / `resolve_kernel` chain) consumes
  those ratings (along with `TolerancePolicy`, `KernelCaps`, and
  static precision/cost metadata) to choose which alternative the
  executor uses.

The Judge side is partly built (per
`project_judge_coverage_expansion_shipped`). Phase 7.6 step 9b
ships multi-impl alternatives in the binding table +
`resolve_kernel`. What's unclear: whether the picker actually
consults Judge ratings today, or whether it falls back to
first-registered / static precision-cost defaults. This audit
traces the picker code path to answer that question and surfaces
the decision points needed to wire Judge ratings into the picker
(if not already wired).

This session produces:

1. A concrete audit of where multi-backend coverage exists today
   (op × dtype × backend matrix from `KernelBindingTable`).
2. A trace through the picker code answering: when there are N
   alternatives at a binding-table key, what determines which gets
   chosen, and where does Judge fit in (today + as architected)?
3. A decision-points doc enumerating the architectural choices
   future sessions need to make (destructive ops, cross-device
   dispatch, fallback semantics, etc.).
4. A scope estimate broken into follow-up sessions, each its own
   coherent unit.

**Do NOT implement.** The temptation is "I see how to wire it,
let me just do it." Resist. The decision points here have downstream
consequences (Phase 7.6 step 9c integration; the Judge profiling
loop's destructive-op semantics; cross-device fallback policy)
that deserve dedicated attention.

## Background — what already exists

Read these in order:

- `project_phase6b_probe_judge_dispatch.md` — the original Phase 6b
  framing (probe → judge → dispatch table → router).
- `project_judge_coverage_expansion_shipped.md` — Judge coverage
  expansion across 28 OpKinds × 5 families.
- `project_phase_7_6_step_4_in_progress.md` — Step 9a (multi-impl
  alternatives in KernelBindingTable) + Step 9b (NodeKernelBinding +
  compile_plan + resolve_kernel + TolerancePolicy).
- `project_phase_7_6_step_9c_parity_audit.md` — the broader
  executor migration; this audit is downstream.
- `project_pipelined_executor_ordering_gap.md` — the in-flight
  in-place ops integration that's a prerequisite for some Judge
  destructive-op work.

## Audit step 1 — where IS multi-backend coverage today?

Concrete: enumerate the binding table at process start, group by
`(op_kind, dtypes)`, and report any key with `>1` registered
alternative.

```rust
// Pseudocode for a one-shot binary that does the audit:
let table = global_bindings();
let mut multi: Vec<(OpKind, Vec<DType>, Vec<BackendId>)> = Vec::new();
for (op_kind, dtypes) in table.all_keys() {
    let alternatives = table.lookup_alternatives_all_backends(op_kind, &dtypes);
    if alternatives.len() > 1 {
        multi.push((op_kind, dtypes, alternatives.iter().map(|a| a.backend).collect()));
    }
}
println!("{} multi-backend keys:", multi.len());
for (op, dts, backends) in multi {
    println!("  {op:?} {dts:?} → {backends:?}");
}
```

**Expected coverage hot spots** (guess; audit will confirm):

- `OpKind::MatMul` × {f32, f16, bf16} × {Cpu, Cuda, Vulkan, Mkl, Aocl}
- Various unary on f32 × {Cpu, Cuda, Vulkan}
- `OpKind::SoftmaxLastDim`, `RmsNormLastDim`, `LayerNormLastDim` on f32 × {Cpu, Cuda, Vulkan}
- `OpKind::Affine` on f32 × {Cpu, Cuda, Vulkan, baracuda variants}
- `OpKind::Cast` pairs × multiple backends
- `OpKind::FusedLinear` × dtypes × backends

**Sparse coverage** (guess): in-place ops (only Cpu + Cuda for now),
reductions (Cpu + Cuda), QMatMul (registered widely but tied to
specific QuantTypes).

Deliverable: a table in the audit doc enumerating every multi-backend
key. This becomes the working set for "which dispatches would benefit
from Judge picking?"

## Audit step 2 — how does the picker currently choose?

Trace from `PipelinedExecutor::compile_one` → `compile_node` →
`resolve_kernel` (or whatever the live names are post-step-9b). For
each step:

- Read the source. Document the current ranking criterion (first-
  registered? precision-then-cost? caps-filtering? Judge consultation?).
- Identify where Judge data could plug in. Is there a `DispatchTable`
  lookup somewhere that's currently bypassed?
- Document the `TolerancePolicy` shape — it controls precision
  filtering at the picker, which is one input to ranking.

Look specifically at:
- `fuel-storage/src/compiled.rs` — `compile_node` entry point
- `fuel-storage/src/kernel.rs` — `KernelBindingTable::lookup_alternatives`
- `fuel-storage/src/plan.rs` — `resolve_kernel` (if it exists per
  the 9b memory)
- `fuel-core/src/judge.rs` — the Judge's `DispatchTable` output

Deliverable: a flow diagram (text is fine) showing the picker's
current decision points and where Judge does/doesn't enter today.

## Decision points the audit must surface

Each of these has downstream session implications. The audit
ENUMERATES them; later sessions DECIDE.

1. **Judge consultation policy.** When the picker has N alternatives,
   does it (a) always consult Judge, (b) consult only when Judge has
   data for this (op, dtype, size class), (c) consult only when
   alternatives' static cost estimates are within some band? The
   choice affects cold-start behavior and the persistence story.

2. **Destructive ops profiling.** The Judge loop runs each candidate
   N times against the same input to measure timing. For destructive
   ops, iteration 2+ sees post-mutation input → timings meaningless.
   Options:
   - Skip destructive ops from Judge enumeration (no picking; first-
     registered wins). Simplest.
   - Clone target before each iteration. Correct but adds memcpy
     cost to the measurement.
   - Skip destructive ops AND emit a clear note in the dispatch
     table that "no Judge data → using fallback." Future-proof.

3. **Cross-device dispatch.** When Judge data shows Vulkan is fastest
   for this op but the input lives on CUDA, what happens? Inject
   `Op::Copy { target: Vulkan }`? Run on CUDA and ignore the
   measurement? Skip Judge's cross-device suggestions entirely?
   Couples to bridge-retirement Phase 4 (cross-device dispatch).

4. **Tolerance budget integration.** Step 9b's `TolerancePolicy`
   filters alternatives by precision. Does Judge's picker run BEFORE
   the policy filter (pick fastest, then drop if precision fails) or
   AFTER (pick fastest among precision-acceptable)? AFTER is the
   architecturally clean default; surface this explicitly.

5. **Fallback when Judge has no data.** Cold start, new shape,
   benchmark-skip flag. Options: first-registered, static-cost rank,
   binding-table caps-based default. Each has trade-offs.

6. **Persistence + invalidation.** Judge reports persist to disk
   (per `PROFILE_REPORT_VERSION`). When does the picker invalidate
   stale data? After driver update? After backend kernel registration
   change? Never (manual flush)?

7. **Measurement frequency.** Profile per shape, per shape-class
   (small/medium/large), or per-op-only? Affects warmup cost vs
   pick fidelity.

## What ISN'T in scope for this session

- Writing the picker code. That's session 2+.
- Building infrastructure for any of the decision points. Audit only.
- Cross-backend kernel writing (e.g., "we should add Vulkan
  InplaceAffine"). That's its own kernel-side audit.
- Touching `populate_dispatch_table` or any Judge runner.

## Deliverables

1. **`docs/session-prompts/judge-alternatives-picking-audit-results.md`**
   (or similar) containing:
   - Multi-backend coverage table (from step 1)
   - Picker flow diagram (from step 2)
   - Decision-points list with options + recommendation per point
   - Scope estimate broken into per-decision follow-up sessions
2. Optional: a one-shot binary (or doc-test) that prints the
   multi-backend coverage table, so future sessions can re-run it
   when coverage changes.
3. Memory updates: new `project_judge_alternatives_audit.md`
   capturing the audit's findings + the recommended sequence.

## Scope estimate

- Step 1 (coverage audit): ~30-60 min, mostly reading binding-table
  registrations + tabulation.
- Step 2 (picker trace): ~60-90 min, reading compile_node /
  resolve_kernel call chain.
- Step 3 (decision-points doc): ~60-90 min, writing.
- Total: 1 focused session, no commits (just docs + memory).

## Why audit-first

Implementation pressure says "just wire Judge into the picker." Two
reasons that's wrong here:

1. **The destructive-ops semantics couple to Phase 4/5 in-place ops
   work** (currently in-flight via the PipelinedExecutor ordering
   integration session). Wiring Judge before that lands risks making
   architectural commitments that conflict.
2. **The picker's existing behavior may already do half of what we
   want** (Step 9b shipped `TolerancePolicy` + `resolve_kernel`; I
   haven't traced whether Judge consultation is wired in or just
   stubbed). Implementing without an audit risks duplicate
   infrastructure.

The audit produces a clear "do A first, then B, then C" sequence
that subsequent sessions can pick up without re-discovering the
landscape.
