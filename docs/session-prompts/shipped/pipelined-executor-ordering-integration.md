# Session prompt — PipelinedExecutor ordering integration

## What this session is for

Switch `PipelinedExecutor::realize` + `realize_many` from raw
`topo_order_multi` to `execution_plan` (which integrates
`derive_ordering`'s pre-run pinning), and auto-invoke
`insert_safety_copies` before the plan walk. Closes the in-place
ops correctness gap created when Phases 1-5 shipped 2026-05-30 —
without this change, in-place ops on autograd-tracked tensors
through PipelinedExecutor can silently produce wrong gradients.

This is a Phase 7.6 step 9c parity gap that the original audit
didn't surface because at audit time (2026-05-19) the only
destructive ops were `Op::Release` / `Op::Move` / `Op::WriteSlice`
/ `Op::ZeroFill`, where ordering naturally falls out of data flow
(Release is a sink, WriteSlice's dest is the output node, ZeroFill
always pairs with Op::Alloc). The 5 unary in-place activations +
`Op::Fused(INPLACE_AFFINE, _)` shipped 2026-05-30 are the first
destructive variants where a target has multiple legitimate
consumers — and that's where the missing ordering-edge pinning
bites.

## The architectural gap (concrete)

PipelinedExecutor's compile loop calls `topo_order_multi` directly
(`fuel-storage/src/pipelined.rs:339-352`). The executor walks that
order and uses `WorkItem.destructive_input` to evict the destroyed
input from the cache AFTER the destructive op runs (line 373).
What it DOESN'T do:

1. Run `derive_ordering` to insert ordering edges that pin the
   destructive op AFTER non-destructive readers of its target.
2. Run `insert_safety_copies` to handle the residual-connection
   cycle case.

Concrete failure mode (would silently produce wrong gradients):

```rust
let x = Tensor::from_f32(...);
let y = x.relu_inplace();
let loss = y.sum_all();
let grads = loss.backward();  // emits Op::Step(x) — reads x
let grad_x = grads.get(&x);
let value = grad_x.realize_through_pipelined();
// PipelinedExecutor topo-sorts x → {Step(x), ReluInplace(x)}
// arbitrarily. If ReluInplace runs first, Step reads mutated bytes →
// wrong gradient. Today: silent corruption. Post-this-session: correct.
```

The legacy `GraphExecutor<B>` already uses `execution_plan` (verified
in `fuel-graph-executor/src/lib.rs:948,990,1040`), so Phase 4's
view-aware ordering is live there. PipelinedExecutor is the gap.

## The 10-LOC change (the easy part)

`fuel-storage/src/pipelined.rs::PipelinedExecutor::realize` (around
line 339):

```rust
// Before:
let g = graph.read().map_err(|_| poisoned("graph lock"))?;
let effective_roots = extend_with_side_effect_roots(&g, &[target]);
let order = if effective_roots.len() == 1 {
    topo_order(&g, target)
} else {
    topo_order_multi(&g, &effective_roots)
};

// After:
{
    let mut g = graph.write().map_err(|_| poisoned("graph lock"))?;
    let effective_roots = extend_with_side_effect_roots(&g, &[target]);
    fuel_graph::opt::insert_safety_copies(&mut g, &effective_roots);
}
let g = graph.read().map_err(|_| poisoned("graph lock"))?;
let effective_roots = extend_with_side_effect_roots(&g, &[target]);
let order = fuel_graph::opt::execution_plan(&g, &effective_roots);
```

Same shape for `realize_many` (around line 418+).

The brief write-lock window for `insert_safety_copies` is safe
because realize is the only caller for the graph at that point.
`insert_safety_copies` is a no-op when no destructive ops exist
(no allocations, just a topo walk + scan), so it's zero-cost for
graphs without in-place ops.

## The regression sweep (the hard part)

The risk: some currently-passing tests may rely on the specific
order `topo_order_multi` produces. Switching to `execution_plan`
preserves topo order when no ordering edges exist (fast path at
line 1487), so most tests should be unaffected. But:

- Tests that exercise existing destructive ops (`Op::Release`,
  `Op::WriteSlice`, `Op::ZeroFill`) will now see ordering edges
  they didn't before. Most should still pass — derive_ordering
  pins destructive ops AFTER readers, which is correct semantics —
  but specific assertions about node visit order may need updating.
- Tests that exercise in-place ops on tape-tracked tensors will
  now BEGIN passing (they previously silently corrupted). The
  in-place-ops-with-multiple-readers regression test (see below)
  is the canary.
- The view-aware alias-set walk in `derive_ordering` adds work for
  every destructive op. Existing benchmarks may show a tiny
  regression (microseconds) on graphs with destructive ops.

Test scopes to sweep (in order of likelihood-to-surface-issues):

1. `cargo test -p fuel-storage --lib` — exercises PipelinedExecutor
   directly. Op::Alloc + Op::ZeroFill + Op::WriteSlice tests live
   here.
2. `cargo test -p fuel-storage --lib --features cuda` —
   CudaStorageBytes round-trips through PipelinedExecutor.
3. `cargo test -p fuel-storage --lib --features vulkan` —
   VulkanStorageBytes round-trips through PipelinedExecutor.
4. `cargo test -p fuel-core --lib` — Tensor::realize_f32 path uses
   PipelinedExecutor for CPU.
5. `cargo test -p fuel-core --lib --features cuda` — Tensor::
   realize_f32_cuda uses PipelinedExecutor (post Phase E.1).
6. `cargo test -p fuel-core --lib --features vulkan` — KvCache
   parity tests exercise the full pipelined path.
7. Live-GPU sweeps (`--features cuda --test '*_live'`, same for
   vulkan).

If any test fails: diagnose whether it was relying on
topo-order-specific behavior (legitimate test fix) or on the
ABSENCE of destructive-ordering pinning (the test was passing by
luck — the ordering edge now reveals a real bug).

## New regression tests to add

Two tests prove the in-place + multi-reader correctness contract:

```rust
// fuel-storage/src/tests/ (or a new file)
#[test]
fn pipelined_inplace_with_multiple_readers_orders_correctly() {
    // Graph:
    //   x = [1, -2, 3, -4]
    //   y = x.relu_inplace()
    //   step_x = Op::Step(x)    // simulates the gradient node
    //
    // Realize both y and step_x via PipelinedExecutor.
    // Expected:
    //   y      = [1, 0, 3, 0]  (post-mutation = relu(x))
    //   step_x = [1, 0, 1, 0]  (sign of ORIGINAL x; would be
    //                           [1, 0, 1, 0] either way for relu's
    //                           specific case, but the principle
    //                           generalizes to silu/tanh where
    //                           pre/post mutation differ meaningfully)
    //
    // Pre-this-session: step_x might be computed from post-mutation
    // bytes (wrong for non-relu activations).
    // Post-this-session: derive_ordering pins ReluInplace after
    // Step(x), so step_x reads pre-mutation bytes.
}

#[test]
fn pipelined_residual_connection_inserts_safety_copy() {
    // Graph:
    //   x = [1, -2, 3, -4]
    //   y = x.relu_inplace()
    //   z = y + x
    //
    // Pre-this-session: PipelinedExecutor panics with a cycle error
    // (or worse, silently corrupts depending on topo).
    // Post-this-session: insert_safety_copies inserts Op::Copy(x) →
    // x_safe, rewires Add's x input to x_safe; z = [1+1, 0+(-2),
    // 3+3, 0+(-4)] = [2, -2, 6, -4].
}
```

## Why this is its own session (not bundled with Phases 1-5)

The Phases 1-5 session (2026-05-30) shipped 7 commits:

- `dd2b7158` Phase 1 — Op IR variants + scheduler integration
- `ce928b0f` Phase 2 — Tensor::*_inplace builders
- `5b9ca5bb` Phase 3 CPU — INPLACE_AFFINE end-to-end
- `7a467218` Phase 3d CUDA — baracuda affine_inplace
- `2a985c27` Phase 3e — 5 unary activations on CPU + CUDA
- `49b01fb3` Phase 4 — view-aware derive_ordering + autograd
- `7a158a10` Phase 5 — insert_safety_copies auto-copy pass

Bundling the PipelinedExecutor integration into that session would
have:

1. Risked scope creep on an already-large session
2. Mixed "build the Op IR + safety machinery" with "wire the
   machinery into the production executor" — two distinct concerns
3. Forced the regression sweep to share attention with feature
   development

The 9c parity audit explicitly recommends "don't bundle phases."
This is closing a Phase 7.6 step 9c parity gap; it deserves its
own session.

## What ISN'T in scope for this session

- **Legacy `GraphExecutor<B>` realize call sites** — already use
  `execution_plan`. No change needed. They're scheduled for
  retirement with Phases F/G/H of the 9c migration.
- **CUDA→CUDA same-device Op::Copy registration** — needed if the
  residual-connection case ever lands a CUDA-side cycle. Today's
  CUDA in-place tests don't exercise that pattern (the pattern
  needs to come from autograd through in-place on a tape-tracked
  tensor + Op::Copy fallback). Defer until a CUDA consumer
  materializes.
- **View-mediated cycle resolution** — `insert_safety_copies`
  handles direct-reader cycles only. View-mediated conflicts (a
  reader of `transpose(x)` while `x.relu_inplace()` mutates x)
  still hit the cycle panic at execution_plan time. Own session
  (would require re-deriving the view from the snapshot for each
  conflicting reader).

## Dependencies + references

- Reads from today (2026-05-30): commits `dd2b7158`, `ce928b0f`,
  `5b9ca5bb`, `7a467218`, `2a985c27`, `49b01fb3`, `7a158a10`.
- Background: `project_phase_7_6_step_9c_parity_audit` memory.
- Architectural framing: `project_inplace_ops_complete` memory.

## Scope estimate

- Core change: ~10 LOC across 2 call sites in `pipelined.rs`
- Regression test additions: ~80 LOC for the 2 tests above
- Test fixes (if any): depends on what the sweep surfaces;
  likely 0-3 tests need order-assertion updates
- Total: 1 focused session, 1 commit
