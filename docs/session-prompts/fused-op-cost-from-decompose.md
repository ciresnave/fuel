# Fused-op Layer-1 cost from its decomposition

**Status:** DESIGN SPEC (2026-07-01, for review). No code yet. Proposed by CireSnave:
*"could the combination of [the base-model ops] that makes up the fused version of a
kernel be used to estimate the cost of that fused kernel until the Judge can verify it
during real runs?"* — answer: yes, and it should be the **default** Layer-1 cost for any
fused/synthesized op.

**Source-of-truth:** [`docs/architecture/04-optimization.md`](../architecture/04-optimization.md)
§"Three layers of cost data" (Layer-1 static / Layer-2 Judge). This spec adds a Layer-1
*default* for the fused-op case and is subordinate to that section.

---

## 0. Motivation — the sentinel is the dangerous value

A fused or imported kernel with no declared cost is registered with `fused_unknown_cost`
(`fuel-dispatch/src/fkc/register.rs:116`) → `CostEstimate::default()` → **zero**. Zero cost
is exactly the failure mode the FKC-cost-unification Part A fixed for GPUs: a mis-priced-at-
zero candidate wins spuriously (a CPU-pinned realize spilled onto an unseeded GPU because the
GPU candidate priced at 0 — see `10-decisions-log.md` 2026-07-01). Every fused op that lacks a
measured Judge cell today carries that same landmine until the Judge profiles it.

**A cost composed from the op's own decomposition replaces zero with a real, bounded
number** — never wildly wrong, directionally safe, and available immediately (no Judge data
required). It is the bridge "until the Judge can verify it during real runs."

## 1. The idea

Every fused op **already carries a total `decompose`** (the recipe principle — fused →
equivalent primitive subgraph; mandatory, build-time, never-panic). So we already know the
exact set of primitive ops a fused kernel is equivalent to. **Compose those primitives'
Layer-1 costs to get the fused op's Layer-1 cost** — reusing the same per-node cost machinery
the optimizer already applies to any candidate path.

The ingredients all exist as of this session:
- **`decompose`** — the recipe principle; the `DecompositionMap` maps a `FusedOpId` to
  `fn(&mut Graph, NodeId, &FusedOpParams) -> NodeId` (emits the primitive subgraph). See
  `fuel-graph/src/runtime_fused.rs:181` (`decompose_region`) for the runtime form.
- **Primitive costs** — `default_cost_for_op_kind(op) -> CostFn`
  (`fuel-dispatch/src/cost.rs:806`), now contract-sourced for the migrated CPU families and
  filled onto every backend via `fill_unset_cost_for_backend`.
- **The ns fold** — `composite_ns(cost, compute_rate, mem_bandwidth)`
  (`fuel-dispatch/src/ranker/cost.rs`), throughput-aware since Part C.
- **The graph + map are in hand** — the optimizer computes fused-op costs while holding the
  `Graph` and the `DecompositionMap`, so it can invoke `decompose` into a scratch subgraph at
  cost time. (The bare `cost: fn(shapes, params, caps)` on `FusedKernelRegistry`'s
  `BackendImpl` — `fuel-dispatch/src/fused.rs:63` — has no graph; see §4 for where the fold
  actually plugs in.)

## 2. The composition — per axis (this is the whole subtlety)

Fusion's *point* is to be cheaper than its decomposition, so the fold is NOT a naive sum on
every axis:

| Axis | Fused vs Σ(primitives) | Estimate to use |
|------|------------------------|-----------------|
| **FLOPs** | ≈ equal (same arithmetic) | `Σ decompose.flops` — essentially exact for algebraic fusions |
| **bytes_moved** | fused is **less** (intermediates stay in registers/shared mem) | **tight:** the fused op's own *boundary* I/O (its declared inputs + final outputs); **loose upper bound:** `Σ decompose.bytes` |
| **kernel_overhead_ns** | fused is **less** (one launch, not N) | **one** launch overhead, not `Σ` |

**Recommended v1:** `flops = Σ decompose.flops`, `bytes_moved =` the fused op's boundary I/O
bytes (from its own input/output shapes — the intermediates are exactly what fusion elides),
`kernel_overhead_ns =` a single launch. This is a *tight* estimate and every input is already
available. The lazy variant (`Σ` on all three) is also acceptable — see §3.

Compose over the decompose subgraph with the optimizer's existing **wall-clock** fold, not a
blind serial sum: a fused kernel's decomposition is usually a data-dependent chain (serial),
but where it has independent branches, use `max(parallel) + serial_remainder` per
04-optimization §"Wall-clock, not strict-serial".

## 3. Why it is directionally safe

If we use the loose `Σ` (over-estimate), the composed cost is a **conservative upper bound**
on the true fused cost. Directionally that makes the optimizer **under-eagerly** fuse — it
misses a win until the Judge refines the number *downward* — rather than make an optimistic,
**irreversible** bad commit. "Missed win until measured" is the safe failure direction under
carry-forward placement; the Judge (Layer 2) then corrects it favorably (the real fused kernel
beats the bound, so it looks even better once measured). The tight v1 (boundary-I/O bytes, one
launch) is close to the true value and still never below zero.

## 4. Where it plugs in

`cost_from_decompose` is an **optimizer-level Layer-1 default**, computed where the `Graph` +
`DecompositionMap` are in hand — NOT inside the bare `BackendImpl::cost` fn pointer (which
lacks the graph). Two viable shapes:

- **(A) Fallback at the use site (recommended).** Wherever the optimizer reads a fused op's
  Layer-1 cost, if the registered `cost` is the `fused_unknown_cost` sentinel (identity-compare
  the fn pointer, as `fill_unset_*` already does for `unknown_cost`), compute
  `cost_from_decompose(fused_node)` instead. Zero code churn on the registry; the sentinel
  simply means "derive me from my recipe."
- **(B) Populate at registration.** When a fused op registers without a declared cost, stamp a
  cost fn that closes over its decompose. Harder — the `cost: fn` is a bare pointer that can't
  capture the map; would need the same signature-widening the Task-F trampoline needs. Prefer
  (A).

Implementation sketch of the fold (shape-only; no permanent graph mutation):
```
fn cost_from_decompose(map, prims_cost, fused_node, caps) -> CostEstimate:
    sub = map.decompose(fused_node)          # emit primitive subgraph into a scratch Graph
    flops = Σ over sub nodes of prims_cost(node.op)(node.shapes, node.dtypes, params, caps).flops
    bytes = boundary_io_bytes(fused_node)    # inputs + final outputs of the FUSED op (tight)
    overhead = caps.kernel_launch_overhead_ns    # ONE launch
    CostEstimate { flops, bytes_moved: bytes, kernel_overhead_ns: overhead }
```

## 5. Layering — this makes Task F optional, not a prerequisite

- **Layer-1 default** — `cost_from_decompose` (this spec). Always available, never zero.
- **Layer-1 override** — a contract-declared `cost:` expression (the deferred **Task F**
  trampoline, `kernel-contract-adoption-plan.md §2.3`) when an author wants to beat the
  composition's precision. Optional.
- **Layer-2** — the Judge's measured latency, overriding both once real-run data exists.

So the fused-op cost pipeline degrades gracefully: measured › declared › composed-from-recipe
› (never) zero.

## 6. Caveats (name them honestly)

- **Algorithm-changing fusions.** For *algebraic* fusions (gelu, rmsnorm, layernorm, softmax)
  the decompose is arithmetically exact → FLOPs match tightly. For fusions that change the
  algorithm (flash-attention trades recompute for O(n) memory), the decompose's FLOPs are an
  approximation (right order of magnitude, slightly low on compute, high on the intermediate
  bytes it *would* have moved). The Judge is precisely the corrector for these; the estimate
  degrades gracefully rather than lying.
- **Runtime-fused ops.** A Tier-2 runtime-registered fused op's decompose is "the region
  re-emitted" (`runtime_fused.rs`), so its primitive set is exactly the source region — the
  composition is immediate and needs no separate authoring. This is where the sentinel-zero is
  most likely today (JIT-adopted kernels), so the win is largest there.
- **Nested fused ops.** If the decompose contains another fused op, recurse (its own
  `cost_from_decompose`) or decompose to the base map first — the base map is the fixpoint of
  `decompose`, so a full lowering terminates at primitives.

## 7. Verification (born-red shape)

- A fused op whose registered cost is the sentinel gets a **nonzero** composed cost (vs the
  `CostEstimate::default()` zero) — the born-red: assert the composed cost > 0 and equals the
  expected fold for a known fusion (e.g. a fused `gelu` composed from its
  erf/mul/add/... decompose primitives).
- Backward-compat: a fused op WITH a measured Judge cell or a declared cost is unchanged (the
  default only fires for the sentinel).
- Regression: no existing placement/rank test changes (fused ops that were priced at zero now
  price higher — assert any test that depended on the zero is updated to the composed value,
  the same "the zero was the bug" reframe as Part A).

## 8. Sequencing

Independent of the in-flight importer-gap work; builds directly on what shipped this session
(Part A registered caps, Part C per-backend throughput, the migrated primitive cost fns). Do
it as a standalone slice: (1) `boundary_io_bytes` + the fold; (2) the sentinel-fallback hook at
the optimizer's fused-op cost read; (3) born-red + the placement-regression sweep. Lands before
or independently of Task F (which it demotes to an optional refinement).
