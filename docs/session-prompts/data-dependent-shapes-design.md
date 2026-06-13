# Data-dependent output shapes — design

**Status**: design / not started (2026-06-13). Proposes a future change to
[`03-ir`](../architecture/03-ir.md); promote the chosen mechanism into `03-ir` when built.

## Problem

Today a `Node` fixes `op / inputs / shape / dtype` at construction
(`fuel-graph/src/lib.rs`), and `Shape` is a concrete dim vector. The architecture set has
no story for **output shapes that depend on runtime values**. Several roadmap items need
one:

- Phase 8.5 dynamic activation sparsity → `Op::NonZeroIndices` (output length = count of
  non-zeros, a function of the data).
- top-k / sort (named highest-value StableHLO vocabulary gap).
- Dynamic MoE routing (per-expert token counts depend on the router's realized argmax).
- NMS / detection post-process (kept-box count is data-dependent).

Under lazy-only with no eager fallback, getting this wrong is paid on every request — and
shape-specialized planning means every distinct prompt length is a different graph and a
different plan (the classic lazy-compilation recompilation cliff).

## What already works (the pattern to generalize)

The KV cache already solves a *restricted* form of "shape that changes at realize time"
without dynamic shapes at all:

- K/V live in **pre-allocated fixed-capacity buffers** `[1, n_kv_heads, max_seq_len,
  head_dim]`.
- Each decode step writes the new token's K/V via `Op::WriteSlice` into that fixed buffer.
- The **live extent is a host scalar** (`cached_len`) threaded into ops as a parameter
  (RoPE offset, slice bounds) — it is *not* a changing tensor shape.

So the graph's ops/shapes/dtypes are identical step-to-step; only a host-scalar parameter
moves. This is the seed of the general answer.

## The autoregressive barrier (why "one DAG at load" can't cover everything)

The whole multi-token inference DAG genuinely cannot be built once at load: each token's
input is the *previous* token's realized-and-sampled output (a host argmax/multinomial
decision that doesn't exist until that step realizes), and the loop length is
data-dependent (stops at EOS). Data-dependent *shapes* are the same phenomenon one level
down — a count that isn't known until some node realizes. The design must therefore say
*where the realize barrier falls* and *how the plan adapts across it*.

## Three mechanisms (use in this preference order)

### 1. Capacity (upper-bound) shape + valid-count side output — DEFAULT

Allocate the worst-case capacity and carry the true count as a second output, consumed
downstream as a host-scalar param (exactly the `cached_len` pattern):

- `Op::NonZeroIndices(x)` → bundle `{ indices: [capacity], count: scalar }` where
  `capacity = x.numel()` (or a tighter static bound). Reuse the **multi-output bundle**
  mechanism (`12-multi-output`: `Op::View`/`Op::ViewOwned`, `Storage.bundle`) that already
  exists — slot 0 the capacity buffer, slot 1 the count.
- Downstream gather/scatter/compute ops honor `count` as a host-scalar param; the tail of
  the capacity buffer is unused, not recomputed.
- **top-k is the easy case**: `k` is known statically, so the output shape is already
  concrete — no dynamic shape needed at all, just the op.
- MoE: capacity = `tokens × top_k`; per-expert counts ride as a small host-side vector.

Keeps `Node.shape` concrete, planning fully static, and the plan cache / structural hash
(Stage 5) reusable across calls. **Cost**: peak memory = worst case. Acceptable for
top-k/MoE; for very sparse `NonZeroIndices` it can over-allocate — that's when mechanism 2
earns its complexity.

### 2. Symbolic dim + realize-time specialization — for when capacity is too wasteful

Introduce a symbolic dim variable `N?` bounded by a static max. The planner plans a
**parametric template** over `N?` and emits a small **adjustment plan**: at the realize
barrier where the producing node's count becomes known, the runtime specializes the
downstream region by selecting among a few pre-planned **size buckets** (sequence
bucketing, generalized) rather than replanning from scratch. This is the "flex region with
a plan for how the DAG adjusts" idea, made concrete:

- The structural hash (Stage 5) keys on bucket id, not exact count → bounded number of
  cached plans, reuse within a bucket.
- The barrier is explicit and few (one per dynamic-count producer), so fusion is only
  broken there, not everywhere.

More machinery (symbolic dims in `Shape`, bucket selection in the runtime, the planner
emitting parametric fragments). Build only when mechanism 1's over-allocation is measured
to matter.

### 3. Host-realize boundary — only at genuine host decisions

Realize the count to the host, then build the next graph region with the now-concrete
shape. This is the autoregressive-sampling pattern generalized. Simplest, but forces a
realize barrier (D2H + loss of cross-barrier fusion), so reserve it for places a host
decision already happens (token sampling, beam management).

## Recommendation

Default to **(1)** — it is the smallest delta, preserves static planning, and reuses two
things the project already has (the multi-output bundle and the host-scalar-extent
pattern). Adopt **(2)** only where capacity waste is measured. Use **(3)** only at real
host-decision points. Record the chosen mechanism per dynamic op in `03-ir` when
implemented, and make sure the Stage 5 structural hash + any sequence-length/bucketing
policy are designed together (they share the "exact shape vs bucket" decision).

## Cross-refs

- `03-ir` (Node/Shape; promote chosen mechanism here)
- `12-multi-output` (the bundle mechanism mechanism 1 reuses)
- `load-time-incremental-planner.md` (Stage 5 structural hash must agree on shape vs
  bucket keying)
- ROADMAP §"Post-wipe resume addendum (2026-06-13)"
